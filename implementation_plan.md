# Implementation Plan: Universal MoE Expert-Aware Streaming Fix

## Problem Summary

Vaayu's MoE streaming is completely inert because the HuggingFace `transformers` library refactored Mixtral's internal module structure. The checkpoint files on disk use old naming (`block_sparse_moe`, per-expert `w1/w2/w3`), while the model instantiated by `from_config()` uses new naming (`mlp`, fused `gate_up_proj/down_proj`). Three systems fail: weight injection, MoE detection, and sparse forward.

See [moe_deep_debug_analysis.md](file:///home/a/.gemini/antigravity-ide/brain/020709ff-c4e6-4587-a4e3-6e6c1c9aa28c/moe_deep_debug_analysis.md) for the full root cause analysis.

## Design Principle: Universal Compatibility

The fix must handle:
- **Dense models** (Llama, Qwen, Gemma) → no change, all MoE paths gated on `self.is_moe`
- **Old-style MoE** (legacy transformers, or models with `nn.ModuleList` experts + `nn.Linear` gate) → existing patch works once detection is broadened
- **New-style MoE** (current transformers: fused `MixtralExperts` + `MixtralTopKRouter`) → new fused forward strategy
- **Any future MoE** (DeepSeek, Qwen MoE, Kimi) → generic heuristics, not architecture-specific hardcoding

---

## Data Shape Reference

### On disk (safetensors checkpoint — per-expert individual weights):
```
model.layers.0.block_sparse_moe.gate.weight         [8, 4096]         (router)
model.layers.0.block_sparse_moe.experts.0.w1.weight  [14336, 4096]    (gate_proj)
model.layers.0.block_sparse_moe.experts.0.w2.weight  [4096, 14336]    (down_proj)
model.layers.0.block_sparse_moe.experts.0.w3.weight  [14336, 4096]    (up_proj)
... × 8 experts
```

### In model (new transformers — fused batched weights):
```
mlp.gate.weight                  [8, 4096]          (MixtralTopKRouter)
mlp.experts.gate_up_proj         [8, 28672, 4096]   (w1 ∥ w3 fused for all 8 experts)
mlp.experts.down_proj            [8, 4096, 14336]   (w2 for all 8 experts)
```

### Fused tensor mapping:
```
gate_up_proj[expert_idx]  = cat(w1.weight, w3.weight, dim=0)   # [28672, 4096]
down_proj[expert_idx]     = w2.weight                           # [4096, 14336]
```

---

## Proposed Changes

### Component 1: MoE Key Remapper — `vaayu/moe_key_remap.py`

> [!IMPORTANT]
> This is the critical piece that makes **all** existing injection paths work without modifying them.

#### [NEW] [moe_key_remap.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_key_remap.py)

A standalone module that builds a key remapping table by comparing checkpoint keys against the live model parameter tree. This runs once at init and provides two functions:

1. **`build_moe_key_remap(model, checkpoint_path, layer_names_dict)`** — probes the first MoE layer shard's header keys and the model's `named_parameters()`. Returns a `MoEKeyRemapper` object.

2. **`MoEKeyRemapper.remap_state_dict(state_dict)`** — takes a loaded state dict with checkpoint keys and returns a new dict with model-compatible keys + fused tensors.

**Detection logic (runs once per model load):**
```python
# 1. Grab one MoE shard's header keys (e.g. layer 0)
disk_keys = parse_safetensors_header(layer_0_shard)

# 2. Grab live model's parameter names for the same layer
model_keys = {name for name, _ in model.model.layers[0].named_parameters()}

# 3. Detect structure type
has_bsm_disk = any("block_sparse_moe" in k for k in disk_keys)
has_bsm_model = any("block_sparse_moe" in k for k in model_keys)
has_mlp_model = any(k.startswith("mlp.") for k in model_keys)

# 4. Determine remap mode
if has_bsm_disk and not has_bsm_model and has_mlp_model:
    mode = "bsm_to_mlp_fused"     # old checkpoint → new model (Mixtral case)
elif has_bsm_disk and has_bsm_model:
    mode = "identity"              # old checkpoint + old model (legacy transformers)
else:
    mode = "identity"              # no remap needed
```

**Fused remapping logic (mode `bsm_to_mlp_fused`):**
```python
def remap_state_dict(self, state_dict: dict) -> dict:
    result = {}
    # 1. Path rename: block_sparse_moe → mlp (for non-expert keys like gate, layernorms)
    # 2. Fuse per-expert w1+w3 → gate_up_proj[expert_idx]
    # 3. Rename w2 → down_proj[expert_idx]
    # 4. Pass through non-MoE keys unchanged (self_attn.*, layernorm.*)
    
    # Group expert tensors by (layer, expert_idx)
    expert_groups = {}  # {expert_idx: {w1: tensor, w2: tensor, w3: tensor}}
    for key, tensor in state_dict.items():
        m = EXPERT_RE.search(key)
        if m:
            expert_idx = int(m.group(1))
            # e.g. "w1.weight" from the key tail
            weight_name = key.rsplit(".", 2)[-2]  # w1, w2, or w3
            expert_groups.setdefault(expert_idx, {})[weight_name] = tensor
        else:
            # Simple path rename: block_sparse_moe → mlp
            new_key = key.replace("block_sparse_moe", "mlp")
            result[new_key] = tensor
    
    # Fuse experts into batched tensors
    if expert_groups:
        n_experts = max(expert_groups.keys()) + 1
        # gate_up_proj: cat(w1, w3) per expert → stack across experts
        gate_up_list = []
        down_list = []
        for eidx in range(n_experts):
            eg = expert_groups[eidx]
            gate_up_list.append(torch.cat([eg["w1"], eg["w3"]], dim=0))
            down_list.append(eg["w2"])
        
        # Derive the fused key prefix from the shard (layer-relative)
        result[f"{layer_prefix}.mlp.experts.gate_up_proj"] = torch.stack(gate_up_list)
        result[f"{layer_prefix}.mlp.experts.down_proj"] = torch.stack(down_list)
    
    return result
```

**Why a separate module, not inline in `move_layer_to_device`:**
- Testable in isolation
- Clean separation — the rest of Vaayu doesn't need to know about checkpoint format differences
- Can be extended for future architecture mismatches (DeepSeek, Kimi) without touching core code

---

### Component 2: Integration into Weight Loading — `vaayu/vaayu_base.py`

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

**`__init__` (after `_detect_moe`, around line 451):**
```python
# Build MoE key remapper (one-time, before any weight loading)
if self.is_moe:
    from .moe_key_remap import build_moe_key_remap
    self._moe_key_remapper = build_moe_key_remap(
        self.model, str(self.checkpoint_path), self.layer_names_dict
    )
else:
    self._moe_key_remapper = None
```

**`move_layer_to_device` (line 1234):**

Before existing path-remapping logic, add:
```python
# Apply MoE key remapping (checkpoint format → live model format)
if self._moe_key_remapper is not None:
    state_dict = self._moe_key_remapper.remap_state_dict(state_dict)
```

This is the minimal touch point — all other loading paths (`_init_pinned_layers`, `inject_to_gpu`, `_prepopulate_ram_cache`) flow through `move_layer_to_device` or `loader_strategy.inject_to_gpu` which calls it. So remapping once here fixes everything.

**`_init_pinned_layers` Phase 1.5 gate pinning (line 1026):**

Before calling `inject_to_gpu` for gate tensors, apply remapping:
```python
if self._moe_key_remapper is not None:
    gate_tensors = self._moe_key_remapper.remap_state_dict(gate_tensors)
```

---

### Component 3: Broadened MoE Detection — `vaayu/moe_patch.py`

#### [MODIFY] [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py)

**`detect_moe_blocks()` — expand the heuristics to handle both structures:**

Current code checks for `isinstance(child, nn.ModuleList)` and `isinstance(child, nn.Linear)`. These fail on the new structure.

New detection algorithm:
```python
def detect_moe_blocks(model: nn.Module) -> list[MoEBlockInfo]:
    results = []
    for mod_path, mod in model.named_modules():
        experts_attr = None
        router_attr = None
        expert_style = None  # "module_list" or "fused"
        
        for child_name, child in mod.named_children():
            # Old style: nn.ModuleList with >4 children
            if isinstance(child, nn.ModuleList) and len(child) > _MIN_EXPERTS:
                experts_attr = child_name
                expert_style = "module_list"
            
            # New style: any module with a num_experts attribute and
            # batched weight tensors (gate_up_proj or similar)
            elif (hasattr(child, 'num_experts') 
                  and getattr(child, 'num_experts', 0) > _MIN_EXPERTS
                  and not isinstance(child, nn.Linear)):
                experts_attr = child_name
                expert_style = "fused"
            
            # Router: nn.Linear named gate/router (old)
            if isinstance(child, nn.Linear) and child_name in _ROUTER_NAMES:
                router_attr = child_name
            
            # Router: any module with a .weight parameter named gate/router (new)
            # MixtralTopKRouter has weight but isn't nn.Linear
            elif (child_name in _ROUTER_NAMES 
                  and hasattr(child, 'weight') 
                  and isinstance(getattr(child, 'weight', None), (nn.Parameter, torch.Tensor))):
                router_attr = child_name
        
        if experts_attr is None or router_attr is None:
            continue
        # ... rest of detection (extract layer_idx, etc.)
        
        info = MoEBlockInfo(...)
        info.expert_style = expert_style  # NEW FIELD
        results.append(info)
    return results
```

**`MoEBlockInfo` dataclass — add `expert_style` field:**
```python
@dataclass
class MoEBlockInfo:
    layer_idx: int
    layer_name: str
    module_path: str
    moe_module: nn.Module
    router_attr: str
    experts_attr: str
    expert_style: str  # NEW: "module_list" or "fused"
```

---

### Component 4: Dual Sparse Forward Strategy — `vaayu/moe_patch.py`

#### [MODIFY] [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py)

**`_make_sparse_moe_forward` — dispatch based on `expert_style`:**

```python
def _make_sparse_moe_forward(block_info, ...):
    if block_info.expert_style == "fused":
        return _make_fused_sparse_forward(block_info, ...)
    else:
        return _make_modulelist_sparse_forward(block_info, ...)
```

**`_make_modulelist_sparse_forward`** — the existing `sparse_forward` (old-style, per-expert modules). Mostly unchanged, already works if detection succeeds.

**`_make_fused_sparse_forward`** — new, for fused expert modules:

```python
def _make_fused_sparse_forward(block_info, expert_cache, offset_index,
                                 checkpoint_path, top_k, device, dtype, vaayu_ref):
    from .persist.safetensor_model_persister import load_expert_sparse
    
    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx
    
    def sparse_forward(hidden_states, **kwargs):
        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn  # SiLU activation
        
        # 1. Run router (pinned in VRAM)
        B_S, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)
        
        # MixtralTopKRouter returns (logits, weights, indices)
        # Direct call: gate returns (logits, top_k_weights, top_k_index)
        _, routing_weights, expert_indices = router(flat_hidden)
        
        unique_experts = expert_indices.flatten().unique().tolist()
        
        # 2. Load active expert weights from disk/cache
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    continue
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw
        
        # 3. Compute sparse MoE manually (no inject needed, raw tensor math)
        #    Mirrors MixtralExperts.forward() but using disk-loaded per-expert tensors
        final_hidden = torch.zeros_like(flat_hidden)
        
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                expert_indices, num_classes=experts_mod.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
        
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue
            
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue
            
            current_state = flat_hidden[token_idx]
            
            # Find w1, w2, w3 tensors from the loaded per-expert dict
            # Keys are like "model.layers.0.block_sparse_moe.experts.0.w1.weight"
            w1 = w2 = w3 = None
            for k, v in raw.items():
                if ".w1." in k: w1 = v.to(device, dtype=dtype, non_blocking=True)
                elif ".w2." in k: w2 = v.to(device, dtype=dtype, non_blocking=True)
                elif ".w3." in k: w3 = v.to(device, dtype=dtype, non_blocking=True)
            
            if w1 is None or w2 is None or w3 is None:
                continue
            
            # Compute: gate = act(x @ w1.T), up = x @ w3.T, out = (gate * up) @ w2.T
            gate = act_fn(torch.nn.functional.linear(current_state, w1))
            up = torch.nn.functional.linear(current_state, w3)
            current_hidden = torch.nn.functional.linear(gate * up, w2)
            
            # Apply routing weights
            current_hidden = current_hidden * routing_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, current_hidden.to(final_hidden.dtype))
            
            # Free GPU tensors immediately
            del w1, w2, w3
        
        # 4. Write routing history for speculative prefetch
        vaayu_ref._expert_history[layer_idx] = unique_experts
        
        # Reshape back and return matching MoE signature
        return final_hidden.view(B_S, S, H)
    
    return sparse_forward
```

> [!IMPORTANT]
> Key difference from existing sparse_forward: The fused forward does **no weight injection into the model**. It loads per-expert w1/w2/w3 from disk, moves them to GPU as temporary tensors, runs `F.linear()` directly, and frees them. This completely sidesteps the path mismatch problem — we never call `set_module_tensor_to_device()` for expert weights.

---

### Component 5: Forward Loop MoE Expert Filtering Fix — `vaayu/vaayu_base.py`

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

The forward loop at [lines 1657-1659](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L1657-L1659) filters expert tensors from the state dict before injection (so the patched MoE forward handles them). This logic is correct but currently never activates because `_moe_blocks` is empty (0 patched).

Once Component 3 fixes detection, this path will activate correctly. However, the filtering regex needs to continue matching the **disk key format** (`block_sparse_moe.experts.N`), not the model key format. Since the remapper from Component 2 runs inside `move_layer_to_device()` (which is called by `inject_to_gpu`), we need the expert filter to run **before** remapping.

**Change:** Move the expert tensor filter to happen in the forward loop before `inject_to_gpu`, which is already where it lives. No change needed — the existing code at line 1657 already filters before calling `inject_to_gpu`. ✅

But we also need to **not remap** expert tensors through `MoEKeyRemapper` when they're filtered out. This is automatic since filtered-out keys never reach `move_layer_to_device`.

---

### Component 6: Return Value Compatibility — `vaayu/moe_patch.py`

#### [MODIFY] [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py)

The new `MixtralSparseMoeBlock.forward()` returns a **single tensor** (not a `(hidden_states, router_logits)` tuple like the old version). The forward loop at [line 1776](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L1776) does:
```python
new_seq = layer_out[0] if isinstance(layer_out, tuple) else layer_out
```

The patched forward replaces `MixtralSparseMoeBlock.forward`, not the entire layer's `MixtralDecoderLayer.forward`. So the return value needs to match what `MixtralDecoderLayer.forward` expects from `self.mlp(...)`.

**In the new transformers**, `MixtralDecoderLayer.forward` calls `self.mlp(hidden_states)` and expects a single tensor back. The existing patch returns `(final_hidden, router_logits)` — this will break.

**Fix:** The fused sparse forward should return **just the hidden states tensor**, matching the new `MixtralSparseMoeBlock.forward` signature.

For old-style models that expect `(hidden_states, router_logits)`, the module_list sparse forward should continue returning the tuple.

> [!IMPORTANT]
> The `expert_style` field on `MoEBlockInfo` drives which return format is used. Each sparse forward function returns the format matching its target architecture.

---

## Files Changed Summary

| File | Change | Risk |
|------|--------|------|
| [moe_key_remap.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_key_remap.py) | **NEW** — checkpoint→model key remapper with fused tensor assembly | Low (new file, no existing code touched) |
| [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py) | **MODIFY** — init remapper, apply in `move_layer_to_device` + gate pinning | Medium (two small insertion points) |
| [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py) | **MODIFY** — broaden detection, add `expert_style` field, add fused sparse forward | Medium (detection logic rewrite + new forward) |

**Files NOT changed (untouched):**
- `expert_cache.py` — works as-is (caches per-expert tensors regardless of structure)
- `expert_offset_index.py` — works as-is (indexes by disk keys, which is exactly what we need)
- `safetensor_model_persister.py` — `load_expert_sparse` works as-is (byte-level reads, format-agnostic)
- `loader_strategy.py` — untouched
- `auto_model.py` — untouched
- Dense model code paths — completely untouched, gated on `self.is_moe`

---

## Verification Plan

### Automated
```bash
# 1. Run Mixtral 8x7B with MoE streaming
source venv/bin/activate && python test_quick.py
```

Expected:
- `[Vaayu-MoE] N MoE blocks patched for sparse streaming` where N = 32
- No `'MixtralDecoderLayer' object has no attribute 'block_sparse_moe'` warnings
- `[VAAYU-IO]` reads should show much less data per layer for MoE layers (expert-fraction, not full)
- Model should produce coherent text (experts are actually being computed)
- Generation time should drop from ~10s/token to ~3-4s/token (reading only 2/8 expert data)

### Manual
- Verify dense model (e.g., Llama) still works unchanged
- Check expert cache hit rates from token 2+: `expert_cache=Nhit/Mmiss`

## Open Questions

> [!IMPORTANT]
> **Q1: w1/w3 fusion order.** The fused `gate_up_proj` tensor is `cat(w1, w3, dim=0)` — meaning the first `intermediate_size` rows are w1 (gate_proj) and the next `intermediate_size` rows are w3 (up_proj). I need to verify this by inspecting the `MixtralExperts.forward()` source, which does `gate, up = F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)`. This means: `gate = x @ gate_up_proj[:intermediate_size].T` and `up = x @ gate_up_proj[intermediate_size:].T`. So yes, **w1 is the first half, w3 is the second half**. The remapper should `torch.cat([w1, w3], dim=0)` in that order.

> [!IMPORTANT]
> **Q2: Expert budget calculation.** The log shows `Expert RAM cache: 0.0 GB budget`. This is because `psutil.virtual_memory().available` minus 8 GB reserve yields 0 or negative on your 27.6 GB RAM system (RAM cache already consumed available memory). This means even when the fix lands, expert caching won't help much. We should either: (a) share budget between LayerRAMCache and ExpertRAMCache, or (b) use a minimum floor for expert cache. **For this fix, I'll add a minimum floor of 1 GB for expert cache when MoE is detected.**
