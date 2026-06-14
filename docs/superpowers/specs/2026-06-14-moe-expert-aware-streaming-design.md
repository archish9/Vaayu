# Design: MoE Expert-Aware Streaming

**Date:** 2026-06-14  
**Status:** Approved  
**Effort:** Very High  
**Risk:** High  
**Impact:** 91s/token → ~2.5s/token for 1T+ MoE models

---

## 1. Problem Statement

Vaayu currently streams entire layer shards from NVMe on every token. For a dense model (e.g., Llama-3 70B), every layer shard contains one transformer layer's weights — this is correct and optimal.

For a **Mixture-of-Experts (MoE)** model (e.g., Mixtral 8x7B, Kimi-K2.6), each layer shard contains:
- Shared attention weights (always needed)
- **All** expert FFN weights (only 2 of N are ever used per token)

This means Vaayu reads 100% of expert weight data but uses only `top_k / num_experts` of it:

| Model | Experts | top-k | Waste ratio | Shard size | Useful bytes |
|-------|---------|-------|-------------|-----------|--------------|
| Mixtral 8x7B | 8 | 2 | 75% | ~1.3 GB | ~325 MB |
| Kimi-K2.6 | 256 | 2 | 99.2% | ~9.7 GB | ~77 MB |

**Latency math at 6.5 GB/s NVMe:**
- Without expert streaming: 594 GB ÷ 6.5 GB/s = **91.3 s/token** (Kimi-K2.6)
- With expert streaming: ~16 GB ÷ 6.5 GB/s = **2.46 s/token**

`vaayu_base.py:690` already logs: `"Expert-aware streaming (Phase 3): pending — currently streams full MoE layer"`.

---

## 2. Architecture Overview

### 2.1 Approach: Monkey-patch + Forward Loop History Tracking

**Why monkey-patch over split layer_names:**
- No disk format changes required — shards stay as-is
- Works immediately with existing `forward()` pipeline
- Generic: detects MoE blocks via `nn.ModuleList` heuristic, not hardcoded architecture names
- Closures inject all dependencies (cache, offsets, checkpoint path) at patch time

**Why forward-loop history tracking:**
- Enables Fix 2 (speculative expert prefetch) to overlap with next layer's attention compute
- Uses existing `ThreadPoolExecutor` in `forward()` — no new thread infrastructure
- History from token N predicts token N+1 experts with 70–90% accuracy (MoE temporal locality)

### 2.2 Pipeline Per Token (MoE model)

```
MAIN THREAD                                    BACKGROUND THREAD
─────────────────────────────────────────────  ─────────────────────────────
layer.K (dense or MoE attention):
  [ExpertRAMCache / NVMe] → GPU
  layer.K.forward()  ────────────────────────► prefetch layer.K+1 attn weights

layer.K+1 (MoE layer):
  wait for attn weights (prefetched) ✅
  inject attn → GPU
  ─── PATCHED MoE FORWARD ───
  run attention (loaded)
  run PINNED router gate → active_experts=[3,7]

  for expert_idx in [3, 7]:
    ExpertRAMCache.get(K+1, expert_idx)
      HIT  → RAM tensors (instant, ~0ms)
      MISS → load_expert_sparse()
               io_uring precise byte range read
               → pinned buffer → CPU tensors

  inject expert [3,7] → GPU
  run sparse expert FFN
  free expert tensors from GPU
  ─── END PATCHED FORWARD ───
  return hidden_state, routing_indices=[3,7]

vaayu_base receives routing_indices:
  _expert_history[K+1] = {3, 7}  ───────────► speculative prefetch layer.K+2:
                                                  predict = _expert_history.get(K+2, {})
                                                  for miss in predict - cache:
                                                    io_uring read expert bytes
                                                    ExpertRAMCache.put(K+2, idx, tensors)
                                               ↑ overlaps with K+2 attn compute

layer.K+2 (next MoE):
  attn already prefetched ✅
  experts already in cache (70–90% hit) ✅
  near-zero stall
```

---

## 3. Component Specifications

### 3.1 `ExpertRAMCache` — `vaayu/persist/expert_cache.py`

LRU cache keyed by `(layer_idx: int, expert_idx: int)`. Separate from `LayerRAMCache` (which caches full layers for dense models).

```python
class ExpertRAMCache:
    def __init__(self, budget_bytes: int): ...
    def get(self, layer_idx: int, expert_idx: int) -> dict | None: ...
    def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None: ...
    def has(self, layer_idx: int, expert_idx: int) -> bool: ...
    def evict_to_budget(self) -> None: ...
    def stats(self) -> str: ...
```

**Budget:** `auto_detect_cache_budget()` result minus `LayerRAMCache` claimed bytes.  
**Eviction:** LRU via `OrderedDict.move_to_end()` + `popitem(last=False)`.  
**Thread safety:** single `threading.Lock` guards all mutations.  
**Clone semantics:** `put()` clones tensors synchronously — expert tensors are small (~2 GB per expert for Kimi 4-bit, ~400 MB for Mixtral BF16), clone is fast.

### 3.2 `ExpertOffsetIndex` — `vaayu/persist/expert_offset_index.py`

Scans safetensors headers at init to build a byte-offset map for every expert tensor in every MoE layer shard. Persisted as JSON alongside the checkpoint to avoid re-scanning on every startup.

```python
class ExpertOffsetIndex:
    def __init__(self, checkpoint_path: str): ...
    def build(self) -> None: ...           # scans headers, writes JSON cache
    def load_cached(self) -> bool: ...     # loads JSON if exists + fresh
    def get_expert_offsets(
        self,
        layer_name: str,
        expert_idx: int
    ) -> dict[str, ExpertTensorMeta]: ...
    # Returns {tensor_key: ExpertTensorMeta(file_offset, byte_length, shape, dtype)}
```

**Detection regex:** `r'\.experts\.(\d+)\.'` matches all MoE architectures:
- Mixtral: `model.layers.3.block_sparse_moe.experts.5.w1`
- DeepSeek: `model.layers.3.mlp.experts.5.gate_proj`
- Kimi: `model.layers.3.feed_forward.experts.5.down_proj`
- Qwen MoE: `model.layers.3.mlp.experts.5.gate_proj`

**JSON cache file:** `<checkpoint_path>/vaayu_expert_offsets.json`  
**Staleness check:** JSON mtime vs oldest shard mtime — rebuild if stale.

### 3.3 Sparse Expert Reader — `vaayu/persist/safetensor_model_persister.py`

New function added to existing file:

```python
def load_expert_sparse(
    file_path: str,
    expert_metas: dict[str, ExpertTensorMeta],
    use_direct_io: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Load only the specified expert tensor byte ranges from a shard file.
    Issues one io_uring_read_into() call per tensor (or groups contiguous
    tensors into a single call when their byte ranges are adjacent).
    Returns {tensor_key: cpu_tensor} using pinned buffer for fast H2D.
    """
```

**Contiguity optimization:** If expert tensor byte ranges are adjacent in the file (common — experts are stored sequentially), merges into a single io_uring call covering `[min_offset, max_offset+max_len]`.

**Buffer strategy:** Reuses `_PINNED_BUFS` double-buffer. Expert tensors for top-k=2 are much smaller than a full layer, so always fit.

### 3.4 MoE Block Patcher — `vaayu/moe_patch.py`

```python
def detect_moe_blocks(model: nn.Module) -> list[MoEBlockInfo]:
    """
    Generic detection: finds any module containing:
      1. An nn.ModuleList with > 4 child modules (expert list)
      2. A sibling Linear layer named 'gate' or 'router' (router)
    Returns list of MoEBlockInfo(layer_idx, module_path, moe_module, router_attr, experts_attr).
    """

def patch_moe_blocks(
    model: nn.Module,
    expert_cache: ExpertRAMCache,
    offset_index: ExpertOffsetIndex,
    checkpoint_path: str,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,  # weakref to VaayuBaseModel — closure writes _expert_history directly
) -> list[MoEBlockInfo]:
    """
    For each detected MoE block:
      1. Save reference to original experts ModuleList
      2. Replace block.forward with sparse_moe_forward closure
      3. Router weights stay pinned (injected separately by _init_pinned_layers)
    Returns list of patched blocks (for logging/debugging).
    """
```

**Sparse MoE forward (closure):**

```python
def _make_sparse_moe_forward(block_info, expert_cache, offset_index,
                              checkpoint_path, top_k, device, dtype,
                              routing_callback):
    layer_name = block_info.layer_name  # e.g. "model.layers.3"
    layer_idx  = block_info.layer_idx

    def sparse_forward(self, hidden_states, **kwargs):
        # 1. Run router (weights permanently in VRAM)
        router_logits = getattr(self, block_info.router_attr)(hidden_states)
        routing_weights, expert_indices = torch.topk(
            router_logits.softmax(dim=-1), top_k, dim=-1
        )

        unique_experts = expert_indices.unique().tolist()

        # 2. Load missing experts (cache-first)
        expert_tensors = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                file_path = checkpoint_path / f"{layer_name}.safetensors"
                raw = load_expert_sparse(str(file_path), metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        # 3. Temporarily inject active expert weights → GPU, run FFN, unload
        # Strategy: for each unique active expert, use set_module_tensor_to_device
        # to inject its weights into the existing expert module (experts_list[eidx]),
        # call the module's forward(), then move it back to meta.
        # This reuses the architecture's own expert MLP forward — no custom kernels.
        experts_list = getattr(self, block_info.experts_attr)
        final_hidden = torch.zeros_like(hidden_states)

        for token_pos in range(hidden_states.shape[1]):
            for k_idx in range(top_k):
                eidx = expert_indices[0, token_pos, k_idx].item()
                weight = routing_weights[0, token_pos, k_idx]
                raw = expert_tensors[eidx]

                # Inject weights into expert module
                expert_mod = experts_list[eidx]
                for param_name, tensor in raw.items():
                    set_module_tensor_to_device(
                        expert_mod, param_name.split('.')[-1],
                        device, value=tensor.to(device, dtype=dtype)
                    )

                # Run expert forward
                expert_out = expert_mod(hidden_states[:, token_pos:token_pos+1, :])
                final_hidden[:, token_pos:token_pos+1, :] += weight * expert_out

                # Unload expert weights back to meta
                expert_mod.to('meta')

        # 4. Write routing result to VaayuBaseModel._expert_history.
        # The forward() loop reads this after layer.forward() returns
        # and submits speculative prefetch using the local executor.
        # Using direct dict write (not callback) avoids executor lifecycle issues
        # (executor is local to each forward() call; patch is applied at init).
        vaayu_ref._expert_history[layer_idx] = unique_experts
        return final_hidden

    return sparse_forward
```

### 3.5 Router Gate Pinning — `vaayu/vaayu_base.py::_init_pinned_layers()`

Added Phase 1.5 between shared layers and transformer layers:

```python
# Phase 1.5: Pin MoE router gates (MoE models only)
if self.is_moe:
    import re
    _gate_re = re.compile(r'\.(gate|router)\b')
    for shard_name in self.layer_names:
        if not shard_name.startswith(layer_prefix):
            continue
        shard_path = Path(self.checkpoint_path) / (shard_name + ".safetensors")
        header = _parse_safetensors_header(shard_path)
        gate_keys = [k for k in header
                     if _gate_re.search(k) and 'experts.' not in k]
        if not gate_keys:
            continue
        tensors = {k: v for k, v in
                   self.loader_strategy.load_from_disk(shard_name, str(self.checkpoint_path)).items()
                   if k in gate_keys}
        size = sum(t.nbytes for t in tensors.values())
        if used + size <= budget:
            inject_to_vram(tensors)
            self.pinned_layer_names.add(shard_name + ".gate")
            used += size
```

### 3.6 Forward Loop Changes — `vaayu/vaayu_base.py::forward()`

Two additions to the existing layer loop (non-MoE paths untouched):

The patched MoE forward writes routing results directly to `self._expert_history` on the `VaayuBaseModel` instance (via `vaayu_ref` closure). The `forward()` loop reads this dict after each layer and submits speculative prefetch using the local `executor`. This cleanly separates concerns: the patch handles sparse loading, the forward loop handles prefetch scheduling.

```python
# __init__ additions:
self._expert_history: dict[int, set[int]] = {}

# In forward() layer loop, after layer.forward() for MoE layers:
if self.is_moe and i in self._expert_history:
    predicted_layer_idx = i  # layer_idx set by patched forward
    pred = self._expert_history.get(predicted_layer_idx + 1, set())
    if pred:
        executor.submit(self._prefetch_experts, predicted_layer_idx + 1, pred)
```

`_prefetch_experts(layer_idx, predicted_experts)` checks `ExpertRAMCache`, issues `load_expert_sparse()` for misses, and calls `expert_cache.put()`. Runs on the existing `ThreadPoolExecutor` inside `forward()` — overlapping with next layer's attention compute.

---

## 4. MoE Detection (existing, unchanged)

`_detect_moe()` in `vaayu_base.py:664` already sets:
- `self.is_moe: bool`
- `self.moe_num_experts: int`
- `self.moe_top_k: int`

All new code gates on `self.is_moe`. Dense model paths are completely untouched.

---

## 5. Initialization Sequence

```
VaayuBaseModel.__init__():
  1. _detect_moe()                          ← existing
  2. _init_pinned_layers()                  ← modified: adds gate pinning
  3. ExpertOffsetIndex(checkpoint_path)     ← new: scan headers, build/load JSON
  4. ExpertRAMCache(budget_bytes)           ← new: allocate cache
  5. patch_moe_blocks(model, ...)           ← new: replace MoE forward()
  6. _prepopulate_expert_cache()            ← new: warm top-N most common experts
```

**Expert cache pre-population:** On first startup (cold cache), prefetch the top-2 most common experts per layer based on statistical frequency (load all experts, sort by index assuming uniform distribution — actually just load experts 0 and 1 as a warm start heuristic). This ensures token 1 has some cache hits.

---

## 6. Error Handling & Fallback

| Failure | Behavior |
|---------|----------|
| `ExpertOffsetIndex` build fails (corrupt shard) | Fall back to loading full layer (safe, slow) |
| `load_expert_sparse` io_uring error | Fall back to `load_layer()` full shard load |
| `detect_moe_blocks` finds no MoE blocks | No patching applied, log warning |
| Expert index not in offset map | Load full shard as fallback |
| `ExpertRAMCache` over budget | Evict LRU, retry put |

All fallbacks degrade to current behavior (full layer streaming) — never crash.

---

## 7. Documentation Plan

### New file: `documentation/moe_expert_aware_streaming.md`
Complete technical reference covering:
- Problem + math (91s → 2.5s)
- All components with class signatures
- Full pipeline diagram
- Monkey-patch internals: detection algorithm, closure injection
- ExpertRAMCache: LRU eviction, budget allocation, thread safety
- ExpertOffsetIndex: header scanning, JSON cache, staleness check
- Sparse reader: contiguity optimization, pinned buffer reuse
- Speculative prefetch: history table, cold-start behavior, accuracy expectations
- Router gate pinning: detection regex, budget accounting
- Debug guide: log messages, expected cache hit rates, how to verify sparse reads
- Debugging checklist for future issues

### Update: `documentation/architecture.md`
Add "Section 6: MoE Expert-Aware Streaming" summarizing the feature and referencing the detailed doc.

### Update: `documentation/optimization_bible.md`
Add Era entry for MoE Expert-Aware Streaming with projected latency numbers.

---

## 8. Verification Plan

1. Run any MoE model (Mixtral 8x7B if available)
2. Verify `ExpertOffsetIndex` JSON written to checkpoint dir
3. Monitor logs: `[Vaayu-MoE] expert cache: Nhit/Mmiss` — expect >70% hits from token 2
4. Monitor NVMe read throughput — should drop from ~full-shard to ~expert-fraction per layer
5. Verify dense model (any non-MoE) unchanged behavior — `is_moe=False` skips all new paths
6. Unit test: `ExpertRAMCache` LRU eviction with mock tensors
7. Unit test: `ExpertOffsetIndex.build()` on a synthetic 2-expert safetensors file

---

## 9. Files Changed

| File | Change type |
|------|------------|
| `vaayu/persist/expert_cache.py` | New |
| `vaayu/persist/expert_offset_index.py` | New |
| `vaayu/moe_patch.py` | New |
| `vaayu/persist/safetensor_model_persister.py` | Modified (add `load_expert_sparse`) |
| `vaayu/vaayu_base.py` | Modified (gate pinning + history tracking + prefetch) |
| `documentation/moe_expert_aware_streaming.md` | New |
| `documentation/architecture.md` | Modified (add Section 6) |
| `documentation/optimization_bible.md` | Modified (add Era entry) |
