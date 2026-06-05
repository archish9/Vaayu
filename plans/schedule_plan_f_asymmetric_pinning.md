# Schedule Plan F: Asymmetric Layer Pinning (Topology Awareness)

**Priority:** 3 (Parallel with Plan E)
**Effort:** Medium
**Risk:** Low-Medium
**Speed Gain:** 15-25% faster prefill, more stable streaming

## Objective
Replace the current "pin only embed/norm/lm_head" strategy with topology-aware asymmetric pinning that also pins the first N transformer layers permanently in VRAM.

## The Problem
Current pinning logic in `vaayu/vaayu_base.py` `_init_pinned_layers()` (lines 568-611) only pins 3 shared layers (~1 GB total). The remaining ~11 GB of available VRAM budget is wasted. Every transformer layer must stream from NVMe even though many could fit permanently.

## Current Code

### `_init_pinned_layers()` — line 568
```python
def _init_pinned_layers(self) -> None:
    budget = self._get_vram_pinning_budget()
    if budget <= 0:
        return
    candidates = [
        self.layer_names_dict["embed"],
        self.layer_names_dict["norm"],
        self.layer_names_dict["lm_head"],
    ]
    # ... loads each candidate into VRAM until budget exhausted
```

**Problem:** `candidates` only has 3 shared layers. Never includes transformer layers like `model.layers.0`, `model.layers.1`, etc.

## Implementation Steps

### Step 1: Add `max_pinned_layers` Parameter

**File:** `vaayu/vaayu_base.py`, `__init__()` (line 112)

Add to signature:
```python
max_pinned_layers=-1,  # -1 = auto (fill budget), 0 = shared only, N = shared + first N
```
Store as `self.max_pinned_layers = max_pinned_layers`.

### Step 2: Rewrite `_init_pinned_layers()`

**File:** `vaayu/vaayu_base.py` — replace lines 568-611 entirely:

```python
def _init_pinned_layers(self) -> None:
    """
    Pre-load layers into VRAM using asymmetric topology-aware pinning.
    
    Pinning priority (highest to lowest):
      1. embed_tokens — gateway for every token entering the model
      2. lm_head — gateway for every token leaving the model
      3. norm — tiny but accessed every token
      4. First N transformer layers — heaviest feature extraction during prefill
    """
    budget = self._get_vram_pinning_budget()
    if budget <= 0:
        return

    # Phase 1: Always pin shared layers (non-negotiable)
    priority_layers = [
        self.layer_names_dict["embed"],
        self.layer_names_dict["norm"],
        self.layer_names_dict["lm_head"],
    ]
    
    # Phase 2: Add early transformer layers
    layer_prefix = self.layer_names_dict["layer_prefix"]
    model_attr = self.model
    for attr_name in layer_prefix.split("."):
        model_attr = getattr(model_attr, attr_name)
    total_layers = len(model_attr)
    
    max_transformer = self.max_pinned_layers
    if max_transformer != 0:  # 0 means shared-only mode
        for i in range(total_layers):
            if max_transformer > 0 and i >= max_transformer:
                break
            priority_layers.append(f"{layer_prefix}.{i}")

    used = 0
    for layer_name in priority_layers:
        try:
            tensors = self.loader_strategy.load_from_disk(
                layer_name, self.checkpoint_path
            )
            size = sum(t.nbytes for t in tensors.values())
            if used + size > budget:
                print(f"[Vaayu-Pin] budget exhausted at {layer_name} "
                      f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB used)")
                break
            self.loader_strategy.inject_to_gpu(
                self.model, layer_name, tensors,
                self.running_device, self.running_dtype,
                self.hf_quantizer,
                getattr(self, "_path_remappings", {}),
            )
            self.pinned_layer_names.add(layer_name)
            used += size
            print(
                f"[Vaayu-Pin] pinned {layer_name}: "
                f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)"
            )
        except Exception as e:
            print(f"[Vaayu-Pin] skip {layer_name}: {e}")

    pinned_xformer = sum(
        1 for n in self.pinned_layer_names 
        if n.startswith(layer_prefix + ".")
    )
    print(
        f"[Vaayu-Pin] {len(self.pinned_layer_names)} layers pinned "
        f"({pinned_xformer} transformer + "
        f"{len(self.pinned_layer_names) - pinned_xformer} shared), "
        f"{used / 1e9:.2f} GB reserved"
    )
```

### Step 3: No Changes Needed in forward()

The existing `forward()` loop already handles pinned layers correctly:
- Line 935: `if layer_name in self.pinned_layer_names:` — skips disk load
- Line 1050: `if layer_name not in self.pinned_layer_names:` — skips free
- Seed prefetch (lines 924-929) already skips pinned layers

### Step 4: VRAM Math for 70B EXL3

```
Per transformer layer (Llama 70B, EXL3 4.5 bpw): ~380 MB
VRAM budget after 4GB reserve:                    ~12 GB
Shared layers (embed + norm + lm_head):           ~1 GB
Remaining for transformer layers:                 ~11 GB
Layers that fit: 11,000 / 380 ≈ 28 layers (out of 80)

→ First 28 layers pinned = 35% of model in VRAM permanently
```

## Key Files to Modify

| File | What Changes |
|------|-------------|
| `vaayu/vaayu_base.py` | Rewrite `_init_pinned_layers()`, add `max_pinned_layers` to `__init__()` |
| `vaayu/auto_model.py` | Pass `max_pinned_layers` through |
| `test_quick.py` | Add `--max_pinned_layers` CLI argument |

## Verification

1. **Pin count:** Log output should show ~28 transformer layers pinned + 3 shared
2. **Prefill benchmark:** Feed a 4,000-token prompt. `io_wait` should drop 30-50% vs shared-only pinning
3. **`[VAAYU-IO] token summary`:** `io_wait` should decrease since 35% of layers skip NVMe
4. **VRAM check:** `nvidia-smi` should show ~11-12 GB used; no OOM during generation
