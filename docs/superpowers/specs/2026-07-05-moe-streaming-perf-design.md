# MoE Streaming Performance: Fix A + B + D Design Spec

**Date:** 2026-07-05
**Model target:** Mixtral-8x7B-Instruct-v0.1 (BF16, ~87 GB)
**Hardware:** 16 GB VRAM, 33 GB RAM, NVMe (~2.2–6.4 GB/s), PCIe Gen5 x8

## Goal

Reduce MoE decode latency from ~15s/token (pre-fix baseline) toward <0.5s/token by fixing three compounding bottlenecks. Fix C (CUDA stream pipelining) is excluded — saves ~225ms at PCIe Gen5, not worth the complexity until telemetry confirms it matters.

## Current State

Fixes from 04-07 reports already applied (commits `214b03c`, `c72d6fb`):
- GPU compute device for standard BF16 models ✓
- Pinned layer bypass in `sparse_forward` ✓
- Speculative prefetch index fix (`_next_layer_idx` lookup) ✓

Remaining bottlenecks:
1. Init-order bug → only 3 layers pinned → 29 layers do inject/free every token
2. Expert cache 46 slots < 64-expert working set → 87% miss rate → ~8s NVMe reads/token
3. Per-layer prefetch fires too late to hide NVMe latency

## Revised Per-Token Estimates (PCIe Gen5)

| Component | Current (post-214b03c) | After Fix A | After A+B | After A+B+D |
|---|---|---|---|---|
| `gpu_load` (inject/free 29 layers) | ~300ms | **~0** | ~0 | ~0 |
| NVMe expert reads (87% miss) | ~8s | ~8s | **~0.2s** | **~0s** |
| Expert H2D + GPU matmul | ~0.55s | ~0.55s | ~0.55s | ~0.55s |
| **Total** | **~9s** | **~8.8s** | **~0.8s** | **~0.55s** |

---

## Fix A — Init Order + Non-Expert Pinning

### Problem

`_init_pinned_layers()` runs at line 476. `_expert_offset_index` is built at line 504. Phase 2 pinning checks `getattr(self, '_expert_offset_index', None) is not None` — always `None` at pin time — so loads full 2.9 GB layers instead of 80 MB non-expert only. Budget exhausts after 3 layers.

### Critical Bug in Original Plan

The original implementation plan adds non-expert-pinned layers to `pinned_layer_names`. In `sparse_forward`:

```python
if layer_name in vaayu_ref.pinned_layer_names:
    return original_forward(hidden_states, **kwargs)  # CRASH: experts on meta device
```

`original_forward` is the pre-patch MoE forward — calls all 8 experts using weight tensors in the module. After non-expert-only pinning, expert tensors are on **meta device**. This crashes every forward pass.

### Fix

Introduce `non_expert_pinned_layer_names: set` alongside `pinned_layer_names`.

- `pinned_layer_names`: fully-pinned layers (all weights in VRAM) → `original_forward` bypass safe
- `non_expert_pinned_layer_names`: non-expert weights in VRAM, experts still on meta → `sparse_forward` must still run

**`vaayu_base.py` changes:**

1. Add `self.non_expert_pinned_layer_names: set = set()` at line ~464 (alongside `pinned_layer_names` init).

2. Before `_init_pinned_layers()`, insert the ExpertOffsetIndex initialization:
   ```python
   self._expert_offset_index = None
   if self.running_device.startswith("cuda") and self.is_moe:
       from .persist.expert_offset_index import ExpertOffsetIndex
       self._expert_offset_index = ExpertOffsetIndex(str(self.checkpoint_path))
       self._expert_offset_index.build_or_load()
   ```
   Then remove `self._expert_offset_index = None` from the existing attribute declarations block at line ~493 — it's now initialized above. The rest of the declarations (`self._expert_history`, `self._expert_cache = None`, `self._moe_blocks`, `self._non_expert_tensor_cache`) and the expert cache creation + `patch_moe_blocks` stay in their current positions after the RAM cache init. Leaving `self._expert_offset_index = None` in the original location would silently overwrite the just-built index — this removal is critical.

3. In `_init_pinned_layers` Phase 2: when `_load_non_expert_strategy` is used, add to `non_expert_pinned_layer_names` instead of `pinned_layer_names`:
   ```python
   if self.is_moe and getattr(self, "_expert_offset_index", None) is not None:
       tensors = self._load_non_expert_strategy(layer_name)
       # ... inject_to_gpu ...
       self.non_expert_pinned_layer_names.add(layer_name)  # NOT pinned_layer_names
   else:
       tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
       # ... inject_to_gpu ...
       self.pinned_layer_names.add(layer_name)
   ```

4. In forward loop (line ~1670), extend the pinned-layer fast-path check:
   ```python
   if layer_name in self.pinned_layer_names or layer_name in self.non_expert_pinned_layer_names:
       _lt_cache_hit = True
       moved_layers = []
       # ... existing prefetch-next-layer logic ...
   ```

**`moe_patch.py`**: no change. After Fix A, `pinned_layer_names` contains only embed/norm/lm_head (shared layers, no MoE block). The `original_forward` bypass in `_make_fused_sparse_forward` never fires for transformer layers. Correct by construction.

### Expected Log

```
[Vaayu-Pin] pinned model.layers.0: 0.08 GB  (cumulative N GB)
...
[Vaayu-Pin] pinned model.layers.31: 0.08 GB  (cumulative ~3.1 GB)
```

`gpu_load` in token summaries drops from ~300ms to **~0ms**.

---

## Fix B — Expert Cache Budget

### Problem

Expert cache budget = `int(total_RAM * 0.50)` = 16.7 GB = 46 expert slots. Per-token working set = 2 experts × 32 layers = 64 unique experts = 22.5 GB. Cache thrashes: 87% miss rate, ~8s NVMe reads per token.

RAM layer cache gets `auto_detect_cache_budget()` ≈ 20 GB but `_prepopulate_ram_cache` is skipped for MoE. 20 GB allocated, never used.

### Fix

**`vaayu_base.py`** — two changes in the `if device.startswith("cuda"):` block:

1. Zero the RAM layer cache budget for MoE (line ~482):
   ```python
   _cache_budget = 0 if self.is_moe else auto_detect_cache_budget()
   self.ram_cache = LayerRAMCache(_cache_budget)
   ```

2. Increase expert cache budget (line ~512). Replace `int(_mem.total * 0.50)` with:
   ```python
   _os_reserve = 6 * 1024**3
   _expert_budget = max(int(_mem.total * 0.50), int(_mem.total - _os_reserve))
   ```
   On 33 GB machine: `max(16.5 GB, 27 GB)` = 27 GB cap. Budget is a cap, not pre-allocation. Actual usage stabilizes at ~22.5 GB (64 experts × 352 MB). Gives ~77 expert slots — covers full working set with margin.

**Sticky eviction** (`expert_cache.py` changes from original plan): deferred. Budget increase alone takes hit rate from 13% → 85–95%. Sticky is a follow-up optimization.

### Expected Result

Expert cache hit rate: 13% → **85–95%** on decode tokens 2+. `compute` time: ~8s → **~0.2–0.5s**.

---

## Fix D — Pre-Token Batch Prefetch

### Problem

Current per-layer prefetch (lines 1855–1867): after layer L completes, submits task to load experts for layer L+1. Head-start ≈ 0 — L+1 starts immediately. Even with correct predictions, the prefetch races and loses.

### Design

At the START of each forward pass (token N), submit prefetch tasks for ALL 32 layers using `_expert_history` from token N-1. Later layers (10–31) get the compute time of all earlier layers as prefetch window — several seconds of head-start at current speeds.

**`vaayu_base.py`** — two changes:

1. Add `_prefetch_all_predicted_experts` method after `_prefetch_experts` (~line 1214):
   ```python
   def _prefetch_all_predicted_experts(self, executor) -> None:
       if not self._expert_history or self._expert_cache is None:
           return
       for layer_idx, expert_list in self._expert_history.items():
           uncached = [e for e in expert_list
                       if not self._expert_cache.has(layer_idx, e)]
           if uncached:
               executor.submit(self._prefetch_experts, layer_idx, uncached)
   ```

2. Call at start of forward pass — first statement inside `with torch.inference_mode(), ThreadPoolExecutor() as executor:` (line 1643):
   ```python
   if self.is_moe and self._token_count > 0:
       self._prefetch_all_predicted_experts(executor)
   ```

**Existing per-layer prefetch** (lines 1855–1867): keep as-is. Handles experts that differ from prediction (routing isn't perfectly stable). The `has()` check in `_prefetch_experts` prevents duplicate loads.

**`_expert_history` safety**: history no longer cleared per token (fixed in `214b03c`). Persists across tokens — safe to use as N-1 prediction.

### Expected Result

Decode tokens 2+: later layers (10–31) find experts pre-loaded. Combined with Fix B's larger cache, miss rate on later layers approaches **~0**.

---

## Verification Plan

Apply in order. Run `python -u test_generate.py > run.log 2>&1` after each.

| Step | Fix | Success Criteria |
|---|---|---|
| 1 | Fix A | Log: 32 transformer layers pinned at ~80 MB each; `gpu_load` ≈ 0s in token summaries |
| 2 | Fix B | Expert cache `hit_rate` jumps from 13% → 85%+; `compute` < 1s/token |
| 3 | Fix D | Later-layer experts show 0 misses; total < 0.5s/token |

**Correctness check after all fixes:**
- First 5 generated tokens match a reference run (hidden states not corrupted by non-expert pinning)
- Output is valid Python (no garbage — `AsyncFileWatcher` present in output for the standard test prompt)

## Files Changed

| File | Fix | Change |
|---|---|---|
| `vaayu/vaayu_base.py` | A | Add `non_expert_pinned_layer_names`; move `_expert_offset_index` build; update Phase 2 pinning; update forward loop check |
| `vaayu/vaayu_base.py` | B | Zero RAM cache for MoE; increase expert budget formula |
| `vaayu/vaayu_base.py` | D | Add `_prefetch_all_predicted_experts`; call at forward start |
| `vaayu/persist/expert_cache.py` | — | No changes (sticky eviction deferred) |
| `vaayu/moe_patch.py` | — | No changes |
