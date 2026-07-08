# Fix ExpertRAMCache OOM for Large MoE Models (Qwen3-235B)

## Problem Summary

Vaayu crashes (OOM kill) and lags the entire system when running Qwen3-235B-A22B (128 experts, top-8, 94 layers) in EXL3 mode. The root cause is **two bugs** that combine to make the expert cache ignore its own RAM budget.

## The Numbers (Your Machine)

| Resource | Value |
|---|---|
| Total RAM | 33.4 GB |
| Pinned layers (embed, norm, lm_head, all attn) | 6.1 GB |
| Expert buffer pool (io_uring) | 0.5 GB |
| CUDA context + KV cache | ~1.5 GB |
| **Available for expert + non-expert cache** | **~25.3 GB** |
| Non-expert tensor cache (94 layers × 43.6 MB) | 4.0 GB |
| **Available for expert cache** | **~21 GB** |

What the current code tries to do:
- `budget_bytes` = 27.0 GB (correct, capped at 50% of total RAM)
- `max_per_layer` = **128** (the total number of experts — this is the bug)
- Worst case allocation = 128 × 94 × 88 MB = **1,034 GB** (the cache never checks the global budget)

Even the top-8 decode working set (8 × 94 × 88 MB = **64.6 GB**) exceeds your 25.3 GB of available RAM.

## Root Cause: Two Bugs

### Bug 1: `vaayu_base.py` — wrong `max_per_layer` for large MoE

[vaayu_base.py lines 528-530](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L528-L530):

```python
# EXL3: all experts per layer fit in budget → max_per_layer = moe_num_experts (8).
_max_per_layer = self.moe_num_experts if _is_exl3 else self.moe_top_k
```

The comment literally says `(8)` — this was written for Mixtral. For Qwen3, `moe_num_experts = 128`, so it allows 128 experts per layer.

### Bug 2: `expert_cache.py` — per-layer path ignores global budget

[expert_cache.py lines 56-66](file:///home/a/code/ai-ml/Vaayu/vaayu/persist/expert_cache.py#L56-L66):

```python
if self.max_per_layer > 0:
    lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
    if expert_idx in lc:
        return
    while len(lc) >= self.max_per_layer:
        _, evicted = lc.popitem(last=False)
        self.current_usage -= sum(t.nbytes for t in evicted.values())
    lc[expert_idx] = cloned
    self.current_usage += raw_size
    return  # ← Never checks budget_bytes!
```

The per-layer eviction only counts *within* a single layer. It never checks `budget_bytes`. So across 94 layers, memory grows unbounded.

---

## Proposed Changes

> [!IMPORTANT]
> The fix needs to make `ExpertRAMCache` **always** enforce the global budget as a hard ceiling, while *also* keeping the per-layer eviction policy as a performance optimization (it prevents one hot layer from evicting all others). Neither policy alone is sufficient.

### [Component 1] ExpertRAMCache — enforce budget as hard limit

#### [MODIFY] [expert_cache.py](file:///home/a/code/ai-ml/Vaayu/vaayu/persist/expert_cache.py)

Redesign the `put()` method so both constraints are enforced:

1. **Per-layer cap** (existing): evict oldest experts *within this layer* if the per-layer count is reached. This is a fairness mechanism — it prevents layer 0 from hogging the entire cache.
2. **Global budget cap** (missing): after the per-layer eviction, also check `current_usage + raw_size > budget_bytes`. If over budget, evict the globally-oldest expert across *all* layers (LRU across all layers). If still over budget after evicting everything possible, **drop the insert** (refuse to cache, which forces the caller to re-read from NVMe next time — slow but safe).

The new `put()` logic:

```python
def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None:
    cloned = {k: v.clone().detach() for k, v in tensors.items()}
    raw_size = sum(t.nbytes for t in cloned.values())
    with self.lock:
        if self.max_per_layer > 0:
            lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
            if expert_idx in lc:
                return

            # 1. Per-layer fairness: evict within this layer
            while len(lc) >= self.max_per_layer:
                _, evicted = lc.popitem(last=False)
                self.current_usage -= sum(t.nbytes for t in evicted.values())

            # 2. Global budget enforcement: evict globally-oldest across ALL layers
            while self.current_usage + raw_size > self.budget_bytes:
                evicted_any = self._evict_global_oldest(exclude_layer=layer_idx)
                if not evicted_any:
                    return  # Can't free anything — refuse insert (safe OOM guard)

            lc[expert_idx] = cloned
            self.current_usage += raw_size
            return

        # Global LRU path (max_per_layer == 0) — unchanged
        key = (layer_idx, expert_idx)
        if key in self.cache:
            return
        while self.current_usage + raw_size > self.budget_bytes and self.cache:
            _, evicted = self.cache.popitem(last=False)
            self.current_usage -= sum(t.nbytes for t in evicted.values())
        if self.current_usage + raw_size > self.budget_bytes:
            return
        self.cache[key] = cloned
        self.current_usage += raw_size
```

New private method `_evict_global_oldest()`:
```python
def _evict_global_oldest(self, exclude_layer: int = -1) -> bool:
    """Evict the single oldest expert across all layers. Returns False if nothing to evict."""
    # Find layer with the oldest insertion (smallest first OrderedDict entry)
    # Skip exclude_layer to avoid evicting from the layer we're inserting into
    oldest_layer = None
    for lid, lc in self._layer_caches.items():
        if lid == exclude_layer or not lc:
            continue
        if oldest_layer is None:
            oldest_layer = lid
            # Note: could track timestamps for true LRU. For now, round-robin
            # across layers works well because each layer is accessed once per token.
    if oldest_layer is None:
        return False
    lc = self._layer_caches[oldest_layer]
    _, evicted = lc.popitem(last=False)
    self.current_usage -= sum(t.nbytes for t in evicted.values())
    return True
```

---

### [Component 2] Budget calculation — derive from actual available RAM

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L509-L534)

The budget calculation has two problems:
1. `_max_per_layer = self.moe_num_experts` assumes all experts are cacheable — hardcoded for 8-expert models.
2. The budget doesn't subtract the non-expert tensor cache (~4 GB for Qwen3) or other fixed allocations.

**New logic:**

```python
_is_exl3 = isinstance(self.loader_strategy, EXL3LoaderStrategy)
_mem = _psutil.virtual_memory()

_expert_mb = 88 if _is_exl3 else 352

# Fixed overhead: pinned layers + expert buffer pool + CUDA + non-expert cache
# Non-expert cache: num_moe_layers × ~44 MB (Qwen3) or ~80 MB (Mixtral)
_num_moe_layers = len(self.layers) - 3  # minus embed, norm, lm_head
_non_expert_estimate = _num_moe_layers * 44 * 1024 * 1024  # conservative
_fixed_overhead = (
    8 * 1024**3  # pinned layers + CUDA context + io_uring buffers + OS
    + _non_expert_estimate
)

# Expert budget = available RAM minus fixed overhead
_expert_budget = max(
    1 * 1024**3,  # floor: at least 1 GB so single-layer decode always works
    int(_mem.available - _fixed_overhead),
)

# max_per_layer: how many experts fit in budget if spread across all layers
# At minimum = top_k (must cache at least the active set for current token)
# At maximum = num_experts (but only if RAM allows)
_per_layer_budget = _expert_budget // _num_moe_layers
_max_per_layer = max(
    self.moe_top_k,
    min(self.moe_num_experts, _per_layer_budget // (_expert_mb * 1024 * 1024)),
)
```

This is **model-agnostic**: it works for Mixtral (8 experts), Qwen3 (128 experts), DeepSeek (256 experts), and any future architecture. It derives `max_per_layer` from actual available memory instead of hardcoding a number.

**On your machine (33.4 GB total, ~25 GB available):**
- `_fixed_overhead` ≈ 8 GB + 4 GB = 12 GB
- `_expert_budget` ≈ 25 - 12 = **13 GB**
- `_per_layer_budget` = 13 GB / 91 layers ≈ **146 MB/layer**
- `_max_per_layer` = max(8, min(128, 146 / 88)) = max(8, 1) = **8** (the top_k)

This means: 8 experts cached per layer × 91 layers × 88 MB = **64 GB theoretical**, but hard-capped at 13 GB by the budget enforcement in Component 1. In practice, the global LRU eviction will keep only the most recently used ~148 experts total in RAM (13 GB / 88 MB ≈ 148 experts). That's roughly 1.5 full layers of experts — more than enough for speculative prefetch.

---

### [Component 3] Non-expert tensor cache — add budget cap

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L506-L508)

The `_non_expert_tensor_cache` is currently an unbounded `dict`. For Qwen3 with 94 layers at ~43.6 MB each, it grows to **4 GB** — more than the available headroom.

Convert it to a simple LRU dict with a size cap. Since the forward loop processes layers sequentially (0→93→0→93…), an LRU naturally keeps the "next" layers warm. A budget of ~2 GB (half the Qwen3 worst-case) is sufficient — the forward loop only needs the current layer's non-expert tensors at any moment, so even a 500 MB cap would work functionally; the rest is prefetch benefit.

```python
# Cap at 2 GB or available/8, whichever is smaller
_ne_budget = min(2 * 1024**3, int(_mem.available / 8))
self._non_expert_tensor_cache = LRUTensorCache(_ne_budget)
```

This can be a simple wrapper using `OrderedDict` with byte tracking (same pattern as `ExpertRAMCache`'s global path).

---

## Open Questions

> [!IMPORTANT]
> **Question 1:** The `_evict_global_oldest` implementation above uses a simple "find any non-empty layer and evict its oldest" strategy. A more sophisticated approach would track a global insertion order across all layers (a single `OrderedDict[(layer, expert)]` alongside the per-layer dicts). This is more complex but gives true LRU. Do you want true global LRU, or is the simpler round-robin approach acceptable for v1?

> [!IMPORTANT]
> **Question 2:** The non-expert tensor cache fix (Component 3) introduces a new `LRUTensorCache` class. Should I add this as a new file, or fold it into `expert_cache.py` as a second class? It's ~30 lines.

## Verification Plan

### Automated Tests
```bash
cd /home/a/code/ai-ml/Vaayu && source venv/bin/activate && python -m pytest tests/test_expert_cache.py -v
cd /home/a/code/ai-ml/Vaayu && source venv/bin/activate && python -m pytest tests/test_moe_patch.py -v
```

### Manual Verification
1. Run `python3 test_generate.py > run.log 2>&1` and confirm it completes without OOM kill.
2. Check `run.log` for the new budget log line — verify `max_per_layer` is computed correctly for Qwen3 (should be 8, not 128).
3. Monitor `htop` during inference — RSS should stay under 25 GB.
4. Check `expert_cache.stats()` output in log — verify `current_usage` stays within `budget_bytes`.
