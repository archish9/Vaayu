# Vaayu: Application-Level RAM Layer Cache

## Goal

Add a **RAM-resident layer cache** so that frequently-reused model layers are served from system memory instead of being re-read from the NVMe SSD on every token. This reduces per-token `io_wait` time without changing the GPU transfer path or the `O_DIRECT` io_uring architecture.

## Background & Problem

Vaayu is an LLM inference framework that runs models too large for VRAM by **streaming layers one-at-a-time** from NVMe → CPU pinned RAM → GPU VRAM. For every single token generated, **all 83 layers** of a Llama-70B model are read from disk, transferred to GPU, computed, and then evicted from VRAM.

### Current Performance (from `o_direct_improvements_report.md`)

| Metric | Value |
|---|---|
| Per-token time | **23.2s** |
| `io_wait` (blocked waiting for NVMe) | **15.8s** |
| `gpu_load` (CPU→GPU PCIe transfer) | **6.9s** |
| `compute` (actual GPU math) | **~0.1s** |
| `other` (overhead) | **0.5s** |
| NVMe throughput | ~5.8 GB/s |
| GPU PCIe bandwidth | ~19 GB/s (Gen 4 x8) |
| Layer size | ~1.6 GB each |
| Total model data per token | ~132 GB (83 layers) |

**The NVMe is the bottleneck.** The GPU finishes transferring a layer in ~0.08s but must wait ~0.19s for the next layer to arrive from disk. This creates 15.8s of `io_wait` per token.

### Why a RAM cache helps

With 32 GB total RAM, after OS + PyTorch + pinned buffers (~8 GB), there is ~24 GB available. That's enough to cache **~15 layers** (~1.6 GB each) in regular system RAM.

If 15 layers are served from RAM (effectively instant) instead of NVMe (0.28s each):
- Layers from NVMe: 68 (down from 83)
- Disk time: 68 × 0.28 = **19.0s** (down from 23.2s)
- Projected `io_wait`: **~12s** (down from 15.8s)
- **Projected per-token time: ~19s** (saving ~4 seconds per token)

---

## System Architecture Context

> [!IMPORTANT]
> The coding agent MUST understand these three systems and how they interact before writing any code.

### 1. The I/O Pipeline (`safetensor_model_persister.py`)

File: [safetensor_model_persister.py](file:///c:/laragon/www/Vaayu/vaayu/persist/safetensor_model_persister.py)

This file handles raw bytes from disk → CPU pinned RAM. It has two paths:

**Path A — Rust io_uring (preferred, when `vaayu_rust` is compiled):**
- Opens the `.safetensors` file with `O_DIRECT` (bypasses Linux page cache entirely)
- Uses `io_uring` to submit async read SQEs directly into a PyTorch pinned-memory buffer
- Achieves ~5.8 GB/s throughput
- The `O_DIRECT` flag means data NEVER enters the Linux page cache — even if you have tons of RAM, the OS cannot cache these reads

**Path B — Python preadv (fallback, when `vaayu_rust` is NOT compiled):**
- Uses `os.preadv` with a 16-thread `ThreadPoolExecutor`
- Reads through the normal filesystem layer (data enters the Linux page cache)
- Achieves ~4.4 GB/s throughput

**Key data structures:**
- `_PINNED_BUFS = [None, None]` — Double-buffer: two pre-allocated PyTorch `pin_memory=True` tensors (~1.7 GB each). One is being filled from disk while the other is being sent to GPU.
- `_BUF_IDX` — Toggles 0/1 to swap between the two buffers each layer load.

**The `_load_pinned()` function (line 63)** is the hot path. It:
1. Parses the safetensors header to get tensor metadata (offsets, shapes, dtypes)
2. Acquires one of the two pinned buffers
3. Reads the file data into the buffer (via io_uring or preadv)
4. Reconstructs tensors as **zero-copy views** into the buffer (no extra allocation)

### 2. The Loader Strategy (`loader_strategy.py`)

File: [loader_strategy.py](file:///c:/laragon/www/Vaayu/vaayu/persist/loader_strategy.py)

This is the abstraction layer between the forward loop and the I/O code. There are two strategies:

- **`StandardSafetensorsStrategy`** — for normal BF16/FP16 models. Its `load_from_disk()` calls `utils.load_layer()` → `ModelPersister.load_model()` → `_load_pinned()`.
- **`EXL3LoaderStrategy`** — for ExLlamaV3 4-bit quantized models. Same disk read path, but applies dtype corrections for the C++ kernels.

Both strategies' `load_from_disk()` ultimately call the same `_load_pinned()` function.

### 3. The Forward Loop (`vaayu_base.py`)

File: [vaayu_base.py](file:///c:/laragon/www/Vaayu/vaayu/vaayu_base.py)

The `forward()` method (line 777) orchestrates the per-token execution:

```
for each layer in [embed, layer.0, layer.1, ..., layer.79, norm, lm_head]:
    1. If layer is PINNED in VRAM (embed/norm/lm_head): skip I/O entirely
    2. If PREFETCHING:
       a. Wait for the background future (disk read) to complete       ← io_wait
       b. Submit the NEXT layer's disk read to the ThreadPoolExecutor
       c. Transfer current layer's tensors from CPU pinned RAM → GPU   ← gpu_load
    3. Run layer.forward() on GPU                                      ← compute
    4. Free layer from VRAM (move to "meta" device)
```

**Key code (lines 850–976):**
- `ThreadPoolExecutor` manages a single background thread for prefetching
- `self._load_from_strategy(layer_name)` is submitted to the executor (line 857, 872, 893)
- `future.result()` blocks until the disk read completes (line 880)
- `self.move_layer_to_device(state_dict)` transfers tensors CPU→GPU (line 902)
- `self.loader_strategy.free_from_gpu(...)` moves tensors to "meta" device to reclaim VRAM (line 974)

**Existing VRAM pinning (lines 497–540):** Embed, norm, and lm_head layers are already permanently pinned in GPU VRAM at init time. These ~3 layers are never read from disk after initialization. The RAM cache is for the **80 decoder layers** that are too large to keep in VRAM.

---

## Proposed Changes

### Overview

Add a new class `LayerRAMCache` that stores decoded tensor dictionaries in regular (non-pinned) system RAM. Integrate it into the `_load_from_strategy` path so that cache hits skip the NVMe read entirely.

```
┌─────────────────────────────────────────────────────┐
│  forward() loop (vaayu_base.py)                     │
│                                                     │
│  _load_from_strategy(layer_name)                    │
│       │                                             │
│       ▼                                             │
│  ┌─────────────────────┐                            │
│  │  LayerRAMCache      │◄── NEW                     │
│  │  (regular RAM, LRU) │                            │
│  └────────┬────────────┘                            │
│      hit? │                                         │
│     ┌─────┴─────┐                                   │
│     │ YES       │ NO                                │
│     ▼           ▼                                   │
│  Return from  load_from_disk()                      │
│  RAM cache    (io_uring/preadv → pinned buffer)     │
│               then STORE copy in RAM cache          │
└─────────────────────────────────────────────────────┘
```

---

### Component 1: `LayerRAMCache` Class

#### [NEW] `vaayu/persist/ram_cache.py`

Create a new file implementing the LRU RAM cache.

```python
"""
Application-level RAM cache for decoded layer tensor dictionaries.

Keeps recently-loaded layers in regular (non-pinned) system RAM so that
subsequent tokens can skip the NVMe read entirely for cached layers.

Design constraints:
  - Budget is set once at init (bytes). Default: auto-detect free RAM minus safety margin.
  - Eviction policy: LRU (least recently used).
  - Stored tensors are CLONES in regular RAM (NOT pinned memory).
    The pinned double-buffer is still used for the CPU→GPU transfer;
    on a cache hit, data is copied from RAM cache → pinned buffer → GPU.
  - Thread-safe: forward() reads cache from main thread while ThreadPoolExecutor
    writes to it from the background prefetch thread.
"""
```

**Class interface:**

```python
class LayerRAMCache:
    def __init__(self, budget_bytes: int):
        """
        Args:
            budget_bytes: Maximum total bytes of tensor data to keep in cache.
                          Recommended: (total_ram - 8GB) as bytes.
                          On a 32GB system: ~24 GB = 25_769_803_776 bytes.
        """

    def get(self, layer_name: str) -> dict | None:
        """
        Returns cached {param_name: tensor} dict if present, else None.
        Tensors are in regular RAM (not pinned). Caller must copy to pinned buffer.
        Updates LRU ordering on hit.
        """

    def put(self, layer_name: str, tensors: dict) -> None:
        """
        Store a CLONE of the tensor dict in the cache.
        If adding this entry would exceed budget, evict LRU entries until it fits.
        The clone is intentional: the original tensors are zero-copy views into
        the pinned double-buffer, which gets overwritten on the next layer load.
        """

    def current_usage(self) -> int:
        """Return current cache size in bytes."""

    def stats(self) -> str:
        """Return human-readable hit/miss/eviction stats string."""
```

**Implementation details:**

1. **Use `OrderedDict`** for O(1) LRU operations. `get()` moves the entry to the end. `put()` appends to the end and pops from the front when over budget.

2. **Thread safety:** Use a `threading.Lock`. The lock is only held during the dict lookup/insert (microseconds), not during I/O or GPU transfer.

3. **Clone strategy:** When storing, clone each tensor to regular (non-pinned) RAM:
   ```python
   cached = {k: v.clone().detach() for k, v in tensors.items()}
   ```
   This is necessary because the original tensors are **views into the pinned double-buffer** (`_PINNED_BUFS`), which gets overwritten on the very next layer load. Without cloning, the cache would contain stale/corrupted data.

4. **Size tracking:** Track `sum(t.nbytes for t in cached.values())` per entry. On `put()`, evict LRU entries until `current_usage + new_entry_size <= budget_bytes`.

5. **Auto-budget detection:** Provide a helper function:
   ```python
   def auto_detect_cache_budget() -> int:
       """
       Returns recommended cache budget in bytes.
       Uses psutil if available, else falls back to /proc/meminfo on Linux.
       Reserves 8 GB for OS + PyTorch + pinned buffers + safety.
       """
   ```
   On a 32 GB system, this should return ~24 GB.

> [!WARNING]  
> The cloning step adds ~0.3–0.5s of CPU memcpy per layer (1.6 GB at ~4 GB/s RAM bandwidth). This happens ONCE per layer on the first token (cold cache). On subsequent tokens, cache hits avoid the 0.28s NVMe read entirely, so the net benefit is still large. The clone cost is paid during the background prefetch thread, overlapping with GPU compute.

---

### Component 2: Integration into Loader Strategy

#### [MODIFY] `vaayu/persist/loader_strategy.py`

The `LoaderStrategy` base class needs a cache-aware wrapper. The cleanest approach: add cache logic to `StandardSafetensorsStrategy.load_from_disk()` and `EXL3LoaderStrategy.load_from_disk()`.

**Changes to `StandardSafetensorsStrategy` (line 93):**

```python
class StandardSafetensorsStrategy(LoaderStrategy):
    def __init__(self, vaayu_base_ref, ram_cache=None):  # ← add ram_cache param
        self._base = vaayu_base_ref
        self._cache = ram_cache  # ← store reference

    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        # 1. Check RAM cache first
        if self._cache is not None:
            cached = self._cache.get(layer_name)
            if cached is not None:
                return cached

        # 2. Cache miss → read from NVMe (existing path)
        from ..utils import load_layer
        tensors = load_layer(checkpoint_path, layer_name)

        # 3. Store in RAM cache for next token
        if self._cache is not None:
            self._cache.put(layer_name, tensors)

        return tensors
```

Apply the same pattern to `EXL3LoaderStrategy.load_from_disk()` (line 259).

**Changes to `build_loader_strategy()` (line 458):**

```python
def build_loader_strategy(checkpoint_path: str, vaayu_base_ref, ram_cache=None) -> LoaderStrategy:
    if _is_exl3_checkpoint(checkpoint_path):
        strategy = EXL3LoaderStrategy()
        strategy._cache = ram_cache  # attach cache
        return strategy
    else:
        return StandardSafetensorsStrategy(vaayu_base_ref, ram_cache=ram_cache)
```

---

### Component 3: Initialization in VaayuBaseModel

#### [MODIFY] `vaayu/vaayu_base.py`

**Changes to `__init__()` (around line 240):**

Add cache initialization before the loader strategy is built:

```python
# ── RAM layer cache ──────────────────────────────────────────────
from .persist.ram_cache import LayerRAMCache, auto_detect_cache_budget

if device.startswith("cuda"):
    cache_budget = auto_detect_cache_budget()
    self.ram_cache = LayerRAMCache(cache_budget)
    print(f"[Vaayu-Cache] RAM layer cache: {cache_budget / 1e9:.1f} GB budget")
else:
    self.ram_cache = None

# Build loader strategy (auto-detects EXL3 vs standard safetensors)
self.loader_strategy: LoaderStrategy = build_loader_strategy(
    str(self.checkpoint_path), self, ram_cache=self.ram_cache  # ← pass cache
)
```

**Changes to `forward()` — add cache stats to the token summary (line 997):**

```python
cache_info = ""
if self.ram_cache is not None:
    cache_info = f"  cache={self.ram_cache.stats()}"
print(f"[VAAYU-IO] token summary: total={_total:.1f}s  io_wait={_io_wait_s:.1f}s  "
      f"gpu_load={_gpu_load_s:.1f}s  compute={_compute_s:.1f}s  "
      f"other={_total - _io_wait_s - _gpu_load_s - _compute_s:.1f}s{cache_info}")
```

---

### Component 4: Cache Hit Data Flow

When the cache has a hit, the returned tensors are in **regular (non-pinned) RAM**. The existing `move_layer_to_device()` in `vaayu_base.py` (line 597) handles this correctly already — it calls `.to(device, dtype, non_blocking=True)` on each tensor, which works for both pinned and non-pinned source tensors. The only difference is:

- **Pinned tensor → GPU:** Uses DMA, ~19 GB/s (0.08s per layer)
- **Non-pinned tensor → GPU:** Uses staged copy, ~8–12 GB/s (~0.13–0.20s per layer)

> [!NOTE]  
> Cache hits trade slightly slower GPU transfer (~0.13s vs ~0.08s) for completely skipping the NVMe read (~0.28s). Net savings per cached layer: **~0.15–0.20s**.

**Optional optimization:** On cache hit, explicitly copy the cached tensors into the pinned double-buffer first, THEN transfer to GPU. This restores the full DMA speed. This is a follow-up optimization and NOT required for the initial implementation.

---

## Files Changed Summary

| File | Change | Type |
|---|---|---|
| `vaayu/persist/ram_cache.py` | New file: `LayerRAMCache` class + `auto_detect_cache_budget()` | NEW |
| `vaayu/persist/loader_strategy.py` | Add `ram_cache` param to strategies + `build_loader_strategy()` | MODIFY |
| `vaayu/vaayu_base.py` | Init cache in `__init__()`, add stats to `forward()` summary | MODIFY |

---

## What NOT to Change

> [!CAUTION]
> These systems must remain untouched:

1. **`safetensor_model_persister.py`** — The pinned-buffer / io_uring / preadv I/O layer. The cache sits ABOVE this layer (in the strategy), not inside it.
2. **`vaayu_rust/src/lib.rs`** — The Rust io_uring module. No changes needed.
3. **The `O_DIRECT` flag** — Must remain. The cache is an application-level cache that works alongside `O_DIRECT`, not a replacement for the Linux page cache.
4. **The VRAM pinning system** (`_init_pinned_layers`) — This pins embed/norm/lm_head in GPU VRAM permanently. The RAM cache is for the 80 decoder layers that cycle through VRAM.
5. **The double-buffer system** (`_PINNED_BUFS`) — Still needed for cache misses and for the CPU→GPU DMA transfer path.
6. **The prefetch pipeline** (ThreadPoolExecutor in `forward()`) — The cache check happens inside `load_from_disk()`, which is called from the executor thread. No changes to the executor/future logic needed.

---

## Expected Performance

### Token 1 (Cold Cache)

All 83 layers read from NVMe. As each layer is loaded, a clone is stored in the RAM cache until it fills up (~15 layers cached).

| Metric | Before | After (Token 1) |
|---|---|---|
| Layers from NVMe | 83 | 83 |
| Layers from RAM cache | 0 | 0 |
| Clone overhead | 0 | ~5s (15 × 0.35s clone, overlapped with GPU) |
| Per-token time | ~23.2s | ~23.2s (no improvement on first token) |

### Token 2+ (Warm Cache)

15 layers served from RAM cache, 68 from NVMe.

| Metric | Before | After (Token 2+) |
|---|---|---|
| Layers from NVMe | 83 | 68 |
| Layers from RAM cache | 0 | 15 |
| Disk time | 23.2s | ~19.0s |
| Projected `io_wait` | 15.8s | ~12s |
| **Per-token time** | **23.2s** | **~19s** |

### With 64 GB RAM (Future Upgrade)

~35 layers cacheable → only 48 from NVMe.

| Metric | Value |
|---|---|
| Layers from NVMe | 48 |
| Layers from RAM cache | 35 |
| Disk time | ~13.4s |
| **Per-token time** | **~14s** |

---

## Verification Plan

### Automated Tests

1. **Unit test for `LayerRAMCache`:**
   - Test LRU eviction order
   - Test budget enforcement
   - Test thread safety with concurrent get/put
   - Test that `get()` returns cloned data (modify original, verify cache is unaffected)

2. **Integration test:** Run Vaayu with `max_new_tokens=3` on any small model and verify:
   - Token 1 log shows `cache=0/X hits`
   - Token 2 log shows `cache=Y/X hits` where Y > 0
   - Token 2 `io_wait` < Token 1 `io_wait`

### Manual Verification

Run on the actual Linux machine with Llama-70B:
```bash
python3 test_quick.py
```

Check the `[VAAYU-IO] token summary` lines:
- Token 1: `cache=0hit/83miss`
- Token 2+: `cache=~15hit/~68miss`, `io_wait` should drop by ~4s
