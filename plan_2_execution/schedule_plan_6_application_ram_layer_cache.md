# Schedule Plan 6: Application-Level RAM Layer Cache

**Phase:** 2 (Core Performance Architecture)
**Priority:** Highest
**Effort:** Medium (1-2 days)
**Risk:** Medium (RAM limit tuning required)
**Impact:** Slices ~4 seconds off the per-token `io_wait` time by serving 15+ layers directly from RAM.

## Objective
Vaayu is currently bottlenecked by reading 83 layers from NVMe on every single token. With 32GB RAM, there's ~24GB of unused system RAM available during generation. This plan introduces an LRU cache (`LayerRAMCache`) to hold fully decoded PyTorch tensor dictionaries in system RAM. When a layer hits the cache, we copy it directly into the pinned buffer (avoiding NVMe entirely).

## Context for the Coding Agent
The cache acts as a middleware in `loader_strategy.py`. `StandardSafetensorsStrategy` and `EXL3LoaderStrategy` will check the cache before calling `load_layer()`. On a hit, they return the cached tensors. On a miss, they load from NVMe, then put a *clone* of the tensors into the RAM cache (evicting the oldest if over budget). The tensors *must* be cloned because the original tensors are views into the volatile pinned double-buffer.

## Implementation Steps

### Step 1: Create `vaayu/persist/ram_cache.py`

```python
import threading
from collections import OrderedDict
import psutil

def auto_detect_cache_budget() -> int:
    """Returns recommended cache budget in bytes (total free RAM minus 8GB safety)."""
    mem = psutil.virtual_memory()
    return max(0, mem.available - (8 * 1024**3))

class LayerRAMCache:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.cache = OrderedDict()
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, layer_name: str) -> dict:
        with self.lock:
            if layer_name in self.cache:
                self.cache.move_to_end(layer_name)
                self.hits += 1
                return self.cache[layer_name]
            self.misses += 1
            return None

    def put(self, layer_name: str, tensors: dict) -> None:
        with self.lock:
            if layer_name in self.cache:
                return

            # Must clone because tensors are views of the pinned buffer which will be overwritten
            cached_tensors = {k: v.clone().detach() for k, v in tensors.items()}
            entry_size = sum(t.nbytes for t in cached_tensors.values())

            # Evict LRU until it fits
            while self.current_usage + entry_size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= sum(t.nbytes for t in evicted.values())

            if entry_size <= self.budget_bytes:
                self.cache[layer_name] = cached_tensors
                self.current_usage += entry_size

    def stats(self) -> str:
        return f"{self.hits}hit/{self.misses}miss"
```

### Step 2: Integrate with Loader Strategies

**File to modify:** `vaayu/persist/loader_strategy.py`

Modify the `LoaderStrategy` base class or the constructors of `StandardSafetensorsStrategy` and `EXL3LoaderStrategy` to accept `ram_cache`.

```python
# In StandardSafetensorsStrategy.load_from_disk
def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
    if hasattr(self, '_cache') and self._cache is not None:
        cached = self._cache.get(layer_name)
        if cached is not None:
            return cached

    from ..utils import load_layer
    tensors = load_layer(checkpoint_path, layer_name)

    if hasattr(self, '_cache') and self._cache is not None:
        self._cache.put(layer_name, tensors)

    return tensors
```
*Do the exact same check/put block in `EXL3LoaderStrategy.load_from_disk`.*

Modify `build_loader_strategy` to accept the cache:
```python
def build_loader_strategy(checkpoint_path: str, vaayu_base_ref, ram_cache=None):
    if _is_exl3_checkpoint(checkpoint_path):
        strategy = EXL3LoaderStrategy()
        strategy._cache = ram_cache
        return strategy
    else:
        strategy = StandardSafetensorsStrategy(vaayu_base_ref)
        strategy._cache = ram_cache
        return strategy
```

### Step 3: Initialize in `vaayu_base.py`

**File to modify:** `vaayu/vaayu_base.py`

> [!IMPORTANT]
> Call `auto_detect_cache_budget()` **after** `_init_pinned_layers()` completes, not at the top of `__init__`. The pinned embed/norm/lm_head layers consume ~3.4 GB of RAM during init. If the budget is measured before that allocation, the cache will be over-budgeted by ~3.4 GB and may cause OOM.

Place the cache init after the `_init_pinned_layers()` call:

```python
# After _init_pinned_layers() call:
from .persist.ram_cache import LayerRAMCache, auto_detect_cache_budget

if device.startswith("cuda"):
    cache_budget = auto_detect_cache_budget()
    self.ram_cache = LayerRAMCache(cache_budget)
    print(f"[Vaayu-Cache] RAM layer cache: {cache_budget / 1e9:.1f} GB budget")
else:
    self.ram_cache = None

self.loader_strategy = build_loader_strategy(
    str(self.checkpoint_path), self, ram_cache=self.ram_cache
)
```

Finally, in `forward()`, add `self.ram_cache.stats()` to the token summary print statement so we can visually verify the hits.

## Verification Plan
1. Run `python test_quick.py` with a generation size of > 3 tokens.
2. Token 1 logs should show `cache=0hit/Xmiss`.
3. Token 2 logs should show `cache=15hit/Ymiss` (or however many layers fit).
4. The `io_wait` metric on Token 2+ should drop by ~3-4 seconds.
