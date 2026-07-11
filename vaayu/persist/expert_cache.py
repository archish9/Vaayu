import threading
from collections import OrderedDict

import torch


class SlabArena:
    """Fixed-size slab pool backing the expert RAM cache (zero-clone path).

    One slab holds one expert's full contiguous file span (aligned); io_uring
    reads land directly in the slab and the cache adopts VIEWS into it — no
    clones anywhere on the disk→RAM→GPU path. Slab size/count are DERIVED from
    the model's offset index and the cache budget (CLAUDE.md §5: model numbers
    are never hardcoded). pin=True allocates CUDA-pinned memory so H2D copies
    run at pinned bandwidth and non_blocking=True is honored.

    Allocation is ONE backing tensor (a single cudaHostAlloc, not thousands),
    sliced into slabs; slab_bytes is _ALIGN-aligned so every slab start stays
    sector-aligned for O_DIRECT.
    """

    def __init__(self, n_slabs: int, slab_bytes: int, pin: bool):
        self.slab_bytes = int(slab_bytes)
        self.n_slabs = int(n_slabs)
        self.pin = bool(pin)
        self._backing = torch.empty(
            self.n_slabs * self.slab_bytes, dtype=torch.uint8, pin_memory=self.pin
        )
        self._free: list[int] = list(range(self.n_slabs))
        self._lock = threading.Lock()

    def acquire(self) -> "int | None":
        with self._lock:
            return self._free.pop() if self._free else None

    def release(self, idx: int) -> None:
        with self._lock:
            self._free.append(idx)

    def tensor(self, idx: int) -> torch.Tensor:
        return self._backing[idx * self.slab_bytes:(idx + 1) * self.slab_bytes]


class ExpertRAMCache:
    """
    LRU cache for MoE expert weight tensors, keyed by (layer_idx, expert_idx).

    Separate from LayerRAMCache (which caches whole layers for dense models).
    Thread-safe: single lock guards all mutations and reads.

    Two eviction constraints enforced simultaneously:
      1. Per-layer cap (max_per_layer): prevents a single hot layer from
         monopolising the cache. Evicts oldest expert *within* that layer.
      2. Global budget (budget_bytes): hard RAM ceiling. Evicts the
         globally-oldest expert across *all* layers (true LRU).

    When max_per_layer == 0, falls back to a flat global LRU (original path).
    """

    def __init__(self, budget_bytes: int, max_per_layer: int = 0,
                 arena: "SlabArena | None" = None):
        self.budget_bytes = budget_bytes
        self.max_per_layer = max_per_layer
        # Zero-clone mode: entries adopted via put_slab() are views into arena
        # slabs; eviction releases the slab back to the arena's freelist.
        self.arena = arena
        self._slab_of: dict[tuple, int] = {}  # (layer, expert) -> slab idx
        # per-layer mode: {layer_idx: OrderedDict[expert_idx, tensors]}
        self._layer_caches: dict[int, OrderedDict] = {} if max_per_layer > 0 else {}
        # True global LRU: tracks (layer_idx, expert_idx) insertion/access order
        # across all layers. Used for cross-layer eviction when budget is exceeded.
        self._global_lru: OrderedDict = OrderedDict()  # key → None (order-only)
        self.cache: OrderedDict = OrderedDict()  # global LRU (used when max_per_layer == 0)
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # Baseline for per-interval delta reporting (see delta_stats).
        self._last_hits = 0
        self._last_misses = 0
        self._last_evictions = 0
        self.prefetch_inserts = 0
        self.prefetch_hits = 0
        self._prefetched: set = set()  # keys inserted by prefetch, not yet used

    def get(self, layer_idx: int, expert_idx: int) -> dict | None:
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                if lc and expert_idx in lc:
                    lc.move_to_end(expert_idx)
                    # Update global LRU access order
                    key = (layer_idx, expert_idx)
                    if key in self._global_lru:
                        self._global_lru.move_to_end(key)
                    self.hits += 1
                    if key in self._prefetched:
                        self._prefetched.discard(key)
                        self.prefetch_hits += 1
                    return dict(lc[expert_idx])
                self.misses += 1
                return None
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                if key in self._prefetched:
                    self._prefetched.discard(key)
                    self.prefetch_hits += 1
                return dict(self.cache[key])
            self.misses += 1
            return None

    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                return lc is not None and expert_idx in lc
            return (layer_idx, expert_idx) in self.cache

    def put(self, layer_idx: int, expert_idx: int, tensors: dict, *, prefetch: bool = False) -> None:
        cloned = {k: v.clone().detach() for k, v in tensors.items()}
        raw_size = sum(t.nbytes for t in cloned.values())
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    return

                # 1. Per-layer fairness: evict oldest within this layer
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self._account_evict(layer_idx, evicted_eidx, evicted)

                # 2. Global budget enforcement: evict globally-oldest across ALL layers
                while self.current_usage + raw_size > self.budget_bytes:
                    if not self._evict_global_oldest():
                        return  # Can't free anything — refuse insert (OOM guard)

                lc[expert_idx] = cloned
                self.current_usage += raw_size
                self._global_lru[(layer_idx, expert_idx)] = None
                self._note_prefetch(layer_idx, expert_idx, prefetch)
                return

            # Global LRU path (max_per_layer == 0) — unchanged
            key = (layer_idx, expert_idx)
            if key in self.cache:
                return
            while self.current_usage + raw_size > self.budget_bytes and self.cache:
                evicted_key, evicted = self.cache.popitem(last=False)
                self._account_evict(evicted_key[0], evicted_key[1], evicted)
            if self.current_usage + raw_size > self.budget_bytes:
                return
            self.cache[key] = cloned
            self.current_usage += raw_size
            self._note_prefetch(layer_idx, expert_idx, prefetch)

    def acquire_slab(self) -> "int | None":
        """Free slab index, evicting LRU entries if the arena is exhausted.
        Returns None when there's no arena or nothing can be evicted (caller
        falls back to the legacy clone path)."""
        if self.arena is None:
            return None
        idx = self.arena.acquire()
        while idx is None:
            with self.lock:
                if not self._evict_global_oldest():
                    return None
            idx = self.arena.acquire()
        return idx

    def put_slab(self, layer_idx: int, expert_idx: int, views: dict, slab_idx: int) -> None:
        """Adopt slab-backed tensor views WITHOUT cloning (zero-copy insert).
        On duplicate key the new slab is released back to the arena. Byte
        accounting uses slab_bytes — the real resident footprint."""
        size = self.arena.slab_bytes
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    self.arena.release(slab_idx)
                    return
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self._account_evict(layer_idx, evicted_eidx, evicted)
                while self.current_usage + size > self.budget_bytes:
                    if not self._evict_global_oldest():
                        self.arena.release(slab_idx)
                        return
                lc[expert_idx] = views
                self.current_usage += size
                self._global_lru[(layer_idx, expert_idx)] = None
                self._slab_of[(layer_idx, expert_idx)] = slab_idx
                return
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.arena.release(slab_idx)
                return
            while self.current_usage + size > self.budget_bytes and self.cache:
                evicted_key, evicted = self.cache.popitem(last=False)
                self._account_evict(evicted_key[0], evicted_key[1], evicted)
            if self.current_usage + size > self.budget_bytes:
                self.arena.release(slab_idx)
                return
            self.cache[key] = views
            self.current_usage += size
            self._slab_of[key] = slab_idx

    def _account_evict(self, layer_idx: int, expert_idx: int, evicted: dict) -> None:
        """Shared eviction bookkeeping: usage, counters, prefetch set, slab release.
        Must be called while self.lock is held."""
        key = (layer_idx, expert_idx)
        slab_idx = self._slab_of.pop(key, None)
        if slab_idx is not None:
            self.current_usage -= self.arena.slab_bytes
            self.arena.release(slab_idx)
        else:
            self.current_usage -= sum(t.nbytes for t in evicted.values())
        self._global_lru.pop(key, None)
        self.evictions += 1
        self._prefetched.discard(key)

    def _note_prefetch(self, layer_idx, expert_idx, prefetch):
        # Must be called while self.lock is held, after a successful insert.
        if prefetch:
            self.prefetch_inserts += 1
            self._prefetched.add((layer_idx, expert_idx))

    def _evict_global_oldest(self) -> bool:
        """Evict the single oldest expert across all layers (true global LRU).

        Returns False if nothing left to evict.
        Must be called while self.lock is held.
        """
        if self.max_per_layer > 0:
            if not self._global_lru:
                return False
            (lid, eidx), _ = self._global_lru.popitem(last=False)
            lc = self._layer_caches.get(lid)
            if lc and eidx in lc:
                evicted = lc.pop(eidx)
                self._account_evict(lid, eidx, evicted)
                # Clean up empty layer dicts to avoid unbounded key growth
                if not lc:
                    del self._layer_caches[lid]
            return True
        # Flat global-LRU mode (max_per_layer == 0): evict oldest from self.cache.
        if not self.cache:
            return False
        (lid, eidx), evicted = self.cache.popitem(last=False)
        self._account_evict(lid, eidx, evicted)
        return True

    def clear(self) -> None:
        with self.lock:
            if self.max_per_layer > 0:
                self._layer_caches.clear()
                self._global_lru.clear()
            else:
                self.cache.clear()
            if self.arena is not None:
                for _slab_idx in self._slab_of.values():
                    self.arena.release(_slab_idx)
            self._slab_of.clear()
            self.current_usage = 0
            self._prefetched.clear()

    def reset_stats(self) -> None:
        """Zero all counters. Called at the prefill→decode boundary so decode
        hit-rate is measured clean, without prefill's ~100%-miss pollution."""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self._last_hits = 0
            self._last_misses = 0
            self._last_evictions = 0
            self.prefetch_inserts = 0
            self.prefetch_hits = 0
            self._prefetched.clear()

    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            pf = (f"{self.prefetch_hits}/{self.prefetch_inserts}"
                  if self.prefetch_inserts else "0/0")
            return (
                f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                f"hit_rate={rate} prefetch_used={pf} "
                f"usage={self.current_usage / 1e9:.2f}GB/"
                f"{self.budget_bytes / 1e9:.2f}GB"
            )

    def delta_stats(self) -> str:
        """Hits/misses/evictions since the previous delta_stats() call, plus the
        per-interval hit rate. Reveals per-token decode behaviour that cumulative
        stats() hides. Advances the baseline each call."""
        with self.lock:
            dh = self.hits - self._last_hits
            dm = self.misses - self._last_misses
            de = self.evictions - self._last_evictions
            self._last_hits = self.hits
            self._last_misses = self.misses
            self._last_evictions = self.evictions
            total = dh + dm
            rate = f"{dh / total:.0%}" if total > 0 else "n/a"
            return f"{dh}hit/{dm}miss/{de}evict hit_rate={rate}"


class ExpertVRAMCache:
    """GPU-resident LRU of per-expert tensor groups, keyed by (layer, expert).

    Fix #7: eliminates per-token host->device copies for hot experts. Stores the
    already-on-GPU {"gate":{...}, "up":{...}, "down":{...}} dicts. Byte-budgeted
    (budget derived from free VRAM, a hardware limit — see caller). Thread-safe;
    a single lock guards all state. Evicted tensors' references are dropped so the
    torch caching allocator can reuse the memory.
    """

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.cache: "OrderedDict[tuple[int, int], dict]" = OrderedDict()
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._last_hits = 0
        self._last_misses = 0
        self._last_evictions = 0

    @staticmethod
    def _group_bytes(groups: dict) -> int:
        total = 0
        for role_name, role in groups.items():
            if role_name == "bc":
                # Cached BC_LinearEXL3 objects (Task 7): C++ wrappers around the
                # tensors already counted below — no separate byte footprint.
                continue
            for t in role.values():
                total += t.nbytes
        return total

    def get(self, layer_idx: int, expert_idx: int):
        with self.lock:
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, layer_idx: int, expert_idx: int, gpu_groups: dict) -> bool:
        """Insert; returns True if the cache now owns gpu_groups, False if refused
        (caller keeps ownership and must free it itself)."""
        size = self._group_bytes(gpu_groups)
        with self.lock:
            key = (layer_idx, expert_idx)
            if key in self.cache:
                return False
            if size > self.budget_bytes:
                return False  # single expert bigger than whole budget — refuse
            while self.current_usage + size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= self._group_bytes(evicted)
                self.evictions += 1
                del evicted  # drop GPU references so the allocator can reuse
            self.cache[key] = gpu_groups
            self.current_usage += size
            return True

    def reset_stats(self) -> None:
        with self.lock:
            self.hits = self.misses = self.evictions = 0
            self._last_hits = self._last_misses = self._last_evictions = 0

    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            return (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                    f"hit_rate={rate} "
                    f"usage={self.current_usage / 1e9:.2f}GB/{self.budget_bytes / 1e9:.2f}GB")

    def delta_stats(self) -> str:
        with self.lock:
            dh = self.hits - self._last_hits
            dm = self.misses - self._last_misses
            de = self.evictions - self._last_evictions
            self._last_hits, self._last_misses, self._last_evictions = self.hits, self.misses, self.evictions
            total = dh + dm
            rate = f"{dh / total:.0%}" if total > 0 else "n/a"
            return f"{dh}hit/{dm}miss/{de}evict hit_rate={rate}"

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.current_usage = 0


class LRUTensorCache:
    """
    Byte-budgeted LRU cache for arbitrary tensor dicts, keyed by string.

    Used for non-expert tensor caching (attention weights, norms, router gates)
    where keys are layer names like "model.layers.3". Prevents the unbounded
    dict growth that exhausted RAM on large MoE models.

    Thread-safe: single lock guards all mutations and reads.
    """

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._sizes: dict[str, int] = {}  # key → byte size
        self.current_usage = 0
        self.lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self._cache

    def get(self, key: str) -> dict | None:
        with self.lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, tensors: dict) -> None:
        raw_size = sum(t.nbytes for t in tensors.values())
        with self.lock:
            if key in self._cache:
                return
            # Evict LRU entries until there's room
            while self.current_usage + raw_size > self.budget_bytes and self._cache:
                evicted_key, _ = self._cache.popitem(last=False)
                self.current_usage -= self._sizes.pop(evicted_key, 0)
            if self.current_usage + raw_size > self.budget_bytes:
                return  # single entry exceeds budget — refuse
            self._cache[key] = tensors
            self._sizes[key] = raw_size
            self.current_usage += raw_size

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()
            self._sizes.clear()
            self.current_usage = 0
