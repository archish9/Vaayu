import threading
from collections import Counter, OrderedDict

import torch

from .safetensor_model_persister import _ALIGN, chunk_decomposition


class SlabArena:
    """Fixed-size slab pool backing the expert RAM cache (zero-clone path).

    One slab holds one expert's full contiguous file span (aligned); io_uring
    reads land directly in the slab and the cache adopts VIEWS into it — no
    clones anywhere on the disk→RAM→GPU path. Slab size/count are DERIVED from
    the model's offset index and the cache budget (CLAUDE.md §5: model numbers
    are never hardcoded). pin=True allocates CUDA-pinned memory so H2D copies
    run at pinned bandwidth and non_blocking=True is honored.

    Allocation is SEVERAL exact-power-of-two pinned chunks, not one backing
    tensor: the caching host allocator charges the next power of two for every
    pinned request, so a single 9.13 GB arena is charged 16 GiB and ~7.9 GB is
    resident but unaddressable. Chunks are charged exactly (measured 2026-08-07:
    9.14 GB charged, 9.245 GB resident, every slab kept, H2D unchanged at 102%
    of the single-tensor figure). slab_bytes is _ALIGN-aligned and every chunk
    base is _ALIGN-aligned, so every slab start stays sector-aligned for
    O_DIRECT; a slab never straddles a chunk boundary, so each chunk's tail
    remainder (chunk_bytes % slab_bytes) is deliberately left unused.
    """

    def __init__(self, n_slabs: int, slab_bytes: int, pin: bool):
        self.slab_bytes = int(slab_bytes)
        self.n_slabs = int(n_slabs)
        self.pin = bool(pin)
        self._chunks: list[torch.Tensor] = []
        self._chunk_first: list[int] = []      # first GLOBAL slab index in chunk
        self._chunk_nslabs: list[int] = []     # slabs this chunk actually holds
        self._chunk_base_ptr: list[int] = []   # cached: data_ptr() per chunk
        placed = 0
        for chunk_bytes in chunk_decomposition(self.n_slabs, self.slab_bytes):
            if placed >= self.n_slabs:
                break
            chunk = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=self.pin)
            if chunk.data_ptr() % _ALIGN != 0:
                # Deviation from the brief's literal allocation, needed because
                # the natural allocation is not ALWAYS _ALIGN-aligned in
                # practice, on either path. Measured directly on this box
                # (torch 2.11.0+cu130): plain (pin=False) torch.empty is only
                # posix_memalign(64)-aligned, never page-aligned, at every size
                # this suite uses. And even pin=True is not immune once the
                # pinned pool is fragmented by other tests' churn (observed via
                # test_slab_arena.py::test_acquire_consumes_pending_read_event
                # failing only as part of the full suite, never alone) -- the
                # evidence doc's "pinned allocations come back 2 MiB-aligned"
                # finding (todo.md, E2) was measured on a fresh pool with the
                # real 8 GiB/512 MiB/16 MiB chunk sizes only, not guaranteed
                # for a fragmented pool or tiny allocations. Pad and slice as a
                # fallback so the invariant holds unconditionally. Only reached
                # off the fast path (it never fires for the production chunk
                # sizes on a fresh pool, per E2), so it never costs the
                # production case the pow2 charge exactness this task exists
                # to preserve -- the untouched branch above still requests the
                # exact chunk_bytes.
                raw = torch.empty(chunk_bytes + _ALIGN, dtype=torch.uint8, pin_memory=self.pin)
                pad = (-raw.data_ptr()) % _ALIGN
                chunk = raw[pad:pad + chunk_bytes]
            # O_DIRECT needs every slab start sector-aligned, and slab starts are
            # offsets from the chunk base -- so the base must be aligned too.
            # Measured 2026-08-07: pinned allocations come back 2 MiB-aligned.
            assert chunk.data_ptr() % _ALIGN == 0, (
                f"chunk base {chunk.data_ptr():#x} is not {_ALIGN}-aligned; "
                f"O_DIRECT slab reads would fail")
            held = min(self.n_slabs - placed, chunk_bytes // self.slab_bytes)
            self._chunks.append(chunk)
            self._chunk_first.append(placed)
            self._chunk_nslabs.append(held)
            self._chunk_base_ptr.append(chunk.data_ptr())
            placed += held
        self.n_slabs = placed
        self._free: list[int] = list(range(self.n_slabs))
        self._lock = threading.Lock()
        # slab idx -> CUDA event marking the most recent async (non_blocking)
        # H2D copy that reads FROM this slab's pinned memory. acquire()
        # synchronizes on it before handing the slab out for a fresh CPU-side
        # write, so an io_uring rewrite can't race an in-flight DMA read of the
        # same bytes (silent torn-read corruption). See record_read().
        self._pending_read: dict[int, "torch.cuda.Event"] = {}

    def acquire(self) -> "int | None":
        with self._lock:
            if not self._free:
                return None
            idx = self._free.pop()
            ev = self._pending_read.pop(idx, None)
        # Sync OUTSIDE the lock: a previously-enqueued H2D copy sourced from this
        # slab may still be draining on the CUDA stream. The caller is about to
        # overwrite the slab CPU-side (io_uring), which the DMA engine cannot
        # see — block until that read completes so it can't read torn bytes.
        # Near-zero cost in the common case (the read finished long ago); the
        # host only actually waits inside the genuine reuse-race window.
        if ev is not None:
            ev.synchronize()
        return idx

    def release(self, idx: int) -> None:
        # NOTE: no event bookkeeping here on purpose — a slab may be released
        # while an H2D read from it is still in flight (e.g. eviction under
        # concurrent stager churn). The pending-read event is kept and consumed
        # by acquire() at the actual point of reuse, which is the only moment
        # the race matters.
        with self._lock:
            self._free.append(idx)

    def _chunk_of(self, idx: int) -> int:
        """Index of the chunk holding global slab `idx`. Linear scan: k is 3 on
        both target geometries and bounded near 5 on any plausible box (chunk
        count tracks log2 of the arena, which is bounded by the RAM budget, not
        by model size)."""
        for c, first in enumerate(self._chunk_first):
            if first <= idx < first + self._chunk_nslabs[c]:
                return c
        raise IndexError(f"slab index {idx} outside arena of {self.n_slabs} slabs")

    def tensor(self, idx: int) -> torch.Tensor:
        c = self._chunk_of(idx)
        off = (idx - self._chunk_first[c]) * self.slab_bytes
        return self._chunks[c][off:off + self.slab_bytes]

    def record_read(self, idx: int) -> None:
        """Mark that an async (non_blocking) H2D copy reading FROM slab `idx`
        was just enqueued on the current CUDA stream. Records an event now;
        acquire() will synchronize on it before reusing this slab. No-op when
        CUDA is unavailable (the race only exists for real async H2D DMA)."""
        if not torch.cuda.is_available():
            return
        ev = torch.cuda.Event()
        ev.record()  # captures all work enqueued so far on the current stream
        with self._lock:
            self._pending_read[idx] = ev

    def slab_index_of(self, tensor: torch.Tensor) -> "int | None":
        """Return the slab index whose backing span contains `tensor`'s storage,
        or None if `tensor` is not a view into this arena (e.g. a legacy cloned
        expert). Pure host-side pointer arithmetic — no CUDA, no lock."""
        base = self._backing.data_ptr()
        off = tensor.data_ptr() - base
        if off < 0 or off >= self.n_slabs * self.slab_bytes:
            return None
        return int(off // self.slab_bytes)


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
        # Pin refcounts: a key with a positive count is CHECKED OUT by a compute
        # thread (its slab-backed views were handed out by get()/put_slab() but
        # the H2D copy has not been issued yet) and must NEVER be chosen as an
        # eviction victim. Without this, a concurrent evictor (esp. the lookahead
        # stager, which drives heavy eviction churn) can release-and-reuse a slab
        # whose bytes another thread is about to DMA to GPU — a silent torn read.
        # The caller unpins once the copy is issued, after which SlabArena's
        # record_read()/acquire() event-sync guards the in-flight-copy window.
        # Guarded by self.lock (same lock as all cache mutations — no new lock).
        self._pinned: dict[tuple, int] = {}

    def pin(self, layer_idx: int, expert_idx: int) -> None:
        """Refcount-pin (layer_idx, expert_idx) so eviction skips it. Safe to
        call for a key not currently in the cache (the pin still blocks a future
        victim selection until unpinned). Idempotent-safe via refcount."""
        with self.lock:
            key = (layer_idx, expert_idx)
            self._pinned[key] = self._pinned.get(key, 0) + 1

    def unpin(self, layer_idx: int, expert_idx: int) -> None:
        """Release one pin. No-op if the key isn't pinned (tolerant of double
        unpin / never-pinned, so callers can unpin defensively in a finally)."""
        with self.lock:
            key = (layer_idx, expert_idx)
            c = self._pinned.get(key)
            if c is None:
                return
            if c <= 1:
                del self._pinned[key]
            else:
                self._pinned[key] = c - 1

    def get(self, layer_idx: int, expert_idx: int, *, pin: bool = False) -> dict | None:
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
                    if pin:
                        self._pinned[key] = self._pinned.get(key, 0) + 1
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
                if pin:
                    self._pinned[key] = self._pinned.get(key, 0) + 1
                return dict(self.cache[key])
            self.misses += 1
            return None

    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                return lc is not None and expert_idx in lc
            return (layer_idx, expert_idx) in self.cache

    def resident_set(self, layer_idx: int) -> set:
        """Experts currently resident for `layer_idx`. Non-mutating (no LRU bump,
        no stat change) — for cache-aware routing's resident mask. One lock."""
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                return set(lc.keys()) if lc else set()
            return {e for (l, e) in self.cache if l == layer_idx}

    def put(self, layer_idx: int, expert_idx: int, tensors: dict, *, prefetch: bool = False) -> None:
        cloned = {k: v.clone().detach() for k, v in tensors.items()}
        raw_size = sum(t.nbytes for t in cloned.values())
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    return

                # 1. Per-layer fairness: evict oldest UNPINNED within this layer
                while len(lc) >= self.max_per_layer:
                    if not self._evict_layer_oldest(layer_idx, lc):
                        return  # every entry in this layer is pinned — refuse
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
            while self.current_usage + raw_size > self.budget_bytes:
                if not self._evict_global_oldest():
                    break
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

    def put_slab(self, layer_idx: int, expert_idx: int, views: dict, slab_idx: int,
                 *, prefetch: bool = False) -> None:
        """Adopt slab-backed tensor views WITHOUT cloning (zero-copy insert).
        On duplicate key the new slab is released back to the arena. Byte
        accounting uses slab_bytes — the real resident footprint.

        NOTE: a caller that intends to DMA this slab to GPU shortly should pin()
        the key BEFORE the read completes (the decode miss path pins in the main
        thread before submitting the load) so a concurrent evictor can't
        reuse the slab in the checked-out-but-not-yet-copied window."""
        key = (layer_idx, expert_idx)
        size = self.arena.slab_bytes
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    self.arena.release(slab_idx)
                    return
                while len(lc) >= self.max_per_layer:
                    if not self._evict_layer_oldest(layer_idx, lc):
                        self.arena.release(slab_idx)
                        return
                while self.current_usage + size > self.budget_bytes:
                    if not self._evict_global_oldest():
                        self.arena.release(slab_idx)
                        return
                lc[expert_idx] = views
                self.current_usage += size
                self._global_lru[key] = None
                self._slab_of[key] = slab_idx
                self._note_prefetch(layer_idx, expert_idx, prefetch)
                return
            if key in self.cache:
                self.arena.release(slab_idx)
                return
            while self.current_usage + size > self.budget_bytes:
                if not self._evict_global_oldest():
                    break
            if self.current_usage + size > self.budget_bytes:
                self.arena.release(slab_idx)
                return
            self.cache[key] = views
            self.current_usage += size
            self._slab_of[key] = slab_idx
            self._note_prefetch(layer_idx, expert_idx, prefetch)

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

    def _evict_layer_oldest(self, layer_idx: int, lc: OrderedDict) -> bool:
        """Evict the oldest UNPINNED expert within one layer's cache.

        Returns False if every entry in the layer is pinned (no valid victim).
        Must be called while self.lock is held.
        """
        for eidx in lc:  # OrderedDict iterates oldest-first
            if (layer_idx, eidx) not in self._pinned:
                evicted = lc.pop(eidx)
                self._account_evict(layer_idx, eidx, evicted)
                return True
        return False

    def _evict_global_oldest(self) -> bool:
        """Evict the single oldest UNPINNED expert across all layers (true global
        LRU). Pinned (checked-out) entries are skipped — they are never valid
        eviction victims.

        Returns False if nothing evictable remains (all remaining entries pinned
        or cache empty). Must be called while self.lock is held.
        """
        if self.max_per_layer > 0:
            victim = None
            for key in self._global_lru:  # oldest-first
                if key not in self._pinned:
                    victim = key
                    break
            if victim is None:
                return False
            del self._global_lru[victim]
            lid, eidx = victim
            lc = self._layer_caches.get(lid)
            if lc and eidx in lc:
                evicted = lc.pop(eidx)
                self._account_evict(lid, eidx, evicted)
                # Clean up empty layer dicts to avoid unbounded key growth
                if not lc:
                    del self._layer_caches[lid]
            return True
        # Flat global-LRU mode (max_per_layer == 0): evict oldest from self.cache.
        victim = None
        for key in self.cache:  # oldest-first
            if key not in self._pinned:
                victim = key
                break
        if victim is None:
            return False
        evicted = self.cache.pop(victim)
        self._account_evict(victim[0], victim[1], evicted)
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
            self._pinned.clear()

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

    def __init__(self, budget_bytes: int, track_evicted_keys: bool = False):
        self.budget_bytes = budget_bytes
        self.cache: "OrderedDict[tuple[int, int], dict]" = OrderedDict()
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefill_evictions = 0   # reactive prefill-headroom evictions
        # Per-(layer, expert) attribution of BUDGET-PRESSURE evictions only, for
        # the expert usage histogram. None unless explicitly enabled (ships dark).
        #
        # Deliberately NOT incremented by evict_until_free(): that is the
        # wholesale prefill wipe (measured at 1362 entries in a single pass),
        # not the LRU failing to afford an expert. Counting it here would drown
        # the signal and make every expert look like a pin candidate.
        self.evicted_keys = Counter() if track_evicted_keys else None
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

    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            return (layer_idx, expert_idx) in self.cache

    def resident_set(self, layer_idx: int) -> set:
        """Experts resident on GPU for `layer_idx`. Non-mutating — for
        cache-aware routing's resident mask. One lock."""
        with self.lock:
            return {e for (l, e) in self.cache if l == layer_idx}

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
                _ekey, evicted = self.cache.popitem(last=False)
                self.current_usage -= self._group_bytes(evicted)
                self.evictions += 1
                if self.evicted_keys is not None:
                    self.evicted_keys[_ekey] += 1
                del evicted  # drop GPU references so the allocator can reuse
            self.cache[key] = gpu_groups
            self.current_usage += size
            return True

    def drop(self, layer_idx: int, expert_idx: int) -> bool:
        """Remove a key that turned out to be unusable (e.g. a re-attach that
        failed structurally). NOT counted in self.evictions — that counter
        tracks budget-pressure evictions only, and this is a correctness
        removal of a stale/broken entry, not the LRU making room. Returns
        whether a key was actually removed."""
        with self.lock:
            key = (layer_idx, expert_idx)
            entry = self.cache.pop(key, None)
            if entry is None:
                return False
            self.current_usage -= self._group_bytes(entry)
            return True

    def evict_until_free(self, free_target_bytes: int) -> int:
        """Evict LRU entries until free VRAM would reach free_target_bytes, then
        return blocks to the driver. Used reactively during prefill: the resident
        cache is not consulted during a prefill pass, so evicting here has no
        correctness impact, and the evicted experts re-warm during decode.

        Reads free VRAM once, evicts enough BYTES to cover the deficit, then calls
        empty_cache() ONCE — dropping Python refs alone leaves blocks in PyTorch's
        caching allocator where mem_get_info still counts them used; only
        empty_cache() reclaims them, so a per-pop re-poll would never see free
        rise and would evict the whole cache. Returns bytes freed."""
        if not torch.cuda.is_available():
            return 0
        with self.lock:
            free, _ = torch.cuda.mem_get_info()
            deficit = free_target_bytes - free
            if deficit <= 0:
                return 0
            freed = 0
            while freed < deficit and self.cache:
                _key, evicted = self.cache.popitem(last=False)   # LRU
                n = self._group_bytes(evicted)
                self.current_usage -= n
                freed += n
                self.prefill_evictions += 1
                del evicted   # drop GPU refs so empty_cache can reclaim
        torch.cuda.empty_cache()   # outside the lock: realize the freed blocks
        return freed

    def reset_stats(self) -> None:
        """Zero hit/miss/eviction counters. Deliberately does NOT clear
        evicted_keys (the per-key eviction attribution used by the expert usage
        report): that mapping is meant to accumulate whole-run totals for the
        report to join against, so a per-turn `evict=` telemetry number reset
        here will legitimately read lower than evicted_keys' cumulative sum."""
        with self.lock:
            self.hits = self.misses = self.evictions = 0
            self._last_hits = self._last_misses = self._last_evictions = 0

    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            return (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                    f"hit_rate={rate} "
                    f"prefill_evict={self.prefill_evictions} "
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
