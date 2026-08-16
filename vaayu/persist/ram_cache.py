import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import psutil


# HARDWARE CONSTANT — RAM held back from the layer cache on this box, covering the
# OS, the CUDA host-side context, PyTorch's own host buffers and the streaming
# working set. It is a property of the machine (32 GB), not of any model.
#
# It is subtracted from TOTAL RAM, deliberately, not from whatever is free at the
# moment of the call. The old form (`available - 8 GB`) sampled a moving quantity
# once at load, so a long-lived server's throughput was decided by whatever else
# happened to be running when it booted: two identical runs against
# Llama-3.3-70B took 9.7 GB / 18 layers / 0.167 tok/s and 16.4 GB / 33 layers /
# 0.230 tok/s -- a 38 % decode swing from luck alone.
RAM_CACHE_RESERVE = 16 * 1024 ** 3

# HARDWARE CONSTANT — never consume the last of free RAM even when the
# deterministic bound would allow it; the allocation itself needs headroom.
_LIVE_FLOOR = 2 * 1024 ** 3


def _bounds(total_bytes, available_bytes, override_bytes):
    """(deterministic_bound, live_bound) in bytes. Pure; see auto_detect_cache_budget."""
    if total_bytes is None or available_bytes is None:
        mem = psutil.virtual_memory()
        total_bytes = mem.total if total_bytes is None else total_bytes
        available_bytes = mem.available if available_bytes is None else available_bytes
    deterministic = (int(total_bytes) - RAM_CACHE_RESERVE
                     if override_bytes is None else int(override_bytes))
    live = int(available_bytes) - _LIVE_FLOOR
    return deterministic, live


def auto_detect_cache_budget(total_bytes=None, available_bytes=None,
                             override_bytes=None) -> int:
    """Bytes the RAM layer cache may hold.

    Derived from TOTAL system RAM minus RAM_CACHE_RESERVE, so the same box gives
    the same answer on every start. Still clamped by what is actually free right
    now -- deterministic is not the same as reckless, and RAM that is genuinely
    occupied cannot be cached into. `override_bytes` replaces the derived figure
    (0 disables the cache) but is clamped by the same live bound.

    Which bound applied is reportable via describe_cache_budget, so a budget that
    collapsed because the box was busy is visible rather than silent
    (CLAUDE.md §5: warn, never silently degrade).
    """
    deterministic, live = _bounds(total_bytes, available_bytes, override_bytes)
    return max(0, min(deterministic, live))


def describe_cache_budget(total_bytes=None, available_bytes=None,
                          override_bytes=None) -> str:
    """One grep-able line naming the budget AND which constraint set it."""
    deterministic, live = _bounds(total_bytes, available_bytes, override_bytes)
    budget = max(0, min(deterministic, live))
    if override_bytes is not None:
        source = "override"
        if live < deterministic:
            source = "override, clamped by available RAM"
    else:
        source = "total RAM" if deterministic <= live else "available RAM"
    return (f"RAM layer cache budget {budget / 1e9:.1f} GB "
            f"(bound by {source}; total-derived {max(0, deterministic) / 1e9:.1f} GB, "
            f"live-derived {max(0, live) / 1e9:.1f} GB)")


def _clone_tensors(tensors: dict) -> dict:
    return {k: v.clone().detach() for k, v in tensors.items()}


class LayerRAMCache:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.cache: OrderedDict = OrderedDict()
        self._pending: set = set()  # names submitted for clone but not yet stored
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        # Single-worker pool: clones run serially, never on the NVMe prefetch thread.
        # max_workers=1 is intentional — parallel clones would fight for DDR4 bandwidth
        # and serialize anyway; one clone at a time is as fast and uses less overhead.
        self._clone_pool = ThreadPoolExecutor(max_workers=1)

    def get(self, layer_name: str) -> dict | None:
        with self.lock:
            if layer_name in self.cache:
                self.hits += 1
                return self.cache[layer_name]
            self.misses += 1
            return None

    def put(self, layer_name: str, tensors: dict) -> None:
        """Non-blocking: reserves budget and returns immediately; clone runs in background.

        Race safety: tensors are views into the pinned double-buffer (A/B). The buffer
        holding these views won't be reused until 2 more NVMe layers complete (~100ms
        minimum). The background clone finishes in ~20ms for a 510MB layer. Safe by 5x.
        For runs of consecutive cache hits, the window before buffer reuse is even
        larger (700ms+) because cache hits do not call _load_pinned or toggle _BUF_IDX.
        """
        with self.lock:
            if layer_name in self.cache or layer_name in self._pending:
                return
            raw_size = sum(t.nbytes for t in tensors.values())
            if self.current_usage + raw_size > self.budget_bytes:
                return
            self._pending.add(layer_name)
            self.current_usage += raw_size  # reserve budget before releasing lock

        def _clone_and_store(name: str = layer_name, src: dict = tensors) -> None:
            cached = _clone_tensors(src)
            with self.lock:
                self._pending.discard(name)
                self.cache[name] = cached

        self._clone_pool.submit(_clone_and_store)

    def put_sync(self, layer_name: str, tensors: dict) -> bool:
        """Synchronous put for use at init time (pre-population).

        Clones inline — safe because no inference is running (no DDR4 contention with
        H2D, no risk of pinned buffer being reused mid-clone).
        Returns True if stored, False if skipped (over budget or already cached).
        """
        with self.lock:
            if layer_name in self.cache or layer_name in self._pending:
                return False
            raw_size = sum(t.nbytes for t in tensors.values())
            if self.current_usage + raw_size > self.budget_bytes:
                return False
            cached = _clone_tensors(tensors)
            self.cache[layer_name] = cached
            self.current_usage += raw_size
            return True

    def stats(self) -> str:
        with self.lock:
            pending = len(self._pending)
        suffix = f"+{pending}pending" if pending else ""
        return f"{self.hits}hit/{self.misses}miss{suffix}"
