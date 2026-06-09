import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import psutil


def auto_detect_cache_budget() -> int:
    mem = psutil.virtual_memory()
    return max(0, mem.available - (8 * 1024 ** 3))


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
