"""Cross-turn KV prefix caching (A1): reuse a session's KV cache across turns,
prefilling only the divergent tail. See
docs/superpowers/specs/2026-07-12-vaayu-prefix-caching-design.md."""
import threading


def longest_common_prefix(a: list, b: list) -> int:
    """Number of leading elements equal in both sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


from collections import OrderedDict


class _Entry:
    __slots__ = ("token_ids", "cache", "nbytes")
    def __init__(self, token_ids, cache, nbytes):
        self.token_ids = token_ids
        self.cache = cache
        self.nbytes = nbytes


class PrefixCacheStore:
    """Thread-safe, byte-budgeted pool of per-session KV caches held in RAM
    between turns (A1). See the design spec for the reuse protocol."""

    def __init__(self, budget_bytes: int, auto_prefix_match: bool = True):
        self.budget_bytes = budget_bytes
        self.auto_prefix_match = auto_prefix_match
        self._pool: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU order
        self._current_bytes = 0
        self._lock = threading.Lock()

    def lookup(self, session_id, input_ids: list):
        """Return (cache_or_None, match_len). A hit REMOVES the entry from the
        pool (checked out to the in-flight request); store() re-inserts it."""
        with self._lock:
            key = None
            if session_id is not None and session_id in self._pool:
                key = session_id
            elif self.auto_prefix_match and session_id is None:
                best, best_m = None, 0
                for k, e in self._pool.items():
                    m = longest_common_prefix(e.token_ids, input_ids)
                    if m > best_m:
                        best, best_m = k, m
                key = best
            if key is None:
                return None, 0
            entry = self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            m = longest_common_prefix(entry.token_ids, input_ids)
            return entry.cache, m

    def store(self, session_id: str, token_ids: list, cache) -> None:
        cache.to_cpu()
        nbytes = cache.resident_bytes()
        with self._lock:
            if session_id in self._pool:
                self._current_bytes -= self._pool.pop(session_id).nbytes
            self._pool[session_id] = _Entry(list(token_ids), cache, nbytes)
            self._current_bytes += nbytes
            while self._current_bytes > self.budget_bytes and len(self._pool) > 1:
                _k, ev = self._pool.popitem(last=False)  # evict LRU
                self._current_bytes -= ev.nbytes

    def drop(self, session_id: str) -> None:
        with self._lock:
            e = self._pool.pop(session_id, None)
            if e is not None:
                self._current_bytes -= e.nbytes
