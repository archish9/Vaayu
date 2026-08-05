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

    def __init__(self, budget_bytes: int, auto_prefix_match: bool = True,
                 compress_bits: int | None = None):
        self.budget_bytes = budget_bytes
        self.auto_prefix_match = auto_prefix_match
        # None = store cache objects as-is (default, unchanged behaviour).
        # An int = TurboQuant-pack entries while they sit in the pool, so the
        # same budget holds ~2x more sessions at 8 bits. Compression happens
        # once per turn here, never on the decode path -- see
        # docs/superpowers/plans/2026-08-02-compress-on-evict-kv.md.
        self.compress_bits = compress_bits
        self.compressions = 0
        self.decompressions = 0
        self._pool: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU order
        self._current_bytes = 0
        self._lock = threading.Lock()
        # Telemetry (chain step 4). Plain ints, mutated under _lock exactly like
        # the pool itself. Mirrors ExpertRAMCache's hits/misses/evictions; the
        # token counters exist because hit/miss alone reports 100% for a cache
        # that saved nothing (see the design spec's "Why token accounting").
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.matched_tokens = 0
        self.lookup_tokens = 0
        # Latch for the capacity warning below: once per store lifetime, not
        # once per eviction. The agentic workload re-stores every session every
        # turn, so per-eviction warning would emit hundreds of lines per run and
        # bury the first one.
        self._capacity_warned = False

    def lookup(self, session_id, input_ids: list, device=None):
        """Return (cache_or_None, match_len). A hit REMOVES the entry from the
        pool (checked out to the in-flight request); store() re-inserts it."""
        with self._lock:
            self.lookup_tokens += len(input_ids)
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
                self.misses += 1
                return None, 0
            entry = self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            m = longest_common_prefix(entry.token_ids, input_ids)
            self.hits += 1
            self.matched_tokens += m
            from .kv_compress import CompressedKV
            if isinstance(entry.cache, CompressedKV):
                if device is None:
                    raise ValueError(
                        f"session {key!r} is stored compressed; lookup() needs a "
                        f"device= to rebuild it. Returning the packed container "
                        f"would hand the caller indices in place of values.")
                from .kv_compress import decompress_cache
                self.decompressions += 1
                return decompress_cache(entry.cache, device), m
            return entry.cache, m

    def store(self, session_id: str, token_ids: list, cache) -> None:
        if self.compress_bits is None:
            cache.to_cpu()
            payload, nbytes = cache, cache.resident_bytes()
        else:
            # Compress-on-evict: once per turn, off the decode path entirely.
            from .kv_compress import compress_cache
            payload = compress_cache(cache, bits_total=self.compress_bits)
            nbytes = payload.nbytes
            self.compressions += 1
        with self._lock:
            if session_id in self._pool:
                self._current_bytes -= self._pool.pop(session_id).nbytes
            self._pool[session_id] = _Entry(list(token_ids), payload, nbytes)
            self._current_bytes += nbytes
            _evicted_here = 0
            while self._current_bytes > self.budget_bytes and len(self._pool) > 1:
                _k, ev = self._pool.popitem(last=False)  # evict LRU
                self._current_bytes -= ev.nbytes
                self.evictions += 1
                _evicted_here += 1
            if _evicted_here and not self._capacity_warned:
                self._capacity_warned = True
                self._warn_over_capacity(nbytes)

    def _warn_over_capacity(self, sess_bytes: int) -> None:
        """Announce that the pool is over capacity, on the first budget-pressure
        eviction. Called under _lock.

        A budget-pressure eviction is unambiguous — the loop above only runs when
        the pool is over budget — and it is the moment cross-turn reuse stops
        working. Measured on-device (Plan J Task 4): one session past capacity
        does not cost a proportional share of the reuse, it costs ALL of it. At
        K=7 against a pool holding 5-6, the final round scored 0 hits, not 5,
        because each recomputed session's re-store cascade-evicts the ones before
        it. Round-2 TTFT stayed at the cold 352 s against 9.8 s at K=6.

        Everything printed is derived from this pool's own state (CLAUDE.md §5):
        the budget it was constructed with and the size of the session just
        stored. No model constant appears here — the two on-device geometries
        differ by ~30% per session and their pool budgets by ~2.7x.
        """
        capacity = int(self.budget_bytes // sess_bytes) if sess_bytes else 0
        if self.compress_bits is None:
            remedy = ("Set prefix_compress=9 to pack pooled KV (~0.55x, so roughly "
                      "double the sessions) — measured cost is ~0.7s of TTFT on the "
                      "hit path, against a full cold prefill if you thrash.")
        else:
            remedy = (f"Pooled KV is already compressed at {self.compress_bits} bits, "
                      f"so the remaining levers are fewer concurrent sessions or less "
                      f"context per session.")
        print(f"[Vaayu-Prefix] WARNING: prefix pool over capacity — evicting under "
              f"budget pressure. Budget {self.budget_bytes / 1e9:.2f} GB holds "
              f"{capacity} session(s) at this session's measured "
              f"{sess_bytes / 1e9:.2f} GB; beyond that, cross-turn reuse collapses to "
              f"zero rather than degrading, because each session's re-store evicts "
              f"the others. {remedy} Run tune_advisor.py with "
              f"POOL_BUDGET_GB={self.budget_bytes / 1e9:.2f} for the full trade-off. "
              f"(a real capacity limit, not a bug)")

    def stats(self) -> str:
        """One grep-able line: pool hit/miss/evict, plus the share of requested
        prompt tokens the cache let us skip.

        Both rates render "n/a" rather than dividing by zero when nothing has
        been looked up yet.

        NOTE on precision: matched_tokens records the raw match length computed
        here. The caller (the generate override in vaayu_base.py that prints
        "[Vaayu-Prefix] HIT") afterward clamps that value down by one position
        when the new prompt is a complete prefix of the cached one, so at least
        one token is left for the model to run. `saved` therefore over-reports
        by at most 1 token per hit, and only in that exact-replay case. The
        clamp stays in the caller on purpose: needing a token to run is a
        generation concern, not a caching one.

        NOTE on hit_rate asymmetry: "hit" means different things on the two
        lookup() paths. On a named session, the key is set whenever the
        session's entry exists in the pool, so a session whose context fully
        diverged still counts as a HIT with 0 matched_tokens (check `saved`
        for the real payoff). On an anonymous lookup (session_id=None, the
        default `generate()` call shape), the key is only set when some pool
        entry shares at least one token, so the same "entry present, nothing
        shared" case counts as a MISS. A mixed workload's hit_rate is
        therefore not directly comparable across the two modes.
        """
        with self._lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            saved = (f"{self.matched_tokens / self.lookup_tokens:.0%}"
                     if self.lookup_tokens > 0 else "n/a")
            out = (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                   f"hit_rate={rate} "
                   f"saved={self.matched_tokens}/{self.lookup_tokens} tok ({saved})")
            if self.compress_bits is not None:
                out += (f" compress={self.compress_bits}b "
                        f"{self.compressions}pack/{self.decompressions}unpack")
            return out

    def reset_stats(self) -> None:
        """Zero the counters, leaving the pool untouched. Called between
        measurement rounds so one round's hit count is measured clean, without
        the previous round's guaranteed-miss cold pass polluting it (same role
        as ExpertRAMCache.reset_stats at the prefill->decode boundary).

        NOTE: when enable_kv_persist=True, kv_persist.py's disk-restore path
        calls store.store() during model construction, before any harness
        obtains its handle — so load-time evictions can already be nonzero by
        the time a caller first reads stats() or calls reset_stats(). A
        "[R0] ...Nevict" line is not necessarily round 0's own activity."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.matched_tokens = 0
            self.lookup_tokens = 0
            self.compressions = 0
            self.decompressions = 0
