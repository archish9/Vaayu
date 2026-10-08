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
        # Hits served from ANOTHER session's entry, which are copied rather
        # than checked out. Counted separately because they are the expensive
        # kind (a copy of the matched prefix) and because a workload that is
        # mostly borrows is one where every session is re-prefilling most of
        # its own context -- visible here, invisible in hits/misses.
        self.borrows = 0
        # Lookups refused because the entry holds recurrent state and the match
        # would restore it at a length it does not describe (42-B). These are
        # MISSES, not failures: the request takes the full prefill it would have
        # taken before hybrids could reuse anything at all. Counted separately
        # because a workload that is mostly these is one where the hybrid prefix
        # cache is buying nothing, which hit_rate alone renders as healthy.
        self.recurrent_partial_misses = 0
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
            if key is None and self.auto_prefix_match:
                # NOT gated on `session_id is None`. An explicit id that misses
                # used to skip this branch entirely, so NAMING a session and
                # missing was strictly worse than not naming it -- a full
                # prefill where a usable prefix was sitting in the pool. Prefix
                # reuse is this engine's largest measured lever (136-377x TTFT);
                # a silent path to zero reuse is a defect, not a nuance.
                best, best_m = None, 0
                for k, e in self._pool.items():
                    m = longest_common_prefix(e.token_ids, input_ids)
                    if m > best_m:
                        best, best_m = k, m
                key = best
            if key is None:
                self.misses += 1
                return None, 0
            entry = self._pool[key]
            m = longest_common_prefix(entry.token_ids, input_ids)
            # A recurrent (linear-attention) entry is servable at exactly ONE
            # length: the L positions its conv/recurrent state summarises. The
            # KV half crops anywhere; the state half cannot be cropped at all,
            # because it is a fixed-size summary with no per-token history. The
            # caller's crop is min(m, prompt_len - 1) (_adopt_prefix_hit) and
            # crop() itself takes a min, so the cache lands on L whenever that
            # clamp is >= L. Refuse below it, and refuse any match that does not
            # consume the WHOLE entry -- the latter is also what keeps `borrowed`
            # False, so the raising copy_prefix stays unreachable. Stored entries
            # always satisfy L == len(token_ids) - 1 (the last sampled token was
            # never fed), so the second clause is an invariant guard: it fires
            # only if some future store breaks that relationship.
            # The same rule serves a cache that DECLARES itself one-length
            # (DeepSeek-V4's preallocated window, S34): it cannot be cropped
            # either, for a different reason, and `cache_is_one_length` is true
            # for both.
            from .batch import cache_is_one_length
            if cache_is_one_length(entry.cache):
                fed = entry.cache.get_seq_length()
                if (m < len(entry.token_ids)
                        or min(m, len(input_ids) - 1) < fed):
                    self.misses += 1
                    self.recurrent_partial_misses += 1
                    return None, 0
            self.hits += 1
            self.matched_tokens += m
            # BORROWED, not checked out. A hit removes the entry because the
            # caller crops it and writes its own decode tokens in -- correct for
            # your OWN context, destructive across sessions: `store` then
            # re-inserts it under the BORROWER's id and the owner's context is
            # gone. Two agents sharing a system prompt therefore destroyed each
            # other's cache every turn, reported as hits with zero evictions.
            # Measured on MiniMax-M2.7 at ~4.17k tokens: the session that kept
            # its entry prefilled 59 new positions (TTFT 15.0 s), the one that
            # lost it prefilled 4172 (190.0 s). So a borrower gets a COPY of the
            # matched prefix -- which is all it matched, and cropping is its next
            # move anyway -- and the owner keeps its entry.
            # Not "someone else's entry" but "an entry this prompt does not
            # fully contain". A prompt that consumes the WHOLE entry is a
            # continuation: it will store a superset back, which still serves
            # the original owner as a complete prefix, so checking it out loses
            # nobody anything -- and that is the unnamed single-user path, where
            # each turn hashes to a fresh id and would otherwise copy forever.
            borrowed = (key != session_id
                        and m < len(entry.token_ids))
            from .kv_compress import CompressedKV
            if isinstance(entry.cache, CompressedKV):
                if device is None:
                    raise ValueError(
                        f"session {key!r} is stored compressed; lookup() needs a "
                        f"device= to rebuild it. Returning the packed container "
                        f"would hand the caller indices in place of values.")
                from .kv_compress import decompress_cache
                self.decompressions += 1
                # Already a fresh object, so a borrow costs nothing extra here.
                rebuilt = decompress_cache(entry.cache, device)
                if borrowed:
                    self.borrows += 1
                    self._pool.move_to_end(key)
                    return rebuilt, m
                self._pool.pop(key)
                self._current_bytes -= entry.nbytes
                return rebuilt, m
            if borrowed:
                self.borrows += 1
                self._pool.move_to_end(key)     # actively useful: keep it MRU
                return entry.cache.copy_prefix(m), m
            self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            return entry.cache, m

    def peek_match(self, session_id, input_ids: list) -> int:
        """Longest prefix this pool COULD serve for `input_ids`. Non-destructive.

        `lookup()` POPS on a hit -- the entry is checked out to the in-flight
        request and only `store()` puts it back -- so it cannot be used to ask
        "how much would this request save?" before deciding anything. This
        answers exactly that and changes nothing: no pop, no counters, so a
        peek can never show up in the hit rate as reuse that did not happen.

        Same key selection `lookup()` uses -- named session first, then the
        best auto-match -- so the two cannot disagree about which entry would
        serve. Used by the server's batch formation (spec 4.3) to compare what
        batching would throw away against what it would buy.
        """
        with self._lock:
            if session_id is not None and session_id in self._pool:
                return longest_common_prefix(
                    self._pool[session_id].token_ids, input_ids)
            if not self.auto_prefix_match:
                return 0
            best_m = 0
            for e in self._pool.values():
                m = longest_common_prefix(e.token_ids, input_ids)
                if m > best_m:
                    best_m = m
            return best_m

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
            remedy = ("Set prefix_compress=9 to pack pooled KV (~0.61x, so roughly "
                      "1.6x the sessions) — measured cost is ~0.7s of TTFT on the "
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
            if self.borrows:
                # Hits served by COPYING another session's prefix. Worth its own
                # number: a workload that is mostly borrows is one where every
                # session keeps re-prefilling its own context, which hit_rate
                # alone renders as a healthy 75%.
                out += f" borrow={self.borrows}"
            if self.recurrent_partial_misses:
                out += f" recurrent_partial={self.recurrent_partial_misses}"
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
            self.borrows = 0
            self.recurrent_partial_misses = 0
            self.compressions = 0
            self.decompressions = 0


def restore_on_abort(store, session_id: str, token_ids: list, cache,
                     prefix_len: int) -> bool:
    """Put a checked-out prefix entry back after a turn died mid-flight.

    lookup() POPS an entry and store() only re-inserts it once generate()
    returns, so an aborted turn -- an SSE client disconnecting, a cancellation,
    an OOM -- loses that entry PERMANENTLY. A single-user script never hits
    this; a server hits it constantly.

    The adopted cache was mutated by however many forward passes completed, so
    it is CROPPED back to a length whose tokens are known before being stored.
    Re-storing the mutated cache against the original token list would be
    corruption, which is worse than losing the entry.

    Never raises: this runs on an exception path, and a failure here would
    replace the real error the caller is about to re-raise. Returns True only
    when an entry actually went back.

    A recurrent (linear-attention) cache cannot be safely cropped (42-B):
    `crop` shortens the KV layers' `_seq_len` only, and the conv/recurrent
    state a gated-delta-net layer keeps is a fixed-size summary that has
    already consumed every position fed to it -- there is no "first N
    tokens" of it to roll back to. If some forwards completed before the
    abort, the state has moved past `prefix_len` while the cropped KV would
    claim only `prefix_len` positions, and the next turn's lookup() reads
    `get_seq_length()` (the KV length) to decide what is safely servable --
    it has no way to see that the state disagrees. Serving that entry would
    resume generation against a state that already consumed tokens the next
    turn is about to re-feed it: silently wrong text, no crash. So a
    recurrent cache that actually needs cropping is dropped instead of
    stored; the no-crop case (the turn aborted before any forward completed,
    so KV and state already agree) is unaffected and still restores.
    """
    if store is None or prefix_len <= 0:
        return False
    try:
        seq_len = int(cache.get_seq_length())
        keep = min(int(prefix_len), seq_len)
        if keep <= 0:
            return False
        from .batch import cache_holds_recurrent_state, cache_is_one_length
        if cache_is_one_length(cache) and not cache_holds_recurrent_state(cache):
            # A DECLARED one-length cache (S34). DeepSeek-V4's attention writes
            # its Compressor / Indexer state before it advances the layer's
            # length, so an interrupted forward can leave the cache changed at
            # an unchanged length: the length cannot vouch for the state.
            print(f"[Vaayu-Prefix] WARNING: aborted turn on session "
                  f"{session_id!r}: its cache is servable at one length only "
                  f"and cannot vouch for its state after an interrupted "
                  f"forward; dropping the entry. The next turn prefills in full.")
            return False
        if keep < seq_len and cache_holds_recurrent_state(cache):
            print(f"[Vaayu-Prefix] WARNING: aborted turn on session "
                  f"{session_id!r} advanced a recurrent (linear-attention) "
                  f"state past the restorable length; dropping the entry "
                  f"rather than storing one whose KV and state disagree. "
                  f"The next turn prefills in full.")
            return False
        cache.crop(keep)
        store.store(session_id, list(token_ids[:keep]), cache)
        return True
    except Exception as e:
        print(f"[Vaayu-Prefix] WARNING: could not restore session {session_id!r} "
              f"after an aborted turn ({type(e).__name__}: {e}); the pool loses "
              f"this entry and the next turn will prefill in full.")
        return False
