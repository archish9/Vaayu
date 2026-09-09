from vaayu.prefix_cache import longest_common_prefix


def test_lcp_identical():
    assert longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_lcp_partial():
    assert longest_common_prefix([1, 2, 3, 9], [1, 2, 3, 8, 8]) == 3


def test_lcp_none():
    assert longest_common_prefix([5, 1], [4, 1]) == 0


def test_lcp_empty():
    assert longest_common_prefix([], [1, 2]) == 0
    assert longest_common_prefix([1, 2], []) == 0


def test_lcp_one_is_prefix_of_other():
    assert longest_common_prefix([1, 2], [1, 2, 3, 4]) == 2


from vaayu.prefix_cache import PrefixCacheStore


class _FakeCache:
    """Stands in for FP8DynamicCache: tracks bytes + records to_cpu()."""
    def __init__(self, nbytes):
        self._nbytes = nbytes
        self.on_cpu = False
    def resident_bytes(self):
        return self._nbytes
    def to_cpu(self):
        self.on_cpu = True
    def copy_prefix(self, length):
        """Lending a prefix to another session copies it (2026-09-09). Any
        cache the pool holds owes this method."""
        out = _FakeCache(self._nbytes)
        out.on_cpu = self.on_cpu
        out.copied_from = self
        out.copied_len = length
        return out


def test_store_lookup_session_hit():
    s = PrefixCacheStore(budget_bytes=10_000)
    c = _FakeCache(1000)
    s.store("s1", [1, 2, 3, 4], c)
    assert c.on_cpu                                 # moved to RAM on store
    kv, m = s.lookup("s1", [1, 2, 3, 9, 9])
    assert kv is c and m == 3


def test_lookup_miss_returns_none():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert s.lookup("nope", [1, 2, 3]) == (None, 0)


def test_auto_prefix_match_when_no_session():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    s.store("s1", [1, 2, 3, 4], _FakeCache(500))
    s.store("s2", [1, 2, 9, 9], _FakeCache(500))
    kv, m = s.lookup(None, [1, 2, 3, 7])           # best LCP is s1 (len 3)
    assert m == 3


def test_auto_match_off_no_session_is_miss():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=False)
    s.store("s1", [1, 2, 3], _FakeCache(500))
    assert s.lookup(None, [1, 2, 3]) == (None, 0)


def test_budget_evicts_lru():
    s = PrefixCacheStore(budget_bytes=1500)
    s.store("s1", [1], _FakeCache(1000))
    s.store("s2", [2], _FakeCache(1000))           # over budget -> evict s1
    assert s.lookup("s1", [1]) == (None, 0)
    kv, _ = s.lookup("s2", [2])
    assert kv is not None


def test_lookup_checks_out_entry():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3], _FakeCache(500))
    s.lookup("s1", [1, 2, 3])                       # checked out
    assert s.lookup("s1", [1, 2, 3]) == (None, 0)   # gone until next store


def test_counters_start_at_zero():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert (s.hits, s.misses, s.evictions) == (0, 0, 0)
    assert (s.matched_tokens, s.lookup_tokens) == (0, 0)


def test_miss_counts_lookup_tokens_only():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.lookup("nope", [1, 2, 3])
    assert s.misses == 1
    assert s.hits == 0
    assert s.lookup_tokens == 3
    assert s.matched_tokens == 0


def test_hit_counts_matched_tokens():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [1, 2, 3, 9, 9])          # shares 3 of 5 requested tokens
    assert s.hits == 1
    assert s.misses == 0
    assert s.matched_tokens == 3
    assert s.lookup_tokens == 5


def test_degenerate_hit_shows_full_hit_rate_but_no_savings():
    """A session whose context fully diverged still hits the pool but saves
    nothing. hit_rate must read 100% while saved reads 0% — this is exactly
    why token accounting exists alongside hit/miss."""
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [9] * 100)                # entry found, zero tokens shared
    assert s.hits == 1
    assert s.matched_tokens == 0
    out = s.stats()
    assert "hit_rate=100%" in out
    assert "saved=0/100 tok (0%)" in out


def test_auto_match_shared_tokens_counts_hit():
    """session_id=None with an entry that shares tokens: counts a HIT and
    matched_tokens equals the shared-prefix length (best_m > 0 path)."""
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    kv, m = s.lookup(None, [1, 2, 3, 9])
    assert kv is not None and m == 3
    assert s.hits == 1
    assert s.misses == 0
    assert s.matched_tokens == 3


def test_auto_match_zero_shared_tokens_counts_miss():
    """session_id=None with an entry that shares ZERO tokens: counts a MISS,
    unlike the named-session path (test_degenerate_hit_shows_full_hit_rate_
    but_no_savings) where the same situation counts as a HIT. This pins the
    documented asymmetry rather than leaving it accidental."""
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    kv, m = s.lookup(None, [9] * 100)
    assert kv is None and m == 0
    assert s.hits == 0
    assert s.misses == 1
    assert s.matched_tokens == 0


def test_evictions_counted_once_per_dropped_entry():
    """One store() call can drop several entries; each is a lost reuse."""
    s = PrefixCacheStore(budget_bytes=1000)
    s.store("s1", [1], _FakeCache(300))
    s.store("s2", [2], _FakeCache(300))
    s.store("s3", [3], _FakeCache(300))      # 900 bytes resident, still under
    assert s.evictions == 0
    s.store("s4", [4], _FakeCache(900))      # 1800 -> must drop s1, s2, s3
    assert s.evictions == 3


def test_stats_format():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [1, 2, 3, 9])             # hit, 3 of 4
    s.lookup("other", [7, 7])                # miss, 2 requested
    assert s.stats() == (
        "1hit/1miss/0evict hit_rate=50% saved=3/6 tok (50%)"
    )


def test_stats_with_no_lookups_is_not_a_division_error():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert s.stats() == "0hit/0miss/0evict hit_rate=n/a saved=0/0 tok (n/a)"


def test_reset_stats_zeroes_all_counters():
    s = PrefixCacheStore(budget_bytes=1000)
    s.store("s1", [1], _FakeCache(600))
    s.store("s2", [2], _FakeCache(600))      # evicts s1
    s.lookup("s2", [2])
    s.lookup("gone", [1, 2, 3])
    assert (s.hits, s.misses, s.evictions) != (0, 0, 0)
    s.reset_stats()
    assert (s.hits, s.misses, s.evictions) == (0, 0, 0)
    assert (s.matched_tokens, s.lookup_tokens) == (0, 0)


def test_reset_stats_does_not_evict_the_pool():
    """Counters are telemetry; resetting them must not touch cached KV."""
    s = PrefixCacheStore(budget_bytes=10_000)
    c = _FakeCache(500)
    s.store("s1", [1, 2, 3], c)
    s.reset_stats()
    kv, m = s.lookup("s1", [1, 2, 3])
    assert kv is c and m == 3


# --- Plan I: compress-on-evict ----------------------------------------------

def _bf16_cache(seq=64, heads=4, dim=128, layers=1):
    import torch
    from vaayu.vaayu_base import FP8DynamicCache
    c = FP8DynamicCache(quantize=False)
    for L in range(layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16)
        c.update(k, k, L)
    return c


def test_compression_off_by_default_stores_the_cache_itself():
    from vaayu.prefix_cache import PrefixCacheStore
    c = _bf16_cache(seq=8)
    s = PrefixCacheStore(budget_bytes=10 ** 9)
    s.store("a", [1, 2, 3], c)
    got, _ = s.lookup("a", [1, 2, 3])
    assert got is c


def test_compressed_store_shrinks_the_entry_and_round_trips():
    import torch
    from vaayu.prefix_cache import PrefixCacheStore
    c = _bf16_cache(seq=64)
    k = c.key_cache[0][..., :64, :].clone()
    raw = c.resident_bytes()

    s = PrefixCacheStore(budget_bytes=10 ** 9, compress_bits=8)
    s.store("a", [1, 2, 3], c)
    assert s._current_bytes < raw * 0.6          # 1.031 vs 2.000 B/elem

    got, m = s.lookup("a", [1, 2, 3], device="cpu")
    assert got is not c                           # rebuilt, not the original
    assert got.get_seq_length(0) == 64
    err = ((got.key_cache[0][..., :64, :].float() - k.float()).norm()
           / k.float().norm()).item()
    assert err < 0.05


def test_compressed_pool_holds_more_sessions_than_bf16():
    """The point of the feature: same budget, more resident sessions."""
    from vaayu.prefix_cache import PrefixCacheStore
    budget = _bf16_cache(seq=64).resident_bytes() * 3 + 1024   # ~3 bf16 sessions
    plain = PrefixCacheStore(budget_bytes=budget)
    comp = PrefixCacheStore(budget_bytes=budget, compress_bits=8)
    for i in range(6):
        plain.store(f"s{i}", [i], _bf16_cache(seq=64))
        comp.store(f"s{i}", [i], _bf16_cache(seq=64))
    assert len(comp._pool) > len(plain._pool)
    assert comp.evictions < plain.evictions


def test_lookup_without_device_on_a_compressed_entry_raises():
    """No silent degradation: never hand a caller packed data."""
    import pytest
    from vaayu.prefix_cache import PrefixCacheStore
    s = PrefixCacheStore(budget_bytes=10 ** 9, compress_bits=8)
    s.store("a", [1], _bf16_cache(seq=8))
    with pytest.raises(ValueError, match="device"):
        s.lookup("a", [1])


def test_stats_reports_compression_and_reset_zeroes_it():
    from vaayu.prefix_cache import PrefixCacheStore
    s = PrefixCacheStore(budget_bytes=10 ** 9, compress_bits=8)
    s.store("a", [1], _bf16_cache(seq=8))
    s.lookup("a", [1], device="cpu")
    line = s.stats()
    assert "compress=8b" in line and "1pack/1unpack" in line
    s.reset_stats()
    assert "0pack/0unpack" in s.stats()
    # off by default -> no compression fragment at all
    assert "compress=" not in PrefixCacheStore(budget_bytes=10 ** 9).stats()


def test_persist_runs_before_store_compresses():
    """kv_persist reads the cache directly; store() may compress it. If the
    order ever flips back, the disk file silently receives packed indices."""
    import types
    from vaayu.vaayu_base import VaayuBaseModel

    order = []

    class _Writer:
        def enqueue_append(self, sid, cache, cfg, toks, length):
            order.append(("persist", type(cache).__name__))

    class _PC:
        def store(self, sid, toks, cache):
            order.append(("store", type(cache).__name__))

    c = _bf16_cache(seq=4)
    fake = types.SimpleNamespace(
        _kv_writer=_Writer(), config=None,
        telemetry=types.SimpleNamespace(dump_json=lambda p: None))
    VaayuBaseModel._persist_and_store(fake, _PC(), "sess", c, [1, 2, 3, 4, 5])
    assert [o[0] for o in order] == ["persist", "store"]


# ── cross-session cannibalisation ───────────────────────────────────────────
#
# Found 2026-09-09 by P10 T10's arms. Two agent sessions that share a system
# prompt and nothing else destroy each other's cached context: a `lookup` for
# session B auto-matches A's entry on the shared prefix, and because a hit
# CHECKS THE ENTRY OUT of the pool, `store` then re-inserts it under B's id.
# A's ~4000-token entry is gone, so A's next turn matches only the system
# prompt again. Measured on MiniMax-M2.7 at ~4.17k tokens: the session that
# kept its entry prefilled 59 new positions (TTFT 15.0 s); the one that lost it
# prefilled 4172 (TTFT 190.0 s). Reported as 3 hits, 0 evictions, so no counter
# says anything is wrong.
#
# `config.py`'s REQUIRE_SESSION_ID says naming a session is what PREVENTS
# exactly this ("two agents sharing a system prompt land on ONE entry and evict
# each other, and reuse collapses to zero") -- these sessions were named.

def test_a_named_session_keeps_its_own_entry_when_another_borrows_the_prefix():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    a_long = [1, 2, 3] + list(range(100, 140))       # A's own long context
    s.store("A", a_long, _FakeCache(500))

    # B shares only [1,2,3] and has no entry of its own -- it may borrow, but
    # borrowing must not take A's entry away.
    kv, m = s.lookup("B", [1, 2, 3] + list(range(900, 930)))
    assert m == 3
    s.store("B", [1, 2, 3] + list(range(900, 930)), _FakeCache(500))

    _kv, m_a = s.lookup("A", a_long + [7])
    assert m_a == len(a_long), (
        f"A's own {len(a_long)}-token entry must survive B's borrow; got {m_a}")


def test_two_sessions_alternating_keep_reusing_their_own_context():
    """The agentic fan-out shape: N sessions, one shared system prompt, each
    coming back for a second turn. Every session must reuse ITS OWN context,
    not re-prefill because a sibling ran in between."""
    s = PrefixCacheStore(budget_bytes=100_000, auto_prefix_match=True)
    shared = [1, 2, 3, 4, 5]
    ctx = {sid: shared + list(range(base, base + 50))
           for sid, base in (("X", 100), ("Y", 500))}

    for sid in ("X", "Y"):                            # turn 1: cold
        s.lookup(sid, ctx[sid])
        s.store(sid, ctx[sid], _FakeCache(1000))

    for sid in ("X", "Y"):                            # turn 2: must be warm
        _kv, m = s.lookup(sid, ctx[sid] + [999])
        assert m == len(ctx[sid]), (
            f"session {sid} re-prefilled {len(ctx[sid]) - m} of its own "
            f"{len(ctx[sid])} tokens")
        s.store(sid, ctx[sid] + [999], _FakeCache(1000))


# ── the copy primitive the borrow path needs ────────────────────────────────

def test_copy_prefix_is_independent_of_its_source():
    """A borrowed prefix must not alias the owner's buffers: the borrower crops
    and then WRITES its own decode tokens into what it got back."""
    import torch
    from vaayu.vaayu_base import FP8DynamicCache

    src = FP8DynamicCache(quantize=False)
    for layer in range(2):
        k = (torch.arange(10, dtype=torch.bfloat16)
             .reshape(1, 1, 10, 1).repeat(1, 2, 1, 4))
        src.update(k, k + 100, layer, {})

    cp = src.copy_prefix(4)
    assert cp.get_seq_length() == 4
    assert src.get_seq_length() == 10, "the source keeps its full length"
    assert cp.quantize is src.quantize
    assert len(cp.key_cache) == len(src.key_cache) == 2
    assert cp.key_cache[0].shape[-2] == 4, "copies only the matched tokens"
    assert torch.equal(cp.key_cache[0][..., :4, :], src.key_cache[0][..., :4, :])
    assert torch.equal(cp.value_cache[1][..., :4, :], src.value_cache[1][..., :4, :])

    cp.key_cache[0][..., 0, :] = -1
    assert not torch.equal(cp.key_cache[0][..., :1, :], src.key_cache[0][..., :1, :]), \
        "writing the copy must not touch the source"


def test_copy_prefix_carries_the_quantized_buffers_too():
    """fp8 mode has four more buffer lists (scales + the dequantized mirror);
    a copy that dropped them would return a cache attention cannot read."""
    import torch
    from vaayu.vaayu_base import FP8DynamicCache

    src = FP8DynamicCache(quantize=True)
    k = (torch.arange(8, dtype=torch.float32)
         .reshape(1, 1, 8, 1).repeat(1, 2, 1, 4))
    src.update(k, k + 5, 0, {})

    cp = src.copy_prefix(3)
    assert cp.get_seq_length() == 3
    for name in src._BUFFER_LISTS:
        assert getattr(cp, name)[0].shape[-2] == 3, name


def test_a_continuation_still_checks_the_entry_out():
    """The unnamed single-user path: `store` resolves an auto id by HASHING the
    prompt, so every turn lands under a different key and matches its own
    previous turn through auto-match. That prompt CONTAINS the whole entry, so
    it will store a superset back -- checking out loses nobody anything, and
    copying it every turn would be pure overhead plus a pool entry per turn."""
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    turn1 = [1, 2, 3, 4]
    s.store("hash-of-turn1", turn1, _FakeCache(500))

    kv, m = s.lookup(None, turn1 + [5, 6])
    assert m == 4
    assert not hasattr(kv, "copied_from"), "a continuation is not a copy"
    assert len(s._pool) == 0, "checked out, as before"
    assert s.borrows == 0


def test_a_partial_cross_session_match_is_counted_as_a_borrow():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    s.store("A", [1, 2, 3] + list(range(50, 90)), _FakeCache(500))

    kv, m = s.lookup("B", [1, 2, 3, 77, 78])
    assert m == 3
    assert kv.copied_len == 3, "copies only what was matched"
    assert s.borrows == 1
    assert list(s._pool) == ["A"], "the owner keeps its entry"
