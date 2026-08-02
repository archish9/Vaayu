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
