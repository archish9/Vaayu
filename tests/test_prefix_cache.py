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
