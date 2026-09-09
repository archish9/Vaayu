"""peek_match: what the pool WOULD serve, without checking the entry out.

lookup() pops on a hit -- only store() puts the entry back -- so the scheduler
cannot use it to price a formation decision. peek_match answers the same
question and changes nothing: no pop, no counters.
"""
from vaayu.prefix_cache import PrefixCacheStore


class FakeCache:
    """Minimum surface store() drives: to_cpu + resident_bytes."""

    def __init__(self, nbytes=1024):
        self._nbytes = nbytes

    def to_cpu(self):
        pass

    def to_device(self, device):
        pass

    def resident_bytes(self):
        return self._nbytes


def _store(pool, session_id, ids):
    pool.store(session_id, list(ids), FakeCache())


def test_peek_reports_the_match_it_would_serve():
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    _store(pool, "a", [1, 2, 3, 4, 5])
    assert pool.peek_match("a", [1, 2, 3, 9, 9]) == 3


def test_peek_does_not_check_the_entry_out():
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    _store(pool, "a", [1, 2, 3, 4, 5])
    pool.peek_match("a", [1, 2, 3])
    cache, m = pool.lookup("a", [1, 2, 3, 4, 5])
    assert cache is not None, "peek must not pop the entry it measured"
    assert m == 5


def test_peek_leaves_the_counters_alone():
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    _store(pool, "a", [1, 2, 3])
    before = (pool.hits, pool.misses, pool.lookup_tokens)
    pool.peek_match("a", [1, 2, 3])
    pool.peek_match("nobody", [7, 7, 7])
    assert (pool.hits, pool.misses, pool.lookup_tokens) == before


def test_peek_falls_back_to_the_best_auto_match_like_lookup_does():
    """An explicit id that MISSES must still see the pool -- naming a session
    and missing is not allowed to be worse than not naming one (the same rule
    lookup() carries)."""
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    _store(pool, "other", [1, 2, 3, 4])
    assert pool.peek_match("unknown-session", [1, 2, 3, 9]) == 3


def test_peek_prefers_the_named_session_over_a_longer_stranger():
    """Same key selection lookup() uses, so the two cannot disagree about
    which entry would serve."""
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    _store(pool, "mine", [1, 2])
    _store(pool, "stranger", [1, 2, 3, 4, 5])
    assert pool.peek_match("mine", [1, 2, 3, 4, 5]) == 2


def test_peek_on_an_empty_pool_is_zero():
    pool = PrefixCacheStore(budget_bytes=10 ** 9)
    assert pool.peek_match("a", [1, 2, 3]) == 0


def test_peek_with_auto_match_off_and_an_unknown_session_is_zero():
    pool = PrefixCacheStore(budget_bytes=10 ** 9, auto_prefix_match=False)
    _store(pool, "other", [1, 2, 3])
    assert pool.peek_match("unknown", [1, 2, 3]) == 0
