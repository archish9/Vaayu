"""An explicit session_id that misses must still auto-prefix-match.

Before this fix, naming a session and missing it was STRICTLY WORSE than not
naming it: the auto-match branch was gated on `session_id is None`.
"""
from vaayu.prefix_cache import PrefixCacheStore


class FakeCache:
    """Minimal stand-in for the three methods PrefixCacheStore calls."""

    def __init__(self, nbytes=1000):
        self._nbytes = nbytes
        self.on_cpu = False

    def to_cpu(self):
        self.on_cpu = True

    def resident_bytes(self):
        return self._nbytes


def _store():
    return PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=True)


def test_an_explicit_id_that_hits_is_used_directly():
    s = _store()
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, m = s.lookup("agent-a", [1, 2, 3, 4, 5])
    assert cache is not None
    assert m == 4


def test_an_explicit_id_that_misses_falls_back_to_the_best_prefix_match():
    s = _store()
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, m = s.lookup("agent-b", [1, 2, 3, 4, 5, 6])
    assert cache is not None, "an explicit miss must not throw away a usable prefix"
    assert m == 4
    assert s.hits == 1


def test_the_fallback_picks_the_longest_common_prefix_not_the_first_entry():
    s = _store()
    s.store("short", [1, 2], FakeCache())
    s.store("long", [1, 2, 3, 4, 5], FakeCache())
    _, m = s.lookup("unknown", [1, 2, 3, 4, 5, 9])
    assert m == 5


def test_a_genuine_miss_with_no_shared_prefix_is_still_a_miss():
    s = _store()
    s.store("agent-a", [1, 2, 3], FakeCache())
    cache, m = s.lookup("agent-b", [90, 91, 92])
    assert cache is None
    assert m == 0
    assert s.misses == 1


def test_auto_prefix_match_disabled_keeps_an_explicit_miss_a_miss():
    s = PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=False)
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, _ = s.lookup("agent-b", [1, 2, 3, 4, 5])
    assert cache is None


def test_an_empty_pool_is_a_miss_not_a_crash():
    s = _store()
    cache, m = s.lookup("agent-a", [1, 2, 3])
    assert (cache, m) == (None, 0)
