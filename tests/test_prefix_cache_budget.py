from vaayu.vaayu_base import compute_prefix_cache_budget


def test_budget_reserves_headroom():
    free = 20 * 1024 ** 3          # 20 GB free
    b = compute_prefix_cache_budget(free)
    assert 0 < b < free            # leaves headroom, never all of RAM


def test_budget_floor_when_low_ram():
    b = compute_prefix_cache_budget(1 * 1024 ** 3)   # only 1 GB free
    assert b >= 0                  # never negative; may be a small floor


def test_budget_monotonic():
    assert (compute_prefix_cache_budget(30 * 1024 ** 3)
            > compute_prefix_cache_budget(10 * 1024 ** 3))
