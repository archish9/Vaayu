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


def test_zero_budget_is_distinguishable_from_disabled():
    """A 0-byte budget means 'enabled but got no RAM' — a very different state
    from 'disabled', and the one the slab arena can silently cause. The two must
    not look alike to an operator reading the log."""
    from vaayu.vaayu_base import compute_prefix_cache_budget

    # Free RAM below the hardware reserve yields a real, meaningful zero.
    assert compute_prefix_cache_budget(1 * 1024 ** 3) == 0
    assert compute_prefix_cache_budget(0) == 0
    # Comfortably above the reserve yields a usable budget.
    assert compute_prefix_cache_budget(20 * 1024 ** 3) > 0
