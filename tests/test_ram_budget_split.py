"""Characterizes how free RAM is split between the expert cache and the prefix
cache. These tests document the pre- and post-rebalance split so the change is
a visible, intentional one rather than an accident."""
import pytest
from vaayu.vaayu_base import (
    compute_expert_budget, compute_prefix_cache_budget,
    _PREFIX_CACHE_RAM_RESERVE, _EXPERT_CACHE_OOM_CEILING,
)

GB = 1024 ** 3

# Geometry stand-ins for the two on-device targets. Values are illustrative
# shapes, not asserted model facts — the point is that BOTH are exercised.
QWEN_LIKE = dict(working_set_bytes=int(8.9 * GB), num_moe_layers=94,
                 expert_bytes=int(11.28 * 1e6), moe_top_k=8, moe_num_experts=128)
MINIMAX_LIKE = dict(working_set_bytes=int(4.4 * GB), num_moe_layers=62,
                    expert_bytes=int(8.9 * 1e6), moe_top_k=8, moe_num_experts=256)


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_expert_budget_is_capped_by_the_oom_ceiling(geo):
    """With plenty of free RAM the expert cache stops at the hardware ceiling."""
    expert, _ne, _mpl = compute_expert_budget(available_bytes=30 * GB, **geo)
    assert expert <= _EXPERT_CACHE_OOM_CEILING


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_expert_budget_shrinks_when_ram_is_tight(geo):
    """The budget tracks available RAM below the ceiling."""
    big, _, _ = compute_expert_budget(available_bytes=30 * GB, **geo)
    small, _, _ = compute_expert_budget(available_bytes=6 * GB, **geo)
    assert small < big


def test_prefix_budget_is_zero_below_the_reserve():
    """CURRENT behavior: free RAM under the reserve yields a 0 budget."""
    assert compute_prefix_cache_budget(_PREFIX_CACHE_RAM_RESERVE - 1) == 0
    assert compute_prefix_cache_budget(0) == 0


def test_prefix_budget_is_free_minus_reserve_above_it():
    free = _PREFIX_CACHE_RAM_RESERVE + 3 * GB
    assert compute_prefix_cache_budget(free) == 3 * GB


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_the_two_budgets_leave_room_for_the_prefix_cache(geo):
    """After the rebalance: once the expert cache (and its mirroring pinned
    slab arena) take their share of a realistic free-RAM figure, the prefix
    cache still gets a real, non-zero budget."""
    free_at_start = 24 * GB          # measured: ~24.5GB free at cache-sizing
    expert, ne, _ = compute_expert_budget(available_bytes=free_at_start, **geo)
    # The expert cache AND its mirroring pinned slab arena (same budget) both
    # come out of free RAM, so subtract the expert budget twice.
    free_for_prefix = free_at_start - expert - expert - ne
    assert compute_prefix_cache_budget(free_for_prefix) > 0


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_prefix_cache_gets_a_real_budget_after_rebalance(geo):
    """After the rebalance, a realistic post-expert-cache free-RAM figure must
    leave the prefix cache enough for MORE THAN ONE resident session — the whole
    point, since budget 0 already serves one (the pool keeps >=1 entry)."""
    free_at_start = 24 * GB
    expert, ne, _ = compute_expert_budget(available_bytes=free_at_start, **geo)
    # The expert cache AND its mirroring pinned slab arena (same budget) both
    # come out of free RAM, so subtract the expert budget twice.
    free_for_prefix = free_at_start - expert - expert - ne
    budget = compute_prefix_cache_budget(free_for_prefix)
    assert budget > 0, "prefix cache still starved"
    # A ~2000-token KV entry is on the order of a few hundred MB; require room
    # for at least two so concurrency is genuinely served, not just one entry.
    assert budget >= 1 * GB
