"""S12 T1: where the decode graphs' memory pool comes from (pure), and the expert VRAM
tier shrinking to pay for it. CPU only: no model, no GPU."""
import pytest
import torch

from vaayu.decode_graphs import graph_pool_charge
from vaayu.persist.expert_cache import ExpertVRAMCache

# A tier as (bytes per expert, experts resident): Qwen3-Next-like and MiniMax-like
# (CLAUDE.md §5 -- a sizing helper is proven on two contrasting geometries).
GEOMETRIES = {"qwen3next_like": (1_769_520, 4251), "minimax_like": (8 * 1024 ** 2, 700)}


@pytest.mark.parametrize("geom", GEOMETRIES.values(), ids=GEOMETRIES.keys())
def test_a_pool_the_tier_covers_shrinks_the_tier(geom):
    per_expert, resident = geom
    tier = per_expert * resident
    pool = tier // 11
    assert graph_pool_charge(pool_bytes=pool, tier_budget=tier, reserve_bytes=0,
                             available_bytes=0) == ("tier", tier - pool)


@pytest.mark.parametrize("geom", GEOMETRIES.values(), ids=GEOMETRIES.keys())
def test_a_pool_larger_than_the_tier_is_refused(geom):
    per_expert, resident = geom
    tier = per_expert * resident
    assert graph_pool_charge(pool_bytes=tier + 1, tier_budget=tier, reserve_bytes=0,
                             available_bytes=10 * tier) == ("refuse", None)


@pytest.mark.parametrize("geom", GEOMETRIES.values(), ids=GEOMETRIES.keys())
def test_an_empty_pool_charges_nothing(geom):
    per_expert, resident = geom
    tier = per_expert * resident
    assert graph_pool_charge(pool_bytes=0, tier_budget=tier, reserve_bytes=0,
                             available_bytes=0) == ("tier", tier)


def test_without_a_tier_the_reserve_must_stay_whole():
    reserve = 3 * 1024 ** 3
    assert graph_pool_charge(pool_bytes=1 << 20, tier_budget=None, reserve_bytes=reserve,
                             available_bytes=reserve) == ("reserve", None)
    assert graph_pool_charge(pool_bytes=1 << 20, tier_budget=None, reserve_bytes=reserve,
                             available_bytes=reserve - 1) == ("refuse", None)


def _entry(nbytes):
    return {"gate": {"w": torch.zeros(nbytes, dtype=torch.uint8)}}


def test_set_budget_evicts_lru_until_usage_fits():
    cache = ExpertVRAMCache(budget_bytes=100)
    for expert in range(5):
        assert cache.put(0, expert, _entry(20))
    assert cache.set_budget(50) == 3
    assert cache.budget_bytes == 50 and cache.current_usage == 40
    assert list(cache.cache) == [(0, 3), (0, 4)], "LRU order: the oldest go first"
    assert cache.evictions == 3


def test_set_budget_above_usage_evicts_nothing():
    cache = ExpertVRAMCache(budget_bytes=100)
    cache.put(0, 0, _entry(20))
    assert cache.set_budget(60) == 0
    assert cache.current_usage == 20 and cache.budget_bytes == 60
