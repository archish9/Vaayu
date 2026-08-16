"""Offline LRU replay for the expert cache (todo row 14).

Answers "what hit rate would N slots have given?" from an access trace captured
during a real decode, so the hit-rate-vs-capacity curve costs ONE GPU run instead
of one run per capacity.

The simulator must reproduce ExpertRAMCache's real policy or the curve is
fiction. That policy (expert_cache.py) is two constraints at once:
  1. per-layer cap (max_per_layer) -- evict oldest expert WITHIN that layer
  2. global budget -- evict the globally-oldest expert across ALL layers
with max_per_layer == 0 meaning flat global LRU.
"""
import pytest

from vaayu.expert_sim import derive_max_per_layer, simulate_lru, sweep_capacity


def test_an_empty_trace_reports_no_lookups():
    r = simulate_lru([], capacity=10, max_per_layer=0)
    assert r["hits"] == 0 and r["misses"] == 0
    assert r["hit_rate"] is None


def test_every_first_touch_is_a_miss():
    trace = [(0, 1), (0, 2), (1, 1)]
    r = simulate_lru(trace, capacity=10, max_per_layer=0)
    assert (r["hits"], r["misses"]) == (0, 3)


def test_an_immediate_repeat_hits():
    r = simulate_lru([(0, 1), (0, 1)], capacity=10, max_per_layer=0)
    assert (r["hits"], r["misses"]) == (1, 1)


def test_the_same_expert_in_a_different_layer_is_a_different_entry():
    r = simulate_lru([(0, 1), (1, 1)], capacity=10, max_per_layer=0)
    assert (r["hits"], r["misses"]) == (0, 2)


def test_global_lru_evicts_the_oldest_when_capacity_is_exceeded():
    # capacity 2: touch 1,2,3 -> evicts 1; touching 1 again must miss.
    trace = [(0, 1), (0, 2), (0, 3), (0, 1)]
    r = simulate_lru(trace, capacity=2, max_per_layer=0)
    assert (r["hits"], r["misses"]) == (0, 4)
    assert r["evictions"] == 2


def test_a_recently_used_entry_survives_eviction():
    # Touch 1, 2, then 1 again (refreshing it), then 3 -> 2 is evicted, not 1.
    trace = [(0, 1), (0, 2), (0, 1), (0, 3), (0, 1), (0, 2)]
    r = simulate_lru(trace, capacity=2, max_per_layer=0)
    # (0,1) hit at index 2 and 4; (0,2) evicted by (0,3) so index 5 misses.
    assert r["hits"] == 2


def test_the_per_layer_cap_binds_before_the_global_budget():
    # Global room for 10, but only 1 per layer: layer 0 can never hold two.
    trace = [(0, 1), (0, 2), (0, 1)]
    r = simulate_lru(trace, capacity=10, max_per_layer=1)
    assert (r["hits"], r["misses"]) == (0, 3)


def test_the_per_layer_cap_does_not_evict_across_layers():
    # 1 per layer, two layers -> both keep their own entry and both hit.
    trace = [(0, 1), (1, 5), (0, 1), (1, 5)]
    r = simulate_lru(trace, capacity=10, max_per_layer=1)
    assert (r["hits"], r["misses"]) == (2, 2)


def test_the_global_budget_still_binds_under_a_generous_per_layer_cap():
    # 8 per layer allowed, but only 2 entries globally.
    trace = [(0, 1), (1, 1), (2, 1), (0, 1)]
    r = simulate_lru(trace, capacity=2, max_per_layer=8)
    assert (r["hits"], r["misses"]) == (0, 4)


def test_hit_rate_is_reported_as_a_fraction():
    r = simulate_lru([(0, 1), (0, 1), (0, 1), (0, 1)], capacity=10, max_per_layer=0)
    assert r["hit_rate"] == pytest.approx(0.75)


# ── the derivation must match vaayu_base.compute_expert_budget ───────────────

def test_max_per_layer_matches_the_engines_own_derivation():
    from vaayu.vaayu_base import compute_expert_budget

    # Qwen-like and MiniMax-like geometries (CLAUDE.md §5: test both).
    for budget, layers, expert_bytes, top_k, n_exp in (
        (9_120_000_000, 94, 11_830_000, 8, 128),      # Qwen3-235B
        (9_120_000_000, 62, 8_880_000, 8, 256),       # MiniMax-M2.7
    ):
        _, _, engine_mpl = compute_expert_budget(
            available_bytes=32 * 1024 ** 3, working_set_bytes=0,
            num_moe_layers=layers, expert_bytes=expert_bytes,
            moe_top_k=top_k, moe_num_experts=n_exp)
        # The engine derives max_per_layer from the budget it computed itself;
        # ours must use the same expression given the same budget.
        ours = derive_max_per_layer(budget_bytes=budget, num_moe_layers=layers,
                                    expert_bytes=expert_bytes, top_k=top_k,
                                    num_experts=n_exp)
        assert isinstance(ours, int) and ours >= top_k
        assert isinstance(engine_mpl, int)


def test_max_per_layer_never_drops_below_top_k():
    # A tiny budget must still admit one token's routing, or decode cannot run.
    assert derive_max_per_layer(budget_bytes=1, num_moe_layers=94,
                                expert_bytes=11_830_000, top_k=8,
                                num_experts=128) == 8


def test_max_per_layer_never_exceeds_the_expert_count():
    assert derive_max_per_layer(budget_bytes=10 ** 13, num_moe_layers=94,
                                expert_bytes=11_830_000, top_k=8,
                                num_experts=128) == 128


def test_a_bigger_budget_yields_a_bigger_per_layer_cap():
    small = derive_max_per_layer(budget_bytes=9_120_000_000, num_moe_layers=94,
                                 expert_bytes=11_830_000, top_k=8, num_experts=128)
    big = derive_max_per_layer(budget_bytes=25_000_000_000, num_moe_layers=94,
                               expert_bytes=11_830_000, top_k=8, num_experts=128)
    assert big > small


# ── the sweep ────────────────────────────────────────────────────────────────

def test_the_sweep_reports_one_row_per_budget_and_hit_rate_is_monotone():
    # More cache can never hit less on the same trace under LRU.
    trace = [(l, (t * 7 + l) % 16) for t in range(40) for l in range(4)]
    rows = sweep_capacity(trace, budgets_bytes=[1e9, 2e9, 4e9, 8e9],
                          num_moe_layers=4, expert_bytes=100_000_000,
                          top_k=2, num_experts=16)
    assert len(rows) == 4
    rates = [r["hit_rate"] for r in rows]
    assert rates == sorted(rates), f"hit rate must be monotone in capacity: {rates}"


def test_the_sweep_carries_the_derived_geometry_for_each_row():
    rows = sweep_capacity([(0, 1)], budgets_bytes=[1e9],
                          num_moe_layers=4, expert_bytes=100_000_000,
                          top_k=2, num_experts=16)
    r = rows[0]
    assert "budget_gb" in r and "max_per_layer" in r and "capacity" in r


# ── the trace hook must record what the cache actually saw ───────────────────

def test_the_trace_hook_records_every_lookup_hit_or_miss():
    from vaayu.persist.expert_cache import ExpertRAMCache

    c = ExpertRAMCache(budget_bytes=10 ** 9)
    assert c.take_trace() == []            # off by default -> nothing recorded
    c.enable_trace()
    c.get(3, 7)
    c.get(3, 7)
    c.get(1, 2)
    assert c.take_trace() == [(3, 7), (3, 7), (1, 2)]


def test_take_trace_stops_recording():
    from vaayu.persist.expert_cache import ExpertRAMCache

    c = ExpertRAMCache(budget_bytes=10 ** 9)
    c.enable_trace()
    c.get(0, 1)
    c.take_trace()
    c.get(0, 2)
    assert c.take_trace() == []


def test_replaying_a_real_caches_own_trace_reproduces_its_hit_count():
    # The simulator is only trustworthy if it agrees with the thing it models.
    from vaayu.persist.expert_cache import ExpertRAMCache

    real = ExpertRAMCache(budget_bytes=4 * 8, max_per_layer=2)  # tiny, forces eviction
    real.enable_trace()
    pattern = [(l, (i * 3 + l) % 5) for i in range(30) for l in range(3)]
    for layer, expert in pattern:
        if real.get(layer, expert) is None:
            real.put(layer, expert, {"w": _one_byte_tensor()})
    trace = real.take_trace()

    sim = simulate_lru(trace, capacity=4, max_per_layer=2)
    assert sim["hits"] == real.hits, (sim, real.hits, real.misses)
    assert sim["misses"] == real.misses


def _one_byte_tensor():
    import torch
    return torch.zeros(8, dtype=torch.uint8)


# ── eviction pinning (group_protect) ─────────────────────────────────────────

def test_group_protect_refuses_to_evict_an_entry_touched_in_the_current_group():
    # One layer, cap 2, a group of 3 lookups: without protection the first would
    # be evicted by the third. The engine pins checked-out entries, so it is not.
    trace = [(0, 1), (0, 2), (0, 3)]
    naive = simulate_lru(trace, capacity=2, max_per_layer=0, group_protect=False)
    pinned = simulate_lru(trace, capacity=2, max_per_layer=0, group_protect=True)
    assert naive["evictions"] == 1
    assert pinned["evictions"] == 0, "all three are checked out in the same group"


def test_pins_are_released_when_the_layer_changes():
    # Group 1 = layer 0, group 2 = layer 1. Layer 0's pins drop at the boundary,
    # so layer 1's inserts may evict them.
    trace = [(0, 1), (0, 2), (1, 5), (1, 6), (1, 7)]
    r = simulate_lru(trace, capacity=2, max_per_layer=0, group_protect=True)
    assert r["evictions"] > 0


def test_group_protect_never_lowers_the_hit_rate():
    trace = [(l, (t * 5 + i) % 12) for t in range(20) for l in range(3) for i in range(4)]
    naive = simulate_lru(trace, capacity=6, max_per_layer=4, group_protect=False)
    pinned = simulate_lru(trace, capacity=6, max_per_layer=4, group_protect=True)
    assert pinned["hit_rate"] >= naive["hit_rate"]


def test_the_sweep_models_pinning_by_default():
    # The engine pins; a sweep that did not model it would under-predict every
    # row, which is exactly the anchor mismatch this option exists to fix.
    trace = [(0, 1), (0, 2), (0, 3)]
    default = sweep_capacity(trace, budgets_bytes=[2e8], num_moe_layers=1,
                             expert_bytes=1e8, top_k=2, num_experts=8)
    off = sweep_capacity(trace, budgets_bytes=[2e8], num_moe_layers=1,
                         expert_bytes=1e8, top_k=2, num_experts=8,
                         group_protect=False)
    assert default[0]["hit_rate"] >= off[0]["hit_rate"]
