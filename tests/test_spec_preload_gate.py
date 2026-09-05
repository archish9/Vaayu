"""S1a: refuse speculation on a MoE target BEFORE the drafter is loaded.

Today `build_speculator` runs first (deliberately -- the expert VRAM cache
must observe the drafter as a real resident cost, bug 12a), and the Fix #9
gate below it needs `_expert_cache.budget_bytes`, which only exists after the
load. So on both actively-tested MoE models the drafter is fully loaded to
VRAM and immediately discarded: wasted load time and I/O on every start.

The circularity is broken by gating on an UPPER BOUND of the budget rather
than the budget itself. The bound asks the real sizing helpers what they would
return under the most generous pre-load assumptions, so it can only over-state
the budget -- and therefore can never refuse a configuration the post-load
gate would have accepted. That invariant is what most of these tests are
about.
"""
import pytest

from vaayu.vaayu_base import (compute_spec_working_set, max_draft_tokens_that_fit,
                              spec_budget_upper_bound, spec_decode_fits,
                              spec_preload_refusal)

MB = 1024 ** 2
GB = 1024 ** 3


def _geo(layers, experts, expert_mb, top_k=8):
    return {"expert_bytes": int(expert_mb * MB), "num_moe_layers": layers,
            "top_k": top_k, "num_experts": experts,
            "working_set_bytes": top_k * layers * int(expert_mb * MB)}


# The two on-device MoE geometries, plus two hypotheticals that exercise the
# branches they do not. These values identify the measured cases (CLAUDE.md
# 5); every helper under test reads its numbers off the geometry it is given.
QWEN = _geo(94, 128, 11.28)
MINIMAX = _geo(62, 256, 8.46)
# A genuine boundary case: num_experts is far above top_k, so the working set
# keeps growing with positions instead of saturating.
MID = _geo(40, 64, 8.0)
# Saturated: one position already routes every expert there is, so no number
# of verify positions costs more than one does.
SATURATED = _geo(12, 8, 4.0)

# RAM free at load time on this box, in the ballpark the real loads report.
# An arm variable, not a hardware constant -- the helpers take it as an
# argument precisely so a different box gets a different answer.
AVAIL = 16 * GB


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen-like", "minimax-like"])
def test_the_two_moe_targets_are_refused_before_any_load(geo):
    reason = spec_preload_refusal(geo, draft_tokens=5, available_bytes=AVAIL)
    assert reason, "the gate that fires after the load must fire before it too"
    # Unactionable without the numbers it was decided on.
    assert "GB" in reason and "draft_tokens" in reason


def test_a_configuration_that_fits_is_not_refused():
    """The gate must not become 'no speculation on MoE, ever'. MID at
    draft_tokens=2 is 3 verify positions and fits the bound with room."""
    assert spec_preload_refusal(MID, draft_tokens=2, available_bytes=AVAIL) is None


@pytest.mark.parametrize("geo", [QWEN, MINIMAX, MID, SATURATED],
                         ids=["qwen-like", "minimax-like", "mid", "saturated"])
@pytest.mark.parametrize("available_gb", [4, 16, 30, 64])
@pytest.mark.parametrize("draft_tokens", [1, 3, 5, 12])
def test_the_preload_gate_never_refuses_what_the_postload_gate_accepts(
        geo, available_gb, draft_tokens):
    """THE invariant. The pre-load gate sees an upper bound on the budget; the
    post-load gate sees the real one, which is never larger. So a refusal here
    must imply a refusal there, for every budget the sizing code could
    produce. If this fails, the bound is wrong and the gate is silently
    turning off a configuration that works."""
    bound = spec_budget_upper_bound(available_gb * GB, geo)
    if spec_preload_refusal(geo, draft_tokens, available_gb * GB) is not None:
        for budget in (0, 1 * GB, bound // 2, bound):
            assert not spec_decode_fits(geo, draft_tokens, budget), \
                f"refused before the load, but would have fit at {budget} bytes"


@pytest.mark.parametrize("geo", [QWEN, MINIMAX, MID, SATURATED],
                         ids=["qwen-like", "minimax-like", "mid", "saturated"])
def test_the_bound_never_exceeds_the_ram_it_was_derived_from(geo):
    """It is an upper bound on a cache that lives in host RAM, so RAM is its
    own ceiling; and more free RAM may never produce a smaller bound, or the
    gate would refuse more on a bigger box."""
    assert spec_budget_upper_bound(AVAIL, geo) <= AVAIL
    assert (spec_budget_upper_bound(8 * GB, geo)
            <= spec_budget_upper_bound(32 * GB, geo))


def test_max_draft_tokens_finds_the_real_boundary():
    bound = spec_budget_upper_bound(AVAIL, MID)
    n = max_draft_tokens_that_fit(MID, bound)
    assert n >= 1
    assert spec_decode_fits(MID, n, bound)
    assert not spec_decode_fits(MID, n + 1, bound)


def test_max_draft_tokens_stops_at_saturation_rather_than_counting_forever():
    """Once one position routes every expert in a layer, more positions cost
    nothing more -- the search is bounded by the geometry, and the value
    returned is the saturation point, beyond which every larger value also
    fits."""
    bound = spec_budget_upper_bound(AVAIL, SATURATED)
    n = max_draft_tokens_that_fit(SATURATED, bound)
    assert n == 1                       # ceil(num_experts / top_k)
    assert spec_decode_fits(SATURATED, n + 50, bound)
    assert (compute_spec_working_set(SATURATED, n + 50)
            == compute_spec_working_set(SATURATED, n + 1))


def test_the_qwen_like_refusal_names_the_alternative_that_actually_helps():
    """D3: 'reduce draft_tokens' was the only advice given, and on this
    geometry it is advice that cannot work -- even one drafted token doubles
    the per-layer expert union past the budget. The lever for this workload is
    batched decode, so the message must say so instead of sending the user
    round a loop."""
    reason = spec_preload_refusal(QWEN, draft_tokens=5, available_bytes=AVAIL)
    assert "no draft_tokens" in reason.lower()
    assert "batch" in reason.lower()


def test_the_minimax_like_refusal_names_the_value_that_would_fit():
    """The other branch, and the one the doc's 'refused by a smaller margin'
    note is about: MiniMax's experts are small enough that a LOWER
    draft_tokens fits the same budget. Fitting is not the same as paying --
    that is S1c's question -- but the refusal must not claim nothing fits when
    something does."""
    bound = spec_budget_upper_bound(AVAIL, MINIMAX)
    viable = max_draft_tokens_that_fit(MINIMAX, bound)
    assert viable >= 1, "premise: this geometry has a fitting draft_tokens"
    reason = spec_preload_refusal(MINIMAX, draft_tokens=5, available_bytes=AVAIL)
    assert f"draft_tokens that could fit is {viable}" in reason
    # And it must not promise that value works: the bound over-states the
    # budget, so the post-load gate may refuse it (measured on Qwen3-235B,
    # 2026-09-05: bound 20.6 GB named draft_tokens=1, real budget 15.8 GB).
    assert "upper bound" in reason and "post-load" in reason
