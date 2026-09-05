"""S1d: the union a verify pass really reads, measured off the model's routing.

The Fix #9 gate sizes a verify pass at min(num_experts, k*top_k) -- a
zero-overlap bound. Measured 2026-09-05, consecutive decode positions overlap
24-51% (Qwen) and 14-35% (MiniMax), and the bound refuses MiniMax's
draft_tokens=3, which measured 1.29x. These helpers replace the assumption
with the model's own `_expert_history`.
"""
import pytest

from vaayu.speculative.adaptive_gate import (
    observed_union_per_layer,
    union_bytes,
    spec_breakeven_acceptance,
    AdaptiveSpecGate,
)

QWEN = {"expert_bytes": 11_830_272, "num_moe_layers": 94, "top_k": 8,
        "num_experts": 128, "working_set_bytes": 8 * 94 * 11_830_272}
MINIMAX = {"expert_bytes": 8_875_020, "num_moe_layers": 62, "top_k": 8,
           "num_experts": 256, "working_set_bytes": 8 * 62 * 8_875_020}


def test_identical_routing_gives_the_single_position_union():
    """Two positions that route the same experts read one position's worth --
    the case the zero-overlap bound gets most wrong."""
    snaps = [{0: [1, 2, 3], 1: [4, 5]}, {0: [1, 2, 3], 1: [4, 5]}]
    assert observed_union_per_layer(snaps, k=2) == pytest.approx(2.5)


def test_disjoint_routing_reproduces_the_bound():
    snaps = [{0: [1, 2]}, {0: [3, 4]}]
    assert observed_union_per_layer(snaps, k=2) == pytest.approx(4.0)


def test_k_of_one_is_the_mean_per_position_width():
    snaps = [{0: [1, 2, 3]}, {0: [4, 5]}]
    assert observed_union_per_layer(snaps, k=1) == pytest.approx(2.5)


def test_windows_slide_rather_than_tile():
    """3 snapshots at k=2 give TWO windows, not one -- tiling would throw away
    half the evidence a short calibration window collected."""
    snaps = [{0: [1]}, {0: [1]}, {0: [2]}]
    # windows: {1} -> 1, {1,2} -> 2  => mean 1.5
    assert observed_union_per_layer(snaps, k=2) == pytest.approx(1.5)


def test_too_few_snapshots_for_the_window_is_zero_not_a_crash():
    """A caller that latches on 0.0 refuses; a caller that crashes takes the
    generation down. Refusing is the safe direction."""
    assert observed_union_per_layer([{0: [1]}], k=4) == 0.0
    assert observed_union_per_layer([], k=1) == 0.0


def test_a_layer_missing_from_one_snapshot_does_not_shrink_the_union():
    """Layers are unioned over the snapshots that HAVE them; a forward that
    skipped a layer must not make that layer look cheap."""
    snaps = [{0: [1, 2], 1: [7]}, {0: [3]}]
    # layer 0: {1,2,3} = 3 ; layer 1: {7} = 1 -> mean 2.0
    assert observed_union_per_layer(snaps, k=2) == pytest.approx(2.0)


@pytest.mark.parametrize("geo,expect_gb", [(QWEN, 13.53), (MINIMAX, 12.78)],
                         ids=["qwen-like", "minimax-like"])
def test_union_bytes_reproduces_the_measured_figures(geo, expect_gb):
    """The 2026-09-05 measurements: Qwen 12.164 experts/layer at k=2 and
    MiniMax 23.232 at k=4. Derived from geometry, never a stored table."""
    per_layer = 12.164 if geo is QWEN else 23.232
    assert union_bytes(per_layer, geo) / 1e9 == pytest.approx(expect_gb, abs=0.01)


def test_breakeven_matches_the_two_measured_models():
    """A round reads union(d+1) and yields 1 + accepted tokens, so it beats
    plain decode's union(1) per token when accepted/drafted exceeds this.
    Measured unions 2026-09-05: Qwen 8.90 -> 13.53 GB at d=1, MiniMax
    4.40 -> 12.78 at d=3."""
    assert spec_breakeven_acceptance(8_900_000_000, 13_530_000_000, 1) == pytest.approx(0.520, abs=0.005)
    assert spec_breakeven_acceptance(4_400_000_000, 12_780_000_000, 3) == pytest.approx(0.635, abs=0.005)


def test_a_free_verify_pass_needs_no_acceptance():
    """If a k-wide pass read no more than a 1-wide one, any acceptance wins."""
    assert spec_breakeven_acceptance(1000, 1000, 3) == 0.0


def test_zero_draft_tokens_is_unsatisfiable_not_a_zero_division():
    """d=0 is not speculation. Returning an unreachable bar makes the caller
    refuse; dividing by zero takes the generation down."""
    assert spec_breakeven_acceptance(1000, 2000, 0) == float("inf")
    assert spec_breakeven_acceptance(0, 2000, 3) == float("inf")


def _calibrate(gate, per_step_experts, layers=None):
    """Feed the gate CALIBRATE_STEPS snapshots of the given routing."""
    from vaayu.speculative import adaptive_gate as ag
    layers = layers if layers is not None else range(2)
    for step in range(ag.CALIBRATE_STEPS):
        gate.observe({l: list(per_step_experts(step)) for l in layers})


def test_it_does_not_draft_while_calibrating():
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    assert gate.should_draft() is False
    assert gate.phase == "calibrate"


def test_routing_that_fits_the_budget_moves_to_trial():
    """MiniMax-like: 8 experts per position, heavy overlap -> a 4-position
    union well inside a 14.7 GB budget."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(62))
    assert gate.phase == "trial"
    assert gate.should_draft() is True


def test_routing_that_does_not_fit_latches_off_before_drafting_once():
    """Disjoint routing at every step: the union grows with k and blows the
    budget. This is the case the old bound assumed always held."""
    gate = AdaptiveSpecGate(QWEN, draft_tokens=5, budget_bytes=1_000_000_000)
    _calibrate(gate, lambda step: range(step * 8, step * 8 + 8), layers=range(94))
    assert gate.phase == "off"
    assert gate.decision == "off_union"
    assert gate.should_draft() is False


def test_acceptance_below_the_bar_latches_off():
    """Qwen's measured regime: it fits, but only 0.286 of drafted tokens are
    accepted against a 0.520 bar, and S1f measured that at 0.836x."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(QWEN, draft_tokens=1, budget_bytes=15_800_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(94))
    assert gate.phase == "trial"
    for _ in range(ag.TRIAL_DRAFTED_TOKENS):
        gate.record_round(drafted=1, accepted=0)
    assert gate.phase == "off"
    assert gate.decision == "off_acceptance"
    assert gate.should_draft() is False


def test_acceptance_above_the_bar_runs():
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(62))
    for _ in range(ag.TRIAL_DRAFTED_TOKENS):
        gate.record_round(drafted=3, accepted=3)
    assert gate.phase == "run"
    assert gate.decision == "run"
    assert gate.should_draft() is True


def test_rounds_that_drafted_nothing_do_not_count_against_acceptance():
    """An empty round is a 1-wide decode producing one token -- economically
    identical to plain decode. Counting it would condemn a merely neutral
    workload. Qwen's 52-of-136 empty rounds are NOT what condemned it."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(62))
    for _ in range(ag.TRIAL_ROUND_CAP - 1):
        gate.record_round(drafted=0, accepted=0)
    assert gate.phase == "trial", "empty rounds must not end the trial early"
    for _ in range(ag.TRIAL_DRAFTED_TOKENS):
        gate.record_round(drafted=3, accepted=3)
    assert gate.decision == "run"


def test_the_round_cap_ends_a_trial_that_never_drafts_enough():
    """A proposer that almost never matches must not leave the gate in trial
    forever -- it decides on what it has."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(62))
    for _ in range(ag.TRIAL_ROUND_CAP):
        gate.record_round(drafted=1, accepted=0)
    assert gate.phase == "off"
    assert gate.decision == "off_acceptance"


def test_status_reports_the_numbers_it_decided_on():
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate(gate, lambda step: range(8), layers=range(62))
    s = gate.status()
    assert "decision" in s
    assert "budget_gb" in s
    assert "observed_union_gb" in s
    assert "breakeven_acceptance" in s
    assert "measured_acceptance" in s
    assert "drafting_rounds" in s
