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
    wall_breakeven_acceptance,
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


def _calibrate_with_overlap(gate, layers):
    """Feed the gate overlapping expert routing: step i routes range(i, i+8).

    Each step adds one new expert, so union(k) = 7 + k experts per layer.
    This produces a meaningful (non-zero) break-even, unlike identical routing.
    """
    from vaayu.speculative import adaptive_gate as ag
    for step in range(ag.CALIBRATE_STEPS):
        gate.observe({l: list(range(step, step + 8)) for l in layers})


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


def test_overlapping_routing_computes_a_meaningful_breakeven():
    """With overlapping routing (step i -> range(i, i+8)), union(k) = 7+k.
    breakeven = (union(k) - union(1)) / union(1) / draft_tokens
             = ((k+7) - 8) / 8 / draft_tokens
             = (draft_tokens) / 8 / draft_tokens = 1/8 = 0.125.
    This holds for any model and draft_tokens (k = draft_tokens+1 always)."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_with_overlap(gate, layers=range(62))
    assert gate.phase == "trial"
    assert gate._breakeven == pytest.approx(0.125, abs=0.001)


def test_acceptance_below_the_bar_latches_off():
    """With overlapping routing (breakeven=0.125), 0 accepted tokens clearly
    fails the bar and the gate turns off."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(QWEN, draft_tokens=1, budget_bytes=15_800_000_000)
    _calibrate_with_overlap(gate, layers=range(94))
    assert gate.phase == "trial"
    for _ in range(ag.TRIAL_DRAFTED_TOKENS):
        gate.record_round(drafted=1, accepted=0)
    assert gate.phase == "off"
    assert gate.decision == "off_acceptance"
    assert gate.should_draft() is False


def test_acceptance_above_the_bar_runs():
    """With overlapping routing (breakeven=0.125), perfect acceptance (1.0)
    clearly exceeds the bar and the gate runs."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate_with_overlap(gate, layers=range(62))
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
    # Calibration alone (no record_round calls yet) moves the gate to
    # "trial" without deciding run/off -- identical routing every step keeps
    # the union well under budget, so this fixture's decision is None, not
    # yet one of the terminal values.
    assert s["decision"] is None
    assert s["budget_gb"] == pytest.approx(14.7, abs=0.05)
    assert s["observed_union_gb"] > 0
    assert s["breakeven_acceptance"] is not None and s["breakeven_acceptance"] >= 0
    assert "measured_acceptance" in s and "drafting_rounds" in s


def _calibrate_needed(gate, draft_tokens):
    """Feed exactly the calibration length THIS gate needs -- never less than
    the CALIBRATE_STEPS floor, never less than one union window
    (draft_tokens + 1) -- using the same overlapping routing as
    _calibrate_with_overlap so the union is a real, non-degenerate number
    rather than the all-identical-routing case above."""
    from vaayu.speculative import adaptive_gate as ag
    needed = max(ag.CALIBRATE_STEPS, draft_tokens + 1)
    for step in range(needed):
        gate.observe({l: list(range(step, step + 8)) for l in range(62)})


@pytest.mark.parametrize("draft_tokens", [1, 3, 8, 12])
def test_a_wide_verify_window_is_not_starved_of_calibration_data(draft_tokens):
    """Fix round 1: CALIBRATE_STEPS=8 is a FLOOR, not the calibration length.
    Before this fix, draft_tokens >= CALIBRATE_STEPS meant _decide_union asked
    observed_union_per_layer for a (draft_tokens+1)-wide window off only
    CALIBRATE_STEPS snapshots -- too few for even one window, so it always
    returned 0.0 and every load at that width latched off_union regardless of
    budget. A generous budget must reach "trial" at every width, including
    draft_tokens >= CALIBRATE_STEPS."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=draft_tokens, budget_bytes=10**12)
    _calibrate_needed(gate, draft_tokens)
    assert gate.phase == "trial", (
        f"draft_tokens={draft_tokens} starved of calibration data instead of "
        f"being judged on budget: {gate.status()}")


def test_a_wide_window_that_genuinely_does_not_fit_still_refuses():
    """The fix must not turn union-fit admission into a rubber stamp: a wide
    window measured correctly (enough snapshots for its window) but bigger
    than the budget still latches off_union."""
    draft_tokens = 12
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=draft_tokens, budget_bytes=1)
    _calibrate_needed(gate, draft_tokens)
    assert gate.phase == "off"
    assert gate.decision == "off_union"


def test_empty_routing_history_reports_a_distinct_decision_not_off_union():
    """A forward that stops writing _expert_history (design doc S8) makes
    every calibration snapshot {} -- observed_union_per_layer then has no
    layers to union over and returns 0.0, same as `_decide_union` used to
    read for a genuinely over-budget union before 800467d's fix covered the
    starved-window case. This is the OTHER way union_k reads 0: no evidence
    at all, not "measured zero and it exceeded the budget" (impossible --
    that requires union_k > budget_bytes > 0). It must not be reported as
    off_union, which a reader would take as a real budget refusal."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    for _ in range(max(8, 3 + 1)):
        gate.observe({})
    assert gate.phase == "off"
    assert gate.decision == "off_no_routing"
    s = gate.status()
    assert s["observed_union_gb"] == 0.0
    assert s["decision"] == "off_no_routing"


def test_an_off_union_refusal_reports_a_real_measurement_not_a_zero():
    """observed_union_gb == 0.0 used to be indistinguishable between "measured
    a real union of zero" (impossible) and "never got enough snapshots to
    measure at all" -- exactly the silent failure mode this fix closes. A
    refusal on union grounds must report the union it actually measured."""
    draft_tokens = 12
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=draft_tokens, budget_bytes=1)
    _calibrate_needed(gate, draft_tokens)
    s = gate.status()
    assert s["decision"] == "off_union"
    assert s["observed_union_gb"] > 0


# ---------------------------------------------------------------------------
# Row 13: the wall-clock bar. The byte break-even prices a verify pass in
# EXPERT BYTES and nothing times a clock, so the gate admitted -- and latched
# -- a configuration measured at -12.9% total / -16.0% decode on MiniMax-M2.7
# (2026-09-07, acceptance 0.6875 against a byte bar of 0.6259). The spike
# (docs/2026-09-10-row13-timed-trial-SPIKE.md) timed real rounds: a
# (d+1)-position verify costs ~(d+1)*alpha plain-decode rounds with alpha in
# 0.725-0.784, so the honest bar is ((1+d)*alpha - 1)/d -- 0.633-0.710 in
# every arm, i.e. systematically ABOVE the byte bar of 0.6259. The byte test
# stays a NECESSARY condition: it is free, and it is what refuses Qwen at
# zero cost, before a single draft is paid for.
# ---------------------------------------------------------------------------

# The MiniMax-M2.7 byte break-even measured at draft_tokens=3 on 2026-09-07
# (P9's end-to-end arm). Fixing it here rather than deriving it from synthetic
# routing is deliberate: the arms below replay a measurement, and an integer
# routing fixture cannot reproduce this ratio.
MINIMAX_BYTE_BAR = 0.6259


def _calibrate_timed(gate, base_s, layers=range(62)):
    """CALIBRATE with a per-round wall-clock delta on every observation."""
    from vaayu.speculative import adaptive_gate as ag
    needed = max(ag.CALIBRATE_STEPS, gate.draft_tokens + 1)
    base = list(base_s) if isinstance(base_s, (list, tuple)) else [base_s] * needed
    for step in range(needed):
        gate.observe({l: list(range(step, step + 8)) for l in layers},
                     elapsed_s=base[step % len(base)])


def _trial_timed(gate, wide_s, acceptance, drafted_per_round=3,
                 draft_overhead_s=None):
    """TRIAL at a given acceptance, timing every drafting round.

    Acceptance is realised exactly: the total accepted count is spread over
    the drafting rounds, never exceeding what each round drafted.

    Follows the gate's own leg schedule (row 13, `even-4`) until it decides.
    Suppressed rounds are left UNTIMED, so no pair forms and these tests
    exercise the single-window rule -- alpha against the wall and byte bars
    -- which is what decides when fewer than TRIAL_PAIRS pairs are timed.
    Drafting-leg rounds past the helper's own `rounds` are untimed no-match
    rounds, so they move neither acceptance nor any median.
    """
    from vaayu.speculative import adaptive_gate as ag
    rounds = ag.TRIAL_DRAFTED_TOKENS // drafted_per_round
    wide = list(wide_s) if isinstance(wide_s, (list, tuple)) else [wide_s] * rounds
    remaining = round(acceptance * drafted_per_round * rounds)
    i = 0
    while gate.phase == "trial":
        if gate.should_draft() and i < rounds:
            take = min(drafted_per_round, remaining)
            remaining -= take
            gate.record_round(drafted_per_round, take,
                              elapsed_s=wide[i % len(wide)],
                              draft_overhead_s=draft_overhead_s)
            i += 1
        else:
            gate.record_round(0, 0)


def _wide_s_for(alpha, base_s, draft_tokens):
    """The per-round wide time that produces this alpha at this base time."""
    return alpha * (1 + draft_tokens) * base_s


def test_alpha_is_the_median_ratio_per_position():
    """alpha = median(wide) / (median(base) * (1 + draft_tokens)) -- the
    per-position cost of a verify pass in plain-decode units."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)      # 1.5 / (0.5 * 4)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


def test_alpha_uses_medians_not_means():
    """One NVMe stall must not move a verdict: the spike saw single-round
    outliers at 1.35x the median. The mean of this fixture is 2.34, not 1.5."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=[0.5] * 7 + [5.0])
    _trial_timed(gate, wide_s=[15.0] + [1.5] * 15, acceptance=1.0)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


@pytest.mark.parametrize("draft_tokens", [1, 3, 5, 8])
@pytest.mark.parametrize("alpha", [0.6, 0.7438, 0.9])
def test_wall_breakeven_is_the_per_position_derivation(alpha, draft_tokens):
    """A round costs (1+d)*alpha base-units and yields 1 + a*d tokens, so it
    breaks even at a* = ((1+d)*alpha - 1)/d."""
    expected = ((1 + draft_tokens) * alpha - 1.0) / draft_tokens
    assert wall_breakeven_acceptance(alpha, draft_tokens) == pytest.approx(expected)


def test_wall_breakeven_is_unsatisfiable_without_drafting():
    """Same contract as spec_breakeven_acceptance: refuse, never divide by
    zero."""
    assert wall_breakeven_acceptance(0.75, 0) == float("inf")


def test_a_verify_pass_cheaper_than_its_positions_needs_no_acceptance():
    """alpha low enough that a wide round costs less than one decode round is
    a bar of zero, not a negative one."""
    assert wall_breakeven_acceptance(0.1, 3) == 0.0


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_the_timed_bar_refuses_what_the_byte_bar_admits(geo):
    """THE defect, replayed. Acceptance 0.6875 clears the measured byte bar of
    0.6259 -- that is exactly the configuration P9 measured at -16% decode --
    and it fails the 0.7068 bar that alpha=0.7801 implies."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    gate._breakeven = MINIMAX_BYTE_BAR      # the measured byte bar, see above
    _trial_timed(gate, wide_s=_wide_s_for(0.7801, 0.5, 3), acceptance=0.6875)
    assert gate.status()["wall_breakeven"] == pytest.approx(0.7068, abs=5e-4)
    assert gate.decision == "off_acceptance"
    assert gate.status()["bar_used"] == "wall"


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_the_byte_bar_stays_necessary_when_it_is_the_higher_one(geo):
    """Bytes remain a necessary condition, and status() says which rule
    decided -- a gate that silently changes that is worse than either rule."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    gate._breakeven = 0.9
    # alpha 0.6 -> a wall bar of 0.4667, well below the byte bar.
    _trial_timed(gate, wide_s=_wide_s_for(0.6, 0.5, 3), acceptance=0.7)
    assert gate.decision == "off_acceptance"
    assert gate.status()["bar_used"] == "bytes"


def test_acceptance_strictly_above_the_timed_bar_runs():
    """A fix that refuses everything is not a fix: the spike's winning arm
    (alpha 0.7438, acceptance 0.877, measured +22.8% pooled) must be admitted."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = MINIMAX_BYTE_BAR
    _trial_timed(gate, wide_s=_wide_s_for(0.7438, 0.5, 3), acceptance=0.877)
    assert gate.decision == "run"
    assert gate.status()["bar_used"] == "wall"


def test_acceptance_exactly_on_the_timed_bar_refuses():
    """Match _decide_acceptance's existing `<=`: breaking even is not worth a
    latched regression. alpha 0.75 at d=3 puts the bar at exactly 2/3, and
    32 accepted of 48 drafted is exactly 2/3."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = 0.0
    _trial_timed(gate, wide_s=_wide_s_for(0.75, 0.5, 3), acceptance=2 / 3)
    assert gate.measured_acceptance == wall_breakeven_acceptance(0.75, 3)
    assert gate.decision == "off_acceptance"


def test_a_short_timing_sample_falls_back_to_the_byte_bar():
    """Fewer than draft_tokens+1 deltas in either list is not a measurement.
    Here CALIBRATE was never timed at all."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_with_overlap(gate, layers=range(62))      # no elapsed_s at all
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    s = gate.status()
    assert s["bar_used"] == "bytes"
    assert s["alpha"] is None and s["wall_breakeven"] is None
    assert gate.decision == "run"          # byte bar 0.125, acceptance 1.0


def test_a_nonpositive_base_median_falls_back_to_the_byte_bar():
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.0)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    s = gate.status()
    assert s["bar_used"] == "bytes"
    assert s["alpha"] is None


def test_omitting_elapsed_s_reproduces_the_byte_only_behaviour():
    """The additive-signature guarantee: every existing caller is unaffected."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=14_700_000_000)
    _calibrate_with_overlap(gate, layers=range(62))
    for _ in range(ag.TRIAL_DRAFTED_TOKENS):
        gate.record_round(drafted=3, accepted=3)
    assert gate.phase == "run" and gate.decision == "run"
    s = gate.status()
    assert s["bar_used"] == "bytes"
    assert s["alpha"] is None and s["wall_breakeven"] is None


def test_a_suppressed_trial_round_never_enters_the_wide_median():
    """PLD drafts 0 or 3, never a partial count. A k=0 round is a 1-wide
    decode, so counting it as wide would understate alpha and bias toward
    ADMIT -- the numerator must contain drafting rounds only."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    for _ in range(20):
        gate.record_round(0, 0, elapsed_s=0.5)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


def test_the_baseline_prefers_the_trials_own_suppressed_rounds():
    """MEASURED 2026-09-10, and it is why this rule exists: CALIBRATE's rounds
    are the FIRST rounds after a prefill, so they run against a cold expert
    cache while TRIAL's run warm. On MiniMax-M2.7's losing arm that inflated
    the baseline enough to drop alpha 0.7801 -> 0.6976 and put the whole timed
    bar UNDER the byte bar, i.e. the fix silently never engaged. A k=0 trial
    round is a 1-wide decode sampled in TRIAL's own regime, which is the
    contemporaneous baseline the ratio actually wants."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=1.0)              # cold: twice as slow
    for _ in range(6):
        gate.record_round(0, 0, elapsed_s=0.5)      # warm 1-wide rounds
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9), \
        "alpha was taken against CALIBRATE's cold rounds, not TRIAL's own"
    assert gate.status()["base_from"] == "trial_suppressed"


def test_the_proposers_own_search_is_not_charged_to_the_baseline():
    """The counterfactual a timed bar compares against is speculation OFF,
    where the proposer never runs at all -- so a k=0 round's baseline is its
    DECODE, not its decode plus a search that found nothing. Measured
    2026-09-10: leaving the search in understates alpha by about the search's
    own share, which on MiniMax-M2.7's losing arm was the whole remaining
    margin (bar 0.6798 against an acceptance of 0.6875)."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=9.0)              # unusable, must not be used
    for _ in range(6):
        gate.record_round(0, 0, elapsed_s=0.55, draft_overhead_s=0.05)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    # 1.5 / ((0.55 - 0.05) * 4) = 0.75, not 1.5 / (0.55 * 4) = 0.6818
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


def test_a_search_that_swallows_the_whole_round_is_not_a_baseline():
    """Defensive: a non-positive remainder is not a decode time. Drop the
    sample rather than divide the bar by it."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    for _ in range(6):
        gate.record_round(0, 0, elapsed_s=0.5, draft_overhead_s=0.5)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["base_from"] == "calibrate"
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


def test_a_drafting_rounds_search_stays_in_the_numerator():
    """The search is a real cost of speculating when it succeeds -- only the
    round that got nothing for it is corrected."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0, draft_overhead_s=0.4)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)


def test_the_calibrate_baseline_is_used_when_the_trial_never_suppresses():
    """A workload that matches an n-gram every round yields no suppressed
    round to measure -- the spike's winning arm had 0% of them. CALIBRATE's
    rounds are then the only baseline there is, and its known bias is toward
    ADMIT, which is the direction that arm needs anyway."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)
    assert gate.status()["base_from"] == "calibrate"


def test_too_few_suppressed_rounds_fall_back_to_calibrate():
    """Same floor as everywhere else: fewer than draft_tokens+1 samples is not
    a measurement, so a handful of suppressed rounds must not outvote a full
    calibration window."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    for _ in range(3):                               # one short of the floor
        gate.record_round(0, 0, elapsed_s=9.0)
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["alpha"] == pytest.approx(0.75, abs=1e-9)
    assert gate.status()["base_from"] == "calibrate"


def test_no_baseline_at_all_falls_back_to_the_byte_bar():
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_with_overlap(gate, layers=range(62))   # untimed calibration
    _trial_timed(gate, wide_s=1.5, acceptance=1.0)
    assert gate.status()["bar_used"] == "bytes"
    assert gate.status()["base_from"] is None


def test_qwen_still_refuses_on_bytes_before_any_wide_timing_exists():
    """Qwen3-235B latches off_union during CALIBRATE, at zero drafting cost.
    The timed bar must not disturb that path -- no wide sample exists yet."""
    gate = AdaptiveSpecGate(QWEN, draft_tokens=5, budget_bytes=1_000_000_000)
    for step in range(max(8, 6)):
        gate.observe({l: list(range(step * 8, step * 8 + 8)) for l in range(94)},
                     elapsed_s=0.5)
    assert gate.decision == "off_union"
    s = gate.status()
    assert s["alpha"] is None and s["bar_used"] is None


# The six spike arms (docs/2026-09-10-row13-timed-trial-SPIKE.md §3), replayed
# against the real decision rule. Their measured pooled ratios (>1 means
# drafting was FASTER) are 0.753, 0.734, 1.007, 0.898, 1.174, 1.228 -- arm 3 is
# a coin flip the design says not to tune to, and it falls on the refuse side
# here, which is the safe direction for a decision that latches.
SPIKE_ARMS = [
    # id,                       alpha,  acceptance, expected decision
    ("loser-think1-loadtime",   0.7801, 0.417, "off_acceptance"),
    ("loser-think1-verify",     0.7821, 0.413, "off_acceptance"),
    ("winner-think1-loadtime",  0.7255, 0.591, "off_acceptance"),
    ("winner-think1-verify",    0.7250, 0.529, "off_acceptance"),
    ("winner-think0-loadtime",  0.7372, 0.845, "run"),
    ("winner-think0-verify",    0.7438, 0.877, "run"),
]


@pytest.mark.parametrize("name,alpha,acceptance,expected",
                         SPIKE_ARMS, ids=[a[0] for a in SPIKE_ARMS])
def test_the_six_measured_arms_replay_to_the_right_decision(name, alpha,
                                                            acceptance, expected):
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = MINIMAX_BYTE_BAR
    _trial_timed(gate, wide_s=_wide_s_for(alpha, 0.5, 3), acceptance=acceptance)
    assert gate.decision == expected, (
        f"{name}: alpha={alpha} acceptance={acceptance} "
        f"wall_bar={gate.status()['wall_breakeven']}")


# ---------------------------------------------------------------------------
# Row 13, `even-4` (2026-09-11). The timed bar changed no decision on device,
# because it was measured in ONE narrow, early window: CALIBRATE's 8 rounds and
# TRIAL's 16 sit in the cheapest stretch of a call, so a correctly-measured bar
# still admitted an arm that lost -19.7% on its own A/B. The fix is a WIDER
# window, not a better bar in the same one: TRIAL alternates legs of
# TRIAL_LEG_ROUNDS drafting and TRIAL_LEG_ROUNDS suppressed rounds, and the
# verdict is `run` iff acceptance clears the BYTE bar (necessary, unchanged)
# AND the MEAN of the per-pair s/token ratios is below 1.
#
# Replayed on the six archived traces: 6/6 correct against the shipped 5/6.
# Routing the pooled ratio through alpha -> wall bar instead scored 4/6 --
# worse than doing nothing -- so the ratio decides directly.
# ---------------------------------------------------------------------------

def _pooled_trial(gate, pairs, supp_s=0.5):
    """Drive TRIAL the way the loop does: ask should_draft() every round,
    draft when told to, run an ordinary 1-wide decode when not.

    `pairs[j] = (wide_s, accepted_per_round)` sets pair j's drafting leg; the
    last entry repeats if the trial runs past len(pairs). Every suppressed
    round takes `supp_s` and yields one token. A pair starts at each
    suppressed -> drafting transition, so this helper needs no knowledge of
    the leg length. Returns the number of trial rounds run.
    """
    d = gate.draft_tokens
    j, prev, rounds = -1, None, 0
    while gate.phase == "trial" and rounds < 10_000:
        drafting = gate.should_draft()
        if drafting and prev is not True:
            j += 1
        prev = drafting
        wide_s, acc = pairs[min(max(j, 0), len(pairs) - 1)]
        if drafting:
            gate.record_round(d, acc, elapsed_s=wide_s)
        else:
            gate.record_round(0, 0, elapsed_s=supp_s)
        rounds += 1
    return rounds


# Per-round drafting-leg times that give a pair ratio r at full acceptance
# (d=3 -> 4 tokens per round) against 0.5 s/token suppressed legs.
def _wide_for_ratio(r, tokens_per_round=4, supp_s=0.5):
    return r * supp_s * tokens_per_round


def test_trial_alternates_drafting_and_suppressed_legs():
    """[D4 S4] x 4 after CALIBRATE: the gate itself schedules the A/B, and the
    loop already honours should_draft() per round (vaayu_base.py)."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_with_overlap(gate, layers=range(62))
    pattern = []
    while gate.phase == "trial":
        drafting = gate.should_draft()
        pattern.append(drafting)
        gate.record_round(3 if drafting else 0, 3 if drafting else 0)
    leg = ag.TRIAL_LEG_ROUNDS
    assert pattern == ([True] * leg + [False] * leg) * ag.TRIAL_PAIRS
    assert gate.decision == "run"


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_the_trial_drafts_48_tokens_over_four_pairs(geo):
    """The widened window keeps TRIAL_DRAFTED_TOKENS: 4 pairs x 4 drafting
    rounds x 3 drafted = 48, so the acceptance estimate the byte bar consumes
    is not loosened -- and suppressed rounds never enter it."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    rounds = _pooled_trial(gate, [(_wide_for_ratio(0.75), 3)])
    assert rounds == 32
    assert gate.status()["drafting_rounds"] == 16
    assert gate.measured_acceptance == 1.0


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_three_pairs_that_favour_drafting_outvote_one_that_does_not(geo):
    """The odd pair comes FIRST -- where the shipped single window would have
    sampled it and nothing else."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    gate._breakeven = MINIMAX_BYTE_BAR
    _pooled_trial(gate, [(_wide_for_ratio(1.5), 3)] + [(_wide_for_ratio(0.75), 3)] * 3)
    s = gate.status()
    assert gate.decision == "run"
    assert s["bar_used"] == "pooled"
    assert s["pooled_ratio"] == pytest.approx(0.9375)     # (1.5 + 3*0.75) / 4
    assert s["pairs_used"] == 4


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_three_pairs_against_drafting_refuse_it(geo):
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    gate._breakeven = MINIMAX_BYTE_BAR
    _pooled_trial(gate, [(_wide_for_ratio(0.75), 3)] + [(_wide_for_ratio(1.5), 3)] * 3)
    s = gate.status()
    assert gate.decision == "off_acceptance"
    assert s["bar_used"] == "pooled"
    assert s["pooled_ratio"] == pytest.approx(1.3125)     # (0.75 + 3*1.5) / 4


def test_the_pooled_estimator_is_a_mean_not_a_median():
    """Row 13's price replay: a median over 3-4 pairs flips when one pair
    moves. Ratios [0.9, 0.9, 0.9, 1.5]: median 0.9 would admit, mean 1.05
    refuses -- and the one expensive pair is a real cost of the call."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = MINIMAX_BYTE_BAR
    _pooled_trial(gate, [(_wide_for_ratio(0.9), 3)] * 3 + [(_wide_for_ratio(1.5), 3)])
    assert gate.status()["pooled_ratio"] == pytest.approx(1.05)
    assert gate.decision == "off_acceptance"


def test_a_pair_ratio_is_seconds_per_token_so_it_carries_realised_acceptance():
    """A drafting round yields accepted + 1 tokens, a suppressed one yields 1
    -- derivable inside the gate from what record_round already receives.
    wide 1.5 s at 1 accepted = 2 tokens = 0.75 s/tok, against 0.5: ratio 1.5."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = 0.0
    _pooled_trial(gate, [(1.5, 1)])
    assert gate.status()["pooled_ratio"] == pytest.approx(1.5)
    assert gate.decision == "off_acceptance"


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_the_byte_bar_stays_necessary_under_the_pooled_rule(geo):
    """Every pair favours drafting (ratio 0.667), but acceptance 2/3 is under
    a 0.9 byte bar: refuse. The byte test is free, sound, and what refuses
    Qwen3-235B before a single draft is paid for."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    gate._breakeven = 0.9
    _pooled_trial(gate, [(1.0, 2)])       # 1.0 s / 3 tokens vs 0.5 s / token
    s = gate.status()
    assert s["pooled_ratio"] == pytest.approx(2 / 3, abs=1e-4)
    assert gate.measured_acceptance == pytest.approx(2 / 3)
    assert gate.decision == "off_acceptance"


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen", "minimax"])
def test_a_slow_proposer_extends_the_trial_by_whole_pairs(geo):
    """PLD drafts 0 or d. A proposer that matches on every other drafting
    round has drafted only 24 tokens after four pairs, so the trial keeps
    going -- whole pairs at a time, so every leg keeps its partner -- until
    it has TRIAL_DRAFTED_TOKENS: 8 pairs, 64 rounds."""
    gate = AdaptiveSpecGate(geo, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5, layers=range(geo["num_moe_layers"]))
    rounds = drafting_seen = 0
    while gate.phase == "trial":
        if gate.should_draft():
            drafting_seen += 1
            k = 3 if drafting_seen % 2 else 0
            gate.record_round(k, k, elapsed_s=1.5 if k else 0.5)
        else:
            gate.record_round(0, 0, elapsed_s=0.5)
        rounds += 1
    assert rounds == 64
    assert gate._drafted == 48
    assert gate.status()["pairs_used"] == 8


def test_the_cap_gives_a_proposer_that_never_matches_as_many_chances_as_before():
    """Only half a trial's rounds may draft now, so the cap doubled: a
    proposer that never matches still gets the shipped 64 drafting-eligible
    rounds before the gate decides on what it has."""
    from vaayu.speculative import adaptive_gate as ag
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    eligible = rounds = 0
    while gate.phase == "trial":
        eligible += gate.should_draft()
        gate.record_round(0, 0, elapsed_s=0.5)
        rounds += 1
    assert rounds == ag.TRIAL_ROUND_CAP
    assert eligible == 64
    assert gate.decision == "off_acceptance"


def test_too_few_timed_pairs_fall_back_to_the_single_window_rule_and_say_so():
    """A pair needs BOTH legs timed. Untimed suppressed legs leave no pair to
    pool, so the gate decides on the shipped rule -- and status() says so.
    A gate that silently changes rule is worse than either rule."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    _calibrate_timed(gate, base_s=0.5)
    gate._breakeven = MINIMAX_BYTE_BAR
    while gate.phase == "trial":
        if gate.should_draft():
            gate.record_round(3, 3, elapsed_s=_wide_s_for(0.7438, 0.5, 3))
        else:
            gate.record_round(0, 0)
    s = gate.status()
    assert s["pairs_used"] == 0 and s["pooled_ratio"] is None
    assert s["bar_used"] == "wall"          # alpha 0.7438 -> 0.6584 > 0.6259
    assert gate.decision == "run"


LEGACY_STATUS_KEYS = {
    "observed_union_gb", "budget_gb", "breakeven_acceptance",
    "measured_acceptance", "drafting_rounds", "alpha", "wall_breakeven",
    "bar_used", "base_from", "decision",
}


def test_status_keeps_every_existing_key_and_adds_the_pooled_ones():
    """These keys reach /v1/models and env.json unchanged."""
    gate = AdaptiveSpecGate(MINIMAX, draft_tokens=3, budget_bytes=15_000_000_000)
    before = gate.status()
    assert LEGACY_STATUS_KEYS <= set(before)
    assert before["pooled_ratio"] is None and before["pairs_used"] == 0
    _calibrate_timed(gate, base_s=0.5)
    _pooled_trial(gate, [(_wide_for_ratio(0.75), 3)])
    after = gate.status()
    assert LEGACY_STATUS_KEYS <= set(after)
    assert isinstance(after["pooled_ratio"], float)
    assert isinstance(after["pairs_used"], int)
