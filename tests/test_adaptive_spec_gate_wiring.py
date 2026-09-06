"""S1d wiring: the gate must actually govern the real generation loop.

The harness is the one tests/test_speculative_reset.py already uses -- it
drives the REAL VaayuBaseModel.speculative_generate with a fake model, so
these tests exercise the shipped loop rather than a copy of it.
"""
import torch

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.speculative import adaptive_gate as ag
from vaayu.vaayu_base import VaayuBaseModel

from .test_speculative_prefix_reuse import _FakeSpecModel, _FakeSpeculator

MINIMAX = {"expert_bytes": 8_875_020, "num_moe_layers": 62, "top_k": 8,
           "num_experts": 256, "working_set_bytes": 8 * 62 * 8_875_020}


class _MoEFake(_FakeSpecModel):
    """The fake plus the four attributes the gate reads off a real MoE model."""

    def __init__(self, pc, budget_bytes=14_700_000_000, experts=range(8)):
        super().__init__(pc)
        self.is_moe = True
        self._moe_geometry = MINIMAX
        self._expert_cache = type("C", (), {"budget_bytes": budget_bytes})()
        self._expert_history = {layer: list(experts) for layer in range(62)}
        self.draft_tokens = 3


class _CountingSpeculator(_FakeSpeculator):
    def __init__(self):
        self.drafts = 0

    def reset(self):
        pass

    def draft(self, generated, past_key_values, num_tokens=5):
        self.drafts += 1
        return super().draft(generated, past_key_values, num_tokens)


class _DraftingSpeculator(_FakeSpeculator):
    """Drafts real k>0 tokens and verifies them with PromptLookupDecoder's OWN
    accept/reject algorithm (copied from prompt_lookup.py, not hardcoded), so
    a test can drive accept/reject scenarios purely by choosing what tokens
    draft() proposes against the fake forward's all-zero (argmax -> token 0)
    verifier logits. `_CountingSpeculator`/`_FakeSpeculator` always draft 0
    tokens, so k is always 0 there and the accept-count arithmetic below can
    never be exercised through them -- this fake exists so it can be."""

    def __init__(self, draft_sequence):
        self.draft_sequence = list(draft_sequence)   # one token-id list per round
        self.drafts = 0

    def reset(self):
        pass

    def draft(self, generated, past_key_values, num_tokens=5):
        self.drafts += 1
        tokens = self.draft_sequence.pop(0)
        return torch.tensor([tokens], dtype=torch.long), {}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        # Identical to PromptLookupDecoder.verify_and_accept: num_accepted is
        # 1..k+1, always including the bonus (full match) or correction
        # (mismatch) token -- the real contract the wiring must translate.
        num_draft = draft_ids.shape[1]
        if num_draft == 0:
            next_token = verifier_logits[:, 0].argmax(dim=-1, keepdim=True)
            return next_token, 1
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def rollback_cache(self, draft_state, num_accepted) -> None:
        pass


def _run(model, max_new_tokens):
    return VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]),
        max_new_tokens=max_new_tokens, session_id="s1")


def test_no_drafting_happens_during_calibration(monkeypatch):
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=4)
    assert model.speculator.drafts == 0, \
        "the gate must observe real routing before paying for a draft"


def test_a_suppressed_round_verifies_exactly_one_position(monkeypatch):
    """A round the gate suppressed must cost exactly what plain decode costs.

    This is the premise the documented exploration-cost bound rests on (todo.md
    P4): CALIBRATE's steps, and every TRIAL round where the proposer matched no
    n-gram, produce `draft_ids` of width 0, so `verify_input` is the single last
    token and the verify forward is the ordinary 1-position decode. If any of
    them ever widened, the "only DRAFTING rounds are paid for" accounting -- and
    the `TRIAL_DRAFTED_TOKENS * breakeven` ceiling derived from it -- would both
    understate the real cost.
    """
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=4)

    prefill, verifies = model.forward_calls[0], model.forward_calls[1:]
    assert prefill == 5, "the prompt is prefilled in one pass"
    assert verifies and set(verifies) == {1}, \
        (f"a suppressed round must verify 1 position, not {sorted(set(verifies))} "
         f"-- widening one makes it cost more than the plain decode it replaces")


def test_a_dense_model_never_builds_a_gate():
    """P7's 5.57x lives on this path. The gate is MoE-only."""
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=3)
    assert model.speculator.drafts > 0, "dense must draft from the first round"
    assert getattr(model, "_spec_gate", None) is None


def test_routing_that_does_not_fit_latches_the_speculator_away(monkeypatch):
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 2)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9), budget_bytes=1)
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=6)
    assert model.speculator is None, \
        "an off decision must disable speculation the way the load gate does"
    assert model.speculative_status["calibration"]["decision"] == "off_union"
    assert model.speculative_status["active"] is False


def test_the_generation_still_completes_after_latching_off(monkeypatch):
    """Latching mid-call must not truncate the answer or crash on the
    speculator it just removed."""
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 2)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9), budget_bytes=1)
    model.speculator = _CountingSpeculator()
    out = _run(model, max_new_tokens=6)
    assert out.shape[1] == 5 + 6


def test_a_fitting_model_speculates_after_calibration(monkeypatch):
    # NOTE: the brief specifies CALIBRATE_STEPS=2 here, matching the other
    # latch-off tests. But observed_union_per_layer needs >= draft_tokens+1
    # (4, for this fake's draft_tokens=3) snapshots to compute a non-zero
    # window (vaayu/speculative/adaptive_gate.py's own comment: "must exceed
    # d+1 or there is no window to measure a union over"); at 2 snapshots
    # _decide_union always sees union_k == 0 and latches off_union regardless
    # of budget, so the literal brief value makes this specific assertion
    # ("a fitting model speculates") unreachable. 4 is the minimum that
    # produces a real window while still leaving a round for should_draft()
    # to fire before max_new_tokens=6 ends the call.
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=6)
    assert model.speculator is not None
    assert model.speculator.drafts > 0


def test_record_round_translates_the_real_accept_contract(monkeypatch):
    """Fix round 1, item 1: the off-by-one that shipped in the first wiring
    (record_round(k, num_accepted) instead of record_round(k, num_accepted-1))
    was invisible to every other test in this file, because their fakes
    always draft 0 tokens -- k is always 0, so the accept-count arithmetic the
    bug lived in never runs. Only an on-device run (S1d Task 6) caught it, on
    Qwen, where it made the anti-regression arm silently report decision="run"
    with an inflated measured_acceptance instead of "off_acceptance".

    Two rounds, driven through the REAL speculative_generate loop:

      round 1: drafts [0, 0, 0]. The fake forward's logits are all-zero, so
               argmax is always token 0 -- every drafted token matches.
               verify_and_accept returns num_accepted=4 (3 drafts + the bonus
               token PromptLookupDecoder always appends on a full match).
               Correct translation into the gate: drafted=3, accepted=3.
      round 2: drafts [7, 0, 0] -- mismatches at position 0, the i=0 edge
               `max(0, num_accepted - 1)` exists for (an immediate rejection
               still returns num_accepted=1, the correction token; there is
               no drafted-token match to count). Correct translation:
               drafted=3, accepted=0.

    Cumulative correct state: drafted=6, accepted=3, measured_acceptance=0.5.
    The bug this guards against would instead record accepted=4 and
    accepted=1 (the raw, un-decremented num_accepted values) -- accepted=5
    over drafted=6, measured_acceptance=0.833 -- so this fails loudly if the
    off-by-one comes back.
    """
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _DraftingSpeculator([[0, 0, 0], [7, 0, 0]])
    # 1 prefill token (seeded before the while loop, no gate involvement) +
    # 4 calibrate rounds (1 token each, draft() not called) + round 1 (4
    # accepted) + round 2 (1 accepted) = 10 tokens -- exactly two trial
    # rounds, no more, so the counts below are not disturbed by a third.
    _run(model, max_new_tokens=10)
    gate = model._spec_gate
    assert gate is not None
    assert gate._drafted == 6, "both rounds drafted k=3, regardless of outcome"
    assert gate._accepted == 3, (
        "3 (all accepted) + 0 (rejected at i=0), NOT 4 + 1 -- the bonus/"
        "correction token verify_and_accept always appends must not be "
        "counted as a drafted-token match")
    assert gate.status()["measured_acceptance"] == 0.5


# ---------------------------------------------------------------------------
# Whole-branch review, 2026-09-05: Critical 1, Important 2/3/4.
# ---------------------------------------------------------------------------

def test_the_gate_is_shared_across_request_boundaries(monkeypatch):
    """Critical 1. The RUN/OFF decision latches for the PROCESS (design doc
    S2), not per request. Before this fix, `_build_spec_gate()` built a FRESH
    gate on every `speculative_generate()` call, so two back-to-back calls
    each restarted CALIBRATE at snapshot 0 -- a server doing short requests
    would never accumulate enough evidence to reach a decision at all, and a
    good configuration would re-pay its calibration cost on every request.

    CALIBRATE_STEPS=4 needs 4 observed snapshots to leave "calibrate". Each
    call below is capped to exactly 2 calibrate rounds. A gate rebuilt per
    call would still have only 2 snapshots after the second call and never
    leave "calibrate"; a gate carried across the boundary reaches 4 and
    moves to "trial" on the second call -- the assertion a rebuild-per-call
    bug fails."""
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _MoEFake(pc)
    model.speculator = _CountingSpeculator()

    _run(model, max_new_tokens=3)   # prefill (1 tok) + 2 calibrate rounds
    gate_after_call_1 = model._spec_gate
    assert gate_after_call_1 is not None
    assert gate_after_call_1.phase == "calibrate"
    assert len(gate_after_call_1._snapshots) == 2, \
        "premise: call 1 must leave the gate exactly 2 snapshots short of done"

    _run(model, max_new_tokens=3)   # 2 more calibrate rounds, on the SAME gate
    assert model._spec_gate is gate_after_call_1, \
        "a fresh gate was built on the second call -- the latch cannot " \
        "survive a request boundary"
    assert model._spec_gate.phase == "trial", \
        "the two calls' snapshots did not accumulate on one gate -- a " \
        "rebuild-per-call gate would still have only 2 snapshots here"


def test_a_dense_model_still_never_gets_a_cached_gate():
    """The caching in _build_spec_gate must not accidentally give dense a
    gate on a second call -- is_moe is checked before the cache lookup."""
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=3)
    _run(model, max_new_tokens=3)
    assert getattr(model, "_spec_gate", None) is None


class _RaisesOnNoneRollback(_FakeSpeculator):
    """StandardDrafterDecoder.rollback_cache does `draft_state.get(...)`
    unconditionally once its own cache is warm (it early-returns only on
    `_drafter_past is None`, which is not the case once the trial has
    drafted). This fake reproduces exactly that contract without needing a
    real weighted drafter: calling it with draft_state=None must never
    happen, so it raises if it does."""

    def rollback_cache(self, draft_state, num_accepted) -> None:
        if draft_state is None:
            raise AttributeError(
                "rollback_cache(None, ...) -- draft_state.get() on None")


def test_rollback_cache_is_skipped_when_draft_state_is_none(monkeypatch):
    """Important 2. Every calibrate round (and every round after an
    off_union/off_acceptance latch) takes the suppressed-draft path with
    draft_state=None -- speculator.draft() is never called, so there is
    nothing for rollback_cache to roll back. Before this fix, Step 6 called
    `speculator.rollback_cache(draft_state, num_accepted)` unconditionally,
    which crashes on any speculator whose rollback_cache is not PLD's `pass`
    (e.g. StandardDrafterDecoder). Reachable on any MoE + weighted drafter,
    starting with the very first calibrate round -- no latch required."""
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _RaisesOnNoneRollback()
    _run(model, max_new_tokens=2)   # one calibrate round is enough to hit it


def test_rollback_cache_is_also_skipped_after_a_mid_call_latch(monkeypatch):
    """Same guard, the scenario the finding names explicitly: after an
    off_union latch, LATER rounds also take the suppressed-draft path with
    draft_state=None, via the local `speculator` reference captured before
    self.speculator was set to None."""
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 2)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9), budget_bytes=1)
    model.speculator = _RaisesOnNoneRollback()
    _run(model, max_new_tokens=6)   # must not raise
    assert model.speculator is None


class _CyclingRoutingFake(_MoEFake):
    """4 disjoint 8-expert sets, cycled every forward() call, so the
    (draft_tokens+1)-wide union genuinely differs from the 1-wide union and
    the trial phase gets a non-degenerate (>0) break-even to fail. The other
    tests' static routing (same experts every step) gives breakeven=0, which
    ANY positive acceptance clears -- it can only ever produce "run", never
    "off_acceptance"."""

    def __init__(self, pc, budget_bytes):
        super().__init__(pc, budget_bytes=budget_bytes)
        self._route_step = 0

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        base = 8 * (self._route_step % 4)
        self._expert_history = {layer: list(range(base, base + 8))
                                 for layer in range(62)}
        self._route_step += 1
        return out


def test_off_acceptance_latches_through_the_real_loop(monkeypatch):
    """Important 3. Every existing latch test uses budget_bytes=1, i.e.
    off_union -- off_acceptance, the path that fires on real models, had
    never run end-to-end through the real loop, which is exactly what hid
    Important 2's rollback_cache(None, ...) crash (off_acceptance is reached
    by mid-call latching from a well-fitting budget, off_union is reached
    before a single draft is paid for).

    4 disjoint 8-expert sets over 4 calibrate snapshots give union(1)=8,
    union(4)=32 -> breakeven = (32/8 - 1)/3 = 1.0 (see _CyclingRoutingFake).
    Each trial round drafts [0, 0, 7]: positions 0-1 match the fake
    forward's all-zero-argmax verifier, position 2 mismatches -- drafted=3,
    accepted=2 every round, measured_acceptance settles at 2/3, well under
    the 1.0 bar.
    """
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 4)
    monkeypatch.setattr(ag, "TRIAL_DRAFTED_TOKENS", 6)
    monkeypatch.setattr(ag, "TRIAL_ROUND_CAP", 10)
    model = _CyclingRoutingFake(PrefixCacheStore(budget_bytes=10 ** 9),
                                budget_bytes=20_000_000_000)
    model.speculator = _DraftingSpeculator([[0, 0, 7], [0, 0, 7]])
    # 1 prefill + 4 calibrate rounds (1 tok each) + 2 trial rounds (3 tok
    # each, drafted=3/accepted=2 -> num_accepted=3 per round) = 11.
    out = _run(model, max_new_tokens=11)
    assert model.speculator is None, \
        "off on measured evidence must disable speculation the same way off_union does"
    assert model.speculative_status["calibration"]["decision"] == "off_acceptance"
    assert out.shape[1] == 5 + 11, \
        "latching mid-call must not truncate the answer"


def test_empty_routing_history_latches_distinctly_and_warns(monkeypatch, capsys):
    """Important 4. A forward that stops writing _expert_history makes every
    calibration snapshot {}. This must latch off on its own distinct
    decision (off_no_routing) -- not off_union, which a reader takes as a
    real budget refusal -- and print a WARNING naming the cause (CLAUDE.md
    S5: warn, never silently degrade), in the style of the other
    [Vaayu-Spec] warnings in vaayu_base.py."""
    monkeypatch.setattr(ag, "CALIBRATE_STEPS", 2)
    model = _MoEFake(PrefixCacheStore(budget_bytes=10 ** 9))
    model._expert_history = {}
    model.speculator = _CountingSpeculator()
    _run(model, max_new_tokens=5)   # prefill(1) + 4 calibrate rounds to decide
    assert model.speculator is None
    assert model.speculative_status["calibration"]["decision"] == "off_no_routing"
    warnings = capsys.readouterr().out
    assert "[Vaayu-Spec] WARNING" in warnings
    assert "_expert_history" in warnings
