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
