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
