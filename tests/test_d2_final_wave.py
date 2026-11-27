"""D2 final-wave fixes: the admission retry's verdict (I2), its probe/quiet (M4),
its idempotence with the elastic tier, and the tier-refund guards (M1)."""
import asyncio
import gc
import types

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import VramPlanError

from .test_elastic_tier import SHORT, rig


class _Once:
    """Stands in for `_assert_context_fits_once`: refuses until gc has run."""

    def __init__(self):
        self.calls, self.collected = 0, False

    def __call__(self, model, n_tokens, **kw):
        self.calls += 1
        if not self.collected:
            raise VramPlanError("Context does not fit.\n")
        raise VramPlanError("Context does not fit.\n")


@pytest.fixture
def refusing(monkeypatch):
    once = _Once()
    monkeypatch.setattr(VaayuBaseModel, "_assert_context_fits_once", once, raising=False)
    real = gc.collect
    ran = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: (ran.append(1), real(*a, **k))[-1])
    return once, ran


def _refusal_text(model=None):
    with pytest.raises(VramPlanError) as ei:
        VaayuBaseModel._assert_context_fits(model or types.SimpleNamespace(), 21566)
    return str(ei.value)


def test_a_measured_reclaim_of_nothing_says_nothing_was_retained(refusing, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 5 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    text = _refusal_text()
    assert "Nothing was retained" in text and "counts memory_allocated only" in text
    assert "reclaimed 0.00 GiB" in capsys.readouterr().out


def test_an_unmeasurable_reclaim_never_claims_nothing_was_retained(refusing, monkeypatch, capsys):
    """I2: memory_allocated raising used to skip gc.collect() AND report a measured 0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "memory_allocated", boom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    once, ran = refusing
    text = _refusal_text()
    assert "Nothing was retained" not in text and "could not be measured" in text
    assert ran, "gc.collect() must run even when the measurement cannot"
    assert once.calls == 2
    assert "could not measure the reclaim" in capsys.readouterr().out


def test_without_cuda_the_reclaim_is_not_a_measurement(refusing, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    text = _refusal_text()
    assert "Nothing was retained" not in text and "could not be measured" in text


def test_the_probe_retry_is_quiet_but_a_request_retry_is_not(refusing, monkeypatch, capsys):
    """M4: the width-2 probe is refused by design at every boot."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _refusal_text(types.SimpleNamespace(_admission_probe=True))
    assert "admission refused" not in capsys.readouterr().out
    _refusal_text(types.SimpleNamespace(_admission_probe=False))
    assert "admission refused" in capsys.readouterr().out


def test_width2_admits_marks_itself_a_probe_and_unmarks(monkeypatch):
    from vaayu.batch import _width2_admits
    seen = []

    class M:
        max_seq_len = 1024

        def _assert_context_fits(self, n, **kw):
            seen.append(self._admission_probe)
            raise VramPlanError("no")

    m = M()
    monkeypatch.setattr("vaayu.batch.expert_giveback_bytes", lambda model: 0)
    assert _width2_admits(m) is False
    assert seen == [True] and m._admission_probe is False


# ── the elastic tier under the retry ───────────────────────────────────────────

def test_a_retried_elastic_admission_is_idempotent(monkeypatch):
    """The wrapper calls `ElasticTier.admit` a SECOND time after a refusal. The REAL
    tier: the first call refuses and empties it (budget 0), gc frees the card, the
    second call must land exactly where ONE admission on that card would -- the
    budget is written absolutely, grants counted once, and nothing is moved twice."""
    card, tier, elastic, budgets = rig("gptoss_like")
    ref_card, ref_tier, ref_elastic, _ = rig("gptoss_like")
    ref_elastic.admit(SHORT, SHORT + 64)

    card.other = card.total              # someone else holds the card: admission refuses
    geo_model = types.SimpleNamespace(
        _vram_plan=elastic.plan, _vram_shape=elastic.shape, _elastic_tier=elastic)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    real = gc.collect

    def freeing(*a, **k):
        card.other = 0                   # the dead request's memory is collected
        return real(*a, **k)

    monkeypatch.setattr(gc, "collect", freeing)
    assert VaayuBaseModel._assert_context_fits(geo_model, SHORT + 64, prompt_tokens=SHORT) \
        == ref_elastic.admit(SHORT, SHORT + 64)
    assert elastic.grants == 1, "a refused admission is not a grant"
    assert tier.budget_bytes == ref_tier.budget_bytes
    assert elastic.granted == ref_elastic.granted and elastic.covering


def test_a_doubly_refused_elastic_admission_leaves_the_tier_at_zero_not_deeper(monkeypatch):
    card, tier, elastic, budgets = rig("gptoss_like")
    card.other = card.total
    model = types.SimpleNamespace(_vram_plan=elastic.plan, _vram_shape=elastic.shape,
                                  _elastic_tier=elastic)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, SHORT + 64, prompt_tokens=SHORT)
    assert tier.budget_bytes == 0 and elastic.grants == 0 and not elastic.covering
    assert all(b == 0 for b in budgets)          # absolute writes, never relative


# ── M1: a failing refund never fails the request ───────────────────────────────

def test_engine_slot_survives_a_raising_refund():
    from vaayu.server.engine import EngineHandle
    from .test_server_engine import FakeModel
    m = FakeModel()

    def refund():
        raise RuntimeError("set_budget failed")

    m._refund_expert_tier = refund

    async def main():
        e = EngineHandle(m)
        async with e.slot():
            return True

    assert asyncio.run(main()) is True
