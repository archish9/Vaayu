"""S35-E: where the engine hands a request to the elastic tier, and where it must not.

Stand-in models and a recording stand-in for ElasticTier -- the decisions themselves
are tested in tests/test_elastic_tier.py against a simulated card. What is pinned
here: the admission branch (single stream only), the two generate entry points, the
batch path's suspend, the prefill net, the forward hook, the layer range and the
load-time build.
"""
import inspect
import types
from unittest.mock import MagicMock

import pytest
import torch

import vaayu.moe_patch as mp
from vaayu.elastic_tier import GRANT_CHUNKS, ElasticTier
from vaayu.persist.expert_cache import ExpertVRAMCache
from vaayu.vaayu_base import (VaayuBaseModel, _VRAM_RESERVE_FLOOR_BYTES,
                              set_tier_layer_range)
from vaayu.vram_plan import VramPlan, VramPlanError, VramShape

GB = 1024 ** 3
MB = 1024 ** 2

# gpt-oss-20b's server-default load (plans/probes/s35e_out/shape-gptoss.log).
SHAPE = VramShape(
    kv_per_token=24576, N=1, chunk=512, hidden=2880, n_heads=64, vocab=201088,
    elem_bytes=2, score_elem_bytes=6.03125, alloc_factor=1.0, head_bytes=1158266976,
    score_bytes_per_key=16384.0, kv_constant_bytes=28286976)
PLAN = VramPlan(max_seq_len=33213, reserve_bytes=9416564832, pinning_budget=0,
                s_max=59901, reason="gpt-oss-20b load log")
EXPERT = 13236480

# (layers, experts, top_k, expert bytes): gpt-oss-like and MiniMax-like (CLAUDE.md §5).
LAYER_GEOMETRIES = {
    "gptoss_like": dict(num_moe_layers=24, num_experts=32, top_k=4, expert_bytes=EXPERT),
    "minimax_like": dict(num_moe_layers=62, num_experts=256, top_k=8, expert_bytes=8 * MB),
}


class Stop(Exception):
    """Raised by a stand-in to end the method under test at the call being pinned."""


class RecordingElastic:
    def __init__(self, covering=True, admit_returns=123, admit_raises=None):
        self.covering = covering
        self.calls = []
        self._returns, self._raises = admit_returns, admit_raises

    def admit(self, prompt_tokens, limit):
        self.calls.append(("admit", prompt_tokens, limit))
        if self._raises is not None:
            raise self._raises
        return self._returns

    def cover(self, context_after):
        self.calls.append(("cover", context_after))

    def suspend(self):
        self.calls.append(("suspend",))
        raise Stop()


@pytest.fixture
def cuda(monkeypatch):
    """A card the admission code can read without one being present."""
    state = {"free": 16 * GB}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (state["free"], 16 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    return state


def model_with(elastic, tier=None):
    return types.SimpleNamespace(
        _vram_plan=PLAN, _vram_shape=SHAPE, _moe_geometry={"expert_bytes": EXPERT},
        _expert_vram_cache=tier, _elastic_tier=elastic)


# ── admission ────────────────────────────────────────────────────────────────────


def test_a_single_stream_request_is_admitted_by_the_elastic_tier(cuda):
    elastic = RecordingElastic(admit_returns=4242)
    model = model_with(elastic)
    assert VaayuBaseModel._assert_context_fits(model, 9000, prompt_tokens=8000) == 4242
    assert elastic.calls == [("admit", 8000, 9000)]


def test_without_a_prompt_length_the_whole_request_is_the_prompt(cuda):
    elastic = RecordingElastic()
    VaayuBaseModel._assert_context_fits(model_with(elastic), 9000)
    assert elastic.calls == [("admit", 9000, 9000)]


def test_the_elastic_tiers_refusal_is_the_requests_refusal(cuda):
    elastic = RecordingElastic(admit_raises=VramPlanError("does not fit"))
    with pytest.raises(VramPlanError, match="does not fit"):
        VaayuBaseModel._assert_context_fits(model_with(elastic), 9000, prompt_tokens=8000)


@pytest.mark.parametrize("kwargs", [dict(N=2), dict(batched_prefill=True),
                                     dict(expert_giveback_bytes=1)])
def test_a_batch_keeps_the_static_admission(cuda, kwargs):
    elastic = RecordingElastic()
    model = model_with(elastic, tier=None)
    VaayuBaseModel._assert_context_fits(model, 1024, **kwargs)
    assert elastic.calls == []


def test_a_load_without_an_elastic_tier_takes_the_static_path(cuda):
    model = model_with(None)
    del model._elastic_tier                       # a stand-in built before S35-E
    assert VaayuBaseModel._assert_context_fits(model, 1024) > 0


def test_without_cuda_nothing_is_asked_of_the_elastic_tier(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elastic = RecordingElastic()
    assert VaayuBaseModel._assert_context_fits(model_with(elastic), 9000) is None
    assert elastic.calls == []


# ── the generate entry points ────────────────────────────────────────────────────


def _entry_stub():
    seen = []

    def fits(n_tokens, **kwargs):
        seen.append((n_tokens, kwargs))
        raise Stop()

    return types.SimpleNamespace(_assert_context_fits=fits), seen


def test_generate_tells_admission_how_long_the_prompt_is():
    stub, seen = _entry_stub()
    with pytest.raises(Stop):
        VaayuBaseModel.generate(stub, input_ids=torch.zeros((1, 11), dtype=torch.long),
                                max_new_tokens=7)
    assert seen == [(18, {"prompt_tokens": 11})]


def test_speculative_generate_asks_for_the_whole_request_at_once():
    """Its loop narrows the tier's layer range for its own duration and restores it
    afterwards, so a grant extended mid-request would be overwritten: it passes no
    prompt length, and the elastic tier then grants the whole request."""
    src = inspect.getsource(VaayuBaseModel.speculative_generate)
    assert "self._assert_context_fits(input_ids.shape[-1] + max_new_tokens)" in src
    assert "prompt_tokens" not in src


def test_batched_generate_takes_the_lend_back_before_its_own_admission():
    elastic = RecordingElastic()
    stub = types.SimpleNamespace(
        _active_attn_impl="sdpa", running_device="cpu", _elastic_tier=elastic,
        _assert_context_fits=lambda *a, **k: pytest.fail("admission ran before suspend"))
    with pytest.raises(Stop):
        next(VaayuBaseModel.batched_generate(stub, [object(), object()]))
    assert elastic.calls == [("suspend",)]


# ── the forward hook (stage 2) ───────────────────────────────────────────────────


def test_forward_covers_the_context_before_the_layer_loop():
    """Source order (forward needs a loaded model to run): the cover call sits after
    the past length and this pass's token count are known and before inference_mode,
    where the layers run. The on-device crossing runs count the extensions."""
    src = inspect.getsource(VaayuBaseModel.forward)
    past = src.index("_past_len_for_pos = (self.get_past_key_values_cache_seq_len(")
    cover = src.index("_elastic.cover(_past_len_for_pos + int(_seq_total))")
    layers = src.index("with torch.inference_mode():")
    assert past < cover < layers
    assert src.index("_seq_total = (sum(_chunk_lens)") < cover


# ── the prefill net ──────────────────────────────────────────────────────────────


class _Driver:
    """mem_get_info that reports `cached` bytes as used until empty_cache() runs."""

    def __init__(self, monkeypatch, free, cached=0):
        self.free, self.cached, self.empties = int(free), int(cached), 0
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (self.free, 16 * GB))
        monkeypatch.setattr(torch.cuda, "empty_cache", self._empty)

    def _empty(self):
        self.empties += 1
        self.free, self.cached = self.free + self.cached, 0


def _net_ref(elastic):
    return types.SimpleNamespace(_moe_prefill_pass=True, _expert_vram_cache=MagicMock(),
                                 _elastic_tier=elastic)


def test_on_a_granted_request_the_net_returns_the_allocators_cache_before_it_evicts(
        monkeypatch):
    """Driver-free under the headroom, but only because the prefill's own blocks are
    cached: on a granted request that is not pressure, and nothing is evicted."""
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    ref = _net_ref(RecordingElastic(covering=True))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 1
    ref._expert_vram_cache.evict_until_free.assert_not_called()


def test_on_a_granted_request_the_net_still_evicts_when_the_cache_was_not_the_reason(
        monkeypatch):
    driver = _Driver(monkeypatch, free=GB, cached=GB // 2)
    ref = _net_ref(RecordingElastic(covering=True))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 1
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_outside_a_grant_the_net_is_todays_eviction(monkeypatch):
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    ref = _net_ref(RecordingElastic(covering=False))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 0
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_a_stand_in_whose_attributes_are_mocks_is_not_a_grant(monkeypatch):
    """A MagicMock model answers every attribute, `covering` included, with a truthy
    mock. The net must run today's eviction for it."""
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    ref = MagicMock()
    ref._moe_prefill_pass = True
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 0
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_with_headroom_to_spare_the_net_touches_nothing(monkeypatch):
    driver = _Driver(monkeypatch, free=mp._PREFILL_VRAM_HEADROOM, cached=GB)
    ref = _net_ref(RecordingElastic(covering=True))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 0
    ref._expert_vram_cache.evict_until_free.assert_not_called()


# ── the layer range ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(LAYER_GEOMETRIES))
def test_the_layer_range_follows_the_budget(name, capsys):
    geo = dict(LAYER_GEOMETRIES[name])
    geo["working_set_bytes"] = geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"]
    blocks = [types.SimpleNamespace(layer_idx=i) for i in range(geo["num_moe_layers"])]
    model = types.SimpleNamespace(_moe_geometry=geo, _moe_blocks=blocks,
                                  _vram_cacheable_layers=None)
    set_tier_layer_range(model, geo["working_set_bytes"])
    assert model._vram_cacheable_layers is None and capsys.readouterr().out == ""
    per_layer = geo["top_k"] * geo["expert_bytes"]
    set_tier_layer_range(model, 3 * per_layer + 1)
    assert model._vram_cacheable_layers == {0, 1, 2}
    assert "restricting the VRAM cache to the first 3/" in capsys.readouterr().out
    set_tier_layer_range(model, 3 * per_layer + 2)          # same range: no second line
    assert capsys.readouterr().out == ""
    set_tier_layer_range(model, 0)
    assert model._vram_cacheable_layers == set()
    set_tier_layer_range(model, geo["working_set_bytes"])
    assert model._vram_cacheable_layers is None


# ── the load-time build ──────────────────────────────────────────────────────────


def _loadable(tier, device="cuda:0", decode_graphs=None, plan=PLAN):
    geo = dict(LAYER_GEOMETRIES["gptoss_like"])
    geo["working_set_bytes"] = geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"]
    return types.SimpleNamespace(
        _expert_vram_cache=tier, _vram_plan=plan, _vram_shape=SHAPE, device=device,
        _moe_geometry=geo, _decode_graphs=decode_graphs, prefill_chunk_size=512,
        _moe_blocks=[types.SimpleNamespace(layer_idx=i) for i in range(24)],
        _vram_cacheable_layers=None)


def test_the_load_builds_an_elastic_tier_from_the_plan_and_the_tier(cuda, capsys):
    tier = ExpertVRAMCache(191 * EXPERT)
    model = _loadable(tier)
    elastic = VaayuBaseModel._build_elastic_tier(model)
    assert isinstance(elastic, ElasticTier)
    assert elastic.base == 191 * EXPERT and elastic.base_floor == _VRAM_RESERVE_FLOOR_BYTES
    assert elastic.grant_step == GRANT_CHUNKS * 512 and elastic.expert_bytes == EXPERT
    assert "elastic expert tier ON" in capsys.readouterr().out
    # its budget writes reach the model's layer range
    elastic._set_budget(4 * EXPERT)
    assert tier.budget_bytes == 4 * EXPERT and model._vram_cacheable_layers == {0}
    # and its readings are the driver's and the allocator's
    cuda["free"] = 5 * GB
    assert elastic._driver_free() == 5 * GB
    assert elastic._allocated is torch.cuda.memory_allocated


@pytest.mark.parametrize("case", ["no tier", "cpu", "no cuda", "floor plan", "decode graphs"])
def test_the_load_builds_none_when_there_is_nothing_to_lend_or_nowhere_to_lend_it(
        case, cuda, monkeypatch, capsys):
    tier = ExpertVRAMCache(191 * EXPERT)
    floor_plan = VramPlan(max_seq_len=8192, reserve_bytes=_VRAM_RESERVE_FLOOR_BYTES,
                          pinning_budget=0, s_max=65536, reason="flash")
    model = {"no tier": lambda: _loadable(None),
             "cpu": lambda: _loadable(tier, device="cpu"),
             "no cuda": lambda: _loadable(tier),
             "floor plan": lambda: _loadable(tier, plan=floor_plan),
             "decode graphs": lambda: _loadable(tier, decode_graphs=object())}[case]()
    if case == "no cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert VaayuBaseModel._build_elastic_tier(model) is None
    out = capsys.readouterr().out
    assert ("elastic expert tier OFF" in out) == (case == "decode graphs")
    assert "elastic expert tier ON" not in out
