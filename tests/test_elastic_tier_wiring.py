"""S35-E: where the engine hands a request to the elastic tier, and where it must not.

Stand-in models and a recording stand-in for ElasticTier -- the decisions themselves
are tested in tests/test_elastic_tier.py against a simulated card. What is pinned
here: the admission branch (single stream only), generate's prompt length, the batch
path's suspend, the prefill net, the forward hook (behaviourally, on the real forward),
the layer range and the load-time build.
"""
import types
from unittest.mock import MagicMock

import pytest
import torch

import vaayu.moe_patch as mp
from tests.test_s33_host_edits import _Cache, make
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
    def __init__(self, lent=True, admit_returns=123, admit_raises=None):
        self.covering, self.lent = True, lent
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


def test_batched_generate_takes_the_lend_back_before_its_own_admission():
    elastic = RecordingElastic()
    stub = types.SimpleNamespace(
        _active_attn_impl="sdpa", running_device="cpu", _elastic_tier=elastic,
        _assert_context_fits=lambda *a, **k: pytest.fail("admission ran before suspend"))
    with pytest.raises(Stop):
        next(VaayuBaseModel.batched_generate(stub, [object(), object()]))
    assert elastic.calls == [("suspend",)]


# ── the forward hook (stage 2) ───────────────────────────────────────────────────


class _RecordingCover:
    """What forward's hook sees: the argument, and how many layers had run by then."""
    covering = True

    def __init__(self, model):
        self.model, self.calls = model, []

    def cover(self, context_after):
        self.calls.append((context_after, sum(len(d.calls) for d in self.model.dec)))


@pytest.mark.parametrize("impl", ["flash_attention_2", "sdpa", "eager"])
@pytest.mark.parametrize("new, past", [(1, 5), (3, 0), (4, 9)])
def test_forward_covers_the_context_it_leaves_behind_before_any_layer_runs(impl, new, past):
    """The REAL forward, on the stand-in model of tests/test_s33_host_edits.py: one
    cover call per pass, with past + this pass's tokens, and before the first layer
    -- on a decode step, a bare first pass and a multi-token pass, under every
    attention kernel (gpt-oss runs eager). The no-OOM argument rests on this order."""
    model = make()
    model._active_attn_impl = impl
    model.kv_cache_mode = "bf16"       # with no cache passed, forward builds its own
    if impl == "eager":
        model.max_seq_len = 64
    model._elastic_tier = _RecordingCover(model)
    VaayuBaseModel.forward(model, input_ids=torch.zeros(1, new, dtype=torch.long),
                           past_key_values=(_Cache(past) if past else None), use_cache=True)
    assert model._elastic_tier.calls == [(past + new, 0)]
    assert sum(len(d.calls) for d in model.dec) > 0, "the layers must have run after it"


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


def test_on_a_lent_request_the_net_returns_the_allocators_cache_before_it_evicts(
        monkeypatch):
    """Driver-free under the headroom, but only because the prefill's own blocks are
    cached: on a request that is lent to that is not pressure, and nothing is evicted."""
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    ref = _net_ref(RecordingElastic(lent=True))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 1
    ref._expert_vram_cache.evict_until_free.assert_not_called()


def test_on_a_lent_request_the_net_still_evicts_when_the_cache_was_not_the_reason(
        monkeypatch):
    driver = _Driver(monkeypatch, free=GB, cached=GB // 2)
    ref = _net_ref(RecordingElastic(lent=True))
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 1
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_on_a_request_that_is_lent_nothing_the_net_is_todays_eviction(monkeypatch):
    """The static plan keeps both its halves: the request was admitted by the elastic
    tier (`covering`), is lent nothing, and the net evicts on the raw reading -- which,
    with the last layer's score buffer still cached, is a reading taken at the peak."""
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    elastic = RecordingElastic(lent=False)
    assert elastic.covering is True
    ref = _net_ref(elastic)
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 0
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_a_stand_in_whose_attributes_are_mocks_is_not_a_grant(monkeypatch):
    """A MagicMock model answers every attribute, `lent` included, with a truthy
    mock. The net must run today's eviction for it."""
    driver = _Driver(monkeypatch, free=GB, cached=3 * GB)
    ref = MagicMock()
    ref._moe_prefill_pass = True
    mp._ensure_prefill_headroom(ref)
    assert driver.empties == 0
    ref._expert_vram_cache.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_with_headroom_to_spare_the_net_touches_nothing(monkeypatch):
    driver = _Driver(monkeypatch, free=mp._PREFILL_VRAM_HEADROOM, cached=GB)
    ref = _net_ref(RecordingElastic(lent=True))
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


def _loadable(tier, device="cuda:0", decode_graphs=None, speculator=None, plan=PLAN,
              geometry="gptoss_like", chunk=512, preallocated=False):
    geo = dict(LAYER_GEOMETRIES[geometry])
    geo["working_set_bytes"] = geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"]
    return types.SimpleNamespace(
        _expert_vram_cache=tier, _vram_plan=plan, _vram_shape=SHAPE, device=device,
        _moe_geometry=geo, _decode_graphs=decode_graphs, speculator=speculator,
        prefill_chunk_size=chunk, kv_cache_preallocated=lambda: preallocated,
        _moe_blocks=[types.SimpleNamespace(layer_idx=i) for i in range(geo["num_moe_layers"])],
        _vram_cacheable_layers=None)


@pytest.mark.parametrize("geometry, chunk", [("gptoss_like", 512), ("minimax_like", 384)])
def test_the_load_builds_an_elastic_tier_from_the_plan_and_the_tier(geometry, chunk, cuda,
                                                                     capsys):
    expert = LAYER_GEOMETRIES[geometry]["expert_bytes"]
    top_k = LAYER_GEOMETRIES[geometry]["top_k"]
    tier = ExpertVRAMCache(191 * expert)
    model = _loadable(tier, geometry=geometry, chunk=chunk)
    elastic = VaayuBaseModel._build_elastic_tier(model)
    assert isinstance(elastic, ElasticTier)
    assert elastic.base == 191 * expert and elastic.base_floor == _VRAM_RESERVE_FLOOR_BYTES
    assert elastic.grant_step == GRANT_CHUNKS * chunk and elastic.expert_bytes == expert
    assert "elastic expert tier ON" in capsys.readouterr().out
    # its budget writes reach the model's layer range: one layer's top_k experts fit
    elastic._set_budget(top_k * expert)
    assert tier.budget_bytes == top_k * expert and model._vram_cacheable_layers == {0}
    # and its readings are the driver's and the allocator's
    cuda["free"] = 5 * GB
    assert elastic._driver_free() == 5 * GB
    assert elastic._allocated is torch.cuda.memory_allocated


@pytest.mark.parametrize("case", ["no tier", "cpu", "no cuda", "floor plan", "decode graphs",
                                  "speculator", "preallocated kv"])
def test_the_load_builds_none_when_there_is_nothing_to_lend_or_nowhere_to_lend_it(
        case, cuda, monkeypatch, capsys):
    tier = ExpertVRAMCache(191 * EXPERT)
    floor_plan = VramPlan(max_seq_len=8192, reserve_bytes=_VRAM_RESERVE_FLOOR_BYTES,
                          pinning_budget=0, s_max=65536, reason="flash")
    model = {"no tier": lambda: _loadable(None),
             "cpu": lambda: _loadable(tier, device="cpu"),
             "no cuda": lambda: _loadable(tier),
             "floor plan": lambda: _loadable(tier, plan=floor_plan),
             "decode graphs": lambda: _loadable(tier, decode_graphs=object()),
             "speculator": lambda: _loadable(tier, speculator=object()),
             "preallocated kv": lambda: _loadable(tier, preallocated=True)}[case]()
    if case == "no cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert VaayuBaseModel._build_elastic_tier(model) is None
    out = capsys.readouterr().out
    assert ("elastic expert tier OFF: decode graphs" in out) == (case == "decode graphs")
    assert ("elastic expert tier OFF: a speculator" in out) == (case == "speculator")
    assert ("elastic expert tier OFF: this model's KV cache" in out) == (case == "preallocated kv")
    assert "elastic expert tier ON" not in out


def test_only_a_model_that_brings_a_preallocated_cache_says_so():
    """The base answers for every cache Vaayu builds (they grow with the context);
    DeepSeek-V4 allocates its cache for max_seq_len in _create_kv_cache."""
    from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4
    assert VaayuBaseModel.kv_cache_preallocated(object()) is False
    assert VaayuDeepseekV4.kv_cache_preallocated(object()) is True
    assert "max_seq_len=self.max_seq_len" in __import__("inspect").getsource(
        VaayuDeepseekV4._create_kv_cache)
