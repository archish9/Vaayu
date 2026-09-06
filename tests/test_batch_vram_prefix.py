"""P6: batched_generate must size the VRAM cacheable prefix for its own N.

The load-time call to compute_vram_layer_range (vaayu_base.py:2578) runs before
any batch exists and knows only one stream's top_k. Measured consequence at N=3
(docs/2026-09-06-P1-vram-resident-dispatch-RESULTS.md §4): VRAM cache hit rate
44.0% -> 0.1% on Qwen and 29.0% -> 2.4% on MiniMax, with near-total eviction
churn. These tests drive the REAL batched_generate against the smallest
model-shaped fake that reaches its decode loop.
"""
from types import SimpleNamespace

import pytest
import torch

from vaayu.batch import BatchRequest
from vaayu.vaayu_base import (VaayuBaseModel, batch_union_per_layer,
                              compute_vram_layer_range)

QWEN_GEO = {"expert_bytes": int(11.83e6), "top_k": 8, "num_moe_layers": 94,
            "num_experts": 128, "working_set_bytes": 8 * 94 * int(11.83e6)}
QWEN_BUDGET = int(4.09e9)
# MiniMax is the case with NO load-time restriction at all (its single-stream
# working set fits), so it is the one that must gain one at N=3.
MINIMAX_GEO = {"expert_bytes": int(8.88e6), "top_k": 8, "num_moe_layers": 62,
               "num_experts": 256, "working_set_bytes": 8 * 62 * int(8.88e6)}
MINIMAX_BUDGET = int(7.45e9)

GEOMETRIES = [("qwen", QWEN_GEO, QWEN_BUDGET, list(range(3, 97))),
              ("minimax", MINIMAX_GEO, MINIMAX_BUDGET, list(range(62)))]


class _FakeCache:
    def __init__(self, s, b=1):
        self._s, self._b = s, b

    def get_seq_length(self):
        return self._s


class _FakeMoEModel:
    """Enough of VaayuBaseModel for batched_generate to reach decode, plus the
    three MoE attributes the prefix sizing reads."""
    _active_attn_impl = "sdpa"
    running_device = "cpu"
    prefill_chunk_size = 0

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    class _Cfg:
        eos_token_id = -999          # unreachable: fake logits are all zero

    class _Telemetry:
        def reset_batch_overlap(self):
            pass

        def dump_json(self, path):
            pass

    def __init__(self, geo, budget, layer_ids, load_time_range):
        self._moe_geometry = geo
        self._moe_blocks = [SimpleNamespace(layer_idx=i) for i in layer_ids]
        self._expert_vram_cache = SimpleNamespace(budget_bytes=budget)
        self._vram_cacheable_layers = load_time_range
        self.tokenizer = self._Tok()
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()
        self.seen_during_decode = []

    def _assert_context_fits(self, n_tokens, N=1):
        pass

    def forward(self, input_ids, attention_mask, position_ids, use_cache=True,
                return_dict=True, past_key_values=None):
        # What the MoE forwards read per layer-visit -- record what they WOULD see.
        self.seen_during_decode.append(self._vram_cacheable_layers)
        return SimpleNamespace(
            past_key_values=_FakeCache(s=attention_mask.shape[1], b=input_ids.shape[0]),
            logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 8))


def _model(geo, budget, layer_ids):
    load_time = compute_vram_layer_range(budget, geo, layer_ids)
    return _FakeMoEModel(geo, budget, layer_ids, load_time), load_time


def _reqs(n, max_new=2):
    return [BatchRequest(prompt_ids=[1, 2, 3, 4 + i], max_new_tokens=max_new,
                         temperature=0.0)
            for i in range(n)]


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_batch_prefix_is_sized_for_the_batch_union(name, geo, budget, layer_ids):
    model, load_time = _model(geo, budget, layer_ids)
    expected = compute_vram_layer_range(
        budget, geo, layer_ids, experts_per_layer=batch_union_per_layer(geo, 3))

    list(VaayuBaseModel.batched_generate(model, _reqs(3)))

    assert model.seen_during_decode, "forward never ran"
    assert all(s == expected for s in model.seen_during_decode)
    assert expected is not None and len(expected) < len(layer_ids)
    # and it is genuinely tighter than what the load-time sizing left in place
    assert load_time is None or len(expected) < len(load_time)


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_load_time_prefix_is_restored_after_the_batch(name, geo, budget, layer_ids):
    model, load_time = _model(geo, budget, layer_ids)
    list(VaayuBaseModel.batched_generate(model, _reqs(3)))
    assert model._vram_cacheable_layers == load_time


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_prefix_is_restored_when_the_generator_is_abandoned(name, geo, budget,
                                                            layer_ids):
    """batched_generate's docstring treats early abandonment as real; a leaked
    batch-sized prefix would silently shrink caching for every later
    single-stream call."""
    model, load_time = _model(geo, budget, layer_ids)
    gen = VaayuBaseModel.batched_generate(model, _reqs(3))
    next(gen)
    gen.close()
    assert model._vram_cacheable_layers == load_time


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_n1_batch_leaves_the_load_time_prefix_untouched(name, geo, budget,
                                                        layer_ids):
    """N=1 is the geometry the load-time sizing was already right for -- it must
    be byte-identical to today, not re-derived into a different set."""
    model, load_time = _model(geo, budget, layer_ids)
    list(VaayuBaseModel.batched_generate(model, _reqs(1)))
    assert all(s == load_time for s in model.seen_during_decode)
    assert model._vram_cacheable_layers == load_time


def test_non_moe_model_is_untouched():
    """Dense: no expert cache, no geometry -- the sizing must not run at all
    (CLAUDE.md §0: MoE work must not touch the dense path)."""
    model = _FakeMoEModel(QWEN_GEO, QWEN_BUDGET, list(range(94)), None)
    del model._expert_vram_cache
    del model._moe_geometry
    list(VaayuBaseModel.batched_generate(model, _reqs(3)))
    assert model._vram_cacheable_layers is None
