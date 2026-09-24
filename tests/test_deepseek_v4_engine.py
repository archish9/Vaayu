"""A5.7a -- the engine change that lets Vaayu's streamed path run DeepSeek-V4.

None of this needs the 157 G checkpoint or a GPU. `VaayuBaseModel.__init__`
loads a model, so every test here builds its subject with `__new__` and sets
only the attributes under test -- the pattern `tests/test_decode_top_k.py:321`
and `tests/test_keepawake_wiring.py:30` already use.
"""
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"

# The shipped config.json carries no `fp4_group_size`; A5.7b's loader is what
# will supply it off the shard headers. A test parametrization, never a
# production literal. Same value and same reason as
# `tests/test_deepseek_v4_model.py:29`.
_TEST_FP4_GROUP_SIZE = 32


def shipped_config(**overrides) -> DeepseekV4Config:
    shipped = json.loads(FIXTURE.read_text())
    shipped.setdefault("fp4_group_size", _TEST_FP4_GROUP_SIZE)
    shipped.update(overrides)
    return DeepseekV4Config(**shipped)


def test_an_fp8_linear_without_a_weight_block_size_names_the_missing_field():
    with pytest.raises(ValueError, match="weight_block_size"):
        modeling.DeepseekV4Linear(64, 32, dtype=torch.float8_e4m3fn,
                                  weight_block_size=None)


def test_an_fp8_linear_with_a_weight_block_size_still_builds():
    linear = modeling.DeepseekV4Linear(64, 32, dtype=torch.float8_e4m3fn,
                                       weight_block_size=[128, 128])
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.scale is not None


def test_a_bf16_linear_needs_no_weight_block_size():
    """The regression the obvious fix would have caused.

    `_expert_dtypes` returns bf16 for any checkpoint whose `fmt` is not
    "e4m3", and a bf16 weight has no per-tile scale to shape. Refusing here
    would refuse a valid checkpoint of this architecture (CLAUDE.md §5).
    """
    linear = modeling.DeepseekV4Linear(64, 32, dtype=torch.bfloat16,
                                       weight_block_size=None)
    assert linear.weight.dtype == torch.bfloat16
    assert linear.scale is None
    assert linear.act_block_size is None


def test_run_embed_defaults_to_calling_the_embedding_unchanged():
    """Every model that is not DeepSeek-V4 must be untouched by this hook."""
    model = VaayuBaseModel.__new__(VaayuBaseModel)
    layer = nn.Embedding(8, 4)
    seq = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.run_embed(layer, seq), layer(seq))


def test_the_layer_major_loop_calls_run_embed_rather_than_the_layer():
    """Pins the call site.

    `VaayuBaseModel.forward` is a ~1400-line loop that needs a loaded model
    and a GPU to run, so this reads the source instead. That is the point:
    the regression this guards against is someone "simplifying"
    `self.run_embed(layer, seq)` back to `layer(seq)`, which would silently
    stop hyper-connections from ever happening and leave every test green.
    """
    source = inspect.getsource(VaayuBaseModel.forward)
    assert "self.run_embed(layer, seq)" in source
    assert "batch[j] = layer(seq)" not in source


def test_the_decoder_layer_accepts_the_loops_use_cache_kwarg():
    """`vaayu_base.py:5209` hardcodes use_cache into every layer call."""
    params = inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters
    assert "use_cache" in params
    assert params["use_cache"].default is None


def test_the_decoder_layer_still_refuses_an_unknown_kwarg():
    """No `**kwargs`. A catch-all would swallow a future engine kwarg."""
    params = inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values())
    assert set(params) == {"self", "hidden_states", "past_key_values",
                           "start_pos", "use_cache"}


def engine(**attrs):
    """A `VaayuDeepseekV4` with only the attributes under test.

    `VaayuBaseModel.__init__` loads a model; these hooks need none of it.
    """
    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def test_the_cache_is_a_deepseek_cache_sized_off_the_engines_max_seq_len():
    # A SMALL max_seq_len on purpose: this cache is PREALLOCATED across every
    # layer, so the shipped 43 at a real context would allocate hundreds of MB
    # in a unit test.
    config = shipped_config(num_hidden_layers=2)
    model = engine(config=config, max_seq_len=256, kv_cache_mode="auto",
                   running_device="cpu", running_dtype=torch.bfloat16)
    cache = model._create_kv_cache()
    assert isinstance(cache, modeling.DeepseekV4Cache)
    assert cache.max_seq_len == 256
    assert len(cache.layers) == config.num_hidden_layers


def test_a_non_default_kv_cache_mode_is_overridden_loudly(capsys):
    """The knob is not silently ignored.

    This cache has no fp8 / bf16 / TurboQuant variant, so honouring
    kv_cache_mode is impossible -- but ignoring it without a word is how a
    user spends an afternoon wondering why their setting did nothing.
    """
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="turboquant:4", running_device="cpu",
                   running_dtype=torch.bfloat16)
    model._create_kv_cache()
    assert "turboquant:4" in capsys.readouterr().out


def test_run_embed_widens_the_residual_to_hc_mult_copies():
    config = shipped_config()
    model = engine(config=config)
    layer = nn.Embedding(8, config.hidden_size)
    seq = torch.tensor([[1, 2, 3]])
    out = model.run_embed(layer, seq)
    assert out.shape == (1, 3, config.hc_mult, config.hidden_size)
    # Every copy is the embedding, unchanged.
    for i in range(config.hc_mult):
        assert torch.equal(out[:, :, i, :], layer(seq))


def test_run_embed_reads_hc_mult_off_the_config_not_a_literal():
    config = shipped_config(hc_mult=3)
    model = engine(config=config)
    layer = nn.Embedding(8, config.hidden_size)
    out = model.run_embed(layer, torch.tensor([[1, 2]]))
    assert out.shape[2] == 3


def test_run_norm_calls_hc_head_before_the_norm_module():
    """Ordering is the whole point: hc_head collapses [b,s,hc,d] -> [b,s,d],
    so a norm that ran first would normalise the wrong axis."""
    calls = []

    class FakeInner:
        def hc_head(self, x):
            calls.append("hc_head")
            return x.sum(dim=2)

    class FakeOuter:
        model = FakeInner()

    class RecordingNorm(nn.Module):
        def forward(self, x):
            calls.append("norm")
            return x

    model = engine(model=FakeOuter())
    seq = torch.zeros(1, 3, 4, 8)
    out = model.run_norm(RecordingNorm(), seq)
    assert calls == ["hc_head", "norm"]
    assert out.shape == (1, 3, 8)


def test_get_pos_emb_args_passes_the_past_length_as_start_pos():
    assert engine().get_pos_emb_args(17, 5) == {"start_pos": 17}


def test_the_mask_and_position_id_hooks_pass_nothing():
    """sparse_attn masks itself by reading -1 out of topk_idxs
    (`kernel.py:322`) and position lives in start_pos, so an HF mask or
    position_ids would be an unexpected kwarg on the decoder layer."""
    model = engine()
    assert model.get_attention_mask_args(torch.ones(1, 1, 4, 4), 0, 4) == {}
    assert model.get_position_ids_args(torch.arange(4)[None], 0, 4) == {}


def test_the_hooks_produce_exactly_the_layers_signature():
    """The union of what the loop passes must be what the layer takes."""
    model = engine()
    emitted = {"use_cache"}
    emitted |= set(model.get_past_key_value_args(object()))
    emitted |= set(model.get_pos_emb_args(0, 4))
    emitted |= set(model.get_attention_mask_args(None, 0, 4))
    emitted |= set(model.get_position_ids_args(None, 0, 4))
    accepted = set(
        inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters)
    accepted -= {"self", "hidden_states"}
    assert emitted == accepted
