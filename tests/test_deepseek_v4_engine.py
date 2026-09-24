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
