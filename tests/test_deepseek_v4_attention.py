"""A5.5 -- MLA + sparse-index attention.

Split the way A5.3's tests are: the structural half needs neither the 157 G
checkpoint nor a GPU, and only the parity half carries both markers.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"


def shipped_config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


def test_rms_norm_matches_the_reference_formula():
    """fp32 internally, cast back on the way out, weight held fp32.

    The checkpoint stores this weight bf16; the reference declares the
    Parameter fp32 and lets `load_state_dict` upcast. Declaring bf16 here
    would silently change the norm's precision on every layer.
    """
    norm = modeling.DeepseekV4RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, 8))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    got = norm(x)
    f = x.float()
    want = (norm.weight * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)))

    assert norm.weight.dtype is torch.float32
    assert got.dtype is torch.bfloat16
    assert torch.equal(got, want.to(torch.bfloat16))


def test_rope_settings_splits_the_two_variants_off_the_config():
    """compress_ratio == 0 disables YaRN and uses the BASE theta; compressed
    layers use compress_rope_theta with YaRN. Two variants, and
    precompute_freqs_cis is lru_cache(2) -- a third call pattern evicts one.
    """
    config = shipped_config()
    plain = modeling.rope_settings(config, 0)
    compressed = modeling.rope_settings(config, 4)

    assert plain["original_seq_len"] == 0, "YaRN off on uncompressed layers"
    assert plain["base"] == config.rope_theta
    assert compressed["base"] == config.compress_rope_theta
    # transformers 5.5.4 folds rope_theta in and renames the length key.
    assert compressed["original_seq_len"] == \
        config.rope_parameters["original_max_position_embeddings"]
    assert compressed["factor"] == config.rope_parameters["factor"]
    assert modeling.rope_settings(config, 128) == compressed


def test_apply_rotary_emb_is_in_place_and_inverse_undoes_it():
    """The reference mutates its slice with no assignment (model.py:505) and
    applies the INVERSE to the attention output (model.py:539). A port that
    returns a new tensor silently drops the rotation at every call site that
    ignores the return value.
    """
    freqs = modeling.precompute_freqs_cis(8, 16, 0, 10000.0, 1.0, 32, 1,
                                          device=torch.device("cpu"))
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    original = x.clone()

    returned = modeling.apply_rotary_emb(x, freqs[:4])
    assert returned is x, "must rotate in place"
    assert not torch.equal(x, original)

    modeling.apply_rotary_emb(x, freqs[:4], inverse=True)
    assert torch.allclose(x.float(), original.float(), atol=2e-2)
