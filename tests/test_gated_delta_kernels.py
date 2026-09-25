"""S3 (2026-09-25): fused Triton kernels on the gated-delta mixers' three seams.

transformers' gated-delta-net mixers decode through `recurrent_gated_delta_rule`,
`causal_conv1d_update` and a gated `norm`, which fall back to torch without
`fla` / `causal_conv1d`. On Qwen3-Next that fallback is 45 launches per mixer per
decode token on a host-bound phase; `vaayu/gated_delta_kernels.py` replaces each
seam with one kernel. These tests pin: which modules get which seam (by surface,
never by name -- CLAUDE.md §5), that each kernel is no further from an fp64
reference than the stock seam, and that the installer falls back to stock,
loudly, whenever its on-device self-check fails.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextRMSNormGated,
    torch_causal_conv1d_update as stock_conv,
    torch_recurrent_gated_delta_rule as stock_rule,
)

import vaayu.gated_delta_kernels as gdk
from vaayu.recurrent_state import install_state_continuation
from tests.test_hybrid_recurrent_state import FAMILIES, T, _cache, _mixer, _qwen3_next, _run

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels are CUDA-only")

# Two contrasting geometries (CLAUDE.md §5's test rule): Qwen3-Next-80B's
# linear-attention shape, and one with non-power-of-2 heads and conv width.
GEOMS = {
    "qwen3next": dict(HV=32, HK=16, K=128, V=128, W=4),
    "odd": dict(HV=6, HK=3, K=48, V=80, W=3),
}
NO_SEAMS = {"rule": 0, "conv": 0, "norm": 0}


def _conv_dim(g):
    return 2 * g["HK"] * g["K"] + g["HV"] * g["V"]


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _counts(model):
    return {s: len(m) for s, m in gdk.gated_delta_seams(model).items()}


def _qwen3_next_model():
    from transformers import Qwen3NextForCausalLM
    cfg, _ = _qwen3_next()
    return Qwen3NextForCausalLM(cfg)


class _RuleAndNormOnly(torch.nn.Module):
    """olmo_hybrid's shape: a gated-delta mixer with no conv-update seam."""

    def __init__(self, norm=None):
        super().__init__()
        self.recurrent_gated_delta_rule = stock_rule
        self.norm = norm if norm is not None else Qwen3NextRMSNormGated(16)


# ── detection (CPU) ──────────────────────────────────────────────────────────

def test_seams_found_on_every_linear_attention_mixer_and_nowhere_else():
    model = _qwen3_next_model()
    n = model.config.layer_types.count("linear_attention")
    assert n > 0
    assert _counts(model) == {"rule": n, "conv": n, "norm": n}


def test_qwen3_5_mixer_exposes_all_three_seams():
    _, mixer = _mixer("qwen3_5")
    assert _counts(mixer) == {"rule": 1, "conv": 1, "norm": 1}


def test_a_mixer_without_conv_update_gets_rule_and_norm_only():
    assert _counts(_RuleAndNormOnly()) == {"rule": 1, "conv": 0, "norm": 1}


def test_no_seams_on_a_dense_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=2))
    # The trap this pins: a dense model's top-level norm HAS weight + variance_epsilon.
    assert hasattr(llama.model.norm, "variance_epsilon")
    assert _counts(llama) == NO_SEAMS
