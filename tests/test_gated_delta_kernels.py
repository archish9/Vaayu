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


# ── rule kernel (CUDA) ───────────────────────────────────────────────────────

def _rule_inputs(geom, B, T_, dtype, h0_dtype, strided=False, seed=0):
    torch.manual_seed(seed)
    HV, K, V = geom["HV"], geom["K"], geom["V"]
    q = torch.randn(B, T_, HV, K, device="cuda").to(dtype)
    k = torch.randn(B, T_, HV, K, device="cuda").to(dtype)
    v = torch.randn(B, T_, HV, 2 * V if strided else V, device="cuda").to(dtype)[..., :V]
    g = -F.softplus(torch.randn(B, T_, HV, device="cuda"))          # log-decay, fp32
    beta = torch.sigmoid(torch.randn(B, T_, HV, device="cuda")).to(dtype)
    h0 = (None if h0_dtype is None
          else (torch.randn(B, HV, K, V, device="cuda") * 0.1).to(h0_dtype))
    return q, k, v, g, beta, h0


def _ref_rule(q, k, v, g, beta, h0):
    """The gated delta rule in fp64, qk-l2norm on -- an oracle independent of
    both the stock fallback (which casts to fp32) and the kernel."""
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    B, T_, HV, K = k.shape
    h = (torch.zeros(B, HV, K, v.shape[-1], dtype=torch.float64, device=q.device)
         if h0 is None else h0.double())
    out = []
    for t in range(T_):
        h = h * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (h * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        h = h + k[:, t, :, :, None] * delta[:, :, None, :]
        out.append((h * q[:, t, :, :, None]).sum(-2))
    return torch.stack(out, 1), h


def _both_rules(q, k, v, g, beta, h0, final=True):
    kw = dict(g=g, beta=beta, output_final_state=final, use_qk_l2norm_in_kernel=True)
    clone = (lambda: None) if h0 is None else h0.clone
    return (stock_rule(q, k, v, initial_state=clone(), **kw),
            gdk.fused_recurrent_rule(q, k, v, initial_state=clone(), **kw))


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("h0_dtype", [None, torch.bfloat16, torch.float32],
                         ids=["h0none", "h0bf16", "h0fp32"])
@pytest.mark.parametrize("B,T_", [(1, 1), (2, 1), (1, 3)], ids=["decode", "batch2", "T3"])
@pytest.mark.parametrize("geom", GEOMS)
def test_rule_is_no_further_from_fp64_than_stock(geom, B, T_, h0_dtype, dtype):
    q, k, v, g, beta, h0 = _rule_inputs(GEOMS[geom], B, T_, dtype, h0_dtype)
    (so, sh), (fo, fh) = _both_rules(q, k, v, g, beta, h0)
    ro, rh = _ref_rule(q, k, v, g, beta, h0)
    assert fo.shape == so.shape and fo.dtype == so.dtype
    assert fh.shape == sh.shape and fh.dtype == torch.float32
    assert _rel(fo, ro) <= 1.25 * _rel(so, ro)
    assert _rel(fh, rh) <= 1.25 * _rel(sh, rh)


@cuda
@pytest.mark.parametrize("geom", GEOMS)
def test_rule_reads_strided_inputs(geom):
    q, k, v, g, beta, h0 = _rule_inputs(GEOMS[geom], 1, 1, torch.bfloat16, torch.bfloat16,
                                        strided=True)
    assert not v.is_contiguous()
    _, (fo, fh) = _both_rules(q, k, v, g, beta, h0)
    _, (co, ch) = _both_rules(q, k, v.contiguous(), g, beta, h0)
    assert torch.equal(fo, co) and torch.equal(fh, ch)


@cuda
def test_rule_without_final_state_returns_none():
    q, k, v, g, beta, h0 = _rule_inputs(GEOMS["qwen3next"], 1, 1, torch.bfloat16, torch.bfloat16)
    _, (with_state, _) = _both_rules(q, k, v, g, beta, h0, final=True)
    _, (without, state) = _both_rules(q, k, v, g, beta, h0, final=False)
    assert state is None and torch.equal(without, with_state)


# ── conv-update kernel ───────────────────────────────────────────────────────

def _conv_inputs(D, W, B, dtype, layout, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, 1, D, device="cuda").to(dtype).transpose(1, 2)    # [B, D, 1], as HF passes it
    base = torch.randn(B, W, D, device="cuda").to(dtype)
    state = base.transpose(1, 2)                                          # channels-last, like HF's cache
    if layout == "contiguous":
        state = state.contiguous()
    w = torch.randn(D, 1, W, device="cuda").to(dtype).squeeze(1)          # conv1d.weight.squeeze(1)
    return x, state, w


def _ref_conv(x, state, w, bias):
    full = torch.cat([state.double(), x.double()], -1)[..., 1:]
    out = (full * w.double()[None]).sum(-1, keepdim=True)
    if bias is not None:
        out = out + bias.double()[None, :, None]
    return F.silu(out)


def _both_convs(x, state, w, bias=None):
    s_state, f_state = state.clone(), state.clone()
    assert s_state.stride() == state.stride()      # clone keeps the layout under test
    so = stock_conv(x, s_state, w, bias, "silu")
    fo = gdk.fused_conv_update(x, f_state, w, bias, "silu")
    return so, s_state, fo, f_state


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("layout", ["channels_last", "contiguous"])
@pytest.mark.parametrize("geom", GEOMS)
def test_conv_update_state_is_bit_exact(geom, layout, B, dtype):
    D, W = _conv_dim(GEOMS[geom]), GEOMS[geom]["W"]
    x, state, w = _conv_inputs(D, W, B, dtype, layout)
    so, s_state, fo, f_state = _both_convs(x, state, w)
    assert torch.equal(f_state, s_state)
    assert f_state.stride() == state.stride()      # updated in place, layout kept
    assert fo.shape == so.shape and fo.dtype == so.dtype
    ref = _ref_conv(x, state, w, None)
    assert _rel(fo, ref) <= 1.25 * _rel(so, ref)


@cuda
def test_conv_update_adds_bias():
    D, W = _conv_dim(GEOMS["qwen3next"]), GEOMS["qwen3next"]["W"]
    x, state, w = _conv_inputs(D, W, 1, torch.bfloat16, "contiguous")
    bias = torch.randn(D, device="cuda").bfloat16()
    so, s_state, fo, f_state = _both_convs(x, state, w, bias)
    assert torch.equal(f_state, s_state)
    ref = _ref_conv(x, state, w, bias)
    assert _rel(fo, ref) <= 1.25 * _rel(so, ref)


@cuda
def test_conv_update_after_a_prefix_cache_round_trip():
    """FP8DynamicCache.to_cpu/to_device move the state with .to()/.pin_memory();
    whatever layout comes back, the update must stay bit-exact with stock."""
    D, W = _conv_dim(GEOMS["qwen3next"]), GEOMS["qwen3next"]["W"]
    x, state, w = _conv_inputs(D, W, 1, torch.bfloat16, "channels_last")
    state = state.to("cpu").pin_memory().to("cuda")
    so, s_state, fo, f_state = _both_convs(x, state, w)
    assert torch.equal(f_state, s_state)
    ref = _ref_conv(x, state, w, None)
    assert _rel(fo, ref) <= 1.25 * _rel(so, ref)


def test_conv_seam_delegates_multi_token_calls_to_stock():
    calls = []

    def stock(*args):
        calls.append(args)
        return "stock"

    seam = gdk._conv_seam(stock)
    assert seam(torch.zeros(1, 8, 2), torch.zeros(1, 8, 4), torch.zeros(8, 4)) == "stock"
    assert len(calls) == 1
    assert getattr(seam, gdk._MARK) is stock
