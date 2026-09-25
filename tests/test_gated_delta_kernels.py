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
    """A synthetic mixer with no conv-update seam (not olmo_hybrid's real shape
    -- see test_olmo_hybrid_mixer_exposes_only_the_rule_seam for that)."""

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


def test_olmo_hybrid_mixer_exposes_only_the_rule_seam():
    """The real family (M1): olmo_hybrid's gated norm is `o_norm`, not `norm`,
    and its conv is a per-q/k/v `ShortConvolution`, not `causal_conv1d_update`
    -- neither matches this module's surface detection."""
    from transformers import OlmoHybridConfig
    from transformers.models.olmo_hybrid.modeling_olmo_hybrid import OlmoHybridGatedDeltaNet
    cfg = OlmoHybridConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4)
    mixer = OlmoHybridGatedDeltaNet(cfg, layer_idx=0)
    assert _counts(mixer) == {"rule": 1, "conv": 0, "norm": 0}


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


# ── gated-norm kernel ────────────────────────────────────────────────────────

def _ref_norm(x, z, w, eps):
    x, z, w = x.double(), z.double(), w.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w * F.silu(z)


def _stock_norm(n, wdtype, seed=0):
    torch.manual_seed(seed)
    norm = Qwen3NextRMSNormGated(n, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(n))
    return norm.cuda().to(wdtype)


@cuda
@pytest.mark.parametrize("xdtype", [torch.bfloat16, torch.float16], ids=["xbf16", "xfp16"])
@pytest.mark.parametrize("rows", [1, 64])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32], ids=["wbf16", "wfp32"])
@pytest.mark.parametrize("geom", GEOMS)
def test_norm_is_no_further_from_fp64_than_stock(geom, wdtype, rows, xdtype):
    n = GEOMS[geom]["V"]
    norm = _stock_norm(n, wdtype)
    x = torch.randn(rows, n, device="cuda").to(xdtype)
    z = torch.randn(rows, n, device="cuda").to(xdtype)
    so = norm(x, z)
    gdk._patch_norm(norm)
    fo = norm(x, z)
    assert fo.shape == so.shape and fo.dtype == so.dtype
    ref = _ref_norm(x, z, norm.weight, 1e-6)
    assert _rel(fo, ref) <= 1.25 * _rel(so, ref)


@cuda
def test_patched_norm_reads_a_weight_replaced_after_install():
    norm = _stock_norm(128, torch.bfloat16)
    gdk._patch_norm(norm)
    # What set_module_tensor_to_device does to a streamed layer's parameter.
    norm.weight = torch.nn.Parameter(torch.randn(128, device="cuda").bfloat16())
    x = torch.randn(4, 128, device="cuda").bfloat16()
    z = torch.randn(4, 128, device="cuda").bfloat16()
    want = type(norm).forward(norm, x, z)
    assert _rel(norm(x, z), want) < 1e-2


def test_patched_norm_hands_gateless_calls_to_stock():
    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.variance_epsilon = 1e-6

        def forward(self, hidden_states, gate=None):
            return "stock"

    norm = Norm()
    gdk._patch_norm(norm)
    assert norm(torch.zeros(2, 4)) == "stock"
    assert hasattr(norm.forward, gdk._MARK)


# ── installer ────────────────────────────────────────────────────────────────

def _seam_fns(model):
    """Every seam callable. Compared with ==, not `is`: an unpatched norm.forward
    is a fresh bound method on every attribute access."""
    s = gdk.gated_delta_seams(model)
    return ([m.recurrent_gated_delta_rule for m in s["rule"]]
            + [m.causal_conv1d_update for m in s["conv"]]
            + [m.norm.forward for m in s["norm"]])


def test_nothing_installed_on_cpu():
    model = _qwen3_next_model()
    before = _seam_fns(model)
    assert gdk.install_gated_delta_kernels(model, "cpu", torch.float32) == NO_SEAMS
    assert all(a == b for a, b in zip(before, _seam_fns(model)))


def test_nothing_installed_without_triton(monkeypatch):
    monkeypatch.setattr(gdk, "triton", None)
    model = _qwen3_next_model()
    before = _seam_fns(model)
    assert gdk.install_gated_delta_kernels(model, "cuda", torch.bfloat16) == NO_SEAMS
    assert all(a == b for a, b in zip(before, _seam_fns(model)))


@cuda
def test_refuses_a_non_current_cuda_device(capsys):
    """I2: Triton launches on `torch.cuda.current_device()`, not the tensors'
    device. On this single-GPU box index 1 is never current, so this must
    refuse without touching anything -- no cuda:1 tensor is ever allocated."""
    model = _qwen3_next_model().cuda().bfloat16()
    before = _seam_fns(model)
    assert gdk.install_gated_delta_kernels(model, "cuda:1", torch.bfloat16) == NO_SEAMS
    assert all(a == b for a, b in zip(before, _seam_fns(model)))
    out = capsys.readouterr().out
    assert "WARNING" in out and "cuda:1" in out and "current CUDA device" in out


@cuda
def test_install_covers_every_seam_and_is_idempotent():
    model = _qwen3_next_model().cuda().bfloat16()
    n = model.config.layer_types.count("linear_attention")
    want = {"rule": n, "conv": n, "norm": n}
    assert gdk.install_gated_delta_kernels(model, "cuda", torch.bfloat16) == want
    first = _seam_fns(model)
    assert all(hasattr(f, gdk._MARK) for f in first)
    assert gdk.install_gated_delta_kernels(model, "cuda", torch.bfloat16) == want
    assert all(a == b for a, b in zip(first, _seam_fns(model)))   # never double-wrapped


@cuda
def test_self_check_specializes_on_the_mixers_real_rule_geometry(monkeypatch):
    """I1: the self-check must compile and run the geometry decode actually
    uses. Spy on `_check_rule`, delegating to the real one, and assert it ran
    at the mixer's own (HV, K, V) -- not the tiny fallback geometry."""
    seen = []
    real_check_rule = gdk._check_rule

    def spy(stock, device, dtype, geom):
        seen.append(geom)
        return real_check_rule(stock, device, dtype, geom)

    monkeypatch.setattr(gdk, "_check_rule", spy)
    _, mixer = _mixer("qwen3_next")
    mixer = mixer.cuda().bfloat16()
    gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16)
    assert seen == [(mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim)]


@cuda
def test_a_seam_that_fails_its_self_check_stays_stock_and_warns(monkeypatch, capsys):
    def wrong(q, k, v, g, beta, initial_state, output_final_state,
              use_qk_l2norm_in_kernel=False):
        B, T_, HV, K = k.shape
        return (torch.zeros_like(v),
                torch.zeros(B, HV, K, v.shape[-1], device=v.device, dtype=torch.float32))

    monkeypatch.setattr(gdk, "fused_recurrent_rule", wrong)
    _, mixer = _mixer("qwen3_next")
    mixer = mixer.cuda().bfloat16()
    stock = mixer.recurrent_gated_delta_rule
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 0, "conv": 1, "norm": 1}
    assert mixer.recurrent_gated_delta_rule is stock
    out = capsys.readouterr().out
    assert "WARNING" in out and "'rule'" in out and "relL2" in out


@cuda
def test_a_seam_whose_kernel_raises_stays_stock_and_warns(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("triton could not compile for this device")

    monkeypatch.setattr(gdk, "fused_conv_update", broken)
    _, mixer = _mixer("qwen3_next")
    mixer = mixer.cuda().bfloat16()
    stock = mixer.causal_conv1d_update
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 1, "conv": 0, "norm": 1}
    assert mixer.causal_conv1d_update is stock
    out = capsys.readouterr().out
    assert "WARNING" in out and "'conv'" in out and "RuntimeError" in out


@cuda
def test_a_conv_seam_with_a_state_mismatch_stays_stock_and_warns(monkeypatch, capsys):
    """M3(b): the CONV self-check's state-mismatch branch (`math.isinf` path),
    untested until now -- a shape/dtype-correct output that does NOT update
    the conv state."""
    def no_state_update(hidden_states, conv_state, weight, bias=None, activation=None):
        B, D, _ = hidden_states.shape
        return torch.zeros(B, D, 1, device=hidden_states.device, dtype=hidden_states.dtype)

    monkeypatch.setattr(gdk, "fused_conv_update", no_state_update)
    _, mixer = _mixer("qwen3_next")
    mixer = mixer.cuda().bfloat16()
    stock = mixer.causal_conv1d_update
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 1, "conv": 0, "norm": 1}
    assert mixer.causal_conv1d_update is stock
    out = capsys.readouterr().out
    assert "WARNING" in out and "'conv'" in out and "not bit-identical" in out


@cuda
def test_a_norm_seam_with_wrong_output_stays_stock_and_warns(monkeypatch, capsys):
    """M3(a): the NORM self-check's relL2-mismatch branch, untested until now
    -- a shape/dtype-correct but numerically wrong fused kernel."""
    def wrong(weight, eps, hidden_states, gate):
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(gdk, "fused_gated_norm", wrong)
    _, mixer = _mixer("qwen3_next")
    mixer = mixer.cuda().bfloat16()
    stock = mixer.norm.forward
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 1, "conv": 1, "norm": 0}
    assert mixer.norm.forward == stock
    out = capsys.readouterr().out
    assert "WARNING" in out and "'norm'" in out and "relL2" in out


@cuda
def test_a_norm_with_different_math_stays_stock_and_warns(capsys):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    mixer = _RuleAndNormOnly(norm=LlamaRMSNorm(16)).cuda()
    stock = mixer.norm.forward
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 1, "conv": 0, "norm": 0}
    assert not hasattr(mixer.norm.forward, gdk._MARK)
    assert mixer.norm.forward == stock
    out = capsys.readouterr().out
    assert "WARNING" in out and "'norm'" in out


@cuda
def test_install_does_not_read_meta_weights():
    """A streamed layer's weights are still on the meta device at install."""
    from accelerate import init_empty_weights
    cfg, cls = _qwen3_next()
    with init_empty_weights():
        mixer = cls(cfg, layer_idx=0)
    assert mixer.norm.weight.is_meta
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.bfloat16) == \
        {"rule": 1, "conv": 1, "norm": 1}


@cuda
@pytest.mark.parametrize("prefill_len", [2, 12], ids=["shorter-than-conv", "prefill12"])
@pytest.mark.parametrize("kind", ["transformers", "vaayu"])
@pytest.mark.parametrize("family", FAMILIES)
def test_full_mixer_decode_is_no_further_from_fp64_than_stock(family, kind, prefill_len):
    """M2/F3: a single seed's fused/stock error ratio ranges 0.75-1.37 across
    seeds -- one draw can fail by noise alone. Gate on the MEAN over 4 seeds."""
    fused_errs, stock_errs = [], []
    for seed in range(4):
        cfg, cpu_mixer = _mixer(family, seed=seed)
        stock = copy.deepcopy(cpu_mixer).cuda().bfloat16()
        fused = copy.deepcopy(cpu_mixer).cuda().bfloat16()
        ref = copy.deepcopy(cpu_mixer).cuda().double()
        assert gdk.install_gated_delta_kernels(fused, "cuda", torch.bfloat16) == \
            {"rule": 1, "conv": 1, "norm": 1}
        torch.manual_seed(seed)
        x = torch.randn(1, prefill_len + 8, cfg.hidden_size, device="cuda")

        def decode(mixer, dtype, cache):
            with torch.no_grad():
                mixer(x[:, :prefill_len].to(dtype), cache_params=cache)
                return torch.cat([mixer(x[:, t:t + 1].to(dtype), cache_params=cache)
                                  for t in range(prefill_len, prefill_len + 8)], 1)

        so = decode(stock, torch.bfloat16, _cache(kind, cfg))
        fo = decode(fused, torch.bfloat16, _cache(kind, cfg))
        ro = decode(ref, torch.float64, DynamicCache(config=cfg))
        fused_errs.append(_rel(fo, ro))
        stock_errs.append(_rel(so, ro))

    assert sum(fused_errs) / 4 <= 1.25 * sum(stock_errs) / 4


@cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_fused_decode_after_a_chunked_prefill_matches_one_forward(family):
    """Row 43's continuation wrapper and these seams compose: decode reads what a
    chunked prefill left exactly as it reads what one prefill forward left."""
    cfg, mixer = _mixer(family)
    mixer = mixer.cuda()
    install_state_continuation(mixer)
    assert gdk.install_gated_delta_kernels(mixer, "cuda", torch.float32) == \
        {"rule": 1, "conv": 1, "norm": 1}
    x = torch.randn(1, T + 4, cfg.hidden_size, device="cuda")
    with torch.no_grad():
        c_full = _cache("vaayu", cfg)
        mixer(x[:, :T], cache_params=c_full)
        want = torch.cat([mixer(x[:, t:t + 1], cache_params=c_full) for t in range(T, T + 4)], 1)
        c_split = _cache("vaayu", cfg)
        _run(mixer, x[:, :T], c_split, [10, 22, 24])
        got = torch.cat([mixer(x[:, t:t + 1], cache_params=c_split) for t in range(T, T + 4)], 1)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
