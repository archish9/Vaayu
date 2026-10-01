"""S14 3c: a model's eager attention cost, measured on its own attention code.

Measured 2026-09-30 (plans/probes/s14c_eager_coeff.py, 8 points per geometry):
the eager transient is exactly c*H*q*k + r*k (+ per-query activations), with
c = 6.0 on gpt-oss-20b and 10.0 on both Llamas, r = 2*H*D*2 (repeat_kv).
"""
import os

import pytest
import torch

from vaayu.eager_attn_cost import (measure_eager_attention_cost,
                                   solve_attention_cost, transient_bytes)

MODELS = "/home/a/code/ai-ml/models"


def _synthetic(n_heads, c, r, s):
    shapes = ((128, 1024), (128, 2048), (256, 2048))
    return [(q, k, c * n_heads * q * k + r * k + s * q) for q, k in shapes]


# Two contrasting geometries (CLAUDE.md §5): gpt-oss-like (H 64, D 64) and
# Llama-3.3-70B-like (H 64, D 128, fp32 softmax).
@pytest.mark.parametrize("n_heads,c,r,s", [
    (64, 6.0, 2 * 64 * 64 * 2, 24_576.0),
    (64, 10.0, 2 * 64 * 128 * 2, 49_152.0),
])
def test_solver_recovers_both_coefficients(n_heads, c, r, s):
    got_c, got_r = solve_attention_cost(_synthetic(n_heads, c, r, s), n_heads)
    assert got_c == pytest.approx(c, rel=1e-9)
    assert got_r == pytest.approx(r, rel=1e-9)


def test_solver_clamps_rounding_noise_on_a_zero_per_key_term():
    """MHA models have no repeat_kv: r is 0 and the allocator's 512-B rounding
    can push the fit a few bytes/key negative. That is noise, not a failure."""
    samples = _synthetic(32, 10.0, -10.0, 8192.0)
    assert solve_attention_cost(samples, 32)[1] == 0.0


@pytest.mark.parametrize("c,r", [(0.0, 16384.0), (100.0, 16384.0), (6.0, -5000.0)])
def test_solver_refuses_an_implausible_fit(c, r):
    with pytest.raises(ValueError, match="implausible"):
        solve_attention_cost(_synthetic(64, c, r, 0.0), 64)


def test_solver_needs_exactly_three_samples():
    with pytest.raises(ValueError, match="3 samples"):
        solve_attention_cost(_synthetic(64, 6.0, 0.0, 0.0)[:2], 64)


def test_no_attention_modules_is_an_error():
    with pytest.raises(ValueError, match="self_attn"):
        measure_eager_attention_cost([], rope_fn=None, make_cache=None, mask_fn=None,
                                     device=torch.device("cpu"), dtype=torch.bfloat16)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_transient_bytes_sees_a_run_below_an_old_high_water_mark():
    """The engine may not reset the process peak counter, so a run smaller than
    an earlier peak would be invisible -- unless the ballast lifts it."""
    dev = torch.device("cuda")
    old = torch.empty(512 * 2 ** 20, dtype=torch.uint8, device=dev)
    del old                                    # the process peak is now >= 512 MiB up
    peak_before = torch.cuda.max_memory_allocated(dev)
    n = 64 * 2 ** 20
    got = transient_bytes(lambda: torch.empty(n, dtype=torch.uint8, device=dev), dev)
    assert abs(got - n) <= 2 * 2 ** 20
    assert torch.cuda.max_memory_allocated(dev) >= peak_before   # never reset


def _attention_case(name):
    from transformers import AutoConfig
    if name == "gpt-oss":
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssAttention as Attn, GptOssRotaryEmbedding as Rope)
        path, layer_idx = f"{MODELS}/gpt-oss-20b", 0
    else:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention as Attn, LlamaRotaryEmbedding as Rope)
        path, layer_idx = f"{MODELS}/Llama-3.2-1B-Instruct", 0
    if not os.path.isdir(path):
        pytest.skip(f"config not present at {path}")
    cfg = AutoConfig.from_pretrained(path)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        template = Attn(cfg, layer_idx)
    return cfg, template, Rope


@cuda
@pytest.mark.parametrize("name,expect_c", [("gpt-oss", 6.0), ("llama-1b", 10.0)])
def test_measures_the_models_own_eager_cost(name, expect_c):
    from vaayu.vaayu_base import FP8DynamicCache, build_eager_causal_mask
    cfg, template, Rope = _attention_case(name)
    dev = torch.device("cuda")
    rope = Rope(cfg, device=dev)

    def rope_fn(pos):
        return rope(torch.empty(1, dtype=torch.bfloat16, device=dev), pos)

    c, r, names = measure_eager_attention_cost(
        [template], rope_fn=rope_fn, make_cache=lambda: FP8DynamicCache(quantize=False),
        mask_fn=build_eager_causal_mask, device=dev, dtype=torch.bfloat16)
    H = cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    # + 2/H: the bf16 mask block is built inside the measured call, as the
    # engine builds it inside every forward.
    assert c == pytest.approx(expect_c + 2 / H, abs=0.2)
    assert r == pytest.approx(2 * H * D * 2, rel=0.05)
    assert names == type(template).__name__


# --- the engine hook (S14 3c) ---------------------------------------------
from types import SimpleNamespace

import vaayu.eager_attn_cost as eac
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import score_elem_bytes


class _Stub:
    attention_score_cost = VaayuBaseModel.attention_score_cost
    _attention_modules_by_class = VaayuBaseModel._attention_modules_by_class

    def __init__(self, impl, layers=()):
        self._active_attn_impl = impl
        self.model = SimpleNamespace(model=SimpleNamespace(layers=list(layers)))
        self.layer_names_dict = {"layer_prefix": "model.layers"}
        self.running_device = "cpu"
        self.running_dtype = torch.bfloat16

    def _precompute_rope(self, position_ids):
        return None


@pytest.mark.parametrize("impl", ["flash_attention_2", "sdpa"])
def test_flash_and_sdpa_keep_the_kernel_table(impl, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("flash/sdpa must not be measured")
    monkeypatch.setattr(eac, "measure_eager_attention_cost", boom)
    assert _Stub(impl).attention_score_cost() == (score_elem_bytes(impl), 0.0, "kernel table")


def test_eager_uses_the_measurement(monkeypatch):
    monkeypatch.setattr(eac, "measure_eager_attention_cost",
                        lambda *a, **kw: (6.0, 16384.0, "GptOssAttention"))
    assert _Stub("eager").attention_score_cost() == (
        6.0, 16384.0, "measured on GptOssAttention")


def test_a_failed_measurement_falls_back_to_the_table_with_the_reason(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(eac, "measure_eager_attention_cost", fail)
    per_elem, per_key, source = _Stub("eager").attention_score_cost()
    assert (per_elem, per_key) == (score_elem_bytes("eager"), 0.0)
    assert "measurement failed" in source and "boom" in source


def test_a_failed_measurement_reason_is_one_short_line(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("line one\nline two " + "x" * 500)
    monkeypatch.setattr(eac, "measure_eager_attention_cost", fail)
    _, _, source = _Stub("eager").attention_score_cost()
    assert "line one" in source and "line two" not in source
    assert "\n" not in source and len(source) < 300


class _Unrebuildable(torch.nn.Module):
    """An attention class that cannot be rebuilt from (config, layer_idx)."""
    def __init__(self, *args, ok=False):
        super().__init__()
        if not ok:
            raise TypeError("custom constructor signature")
        self.config = SimpleNamespace(num_attention_heads=8)
        self.layer_idx = 0


def test_unrebuildable_attention_falls_back_with_the_reason():
    layer = SimpleNamespace(self_attn=_Unrebuildable(ok=True))
    per_elem, per_key, source = _Stub("eager", [layer]).attention_score_cost()
    assert (per_elem, per_key) == (score_elem_bytes("eager"), 0.0)
    assert "TypeError" in source and "custom constructor signature" in source


class _AttnA:
    pass


class _AttnB:
    pass


def test_attention_modules_by_class_takes_the_first_of_each_class_in_order():
    a0, a1, b2 = _AttnA(), _AttnA(), _AttnB()
    layers = [SimpleNamespace(self_attn=a0), SimpleNamespace(self_attn=a1),
              SimpleNamespace(self_attn=b2), SimpleNamespace()]   # last: no attention
    assert _Stub("eager", layers)._attention_modules_by_class() == [a0, b2]


def _ds_stub(**cfg):
    return SimpleNamespace(_active_attn_impl="eager", running_dtype=torch.bfloat16,
                           config=SimpleNamespace(**cfg))


@pytest.mark.parametrize("cfg,expect", [
    # DeepSeek-V4-Flash: 2 x 64 x 2 B / (64 x 4) = 1.0 B per score element
    (dict(num_hidden_layers=5, compress_ratios=[0, 4, 128, 4, 0, 0], index_n_heads=64,
          num_attention_heads=64), 1.0),
    # a contrasting geometry: fewer indexer heads, coarser compression
    (dict(num_hidden_layers=3, compress_ratios=[8, 16, 8], index_n_heads=32,
          num_attention_heads=128), 2 * 32 * 2 / (128 * 8)),
])
def test_deepseek_attention_cost_is_derived_from_its_config(cfg, expect):
    from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4
    per_elem, per_key, source = VaayuDeepseekV4.attention_score_cost(_ds_stub(**cfg))
    assert per_elem == pytest.approx(expect)
    assert per_key == 0.0
    assert source.startswith("derived from config")


def test_deepseek_attention_cost_falls_back_without_indexer_fields():
    from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4
    per_elem, per_key, source = VaayuDeepseekV4.attention_score_cost(
        _ds_stub(num_hidden_layers=2, compress_ratios=[0, 0], num_attention_heads=64))
    assert (per_elem, per_key) == (score_elem_bytes("eager"), 0.0)
    assert source.startswith("kernel table") and "DSA" in source
