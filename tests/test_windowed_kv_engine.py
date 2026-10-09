"""S35 engine wiring: which layers a load windows, the cache it builds, and the
mask each windowed layer is handed. No checkpoint, no GPU."""
import types
from types import SimpleNamespace

import pytest
import torch

from vaayu.vaayu_base import (FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel,
                              build_eager_causal_mask)
from vaayu.windowed_kv import WindowedKVCache, kv_windows_of


def _layer(window, idx=None):
    return SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window, layer_idx=idx))


def _model(windows_per_layer, *, mode="auto", chunk=8):
    layers = [_layer(w, i) for i, w in enumerate(windows_per_layer)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers),
                           layer_names_dict={"layer_prefix": "layers"},
                           kv_cache_mode=mode, loader_strategy=object(),
                           prefill_chunk_size=chunk)


# ── the window map ───────────────────────────────────────────────────────────

def test_the_window_map_is_read_off_the_attention_modules():
    m = _model([4, None, 4, None, 16])
    assert kv_windows_of(m) == {0: 4, 2: 4, 4: 16}
    assert m._kv_windows == {0: 4, 2: 4, 4: 16}, "cached for the load"


def test_a_model_without_windows_maps_nothing():
    assert kv_windows_of(_model([None, None, None])) == {}


@pytest.mark.parametrize("mode,chunk", [("turboquant", 8), ("turboquant:9", 8), ("bf16", 0)],
                         ids=["turboquant", "turboquant-bits", "chunk-0"])
def test_no_windowing_without_a_chunk_or_under_turboquant(mode, chunk):
    """TurboQuant's own update never windows; an unchunked prefill would put the
    whole prompt in one update, which the plan's constant cannot bound."""
    assert kv_windows_of(_model([4, None], mode=mode, chunk=chunk)) == {}


def test_no_tree_yet_is_not_cached():
    m = SimpleNamespace(kv_cache_mode="auto", loader_strategy=object(), prefill_chunk_size=8)
    assert kv_windows_of(m) == {}
    assert not hasattr(m, "_kv_windows")


# ── the cache factory ────────────────────────────────────────────────────────

def _stub(mode, windows, chunk=8):
    return SimpleNamespace(kv_cache_mode=mode, loader_strategy=object(),
                           _kv_windows=windows, prefill_chunk_size=chunk)


@pytest.mark.parametrize("mode,quantize", [("auto", False), ("bf16", False), ("fp8", True)])
def test_a_windowed_load_gets_the_windowed_cache(mode, quantize):
    c = VaayuBaseModel._create_kv_cache(_stub(mode, {1: 4}))
    assert type(c) is WindowedKVCache
    assert c.windows == {1: 4} and c.chunk == 8 and c.quantize is quantize


@pytest.mark.parametrize("mode,cls", [("auto", FP8DynamicCache), ("fp8", FP8DynamicCache),
                                      ("turboquant:9", TurboQuantKVCache)])
def test_a_load_without_windows_keeps_its_cache_class(mode, cls):
    assert type(VaayuBaseModel._create_kv_cache(_stub(mode, {}))) is cls


# ── the mask hook ────────────────────────────────────────────────────────────

class _Engine:
    layer_attention_window = staticmethod(VaayuBaseModel.layer_attention_window)
    _windowed_mask = VaayuBaseModel._windowed_mask
    get_attention_mask_args = VaayuBaseModel.get_attention_mask_args

    def __init__(self, windows, impl="eager", dtype=torch.float32):
        self._kv_windows = windows
        self._active_attn_impl = impl
        self._windowed_mask_cache = None
        self.running_dtype = dtype
        self.running_device = "cpu"


@pytest.mark.parametrize("len_p,len_s", [(0, 6), (9, 1), (3, 4), (20, 5)])
def test_a_windowed_cache_layer_gets_exactly_its_keys(len_p, len_s):
    W = 4
    eng = _Engine({1: W})
    full = build_eager_causal_mask(len_s, len_p + len_s, torch.float32, "cpu")
    out = eng.get_attention_mask_args(full, len_p, len_s, layer=_layer(W, 1))["attention_mask"]
    keys = min(len_p, W - 1) + len_s
    assert out.shape[-2:] == (len_s, keys)
    blocked = torch.finfo(torch.float32).min
    for r in range(len_s):
        q = len_p + r                                     # absolute query position
        for c in range(keys):
            k = len_p + len_s - keys + c                  # absolute key position
            allowed = k <= q and k > q - W
            assert out[0, 0, r, c] == (0 if allowed else blocked), (r, c)


def test_a_layer_outside_the_map_keeps_the_full_width():
    """TurboQuant, or a load whose map is empty: the cache stores every position."""
    eng = _Engine({})
    full = build_eager_causal_mask(1, 10, torch.float32, "cpu")
    out = eng.get_attention_mask_args(full, 9, 1, layer=_layer(4, 1))["attention_mask"]
    assert out.shape[-1] == 10


def test_sdpa_without_a_mask_gets_an_explicit_windowed_one():
    eng = _Engine({0: 4}, impl="sdpa")
    out = eng.get_attention_mask_args(None, 0, 10, layer=_layer(4, 0))["attention_mask"]
    assert out.dtype is torch.bool and out.shape == (1, 1, 10, 10)
    for q in range(10):
        for k in range(10):
            assert bool(out[0, 0, q, k]) == (k <= q and k > q - 4), (q, k)


@pytest.mark.parametrize("impl,len_s,window", [("flash_attention_2", 10, 4), ("sdpa", 1, 4),
                                               ("sdpa", 10, None), ("eager", 10, 4)])
def test_no_mask_stays_no_mask_where_the_kernel_handles_it(impl, len_s, window):
    eng = _Engine({0: 4} if window else {}, impl=impl)
    assert eng.get_attention_mask_args(None, 0, len_s, layer=_layer(window, 0)) == {}


def test_a_real_gpt_oss_attention_layer_matches_the_full_length_cache():
    """GptOssAttention (eager, sinks, fp32): the windowed cache + its mask slice
    against today's full-length cache + full-width mask, across spills."""
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import (GptOssAttention,
                                                              GptOssRotaryEmbedding)
    W = 4
    cfg = GptOssConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=8, num_hidden_layers=2, sliding_window=W,
                       layer_types=["sliding_attention", "full_attention"],
                       num_local_experts=4, num_experts_per_tok=2, vocab_size=64,
                       intermediate_size=32)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    attn = GptOssAttention(cfg, layer_idx=0).float().eval()
    torch.nn.init.normal_(attn.sinks)
    rot = GptOssRotaryEmbedding(cfg)
    layer = SimpleNamespace(self_attn=attn)
    old, new = _Engine({}), _Engine({0: W})
    ref = FP8DynamicCache(quantize=False)
    win = WindowedKVCache(quantize=False, windows={0: W}, chunk=3)
    past = 0
    for n in [3, 3, 2] + [1] * 20:
        x = torch.randn(1, n, 32)
        pe = rot(x, torch.arange(past, past + n)[None])
        full = build_eager_causal_mask(n, past + n, torch.float32, "cpu")
        with torch.no_grad():
            o_ref, _ = attn(x, pe, old.get_attention_mask_args(full, past, n, layer=layer)
                            ["attention_mask"], past_key_values=ref)
            o_win, _ = attn(x, pe, new.get_attention_mask_args(full, past, n, layer=layer)
                            ["attention_mask"], past_key_values=win)
        torch.testing.assert_close(o_win, o_ref, rtol=0, atol=1e-6)
        past += n
    assert win._ring_start[0] > 0, "the run must cross a spill"
