"""Row 30 / S2, RC4: KV caching on a HYBRID-attention model.

`qwen3_next` (the S2 target) interleaves gated-delta-net layers with real
attention — 36 `linear_attention` + 12 `full_attention` at 48 layers. That
breaks two assumptions Vaayu has held everywhere:

1. **Every layer needs a KV layer object.** It does not: linear-attention layers
   need a `LinearAttentionLayer` holding `conv_states`/`recurrent_states`, and
   transformers only builds those when the cache is given the `config`. Without
   it, `Qwen3NextGatedDeltaNet` hits
   `AttributeError: 'DynamicLayer' object has no attribute 'conv_states'`.
2. **Layers call `update()` in order from 0.** They do not: on this model the
   first layer to store KV is layer **3**, and Vaayu's own ring buffer
   (`FP8DynamicCache`, what `auto` resolves to) appends its per-layer buffers
   positionally, so it raised `IndexError: list index out of range`.

Every model on device today (Qwen3-235B, MiniMax-M2.7, both Llamas) is uniform
attention with no sliding window, so the uniform cases below are the regression
guard: they must keep producing exactly what they produce now.
"""
import types

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel

B, H, D = 1, 2, 8


def _hybrid_config(num_layers=8, interval=4):
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    return Qwen3NextConfig(num_hidden_layers=num_layers, full_attention_interval=interval)


def _uniform_config(num_layers=4):
    from transformers.models.llama.configuration_llama import LlamaConfig
    return LlamaConfig(num_hidden_layers=num_layers)


def _model(mode, config):
    """Stand-in carrying only what _create_kv_cache reads."""
    return types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=object(), config=config)


def _kv(n=1):
    return torch.randn(B, H, n, D), torch.randn(B, H, n, D)


# ── the ring buffer with non-contiguous layer indices ────────────────────────

@pytest.mark.parametrize("quantize", [False, True])
def test_ring_buffer_addresses_non_contiguous_layers(quantize):
    if quantize and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs FP8 support")
    cache = FP8DynamicCache(quantize=quantize)
    kv_layers = [3, 7, 11]  # what a full_attention_interval=4 model actually updates

    for idx in kv_layers:
        k, v = _kv()
        k_out, v_out = cache.update(k, v, layer_idx=idx)
        assert k_out.shape == (B, H, 1, D)
        assert cache.get_seq_length(idx) == 1

    # A second token on the SAME layers appends rather than restarting.
    for idx in kv_layers:
        cache.update(*_kv(), layer_idx=idx)
        assert cache.get_seq_length(idx) == 2

    # Layers that never stored anything report zero, not garbage.
    for idx in (0, 1, 2, 4):
        assert cache.get_seq_length(idx) == 0


def test_ring_buffer_keeps_each_layer_separate_when_indices_skip():
    """Values must land under their own layer index — a positional append would
    silently put layer 7's tokens where layer 3's belong."""
    cache = FP8DynamicCache(quantize=False)
    k3, v3 = _kv()
    k7, v7 = _kv()
    cache.update(k3, v3, layer_idx=3)
    cache.update(k7, v7, layer_idx=7)
    got3, _ = cache.update(torch.zeros(B, H, 0, D), torch.zeros(B, H, 0, D), layer_idx=3)
    got7, _ = cache.update(torch.zeros(B, H, 0, D), torch.zeros(B, H, 0, D), layer_idx=7)
    assert torch.equal(got3, k3)
    assert torch.equal(got7, k7)


def test_ring_buffer_crop_survives_the_gaps():
    cache = FP8DynamicCache(quantize=False)
    for idx in (3, 7):
        for _ in range(5):
            cache.update(*_kv(), layer_idx=idx)
    cache.crop(3)
    assert cache.get_seq_length(3) == 3 and cache.get_seq_length(7) == 3
    cache.crop_front(2)
    assert cache.get_seq_length(3) == 2 and cache.get_seq_length(7) == 2


def test_ring_buffer_contiguous_layers_unchanged():
    """Regression guard: the uniform-attention path every on-device model takes."""
    cache = FP8DynamicCache(quantize=False)
    for idx in range(4):
        for step in range(3):
            cache.update(*_kv(), layer_idx=idx)
            assert cache.get_seq_length(idx) == step + 1
    assert len(cache.key_cache) == 4


# ── the config reaches the cache ─────────────────────────────────────────────

@pytest.mark.parametrize("mode,cls", [
    ("fp16", DynamicCache),
    ("bf16", FP8DynamicCache),
    ("turboquant", TurboQuantKVCache),
])
def test_created_cache_has_hybrid_layer_types(mode, cls):
    cfg = _hybrid_config(num_layers=8, interval=4)
    cache = VaayuBaseModel._create_kv_cache(_model(mode, cfg))
    assert isinstance(cache, cls)
    names = [type(l).__name__ for l in cache.layers]
    assert names.count("LinearAttentionLayer") == 6, names
    assert names.count("DynamicLayer") == 2, names
    # The linear-attention layers carry the state the gated delta net reads.
    assert hasattr(cache.layers[0], "conv_states")
    assert not hasattr(cache.layers[3], "conv_states")


@pytest.mark.parametrize("mode", ["fp16", "bf16", "turboquant"])
def test_uniform_model_cache_is_unchanged(mode):
    """Regression guard for the four checkpoints on disk: none declares
    `layer_types` and none has a sliding window, so nothing about their cache
    may change. Handing transformers a config MATERIALIZES `cache.layers`,
    flipping `len(cache)` from 0 to num_hidden_layers and `is_initialized` to
    True before a token is stored — observable, so a uniform model must not get
    one."""
    cache = VaayuBaseModel._create_kv_cache(_model(mode, _uniform_config(4)))
    assert len(cache) == 0, "uniform model gained eagerly-built cache layers"
    assert cache.layers == []
    assert not cache.is_initialized


def test_non_uniform_detection():
    from vaayu.vaayu_base import cache_config_if_non_uniform
    assert cache_config_if_non_uniform(None) is None
    assert cache_config_if_non_uniform(_uniform_config(4)) is None
    hyb = _hybrid_config(8, 4)
    assert cache_config_if_non_uniform(hyb) is hyb
    sliding = _uniform_config(4)
    sliding.sliding_window = 512
    assert cache_config_if_non_uniform(sliding) is sliding


def test_missing_config_still_builds_a_cache():
    """Engines and tests construct partial stand-ins; a cache must still come
    back rather than raising on a model object with no `config`."""
    stub = types.SimpleNamespace(kv_cache_mode="bf16", loader_strategy=object())
    assert isinstance(VaayuBaseModel._create_kv_cache(stub), FP8DynamicCache)
