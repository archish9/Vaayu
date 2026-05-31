import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch


# Skip entire file if FP8 not available (PyTorch < 2.1)
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn not available"
)


@pytest.fixture
def cache():
    from vaayu.vaayu_base import FP8DynamicCache
    return FP8DynamicCache()


def _kv(batch=1, heads=4, tokens=3, head_dim=8, val=1.0):
    """Return (key, value) BF16 tensors filled with val."""
    shape = (batch, heads, tokens, head_dim)
    return torch.full(shape, val, dtype=torch.bfloat16), \
           torch.full(shape, val * 0.5, dtype=torch.bfloat16)


def test_update_stores_fp8(cache):
    k, v = _kv(tokens=1)
    cache.update(k, v, layer_idx=0)
    assert cache.key_cache[0].dtype == torch.float8_e4m3fn
    assert cache.value_cache[0].dtype == torch.float8_e4m3fn


def test_update_scales_shape(cache):
    batch, heads, tokens, head_dim = 1, 4, 1, 8
    k, v = _kv(batch, heads, tokens, head_dim)
    cache.update(k, v, layer_idx=0)
    # scale shape: [batch, heads, tokens, 1]
    assert cache._key_scales[0].shape == (batch, heads, tokens, 1)
    assert cache._value_scales[0].shape == (batch, heads, tokens, 1)


def test_update_returns_correct_shape(cache):
    batch, heads, tokens, head_dim = 1, 4, 1, 8
    k, v = _kv(batch, heads, tokens, head_dim)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (batch, heads, tokens, head_dim)
    assert v_out.shape == (batch, heads, tokens, head_dim)


def test_update_returns_bfloat16(cache):
    k, v = _kv(tokens=1)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.bfloat16


def test_dequant_approximate(cache):
    """Dequantized output should be close to original (FP8 has ~3 decimal digits)."""
    k, v = _kv(tokens=1, val=2.0)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert torch.allclose(k_out.float(), k.float(), atol=0.02)
    assert torch.allclose(v_out.float(), v.float(), atol=0.02)


def test_accumulation_across_decode_steps(cache):
    """Three sequential decode steps → seq_len grows to 3."""
    for step in range(3):
        k, v = _kv(tokens=1, val=float(step + 1))
        cache.update(k, v, layer_idx=0)
    assert cache.key_cache[0].shape[-2] == 3
    assert cache.value_cache[0].shape[-2] == 3
    assert cache._key_scales[0].shape[-2] == 3


def test_get_seq_length(cache):
    k, v = _kv(tokens=1)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length() == 2


def test_multi_layer(cache):
    """Different layer_idx entries are independent."""
    k0, v0 = _kv(tokens=1, val=1.0)
    k1, v1 = _kv(tokens=1, val=2.0)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)
    assert len(cache.key_cache) == 2
    assert cache.key_cache[0].shape[-2] == 1
    assert cache.key_cache[1].shape[-2] == 1


def test_prefill_then_decode(cache):
    """Prefill (4 tokens) + 2 decode steps → seq_len 6."""
    k_pre, v_pre = _kv(tokens=4, val=1.0)
    cache.update(k_pre, v_pre, layer_idx=0)
    for _ in range(2):
        k, v = _kv(tokens=1, val=2.0)
        cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length() == 6


def test_returned_shape_grows_with_seq(cache):
    """After decode steps, returned K/V shape includes ALL past tokens."""
    k, v = _kv(tokens=4, val=1.0)
    cache.update(k, v, layer_idx=0)  # prefill
    k2, v2 = _kv(tokens=1, val=2.0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)  # decode
    # Output must include all 5 tokens for the attention layer
    assert k_out.shape[-2] == 5
    assert v_out.shape[-2] == 5


def test_zero_activations_no_division_by_zero(cache):
    """All-zero K/V should not cause NaN (scale clamped to 1e-12)."""
    k = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
    v = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_float16_input_returns_float16(cache):
    """Output dtype matches input dtype — float16 in → float16 out."""
    k = torch.full((1, 4, 1, 8), 1.0, dtype=torch.float16)
    v = torch.full((1, 4, 1, 8), 0.5, dtype=torch.float16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.dtype == torch.float16
    assert v_out.dtype == torch.float16


def test_out_of_order_layer_raises():
    """Layer indices must be filled sequentially (0, 1, 2, ...). Skipping raises."""
    from vaayu.vaayu_base import FP8DynamicCache
    c = FP8DynamicCache()
    k, v = _kv(tokens=1)
    # Fill layer 0 first — OK
    c.update(k, v, layer_idx=0)
    # Attempt layer 2 without layer 1 — should raise IndexError (list, not dict)
    with pytest.raises((IndexError, AssertionError)):
        c.update(k, v, layer_idx=2)
