"""Fix #8: FP8 KV ring buffer must match cat-based accumulation, without O(n^2)."""
import torch
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="needs FP8 support"
)


def _ref_quant(k):
    scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    fp8 = (k / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.to(k.dtype) * scale


def test_ring_matches_reference_accumulation():
    from vaayu.vaayu_base import FP8DynamicCache
    cache = FP8DynamicCache()
    B, H, D = 1, 4, 8
    torch.manual_seed(0)
    ref_k = []
    # 20 single-token updates crossing the initial capacity (forces a doubling)
    for step in range(20):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        ref_k.append(_ref_quant(k))
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == step + 1
        assert k_out.shape == (B, H, step + 1, D)
    expected = torch.cat(ref_k, dim=-2)
    # Final returned k_out must equal the reference dequantized accumulation.
    torch.testing.assert_close(k_out, expected, rtol=0, atol=0)


def test_multi_token_prefill_then_decode():
    from vaayu.vaayu_base import FP8DynamicCache
    cache = FP8DynamicCache()
    B, H, D = 1, 4, 8
    k0 = torch.randn(B, H, 5, D); v0 = torch.randn(B, H, 5, D)  # prefill 5 tokens
    cache.update(k0, v0, layer_idx=0)
    assert cache.get_seq_length(0) == 5
    k1 = torch.randn(B, H, 1, D); v1 = torch.randn(B, H, 1, D)  # decode 1
    k_out, _ = cache.update(k1, v1, layer_idx=0)
    assert cache.get_seq_length(0) == 6
    assert k_out.shape == (B, H, 6, D)
