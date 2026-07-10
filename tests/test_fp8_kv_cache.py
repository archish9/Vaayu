import torch
from vaayu.vaayu_base import FP8DynamicCache


def _random_kv(batch=1, heads=2, seq=1, dim=4):
    torch.manual_seed(0)
    return torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim)


def _reference_dequant(cache: FP8DynamicCache, layer_idx: int, dtype: torch.dtype):
    """What update() used to compute: full re-dequant of the whole accumulated cache.

    Fix #8: key_cache/_key_scales are now capacity buffers (>= seq_len, with unused
    tail slack from geometric doubling), so slice to the actual used length before
    dequantizing -- matching what update() itself returns.
    """
    n = cache.get_seq_length(layer_idx)
    k_out = cache.key_cache[layer_idx][..., :n, :].to(dtype) * cache._key_scales[layer_idx][..., :n, :]
    v_out = cache.value_cache[layer_idx][..., :n, :].to(dtype) * cache._value_scales[layer_idx][..., :n, :]
    return k_out, v_out


def test_update_matches_full_redequant_first_call():
    cache = FP8DynamicCache()
    k, v = _random_kv(seq=5)  # prefill-style multi-token first call
    k_out, v_out = cache.update(k, v, layer_idx=0)
    ref_k, ref_v = _reference_dequant(cache, 0, k.dtype)
    assert torch.equal(k_out, ref_k)
    assert torch.equal(v_out, ref_v)


def test_update_matches_full_redequant_across_decode_steps():
    cache = FP8DynamicCache()
    k0, v0 = _random_kv(seq=3)
    cache.update(k0, v0, layer_idx=0)

    for _ in range(5):
        k, v = _random_kv(seq=1)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        ref_k, ref_v = _reference_dequant(cache, 0, k.dtype)
        assert torch.equal(k_out, ref_k)
        assert torch.equal(v_out, ref_v)


def test_update_multi_layer_independent():
    cache = FP8DynamicCache()
    k0, v0 = _random_kv(seq=2)
    k1, v1 = _random_kv(seq=4)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)

    ref_k0, ref_v0 = _reference_dequant(cache, 0, k0.dtype)
    ref_k1, ref_v1 = _reference_dequant(cache, 1, k1.dtype)
    assert torch.equal(cache._k_out_cache[0], ref_k0)
    assert torch.equal(cache._v_out_cache[0], ref_v0)
    assert torch.equal(cache._k_out_cache[1], ref_k1)
    assert torch.equal(cache._v_out_cache[1], ref_v1)


def test_get_seq_length_unaffected():
    cache = FP8DynamicCache()
    k, v = _random_kv(seq=3)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 3
    k2, v2 = _random_kv(seq=1)
    cache.update(k2, v2, layer_idx=0)
    assert cache.get_seq_length(0) == 4


def test_trim_kv_cache_keeps_out_cache_in_sync():
    """_trim_kv_cache (keep first N) must trim _k_out_cache/_v_out_cache identically."""
    cache = FP8DynamicCache()
    k, v = _random_kv(seq=6)
    cache.update(k, v, layer_idx=0)

    target = 4
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :target, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :target, :]
        cache._key_scales[i] = cache._key_scales[i][:, :, :target, :]
        cache._value_scales[i] = cache._value_scales[i][:, :, :target, :]
        cache._k_out_cache[i] = cache._k_out_cache[i][:, :, :target, :]
        cache._v_out_cache[i] = cache._v_out_cache[i][:, :, :target, :]

    ref_k, ref_v = _reference_dequant(cache, 0, k.dtype)
    assert cache._k_out_cache[0].shape[-2] == target
    assert torch.equal(cache._k_out_cache[0], ref_k)
    assert torch.equal(cache._v_out_cache[0], ref_v)


def test_trim_kv_cache_front_keeps_out_cache_in_sync():
    """_trim_kv_cache_front (drop oldest, keep last N) must trim _k_out_cache/_v_out_cache identically."""
    cache = FP8DynamicCache()
    k, v = _random_kv(seq=6)
    cache.update(k, v, layer_idx=0)

    target = 4
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, -target:, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, -target:, :]
        cache._key_scales[i] = cache._key_scales[i][:, :, -target:, :]
        cache._value_scales[i] = cache._value_scales[i][:, :, -target:, :]
        cache._k_out_cache[i] = cache._k_out_cache[i][:, :, -target:, :]
        cache._v_out_cache[i] = cache._v_out_cache[i][:, :, -target:, :]

    ref_k, ref_v = _reference_dequant(cache, 0, k.dtype)
    assert cache._k_out_cache[0].shape[-2] == target
    assert torch.equal(cache._k_out_cache[0], ref_k)
    assert torch.equal(cache._v_out_cache[0], ref_v)
