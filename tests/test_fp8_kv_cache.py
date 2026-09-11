import types

import torch
from vaayu.vaayu_base import FP8DynamicCache


def _random_kv(batch=1, heads=2, seq=1, dim=4):
    torch.manual_seed(0)
    return torch.randn(batch, heads, seq, dim), torch.randn(batch, heads, seq, dim)


def _bf16_kv(batch=1, heads=4, seq=8, dim=16):
    torch.manual_seed(0)
    return (torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16),
            torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16))


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


# --- unquantized (bf16-only ring buffer) mode ---


def test_unquantized_mode_allocates_no_fp8_or_scale_buffers():
    cache = FP8DynamicCache(quantize=False)
    k, v = _bf16_kv()
    cache.update(k, v, layer_idx=0)
    assert cache.quantize is False
    assert cache.key_cache[0].dtype == k.dtype      # stored in the source dtype
    assert cache._key_scales == []                  # no scales allocated
    assert cache._value_scales == []
    assert cache._k_out_cache == []                 # no dequantized mirror
    assert cache._v_out_cache == []


def test_unquantized_mode_is_lossless():
    """fp8 is lossy; bf16 storage is not. Round-trip must be exact."""
    cache = FP8DynamicCache(quantize=False)
    k, v = _bf16_kv()
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_unquantized_mode_is_two_bytes_per_element():
    cache = FP8DynamicCache(quantize=False)
    k, v = _bf16_kv(seq=8)
    cache.update(k, v, layer_idx=0)
    assert cache.resident_bytes() == 2 * k.numel() * 2   # K and V, 2 bytes each


def test_quantized_mode_is_unchanged():
    """Regression guard: the fp8 path must keep all six buffers."""
    cache = FP8DynamicCache()                       # default quantize=True
    k, v = _bf16_kv()
    cache.update(k, v, layer_idx=0)
    assert cache.quantize is True
    assert cache.key_cache[0].dtype == torch.float8_e4m3fn
    assert len(cache._key_scales) == 1 and len(cache._k_out_cache) == 1


def test_unquantized_mode_appends_across_steps():
    """The ring buffer must still work -- that is the whole reason this class exists."""
    cache = FP8DynamicCache(quantize=False)
    k1, v1 = _bf16_kv(seq=3)
    k2, v2 = _bf16_kv(seq=1)
    cache.update(k1, v1, layer_idx=0)
    k_out, _ = cache.update(k2, v2, layer_idx=0)
    assert cache.get_seq_length(0) == 4
    assert k_out.shape[-2] == 4
    assert torch.equal(k_out[..., :3, :], k1)
    assert torch.equal(k_out[..., 3:, :], k2)


def test_unquantized_mode_crop_and_to_cpu():
    cache = FP8DynamicCache(quantize=False)
    k, v = _bf16_kv(seq=6)
    cache.update(k, v, layer_idx=0)
    cache.crop(4)
    assert cache.get_seq_length(0) == 4
    cache.to_cpu()
    assert cache.key_cache[0].device.type == "cpu"
    assert cache.key_cache[0].shape[-2] == 4        # compacted to used length


def _cache_for_mode(mode):
    """_create_kv_cache reads only kv_cache_mode + loader_strategy off self."""
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=None)
    return VaayuBaseModel._create_kv_cache(fake)


def test_create_kv_cache_bf16_mode():
    cache = _cache_for_mode("bf16")
    assert isinstance(cache, FP8DynamicCache)
    assert cache.quantize is False


def test_create_kv_cache_fp8_and_fp16_modes_unchanged():
    from transformers import DynamicCache
    fp8 = _cache_for_mode("fp8")
    assert isinstance(fp8, FP8DynamicCache) and fp8.quantize is True
    fp16 = _cache_for_mode("fp16")
    assert type(fp16) is DynamicCache
    # non-EXL3 loader_strategy -> auto resolves to bf16 too (row 3, 2026-09-11)
    auto = _cache_for_mode("auto")
    assert type(auto) is FP8DynamicCache and auto.quantize is False


def test_auto_selects_bf16_for_exl3():
    """auto used to resolve to fp8 for EXL3 models — 3.016 bytes/element, lossy,
    and slower than storing bf16 outright (Plan E RESULTS)."""
    from vaayu.persist.loader_strategy import EXL3LoaderStrategy
    from vaayu.vaayu_base import VaayuBaseModel

    fake = types.SimpleNamespace(kv_cache_mode="auto",
                                 loader_strategy=EXL3LoaderStrategy())
    cache = VaayuBaseModel._create_kv_cache(fake)
    assert isinstance(cache, FP8DynamicCache)
    assert cache.quantize is False


def test_unquantized_mode_crop_front():
    cache = FP8DynamicCache(quantize=False)
    k, v = _bf16_kv(seq=6)
    cache.update(k, v, layer_idx=0)
    cache.crop_front(2)
    k_out, _ = cache.update(*_bf16_kv(seq=1), layer_idx=0)
    assert cache.get_seq_length(0) == 3
    assert torch.equal(k_out[..., :2, :], k[..., 4:, :])
