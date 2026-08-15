"""Cache-format-agnostic KV extraction.

NOT a production path. `KVCacheManager` used to live here -- a save/restore
snapshot manager superseded by PrefixCacheStore and constructed only by its own
test file. It was deleted 2026-08-15 (audit F6): two separate plan tasks had
been spent hardening an unreachable code path, a tax every future KV-format
change would have paid again. Its INT8 helpers went with it, having no other
caller.

`_extract_kv_as_bf16` survives because it is not dead: tests/test_turboquant.py
uses it as an INDEPENDENT ORACLE for the packed-indices corruption case -- a
TurboQuantKVCache is an FP8DynamicCache with quantize=False, so a class-only
check hands back packed indices interpreted as values, with no crash and no
warning. Keep it correct for every cache class; it has no production caller,
so nothing else will notice if it rots.
"""
import torch


def _extract_kv_as_bf16(cache, layer_idx: int):
    """
    Extract key/value tensors from cache as BF16 CPU tensors.
    Handles both FP8DynamicCache (needs dequantization) and standard DynamicCache.
    """
    from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache
    # Checked FIRST: a TurboQuantKVCache is an FP8DynamicCache with
    # quantize=False, so it would otherwise slip through the guard below and
    # return packed INDICES interpreted as values — no crash, no warning.
    if isinstance(cache, TurboQuantKVCache):
        k, v = cache.reconstruct_layer(layer_idx, dtype=torch.bfloat16)
        return k.cpu(), v.cpu()
    k = cache.key_cache[layer_idx]
    v = cache.value_cache[layer_idx]
    # The class alone does not imply quantization: FP8DynamicCache(quantize=False)
    # stores bf16 directly and allocates no scales, so a class-only check would
    # IndexError on the empty scale lists (or scale data that isn't fp8).
    if isinstance(cache, FP8DynamicCache) and getattr(cache, "quantize", True):
        ks = cache._key_scales[layer_idx]
        vs = cache._value_scales[layer_idx]
        k = k.to(torch.bfloat16) * ks
        v = v.to(torch.bfloat16) * vs
    return k.cpu(), v.cpu()
