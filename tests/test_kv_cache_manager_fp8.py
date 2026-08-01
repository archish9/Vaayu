"""KV snapshot restore must preserve the model's cache class (FP8 stays FP8)."""
import pytest
import torch

from vaayu.kv_cache_manager import KVCacheManager
from vaayu.vaayu_base import FP8DynamicCache


class _FakeModel:
    def _create_kv_cache(self):
        return FP8DynamicCache()


def _filled_fp8_cache(n_layers=2, seq=5):
    c = FP8DynamicCache()
    for li in range(n_layers):
        k = torch.randn(1, 2, seq, 8, dtype=torch.bfloat16)
        v = torch.randn(1, 2, seq, 8, dtype=torch.bfloat16)
        c.update(k, v, li)
    return c


def test_load_snapshot_restores_fp8_cache_class():
    m = KVCacheManager(_FakeModel(), compress_kv=False)
    m.save_snapshot(_filled_fp8_cache())
    restored = m.load_snapshot(device="cpu")
    assert isinstance(restored, FP8DynamicCache)
    assert restored.get_seq_length() == 5


def test_roundtrip_values_close():
    src = _filled_fp8_cache()
    m = KVCacheManager(_FakeModel(), compress_kv=False)
    m.save_snapshot(src)
    restored = m.load_snapshot(device="cpu")
    k_src, _ = src.update(torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16),
                          torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16), 0)
    k_res, _ = restored.update(torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16),
                               torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16), 0)
    # FP8 roundtrips twice (save dequant -> restore requant); tolerance is loose
    assert torch.allclose(k_src.float(), k_res.float(), atol=0.15, rtol=0.15)


def test_extract_handles_unquantized_cache():
    """An unquantized FP8DynamicCache has no scales — extraction must not
    assume they exist just because the class matches."""
    from vaayu.kv_cache_manager import _extract_kv_as_bf16

    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    c.update(k, v, 0)
    k_out, v_out = _extract_kv_as_bf16(c, 0)
    assert k_out.dtype == torch.bfloat16
    assert torch.equal(k_out, k.cpu())      # lossless, no scale multiply
    assert torch.equal(v_out, v.cpu())


def test_fallback_without_model_factory():
    class _NoFactory: pass
    m = KVCacheManager(_NoFactory(), compress_kv=False)
    m.save_snapshot(_filled_fp8_cache())
    restored = m.load_snapshot(device="cpu")
    from transformers import DynamicCache
    assert isinstance(restored, DynamicCache)
