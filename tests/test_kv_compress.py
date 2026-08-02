"""CompressedKV: TurboQuant packing of a bf16 ring buffer for RAM-pool storage."""
import pytest
import torch

from vaayu.kv_compress import CompressedKV, compress_cache, decompress_cache
from vaayu.vaayu_base import FP8DynamicCache

DIM = 128


def _cache(layers=3, heads=4, seq=32, dim=DIM, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = FP8DynamicCache(quantize=False)
    for L in range(layers):
        k = torch.randn(1, heads, seq, dim, generator=g).to(dtype)
        v = torch.randn(1, heads, seq, dim, generator=g).to(dtype)
        c.update(k, v, L)
    return c


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_roundtrip_preserves_shape_dtype_and_lengths():
    c = _cache()
    out = decompress_cache(compress_cache(c), device="cpu")
    assert out.get_seq_length(0) == c.get_seq_length(0)
    assert len(out.key_cache) == len(c.key_cache)
    for L in range(len(c.key_cache)):
        u = c._seq_len[L]
        assert out.key_cache[L][..., :u, :].shape == c.key_cache[L][..., :u, :].shape
        assert out.key_cache[L].dtype == c.key_cache[L].dtype


def test_roundtrip_is_accurate_at_8_bits():
    """8 bits/channel is this feature's default because Plan H's on-device gate
    failed at 4 and first passed at 8 (RESULTS 2026-08-02, section 1)."""
    c = _cache(seq=64)
    out = decompress_cache(compress_cache(c, bits_total=8), device="cpu")
    for L in range(len(c.key_cache)):
        u = c._seq_len[L]
        assert _rel(out.key_cache[L][..., :u, :], c.key_cache[L][..., :u, :]) < 0.05
        assert _rel(out.value_cache[L][..., :u, :], c.value_cache[L][..., :u, :]) < 0.05


def test_four_bits_is_much_worse_and_much_smaller():
    c = _cache(seq=64)
    p8, p4 = compress_cache(c, bits_total=8), compress_cache(c, bits_total=4)
    assert p4.nbytes < p8.nbytes
    e8 = _rel(decompress_cache(p8, "cpu").key_cache[0], c.key_cache[0])
    e4 = _rel(decompress_cache(p4, "cpu").key_cache[0], c.key_cache[0])
    assert e4 > e8


def test_nbytes_beats_the_bf16_cache_and_matches_the_rate():
    """8 bits: 7 index bits + 1 sign bit + two fp16 norms per (token, head)
    = (7 + 1)/8 + 4/128 = 1.031 bytes/element, vs bf16's 2.000."""
    c = _cache(layers=2, heads=4, seq=64)
    p = compress_cache(c, bits_total=8)
    elems = sum(c._seq_len) * 2 * 4 * DIM
    assert p.nbytes / elems == pytest.approx(1.031, abs=0.002)
    assert p.nbytes < c.resident_bytes()


def test_compressed_container_is_not_a_cache():
    """Structural guard: CompressedKV must never be usable where a Cache is
    expected, so no code path can read packed indices as values."""
    from transformers import DynamicCache
    p = compress_cache(_cache())
    assert not isinstance(p, DynamicCache)
    assert not hasattr(p, "update")
    assert not hasattr(p, "key_cache")


def test_decompressed_cache_accepts_further_appends():
    """A restored cache must behave like a live one: the next decode step
    appends to it."""
    c = _cache(seq=16)
    out = decompress_cache(compress_cache(c), device="cpu")
    k = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16)
    ko, _ = out.update(k, k, 0)
    assert ko.shape[-2] == 17
    assert out.get_seq_length(0) == 17


def test_uneven_layer_lengths_are_preserved():
    c = FP8DynamicCache(quantize=False)
    for L, seq in enumerate((5, 9, 3)):
        k = torch.randn(1, 4, seq, DIM, dtype=torch.bfloat16)
        c.update(k, k, L)
    out = decompress_cache(compress_cache(c), device="cpu")
    assert [out.get_seq_length(L) for L in range(3)] == [5, 9, 3]


def test_refuses_a_quantized_fp8_cache():
    """fp8 caches store scales, not raw values -- compressing one would read
    fp8 bytes as if they were values."""
    c = FP8DynamicCache()             # quantize=True
    k = torch.randn(1, 4, 8, DIM, dtype=torch.float16)
    c.update(k, k, 0)
    with pytest.raises(ValueError, match="bf16"):
        compress_cache(c)


def test_compression_is_deterministic():
    c = _cache()
    a, b = compress_cache(c), compress_cache(c)
    for x, y in zip(a.k_idx, b.k_idx):
        assert torch.equal(x, y)
