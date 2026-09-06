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


def _rate(dim, bits, m, heads, seq):
    """Bytes per stored element: packed residual + exact outlier channels.

    Per (token, head): (dim-m) codes at bits-1 bits, (dim-m) QJL sign bits, two
    fp16 norms, and m channels kept in the cache dtype. Once per LAYER, shared
    by every head and not paid per token: the m int16 channel indices.
    """
    r = dim - m
    per_token = r * (bits - 1) / 8 + r / 8 + 4 + 2 * m
    return (per_token * seq * heads + 2 * m) / (heads * seq * dim)


def test_nbytes_beats_the_bf16_cache_and_matches_the_rate():
    c = _cache(layers=2, heads=4, seq=64)
    p = compress_cache(c, bits_total=8)
    elems = sum(c._seq_len) * 2 * 4 * DIM
    rate = _rate(DIM, 8, p.outlier_channels, 4, 64)
    assert p.nbytes / elems == pytest.approx(rate, abs=0.002)
    assert p.nbytes < c.resident_bytes()


def test_uniform_rate_without_the_split_is_unchanged():
    """outlier_channels=0 is exactly the pre-split format: 7 index bits + 1 sign
    bit + two fp16 norms per (token, head) = (7 + 1)/8 + 4/128 = 1.031."""
    c = _cache(layers=2, heads=4, seq=64)
    p = compress_cache(c, bits_total=8, outlier_channels=0)
    elems = sum(c._seq_len) * 2 * 4 * DIM
    assert p.nbytes / elems == pytest.approx(1.031, abs=0.002)
    assert all(t is None for t in p.k_chan + p.v_chan + p.k_exact + p.v_exact)


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


# --- model flag (Task 4) ----------------------------------------------------

def test_prefix_compress_flag_reaches_the_store():
    import types
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(enable_prefix_cache=True, prefix_compress=8)
    store = VaayuBaseModel._build_prefix_store(fake, budget_bytes=10 ** 9)
    assert store.compress_bits == 8


def test_prefix_compress_defaults_to_off():
    import types
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(enable_prefix_cache=True, prefix_compress=None)
    assert VaayuBaseModel._build_prefix_store(fake, budget_bytes=10 ** 9).compress_bits is None


def test_below_nine_bits_warns(capsys):
    """8 bits measured 0.0612 worst-layer attention-score error on MiniMax
    against a 0.05 bar (Plan I gate); a user asking for it gets told, not
    silently degraded."""
    import types
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(enable_prefix_cache=True, prefix_compress=8)
    VaayuBaseModel._build_prefix_store(fake, budget_bytes=10 ** 9)
    assert "WARNING" in capsys.readouterr().out


def test_default_bits_is_nine_and_is_the_quantizer_ceiling():
    """9 is the shipped width and the highest one that exists: make_codebook
    raises above b_mse=8. 8 fails the gate on both models against real decode
    queries, so there is no cheaper width and no more expensive one either."""
    import pytest as _pytest
    from vaayu.kv_compress import _DEFAULT_BITS
    from vaayu import turboquant
    assert _DEFAULT_BITS == 9
    assert compress_cache(_cache(seq=8)).bits_total == 9
    with _pytest.raises(ValueError):
        turboquant.make_codebook(9, DIM)


def test_default_outliers_is_eight_and_costs_the_measured_rate():
    """Uniform 9-bit FAILS the real-query quality bar on Qwen3-235B (worst
    sampled layer 0.0683 vs 0.05). m=8 exact channels put both models under it
    (Qwen 0.0281, MiniMax 0.0263), measured through compress_cache itself. That
    costs ~0.055 B/elem over uniform 9-bit's 1.156, and m is non-monotonic, so
    it is not a knob to trim."""
    from vaayu.kv_compress import _DEFAULT_OUTLIERS
    assert _DEFAULT_OUTLIERS == 8
    c = _cache(seq=64)
    p = compress_cache(c)
    assert p.outlier_channels == 8
    elems = sum(c._seq_len) * 2 * 4 * DIM
    assert p.nbytes / elems == pytest.approx(_rate(DIM, 9, 8, 4, 64), abs=0.002)
    assert p.nbytes / elems < 2.0 * 0.65        # still well under 0.65x bf16


def test_outlier_channels_round_trip_exactly():
    """The whole point of the split: the kept channels carry no quantization
    error at all, in the cache's own dtype."""
    c = _cache(seq=64)
    p = compress_cache(c)
    out = decompress_cache(p, device="cpu")
    for L in range(len(c.key_cache)):
        u = c._seq_len[L]
        chan = p.k_chan[L].long()
        src = c.key_cache[L][..., :u, :].index_select(-1, chan)
        got = out.key_cache[L][..., :u, :].index_select(-1, chan)
        assert torch.equal(src, got)


def test_the_split_is_more_accurate_where_channel_scales_are_uneven():
    """The split buys accuracy from channel-scale structure, which real K has and
    iid noise does not -- on plain Gaussian data it is a wash (0.0162 vs 0.0161),
    and that is expected, not a regression. What it must do is win where a few
    channels carry most of the inner product, which is the case the on-device
    gate fails in."""
    g = torch.Generator().manual_seed(11)
    k = torch.randn(1, 4, 64, DIM, generator=g)
    k[..., ::16] *= 20.0
    c = FP8DynamicCache(quantize=False)
    c.update(k.to(torch.bfloat16), k.to(torch.bfloat16), 0)
    ref = c.key_cache[0][..., :64, :]
    split = decompress_cache(compress_cache(c), device="cpu")
    uniform = decompress_cache(compress_cache(c, outlier_channels=0), device="cpu")
    assert _rel(split.key_cache[0][..., :64, :], ref) < \
           _rel(uniform.key_cache[0][..., :64, :], ref)


def test_channels_are_picked_once_per_layer_not_per_head():
    """One index list per layer, shared by every head. Per-(layer, head)
    selection was measured and is WORSE: at m=8 it takes MiniMax-M2.7's worst
    sampled layer to 0.0725, over the 0.05 bar and worse than not splitting at
    all, where the shared choice reaches 0.0272."""
    g = torch.Generator().manual_seed(7)
    k = torch.randn(1, 2, 32, DIM, generator=g)
    k[0, 0, :, 3] *= 50.0
    k[0, 1, :, 99] *= 50.0
    c = FP8DynamicCache(quantize=False)
    c.update(k.to(torch.bfloat16), k.to(torch.bfloat16), 0)
    chan = compress_cache(c).k_chan[0]
    assert chan.dim() == 1 and chan.shape[0] == 8
    # Both heads' outliers are in the one shared set.
    assert {3, 99} <= set(chan.tolist())


def test_m_is_snapped_to_a_bit_packable_residual():
    """(head_dim - m) must stay divisible by 8: pack_codes otherwise falls back
    to one byte per code and the payload roughly DOUBLES -- at head_dim 128,
    m=4 costs 2.031 B/elem, worse than storing bf16. m is snapped up rather than
    taken literally, and head_dim is not assumed to be 128."""
    c = _cache(layers=1, heads=2, seq=16)
    p = compress_cache(c, outlier_channels=4)
    assert p.k_chan[0].shape[0] == 8
    elems = sum(c._seq_len) * 2 * 2 * DIM
    assert p.nbytes / elems == pytest.approx(_rate(DIM, 9, 8, 2, 16), abs=0.002)

