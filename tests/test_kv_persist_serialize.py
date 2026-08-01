"""Round-trip a real FP8DynamicCache through pack_record / read_records /
build_cache. CPU-only, all three geometries."""
import io
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache

GEOMS = [
    pytest.param(3, 8, 16, id="qwen-like"),
    pytest.param(4, 4, 16, id="minimax-like"),
    pytest.param(2, 1, 64, id="mla-like"),
]


def _make_cache(n_layers, heads, dim, seq):
    c = FP8DynamicCache()
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        c.update(k, v, L)
    return c


@pytest.mark.parametrize("n_layers,heads,dim", GEOMS)
def test_pack_read_build_roundtrip(n_layers, heads, dim):
    seq = 5
    c = _make_cache(n_layers, heads, dim, seq)
    cfg = types.SimpleNamespace(vocab_size=1000)
    toks = [10, 11, 12, 13, 14]
    h = kp.build_header(c, cfg, "s", nrec=seq)

    buf = io.BytesIO()
    kp.write_header(buf, h)
    for p in range(seq):
        rec = kp.pack_record(c, p, toks[p], h)
        assert len(rec) == kp.record_bytes(h)
        buf.write(rec)

    buf.seek(kp.record_start(h))
    got_toks, buffers = kp.read_records(buf, h)
    assert got_toks == toks

    c2 = kp.build_cache(h, buffers)
    for L in range(n_layers):
        # persisted fp8 bytes + scales are byte-identical
        assert torch.equal(c2.key_cache[L][..., :seq, :].to(torch.float16),
                           c.key_cache[L][..., :seq, :].to(torch.float16))
        assert torch.equal(c2._key_scales[L][..., :seq, :].to(torch.float16),
                           c._key_scales[L][..., :seq, :].to(torch.float16))
        # derived dequant output recomputed to match
        assert torch.allclose(c2._k_out_cache[L][..., :seq, :],
                             c._k_out_cache[L][..., :seq, :])
        assert c2._seq_len[L] == seq


def _make_bf16_cache(n_layers, heads, dim, seq):
    c = FP8DynamicCache(quantize=False)
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16)
        v = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16)
        c.update(k, v, L)
    return c


@pytest.mark.parametrize("n_layers,heads,dim", GEOMS)
def test_bf16_pack_read_build_roundtrip(n_layers, heads, dim):
    """An unquantized cache carries no scales; the format must round-trip it
    losslessly and rebuild a cache that is still quantize=False."""
    seq = 5
    c = _make_bf16_cache(n_layers, heads, dim, seq)
    cfg = types.SimpleNamespace(vocab_size=1000)
    toks = [10, 11, 12, 13, 14]
    h = kp.build_header(c, cfg, "s", nrec=seq)
    assert h.quant_sig == kp.QUANT_SIG_BF16

    buf = io.BytesIO()
    kp.write_header(buf, h)
    for p in range(seq):
        rec = kp.pack_record(c, p, toks[p], h)
        assert len(rec) == kp.record_bytes(h)
        buf.write(rec)

    buf.seek(kp.record_start(h))
    got_toks, buffers = kp.read_records(buf, h)
    assert got_toks == toks

    c2 = kp.build_cache(h, buffers)
    assert c2.quantize is False
    assert c2._key_scales == [] and c2._value_scales == []
    assert c2._k_out_cache == [] and c2._v_out_cache == []
    for L in range(n_layers):
        # bf16 storage is lossless: bytes must come back EXACTLY equal
        assert torch.equal(c2.key_cache[L][..., :seq, :], c.key_cache[L][..., :seq, :])
        assert torch.equal(c2.value_cache[L][..., :seq, :], c.value_cache[L][..., :seq, :])
        assert c2._seq_len[L] == seq


def test_bf16_record_bytes_excludes_scale_storage():
    """The scale term must vanish, not be charged at the KV dtype — otherwise
    record_bytes over-counts and every seek into the file lands mid-record."""
    seq, heads, dim, n_layers = 2, 4, 16, 3
    cfg = types.SimpleNamespace(vocab_size=1000)
    h_bf16 = kp.build_header(_make_bf16_cache(n_layers, heads, dim, seq), cfg, "s", nrec=seq)
    expected = 4 + n_layers * (2 * heads * dim * 2)      # token id + K/V bf16, no scales
    assert kp.record_bytes(h_bf16) == expected


def test_bf16_restored_cache_accepts_further_updates():
    """A restored prefix is appended to on the next turn — the rebuilt cache must
    behave like a live one, not just hold bytes."""
    seq, heads, dim = 3, 4, 16
    c = _make_bf16_cache(2, heads, dim, seq)
    cfg = types.SimpleNamespace(vocab_size=1000)
    h = kp.build_header(c, cfg, "s", nrec=seq)
    buf = io.BytesIO()
    kp.write_header(buf, h)
    for p in range(seq):
        buf.write(kp.pack_record(c, p, p, h))
    buf.seek(kp.record_start(h))
    _toks, buffers = kp.read_records(buf, h)
    c2 = kp.build_cache(h, buffers)

    k = torch.randn(1, heads, 1, dim, dtype=torch.bfloat16)
    v = torch.randn(1, heads, 1, dim, dtype=torch.bfloat16)
    out_k, _out_v = c2.update(k, v, 0)
    assert out_k.shape[-2] == seq + 1
    assert torch.equal(out_k[..., -1, :], k[..., 0, :])
    assert c2._seq_len[0] == seq + 1
