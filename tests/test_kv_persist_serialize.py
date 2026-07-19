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
