"""Format helpers: header round-trip, record sizing, byte IO — CPU only,
parametrized over Qwen-like, MiniMax-like, and MLA-like geometries (§5)."""
import dataclasses
import io
import types
import torch
import pytest
from vaayu import kv_persist as kp


class _Cache:
    """Minimal stand-in exposing the FP8DynamicCache buffer lists we serialize."""
    def __init__(self, n_layers, heads, dim, kv_dt, sc_dt):
        z = lambda d, last: [torch.zeros((1, heads, 3, last), dtype=d) for _ in range(n_layers)]
        self.key_cache = z(kv_dt, dim)
        self.value_cache = z(kv_dt, dim)
        self._key_scales = z(sc_dt, 1)
        self._value_scales = z(sc_dt, 1)


GEOMS = [
    pytest.param(94, 8, 128, id="qwen-like"),
    pytest.param(62, 4, 128, id="minimax-like"),
    pytest.param(61, 1, 576, id="mla-like"),   # MLA: 1 kv head, large latent dim
]


@pytest.mark.parametrize("n_layers,heads,dim", GEOMS)
def test_header_roundtrip_and_record_bytes(n_layers, heads, dim):
    cache = _Cache(n_layers, heads, dim, torch.float8_e4m3fn, torch.float16)
    cfg = types.SimpleNamespace(vocab_size=151936)
    h = kp.build_header(cache, cfg, "sess-A", nrec=17)
    assert h.num_layers == n_layers and h.num_kv_heads == heads and h.head_dim == dim
    assert h.quant_sig == kp.QUANT_SIG and h.schema_ver == kp.SCHEMA_VER
    # record = 4 (tok) + layers*(2*heads*dim*1byte_fp8 + 2*heads*2byte_fp16)
    expected = 4 + n_layers * (2 * heads * dim * 1 + 2 * heads * 2)
    assert kp.record_bytes(h) == expected

    buf = io.BytesIO()
    kp.write_header(buf, h)
    buf.seek(0)
    h2 = kp.read_header(buf)
    assert h2 == h
    assert kp.record_start(h) == buf.tell()  # records begin right after header+sid


def test_unquantized_cache_gets_its_own_quant_sig():
    """A bf16 ring buffer is a different on-disk scheme than fp8+scales. It must
    never claim the fp8 signature, or a bf16 file would be read back as fp8.

    Both layouts are now readable (bf16 serialization landed), so the signature's
    job is no longer to get the file rejected — it is what tells the reader which
    layout to parse. That is what has_scales keys off."""
    from vaayu.vaayu_base import FP8DynamicCache
    c = FP8DynamicCache(quantize=False)
    c.update(torch.randn(1, 4, 3, 16, dtype=torch.bfloat16),
             torch.randn(1, 4, 3, 16, dtype=torch.bfloat16), 0)
    cfg = types.SimpleNamespace(vocab_size=1000)
    h = kp.build_header(c, cfg, "sess-bf16", nrec=3)
    assert h.quant_sig == kp.QUANT_SIG_BF16
    assert h.quant_sig != kp.QUANT_SIG
    assert kp.has_scales(h) is False            # parsed as bf16, never as fp8
    assert kp.header_matches(h, cfg) is True    # and it IS restorable

    q = FP8DynamicCache()
    q.update(torch.randn(1, 4, 3, 16, dtype=torch.float16),
             torch.randn(1, 4, 3, 16, dtype=torch.float16), 0)
    hq = kp.build_header(q, cfg, "sess-fp8", nrec=3)
    assert kp.has_scales(hq) is True
    # Scales cost real bytes — compare the SAME header with only the signature
    # flipped (fp8 vs bf16 records differ in KV width too, so the two real
    # headers are not comparable).
    assert kp.record_bytes(hq) > kp.record_bytes(
        dataclasses.replace(hq, quant_sig=kp.QUANT_SIG_BF16))


def test_read_header_rejects_bad_magic():
    buf = io.BytesIO(b"NOTVAAYU" + b"\x00" * 40)
    assert kp.read_header(buf) is None


def test_to_from_bytes_roundtrip_fp8_and_fp16():
    for dt in (torch.float8_e4m3fn, torch.float16, torch.bfloat16):
        t = torch.arange(6, dtype=torch.float16).reshape(2, 3).to(dt)
        r = kp.from_bytes(kp.to_bytes(t), dt, (2, 3))
        assert r.dtype == dt and r.shape == (2, 3)
        assert torch.equal(r.to(torch.float16), t.to(torch.float16))
