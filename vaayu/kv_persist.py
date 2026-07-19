"""Disk persistence tier for the KV prefix cache (Subsystem A).

Serializes each session's compressed FP8 KV cache to a per-session append-only
file so the RAM prefix cache survives a process restart. Model-agnostic: every
size is derived from the live FP8DynamicCache buffers + config (CLAUDE.md §5)."""
import struct
import zlib
from dataclasses import dataclass

import torch

MAGIC = b"VAAYUKV1"
SCHEMA_VER = 1
# quant scheme signature: changes whenever FP8DynamicCache.update's scheme changes.
_QUANT_SIG_UNSIGNED = zlib.crc32(b"fp8_e4m3fn:448.0:per-head-per-token") & 0xFFFFFFFF
QUANT_SIG = struct.unpack('<i', struct.pack('<I', _QUANT_SIG_UNSIGNED))[0]

_DTYPE_CODES = {
    torch.float8_e4m3fn: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float32: 4,
}
_CODE_DTYPES = {v: k for k, v in _DTYPE_CODES.items()}
_DTYPE_SIZE = {torch.float8_e4m3fn: 1, torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}

# fixed header: magic(8) + 9 int32 fields. nrec is the 9th int32.
_FIXED_INTS = 9
_FIXED_BYTES = 8 + _FIXED_INTS * 4          # 44
NREC_OFFSET = 8 + 8 * 4                      # 40


def dtype_code(dt: torch.dtype) -> int:
    return _DTYPE_CODES[dt]


def code_dtype(c: int) -> torch.dtype:
    return _CODE_DTYPES[c]


@dataclass(eq=True)
class KvHeader:
    schema_ver: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    kv_dtype_code: int
    scale_dtype_code: int
    quant_sig: int
    vocab: int
    nrec: int
    session_id: str


def build_header(cache, config, session_id: str, nrec: int) -> KvHeader:
    key0 = cache.key_cache[0]            # [1, H, cap, D]
    return KvHeader(
        schema_ver=SCHEMA_VER,
        num_layers=len(cache.key_cache),
        num_kv_heads=int(key0.shape[1]),
        head_dim=int(key0.shape[3]),
        kv_dtype_code=dtype_code(cache.key_cache[0].dtype),
        scale_dtype_code=dtype_code(cache._key_scales[0].dtype),
        quant_sig=QUANT_SIG,
        vocab=int(getattr(config, "vocab_size", 0)),
        nrec=nrec,
        session_id=session_id,
    )


def record_bytes(h: KvHeader) -> int:
    ks = _DTYPE_SIZE[code_dtype(h.kv_dtype_code)]
    ss = _DTYPE_SIZE[code_dtype(h.scale_dtype_code)]
    return 4 + h.num_layers * (2 * h.num_kv_heads * h.head_dim * ks + 2 * h.num_kv_heads * ss)


def record_start(h: KvHeader) -> int:
    return _FIXED_BYTES + 4 + len(h.session_id.encode("utf-8"))


def write_header(f, h: KvHeader) -> None:
    f.write(MAGIC)
    f.write(struct.pack(
        "<9i", h.schema_ver, h.num_layers, h.num_kv_heads, h.head_dim,
        h.kv_dtype_code, h.scale_dtype_code, h.quant_sig, h.vocab, h.nrec))
    sid = h.session_id.encode("utf-8")
    f.write(struct.pack("<i", len(sid)))
    f.write(sid)


def read_header(f):
    magic = f.read(8)
    if magic != MAGIC:
        return None
    vals = struct.unpack("<9i", f.read(36))
    slen = struct.unpack("<i", f.read(4))[0]
    sid = f.read(slen).decode("utf-8")
    return KvHeader(*vals, session_id=sid)


def to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def from_bytes(buf: bytes, dt: torch.dtype, shape) -> torch.Tensor:
    u = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    return u.view(dt).reshape(shape)


def pack_record(cache, pos: int, token_id: int, h: KvHeader) -> bytes:
    out = bytearray()
    out += struct.pack("<i", int(token_id))
    for L in range(h.num_layers):
        out += to_bytes(cache.key_cache[L][0, :, pos, :])
        out += to_bytes(cache.value_cache[L][0, :, pos, :])
        out += to_bytes(cache._key_scales[L][0, :, pos, :])
        out += to_bytes(cache._value_scales[L][0, :, pos, :])
    return bytes(out)


def read_records(f, h: KvHeader):
    H, D, Ln, nrec = h.num_kv_heads, h.head_dim, h.num_layers, h.nrec
    kdt, sdt = code_dtype(h.kv_dtype_code), code_dtype(h.scale_dtype_code)
    kb = H * D * _DTYPE_SIZE[kdt]
    sb = H * 1 * _DTYPE_SIZE[sdt]
    key = [torch.zeros((1, H, nrec, D), dtype=kdt) for _ in range(Ln)]
    val = [torch.zeros((1, H, nrec, D), dtype=kdt) for _ in range(Ln)]
    ks = [torch.zeros((1, H, nrec, 1), dtype=sdt) for _ in range(Ln)]
    vs = [torch.zeros((1, H, nrec, 1), dtype=sdt) for _ in range(Ln)]
    toks = []
    for p in range(nrec):
        toks.append(struct.unpack("<i", f.read(4))[0])
        for L in range(Ln):
            key[L][0, :, p, :] = from_bytes(f.read(kb), kdt, (H, D))
            val[L][0, :, p, :] = from_bytes(f.read(kb), kdt, (H, D))
            ks[L][0, :, p, :] = from_bytes(f.read(sb), sdt, (H, 1))
            vs[L][0, :, p, :] = from_bytes(f.read(sb), sdt, (H, 1))
    return toks, {"key": key, "value": val, "kscale": ks, "vscale": vs}


def build_cache(h: KvHeader, buffers: dict):
    from vaayu.vaayu_base import FP8DynamicCache
    sdt = code_dtype(h.scale_dtype_code)
    c = FP8DynamicCache()
    c.key_cache = buffers["key"]
    c.value_cache = buffers["value"]
    c._key_scales = buffers["kscale"]
    c._value_scales = buffers["vscale"]
    c._k_out_cache = [buffers["key"][L].to(sdt) * buffers["kscale"][L]
                      for L in range(h.num_layers)]
    c._v_out_cache = [buffers["value"][L].to(sdt) * buffers["vscale"][L]
                      for L in range(h.num_layers)]
    c._seq_len = [h.nrec] * h.num_layers
    return c
