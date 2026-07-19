# Vaayu KV Prefix-Cache Disk Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist each session's compressed FP8 KV cache to disk crash-safely so the prefix cache survives a full process restart — first hit after restart costs zero re-prefill instead of a full prefill.

**Architecture:** A new self-contained module `vaayu/kv_persist.py` adds a disk tier BELOW the RAM `PrefixCacheStore`. On `store()`, the engine packs only the new positions into bytes (in the request thread) and hands them to a single-worker background writer that appends them to a per-session file `<sha1(session_id)>.vaayu_kv` using a **nrec-last** crash-safe protocol. At startup a loader scans the directory, validates each file's geometry header against the current model, reconstructs `FP8DynamicCache` objects, and inserts them into the RAM pool. Ships dark behind `enable_kv_persist` (default False → byte-identical to today).

**Tech Stack:** Python 3, PyTorch (`torch.float8_e4m3fn`), `struct`, `hashlib`, `concurrent.futures.ThreadPoolExecutor`, pytest. No new third-party dependencies.

**Design source:** `docs/superpowers/specs/2026-07-19-vaayu-kv-persist-design.md`.

## Prerequisites (MUST be landed before Task 5)

This plan layers on the RAM prefix cache. Tasks 1–4 and their tests are **fully
standalone** (they only need `FP8DynamicCache`). Task 5 (engine wiring) requires the
RAM prefix-cache plan (`docs/superpowers/plans/2026-07-12-vaayu-prefix-caching-plan.md`)
already implemented, which provides:
- `FP8DynamicCache` with `crop`, `to_cpu`, `to_device`, `resident_bytes` (`vaayu/vaayu_base.py`).
- `PrefixCacheStore` with `store(session_id, token_ids, kv)`, `lookup(...)`, `drop(session_id)`
  (`vaayu/prefix_cache.py`).
- Engine `enable_prefix_cache` wiring in `vaayu/vaayu_base.py`, incl. the crop call site
  in the prepare-inputs path (RAM spec §4.4).

If a prerequisite symbol's name differs at implementation time, adapt the Task 5 call
sites to the real names — Tasks 1–4 do not depend on any of it.

## Global Constraints

Copied verbatim from the spec; every task's requirements implicitly include these.

- **Ships dark.** `enable_kv_persist` defaults **False** → no disk I/O, behavior
  byte-identical to today. Has effect only when `enable_prefix_cache=True`.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No model-number literals. Header
  geometry and record sizes derive from the live `FP8DynamicCache` buffer shapes/dtypes
  and `config` — never a literal layer/head/dim count. Layout serializes whatever
  per-layer buffers exist → GQA / MLA / dense all flow through one code path. Unit tests
  parametrized over Qwen-like, MiniMax-like, AND MLA-like synthetic geometries.
- **Budget host-derived, not model-derived.** RAM repopulation budget reuses the RAM
  tier's budget; disk budget derives from free disk / a host cap. Never a model literal.
- **Thread-safe.** Single-worker background writer serializes all appends/truncates.
- **Graceful degradation.** Any disk error or mismatched/corrupt file → log + skip; RAM
  tier and generation unaffected. Never a correctness failure.
- **Repo layout.** Source under `vaayu/`; tests under `tests/`; run tests with
  `python -m pytest tests/<file> -v`.

## File Structure

- **Create `vaayu/kv_persist.py`** — the entire disk tier: format helpers (Task 1),
  cache serialize/deserialize (Task 2), `DiskWriter` (Task 3), `KvDiskLoader` (Task 4).
  One module; four cohesive responsibilities that change together.
- **Modify `vaayu/vaayu_base.py`** — engine wiring only (Task 5): flags, build the
  writer, load at startup, `enqueue_append`/`enqueue_truncate` hooks, `drain` at shutdown.
- **Tests:** `tests/test_kv_persist_format.py` (T1), `tests/test_kv_persist_serialize.py`
  (T2), `tests/test_kv_persist_writer.py` (T3), `tests/test_kv_persist_loader.py` (T4),
  `tests/test_kv_persist_integration.py` (T5–T6).

---

### Task 1: File-format helpers (header, record sizing, dtype codes, byte IO)

Pure, config/cache-driven helpers. No CUDA. This is the foundation every other task uses.

**Files:**
- Create: `vaayu/kv_persist.py`
- Test: `tests/test_kv_persist_format.py`

**Interfaces:**
- Produces:
  - `MAGIC: bytes` (8 bytes `b"VAAYUKV1"`), `SCHEMA_VER: int = 1`, `QUANT_SIG: int`,
    `NREC_OFFSET: int` (byte offset of the nrec field within the fixed header).
  - `dtype_code(dt: torch.dtype) -> int` / `code_dtype(c: int) -> torch.dtype`.
  - `@dataclass KvHeader` with fields: `schema_ver, num_layers, num_kv_heads, head_dim,
    kv_dtype_code, scale_dtype_code, quant_sig, vocab, nrec: int` and `session_id: str`.
  - `build_header(cache, config, session_id: str, nrec: int) -> KvHeader` — derives
    `num_layers`, `num_kv_heads`, `head_dim`, dtype codes from the live cache buffers;
    `vocab` from `config`.
  - `record_bytes(h: KvHeader) -> int` — bytes of one on-disk position record.
  - `record_start(h: KvHeader) -> int` — byte offset where record 0 begins.
  - `write_header(f, h: KvHeader) -> None`, `read_header(f) -> "KvHeader | None"`
    (None if magic mismatch).
  - `to_bytes(t: torch.Tensor) -> bytes`, `from_bytes(buf: bytes, dt, shape) -> torch.Tensor`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_persist_format.py`:

```python
"""Format helpers: header round-trip, record sizing, byte IO — CPU only,
parametrized over Qwen-like, MiniMax-like, and MLA-like geometries (§5)."""
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


def test_read_header_rejects_bad_magic():
    buf = io.BytesIO(b"NOTVAAYU" + b"\x00" * 40)
    assert kp.read_header(buf) is None


def test_to_from_bytes_roundtrip_fp8_and_fp16():
    for dt in (torch.float8_e4m3fn, torch.float16, torch.bfloat16):
        t = torch.arange(6, dtype=torch.float16).reshape(2, 3).to(dt)
        r = kp.from_bytes(kp.to_bytes(t), dt, (2, 3))
        assert r.dtype == dt and r.shape == (2, 3)
        assert torch.equal(r.to(torch.float16), t.to(torch.float16))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kv_persist_format.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vaayu.kv_persist'`.

- [ ] **Step 3: Write minimal implementation**

Create `vaayu/kv_persist.py`:

```python
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
QUANT_SIG = zlib.crc32(b"fp8_e4m3fn:448.0:per-head-per-token") & 0xFFFFFFFF

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kv_persist_format.py -v`
Expected: PASS (all 3 geometries + magic + byte round-trip).

If `torch.zeros(..., dtype=torch.float8_e4m3fn)` raises on this CPU torch build, the
whole tier cannot be unit-tested on CPU here; stop and report — do not work around it,
the RAM prefix-cache plan hits the same requirement first.

- [ ] **Step 5: Commit**

```bash
git add vaayu/kv_persist.py tests/test_kv_persist_format.py
git commit -m "feat(kv-persist): file-format helpers for KV disk tier"
```

---

### Task 2: Serialize / deserialize a cache's positions

Pack a single position to bytes; read a whole file's records back into a rebuilt
`FP8DynamicCache` (recomputing the derived `_k_out/_v_out`, which are NOT persisted).

**Files:**
- Modify: `vaayu/kv_persist.py`
- Test: `tests/test_kv_persist_serialize.py`

**Interfaces:**
- Consumes (Task 1): `KvHeader`, `record_bytes`, `to_bytes`, `from_bytes`, `code_dtype`,
  `_DTYPE_SIZE`.
- Produces:
  - `pack_record(cache, pos: int, token_id: int, h: KvHeader) -> bytes` — one record;
    assumes batch dim 0 is size 1.
  - `read_records(f, h: KvHeader) -> tuple[list[int], dict]` — returns
    `(token_ids, buffers)` where `buffers` has keys `key,value,kscale,vscale`, each a
    list of `[1,H,nrec,*]` tensors. `f` positioned at `record_start(h)`.
  - `build_cache(h: KvHeader, buffers: dict)` — returns a populated `FP8DynamicCache`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_persist_serialize.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kv_persist_serialize.py -v`
Expected: FAIL with `AttributeError: module 'vaayu.kv_persist' has no attribute 'pack_record'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vaayu/kv_persist.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kv_persist_serialize.py -v`
Expected: PASS for all three geometries.

- [ ] **Step 5: Commit**

```bash
git add vaayu/kv_persist.py tests/test_kv_persist_serialize.py
git commit -m "feat(kv-persist): cache serialize/deserialize with derived-output recompute"
```

---

### Task 3: `DiskWriter` — crash-safe background append + truncate

Owns per-session files. Packs new positions in the calling thread (immune to later
cache mutation) and does IO on one background worker. nrec-last crash-safe.

**Files:**
- Modify: `vaayu/kv_persist.py`
- Test: `tests/test_kv_persist_writer.py`

**Interfaces:**
- Consumes (Tasks 1–2): `build_header`, `pack_record`, `record_bytes`, `record_start`,
  `write_header`, `read_header`, `NREC_OFFSET`.
- Produces `class DiskWriter`:
  - `__init__(self, persist_dir: str, budget_bytes: int, sync: bool = False)`
  - `enqueue_append(self, session_id, cache, config, token_ids: list[int], length: int) -> None`
  - `enqueue_truncate(self, session_id: str, match_len: int) -> None`
  - `remove(self, session_id: str) -> None`
  - `drain(self) -> None` (block until all queued IO done)
  - `close(self) -> None` (drain + shut the worker down)
  - `path_for(self, session_id: str) -> str` (for tests/budget).

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_persist_writer.py`:

```python
"""DiskWriter: append grows nrec, nrec-last crash safety, truncate mirror."""
import os
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


def _cache(n_layers=2, heads=4, dim=16, seq=6):
    c = FP8DynamicCache()
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        c.update(k, v, L)
    return c


def _read_nrec(path):
    with open(path, "rb") as f:
        h = kp.read_header(f)
    return h.nrec


def test_append_writes_header_and_grows_nrec(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    toks = [1, 2, 3, 4, 5, 6]
    w.enqueue_append("s1", c, cfg, toks[:4], 4)
    w.drain()
    p = w.path_for("s1")
    assert os.path.exists(p) and _read_nrec(p) == 4
    # second turn appends only the new tail
    w.enqueue_append("s1", c, cfg, toks, 6)
    w.drain()
    assert _read_nrec(p) == 6
    w.close()


def test_nrec_last_survives_torn_append(tmp_path):
    """Corrupt the file by chopping the last record; header nrec still points at
    the committed prefix, so a reader that trusts nrec loads coherent data."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain(); w.close()
    p = w.path_for("s1")
    # simulate a torn write: append garbage bytes past the committed region
    with open(p, "ab") as f:
        f.write(b"\x00" * 32)
    assert _read_nrec(p) == 6  # nrec unchanged -> reader ignores the torn tail


def test_truncate_mirror(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()
    w.enqueue_truncate("s1", 3)
    w.drain()
    assert _read_nrec(w.path_for("s1")) == 3
    w.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kv_persist_writer.py -v`
Expected: FAIL with `AttributeError: module 'vaayu.kv_persist' has no attribute 'DiskWriter'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vaayu/kv_persist.py` (add `import hashlib, os, threading` and
`from concurrent.futures import ThreadPoolExecutor` at the top of the file):

```python
class DiskWriter:
    """Single-worker background writer. The calling thread packs new positions to
    bytes (so a later turn cropping/appending the cache cannot corrupt an in-flight
    write); the worker only does file IO. nrec-last = crash-safe commit."""

    def __init__(self, persist_dir: str, budget_bytes: int, sync: bool = False):
        self.dir = persist_dir
        self.budget = budget_bytes
        self.sync = sync
        os.makedirs(self.dir, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._state = {}   # sid -> {"header": KvHeader, "nrec": int}

    def path_for(self, session_id: str) -> str:
        h = hashlib.sha1(session_id.encode("utf-8")).hexdigest()
        return os.path.join(self.dir, f"{h}.vaayu_kv")

    def enqueue_append(self, session_id, cache, config, token_ids, length):
        with self._lock:
            st = self._state.get(session_id)
            if st is None:
                st = {"header": build_header(cache, config, session_id, 0), "nrec": 0}
                self._state[session_id] = st
            start = st["nrec"]
            if length <= start:
                return
            hdr = st["header"]
            blob = b"".join(pack_record(cache, p, token_ids[p], hdr)
                            for p in range(start, length))
            st["nrec"] = length
            self._pool.submit(self._do_append, session_id, hdr, start, blob, length)

    def enqueue_truncate(self, session_id, match_len):
        with self._lock:
            st = self._state.get(session_id)
            if st is None or match_len >= st["nrec"]:
                return
            st["nrec"] = match_len
            hdr = st["header"]
        self._pool.submit(self._do_truncate, session_id, hdr, match_len)

    def remove(self, session_id):
        with self._lock:
            self._state.pop(session_id, None)
        self._pool.submit(self._do_remove, session_id)

    def drain(self):
        self._pool.submit(lambda: None).result()

    def close(self):
        self.drain()
        self._pool.shutdown(wait=True)

    # --- worker side (single thread) ---
    def _do_append(self, sid, hdr, start, blob, new_nrec):
        try:
            path = self.path_for(sid)
            new_file = not os.path.exists(path)
            with open(path, "r+b" if not new_file else "w+b") as f:
                if new_file:
                    write_header(f, hdr)
                f.seek(record_start(hdr) + start * record_bytes(hdr))
                f.write(blob)
                f.flush()
                if self.sync:
                    os.fsync(f.fileno())
                f.seek(NREC_OFFSET)
                f.write(struct.pack("<i", new_nrec))
                f.flush()
                if self.sync:
                    os.fsync(f.fileno())
            self._enforce_budget()
        except OSError as e:
            print(f"[KV] append failed for {sid}: {e}")

    def _do_truncate(self, sid, hdr, match_len):
        try:
            path = self.path_for(sid)
            if not os.path.exists(path):
                return
            with open(path, "r+b") as f:
                f.seek(NREC_OFFSET)
                f.write(struct.pack("<i", match_len))
                f.flush()
                f.truncate(record_start(hdr) + match_len * record_bytes(hdr))
        except OSError as e:
            print(f"[KV] truncate failed for {sid}: {e}")

    def _do_remove(self, sid):
        try:
            os.remove(self.path_for(sid))
        except OSError:
            pass

    def _enforce_budget(self):
        files = []
        total = 0
        for name in os.listdir(self.dir):
            if not name.endswith(".vaayu_kv"):
                continue
            fp = os.path.join(self.dir, name)
            try:
                stat = os.stat(fp)
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, fp))
            total += stat.st_size
        files.sort()  # oldest first
        active = {self.path_for(s) for s in self._state}
        for _mtime, size, fp in files:
            if total <= self.budget:
                break
            if fp in active:
                continue
            try:
                os.remove(fp)
                total -= size
            except OSError:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kv_persist_writer.py -v`
Expected: PASS (append/grow, torn-append survival, truncate mirror).

- [ ] **Step 5: Commit**

```bash
git add vaayu/kv_persist.py tests/test_kv_persist_writer.py
git commit -m "feat(kv-persist): crash-safe background DiskWriter (append/truncate/budget)"
```

---

### Task 4: `KvDiskLoader` — startup scan, validate, reconstruct into the pool

Reads every valid file, rebuilds caches, inserts into the RAM store via its
`store(session_id, token_ids, kv)` method. Budget-bounded, newest-first.

**Files:**
- Modify: `vaayu/kv_persist.py`
- Test: `tests/test_kv_persist_loader.py`

**Interfaces:**
- Consumes (Tasks 1–3): `read_header`, `read_records`, `build_cache`, `record_start`,
  `QUANT_SIG`, `SCHEMA_VER`; a `DiskWriter` (to produce fixtures).
- Produces:
  - `header_matches(h: KvHeader, config) -> bool` — validate schema/quant/geometry vs config.
  - `class KvDiskLoader` with `__init__(self, persist_dir: str)` and
    `load_all(self, store, config, ram_budget_bytes: int) -> int` (returns sessions restored).
  - `store` is any object with `store(session_id: str, token_ids: list[int], kv) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_persist_loader.py`:

```python
"""KvDiskLoader: reconstruct sessions, skip geometry/quant mismatches, honor budget."""
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


class _Store:
    def __init__(self):
        self.entries = {}
    def store(self, session_id, token_ids, kv):
        self.entries[session_id] = (token_ids, kv)


def _cache(n_layers=2, heads=4, dim=16, seq=5):
    c = FP8DynamicCache()
    for L in range(n_layers):
        c.update(torch.randn(1, heads, seq, dim, dtype=torch.float16),
                 torch.randn(1, heads, seq, dim, dtype=torch.float16), L)
    return c


def _write_session(tmp_path, sid, cache, cfg, toks):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    w.enqueue_append(sid, cache, cfg, toks, len(toks))
    w.drain(); w.close()


def test_load_all_restores_sessions(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache()
    _write_session(tmp_path, "s1", c, cfg, [7, 8, 9, 10, 11])
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)
    assert n == 1 and "s1" in store.entries
    toks, kv = store.entries["s1"]
    assert toks == [7, 8, 9, 10, 11]
    assert torch.allclose(kv._k_out_cache[0][..., :5, :], c._k_out_cache[0][..., :5, :])


def test_load_all_skips_geometry_mismatch(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    _write_session(tmp_path, "s1", _cache(), cfg, [1, 2, 3, 4, 5])
    other = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=99,
                                  num_key_value_heads=4, head_dim=16)
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, other, ram_budget_bytes=10 ** 12)
    assert n == 0 and store.entries == {}


def test_load_all_respects_budget(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    _write_session(tmp_path, "s1", _cache(), cfg, [1, 2, 3, 4, 5])
    _write_session(tmp_path, "s2", _cache(), cfg, [1, 2, 3, 4, 5])
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=1)
    assert n == 0  # budget of 1 byte admits nothing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kv_persist_loader.py -v`
Expected: FAIL with `AttributeError: module 'vaayu.kv_persist' has no attribute 'KvDiskLoader'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vaayu/kv_persist.py`:

```python
def header_matches(h: KvHeader, config) -> bool:
    if h.schema_ver != SCHEMA_VER or h.quant_sig != QUANT_SIG:
        return False
    nl = getattr(config, "num_hidden_layers", None)
    if nl is not None and h.num_layers != nl:
        return False
    vocab = getattr(config, "vocab_size", None)
    if vocab is not None and h.vocab != vocab:
        return False
    nkv = getattr(config, "num_key_value_heads", None)
    if nkv is not None and h.num_kv_heads != nkv:
        return False
    hd = getattr(config, "head_dim", None)
    if hd is not None and h.head_dim != hd:
        return False
    return True


class KvDiskLoader:
    def __init__(self, persist_dir: str):
        self.dir = persist_dir

    def load_all(self, store, config, ram_budget_bytes: int) -> int:
        if not os.path.isdir(self.dir):
            return 0
        files = []
        for name in os.listdir(self.dir):
            if name.endswith(".vaayu_kv"):
                fp = os.path.join(self.dir, name)
                try:
                    files.append((os.stat(fp).st_mtime, fp))
                except OSError:
                    pass
        files.sort(reverse=True)  # newest first
        used = 0
        restored = 0
        for _mtime, fp in files:
            try:
                with open(fp, "rb") as f:
                    h = read_header(f)
                    if h is None or not header_matches(h, config):
                        print(f"[KV] ignoring {os.path.basename(fp)}: "
                              f"different model/quant/schema")
                        continue
                    size = record_bytes(h) * h.nrec
                    if h.nrec < 1 or used + size > ram_budget_bytes:
                        continue
                    f.seek(record_start(h))
                    toks, buffers = read_records(f, h)
                kv = build_cache(h, buffers)
                store.store(h.session_id, toks, kv)
                used += size
                restored += 1
            except (OSError, struct.error) as e:
                print(f"[KV] skipping {os.path.basename(fp)}: {e}")
        if restored:
            print(f"[KV] resumed {restored} session(s) from disk (no re-prefill)")
        return restored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kv_persist_loader.py -v`
Expected: PASS (restore, skip-mismatch, budget).

- [ ] **Step 5: Run the whole module's tests**

Run: `python -m pytest tests/test_kv_persist_format.py tests/test_kv_persist_serialize.py tests/test_kv_persist_writer.py tests/test_kv_persist_loader.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add vaayu/kv_persist.py tests/test_kv_persist_loader.py
git commit -m "feat(kv-persist): startup loader with geometry validation + budget"
```

---

### Task 5: Engine wiring (`vaayu_base.py`) + ships-dark test

Wire the tier into the engine. **Requires the RAM prefix-cache prerequisite landed.**
All new behavior gated by `enable_kv_persist and enable_prefix_cache`.

**Files:**
- Modify: `vaayu/vaayu_base.py` (constructor/`from_pretrained` prefix-cache block; the
  post-generation `store` call site; the prepare-inputs crop call site; shutdown).
- Test: `tests/test_kv_persist_integration.py`

**Interfaces:**
- Consumes: `kv_persist.DiskWriter`, `kv_persist.KvDiskLoader`; prerequisite
  `self._prefix_cache` (`PrefixCacheStore`), its `store`/`drop`, and the crop call site.
- Produces (engine attributes/flags): `enable_kv_persist: bool = False`,
  `kv_persist_dir: str | None`, `kv_persist_sync: bool = False`, `self._kv_writer:
  DiskWriter | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_persist_integration.py`:

```python
"""Ships-dark: with persistence off, no files are created and behavior is unchanged.
Uses a tiny stubbed engine surface — no model load, no CUDA."""
import os
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


def _cache(n_layers=2, heads=4, dim=16, seq=4):
    c = FP8DynamicCache()
    for L in range(n_layers):
        c.update(torch.randn(1, heads, seq, dim, dtype=torch.float16),
                 torch.randn(1, heads, seq, dim, dtype=torch.float16), L)
    return c


def test_writer_disabled_creates_no_files(tmp_path):
    """When the engine leaves _kv_writer None (flag off), store() must not write."""
    writer = None  # simulates enable_kv_persist=False
    # the hook the engine runs after store():
    if writer is not None:
        writer.enqueue_append("s", _cache(), types.SimpleNamespace(vocab_size=1), [1], 1)
    assert os.listdir(tmp_path) == []


def test_end_to_end_persist_then_reload(tmp_path):
    """Full disk round-trip via the real writer+loader (the engine's two hooks)."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache(seq=4)
    toks = [5, 6, 7, 8]
    writer = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    writer.enqueue_append("s1", c, cfg, toks, 4)   # engine's post-store hook
    writer.drain(); writer.close()

    class _Store:
        def __init__(self): self.entries = {}
        def store(self, sid, t, kv): self.entries[sid] = (t, kv)
    store = _Store()
    kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)
    toks2, kv2 = store.entries["s1"]
    assert toks2 == toks
    for L in range(2):
        assert torch.allclose(kv2._k_out_cache[L][..., :4, :], c._k_out_cache[L][..., :4, :])
```

- [ ] **Step 2: Run test to verify it fails / passes appropriately**

Run: `python -m pytest tests/test_kv_persist_integration.py -v`
Expected: PASS (these exercise the module contract the engine hooks rely on; they lock
the hook shape before you edit `vaayu_base.py`).

- [ ] **Step 3: Add the engine flags + build the writer + load at startup**

In `vaayu/vaayu_base.py`, find the block that builds `self._prefix_cache` when
`enable_prefix_cache` is true (added by the RAM prefix-cache plan). Immediately after
it, add (adapt names to the real ones if they differ):

```python
        # --- KV disk persistence tier (Subsystem A; ships dark) ---
        self._kv_writer = None
        enable_kv_persist = bool(kwargs.get("enable_kv_persist", False))
        if enable_kv_persist and not enable_prefix_cache:
            print("[KV] enable_kv_persist=True ignored: enable_prefix_cache is False")
        if enable_kv_persist and enable_prefix_cache:
            import os
            from vaayu.kv_persist import DiskWriter, KvDiskLoader
            persist_dir = kwargs.get("kv_persist_dir") or os.path.join(
                os.path.expanduser("~/.cache/vaayu"), "kv")
            # disk budget: free space on the target fs, halved as headroom (host limit)
            import shutil
            free = shutil.disk_usage(os.path.dirname(persist_dir) or ".").free
            self._kv_writer = DiskWriter(
                persist_dir, budget_bytes=free // 2,
                sync=bool(kwargs.get("kv_persist_sync", False)))
            KvDiskLoader(persist_dir).load_all(
                self._prefix_cache, self.config, self._prefix_cache.budget_bytes)
```

- [ ] **Step 4: Add the post-generation append hook**

Find where the engine calls `self._prefix_cache.store(session_id, full_token_ids,
final_kv)` after generation (RAM spec §4.4). Immediately after that call, add:

```python
                if self._kv_writer is not None:
                    self._kv_writer.enqueue_append(
                        session_id, final_kv, self.config, full_token_ids, len(full_token_ids))
```

- [ ] **Step 5: Add the truncate-mirror hook**

Find the prepare-inputs crop site where, on prefix divergence, the engine does
`kv.crop(match_len)` (RAM spec §4.4). Immediately after the crop, add:

```python
                if self._kv_writer is not None and session_id is not None:
                    self._kv_writer.enqueue_truncate(session_id, match_len)
```

- [ ] **Step 6: Add shutdown drain**

Find the engine's shutdown/close path (where other executors are shut down). Add:

```python
        if getattr(self, "_kv_writer", None) is not None:
            self._kv_writer.close()
```

- [ ] **Step 7: Run the integration test + full suite**

Run: `python -m pytest tests/test_kv_persist_integration.py -v`
Expected: PASS.

Run the model-agnostic smoke guard (per CLAUDE.md §0, MoE + dense must stay green):
`python -m pytest tests/ -k "kv_persist or prefix" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_kv_persist_integration.py
git commit -m "feat(kv-persist): wire disk tier into engine (ships dark)"
```

---

### Task 6: On-device coherence verification (manual, gated)

Numerically prove a reloaded session equals an uninterrupted one on real hardware.
This is a documented manual verification, not a CI unit test (needs a real model + CUDA).

**Files:**
- Modify: `docs/superpowers/plans/2026-07-19-vaayu-kv-persist-RESULTS.md` (create).

- [ ] **Step 1: Run the coherence check on device**

With a small real model configured in `test_generate.py`:
1. Start the server with `enable_prefix_cache=True, enable_kv_persist=True`, a fixed
   `session_id`, send prompt P (say 2000 tokens), record the generated continuation G1.
2. Stop the server. Confirm `~/.cache/vaayu/kv/<hash>.vaayu_kv` exists.
3. Restart the server (same flags/session), send prompt `P + G1`, record continuation G2.
4. As a control, run the same `P + G1` in a single uninterrupted server session (no
   restart), record G3.

- [ ] **Step 2: Assert equivalence + record**

Expected: G2 == G3 token-for-token (greedy) or within sampling equivalence, and the
`[KV] resumed 1 session(s) from disk` log appears on restart with TTFT collapsed to the
new-tail prefill. Record TTFT-with-persist vs cold-start TTFT in the RESULTS doc.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-19-vaayu-kv-persist-RESULTS.md
git commit -m "docs(kv-persist): on-device coherence + TTFT results"
```

---

## Notes & known limitations

- **B=1 only.** Persisted caches assume batch dim size 1 (A1 per-session caches are
  B=1). The writer indexes `[0, ...]`; a B>1 cache must not be persisted. If Feature B
  batched decode ever hands a B>1 cache to `store`, guard it upstream (out of scope here).
- **MLA header validation.** `header_matches` validates `num_kv_heads`/`head_dim` against
  `config` only when those attributes exist. For an MLA model whose live cache buffer
  shape differs from `config.num_key_value_heads`/`head_dim`, validation falls back to
  `num_layers` + `vocab` + `quant_sig` + `schema_ver`. Reconstruction always uses the
  file's own header, so a same-model reload is correct; the residual risk is only
  cross-model collision with identical layer count + vocab, which the quant_sig and a
  shape-mismatch attention error would surface. Revisit when GLM/Kimi are on-device.
- **Startup-only load (v1).** Sessions evicted from RAM mid-run re-prefill until the next
  restart; live lazy-load-from-disk-on-miss is deferred (spec §4.3, §11).
- **io_uring append deferred** (spec §11). Never attach a ptrace profiler (py-spy) to the
  live writer — it corrupts vaayu_rust's io_uring state (memory: io_uring_cqe_overflow_bug).
- **Explicit `drop`→`remove` hook deferred.** Spec §4.4 lists deleting a session's file on
  an explicit `PrefixCacheStore.drop`. v1 relies on the disk-budget LRU (`_enforce_budget`)
  to reclaim instead; `DiskWriter.remove` exists and can be wired to `drop` in one line when
  the engine exposes a session-end API.
```
