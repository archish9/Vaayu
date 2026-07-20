"""Disk persistence tier for the KV prefix cache (Subsystem A).

Serializes each session's compressed FP8 KV cache to a per-session append-only
file so the RAM prefix cache survives a process restart. Model-agnostic: every
size is derived from the live FP8DynamicCache buffers + config (CLAUDE.md §5)."""
import hashlib
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
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
        self._lock = threading.Lock()   # guards _state dict structure only
        self._state = {}   # sid -> {"header": KvHeader, "nrec": int, "lock": Lock, "future": Future|None}

    def path_for(self, session_id: str) -> str:
        h = hashlib.sha1(session_id.encode("utf-8")).hexdigest()
        return os.path.join(self.dir, f"{h}.vaayu_kv")

    def register_existing(self, session_id, header, nrec):
        """Seed writer state for a session restored from disk at startup, so
        the first post-restart append/truncate operates on the TRUE committed
        nrec instead of treating it as brand-new (nrec=0). Without this, the
        first append after a restart would overwrite the whole file from
        position 0 under a stale-high on-disk nrec (a crash-safety hole) and
        rewrite the entire sequence instead of just the new tail."""
        with self._lock:
            if session_id not in self._state:
                self._state[session_id] = {
                    "header": header, "nrec": nrec,
                    "lock": threading.Lock(), "future": None,
                }

    def enqueue_append(self, session_id, cache, config, token_ids, length):
        # self._lock (global) only guards the _state dict structure: atomically
        # look up or create this session's entry -- including its per-session
        # lock -- so two threads can't race to install two different locks for a
        # new session.
        with self._lock:
            st = self._state.get(session_id)
            if st is None:
                st = {"header": build_header(cache, config, session_id, 0),
                      "nrec": 0, "lock": threading.Lock(), "future": None}
                self._state[session_id] = st
        # st["lock"] (per-session) makes wait / read-start / pack / submit one
        # atomic critical section for THIS session only -- different sessions
        # still pack fully in parallel (the parallel-fan-out case; pack_record
        # does real .cpu()/.contiguous() work and never runs under the GLOBAL
        # lock). nrec is NOT advanced here: it only moves forward in _do_append
        # on a confirmed write.
        with st["lock"]:
            # Wait for THIS session's own previously-submitted op to finish before
            # reading nrec. Same-session callers are already serialized by st["lock"]
            # and the single worker is strictly FIFO, so once the prior future
            # resolves, nrec reflects every op submitted before us -- there is no
            # stale-read window in either direction (append-then-truncate or
            # truncate-then-append, C1). The wait blocks only on a genuinely still-
            # pending SAME-session op (rare: seconds of generation separate real
            # truncate/append calls); other sessions are wholly unaffected. Swallow
            # the exception: the worker already logs any failure, and nrec simply
            # stays unadvanced on failure (Repro A).
            prior = st["future"]
            if prior is not None:
                try:
                    prior.result()
                except Exception:
                    pass
            start = st["nrec"]
            if length <= start:
                return
            hdr = st["header"]
            try:
                blob = b"".join(pack_record(cache, p, token_ids[p], hdr)
                                for p in range(start, length))
            except Exception as e:
                print(f"[KV] pack failed for {session_id}: {e}")
                return
            st["future"] = self._pool.submit(
                self._do_append, session_id, hdr, start, blob, length)

    def enqueue_truncate(self, session_id, match_len):
        with self._lock:
            st = self._state.get(session_id)
        if st is None:
            return
        # st["lock"] serializes this truncate against the same session's appends
        # (Repro B). nrec is NOT advanced here: like append, it moves ONLY in the
        # worker (_do_truncate) on a confirmed disk truncate. But we FIRST wait for
        # this session's own prior in-flight op to finish, so st["nrec"] read below
        # is the true post-prior-op value -- an earlier in-flight append's worker
        # callback can no longer land after this and overwrite nrec back up (round-2
        # bug), and a following pipelined append will read the post-truncate nrec so
        # it starts at the truncated boundary rather than skipping over the bytes
        # this truncate physically removes (C1). The early-return guard stays valid:
        # after the wait, nrec == true disk length, so match_len >= nrec means there
        # is nothing at/after match_len to truncate.
        with st["lock"]:
            prior = st["future"]
            if prior is not None:
                try:
                    prior.result()
                except Exception:
                    pass
            if match_len >= st["nrec"]:
                return
            hdr = st["header"]
            st["future"] = self._pool.submit(
                self._do_truncate, session_id, hdr, match_len)

    def remove(self, session_id):
        with self._lock:
            st = self._state.get(session_id)
        if st is not None:
            # Take the session lock BEFORE popping/submitting so a remove can't
            # interleave with an in-flight append/truncate critical section for
            # the same session: once we hold st["lock"], any such op has already
            # submitted, so the delete is ordered after it on the FIFO worker
            # queue -- no stale-start recreate of a gapped file (Repro B). Wait
            # for any still-pending prior op too, keeping the pattern uniform with
            # append/truncate (not strictly required -- st["lock"] already excludes
            # interleaving -- but it makes the delete strictly ordered after any
            # pending write).
            with st["lock"]:
                prior = st["future"]
                if prior is not None:
                    try:
                        prior.result()
                    except Exception:
                        pass
                with self._lock:
                    self._state.pop(session_id, None)
                self._pool.submit(self._do_remove, session_id)
        else:
            # No tracked session (e.g. a stale file from a previous process);
            # nothing can be mid-critical-section for it, so just delete.
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
            # Advance nrec ONLY now, on a confirmed durable write. The caller
            # never set it optimistically, so a failure below leaves nrec
            # untouched -- nothing to roll back, and a pipelined next append
            # re-covers this range from the true committed start (Repro A).
            with self._lock:
                st = self._state.get(sid)
                if st is not None:
                    st["nrec"] = new_nrec
            self._enforce_budget()
        except Exception as e:
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
                if self.sync:
                    os.fsync(f.fileno())
                f.truncate(record_start(hdr) + match_len * record_bytes(hdr))
                if self.sync:
                    os.fsync(f.fileno())
            # Set nrec ONLY now, worker-side, on a confirmed disk truncate --
            # same single-writer pattern as _do_append (under self._lock, on this
            # one worker thread, right after this method's own disk mutation). This
            # keeps every nrec write in true execution order with the actual disk
            # writes, so a truncate can never be reordered against an append.
            with self._lock:
                st = self._state.get(sid)
                if st is not None:
                    st["nrec"] = match_len
        except Exception as e:
            print(f"[KV] truncate failed for {sid}: {e}")

    def _do_remove(self, sid):
        try:
            os.remove(self.path_for(sid))
        except OSError:
            pass  # already gone -- remove() is idempotent, this is expected
        except Exception as e:
            print(f"[KV] remove failed for {sid}: {e}")

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
        with self._lock:
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

    def load_all(self, store, config, ram_budget_bytes: int, writer=None) -> int:
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
                    if h.nrec < 1:
                        continue
                    if used + size > ram_budget_bytes:
                        print(f"[KV] skipping {os.path.basename(fp)}: "
                              f"{size} bytes would exceed RAM budget "
                              f"({used} used + {size} > {ram_budget_bytes})")
                        continue
                    f.seek(record_start(h))
                    toks, buffers = read_records(f, h)
                kv = build_cache(h, buffers)
                store.store(h.session_id, toks, kv)
                if writer is not None:
                    writer.register_existing(h.session_id, h, h.nrec)
                used += size
                restored += 1
            except Exception as e:
                print(f"[KV] skipping {os.path.basename(fp)}: {e}")
        if restored:
            print(f"[KV] resumed {restored} session(s) from disk (no re-prefill)")
        return restored
