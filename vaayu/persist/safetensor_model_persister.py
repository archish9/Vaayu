import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister

_DTYPE_MAP = {
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "F32":  torch.float32,
    "F64":  torch.float64,
    "I8":   torch.int8,
    "I16":  torch.int16,
    "I32":  torch.int32,
    "I64":  torch.int64,
    "U8":   torch.uint8,
    "BOOL": torch.bool,
}

_ALIGN = 4096  # O_DIRECT sector-alignment requirement

def _align_down(x: int) -> int:
    return x & ~(_ALIGN - 1)

def _align_up(x: int) -> int:
    return (x + _ALIGN - 1) & ~(_ALIGN - 1)


_USE_PINNED = torch.cuda.is_available()
if _USE_PINNED:
    print(">>>> pinned-memory loader active — direct NVMe→RAM→GPU path")
else:
    print(">>>> CUDA not available — using Python safetensors loader")

try:
    import vaayu_rust
    _USE_IOURING = True
    print(">>>> Rust io_uring reader loaded — O_DIRECT page-cache bypass active (QD32)")
except ImportError:
    _USE_IOURING = False
    print(">>>> Rust io_uring NOT available — preadv QD16 fallback active")

# ── Global state for the loader ──────────────────────────────────────
_NUM_IO_THREADS = 16            # QD16 = sweet spot for Samsung 990 PRO (preadv fallback)
_EXECUTOR = None                # Lazily initialized ThreadPoolExecutor (preadv fallback)
_PINNED_BUFS = [None, None]     # Double buffer (A/B)
_PINNED_SIZES = [0, 0]          # Allocated size of each buffer
_BUF_IDX = 0                    # Toggles 0 → 1 → 0 → 1 …
_BUF_LOCK = __import__("threading").Lock()  # Guards _BUF_IDX toggle under concurrent prefetch

_RING_INITIALIZED = False   # True after vaayu_rust.init_ring() succeeds
_FILES_REGISTERED = False   # True after register_files() succeeds


def _ensure_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_NUM_IO_THREADS)


def _init_iouring_ring(queue_depth: int = 32) -> bool:
    try:
        vaayu_rust.init_ring(queue_depth)
        print(f"[VAAYU-IO] persistent ring initialized  qd={queue_depth}")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] init_ring failed: {e} — per-call ring fallback active")
        return False


def _register_iouring_files(model_dir: str) -> bool:
    try:
        paths = [str(p) for p in Path(model_dir).glob("*.safetensors")]
        if not paths:
            print(f"[VAAYU-IO] no .safetensors files in {model_dir} — file registration skipped")
            return False
        vaayu_rust.register_files(paths)
        print(f"[VAAYU-IO] {len(paths)} files registered with io_uring")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] register_files failed: {e} — fd registration disabled")
        return False


def _load_pinned(file_path: str) -> dict:
    global _PINNED_BUFS, _PINNED_SIZES, _BUF_IDX

    # ── 1. Parse safetensors header ─────────────────────────────────
    with open(file_path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

    tensors_meta = {k: v for k, v in hdr.items() if k != '__metadata__'}
    if not tensors_meta:
        return {}

    total_size = max(v['data_offsets'][1] for v in tensors_meta.values())
    file_size = os.path.getsize(file_path)

    # ── 2. Acquire pinned buffer (double-buffer swap) ───────────────
    with _BUF_LOCK:
        buf_idx = _BUF_IDX
        _BUF_IDX = 1 - _BUF_IDX

    if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < total_size:
        alloc_size = total_size + (100 * 1024 * 1024) + _ALIGN  # pad +100 MB + O_DIRECT padding
        _PINNED_BUFS[buf_idx] = torch.empty(alloc_size, dtype=torch.uint8,
                                            pin_memory=True)
        _PINNED_SIZES[buf_idx] = alloc_size
        print(f"[VAAYU-IO] alloc  buf={buf_idx}  size={alloc_size / (1024**2):.1f}MB")
        # Re-register buffers with io_uring after alloc (only when both buffers are allocated)
        if (_USE_IOURING and _RING_INITIALIZED
                and _PINNED_BUFS[0] is not None and _PINNED_BUFS[1] is not None):
            try:
                vaayu_rust.reregister_buffers(
                    _PINNED_BUFS[0].data_ptr(), _PINNED_SIZES[0],
                    _PINNED_BUFS[1].data_ptr(), _PINNED_SIZES[1],
                )
            except Exception as e:
                print(f"[VAAYU-IO] reregister_buffers failed: {e}")

    buf = _PINNED_BUFS[buf_idx]

    if _USE_IOURING:
        # ── 3a. Rust io_uring read (zero-copy into pinned buffer) ───
        aligned_offset = _align_down(data_start)
        prefix_pad = data_start - aligned_offset
        aligned_length = _align_up(total_size + prefix_pad)
        print(f"[VAAYU-IO] io  file={os.path.basename(file_path)}"
              f"  data_start={data_start}  total={total_size}  file_size={file_size}")
        print(f"[VAAYU-IO] align  raw_off={data_start}→aligned_off={aligned_offset}"
              f"  pad={prefix_pad}  raw_len={total_size}→aligned_len={aligned_length}")
        vaayu_rust.io_uring_read_into(
            file_path,
            aligned_offset,
            aligned_length,
            buf.data_ptr(),
            32,
        )
        buf_view = buf[prefix_pad:]
    else:
        # ── 3b. Fallback: Python preadv (QD16) ──────────────────────
        print(f"[VAAYU-IO] preadv  file={os.path.basename(file_path)}"
              f"  data_start={data_start}  total={total_size}  threads={_NUM_IO_THREADS}")
        _ensure_executor()
        view = memoryview(buf.numpy())
        fd = os.open(file_path, os.O_RDONLY)
        try:
            chunk_size = (total_size + _NUM_IO_THREADS - 1) // _NUM_IO_THREADS

            def _read_chunk(i):
                offset = i * chunk_size
                length = min(chunk_size, total_size - offset)
                if length > 0:
                    os.preadv(fd, [view[offset:offset + length]],
                              data_start + offset)

            list(_EXECUTOR.map(_read_chunk, range(_NUM_IO_THREADS)))
        finally:
            os.close(fd)

    # ── 4. Reconstruct tensors as zero-copy views ───────────────────
    result = {}
    source_buf = buf_view if _USE_IOURING else buf
    for name, meta in tensors_meta.items():
        dtype = _DTYPE_MAP.get(meta['dtype'], torch.float32)
        start, end = meta['data_offsets']
        result[name] = source_buf[start:end].view(dtype).reshape(meta['shape'])

    return result


class SafetensorModelPersister(ModelPersister):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_persist_exist(self, layer_name, saving_path):
        safetensor_exists = Path(saving_path / (layer_name + '.safetensors')).exists()
        done_marker_exists = Path(saving_path / (layer_name + '.safetensors.done')).exists()
        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + '.safetensors'))
        print(f"saved as: {saving_path / (layer_name + '.safetensors')}")
        (saving_path / (layer_name + '.safetensors.done')).touch()

    def load_model(self, layer_name, path):
        global _RING_INITIALIZED, _FILES_REGISTERED
        if _USE_IOURING and not _RING_INITIALIZED:
            _RING_INITIALIZED = _init_iouring_ring(32)
        if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
            _FILES_REGISTERED = _register_iouring_files(str(path))

        file_path = str(Path(path) / (layer_name + ".safetensors"))

        if _USE_PINNED:
            t = time.time()
            result = _load_pinned(file_path)
            elapsed = time.time() - t
            tag = "iouring" if _USE_IOURING else "pinned"
            size_mb = sum(v.nbytes for v in result.values()) / (1024**2)
            gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
            print(f"[VAAYU-IO] {tag}  {elapsed:.3f}s  {size_mb:.1f}MB  {gbps:.2f}GB/s  {layer_name}")
            return result
        else:
            t = time.time()
            result = load_file(file_path, device="cpu")
            elapsed = time.time() - t
            size_mb = sum(v.nbytes for v in result.values()) / (1024**2)
            gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
            print(f"[VAAYU-IO] python  {elapsed:.3f}s  {size_mb:.1f}MB  {gbps:.2f}GB/s  {layer_name}")
            return result
