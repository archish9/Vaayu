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

_USE_PINNED = torch.cuda.is_available()
if _USE_PINNED:
    print(">>>> pinned-memory loader active — direct NVMe→RAM→GPU path")
else:
    print(">>>> CUDA not available — using Python safetensors loader")

try:
    import vaayu_rust
    _USE_IOURING = True
    print(">>>> Rust io_uring reader loaded — targeting 6+ GB/s NVMe bandwidth")
except ImportError:
    _USE_IOURING = False

# ── Global state for the loader ──────────────────────────────────────
_NUM_IO_THREADS = 16            # QD16 = sweet spot for Samsung 990 PRO (preadv fallback)
_EXECUTOR = None                # Lazily initialized ThreadPoolExecutor (preadv fallback)
_PINNED_BUFS = [None, None]     # Double buffer (A/B)
_PINNED_SIZES = [0, 0]          # Allocated size of each buffer
_BUF_IDX = 0                    # Toggles 0 → 1 → 0 → 1 …


def _ensure_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_NUM_IO_THREADS)


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

    # ── 2. Acquire pinned buffer (double-buffer swap) ───────────────
    buf_idx = _BUF_IDX
    _BUF_IDX = 1 - _BUF_IDX

    if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < total_size:
        alloc_size = total_size + (100 * 1024 * 1024)   # pad +100 MB
        _PINNED_BUFS[buf_idx] = torch.empty(alloc_size, dtype=torch.uint8,
                                            pin_memory=True)
        _PINNED_SIZES[buf_idx] = alloc_size

    buf = _PINNED_BUFS[buf_idx]

    if _USE_IOURING:
        # ── 3a. Rust io_uring read (zero-copy into pinned buffer) ───
        vaayu_rust.io_uring_read_into(
            file_path,
            data_start,
            total_size,
            buf.data_ptr(),
            32,                 # queue depth (SQEs)
        )
    else:
        # ── 3b. Fallback: Python preadv (QD16) ──────────────────────
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
    for name, meta in tensors_meta.items():
        dtype = _DTYPE_MAP.get(meta['dtype'], torch.float32)
        start, end = meta['data_offsets']
        result[name] = buf[start:end].view(dtype).reshape(meta['shape'])

    return result


class SafetensorModelPersister(ModelPersister):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_persist_exist(self, layer_name, saving_path):
        safetensor_exists = Path(saving_path / (layer_name + 'safetensors')).exists()
        done_marker_exists = Path(saving_path / (layer_name + 'safetensors.done')).exists()
        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))
        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")
        (saving_path / (layer_name + 'safetensors.done')).touch()

    def load_model(self, layer_name, path):
        file_path = str(Path(path) / (layer_name + ".safetensors"))

        if _USE_PINNED:
            t = time.time()
            result = _load_pinned(file_path)
            elapsed = time.time() - t
            tag = "iouring" if _USE_IOURING else "pinned"
            print(f"[VAAYU-IO] {tag} {elapsed:.3f}s  {layer_name}")
            return result
        else:
            t = time.time()
            result = load_file(file_path, device="cpu")
            elapsed = time.time() - t
            print(f"[VAAYU-IO] python {elapsed:.3f}s  {layer_name}")
            return result
