import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

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
# FP8 dtypes added in PyTorch 2.1 — add if available
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_MAP["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _DTYPE_MAP["F8_E5M2"] = torch.float8_e5m2

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
_BUFREG_DISABLED = False    # True after first EFAULT — CUDA-pinned memory is
                            # incompatible with IORING_REGISTER_BUFFERS (the kernel's
                            # get_user_pages_fast() fails on NVIDIA-managed pages).
                            # Reads still work via IORING_OP_READ; only the minor
                            # IORING_OP_READ_FIXED zero-copy path is lost.

_LAST_PINNED_TIMINGS: dict = {}
_PINNED_TIMINGS_LOCK = threading.Lock()

_USE_DIRECT_IO: bool = True  # safe default; overridden by init_model_io() (Plan 5)


def _should_use_direct_io(model_dir: str) -> bool:
    """
    O_DIRECT wins when the page cache can't meaningfully cache the model.
    The crossover point is ~40% cacheable (empirically validated).
    """
    try:
        paths = list(Path(model_dir).glob("*.safetensors"))
        model_total_bytes = sum(os.path.getsize(p) for p in paths)
        mem = psutil.virtual_memory()

        # Reserve 8GB for OS + PyTorch + pinned buffers
        available_for_cache = max(0, mem.available - (8 * 1024**3))
        cacheable_fraction = available_for_cache / max(1, model_total_bytes)

        use_direct = cacheable_fraction < 0.40

        if not use_direct:
            print(f"[Vaayu] Sufficient RAM detected ({mem.available / 1024**3:.1f} GB free). "
                  f"Disabling O_DIRECT to leverage page cache for {cacheable_fraction*100:.0f}% of model.")
        else:
            print(f"[Vaayu] Limited RAM ({mem.available / 1024**3:.1f} GB free). "
                  f"Using O_DIRECT to avoid wasteful page cache copies.")

        return use_direct
    except Exception as e:
        print(f"[Vaayu] Failed to detect RAM ({e}), defaulting to O_DIRECT = True")
        return True


def get_and_clear_pinned_timings(file_path: str) -> dict:
    with _PINNED_TIMINGS_LOCK:
        return _LAST_PINNED_TIMINGS.pop(file_path, {})


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


def _register_iouring_files(model_dir: str, use_direct_io: bool = True) -> bool:
    try:
        paths = [str(p) for p in Path(model_dir).glob("*.safetensors")]
        if not paths:
            print(f"[VAAYU-IO] no .safetensors files in {model_dir} — file registration skipped")
            return False
        vaayu_rust.register_files(paths, use_direct_io)
        print(f"[VAAYU-IO] {len(paths)} files registered with io_uring")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] register_files failed: {e} — fd registration disabled")
        return False


def init_model_io(model_dir: str, use_direct_io: bool) -> None:
    """Called once at model load. Initializes ring and registers files explicitly."""
    global _RING_INITIALIZED, _FILES_REGISTERED, _USE_DIRECT_IO
    _USE_DIRECT_IO = use_direct_io
    if _USE_IOURING and not _RING_INITIALIZED:
        _RING_INITIALIZED = _init_iouring_ring(32)
    if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
        _FILES_REGISTERED = _register_iouring_files(model_dir, use_direct_io)


import atexit

def _shutdown_io() -> None:
    if _USE_IOURING and _FILES_REGISTERED:
        try:
            vaayu_rust.unregister_files()
        except Exception:
            pass

atexit.register(_shutdown_io)


def _load_pinned(file_path: str) -> dict:
    global _PINNED_BUFS, _PINNED_SIZES, _BUF_IDX, _BUFREG_DISABLED

    # ── 1. Parse safetensors header ─────────────────────────────────
    _t_hdr = time.perf_counter()
    with open(file_path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    _header_parse_us = int((time.perf_counter() - _t_hdr) * 1e6)

    tensors_meta = {k: v for k, v in hdr.items() if k != '__metadata__'}
    if not tensors_meta:
        return {}
    _io_wait_us = 0
    _io_submit_us = 0

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
        # NOTE: CUDA-pinned memory (pin_memory=True) is incompatible with
        # IORING_REGISTER_BUFFERS — kernel get_user_pages_fast() returns EFAULT.
        # After the first failure we permanently disable to avoid log spam.
        if (_USE_IOURING and _RING_INITIALIZED and not _BUFREG_DISABLED
                and _PINNED_BUFS[0] is not None and _PINNED_BUFS[1] is not None):
            try:
                vaayu_rust.reregister_buffers(
                    _PINNED_BUFS[0].data_ptr(), _PINNED_SIZES[0],
                    _PINNED_BUFS[1].data_ptr(), _PINNED_SIZES[1],
                )
            except Exception as e:
                _BUFREG_DISABLED = True
                print(f"[VAAYU-IO] buffer registration disabled: CUDA-pinned memory "
                      f"incompatible with IORING_REGISTER_BUFFERS ({e}). "
                      f"Using IORING_OP_READ (no perf impact for large reads).")

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
        _t_io = time.perf_counter()
        vaayu_rust.io_uring_read_into(
            file_path,
            aligned_offset,
            aligned_length,
            buf.data_ptr(),
            32,
            _USE_DIRECT_IO,
        )
        _io_wait_us = int((time.perf_counter() - _t_io) * 1e6)
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

            _t_io = time.perf_counter()
            list(_EXECUTOR.map(_read_chunk, range(_NUM_IO_THREADS)))
            _io_wait_us = int((time.perf_counter() - _t_io) * 1e6)
        finally:
            os.close(fd)

    # ── 4. Reconstruct tensors as zero-copy views ───────────────────
    _t_recon = time.perf_counter()
    result = {}
    source_buf = buf_view if _USE_IOURING else buf
    for name, meta in tensors_meta.items():
        dtype = _DTYPE_MAP.get(meta['dtype'], torch.float32)
        start, end = meta['data_offsets']
        result[name] = source_buf[start:end].view(dtype).reshape(meta['shape'])
    _tensor_reconstruct_us = int((time.perf_counter() - _t_recon) * 1e6)

    with _PINNED_TIMINGS_LOCK:
        _LAST_PINNED_TIMINGS[file_path] = {
            "header_parse_us": _header_parse_us,
            "io_submit_us": _io_submit_us,
            "io_wait_us": _io_wait_us,
            "tensor_reconstruct_us": _tensor_reconstruct_us,
        }

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


def load_expert_sparse(
    file_path: str,
    expert_metas: dict,
    use_direct_io: bool | None = None,
) -> dict:
    """
    Load only the specified expert tensor byte ranges from a safetensors shard.

    Groups contiguous byte ranges (gap ≤ _ALIGN) into a single io_uring call
    to minimise syscall overhead. Falls back to preadv when io_uring unavailable.

    Args:
        file_path: absolute path to the .safetensors shard
        expert_metas: {tensor_key: ExpertTensorMeta} from ExpertOffsetIndex
        use_direct_io: O_DIRECT flag; None = use global _USE_DIRECT_IO

    Returns:
        {tensor_key: cpu_tensor} — cloned tensors safe to hold beyond buffer reuse
    """
    if not expert_metas:
        return {}

    if use_direct_io is None:
        use_direct_io = _USE_DIRECT_IO

    # Sort by file offset to enable contiguity detection
    sorted_metas = sorted(expert_metas.items(), key=lambda x: x[1].file_offset)

    # Group contiguous ranges (gap <= _ALIGN → merge into one read)
    groups: list[list] = []
    current_group = [sorted_metas[0]]
    for key, meta in sorted_metas[1:]:
        prev_meta = current_group[-1][1]
        prev_end = prev_meta.file_offset + prev_meta.byte_length
        if meta.file_offset - prev_end <= _ALIGN:
            current_group.append((key, meta))
        else:
            groups.append(current_group)
            current_group = [(key, meta)]
    groups.append(current_group)

    result = {}

    for group in groups:
        group_start = group[0][1].file_offset
        last_meta = group[-1][1]
        group_end = last_meta.file_offset + last_meta.byte_length
        read_len = group_end - group_start

        aligned_offset = _align_down(group_start)
        prefix_pad = group_start - aligned_offset
        aligned_length = _align_up(read_len + prefix_pad)

        buf = torch.empty(aligned_length, dtype=torch.uint8, pin_memory=_USE_PINNED)

        if _USE_IOURING:
            vaayu_rust.io_uring_read_into(
                file_path, aligned_offset, aligned_length,
                buf.data_ptr(), 8, use_direct_io,
            )
            data_buf = buf[prefix_pad:]
        else:
            fd = os.open(file_path, os.O_RDONLY)
            try:
                view = memoryview(buf.numpy())
                os.preadv(fd, [view[:aligned_length]], aligned_offset)
            finally:
                os.close(fd)
            data_buf = buf[prefix_pad:]

        for key, meta in group:
            rel_start = meta.file_offset - group_start
            rel_end = rel_start + meta.byte_length
            dtype = _DTYPE_MAP.get(meta.dtype, torch.float32)
            result[key] = data_buf[rel_start:rel_end].view(dtype).reshape(meta.shape).clone()

    return result
