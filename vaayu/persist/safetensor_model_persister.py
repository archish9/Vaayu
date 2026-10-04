import json
import os
import struct
import threading
from contextlib import contextmanager
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
# MXFP8-style block-scale dtype (E8M0), added to PyTorch in 2.7 — add if available
if hasattr(torch, "float8_e8m0fnu"):
    _DTYPE_MAP["F8_E8M0"] = torch.float8_e8m0fnu


def _resolve_dtype(dtype_str: str, tensor_name: str, file_path: str) -> "torch.dtype":
    """Map a safetensors on-disk dtype string to its torch dtype, or raise.

    The five readers below used to fall back to `torch.float32` for any dtype
    missing from `_DTYPE_MAP` (`.get(meta['dtype'], torch.float32)`). That
    silently reinterpreted the tensor's bytes at the wrong element width and
    only surfaced as a confusing reshape error far from the real cause (e.g. a
    128-byte F8_E8M0 tensor read as 32 float32 elements instead of 128 uint8
    elements). CLAUDE.md §5: warn/raise, don't silently degrade.
    """
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise RuntimeError(
            f"Unmapped safetensors dtype {dtype_str!r} for tensor "
            f"{tensor_name!r} in {file_path!r} — add it to _DTYPE_MAP in "
            f"safetensor_model_persister.py"
        )
    return dtype

_ALIGN = 4096  # O_DIRECT sector-alignment requirement
_TENSOR_ALIGN = 16  # CUDA uint4/float4 vector-load width. EXL3's GEMV kernel
                    # (exl3_gemv_kernel.cuh) dereferences the weight buffer as
                    # uint4* for coalesced loads — any GPU tensor whose data_ptr()
                    # isn't a multiple of 16 faults with cudaErrorMisalignedAddress.
                    # A CUDA/kernel-contract constant, not a model number.

def _align_down(x: int) -> int:
    return x & ~(_ALIGN - 1)

def _align_up(x: int) -> int:
    return (x + _ALIGN - 1) & ~(_ALIGN - 1)

def _align_up_to(x: int, align: int) -> int:
    return (x + align - 1) & ~(align - 1)


def largest_pow2_at_most(n: int) -> int:
    """Greatest power of two <= n (0 for n <= 0).

    PyTorch's caching host allocator charges every pinned allocation the next
    power of two >= the request, and charges an exact power of two exactly
    (measured 2026-08-03, torch 2.11.0+cu130: 128 MiB -> 128 MiB, 129 MiB ->
    256 MiB). Sizing a pinned buffer with this keeps the charge equal to the
    request. Allocator behaviour, not a model number (CLAUDE.md §5).
    """
    n = int(n)
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def chunk_decomposition(n_slabs: int, slab_bytes: int) -> list[int]:
    """Split an arena of `n_slabs` into exact-power-of-two chunk sizes.

    The caching host allocator charges the NEXT power of two for every pinned
    request and charges an exact power of two exactly, so ONE backing tensor a
    hair over a power of two costs nearly double and the surplus is never
    addressable (measured: a 9.13 GB arena is charged 16 GiB). Several exact-pow2
    chunks are each charged exactly.

    A slab must not straddle a chunk boundary — `tensor()` would hand io_uring a
    span crossing two allocations — so each chunk contributes only
    `floor(chunk_bytes / slab_bytes)` slabs and its tail remainder is left
    unused. `slab_bytes` is _ALIGN-aligned but is not a power of two, so that
    remainder is unavoidable; measured at ~15 MB total across the whole arena.

    Largest chunk first: take the largest power of two that does not exceed what
    is still needed, then recurse on the remainder. When that candidate cannot
    hold even one slab (a model whose experts are larger than the remaining
    need), fall back to the smallest power of two that holds one.

    Returns [] for a non-positive slab count. Allocator behaviour, not a model
    number (CLAUDE.md §5).
    """
    n_slabs = int(n_slabs)
    slab_bytes = int(slab_bytes)
    if n_slabs <= 0 or slab_bytes <= 0:
        return []
    chunks: list[int] = []
    left = n_slabs
    while left > 0:
        need = left * slab_bytes
        chunk = largest_pow2_at_most(need)
        capacity = (chunk // slab_bytes) if chunk else 0
        if capacity == 0:
            # The remaining need is smaller than one slab's own power-of-two
            # footprint; take the smallest chunk that holds a slab.
            chunk = next_pow2_at_least(slab_bytes)
            capacity = chunk // slab_bytes
        chunks.append(chunk)
        left -= min(left, capacity)
    return chunks


def next_pow2_at_least(n: int) -> int:
    """Least power of two >= n (0 for n <= 0). This is exactly what the caching
    host allocator will charge for a pinned request of n bytes."""
    n = int(n)
    if n <= 0:
        return 0
    return 1 << (n - 1).bit_length()


def pinned_alloc_size(total_size: int) -> int:
    """Bytes to request for a pinned layer-staging buffer holding `total_size`
    bytes of tensor data.

    Requests the power of two the allocator charges anyway, so the whole block is
    usable instead of the surplus being dead pages. `2 * _ALIGN` of slack covers
    the O_DIRECT worst case: io_uring reads `_align_up(total_size + prefix_pad)`
    with `prefix_pad < _ALIGN`. This replaces a flat +100 MB reallocation pad —
    power-of-two sizing already steps by doubling, and that pad was the sole
    reason a ~0.1 MB shard was charged a 128 MiB block.
    """
    return next_pow2_at_least(int(total_size) + 2 * _ALIGN)


_USE_PINNED = torch.cuda.is_available()
if _USE_PINNED:
    print(">>>> pinned-memory loader active — direct NVMe→RAM→GPU path")
else:
    print(">>>> CUDA not available — using Python safetensors loader")

try:
    import vaayu_rust
    _USE_IOURING = True
    # One io_uring wave for all of an expert's byte groups (S22-A2). None on an
    # extension built before it existed: load_expert_into_slab then issues one
    # round trip per group, and init_model_io says so.
    _READ_SEGMENTS = getattr(vaayu_rust, "io_uring_read_segments", None)
    print(">>>> Rust io_uring reader loaded — O_DIRECT page-cache bypass active (QD32)")
except ImportError:
    _USE_IOURING = False
    _READ_SEGMENTS = None
    print(">>>> Rust io_uring NOT available — preadv QD16 fallback active")

# ── Global state for the loader ──────────────────────────────────────
_NUM_IO_THREADS = 16            # QD16 = sweet spot for Samsung 990 PRO (preadv fallback)
_EXECUTOR = None                # Lazily initialized ThreadPoolExecutor (preadv fallback)
_PINNED_BUFS = [None, None]     # Double buffer (A/B)
_PINNED_SIZES = [0, 0]          # Allocated size of each buffer
_BUF_IDX = 0                    # Toggles 0 → 1 → 0 → 1 …
_BUF_LOCK = __import__("threading").Lock()  # Guards _BUF_IDX toggle under concurrent prefetch

# Staging for WHOLE-LAYER fused expert reads, deliberately separate from the A/B
# pair above. See _staging_buffer for why sharing them corrupts a layer.
_FUSED_BUF = [None]
_FUSED_SIZE = [0]
_FUSED_LOCK = threading.Lock()

import queue as _queue
_EXPERT_BUF_POOL: "_queue.Queue[torch.Tensor]" = _queue.Queue()
_EXPERT_PINNED_SIZE = 0
_EXPERT_POOL_N = 8  # cap on pool size; actual size is RAM-budgeted, see init_model_io.
                    # Tried raising to 32 (2026-07-18, matching the ring's QD32) to test
                    # whether the pool cap was starving the ring: on-device A/B showed
                    # n=16 (RAM-budget-limited before hitting 32) made TTFT WORSE
                    # (24.9s->28.9s total, 27.3s->31.4s TTFT) — reverted. Root cause of
                    # the prefill disk/compute serialization is NOT pool size; see
                    # moe_perf_rootcause memory for the live investigation.
_EXPERT_POOL_ACTUAL_N = 1  # actual buffer count allocated, set by init_model_io

# HARDWARE constant (CLAUDE.md §5) — the NVMe's READ-CONCURRENCY knee, not a
# model number. Samsung SSD 990 PRO 1TB, measured 2026-09-03 through THIS path
# (plans/probes/p8_samepath.py: real expert byte ranges, O_DIRECT, registered
# fds, pinned buffers — the only difference from decode is that nothing computes
# between reads). Aggregate bandwidth is FLAT at ~7.09 GB/s for 2-4 rings at
# every caller concurrency k from 2 to 16, and collapses to ~5.15 GB/s at 8
# rings — reproduced on both models (Qwen 11.80 MB reads, MiniMax 8.85 MB) and
# independent of the ring queue depth (identical curve at qd=1,2,4,8,16,32) and
# of file locality (identical for scattered and file-adjacent reads).
#
# The ring pool is what actually throttles DEVICE concurrency: a caller queues on
# a ring's Mutex. Sizing it to get_expert_pool_size() therefore over-drove the
# drive past its knee, and was the ~5.12 GB/s aggregate ceiling row P8 was opened
# to explain. Caller-side concurrency (buffer pool, MoE executor width, in-flight
# staging slabs) is a RAM budget and stays at get_expert_pool_size().
#
# 2 NOT 3, corrected 2026-09-21 (row 49). P8's standalone probe reported that
# flat 2-4 zone and this constant took 3 from the middle of it; IN PATH the zone
# is NOT flat and never reaches 7.09 GB/s. Swept through real decode with a live
# model (MiniMax-M2.7, ABBA, greedy ids identical at every n): delivered-while-
# busy 6.08 (n=2) / 6.03 (n=3) / 5.90 (n=4) GB/s, and n=1 collapses to 5.50.
# Throughput vs n=3: n=2 +1.89 % decode and +1.85 % on a 2k prefill, n=4
# -2.07 %, n=1 -5.70 %. Null on Qwen3-Next (its disk is 0.42 % of a decode
# token) and on Llama-3.3-70B dense, both inside their own arm spread. The
# difference from the probe is everything it left out: 8 loader threads, the GPU
# stream and the expert-phase host work all contending inside a real token.
# docs/2026-09-21-row49-ringpool-AB-RESULTS.md.
_NVME_READ_CONCURRENCY = 2

_RING_INITIALIZED = False   # True after vaayu_rust.init_ring_pool() succeeds
_FILES_REGISTERED = False   # True after register_files() succeeds
_BUFREG_DISABLED = False    # True after first EFAULT — CUDA-pinned memory is
                            # incompatible with IORING_REGISTER_BUFFERS (the kernel's
                            # get_user_pages_fast() fails on NVIDIA-managed pages).
                            # Reads still work via IORING_OP_READ; only the minor
                            # IORING_OP_READ_FIXED zero-copy path is lost.

_LAST_PINNED_TIMINGS: dict = {}
_PINNED_TIMINGS_LOCK = threading.Lock()

_USE_DIRECT_IO: bool = True  # safe default; overridden by init_model_io() (Plan 5)

# Total bytes this process has REQUESTED of the NVMe, for the token
# summary's gb_tok= field and the server's requests.jsonl. Counted at read
# submission (aligned lengths), not at completion.
#
# MONOTONIC ON PURPOSE -- there is deliberately no reset. Two consumers take
# different windows over this one counter: vaayu_base subtracts a baseline it
# recorded at the top of a forward (per TOKEN), and the server subtracts one it
# recorded before generate() (per REQUEST). A counter that reset itself per
# token would leave the server reporting a fraction of one token as a whole
# request's disk traffic.
#
# Counted HERE, inside the functions that actually issue the reads, rather than
# at their callers: every caller-side cache (LoaderStrategy._cache,
# _non_expert_tensor_cache, ExpertRAMCache) returns before reaching this
# module, so a cache hit contributes zero by construction instead of by a flag
# someone has to remember to set.
#
# Note: a read that raises before completing is still counted. This is
# acceptable because such paths abort the process.
#
# Locked because these run on ThreadPoolExecutor workers and `+=` on a module
# global is a load-add-store, not an atomic. One uncontended acquisition per
# read (~190/token) is nothing beside an 11.9 MB O_DIRECT read.
_read_bytes = 0
_read_bytes_lock = threading.Lock()


def note_read_bytes(n: int) -> None:
    """Record `n` bytes requested of the drive (aligned read length at submission)."""
    global _read_bytes
    with _read_bytes_lock:
        _read_bytes += n


def read_bytes() -> int:
    """Total checkpoint bytes requested of the drive so far (aligned lengths, at
    submission). Monotonic; never reset."""
    with _read_bytes_lock:
        return _read_bytes


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


def _init_iouring_ring(pool_size: int, queue_depth: int = 32) -> bool:
    try:
        n = pool_size
        vaayu_rust.init_ring_pool(n, queue_depth)
        print(f"[VAAYU-IO] persistent ring pool initialized  n={n}  qd={queue_depth}")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] init_ring_pool failed: {e} — per-call ring fallback active")
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


def _warn_if_stale_extension() -> None:
    """Say so, once per load, when the installed vaayu_rust predates the
    one-wave expert read: nothing breaks, but every expert costs one disk round
    trip per byte group instead of one in total."""
    if _USE_IOURING and _READ_SEGMENTS is None:
        print("[VAAYU-IO] WARNING: the installed vaayu_rust has no io_uring_read_segments — "
              "expert reads issue one round trip per byte group. Rebuild it: "
              "`maturin develop --release` in vaayu_rust/.")


def init_model_io(model_dir: str, use_direct_io: bool) -> None:
    """Called once at model load. Initializes ring pool and registers files explicitly.

    Buffer-pool sizing runs BEFORE ring init: the ring pool is capped at
    get_expert_pool_size(), since a ring no caller can reach is dead weight, so
    that number must be known first. It is otherwise sized to the DRIVE — see
    _NVME_READ_CONCURRENCY for why matching the caller pool was the P8 ceiling.
    """
    global _RING_INITIALIZED, _FILES_REGISTERED, _USE_DIRECT_IO
    _USE_DIRECT_IO = use_direct_io
    _warn_if_stale_extension()

    global _EXPERT_PINNED_SIZE, _EXPERT_POOL_ACTUAL_N
    if _USE_PINNED and _EXPERT_BUF_POOL.empty():
        # Buffer must fit the largest single contiguous read group per expert, not
        # a flat guess: it is the checkpoint's largest aligned expert span, derived
        # from its offset index (CLAUDE.md §5). The guesses it replaces -- 32 MB
        # EXL3, 512 MB anything else -- gave every non-EXL3 checkpoint ONE buffer
        # on this box's 0.5 GB budget, so one loader worker and one ring: every
        # Qwen3-Next expert read (~1.8 MB) ran serially, 66 % of a 2k prefill
        # (row 44, 2026-09-19: 86.3 -> 57.1 s with a full-width pool). A
        # checkpoint with no per-expert tensors never calls the expert loader and
        # keeps the old sizing, so its ring count does not move.
        _EXPERT_PINNED_SIZE = expert_read_buffer_bytes(model_dir)
        if _EXPERT_PINNED_SIZE == 0:
            try:
                from .loader_strategy import _is_exl3_checkpoint
                _is_exl3 = _is_exl3_checkpoint(model_dir)
            except Exception:
                _is_exl3 = False
            _EXPERT_PINNED_SIZE = (32 if _is_exl3 else 512) * 1024 * 1024

        # Budget pool size against available RAM. Reserve 27 GB for: layer load
        # double-buffer (~6 GB), expert RAM cache (~16.5 GB), non-expert tensor
        # cache (~2.5 GB), CUDA/Python overhead (~2 GB). Pool gets the remainder,
        # capped at _EXPERT_POOL_N. On tight machines this yields 1-2 buffers
        # (still parallel vs the old single-locked buffer); on 64+ GB machines 8.
        _avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        _pool_budget_gb = max(0.5, _avail_gb - 27.0)
        _buf_mb = _EXPERT_PINNED_SIZE / (1024 * 1024)
        _n_bufs = max(1, min(_EXPERT_POOL_N, int(_pool_budget_gb * 1024 / _buf_mb)))

        for _ in range(_n_bufs):
            _EXPERT_BUF_POOL.put(
                torch.empty(_EXPERT_PINNED_SIZE, dtype=torch.uint8, pin_memory=True)
            )
        _EXPERT_POOL_ACTUAL_N = _n_bufs
        total_mb = _n_bufs * _EXPERT_PINNED_SIZE / (1024 ** 2)
        print(
            f"[VAAYU-IO] alloc expert buffer pool"
            f"  n={_n_bufs}  size={_EXPERT_PINNED_SIZE / (1024**2):.1f}MB each"
            f"  total={total_mb:.1f}MB  (avail={_avail_gb:.1f}GB)"
        )

    if _USE_IOURING and not _RING_INITIALIZED:
        _RING_INITIALIZED = _init_iouring_ring(_ring_pool_size(), 32)
    if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
        _FILES_REGISTERED = _register_iouring_files(model_dir, use_direct_io)


def get_expert_pool_size() -> int:
    """Actual pinned-buffer pool size (RAM-budgeted, may be < _EXPERT_POOL_N).

    Concurrent load_expert_sparse() callers beyond this count block on
    _EXPERT_BUF_POOL.get() rather than gaining real I/O parallelism — callers
    doing concurrent dispatch (e.g. a ThreadPoolExecutor) should cap their
    worker count at this value.
    """
    return max(1, _EXPERT_POOL_ACTUAL_N)


def _ring_pool_size() -> int:
    """How many io_uring rings to build = how many reads the DRIVE sees at once.

    Sized to _NVME_READ_CONCURRENCY (a hardware constant), capped at the caller
    pool so a RAM-tight box does not build rings nothing can reach.
    `VAAYU_RING_POOL_N` overrides it — the knob P8's A/B arms were run with, and
    the way to reproduce the pre-fix behaviour (`VAAYU_RING_POOL_N=8`).
    """
    n = _NVME_READ_CONCURRENCY
    raw = os.environ.get("VAAYU_RING_POOL_N")
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = _NVME_READ_CONCURRENCY
        if n < 1:
            n = _NVME_READ_CONCURRENCY
    return max(1, min(n, get_expert_pool_size()))


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

    if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < pinned_alloc_size(total_size):
        alloc_size = pinned_alloc_size(total_size)
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

    _bytes_read_this_call = 0
    if _USE_IOURING:
        # ── 3a. Rust io_uring read (zero-copy into pinned buffer) ───
        aligned_offset = _align_down(data_start)
        prefix_pad = data_start - aligned_offset
        aligned_length = _align_up(total_size + prefix_pad)
        _bytes_read_this_call = aligned_length
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
            _bytes_read_this_call = total_size
        finally:
            os.close(fd)

    # ── 4. Reconstruct tensors as zero-copy views ───────────────────
    _t_recon = time.perf_counter()
    result = {}
    source_buf = buf_view if _USE_IOURING else buf
    for name, meta in tensors_meta.items():
        dtype = _resolve_dtype(meta['dtype'], name, file_path)
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

    note_read_bytes(_bytes_read_this_call)
    return result


def non_expert_ranges(file_path: str, layer_name: str, offset_index) -> list[tuple]:
    """[(key, hdr_meta, abs_file_offset, byte_len)] for a MoE layer shard's
    NON-expert tensors -- exactly the set `load_non_expert_tensors` reads, from
    the header alone (S14 P sizes a tier entry with it before deciding to read)."""
    with open(file_path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    data_start = 8 + hdr_len

    # Collect disk-format expert tensor keys from offset_index
    expert_keys: set[str] = set()
    layer_experts = offset_index.index.get(layer_name, {})
    for _eidx, expert_metas in layer_experts.items():
        expert_keys.update(expert_metas.keys())

    # Build list of non-expert (key, hdr_meta, abs_file_offset, byte_len)
    non_expert: list[tuple] = []
    for key, meta in hdr.items():
        if key == '__metadata__':
            continue
        if key in expert_keys:
            continue
        start, end = meta['data_offsets']
        non_expert.append((key, meta, data_start + start, end - start))
    return non_expert


def load_non_expert_tensors(
    file_path: str,
    layer_name: str,
    offset_index,
) -> dict:
    """
    Load only non-expert tensors from a MoE layer shard via targeted byte-range reads.

    Skips all expert tensor byte ranges in the file, reading only attention weights,
    norms, and the router gate (~170 MB instead of 2.77 GB per Mixtral layer).

    Args:
        file_path: absolute path to the .safetensors shard
        layer_name: shard stem used as key in offset_index (e.g. "model.layers.3")
        offset_index: ExpertOffsetIndex instance (or None → falls back to full load)

    Returns:
        {tensor_key: cpu_tensor} for non-expert tensors only
    """
    if offset_index is None:
        return _load_pinned(file_path)

    non_expert = non_expert_ranges(file_path, layer_name, offset_index)

    if not non_expert:
        return {}

    result = _read_grouped_ranges(file_path, non_expert)
    size_mb = sum(v.nbytes for v in result.values()) / (1024**2)
    print(f"[VAAYU-IO] non-expert  {size_mb:.1f}MB  {layer_name}")
    return result


def load_fused_layer_tensors(file_path: str, metas: dict) -> dict:
    """Load WHOLE fused expert tensors -- one blob per projection, every expert
    of the layer inside it -- for a kernel that routes internally.

    Deliberately not `load_expert_sparse`: that reader serves one expert per
    call out of a pinned pool sized to the largest single expert (13.27 MB on
    gpt-oss-20b), and a whole fused layer is 423 MB. This path takes the same
    grouped reader the non-expert tensors use, whose buffer grows to the read.

    `metas` is {tensor_key: ExpertTensorMeta}, i.e. ExpertOffsetIndex's
    `fused_layer_tensors[layer_name]`.
    """
    if not metas:
        return {}
    items = [(key, {"dtype": m.dtype, "shape": m.shape}, m.file_offset, m.byte_length)
             for key, m in metas.items()]
    return _read_grouped_ranges(file_path, items, dedicated=True)


def release_fused_staging_buffer() -> int:
    """Free the dedicated fused-layer staging buffer; returns the bytes released.

    The routed-delivery self-check (vaayu/routed_fused3d.py) reads one whole layer
    through `load_fused_layer_tensors`, and on a routed install nothing reads that
    way again -- so the buffer (next power of two above one fused layer, 512 MiB on
    gpt-oss-20b) would stay pinned, unswappable and unread: 47-m5's bug class. The
    whole-layer path re-allocates it lazily on its next visit, so this is always
    safe. Dropping the reference is not enough on its own: the host caching
    allocator keeps a freed pinned block, so it is emptied too. Idempotent.
    """
    with _FUSED_LOCK:
        released = _FUSED_SIZE[0] if _FUSED_BUF[0] is not None else 0
        _FUSED_BUF[0] = None
        _FUSED_SIZE[0] = 0
        if released and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    if released:
        print(f"[VAAYU-IO] free   fused-layer staging {released / (1024**2):.1f}MB")
    return released


@contextmanager
def _staging_buffer(aligned_length: int, dedicated: bool):
    """Yield a pinned staging buffer large enough for one grouped read.

    The A/B pair alternates between exactly TWO users -- the layer prefetch
    thread and the main thread -- and that alternation IS the safety argument:
    each clones out of its own buffer while the other writes into the other one.
    A THIRD concurrent user silently breaks it, and a whole-layer fused expert
    read is exactly that: it runs on the main thread inside a layer's forward,
    while the prefetcher is already reading the NEXT layer's non-expert tensors.
    Measured on gpt-oss-20b 2026-09-21: the model's output went NaN from layer 10
    -- the second STREAMED layer -- deterministically, because layer 9's 423 MB
    expert read landed in the buffer layer 10's attention weights were being
    cloned out of. The split file was byte-perfect; the corruption was in RAM.

    So that reader gets its own buffer, and the lock is held across the read AND
    the clone, not just the index toggle.
    """
    global _BUF_IDX
    if dedicated:
        with _FUSED_LOCK:
            if _FUSED_BUF[0] is None or _FUSED_SIZE[0] < aligned_length:
                alloc_size = next_pow2_at_least(aligned_length)
                _FUSED_BUF[0] = torch.empty(alloc_size, dtype=torch.uint8, pin_memory=True)
                _FUSED_SIZE[0] = alloc_size
                print(f"[VAAYU-IO] alloc  fused-layer staging "
                      f"{alloc_size / (1024**2):.1f}MB")
            yield _FUSED_BUF[0]
        return

    with _BUF_LOCK:
        buf_idx = _BUF_IDX
        _BUF_IDX = 1 - _BUF_IDX
    if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < aligned_length:
        # Same policy as pinned_alloc_size: ask for the power of two the
        # caching host allocator charges anyway, so the whole block is
        # usable. No O_DIRECT slack on top -- unlike the expert path's
        # total_size, aligned_length is already _align_up(read_len +
        # prefix_pad), i.e. exactly what io_uring writes below.
        alloc_size = next_pow2_at_least(aligned_length)
        _PINNED_BUFS[buf_idx] = torch.empty(alloc_size, dtype=torch.uint8, pin_memory=True)
        _PINNED_SIZES[buf_idx] = alloc_size
        print(f"[VAAYU-IO] alloc  buf={buf_idx}  size={alloc_size / (1024**2):.1f}MB (non-expert)")
    yield _PINNED_BUFS[buf_idx]


def _read_grouped_ranges(file_path: str, items: list, dedicated: bool = False) -> dict:
    """Read a set of byte ranges out of one shard into CPU tensors.

    `items` is [(key, {'dtype','shape'}, absolute_file_offset, byte_length)].
    Adjacent ranges (gap <= _ALIGN) are coalesced into one io_uring call, and
    the pinned staging area grows to the largest group. Returns {key: cloned cpu
    tensor}, safe to hold after the buffer is reused.

    `dedicated=True` stages through a buffer of this reader's own instead of the
    shared A/B pair -- required for any caller that can run CONCURRENTLY with the
    layer prefetcher (see _staging_buffer).
    """
    if not items:
        return {}
    non_expert = sorted(items, key=lambda x: x[2])

    # Group contiguous ranges (gap <= _ALIGN) to minimise io_uring calls
    groups: list[list] = []
    current_group = [non_expert[0]]
    for item in non_expert[1:]:
        prev_end = current_group[-1][2] + current_group[-1][3]
        if item[2] - prev_end <= _ALIGN:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
    groups.append(current_group)

    result: dict = {}

    # ONE staging slot for the whole call, sized to the largest group -- not one
    # per group. The A/B pair's safety argument is that a caller holds its slot
    # while the OTHER caller writes the other one; taking a fresh slot per group
    # cycles both within a single call and overwrites what the other caller is
    # still holding. gpt-oss-20b is where that surfaced: its layer files
    # interleave the small non-expert tensors between six ~70 MB expert blobs, so
    # one non-expert read is a dozen groups, and it clobbered the 1.16 GB
    # embedding the main thread had read but not yet injected (measured
    # 2026-09-21: absmax 118.5 at read, 1.36e38 at inject, logits garbage).
    # Every checkpoint whose non-expert tensors sit contiguously -- Qwen3-Next,
    # MiniMax -- is one group and never saw it.
    _plan = []
    for group in groups:
        group_start = group[0][2]
        group_end = group[-1][2] + group[-1][3]
        aligned_offset = _align_down(group_start)
        prefix_pad = group_start - aligned_offset
        aligned_length = _align_up(group_end - group_start + prefix_pad)
        _plan.append((group, group_start, aligned_offset, prefix_pad, aligned_length))

    with _staging_buffer(max(p[4] for p in _plan), dedicated) as buf:
        for group, group_start, aligned_offset, prefix_pad, aligned_length in _plan:
            note_read_bytes(aligned_length)
            if _USE_IOURING:
                vaayu_rust.io_uring_read_into(
                    file_path, aligned_offset, aligned_length,
                    buf.data_ptr(), 8, _USE_DIRECT_IO,
                )
            else:
                fd = os.open(file_path, os.O_RDONLY)
                try:
                    view = memoryview(buf.numpy())
                    os.preadv(fd, [view[:aligned_length]], aligned_offset)
                finally:
                    os.close(fd)
            data_buf = buf[prefix_pad:]

            for key, meta, file_offset, byte_len in group:
                rel_start = file_offset - group_start
                rel_end = rel_start + byte_len
                dtype = _resolve_dtype(meta['dtype'], key, file_path)
                tensor = data_buf[rel_start:rel_end].view(dtype).reshape(meta['shape'])
                result[key] = tensor.clone()

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
            note_read_bytes(sum(v.nbytes for v in result.values()))
            gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
            print(f"[VAAYU-IO] python  {elapsed:.3f}s  {size_mb:.1f}MB  {gbps:.2f}GB/s  {layer_name}")
            return result


def _group_contiguous_ranges(expert_metas: dict) -> list:
    """Sort one expert's tensor metas by file offset and group ranges whose
    gap is <= _ALIGN into single contiguous reads. Checkpoints may store
    tensors grouped by weight-type across ALL experts rather than per-expert
    (confirmed: this EXL3 checkpoint stores every expert's small suh/svh
    tensors in one block, then every expert's large trellis tensors in a
    separate block far away) — multiple groups with large gaps between them
    is the expected case, not an edge case. Never assume one expert's
    tensors are file-contiguous."""
    sorted_metas = sorted(expert_metas.items(), key=lambda x: x[1].file_offset)
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
    return groups


def load_expert_sparse(
    file_path: str,
    expert_metas: dict,
    use_direct_io: bool | None = None,
) -> dict:
    """
    Load only the specified expert tensor byte ranges from a safetensors shard.

    Groups contiguous byte ranges (gap ≤ _ALIGN) into a single io_uring call
    to minimise syscall overhead. Falls back to preadv when io_uring unavailable.

    Uses a pool of pinned buffers (_EXPERT_BUF_POOL) — each concurrent caller
    gets its own pinned buffer via blocking queue.get(). Pool size = max_workers
    in the ThreadPoolExecutor that calls this, so no starvation. Pinned memory
    ensures O_DIRECT-compatible alignment and fast PCIe DMA to GPU.

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

    groups = _group_contiguous_ranges(expert_metas)

    result = {}

    for group in groups:
        group_start = group[0][1].file_offset
        last_meta = group[-1][1]
        group_end = last_meta.file_offset + last_meta.byte_length
        read_len = group_end - group_start

        aligned_offset = _align_down(group_start)
        prefix_pad = group_start - aligned_offset
        aligned_length = _align_up(read_len + prefix_pad)
        note_read_bytes(aligned_length)     # what the read below actually asks for

        # Acquire a pinned buffer from the pool (blocks until one is available).
        # Pool has _EXPERT_POOL_N buffers = max_workers, so no starvation.
        # Falls back to unpinned + buffered IO only if pool was never initialized.
        _pool_buf = None
        if _USE_PINNED and _EXPERT_PINNED_SIZE > 0:
            _pool_buf = _EXPERT_BUF_POOL.get()
            if aligned_length > _EXPERT_PINNED_SIZE:
                _EXPERT_BUF_POOL.put(_pool_buf)
                raise RuntimeError(
                    f"Expert size {aligned_length} exceeds buffer size {_EXPERT_PINNED_SIZE}"
                )
            buf = _pool_buf[:aligned_length]
        else:
            # Pool not initialized (CUDA unavailable or init_model_io not called yet).
            # torch.empty is malloc-aligned (16 bytes), not 512-byte sector-aligned.
            # O_DIRECT requires sector-aligned addresses → force buffered IO.
            buf = torch.empty(aligned_length, dtype=torch.uint8)
            use_direct_io = False

        try:
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
                shape = meta.shape
                dt = _resolve_dtype(meta.dtype, key, file_path)

                tensor = data_buf[rel_start:rel_end].view(dt).view(shape)
                result[key] = tensor.clone()

        finally:
            if _pool_buf is not None:
                _EXPERT_BUF_POOL.put(_pool_buf)

    return result


def _group_layout(group: list) -> tuple[int, int, int, int, int]:
    """Shared sizing/reading arithmetic for one contiguous tensor group —
    used by BOTH expert_span_bytes (sizing) and load_expert_into_slab
    (reading), so they can't drift apart the way sizing/reading arithmetic
    once did in this codebase (see commit ef24551).

    load_expert_into_slab has TWO write modes (see its docstring for why):
    `tensor_align=True` individually re-packs each tensor onto a
    _TENSOR_ALIGN boundary in a transient scratch buffer, writing only
    `packed_span` bytes into the persistent slab; `tensor_align=False`
    writes the raw group read directly into the persistent slab, needing
    `raw_aligned_length` bytes there. **Every production caller passes
    `tensor_align=False`** — the repacking mode is exercised only by
    `tests/` (see load_expert_into_slab's docstring). Sizing still covers
    BOTH modes: the same arena slabs back both callers, the mode is a
    per-call argument, and under-sizing for a mode a future caller turns
    on would overflow the slab silently,
    so `slab_span = max(raw_aligned_length, packed_span)` and both
    functions always advance by `slab_span` regardless of which mode a
    given call actually uses.

    Returns (aligned_offset, prefix_pad, raw_aligned_length, packed_span, slab_span):
      aligned_offset     — file offset to start the O_DIRECT read from
                            (a multiple of _ALIGN).
      prefix_pad         — bytes between aligned_offset and the group's
                            real first byte within the raw read.
      raw_aligned_length — bytes read from disk for this group (O_DIRECT
                            sector-aligned length).
      packed_span         — bytes this group occupies once each tensor is
                            individually re-packed at a _TENSOR_ALIGN
                            boundary (16 bytes) — no _ALIGN (4096-byte)
                            rounding needed, since this layout is never an
                            O_DIRECT read target itself.
      slab_span           — max(raw_aligned_length, packed_span): what
                            BOTH expert_span_bytes and load_expert_into_slab
                            (in either mode) must reserve/advance per group.
    """
    group_start = group[0][1].file_offset
    last_meta = group[-1][1]
    group_end = last_meta.file_offset + last_meta.byte_length
    aligned_offset = _align_down(group_start)
    prefix_pad = group_start - aligned_offset
    raw_aligned_length = _align_up(group_end - group_start + prefix_pad)
    packed_span = sum(_align_up_to(meta.byte_length, _TENSOR_ALIGN) for _key, meta in group)
    slab_span = max(raw_aligned_length, packed_span)
    return aligned_offset, prefix_pad, raw_aligned_length, packed_span, slab_span


def expert_span_bytes(expert_metas: dict) -> int:
    """Byte length the PERSISTENT slab needs to hold ALL of one expert's
    tensors, covering EITHER of load_expert_into_slab's two write modes
    (see _group_layout) — NOT the transient scratch buffer's read length
    when tensor_align=True (that never counts against slab/cache capacity).
    Expert tensors are NOT assumed file-contiguous (see
    _group_contiguous_ranges) — this is the SUM of each contiguous group's
    own slab_span, not the raw span from the first tensor's start to the
    last tensor's end (that span can be enormous when a checkpoint stores
    tensors grouped by weight-type across many experts rather than
    per-expert, e.g. a 128-expert checkpoint whose small metadata tensors
    and large trellis tensors live in two far-apart blocks). Model-agnostic:
    derived from the offset index, never hardcoded."""
    total = 0
    for group in _group_contiguous_ranges(expert_metas):
        _, _, _, _, slab_span = _group_layout(group)
        total += slab_span
    return total


def expert_read_buffer_bytes(model_dir: str) -> int:
    """Bytes one expert-loader buffer needs: the checkpoint's largest aligned
    expert span (`compute_slab_bytes`), read off its expert offset index.
    0 when the checkpoint has no per-expert tensors.

    The cached index is used when present; otherwise the headers are scanned
    WITHOUT writing a cache, so a checkpoint with no experts (dense) gains no
    file. The engine's own build_or_load() writes it for MoE checkpoints."""
    from .expert_offset_index import ExpertOffsetIndex
    idx = ExpertOffsetIndex(model_dir)
    if not idx.load_cached():
        idx.build(write_cache=False)
    return compute_slab_bytes(idx) if idx.index else 0


def compute_slab_bytes(offset_index) -> int:
    """Max aligned expert span across the whole model — every slab must fit the
    largest expert. Scans the in-RAM offset index (cheap: ~10k dict entries)."""
    worst = 0
    for layer_experts in offset_index.index.values():
        for expert_metas in layer_experts.values():
            worst = max(worst, expert_span_bytes(expert_metas))
    # H2: slab starts are multiples of slab_bytes from an aligned chunk base, so
    # SlabArena's O_DIRECT premise needs slab_bytes itself _ALIGN-aligned. It is
    # NOT automatic: _group_layout returns max(raw_aligned_length, packed_span),
    # and packed_span only rounds each tensor to _TENSOR_ALIGN (16) -- a group of
    # many small tensors makes it the larger of the two and 16-aligned is not
    # 4096-aligned. Costs at most _ALIGN-1 bytes per slab.
    return _align_up(worst)


def load_expert_into_slab(
    file_path: str,
    expert_metas: dict,
    slab: "torch.Tensor",
    allow_direct: bool,
    tensor_align: bool,
) -> dict:
    """Read one expert's tensors into `slab` (uint8) and return
    {tensor_key: view-into-slab}. No clone in the cache (see
    ExpertRAMCache.put_slab). Expert tensors are NOT assumed file-contiguous
    (see _group_contiguous_ranges) — this issues ONE read per contiguous
    group; the slab's layout does not mirror the file's layout (slab_bytes
    equals the sum of each group's own slab_span, per expert_span_bytes /
    compute_slab_bytes / _group_layout).

    tensor_align selects one of two write modes.

    **NO PRODUCTION CALLER PASSES tensor_align=True** (verified 2026-09-03,
    row P6). All three — stage_read_expert (vaayu_base.py), and both sites
    in moe_patch.py — pass False; the True branch below is live only in
    tests/. It stopped being the prefill stager's mode in commit c5014cd
    (2026-07-13, "zero-copy prefill staging + GPU-side 16-byte alignment",
    bug B1): the stager now reads RAW like decode does, and the 16-byte
    alignment EXL3's uint4* kernel needs is applied on the GPU instead, by
    gpu_repack_aligned in moe_patch.py's staged-consume path. This
    docstring claimed otherwise for six weeks and is what produced (and
    wasted) perf row R4/P6 — do not re-stale it. The branch is kept because
    the two modes' sizing is shared (see _group_layout) and four tests
    assert its behaviour.

    The mode exists at all because only ONE caller ever needed the stronger
    alignment guarantee, and paying for it unconditionally regressed decode
    throughput in real-hardware testing (2026-07-12: avg decode token time
    rose from ~1.0-1.2s to ~1.56s after tensor_align was applied
    unconditionally, from added _EXPERT_BUF_POOL contention + per-tensor
    copy_ overhead on decode's cache-miss path, which is hit 350-400x/token):

    - tensor_align=True (TEST-ONLY since 2026-07-13; formerly the prefill
      stager, via stage_read_expert): the raw group read lands in a
      TRANSIENT scratch buffer drawn from
      _EXPERT_BUF_POOL (the same pinned-buffer pool load_expert_sparse
      already uses, bounded by the shared _moe_executor's worker count — no
      oversubscription), and each tensor is individually re-packed (one
      scratch→slab copy_, no clone — the two buffers are always disjoint)
      at a _TENSOR_ALIGN-aligned offset in `slab`. A group's raw read only
      guarantees each tensor is itemsize-aligned (safe for .view(dtype)),
      not _TENSOR_ALIGN-aligned, since that position is derived from the
      file's own byte layout and is essentially arbitrary mod
      _TENSOR_ALIGN — invisible on CPU, but until B1 (2026-07-13) Task 9's
      one-copy GPU rebuild (views_from_packed_copy) resliced these exact
      views straight into a CUDA tensor for EXL3's kernel, which
      dereferences the weight buffer as uint4* — a misaligned slice there
      faults with cudaErrorMisalignedAddress. That is what this mode
      existed for; gpu_repack_aligned now stands between the two and does
      the aligning on the GPU. Keeping the raw region OUT of `slab`
      matters because `slab` is a long-lived, cache-capacity-counted
      buffer, while the raw region is only needed for this one read.

    - tensor_align=False (EVERY production caller: decode's cache-miss path
      AND, since 2026-07-13, the prefill stager): the raw group read lands
      DIRECTLY in `slab` at its natural file-relative offset — zero extra
      copies, no scratch pool. Views into it are itemsize-aligned only, NOT
      _TENSOR_ALIGN-aligned, so they are safe as copy SOURCES but not as
      EXL3 kernel inputs. Both consumers respect that: decode's cached
      experts go through per-component `.to(device)` copies in the compute
      loop (Task 7), which always allocate a fresh, naturally aligned GPU
      destination; the prefill stager's slabs go through
      views_from_packed_copy (which preserves the raw, unaligned offsets)
      and then gpu_repack_aligned, which is what re-packs onto 16-byte CUDA
      offsets for the kernel.

    Either mode advances by the SAME per-group `slab_span` (see
    _group_layout) so the two modes can safely share the same arena/slab
    sizing regardless of which mode any given call uses. The returned
    views are valid until the slab is recycled by cache eviction —
    consumers must use them within the same layer forward.

    allow_direct: whether this call may use O_DIRECT for the read (subject,
    in tensor_align=True mode, to the scratch buffer actually being
    pinned — see below).
    """
    use_direct = _USE_DIRECT_IO and allow_direct
    out: dict = {}
    slab_off = 0
    if not tensor_align and _READ_SEGMENTS is not None:
        # One io_uring wave for the whole expert: every group's read is issued
        # together, on one ring, instead of one round trip per group. Same slab
        # layout, same views, same bytes counted as the loop below.
        base = slab.data_ptr()
        segments = []
        layouts = []
        for group in _group_contiguous_ranges(expert_metas):
            aligned_offset, prefix_pad, raw_aligned_length, _packed_span, slab_span = _group_layout(group)
            if slab_off + slab_span > slab.nbytes:
                raise RuntimeError(
                    f"expert span {slab_off + slab_span}B exceeds slab {slab.nbytes}B "
                    f"(compute_slab_bytes undersized?)"
                )
            segments.append((aligned_offset, raw_aligned_length, base + slab_off))
            layouts.append((group, slab_off, prefix_pad, raw_aligned_length))
            slab_off += slab_span
        for _aligned_offset, raw_aligned_length, _dst in segments:
            note_read_bytes(raw_aligned_length)
        _READ_SEGMENTS(file_path, segments, 8, use_direct)
        for group, group_off, prefix_pad, raw_aligned_length in layouts:
            group_start = group[0][1].file_offset
            data = slab[group_off:group_off + raw_aligned_length][prefix_pad:]
            for key, meta in group:
                rel = meta.file_offset - group_start
                dt = _resolve_dtype(meta.dtype, key, file_path)
                out[key] = data[rel:rel + meta.byte_length].view(dt).view(meta.shape)
        return out
    for group in _group_contiguous_ranges(expert_metas):
        group_start = group[0][1].file_offset
        aligned_offset, prefix_pad, raw_aligned_length, packed_span, slab_span = _group_layout(group)
        note_read_bytes(raw_aligned_length)  # both write modes read this much
        if slab_off + slab_span > slab.nbytes:
            raise RuntimeError(
                f"expert span {slab_off + slab_span}B exceeds slab {slab.nbytes}B "
                f"(compute_slab_bytes undersized?)"
            )

        if not tensor_align:
            # Zero-copy: raw read lands directly in `slab` at its natural
            # file-relative offset. No scratch pool, no repacking.
            dest = slab[slab_off:slab_off + raw_aligned_length]
            if _USE_IOURING:
                vaayu_rust.io_uring_read_into(
                    file_path, aligned_offset, raw_aligned_length,
                    dest.data_ptr(), 8, use_direct,
                )
            else:
                fd = os.open(file_path, os.O_RDONLY)
                try:
                    view = memoryview(dest.numpy())
                    os.preadv(fd, [view[:raw_aligned_length]], aligned_offset)
                finally:
                    os.close(fd)
            data = dest[prefix_pad:]
            for key, meta in group:
                rel = meta.file_offset - group_start
                dt = _resolve_dtype(meta.dtype, key, file_path)
                out[key] = data[rel:rel + meta.byte_length].view(dt).view(meta.shape)
            slab_off += slab_span
            continue

        # Raw read into a transient scratch buffer (same pool as
        # load_expert_sparse). Pool exhausted/uninitialized falls back to
        # plain (non-pinned) torch.empty, forcing this read's use_direct
        # off — O_DIRECT requires a sector-aligned destination, which only
        # the pinned pool buffers guarantee (mirrors load_expert_sparse).
        _pool_buf = None
        _use_direct_this_group = use_direct
        if _USE_PINNED and _EXPERT_PINNED_SIZE > 0:
            _pool_buf = _EXPERT_BUF_POOL.get()
            if raw_aligned_length > _EXPERT_PINNED_SIZE:
                _EXPERT_BUF_POOL.put(_pool_buf)
                raise RuntimeError(
                    f"Expert group size {raw_aligned_length} exceeds scratch "
                    f"buffer size {_EXPERT_PINNED_SIZE}"
                )
            raw = _pool_buf[:raw_aligned_length]
        else:
            raw = torch.empty(raw_aligned_length, dtype=torch.uint8)
            _use_direct_this_group = False

        try:
            if _USE_IOURING:
                vaayu_rust.io_uring_read_into(
                    file_path, aligned_offset, raw_aligned_length,
                    raw.data_ptr(), 8, _use_direct_this_group,
                )
            else:
                fd = os.open(file_path, os.O_RDONLY)
                try:
                    view = memoryview(raw.numpy())
                    os.preadv(fd, [view[:raw_aligned_length]], aligned_offset)
                finally:
                    os.close(fd)

            data_raw = raw[prefix_pad:]
            packed = slab[slab_off:slab_off + packed_span]
            packed_off = 0
            for key, meta in group:
                rel = meta.file_offset - group_start
                dt = _resolve_dtype(meta.dtype, key, file_path)
                chunk = data_raw[rel:rel + meta.byte_length]
                dst_slice = packed[packed_off:packed_off + meta.byte_length]
                dst_slice.copy_(chunk)
                out[key] = dst_slice.view(dt).view(meta.shape)
                packed_off += _align_up_to(meta.byte_length, _TENSOR_ALIGN)
        finally:
            if _pool_buf is not None:
                _EXPERT_BUF_POOL.put(_pool_buf)

        slab_off += slab_span

    return out


def gpu_repack_aligned(src_views: dict, device) -> dict:
    """Copy each src view into ONE fresh contiguous buffer at _TENSOR_ALIGN
    boundaries, so the result is uint4-safe for EXL3's kernel regardless of the
    source's alignment. Used by the prefill staged-consume path (B1): staging
    now reads raw (tensor_align=False, zero CPU copy), whose GPU views land at
    arbitrary mod-_TENSOR_ALIGN offsets and would fault EXL3's uint4* load; this
    re-slices them onto aligned offsets in a single allocation. CUDA allocations
    are >=256-byte aligned, so the buffer base and every packed offset are
    16-safe by construction. `device` is passed through to torch.empty so unit
    tests can exercise the byte-exact packing on CPU."""
    packed_span = sum(_align_up_to(v.nbytes, _TENSOR_ALIGN) for v in src_views.values())
    dst = torch.empty(packed_span, dtype=torch.uint8, device=device)
    out = {}
    off = 0
    for key, v in src_views.items():
        nb = v.nbytes
        typed = dst[off:off + nb].view(v.dtype).view(v.shape)
        typed.copy_(v)
        out[key] = typed
        off += _align_up_to(nb, _TENSOR_ALIGN)
    return out


def views_from_packed_copy(src_views: dict, src_base: "torch.Tensor", dst: "torch.Tensor") -> dict:
    """Rebuild {key: view} inside `dst`, a copy of src_base's used prefix
    (typically a single H2D transfer of a pinned slab). Each view keeps its
    byte offset relative to src_base; dtype/shape are re-applied.

    **This function preserves the SOURCE's alignment; it does not create
    any.** `off` is carried over verbatim, so each returned view is exactly
    as aligned as its source view was. Production sources come from
    load_expert_into_slab(tensor_align=False) — raw reads, whose offsets
    are derived from the file's own byte layout and are itemsize-aligned
    only. **The results are therefore valid as copy SOURCES but NOT as EXL3
    kernel inputs**, which need 16 bytes because the kernel dereferences its
    weight buffer as uint4* and faults with cudaErrorMisalignedAddress
    otherwise. moe_patch.py's staged-consume path passes every dict from
    here through gpu_repack_aligned before the kernel sees it; that is where
    the alignment comes from (B1, 2026-07-13). An earlier version of this
    docstring claimed alignment was guaranteed here by construction —
    true only while the stager still used tensor_align=True, which it has
    not since B1."""
    assert src_base.dtype == torch.uint8 and dst.dtype == torch.uint8, \
        f"views_from_packed_copy requires uint8 buffers (byte-offset arithmetic); got src_base={src_base.dtype}, dst={dst.dtype}"
    base_ptr = src_base.data_ptr()
    out = {}
    for key, v in src_views.items():
        off = v.data_ptr() - base_ptr
        out[key] = dst[off:off + v.nbytes].view(v.dtype).view(v.shape)
    return out
