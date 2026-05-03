# Rust io_uring Zero-Copy NVMe Reader

Replace the Python `os.preadv` + `ThreadPoolExecutor` I/O path with a Rust io_uring reader that writes directly into Python's pre-allocated pinned memory buffer. **Zero memory copies across the PyO3 boundary.**

## Background

| | Old Rust module (broken, removed) | New Rust module (this plan) |
|---|---|---|
| Who allocates memory? | Rust (`Vec<u8>`) | Python (pinned `torch` buffer) |
| What crosses PyO3 boundary? | 1.6 GB of data (3 copies) | One integer (a pointer) |
| I/O mechanism | `mmap` + `MAP_POPULATE` | `io_uring` submission ring |
| Memory copies | 3 (4.8 GB wasted) | **0** |
| Expected bandwidth | 1.7 GB/s (measured) | **6-7 GB/s** (target) |

### Why io_uring over preadv threads

| | Python preadv (current) | Rust io_uring (proposed) |
|---|---|---|
| Syscalls per layer | 16 (one per thread) | 1 (`io_uring_enter`) |
| Thread overhead | 16 threads created/scheduled | Single thread, no scheduling |
| GIL interaction | Released per-thread, reacquired on completion | Released once for entire read |
| Measured bandwidth | 4.4 GB/s | Target: 6-7 GB/s |
| Context switches | ~32 per layer | ~1 per layer |

## Prerequisites

> [!IMPORTANT]
> - **Linux kernel 5.1+** for io_uring support (system has 6.17 ✅)
> - **Rust 1.95** installed (system has 1.95 ✅)
> - **maturin** for building PyO3 modules (`pip install maturin`)
> - GPU should be moved to **PCI_E1** first for maximum benefit

## Proposed Changes

---

### Rust Crate: `vaayu_rust/`

#### [MODIFY] [Cargo.toml](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/Cargo.toml)

Replace all dependencies. Remove `memmap2`, `safetensors`, `numpy` (no longer needed). Add `io-uring` crate.

```toml
[package]
name = "vaayu_rust"
version = "0.2.0"
edition = "2021"

[lib]
name = "vaayu_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module", "abi3-py38"] }
io-uring = "0.7"

[profile.release]
opt-level = 3
lto = "thin"
```

> [!NOTE]
> The `io-uring` crate is a pure Rust wrapper around the kernel's io_uring interface. Unlike the Python `liburing` package (which crashed with "Illegal Instruction" due to Zig compilation), this crate compiles natively with `rustc` using the correct CPU instruction set.

#### [MODIFY] [lib.rs](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/src/lib.rs)

Complete rewrite. The old `fast_load` function (mmap + triple copy) is deleted entirely. New function: `io_uring_read_into`.

**Design:**
1. Python passes: file path, data offset, total bytes, destination pointer, number of SQEs (queue depth)
2. Rust opens the file with `O_DIRECT` (bypass page cache, DMA straight to pinned memory)
3. Rust creates an io_uring instance and submits `QD` read operations as `ReadFixed` SQEs
4. Each SQE reads a chunk of the file into `ptr + offset`
5. Rust waits for all CQEs (completions)
6. Returns `Ok(())` — data is already in Python's buffer

```rust
use io_uring::{opcode, types, IoUring};
use pyo3::prelude::*;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;

/// Read a file region directly into a caller-provided buffer using io_uring.
///
/// # Arguments
/// * `path`       - Path to the .safetensors file
/// * `offset`     - Byte offset into the file where tensor data starts (after header)
/// * `length`     - Total bytes to read
/// * `buf_ptr`    - Raw pointer to the destination buffer (Python's pinned memory)
/// * `queue_depth` - Number of parallel io_uring SQEs (e.g. 32)
///
/// # Safety
/// The caller must ensure `buf_ptr` points to at least `length` bytes of valid,
/// writable memory that remains alive for the duration of this call.
#[pyfunction]
fn io_uring_read_into(
    py: Python<'_>,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
    queue_depth: u32,
) -> PyResult<()> {
    let path = path.to_owned();

    py.allow_threads(move || {
        // Open with O_DIRECT to bypass page cache (DMA straight to pinned mem)
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                format!("open {}: {}", path, e)))?;

        let fd = types::Fd(file.as_raw_fd());
        let ptr = buf_ptr as *mut u8;
        let qd = queue_depth.min(64) as usize;
        let chunk_size = (length + qd - 1) / qd;
        // Align chunk_size to 4096 for O_DIRECT
        let chunk_size = (chunk_size + 4095) & !4095;

        let mut ring = IoUring::new(qd as u32)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("io_uring_setup: {}", e)))?;

        // Submit all read SQEs
        let mut submitted = 0;
        let mut file_off = offset;
        let mut buf_off = 0usize;

        while buf_off < length {
            let this_len = chunk_size.min(length - buf_off);
            let read_op = opcode::Read::new(fd, unsafe { ptr.add(buf_off) }, this_len as u32)
                .offset(file_off)
                .build()
                .user_data(submitted as u64);

            unsafe { ring.submission().push(&read_op) }
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SQ full"))?;

            file_off += this_len as u64;
            buf_off += this_len;
            submitted += 1;
        }

        // Submit all at once — single syscall
        ring.submit_and_wait(submitted)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("io_uring submit: {}", e)))?;

        // Verify all completions
        let cq = ring.completion();
        let mut completed = 0;
        for cqe in cq {
            if cqe.result() < 0 {
                return Err(pyo3::exceptions::PyIOError::new_err(
                    format!("io_uring read error: {}", cqe.result())));
            }
            completed += 1;
        }

        if completed != submitted {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("incomplete: {}/{} CQEs", completed, submitted)));
        }

        Ok(())
    })
}

/// Fallback: simple pread without io_uring (for testing/compatibility)
#[pyfunction]
fn pread_into(
    py: Python<'_>,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
) -> PyResult<()> {
    let path = path.to_owned();
    py.allow_threads(move || {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        let ptr = buf_ptr as *mut u8;
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, length) };
        let n = unsafe { libc::pread(file.as_raw_fd(), buf.as_mut_ptr() as *mut _, length, offset as i64) };
        if n < 0 {
            return Err(pyo3::exceptions::PyIOError::new_err("pread failed"));
        }
        Ok(())
    })
}

#[pymodule]
fn vaayu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io_uring_read_into, m)?)?;
    m.add_function(wrap_pyfunction!(pread_into, m)?)?;
    Ok(())
}
```

> [!WARNING]
> **`O_DIRECT` requirement:** The destination buffer MUST be page-aligned (4096 bytes). PyTorch's `pin_memory=True` allocates via `cudaHostAlloc` which returns page-aligned memory, so this should work. We'll verify this in the benchmark phase. If alignment issues arise, we fall back to non-`O_DIRECT` reads.

> [!IMPORTANT]
> **`libc` crate dependency:** The code uses `libc::O_DIRECT` and `libc::pread`. Add `libc = "0.2"` to Cargo.toml dependencies.

Updated Cargo.toml dependencies:
```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module", "abi3-py38"] }
io-uring = "0.7"
libc = "0.2"
```

---

### Python Integration: `vaayu/persist/safetensor_model_persister.py`

#### [MODIFY] [safetensor_model_persister.py](file:///home/a/code/ai-ml/Vaayu/vaayu/persist/safetensor_model_persister.py)

Modify `_load_pinned()` to use the Rust io_uring reader instead of the Python ThreadPoolExecutor. **Keep the Python preadv path as a fallback.**

Changes:
1. At module level: try `import vaayu_rust`, set a flag
2. Replace the "Multi-threaded parallel read" section (lines 76-88) with a single `vaayu_rust.io_uring_read_into()` call
3. Keep the ThreadPoolExecutor path as fallback if `vaayu_rust` is not installed

```python
# At top of file, after existing imports:
try:
    import vaayu_rust
    _USE_IOURING = True
    print(">>>> Rust io_uring reader loaded — targeting 6+ GB/s NVMe bandwidth")
except ImportError:
    _USE_IOURING = False

# Inside _load_pinned(), replace the parallel read section:
def _load_pinned(file_path: str) -> dict:
    global _PINNED_BUFS, _PINNED_SIZES, _BUF_IDX
    # ... (header parsing stays the same — lines 50-60) ...
    # ... (buffer acquisition stays the same — lines 62-73) ...

    if _USE_IOURING:
        # ── 3a. Rust io_uring read (zero-copy into pinned buffer) ───
        vaayu_rust.io_uring_read_into(
            file_path,
            data_start,         # offset past safetensor header
            total_size,         # bytes to read
            buf.data_ptr(),     # raw pointer to pinned memory
            32,                 # queue depth (SQEs)
        )
    else:
        # ── 3b. Fallback: Python preadv (existing code) ─────────────
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

    # ── 4. Reconstruct tensors (stays exactly the same) ─────────
    # ...
```

> [!NOTE]
> The `memoryview(buf.numpy())` line (current line 73) is only needed for the Python preadv fallback. The Rust path uses `buf.data_ptr()` directly. We keep both paths.

---

## Open Questions

> [!IMPORTANT]
> **O_DIRECT + pinned memory alignment:** `O_DIRECT` requires the destination buffer to be 512-byte or 4096-byte aligned. PyTorch's `cudaHostAlloc` (used by `pin_memory=True`) typically returns page-aligned memory, but this needs verification. If it fails, we have two options:
> 1. Drop `O_DIRECT` (still use io_uring but through page cache — less benefit)
> 2. Allocate aligned memory manually with `posix_memalign` in Rust and copy once (still 1 copy vs 3 copies before)

> [!IMPORTANT]
> **O_DIRECT and read length alignment:** `O_DIRECT` also requires read lengths to be multiples of 512 bytes. Safetensor data sizes should be aligned (tensor sizes are always multiples of element size), but we need to handle the last chunk carefully by rounding up and truncating.

> [!IMPORTANT]
> **io_uring queue depth tuning:** The optimal QD for io_uring may differ from the thread-based QD16. We should benchmark QD 8, 16, 32, 64 to find the sweet spot. Starting with QD32 as default.

## Build & Install Workflow

```bash
# From the Vaayu project root:
cd vaayu_rust

# Install maturin if not already installed
pip install maturin

# Build and install the Rust module into the active venv
maturin develop --release

# Verify it loads
python -c "import vaayu_rust; print('io_uring reader loaded')"

# Run inference — should auto-detect and use io_uring
cd ..
python test_quick.py
```

## Verification Plan

### Phase 1: Build Verification
```bash
cd vaayu_rust && maturin develop --release
python -c "import vaayu_rust; print(dir(vaayu_rust))"
```
Expected: Module loads, `io_uring_read_into` and `pread_into` functions visible.

### Phase 2: Standalone Benchmark
Create a quick benchmark script that reads a single layer file using:
1. Python `os.preadv` (current — baseline)
2. Rust `io_uring_read_into` (new)
3. Rust `pread_into` (fallback)

Compare bandwidth (GB/s) and latency (ms per layer) for each.

```bash
python benchmark_rust_io.py
```

Expected results:
| Method | Bandwidth | Per-Layer Time |
|---|---|---|
| Python preadv QD16 | 4.4 GB/s | 0.35s |
| Rust io_uring QD32 | ~6-7 GB/s | ~0.24s |
| Rust pread (single) | ~2.8 GB/s | ~0.58s |

### Phase 3: Full Inference Test
```bash
python test_quick.py
```
Compare token summary line:
- Before: `total=29.6s  io_wait=8.8s  gpu_load=20.1s`
- After (x4 GPU): `total=~25s  io_wait=~4s  gpu_load=20.1s`
- After (x16 GPU): `total=~21s  io_wait=~15s  gpu_load=~5.4s`

### Phase 4: QD Sweep
Test io_uring with QD 8, 16, 32, 64 to find the optimal queue depth:
```bash
for qd in 8 16 32 64; do
    python benchmark_rust_io.py --qd $qd
done
```

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| io_uring kernel support issues | `pread_into` fallback function included |
| O_DIRECT alignment failure | Remove `O_DIRECT` flag, use buffered io_uring |
| Rust module fails to build | Python preadv path preserved, no regression |
| Pinned memory pointer invalid | Assertion check: verify `data_ptr() % 4096 == 0` before calling Rust |
| io_uring crate "Illegal Instruction" | Unlike Python liburing (Zig-compiled), this is pure Rust compiled with `rustc` — should not happen |

## Summary

| Component | Change |
|---|---|
| `vaayu_rust/Cargo.toml` | Replace deps: remove `memmap2/safetensors/numpy`, add `io-uring/libc` |
| `vaayu_rust/src/lib.rs` | Complete rewrite: `io_uring_read_into()` + `pread_into()` fallback |
| `safetensor_model_persister.py` | Add `try: import vaayu_rust` + conditional io_uring path |
| `pyproject.toml` | No changes needed |
| `vaayu_base.py` | **No changes** — prefetch pipeline stays exactly the same |

**Estimated time:** ~1 day (build + benchmark + integrate + QD tuning)
**Risk level:** Low — Python fallback preserved, no changes to inference logic
**Expected gain:** Disk bandwidth 4.4 → 6+ GB/s → saves ~5-8s per token
