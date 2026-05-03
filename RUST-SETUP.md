# Vaayu Rust io_uring Acceleration — Setup & Run Guide

Everything needed to build `vaayu_rust`, run inference, and read the diagnostic logs.

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Install Rust Toolchain](#3-step-1--install-rust-toolchain)
4. [Step 2 — Install maturin](#4-step-2--install-maturin)
5. [Step 3 — Build vaayu_rust](#5-step-3--build-vaayu_rust)
6. [Step 4 — Verify the Build](#6-step-4--verify-the-build)
7. [Step 5 — Run Inference](#7-step-5--run-inference)
8. [Reading the Logs](#8-reading-the-logs)
9. [How It Works — Zero-Copy Architecture](#9-how-it-works--zero-copy-architecture)
10. [Rebuilding After Code Changes](#10-rebuilding-after-code-changes)
11. [Fallback Behaviour (No Rust)](#11-fallback-behaviour-no-rust)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What Was Built

A Rust PyO3 module that uses Linux `io_uring` to read safetensor files directly into Python's pre-allocated pinned memory — **zero memory copies** across the Rust-Python boundary.

| File | What It Does |
|---|---|
| `vaayu_rust/src/lib.rs` | `io_uring_read_into()` — reads file data via io_uring directly into a pointer (Python's pinned buffer). `pread_into()` — single-threaded pread fallback. |
| `vaayu_rust/Cargo.toml` | Rust crate manifest — deps: `pyo3`, `io-uring`, `libc` |
| `vaayu_rust/pyproject.toml` | Maturin build config — compiles `lib.rs` into `vaayu_rust.so` |
| `vaayu/persist/safetensor_model_persister.py` | Modified — uses `vaayu_rust.io_uring_read_into()` if available, falls back to Python `os.preadv` QD16 threads if not |

The entrypoint (`test_quick.py`) is **unchanged** — you still run `python3 test_quick.py` exactly as before.

### Key Design: No Data Crosses the PyO3 Boundary

Unlike the previous (broken) Rust module that returned 1.6 GB of tensor data through PyO3 (causing triple memory copies), the new module only receives **one integer** — a raw pointer to Python's pinned memory buffer — and writes directly into it:

```
Python allocates pinned buffer → passes pointer (integer) to Rust
Rust uses io_uring to read NVMe → writes directly into that pointer
Rust returns Ok(()) → data is already in Python's buffer
```

**Zero copies. Zero allocations in Rust.**

---

## 2. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Linux kernel | 5.1+ | io_uring support. System has 6.17 ✅ |
| Python | 3.8+ | Project uses 3.12.3 |
| PyTorch | 2.1.0+ | Needed for pinned memory allocation |
| Rust | 1.70+ | Installed in Step 1 — only needed at build time |
| maturin | 1.5–1.x | Installed in Step 2 — only needed at build time |
| gcc / glibc | system default | Linux: already present |

> **Rust is only needed to build.** Once `vaayu_rust.so` is compiled, Rust and maturin can be uninstalled. `python3 test_quick.py` has no Rust dependency at runtime — it falls back to Python `preadv` automatically.

---

## 3. Step 1 — Install Rust Toolchain

```bash
# Download and run the Rust installer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

When prompted, choose **option 1** (default installation).

After installation completes:

```bash
# Load Rust into the current shell (only needed once per terminal session)
source "$HOME/.cargo/env"

# Verify
rustc --version   # should print e.g. rustc 1.95.0
cargo --version   # should print e.g. cargo 1.95.0
```

> **Subsequent terminal sessions**: `$HOME/.cargo/env` is automatically sourced if you chose to modify your shell profile during installation. If `cargo` is not found in a new terminal, run `source "$HOME/.cargo/env"` again.

---

## 4. Step 2 — Install maturin

Maturin is the build tool that compiles Rust into a Python extension (`.so`).

Install it **inside the project's venv**:

```bash
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate

pip install "maturin>=1.5,<2"

# Verify
maturin --version   # should print maturin 1.x.x
```

---

## 5. Step 3 — Build vaayu_rust

```bash
# Make sure you are in the venv and cargo is on PATH
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate
source "$HOME/.cargo/env"

# Build and install directly into the active venv
cd vaayu_rust
maturin develop --release
```

**What `maturin develop --release` does:**

1. Compiles `src/lib.rs` with full optimisations (`opt-level=3`, `lto=thin`)
2. Downloads Rust crate dependencies (`io-uring`, `libc`, `pyo3`) from crates.io on first run
3. Produces `vaayu_rust.cpython-312-x86_64-linux-gnu.so` (or similar)
4. Installs it directly into the venv's `site-packages` so `import vaayu_rust` works

**First build time:** ~2–5 minutes (crate downloads + compilation of all dependencies).  
**Subsequent builds:** ~10–30 seconds (only changed files recompile).

You should see output ending with something like:
```
🔗 Compiling vaayu_rust v0.2.0 (/home/a/code/ai-ml/Vaayu/vaayu_rust)
    Finished `release` profile [optimized] target(s) in 42.3s
📦 Built wheel for CPython 3.12
✏️  Setting installed package as editable
🛠 Installed vaayu_rust-0.2.0
```

---

## 6. Step 4 — Verify the Build

```bash
# Return to project root
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate

# Check module loads and functions are available
python3 -c "import vaayu_rust; print(dir(vaayu_rust))"
```

Expected output:
```
['__all__', ..., 'io_uring_read_into', 'pread_into']
```

Run a quick benchmark to confirm io_uring reads work:

```bash
python3 -c "
import time, struct, json, torch

# Setup: allocate pinned buffer
buf = torch.empty(2 * 1024 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
ptr = buf.data_ptr()
print(f'Buffer pointer: {hex(ptr)} (aligned: {ptr % 4096 == 0})')

path = '/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.0.safetensors'

# Parse header to get data offset and size
with open(path, 'rb') as f:
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    data_start = 8 + hdr_len
meta = {k: v for k, v in hdr.items() if k != '__metadata__'}
total = max(v['data_offsets'][1] for v in meta.values())
print(f'Layer size: {total / 1024**2:.0f} MB, data_start: {data_start}')

# Rust io_uring read
import vaayu_rust
t = time.time()
vaayu_rust.io_uring_read_into(path, data_start, total, ptr, 32)
rust_t = time.time() - t
print(f'Rust io_uring : {rust_t:.3f}s  ({total/rust_t/1e9:.2f} GB/s)')

# Python preadv comparison
import os
t = time.time()
view = memoryview(buf.numpy())
fd = os.open(path, os.O_RDONLY)
chunk = (total + 15) // 16
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(16) as ex:
    def rd(i):
        o = i * chunk
        l = min(chunk, total - o)
        if l > 0: os.preadv(fd, [view[o:o+l]], data_start + o)
    list(ex.map(rd, range(16)))
os.close(fd)
py_t = time.time() - t
print(f'Python preadv : {py_t:.3f}s  ({total/py_t/1e9:.2f} GB/s)')
print(f'Speedup: {py_t/rust_t:.1f}x')
"
```

---

## 7. Step 5 — Run Inference

Nothing changes from the standard run command:

```bash
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate

python3 test_quick.py
```

The startup logs tell you which loader is active:

```
>>>> pinned-memory loader active — direct NVMe→RAM→GPU path
>>>> Rust io_uring reader loaded — targeting 6+ GB/s NVMe bandwidth    ← io_uring active
```

If Rust is not installed:
```
>>>> pinned-memory loader active — direct NVMe→RAM→GPU path            ← preadv fallback
```

---

## 8. Reading the Logs

### 8.1 Per-Layer I/O Logs

Printed once per layer per token:

```
[VAAYU-IO] iouring 0.250s  model.layers.0     ← Rust io_uring path
[VAAYU-IO] iouring 0.241s  model.layers.1
...
```

or (if Rust not installed):

```
[VAAYU-IO] pinned  0.350s  model.layers.0     ← Python preadv fallback
```

| Field | Meaning |
|---|---|
| `iouring` / `pinned` / `python` | Which loader handled this layer |
| `0.250s` | Wall time for the read (NVMe → pinned RAM) |
| `model.layers.0` | Layer name |

### 8.2 Per-Token Summary Log

Printed once per token at the end of each `forward()` call:

```
[VAAYU-IO] token summary: total=29.6s  io_wait=8.8s  gpu_load=20.1s  compute=0.1s  other=0.5s
```

| Field | What it measures |
|---|---|
| `total` | Wall time for the entire `forward()` call |
| `io_wait` | Time the main thread was blocked waiting for prefetch `future.result()` |
| `gpu_load` | Time spent in `move_layer_to_device()` — CPU→GPU weight transfer via PCIe |
| `compute` | Time spent in `layer.forward()` — actual GPU computation |
| `other` | Everything else (cleanup, tqdm, Python overhead) |

---

## 9. How It Works — Zero-Copy Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python (safetensor_model_persister.py)                 │
│                                                         │
│  1. Parse safetensor header (get data_start, total_size)│
│  2. Acquire pinned buffer from double-buffer pool       │
│  3. Get raw pointer: ptr = buf.data_ptr()               │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  vaayu_rust.io_uring_read_into(                   │  │
│  │      path, data_start, total_size, ptr, 32        │  │
│  │  )                                                │  │
│  │                                                   │  │
│  │  Rust receives ptr as integer → casts to *mut u8  │  │
│  │  Opens file with O_DIRECT (bypass page cache)     │  │
│  │  Creates io_uring ring with 32 SQEs               │  │
│  │  Submits all reads → single io_uring_enter()      │  │
│  │  NVMe DMA writes directly into ptr                │  │
│  │  Returns Ok(()) — nothing crosses the boundary    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  4. Slice buffer into tensor views (zero-copy)          │
│  5. Transfer to GPU: tensor.to('cuda')                  │
└─────────────────────────────────────────────────────────┘
```

### Why This Is Different From the Old Rust Module

| | Old Rust (removed) | New Rust (current) |
|---|---|---|
| I/O method | `mmap` + `MAP_POPULATE` | `io_uring` + `O_DIRECT` |
| Memory allocation | Rust allocates `Vec<u8>` | Python allocates pinned buffer |
| Data across PyO3 | 1.6 GB (via NumPy array) | 1 integer (a pointer) |
| Memory copies | **3** (4.8 GB wasted) | **0** |
| Dependencies | `memmap2, safetensors, numpy` | `io-uring, libc` |
| Result | 2.3s/layer (190s/token) 💀 | Target: 0.25s/layer (~21s/token) |

---

## 10. Rebuilding After Code Changes

After any change to `vaayu_rust/src/lib.rs` or `vaayu_rust/Cargo.toml`:

```bash
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
source ../venv/bin/activate
source "$HOME/.cargo/env"

maturin develop --release
```

After any change to Python files (`safetensor_model_persister.py`, `vaayu_base.py`, etc.):

```bash
# No rebuild needed — Python files are loaded directly from source.
# Just run test_quick.py again.
python3 test_quick.py
```

---

## 11. Fallback Behaviour (No Rust)

If `vaayu_rust` is not installed, the persister automatically falls back to the Python `preadv` QD16 reader:

```python
# safetensor_model_persister.py — automatic at import time
try:
    import vaayu_rust
    _USE_IOURING = True
    print(">>>> Rust io_uring reader loaded — targeting 6+ GB/s NVMe bandwidth")
except ImportError:
    _USE_IOURING = False
```

Everything works — inference runs at 4.4 GB/s instead of the io_uring target of 6+ GB/s. The `[VAAYU-IO]` logs will show `pinned` instead of `iouring`.

---

## 12. Troubleshooting

### `cargo: command not found` after Rust install

```bash
source "$HOME/.cargo/env"
# Then retry the maturin develop command
```

### `maturin: command not found`

```bash
source venv/bin/activate
pip install "maturin>=1.5,<2"
```

### Build error: `error[E0432]: unresolved import`

Cargo dependency version mismatch. Try:

```bash
cd vaayu_rust
cargo update          # updates Cargo.lock to compatible versions
maturin develop --release
```

### Build error: `error: linker 'cc' not found`

Missing C linker (needed by PyO3). On Ubuntu/Debian:

```bash
sudo apt-get install build-essential
```

### `ModuleNotFoundError: No module named 'vaayu_rust'` at runtime

The `.so` was not installed into the active venv. Confirm you are in the right venv and rebuilt:

```bash
which python3                    # should point into .../Vaayu/venv/...
pip show vaayu-rust              # should show version 0.2.0
```

If `pip show` returns nothing:

```bash
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
maturin develop --release
```

### `RuntimeError: io_uring_setup: ...`

io_uring is not supported on your kernel (requires 5.1+). Check:

```bash
uname -r    # must be >= 5.1
```

If on an older kernel, the Python preadv fallback will be used automatically.

### `PyIOError: open ...: Invalid argument` (O_DIRECT failure)

`O_DIRECT` requires the destination buffer to be page-aligned (4096 bytes). PyTorch's `cudaHostAlloc` usually provides this, but if it fails:

1. Verify alignment: `python3 -c "import torch; b=torch.empty(1024,dtype=torch.uint8,pin_memory=True); print(b.data_ptr() % 4096)"`  
   → Should print `0`

2. If non-zero, modify `lib.rs` to remove `O_DIRECT`:
   ```rust
   // Change this:
   .custom_flags(libc::O_DIRECT)
   // To this:
   // (remove the line — use buffered I/O through page cache)
   ```
   Then rebuild with `maturin develop --release`.

### `io_uring read error: -22` (EINVAL)

`O_DIRECT` requires read lengths to be multiples of 512 bytes. The chunk alignment in `lib.rs` handles this (`(chunk_size + 4095) & !4095`), but if a safetensor file has unusual data sizes, you may hit this. Remove `O_DIRECT` as described above.

### Per-layer times look the same as before (no speedup)

Filesystem page cache is warm — the OS already has the file in RAM from a prior run. To force cold read:

```bash
# Drop page cache (requires root)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
python3 test_quick.py
```

### `io_wait` is still high even with io_uring

With `prefetching=True` (default), `io_wait` measures time the main thread blocks on `future.result()`. If `io_wait` is still high, the NVMe is the bottleneck — io_uring can only read as fast as the physical drive allows. Check the per-layer times:
- `0.35s/layer` = 4.4 GB/s (same as Python — io_uring didn't help, likely cached reads)
- `0.25s/layer` = 6.5 GB/s (io_uring working correctly)
- `0.22s/layer` = 7.4 GB/s (theoretical max for Samsung 990 PRO)

---

## Quick Reference

```bash
# One-time setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
cd /home/a/code/ai-ml/Vaayu && source venv/bin/activate
pip install "maturin>=1.5,<2"
cd vaayu_rust && maturin develop --release && cd ..

# Every run
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate
python3 test_quick.py

# After Rust code changes only
cd vaayu_rust && maturin develop --release && cd ..
python3 test_quick.py

# Verify io_uring loader active
python3 -c "import vaayu_rust; print('io_uring_read_into' in dir(vaayu_rust))"
```
