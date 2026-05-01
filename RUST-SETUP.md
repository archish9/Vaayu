# Vaayu Rust Acceleration — Setup & Run Guide

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
9. [Expected Performance](#9-expected-performance)
10. [Rebuilding After Code Changes](#10-rebuilding-after-code-changes)
11. [Fallback Behaviour (No Rust)](#11-fallback-behaviour-no-rust)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What Was Built

Three components were added or changed:

| File | What Changed |
|---|---|
| `vaayu_rust/src/lib.rs` | New Rust extension — `fast_load()` loads a `.safetensors` file via `mmap` with `MAP_POPULATE`, releasing the Python GIL during all I/O |
| `vaayu_rust/Cargo.toml` | Rust crate manifest — deps: `pyo3`, `memmap2`, `safetensors`, `numpy` |
| `vaayu_rust/pyproject.toml` | Maturin build config — compiles `lib.rs` into `vaayu_rust.so` |
| `vaayu/persist/safetensor_model_persister.py` | Modified — tries `vaayu_rust.fast_load()` first, falls back to Python `safetensors.torch.load_file()` if Rust module not found |
| `vaayu/vaayu_base.py` | `load_layer_to_cpu()` — removed `pin_memory()` step (no-op for mmap memory); added per-token I/O timing accumulators |

The entrypoint (`test_quick.py`) is **unchanged** — you still run `python3 test_quick.py` exactly as before.

---

## 2. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.8+ | Project uses 3.12.3 |
| PyTorch | 2.1.0+ | Already in `requirements.txt` |
| numpy | any | Already installed (required by vaayu) |
| Rust | 1.70+ | Installed in Step 1 — only needed at build time |
| maturin | 1.5–1.x | Installed in Step 2 — only needed at build time |
| gcc / glibc | system default | Linux: already present |
| Internet | build only | Rust crates downloaded once during first build |

> **Rust is only needed to build.** Once `vaayu_rust.so` is compiled, Rust and maturin can be uninstalled. `python3 test_quick.py` has no Rust dependency at runtime.

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
rustc --version   # should print e.g. rustc 1.82.0 (f6e511eec 2024-10-15)
cargo --version   # should print e.g. cargo 1.82.0 (8f40fc59f 2024-08-21)
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
2. Downloads Rust crate dependencies from crates.io on first run (~50MB, cached afterwards)
3. Produces `vaayu_rust.cpython-312-x86_64-linux-gnu.so` (or similar)
4. Installs it directly into the venv's `site-packages` so `import vaayu_rust` works

**First build time:** ~2–5 minutes (crate downloads + compilation of all dependencies).  
**Subsequent builds:** ~10–30 seconds (only changed files recompile).

You should see output ending with something like:
```
🔗 Compiling vaayu_rust v0.1.0 (/home/a/code/ai-ml/Vaayu/vaayu_rust)
   Compiling vaayu_rust v0.1.0
    Finished `release` profile [optimized] target(s) in 42.3s
📦 Built wheel for CPython 3.12 to /tmp/.../vaayu_rust-0.1.0-cp312-cp312-linux_x86_64.whl
✏️  Setting installed package as editable
🛠 Installed vaayu_rust-0.1.0
```

---

## 6. Step 4 — Verify the Build

```bash
# Return to project root
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate

python3 -c "import vaayu_rust; print('vaayu_rust loaded OK')"
```

Expected output:
```
vaayu_rust loaded OK
```

Run a quick load benchmark to confirm the module works and measure baseline per-layer time:

```bash
python3 -c "
import time, vaayu_rust
from safetensors.torch import load_file

path = '/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.0.safetensors'

# Rust path
t = time.time()
d = vaayu_rust.fast_load(path)
rust_t = time.time() - t
print(f'Rust  fast_load : {rust_t:.3f}s  ({len(d)} tensors)')

# Python path
t = time.time()
d2 = load_file(path, device='cpu')
py_t = time.time() - t
print(f'Python load_file: {py_t:.3f}s  ({len(d2)} tensors)')

print(f'Speedup: {py_t/rust_t:.1f}x')
"
```

Expected output (approximate):
```
Rust  fast_load : 0.280s  (N tensors)
Python load_file: 0.950s  (N tensors)
Speedup: 3.4x
```

---

## 7. Step 5 — Run Inference

Nothing changes from the standard run command:

```bash
cd /home/a/code/ai-ml/Vaayu
source venv/bin/activate

python3 test_quick.py
```

The first line printed by `safetensor_model_persister.py` at import time tells you which loader is active:

```
>>>> vaayu_rust loaded — using mmap fast_load        ← Rust active
```

or

```
>>>> vaayu_rust not found — using Python safetensors loader   ← fallback active
```

---

## 8. Reading the Logs

### 8.1 Per-Layer I/O Logs

Printed once per layer per token:

```
[VAAYU-IO] rust   0.287s  model.layers.0
[VAAYU-IO] rust   0.271s  model.layers.1
...
```

Format: `[VAAYU-IO] <loader>  <seconds>s  <layer_name>`

| Field | Meaning |
|---|---|
| `rust` / `python` | Which loader handled this layer |
| `0.287s` | Wall time from `fast_load()` call to return (includes mmap setup + data copy to Python) |
| `model.layers.0` | Layer shard filename (without `.safetensors`) |

> With prefetching enabled (the default), layers load in a background thread. The time shown is the actual disk load time running in parallel with GPU compute. The `io_wait` in the token summary (§8.2) is the time the main thread blocked waiting for the result — these two numbers are different.

### 8.2 Per-Token Summary Log

Printed once per token at the end of each `forward()` call:

```
[VAAYU-IO] token summary: total=56.4s  io_wait=42.1s  gpu_load=8.3s  compute=4.2s  other=1.8s
```

| Field | What it measures |
|---|---|
| `total` | Wall time for the entire `forward()` call |
| `io_wait` | Time the main thread was blocked waiting on `future.result()` (prefetch) or `load_layer_to_cpu()` (no prefetch). With prefetch + fast I/O this approaches 0. |
| `gpu_load` | Time spent in `move_layer_to_device()` — CPU→GPU weight transfer via PCIe |
| `compute` | Time spent in `layer.forward()` — actual GPU attention + FFN compute |
| `other` | Unaccounted (embedding, norm, lm_head, clean_memory, RoPE, overhead) |

**Baseline (Python loader, no prefetch overlap):**
```
io_wait ≈ 50s   gpu_load ≈ 4s   compute ≈ 4s   total ≈ 58s
```

**Target (Rust loader, prefetch hiding I/O):**
```
io_wait ≈ 0-5s  gpu_load ≈ 4s   compute ≈ 4s   total ≈ 10-15s
```

### 8.3 KV Cache Effect (Token 2+)

With KV cache enabled (current code), tokens 2+ only push **1 token** through all 83 layers instead of the full growing sequence. This appears in the logs as shorter `compute` times from token 2 onward — compute drops from ~4s (83 layers × full seq) to ~0.4s (83 layers × 1 token).

Combined effect after all optimisations:
- Token 1: ~10-15s (full prompt load + compute)
- Token 2+: ~10-15s (I/O dominated) — eventually bottlenecked purely by NVMe throughput

### 8.4 Log Flow Per Token (Full Example)

```
[Vaayu] inference start: 2026-05-01 16:46:20
>>>> vaayu_rust loaded — using mmap fast_load       ← printed once at import
[VAAYU-IO] rust   0.282s  model.embed_tokens
[VAAYU-IO] rust   0.271s  model.layers.0
[VAAYU-IO] rust   0.268s  model.layers.1
...  (83 lines total)
[VAAYU-IO] rust   0.275s  lm_head
running layers(cuda): 100%|████| 83/83 [00:14<00:00, ...]
[Vaayu] inference end:   2026-05-01 16:46:34
[Vaayu] total execution time: 14.231s
[VAAYU-IO] token summary: total=14.2s  io_wait=3.1s  gpu_load=4.8s  compute=4.2s  other=2.1s
```

---

## 9. Expected Performance

### Per-Token Timing (Llama 3.3 70B, 83 layers, RTX 5060 Ti 16GB)

| State | io_wait | gpu_load | compute | total/token |
|---|---|---|---|---|
| **Before** (Python loader, KV cache off) | ~50s | ~4s | ~4s | ~58s |
| **After Phase 1** (Rust mmap loader) | ~15s | ~4s | ~4s | ~23s |
| **After Phase 1+2** (Rust + prefetch overlap) | ~0s | ~4s | ~4s | ~8s |
| **After Phase 1+2 + KV cache (token 2+)** | ~0s | ~4s | ~0.4s | ~5s |

> Numbers are estimates based on Samsung 990 PRO NVMe at ~7 GB/s sequential read and RTX 5060 Ti GPU compute benchmarks. Actual results vary with filesystem cache state (warm vs cold), PCIe bandwidth utilisation, and thermal throttling.

### Why Token 1 Is Always Slower

Token 1 processes the full prompt (9 tokens in the test) through all 83 layers. KV cache has no past state yet, so there is no shortcut. Every subsequent token only processes 1 new token.

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

If `vaayu_rust` is not installed (Rust not built, or `.so` not in venv), the persister automatically falls back to the Python `safetensors` loader:

```python
# safetensor_model_persister.py — automatic at import time
try:
    import vaayu_rust
    _RUST_LOADER = True
    print(">>>> vaayu_rust loaded — using mmap fast_load")
except ImportError:
    _RUST_LOADER = False
    print(">>>> vaayu_rust not found — using Python safetensors loader")
```

Everything works — inference is just slower. The `[VAAYU-IO]` logs will show `python` instead of `rust`.

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
pip show vaayu-rust              # should show version 0.1.0
```

If `pip show` returns nothing:

```bash
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
maturin develop --release
```

### `RuntimeError: mmap failed: Operation not permitted`

Some environments restrict `mmap` with `MAP_POPULATE` (Docker with `--security-opt=no-new-privileges`, certain SELinux policies). Remove the `.populate()` call in `src/lib.rs`:

```rust
// Change this:
let mmap = unsafe { MmapOptions::new().populate().map(&file)? };

// To this:
let mmap = unsafe { MmapOptions::new().map(&file)? };
```

Then rebuild with `maturin develop --release`. Performance will be slightly lower (no kernel readahead hint) but correct.

### Tensors differ between Rust and Python loader

Run the correctness check:

```bash
python3 -c "
import torch, vaayu_rust
from safetensors.torch import load_file
import numpy as np

path = '/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.0.safetensors'

py_dict = load_file(path, device='cpu')
rust_raw = vaayu_rust.fast_load(path)

import sys
sys.path.insert(0, '.')
from vaayu.persist.safetensor_model_persister import _raw_to_tensor
rust_dict = {k: _raw_to_tensor(d, dtype, shape) for k, (d, dtype, shape) in rust_raw.items()}

all_match = True
for name in py_dict:
    if not torch.equal(py_dict[name], rust_dict[name]):
        print(f'MISMATCH: {name}')
        all_match = False

print('All tensors match!' if all_match else 'TENSORS DO NOT MATCH — check dtype handling in _raw_to_tensor')
"
```

### Per-layer times look the same as before (no speedup)

Filesystem page cache is warm — the OS already has the file in RAM from a prior run. To force cold read:

```bash
# Drop page cache (requires root)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
python3 test_quick.py
```

Or compare only the first run after a reboot.

### `io_wait` is still high even with Rust

With `prefetching=True` (default), `io_wait` measures time the main thread blocks on `future.result()` — i.e., how much I/O was not hidden by the prefetch pipeline. If `io_wait` is still high after switching to the Rust loader, the bottleneck has shifted to `gpu_load` (PCIe transfer). Check the `gpu_load` value in the token summary log — if it is large, the NVMe→CPU→GPU pipeline is the remaining bottleneck (Phase 2 of the implementation plan addresses this with `io_uring` prefetch).

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

# Verify Rust loader active
python3 -c "import vaayu_rust; print('OK')"

# Correctness benchmark
python3 -c "
import time, vaayu_rust
from safetensors.torch import load_file
p = '/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.0.safetensors'
t=time.time(); vaayu_rust.fast_load(p); print(f'Rust : {time.time()-t:.3f}s')
t=time.time(); load_file(p,device='cpu'); print(f'Py   : {time.time()-t:.3f}s')
"
```
