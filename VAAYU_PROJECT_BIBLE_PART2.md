# VAAYU PROJECT BIBLE — PART 2
### The I/O Bottleneck: Discovery, Diagnosis, and the Road to 4.4 GB/s

*This is Part 2 of 4. Read Part 1 first (VAAYU_PROJECT_BIBLE.md).*

---

## 5. The I/O Bottleneck — Core Problem

This is the central, dominant problem of the entire project. Everything else is noise by comparison. This section documents the discovery process in full, including all dead ends.

---

### 5.1 Baseline: Where We Started (~116–190 seconds per token)

The initial state of the codebase used Python's standard `safetensors.torch.load_file()` to load each layer, combined with per-layer memory allocation. Two real runs were logged.

**Run log (before any I/O optimization, token 1):**
```
[Vaayu] inference start: 2026-05-02 15:56:42
[VAAYU-IO] pinned 2.352s  model.embed_tokens
[VAAYU-IO] pinned 0.536s  model.layers.0
[VAAYU-IO] pinned 1.609s  model.layers.1
[VAAYU-IO] pinned 1.592s  model.layers.2
[VAAYU-IO] pinned 1.605s  model.layers.3
...
[VAAYU-IO] pinned 1.555s  model.layers.5
[VAAYU-IO] pinned 1.526s  model.layers.6
...
[VAAYU-IO] pinned 1.733s  model.layers.58
...
[VAAYU-IO] pinned 1.734s  model.layers.68
[VAAYU-IO] pinned 0.001s  model.norm
[VAAYU-IO] pinned 2.065s  lm_head
[Vaayu] inference end:   2026-05-02 15:58:39
[Vaayu] total execution time: 116.823s
[VAAYU-IO] token summary: total=116.8s  io_wait=88.4s  gpu_load=20.5s  compute=0.2s  other=7.8s
```

**Key observations from this log:**
- `io_wait=88.4s` — **76% of the token time was waiting for disk I/O**
- Layers 45–53 showed anomalously fast times (0.166–0.175s). This is the Linux page cache serving data from RAM — those layers were still cached from a recent previous run
- `gpu_load=20.5s` — GPU PCIe transfer was already 18% of the total
- `compute=0.2s` — GPU math was essentially free

**The 1.55s per layer breakdown (pre-fix analysis):**
```
~0.50s  — torch.empty(1.6GB, pin_memory=True)  [kernel locks RAM every call]
~1.05s  — actual f.readinto(buf) from NVMe
```

The per-layer allocation of 1.6 GB of pinned memory was costing **~0.5 seconds per layer × 83 layers = ~41.5 seconds** of pure overhead that had nothing to do with disk I/O.

---

### 5.2 Hardware Benchmark: `dd` — Establishing the True NVMe Speed

Before writing any code, the true hardware ceiling was measured using `dd` — a raw kernel-level file copy tool that completely bypasses Python, the page cache, and all application overhead.

**Method: drop caches, then read a 1.6 GB layer file to /dev/null:**
```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
dd if=/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.10.safetensors \
   of=/dev/null bs=1M status=progress
```

**Result:**
```
1711309840 bytes (1.7 GB, 1.6 GiB) copied, 0.613746 s, 2.8 GB/s
```

**This was the first critical discovery: the NVMe was running at 2.8 GB/s, not 7.4 GB/s.**

This was immediately assumed to be a hardware configuration problem. However, inspection of the PCIe link confirmed:
```bash
cat /sys/class/nvme/nvme0/device/current_link_speed  # → 16.0 GT/s
cat /sys/class/nvme/nvme0/device/current_link_width  # → 4
cat /sys/class/nvme/nvme0/device/max_link_speed       # → 16.0 GT/s
cat /sys/class/nvme/nvme0/device/max_link_width       # → 4
```

The drive is physically connected at **PCIe Gen 4 x4** — exactly the maximum it supports. There are no BIOS settings to fix.

**Root cause of 2.8 GB/s ceiling: Queue Depth 1.**

The `dd` tool, like standard Python `read()`, reads the file sequentially with **one outstanding I/O request at a time** (Queue Depth 1). At QD1, Samsung's NVMe controller architecture can only sustain ~2.5–3.0 GB/s because the controller's internal flash chips cannot be parallelized with only one request in flight. The 7.4 GB/s rated speed requires QD ≥ 2 with multiple threads (per Samsung's own datasheet — the specification is measured at QD2 multi-thread, not QD32 as was initially reported in the research paper).

**Also confirmed with `fio` at QD32:**
```bash
fio --name=test --ioengine=libaio --iodepth=32 --rw=read \
    --bs=1M --direct=1 --size=1G --numjobs=1 \
    --filename=model.layers.10.safetensors
```
The `fio` result showed higher throughput at depth 32, confirming the drive is capable of more — but only when software submits parallel requests.

---

### 5.3 Fix #1 — Global Pre-Allocated Pinned Memory Buffer

**Problem:** `torch.empty(1.6GB, pin_memory=True)` was being called inside the layer-loading loop — once per layer. This forces the Linux kernel to find 1.6 GB of contiguous physical RAM and register it as non-pageable memory on every call. This takes ~0.5 seconds even on a system with plenty of free RAM.

**Benchmark confirming the cost:**
```python
for i in range(3):
    t = time.time()
    buf = torch.empty(1024*1024*1600, dtype=torch.uint8, pin_memory=True)
    print(f'pin_memory allocate: {time.time()-t:.3f}s')
    # Output: ~0.5s on every iteration
```

**Solution: Allocate two buffers once at startup, reuse them forever.**

The reason two buffers are needed (double-buffering): Vaayu's prefetch thread loads Layer N+1 into a buffer while the main GPU thread is transferring Layer N over PCIe. If there is only one buffer, the prefetch would overwrite data that the GPU transfer hasn't finished reading yet. Two buffers alternate: while buffer A is being sent to the GPU, the loader fills buffer B, and vice versa.

**Implementation in `safetensor_model_persister.py`:**
```python
# Allocated once at module load time — 1.7 GB each, pinned
_PINNED_BUF_SIZE = 1024 * 1024 * 1800  # 1.8 GB (max layer size with headroom)
_PINNED_BUFS = [
    torch.empty(_PINNED_BUF_SIZE, dtype=torch.uint8, pin_memory=True),
    torch.empty(_PINNED_BUF_SIZE, dtype=torch.uint8, pin_memory=True),
]
_BUF_IDX = 0  # toggles 0/1 between layers

def _load_pinned(path: str) -> dict:
    global _BUF_IDX
    buf = _PINNED_BUFS[_BUF_IDX]
    _BUF_IDX = 1 - _BUF_IDX  # flip for next call
    # ... read file into buf, slice tensors from it ...
```

**Impact of this fix alone:** Eliminates the 0.5s × 83 layers = **~41.5s allocation overhead per token**.

---

### 5.4 Fix #2 — Multi-Threaded Parallel I/O (QD16)

**The Insight:** Python's `file.read()` / `f.readinto()` uses synchronous I/O — one request in flight at a time (Queue Depth 1). This caps NVMe throughput at ~2.8 GB/s. But if we split the file into N chunks and read all N chunks simultaneously from N different threads, the kernel sees N independent read requests in flight — exactly the higher queue depth the NVMe controller needs to engage its internal parallelism.

**Key technical detail:** `os.preadv()` — a POSIX syscall for reading a file at a given offset into a scatter/gather buffer — releases the Python GIL during the kernel call. This means N threads running `os.preadv()` simultaneously are true parallel kernel I/O operations, not GIL-serialized Python code.

**Thread count sweep benchmark (all cold reads, no page cache):**

```
QD  1:  avg 1.536s  (1.04 GB/s)   single thread baseline (worse — cold vs dd)
QD  2:  avg 0.747s  (2.13 GB/s)
QD  4:  avg 0.602s  (2.65 GB/s)
QD  8:  avg 0.419s  (3.80 GB/s)   major jump
QD 12:  avg 0.376s  (4.24 GB/s)
QD 16:  avg 0.362s  (4.40 GB/s)   sweet spot begins
QD 20:  avg 0.360s  (4.43 GB/s)   best
QD 24:  avg 0.362s  (4.40 GB/s)   no improvement
QD 32:  avg 0.361s  (4.42 GB/s)   no improvement
```

**Conclusion: QD16 is the practical optimum.** Beyond 16 threads, the NVMe controller is saturated and adding more threads introduces scheduling overhead with no bandwidth gain. QD20 technically scores highest but the 0.3% improvement over QD16 is within noise.

**Implementation:**
```python
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=16)  # permanent pool, created once

def _preadv_chunked(fd, view, offset, size, n_threads=16):
    chunk = (size + n_threads - 1) // n_threads
    def read_chunk(i):
        off = i * chunk
        length = min(chunk, size - off)
        if length > 0:
            os.preadv(fd, [view[off:off+length]], offset + off)
    list(_IO_EXECUTOR.map(read_chunk, range(n_threads)))
```

**Impact:** Per-layer read time drops from ~1.05s (QD1) to ~0.36s (QD16) — a **65% reduction** in pure I/O time.

---

### 5.5 io_uring Exploration — A Dead End (For Now)

A technical reviewer of the research paper noted that the Python threading approach is a workaround for Python's lack of native async file I/O. The correct solution would be `io_uring` — Linux's modern async I/O interface that allows a single thread to submit 32+ read requests to the kernel simultaneously in one syscall, with zero thread coordination overhead. On a Samsung 990 PRO with `io_uring`, 6+ GB/s sequential reads should be achievable.

**Kernel version check:**
```bash
uname -r  # → 6.17
```
Kernel 6.17 supports `io_uring`. Minimum required is 6.14.

**Python binding attempt — `liburing` package:**
```bash
pip install liburing
python3 -c "import liburing; print(dir(liburing))"
# → Imported successfully

python3 -c "
from liburing import Ring, io_uring_queue_init, io_uring_get_sqe, \
    io_uring_prep_read, io_uring_submit, io_uring_wait_cqe
ring = Ring()
io_uring_queue_init(8, ring)
# small read test...
"
# → Illegal instruction (core dumped)
```

**Result: `liburing` Python package crashes with `Illegal instruction`.**

The package uses a Zig backend compiled with CPU intrinsics that the i5-14600K does not support. There is no published prebuilt for this CPU variant. Building from source may fix it but was not pursued.

**Alternative `aiofile` package (uses `caio` / Linux AIO):**
```bash
pip install aiofile
python3 -c "import caio; print(dir(caio))"
```
The `caio` package provides Linux's older `libaio` interface, not `io_uring`. It was not benchmarked.

**Current status:** `io_uring` from Python is blocked by the `liburing` crash. The Rust `vaayu_rust` module (described in the next section) implements `io_uring` at the native level via the `io-uring` Rust crate, which does not have the Zig/CPU-intrinsic issue.

**File fragmentation check (related):**
```bash
filefrag /home/a/code/ai-ml/models/.../model.layers.{0,10,20,30,40,50,60,70,79}.safetensors
# → 4–12 extents per file (average ~6)
```
Layer files have mild fragmentation — 4–12 physical disk extents per 1.6 GB file. A perfectly contiguous file would have 1 extent. This contributes slightly to not hitting the full 4.4+ GB/s possible with truly sequential reads. You do not need to re-run `filefrag` — the files won't change unless the model is re-downloaded or re-split.

---

### 5.6 Combined Result After Fix #1 + Fix #2

After implementing both the global double-buffer and the QD16 threaded reader, the inference times collapsed:

**Run log (QD16 + double buffer, token 1):**
```
[Vaayu] inference start: 2026-05-03 10:46:21
[VAAYU-IO] pinned 1.274s  model.embed_tokens
[VAAYU-IO] pinned 0.785s  model.layers.0       ← cold read (no cache)
[VAAYU-IO] pinned 0.346s  model.layers.1
[VAAYU-IO] pinned 0.363s  model.layers.2
[VAAYU-IO] pinned 0.363s  model.layers.3
[VAAYU-IO] pinned 0.362s  model.layers.4
...
[VAAYU-IO] pinned 0.364s  model.layers.78
[VAAYU-IO] pinned 0.364s  model.layers.79
[VAAYU-IO] pinned 0.001s  model.norm
[VAAYU-IO] pinned 0.536s  lm_head
[Vaayu] inference end:   2026-05-03 10:46:53
[Vaayu] total execution time: 32.672s
[VAAYU-IO] token summary: total=32.7s  io_wait=2.0s  gpu_load=20.4s  compute=0.2s  other=10.1s
```

**Token 2 (page cache warming up):**
```
[Vaayu] total execution time: 32.044s
[VAAYU-IO] token summary: total=32.0s  io_wait=1.7s  gpu_load=20.4s  compute=0.1s  other=9.8s
```

**Token 3 and beyond (fully warm):**
```
[Vaayu] total execution time: 31.077s
[VAAYU-IO] token summary: total=31.1s  io_wait=0.9s  gpu_load=20.4s  compute=0.0s  other=9.7s

[Vaayu] total execution time: 31.241s
[VAAYU-IO] token summary: total=31.2s  io_wait=0.8s  gpu_load=20.4s  compute=0.0s  other=9.8s
```

**Before vs After summary:**

| Metric | Before | After | Change |
|---|---|---|---|
| **Per-layer time** | ~1.55s | ~0.36s | **-77%** |
| **io_wait per token** | ~88.4s | ~0.9s | **-99%** |
| **Total per-token time** | ~116.8s | ~31.1s | **-73%** |
| **NVMe throughput** | ~1.04 GB/s | ~4.4 GB/s | **+323%** |
| **Layers 45–53 anomaly** | 0.166s (cached) | 0.073s (still cached) | Normal behavior |

**Critical observation:** `io_wait=0.9s` on a warm cache run means **the NVMe I/O is no longer the dominant bottleneck**. The 31s total is now dominated by `gpu_load=20.4s`.

The `gpu_load` time is the PCIe bus transfer time — moving 1.6 GB from CPU pinned RAM into GPU VRAM for each layer. This is now the new bottleneck.

---

### 5.7 The New Bottleneck Revealed: GPU in Wrong PCIe Slot

After the I/O was fixed, the `gpu_load=20.4s` component became the primary concern. Analysis:

**Expected PCIe Gen 5 x16 bandwidth:** ~64 GB/s  
**Expected transfer time for 132 GB:** ~2 seconds total  
**Actual observed:** ~20.4 seconds total

This is a **10× discrepancy**. The calculation implies the GPU is transferring at only ~6.5 GB/s — consistent with **PCIe Gen 4 x4** or **PCIe Gen 3 x4** speeds, not PCIe 5.0 x16.

**Root cause found via `lspci -tv`:**
```bash
lspci -tv | grep -A5 "NVIDIA"
```
Output showed the GPU connected through the **Z790 chipset** on a PCIe 4.0 x4 link (via PCI_E3 slot), not directly to the CPU on the PCIe 5.0 x16 slot (PCI_E1).

**The motherboard layout:**
```
MSI Z790 Gaming Plus WiFi — PCIe slot layout:
┌──────────────────────────────────────────┐
│  M.2_1 — PCIe 4.0 x4 (FROM CPU)        │  ← Samsung 990 PRO installed here ✓
├──────────────────────────────────────────┤
│  PCI_E1 — PCIe 5.0 x16 (FROM CPU)      │  ← GPU SHOULD be here
│  (Metal-armored, topmost large slot)     │
├──────────────────────────────────────────┤
│  PCI_E2 — PCIe 3.0 x1 (tiny slot)      │
├──────────────────────────────────────────┤
│  PCI_E3 — PCIe 4.0 x4 (FROM CHIPSET)   │  ← GPU was actually here ← WRONG
├──────────────────────────────────────────┤
│  PCI_E4 — PCIe 3.0 x1 (tiny slot)      │
└──────────────────────────────────────────┘
```

**Impact of this hardware finding:**

| PCIe Slot | Bandwidth | Transfer time for 132 GB | Expected token time |
|---|---|---|---|
| PCI_E3 (current) | ~6.5 GB/s (PCIe 4.0 x4 via chipset) | ~20s | ~31s |
| PCI_E1 (correct) | ~64 GB/s (PCIe 5.0 x16 from CPU) | ~2s | **~12-14s** |

**Action required:** Move GPU from PCI_E3 to PCI_E1. After moving, verify:
```bash
nvidia-smi -q | grep -A 10 "Link Width"
# Expected:
#   Max: 16x
#   Current: 16x  (currently shows 4x)
```

**This single hardware change is projected to drop per-token time from ~31s to ~14s.**  
The software code changes (QD16 threading, double buffer) are already in place and do not need to be redone after the GPU move.

---

### 5.8 Additional Python Overhead Found: `gc.collect()` in the Loop

While analyzing the `other=9.8s` category in the token summary, a profiler showed that `gc.collect()` was being called 83 times per token — once per layer — inside the layer eviction loop. Each call takes ~17ms, and because Python's garbage collector must inspect all live objects (including large pinned tensors), the total cost was:

**~17ms × 83 layers = ~1.41 seconds of GC overhead per token wasted doing nothing useful.**

The call was defensive code meant to prevent memory leaks when layers are evicted from GPU VRAM. However, because the pinned double-buffer is now globally allocated and never freed, and because PyTorch's CUDA allocator handles GPU memory independently, the explicit `gc.collect()` is unnecessary.

**Fix:** Remove `gc.collect()` from the layer eviction loop in `vaayu_base.py`.

**Additional finding — `move_layer_to_device` using synchronous `.to("cuda")`:**

The layer transfer code was using:
```python
tensor.to("cuda")  # blocks until transfer completes
```

This should be:
```python
tensor.to("cuda", non_blocking=True)  # submits transfer, returns immediately
```

With `non_blocking=True`, the PCIe DMA engine starts copying immediately and the CPU can continue preparing the next operation. The actual synchronization happens automatically when the GPU needs the data. This allows overlap between PCIe transfer and the next layer's NVMe prefetch.

---

*[Continued in Part 3: The Rust io_uring Module Architecture + Complete Performance Timeline]*
