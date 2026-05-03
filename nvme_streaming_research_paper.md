# Deep-Dive: Unlocking Maximum NVMe Speed for LLM Layer Streaming
*A Comprehensive Analysis of the I/O Bottleneck on the i5-14600K and Samsung 990 PRO*

---

## 1. The 132GB Dilemma: Why Layer Streaming Exists

Running a 70-Billion parameter model (Llama 3.3 70B) requires approximately **132 GB** of storage for the model weights alone.

Our target system is a high-end consumer desktop:
| Component | Specification |
|---|---|
| **CPU** | Intel Core i5-14600K (6 P-Cores + 8 E-Cores = 14 Cores, 20 Threads) |
| **RAM** | 32GB DDR5 @ 6000MHz (2x16GB, Dual-Channel) |
| **GPU** | Nvidia RTX 5060 Ti (16GB VRAM) |
| **Storage** | Samsung 990 PRO 1TB NVMe (PCIe Gen 4 x4) |
| **Motherboard** | MSI Z790 GAMING PLUS WIFI |

Because 132 GB cannot physically fit into the 16GB of GPU VRAM, nor can it fit into the 32GB of system RAM, the entire model must live permanently on the NVMe SSD. To generate a single token, the system must stream all 83 layers (each ~1.6 GB) from the SSD into the GPU, process the computation, then overwrite the GPU buffer with the next layer.

**This means for every single token generated, the computer must read ~132 GB of data from the SSD.** The read speed of the NVMe drive is therefore the absolute dominant bottleneck for the entire inference pipeline.

## 2. The Advertised Speed vs. Reality

The Samsung 990 PRO box advertises sequential read speeds of **7,450 MB/s**. If it actually sustained that speed, reading 132 GB would take ~18 seconds per token. However, in our initial Python implementation, loading all 83 layers took over **100 seconds**, meaning the drive was effectively operating at around 1.3 GB/s.

To isolate whether this was a hardware problem or a software problem, we bypassed Python entirely and tested the bare-metal hardware using the Linux `dd` command:

```bash
# Clear the Linux page cache first, so we measure actual SSD speed
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Read a 1.6 GB layer file as fast as the hardware allows
dd if=model.layers.10.safetensors of=/dev/null bs=1M status=progress
```

**Result:** `1711309840 bytes (1.7 GB) copied, 0.60479 s, 2.8 GB/s`

The hardware-level maximum was **2.8 GB/s** — far below the advertised 7.4 GB/s, but also far above the 1.3 GB/s Python was achieving. This told us two things: Python was adding massive overhead, and the drive itself wasn't hitting its rated speed even outside of Python.

We also verified that the hardware link was not misconfigured:
```bash
cat /sys/class/nvme/nvme0/device/current_link_speed
# Output: 16.0 GT/s PCIe

cat /sys/class/nvme/nvme0/device/current_link_width
# Output: 4
```

This confirmed the Samsung 990 PRO was correctly running at its maximum **PCIe Gen 4 x4** physical link speed (16 GT/s × 4 lanes = theoretical 8 GB/s). The bottleneck was not the physical connection.

## 3. Understanding Queue Depth: The Real Speed Limiter

The reason a 7.4 GB/s drive only delivers 2.8 GB/s lies in a concept called **Queue Depth (QD)**.

### How NVMe drives work internally
A modern NVMe SSD like the 990 PRO contains multiple NAND flash chips arranged in parallel channels. The SSD controller can read from many chips simultaneously — but only if the host software asks it to. The number of outstanding (in-flight) read requests at any given moment is the Queue Depth.

### What happens at QD1 (standard Python behavior)
When Python reads a file using `read()` or `readinto()`, it issues a single synchronous system call. The operating system sends one read command to the SSD, and Python blocks (sleeps) until the data comes back. Only then does it send the next request.

This is **Queue Depth 1**: one request at a time. At QD1, the SSD can only activate one internal flash channel at a time. The result is a hard ceiling of approximately **2.5–3.0 GB/s** on consumer NVMe drives, regardless of their rated speed.

### How Samsung gets 7.4 GB/s
Samsung's official 7,450 MB/s specification is measured at **QD2 with multiple threads** (per their datasheet). Common benchmarking tools like CrystalDiskMark use QD8 single-thread for their "SEQ1M" test. At these higher queue depths, the SSD controller can dispatch reads to multiple flash chips simultaneously, saturating the PCIe Gen 4 x4 bus.

The key insight: **QD1 is the trap.** Python's synchronous I/O model locks us into QD1. To get more speed from the same drive, we must find a way to have multiple read requests in-flight simultaneously.

## 4. Diagnosing the Python-Side Overhead

Even compared to the 2.8 GB/s hardware baseline (0.6s per layer), Python was taking **1.55 seconds** per layer — nearly 3× slower. Profiling revealed two distinct sources of overhead:

### 4.1 The Pinned Memory Allocation Penalty (~0.5s per layer)

To transfer data from CPU RAM to GPU VRAM efficiently, PyTorch requires the source data to reside in **pinned (page-locked) memory**. Pinned memory cannot be swapped to disk by the OS, which allows the GPU to use Direct Memory Access (DMA) for fast, non-blocking transfers.

The naive implementation allocated a fresh pinned buffer for every layer:
```python
buf = torch.empty(total_size, dtype=torch.uint8, pin_memory=True)
```

This single line forces the Linux kernel to:
1. Find 1.6 GB of contiguous physical pages
2. Mark all those pages as non-swappable (`mlock`)
3. Update the page tables

Benchmarking this in isolation revealed it takes **~0.5 seconds** each time:

```
pin_memory allocate: 0.488s
pin_memory allocate: 0.361s   (warm)
unpinned allocate:   0.000s   (comparison)
```

Across 83 layers, this wastes **~40 seconds** per token just on memory management.

### 4.2 The QD1 Sequential Read (~1.0s per layer)

After allocation, the actual file read (`readinto()`) took ~1.0 seconds for a 1.6 GB layer at the QD1 baseline of ~1.6 GB/s effective throughput (lower than the `dd` baseline due to additional Python/PyTorch overhead in the read path).

### Breakdown Summary

| Phase | Time per Layer | Time per Token (83 layers) |
|---|---|---|
| Pinned memory allocation | 0.50s | 41.5s |
| Sequential file read (QD1) | 1.05s | 87.2s |
| **Total I/O** | **1.55s** | **128.7s** |

## 5. Solution Part 1: Pre-Allocated Double Buffer

The fix for the allocation overhead is straightforward: **allocate the pinned buffers once at startup and reuse them forever.**

We use a **double-buffer** architecture because Vaayu's inference engine uses a prefetch pipeline. While the GPU processes Layer N (reading from Buffer A), a background thread reads Layer N+1 from the SSD into Buffer B. On the next iteration, the roles swap. This prevents the prefetch thread from overwriting data that the GPU is still consuming.

The two buffers consume a total of ~3.4 GB of the system's 32 GB DDR5 RAM — a modest and completely safe allocation.

After implementing the global buffer, profiling confirmed the allocation overhead dropped to **zero** for all layers after the initial startup:

```
Allocating pinned buffer once...
Alloc took 0.493s            ← one-time cost at startup

Reading into preallocated buffer 5 times...
Iter 0 readinto: 0.150s      ← pure I/O, no allocation
Iter 1 readinto: 0.149s
Iter 2 readinto: 0.150s
Iter 3 readinto: 0.149s
Iter 4 readinto: 0.149s
```

## 6. Solution Part 2: Breaking the QD1 Barrier with Multi-Threaded Reads

With the allocation overhead eliminated, the remaining bottleneck was the QD1 sequential read. We needed to submit multiple read requests simultaneously, but Python's `read()` API is inherently synchronous and blocking.

### Why threads solve this

The critical property of the `os.preadv()` system call is that it **releases the Python GIL** while blocked in the kernel waiting for the I/O to complete. This means multiple Python threads can have concurrent `preadv` calls in-flight at the same time — each one holding an independent read request open in the kernel.

By splitting a 1.6 GB file into N equal chunks and dispatching N threads to read their respective chunks via `os.preadv`, we effectively create **Queue Depth N** from the NVMe driver's perspective. Each thread's `preadv` call results in an independent I/O submission to the NVMe submission queue.

This is fundamentally a **workaround for Python's lack of native asynchronous file I/O** (like `io_uring`). Each thread exists solely to hold open a blocking syscall, creating the illusion of concurrent I/O from what is otherwise a synchronous language runtime.

### The benchmark

We tested with 20 threads (matching the i5-14600K's hardware thread count as a reasonable concurrency level):

```python
from concurrent.futures import ThreadPoolExecutor
import os

fd = os.open(layer_path, os.O_RDONLY)
chunk_size = file_size // 20

def read_chunk(i):
    offset = i * chunk_size
    length = min(chunk_size, file_size - offset)
    os.preadv(fd, [pinned_view[offset:offset+length]], data_start + offset)

with ThreadPoolExecutor(max_workers=20) as executor:
    list(executor.map(read_chunk, range(20)))
```

**Results:**
| Method | Time (1.6 GB layer) | Effective Bandwidth |
|---|---|---|
| Single-thread `readinto` (QD1) | 0.598s | 2.8 GB/s |
| 20-thread `preadv` (QD20) | 0.378s | 4.4 GB/s |

The multi-threaded approach pushed the Samsung 990 PRO from 2.8 GB/s to **4.4 GB/s** — a 57% improvement in raw read bandwidth.

### Why 4.4 GB/s and not 7.4 GB/s?

The 4.4 GB/s result suggests there is still headroom. Several factors likely contribute to the gap:

1. **Thread coordination overhead:** The `ThreadPoolExecutor` has non-trivial scheduling costs. Waking 20 threads, dispatching work items, and collecting futures involves GIL acquisition on the Python side.
2. **Chunk granularity:** Each thread reads an 80 MB chunk. Smaller chunks (1–4 MB) at higher QD often perform better on NVMe controllers because they better match the internal page-group size of the flash.
3. **File fragmentation:** `filefrag` analysis revealed that the layer files have 4–12 extents (non-contiguous regions on disk). A perfectly contiguous file has 1 extent. While fragmentation matters less on SSDs than spinning disks, it can still cause the NVMe controller to issue internal seeks between flash blocks.
4. **The `io_uring` alternative:** Linux's `io_uring` interface allows a single thread to submit dozens of I/O requests to the kernel in a single syscall, completely avoiding thread creation and GIL overhead. This could potentially push bandwidth to 6+ GB/s.

## 7. How the i5-14600K's Architecture Helps

The Intel Core i5-14600K has a hybrid architecture: 6 Performance cores (P-cores) with Hyper-Threading and 8 Efficiency cores (E-cores), for 20 total hardware threads. This architecture is well-suited to the layer streaming workload:

**Thread scheduling considerations:** The 20 I/O reader threads spend most of their time blocked in kernel space waiting for the SSD. They consume very little actual CPU compute. The Linux CFS scheduler, combined with Intel's Thread Director hints on hybrid CPUs, *can* preferentially schedule low-priority sleeping threads on E-cores while keeping P-cores available for compute-heavy work (like PyTorch's CUDA dispatch thread). However, this behavior is heuristic and not guaranteed without explicit affinity control (e.g., `taskset` or `sched_setaffinity`). If deterministic core assignment is needed, explicit pinning should be configured.

**DDR5 bandwidth:** The 6000MHz DDR5 memory provides approximately 80–90 GB/s of memory bandwidth. When the NVMe drive pushes 4.4 GB/s into the pinned buffer, the RAM absorbs this effortlessly — it's operating at less than 6% of its capacity. There is zero risk of memory bandwidth becoming a bottleneck in this workload.

## 8. Disk Fragmentation Analysis

We used `filefrag` to check whether the layer files are stored contiguously on disk:

```bash
filefrag model.layers.{0,10,20,30,40,50,60,70,79}.safetensors
```

| Layer | Extents | Assessment |
|---|---|---|
| Layer 0 | 12 | Moderately fragmented |
| Layer 10 | 5 | Mildly fragmented |
| Layer 20 | 6 | Mildly fragmented |
| Layer 30 | 5 | Mildly fragmented |
| Layer 40 | 5 | Mildly fragmented |
| Layer 50 | 7 | Mildly fragmented |
| Layer 60 | 4 | Mildly fragmented |
| Layer 70 | 4 | Mildly fragmented |
| Layer 79 | 5 | Mildly fragmented |

A perfectly contiguous file has **1 extent**. Our files average ~6 extents, meaning each 1.6 GB file is split across ~6 non-adjacent physical regions. This was likely caused by writing the files to a partially-full filesystem. While NVMe SSDs handle non-sequential access far better than spinning disks, fragmentation still forces the controller to issue internal redirects between flash blocks, which can slightly reduce throughput at high queue depths.

**This is a one-time check.** The files won't fragment further unless they are rewritten or the filesystem runs critically low on space.

## 9. Combined Results

| Optimization | Per-Layer Time | Per-Token Time (83 layers) | Bandwidth |
|---|---|---|---|
| **Baseline** (naive pinned alloc + QD1 read) | 1.55s | ~129s | ~1.0 GB/s |
| **+ Global Double Buffer** (eliminate alloc) | 1.05s | ~87s | ~1.6 GB/s |
| **+ Multi-Threaded Read** (QD20) | ~0.38s | ~32s | ~4.4 GB/s |

The combined optimizations reduce per-token I/O time from **~129 seconds to ~32 seconds** — a **4× speedup** — using purely software changes with zero hardware modifications.

## 10. The Thread Count Sweet Spot

After implementing the multi-threaded reader, we ran a comprehensive sweep across thread counts to find the optimal Queue Depth:

| Threads (QD) | Avg Time per Layer | Bandwidth | Notes |
|---|---|---|---|
| 1 | 1.536s | 1.04 GB/s | Baseline — worst case |
| 2 | 0.747s | 2.13 GB/s | |
| 4 | 0.602s | 2.65 GB/s | |
| 8 | 0.419s | 3.80 GB/s | Major jump |
| 12 | 0.376s | 4.24 GB/s | |
| **16** | **0.362s** | **4.40 GB/s** | **Sweet spot** |
| 20 | 0.360s | 4.43 GB/s | Negligible gain over 16 |
| 24 | 0.362s | 4.40 GB/s | Diminishing returns |
| 32 | 0.361s | 4.42 GB/s | No improvement |

**QD16 delivers 99.3% of peak bandwidth.** Beyond 16 threads, the NVMe controller's internal parallelism is fully saturated and additional threads only add scheduling overhead. We chose 16 threads for the production implementation.

## 11. Live Inference Results — NVMe I/O is Solved

After deploying both the double-buffered pinned memory and the QD16 multi-threaded reader, we ran a full inference session (7 tokens) and examined the per-token timing breakdown:

```
[VAAYU-IO] token summary: total=31.2s  io_wait=0.9s  gpu_load=20.4s  compute=0.1s  other=9.8s
```

| Phase | Time | % of Total | Status |
|---|---|---|---|
| `gpu_load` (CPU→GPU PCIe transfer) | **20.4s** | 66% | 🔴 New dominant bottleneck |
| `other` (framework + cleanup overhead) | 9.8s | 31% | 🟡 Partially fixable |
| `io_wait` (NVMe read — blocked on disk) | **0.9s** | 3% | ✅ **Solved** |
| `compute` (actual GPU math) | 0.1s | <1% | ✅ Optimal |

**The NVMe I/O bottleneck is effectively eliminated.** The `io_wait` of 0.9 seconds means the prefetch pipeline successfully overlaps almost all disk I/O with GPU work. Out of 83 layers × 0.36s each = ~30s of total disk reads, only 0.9s was time the GPU actually had to *wait* for data.

But the per-token time is still 31 seconds, and now **66% of that time is spent transferring data from CPU RAM to GPU VRAM** over the PCIe bus.

## 12. The PCIe Slot Discovery: A 15-Second Hardware Bug

### The Investigation

With NVMe I/O solved, we profiled the GPU transfer path. A 1.6 GB layer transfer from pinned CPU memory to GPU VRAM was benchmarked:

```python
buf = torch.empty(1632 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
gpu = buf.to('cuda:0')
torch.cuda.synchronize()
# Result: 0.250s  (6.4 GB/s)
```

**6.4 GB/s** for a CPU→GPU transfer seemed low. The RTX 5060 Ti supports PCIe Gen 5 x16 (theoretical ~63 GB/s). Even at Gen 4 x16, practical throughput should be ~25 GB/s. We queried `nvidia-smi`:

```
PCIe Generation
    Max                  : 4        ← Motherboard caps at Gen 4
    Current              : 1        ← Power-saving idle speed
    Device Max           : 5        ← GPU supports Gen 5
Link Width
    Max                  : 16x      ← GPU supports x16
    Current              : 4x       ← ❌ Only 4 lanes active!
```

The GPU was reporting **x4 link width** — not x16. Even under sustained load, the width stayed at x4. This means the GPU was physically limited to 4 PCIe lanes instead of the 16 it was designed for.

### Root Cause: Wrong Physical Slot

We examined the PCI bus topology:

```
lspci -tv

-[0000:00]
  +-06.0-[01]----00.0  Samsung NVMe SSD            ← CPU-direct (Gen4 x4) ✅
  +-1c.4-[04]--+-00.0  NVIDIA RTX 5060 Ti          ← Chipset bridge ❌
```

The GPU was connected through **bridge `00:1c.4`** — a Z790 chipset PCIe root port. Checking this bridge's capabilities:

```bash
cat /sys/bus/pci/devices/0000:00:1c.4/max_link_width
# Output: 4
```

The bridge itself is **hardware-limited to 4 lanes**. The GPU physically cannot negotiate x16 through this bridge, regardless of any software or BIOS setting.

Cross-referencing with the MSI Z790 GAMING PLUS WIFI motherboard layout:

```
┌─────────────────────────────────────────┐
│              CPU Socket (i5-14600K)     │
├─────────────────────────────────────────┤
│  M.2_1 (Samsung 990 PRO — CPU direct)  │
├═════════════════════════════════════════╡
│  PCI_E1 — PCIe 5.0 x16 (FROM CPU)     │  ← CORRECT slot for GPU
│  (Metal-armored, topmost large slot)    │
├─────────────────────────────────────────┤
│  PCI_E2 — PCIe 3.0 x1                 │
├─────────────────────────────────────────┤
│  PCI_E3 — PCIe 4.0 x4 (FROM CHIPSET)  │  ← GPU was installed HERE
├─────────────────────────────────────────┤
│  PCI_E4 — PCIe 3.0 x1                 │
└─────────────────────────────────────────┘
```

The GPU was installed in **PCI_E3**, a chipset-routed x4 slot designed for add-in cards like capture cards or RAID controllers — not graphics cards. The correct slot is **PCI_E1**, which connects directly to the CPU's 16 PCIe 5.0 lanes.

### The Impact

| Configuration | Bandwidth | gpu_load (83 layers) | Total Token Time |
|---|---|---|---|
| PCI_E3 (Gen4 x4) — current | 6.4 GB/s | 20.4s | ~31s |
| PCI_E1 (Gen4 x16) — correct | ~25 GB/s | ~5.4s | **~16s** |

The math: 83 layers × 1.632 GB = 135.5 GB total transfer per token.
- At 6.4 GB/s: 135.5 / 6.4 = 21.2s (measured: 20.4s — operating at **96% of the ceiling**)
- At 25 GB/s: 135.5 / 25 = 5.4s

**Moving the GPU to PCI_E1 would save ~15 seconds per token — more than all software optimizations combined.**

*Note: The link will negotiate Gen 4 (not Gen 5) because the i5-14600K's PCIe 5.0 lanes are limited by motherboard trace quality and BIOS defaults. Gen 4 x16 at ~25 GB/s is the realistic expectation.*

## 13. The `clean_memory()` Overhead: Death by a Thousand GC Calls

### The Problem

After every single layer's forward pass, the inference loop calls:

```python
layer.to("meta")         # Move parameters to meta device (free GPU memory)
layer.to("meta")         # Duplicate call (bug)
clean_memory()           # gc.collect() + malloc_trim() + cuda.empty_cache()
```

This `clean_memory()` function runs **83 times per token**:

```python
def clean_memory():
    gc.collect()                          # Scan every Python object for reference cycles
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # Return freed pages to the OS
    torch.cuda.empty_cache()              # Release cached GPU memory blocks
```

### Benchmarking the Cost

```
clean_memory() × 83 = 1.472s
  gc.collect() × 83 = 1.454s     ← 99% of the cost
  cuda.empty_cache() × 83 = 0.000s
  malloc_trim() × 83 = 0.018s
```

`gc.collect()` is a full garbage collector sweep. It traverses every Python object in the process looking for reference cycles. PyTorch tensors use reference counting and rarely create cycles — calling `gc.collect()` 83 times per token is almost entirely wasted work.

The duplicate `layer.to("meta")` call is a separate bug — the second call is a no-op but still iterates all parameters.

### The Fix

Reduce GC frequency from every layer to every 20 layers:

```python
layer.to("meta")                # Keep this — frees GPU memory
if i % 20 == 19:
    clean_memory()              # Full GC only every 20 layers
```

**Expected saving: ~1.3 seconds per token.**

## 14. The `move_layer_to_device` Transfer Path

The function that transfers each layer's tensors from CPU to GPU currently uses synchronous transfers:

```python
# Current code (line 508)
standard_params[param_name] = remapped_state_dict[param_name].to(
    self.running_device, dtype=self.running_dtype
)
```

Each `.to(device)` call blocks Python until the DMA transfer completes. For a typical layer with 9 tensors:

| Tensor | Shape | Size |
|---|---|---|
| `mlp.down_proj.weight` | [8192, 28672] | 448 MB |
| `mlp.gate_proj.weight` | [28672, 8192] | 448 MB |
| `mlp.up_proj.weight` | [28672, 8192] | 448 MB |
| `self_attn.q_proj.weight` | [8192, 8192] | 128 MB |
| `self_attn.o_proj.weight` | [8192, 8192] | 128 MB |
| `self_attn.k_proj.weight` | [1024, 8192] | 16 MB |
| `self_attn.v_proj.weight` | [1024, 8192] | 16 MB |
| `input_layernorm.weight` | [8192] | 0.016 MB |
| `post_attention_layernorm.weight` | [8192] | 0.016 MB |
| **Total** | | **1,632 MB** |

Using `non_blocking=True` (which requires pinned source memory — which we now have) allows CUDA to pipeline the transfers:

```python
# Optimized
standard_params[param_name] = remapped_state_dict[param_name].to(
    self.running_device, dtype=self.running_dtype, non_blocking=True
)
```

Benchmarked improvement: ~10% per-layer transfer time reduction (0.264s → 0.241s). Modest, but it adds up across 83 layers.

## 15. Future Work: `io_uring` for Maximum NVMe Throughput

The current multi-threaded approach using `os.preadv` + `ThreadPoolExecutor` is a workaround for Python's lack of native async file I/O. The Linux kernel provides a superior interface: **`io_uring`**.

`io_uring` allows a single thread to:
1. Prepare dozens of read requests in a userspace submission queue
2. Submit them all to the kernel with a single `io_uring_enter()` syscall
3. Harvest completions from a shared completion ring without any additional syscalls

This eliminates:
- Thread creation and destruction overhead
- GIL contention during thread wakeup
- `ThreadPoolExecutor` scheduling latency
- Context switching between 16+ threads

**Current status:** The Python binding (`liburing`) crashes with "Illegal Instruction" on the i5-14600K — the Zig-compiled backend uses CPU extensions this processor doesn't support. Building from source with `-march=native` may resolve this. An `io_uring`-based loader could potentially push read bandwidth from 4.4 GB/s toward **6+ GB/s**.

## 16. Full Optimization Summary

| Optimization | Type | Per-Token Impact | Status |
|---|---|---|---|
| Pre-allocated double-buffered pinned memory | Software | -41.5s | ✅ Deployed |
| Multi-threaded QD16 `preadv` reader | Software | -55s | ✅ Deployed |
| GPU in PCI_E1 slot (Gen4 x16) | Hardware | **-15s** | 🔧 Pending |
| Reduce `gc.collect()` frequency | Software | -1.3s | 🔧 Pending |
| `non_blocking=True` GPU transfers | Software | -2s | 🔧 Pending |
| Remove duplicate `layer.to("meta")` | Software (bug fix) | -0.2s | 🔧 Pending |
| `io_uring` NVMe reader | Software | TBD | 🔬 Experimental |

### End-to-End Timeline

| Stage | Per-Token Time | Primary Bottleneck |
|---|---|---|
| **Original** (naive implementation) | ~129s | NVMe read at QD1 + pinned alloc |
| **After NVMe optimizations** | ~31s | GPU PCIe transfer (Gen4 x4) |
| **After GPU slot fix** (projected) | ~16s | GPU PCIe transfer (Gen4 x16) |
| **After all software fixes** (projected) | **~14s** | PCIe bandwidth ceiling |

The journey from 129 seconds to a projected 14 seconds per token represents a **9× speedup** through a combination of systems-level software engineering and hardware configuration — all without modifying the model weights, changing precision, or sacrificing output quality.
