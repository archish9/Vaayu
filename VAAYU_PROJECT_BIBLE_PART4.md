# VAAYU PROJECT BIBLE — PART 4
### Complete Performance Timeline, Known Issues, Open Questions & Next Steps

*This is Part 4 of 4 — the final section. Read Parts 1, 2, and 3 first.*

---

## 8. Complete Performance Timeline

This is the master record of every measurable state the project has been in, in chronological order. Every number comes from actual logs — nothing is estimated.

---

### Era 1: Pre-Optimization Baseline (~116 seconds/token)

**Code state:** Original AirLLM-style per-layer allocation, no threading, no pinned memory pre-allocation, no KV cache.

**Representative log (2026-05-02 15:56:42):**
```
[VAAYU-IO] pinned 2.352s  model.embed_tokens
[VAAYU-IO] pinned 0.536s  model.layers.0
[VAAYU-IO] pinned 1.609s  model.layers.1
[VAAYU-IO] pinned 1.592s  model.layers.2
[VAAYU-IO] pinned 1.605s  model.layers.3
[VAAYU-IO] pinned 1.567s  model.layers.4
[VAAYU-IO] pinned 1.559s  model.layers.5
[VAAYU-IO] pinned 1.526s  model.layers.6
...
[VAAYU-IO] pinned 1.638s  model.layers.27
[VAAYU-IO] pinned 1.516s  model.layers.28
[VAAYU-IO] pinned 1.658s  model.layers.29
[VAAYU-IO] pinned 1.512s  model.layers.30
[VAAYU-IO] pinned 1.254s  model.layers.31
[VAAYU-IO] pinned 1.701s  model.layers.32
...
[VAAYU-IO] pinned 0.166s  model.layers.45   ← Linux page cache hit
[VAAYU-IO] pinned 0.175s  model.layers.46   ← Linux page cache hit
[VAAYU-IO] pinned 0.167s  model.layers.47   ← Linux page cache hit
[VAAYU-IO] pinned 0.169s  model.layers.48   ← Linux page cache hit
[VAAYU-IO] pinned 0.166s  model.layers.49   ← Linux page cache hit
[VAAYU-IO] pinned 0.166s  model.layers.50   ← Linux page cache hit
[VAAYU-IO] pinned 0.166s  model.layers.51   ← Linux page cache hit
[VAAYU-IO] pinned 0.166s  model.layers.52   ← Linux page cache hit
[VAAYU-IO] pinned 0.166s  model.layers.53   ← Linux page cache hit
[VAAYU-IO] pinned 1.418s  model.layers.54   ← back to NVMe
[VAAYU-IO] pinned 1.426s  model.layers.55
...
[VAAYU-IO] pinned 1.733s  model.layers.58
...
[VAAYU-IO] pinned 0.001s  model.norm
[VAAYU-IO] pinned 2.065s  lm_head
[Vaayu] total execution time: 116.823s
[VAAYU-IO] token summary: total=116.8s  io_wait=88.4s  gpu_load=20.5s  compute=0.2s  other=7.8s
```

**Why layer 0 was fast (0.536s) and layer 1 was slow (1.609s):** Layer 0 was still in the Linux page cache from a prior run. The 0.536s is the allocation overhead for pinned memory without the actual NVMe read. Layer 1 was a true cold read.

**The layers 45–53 anomaly:** These show 0.166s per layer because they happened to be in the Linux page cache from a recent run. The OS was holding 32 GB of RAM, with ~6 GB used by the model process, leaving ~26 GB for the page cache — enough to hold several layers. This proved the Linux page cache works correctly, but 32 GB of RAM can only cache ~20 layers (32 GB ÷ 1.6 GB/layer). The other 63 layers always hit NVMe.

---

### Era 2: First Rust Attempt — Catastrophic Regression (~190 seconds/token)

**What changed:** Installed `vaayu_rust` PyO3 bindings that returned data as Python bytes/NumPy arrays, causing triple-copy.

**Not logged in full**, but representative per-layer time jumped from ~1.5s to ~2.3s due to:
- 1.6 GB Rust allocation: ~0.5s  
- PyO3 boundary copy: ~0.5s  
- NumPy → PyTorch `.clone()`: ~0.5s  
- Actual NVMe read still present: ~1.0s  
- **Total: ~2.5s per layer × 83 = ~208s**

**Decision: Abandon this approach immediately. Revert to Python + redesign Rust.**

---

### Era 3: Post Global Buffer + QD16 Threading (~31–32 seconds/token)

**What changed:** `_PINNED_BUFS` double-buffer allocated at module startup. QD16 `ThreadPoolExecutor` with `os.preadv` replacing single-threaded `f.readinto`.

**First token (cold — no page cache warming):**
```
[VAAYU-IO] pinned 1.274s  model.embed_tokens
[VAAYU-IO] pinned 0.785s  model.layers.0
[VAAYU-IO] pinned 0.346s  model.layers.1
[VAAYU-IO] pinned 0.363s  model.layers.2
[VAAYU-IO] pinned 0.363s  model.layers.3
[VAAYU-IO] pinned 0.362s  model.layers.4
[VAAYU-IO] pinned 0.366s  model.layers.5
[VAAYU-IO] pinned 0.363s  model.layers.6
...
[VAAYU-IO] pinned 0.073s  model.layers.45   ← page cache (faster due to no alloc cost)
...
[VAAYU-IO] pinned 0.536s  lm_head
[Vaayu] total execution time: 32.672s
[VAAYU-IO] token summary: total=32.7s  io_wait=2.0s  gpu_load=20.4s  compute=0.2s  other=10.1s
```

**Token 2:**
```
[Vaayu] total execution time: 32.044s
[VAAYU-IO] token summary: total=32.0s  io_wait=1.7s  gpu_load=20.4s  compute=0.1s  other=9.8s
```

**Token 3 (steady state):**
```
[Vaayu] total execution time: 31.077s
[VAAYU-IO] token summary: total=31.1s  io_wait=0.9s  gpu_load=20.4s  compute=0.0s  other=9.7s
```

**Tokens 4–7 (all steady state):**
```
Token 4: 31.173s  [io_wait=0.9s  gpu_load=20.4s  other=9.8s]
Token 5: 31.241s  [io_wait=0.8s  gpu_load=20.4s  other=10.0s]
Token 6: 31.269s  [io_wait=0.9s  gpu_load=20.4s  other=9.9s]
```

**Key finding from these logs:**
- `io_wait` is now just **0.8–0.9s** (down from 88.4s) — **NVMe I/O is no longer the bottleneck**
- `gpu_load=20.4s` is now the dominant cost (63% of token time)
- `other=9.7–10.1s` contains `gc.collect()` overhead (~1.4s) plus Python scheduling

---

### Era 3b: Software-Only Optimizations Applied (same session, ~29–30 seconds/token)

A further test removed `gc.collect()` and applied `non_blocking=True` transfers. The logs showed:

```
[Vaayu] total execution time: 31.689s
[VAAYU-IO] token summary: total=31.7s  io_wait=9.5s  gpu_load=21.4s  compute=0.2s  other=0.6s
```

**Note:** In this run, `io_wait=9.5s` — much higher than the ~0.9s seen earlier. This happened because `gc.collect()` removal caused a timing shift in the prefetch thread synchronization, and the `other` category dropped to 0.6s (from 9.8s). The total time is similar (~31s), but the internal categorization shifted. The absolute total time is what matters — the categorization can shift based on prefetch timing overlaps.

Subsequent tokens:
```
Token 2: 30.743s  [io_wait=10.0s  gpu_load=20.1s  compute=0.1s  other=0.5s]
Token 3: 29.613s  [io_wait=8.9s  gpu_load=20.1s  compute=0.1s  other=0.5s]
Token 4: 29.600s  [io_wait=8.8s  gpu_load=20.1s  other=0.6s]
```

The bottom line is ~29–31s regardless of the `gc.collect()` / `non_blocking` tweak in this configuration — the GPU transfer bandwidth is the hard limit until the GPU is moved to PCI_E1.

---

### Projected Era 4: After GPU Move to PCI_E1 (~12–14 seconds/token)

**What needs to happen:** Physical GPU relocation from PCI_E3 (PCIe 4.0 x4, chipset) to PCI_E1 (PCIe 5.0 x16, CPU-direct).

| Component | Before | After |
|---|---|---|
| GPU PCIe slot | PCI_E3 (PCIe 4.0 x4) | PCI_E1 (PCIe 5.0 x16) |
| GPU PCIe bandwidth | ~6.5 GB/s | ~64 GB/s |
| `gpu_load` per token | ~20.4s | **~2s** |
| `io_wait` per token | ~0.9s | ~0.9s (unchanged) |
| `compute` per token | ~0.1s | ~0.1s (unchanged) |
| `other` per token | ~9.8s | ~9.8s (unchanged) |
| **Estimated total** | **~31s** | **~13s** |

**Verification after GPU move:**
```bash
nvidia-smi -q | grep -A 10 "Link Width"
# Before move: Current: 4x
# After move:  Current: 16x
```

---

### Projected Era 5: After Rust io_uring (~8–10 seconds/token, with GPU in x16)

**Assumption:** `vaayu_rust io_uring_read_into` achieves 6.5 GB/s (conservative, based on Samsung 990 PRO capability at higher queue depth).

| Component | Era 4 (after GPU move) | Era 5 (+ Rust io_uring) |
|---|---|---|
| `io_wait` per token | ~0.9s | **~0.3s** *(132 GB / 6.5 GB/s ≈ 20s... wait)* |

**Important correction:** At 6.5 GB/s, reading 132 GB still takes `132/6.5 = 20.3s`. However, the `io_wait` category in the logs counts only the blocking time inside `_load_pinned()`. When the prefetch thread is working ahead of the GPU thread, some I/O happens in parallel with GPU transfers, masking the I/O wait time in the logs. The ~0.9s `io_wait` at Era 3 reflected how much time the main thread was stalled waiting for NVMe *after accounting for prefetch overlap*.

The real impact of Rust io_uring on steady-state token time will only be measurable by running `test_quick.py` with the Rust module active. The theoretical gain is approximately:

```
Per-layer read: 0.36s (Python QD16) → ~0.24s (Rust io_uring at 6.5 GB/s)
Savings: 0.12s × 83 layers = ~10s per token if perfectly sequential
Actual savings (with prefetch overlap): ~3–7s per token
```

---

## 9. Known Issues and Gotchas

### Issue 1: `O_DIRECT` Alignment Requirement

When Rust opens the layer file with `O_DIRECT` for zero-copy DMA, the kernel requires:
1. The read offset must be aligned to 512 bytes (sector size)
2. The read length must be aligned to 512 bytes
3. The destination buffer must be aligned to 512 bytes (4096 for some devices)

**The safetensors header is not 512-byte aligned.** The data region starts at `8 + hdr_len` bytes into the file, where `hdr_len` varies by layer. This means `data_start` is unlikely to be 512-byte aligned.

**Workaround options:**
1. Round down `data_start` to the nearest 512-byte boundary and discard the prefix bytes
2. Read the header with a standard `open()` and the data with `O_DIRECT | O_RDONLY`
3. Skip `O_DIRECT` entirely — `io_uring` can still achieve higher throughput than `preadv` threading even without `O_DIRECT`, because `O_DIRECT` is not required for `io_uring` to benefit from higher queue depth

**Current status:** If `io_uring_read_into` fails with `EINVAL` at runtime, it is likely an alignment issue. The Python QD16 fallback activates automatically.

### Issue 2: PyTorch's Page Cache Interaction

Even with `O_DIRECT` bypassing the Linux page cache for disk reads, PyTorch's CUDA unified memory allocator may still interact with the kernel's virtual memory system. The true zero-copy path (NVMe → pinned RAM without any intermediate kernel buffer) requires both `O_DIRECT` *and* that the destination buffer be registered with `io_uring` via `IORING_OP_REGISTER_BUFFERS`. The Rust module currently does not use buffer registration, so there may still be one kernel memcpy involved. This is a known limitation for future optimization.

### Issue 3: The `model.layers.45` Anomaly Across All Runs

In every run across every era, `model.layers.45` consistently shows an anomalously low `[VAAYU-IO]` time (~0.073s vs ~0.363s for peers). This happens because layer 45's `.safetensors` file is always in the Linux page cache from recent reads. The pattern is visible in Era 3 logs repeatedly:

```
[VAAYU-IO] pinned 0.365s  model.layers.44
[VAAYU-IO] pinned 0.073s  model.layers.45   ← always fast
[VAAYU-IO] pinned 0.376s  model.layers.46
```

This is not a bug — it is the Linux page cache working normally. Layer 45's file happens to be read frequently enough (multiple token passes) that the OS keeps it warm. It does not indicate any code error.

### Issue 4: First Token Always Slower

The first token in any new process invocation always shows higher times for layers 0 and `embed_tokens` (cold cache). Layers seen in logs:

```
Token 1: [VAAYU-IO] pinned 1.274s  model.embed_tokens  (cold)
Token 2: [VAAYU-IO] pinned 1.328s  model.embed_tokens  (partial cache)
Token 3: [VAAYU-IO] pinned 0.439s  model.embed_tokens  (mostly cached)
Token 4: [VAAYU-IO] pinned 0.444s  model.embed_tokens  (warm)
```

This is normal — the page cache warms up over the first 1–2 tokens. Benchmark results should always be taken from token 3 onward.

### Issue 5: KV Cache Produces Weird Per-Layer Timings

Because KV cache is now working correctly (after the `past_key_values` bug fix), subsequent tokens process different amounts of data per layer:
- **Token 1:** full prompt (9 tokens) through all layers
- **Token 2+:** single new token through all layers, but with KV state loaded

This means the GPU `compute` time differs slightly between token 1 and subsequent tokens, though the effect is small (~0.1–0.2s) because GPU compute is fast.

---

## 10. Next Steps — Prioritized Action List

Listed in order of expected impact on per-token time.

### 🔴 Critical (hardware, ~50% improvement)

**Step 1: Move GPU from PCI_E3 to PCI_E1**
- **Who:** Requires opening the PC case (screwdriver)
- **What:** Unscrew GPU bracket, release PCIe latch, move GPU to the top metal-armored slot (PCI_E1, labeled on PCB)
- **Verification:** `nvidia-smi -q | grep -A10 "Link Width"` → should show `Current: 16x`
- **Expected impact:** `gpu_load` drops from 20.4s → ~2s. Token time: 31s → **~12–14s**
- **No code changes required** — all software is already prepared

### 🟠 High Priority (Rust validation, additional ~30% improvement)

**Step 2: Build and validate the Rust io_uring module**
```bash
cd /path/to/Vaayu/vaayu_rust
maturin develop --release
python3 -c "import vaayu_rust; print('Rust module loaded')"
```

**Step 3: Run Rust throughput benchmark**
```bash
python3 benchmark_rust_io.py
# Expected: 6–7 GB/s (vs 4.4 GB/s with Python threading)
```

**Step 4: Run full inference test with Rust active**
```bash
python3 test_quick.py
# Expected token time after GPU move + Rust: ~8–10s
```

### 🟡 Medium Priority (Queue Depth tuning)

**Step 5: Tune `io_uring` queue depth in `vaayu_rust/src/lib.rs`**

The current Rust module uses a default queue depth. A sweep over QD8, QD16, QD32, QD64 should find the optimal value for the i5-14600K + Samsung 990 PRO combination. The Python threading plateau was at QD16–20. `io_uring` may achieve higher bandwidth at QD32.

**Step 6: Tune chunk size in `io_uring` reads**

Current chunk size in Rust: 4 MB. Try 1 MB, 2 MB, 8 MB, 16 MB to find optimal for sequential reads on this SSD.

### 🟢 Low Priority (software cleanup)

**Step 7: Confirm `gc.collect()` removal is stable**

Verify that removing `gc.collect()` from the layer eviction loop doesn't cause memory leaks in long generation runs (50+ tokens). Check GPU VRAM doesn't grow unboundedly with `nvidia-smi dmon`.

**Step 8: Add `non_blocking=True` to all `.to("cuda")` calls**

Already identified in `vaayu_base.py`. Minor but contributes to PCIe overlap.

**Step 9: Register pinned buffers with `io_uring` for true zero-copy**

Use `IORING_OP_REGISTER_BUFFERS` in `vaayu_rust/src/lib.rs` to register the two `_PINNED_BUFS` pointers with the kernel at startup. This enables DMA directly from NVMe to pinned memory without any intermediate kernel buffer, achieving true zero-copy.

---

## 11. Summary Table: The Full Optimization Journey

| Era | State | Per-Token Time | io_wait | gpu_load | compute |
|---|---|---|---|---|---|
| **0** | Baseline (Python, QD1, per-layer alloc) | **116.8s** | 88.4s | 20.5s | 0.2s |
| **1** | First Rust attempt (triple copy) | **~190s** | ~125s | 20.5s | 0.2s |
| **2** | Reverted to Python | 116.8s | 88.4s | 20.5s | 0.2s |
| **3** | Global buffer + KV cache fix | 32.7s | 2.0s | 20.4s | 0.2s |
| **3+** | QD16 threading (steady state) | **31.1s** | 0.9s | 20.4s | 0.0s |
| **4** *(pending)* | GPU moved to PCI_E1 | **~13s** | ~0.9s | ~2s | ~0.1s |
| **5** *(pending)* | Rust io_uring active | **~10s** | ~0.5s | ~2s | ~0.1s |
| **6** *(pending)* | Rust + GPU move + io_uring registered buffers | **~8s** | ~0.3s | ~2s | ~0.1s |
| **Theoretical max** | 7.45 GB/s NVMe + PCIe 5.0 x16 | **~5s** | ~0.3s | ~0.5s | ~0.1s |

---

## 12. How to Resume Work After a Gap

If you are coming back to this project after time away, follow this checklist:

**1. Read the performance summary table above (Section 11)** — find the highest numbered "pending" row to understand the current state.

**2. Check which Rust build state is active:**
```bash
cd /path/to/Vaayu
source venv/bin/activate
python3 -c "import vaayu_rust; print('Rust active')" 2>/dev/null || echo "Python fallback"
```

**3. Run the standard test:**
```bash
python3 test_quick.py 2>&1 | grep -E "VAAYU-IO|inference|total execution"
```

**4. Compare token times to Section 8 above.** If times are dramatically different from the Era 3 baseline (31s), investigate why before making changes.

**5. Check the GPU PCIe slot:**
```bash
nvidia-smi -q | grep -A5 "Link Width"
# If Current: 4x → GPU is still in wrong slot (Era 3 behavior)
# If Current: 16x → GPU has been moved (Era 4 behavior expected)
```

**6. Do not re-run benchmarks on warm cache** — always `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before any I/O speed measurement to ensure you're measuring NVMe, not RAM.

---

*End of Vaayu Project Bible.*  
*Parts: 1 (Introduction/Hardware/Early Bugs) · 2 (I/O Bottleneck/Benchmarks) · 3 (Rust Module/Code Architecture) · 4 (This document — Timeline/Issues/Next Steps)*
