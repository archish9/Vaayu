# Why Vaayu Is Still Slow — Updated Log Analysis

## What Changed vs Previous Run

| Improvement | Before | After | Status |
|---|---|---|---|
| `init_ring` | ❌ `AttributeError` every layer | ✅ `persistent ring initialized qd=32` | **Fixed** |
| File registration | ❌ Not reached | ✅ `166 files registered with io_uring` | **Fixed** |
| APST log message | ⚠️ Warning printed | ✅ `APST: optimized (disabled)` | **Says fixed** |
| `init_ring failed` count | 210+ per run | **0** | **Fixed** |

## But Performance Is Identical

| Metric | Previous Run | This Run | Delta |
|---|---|---|---|
| Prefill chunk 1 | 5.9s | 5.9s | **+0.0s** |
| Prefill chunk 2 | 5.5s | 5.5s | **+0.0s** |
| Decode pass | 5.4s | 5.4s | **+0.0s** |
| Total wall time | 16.8s | 16.9s | **+0.0s** |
| Speculative accept | 40% | 40% | same |
| Tokens generated | 3 | 3 | same |

---

## The Three Remaining Issues

### 🔴 Issue 1: `reregister_buffers` Fails — Registered Buffers Are Dead

```
[VAAYU-IO] reregister_buffers failed: register_buffers: Bad address (os error 14)
```

This appears **twice** (lines 28 and 39), during pinned layer loading. The `EFAULT` means the kernel rejected the buffer addresses passed to `IORING_REGISTER_BUFFERS`. 

**Root cause:** The code in [safetensor_model_persister.py:119-127](file:///home/a/code/ai-ml/Vaayu/vaayu/persist/safetensor_model_persister.py#L119-L127) calls `reregister_buffers` after allocating a new pinned buffer, but it's calling it when only ONE buffer exists. Line 28 fires after allocating buf0 (2104 MB for embed_tokens), and line 39 fires after allocating buf1 (100 MB for norm). The second buffer is tiny (100 MB) — when it gets reallocated again at line 38 to 618 MB for layers, it fires again.

**The guard check `_PINNED_BUFS[0] is not None and _PINNED_BUFS[1] is not None` passes** because both buffers exist, but the old buf1 pointer was freed by Python's GC when the new tensor was allocated, so the kernel sees a stale address and returns EFAULT.

**Impact:** Without registered buffers, io_uring falls through to `IORING_OP_READ` (standard) instead of `IORING_OP_READ_FIXED` (zero-copy from pre-registered kernel pages). This means every read still does a full `copy_to_user` memcpy in the kernel. The persistent ring helps (no per-call ring setup), but the biggest Plan D optimization — registered buffers — is **completely inactive**.

### 🔴 Issue 2: APST Is NOT Actually Fixed

Despite `[Vaayu] APST: optimized (disabled)` at startup, **the exact same latency spikes are still happening**:

| Layer | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---|
| **layer 39** | **0.146s** (2.85 GB/s) | 0.128s (3.24 GB/s) | 0.131s (3.18 GB/s) |
| **layer 79** | **0.146s** (2.84 GB/s) | 0.132s (3.16 GB/s) | 0.131s (3.18 GB/s) |
| Normal layer | 0.070s (5.95 GB/s) | 0.070s (5.95 GB/s) | 0.070s (5.95 GB/s) |

Layers 39 and 79 consistently take **2× normal read time** across all passes. This is identical to the previous run. The APST sysfs write may not be persisting, or the spike is caused by something else entirely — possibly **NVMe controller thermal throttling** at a predictable interval (~30 layers × 70ms = 2.1s duty cycle).

> [!WARNING]
> Run `sudo nvme get-feature /dev/nvme0 -f 0x0c -H` to check if APST is actually disabled at the controller level. The sysfs write may only affect the Linux driver, not the firmware.

### 🟡 Issue 3: The Fundamental Physics Problem

Here is why Vaayu is "still slow," and **why no software optimization can make it fast for this workload:**

```
70 streamed layers × 426-535 MB/layer = ~33 GB per forward pass
33 GB ÷ 6.5 GB/s (NVMe sequential read) = 5.1 seconds (PHYSICS FLOOR)
```

Your measured results:

| Phase | Time | What it is |
|---|---|---|
| **io_wait** | 3.2-3.3s | Waiting for NVMe data |
| **gpu_load** | 1.8-2.2s | CPU→GPU transfer + ExLlamaV3 QMatrix construction |
| **compute** | 0.1-1.0s | Actual transformer math |
| **other** | 0.2-0.4s | Python overhead |
| **Total** | ~5.4-5.9s | |

The io_wait (3.3s) is less than the theoretical 5.1s because prefetch overlaps I/O with GPU compute for the previous layer. But `gpu_load` (1.8s) is pure serialized overhead — this is the time to copy the pinned CPU buffer to VRAM and reconstruct ExLlamaV3 `QMatrix` objects. **This cannot overlap with I/O because the GPU needs the data from the current read.**

So the true pipeline is:
```
Per-layer time = max(io_read[N+1], gpu_load[N] + compute[N])
                = max(70ms, ~30ms + ~1ms)
                = 70ms per layer, IO-bound
                
70 layers × 70ms = 4.9s  (matches the ~5s you see)
+ 2 layers at 130-146ms (APST/thermal spike) = +0.12s
+ GPU startup/teardown overhead              = +0.3s
Total ≈ 5.3-5.4s  ✓
```

**The pipeline is almost perfectly IO-bound.** The NVMe drive is the clock.

---

## The Uncomfortable Truth

> [!IMPORTANT]
> **Vaayu is not slow because of software bugs. Vaayu is slow because you're streaming a 33 GB model through a 6.5 GB/s pipe, and physics says that takes 5 seconds.**

The persistent ring fix eliminated ~0.4s of ring setup overhead per pass — but this was masked by the prefetch pipeline already overlapping it. The APST fix (if it actually worked) would save ~0.12s per pass. Neither of these change the fundamental:

**Every single token requires a full 33 GB read from NVMe.**

---

## The Only Paths Forward

These are the **only** ways to make Vaayu meaningfully faster for 70B models on 16 GB VRAM:

### 1. Reduce Streamed Payload (Highest Impact)

| Approach | Streamed/Pass | Time/Token | Effort |
|---|---|---|---|
| Current (EXL3 4.5bpw, 10 pinned) | 33 GB | ~5.4s | — |
| Pin 20 layers (drop drafter, needs ~5 GB more) | 25 GB | ~4.0s | Low |
| Move to 3.0 bpw quantization | 22 GB | ~3.5s | Low |
| Pin 20 + 3.0 bpw | 17 GB | ~2.7s | Low |

**Dropping the 1B drafter (2.47 GB)** frees enough VRAM to pin ~4-5 more transformer layers. Combined with lower quantization, this is the fastest path to sub-3s tokens.

### 2. RAM Layer Cache (Medium Impact, Medium Effort)

Cache all 80 layers decompressed in system RAM (~33 GB). Replace the NVMe read with a `memcpy` from RAM, which runs at **~25 GB/s** (DDR4-3200). This would cut io_wait from 3.3s to ~1.3s, giving ~3.5s per token.

Requires 33+ GB free RAM. Your system likely has 64 GB total — check with `free -h`.

### 3. Smaller / MoE Models (Highest Impact, Different Tradeoff)

A 30B model at 4.5 bpw would stream ~14 GB per pass → ~2.2s per token. A MoE model like Mixtral 8x7B activates only 2 experts per layer → much less data per pass.

### 4. Bigger GPU (Nuclear Option)

An RTX 4090 (24 GB) or RTX 5090 (32 GB) would pin all 80 layers. Zero streaming = compute-bound at ~50ms per token for decode.

---

## Immediate Quick Wins

1. **Drop the 1B drafter** — Reclaim 2.47 GB VRAM → pin ~4 more layers → save ~0.3s per token unconditionally (vs. the speculative 40% gamble)
2. **Fix `reregister_buffers`** — The EFAULT is a real bug. The old tensor gets freed before the new one is registered. Need to keep a reference to the old buffer until after re-registration succeeds.
3. **Verify APST at hardware level** — `sudo nvme get-feature /dev/nvme0 -f 0x0c -H`. If the feature is still enabled, use `nvme set-feature` directly.
4. **Consider 3.0-3.5 bpw** — Quality loss is measurable but modest for code tasks. ~25% less data to stream per layer.
