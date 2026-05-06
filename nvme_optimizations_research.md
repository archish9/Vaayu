# NVMe I/O Optimizations Beyond O_DIRECT

Research into software-level optimizations that can push NVMe throughput from the current **4.4 GB/s** toward the Samsung 990 PRO's rated **7.4 GB/s**.

---

## Currently Implemented ✅

| Optimization | Status | Impact |
|---|---|---|
| O_DIRECT | ✅ Deployed | Bypass page cache |
| io_uring async I/O | ✅ Deployed | Single-syscall batch submission |
| Pre-allocated pinned buffers | ✅ Deployed | Zero allocation overhead |
| Zero-copy PyO3 boundary | ✅ Deployed | Rust writes into Python buffer |
| QD32 io_uring | ✅ Deployed | Saturate NVMe internal parallelism |

---

## New Optimizations Available

### 1. 🔴 Registered Buffers (`IORING_REGISTER_BUFFERS`) — HIGH IMPACT

**What it does:** Pre-register your pinned memory buffer with the kernel so it maps the pages **once** at registration time, instead of mapping/unmapping them on every I/O operation.

**Why it helps:** Currently, for every `Read` SQE, the kernel:
1. Looks up your `buf_ptr` in the page tables
2. Pins the pages (prevents swap)
3. Sets up DMA scatter-gather list
4. After completion, unpins the pages

With registered buffers, steps 1-2-4 happen **once** at startup. Every subsequent read uses `ReadFixed` which skips the per-I/O page pinning entirely.

**Expected gain:** 5-15% throughput improvement (reduces CPU overhead per I/O, allowing higher effective QD)

**Implementation in your Rust module:**
```rust
// At init time — register the pinned buffer ONCE
let iovec = libc::iovec {
    iov_base: buf_ptr as *mut libc::c_void,
    iov_len: length,
};
ring.submitter().register_buffers(&[iovec])?;

// For each chunk — use ReadFixed instead of Read
let read_op = opcode::ReadFixed::new(
    fd,
    unsafe { ptr.add(buf_off) },
    this_len as u32,
    0  // buf_index = 0 (first registered buffer)
)
.offset(file_off)
.build();
```

**Complexity:** Low — ~15 lines of Rust changes in `lib.rs`

---

### 2. 🔴 Fixed File Descriptors (`IORING_REGISTER_FILES`) — MEDIUM IMPACT

**What it does:** Pre-register the file descriptor with io_uring so the kernel doesn't look up the fd table on every SQE submission.

**Why it helps:** Every `Read` SQE currently passes a raw fd. The kernel must:
1. Lock the file descriptor table
2. Look up the fd
3. Increment the reference count
4. After completion, decrement the reference count

With fixed files, the fd is resolved **once** at registration. Every SQE uses `types::Fixed(0)` instead of `types::Fd(raw_fd)`.

**Expected gain:** 3-8% (reduces fd lookup overhead, mostly helps at high QD)

**Implementation:**
```rust
ring.submitter().register_files(&[raw_fd])?;

// Use Fixed(0) instead of Fd(raw_fd)
let read_op = opcode::ReadFixed::new(
    types::Fixed(0),  // ← registered file index
    ...
)
```

**Complexity:** Very low — ~5 lines of Rust changes

---

### 3. 🟡 I/O Scheduler: `none` — LOW-MEDIUM IMPACT

**What it does:** Sets the Linux block I/O scheduler to pass-through mode (FIFO).

**Why it helps:** The default scheduler (`mq-deadline`) adds CPU overhead to reorder and merge I/O requests. NVMe drives handle this internally far better than the kernel can. The `none` scheduler eliminates this unnecessary software layer.

**How to check current setting:**
```bash
cat /sys/block/nvme0n1/queue/scheduler
# Output example: [mq-deadline] none kyber bfq
```

**How to apply (temporary):**
```bash
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
```

**How to apply (permanent via udev rule):**
```bash
# /etc/udev/rules.d/60-nvme-scheduler.rules
ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"
```

**Expected gain:** 2-5% throughput improvement, lower latency jitter

**Complexity:** Zero code changes — one-time system config

---

### 4. 🟡 Disable NVMe Power Management (APST) — LOW-MEDIUM IMPACT

**What it does:** Prevents the Samsung 990 PRO from entering low-power sleep states between token generations.

**Why it helps:** Between tokens (during your ~29s inference), the NVMe may briefly idle during GPU compute phases and drop into a lower power state. When the next layer read arrives, it must "wake up" — introducing **50-500μs of extra latency** per wakeup. Over 83 layers with intermittent idle gaps, this adds up.

**How to apply:**
```bash
# Add to /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nvme_core.default_ps_max_latency_us=0"

# Then:
sudo update-grub && sudo reboot
```

**Expected gain:** 1-3% (eliminates wake-up latency spikes)

**Trade-off:** Drive runs slightly hotter, draws slightly more power. Irrelevant for desktop.

**Complexity:** Zero code changes — one-time GRUB config

---

### 5. 🟡 Increase Read-Ahead (for preadv fallback path) — CONDITIONAL

**What it does:** Tells the kernel to speculatively pre-read ahead of your actual read position.

> [!WARNING]
> This does NOT help with O_DIRECT. O_DIRECT bypasses the page cache, and read-ahead is a page-cache feature. This only benefits the **preadv fallback path** (when Rust module is not loaded).

**How to apply:**
```bash
echo 4096 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb
```

**Skip this** if you're always using the io_uring O_DIRECT path.

---

### 6. 🟠 SQPOLL (`IORING_SETUP_SQPOLL`) — MEDIUM IMPACT, HIGH COMPLEXITY

**What it does:** Creates a dedicated **kernel polling thread** that continuously checks the submission queue for new SQEs. Your application never needs to call `io_uring_enter()` (the syscall) — you just write SQEs to the ring and the kernel thread picks them up.

**Why it helps:** Eliminates the `io_uring_enter()` syscall entirely. Your Rust code submits I/O by writing to shared memory — zero syscalls, zero context switches.

**Expected gain:** 5-10% for high-frequency submissions. Less impactful for your use case because you submit all SQEs in one batch anyway (the current `submit_and_wait` is already one syscall).

**Implementation:**
```rust
let mut ring = IoUring::builder()
    .setup_sqpoll(2000)  // Poll for 2000ms before sleeping
    .build(qd as u32)?;
```

**Trade-offs:**
- Requires `CAP_SYS_NICE` or root
- Dedicates one CPU core to polling (burns CPU cycles even when idle)
- Most useful for sustained, continuous I/O — less useful for burst-then-idle patterns

**My recommendation:** Skip for now. Your current batch-submit pattern (one `submit_and_wait` per layer) means you're already at one syscall per layer. SQPOLL saves one syscall per 0.35s — negligible.

---

### 7. 🔴 NVMe Passthrough (`OP_URING_CMD`) — HIGHEST IMPACT, HIGHEST COMPLEXITY

**What it does:** Bypasses the **entire Linux block layer** — no I/O scheduler, no bio/request processing, no block-layer statistics. Your application sends raw NVMe commands directly to the drive's hardware submission queues.

**Expected gain:** **16-40% higher throughput** compared to standard io_uring block-path. This is the nuclear option.

**How it works:**
```
Standard io_uring path:
  Application → io_uring → Block Layer → NVMe Driver → Hardware

NVMe Passthrough:
  Application → io_uring → NVMe Driver → Hardware  (block layer SKIPPED)
```

**Requirements:**
- Uses character device `/dev/ng0n1` instead of block device `/dev/nvme0n1p*`
- Requires raw device access (root or appropriate permissions)
- Reads are sector-addressed, not file-offset-addressed (you must know where your files physically sit on disk)
- Requires kernel ≥ 6.1 (you have 6.17 ✅)

**The catch for Vaayu:** Your model is stored as `.safetensors` files on a filesystem (ext4/btrfs). NVMe passthrough bypasses the filesystem entirely — you'd need to either:
1. **Map file offsets to LBA addresses** using `FIEMAP` ioctl, then read raw sectors
2. **Store the model on a raw partition** without a filesystem (extreme but eliminates all filesystem overhead)

**My recommendation:** This is the single highest-impact optimization available but requires significant architectural work. Consider it after exhausting options #1-4.

---

## Recommended Implementation Order

| Priority | Optimization | Gain | Effort | Risk |
|---|---|---|---|---|
| **1** | I/O Scheduler → `none` | 2-5% | 1 command | Zero |
| **2** | Disable APST | 1-3% | GRUB edit | Zero |
| **3** | Registered Buffers | 5-15% | ~15 lines Rust | Low |
| **4** | Fixed File Descriptors | 3-8% | ~5 lines Rust | Low |
| **5** | SQPOLL | 5-10% | ~3 lines Rust | Medium |
| **6** | NVMe Passthrough | 16-40% | Major refactor | High |

**Combined projected impact of #1-4:** ~15-25% throughput improvement
```
Current:  4.4 GB/s → ~5.3-5.5 GB/s (with optimizations #1-4)
Disk time: 29s → ~24-25s per token
```

**With NVMe Passthrough (#6) on top:** potentially ~6.5-7.0 GB/s → ~20s per token
