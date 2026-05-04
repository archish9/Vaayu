# VAAYU PROJECT BIBLE — PART 3
### The Rust io_uring Module + Code Architecture Deep Dive

*This is Part 3 of 4. Read Parts 1 and 2 first.*

---

## 6. The Rust `vaayu_rust` Module — Zero-Copy Architecture

After the Python threading approach plateaued at 4.4 GB/s (leaving ~3 GB/s of headroom to the drive's theoretical maximum), the decision was made to implement a production-grade `io_uring`-based reader in Rust. The Python `liburing` bindings were crashing (`Illegal instruction`), but a native Rust implementation using the `io-uring` crate does not have that limitation.

---

### 6.1 Design Philosophy: Rust Orchestrates, Python Owns

The critical lesson from the failed first Rust attempt (Section 4.3 in Part 1) was that **Rust must not own any data**. The moment Rust allocates a `Vec<u8>`, reads into it, and returns it across the PyO3 boundary to Python, three full copies of the 1.6 GB layer data are made, negating any speed benefit.

**The correct zero-copy design:**

```
Python allocates pinned buffer (via torch.empty(..., pin_memory=True))
         ↓
Python passes the raw memory pointer to Rust
         ↓
Rust submits io_uring read operations targeting that pointer directly
         ↓
Linux io_uring DMA writes data from NVMe directly into the pinned buffer
         ↓
Rust returns (no data moved across the PyO3 boundary)
         ↓
Python slices tensors out of the pinned buffer it already owns
```

At no point does data cross the PyO3 boundary. Rust is purely an orchestration layer — it manages the submission ring, file descriptors, and completion events, but never touches the data itself.

---

### 6.2 Rust Module: `vaayu_rust/src/lib.rs`

**Cargo.toml dependencies:**
```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
io-uring = "0.6"

[lib]
name = "vaayu_rust"
crate-type = ["cdylib"]
```

**Key exposed functions:**

```rust
use pyo3::prelude::*;
use io_uring::{IoUring, opcode, types};

/// Zero-copy read using io_uring.
/// buf_ptr: raw pointer to PyTorch pinned memory (page-aligned)
/// length: bytes to read
/// offset: byte offset within the file
#[pyfunction]
pub fn io_uring_read_into(
    fd: i32,
    buf_ptr: usize,   // Python passes int(buf.data_ptr())
    length: usize,
    offset: u64,
) -> PyResult<i32> {
    let mut ring = IoUring::new(32)?;
    
    // Safety: Python guarantees buf_ptr is a valid pinned memory allocation
    // of at least `length` bytes
    let buf = unsafe {
        std::slice::from_raw_parts_mut(buf_ptr as *mut u8, length)
    };
    
    // Split into chunks for QD > 1
    let chunk_size = 1024 * 1024 * 4;  // 4 MB chunks → QD = length/chunk_size
    let mut submitted = 0usize;
    let mut completed = 0usize;
    let mut file_offset = offset;
    
    while completed < length {
        // Fill the submission queue
        while submitted - completed < 32 && file_offset < offset + length as u64 {
            let this_chunk = chunk_size.min((offset + length as u64 - file_offset) as usize);
            let buf_start = (file_offset - offset) as usize;
            
            let read_e = opcode::Read::new(
                types::Fd(fd),
                buf[buf_start..].as_mut_ptr(),
                this_chunk as u32,
            )
            .offset(file_offset)
            .build()
            .user_data(file_offset);
            
            unsafe { ring.submission().push(&read_e)?; }
            ring.submit()?;
            file_offset += this_chunk as u64;
            submitted += this_chunk;
        }
        
        // Drain completions
        ring.submit_and_wait(1)?;
        for cqe in ring.completion() {
            completed += cqe.result() as usize;
        }
    }
    
    Ok(length as i32)
}

/// Synchronous pread fallback (for kernels / systems where io_uring fails)
#[pyfunction]
pub fn pread_into(fd: i32, buf_ptr: usize, length: usize, offset: u64) -> PyResult<i32> {
    use std::os::unix::io::FromRawFd;
    // Uses pread(2) syscall directly — safer fallback, ~QD1 speed
    ...
}
```

**Python-side integration in `safetensor_model_persister.py`:**

```python
# Try to import Rust module at startup
try:
    import vaayu_rust
    _RUST_AVAILABLE = True
    print(">>>> vaayu_rust io_uring module loaded")
except ImportError:
    _RUST_AVAILABLE = False
    print(">>>> vaayu_rust not available, using Python threading fallback")

def _load_layer_fast(path: str, buf_idx: int) -> dict:
    buf = _PINNED_BUFS[buf_idx]
    
    # Parse safetensors header
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    
    total_size = max(v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__')
    view = memoryview(buf.numpy())
    
    if _RUST_AVAILABLE:
        # Zero-copy path: Rust io_uring writes directly into pinned buf
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        vaayu_rust.io_uring_read_into(fd, buf.data_ptr(), total_size, data_start)
        os.close(fd)
    else:
        # Python fallback: QD16 threading
        fd = os.open(path, os.O_RDONLY)
        _preadv_chunked(fd, view[:total_size], data_start, total_size, n_threads=16)
        os.close(fd)
    
    # Slice tensors from buffer (no copy — just views into pinned memory)
    tensors = {}
    for name, meta in hdr.items():
        if name == '__metadata__':
            continue
        dtype = _DTYPE_MAP[meta['dtype']]
        start, end = meta['data_offsets']
        tensor_bytes = buf[start:end]
        tensors[name] = tensor_bytes.view(dtype).reshape(meta['shape'])
    
    return tensors
```

---

### 6.3 Build Process: Compiling `vaayu_rust`

**Prerequisites:**
```bash
# Install Rust toolchain (if not present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin (the PyO3 build tool)
pip install maturin
```

**Build and install as an editable Python package:**
```bash
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
maturin develop --release
# → builds vaayu_rust.so and installs it into the active venv
```

**Verify the build:**
```bash
python3 -c "
import vaayu_rust
print(dir(vaayu_rust))
# Should show: ['io_uring_read_into', 'pread_into', ...]
"
```

**Note:** If `maturin develop` fails with an `O_DIRECT` alignment error at runtime, see Section 8 (Known Issues) in Part 4.

---

### 6.4 Expected Performance Gain from Rust io_uring

The Python threading approach achieves 4.4 GB/s because each of the 16 threads holds one `preadv` syscall open simultaneously. The overhead is:
- Thread wakeup/scheduling latency (~100–200 μs per thread per layer)
- Python GIL acquisition on wakeup (~50 μs per thread)
- ThreadPoolExecutor overhead
- Chunk alignment decisions in Python

`io_uring` eliminates all of these. A single Rust thread can submit 32 reads simultaneously in one syscall, with responses handled via the completion ring — no thread context switches, no GIL, no Python overhead.

**Projected performance:**

| Method | Throughput | Time per layer | Time per token |
|---|---|---|---|
| Python QD1 (baseline) | 1.04 GB/s | 1.54s | ~116s |
| Python QD16 threading | 4.40 GB/s | 0.36s | ~31s |
| Rust io_uring (projected) | 6–7 GB/s | ~0.24s | **~20s** |
| Theoretical max (7.45 GB/s) | 7.45 GB/s | ~0.22s | ~18.5s |

**After moving GPU to PCI_E1 (PCIe 5.0 x16):**

| Method | Projected token time | Notes |
|---|---|---|
| Python QD16 + GPU in x16 | ~12–14s | GPU transfer drops from 20s → 2s |
| Rust io_uring + GPU in x16 | **~8–10s** | Combined optimal state |

---

## 7. Complete Code Architecture

This section documents the codebase structure as it exists after all the changes described in this bible.

---

### 7.1 Repository Layout

```
Vaayu/
├── vaayu/                          # Core Python inference engine
│   ├── vaayu_base.py               # Main model class — forward pass, KV cache, layer loop
│   ├── persist/
│   │   └── safetensor_model_persister.py  # I/O engine — double buffer, QD16, Rust integration
│   ├── modeling_llama.py           # Llama decoder layer (modified for KV cache)
│   └── cache_utils.py              # DynamicCache wrapper
│
├── vaayu_rust/                     # Rust io_uring module
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs                  # io_uring_read_into() and pread_into()
│   └── target/release/             # Build artifacts (after maturin build)
│
├── test_quick.py                   # Standard inference test — "What is the capital of India?"
├── benchmark_io_uring.py           # I/O benchmark comparing threading vs io_uring
├── benchmark_rust_io.py            # Rust-specific throughput benchmark
│
└── docs/
    ├── VAAYU_PROJECT_BIBLE.md      # Part 1 — Introduction, hardware, early bugs
    ├── VAAYU_PROJECT_BIBLE_PART2.md  # Part 2 — I/O bottleneck, benchmarks
    ├── VAAYU_PROJECT_BIBLE_PART3.md  # Part 3 — Rust module, code architecture
    ├── VAAYU_PROJECT_BIBLE_PART4.md  # Part 4 — Performance timeline, open issues, next steps
    ├── nvme_streaming_research_paper.md  # Technical deep-dive paper (corrected version)
    └── hardware_software_repot.md   # Hardware findings summary
```

---

### 7.2 `vaayu_base.py` — Key Code Sections

**The layer-streaming loop (simplified):**

```python
def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
    # Step 1: Embed tokens (loads embed_tokens layer from NVMe)
    hidden_states = self._load_and_run_embed(input_ids)
    
    # Step 2: Stream 80 decoder layers
    kv_cache = past_key_values or DynamicCache()
    
    for layer_idx in range(self.num_layers):
        # Prefetch next layer in background thread while GPU processes current
        future = self._executor.submit(self._load_layer, layer_idx + 1)
        
        # Move current layer to GPU
        layer = self._current_layer.to(self.device, non_blocking=True)
        
        # Run decoder layer forward pass — CRITICAL: use 'past_key_values' (plural)
        layer_out = layer(
            hidden_states,
            past_key_values=kv_cache,   # ← was 'past_key_value' (bug, now fixed)
            use_cache=True,              # ← must not be overridden to False
            **decoder_kwargs,
        )
        hidden_states = layer_out[0]
        
        # Evict layer from GPU immediately
        del layer
        # DO NOT call gc.collect() here — it costs 17ms × 83 = 1.41s per token
        
        # Wait for prefetch
        self._current_layer = future.result()
    
    # Step 3: Run norm + lm_head
    ...
    
    return logits, kv_cache
```

**Files modified for KV cache fix:**
- `vaayu/vaayu_base.py` — line ~659: `past_key_value` → `past_key_values`
- `vaayu/vaayu_base.py` — removed `use_cache=False` override

---

### 7.3 `safetensor_model_persister.py` — Current State

This is the most heavily modified file. The active code path as of the last session:

```python
# ─── Global state (module-level, allocated once at import time) ────────────
_PINNED_BUF_SIZE = 1024 * 1024 * 1800      # 1.8 GB
_PINNED_BUFS = [
    torch.empty(_PINNED_BUF_SIZE, dtype=torch.uint8, pin_memory=True),
    torch.empty(_PINNED_BUF_SIZE, dtype=torch.uint8, pin_memory=True),
]
_BUF_IDX = 0
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=16)

# Try Rust module
try:
    import vaayu_rust
    _RUST_IO = True
except ImportError:
    _RUST_IO = False

print(f">>>> pinned-memory loader active — direct NVMe→RAM→GPU path (QD16)")

# ─── dtype mapping (safetensors → torch) ──────────────────────────────────
_DTYPE_MAP = {
    'BF16': torch.bfloat16,
    'F16':  torch.float16,
    'F32':  torch.float32,
    'I32':  torch.int32,
}

# ─── Core load function ────────────────────────────────────────────────────
def _load_pinned(path: str) -> dict[str, torch.Tensor]:
    global _BUF_IDX
    buf = _PINNED_BUFS[_BUF_IDX]
    _BUF_IDX ^= 1
    
    t0 = time.perf_counter()
    
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    
    tensors_meta = {k: v for k, v in hdr.items() if k != '__metadata__'}
    total_size = max(v['data_offsets'][1] for v in tensors_meta.values())
    
    view = memoryview(buf.numpy())
    
    if _RUST_IO:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        vaayu_rust.io_uring_read_into(fd, buf.data_ptr(), total_size, data_start)
        os.close(fd)
    else:
        # QD16 threading fallback
        fd = os.open(path, os.O_RDONLY)
        chunk = (total_size + 15) // 16
        def _read(i):
            off = i * chunk
            n = min(chunk, total_size - off)
            if n > 0:
                os.preadv(fd, [view[off:off+n]], data_start + off)
        list(_IO_EXECUTOR.map(_read, range(16)))
        os.close(fd)
    
    elapsed = time.perf_counter() - t0
    layer_name = path.split('/')[-1].replace('.safetensors', '')
    print(f'[VAAYU-IO] pinned {elapsed:.3f}s  {layer_name}')
    
    # Slice tensors from pinned buffer — zero copy, just views
    result = {}
    for name, meta in tensors_meta.items():
        start, end = meta['data_offsets']
        nbytes = end - start
        raw = buf.narrow(0, start, nbytes)
        dtype = _DTYPE_MAP[meta['dtype']]
        result[name] = raw.view(dtype).reshape(meta['shape'])
    
    return result
```

---

### 7.4 The `[VAAYU-IO]` Token Summary Format

Every token generates a final summary line after all 83 layers complete:

```
[VAAYU-IO] token summary: total=31.1s  io_wait=0.9s  gpu_load=20.4s  compute=0.0s  other=9.7s
```

**Breakdown of categories:**
- `total` — wall clock time for the entire token (from first layer start to lm_head done)
- `io_wait` — cumulative time blocked in `_load_pinned()` waiting for NVMe reads
- `gpu_load` — cumulative time transferring tensors from pinned RAM to GPU VRAM (`.to("cuda")`)
- `compute` — cumulative time for GPU matrix operations (attention, FFN, etc.)
- `other` — everything else: Python overhead, tensor slicing, prefetch scheduling, the GC

This breakdown makes it trivially clear where time is being spent. After the I/O fixes, `io_wait` dropped from 88.4s to 0.9s — a 98.9% reduction. The new bottleneck is `gpu_load` at 20.4s, directly caused by the GPU being in the wrong PCIe slot.

---

*[Continued in Part 4: Complete Performance Timeline, Known Issues, and Next Steps]*
