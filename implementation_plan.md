# O_DIRECT io_uring Integration — Implementation Plan

Enable true page-cache bypass for NVMe reads by opening layer files with `O_DIRECT` in the Rust `io_uring` module, and implementing the "Read Wide, Slice Narrow" alignment strategy on the Python side.

## Proposed Changes

### Rust Module — `vaayu_rust`

#### [MODIFY] [lib.rs](file:///c:/laragon/www/AI-ML/Vaayu/vaayu_rust/src/lib.rs)

The `io_uring_read_into` function currently opens files with `File::open()` (standard buffered I/O). This must change to `libc::open()` with `O_RDONLY | O_DIRECT` to bypass the Linux page cache entirely.

**Current code (line 30–32):**
```rust
let file = File::open(&path)
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(
        format!("open {}: {}", path, e)))?;
```

**New code:**
```rust
let c_path = std::ffi::CString::new(path.as_str())
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
let raw_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
if raw_fd < 0 {
    return Err(pyo3::exceptions::PyIOError::new_err(
        format!("open O_DIRECT {}: errno {}", path,
                std::io::Error::last_os_error())));
}
```

Additional changes in `io_uring_read_into`:
- Use `raw_fd` directly with `types::Fd(raw_fd)` instead of `file.as_raw_fd()`
- Add `unsafe { libc::close(raw_fd); }` at the end of the function (before `Ok(())`) to clean up the file descriptor, wrapped in a scope guard pattern to ensure it closes even on error
- The chunk alignment logic on line 39 (`(chunk_size + 4095) & !4095`) is already correct and stays as-is

The `pread_into` fallback function remains unchanged — it does not use `O_DIRECT` and serves as a safe fallback.

No changes to function signatures. The Rust module stays "dumb" — it receives pre-aligned offset/length from Python and just executes the read.

---

### Python Integration — `safetensor_model_persister.py`

#### [MODIFY] [safetensor_model_persister.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/persist/safetensor_model_persister.py)

**Change 1: Add alignment constants and helper (after line 24)**

```python
_ALIGN = 4096  # O_DIRECT requires sector-aligned offset, length, and buffer

def _align_down(x):
    """Round x down to nearest _ALIGN boundary."""
    return x & ~(_ALIGN - 1)

def _align_up(x):
    """Round x up to nearest _ALIGN boundary."""
    return (x + _ALIGN - 1) & ~(_ALIGN - 1)
```

**Change 2: Implement "Read Wide, Slice Narrow" in the io_uring path (lines 80–88)**

Current code passes raw `data_start` and `total_size` to Rust:
```python
vaayu_rust.io_uring_read_into(
    file_path, data_start, total_size, buf.data_ptr(), 32,
)
```

New code computes aligned boundaries and adjusts the buffer view:
```python
# "Read Wide, Slice Narrow" — pad offset/length to 4096 boundaries
aligned_offset = _align_down(data_start)
prefix_pad = data_start - aligned_offset          # bytes of junk before real data
aligned_length = _align_up(total_size + prefix_pad)

vaayu_rust.io_uring_read_into(
    file_path,
    aligned_offset,     # ← 4096-aligned file offset
    aligned_length,     # ← 4096-aligned read length
    buf.data_ptr(),     # ← already page-aligned (pin_memory=True)
    32,                 # QD32
)

# Slide the view past the alignment prefix so tensor offsets are correct
buf_view = buf[prefix_pad:]
```

**Change 3: Update buffer allocation to account for alignment padding (line 73)**

Current:
```python
alloc_size = total_size + (100 * 1024 * 1024)   # pad +100 MB
```

New:
```python
alloc_size = total_size + (100 * 1024 * 1024) + _ALIGN  # +4096 for O_DIRECT padding
```

The extra 4096 bytes accounts for the worst-case alignment padding. This is negligible relative to the 100 MB headroom already allocated.

**Change 4: Use `buf_view` instead of `buf` for tensor slicing (line 113)**

Current:
```python
result[name] = buf[start:end].view(dtype).reshape(meta['shape'])
```

New — only when io_uring path was used:
```python
source = buf_view if _USE_IOURING else buf
result[name] = source[start:end].view(dtype).reshape(meta['shape'])
```

This ensures the tensor offsets from the safetensors header (which are relative to `data_start`, not `aligned_offset`) map correctly to the buffer contents.

> [!NOTE]
> The Python `preadv` fallback path (lines 89–106) does **not** use `O_DIRECT` and does **not** need alignment changes. It continues to work with raw `data_start` and `total_size` as before.

---

## Verification Plan

### Automated Tests

**Step 1:** Rebuild the Rust module on the Linux dev machine:
```bash
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
maturin develop --release
python3 -c "import vaayu_rust; print('✓ Rust module loaded')"
```

**Step 2:** Run the standard inference test and confirm no `EINVAL` errors:
```bash
python3 test_quick.py 2>&1 | head -30
```
- Should see `>>>> Rust io_uring reader loaded` in startup
- Should see `[VAAYU-IO] iouring 0.XXXs` per layer (not `pinned`)
- No `io_uring read error` or `PyIOError` exceptions

**Step 3:** Compare per-layer times against the Python QD16 baseline:
```bash
# Drop caches first for fair comparison
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
python3 test_quick.py 2>&1 | grep "VAAYU-IO"
```
- Python QD16 baseline: ~0.36s per layer
- Expected with io_uring + O_DIRECT: **~0.24–0.28s per layer**
- If times are *worse* than 0.36s, `O_DIRECT` alignment is likely wrong — check `dmesg` for kernel errors

**Step 4:** Verify page cache bypass:
```bash
# Clear caches
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
# Run one token
python3 test_quick.py 2>&1 | grep "token summary"
# Check that io_wait does NOT decrease dramatically on token 2
# (if O_DIRECT works, there's no page cache warming — every token reads from NVMe)
```
With `O_DIRECT`, token 2 should have similar `io_wait` to token 1 (no page cache benefit). This confirms the bypass is working.

### Manual Verification

- After GPU is physically moved to PCI_E1, re-run `test_quick.py` and compare against the Era 4/5 projections in the Project Bible Part 4
