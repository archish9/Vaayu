# Plan D: Deep Rust io_uring Optimizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-SQE kernel overhead by adding a persistent `IoUring` ring with `IORING_REGISTER_BUFFERS` and `IORING_REGISTER_FILES`, cutting CPU overhead on model layer loads by 10–15%.

**Architecture:** A module-level `static RING_STATE: Mutex<Option<RingState>>` holds the persistent ring, buffer registrations, and file descriptor pool. `io_uring_read_into` auto-dispatches to `ReadFixed`+`Fixed` when both are registered, falling back gracefully to the existing `Read`+`Fd` path. Python calls five new Rust functions at startup and after buffer reallocs; the `io_uring_read_into` call site is unchanged.

**Tech Stack:** Rust (PyO3 0.21, io-uring 0.7, libc 0.2), Python 3.8+, maturin 1.5+, Linux kernel ≥ 5.1

---

## File Map

| File | Change |
|------|--------|
| `vaayu_rust/src/lib.rs` | Add structs, static, 5 new pyfunctions; extract `legacy_read`; add `persistent_read`; update dispatch + module exports |
| `vaayu/persist/safetensor_model_persister.py` | Add 3 globals, 2 helpers; modify `load_model` (lazy init) and `_load_pinned` (re-register after realloc) |
| `tests/test_iouring_persistent.py` | New integration test (run on Linux target) |

---

## Task 1: Add RingState struct and global static to lib.rs

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

- [ ] **Step 1: Write failing import test**

Create `tests/test_iouring_persistent.py`:

```python
import vaayu_rust

def test_init_ring_exists():
    assert hasattr(vaayu_rust, "init_ring"), "init_ring not found"

def test_register_buffers_exists():
    assert hasattr(vaayu_rust, "register_buffers"), "register_buffers not found"

def test_reregister_buffers_exists():
    assert hasattr(vaayu_rust, "reregister_buffers"), "reregister_buffers not found"

def test_register_files_exists():
    assert hasattr(vaayu_rust, "register_files"), "register_files not found"

def test_unregister_files_exists():
    assert hasattr(vaayu_rust, "unregister_files"), "unregister_files not found"

if __name__ == "__main__":
    test_init_ring_exists()
    test_register_buffers_exists()
    test_reregister_buffers_exists()
    test_register_files_exists()
    test_unregister_files_exists()
    print("all attribute checks passed")
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd vaayu_rust && maturin develop --release 2>&1 | tail -5
cd .. && python tests/test_iouring_persistent.py
```

Expected: `AssertionError: init_ring not found`

- [ ] **Step 3: Add imports and data structures to lib.rs**

At the top of `vaayu_rust/src/lib.rs`, replace the existing `use` block with:

```rust
use io_uring::{opcode, types, IoUring};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::sync::Mutex;
```

After the `use` block, before the first `#[pyfunction]`, add:

```rust
struct BufRegistrations {
    ptr0: usize,
    len0: usize,
    ptr1: usize,
    len1: usize,
}

struct RingState {
    ring: IoUring,
    queue_depth: usize,
    buf_regs: Option<BufRegistrations>,
    file_map: HashMap<String, u32>,
    raw_fds: Vec<RawFd>,
}

// Safety: IoUring wraps a file descriptor and mmap'd memory.
// The Mutex ensures exclusive access — never shared across threads simultaneously.
unsafe impl Send for RingState {}

static RING_STATE: Mutex<Option<RingState>> = Mutex::new(None);
```

- [ ] **Step 4: Commit**

```bash
git add vaayu_rust/src/lib.rs tests/test_iouring_persistent.py
git commit -m "feat(rust): add RingState struct and RING_STATE global static"
```

---

## Task 2: Implement init_ring()

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

- [ ] **Step 1: Add init_ring() function**

Add after the `RING_STATE` static (before `io_uring_read_into`):

```rust
#[pyfunction]
fn init_ring(queue_depth: u32) -> PyResult<()> {
    let mut guard = RING_STATE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }
    let qd = (queue_depth.min(64)) as usize;
    let ring = IoUring::new(qd as u32)
        .map_err(|e| PyRuntimeError::new_err(format!("io_uring_setup: {}", e)))?;
    *guard = Some(RingState {
        ring,
        queue_depth: qd,
        buf_regs: None,
        file_map: HashMap::new(),
        raw_fds: Vec::new(),
    });
    eprintln!("[vaayu_rust] persistent ring initialized  qd={}", qd);
    Ok(())
}
```

- [ ] **Step 2: Register init_ring in the module**

In the `#[pymodule]` function at the bottom of `lib.rs`, add before the final `Ok(())`:

```rust
m.add_function(wrap_pyfunction!(init_ring, m)?)?;
```

- [ ] **Step 3: Build**

```bash
cd vaayu_rust && maturin develop --release
```

Expected: compiles without errors.

- [ ] **Step 4: Add and run test for init_ring**

Append to `tests/test_iouring_persistent.py`:

```python
def test_init_ring():
    # Idempotent — calling twice must not error
    vaayu_rust.init_ring(32)
    vaayu_rust.init_ring(32)
    print("init_ring: ok")
```

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_iouring_persistent.py').read()); test_init_ring()"
```

Expected: prints `init_ring: ok`

- [ ] **Step 5: Commit**

```bash
git add vaayu_rust/src/lib.rs tests/test_iouring_persistent.py
git commit -m "feat(rust): implement init_ring() — creates persistent IoUring ring singleton"
```

---

## Task 3: Implement register_buffers() and reregister_buffers()

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

- [ ] **Step 1: Add do_register_buffers helper and both pyfunctions**

Add after `init_ring`:

```rust
fn do_register_buffers(
    state: &mut RingState,
    ptr0: usize, len0: usize,
    ptr1: usize, len1: usize,
) -> PyResult<()> {
    // Unregister existing (no-op if not registered)
    let _ = state.ring.submitter().unregister_buffers();
    let iovecs = [
        libc::iovec { iov_base: ptr0 as *mut libc::c_void, iov_len: len0 },
        libc::iovec { iov_base: ptr1 as *mut libc::c_void, iov_len: len1 },
    ];
    unsafe { state.ring.submitter().register_buffers(&iovecs) }
        .map_err(|e| PyRuntimeError::new_err(format!("register_buffers: {}", e)))?;
    state.buf_regs = Some(BufRegistrations { ptr0, len0, ptr1, len1 });
    eprintln!(
        "[vaayu_rust] buffers registered  buf0=0x{:x}+{}  buf1=0x{:x}+{}",
        ptr0, len0, ptr1, len1
    );
    Ok(())
}

#[pyfunction]
fn register_buffers(ptr0: usize, len0: usize, ptr1: usize, len1: usize) -> PyResult<()> {
    let mut guard = RING_STATE.lock().unwrap();
    let state = guard
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err("ring not initialized — call init_ring() first"))?;
    do_register_buffers(state, ptr0, len0, ptr1, len1)
}

#[pyfunction]
fn reregister_buffers(ptr0: usize, len0: usize, ptr1: usize, len1: usize) -> PyResult<()> {
    let mut guard = RING_STATE.lock().unwrap();
    let state = guard
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err("ring not initialized"))?;
    do_register_buffers(state, ptr0, len0, ptr1, len1)
}
```

- [ ] **Step 2: Register both functions in the module**

In the `#[pymodule]` function, add:

```rust
m.add_function(wrap_pyfunction!(register_buffers, m)?)?;
m.add_function(wrap_pyfunction!(reregister_buffers, m)?)?;
```

- [ ] **Step 3: Build**

```bash
cd vaayu_rust && maturin develop --release
```

Expected: no errors.

- [ ] **Step 4: Add and run buffer registration test**

Append to `tests/test_iouring_persistent.py`:

```python
def test_register_buffers():
    import torch
    vaayu_rust.init_ring(32)
    buf0 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    buf1 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    vaayu_rust.register_buffers(
        buf0.data_ptr(), buf0.nbytes,
        buf1.data_ptr(), buf1.nbytes,
    )
    # reregister after simulated realloc
    buf0b = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    vaayu_rust.reregister_buffers(
        buf0b.data_ptr(), buf0b.nbytes,
        buf1.data_ptr(), buf1.nbytes,
    )
    print("register_buffers / reregister_buffers: ok")
```

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_iouring_persistent.py').read()); test_register_buffers()"
```

Expected: prints `register_buffers / reregister_buffers: ok`

- [ ] **Step 5: Commit**

```bash
git add vaayu_rust/src/lib.rs tests/test_iouring_persistent.py
git commit -m "feat(rust): implement register_buffers() and reregister_buffers() via IORING_REGISTER_BUFFERS"
```

---

## Task 4: Implement register_files() and unregister_files()

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

- [ ] **Step 1: Add register_files() and unregister_files()**

Add after `reregister_buffers`:

```rust
#[pyfunction]
fn register_files(py: Python<'_>, paths: Vec<String>) -> PyResult<()> {
    py.allow_threads(move || {
        let mut guard = RING_STATE.lock().unwrap();
        let state = guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ring not initialized"))?;

        // Clear any existing file registration
        if !state.raw_fds.is_empty() {
            let _ = state.ring.submitter().unregister_files();
            for &fd in &state.raw_fds {
                unsafe { libc::close(fd) };
            }
            state.raw_fds.clear();
            state.file_map.clear();
        }

        let mut raw_fds: Vec<RawFd> = Vec::with_capacity(paths.len());
        let mut file_map: HashMap<String, u32> = HashMap::new();

        for (idx, path) in paths.iter().enumerate() {
            let c_path = std::ffi::CString::new(path.as_str())
                .map_err(|e| PyValueError::new_err(format!("invalid path {}: {}", path, e)))?;
            let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
            if fd < 0 {
                for &f in &raw_fds {
                    unsafe { libc::close(f) };
                }
                return Err(PyIOError::new_err(format!(
                    "open {} failed: {}",
                    path,
                    std::io::Error::last_os_error()
                )));
            }
            raw_fds.push(fd);
            file_map.insert(path.clone(), idx as u32);
        }

        state
            .ring
            .submitter()
            .register_files(&raw_fds)
            .map_err(|e| {
                for &f in &raw_fds {
                    unsafe { libc::close(f) };
                }
                PyRuntimeError::new_err(format!("register_files: {}", e))
            })?;

        state.raw_fds = raw_fds;
        state.file_map = file_map;
        eprintln!("[vaayu_rust] {} files registered", state.raw_fds.len());
        Ok(())
    })
}

#[pyfunction]
fn unregister_files(py: Python<'_>) -> PyResult<()> {
    py.allow_threads(|| {
        let mut guard = RING_STATE.lock().unwrap();
        if let Some(ref mut state) = *guard {
            let _ = state.ring.submitter().unregister_files();
            for &fd in &state.raw_fds {
                unsafe { libc::close(fd) };
            }
            state.raw_fds.clear();
            state.file_map.clear();
            eprintln!("[vaayu_rust] files unregistered");
        }
        Ok(())
    })
}
```

- [ ] **Step 2: Register both in the module**

```rust
m.add_function(wrap_pyfunction!(register_files, m)?)?;
m.add_function(wrap_pyfunction!(unregister_files, m)?)?;
```

- [ ] **Step 3: Build**

```bash
cd vaayu_rust && maturin develop --release
```

- [ ] **Step 4: Add and run file registration test**

Append to `tests/test_iouring_persistent.py`:

```python
def test_register_files():
    import os, tempfile
    vaayu_rust.init_ring(32)
    # Create two small test files
    files = []
    for i in range(2):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
        f.write(b"\x00" * 4096)  # 4KB aligned
        f.close()
        files.append(f.name)
    try:
        vaayu_rust.register_files(files)
        vaayu_rust.unregister_files()
        print("register_files / unregister_files: ok")
    finally:
        for p in files:
            os.unlink(p)
```

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_iouring_persistent.py').read()); test_register_files()"
```

Expected: `register_files / unregister_files: ok`

- [ ] **Step 5: Commit**

```bash
git add vaayu_rust/src/lib.rs tests/test_iouring_persistent.py
git commit -m "feat(rust): implement register_files() and unregister_files() via IORING_REGISTER_FILES"
```

---

## Task 5: Extract legacy_read() from io_uring_read_into

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

This is a pure refactor. No behavior changes. The existing `io_uring_read_into` body moves verbatim into a `legacy_read` free function.

- [ ] **Step 1: Add legacy_read() above io_uring_read_into**

Add this function immediately before `io_uring_read_into`:

```rust
fn legacy_read(path: &str, offset: u64, length: usize, buf_ptr: usize, queue_depth: u32) -> PyResult<()> {
    let c_path = std::ffi::CString::new(path)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let raw_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
    if raw_fd < 0 {
        return Err(PyIOError::new_err(format!(
            "open O_DIRECT {}: errno {}",
            path,
            std::io::Error::last_os_error()
        )));
    }
    let file = unsafe { File::from_raw_fd(raw_fd) };
    eprintln!("[vaayu_rust] open O_DIRECT  path={}  offset={}  length={}", path, offset, length);

    let fd = types::Fd(file.as_raw_fd());
    let ptr = buf_ptr as *mut u8;
    let qd = queue_depth.min(64) as usize;
    let chunk_size = (length + qd - 1) / qd;
    let chunk_size = (chunk_size + 4095) & !4095;
    let total_chunks = (length + chunk_size - 1) / chunk_size;
    eprintln!(
        "[vaayu_rust] qd={}  chunk_size={}  total_chunks={}  total_bytes={}",
        qd, chunk_size, total_chunks, length
    );

    let mut ring = IoUring::new(qd as u32)
        .map_err(|e| PyRuntimeError::new_err(format!("io_uring_setup: {}", e)))?;

    let mut submitted = 0usize;
    let mut file_off = offset;
    let mut buf_off = 0usize;

    while buf_off < length {
        let this_len = chunk_size.min(length - buf_off);
        let read_op = opcode::Read::new(fd, unsafe { ptr.add(buf_off) }, this_len as u32)
            .offset(file_off)
            .build()
            .user_data(submitted as u64);
        unsafe { ring.submission().push(&read_op) }
            .map_err(|_| PyRuntimeError::new_err("SQ full"))?;
        file_off += this_len as u64;
        buf_off += this_len;
        submitted += 1;
    }

    eprintln!("[vaayu_rust] submitted {} SQEs", submitted);
    ring.submit_and_wait(submitted)
        .map_err(|e| PyRuntimeError::new_err(format!("io_uring submit: {}", e)))?;

    let cq = ring.completion();
    let mut completed = 0usize;
    for cqe in cq {
        if cqe.result() < 0 {
            eprintln!(
                "[vaayu_rust] CQE error  idx={}  result={}  errno={}",
                cqe.user_data(),
                cqe.result(),
                -cqe.result()
            );
            return Err(PyIOError::new_err(format!(
                "io_uring read error  CQE[{}]  result={}  errno={}",
                cqe.user_data(),
                cqe.result(),
                -cqe.result()
            )));
        }
        completed += 1;
    }

    eprintln!("[vaayu_rust] done {}/{} CQEs ok", completed, submitted);
    if completed != submitted {
        return Err(PyRuntimeError::new_err(format!(
            "incomplete: {}/{} CQEs",
            completed,
            submitted
        )));
    }
    Ok(())
}
```

- [ ] **Step 2: Replace io_uring_read_into body with a call to legacy_read**

Replace the entire body of `io_uring_read_into` (the `py.allow_threads` closure contents) with a delegation:

```rust
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
    py.allow_threads(move || legacy_read(&path, offset, length, buf_ptr, queue_depth))
}
```

- [ ] **Step 3: Build and verify no regression**

```bash
cd vaayu_rust && maturin develop --release
```

Expected: compiles without errors. Existing read behavior unchanged.

- [ ] **Step 4: Commit**

```bash
git add vaayu_rust/src/lib.rs
git commit -m "refactor(rust): extract legacy_read() from io_uring_read_into — no behavior change"
```

---

## Task 6: Add persistent_read() and update io_uring_read_into dispatch

**Files:**
- Modify: `vaayu_rust/src/lib.rs`

- [ ] **Step 1: Add persistent_read() before legacy_read()**

```rust
fn persistent_read(
    state: &mut RingState,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
) -> PyResult<()> {
    // Determine which registered buffer index buf_ptr falls into (None = use Read)
    let buf_index: Option<u16> = state.buf_regs.as_ref().and_then(|br| {
        if buf_ptr >= br.ptr0 && buf_ptr < br.ptr0 + br.len0 {
            Some(0)
        } else if buf_ptr >= br.ptr1 && buf_ptr < br.ptr1 + br.len1 {
            Some(1)
        } else {
            None
        }
    });

    // Determine whether to use a fixed (pre-registered) fd
    let fixed_fd_idx: Option<u32> = state.file_map.get(path).copied();

    // If path not registered, open a fresh fd (dropped = closed at end of fn)
    let _fresh_file: Option<File>;
    let fresh_raw_fd: RawFd;
    if fixed_fd_idx.is_none() {
        let c_path = std::ffi::CString::new(path)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        fresh_raw_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
        if fresh_raw_fd < 0 {
            return Err(PyIOError::new_err(format!(
                "open O_DIRECT {}: {}",
                path,
                std::io::Error::last_os_error()
            )));
        }
        _fresh_file = Some(unsafe { File::from_raw_fd(fresh_raw_fd) });
    } else {
        fresh_raw_fd = -1;
        _fresh_file = None;
    }

    let ptr = buf_ptr as *mut u8;
    let qd = state.queue_depth;
    let chunk_size = (length + qd - 1) / qd;
    let chunk_size = (chunk_size + 4095) & !4095;
    let total_chunks = (length + chunk_size - 1) / chunk_size;
    eprintln!(
        "[vaayu_rust] persistent  buf_index={:?}  fixed_fd={:?}  qd={}  chunk={}  total_chunks={}",
        buf_index, fixed_fd_idx, qd, chunk_size, total_chunks
    );

    let mut submitted = 0usize;
    let mut file_off = offset;
    let mut buf_off = 0usize;

    while buf_off < length {
        let this_len = chunk_size.min(length - buf_off) as u32;
        let sqe_ptr = unsafe { ptr.add(buf_off) };

        let entry = match (buf_index, fixed_fd_idx) {
            (Some(bi), Some(fi)) => opcode::ReadFixed::new(types::Fixed(fi), sqe_ptr, this_len, bi)
                .offset(file_off)
                .build()
                .user_data(submitted as u64),
            (Some(bi), None) => opcode::ReadFixed::new(types::Fd(fresh_raw_fd), sqe_ptr, this_len, bi)
                .offset(file_off)
                .build()
                .user_data(submitted as u64),
            (None, Some(fi)) => opcode::Read::new(types::Fixed(fi), sqe_ptr, this_len)
                .offset(file_off)
                .build()
                .user_data(submitted as u64),
            (None, None) => opcode::Read::new(types::Fd(fresh_raw_fd), sqe_ptr, this_len)
                .offset(file_off)
                .build()
                .user_data(submitted as u64),
        };

        unsafe { state.ring.submission().push(&entry) }
            .map_err(|_| PyRuntimeError::new_err("SQ full"))?;

        file_off += this_len as u64;
        buf_off += this_len as usize;
        submitted += 1;
    }

    state
        .ring
        .submit_and_wait(submitted)
        .map_err(|e| PyRuntimeError::new_err(format!("io_uring submit: {}", e)))?;

    let cq = state.ring.completion();
    let mut completed = 0usize;
    for cqe in cq {
        if cqe.result() < 0 {
            return Err(PyIOError::new_err(format!(
                "io_uring read error  CQE[{}]  result={}  errno={}",
                cqe.user_data(),
                cqe.result(),
                -cqe.result()
            )));
        }
        completed += 1;
    }

    if completed != submitted {
        return Err(PyRuntimeError::new_err(format!(
            "incomplete: {}/{} CQEs",
            completed,
            submitted
        )));
    }
    Ok(())
}
```

- [ ] **Step 2: Update io_uring_read_into to dispatch**

Replace the current `io_uring_read_into` with:

```rust
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
        let mut guard = RING_STATE.lock().unwrap();
        if let Some(ref mut state) = *guard {
            persistent_read(state, &path, offset, length, buf_ptr)
        } else {
            drop(guard);
            legacy_read(&path, offset, length, buf_ptr, queue_depth)
        }
    })
}
```

- [ ] **Step 3: Build**

```bash
cd vaayu_rust && maturin develop --release
```

Expected: no errors. If `opcode::ReadFixed` is not found, check `io-uring` version — it is present since 0.4.

- [ ] **Step 4: Add and run end-to-end read test with persistent ring**

Append to `tests/test_iouring_persistent.py`:

```python
def test_persistent_read_correctness():
    """Write known bytes to a temp file, read back via persistent ring, verify content."""
    import os, tempfile, struct
    import torch

    # Write 8192 bytes (2 aligned pages) with a recognizable pattern
    payload = bytes(range(256)) * 32  # 8192 bytes
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
    f.write(payload)
    f.close()
    path = f.name

    try:
        vaayu_rust.init_ring(32)

        buf0 = torch.zeros(1024 * 1024, dtype=torch.uint8, pin_memory=True)
        buf1 = torch.zeros(1024 * 1024, dtype=torch.uint8, pin_memory=True)
        vaayu_rust.register_buffers(
            buf0.data_ptr(), buf0.nbytes,
            buf1.data_ptr(), buf1.nbytes,
        )
        vaayu_rust.register_files([path])

        length = len(payload)
        # offset 0, length must be sector-aligned (already 8192)
        vaayu_rust.io_uring_read_into(path, 0, length, buf0.data_ptr(), 32)

        result = bytes(buf0[:length].numpy())
        assert result == payload, f"Data mismatch: first byte got {result[0]}, want {payload[0]}"
        print("persistent_read_correctness: ok")
    finally:
        vaayu_rust.unregister_files()
        os.unlink(path)
```

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_iouring_persistent.py').read()); test_persistent_read_correctness()"
```

Expected: `persistent_read_correctness: ok`

If you see `EINVAL` or `EFAULT` from the kernel: check `dmesg | tail -20` for context. Common causes:
- Buffer ptr not page-aligned (pinned PyTorch tensors are always aligned — should not occur)
- Kernel < 5.1 (check with `uname -r`)
- O_DIRECT requires sector-aligned offset and length (test uses 8192 bytes at offset 0 — should be fine)

- [ ] **Step 5: Commit**

```bash
git add vaayu_rust/src/lib.rs tests/test_iouring_persistent.py
git commit -m "feat(rust): add persistent_read() with ReadFixed+Fixed dispatch; update io_uring_read_into"
```

---

## Task 7: Update safetensor_model_persister.py

**Files:**
- Modify: `vaayu/persist/safetensor_model_persister.py`

- [ ] **Step 1: Add new globals and helper functions**

After the existing globals block (after line 55, after `_BUF_LOCK`), add:

```python
_RING_INITIALIZED = False   # True after vaayu_rust.init_ring() succeeds
_BUFS_REGISTERED  = False   # True after register_buffers() succeeds
_FILES_REGISTERED = False   # True after register_files() succeeds


def _init_iouring_ring(queue_depth: int = 32) -> bool:
    try:
        vaayu_rust.init_ring(queue_depth)
        print(f"[VAAYU-IO] persistent ring initialized  qd={queue_depth}")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] init_ring failed: {e} — per-call ring fallback active")
        return False


def _register_iouring_files(model_dir: str) -> bool:
    try:
        paths = [str(p) for p in Path(model_dir).glob("*.safetensors")]
        if not paths:
            print(f"[VAAYU-IO] no .safetensors files in {model_dir} — file registration skipped")
            return False
        vaayu_rust.register_files(paths)
        print(f"[VAAYU-IO] {len(paths)} files registered with io_uring")
        return True
    except Exception as e:
        print(f"[VAAYU-IO] register_files failed: {e} — fd registration disabled")
        return False
```

- [ ] **Step 2: Modify _load_pinned to call reregister_buffers after alloc**

In `_load_pinned`, the existing realloc block is at lines 85-90:

```python
if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < total_size:
    alloc_size = total_size + (100 * 1024 * 1024) + _ALIGN
    _PINNED_BUFS[buf_idx] = torch.empty(alloc_size, dtype=torch.uint8, pin_memory=True)
    _PINNED_SIZES[buf_idx] = alloc_size
    print(f"[VAAYU-IO] alloc  buf={buf_idx}  size={alloc_size / (1024**2):.1f}MB")
```

Replace it with:

```python
if _PINNED_BUFS[buf_idx] is None or _PINNED_SIZES[buf_idx] < total_size:
    alloc_size = total_size + (100 * 1024 * 1024) + _ALIGN
    _PINNED_BUFS[buf_idx] = torch.empty(alloc_size, dtype=torch.uint8, pin_memory=True)
    _PINNED_SIZES[buf_idx] = alloc_size
    print(f"[VAAYU-IO] alloc  buf={buf_idx}  size={alloc_size / (1024**2):.1f}MB")
    # Re-register buffers with io_uring after alloc (only when both are allocated)
    if _USE_IOURING and _RING_INITIALIZED and _PINNED_BUFS[0] is not None and _PINNED_BUFS[1] is not None:
        global _BUFS_REGISTERED
        try:
            vaayu_rust.reregister_buffers(
                _PINNED_BUFS[0].data_ptr(), _PINNED_SIZES[0],
                _PINNED_BUFS[1].data_ptr(), _PINNED_SIZES[1],
            )
            _BUFS_REGISTERED = True
        except Exception as e:
            print(f"[VAAYU-IO] reregister_buffers failed: {e}")
            _BUFS_REGISTERED = False
```

- [ ] **Step 3: Modify load_model for lazy one-time init**

At the top of `SafetensorModelPersister.load_model`, before the existing logic, add:

```python
def load_model(self, layer_name, path):
    global _RING_INITIALIZED, _FILES_REGISTERED
    if _USE_IOURING and not _RING_INITIALIZED:
        _RING_INITIALIZED = _init_iouring_ring(32)
    if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
        _FILES_REGISTERED = _register_iouring_files(str(path))
    # ... existing body unchanged below ...
```

The full updated `load_model` method:

```python
def load_model(self, layer_name, path):
    global _RING_INITIALIZED, _FILES_REGISTERED
    if _USE_IOURING and not _RING_INITIALIZED:
        _RING_INITIALIZED = _init_iouring_ring(32)
    if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
        _FILES_REGISTERED = _register_iouring_files(str(path))

    file_path = str(Path(path) / (layer_name + ".safetensors"))

    if _USE_PINNED:
        t = time.time()
        result = _load_pinned(file_path)
        elapsed = time.time() - t
        tag = "iouring" if _USE_IOURING else "pinned"
        size_mb = sum(v.nbytes for v in result.values()) / (1024**2)
        gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
        print(f"[VAAYU-IO] {tag}  {elapsed:.3f}s  {size_mb:.1f}MB  {gbps:.2f}GB/s  {layer_name}")
        return result
    else:
        t = time.time()
        result = load_file(file_path, device="cpu")
        elapsed = time.time() - t
        size_mb = sum(v.nbytes for v in result.values()) / (1024**2)
        gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
        print(f"[VAAYU-IO] python  {elapsed:.3f}s  {size_mb:.1f}MB  {gbps:.2f}GB/s  {layer_name}")
        return result
```

- [ ] **Step 4: Move `global _BUFS_REGISTERED` to top of _load_pinned**

Replace the existing global declaration line in `_load_pinned`:

```python
# Before
global _PINNED_BUFS, _PINNED_SIZES, _BUF_IDX

# After — add _BUFS_REGISTERED and remove the inner `global _BUFS_REGISTERED` added in step 2
global _PINNED_BUFS, _PINNED_SIZES, _BUF_IDX, _BUFS_REGISTERED
```

Then remove the standalone `global _BUFS_REGISTERED` line that was added inside the `if` block in step 2 (it is now redundant).

- [ ] **Step 5: Rebuild**

```bash
cd vaayu_rust && maturin develop --release
```

- [ ] **Step 6: Commit**

```bash
git add vaayu/persist/safetensor_model_persister.py
git commit -m "feat(python): wire persistent io_uring ring init and buffer/file registration into safetensor loader"
```

---

## Task 8: Full integration test and benchmark

**Files:**
- Modify: `tests/test_iouring_persistent.py`

- [ ] **Step 1: Add full test runner to test file**

Append to `tests/test_iouring_persistent.py`:

```python
def test_fallback_when_path_not_registered():
    """io_uring_read_into must succeed even for a file not in register_files list."""
    import os, tempfile
    import torch

    payload = bytes(range(256)) * 32  # 8192 bytes
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    f.write(payload)
    f.close()
    path = f.name

    try:
        vaayu_rust.init_ring(32)
        buf0 = torch.zeros(1024 * 1024, dtype=torch.uint8, pin_memory=True)
        buf1 = torch.zeros(1024 * 1024, dtype=torch.uint8, pin_memory=True)
        vaayu_rust.register_buffers(
            buf0.data_ptr(), buf0.nbytes,
            buf1.data_ptr(), buf1.nbytes,
        )
        # Intentionally do NOT register this file — exercises the ReadFixed+fresh-Fd path
        vaayu_rust.io_uring_read_into(path, 0, len(payload), buf0.data_ptr(), 32)
        result = bytes(buf0[:len(payload)].numpy())
        assert result == payload
        print("fallback_when_path_not_registered: ok")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("=== Plan D integration tests ===")
    test_init_ring_exists()
    test_register_buffers_exists()
    test_reregister_buffers_exists()
    test_register_files_exists()
    test_unregister_files_exists()
    test_init_ring()
    test_register_buffers()
    test_register_files()
    test_persistent_read_correctness()
    test_fallback_when_path_not_registered()
    print("=== all tests passed ===")
```

- [ ] **Step 2: Run all tests**

```bash
python tests/test_iouring_persistent.py
```

Expected output:
```
=== Plan D integration tests ===
[vaayu_rust] persistent ring initialized  qd=32
[vaayu_rust] buffers registered  buf0=0x...  buf1=0x...
[vaayu_rust] 2 files registered
[vaayu_rust] persistent  buf_index=Some(0)  fixed_fd=Some(0)  ...
persistent_read_correctness: ok
[vaayu_rust] persistent  buf_index=Some(0)  fixed_fd=None  ...
fallback_when_path_not_registered: ok
=== all tests passed ===
```

If any test fails with `EINVAL` from `register_buffers`:
- Check `dmesg | tail -20` for kernel rejection reason
- Verify kernel ≥ 5.1 with `uname -r`
- Pinned memory must be page-aligned: PyTorch `pin_memory=True` guarantees this

If any CQE returns `result < 0`:
- `-22` = `EINVAL`: offset or length not sector-aligned (must be multiples of 4096 for O_DIRECT)
- `-9` = `EBADF`: fixed fd index out of range — check `register_files` was called
- `-14` = `EFAULT`: buf ptr outside registered buffer range — check `register_buffers` called with correct sizes

- [ ] **Step 3: Benchmark (Linux)**

Run this timing comparison on the target Linux machine, substituting an actual model layer path:

```bash
# Before (comment out the lazy init block in load_model, rebuild with per-call ring)
python -c "
import time, vaayu_rust, torch
buf = torch.zeros(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
path = '/path/to/model/layer0.safetensors'
import os; size = os.path.getsize(path)
t = time.perf_counter()
for _ in range(5):
    vaayu_rust.io_uring_read_into(path, 0, size, buf.data_ptr(), 32)
print(f'per-call ring: {(time.perf_counter()-t)/5*1000:.1f}ms avg')
"

# After (with persistent ring initialized)
python -c "
import time, vaayu_rust, torch
vaayu_rust.init_ring(32)
buf0 = torch.zeros(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
buf1 = torch.zeros(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
vaayu_rust.register_buffers(buf0.data_ptr(), buf0.nbytes, buf1.data_ptr(), buf1.nbytes)
path = '/path/to/model/layer0.safetensors'
vaayu_rust.register_files([path])
import os; size = os.path.getsize(path)
t = time.perf_counter()
for _ in range(5):
    vaayu_rust.io_uring_read_into(path, 0, size, buf0.data_ptr(), 32)
print(f'persistent ring: {(time.perf_counter()-t)/5*1000:.1f}ms avg')
"
```

Expected: 10–15% reduction in avg ms (CPU overhead; wall-clock improvement varies with NVMe saturation).

- [ ] **Step 4: Final commit**

```bash
git add tests/test_iouring_persistent.py
git commit -m "test: add Plan D integration tests for persistent io_uring ring with REGISTER_BUFFERS and REGISTER_FILES"
```

---

## Verification Checklist

- [ ] `python tests/test_iouring_persistent.py` — all tests pass on Linux target
- [ ] No `EINVAL`/`EFAULT` in dmesg during any test run
- [ ] `completed == submitted` holds for every CQE drain (confirmed by eprintln logs)
- [ ] Realloc path exercised: `test_register_buffers` triggers `reregister_buffers` with a larger buffer
- [ ] Fallback path exercised: `test_fallback_when_path_not_registered` succeeds without file in registry
- [ ] Benchmark shows ≥ 10% CPU overhead reduction on model layer reads
