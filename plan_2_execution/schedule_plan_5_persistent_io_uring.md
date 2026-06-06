# Schedule Plan 5: Persistent io_uring Ring + Lifecycle

**Phase:** 2 (Core Performance Architecture)
**Priority:** High
**Effort:** Medium (1 day)
**Risk:** Medium (Requires modifying Rust bindings)
**Impact:** Eliminates 83 ring creation/destruction cycles per token and enables Fixed Files and Registered Buffers for a 12-25% throughput improvement.

## Objective
The current Rust `io_uring` backend creates a new ring, opens the `.safetensors` file, reads it, and drops everything on *every single layer read* (83 times per token). We need to create a persistent lifecycle API: initialize the ring and register buffers/files once at model load, read from the persistent state during generation, and cleanly shut down on exit.

## Context for the Coding Agent
This requires modifying both the Rust extension (`vaayu_rust/src/lib.rs`) and the Python I/O orchestrator (`vaayu/persist/safetensor_model_persister.py`). The state needs to be managed globally in Rust (e.g., using `lazy_static` or passing a state pointer back to Python) so that the `read_layer` function can use the pre-registered file descriptors and pinned PyTorch buffers without re-initializing them.

## Implementation Steps

### Step 1: Update Rust `lib.rs`

Modify `vaayu_rust/src/lib.rs` to expose three new functions instead of the single `io_uring_read_into` function. 

*Note: For state persistence in Rust exposed to Python, you can use a Python class wrapped via PyO3, or a global `Mutex<Option<IoUringState>>`.*

```rust
use pyo3::prelude::*;
use std::sync::Mutex;
// Note: io_uring and libc imports are required.
// Assume a global state for simplicity (or use a PyClass):

/*
struct GlobalIoRing {
    ring: io_uring::IoUring,
    files: Vec<std::fs::File>,
    // buffers...
}
lazy_static! {
    static ref IO_RING: Mutex<Option<GlobalIoRing>> = Mutex::new(None);
}
*/

#[pyfunction]
fn init_io_ring(
    file_paths: Vec<String>,
    buf_ptr_a: usize,
    buf_ptr_b: usize,
    buf_size: usize,
    queue_depth: u32,
    use_direct_io: bool,
) -> PyResult<()> {
    // 1. Open all files using libc::open with flags based on use_direct_io
    // 2. Create the io_uring ring
    // 3. Register the fixed files
    // 4. Register the two pinned buffers (Fixed Buffers)
    // 5. Store in global state or return a handle
    Ok(())
}

#[pyfunction]
fn read_layer(
    file_index: u32,
    offset: u64,
    length: usize,
    buf_index: u16,
) -> PyResult<()> {
    // 1. Fetch ring from state
    // 2. Submit ReadFixed SQE targeting the specific file_index and buf_index
    // 3. Wait for CQE
    Ok(())
}

#[pyfunction]
fn shutdown_io_ring() -> PyResult<()> {
    // Drop the ring, close files, unregister buffers
    Ok(())
}
```
*Be sure to run `maturin develop --release` inside the Rust directory to compile the changes.*

### Step 2: Python Lifecycle Management

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

In the initializer (where the paths and pinned buffers are created), add:

```python
import vaayu_rust
import atexit

# Assuming _PINNED_BUFS are allocated
all_paths = [str(p) for p in self._layer_paths]

vaayu_rust.init_io_ring(
    all_paths,
    _PINNED_BUFS[0].data_ptr(),
    _PINNED_BUFS[1].data_ptr(),
    _PINNED_BUFS[0].numel() * _PINNED_BUFS[0].element_size(),
    32,   # queue depth
    self.use_direct_io  # from Plan 3
)

atexit.register(vaayu_rust.shutdown_io_ring)
```

### Step 3: Replace the Hot Loop Call

In `_load_pinned()`, replace the call to `io_uring_read_into` with:

```python
vaayu_rust.read_layer(
    file_index=layer_idx,   # You must pass the index matching all_paths
    offset=header_size,
    length=data_size,
    buf_index=_BUF_IDX
)
```

## Verification Plan
1. Ensure `maturin develop --release` succeeds.
2. Run `python test_quick.py`.
3. Check `nvidia-smi` and the telemetry logs (if Plan 2 is done) to confirm `io_submit_us` and `io_wait_us` have dropped, and generation completes successfully without segfaults.
