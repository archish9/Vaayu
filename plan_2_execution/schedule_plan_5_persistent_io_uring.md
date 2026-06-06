# Schedule Plan 5: Persistent io_uring Ring + Lifecycle

**Phase:** 2 (Core Performance Architecture)
**Priority:** High
**Effort:** Low (2-3 hours) — core is already implemented
**Risk:** Low
**Impact:** Completes the lifecycle wiring and enables the `use_direct_io` flag (prerequisite for Plan 3).

## Current State (Verified 2026-06-06)

> [!IMPORTANT]
> The persistent ring is **already implemented**. The original plan spec was written before this work landed. Do NOT re-implement from scratch.

**What already exists in `vaayu_rust/src/lib.rs`:**
- `RING_STATE: Mutex<Option<RingState>>` — global persistent ring state
- `init_ring(queue_depth)` — creates persistent `IoUring` ring
- `register_buffers(ptr0, len0, ptr1, len1)` — registers pinned PyTorch buffers with io_uring
- `reregister_buffers(...)` — re-registers after buffer reallocation
- `register_files(paths)` — registers all `.safetensors` file descriptors
- `unregister_files()` — unregisters fds
- `persistent_read(...)` — private fn: uses `ReadFixed` + `Fixed(fd)` when registered, falls back gracefully
- `io_uring_read_into(path, offset, length, buf_ptr, queue_depth)` — already routes to `persistent_read()` if ring is initialized, falls back to `legacy_read()` if not

**What already exists in `vaayu/persist/safetensor_model_persister.py`:**
- `_RING_INITIALIZED`, `_FILES_REGISTERED` flags
- `_init_iouring_ring(queue_depth=32)` — calls `vaayu_rust.init_ring()`
- `_register_iouring_files(model_dir)` — calls `vaayu_rust.register_files()`
- Both are called lazily on the first `_load_pinned()` call (lines 208–211)

**What is still missing (the actual work):**

1. `persistent_read` and `legacy_read` hardcode `libc::O_RDONLY | libc::O_DIRECT` — no adaptive flag. Plan 3 depends on this.
2. No `atexit` shutdown — `unregister_files()` is never called at process exit.
3. Ring init is lazy (first layer load). Moving it to explicit model-load time eliminates setup overhead on Token 1.

---

## Implementation Steps

### Step 1: Add `use_direct_io` flag to Rust open calls

**File to modify:** `vaayu_rust/src/lib.rs`

`O_DIRECT` is hardcoded in two places. Both need the flag:

**In `persistent_read` (line ~206) — the fresh-file fallback path:**
```rust
fn persistent_read(
    state: &mut RingState,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
    use_direct_io: bool,  // NEW
) -> PyResult<()> {
    // ...
    let flags = if use_direct_io {
        libc::O_RDONLY | libc::O_DIRECT
    } else {
        libc::O_RDONLY
    };
    fresh_raw_fd = unsafe { libc::open(c_path.as_ptr(), flags) };
```

**In `legacy_read` (line ~305) — the no-ring fallback:**
```rust
fn legacy_read(path: &str, offset: u64, length: usize, buf_ptr: usize, queue_depth: u32, use_direct_io: bool) -> PyResult<()> {
    let flags = if use_direct_io {
        libc::O_RDONLY | libc::O_DIRECT
    } else {
        libc::O_RDONLY
    };
    let raw_fd = unsafe { libc::open(c_path.as_ptr(), flags) };
```

**In `register_files` (line ~93) — for pre-registered fds, O_DIRECT must match what was used at registration time:**
```rust
fn register_files(py: Python<'_>, paths: Vec<String>, use_direct_io: bool) -> PyResult<()> {
    // open each path with the correct flags before registering
    let flags = if use_direct_io { libc::O_RDONLY | libc::O_DIRECT } else { libc::O_RDONLY };
```

**Update `io_uring_read_into` to pass the flag through:**
```rust
#[pyfunction]
fn io_uring_read_into(
    py: Python<'_>,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
    queue_depth: u32,
    use_direct_io: bool,  // NEW — defaults to true in Python call
) -> PyResult<()> {
    let path = path.to_owned();
    py.allow_threads(move || {
        let mut guard = RING_STATE.lock().unwrap();
        if let Some(ref mut state) = *guard {
            persistent_read(state, &path, offset, length, buf_ptr, use_direct_io)
        } else {
            drop(guard);
            legacy_read(&path, offset, length, buf_ptr, queue_depth, use_direct_io)
        }
    })
}
```

Run `maturin develop --release` after this change.

---

### Step 2: Pass `use_direct_io` from Python

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

The `_should_use_direct_io()` heuristic (from Plan 3) computes the flag once at init. Store it as a module-level variable and pass it through:

```python
_USE_DIRECT_IO: bool = True  # set by init_model_io() below, default safe

def init_model_io(model_dir: str, use_direct_io: bool) -> None:
    """Called once at model load. Initializes ring, registers files and buffers."""
    global _RING_INITIALIZED, _FILES_REGISTERED, _USE_DIRECT_IO
    _USE_DIRECT_IO = use_direct_io

    if _USE_IOURING and not _RING_INITIALIZED:
        _RING_INITIALIZED = _init_iouring_ring(32)
    if _USE_IOURING and _RING_INITIALIZED and not _FILES_REGISTERED:
        _FILES_REGISTERED = _register_iouring_files(model_dir, use_direct_io)
```

Update `_register_iouring_files` to accept and pass the flag:
```python
def _register_iouring_files(model_dir: str, use_direct_io: bool = True) -> bool:
    try:
        paths = [str(p) for p in Path(model_dir).glob("*.safetensors")]
        vaayu_rust.register_files(paths, use_direct_io)  # pass flag to Rust
        ...
```

Update the hot path in `_load_pinned` to pass `_USE_DIRECT_IO`:
```python
vaayu_rust.io_uring_read_into(
    file_path,
    aligned_offset,
    aligned_length,
    buf.data_ptr(),
    32,
    _USE_DIRECT_IO,   # NEW — was missing, Rust now requires it
)
```

Remove the lazy init from `_load_pinned()` (lines 207–211) — replace with explicit call at model load.

---

### Step 3: Explicit init at model load (not lazy)

**File to modify:** `vaayu/vaayu_base.py`

In `__init__()`, after `checkpoint_path` is resolved and before building the loader strategy:

```python
from .persist.safetensor_model_persister import init_model_io
from .persist.ram_cache import _should_use_direct_io  # Plan 3 heuristic

if device.startswith("cuda"):
    use_direct = _should_use_direct_io(str(self.checkpoint_path))
    init_model_io(str(self.checkpoint_path), use_direct_io=use_direct)
```

---

### Step 4: Register atexit shutdown

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

Add at module level, after the `_USE_IOURING` block:

```python
import atexit

def _shutdown_io():
    if _USE_IOURING and _FILES_REGISTERED:
        try:
            vaayu_rust.unregister_files()
        except Exception:
            pass

atexit.register(_shutdown_io)
```

---

## Verification Plan

1. `maturin develop --release` succeeds with no type errors.
2. Run `python test_quick.py`. Check logs:
   - `[VAAYU-IO] persistent ring initialized` appears at startup (not mid-generation)
   - `[VAAYU-IO] files registered with io_uring` appears at startup
   - `[vaayu_rust] persistent buf_index=Some(...) fixed_fd=Some(...)` in stderr — confirms both `ReadFixed` + `Fixed(fd)` are active
3. No `EFAULT` or `O_DIRECT` errors in output.
4. Generation completes successfully end-to-end.
