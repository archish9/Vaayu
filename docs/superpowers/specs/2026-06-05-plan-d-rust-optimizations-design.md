# Plan D: Deep Rust io_uring Optimizations — Design Spec

**Date:** 2026-06-05  
**Status:** Approved  
**Phase:** Final polish — executes after Plan H (Speculative Decoding)  
**Files touched:** `vaayu_rust/src/lib.rs`, `vaayu/persist/safetensor_model_persister.py`

---

## Objective

Eliminate per-SQE kernel overhead in the io_uring read path by pre-registering pinned memory buffers (`IORING_REGISTER_BUFFERS`) and file descriptors (`IORING_REGISTER_FILES`). Expected result: 10-15% reduction in CPU overhead on model layer loads, making scattered expert-chunk reads under EXL3 + speculative decoding more efficient.

---

## Architecture

### Rust Global State

A module-level `static RING_STATE: Mutex<Option<RingState>>` holds the persistent ring and all registration metadata.

```
RingState {
    ring: IoUring,
    queue_depth: usize,
    buf_regs: Option<BufRegistrations>,
    file_map: HashMap<String, u32>,   // path → fixed fd index
    raw_fds: Vec<RawFd>,              // kept open; closed on unregister_files
}

BufRegistrations {
    ptr0: usize, len0: usize,
    ptr1: usize, len1: usize,
}
```

Ring is created once via `init_ring()` and lives for the process lifetime. The Python-side `_BUF_LOCK` prevents concurrent `_load_pinned` calls, so Mutex contention on the ring is never hot.

---

## Rust API Surface

All functions are Python-callable via PyO3.

| Function | Purpose |
|----------|---------|
| `init_ring(queue_depth: u32)` | Create persistent IoUring ring. Called once at first model load. |
| `register_buffers(ptr0, len0, ptr1, len1)` | Register both pinned buffers with the ring via `IORING_REGISTER_BUFFERS`. |
| `reregister_buffers(ptr0, len0, ptr1, len1)` | Replace buffer registration after Python reallocs a pinned buffer. |
| `register_files(paths: Vec<String>)` | Open all `.safetensors` files with `O_RDONLY \| O_DIRECT`, register fds via `IORING_REGISTER_FILES`. |
| `unregister_files()` | Close all pre-opened fds, clear fd map. |
| `io_uring_read_into(py, path, offset, length, buf_ptr, queue_depth)` | **Unchanged signature.** Internally dispatches to fixed or non-fixed path. |
| `pread_into(...)` | Unchanged fallback. |

### `io_uring_read_into` Dispatch Logic

```
if RING_STATE has ring + buf_regs + path in file_map:
    → ReadFixed(buf_index) + Fixed(fd_index)       # fully optimized path
elif ring + buf_regs:
    → ReadFixed(buf_index) + Fd(fresh_open)        # buffer optimized only
else:
    → Read(fresh_buf_ptr) + Fd(fresh_open)         # old path (fallback)
```

`buf_index` (0 or 1) is determined by checking which registered buffer range `buf_ptr` falls into:
- `buf_regs.ptr0 <= buf_ptr < ptr0 + len0` → index 0
- `buf_regs.ptr1 <= buf_ptr < ptr1 + len1` → index 1
- Neither → fallback to `Read`

The `queue_depth` argument to `io_uring_read_into` is retained for API compatibility but ignored when using the persistent ring (ring was sized at `init_ring` time).

---

## Python Side Changes (`safetensor_model_persister.py`)

### New module-level globals

```python
_RING_INITIALIZED = False
_FILES_REGISTERED = False
```

### New init helpers

```python
def _init_iouring_ring(queue_depth: int = 32):
    vaayu_rust.init_ring(queue_depth)

def _register_iouring_files(model_dir: str):
    paths = [str(p) for p in Path(model_dir).glob("*.safetensors")]
    vaayu_rust.register_files(paths)
```

### `load_model` — lazy one-time init

```python
def load_model(self, layer_name, path):
    global _RING_INITIALIZED, _FILES_REGISTERED
    if _USE_IOURING and not _RING_INITIALIZED:
        _init_iouring_ring(32)
        _RING_INITIALIZED = True
    if _USE_IOURING and not _FILES_REGISTERED:
        _register_iouring_files(str(path))
        _FILES_REGISTERED = True
    # ... existing logic unchanged
```

### `_load_pinned` — reregister after buffer realloc

After the buffer realloc block (current lines 85-90), add:

```python
if _USE_IOURING and _RING_INITIALIZED:
    vaayu_rust.reregister_buffers(
        _PINNED_BUFS[0].data_ptr(), _PINNED_SIZES[0],
        _PINNED_BUFS[1].data_ptr(), _PINNED_SIZES[1],
    )
```

No changes to the `io_uring_read_into` call site — dispatch is transparent inside Rust.

---

## Error Handling

| Scenario | Behavior |
|----------|---------|
| `register_buffers` returns `EINVAL`/`EFAULT` | Python catches `PyRuntimeError`, sets `_RING_INITIALIZED = False`, falls back to old path. Logged to stderr. |
| `register_files` returns error | Python catches, sets `_FILES_REGISTERED = False`. Ring still usable with buffer registration only. |
| `buf_ptr` out of registered range at read time | Rust detects via range check, silently uses `Read` + fresh `Fd` for that call. Covers realloc race window. |
| `unregister_files` not called on exit | Kernel reclaims fds on process exit. Acceptable. |

Kernel version requirement: `IORING_REGISTER_BUFFERS` and `IORING_REGISTER_FILES` both stable since Linux 5.1.

---

## Verification

1. Run inference with `RUST_LOG=debug`. Confirm no `EINVAL`/`EFAULT` in stderr or dmesg.
2. Confirm `completed == submitted` still holds in CQE drain loop for all layer loads.
3. Benchmark: compare CPU time (not wall-clock GB/s, which is NVMe-bound) before and after. Expect 10-15% reduction in syscall overhead on the io_uring submission path.
4. Test realloc path: load a layer larger than initial alloc, confirm `reregister_buffers` is called and subsequent loads still use `ReadFixed`.
