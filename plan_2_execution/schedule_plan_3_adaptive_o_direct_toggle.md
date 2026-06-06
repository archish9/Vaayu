# Schedule Plan 3: Adaptive O_DIRECT Toggle

**Phase:** 1 (Measurement & Quick Wins)
**Priority:** High
**Effort:** Low (2 hours)
**Risk:** Low
**Impact:** Auto-optimizes I/O on machines with abundant RAM by relying on the Linux page cache, while protecting low-RAM systems with `O_DIRECT`.

## Objective
Currently, the Rust backend hardcodes `O_DIRECT` for all NVMe reads. `O_DIRECT` bypasses the Linux page cache, which is necessary on 32GB RAM machines to prevent memory trashing. However, on machines with 64GB+ of RAM, the Linux page cache could naturally hold the entire model (or a large portion of it), making subsequent reads practically instant. We need an adaptive toggle based on available system memory.

## Context for the Coding Agent
The `io_uring` implementation in `vaayu_rust` opens files with `libc::O_RDONLY | libc::O_DIRECT`. We need to pass a boolean flag from Python at initialization to tell Rust whether to include the `O_DIRECT` flag based on a heuristic of available system RAM vs total model size.

## Implementation Steps

### Step 1: Create RAM Heuristic

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

Add a function to check available RAM using `psutil`.

```python
import psutil
import os

def _should_use_direct_io(model_paths: list[str]) -> bool:
    """
    O_DIRECT wins when the page cache can't meaningfully cache the model.
    The crossover point is ~40% cacheable.
    """
    try:
        model_total_bytes = sum(os.path.getsize(p) for p in model_paths)
        mem = psutil.virtual_memory()
        
        # Reserve 8GB for OS + PyTorch
        available_for_cache = max(0, mem.available - (8 * 1024**3))  
        cacheable_fraction = available_for_cache / max(1, model_total_bytes)
        
        use_direct = cacheable_fraction < 0.40
        
        if not use_direct:
            print(f"[Vaayu] Sufficient RAM detected ({mem.available / 1024**3:.1f} GB free). "
                  f"Disabling O_DIRECT to leverage page cache for {cacheable_fraction*100:.0f}% of model.")
        else:
            print(f"[Vaayu] Limited RAM ({mem.available / 1024**3:.1f} GB free). "
                  f"Using O_DIRECT to avoid wasteful page cache copies.")
        
        return use_direct
    except Exception as e:
        print(f"[Vaayu] Failed to detect RAM ({e}), defaulting to O_DIRECT = True")
        return True
```

### Step 2: Pass Flag to Rust Backend

Currently, `vaayu_rust.io_uring_read_into` handles opening the file directly. 
*Note: If Plan 5 (Persistent io_uring Ring) has been implemented, this flag should be passed to `init_io_ring()` instead.*

Assuming we are passing it to the read function (or init function), update the Python call in `_load_pinned()`:

```python
# Check this ONCE at model init and store it
self.use_direct_io = _should_use_direct_io(self._layer_paths)

# Later in _load_pinned:
if hasattr(vaayu_rust, 'io_uring_read_into'):
    vaayu_rust.io_uring_read_into(path, offset, length, buf.data_ptr(), self.use_direct_io)
```

### Step 3: Update `vaayu_rust/src/lib.rs`

Modify the Rust function signature and file open logic:

```rust
#[pyfunction]
fn io_uring_read_into(
    file_path: &str,
    offset: u64,
    length: usize,
    buffer_ptr: usize,
    use_direct_io: bool, // NEW ARGUMENT
) -> PyResult<()> {
    
    // ...
    let flags = if use_direct_io {
        libc::O_RDONLY | libc::O_DIRECT
    } else {
        libc::O_RDONLY
    };
    
    let fd = unsafe {
        libc::open(
            file_path_c.as_ptr(),
            flags
        )
    };
    // ...
}
```

*Don't forget to run `maturin develop --release` to rebuild the Rust extension.*

## Verification Plan
1. Run `python test_quick.py`.
2. Check the console output for the log `[Vaayu] Limited RAM (...) Using O_DIRECT`.
3. If running on a 32GB machine, it should default to `True`.
