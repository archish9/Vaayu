# Schedule Plan 3: Adaptive O_DIRECT Toggle

**Phase:** 1 (Measurement & Quick Wins)
**Priority:** High
**Effort:** Low (2 hours)
**Risk:** Low
**Impact:** Auto-optimizes I/O on machines with abundant RAM by relying on the Linux page cache, while protecting low-RAM systems with `O_DIRECT`.

## Objective
Currently, the Rust backend hardcodes `O_DIRECT` for all NVMe reads. `O_DIRECT` bypasses the Linux page cache, which is necessary on 32GB RAM machines to prevent memory trashing. However, on machines with 64GB+ of RAM, the Linux page cache could naturally hold the entire model (or a large portion of it), making subsequent reads practically instant. We need an adaptive toggle based on available system memory.

## Context for the Coding Agent

> [!IMPORTANT]
> `O_DIRECT` is **not** set in `io_uring_read_into`. It is hardcoded in two internal Rust functions: `persistent_read` (line ~206 in `lib.rs`) and `legacy_read` (line ~305). Both open files with `libc::O_RDONLY | libc::O_DIRECT`. The Python flag must be passed through `register_files()` and `io_uring_read_into()` down into these functions.
>
> Plan 5 handles the Rust changes. Plan 3 only implements the Python-side heuristic and ensures the flag is plumbed correctly. Implement Plan 3 first to define `_should_use_direct_io`, then Plan 5 to wire it into Rust.

## Implementation Steps

### Step 1: Create RAM Heuristic

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

Add a function to check available RAM using `psutil`. Place it near the top, before `_load_pinned`.

```python
import psutil
import os

def _should_use_direct_io(model_dir: str) -> bool:
    """
    O_DIRECT wins when the page cache can't meaningfully cache the model.
    The crossover point is ~40% cacheable (empirically validated).
    """
    try:
        paths = list(Path(model_dir).glob("*.safetensors"))
        model_total_bytes = sum(os.path.getsize(p) for p in paths)
        mem = psutil.virtual_memory()

        # Reserve 8GB for OS + PyTorch + pinned buffers
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

### Step 2: Call heuristic at model load, store as module-level flag

**File to modify:** `vaayu/persist/safetensor_model_persister.py`

Add a module-level variable that will be set by `init_model_io()` (defined in Plan 5):

```python
_USE_DIRECT_IO: bool = True  # safe default; overridden by init_model_io()
```

This variable is read by Plan 5's `init_model_io()` and passed to Rust. No changes to `_load_pinned()` are needed in Plan 3 — Plan 5 handles plumbing it to `io_uring_read_into`.

### Step 3: Call from `vaayu_base.py`

**File to modify:** `vaayu/vaayu_base.py`

In `__init__()`, after `checkpoint_path` is resolved (before loader strategy build):

```python
from .persist.safetensor_model_persister import _should_use_direct_io

if device.startswith("cuda"):
    self._use_direct_io = _should_use_direct_io(str(self.checkpoint_path))
```

Pass `self._use_direct_io` to `init_model_io()` in Plan 5.

> [!NOTE]
> The Rust changes (adding `use_direct_io: bool` param to `persistent_read`, `legacy_read`, `register_files`, and `io_uring_read_into`) are described in **Plan 5 Step 1**. Do not duplicate them here.

## Dependency

`psutil` is not yet in `requirements.txt`. Add it before implementing:
```
psutil>=5.9
```
Plan 6 also uses `psutil` — add once, covers both.

## Verification Plan
1. Run `python test_quick.py`.
2. On a 32GB machine: `[Vaayu] Limited RAM (...) Using O_DIRECT` should appear at startup.
3. On a 64GB+ machine: `[Vaayu] Sufficient RAM (...) Disabling O_DIRECT` should appear.
4. Generation completes correctly in both cases.
