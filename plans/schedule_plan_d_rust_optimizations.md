# Schedule Plan D: Deep Rust Optimizations

**Priority:** 4 (The Final Polish)
**Effort:** Medium
**Risk:** Low

## Objective
Squeeze the absolute maximum capability out of the `io_uring` API and the Linux kernel now that the final access patterns (EXL3 compressed chunks) are locked in. 

## The Target
Because Phase B and Phase C will change our I/O from "large sequential reads" to "smaller, scattered expert chunk reads," we want to completely eliminate the per-SQE kernel overhead.

## Implementation Steps

### 1. `IORING_REGISTER_BUFFERS`
Currently, every `Read` operation forces the kernel to look up our pinned buffer pointer, lock the pages, map them for DMA, and then unmap them.
- **Action:** In `vaayu_rust/src/lib.rs`, add a module initialization step that registers the two `_PINNED_BUFS` pointers using `ring.submitter().register_buffers()`.
- **Action:** Switch the SQE opcodes from `opcode::Read` to `opcode::ReadFixed` passing index `0` or `1`.

### 2. `IORING_REGISTER_FILES`
Currently, passing a raw file descriptor requires the kernel to lock the fd table and increment reference counts on every I/O submission.
- **Architectural prerequisite:** The current Rust implementation opens each file fresh per layer call using `libc::open(..., O_RDONLY | O_DIRECT)` and closes it at function exit. `IORING_REGISTER_FILES` requires all file descriptors to be opened **once at module startup** and kept open for the lifetime of the process. This is a structural change to `lib.rs` — file management must be lifted out of `io_uring_read_into` and into a separate init/teardown API exposed to Python.
- **Action:** Add `vaayu_rust.register_files(file_paths: List[str])` and `vaayu_rust.unregister_files()` Python-callable functions to manage the persistent fd pool.
- **Action:** Switch the SQE targets from `types::Fd(raw_fd)` to `types::Fixed(fd_index)`.

## Verification
- These changes are highly sensitive to kernel versions. Ensure no `EINVAL` or `EFAULT` errors occur during execution.
- Measure the final throughput benchmark. We expect a final 10-15% reduction in CPU overhead, helping the system gracefully handle the scattered reads required by the MoE / EXL3 setup.
