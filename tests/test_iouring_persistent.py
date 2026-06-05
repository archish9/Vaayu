"""
Plan D integration tests — persistent io_uring ring with IORING_REGISTER_BUFFERS and IORING_REGISTER_FILES.

Run on the Linux target machine (requires io_uring + CUDA + PyTorch pin_memory):
    python tests/test_iouring_persistent.py

Benchmark (substitute real model layer path):
    # Before (per-call ring, no registrations):
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

    # After (persistent ring + registered buffers + registered files):
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

Expected: 10-15% reduction in avg ms (CPU overhead; NVMe wall-clock may vary).

Error reference:
    CQE result -22 (EINVAL): offset or length not sector-aligned (must be multiples of 4096 for O_DIRECT)
    CQE result  -9 (EBADF):  fixed fd index out of range — check register_files was called
    CQE result -14 (EFAULT): buf ptr outside registered buffer range — check register_buffers sizes
    EINVAL from register_buffers: check kernel >= 5.1 (uname -r)
"""
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

def test_init_ring():
    # Idempotent — calling twice must not error
    vaayu_rust.init_ring(32)
    vaayu_rust.init_ring(32)
    print("init_ring: ok")

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

def test_persistent_read_correctness():
    """Write known bytes to a temp file, read back via persistent ring, verify content."""
    import os, tempfile
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
        vaayu_rust.io_uring_read_into(path, 0, length, buf0.data_ptr(), 32)

        result = bytes(buf0[:length].numpy())
        assert result == payload, f"Data mismatch: first byte got {result[0]}, want {payload[0]}"
        print("persistent_read_correctness: ok")
    finally:
        vaayu_rust.unregister_files()
        os.unlink(path)


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
    print("all attribute checks passed")
    test_init_ring()
    test_register_buffers()
    test_register_files()
    test_persistent_read_correctness()
    test_fallback_when_path_not_registered()
    print("=== all tests passed ===")
