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
