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

if __name__ == "__main__":
    test_init_ring_exists()
    test_register_buffers_exists()
    test_reregister_buffers_exists()
    test_register_files_exists()
    test_unregister_files_exists()
    print("all attribute checks passed")
    test_init_ring()
