"""S22-A2: `vaayu_rust.io_uring_read_segments` -- several regions of one file, each
into its own destination, in one io_uring wave.

The wave itself (one submit for every segment, short reads, error drain) is unit
tested in Rust (`cargo test` in vaayu_rust/). These tests cover what only Python
can see: the installed extension has the symbol, the call delivers the right
bytes with and without a ring pool, and the production combination -- ring pool
+ registered fds + O_DIRECT -- works through the Python entry point.
"""
import ctypes
import os
import subprocess
import sys
import textwrap

import pytest

vaayu_rust = pytest.importorskip("vaayu_rust")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _payload(n: int) -> bytes:
    return bytes((i * 7 + 3) % 251 for i in range(n))


def test_extension_exposes_read_segments():
    assert hasattr(vaayu_rust, "io_uring_read_segments"), (
        "stale vaayu_rust: run `pip install -U vaayu-rust`")


def test_read_segments_delivers_each_segment(tmp_path):
    payload = _payload(300_000)
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    spans = [(0, 4096), (100_000, 50_000), (250_000, 8192)]
    bufs = [ctypes.create_string_buffer(n) for _, n in spans]
    vaayu_rust.io_uring_read_segments(
        str(f), [(off, n, ctypes.addressof(b)) for (off, n), b in zip(spans, bufs)], 8, False)
    for (off, n), b in zip(spans, bufs):
        assert b.raw == payload[off:off + n]


def test_read_segments_with_no_segments_is_a_no_op(tmp_path):
    vaayu_rust.io_uring_read_segments(str(tmp_path / "never-opened.bin"), [], 8, False)


def test_missing_file_raises_and_the_next_read_works(tmp_path):
    payload = _payload(16_384)
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    buf = ctypes.create_string_buffer(8192)
    seg = [(4096, 8192, ctypes.addressof(buf))]
    with pytest.raises(OSError):
        vaayu_rust.io_uring_read_segments(str(tmp_path / "missing.bin"), seg, 8, False)
    vaayu_rust.io_uring_read_segments(str(f), seg, 8, False)
    assert buf.raw == payload[4096:4096 + 8192]


# The production combination, in a subprocess: the ring pool is process-global,
# so initialising it here would change how every later test in this process reads.
POOLED = textwrap.dedent(
    r"""
    import ctypes, mmap, os, sys
    import vaayu_rust

    path, direct = sys.argv[1], sys.argv[2] == "direct"
    payload = open(path, "rb").read()
    vaayu_rust.init_ring_pool(2, 32)
    if direct:
        vaayu_rust.register_files([path], True)
    # 4096-aligned offsets and lengths, page-aligned destinations: O_DIRECT's contract.
    spans = [(0, 4096), (1 << 20, 1_576_960), (3 << 20, 200_704)]
    maps = [mmap.mmap(-1, n) for _, n in spans]
    segs = [(off, n, ctypes.addressof(ctypes.c_char.from_buffer(m))) for (off, n), m in zip(spans, maps)]
    for _ in range(8):          # more calls than rings: every ring serves several
        vaayu_rust.io_uring_read_segments(path, segs, 8, direct)
        for (off, n), m in zip(spans, maps):
            assert m[:n] == payload[off:off + n], f"segment at {off} got the wrong bytes"
    print("POOLED-OK")
    """
)


@pytest.mark.parametrize("mode", ["buffered", "direct"])
def test_pooled_read_segments_in_a_subprocess(tmp_path, mode):
    f = tmp_path / "blob.bin"
    f.write_bytes(_payload(4 << 20))
    path = os.path.realpath(str(f))      # register_files keys are canonical absolute paths
    if mode == "direct":
        try:
            os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
        except OSError:
            pytest.skip("this filesystem refuses O_DIRECT")
    proc = subprocess.run([sys.executable, "-c", POOLED, path, mode],
                          capture_output=True, text=True, cwd=REPO, timeout=120)
    assert proc.returncode == 0 and "POOLED-OK" in proc.stdout, proc.stderr[-2000:]
