"""R1: the io_uring read path must be silent unless VAAYU_RUST_TRACE asks.

Two unconditional `eprintln!` per read on the decode hot path is ~850 unbuffered
`write(2)` syscalls per token at Qwen's 422 reads/token, each taking the stderr
lock the I/O threads contend on -- and `vaayu/server/logs.py` now tees fd 2
through a GIL-taking Python pump thread. M1 needs the trace BACK for one arm,
so this is a gate, not a deletion.

Runs in a subprocess on purpose: the gate is resolved once per process from the
environment, so "off" and "on" cannot both be observed from one interpreter.
"""
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("vaayu_rust")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `init_pool` picks which read path runs: with a pool -> persistent_read,
# without -> legacy_read. Both are hot paths and both must be gated.
READER = textwrap.dedent(
    r"""
    import ctypes, os, sys, tempfile
    import vaayu_rust

    init_pool = sys.argv[1] == "pool"
    payload = b"x" * 65536
    fd, path = tempfile.mkstemp()
    os.write(fd, payload)
    os.close(fd)
    try:
        if init_pool:
            vaayu_rust.init_ring_pool(1, 8)
        # Everything before this marker is once-per-process setup logging, which
        # is not what R1 is about; the test only reads what follows it.
        sys.stderr.write("---READ-STARTS-HERE---\n")
        sys.stderr.flush()
        buf = ctypes.create_string_buffer(len(payload))
        # use_direct_io=False: a buffered read has no 4096-alignment contract on
        # the destination, so the probe needs no aligned allocator.
        vaayu_rust.io_uring_read_into(
            path, 0, len(payload), ctypes.addressof(buf), 8, False
        )
        assert buf.raw == payload, "read did not deliver the file"
    finally:
        os.unlink(path)
    """
)


def _run(pool: bool, trace: str | None):
    env = dict(os.environ)
    env.pop("VAAYU_RUST_TRACE", None)
    if trace is not None:
        env["VAAYU_RUST_TRACE"] = trace
    proc = subprocess.run(
        [sys.executable, "-c", READER, "pool" if pool else "nopool"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=120,
    )
    assert proc.returncode == 0, f"reader failed: {proc.stderr}"
    marker = "---READ-STARTS-HERE---"
    assert marker in proc.stderr, f"probe never reached the read: {proc.stderr}"
    return proc.stderr.split(marker, 1)[1]


@pytest.mark.parametrize("pool", [True, False], ids=["persistent", "legacy"])
def test_read_path_is_silent_by_default(pool):
    stderr = _run(pool, trace=None)
    noisy = [ln for ln in stderr.splitlines() if "[vaayu_rust]" in ln]
    assert noisy == [], f"hot path logged with no VAAYU_RUST_TRACE set: {noisy}"


@pytest.mark.parametrize("pool", [True, False], ids=["persistent", "legacy"])
def test_read_path_traces_when_env_asks(pool):
    stderr = _run(pool, trace="1")
    assert "[vaayu_rust]" in stderr, (
        "VAAYU_RUST_TRACE=1 produced no trace -- M1's ring-idle arm needs it back"
    )
