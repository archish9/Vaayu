"""G1: the on-device HTTP gate must refuse to run under pytest's capture.

`tests/test_server_smoke.py` asserts on `server.log`, which is written by
`capture_stdio`'s fd-level pump. Under pytest's default capture a Python
`print()` never reaches fd 1, so that log holds only `vaayu_rust`'s native
writes and the gate fails with `no token summary carried a gb_tok= field` --
reading exactly like the byte-counter defect it exists to catch (2026-09-05).

These tests do not need a checkpoint: the guard fires before the module's
`client` fixture loads one.
"""
import os
import subprocess
import sys

import tests.test_server_smoke as smoke


def test_only_capture_no_lets_a_print_reach_the_pump():
    """Measured over all four modes 2026-09-05 (see the docstring of the
    predicate): `fd`, `sys` and `tee-sys` all rebind sys.stdout away from
    fd 1. `tee-sys` does too, despite the name -- do not widen this."""
    assert smoke._capture_hides_python_prints("fd")
    assert smoke._capture_hides_python_prints("sys")
    assert smoke._capture_hides_python_prints("tee-sys")
    assert not smoke._capture_hides_python_prints("no")


def test_the_gate_refuses_to_run_without_dash_s():
    """The whole point: without `-s` the operator gets a message naming `-s`,
    not an assertion about gb_tok. SERVER_CKPT points at a path that does not
    exist -- if the guard did not fire first, the run would try to load it."""
    env = dict(os.environ, SERVER_CKPT="/nonexistent/checkpoint-that-is-never-loaded")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_server_smoke.py::test_health",
         "-q", "-p", "no:cacheprovider"],
        env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out[-2000:]
    assert smoke._CAPTURE_REFUSAL in out, out[-3000:]
    assert "re-run with -s" in out, out[-3000:]
    # It must refuse BEFORE the load: the message is the whole value here,
    # and a run that reaches the loader has already paid for a checkpoint.
    assert "loading /nonexistent" not in out, out[-3000:]
    assert "gb_tok" not in out, out[-3000:]


def test_the_gate_still_skips_with_no_checkpoint():
    """The guard must not turn the default suite red: no SERVER_CKPT still
    skips the whole module, capture or not."""
    env = {k: v for k, v in os.environ.items() if k != "SERVER_CKPT"}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_server_smoke.py",
         "-q", "-p", "no:cacheprovider"],
        env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    assert "skipped" in out, out[-2000:]
