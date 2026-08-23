"""Logs on disk: one rotating server log, one file per session.

Two capture problems, not one. Python `print()` goes through sys.stdout, but
vaayu_rust logs with `eprintln!` straight to fd 2 (vaayu_rust/src/lib.rs:72) --
replacing sys.stderr in Python never sees it, and the io_uring lines are exactly
what the disk-streaming path is debugged with. So the capture is at fd level.

`session_id` arrives from an HTTP body, so it is untrusted input that becomes a
FILENAME. That is the sharpest edge in this file.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from vaayu.server.logs import (UNNAMED, sanitize_session, session_event,
                               session_log_path)


# ── session_id -> filename ───────────────────────────────────────────────────

def test_an_ordinary_id_is_used_as_is():
    assert sanitize_session("agent-7-turn-3") == "agent-7-turn-3"


def test_an_unnamed_request_gets_one_shared_file():
    assert sanitize_session(None) == UNNAMED
    assert sanitize_session("") == UNNAMED


def test_path_traversal_cannot_escape_the_log_directory(tmp_path):
    evil = "../../../../etc/cron.d/pwned"
    safe = sanitize_session(evil)
    assert "/" not in safe and ".." not in safe
    path = session_log_path(tmp_path, evil)
    assert tmp_path in path.parents, f"{path} escaped {tmp_path}"


def test_separators_and_nul_are_stripped():
    for hostile in ("a/b", "a\\b", "a\x00b", "/etc/passwd", "."):
        safe = sanitize_session(hostile)
        assert "/" not in safe and "\\" not in safe and "\x00" not in safe
        assert safe not in (".", "..")


def test_two_different_hostile_ids_do_not_collide():
    # Stripping unsafe characters alone would map ../a and ..\a onto one file,
    # silently interleaving two sessions' logs.
    assert sanitize_session("../../a") != sanitize_session("..\\..\\a")


def test_a_very_long_id_is_bounded_but_still_distinct():
    a, b = sanitize_session("x" * 500), sanitize_session("x" * 499)
    assert len(a) <= 120 and a != b


def test_unicode_survives_without_becoming_a_path():
    safe = sanitize_session("agent-café-7")
    assert "/" not in safe and safe


# ── writing ──────────────────────────────────────────────────────────────────

def test_an_event_lands_in_that_session_file(tmp_path):
    session_event(tmp_path, "agent-7", "REQ", prompt=44, cap=8148)
    text = session_log_path(tmp_path, "agent-7").read_text()
    assert "REQ" in text and "prompt=44" in text and "cap=8148" in text


def test_events_append_rather_than_truncate(tmp_path):
    session_event(tmp_path, "agent-7", "REQ", prompt=44)
    session_event(tmp_path, "agent-7", "DONE", tokens=996)
    text = session_log_path(tmp_path, "agent-7").read_text()
    assert "REQ" in text and "DONE" in text


def test_two_sessions_do_not_share_a_file(tmp_path):
    session_event(tmp_path, "a", "REQ", prompt=1)
    session_event(tmp_path, "b", "REQ", prompt=2)
    assert "prompt=2" not in session_log_path(tmp_path, "a").read_text()


def test_every_line_is_timestamped(tmp_path):
    session_event(tmp_path, "a", "REQ", prompt=1)
    first = session_log_path(tmp_path, "a").read_text().splitlines()[0]
    assert first[:4].isdigit(), f"expected an ISO date, got {first!r}"


def test_logging_never_raises_into_the_request_path(tmp_path):
    # A full disk or a read-only mount must not turn into a 500 on generation.
    unwritable = tmp_path / "nope"
    unwritable.write_text("i am a file, not a directory")
    session_event(unwritable, "a", "REQ", prompt=1)      # must not raise


# ── fd-level capture (subprocess: it rewires fd 1 and 2) ─────────────────────

CAPTURE_SCRIPT = """
import os, sys
sys.path.insert(0, {repo!r})
from vaayu.server.logs import capture_stdio
restore = capture_stdio({path!r})
print("python-stdout-line")
os.write(2, b"[vaayu_rust] fd2-line\\n")      # what eprintln! does
sys.stdout.flush()
restore()
print("after-restore-goes-to-terminal")
"""


def test_fd_level_capture_gets_python_and_raw_fd_writes(tmp_path):
    log = tmp_path / "server.log"
    out = subprocess.run(
        [sys.executable, "-c", CAPTURE_SCRIPT.format(
            repo=str(Path(__file__).resolve().parents[1]), path=str(log))],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    written = log.read_text()
    assert "python-stdout-line" in written
    assert "[vaayu_rust] fd2-line" in written, (
        "a Python-level tee misses raw fd writes -- that is the whole point")
    assert "after-restore-goes-to-terminal" not in written, "restore() must detach"


def test_capture_still_reaches_the_terminal(tmp_path):
    log = tmp_path / "server.log"
    out = subprocess.run(
        [sys.executable, "-c", CAPTURE_SCRIPT.format(
            repo=str(Path(__file__).resolve().parents[1]), path=str(log))],
        capture_output=True, text=True, timeout=60)
    assert "python-stdout-line" in out.stdout, (
        "capturing must not blind the console you are watching")


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_defaults_are_a_logs_dir_with_capture_on():
    from vaayu.server.app import ServerConfig
    cfg = ServerConfig(model_path="/x")
    assert cfg.log_dir == "logs"
    assert cfg.capture_stdio is True


def test_a_request_writes_req_and_done_for_its_session(tmp_path):
    import asyncio

    import torch

    from vaayu.server.engine import EngineHandle
    from vaayu.server.schemas import ChatCompletionRequest
    from .test_server_engine import FakeModel

    class M(FakeModel):
        def __init__(self):
            super().__init__()
            self.tokenizer.decode = lambda ids, **k: "x" * len(ids)
            self.config = type("C", (), {"eos_token_id": 2})()

        def generate(self, **kw):
            return torch.cat([kw["input_ids"], torch.tensor([[7, 7, 7]])], dim=1)

    e = EngineHandle(M(), log_dir=tmp_path)
    req = ChatCompletionRequest(model="v", session_id="agent-7", max_tokens=16,
                                messages=[{"role": "user", "content": "hi"}])
    asyncio.run(e.generate(req))

    text = session_log_path(tmp_path, "agent-7").read_text()
    assert "REQ" in text and "DONE" in text
    assert "tokens=3" in text, text
    assert "finish=" in text and "wall=" in text


def test_an_unnamed_request_is_logged_under_the_shared_file(tmp_path):
    import asyncio

    import torch

    from vaayu.server.engine import EngineHandle
    from vaayu.server.schemas import ChatCompletionRequest
    from .test_server_engine import FakeModel

    class M(FakeModel):
        def __init__(self):
            super().__init__()
            self.tokenizer.decode = lambda ids, **k: "x"
            self.config = type("C", (), {"eos_token_id": 2})()

        def generate(self, **kw):
            return torch.cat([kw["input_ids"], torch.tensor([[7]])], dim=1)

    e = EngineHandle(M(), log_dir=tmp_path)
    asyncio.run(e.generate(ChatCompletionRequest(
        model="v", max_tokens=8, messages=[{"role": "user", "content": "hi"}])))
    assert session_log_path(tmp_path, None).exists()


def test_no_log_dir_means_no_files_and_no_errors(tmp_path):
    import asyncio

    import torch

    from vaayu.server.engine import EngineHandle
    from vaayu.server.schemas import ChatCompletionRequest
    from .test_server_engine import FakeModel

    class M(FakeModel):
        def __init__(self):
            super().__init__()
            self.tokenizer.decode = lambda ids, **k: "x"
            self.config = type("C", (), {"eos_token_id": 2})()

        def generate(self, **kw):
            return torch.cat([kw["input_ids"], torch.tensor([[7]])], dim=1)

    e = EngineHandle(M())                       # library default: no logging
    asyncio.run(e.generate(ChatCompletionRequest(
        model="v", max_tokens=8, messages=[{"role": "user", "content": "hi"}])))
    assert not list(tmp_path.iterdir())
