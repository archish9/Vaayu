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


# `print()` to a non-tty is block-buffered, so a message can still be sitting in
# Python's buffer when restore() detaches -- and then it flushes to the console
# the dup2 just restored, never reaching the log. That is the failure mode on a
# server started with its output redirected to a file, which is how this one is
# meant to be run (B10 Fix 2).
BUFFERED_SCRIPT = """
import os, sys
sys.path.insert(0, {repo!r})
from vaayu.server.logs import capture_stdio
restore = capture_stdio({path!r})
print("buffered-before-restore")          # deliberately NOT flushed
restore()
"""


def test_restore_flushes_python_buffered_output_into_the_log(tmp_path):
    log = tmp_path / "server.log"
    out = subprocess.run(
        [sys.executable, "-c", BUFFERED_SCRIPT.format(
            repo=str(Path(__file__).resolve().parents[1]), path=str(log))],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "buffered-before-restore" in log.read_text(), (
        "restore() detached before flushing, so the message went to the console "
        "instead of the log")


# ── run directories ──────────────────────────────────────────────────────────

def test_a_run_directory_is_created_with_a_sessions_subdir(tmp_path):
    from vaayu.server.logs import new_run_dir
    run = new_run_dir(tmp_path)
    assert run.is_dir() and run.name.startswith("run-")
    assert (run / "sessions").is_dir()
    assert run.parent == tmp_path


def test_two_starts_get_two_directories(tmp_path):
    from vaayu.server.logs import new_run_dir
    a, b = new_run_dir(tmp_path), new_run_dir(tmp_path)
    assert a != b, "a second start must not append to the first run's files"


def test_latest_points_at_the_newest_run(tmp_path):
    from vaayu.server.logs import new_run_dir
    new_run_dir(tmp_path)
    b = new_run_dir(tmp_path)
    link = tmp_path / "latest"
    assert link.is_symlink() and link.resolve() == b.resolve()


def test_latest_is_a_relative_link_so_the_bundle_survives_being_moved(tmp_path):
    # The whole point is that a user zips this directory and mails it. An
    # absolute target would resolve to the maintainer's disk, or to nothing.
    import os
    from vaayu.server.logs import new_run_dir
    run = new_run_dir(tmp_path)
    assert os.readlink(tmp_path / "latest") == run.name


def test_a_failed_symlink_does_not_stop_the_run(tmp_path, monkeypatch, capsys):
    import os
    from vaayu.server import logs
    monkeypatch.setattr(os, "symlink",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    run = logs.new_run_dir(tmp_path)             # must not raise
    assert run.is_dir()
    assert "WARNING" in capsys.readouterr().out


# ── line treatment ───────────────────────────────────────────────────────────

def test_every_line_gets_a_timestamp_and_a_session_field():
    from vaayu.server.logs import format_line
    out = format_line("hello", session=None, stamp="2026-09-03T17:03:11.000Z")
    assert out == "2026-09-03T17:03:11.000Z session=- hello\n"


def test_a_named_session_is_recorded_on_the_line():
    from vaayu.server.logs import format_line
    out = format_line("hello", session="agent-7", stamp="S")
    assert out == "S session=agent-7 hello\n"


def test_tqdm_redraws_collapse_to_the_final_frame():
    # A prefill emits hundreds of \r-framed progress redraws. Kept raw they
    # collapse into one unreadable megaline in any editor.
    from vaayu.server.logs import format_line
    raw = "layers  1%|  | 1/94\rlayers 50%|##| 47/94\rlayers 100%|####| 94/94"
    out = format_line(raw, stamp="S")
    assert "94/94" in out
    assert "1/94" not in out and "47/94" not in out
    assert out.count("\n") == 1


def test_ansi_colour_is_stripped():
    from vaayu.server.logs import format_line
    out = format_line("\x1b[31mred\x1b[0m", stamp="S")
    assert out == "S session=- red\n"
    assert "\x1b" not in out


def test_a_line_that_is_empty_after_treatment_is_dropped():
    from vaayu.server.logs import format_line
    assert format_line("\r", stamp="S") == ""
    assert format_line("   ", stamp="S") == ""


def test_a_real_timestamp_is_iso8601_utc():
    from vaayu.server.logs import format_line
    out = format_line("x")
    stamp = out.split(" ", 1)[0]
    assert stamp.endswith("Z") and stamp[4] == "-" and stamp[10] == "T"


CAPTURE_SCRIPT_TQDM = """
import os, sys
sys.path.insert(0, {repo!r})
from vaayu.server.logs import capture_stdio
restore = capture_stdio({path!r})
sys.stdout.write("frame 1/3\\rframe 2/3\\rframe 3/3\\n")
sys.stdout.flush()
restore()
"""


def test_the_file_is_treated_but_the_console_is_not(tmp_path):
    import subprocess, sys
    from pathlib import Path
    log = tmp_path / "server.log"
    out = subprocess.run(
        [sys.executable, "-c", CAPTURE_SCRIPT_TQDM.format(
            repo=str(Path(__file__).resolve().parents[1]), path=str(log))],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    written = log.read_text()
    assert "frame 3/3" in written and "frame 1/3" not in written, written
    assert written.splitlines()[0][:4].isdigit(), "file lines must be stamped"
    # The console keeps its live animation: a user watching the server must not
    # have their terminal degraded to buy a readable log file.
    assert "frame 1/3" in out.stdout, "the console tee must stay raw"
