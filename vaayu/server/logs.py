"""Logs on disk: one rotating server log, one file per session.

Two kinds of output have to be captured, and they need different mechanisms:

* Python `print()` and uvicorn's logging go through `sys.stdout` / `sys.stderr`.
* `vaayu_rust` logs with `eprintln!` (vaayu_rust/src/lib.rs) straight to fd 2.
  Rebinding `sys.stderr` never sees those, and they are the io_uring lines the
  disk-streaming path is actually debugged with.

So the capture is at the file-descriptor level: fd 1 and 2 are redirected into a
pipe and a daemon thread writes every line to the log AND back to the real
terminal. Nothing that was visible before becomes invisible.

`session_id` arrives in an HTTP body and becomes a FILENAME here, so it is
sanitized rather than trusted -- `"../../etc/cron.d/x"` must not be a path.
"""
import hashlib
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

UNNAMED = "_unnamed"
_MAX_NAME = 120
_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_write_lock = threading.Lock()

_marker_lock = threading.Lock()
_current_session: str | None = None
_overlap_warned = False

# The one session generating right now, or None when the engine is idle.
#
# This is sound ONLY because ServerEngine.slot() runs exactly one generation at
# a time under its lock (vaayu/server/engine.py) -- deliberately, because the
# engine is not thread-safe across the KV cache, the expert VRAM cache and the
# slab arena. That is what lets the whole process-wide stream, including
# print() from inside vaayu_base and eprintln! from vaayu_rust -- neither of
# which carries any session context -- be attributed to one request.
#
# >>> THIS ASSUMPTION HAS AN EXPIRY DATE. <<<
# Batched decode (Feature B, N>1) breaks it: several sessions would then share
# one forward and one output stream. When that lands, this marker must become a
# SET, and any line that cannot be attributed to exactly one session must be
# tagged `session=*` rather than guessed. The overlap path below already tags
# `*` and warns, so the day it happens is visible in the log rather than silent
# -- but the fix is a set, not a louder warning.


_pump_active = False
# True only while a capture_stdio pump thread is actually reading fd 1/2.
# drain() below is a no-op unless this is set: with no pump installed
# (CAPTURE_STDIO=False, or any library caller of EngineHandle that never
# wired up capture_stdio at all) fd 1 is not our pipe -- writing the drain
# token would print garbage straight to the real console and then block for
# the whole timeout on every marker change, on every request.
_drain_lock = threading.Lock()
_drain_event = threading.Event()
_DRAIN_TOKEN = "\x00__vaayu_drain__\x00"

# Bounded so a stuck or dead pump can never turn a marker change into a
# stalled request -- this runs on the generation path, at every set/clear of
# the session marker. 2s is generous next to how fast an uncontended pipe
# actually drains (microseconds, in practice) and small next to the
# multi-second-per-request scale this engine runs at.
_DRAIN_TIMEOUT = 2.0


def drain(timeout: float = _DRAIN_TIMEOUT) -> bool:
    """Block until every byte written to fd 1 so far has been read by the pump.

    `_emit` reads the session marker when a line is DRAINED from the pipe, not
    when it was WRITTEN -- the writers (print() inside vaayu_base, eprintln!
    in vaayu_rust) carry no session context at all, which is the entire
    reason the marker exists. Without this barrier the marker can move to the
    next session while the previous session's last lines are still sitting
    unread in the pipe, and they get attributed to whoever holds the marker
    when the pump is finally scheduled -- wrong, not just late.

    The pipe is FIFO: everything written before this sentinel is read before
    it, so writing one unique token and waiting for `_emit` to see it is a
    correct drain barrier with no new synchronisation between writer and
    reader required.

    A no-op when no pump is installed (see `_pump_active` above). Never
    raises; returns whether the drain was observed within `timeout`.
    """
    if not _pump_active:
        return True
    with _drain_lock:
        _drain_event.clear()
        try:
            print(_DRAIN_TOKEN, flush=True)
        except OSError:
            return False
        return _drain_event.wait(timeout)


def set_current_session(session_id) -> None:
    """Mark the session that owns the output stream from here until cleared."""
    global _current_session, _overlap_warned
    # BEFORE the lock, always: _emit() calls current_session(), which takes
    # _marker_lock too, so draining while holding it would deadlock the pump
    # against this thread and hang the server on every request.
    drain()
    with _marker_lock:
        marker = sanitize_session(session_id) if session_id else UNNAMED
        if _current_session is not None and _current_session != marker:
            # Two generations in flight. Do NOT raise: this runs on the
            # generation path, and a logging failure must never become a 500.
            # Do NOT guess either -- attribute to nobody.
            if not _overlap_warned:
                _overlap_warned = True
                try:
                    print(f"[Vaayu-Server] WARNING: session {marker!r} started "
                          f"while {_current_session!r} was still generating. "
                          f"Per-session log attribution is no longer sound and "
                          f"lines are being tagged `session=*`. This is the "
                          f"batched-decode tripwire: the marker needs to become a "
                          f"set (see vaayu/server/logs.py).")
                except OSError:
                    pass  # same rule as everywhere else on this path: never raise
            _current_session = "*"
            return
        _current_session = marker


def clear_current_session() -> None:
    global _current_session
    drain()  # same deadlock note as set_current_session -- before the lock
    with _marker_lock:
        _current_session = None


def current_session() -> str | None:
    with _marker_lock:
        return _current_session

# CSI sequences: colour, cursor moves, erase-line. tqdm and rich emit these on
# every redraw; in a file they are noise that breaks grep and line counting.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def format_line(raw: str, session: str | None = None,
                stamp: str | None = None) -> str:
    """One captured line, ready for the log FILE. Returns "" to drop it.

    Pure apart from reading the clock when `stamp` is None, so the whole of the
    treatment is testable without rewiring a file descriptor.

    Three things happen, and all three are for the file only -- the console tee
    is written raw, upstream of this (spec 9.4):

      * `\\r`-framed redraws collapse to the final frame. A 512-token prefill
        emits hundreds of them; kept raw they become one unreadable megaline.
      * ANSI CSI sequences are stripped.
      * The line is prefixed `<ISO8601 UTC> session=<id|->`, which is what makes
        `server.log` alignable against a session file, and what makes a stall
        visible as a gap.
    """
    line = raw.rstrip("\r\n")
    if "\r" in line:
        frames = [f for f in line.split("\r") if f.strip()]
        line = frames[-1] if frames else ""
    line = _ANSI.sub("", line)
    if not line.strip():
        return ""
    if stamp is None:
        stamp = (datetime.now(timezone.utc)
                 .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    return f"{stamp} session={session or '-'} {line}\n"


def sanitize_session(session_id) -> str:
    """A session id turned into one safe filename stem.

    Unsafe characters are replaced rather than dropped, and anything that had to
    be changed keeps a short hash of the ORIGINAL: stripping alone would map
    `../a` and `..\\a` onto one file and silently interleave two sessions.
    """
    if not session_id:
        return UNNAMED
    cleaned = _SAFE.sub("_", session_id)
    # Collapse dot runs as well: "_.._.." carries no separator so it cannot
    # traverse, but a name containing ".." invites exactly one wrong assumption
    # from the next reader of this directory.
    cleaned = re.sub(r"\.{2,}", "_", cleaned).strip(".") or "_"
    if cleaned != session_id or len(cleaned) > _MAX_NAME:
        digest = hashlib.sha1(session_id.encode("utf-8", "replace")).hexdigest()[:10]
        cleaned = f"{cleaned[:_MAX_NAME - 11]}-{digest}"
    return cleaned


def session_log_path(log_dir, session_id) -> Path:
    return Path(log_dir) / "sessions" / f"{sanitize_session(session_id)}.log"


def session_event(log_dir, session_id, event: str, **fields) -> None:
    """Append one `TIME EVENT k=v ...` line for this session.

    Never raises: a read-only mount or a full disk must not become a 500 on the
    generation path. A lost log line is cheaper than a lost response.
    """
    path = session_log_path(log_dir, session_id)
    stamp = datetime.now().isoformat(timespec="seconds")
    body = " ".join(f"{k}={v}" for k, v in fields.items())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, open(path, "a") as f:
            f.write(f"{stamp} {event:<6} {body}\n")
    except OSError:
        pass


_session_sink: dict = {"id": None, "fh": None}


def _session_write(run_dir: Path, session: str, data: bytes) -> None:
    """Append one already-formatted line to `sessions/<session>.log`.

    One cached handle rather than open-append-close per line: a decode emits
    thousands of lines and the reopen cost lands on the generation path.
    Never raises, for the same reason session_event does not.
    """
    try:
        if _session_sink["id"] != session:
            if _session_sink["fh"] is not None:
                _session_sink["fh"].close()
            path = run_dir / "sessions" / f"{session}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            _session_sink["id"] = session
            _session_sink["fh"] = open(path, "ab", buffering=0)
        _session_sink["fh"].write(data)
    except OSError:
        _session_sink["id"], _session_sink["fh"] = None, None


# Startup warnings print ONCE, so a tailed log -- which is what users paste --
# shows none of them, including the APST and foreign-GPU-process warnings that
# are the two live hardware taxes on this box. env.json captures them
# structurally instead.
#
# Harvested here rather than plumbed from the print sites for two reasons: the
# sites are scattered through vaayu_base.py and moe_patch.py and editing a dozen
# of them to call into the server package is exactly the coupling CLAUDE.md 3
# forbids -- and it would STILL miss every vaayu_rust warning, which reaches
# fd 2 directly where no Python hook can see it. The pump sees all of it.
_WARNING_LIMIT = 200
_warnings: list = []


def note_if_warning(formatted: str) -> None:
    """Record `formatted` if it looks like a warning. Cheap, called per line."""
    if "WARNING" not in formatted:
        return
    # Drop the `<stamp> session=<id> ` prefix format_line added: env.json wants
    # the message, and the timestamp is already in server.log.
    parts = formatted.rstrip("\n").split(" ", 2)
    message = parts[2] if len(parts) == 3 else formatted.rstrip("\n")
    if message in _warnings:
        return
    _warnings.append(message)
    if len(_warnings) > _WARNING_LIMIT:
        del _warnings[0]


def collected_warnings() -> list:
    """Every distinct WARNING line seen so far, oldest first."""
    return list(_warnings)


def capture_stdio(log_path, rotate_mb: int = 50):
    """Tee fd 1 and 2 into `log_path`. Returns a restore callable.

    Rotation is a size check on the pump thread rather than logging's
    RotatingFileHandler: the bytes arriving here were never logging records.
    """
    log_path = Path(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sink = open(log_path, "ab", buffering=0)
    except OSError:
        return lambda: None          # logging must never stop the server booting

    saved_out, saved_err = os.dup(1), os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    limit = rotate_mb * 1024 ** 2
    state = {"sink": sink, "size": log_path.stat().st_size if log_path.exists() else 0}
    run_dir = log_path.parent

    def _emit(raw: str) -> None:
        """One assembled line -> the log file, and the session file if any."""
        if raw == _DRAIN_TOKEN:
            # drain()'s barrier: the pipe is FIFO, so seeing this means every
            # byte written before drain() was called has now been read. Not a
            # real line -- never formatted, never logged.
            _drain_event.set()
            return
        session = current_session()
        text = format_line(raw, session=session)
        if not text:
            return
        note_if_warning(text)
        data = text.encode("utf-8", "replace")
        try:
            if state["size"] + len(data) > limit:
                state["sink"].close()
                os.replace(log_path, log_path.with_suffix(".log.1"))
                state["sink"] = open(log_path, "ab", buffering=0)
                state["size"] = 0
            state["sink"].write(data)
            state["size"] += len(data)
        except OSError:
            pass
        if session and session != "*":
            _session_write(run_dir, session, data)

    def pump():
        buf = b""
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                try:
                    os.write(saved_out, chunk)  # RAW: the console keeps its
                except OSError:                 # animation and its colour
                    pass                        # console already gone
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    _emit(raw.decode("utf-8", "replace"))
        if buf:                                 # a last line with no newline
            _emit(buf.decode("utf-8", "replace"))
        state["sink"].close()

    thread = threading.Thread(target=pump, daemon=True, name="vaayu-log-pump")
    thread.start()
    global _pump_active
    _pump_active = True

    def restore():
        global _pump_active
        _pump_active = False
        # Order matters. Restoring fd 1/2 closes the pipe's write ends, which is
        # what lets the pump see EOF -- so JOIN before closing the saved console
        # fds. Closing them first raced the pump onto a dead descriptor and the
        # last lines were lost with an Errno 9 traceback.
        #
        # Flush FIRST, for the same reason in the other direction: `print()` to a
        # non-tty is block-buffered, so a message can still be in Python's buffer
        # here, and after the dup2 it would flush to the console rather than into
        # the log. That is exactly the case a failed startup hits.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass                            # already closed or detached
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        thread.join(timeout=5)
        os.close(saved_out)
        os.close(saved_err)

    return restore


def new_run_dir(root) -> Path:
    """One directory per server process: `<root>/run-<UTC>/`, with `sessions/`.

    A run is never split across directories and two runs never share one, which
    is what makes "which byte range of server.log was that run?" -- the question
    that cost the 2026-09-03 diagnosis a `grep` and an `awk` -- stop existing.
    """
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%SZ")
    run, n = root / stamp, 0
    while run.exists():                 # two starts inside one second
        n += 1
        run = root / f"{stamp}-{n}"
    run.mkdir(parents=True)
    (run / "sessions").mkdir()
    _point_latest(root, run)
    return run


def _point_latest(root: Path, run: Path) -> None:
    """Repoint `<root>/latest` at `run`, atomically, with a RELATIVE target.

    Relative because this directory gets zipped and mailed: an absolute target
    would point at the reporter's disk. Atomic because `tail -f logs/latest/...`
    must never observe the link missing.

    A failure here is not fatal -- the run directory is the artifact; the
    symlink is a convenience for the person watching the console.
    """
    link, tmp = root / "latest", root / ".latest.tmp"
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(run.name, tmp)
        os.replace(tmp, link)
    except OSError as e:
        print(f"[Vaayu-Server] WARNING: could not point {link} at {run.name} "
              f"({type(e).__name__}: {e}). The run directory itself is fine; "
              f"use it directly instead of logs/latest.")
