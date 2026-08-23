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
import threading
from datetime import datetime
from pathlib import Path

UNNAMED = "_unnamed"
_MAX_NAME = 120
_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_write_lock = threading.Lock()


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

    def pump():
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                try:
                    os.write(saved_out, chunk)      # keep the console alive
                except OSError:
                    pass                            # console already gone
                try:
                    if state["size"] + len(chunk) > limit:
                        state["sink"].close()
                        os.replace(log_path, log_path.with_suffix(".log.1"))
                        state["sink"] = open(log_path, "ab", buffering=0)
                        state["size"] = 0
                    state["sink"].write(chunk)
                    state["size"] += len(chunk)
                except OSError:
                    pass
        state["sink"].close()

    thread = threading.Thread(target=pump, daemon=True, name="vaayu-log-pump")
    thread.start()

    def restore():
        # Order matters. Restoring fd 1/2 closes the pipe's write ends, which is
        # what lets the pump see EOF -- so JOIN before closing the saved console
        # fds. Closing them first raced the pump onto a dead descriptor and the
        # last lines were lost with an Errno 9 traceback.
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        thread.join(timeout=5)
        os.close(saved_out)
        os.close(saved_err)

    return restore
