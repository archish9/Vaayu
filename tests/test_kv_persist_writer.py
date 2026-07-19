"""DiskWriter: append grows nrec, nrec-last crash safety, truncate mirror."""
import builtins
import os
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


def _cache(n_layers=2, heads=4, dim=16, seq=6):
    c = FP8DynamicCache()
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        c.update(k, v, L)
    return c


def _read_nrec(path):
    with open(path, "rb") as f:
        h = kp.read_header(f)
    return h.nrec


def test_append_writes_header_and_grows_nrec(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    toks = [1, 2, 3, 4, 5, 6]
    w.enqueue_append("s1", c, cfg, toks[:4], 4)
    w.drain()
    p = w.path_for("s1")
    assert os.path.exists(p) and _read_nrec(p) == 4
    # second turn appends only the new tail
    w.enqueue_append("s1", c, cfg, toks, 6)
    w.drain()
    assert _read_nrec(p) == 6
    w.close()


def test_nrec_last_survives_torn_append(tmp_path):
    """Corrupt the file by chopping the last record; header nrec still points at
    the committed prefix, so a reader that trusts nrec loads coherent data."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain(); w.close()
    p = w.path_for("s1")
    # simulate a torn write: append garbage bytes past the committed region
    with open(p, "ab") as f:
        f.write(b"\x00" * 32)
    assert _read_nrec(p) == 6  # nrec unchanged -> reader ignores the torn tail


def test_truncate_mirror(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()
    w.enqueue_truncate("s1", 3)
    w.drain()
    assert _read_nrec(w.path_for("s1")) == 3
    w.close()


# --- Fix-round regression tests (task-3 reviewer findings #1-#7) ---


def test_failed_append_rolls_back_nrec_so_next_append_retries(tmp_path, monkeypatch):
    """A worker-side append failure must roll back the in-memory nrec so the
    NEXT append for the same session retries the failed range from scratch,
    instead of leaving on-disk nrec pointing past a gap of never-written
    bytes that a reader would trust as real data (finding #1)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=12)
    toks = list(range(101, 113))   # distinctive, non-zero token ids
    target = w.path_for("s1")

    real_open = builtins.open
    fail_box = {"armed": True}

    def flaky_open(file, *a, **kw):
        if fail_box["armed"] and file == target:
            fail_box["armed"] = False
            raise OSError("simulated disk failure")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", flaky_open)

    # First append (positions 0..6): the worker's open() fails entirely, so
    # nothing lands on disk.
    w.enqueue_append("s1", c, cfg, toks[:6], 6)
    w.drain()
    assert not os.path.exists(target)

    # Second append (positions 0..10) must retry from 0 (not from 6) --
    # proves the in-memory nrec rolled back after the failure above.
    w.enqueue_append("s1", c, cfg, toks[:10], 10)
    w.drain()
    w.close()

    with open(target, "rb") as f:
        h = kp.read_header(f)
        got_toks, _ = kp.read_records(f, h)
    assert h.nrec == 10
    assert got_toks == toks[:10]   # every position holds real data, no gap


def test_truncate_and_remove_submit_while_holding_lock(tmp_path, monkeypatch):
    """enqueue_truncate/remove must submit to the worker pool while still
    holding self._lock (matching enqueue_append's commit section), so the
    single-worker FIFO queue always processes tasks in the same order their
    callers committed the corresponding _state change -- not the order calls
    happen to reach pool.submit() after releasing the lock (finding #2)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()

    held_at_submit = []
    real_submit = w._pool.submit

    def spy_submit(fn, *a, **kw):
        held_at_submit.append(w._lock.locked())
        return real_submit(fn, *a, **kw)

    monkeypatch.setattr(w._pool, "submit", spy_submit)
    w.enqueue_truncate("s1", 3)
    w.remove("s2")
    monkeypatch.setattr(w._pool, "submit", real_submit)  # stop spying before drain()

    w.drain()
    w.close()
    assert held_at_submit == [True, True]


def test_enforce_budget_snapshot_taken_under_lock(tmp_path, monkeypatch):
    """_enforce_budget's `active = {self.path_for(s) for s in self._state}`
    must run under self._lock -- it executes on the worker thread after every
    successful append, while enqueue_append/enqueue_truncate/remove mutate
    self._state (e.g. for other sessions -- ordinary parallel fan-out) from
    other threads under that same lock. Without it, a concurrent size change
    during the dict comprehension raises `RuntimeError: dictionary changed
    size during iteration` on the worker thread (finding #3)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=2)
    w.enqueue_append("s1", c, cfg, [1, 2], 2)
    w.drain()

    saw_locked = []
    real_path_for = w.path_for

    def spying_path_for(session_id):
        saw_locked.append(w._lock.locked())
        return real_path_for(session_id)

    monkeypatch.setattr(w, "path_for", spying_path_for)
    w._enforce_budget()
    w.close()

    assert saw_locked and all(saw_locked)


def test_truncate_fsyncs_when_sync_enabled(tmp_path, monkeypatch):
    """_do_truncate must fsync (both the nrec write and the file truncate)
    when sync=True, matching the durability promise _do_append already makes
    (finding #4)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12, sync=True)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()

    calls = []
    real_fsync = os.fsync

    def counting_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(kp.os, "fsync", counting_fsync)
    w.enqueue_truncate("s1", 3)
    w.drain()
    w.close()

    assert len(calls) == 2   # nrec write + f.truncate(), each fsync'd


def test_packing_runs_outside_the_lock(tmp_path, monkeypatch):
    """pack_record does real CPU/tensor-copy work (.cpu()/.contiguous()) and
    must run OUTSIDE self._lock, so concurrent DIFFERENT sessions' packing
    isn't serialized behind one global lock -- the parallel-fan-out case
    CLAUDE.md calls out as this box's reason for existing (finding #5)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=4)

    saw_locked = []
    real_pack_record = kp.pack_record

    def spying_pack_record(*a, **kw):
        saw_locked.append(w._lock.locked())
        return real_pack_record(*a, **kw)

    monkeypatch.setattr(kp, "pack_record", spying_pack_record)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4], 4)
    w.drain()
    w.close()

    assert saw_locked and not any(saw_locked)


def test_non_oserror_worker_exception_is_caught_and_logged(tmp_path, monkeypatch, capsys):
    """Any exception raised inside a worker method -- not just OSError --
    must be caught and logged, never vanish silently via an unobserved
    Future (finding #6). record_bytes is patched to simulate a non-OSError
    bug reached only from the worker side (_do_append/_do_truncate)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=4)

    def boom(hdr):
        raise ValueError("simulated non-OSError worker failure")

    monkeypatch.setattr(kp, "record_bytes", boom)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4], 4)
    w.drain()   # must not raise/hang even though the worker's write blew up
    w.close()

    assert "append failed" in capsys.readouterr().out


def test_pack_failure_does_not_propagate_to_caller(tmp_path, capsys):
    """A packing exception (bad shapes, short token_ids, etc.) must be caught
    and logged inside enqueue_append, not propagate synchronously into
    whatever thread called it -- plausibly the generation loop (finding #7)."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)

    # token_ids has only 2 entries but length=6 asks pack_record to index up
    # to position 5 -> IndexError while packing, in the calling thread.
    w.enqueue_append("s1", c, cfg, [1, 2], 6)   # must not raise
    w.drain()
    w.close()

    assert "pack failed" in capsys.readouterr().out
