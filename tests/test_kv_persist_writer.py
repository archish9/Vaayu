"""DiskWriter: append grows nrec, nrec-last crash safety, truncate mirror."""
import builtins
import os
import threading
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


def test_failed_append_leaves_nrec_unadvanced_so_next_append_retries(tmp_path, monkeypatch):
    """A worker-side append failure must leave the in-memory nrec unadvanced so
    the NEXT append for the same session retries the failed range from scratch,
    instead of leaving on-disk nrec pointing past a gap of never-written bytes
    that a reader would trust as real data (finding #1). Under the redesign nrec
    only ever advances in the worker on a confirmed write, so a failure simply
    never advances it -- there is nothing to roll back (this test previously
    asserted a rollback; the observable behavior is identical, the mechanism is
    now failure-never-advances rather than advance-then-roll-back)."""
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


def test_truncate_and_remove_submit_while_holding_session_lock(tmp_path, monkeypatch):
    """enqueue_truncate/remove must submit to the worker pool while still holding
    the SESSION's lock (st["lock"]) -- the lock that now serializes a session's
    operations against enqueue_append -- so the single-worker FIFO queue always
    processes tasks in the same order their callers committed the corresponding
    _state change, not the order calls happen to reach pool.submit() after
    releasing a lock (finding #2, re-verified under the per-session-lock redesign).
    Note self._lock (global) is deliberately NOT held at submit time now: it only
    guards the _state dict structure, so it must not serialize disk IO."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()

    # Capture the session's own lock object -- remove() pops "s1" from _state
    # before it submits, so we can't look it up via _state at submit time.
    lock_s1 = w._state["s1"]["lock"]
    held_session = []
    held_global = []
    real_submit = w._pool.submit

    def spy_submit(fn, *a, **kw):
        held_session.append(lock_s1.locked())
        held_global.append(w._lock.locked())
        return real_submit(fn, *a, **kw)

    monkeypatch.setattr(w._pool, "submit", spy_submit)
    w.enqueue_truncate("s1", 3)   # submits _do_truncate while holding st["lock"]
    w.remove("s1")                # submits _do_remove while holding st["lock"]
    monkeypatch.setattr(w._pool, "submit", real_submit)  # stop spying before drain()

    w.drain()
    w.close()
    assert held_session == [True, True]   # both submit under the session lock
    assert held_global == [False, False]  # and NOT under the global lock


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


# --- Second fix round (redesign) regression tests: Repro A + Repro B ---


def test_pipelined_append_after_failed_write_has_no_gap(tmp_path, monkeypatch):
    """Repro A: turn 2's enqueue_append is issued BEFORE turn 1's failed
    background write is processed -- no drain() between them, which is the normal
    pipelined case (turn 2 doesn't wait for turn 1's disk IO). Because nrec now
    advances only on a CONFIRMED worker success (never optimistically in the
    caller), turn 2 re-reads the true committed start (0) and re-packs the full
    range, so the final file has real data at every position -- no zero-byte gap
    that a reader would trust as valid KV state."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=12)
    toks = list(range(101, 113))   # distinctive, non-zero token ids
    target = w.path_for("s1")

    real_open = builtins.open
    fail_box = {"armed": True}

    def flaky_open(file, *a, **kw):
        # Fail only the FIRST open of the target file (turn 1's worker write).
        if fail_box["armed"] and file == target:
            fail_box["armed"] = False
            raise OSError("simulated disk failure")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", flaky_open)

    # Two appends for the SAME session with NO drain() between: turn 1 (len 6)
    # fails on the worker; turn 2 (len 10) is enqueued before that failure lands.
    w.enqueue_append("s1", c, cfg, toks[:6], 6)
    w.enqueue_append("s1", c, cfg, toks[:10], 10)
    w.drain()
    w.close()

    with open(target, "rb") as f:
        h = kp.read_header(f)
        got_toks, _ = kp.read_records(f, h)
    assert h.nrec == 10
    assert got_toks == toks[:10]   # every position holds real data, no gap


def test_remove_racing_paused_append_leaves_no_gap(tmp_path, monkeypatch):
    """Repro B: remove() is issued WHILE an enqueue_append for the same session
    is paused mid-critical-section (mid-pack). The per-session lock must make
    remove wait until the append has submitted its write, so on the single-worker
    FIFO the delete is ordered AFTER the append -- the file ends deleted, never
    recreated with a leading zero-byte gap from a stale start captured before the
    remove."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)

    # Seed a committed prefix so the session has an st entry (and a file).
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4], 4)
    w.drain()

    pack_entered = threading.Event()
    release_pack = threading.Event()
    real_pack_record = kp.pack_record

    def blocking_pack_record(*a, **kw):
        pack_entered.set()          # signal: append is inside its critical section
        release_pack.wait(5)        # hold the session lock until the test says go
        return real_pack_record(*a, **kw)

    monkeypatch.setattr(kp, "pack_record", blocking_pack_record)

    # Thread A: a second append that blocks inside packing while holding st["lock"].
    def do_append():
        w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)

    ta = threading.Thread(target=do_append)
    ta.start()
    assert pack_entered.wait(5)     # append is now paused mid-pack, holding st["lock"]

    # Thread B: remove() must BLOCK on st["lock"] until the append releases it.
    removed = threading.Event()

    def do_remove():
        w.remove("s1")
        removed.set()

    tb = threading.Thread(target=do_remove)
    tb.start()
    assert not removed.wait(0.3)    # remove is excluded while append holds st["lock"]

    # Release the append: it finishes packing + submits, then frees st["lock"].
    release_pack.set()
    ta.join(5)
    assert removed.wait(5)          # remove now proceeds, ordered after the append

    monkeypatch.setattr(kp, "pack_record", real_pack_record)
    w.drain()
    w.close()

    # FIFO order was append-then-remove -> the file ends deleted (no gapped
    # recreate). No orphaned file with zero-byte positions can exist.
    assert not os.path.exists(w.path_for("s1"))


# --- Third fix round: truncate racing an in-flight append's WORKER execution ---


def test_truncate_racing_inflight_append_worker_no_desync_no_gap(tmp_path, monkeypatch):
    """A truncate must not leave nrec desynced from disk (or a zero-byte gap) when
    a same-session append's WORKER-side execution is still in flight.

    Round-2 bug: enqueue_truncate advanced nrec optimistically in the CALLER, while
    _do_append advanced it in the WORKER. An earlier-submitted append's worker
    callback could land AFTER the truncate's synchronous caller-side write and
    overwrite nrec back up -- desyncing it from the disk the truncate had cut. A
    later append then read a stale start and left a zero-byte gap (reviewer repro:
    final nrec=20 with a token gap at [3,15)).

    Fourth-round design: enqueue_truncate now WAITS for this session's own prior
    in-flight future before touching nrec. So it can no longer be interleaved with
    the append worker at all -- it blocks on the append's future until that append
    has fully committed (disk + nrec), then reads the true nrec and truncates. This
    test proves that blocking: the truncate, run on a side thread, is held until the
    paused append worker is released, and the final state is nrec==disk==3 with no
    gap. (It differs from test_remove_racing_paused_append_leaves_no_gap, which
    pauses PACKING in the caller; here we pause the WORKER's _do_append.)"""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=20)
    toks = list(range(101, 121))   # 20 distinctive, non-zero token ids
    target = w.path_for("s1")

    # Seed a committed prefix of 5 so a later append can grow it while an even
    # smaller truncate (to 3) contends with that append's worker.
    w.enqueue_append("s1", c, cfg, toks[:5], 5)
    w.drain()   # disk nrec=5, mem nrec=5

    entered = threading.Event()
    release = threading.Event()
    real_do_append = kp.DiskWriter._do_append

    def paused_do_append(self, *a, **kw):
        entered.set()          # worker is now inside _do_append...
        release.wait(5)        # ...held here, BEFORE any disk write / nrec update
        return real_do_append(self, *a, **kw)

    monkeypatch.setattr(kp.DiskWriter, "_do_append", paused_do_append)

    # Growing append 5 -> 15: its worker enters _do_append and pauses before writing.
    w.enqueue_append("s1", c, cfg, toks[:15], 15)
    assert entered.wait(5)

    # Issue a truncate to 3 while that append's WORKER is paused. Under the new
    # design enqueue_truncate must BLOCK on the append's future, so run it on a side
    # thread and assert it does not complete while the append worker is held.
    trunc_done = threading.Event()

    def do_trunc():
        w.enqueue_truncate("s1", 3)
        trunc_done.set()

    tt = threading.Thread(target=do_trunc)
    tt.start()
    assert not trunc_done.wait(0.3)   # truncate is blocked on the append's future

    # Release the append: the worker runs _do_append (disk->15, then nrec), which
    # resolves the future the truncate is waiting on. The truncate then unblocks,
    # reads the true nrec (15), and submits _do_truncate (disk->3, then nrec). Both
    # nrec writes are worker-side in FIFO order, so nrec settles at 3 == disk.
    release.set()
    assert trunc_done.wait(5)
    tt.join(5)
    w.drain()
    monkeypatch.setattr(kp.DiskWriter, "_do_append", real_do_append)  # unpause

    # Directly assert NO desync, BEFORE any further append can mask it
    # (round-2 bug at this point would be: mem=15 but disk=3).
    with open(target, "rb") as f:
        h_mid = kp.read_header(f)
    assert w._state["s1"]["nrec"] == h_mid.nrec == 3

    # A subsequent append reads st["nrec"] as its start. If the race had left mem=15
    # it would seek to position 15 over a 3-record file, zero-filling positions
    # [3,15) -- exactly the reviewer's gap. With the fix start=3, so the append is
    # contiguous and every position holds real data.
    w.enqueue_append("s1", c, cfg, toks[:20], 20)
    w.drain()
    w.close()

    with open(target, "rb") as f:
        h = kp.read_header(f)
        got_toks, _ = kp.read_records(f, h)
    # (a) in-memory nrec matches the on-disk header (no desync) ...
    assert w._state["s1"]["nrec"] == h.nrec == 20
    # ... (b) and every position up to nrec holds real, non-zero token data (no gap).
    assert got_toks == toks[:20]
    assert 0 not in got_toks       # belt-and-suspenders: no zero-byte gap anywhere


# --- Fourth fix round: pipelined truncate-then-append (C1, the normal case) ---


def test_pipelined_truncate_then_append_no_gap(tmp_path, monkeypatch):
    """C1: a truncate immediately followed by a same-session append with NO drain
    between them -- the feature's NORMAL crop-to-(L-1)-then-extend continuation
    pattern. The truncate's WORKER execution is deliberately delayed to force the
    pipelined condition.

    Round-3 bug: enqueue_truncate no longer lowered nrec synchronously (it moved to
    the worker), so a pipelined append read a stale-HIGH nrec as its start, skipped
    the truncated boundary, and -- because the truncate's physical f.truncate() had
    already removed those bytes -- left a real zero-byte gap (reviewer repro:
    nrec=15, gap at [3,10)).

    Fourth-round fix: enqueue_append WAITS for this session's own prior future (the
    truncate) before reading start. So the append blocks until the truncate has
    committed (disk + nrec), then reads the POST-truncate nrec (3) and packs from the
    truncated boundary -- contiguous, no gap. We prove the append blocks by pausing
    the truncate worker and asserting the append (on a side thread) does not complete
    until the truncate is released."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=15)
    toks = list(range(101, 116))   # 15 distinctive, non-zero token ids
    target = w.path_for("s1")

    # Seed a committed prefix of 10.
    w.enqueue_append("s1", c, cfg, toks[:10], 10)
    w.drain()   # disk nrec=10, mem nrec=10

    entered = threading.Event()
    release = threading.Event()
    real_do_truncate = kp.DiskWriter._do_truncate

    def paused_do_truncate(self, *a, **kw):
        entered.set()          # worker is now inside _do_truncate...
        release.wait(5)        # ...held BEFORE it writes nrec / physically truncates
        return real_do_truncate(self, *a, **kw)

    monkeypatch.setattr(kp.DiskWriter, "_do_truncate", paused_do_truncate)

    # Crop to 3: caller submits _do_truncate, whose worker enters and pauses.
    w.enqueue_truncate("s1", 3)
    assert entered.wait(5)

    # Extend to 15 with NO drain between -- the pipelined normal case. Under the fix
    # enqueue_append must BLOCK on the truncate's future, so run it on a side thread
    # and assert it does not complete while the truncate worker is held.
    append_done = threading.Event()

    def do_append():
        w.enqueue_append("s1", c, cfg, toks[:15], 15)
        append_done.set()

    ta = threading.Thread(target=do_append)
    ta.start()
    assert not append_done.wait(0.3)   # append is blocked on the truncate's future

    # Release the truncate worker: it writes disk nrec=3 and physically shrinks the
    # file to 3 records, then updates mem nrec=3. Its future resolves, unblocking the
    # append, which reads start=3 (POST-truncate) and packs positions 3..14.
    release.set()
    assert append_done.wait(5)
    ta.join(5)
    w.drain()
    monkeypatch.setattr(kp.DiskWriter, "_do_truncate", real_do_truncate)  # unpause
    w.close()

    with open(target, "rb") as f:
        h = kp.read_header(f)
        got_toks, _ = kp.read_records(f, h)
    # nrec matches reality (no desync) and every position holds real data (no gap
    # where the truncate physically removed bytes).
    assert w._state["s1"]["nrec"] == h.nrec == 15
    assert got_toks == toks[:15]
    assert 0 not in got_toks
