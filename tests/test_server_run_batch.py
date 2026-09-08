"""run_batch: one batched_generate, fanned out to N per-job queues.

The generator contract is the #1 correctness risk on this row: its own
docstring says abandoning it leaves the model in batched-decode mode until GC,
and a client disconnect is exactly that abandonment. run_batch owns the
generator so no client can abandon it, and closes it explicitly on every path.
"""
import asyncio

import pytest

from vaayu.server.engine import CancelToken, EngineHandle
from vaayu.server.schemas import ChatCompletionRequest

PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class CharTokenizer:
    chat_template = PLAIN_TEMPLATE
    pad_token_id = 0

    def apply_chat_template(self, messages, **kw):
        return [1] * 10

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _Config:
    def __init__(self, eos_token_id=2):
        self.eos_token_id = eos_token_id


class BatchModel:
    """Yields a fixed script of (stream_id, token_id), honouring should_stop."""

    def __init__(self, script, max_seq_len=8192):
        self.script = script
        self.max_seq_len = max_seq_len
        self.tokenizer = CharTokenizer()
        self.speculator = None
        self.running_device = "cpu"
        self._vram_plan = object()
        self.closed = False
        self.seen_requests = None
        self.batch_flag_on_exit = None
        self._batch_decode_active = False
        # _finish() reads eos_token_id off here; the brief's BatchModel omits
        # it (an AttributeError on every job, unrelated to what this test
        # exercises), so it is added here matching test_server_engine.py's
        # `_Config` pattern -- default 2, which none of these tests' char
        # tokens collide with.
        self.config = _Config()

    def batched_generate(self, requests, reclaim=False, should_stop=None):
        self.seen_requests = requests
        self._batch_decode_active = True
        try:
            for i, tok in self.script:
                if should_stop is not None and should_stop(i):
                    continue
                yield (i, tok)
        finally:
            self._batch_decode_active = False
            self.closed = True


class _Job:
    def __init__(self, session_key, prompt_ids, cap):
        self.session_key = session_key
        self.prompt_ids = prompt_ids
        self.cap = cap
        self.cancel = CancelToken()
        self.out = asyncio.Queue()
        self.req = ChatCompletionRequest(
            model="vaayu", session_id=session_key,
            messages=[{"role": "user", "content": "hi"}])


async def _drain(job):
    """Everything a client would see, up to the end sentinel."""
    seen = []
    while True:
        item = await job.out.get()
        if item is None:
            return seen
        seen.append(item)


def _deltas(seen):
    return "".join(p for kind, p in seen if kind == "delta")


def _final(seen):
    return next(p for kind, p in seen if kind == "final")


def test_each_job_gets_only_its_own_tokens():
    script = [(0, ord("a")), (1, ord("x")), (0, ord("b")), (1, ord("y"))]
    e = EngineHandle(BatchModel(script))
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert _deltas(a) == "ab"
    assert _deltas(b) == "xy"


def test_each_job_gets_a_final_result_with_its_own_counts():
    script = [(0, ord("a")), (0, ord("b")), (1, ord("z"))]
    e = EngineHandle(BatchModel(script))
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert _final(a).completion_tokens == 2
    assert _final(a).prompt_tokens == 10
    assert _final(b).completion_tokens == 1


def test_a_cancelled_job_is_retired_and_the_others_finish():
    script = [(0, ord("a")), (1, ord("x")), (0, ord("b")), (1, ord("y"))]
    model = BatchModel(script)
    e = EngineHandle(model)
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]
    jobs[0].cancel.cancel()

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert _deltas(a) == "", "a disconnected client's tokens are not produced"
    assert _deltas(b) == "xy", "the rest of the batch is unaffected"


def test_the_generator_is_always_closed(monkeypatch):
    """No client holds it, and run_batch closes it explicitly rather than
    leaving _batch_decode_active set until GC.

    A naive version of this test -- a short script, drained to exhaustion --
    cannot tell explicit closure from CPython's own refcounting: once _pump's
    frame returns, its only reference to the generator drops to zero and
    Python collects it immediately either way, so `model.closed` reads True
    whether or not `run_batch` ever calls `.close()`. Two things make this
    version actually discriminate:

    1. The model stashes its OWN reference to the generator (`_last_gen`),
       kept alive by `model` for the rest of this test -- so nothing but an
       explicit `.close()` call can run its `finally` early.
    2. The bridge queue is made to fail relaying the 3rd token, so `_pump`'s
       consuming loop exits with the generator still suspended mid-yield
       (real abandonment), not naturally exhausted -- the script never runs
       out on its own (1000 entries, cap far higher).

    Verified empirically per review: with `gen.close()` temporarily deleted
    from `run_batch`, this test fails (`closed` reads False); restored, it
    passes. See task-6-report.md for the transcript.
    """
    import vaayu.server.engine as engine_mod

    class SelfKeepingModel(BatchModel):
        def __init__(self):
            super().__init__(script=None)

        def batched_generate(self, requests, reclaim=False, should_stop=None):
            def _inner():
                self._batch_decode_active = True
                try:
                    for k in range(1000):
                        i = 0
                        tok = ord("a") + (k % 5)
                        if should_stop is not None and should_stop(i):
                            continue
                        yield (i, tok)
                finally:
                    self._batch_decode_active = False
                    self.closed = True
            gen = _inner()
            self._last_gen = gen          # outlives _pump's stack frame
            return gen

    real_queue_cls = engine_mod._queue.Queue

    class _FlakyQueue(real_queue_cls):
        """Fails relaying the 3rd token, simulating the bridge breaking while
        `_pump` is mid-iteration -- forcing its for loop to exit with the
        generator still suspended, not exhausted."""
        _tok_puts = 0

        def put(self, item, *a, **kw):
            if isinstance(item, tuple) and item[0] == "tok":
                type(self)._tok_puts += 1
                if type(self)._tok_puts == 3:
                    raise RuntimeError("bridge broke")
            return super().put(item, *a, **kw)

    model = SelfKeepingModel()
    e = EngineHandle(model)
    jobs = [_Job("s0", [1] * 10, 100_000)]

    _FlakyQueue._tok_puts = 0
    # pytest's monkeypatch, not a manual assign/restore: same reach (this IS
    # the stdlib queue module's Queue, patched process-wide for the duration),
    # but teardown is owned by pytest, so it survives a KeyboardInterrupt or
    # an internal error between the assignment and a manual try/finally.
    monkeypatch.setattr(engine_mod._queue, "Queue", _FlakyQueue)

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    asyncio.run(go())

    assert model.closed is True
    assert model._batch_decode_active is False


def test_a_failed_thread_start_leaves_the_engine_idle(monkeypatch):
    """`_live_cancels`/`_live_thread` used to be published BEFORE
    `thread.start()`, with cleanup only in the `finally` `start()` raising
    never reaches -- so a failed start left `live_cancels` populated forever
    (the engine reporting a generation in flight when none exists) and
    `_live_thread` pointing at a Thread that was never started, which
    `quiesce()` -- the teardown / client-disconnect path -- would then try to
    `.join()`, raising "cannot join thread before it is started". Publishing
    only happens after `start()` succeeds now; this proves both halves: the
    engine reports idle, and `quiesce()` does not raise."""
    import vaayu.server.engine as engine_mod

    class _NeverStarts(engine_mod.threading.Thread):
        def start(self):
            raise RuntimeError("no threads available")

    monkeypatch.setattr(engine_mod.threading, "Thread", _NeverStarts)

    e = EngineHandle(BatchModel([(0, ord("a"))]))
    jobs = [_Job("s0", [1] * 10, 8)]

    async def go():
        with pytest.raises(RuntimeError, match="no threads available"):
            await e.run_batch(jobs)

    asyncio.run(go())

    assert e.live_cancels == [], "must report idle, not a phantom generation"
    assert e._live_thread is None
    e.quiesce()  # must not raise "cannot join thread before it is started"


def test_an_engine_error_reaches_every_job():
    class Boom(BatchModel):
        def batched_generate(self, requests, reclaim=False, should_stop=None):
            yield (0, ord("a"))
            raise RuntimeError("CUDA OOM")

    e = EngineHandle(Boom([]))
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert any(kind == "error" for kind, _p in a)
    assert any(kind == "error" for kind, _p in b), "no job may hang on a failure"


def test_a_finalisation_failure_for_one_job_does_not_starve_the_rest():
    """A raise while finalising job i (parse_tool_calls, _finish, _log, or
    building GenerationResult) used to propagate straight out of the fan-out
    for loop, so jobs i+1..N never got a ("final")/("error") item OR their
    None sentinel -- their clients would block on out.get() forever while
    run_batch re-raised through the caller's lock. job 0 is rigged to fail
    finalising (via cap=3, a marker _finish below raises on); job 1 must
    still get its normal final result and sentinel."""
    class FlakyFinishEngine(EngineHandle):
        def _finish(self, new_ids, cap, tool_calls, cancel=None):
            if cap == 3:
                raise RuntimeError("boom finalising job 0")
            return super()._finish(new_ids, cap, tool_calls, cancel=cancel)

    script = [(0, ord("a")), (1, ord("x")), (1, ord("y"))]
    e = FlakyFinishEngine(BatchModel(script))
    jobs = [_Job("s0", [1] * 10, 3), _Job("s1", [1] * 10, 8)]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert any(kind == "error" for kind, _p in a), "job 0's own finalisation failure"
    assert _final(b).completion_tokens == 2, "job 1 must still finalise normally"


def test_a_keyboardinterrupt_finalising_one_job_still_serves_the_rest_and_propagates():
    """The BaseException sibling of the test above: a raise while finalising
    job i that is NOT an Exception (CancelledError, KeyboardInterrupt) must
    not be swallowed as an ordinary client error -- that would silently drop
    a real cancellation -- but it also must not starve jobs i+1..N. Every
    job must still get its item and sentinel, the engine must be left idle
    afterward, and the KeyboardInterrupt must propagate out of run_batch
    once every job is served."""
    class FlakyFinishEngine(EngineHandle):
        def _finish(self, new_ids, cap, tool_calls, cancel=None):
            if cap == 3:
                raise KeyboardInterrupt()
            return super()._finish(new_ids, cap, tool_calls, cancel=cancel)

    script = [(0, ord("a")), (1, ord("x")), (1, ord("y"))]
    e = FlakyFinishEngine(BatchModel(script))
    jobs = [_Job("s0", [1] * 10, 3), _Job("s1", [1] * 10, 8)]

    async def go():
        with pytest.raises(KeyboardInterrupt):
            await e.run_batch(jobs)
        # Reached only if run_batch's finally already ran to completion --
        # a hang here (no sentinel) would fail the test via timeout instead.
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    assert any(kind == "error" for kind, _p in a), "job 0's own finalisation failure"
    assert _final(b).completion_tokens == 2, "job 1 must still finalise normally"
    assert e.live_cancels == [], "must report idle after propagating"
    assert e._live_thread is None


def test_stop_strings_are_honoured_per_stream():
    script = [(0, ord("a")), (0, ord("E")), (0, ord("N")), (0, ord("D")),
              (0, ord("z")), (1, ord("q"))]
    e = EngineHandle(BatchModel(script))
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]
    jobs[0].req.stop = ["END"]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, _b = asyncio.run(go())
    assert _final(a).text == "a"


def test_every_cancel_token_is_published_for_shutdown():
    """cancel_all() is what SIGTERM and _teardown reach the GPU through. With
    a batch running it must reach EVERY member, not one."""
    model = BatchModel([(0, ord("a")), (1, ord("x"))])
    e = EngineHandle(model)
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]
    seen = {}

    async def go():
        task = asyncio.ensure_future(e.run_batch(jobs))
        await asyncio.sleep(0)
        seen["live"] = len(e.live_cancels)
        await task

    asyncio.run(go())
    assert seen["live"] == 2
