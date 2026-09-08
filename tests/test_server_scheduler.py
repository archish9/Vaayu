"""The dispatcher: forms batches, and never sends a batch of one through
batched_generate.

Item 1 of the build handoff, and the highest-value rule on the row: Qwen's
admission grants N=1, and driving that single request through batched_generate
measured 573.88 s against 177.14 s serial -- a 3.24x regression for every Qwen
user. It is structural here, not a comment, and this file is what pins it.
"""
import asyncio

import pytest

from vaayu.server.engine import CancelToken, EngineHandle, GenerationResult, QueueFull
from vaayu.server.scheduler import BatchScheduler, Job
from vaayu.server.schemas import ChatCompletionRequest

PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class CharTokenizer:
    chat_template = PLAIN_TEMPLATE
    pad_token_id = 0

    def apply_chat_template(self, messages, **kw):
        return [1] * 10

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class RecordingModel:
    """Records which path each run took. batched_generate raises if entered
    at N=1 -- the rule under test."""

    def __init__(self, attn="sdpa"):
        self.max_seq_len = 8192
        self.tokenizer = CharTokenizer()
        self.speculator = None
        self.running_device = "cpu"
        self._vram_plan = object()
        self._active_attn_impl = attn
        self._prefix_cache = None
        self.widths = []

    def batched_generate(self, requests, reclaim=False, should_stop=None):
        if len(requests) == 1:
            raise AssertionError("N=1 must never reach batched_generate")
        self.widths.append(len(requests))
        for i in range(len(requests)):
            yield (i, ord("a") + i)


def _job(session, prompt_ids=None, cap=4):
    req = ChatCompletionRequest(model="vaayu", session_id=session,
                                messages=[{"role": "user", "content": "hi"}])
    return Job(req=req, session_key=session,
               prompt_ids=prompt_ids or [1] * 10, cap=cap,
               cancel=CancelToken())


async def _drain(job):
    seen = []
    while True:
        item = await job.out.get()
        if item is None:
            return seen
        seen.append(item)


def test_a_single_waiting_request_runs_serially(monkeypatch):
    model = RecordingModel()
    engine = EngineHandle(model)
    served = []

    async def fake_stream(req, cancel=None, prepared=None):
        served.append(req.session_key)
        yield ("delta", "hi")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        job = _job("s0")
        await sched.submit(job)
        out = await _drain(job)
        await sched.stop()
        return out

    out = asyncio.run(go())
    assert served == ["s0"]
    assert model.widths == [], "no batch was formed for one request"
    assert ("delta", "hi") in out
    # The rule itself, not just its side effect: a single job must never even
    # be OFFERED to run_batch, so neither counter can have moved. Without
    # this, an N=1 job routed into run_batch (which raises AssertionError,
    # caught, and falls back to serial) would still satisfy `served` and
    # `widths` above -- this is what actually pins the rule.
    assert sched.formations_refused == 0 and sched.batches_run == 0


def test_max_batch_size_one_never_batches(monkeypatch):
    model = RecordingModel()
    engine = EngineHandle(model)

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=1,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        jobs = [_job("s0"), _job("s1")]
        for j in jobs:
            await sched.submit(j)
        for j in jobs:
            await _drain(j)
        await sched.stop()

    asyncio.run(go())
    assert model.widths == [], "MAX_BATCH_SIZE=1 is today's behaviour exactly"
    assert sched.formations_refused == 0 and sched.batches_run == 0


def test_a_non_sdpa_load_disables_batching(monkeypatch):
    """An operator who set flash_attention_2 gets the serialized path, never a
    live assert (build handoff item 2)."""
    model = RecordingModel(attn="flash_attention_2")
    engine = EngineHandle(model)

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=3,
                           prefill_chunk_size=512)
    assert sched.batching_enabled is False

    async def go():
        await sched.start()
        jobs = [_job("s0"), _job("s1")]
        for j in jobs:
            await sched.submit(j)
        for j in jobs:
            await _drain(j)
        await sched.stop()

    asyncio.run(go())
    assert model.widths == []


def test_two_divergent_cold_requests_form_a_batch():
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=4)
    jobs = [_job("s0", prompt_ids=[1, 2, 3] + [10] * 20),
            _job("s1", prompt_ids=[1, 2, 3] + [99] * 20)]

    async def go():
        await sched.start()
        for j in jobs:
            await sched.submit(j)
        out = [await _drain(j) for j in jobs]
        await sched.stop()
        return out

    out = asyncio.run(go())
    assert model.widths == [2]
    assert all(any(k == "delta" for k, _p in o) for o in out)


def test_the_queue_refuses_past_its_depth():
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=1, max_batch_size=1,
                           prefill_chunk_size=512)

    async def go():
        # A dispatcher must exist for a submit to be accepted at all (see
        # `submit`'s liveness guard), and neither submit() nor stop() yields
        # before its own first real suspension point, so the loop cannot
        # have picked s0 up before the depth is tested.
        await sched.start()
        await sched.submit(_job("s0"))          # occupies the depth
        with pytest.raises(QueueFull):
            await sched.submit(_job("s1"))
        await sched.stop()

    asyncio.run(go())


def test_a_gate_two_refusal_falls_back_to_serial(monkeypatch):
    """batched_generate re-checks admission at the real width and may still
    refuse. That is a fallback, never a 500."""
    model = RecordingModel()

    def refusing(requests, reclaim=False, should_stop=None):
        raise RuntimeError("insufficient VRAM for N=2")
        yield  # pragma: no cover -- makes this a generator function

    model.batched_generate = refusing
    engine = EngineHandle(model)
    served = []

    async def fake_stream(req, cancel=None, prepared=None):
        served.append(req.session_key)
        yield ("delta", "s")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=4)
    jobs = [_job("s0", prompt_ids=[1, 2, 3] + [10] * 20),
            _job("s1", prompt_ids=[1, 2, 3] + [99] * 20)]

    async def go():
        await sched.start()
        for j in jobs:
            await sched.submit(j)
        out = [await _drain(j) for j in jobs]
        await sched.stop()
        return out

    out = asyncio.run(go())
    assert served == ["s0", "s1"], "both ran, serially, after the refusal"
    assert sched.formations_refused >= 1
    assert all(any(k == "delta" for k, _p in o) for o in out)
    assert all(not any(k == "error" for k, _p in o) for o in out), (
        "the refusal must never reach a client -- only the serial retry's "
        "own output may")


def test_prepared_is_passed_through_on_the_streaming_path(monkeypatch):
    """Correction 1: _run_serial must not re-render the chat template and
    re-tokenize a prompt the route already tokenized."""
    model = RecordingModel()
    engine = EngineHandle(model)
    seen = []

    async def fake_stream(req, cancel=None, prepared=None):
        seen.append(prepared)
        yield ("delta", "x")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        job = _job("s0")
        job.prepared = ([1, 2, 3], 4)
        await sched.submit(job)
        await _drain(job)
        await sched.stop()

    asyncio.run(go())
    assert seen == [([1, 2, 3], 4)]


def test_stream_false_routes_to_generate(monkeypatch):
    """Correction 2: a non-streaming job calls engine.generate() (its own
    speculative dispatch, a stream=false requests.jsonl row), never
    engine.stream()."""
    model = RecordingModel()
    engine = EngineHandle(model)
    calls = []

    async def fake_generate(req, cancel=None):
        calls.append(req.session_key)
        return GenerationResult(text="hi", prompt_tokens=1, completion_tokens=1,
                                finish_reason="stop")

    async def fake_stream(req, cancel=None, prepared=None):
        raise AssertionError("stream=False must not call stream()")
        yield  # pragma: no cover -- makes this a generator function

    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        job = _job("s0")
        job.stream = False
        await sched.submit(job)
        out = await _drain(job)
        await sched.stop()
        return out

    out = asyncio.run(go())
    assert calls == ["s0"]
    assert any(kind == "final" for kind, _p in out)


def test_depth_counts_the_running_batch(monkeypatch):
    """Correction 3: depth must count the batch that is already running, not
    only the waiting queue -- the same promise today's slot() makes."""
    model = RecordingModel()
    engine = EngineHandle(model)
    resume = asyncio.Event()

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")
        await resume.wait()

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        job = _job("s0")
        await sched.submit(job)
        # The first delta can only exist once _run has started (and set
        # `_running`), so this is a synchronisation point, not a guess.
        first = await job.out.get()
        depth_while_running = sched.depth
        resume.set()
        while True:
            item = await job.out.get()
            if item is None:
                break
        await sched.stop()
        return first, depth_while_running

    first, depth_while_running = asyncio.run(go())
    assert first == ("delta", "x")
    assert depth_while_running == 1, "the running job must still count"


def test_an_empty_first_delta_does_not_stall_other_members():
    """Critical 1's regression test. A member whose first token decodes to
    an empty delta (EOS decoded away, a half multi-byte character, a stop
    match, an immediately-retiring cap) must not hold every other member's
    real output back until the whole batch finishes -- run_batch forwards
    live, per item, straight onto the real queues, with no buffering stage
    in between.

    Job 0 gets exactly one token (0xFFFD -- CharTokenizer.decode joins it as
    the U+FFFD replacement character, which _Detok.feed treats as a split
    multi-byte character and answers with "", so nothing is ever queued for
    it) and cap=1 retires it there; job 0 gets nothing else until run_batch's
    very last step (its "final" + None, once the ENTIRE batch is done). Job
    1 keeps producing real tokens throughout. If job 1's first delta is only
    observable AFTER job 0 has already produced something, the two are being
    held together rather than forwarded independently.
    """
    class HalfCharModel(RecordingModel):
        def batched_generate(self, requests, reclaim=False, should_stop=None):
            self.widths.append(len(requests))
            yield (0, 0xFFFD)
            for ch in "zyx":
                yield (1, ord(ch))

    model = HalfCharModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=4)
    jobs = [_job("s0", prompt_ids=[1, 2, 3] + [10] * 20, cap=1),
            _job("s1", prompt_ids=[1, 2, 3] + [99] * 20, cap=8)]

    async def go():
        await sched.start()
        for j in jobs:
            await sched.submit(j)
        first1 = await asyncio.wait_for(jobs[1].out.get(), timeout=5)
        job0_untouched = jobs[0].out.empty()
        for j in jobs:
            while True:
                item = await j.out.get()
                if item is None:
                    break
        await sched.stop()
        return first1, job0_untouched

    first1, job0_untouched = asyncio.run(go())
    assert first1 == ("delta", "z")
    assert job0_untouched, (
        "job 0 (empty first delta, then nothing until the batch's very end) "
        "must still be untouched when job 1's first real delta arrives")


def test_cancelling_the_dispatch_task_ends_it(monkeypatch):
    """Regression test for review finding B (2026-09-08).

    Task 8 wires this scheduler into the app lifespan; whatever mechanism it
    uses to stop the dispatcher, a cancelled dispatch task must actually end
    rather than swallow the CancelledError and keep looping -- otherwise
    shutdown hangs forever waiting for a task that never finishes.

    `fake_stream` yields one delta then blocks "forever" on `hang`, so the
    task is genuinely suspended mid-run (inside _run_serial's own stream()
    consumption) when the test cancels it.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    hang = asyncio.Event()

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")
        await hang.wait()

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        job = _job("s0")
        await sched.submit(job)
        first = await job.out.get()          # proves _run has actually started
        sched._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(sched._task, timeout=5)
        rest = []
        while not job.out.empty():
            rest.append(job.out.get_nowait())
        return first, rest

    first, rest = asyncio.run(go())
    assert first == ("delta", "x")
    assert any(k == "error" for k, _p in rest), (
        "the batch's own client must still be notified before the "
        "dispatcher ends")
    assert None in rest, "and still get a sentinel"


def test_a_job_cancelled_while_queued_still_gets_a_sentinel():
    """Regression test for review finding C (2026-09-08).

    `_form` drops an already-cancelled job before it can become a batch
    member (it would waste a real batch slot on a stream nobody is
    listening to). It must still deliver the job's sentinel -- Task 8's
    non-streaming route awaits `job.out.get()` after its own disconnect
    watcher fires, and a job dropped with nothing written to its queue
    would leave that await hanging forever.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)
    job = _job("s0")
    job.cancel.cancel()
    sched._waiting.append(job)

    async def go():
        members = await sched._form()
        out = await asyncio.wait_for(_drain(job), timeout=5)
        return members, out

    members, out = asyncio.run(go())
    assert members == []
    assert sched._waiting == []
    assert out == []


def test_a_job_still_queued_when_the_dispatcher_stops_still_gets_a_sentinel(monkeypatch):
    """Regression test for review finding 1 (2026-09-08).

    Finding C's failure mode (a queued job dropped with nothing written to
    its queue) by a different route: a job still sitting in `_waiting` when
    the dispatch loop exits -- graceful stop() with requests still queued,
    or a re-raised CancelledError leaving mid-iteration -- must still get a
    sentinel. Every job that ever entered submit() must end with one.

    `running_job` is submitted and drained to completion FIRST, which
    parks the dispatch loop on `_arrived.wait()` (nothing left in
    `_waiting`, `_stopping` still False). `submit()` and `stop()` are then
    both synchronous up to their own first real suspension point (neither
    has an `await` that yields control before it), so `queued_job` is
    guaranteed to still be sitting in `_waiting`, never picked up, at the
    instant `_stopping` becomes True.
    """
    model = RecordingModel()
    engine = EngineHandle(model)

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=1,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        running_job = _job("s0")
        await sched.submit(running_job)
        await _drain(running_job)
        queued_job = _job("s1")
        await sched.submit(queued_job)
        await sched.stop()
        return await asyncio.wait_for(_drain(queued_job), timeout=5)

    out = asyncio.run(go())
    assert any(k == "error" for k, _p in out), (
        "a job dropped this way gets an error item, not silence")


def test_stop_after_the_dispatch_task_was_cancelled_returns_cleanly():
    """Regression test for review finding 2 (2026-09-08).

    Once the dispatch task has been cancelled directly (Task 8's
    app-lifespan path, not stop()'s own cooperative one), `stop()`'s bare
    `await self._task` re-raises that CancelledError and never clears
    `self._task` -- surfacing in the lifespan's shutdown code as an
    exception. `stop()` must return cleanly and leave `_task` cleared
    regardless of which path cancelled the dispatcher.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        sched._task.cancel()
        await asyncio.wait_for(sched.stop(), timeout=5)
        return sched._task

    task_after = asyncio.run(go())
    assert task_after is None


def test_submit_after_stop_refuses_rather_than_enqueueing():
    """Regression test for review finding 1 (2026-09-08).

    `submit()` had no liveness guard: it appended to `_waiting` and set
    `_arrived` whether or not a dispatcher was alive to drain it. A request
    racing the app lifespan's shutdown lands here -- and its
    `await job.out.get()` would then wait forever on a queue nobody owns.
    The honest answer is a refusal (QueueFull -> 503), not a hang.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        await sched.stop()
        with pytest.raises(QueueFull):
            await sched.submit(_job("s0"))
        assert sched._waiting == [], "and nothing was left behind in the queue"

    asyncio.run(go())


def test_submit_refuses_once_the_dispatch_task_has_died(monkeypatch):
    """Regression test for review finding 1 (2026-09-08), second route.

    An exception escaping `_form`/`_admit` kills the dispatch task, and
    `start()` deliberately does not resurrect it (`start()` only checks
    `self._task is None`). Every LATER request would then hang forever.
    A dead dispatcher must be a loud refusal.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    def exploding_form():
        raise RuntimeError("boom in formation")

    monkeypatch.setattr(sched, "_form", exploding_form)

    async def go():
        await sched.start()
        first = _job("s0")
        await sched.submit(first)
        # The loop's own finally still strands `first` with a sentinel...
        await asyncio.wait_for(_drain(first), timeout=5)
        assert sched._task.done(), "the dispatch task died, as this test needs"
        with pytest.raises(RuntimeError):
            await sched._task          # retrieve it; don't warn at GC
        # ...but the NEXT request must be refused, not silently queued.
        with pytest.raises(QueueFull):
            await sched.submit(_job("s1"))
        assert sched._waiting == []

    asyncio.run(go())


def test_stop_propagates_the_callers_own_cancellation(monkeypatch):
    """Regression test for review finding 2 (2026-09-08).

    `except asyncio.CancelledError: pass` cannot tell "the dispatch task was
    cancelled" (finding 2 of the previous round -- still swallowed) from
    "the coroutine calling stop() was ITSELF cancelled". Swallowing the
    latter breaks cancellation for the app lifespan: the shutdown coroutine
    would sail on as though it had never been cancelled.
    """
    model = RecordingModel()
    engine = EngineHandle(model)
    blocked = asyncio.Event()

    async def fake_stream(req, cancel=None, prepared=None):
        yield ("delta", "x")
        await blocked.wait()             # never set: the batch never finishes

    monkeypatch.setattr(engine, "stream", fake_stream)
    sched = BatchScheduler(engine, queue_depth=8, max_batch_size=2,
                           prefill_chunk_size=512)

    async def go():
        await sched.start()
        dispatch_task = sched._task
        job = _job("s0")
        await sched.submit(job)
        await job.out.get()              # proves _run is under way and stuck
        stopper = asyncio.ensure_future(sched.stop())
        await asyncio.sleep(0.05)        # let stop() reach `await task`
        assert not stopper.done()
        stopper.cancel()                 # cancel the CALLER of stop()
        with pytest.raises(asyncio.CancelledError):
            await stopper
        dispatch_task.cancel()           # test cleanup only
        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
