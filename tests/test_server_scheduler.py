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
        await sched.submit(_job("s0"))          # occupies the depth
        with pytest.raises(QueueFull):
            await sched.submit(_job("s1"))

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
