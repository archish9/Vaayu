"""The dispatcher: forms batches, and never sends a batch of one through
batched_generate.

Item 1 of the build handoff, and the highest-value rule on the row: Qwen's
admission grants N=1, and driving that single request through batched_generate
measured 573.88 s against 177.14 s serial -- a 3.24x regression for every Qwen
user. It is structural here, not a comment, and this file is what pins it.
"""
import asyncio

import pytest

from vaayu.server.engine import CancelToken, EngineHandle, QueueFull
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
