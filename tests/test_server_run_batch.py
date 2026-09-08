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


def test_the_generator_is_always_closed():
    """No client holds it, and run_batch closes it explicitly rather than
    leaving _batch_decode_active set until GC."""
    model = BatchModel([(0, ord("a")), (1, ord("x"))])
    e = EngineHandle(model)
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]
    asyncio.run(e.run_batch(jobs))
    assert model.closed is True
    assert model._batch_decode_active is False


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
