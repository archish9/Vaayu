"""Batching must never ship silently inert (CLAUDE.md section 8).

P10 exists because P6 and P8 shipped green and delivered zero to users. The
same mistake here would be a MAX_BATCH_SIZE > 1 that never forms a batch --
which is Qwen3-235B's situation today -- discovered by a probe months later.
"""
import asyncio

from vaayu.server.engine import CancelToken, EngineHandle
from vaayu.server.scheduler import BatchScheduler, Job
from vaayu.server.schemas import ChatCompletionRequest


class Model:
    def __init__(self, attn):
        self.max_seq_len = 8192
        self.running_device = "cpu"
        self.speculator = None
        self._vram_plan = object()
        self._active_attn_impl = attn
        self._prefix_cache = None
        self.tokenizer = type("T", (), {
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
            "pad_token_id": 0,
            "apply_chat_template": lambda self, m, **k: [1] * 10})()


def _job(session):
    req = ChatCompletionRequest(model="vaayu", session_id=session,
                                messages=[{"role": "user", "content": "hi"}])
    return Job(req=req, session_key=session, prompt_ids=[1] * 10, cap=4,
               cancel=CancelToken())


def test_a_non_sdpa_load_warns_at_startup(capsys):
    BatchScheduler(EngineHandle(Model("flash_attention_2")), queue_depth=8,
                   max_batch_size=3, prefill_chunk_size=512)
    out = capsys.readouterr().out
    assert "MAX_BATCH_SIZE" in out and "sdpa" in out
    assert "WARNING" in out


def test_max_batch_size_one_says_nothing(capsys):
    BatchScheduler(EngineHandle(Model("sdpa")), queue_depth=8,
                   max_batch_size=1, prefill_chunk_size=512)
    assert capsys.readouterr().out == "", "the default must be silent"


def test_a_batching_load_says_nothing_at_startup(capsys):
    BatchScheduler(EngineHandle(Model("sdpa")), queue_depth=8,
                   max_batch_size=2, prefill_chunk_size=512)
    assert capsys.readouterr().out == ""


def test_never_admitting_prints_once(capsys):
    """Qwen3-235B's situation: sdpa is being paid for, and no batch ever
    forms. Once, not per request."""
    sched = BatchScheduler(EngineHandle(Model("sdpa")), queue_depth=8,
                           max_batch_size=2, prefill_chunk_size=512)
    capsys.readouterr()

    async def go():
        # Two identical warm-prefix jobs never beat serial, so formation
        # always returns width 1.
        for _ in range(BatchScheduler.NEVER_ADMITTED_AFTER + 3):
            sched._waiting = [_job("a"), _job("b")]
            for j in sched._waiting:
                j.peek_len = len(j.prompt_ids)
            await sched._form()

    asyncio.run(go())
    out = capsys.readouterr().out
    assert out.count("never admitted") == 1, out
