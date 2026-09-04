"""Every generation writes exactly one requests.jsonl row -- including the
ones that raised, and the ones a client abandoned mid-stream.

Audit item 9: a failed request used to leave a REQ line with no DONE, and its
traceback landed in server.log unattributed.
"""
import asyncio
import json

import pytest
import torch

from vaayu.server.engine import EngineHandle
from vaayu.server.schemas import ChatCompletionRequest

from .test_server_engine import FakeModel


class M(FakeModel):
    """FakeModel plus the counters reqlog.snapshot reads."""
    def __init__(self, raise_on_generate=None):
        super().__init__()
        self.tokenizer.decode = lambda ids, **k: "x" * len(ids)
        self.config = type("C", (), {"eos_token_id": 2})()
        self.raise_on_generate = raise_on_generate
        self._expert_disk_total_s = 0.0
        self._expert_cache = type("C", (), {"hits": 0, "misses": 0})()
        self._expert_vram_cache = type("C", (), {"hits": 0, "misses": 0})()

    def generate(self, **kw):
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        new = torch.tensor([[7, 7, 7]])
        streamer = kw.get("streamer")
        if streamer is not None:
            # stream_generate's consumer loop is `for text_chunk in streamer`,
            # fed by a queue.get() with no timeout. A real model.generate()
            # feeds it per-token; this fake must too, or that loop blocks
            # forever on a queue nothing ever fills. One prompt put (consumed
            # by skip_prompt), one put for the real tokens, then end().
            streamer.put(kw["input_ids"])
            streamer.put(new)
            streamer.end()
        return torch.cat([kw["input_ids"], new], dim=1)


def _req(**over):
    base = {"model": "v", "session_id": "s", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def rows(log_dir):
    p = log_dir / "requests.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def test_a_successful_generation_writes_one_row(tmp_path):
    e = EngineHandle(M(), log_dir=tmp_path)
    asyncio.run(e.generate(_req()))
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["error"] is None
    assert r[0]["tokens"] == 3
    assert r[0]["stream"] is False
    assert r[0]["session"] == "s"
    assert r[0]["decode"]["bytes_read_gb"] is not None


def test_a_raising_generation_still_writes_a_row(tmp_path):
    e = EngineHandle(M(raise_on_generate=RuntimeError("CUDA OOM")), log_dir=tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(e.generate(_req()))
    r = rows(tmp_path)
    assert len(r) == 1, "a failed request left no row -- audit item 9"
    assert r[0]["finish"] == "error"
    assert r[0]["error"]["type"] == "RuntimeError"
    assert "CUDA OOM" in r[0]["error"]["message"]
    assert "test_reqlog_engine.py" in r[0]["error"]["traceback"]
    assert r[0]["tokens"] == 0
    assert r[0]["decode"]["gb_per_token"] is None, "0 tokens must not divide"


def test_two_requests_write_two_rows(tmp_path):
    e = EngineHandle(M(), log_dir=tmp_path)
    asyncio.run(e.generate(_req()))
    asyncio.run(e.generate(_req()))
    assert len(rows(tmp_path)) == 2


def test_a_library_caller_with_no_log_dir_writes_nothing(tmp_path):
    e = EngineHandle(M(), log_dir=None)
    asyncio.run(e.generate(_req()))
    assert not (tmp_path / "requests.jsonl").exists()


def test_an_abandoned_stream_writes_a_row(tmp_path):
    """Starlette closes an abandoned SSE generator with GeneratorExit. That is
    a normal outcome, not a failure, so the row carries a null error -- but it
    is still a row."""
    e = EngineHandle(M(), log_dir=tmp_path)

    async def main():
        gen = e.stream(_req())
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(main())
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["stream"] is True
    assert r[0]["error"] is None, "a client walking away is not an error"


def test_a_completed_stream_records_ttft(tmp_path):
    e = EngineHandle(M(), log_dir=tmp_path)

    async def main():
        async for _kind, _payload in e.stream(_req()):
            pass

    asyncio.run(main())
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["ttft_s"] is not None and r[0]["ttft_s"] >= 0
