"""Every generation writes exactly one requests.jsonl row -- including the
ones that raised, and the ones a client abandoned mid-stream.

Audit item 9: a failed request used to leave a REQ line with no DONE, and its
traceback landed in server.log unattributed.
"""
import asyncio
import json

import pytest
import torch

import vaayu.server.engine as engine_mod
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


def test_a_finalize_phase_failure_in_stream_is_recorded_as_an_error(tmp_path, monkeypatch):
    """parse_tool_calls runs AFTER the consumer loop's own try/except -- in
    the finalize phase stream()'s outer except now covers. Before that outer
    except existed, a failure here reached the row-writing `finally` with
    `error` still None: a real traceback, logged as a clean request."""
    def boom(*a, **kw):
        raise RuntimeError("boom in finalize")
    monkeypatch.setattr(engine_mod, "parse_tool_calls", boom)
    e = EngineHandle(M(), log_dir=tmp_path)

    async def main():
        async for _kind, _payload in e.stream(_req()):
            pass

    with pytest.raises(RuntimeError, match="boom in finalize"):
        asyncio.run(main())
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["finish"] == "error"
    assert r[0]["error"]["type"] == "RuntimeError"
    assert "boom in finalize" in r[0]["error"]["message"]


def test_an_abandoned_stream_reports_finish_cancelled(tmp_path):
    """finish="error" beside error=null is incoherent to a maintainer reading
    the row cold. A disconnect is not an error, but it IS a cancellation."""
    e = EngineHandle(M(), log_dir=tmp_path)

    async def main():
        gen = e.stream(_req())
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(main())
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["finish"] == "cancelled"
    assert r[0]["error"] is None


def test_a_logging_failure_does_not_replace_the_real_exception(tmp_path, monkeypatch):
    """_write_request_row runs from a `finally` on the generation path. If
    building the row raises (a counter access blowing up), that must not
    replace the real exception already propagating -- a CUDA OOM becoming a
    confusing logging traceback instead."""
    real_snapshot = engine_mod.reqlog.snapshot
    calls = {"n": 0}

    def flaky_snapshot(model):
        calls["n"] += 1
        if calls["n"] > 1:      # the `before` snapshot is real; `after` blows up
            raise RuntimeError("counter access exploded")
        return real_snapshot(model)

    monkeypatch.setattr(engine_mod.reqlog, "snapshot", flaky_snapshot)
    e = EngineHandle(M(raise_on_generate=RuntimeError("CUDA OOM")), log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        asyncio.run(e.generate(_req()))


def test_a_prepare_failure_still_writes_a_row_non_streaming(tmp_path):
    """prepare() -> render_prompt() -> apply_chat_template() runs Jinja over
    the messages, and a malformed sequence raises whatever the template
    raises -- none of the three types the route names. It used to sit OUTSIDE
    generate()'s try, so this class of failure ended with no row at all, on
    the one path whose whole contract is that a request never ends without
    one."""
    m = M()
    def boom(messages, **kw):
        raise ValueError("Jinja: unknown role 'tool' with no matching call")
    m.tokenizer.apply_chat_template = boom
    e = EngineHandle(m, log_dir=tmp_path)

    with pytest.raises(ValueError, match="Jinja"):
        asyncio.run(e.generate(_req()))
    r = rows(tmp_path)
    assert len(r) == 1, "a prepare() failure left no row"
    assert r[0]["finish"] == "error"
    assert r[0]["error"]["type"] == "ValueError"
    assert r[0]["prompt_tokens"] == 0     # never got one; not a NameError


def test_stream_tags_the_three_refusal_types_refused_like_generate_does(tmp_path):
    """V1 sibling 1: same vocabulary on both of the engine's own methods.

    generate() explicitly tags ContextTooLong / TemplateLacksToolSupport /
    NoChatTemplate as finish="refused"; stream()'s fallback prepare() had no
    such branch, so the same rejection landed as "error" there. Unreachable
    from HTTP today (the route always passes `prepared=`), reachable for any
    other caller of stream() -- and the asymmetry is exactly what V1 is about.
    The prompt is genuinely oversized rather than a monkeypatched raise, so it
    is prepare()'s own ContextTooLong.
    """
    m = M()
    m.tokenizer.apply_chat_template = lambda messages, **kw: [1] * 100000
    e = EngineHandle(m, log_dir=tmp_path)

    async def main():
        async for _kind, _payload in e.stream(_req()):
            pass

    with pytest.raises(engine_mod.ContextTooLong):
        asyncio.run(main())
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["finish"] == "refused", (
        f"stream() filed a rejection as {r[0]['finish']!r}; generate() calls "
        f"the same one 'refused'")
    assert r[0]["error"]["type"] == "ContextTooLong"


def test_a_prepare_failure_still_writes_a_row_streaming(tmp_path):
    """stream()'s own fallback path, for callers that do not pass `prepared`."""
    m = M()
    def boom(messages, **kw):
        raise ValueError("Jinja: unknown role")
    m.tokenizer.apply_chat_template = boom
    e = EngineHandle(m, log_dir=tmp_path)

    async def main():
        async for _kind, _payload in e.stream(_req()):
            pass

    with pytest.raises(ValueError, match="Jinja"):
        asyncio.run(main())
    assert len(rows(tmp_path)) == 1


def test_a_cancelled_non_streaming_request_records_finish_cancelled(tmp_path):
    """`finish_reason()` maps cancelled -> "length" on purpose: the OpenAI wire
    vocabulary has no other word. requests.jsonl is NOT the wire, and
    documentation/server.md promises `cancelled` there -- otherwise a
    disconnect is indistinguishable from a truncated answer."""
    e = EngineHandle(M(), log_dir=tmp_path)
    cancel = engine_mod.CancelToken()
    cancel.cancel()

    result = asyncio.run(e.generate(_req(), cancel=cancel))
    # The WIRE value is unchanged -- that is the OpenAI vocabulary's problem.
    assert result.finish_reason == "length"
    r = rows(tmp_path)
    assert len(r) == 1
    assert r[0]["finish"] == "cancelled"


class _Pool:
    """The two counters the request row reads off PrefixCacheStore."""
    def __init__(self):
        self.hits = 0
        self.matched_tokens = 0


class PM(M):
    """M plus a prefix pool. Each generate() plays one lookup: the next entry of
    `matches` is how many prompt tokens it matched (0 = a miss)."""
    def __init__(self, matches, **kw):
        super().__init__(**kw)
        self._prefix_cache = _Pool()
        self._matches = list(matches)

    def generate(self, **kw):
        m = self._matches.pop(0)
        if m:
            self._prefix_cache.hits += 1
            self._prefix_cache.matched_tokens += m
        return super().generate(**kw)


def test_a_hit_row_says_how_many_prompt_tokens_matched(tmp_path):
    """S19 (ii): `prefix_hit: true` for a 44-token match on a 7.5k prompt hid a reuse
    collapse (row 31). The row carries the match length next to prompt_tokens."""
    e = EngineHandle(PM([44]), log_dir=tmp_path)
    asyncio.run(e.generate(_req()))
    r = rows(tmp_path)[0]
    assert r["prefix_hit"] is True
    assert r["prefix_match_tokens"] == 44
    assert r["prompt_tokens"] > 0


def test_a_miss_row_reports_zero_matched_tokens(tmp_path):
    e = EngineHandle(PM([0]), log_dir=tmp_path)
    asyncio.run(e.generate(_req()))
    r = rows(tmp_path)[0]
    assert r["prefix_hit"] is False and r["prefix_match_tokens"] == 0


def test_the_match_is_this_requests_own_not_the_pools_running_total(tmp_path):
    e = EngineHandle(PM([44, 7, 0]), log_dir=tmp_path)
    for _ in range(3):
        asyncio.run(e.generate(_req()))
    assert [r["prefix_match_tokens"] for r in rows(tmp_path)] == [44, 7, 0]


def test_a_streamed_hit_row_carries_the_match_too(tmp_path):
    e = EngineHandle(PM([31]), log_dir=tmp_path)

    async def main():
        async for _kind, _payload in e.stream(_req()):
            pass

    asyncio.run(main())
    r = rows(tmp_path)[0]
    assert r["stream"] is True and r["prefix_match_tokens"] == 31


def test_a_model_with_no_prefix_pool_reports_zero_not_null(tmp_path):
    """DeepSeek-V4 has no prefix cache; a dense load with it off has none either.
    An engine row is never null here -- null means "refused before generation"."""
    e = EngineHandle(M(), log_dir=tmp_path)
    asyncio.run(e.generate(_req()))
    r = rows(tmp_path)[0]
    assert r["prefix_hit"] is False and r["prefix_match_tokens"] == 0


def test_a_failed_request_still_records_what_it_matched(tmp_path):
    e = EngineHandle(PM([12], raise_on_generate=RuntimeError("CUDA OOM")), log_dir=tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(e.generate(_req()))
    r = rows(tmp_path)[0]
    assert r["finish"] == "error" and r["prefix_match_tokens"] == 12
