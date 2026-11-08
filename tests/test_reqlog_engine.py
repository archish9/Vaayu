"""Every generation writes exactly one requests.jsonl row -- including the
ones that raised, and the ones a client abandoned mid-stream.

Audit item 9: a failed request used to leave a REQ line with no DONE, and its
traceback landed in server.log unattributed.
"""
import asyncio
import contextlib
import json

import pytest
import torch

import vaayu.server.engine as engine_mod
from vaayu.server.engine import EngineHandle
from vaayu.server.schemas import ChatCompletionRequest, make_completion_id
from vaayu.server.wire import WireMeter

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
    """A load with the prefix pool off (`KV_CACHE_MODE="fp16"`, or the flag) has none.
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


# ── the one id that joins a row to the client's own log (G5) ─────────────────

def _drive_sse(e, req):
    """The SSE lines a client would get, with the meter the ROUTE mints.

    The route is where the meter is created -- it is the one frame holding
    both the `Request` and the completion id -- so a row carrying an `id`
    exists only downstream of `_sse` plus the dispatcher. `e.stream()` on its
    own is the library path, which has no meter by design (see the library
    caller test below).
    """
    import vaayu.server.app as app_mod
    from vaayu.server.scheduler import BatchScheduler

    async def main():
        sched = BatchScheduler(e, queue_depth=8, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        try:
            gen = app_mod._sse(e, req, "m", None, prepared=e.prepare(req),
                               scheduler=sched,
                               meter=WireMeter(make_completion_id()))
            return [line async for line in gen]
        finally:
            await sched.stop()

    return asyncio.run(main())


def test_an_engine_row_carries_the_completion_id_the_client_saw(tmp_path):
    """G5: the reporter quotes the id their client logged; the user finds the
    row. Rows key on `session` plus a second-resolution `ts`, which joins a
    fan-out workload's concurrent requests to each other."""
    e = EngineHandle(M(), log_dir=tmp_path)
    _drive_sse(e, _req())
    row = rows(tmp_path)[0]
    assert row["id"].startswith("chatcmpl-")
    # Keys PRESENT, values deliberately not asserted here: the `wire` block is
    # filled for exactly this (streaming) row by the task that measures the
    # socket, so asserting null would make this test fail then. The null
    # assertion belongs to the non-streaming row below.
    assert {"wire", "cancel_reason", "client"} <= row.keys()
    assert row["cancel_reason"] is None


def test_the_streamed_id_matches_the_row_id(tmp_path):
    """Minting before generation is the whole point: a second id would join
    nothing."""
    e = EngineHandle(M(), log_dir=tmp_path)
    lines = _drive_sse(e, _req())
    row = rows(tmp_path)[0]
    chunk_ids = {json.loads(line[6:])["id"] for line in lines
                 if line.startswith("data: {")}
    assert chunk_ids == {row["id"]}


def test_a_non_streaming_row_has_an_id_and_a_null_wire(tmp_path):
    """`wire` is null where there was no stream at all. A zero-filled block
    would read as "the socket sent nothing", which is a different claim from
    "this request had no socket frame" -- and the engine start the meter
    records on EVERY request must not be enough to invent one."""
    e = EngineHandle(M(), log_dir=tmp_path)
    asyncio.run(e.generate(_req(), meter=WireMeter(make_completion_id())))
    row = rows(tmp_path)[0]
    assert row["stream"] is False
    assert row["id"].startswith("chatcmpl-") and row["wire"] is None


def test_the_row_can_rebase_wire_times_onto_ttft(tmp_path):
    """Wire times start at request ARRIVAL; `ttft_s`/`wall_s` start at the
    engine's own `started`, after admission. Without the offset a reader
    comparing them on one row is wrong by the queue wait.

    The offset is read off the meter rather than off `row["wire"]` because
    nothing has put a byte on the wire yet: `wire` is still null on every row
    until the socket measurements land, and then this reads
    `row["wire"]["engine_started_s"] >= 0` directly.
    """
    e = EngineHandle(M(), log_dir=tmp_path)
    meter = WireMeter(make_completion_id())
    asyncio.run(e.generate(_req(), meter=meter))
    row = rows(tmp_path)[0]
    assert "wire" in row and row["wire"] is None
    meter.note_chunk(1)        # what `_sse` will do; `row()` is None without it
    offset = meter.row()["engine_started_s"]
    assert offset >= 0, "the engine cannot have started before the request arrived"
    assert row["wall_s"] >= 0, "nothing to rebase onto"


def test_a_library_caller_without_a_meter_still_streams_and_still_rows(tmp_path):
    """`EngineHandle` is importable without a server, a route or a socket, so
    `meter=None` must be a no-op on the generation path rather than an
    AttributeError -- and the row is still written, with a null id."""
    e = EngineHandle(M(), log_dir=tmp_path)

    async def main():
        return [x async for x in e.stream(_req())]       # no meter=

    out = asyncio.run(main())
    assert any(k == "delta" for k, _ in out)
    row = rows(tmp_path)[0]
    assert row["id"] is None and row["wire"] is None


# ── the same id, through the REAL route ─────────────────────────────────────

@contextlib.contextmanager
def _route_client(log_dir, monkeypatch):
    """A `TestClient` over the real `/v1/chat/completions`, with this file's
    streaming fake behind it.

    `_drive_sse` above enters at `_sse`, which is DOWNSTREAM of the frame that
    mints the id -- so it cannot see the route failing to hand the meter over,
    and three one-line route mutants (dropping `meter=` from the `_sse` call,
    nulling `job.meter` on the non-streaming path, minting a second id for the
    response body) all passed against it. These two tests post through the
    route itself, which is the only place that hand-off exists.

    `no_real_load`'s pattern, as `tests/test_reqlog_refusals.py` uses it:
    without it `create_app`'s lifespan calls `AutoModel.from_pretrained` on a
    checkpoint that is not here.
    """
    from fastapi.testclient import TestClient

    from vaayu.server.app import ServerConfig, create_app

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return M()

    import vaayu.auto_model as am
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ga, "audit", lambda **kw: [])

    cfg = ServerConfig(model_path="/ckpt", log_dir=str(log_dir.parent),
                       capture_stdio=False, default_stream=False)
    app = create_app(cfg)
    with TestClient(app) as c:
        # Point the rows at the directory this test owns; the lifespan built
        # its engine against its own new_run_dir(). The scheduler looks every
        # attribute up on `self.engine` per dispatch, so repointing that one
        # reference is enough.
        app.state.engine = EngineHandle(M(), log_dir=log_dir)
        app.state.scheduler.engine = app.state.engine
        yield c


def _run_dir(tmp_path):
    d = tmp_path / "run-test"
    (d / "sessions").mkdir(parents=True)
    return d


_ROUTE_BODY = {"model": "m", "session_id": "s", "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]}


def test_the_non_streaming_body_id_is_the_row_id(tmp_path, monkeypatch):
    """The id the client keeps is the id the row carries. Minting a second one
    for the response body is what made a reporter's quoted id join nothing."""
    run_dir = _run_dir(tmp_path)
    with _route_client(run_dir, monkeypatch) as c:
        r = c.post("/v1/chat/completions", json={**_ROUTE_BODY, "stream": False})
    assert r.status_code == 200
    row = rows(run_dir)[-1]
    assert row["stream"] is False
    assert r.json()["id"] == row["id"]


def test_the_streamed_chunk_ids_are_the_row_id(tmp_path, monkeypatch):
    """Same property on the streaming path, where the id reaches the client as
    every chunk's `id` and reaches the row through `job.meter`."""
    run_dir = _run_dir(tmp_path)
    with _route_client(run_dir, monkeypatch) as c:
        r = c.post("/v1/chat/completions", json={**_ROUTE_BODY, "stream": True})
    assert r.status_code == 200
    chunk_ids = {json.loads(line[6:])["id"] for line in r.text.splitlines()
                 if line.startswith("data: {")}
    row = rows(run_dir)[-1]
    assert row["stream"] is True
    assert chunk_ids == {row["id"]}
