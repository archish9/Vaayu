"""A refused request is still a request, and still gets a row.

These never reach the engine: REQUIRE_SESSION_ID and the streaming path's
prepare() both run in the route (the latter deliberately, so ContextTooLong is
a 400 rather than an SSE internal_error -- B8).
"""
import json

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app
from vaayu.server.engine import EngineHandle

from .test_server_engine import FakeModel


def rows(log_dir):
    from pathlib import Path
    p = Path(log_dir) / "requests.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


class M(FakeModel):
    def __init__(self):
        super().__init__()
        self.tokenizer.decode = lambda ids, **k: "x" * len(ids)
        # FakeModel's inherited tokenizer stub always renders 10 ids, ignoring
        # the messages -- fine for the other suites, but these tests need
        # prepare() to actually see prompt length, or the oversized-prompt
        # cases below never raise ContextTooLong at all.
        self.tokenizer.apply_chat_template = lambda messages, **kw: [1] * sum(
            len(m.get("content") or "") for m in messages)
        self.config = type("C", (), {"eos_token_id": 2})()
        self._expert_disk_total_s = 0.0
        self._expert_cache = type("C", (), {"hits": 0, "misses": 0})()
        self._expert_vram_cache = type("C", (), {"hits": 0, "misses": 0})()

    def generate(self, **kw):
        import torch
        return torch.cat([kw["input_ids"], torch.tensor([[7, 7, 7]])], dim=1)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run-test"
    (d / "sessions").mkdir(parents=True)
    return d


@pytest.fixture
def no_real_load(monkeypatch):
    """Stop `create_app`'s lifespan from loading a real checkpoint.

    The established pattern in this suite -- `tests/test_delete_original.py`'s
    `captured` fixture does exactly this. Without it, entering `TestClient`
    runs the lifespan hook, which calls `AutoModel.from_pretrained(model_path)`
    and takes minutes here, or fails outright in CI where no checkpoint exists.
    Do not invent a second pattern for this.
    """
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return M()

    import vaayu.auto_model as am
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ga, "audit", lambda **kw: [])


def _client(run_dir, require_session_id):
    cfg = ServerConfig(model_path="/ckpt", log_dir=str(run_dir.parent),
                       capture_stdio=False,
                       require_session_id=require_session_id,
                       default_stream=False)
    app = create_app(cfg)
    with TestClient(app) as c:
        # Point the engine's rows at THIS test's run directory. The lifespan
        # built one against its own new_run_dir(); rebinding here keeps the
        # assertions reading a directory the test owns.
        app.state.engine = EngineHandle(M(), log_dir=run_dir)
        # The dispatcher (P10) was built against the lifespan's OWN engine;
        # every attribute it reads is looked up on `self.engine` fresh on
        # each dispatch, so repointing that one reference is enough to make
        # it dispatch through the engine just swapped in above.
        app.state.scheduler.engine = app.state.engine
        yield c


@pytest.fixture
def client(run_dir, no_real_load):
    yield from _client(run_dir, require_session_id=False)


@pytest.fixture
def client_requiring_session(run_dir, no_real_load):
    yield from _client(run_dir, require_session_id=True)


def _full_queue_client(run_dir):
    """Same as `_client`, but the queue is built at depth 0 so the ordinary
    admission check trips on the very first request -- an honest QueueFull
    trigger rather than a monkeypatched exception, since one is available.

    Admission is the DISPATCHER's now (P10): `BatchScheduler.submit` and the
    route's own eager depth check both key off `queue_depth`, and the
    scheduler's own copy is captured once at construction from
    `ServerConfig.queue_depth` -- not read off whatever engine happens to be
    bound later, so this has to be `0` on the CONFIG, not just on a swapped-in
    `EngineHandle`.
    """
    cfg = ServerConfig(model_path="/ckpt", log_dir=str(run_dir.parent),
                       capture_stdio=False, require_session_id=False,
                       default_stream=False, queue_depth=0)
    app = create_app(cfg)
    with TestClient(app) as c:
        app.state.engine = EngineHandle(M(), log_dir=run_dir, queue_depth=0)
        app.state.scheduler.engine = app.state.engine
        yield c


@pytest.fixture
def full_queue_client(run_dir, no_real_load):
    yield from _full_queue_client(run_dir)


def test_a_missing_session_id_writes_a_refusal_row(client_requiring_session, run_dir):
    r = client_requiring_session.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    rs = rows(run_dir)
    assert len(rs) == 1
    assert rs[0]["finish"] == "refused"
    assert rs[0]["error"]["type"] == "MissingSessionId"
    assert rs[0]["tokens"] == 0


def test_an_oversized_prompt_writes_a_refusal_row(client, run_dir):
    r = client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": False,
        "messages": [{"role": "user", "content": "x " * 100000}]})
    assert r.status_code == 400
    rs = rows(run_dir)
    assert len(rs) == 1
    assert rs[0]["error"]["type"] == "ContextTooLong"


def test_an_oversized_streaming_prompt_writes_a_refusal_row(client, run_dir):
    r = client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": True,
        "messages": [{"role": "user", "content": "x " * 100000}]})
    assert r.status_code == 400
    assert len(rows(run_dir)) == 1


def test_a_successful_request_writes_exactly_one_row(client, run_dir):
    r = client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert len(rows(run_dir)) == 1              # the engine's, not two


def test_a_logging_failure_does_not_turn_a_refusal_into_a_500(
        client_requiring_session, run_dir, monkeypatch):
    """`_refusal_row` must swallow its own failures -- see the docstring on
    `_refusal_row` in app.py. Patching `reqlog.write_row` itself (rather than
    e.g. `resolve_stream`) is what actually lands inside `_refusal_row`'s own
    try/except: `resolve_stream(req, cfg)` is evaluated as a call-site
    argument in `chat_completions`, one frame up, so it is not what this
    function's wrapper protects -- this is the failure this helper's own
    body IS responsible for.
    """
    import vaayu.server.app as app_mod

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_mod.reqlog, "write_row", boom)
    r = client_requiring_session.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    # write_row was replaced outright, so nothing landed on disk either --
    # the point is only that the client still got its clean refusal.
    assert rows(run_dir) == []


def test_a_full_queue_writes_a_refusal_row_non_streaming(full_queue_client, run_dir):
    r = full_queue_client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    rs = rows(run_dir)
    assert len(rs) == 1
    assert rs[0]["finish"] == "refused"
    assert rs[0]["error"]["type"] == "QueueFull"


def test_a_full_queue_writes_a_refusal_row_streaming(full_queue_client, run_dir):
    r = full_queue_client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    rs = rows(run_dir)
    assert len(rs) == 1
    assert rs[0]["finish"] == "refused"
    assert rs[0]["error"]["type"] == "QueueFull"


def _sse_req(**over):
    from vaayu.server.schemas import ChatCompletionRequest
    base = {"model": "m", "session_id": "s", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_the_sse_admission_race_writes_a_refusal_row(run_dir):
    """`BatchScheduler.submit` raises QueueFull BEFORE the job ever reaches
    the dispatcher, so engine.stream is never entered and its row-writing
    `finally` never runs. The route's eager depth check cannot see this race
    -- it is the reason _sse has a QueueFull handler at all -- so the row is
    that handler's to write."""
    import asyncio

    import vaayu.server.app as app_mod
    from vaayu.server.scheduler import BatchScheduler

    e = EngineHandle(M(), log_dir=run_dir, queue_depth=0)

    async def main():
        req = _sse_req()
        sched = BatchScheduler(e, queue_depth=0, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        chunks = [c async for c in app_mod._sse(e, req, "m", None,
                                                prepared=e.prepare(req),
                                                scheduler=sched)]
        await sched.stop()
        return chunks

    chunks = asyncio.run(main())
    assert any("server_busy" in c for c in chunks)
    rs = rows(run_dir)
    assert len(rs) == 1, "the request that lost the admission race left no row"
    assert rs[0]["finish"] == "refused"
    assert rs[0]["error"]["type"] == "QueueFull"
    assert rs[0]["stream"] is True


def test_the_streaming_row_is_written_before_the_loop_finalizes(run_dir):
    """The production shape: _sse `break`s out of `async for ... engine.stream`
    on the final event. Python does not close a generator on `break`, so
    engine.stream stayed suspended at its last yield until GC dropped it and
    asyncio scheduled the finalizer -- i.e. AFTER `async with engine.slot`
    released the lock. wall_s then disagreed with server.log's DONE line, and
    the row's `after` counter snapshot could pick up the next admitted
    request's NVMe reads. Reading the file with no await in between is what
    makes this fail before the fix: the finalizer cannot have run yet.

    The existing tests both run the generator to exhaustion or call aclose()
    explicitly; neither reproduces this shape.
    """
    import asyncio
    import time

    import vaayu.server.app as app_mod

    from .test_reqlog_engine import M as StreamingM

    # This file's M returns straight from generate(); only test_reqlog_engine's
    # feeds the streamer stream_generate's consumer loop blocks on.
    e = EngineHandle(StreamingM(), log_dir=run_dir)

    async def main():
        t0 = time.perf_counter()
        async for _chunk in app_mod._sse(e, _sse_req(), "m", None):
            pass
        elapsed = time.perf_counter() - t0
        return rows(run_dir), elapsed

    rs, elapsed = asyncio.run(main())
    assert len(rs) == 1, "the row was still pending in the event loop's finalizer"
    assert rs[0]["stream"] is True
    # Timed inside the slot, so it cannot exceed the whole consumption.
    assert rs[0]["wall_s"] is not None and rs[0]["wall_s"] <= elapsed


def test_a_jinja_error_on_the_streaming_path_writes_a_row(run_dir, no_real_load,
                                                          monkeypatch):
    """prepare() raises more than the three types the route names:
    apply_chat_template runs Jinja over the messages, and a malformed sequence
    raises whatever the template raises. On the streaming path that lands
    BEFORE any StreamingResponse exists, so it left a 500 with no row and no
    SSE generator at all."""
    cfg = ServerConfig(model_path="/ckpt", log_dir=str(run_dir.parent),
                       capture_stdio=False, require_session_id=False,
                       default_stream=False)
    app = create_app(cfg)
    with TestClient(app) as c:
        m = M()

        def boom(messages, **kw):
            raise ValueError("Jinja: unknown role 'tool' with no matching call")

        m.tokenizer.apply_chat_template = boom
        app.state.engine = EngineHandle(m, log_dir=run_dir)
        with pytest.raises(ValueError, match="Jinja"):
            c.post("/v1/chat/completions", json={
                "model": "m", "session_id": "s", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]})
    rs = rows(run_dir)
    assert len(rs) == 1, "a Jinja error on the streaming path left no row"
    # V1: it RAISED and produces a 500, so the vocabulary word is `error`.
    # `refused` is documentation/server.md's word for a rejection before
    # generation started -- an oversized prompt, a missing session_id, a full
    # queue, all 400/503 and all client-fixable. This row said `refused` only
    # because _refusal_row hardcoded it when every caller really was one.
    assert rs[0]["finish"] == "error"
    assert rs[0]["error"]["type"] == "ValueError"


def test_a_jinja_error_gets_the_same_finish_on_both_paths(run_dir, no_real_load):
    """V1: one failure, one word for it.

    A maintainer filters a mailed bundle by `finish`. While the streaming path
    said `refused` and the non-streaming path `error` for the SAME template
    failure, filtering on `error` under-counted them by the streaming share of
    traffic -- and DEFAULT_STREAM is True, so that is most of it.
    """
    cfg = ServerConfig(model_path="/ckpt", log_dir=str(run_dir.parent),
                       capture_stdio=False, require_session_id=False,
                       default_stream=False)
    app = create_app(cfg)
    with TestClient(app) as c:
        m = M()

        def boom(messages, **kw):
            raise ValueError("Jinja: unknown role 'tool' with no matching call")

        m.tokenizer.apply_chat_template = boom
        app.state.engine = EngineHandle(m, log_dir=run_dir)
        body = {"model": "m", "session_id": "s",
                "messages": [{"role": "user", "content": "hi"}]}
        # Streaming: the route re-raises, so the 500 is TestClient's exception.
        with pytest.raises(ValueError, match="Jinja"):
            c.post("/v1/chat/completions", json=dict(body, stream=True))
        # Non-streaming: generate()'s finally writes the row, the route answers
        # 500 rather than re-raising.
        r = c.post("/v1/chat/completions", json=dict(body, stream=False))
        assert r.status_code == 500, r.text[:200]

    streamed, non_streamed = rows(run_dir)
    assert streamed["stream"] is True and non_streamed["stream"] is False
    assert streamed["finish"] == non_streamed["finish"], (
        f"same failure, two words: streaming={streamed['finish']!r} "
        f"non-streaming={non_streamed['finish']!r}")
    assert streamed["finish"] == "error"


def test_a_refusal_row_has_the_same_keys_as_an_engine_row(client_requiring_session,
                                                          client, run_dir):
    """A maintainer's script does row["prefix_hit"]. Omitting the key on a
    refusal row rather than nulling it made that a KeyError depending on which
    kind of row it hit -- and documentation/server.md documents only the
    engine row's shape."""
    r = client_requiring_session.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    r = client.post("/v1/chat/completions", json={
        "model": "m", "session_id": "s", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200

    refusal, engine_row = rows(run_dir)
    assert refusal["finish"] == "refused"
    assert set(refusal) == set(engine_row), (
        f"only in one: {set(refusal) ^ set(engine_row)}")
    for k in ("prefix_hit", "ttft_s", "decode_tok_s", "prompt_tokens", "cap",
              "decode"):
        assert refusal[k] is None


@pytest.mark.parametrize("consumed,warning", [(1, None), (2, None), (3, "w")])
def test_a_disconnect_before_generation_starts_still_writes_a_row(
        run_dir, consumed, warning):
    """`_sse` yields the queue-position comment, an optional warning and the
    role chunk BEFORE entering the try that owns row-writing. Starlette closes
    an abandoned generator with GeneratorExit at whichever of those it is
    suspended on, so the request ended with no row at all -- the one known
    exception to the "a request never ends without a row" contract README.md
    and documentation/server.md both state.

    engine.stream's own row-writing `finally` cannot cover this: it runs only
    once that generator's BODY has started, which is after all three yields.
    """
    import asyncio

    import vaayu.server.app as app_mod
    from vaayu.server.scheduler import BatchScheduler

    from .test_reqlog_engine import M as StreamingM

    e = EngineHandle(StreamingM(), log_dir=run_dir)

    async def main():
        req = _sse_req()
        sched = BatchScheduler(e, queue_depth=8, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        gen = app_mod._sse(e, req, "m", warning, prepared=e.prepare(req),
                           scheduler=sched)
        for _ in range(consumed):
            await gen.__anext__()
        await gen.aclose()          # exactly what Starlette does on disconnect
        await sched.stop()

    asyncio.run(main())
    rs = rows(run_dir)
    assert len(rs) == 1, "a client that vanished pre-generation left no row"
    assert rs[0]["finish"] == "cancelled"
    assert rs[0]["stream"] is True
    # documentation/server.md: only `error`/`refused` rows carry an error object.
    assert rs[0]["error"] is None


def test_a_disconnect_while_queued_still_writes_a_row(run_dir):
    """The same hole, one step further in: a request whose JOB the dispatcher
    has already picked up, but which is blocked behind the engine lock, is
    abandoned by cancelling the ASGI task. Since P10 that cancellation no
    longer raises inside `_sse` itself -- clients hold a queue, not the
    generation generator (scheduler.py's own module docstring) -- so the row
    is not `_refusal_row`'s to write here. It is the dispatcher's, once
    unblocked: `_run_serial` calls `engine.stream` regardless of
    `job.cancel.cancelled` (only `_form` drops a job pre-run, and this one has
    already left `_waiting`), so it starts with `cancel` already set and runs
    to its ordinary success tail -- `row["error"]` stays `None`, and
    `finish_reason` maps a cancelled run to the WIRE value `"length"` there
    (stopping.py's `finish_reason` docstring), not the row-only `"cancelled"`
    enrichment `stream()`'s GeneratorExit branch adds. That enrichment is a
    per-branch thing, not a blanket `cancel.cancelled` check, so a run that
    reaches the success tail already cancelled does not get it -- pre-existing
    and unrelated to P10, noted here rather than changed.
    """
    import asyncio

    import vaayu.server.app as app_mod
    from vaayu.server.scheduler import BatchScheduler

    from .test_reqlog_engine import M as StreamingM

    e = EngineHandle(StreamingM(), log_dir=run_dir)

    async def main():
        req = _sse_req()
        sched = BatchScheduler(e, queue_depth=8, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        await e._lock.acquire()             # hold the lock so the dispatcher blocks
        gen = app_mod._sse(e, req, "m", None, prepared=e.prepare(req),
                           scheduler=sched)
        await gen.__anext__()               # queue position
        await gen.__anext__()               # role chunk

        async def drain():
            async for _ in gen:
                pass

        t = asyncio.create_task(drain())
        # Long enough for the job to be submitted, picked up by the dispatch
        # loop, and blocked on `engine._lock` -- not just enqueued.
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        e._lock.release()
        # Let the dispatcher actually finish the now-cancelled run -- it,
        # not this generator, owns writing the row.
        await sched.stop()

    asyncio.run(main())
    rs = rows(run_dir)
    assert len(rs) == 1, "a client that vanished while queued left no row"
    assert rs[0]["finish"] == "length"
    assert rs[0]["error"] is None
