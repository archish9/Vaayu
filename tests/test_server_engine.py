"""EngineHandle: admission gate and request-to-kwargs mapping.

Exercised against a fake model. No torch model, no checkpoint -- the parts that
need a real GPU are covered by test_server_smoke.py at the repo root.
"""
import asyncio
import time

import pytest

from vaayu.server.engine import EngineHandle, QueueFull
from vaayu.server.schemas import ChatCompletionRequest

HERMES_TEMPLATE = "tools <tool_call> {% for m in messages %}{{ m.content }}{% endfor %}"
PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class FakeTokenizer:
    def __init__(self, chat_template=PLAIN_TEMPLATE):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, **kw):
        return [1] * 10


class FakeModel:
    def __init__(self, max_seq_len=8192, chat_template=PLAIN_TEMPLATE, speculator=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FakeTokenizer(chat_template)
        self.speculator = speculator
        self.running_device = "cpu"
        self._vram_plan = object()      # not the CPU fallback path

    def get_prompt_renderer(self):
        return None


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_max_seq_len_is_read_off_the_model_not_stored():
    e = EngineHandle(FakeModel(max_seq_len=71981))
    assert e.max_seq_len == 71981


def test_prepare_returns_the_prompt_ids_and_the_resolved_cap():
    e = EngineHandle(FakeModel(max_seq_len=100))
    ids, cap = e.prepare(_req())
    assert ids == [1] * 10
    assert cap == 90, "an omitted max_tokens takes the whole remaining window"


def test_prepare_clamps_a_requested_cap_to_the_window():
    e = EngineHandle(FakeModel(max_seq_len=100))
    _, cap = e.prepare(_req(max_tokens=10_000))
    assert cap == 90


def test_greedy_requests_do_not_sample():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.0), [1, 2, 3])
    assert kw["do_sample"] is False
    assert "top_p" not in kw and "top_k" not in kw


def test_sampling_parameters_are_passed_only_when_sampling():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.7, top_p=0.9, top_k=40), [1, 2, 3])
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["top_k"] == 40


def test_stop_strings_require_the_tokenizer_and_are_passed_together():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(stop=["```"]), [1, 2, 3])
    assert kw["stop_strings"] == ["```"]
    assert kw["tokenizer"] is e.model.tokenizer


def test_the_tool_close_marker_is_added_as_a_stop_string():
    # Otherwise decode runs past the call and hallucinates the tool's result.
    e = EngineHandle(FakeModel(chat_template=HERMES_TEMPLATE))
    kw = e.build_generate_kwargs(
        _req(tools=[{"type": "function", "function": {"name": "f"}}]), [1, 2, 3])
    assert "</tool_call>" in kw["stop_strings"]


def test_session_id_is_passed_straight_through():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(session_id="agent-7"), [1, 2, 3])
    assert kw["session_id"] == "agent-7"


def test_an_omitted_session_id_is_none_so_auto_prefix_match_runs():
    e = EngineHandle(FakeModel())
    assert e.build_generate_kwargs(_req(), [1, 2, 3])["session_id"] is None


def test_no_notice_when_there_is_no_speculator():
    e = EngineHandle(FakeModel(speculator=None))
    assert e.notice_if_speculation_skipped(_req(top_p=0.5)) is None


def test_a_sampled_request_is_told_speculation_is_skipped_for_it():
    # stream_generate dispatches per request now, so top_p is NOT ignored --
    # the request takes the ordinary path and gets it. Saying "IGNORED" here
    # would send the caller to unload a drafter that is costing them nothing.
    e = EngineHandle(FakeModel(speculator=object()))
    msg = e.notice_if_speculation_skipped(_req(temperature=0.7, top_p=0.5))
    assert msg is not None and "top_p" in msg
    assert "IGNORED" not in msg
    assert "drafter_model_path" not in msg


def test_a_request_with_stop_strings_gets_the_same_notice():
    e = EngineHandle(FakeModel(speculator=object()))
    assert "stop" in e.notice_if_speculation_skipped(_req(stop=["```"]))


def test_a_greedy_request_gets_no_notice_because_it_does_speculate():
    e = EngineHandle(FakeModel(speculator=object()))
    assert e.notice_if_speculation_skipped(_req(temperature=0.0)) is None


def test_the_cpu_max_seq_len_fallback_is_reported_at_construction(capsys):
    # vaayu_base falls back to max_seq_len=512 when there is no VramPlan (CPU /
    # no CUDA). Defaulting the cap to "the remaining window" would silently
    # inherit a 512-token window, which is the very bug this server removes.
    m = FakeModel(max_seq_len=512)
    m._vram_plan = None
    EngineHandle(m)
    assert "512" in capsys.readouterr().out


def test_prefix_stats_reports_disabled_when_there_is_no_pool():
    assert EngineHandle(FakeModel()).prefix_stats() == "disabled"


def test_prefix_stats_delegates_to_the_pool():
    m = FakeModel()
    m._prefix_cache = type("P", (), {"stats": lambda self: "hits=3 misses=1"})()
    assert EngineHandle(m).prefix_stats() == "hits=3 misses=1"


def test_requests_beyond_the_queue_depth_are_refused():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=2)
        held = []

        async def hold():
            async with e.slot():
                held.append(1)
                await asyncio.sleep(0.2)

        a = asyncio.create_task(hold())
        b = asyncio.create_task(hold())
        await asyncio.sleep(0.05)
        with pytest.raises(QueueFull):
            async with e.slot():
                pass
        await asyncio.gather(a, b)
        assert len(held) == 2

    asyncio.run(main())


def test_the_gate_serialises_so_the_model_is_never_touched_concurrently():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=8)
        order, live = [], 0

        async def work(tag):
            nonlocal live
            async with e.slot():
                live += 1
                assert live == 1, "two requests were inside the gate at once"
                order.append(tag)
                await asyncio.sleep(0.02)
                live -= 1

        await asyncio.gather(*(work(i) for i in range(4)))
        assert sorted(order) == [0, 1, 2, 3]

    asyncio.run(main())


def test_a_slot_is_released_even_when_the_body_raises():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=1)
        with pytest.raises(ValueError):
            async with e.slot():
                raise ValueError("boom")
        assert e.depth == 0
        async with e.slot():
            pass

    asyncio.run(main())


def test_no_kwarg_stream_generate_builds_itself_survives_into_its_kwargs():
    # stream_generate constructs these; a duplicate arrives as
    # "TypeError: dict() got multiple values for keyword argument 'use_cache'".
    # That shipped past every fake-backed test and was caught only on-device.
    import inspect

    from vaayu.server.engine import STREAM_GENERATE_OWNS
    from vaayu.streaming import stream_generate

    src = inspect.getsource(stream_generate)
    for name in STREAM_GENERATE_OWNS:
        assert f"{name}=" in src, (
            f"{name} is listed as owned by stream_generate but does not appear "
            f"in its source -- the set has drifted")

    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.7, top_p=0.9), [1, 2, 3])
    for name in STREAM_GENERATE_OWNS:
        kw.pop(name, None)
    assert set(kw) <= {"session_id", "top_k", "stop_strings", "tokenizer"}


def test_user_reaches_the_prefix_cache_as_the_session_id():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(user="agent-7"), [1, 2, 3])
    assert kw["session_id"] == "agent-7"


def test_session_id_still_wins_over_user():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(user="alice", session_id="agent-7"), [1, 2, 3])
    assert kw["session_id"] == "agent-7"


# ── streaming reports real generated ids (B2) and honours cancellation (B5) ──


class _Config:
    def __init__(self, eos_token_id=2):
        self.eos_token_id = eos_token_id


class _CountingTokenizer(FakeTokenizer):
    """A tokenizer whose re-tokenization of decoded text is SHORTER than the
    ids that produced it -- which is what really happens, since the streamed
    text is decoded with skip_special_tokens=True (EOS gone) and merges differ
    across a decode boundary. The old code counted tokens this way.
    """

    def __call__(self, text, **kw):
        return {"input_ids": [0] * (len(text.split()) - 1)}


def _streaming_engine(generated_ids, *, eos_token_id=2, chunks=("a b c d e",)):
    """An EngineHandle whose stream_generate is replaced by a fake that emits
    `chunks` and reports `generated_ids` as the ids it actually produced."""
    import vaayu.streaming as streaming_mod

    model = FakeModel(max_seq_len=100)
    model.tokenizer = _CountingTokenizer()
    model.config = _Config(eos_token_id)

    def _fake(m, input_ids, max_new_tokens=None, token_ids_out=None, **kw):
        if token_ids_out is not None:
            token_ids_out.extend(generated_ids)
        yield from chunks

    return EngineHandle(model), streaming_mod, _fake


async def _run_stream(engine, req, cancel=None):
    final = None
    async for kind, payload in engine.stream(req, cancel=cancel):
        if kind == "final":
            final = payload
    return final


def test_stream_counts_real_generated_ids_not_a_retokenization(monkeypatch):
    e, mod, fake = _streaming_engine([10, 11, 12, 13, 14])
    monkeypatch.setattr(mod, "stream_generate", fake)
    final = asyncio.run(_run_stream(e, _req(max_tokens=5)))
    # Re-tokenizing "a b c d e" through _CountingTokenizer gives 4. The model
    # generated 5.
    assert final.completion_tokens == 5


def test_a_stream_that_ran_to_the_cap_reports_length_not_stop(monkeypatch):
    # THE B2 REGRESSION. Five ids generated against cap=5, none of them EOS:
    # the answer IS cut off. The old count (a re-tokenization yielding 4) made
    # `produced < cap` true and reported "stop" -- telling a coding agent its
    # truncated file was complete.
    e, mod, fake = _streaming_engine([10, 11, 12, 13, 14])
    monkeypatch.setattr(mod, "stream_generate", fake)
    final = asyncio.run(_run_stream(e, _req(max_tokens=5)))
    assert final.finish_reason == "length"


def test_a_stream_that_ended_on_eos_reports_stop(monkeypatch):
    e, mod, fake = _streaming_engine([10, 11, 2], eos_token_id=2)
    monkeypatch.setattr(mod, "stream_generate", fake)
    final = asyncio.run(_run_stream(e, _req(max_tokens=5)))
    assert final.finish_reason == "stop"


def test_a_cancelled_stream_is_not_reported_as_stop(monkeypatch):
    from vaayu.server.engine import CancelToken

    e, mod, fake = _streaming_engine([10, 11])
    monkeypatch.setattr(mod, "stream_generate", fake)
    cancel = CancelToken()
    cancel.cancel()
    final = asyncio.run(_run_stream(e, _req(max_tokens=5), cancel=cancel))
    assert final.finish_reason == "length"


def test_a_cancelled_non_stream_run_is_not_reported_as_stop():
    from vaayu.server.engine import CancelToken

    model = FakeModel()
    model.config = _Config()
    e = EngineHandle(model)
    cancel = CancelToken()
    cancel.cancel()
    assert e._finish([10, 11], cap=5, tool_calls=[], cancel=cancel) == "length"
    assert e._finish([10, 11], cap=5, tool_calls=[]) == "stop", (
        "an uncancelled short run is still a stop-string stop")


# ── the slot outlives the generation on every exit path (B4) ─────────────────


def _blocking_fake(engine, cancel, record, *, tick=0.02):
    """A stream_generate stand-in that keeps producing until `cancel` fires. It
    records its own thread, and whether the engine lock was still held at the
    moment it stopped."""
    import threading

    def _fake(m, input_ids, max_new_tokens=None, token_ids_out=None, **kw):
        record["thread"] = threading.current_thread()
        try:
            while not cancel.cancelled:
                yield "tok "
                time.sleep(tick)
        finally:
            if token_ids_out is not None:
                token_ids_out.extend([10, 11])
            record["lock_at_exit"] = engine._lock.locked()
    return _fake


def test_sse_holds_the_slot_until_the_abandoned_worker_thread_has_stopped(monkeypatch):
    # B4. Starlette closes the SSE generator when the client goes away.
    #
    # Since P10, admission runs through BatchScheduler, not engine.slot()
    # directly -- clients hold a QUEUE, not the generation generator itself
    # (scheduler.py's own module docstring: this is why an abandoned client
    # cannot corrupt a batch). So `_sse`'s own aclose() no longer
    # synchronously waits for the worker to stop; the dispatcher does, in its
    # own task, still holding `engine._lock` for the whole run. That is the
    # property this test now checks: the lock stays held until the worker
    # thread the dispatcher abandoned actually exits, and the scheduler's
    # queue is empty once the dispatcher (not the client) is done with it --
    # only then may the next queued request be admitted.
    from vaayu.server.app import _sse
    from vaayu.server.engine import CancelToken
    from vaayu.server.scheduler import BatchScheduler

    e, mod, _ = _streaming_engine([10, 11])
    cancel = CancelToken()
    record = {}
    monkeypatch.setattr(mod, "stream_generate", _blocking_fake(e, cancel, record))

    async def main():
        req = _req(max_tokens=50)
        sched = BatchScheduler(e, queue_depth=8, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        agen = _sse(e, req, "vaayu", None, cancel=cancel,
                    prepared=e.prepare(req), scheduler=sched)
        seen = 0
        async for _ in agen:
            seen += 1
            if seen >= 4:
                break
        await agen.aclose()             # exactly what Starlette does on disconnect
        # The dispatcher, not this generator, owns finishing the abandoned
        # run -- wait for it the same way the app lifespan's shutdown does.
        await sched.stop()
        return record["thread"].is_alive(), sched.depth

    alive_after_close, depth_after_close = asyncio.run(main())
    record["thread"].join(timeout=5)
    # NOTE: `alive_after_close` is read AFTER `sched.stop()`, which itself
    # already waits for the dispatcher's current run (worker join included)
    # to finish -- so this assertion can no longer, by itself, distinguish
    # "held correctly" from "`_sse` returned early and `stop()` masked it".
    # `record["lock_at_exit"]` (below) is what still pins the real property:
    # it is read on the WORKER THREAD, at the moment it observes the
    # cancellation and exits, independent of anything `stop()` does.
    assert not alive_after_close, (
        "the dispatcher stopped while the generation worker was still "
        "running -- the lock is free and a queued request will be admitted")
    assert record["lock_at_exit"] is True, (
        f"the engine lock was {record['lock_at_exit']} when the worker stopped; "
        f"it must still be held, or two generations can overlap")
    assert depth_after_close == 0


def test_a_stream_that_finishes_normally_still_releases_the_slot(monkeypatch):
    # Guards the fix above: quiescing at slot exit must not wait on a worker
    # that has already gone, and must leave the gate reusable. Exercised
    # through the dispatcher (P10) rather than _sse holding engine.slot()
    # directly -- same property: the engine's lock is free and the
    # scheduler's queue is empty once the run is done.
    from vaayu.server.app import _sse
    from vaayu.server.engine import CancelToken
    from vaayu.server.scheduler import BatchScheduler

    e, mod, fake = _streaming_engine([10, 11, 2], eos_token_id=2)
    monkeypatch.setattr(mod, "stream_generate", fake)

    async def main():
        req = _req(max_tokens=5)
        sched = BatchScheduler(e, queue_depth=8, max_batch_size=1,
                               prefill_chunk_size=512)
        await sched.start()
        events = [ev async for ev in _sse(e, req, "vaayu", None,
                                          cancel=CancelToken(),
                                          prepared=e.prepare(req), scheduler=sched)]
        await sched.stop()
        async with e._lock:              # the gate is usable again
            pass
        return events, sched.depth

    events, depth_after = asyncio.run(main())
    assert events[-1] == "data: [DONE]\n\n"
    assert depth_after == 0


# ── the cancel token stays published until the worker exits (B9) ─────────────


def _self_cancelling_fake(engine, record, *, tick=0.01, rounds=400):
    """A stream_generate stand-in that can ONLY be stopped through the token the
    engine has published -- whatever the caller passed in. It records that token
    at the moment it exits. Bounded so a worker that never gets cancelled cannot
    outlive the test."""
    import threading

    def _fake(m, input_ids, max_new_tokens=None, token_ids_out=None, **kw):
        record["thread"] = threading.current_thread()
        try:
            for _ in range(rounds):
                token = engine.live_cancel
                if token is not None and token.cancelled:
                    return
                yield "tok "
                time.sleep(tick)
        finally:
            record["live_cancel_at_exit"] = engine.live_cancel
            if token_ids_out is not None:
                token_ids_out.extend([10, 11])
    return _fake


def test_an_abandoned_stream_keeps_its_token_published_until_the_worker_exits(monkeypatch):
    # B9. Closing the stream raises GeneratorExit at its `yield`, and the
    # `finally` used to clear _live_cancel there and skip thread.join()
    # entirely -- so cancel_all() (the SIGTERM handler, _teardown) saw None and
    # reported idle with a generation still on the GPU.
    e, mod, _ = _streaming_engine([10, 11])
    record = {}
    monkeypatch.setattr(mod, "stream_generate", _self_cancelling_fake(e, record))

    async def main():
        agen = e.stream(_req(max_tokens=50))
        seen = 0
        async for _kind, _payload in agen:
            seen += 1
            if seen >= 3:
                break
        await agen.aclose()             # what closing an abandoned stream does
        return record["thread"].is_alive(), e.live_cancel

    alive_after_close, live_after_close = asyncio.run(main())
    record["thread"].join(timeout=5)
    assert not alive_after_close, (
        "the stream returned while its generation worker was still running -- "
        "the token was cleared and the join skipped, so shutdown sees an idle "
        "engine with the GPU busy")
    assert record["live_cancel_at_exit"] is not None, (
        "the token was already cleared when the worker exited; anything asking "
        "cancel_all() in that window is told the engine is idle")
    assert live_after_close is None, "and it must be cleared once the worker is gone"


def test_cancel_all_stops_a_stream_started_without_a_caller_token(monkeypatch):
    # A stream published whatever it was handed, so a caller that passed no
    # token published None -- cancel_all() then had nothing to cancel and
    # reported idle for the whole generation.
    e, mod, _ = _streaming_engine([10, 11])
    record = {}
    monkeypatch.setattr(mod, "stream_generate", _self_cancelling_fake(e, record))

    async def main():
        agen = e.stream(_req(max_tokens=50))
        try:
            async for _kind, _payload in agen:
                break
            assert e.live_cancel is not None, (
                "nothing is published, so cancel_all() cannot stop this run")
            e.cancel_all()
            async for _kind, _payload in agen:
                pass
        finally:
            await agen.aclose()

    asyncio.run(main())
    record["thread"].join(timeout=5)
    assert not record["thread"].is_alive()


def test_a_normally_finished_stream_clears_its_published_token(monkeypatch):
    # Guard: the token the stream now creates for itself must not read back as
    # cancelled, or every normal answer would be labelled "length".
    e, mod, fake = _streaming_engine([10, 11, 2], eos_token_id=2)
    monkeypatch.setattr(mod, "stream_generate", fake)
    final = asyncio.run(_run_stream(e, _req(max_tokens=5)))
    assert final.finish_reason == "stop"
    assert e.live_cancel is None and e._live_thread is None


# ── the non-streaming route publishes a joinable worker too (B16) ────────────


class _Ids(list):
    """A stand-in for the generate() output tensor: sliceable, and the slice
    still answers .tolist() the way _decode_new expects."""

    def __getitem__(self, key):
        got = list.__getitem__(self, key)
        return _Ids(got) if isinstance(key, slice) else got

    def tolist(self):
        return list(self)


class _DecodingTokenizer(FakeTokenizer):
    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def _blocking_generate_engine(record, *, rounds=400, tick=0.01):
    """An EngineHandle whose model.generate blocks until the token the ENGINE
    published is cancelled -- not one the test holds. Bounded so a worker that
    never gets cancelled cannot outlive the test."""
    import threading

    model = FakeModel(max_seq_len=100)
    model.tokenizer = _DecodingTokenizer()
    model.config = _Config()
    engine = EngineHandle(model)

    def _generate(**kwargs):
        record["thread"] = threading.current_thread()
        record["live_thread_while_running"] = engine._live_thread
        record["live_cancel_while_running"] = engine.live_cancel
        try:
            for _ in range(rounds):
                token = engine.live_cancel
                if token is not None and token.cancelled:
                    break
                time.sleep(tick)
        finally:
            record["live_cancel_at_exit"] = engine.live_cancel
        return _Ids([_Ids([1] * 10 + [10, 11])])

    model.generate = _generate
    return engine, record


def test_a_cancelled_non_stream_handler_waits_for_its_worker_to_exit():
    # B16, B9's shape on the other route. The async `finally` cleared
    # _live_cancel when `await run_in_executor(...)` raised, and a handler task
    # cancelled by uvicorn's graceful-shutdown timeout raises CancelledError
    # there while the executor thread is still inside model.generate. The token
    # was gone, so nothing could stop it, and slot()'s quiesce() had no thread
    # to join -- it released the lock under a live generation.
    e, record = _blocking_generate_engine({})

    async def main():
        async def call():
            async with e.slot():
                return await e.generate(_req(max_tokens=50))

        task = asyncio.create_task(call())
        while "thread" not in record:
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return record["thread"].is_alive(), e.live_cancel, e._lock.locked()

    alive_after_cancel, live_after_cancel, locked_after = asyncio.run(main())
    record["thread"].join(timeout=10)
    assert not alive_after_cancel, (
        "the cancelled handler returned while its generation worker was still "
        "running -- the slot is free and a queued request will be admitted")
    assert record["live_cancel_at_exit"] is not None, (
        "the token was already cleared when the worker exited; anything asking "
        "cancel_all() in that window is told the engine is idle")
    assert live_after_cancel is None, "and it must be cleared once the worker is gone"
    assert locked_after is False


def test_cancel_all_stops_a_generate_started_without_a_caller_token():
    # generate() published whatever it was handed, so a caller passing no token
    # published None: cancel_all() had nothing to cancel for the WHOLE run.
    e, record = _blocking_generate_engine({})

    async def main():
        task = asyncio.create_task(e.generate(_req(max_tokens=50)))
        while "thread" not in record:
            await asyncio.sleep(0.005)
        assert e.live_cancel is not None, (
            "nothing is published, so cancel_all() cannot stop this run")
        e.cancel_all()
        await task

    asyncio.run(main())
    record["thread"].join(timeout=10)
    assert not record["thread"].is_alive()


def test_generate_publishes_a_thread_quiesce_can_join():
    # quiesce() waits on _live_thread. generate() published none, so B4's fix
    # covered only stream(): the lock was released under a live generation.
    e, record = _blocking_generate_engine({})

    async def main():
        task = asyncio.create_task(e.generate(_req(max_tokens=50)))
        while "thread" not in record:
            await asyncio.sleep(0.005)
        e.cancel_all()
        await task

    asyncio.run(main())
    record["thread"].join(timeout=10)
    assert record["live_thread_while_running"] is record["thread"], (
        "quiesce() has nothing to join while generate() is running")
    assert e._live_thread is None, "and it must be cleared once the worker is gone"


def test_a_normally_finished_generate_still_reports_stop():
    # Guard: the token generate() now creates for itself must not read back as
    # cancelled, or every normal answer would be labelled "length".
    e, record = _blocking_generate_engine({}, rounds=0)
    result = asyncio.run(e.generate(_req(max_tokens=5)))
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 2
    assert e.live_cancel is None and e._live_thread is None


# ── a template without tool support must still refuse tools ─────────────────

def test_tool_format_is_none_for_a_template_without_tool_support():
    e = EngineHandle(FakeModel(chat_template=PLAIN_TEMPLATE))
    assert e.tool_format is None, "a plain template must still refuse tools"


def test_a_speculator_plus_temperature_reports_the_skip_not_a_loss():
    # `temperature` is a declared, unread parameter of that loop and every
    # token in it is an argmax, so a sampled request cannot run there. It is
    # now sent to the ordinary path instead of quietly getting greedy text.
    e = EngineHandle(FakeModel(speculator=object()))
    msg = e.notice_if_speculation_skipped(_req(temperature=0.7))
    assert msg is not None and "temperature" in msg


# ── the non-streaming route speculates too ───────────────────────────────────
# It called `model.generate` directly, so speculation -- dispatched inside
# `stream_generate` -- never applied to it. Measured through the server on
# 2026-09-06: 10 draft rounds streamed, 0 non-streamed, same greedy prompt in
# the same process. Both routes now ask `choose_speculative_target`.

class _DecodingTokenizer(_CountingTokenizer):
    """`generate()`'s route decodes the ids it got back; the streaming fakes
    never needed that."""

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


class _TwoPathFakeModel(FakeModel):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.tokenizer = _DecodingTokenizer()
        self.config = _Config()
        self.called = None

    def _out(self):
        import torch
        return torch.tensor([[1] * 12])         # prompt is 10 ids; 2 generated

    def generate(self, **kwargs):
        self.called = "generate"
        return self._out()

    def speculative_generate(self, **kwargs):
        self.called = "speculative_generate"
        return self._out()


def _nonstreaming_path(**over):
    model = _TwoPathFakeModel(max_seq_len=100, speculator=object())
    e = EngineHandle(model)
    asyncio.run(e.generate(_req(max_tokens=5, **over)))
    return model.called


def test_a_greedy_non_streaming_request_speculates():
    assert _nonstreaming_path(temperature=0.0) == "speculative_generate"


def test_a_sampled_non_streaming_request_takes_the_ordinary_path():
    assert _nonstreaming_path(temperature=0.7) == "generate"


def test_a_non_streaming_request_with_stop_strings_takes_the_ordinary_path():
    assert _nonstreaming_path(temperature=0.0, stop=["```"]) == "generate"


def test_a_non_streaming_request_without_a_speculator_is_unaffected():
    model = _TwoPathFakeModel(max_seq_len=100, speculator=None)
    asyncio.run(EngineHandle(model).generate(_req(max_tokens=5)))
    assert model.called == "generate"


def test_prepare_renders_with_the_models_own_renderer_when_it_has_one():
    class Renderer:
        def render(self, messages, enable_thinking):
            return "xyz"

    class TextTokenizer(FakeTokenizer):
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [ord(c) for c in text]}

    model = FakeModel(chat_template=None)
    model.tokenizer = TextTokenizer(None)
    renderer = Renderer()
    model.get_prompt_renderer = lambda: renderer

    ids, _ = EngineHandle(model).prepare(_req())

    assert ids == [120, 121, 122]


# ── tool-format discovered by self-probing the template (model-agnostic) ──────
# A tokenizer that renders MiniMax's native XML tool call when the probe asks
# for a string render, and ids otherwise. The probe sends an assistant message
# carrying tool_calls (the call render) and a plain one (to strip the header).
_MM_CALL = (']~b]ai\n\n<minimax:tool_call>\n<invoke name="ZZNAMEZZ">\n'
            '<parameter name="KKKEYKK">VVVALVV</parameter>\n</invoke>\n'
            '</minimax:tool_call>[e~[\n')
_MM_PLAIN = ']~b]ai\nPLAINTEXTPAYLOAD[e~[\n'


class _MiniMaxTokenizer:
    # Non-empty, and WITHOUT a <tool_call> marker, so the registry fallback would
    # mis-type this as GENERIC_JSON -- only the probe gets it right.
    chat_template = "tools go here, call with <minimax:tool_call> ... {{ tools }}"

    def apply_chat_template(self, messages, **kw):
        if kw.get("tokenize", True) is False:
            return _MM_CALL if messages[-1].get("tool_calls") else _MM_PLAIN
        return [1] * 10


def test_tool_format_is_discovered_by_probing_the_template_not_the_marker_table():
    e = EngineHandle(FakeModel())
    e.model.tokenizer = _MiniMaxTokenizer()
    fmt = e.tool_format
    assert fmt is not None and fmt.grammar == "xml"
    assert fmt.open_marker == "<minimax:tool_call>"


def test_the_probed_xml_close_marker_is_added_as_a_stop_string():
    e = EngineHandle(FakeModel())
    e.model.tokenizer = _MiniMaxTokenizer()
    kw = e.build_generate_kwargs(
        _req(tools=[{"type": "function", "function": {"name": "f"}}]), [1, 2, 3])
    assert "</minimax:tool_call>" in kw["stop_strings"]
