"""Stopping the server must actually stop it.

Observed: the process was killed and the GPU stayed held. Root cause is that
vaayu.streaming starts its generation worker as a NON-daemon thread, so on
SIGTERM the interpreter waits for a generation that may have thousands of
tokens left to run -- the process never exits, so CUDA never releases.

Teardown therefore has to (1) stop generating, (2) flush the disk KV writer,
(3) drop the model, and (4) never be blockable by a worker thread.
"""
import threading

import torch
from fastapi.testclient import TestClient

from vaayu.server.app import _teardown, create_app
from vaayu.server.engine import CancelToken, EngineHandle
from vaayu.server.schemas import ChatCompletionRequest
from vaayu.streaming import stream_generate

from .test_server_app import FakeEngine


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


# ── the engine knows what is running ─────────────────────────────────────────

class _RecordingModel:
    """Captures the CancelToken's state at the moment generate() is entered."""

    class _Tok:
        chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

        def apply_chat_template(self, messages, **kw):
            return [1, 2, 3]

        def decode(self, ids, **kw):
            return "x" * len(ids)

    max_seq_len = 8192
    running_device = "cpu"
    speculator = None
    _vram_plan = object()
    config = type("C", (), {"eos_token_id": 2})()

    def __init__(self):
        self.tokenizer = self._Tok()
        self.entered = threading.Event()
        self.saw_cancel = None

    def generate(self, **kw):
        crit = kw["stopping_criteria"]
        ids = kw["input_ids"]
        self.entered.set()
        for _ in range(10_000):
            if bool(crit(ids, None).any()):
                break
            ids = torch.cat([ids, torch.tensor([[7]])], dim=1)
        self.saw_cancel = True
        return ids


def test_cancel_all_is_a_no_op_when_nothing_is_running():
    EngineHandle(_RecordingModel()).cancel_all()      # must not raise


def test_cancel_all_trips_the_live_request():
    import asyncio

    model = _RecordingModel()
    e = EngineHandle(model)

    async def main():
        task = asyncio.create_task(e.generate(_req(max_tokens=9000),
                                              cancel=CancelToken()))
        await asyncio.sleep(0.05)
        e.cancel_all()
        return await asyncio.wait_for(task, timeout=5)

    result = asyncio.run(main())
    assert model.saw_cancel is True
    assert result.completion_tokens < 9000, "cancel_all stopped it short"


def test_the_live_token_is_forgotten_once_the_request_finishes():
    import asyncio

    e = EngineHandle(_RecordingModel())
    token = CancelToken()
    token.cancel()                                   # returns immediately
    asyncio.run(e.generate(_req(max_tokens=10), cancel=token))
    e.cancel_all()                                   # must not raise on a stale ref
    assert e.live_cancel is None


# ── teardown ─────────────────────────────────────────────────────────────────

class _RecordingWriter:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_teardown_flushes_the_disk_kv_writer():
    app = create_app(engine=FakeEngine())
    writer = _RecordingWriter()
    app.state.engine.model._kv_writer = writer
    _teardown(app)
    assert writer.closed is True, "an unflushed writer loses the last appends"


def test_teardown_survives_a_model_with_no_kv_writer():
    app = create_app(engine=FakeEngine())
    _teardown(app)                                   # must not raise


def test_teardown_drops_the_engine_so_the_model_can_be_collected():
    app = create_app(engine=FakeEngine())
    _teardown(app)
    assert app.state.engine is None


def test_teardown_cancels_whatever_is_still_generating():
    app = create_app(engine=EngineHandle(_RecordingModel()))
    token = CancelToken()
    app.state.engine._live_cancels = [token]
    _teardown(app)
    assert token.cancelled is True


def test_the_lifespan_runs_teardown_on_shutdown():
    app = create_app(engine=FakeEngine())
    with TestClient(app):
        assert app.state.engine is not None
    assert app.state.engine is None, "leaving the lifespan must tear down"


# ── the thread that used to block exit ───────────────────────────────────────

def test_the_generation_worker_is_a_daemon_thread():
    """A non-daemon worker holds the interpreter open after SIGTERM, which is
    exactly how a killed server kept the GPU."""
    seen = {}

    class Tok:
        def decode(self, ids, **kw):
            return ""

    class Model:
        tokenizer = Tok()
        speculator = None

        def generate(self, **kw):
            seen["daemon"] = threading.current_thread().daemon
            kw["streamer"].end()

    list(stream_generate(Model(), torch.tensor([[1]]), max_new_tokens=1))
    assert seen["daemon"] is True


# ── SIGTERM must cancel BEFORE uvicorn waits for the request ─────────────────
# uvicorn's Server.shutdown() awaits _wait_tasks_to_complete() and only then
# runs the lifespan shutdown, with timeout_graceful_shutdown defaulting to None.
# So a teardown-time cancel is unreachable while a generation is running: the
# signal handler is the only place early enough.

def test_the_signal_handler_cancels_then_delegates_to_uvicorns_own():
    import signal

    from vaayu.server.app import _install_signal_cancel

    engine = FakeEngine()
    delegated = []
    prev = signal.signal(signal.SIGTERM, lambda s, f: delegated.append(s))
    try:
        restore = _install_signal_cancel(engine)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
        handler(signal.SIGTERM, None)
        assert engine.cancelled_all is True, "generation must stop on the signal"
        assert delegated == [signal.SIGTERM], (
            "uvicorn's own handler drives the graceful shutdown -- chain to it, "
            "do not replace it")
        restore()
        assert signal.getsignal(signal.SIGTERM) is not handler
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_installing_from_a_non_main_thread_is_a_no_op_not_a_crash():
    # TestClient and any embedded use run the lifespan off the main thread,
    # where signal.signal raises ValueError.
    import threading

    from vaayu.server.app import _install_signal_cancel

    out = {}

    def run():
        out["restore"] = _install_signal_cancel(FakeEngine())

    t = threading.Thread(target=run)
    t.start()
    t.join()
    out["restore"]()          # must be callable and harmless
