"""
Unit tests for vaayu/streaming.py's stream_generate().
Tests are designed to run without a real GPU or model checkpoint.
"""
import threading

from vaayu.streaming import stream_generate


class _FakeModel:
    """Minimal stand-in: only what stream_generate touches (.tokenizer,
    .speculator, .generate)."""
    tokenizer = object()
    speculator = None

    def __init__(self, behavior):
        self._behavior = behavior

    def generate(self, **kwargs):
        self._behavior(kwargs["streamer"])


def _happy_path(streamer):
    streamer.on_finalized_text("hello ", stream_end=False)
    streamer.on_finalized_text("world", stream_end=True)


def _crash_path(streamer):
    raise RuntimeError("boom")


def test_stream_generate_yields_chunks_normally():
    model = _FakeModel(_happy_path)
    chunks = list(stream_generate(model, input_ids=None, max_new_tokens=1))
    assert chunks == ["hello ", "world"]


def test_stream_generate_propagates_exception_instead_of_hanging():
    """A background-thread exception (e.g. the io_uring CQE overflow crash)
    must propagate to the caller, not deadlock stream_generate's consumer
    forever on the streamer's unbounded queue."""
    model = _FakeModel(_crash_path)
    result = {}

    def _drain():
        try:
            list(stream_generate(model, input_ids=None, max_new_tokens=1))
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "stream_generate hung instead of propagating the exception"
    assert isinstance(result.get("exc"), RuntimeError)
    assert str(result["exc"]) == "boom"
