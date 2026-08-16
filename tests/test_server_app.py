"""HTTP surface, exercised against a fake engine -- no model, no GPU."""
import json

from fastapi.testclient import TestClient

from vaayu.server.app import create_app
from vaayu.server.chat import TemplateLacksToolSupport
from vaayu.server.engine import GenerationResult, QueueFull
from vaayu.server.schemas import ToolCall, ToolCallFunction
from vaayu.server.stopping import ContextTooLong


class FakeEngine:
    """Implements only what app.py touches."""

    def __init__(self, result=None, raises=None, deltas=None):
        self.max_seq_len = 8192
        self.queue_depth = 8
        self.depth = 0
        self.model = type("M", (), {"name_or_path": "fake"})()
        self._result = result or GenerationResult(
            text="hello", prompt_tokens=3, completion_tokens=2, finish_reason="stop")
        self._raises = raises
        self._deltas = deltas or ["hel", "lo"]
        self.warned = None
        self.tool_format = None

    def warn_if_speculative_sampling_ignored(self, req):
        return self.warned

    def slot(self):
        engine = self

        class _Slot:
            async def __aenter__(self):
                if isinstance(engine._raises, QueueFull):
                    raise engine._raises

            async def __aexit__(self, *a):
                return False

        return _Slot()

    async def generate(self, req):
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        return self._result

    async def stream(self, req):
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        for d in self._deltas:
            yield ("delta", d)
        yield ("final", self._result)

    def prefix_stats(self):
        return "hits=1 misses=0"


def _client(engine):
    return TestClient(create_app(engine=engine))


BODY = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}


def test_health_reports_the_queue_and_the_prefix_cache():
    r = _client(FakeEngine()).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["queue_depth"] == 8
    assert body["max_seq_len"] == 8192
    assert "hits=1" in body["prefix_cache"]


def test_models_lists_the_one_loaded_model_with_its_derived_window():
    body = _client(FakeEngine()).get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["max_seq_len"] == 8192


def test_a_non_streaming_completion_returns_the_openai_shape():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "chat.completion"
    assert b["choices"][0]["message"]["content"] == "hello"
    assert b["choices"][0]["finish_reason"] == "stop"
    assert b["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert b["id"].startswith("chatcmpl-")


def test_tool_calls_are_returned_with_finish_reason_tool_calls():
    call = ToolCall(id="call_x", function=ToolCallFunction(name="f", arguments="{}"))
    engine = FakeEngine(result=GenerationResult(
        text="", prompt_tokens=3, completion_tokens=9,
        finish_reason="tool_calls", tool_calls=[call]))
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert b["choices"][0]["finish_reason"] == "tool_calls"
    assert b["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


def test_hitting_the_cap_is_reported_as_length_not_hidden():
    engine = FakeEngine(result=GenerationResult(
        text="def f(", prompt_tokens=3, completion_tokens=8, finish_reason="length"))
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert b["choices"][0]["finish_reason"] == "length"


def test_an_oversized_prompt_is_a_400_carrying_the_engine_message():
    engine = FakeEngine(raises=ContextTooLong("prompt is 9000 tokens but the window is 8192"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400
    assert "9000" in r.json()["error"]["message"]


def test_tools_against_a_template_without_tool_support_is_a_400():
    engine = FakeEngine(raises=TemplateLacksToolSupport("no tool support"))
    body = dict(BODY, tools=[{"type": "function", "function": {"name": "f"}}])
    r = _client(engine).post("/v1/chat/completions", json=body)
    assert r.status_code == 400


def test_a_full_queue_is_a_503_with_retry_after():
    engine = FakeEngine(raises=QueueFull("8 requests are already queued"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 503
    assert "Retry-After" in r.headers


def test_an_engine_crash_is_a_500_not_a_hang():
    engine = FakeEngine(raises=RuntimeError("CUDA error"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 500
    assert "CUDA error" in r.json()["error"]["message"]


def test_a_speculative_sampling_warning_is_surfaced_on_the_response():
    engine = FakeEngine()
    engine.warned = "top_p will be IGNORED"
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert "top_p" in b["warnings"][0]


def test_n_greater_than_one_is_a_422_from_validation():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, n=4))
    assert r.status_code == 422
