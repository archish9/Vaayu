"""HTTP surface, exercised against a fake engine -- no model, no GPU."""
import json

from fastapi.testclient import TestClient

from vaayu.server.app import create_app
from vaayu.server.chat import NoChatTemplate, TemplateLacksToolSupport
from vaayu.server.engine import GenerationResult, QueueFull
from vaayu.server.schemas import ToolCall, ToolCallFunction
from vaayu.server.stopping import ContextTooLong
from vaayu.server.tools import GENERIC_JSON as _GENERIC_JSON
from vaayu.server.tools import HERMES as _HERMES


_PREPARE_ERRORS = (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate)


class FakeEngine:
    """Implements only what app.py touches."""

    def __init__(self, result=None, raises=None, deltas=None):
        self.max_seq_len = 8192
        self.queue_depth = 8
        self.depth = 0
        self.model = type("M", (), {"name_or_path": "fake"})()
        # The real model sets this on every load path, so a fake without it
        # would be testing a shape the server cannot produce (S1b).
        self.model.speculative_status = {"requested": None, "draft_tokens": 5,
                                         "active": False, "refused_because": None}
        self._result = result or GenerationResult(
            text="hello", prompt_tokens=3, completion_tokens=2, finish_reason="stop")
        self._raises = raises
        self._deltas = deltas or ["hel", "lo"]
        self.warned = None
        self.tool_format = None
        self.cancel = None          # the CancelToken the route handed us
        self.cancelled_all = False  # set by shutdown

    def warn_if_speculative_sampling_ignored(self, req):
        return self.warned

    # The real EngineHandle raises ContextTooLong / TemplateLacksToolSupport /
    # NoChatTemplate from HERE, and generate()/stream() call it. Modelling that
    # is what makes the streaming path's status mapping testable at all.
    def prepare(self, req):
        if isinstance(self._raises, _PREPARE_ERRORS):
            raise self._raises
        return [1, 2, 3], 16

    def slot(self, session_key=None):
        engine = self

        class _Slot:
            async def __aenter__(self):
                if isinstance(engine._raises, QueueFull):
                    raise engine._raises

            async def __aexit__(self, *a):
                return False

        return _Slot()

    async def generate(self, req, cancel=None):
        self.cancel = cancel
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        return self._result

    async def stream(self, req, cancel=None, prepared=None):
        self.cancel = cancel
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        for d in self._deltas:
            yield ("delta", d)
        yield ("final", self._result)

    def cancel_all(self):
        self.cancelled_all = True

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


def test_models_reports_whether_speculation_is_actually_running():
    """S1b: a user who sets DRAFTER_MODEL_PATH on a MoE target gets it refused
    at load, and before this had no way to find out short of reading a startup
    line that has already scrolled away."""
    e = FakeEngine()
    e.model.speculative_status = {
        "requested": "/models/Qwen2.5-Coder-1.5B", "draft_tokens": 5,
        "active": False, "refused_because": "verify working set exceeds budget"}
    body = _client(e).get("/v1/models").json()
    spec = body["data"][0]["speculative"]
    assert spec["active"] is False
    assert "exceeds budget" in spec["refused_because"]


def test_models_reports_speculation_off_when_none_was_asked_for():
    """The field must be present and honest on an ordinary load too -- an
    absent key reads as 'unknown', which is the state this fixes."""
    body = _client(FakeEngine()).get("/v1/models").json()
    assert body["data"][0]["speculative"]["active"] is False
    assert body["data"][0]["speculative"]["requested"] is None


def test_models_reports_the_runtime_calibration_too():
    e = FakeEngine()
    e.model.speculative_status = {
        "requested": "prompt_lookup", "draft_tokens": 3, "active": True,
        "refused_because": None,
        "calibration": {"observed_union_gb": 12.9, "budget_gb": 14.7,
                        "breakeven_acceptance": 0.635,
                        "measured_acceptance": 0.907,
                        "drafting_rounds": 17, "decision": "run"}}
    spec = _client(e).get("/v1/models").json()["data"][0]["speculative"]
    assert spec["calibration"]["decision"] == "run"


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


def _sse_events(raw: str):
    """Parse an SSE body into the list of JSON payloads, dropping [DONE]."""
    out = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def test_a_stream_emits_role_then_content_then_a_terminating_done():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")
    events = _sse_events(r.text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[0]["object"] == "chat.completion.chunk"
    contents = [e["choices"][0]["delta"].get("content") for e in events]
    assert "".join(c for c in contents if c) == "hello"


def test_the_last_stream_event_carries_the_finish_reason():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    events = _sse_events(r.text)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"][0]["delta"] == {}


def test_usage_is_omitted_unless_asked_for():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert all("usage" not in e for e in _sse_events(r.text))


def test_usage_is_included_when_stream_options_asks():
    body = dict(BODY, stream=True, stream_options={"include_usage": True})
    events = _sse_events(_client(FakeEngine()).post("/v1/chat/completions", json=body).text)
    assert events[-1]["usage"]["total_tokens"] == 5


def test_a_streamed_tool_call_is_emitted_once_and_complete():
    # Buffered, never emitted as partial JSON -- partial tool-call JSON breaks
    # clients that try to parse each delta.
    call = ToolCall(id="call_x", function=ToolCallFunction(name="f", arguments='{"a":1}'))
    engine = FakeEngine(
        deltas=["<tool_call>", '{"name":"f",'],
        result=GenerationResult(text="", prompt_tokens=3, completion_tokens=9,
                                finish_reason="tool_calls", tool_calls=[call]))
    engine.tool_format = _HERMES
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "f"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    with_calls = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert len(with_calls) == 1
    assert with_calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def _streamed_content(events):
    return "".join(e["choices"][0]["delta"].get("content", "") for e in events)


def test_a_fenced_json_answer_survives_the_stream_intact():
    # B7, streaming half: for GENERIC_JSON the "open marker" is a plain ```json
    # fence, which an ordinary code answer opens too. Suppressing from there on
    # deleted the block AND everything after it, and the final event carries no
    # content -- so the client never saw it at all.
    txt = 'Here is package.json:\n```json\n{"name": "my-app"}\n```\ndone'
    engine = FakeEngine(deltas=[txt], result=GenerationResult(
        text=txt, prompt_tokens=3, completion_tokens=9, finish_reason="stop"))
    engine.tool_format = _GENERIC_JSON
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "read_file"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    assert _streamed_content(events) == txt


def test_a_real_fenced_tool_call_is_still_not_duplicated_as_content():
    # The other side of the same buffer: when the block DID parse as a call it
    # is delivered as tool_calls, so it must not also appear in the prose.
    call = ToolCall(id="call_y", function=ToolCallFunction(name="read_file", arguments='{}'))
    txt = 'Sure.\n```json\n{"name": "read_file", "arguments": {}}\n```'
    engine = FakeEngine(deltas=[txt], result=GenerationResult(
        text="Sure.", prompt_tokens=3, completion_tokens=9,
        finish_reason="tool_calls", tool_calls=[call]))
    engine.tool_format = _GENERIC_JSON
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "read_file"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    assert _streamed_content(events) == "Sure.\n"
    assert len([e for e in events if e["choices"][0]["delta"].get("tool_calls")]) == 1


def test_an_engine_crash_mid_stream_emits_an_error_event_and_closes():
    engine = FakeEngine(raises=RuntimeError("io_uring read failed"))
    r = _client(engine).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert "io_uring read failed" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_a_queue_position_comment_precedes_the_first_event():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert r.text.startswith(": vaayu queue")


# ── the resolved session id is echoed, so a dropped one is visible ────────────
# A misspelled or unsupported session field is silently ignored by pydantic
# (extra="ignore", which off-the-shelf clients need), so a client could pin
# every turn to nothing and never know. The header is the only outside tell.

def test_the_resolved_session_id_comes_back_on_a_header():
    r = _client(FakeEngine()).post("/v1/chat/completions",
                                   json={**BODY, "session_id": "agent-7"})
    assert r.headers["x-vaayu-session-id"] == "agent-7"


def test_the_header_reports_the_user_fallback_that_was_actually_used():
    r = _client(FakeEngine()).post("/v1/chat/completions",
                                   json={**BODY, "user": "agent-7"})
    assert r.headers["x-vaayu-session-id"] == "agent-7"


def test_no_usable_session_id_is_reported_as_auto_not_omitted():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=BODY)
    assert r.headers["x-vaayu-session-id"] == "(auto)", (
        "a missing header is indistinguishable from an old server; say (auto) "
        "so a client that meant to pin a session can see that it did not")


def test_a_misspelled_session_field_shows_up_as_auto():
    r = _client(FakeEngine()).post("/v1/chat/completions",
                                   json={**BODY, "sessionId": "agent-7"})
    assert r.status_code == 200, "extra fields stay ignored, not 422"
    assert r.headers["x-vaayu-session-id"] == "(auto)"


def test_the_streaming_path_carries_the_same_header():
    with _client(FakeEngine()).stream(
            "POST", "/v1/chat/completions",
            json={**BODY, "stream": True, "user": "agent-7"}) as r:
        assert r.headers["x-vaayu-session-id"] == "agent-7"
        r.read()


def test_the_session_is_echoed_in_the_body_as_well_as_the_header():
    # Was "the body stays OpenAI-shaped, the echo lives only in the header".
    # Superseded: the session is now reported in both, so a client that never
    # reads headers can still confirm which session it was served under. The
    # extra key is additive -- OpenAI clients ignore unknown response fields.
    r = _client(FakeEngine()).post("/v1/chat/completions",
                                   json={**BODY, "user": "agent-7"})
    assert r.json()["session_id"] == "agent-7"
    assert r.headers["x-vaayu-session-id"] == "agent-7"


# ── B8: the streaming path must map request errors to the same statuses ──────
#
# `prepare()` used to run INSIDE `_sse`, i.e. after the StreamingResponse was
# returned, so every one of these arrived as a 200 carrying an SSE event typed
# "internal_error". DEFAULT_STREAM is True, so that is what a default client saw
# for a prompt that is simply too long.

def test_an_oversized_prompt_is_a_400_when_streaming_too():
    engine = FakeEngine(raises=ContextTooLong("prompt is 9000 tokens but the window is 8192"))
    r = _client(engine).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 400, "a too-long prompt is the client's error on both paths"
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "9000" in r.json()["error"]["message"]


def test_streamed_tools_against_a_template_without_tool_support_is_a_400():
    engine = FakeEngine(raises=TemplateLacksToolSupport("no tool support"))
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "f"}}])
    r = _client(engine).post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_a_streamed_request_with_no_chat_template_is_a_400():
    engine = FakeEngine(raises=NoChatTemplate("this tokenizer has no chat template"))
    r = _client(engine).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 400


def test_a_queue_full_losing_the_admission_race_is_an_honest_sse_error_type():
    # The route's eager depth check catches the ordinary full queue with a real
    # 503. This is the race it cannot catch: depth was under the limit at the
    # check and slot() raised anyway. Headers are long gone by then, so the
    # status cannot be fixed -- but the event type can be honest.
    engine = FakeEngine(raises=QueueFull("8 requests are already queued"))
    engine.depth = 0
    r = _client(engine).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200, "headers were already sent; only the event type is left"
    errors = [e for e in _sse_events(r.text) if "error" in e]
    assert errors, "the stream must say something went wrong"
    assert errors[0]["error"]["type"] == "server_busy", \
        "a busy engine is not an internal error"


def test_a_real_engine_crash_still_streams_as_internal_error():
    # Guard: the mapping must not swallow genuine faults into a client error.
    engine = FakeEngine(raises=RuntimeError("CUDA error"))
    r = _client(engine).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200
    errors = [e for e in _sse_events(r.text) if "error" in e]
    assert errors[0]["error"]["type"] == "internal_error"


# ── startup failure must not leave fd 1/2 in the log pipe (B10) ──────────────
#
# Subprocess, for the reason test_logs.py's capture tests are: this rewires fd 1
# and 2 for real, and against the pre-fix app.py it leaves them rewired -- which
# in-process would put pytest's own output down a pipe for the rest of the run.
# The verdict goes to a FILE rather than stdout: stdout is the thing under test,
# and the log pump interleaves its tee with anything written there.

FAILED_STARTUP_SCRIPT = """
import os, sys
sys.path.insert(0, {repo!r})

def say(msg):
    with open({verdict!r}, "w") as f:
        f.write(msg)
    sys.exit(0 if msg == "OK" else 1)

def fds():
    return set(os.listdir("/proc/self/fd"))

from fastapi.testclient import TestClient
from vaayu.server import app as app_mod

def refuse(config):
    # What _check_gpu_is_exclusive does: print the report, then SystemExit. It
    # is the FIRST thing after capture_stdio, so nothing else can have failed.
    print("[Vaayu-Server] refusing to load -- the GPU is not exclusive")
    raise SystemExit("refusing")

app_mod._check_gpu_is_exclusive = refuse

cfg = app_mod.ServerConfig(model_path="/nonexistent", log_dir={log_dir!r})
app = app_mod.create_app(cfg)        # engine=None -> the real startup path runs

before_stat, before_fds = (os.fstat(1), os.fstat(2)), fds()
try:
    # TestClient re-raises a lifespan SystemExit as CancelledError out of the
    # portal task group, so catch broadly: the exception TYPE is not what this
    # test is about, the fd table is.
    with TestClient(app):
        pass
except BaseException:
    pass
else:
    say("FAIL: the startup did not fail")

after_stat, after_fds = (os.fstat(1), os.fstat(2)), fds()
for fd, (b, a) in enumerate(zip(before_stat, after_stat), start=1):
    if (b.st_dev, b.st_ino) != (a.st_dev, a.st_ino):
        say("FAIL: fd %d is still in the log pipe after a failed startup" % fd)
if after_fds - before_fds:
    say("FAIL: leaked fds %s" % sorted(after_fds - before_fds))
say("OK")
"""


def test_a_failed_startup_restores_fd_1_and_2(tmp_path):
    # The pre-yield section -- the GPU audit, the load, EngineHandle -- sits
    # outside any try, so a failure there skips restore_stdio() entirely: fds 1
    # and 2 stay in the pipe, saved_out/saved_err leak, and whatever uvicorn
    # prints next reaches the console only through a daemon pump thread the
    # interpreter is about to kill.
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    verdict = tmp_path / "verdict.txt"
    out = subprocess.run(
        [_sys.executable, "-c", FAILED_STARTUP_SCRIPT.format(
            repo=str(_Path(__file__).resolve().parents[1]),
            log_dir=str(tmp_path), verdict=str(verdict))],
        capture_output=True, text=True, timeout=120)
    assert verdict.exists(), f"the probe died before deciding:\n{out.stdout}\n{out.stderr}"
    assert verdict.read_text() == "OK", verdict.read_text()


# ── the stream emits the prose that FOLLOWS a real tool call (B15) ───────────


def _content_and_call_order(events):
    """The delta kinds in order, so a prose chunk emitted AFTER the tool_calls
    chunk is distinguishable from one emitted before it."""
    out = []
    for e in events:
        d = e["choices"][0]["delta"]
        if d.get("content"):
            out.append("content")
        if d.get("tool_calls"):
            out.append("tool_calls")
    return out


def test_prose_after_a_marked_tool_call_is_streamed_not_dropped():
    # B15. Once _sse starts buffering at the open marker it buffers to the end,
    # and the buffer is discarded whole when the final event carries a call --
    # taking anything the model wrote AFTER the call with it. The non-streaming
    # route returns that prose, so the two routes disagreed about the same
    # generation.
    call = ToolCall(id="call_z", function=ToolCallFunction(name="f", arguments='{}'))
    engine = FakeEngine(
        deltas=['Sure. ', '<tool_call>{"name":"f","arguments":{}}</tool_call>',
                ' trailing prose'],
        result=GenerationResult(text="Sure.  trailing prose", prompt_tokens=3,
                                completion_tokens=9, finish_reason="tool_calls",
                                tool_calls=[call]))
    engine.tool_format = _HERMES
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "f"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    assert _streamed_content(events) == "Sure.  trailing prose"
    assert _content_and_call_order(events) == ["content", "content", "tool_calls"], (
        "the trailing prose must arrive before the tool_calls chunk")


def test_prose_after_a_fenced_tool_call_is_streamed_not_dropped():
    call = ToolCall(id="call_w", function=ToolCallFunction(name="read_file", arguments='{}'))
    txt = 'Sure.\n```json\n{"name": "read_file", "arguments": {}}\n```\nAfterwards, done.'
    engine = FakeEngine(deltas=[txt], result=GenerationResult(
        text="Sure.\n\nAfterwards, done.", prompt_tokens=3, completion_tokens=9,
        finish_reason="tool_calls", tool_calls=[call]))
    engine.tool_format = _GENERIC_JSON
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "read_file"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    assert _streamed_content(events) == "Sure.\n\nAfterwards, done."


def test_a_call_with_no_prose_before_it_still_streams_the_prose_after_it():
    # Nothing was emitted before the marker, so the whole cleaned text is the
    # remainder -- and it must be sent exactly once.
    call = ToolCall(id="call_v", function=ToolCallFunction(name="f", arguments='{}'))
    engine = FakeEngine(
        deltas=['<tool_call>{"name":"f","arguments":{}}</tool_call>', ' tail'],
        result=GenerationResult(text="tail", prompt_tokens=3, completion_tokens=9,
                                finish_reason="tool_calls", tool_calls=[call]))
    engine.tool_format = _HERMES
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "f"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    assert _streamed_content(events) == "tail"


def test_leading_whitespace_before_a_call_does_not_misalign_the_remainder():
    # The parser strips the JOINED string, so the prefix the stream already sent
    # is not a literal prefix of final.text. A naive final.text[len(sent):] cuts
    # in the wrong place; this is the case that catches it.
    call = ToolCall(id="call_u", function=ToolCallFunction(name="f", arguments='{}'))
    engine = FakeEngine(
        deltas=['  \nHi\n', '<tool_call>{"name":"f","arguments":{}}</tool_call>',
                '\nbye'],
        result=GenerationResult(text="Hi\n\nbye", prompt_tokens=3,
                                completion_tokens=9, finish_reason="tool_calls",
                                tool_calls=[call]))
    engine.tool_format = _HERMES
    body = dict(BODY, stream=True,
                tools=[{"type": "function", "function": {"name": "f"}}])
    events = _sse_events(_client(engine).post("/v1/chat/completions", json=body).text)
    streamed = _streamed_content(events)
    assert streamed.strip() == "Hi\n\nbye"
    assert streamed.count("bye") == 1, "the remainder must not be sent twice"


def test_a_started_server_writes_env_json_into_its_run_directory(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from vaayu.server.app import ServerConfig, create_app
    from .test_server_engine import FakeModel
    from vaayu.server.engine import EngineHandle

    cfg = ServerConfig(model_path="/x", log_dir=str(tmp_path), capture_stdio=False)
    app = create_app(cfg, engine=EngineHandle(FakeModel()))
    with TestClient(app):
        pass
    runs = [p for p in tmp_path.iterdir() if p.name.startswith("run-")]
    assert len(runs) == 1
    data = json.loads((runs[0] / "env.json").read_text())
    assert "model" in data and "budgets" in data and "collect_errors" in data
    assert data["config"]["model_path"]["name"] == "x"
