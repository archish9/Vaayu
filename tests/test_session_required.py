"""`session_id` is mandatory, and every response says which one was used.

Why mandatory: without an id the prefix pool keys on a hash of the first 64
tokens (vaayu_base._auto_session_id), so two agents sharing a system prompt
collide on ONE entry and evict each other -- and concurrent calls on one id
mutate the same cache object (vaayu_base.py:4185). An engine built for parallel
fan-out cannot leave that to chance.

Why it is still satisfiable by an OpenAI-native client: `user` counts. That is
the whole reason the fallback exists.
"""
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app

from .test_server_app import FakeEngine

BODY = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}


def _client(engine=None, **cfg_over):
    cfg = ServerConfig(model_path="/ckpt", capture_stdio=False, **cfg_over)
    app = create_app(cfg, engine=engine or FakeEngine())
    return TestClient(app)


# ── the requirement ──────────────────────────────────────────────────────────

def test_requiring_a_session_is_the_default():
    assert ServerConfig(model_path="/x").require_session_id is True


def test_a_request_with_no_session_is_refused():
    r = _client().post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400
    body = r.json()["error"]["message"]
    assert "session_id" in body
    assert "user" in body, "name the OpenAI-native alternative, not just ours"


def test_the_refusal_explains_why_rather_than_just_saying_no():
    r = _client().post("/v1/chat/completions", json=BODY)
    msg = r.json()["error"]["message"].lower()
    assert "prefix" in msg or "cache" in msg or "collide" in msg


def test_an_explicit_session_id_is_accepted():
    r = _client().post("/v1/chat/completions", json={**BODY, "session_id": "agent-7"})
    assert r.status_code == 200


def test_the_openai_user_field_satisfies_the_requirement():
    r = _client().post("/v1/chat/completions", json={**BODY, "user": "agent-7"})
    assert r.status_code == 200, "an OpenAI-native client must not be locked out"


def test_an_empty_session_id_does_not_count():
    r = _client().post("/v1/chat/completions", json={**BODY, "session_id": ""})
    assert r.status_code == 400


def test_streaming_is_refused_the_same_way():
    r = _client().post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 400, "the check must precede the SSE body, not fail inside it"


def test_the_requirement_can_be_relaxed():
    r = _client(require_session_id=False).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200


# ── the echo ─────────────────────────────────────────────────────────────────

def test_the_response_body_carries_the_session_id():
    body = _client().post("/v1/chat/completions",
                          json={**BODY, "session_id": "agent-7",
                                "stream": False}).json()
    assert body["session_id"] == "agent-7"


def test_the_body_reports_the_user_fallback_that_was_actually_used():
    body = _client().post("/v1/chat/completions",
                          json={**BODY, "user": "agent-7", "stream": False}).json()
    assert body["session_id"] == "agent-7"


def test_every_stream_chunk_identifies_its_session():
    with _client().stream("POST", "/v1/chat/completions",
                          json={**BODY, "stream": True, "session_id": "agent-7"}) as r:
        chunks = [ln for ln in r.iter_lines() if ln.startswith("data: ") and "[DONE]" not in ln]
    import json
    assert chunks, "premise: the stream produced chunks"
    for raw in chunks:
        assert json.loads(raw[6:])["session_id"] == "agent-7"


def test_the_header_still_agrees_with_the_body():
    r = _client().post("/v1/chat/completions",
                       json={**BODY, "session_id": "agent-7", "stream": False})
    assert r.headers["x-vaayu-session-id"] == r.json()["session_id"] == "agent-7"
