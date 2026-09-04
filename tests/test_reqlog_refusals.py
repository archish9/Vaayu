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
        yield c


@pytest.fixture
def client(run_dir, no_real_load):
    yield from _client(run_dir, require_session_id=False)


@pytest.fixture
def client_requiring_session(run_dir, no_real_load):
    yield from _client(run_dir, require_session_id=True)


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
