"""A mailed `logs/run-*/` folder carries no prompt and no completion text.

`documentation/server.md` tells a user to zip `logs/run-<timestamp>/` and send
it, and the whole wire-telemetry plan (2026-11-08) rests on that being safe to
ask: run E was diagnosed from a folder plus the reporter's own bug report, and
the fields added by that plan exist so the folder alone is enough next time. If
the folder carried the request body, "mail me your logs" would stop being a
reasonable thing to ask a stranger.

That property held by luck of what this server happens to log. This makes it
hold on purpose. It is a GUARD, not a red-first TDD test: it is expected to
pass the first time it is run, and a failure is a real content leak.

Three canaries, three different leak surfaces:

  * the PROMPT text, which reaches `prepare`/`render_prompt` and every
    traceback raised under them;
  * the COMPLETION text, which reaches the response body, the SSE chunks, the
    tool-call parser and the `finish` resolution;
  * the raw USER-AGENT's private tail and an `x-forwarded-for` ADDRESS, which
    reach the route's headers. `vaayu/server/wire.py:client_shape` derives a
    coarse `{family, version, forwarded}` from them and discards the input;
    nothing else may write them down. This path had no privacy test of its own
    before this file (the Task 6 review found the briefed one vacuous).

**Run in a subprocess, deliberately.** `capture_stdio` rewires fds 1 and 2
process-wide, so `tests/test_logs.py` already drives it this way -- and
`server.log` is the file where an arbitrary `print()` would leak, so a test
that skipped it would miss the widest surface. The child builds the app the way
the real `__main__` does (`capture_stdio=True`, a run directory created by the
lifespan), so the folder it leaves behind is the artifact a user would mail:
`env.json`, `server.log`, `requests.jsonl` and `sessions/*.log`.

**Known limitation, not fixed here.** This cannot catch uvicorn's own access
log, because no real uvicorn serves under pytest: `vaayu/server/__main__.py`
calls `uvicorn.run(..., log_level="info")` with no `access_log=False`, uvicorn's
access line carries the peer `client_addr`, and `CAPTURE_STDIO = True` tees it
into `server.log`. That is a real pre-existing hole in the same invariant and it
is recorded as a `[bug]` row in `todo.md`; it is the user's call, because access
logs are genuinely useful.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PROMPT_CANARY = "zqxPROMPTCANARY"
REPLY_CANARY = "zqxREPLYCANARY"
UA_TAIL = "alice-laptop.local"          # a hostname a real user-agent can carry
PEER_IP = "198.51.100.77"               # TEST-NET-2, never a real peer
CANARIES = (PROMPT_CANARY, REPLY_CANARY, UA_TAIL, PEER_IP)

# The child drives the REAL route with the REAL row writers behind a fake
# model, then prints the run directory once the lifespan has restored stdio.
# It asserts the canaries actually reached the wire first: a scan of a folder
# produced by a request whose reply never contained the canary proves nothing.
_DRIVER = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
root, prompt_canary, reply_canary, ua_tail, peer_ip = sys.argv[2:7]

from fastapi.testclient import TestClient

import vaayu.auto_model as am
import vaayu.gpu_audit as ga
from tests.test_reqlog_engine import M
from vaayu.server.app import ServerConfig, create_app


def _render(messages, **kw):
    """The fake template. Returns TEXT for `tokenize=False`, ids otherwise.

    The text leg matters: `render_prompt` diffs two text renders through
    `generation_tail` whenever thinking is off (the server's default), so
    without it the prompt canary would never be rendered at all and the shared
    fake raises `AttributeError` on a list.
    """
    text = "".join(str(m.get("content", "")) for m in messages)
    return text if kw.get("tokenize", True) is False else [1] * 10


class CanaryModel(M):
    """M, but the prompt is really rendered and the completion carries a canary."""
    def __init__(self):
        super().__init__()
        self.tokenizer.apply_chat_template = (
            lambda messages, **kw: _render(messages, **kw))
        self.tokenizer.decode = lambda ids, **k: " %s and some tail" % reply_canary


class FakeAutoModel:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return CanaryModel()


am.AutoModel = FakeAutoModel
ga.audit = lambda **kw: []

cfg = ServerConfig(model_path="/ckpt", log_dir=root, capture_stdio=True,
                   default_stream=False)
app = create_app(cfg)

body = {"model": "m", "session_id": "canary", "max_tokens": 16,
        "messages": [{"role": "user", "content": "please echo " + prompt_canary}]}
headers = {"user-agent": "curl/8.5.0 (%s)" % ua_tail, "x-forwarded-for": peer_ip}

with TestClient(app) as c:
    run_dir = app.state.run_dir
    plain = c.post("/v1/chat/completions", json=dict(body, stream=False),
                   headers=headers)
    streamed = c.post("/v1/chat/completions", json=dict(body, stream=True),
                      headers=headers)
    # A route-level refusal too: `_refusal_row` is a SECOND row writer with its
    # own key set, and it is the only row some requests ever get.
    refused = c.post("/v1/chat/completions",
                     json={k: v for k, v in body.items() if k != "session_id"},
                     headers=headers)

assert plain.status_code == 200, plain.text
assert streamed.status_code == 200, streamed.text
assert refused.status_code == 400, refused.text
# Anti-vacuity: the canary must have been ON the wire in both directions.
assert reply_canary in plain.json()["choices"][0]["message"]["content"]
assert reply_canary in streamed.text
print(json.dumps({"run_dir": str(run_dir)}))
'''


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("logroot")
    out = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(REPO), str(root),
         PROMPT_CANARY, REPLY_CANARY, UA_TAIL, PEER_IP],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert out.returncode == 0, (
        f"the driver failed, so nothing was scanned:\n{out.stdout}\n{out.stderr}")
    # The driver's own stdout is teed into server.log until the lifespan
    # restores fd 1, so the run directory is the LAST line, not the only one.
    reported = json.loads(out.stdout.strip().splitlines()[-1])["run_dir"]
    return Path(reported)


def _files(run_dir):
    return sorted(p for p in run_dir.rglob("*") if p.is_file())


def test_the_run_directory_holds_the_whole_support_artifact(run_dir):
    """Name what was scanned, so the scan below cannot pass by being empty.

    Reddened by dropping `envinfo.write` or the `capture_stdio` call in
    `app.py`'s lifespan, or `session_event`'s write in `logs.py` -- each of
    which would silently shrink what the next test looks at.
    """
    names = {str(p.relative_to(run_dir)) for p in _files(run_dir)}
    assert "env.json" in names
    assert "server.log" in names
    assert "requests.jsonl" in names
    assert any(n.startswith("sessions/") for n in names), names
    assert all(p.stat().st_size > 0 for p in _files(run_dir)), names


def test_the_rows_prove_all_three_request_paths_ran(run_dir):
    """The three rows the scan depends on, and the derived client on each.

    `client` being the parsed shape is what makes the UA/IP half of the scan a
    result rather than a coincidence: it proves the route DID read both headers
    (`app.py`'s `client_shape(request.headers.get("user-agent"), forwarded=...)`
    call) and still wrote neither value down. Dropping `client=shape` from the
    `WireMeter(...)` construction, or the `forwarded=` argument, reddens it.
    """
    rows = [json.loads(line) for line
            in (run_dir / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 3, rows
    # The refusal is `stream: False` too (it is answered before `stream` is
    # resolved from a body that omitted it), so split on `finish` first.
    refused = [r for r in rows if r["finish"] == "refused"]
    served = [r for r in rows if r["finish"] != "refused"]
    streamed = [r for r in served if r["stream"]]
    plain = [r for r in served if not r["stream"]]
    assert len(streamed) == 1 and streamed[0]["wire"] is not None
    assert len(plain) == 1 and plain[0]["wire"] is None
    assert len(refused) == 1 and refused[0]["wire"] is None
    for r in rows:
        assert r["client"] == {"family": "curl", "version": "8",
                               "forwarded": True}, r


@pytest.mark.parametrize("canary", CANARIES)
def test_no_file_in_the_run_directory_contains_a_canary(run_dir, canary):
    """The invariant itself, one canary at a time so a failure names which.

    Reddened by any production line that writes request or client input into
    the folder: a `"prompt"`/`"completion"` key in either row writer's dict
    (`engine.py:_write_request_row`, `app.py:_refusal_row`), a `print()` that
    interpolates `req.messages`, or `wire.py:client_shape` returning the
    user-agent it was given instead of the family it derived.
    """
    for path in _files(run_dir):
        body = path.read_bytes().decode("utf-8", "replace")
        assert canary not in body, (
            f"{canary!r} reached {path.relative_to(run_dir)} -- a mailed log "
            f"folder must carry no prompt, completion or client input")
