"""On-device gate: the server against a real checkpoint.

    SERVER_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
      systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
      ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s

`-s` is REQUIRED, and this module refuses to run without it: the assertions
below read `server.log`, which is written by `capture_stdio`'s fd-level pump,
and pytest's capture keeps Python `print()` off fd 1 (see the guard fixture).

Point SERVER_CKPT at an EXL3 MoE checkpoint to exercise that path.
Skipped entirely unless SERVER_CKPT is set -- this must never run as part of
the default suite; it needs a real checkpoint and several minutes.

ONE model load for the whole file: freed pinned host blocks are cached rather
than returned, and a second load in the same process is OOM-killed under a 22 GB
cgroup cap (todo.md, "Never load a model twice in one pytest process").
"""
import json
import os

import pytest

from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app

CKPT = os.environ.get("SERVER_CKPT")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="on-device gate: set SERVER_CKPT to a checkpoint directory to run it")

PROMPT = "Write a Python function that reverses a string. Reply with code only."


_CAPTURE_REFUSAL = "this gate is blind under pytest's capture"


def _capture_hides_python_prints(mode: str) -> bool:
    """True when pytest's `--capture=<mode>` keeps `print()` off fd 1.

    `capture_stdio` (vaayu/server/logs.py) tees fd 1 into server.log, which is
    what every server.log assertion in this module reads. Pytest rebinds
    sys.stdout to its own object, so a Python print never reaches the pump --
    while vaayu_rust's native fd-1 writes still tee, which is why the log
    looks written rather than empty and the failure reads as a byte-counter
    defect (G1, 2026-09-05).

    Measured over all four modes, printing through a live pump: only `no`
    (i.e. `-s`) tees a print. `fd`, `sys` and `tee-sys` all hide it --
    `tee-sys` included, despite the name. Do not widen this on the name alone.
    """
    return mode != "no"


@pytest.fixture(scope="module", autouse=True)
def uncaptured_stdout(pytestconfig):
    """FAIL, not skip, when run without `-s`.

    A skipped gate is a gate nobody notices is not running -- the failure mode
    `7d806e7` already caused once by deleting the gate scripts outright. This
    fails the whole module with one message instead, before any checkpoint is
    loaded (the `client` fixture depends on it explicitly, so the ordering is
    not left to autouse resolution).
    """
    mode = pytestconfig.getoption("capture")
    if _capture_hides_python_prints(mode):
        pytest.fail(
            f"{_CAPTURE_REFUSAL} (--capture={mode}): pytest rebinds sys.stdout, "
            f"so server.log gets no [VAAYU-IO]/[Vaayu-*] line and the log-reading "
            f"assertions here fail for a reason that has nothing to do with the "
            f"server. re-run with -s.", pytrace=False)


@pytest.fixture(scope="module")
def client(uncaptured_stdout):
    """ONE load for the whole module.

    Never load a model twice in one pytest process: freed pinned host blocks
    are cached rather than returned to the OS, and the second load is
    OOM-killed (exit 137) under a 22 GB cgroup cap.

    delete_original=False because the server's default is True and running
    this gate against a not-yet-split checkpoint DELETED its 2.3 GB source
    file (measured 2026-08-23 on Llama-3.2-1B); the loader then stopped
    recognising the directory as a local checkpoint and fell through to a
    HuggingFace download. A gate may not consume the thing it is testing.
    """
    # SERVER_DECODE_TOP_K=auto (or an int) runs this gate with the decode
    # expert budget on; unset keeps the server default (None).
    _dtk = os.environ.get("SERVER_DECODE_TOP_K")
    # SERVER_USE_PROMPT_LOOKUP=1 opts speculation in (row 41: a hybrid model
    # must load with it REFUSED and still pass); unset keeps "auto".
    _pld = os.environ.get("SERVER_USE_PROMPT_LOOKUP")
    cfg = ServerConfig(model_path=CKPT, served_model_name="vaayu-smoke",
                       delete_original=False,
                       decode_top_k=(None if not _dtk
                                     else "auto" if _dtk == "auto" else int(_dtk)),
                       use_prompt_lookup=("auto" if not _pld else _pld == "1"))
    with TestClient(create_app(cfg)) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    print(json.dumps(h, indent=2))
    assert h["status"] == "ready"
    assert "decode_top_k" in h, "the /health decode_top_k field is missing"
    # The derived window, NOT a literal: only that it is a real number and
    # bigger than the CPU fallback this server warns about.
    assert h["max_seq_len"] > 512, f"max_seq_len={h['max_seq_len']}"


# REQUIRE_SESSION_ID has been on since the session-id commit, and every
# request without one is a 400. This gate sent none, so from that commit
# it failed at the first call -- and its oversized-prompt check went on
# "passing" on the wrong 400. The id also keys the prefix cache, which
# test_prefix_reuse below asserts on.
BODY = {"model": "vaayu-smoke", "session_id": "smoke",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 48, "temperature": 0.0}


def test_non_stream(client):
    # ── 1. non-streaming ──
    # stream=False must be EXPLICIT: DEFAULT_STREAM is True, so a body that
    # omits the key gets SSE back and this step's r.json() fails on it --
    # which is what this gate did from the day that default landed.
    r = client.post("/v1/chat/completions", json=dict(BODY, stream=False))
    assert r.status_code == 200, r.text[:200]
    b = r.json()
    text = b["choices"][0]["message"].get("content") or ""
    print(f"--- non-stream ({b['usage']['completion_tokens']} tokens, "
          f"finish_reason={b['choices'][0]['finish_reason']}) ---\n{text}\n---")
    assert b["usage"]["completion_tokens"] > 0
    assert b["choices"][0]["finish_reason"] in {"stop", "length", "tool_calls"}


def test_stream(client):
    # ── 2. streaming, and prefix reuse on the second turn ──
    # Ask for usage: the stream's completion_tokens and finish_reason were
    # unchecked here while B2 was live -- the streamed path is the DEFAULT
    # one, and this gate watched only that text came out of it.
    with client.stream("POST", "/v1/chat/completions",
                       json=dict(BODY, stream=True,
                                 stream_options={"include_usage": True})) as s:
        assert s.status_code == 200
        chunks, saw_done = [], False
        stream_finish, stream_usage = None, None
        for line in s.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                saw_done = True
                continue
            ev = json.loads(payload)
            if "error" in ev:
                print(f"stream error event: {ev['error']}")
                continue
            if ev.get("usage"):
                stream_usage = ev["usage"]
            if ev["choices"][0].get("finish_reason"):
                stream_finish = ev["choices"][0]["finish_reason"]
            piece = ev["choices"][0]["delta"].get("content")
            if piece:
                chunks.append(piece)
    streamed = "".join(chunks)
    print(f"--- streamed ({len(chunks)} deltas, finish_reason={stream_finish}, "
          f"usage={stream_usage}) ---\n{streamed}\n---")
    assert saw_done, "stream did not terminate with [DONE]"
    assert len(streamed) > 0, "stream produced no text"
    assert stream_finish in {"stop", "length", "tool_calls"}, str(stream_finish)
    # B2: this count came from re-tokenizing the decoded text, so it did not
    # match the ids the model generated. It is now the real id count, and
    # `cap` is the only bound it may not exceed.
    assert (stream_usage is not None
            and 0 < stream_usage["completion_tokens"] <= BODY["max_tokens"]), \
        str(stream_usage)
    # A run that used every admissible token is truncated, and must say so.
    if stream_usage and stream_usage["completion_tokens"] == BODY["max_tokens"]:
        assert stream_finish == "length", str(stream_finish)


def test_oversized_prompt_refused(client):
    # ── 3. an oversized prompt is refused, not truncated ──
    h = client.get("/health").json()
    huge = {"model": "vaayu-smoke", "session_id": "smoke-huge",
            "messages": [{"role": "user", "content": "x " * (h["max_seq_len"] + 1000)}]}
    # stream=False for the same reason test_non_stream needs it.
    r = client.post("/v1/chat/completions", json=dict(huge, stream=False))
    assert r.status_code == 400, r.text[:200]
    # The STREAMING form was B8: prepare() ran inside the generator, so the
    # response was already a 200 by the time ContextTooLong was raised and
    # it arrived as an "internal_error" SSE event -- on the DEFAULT path,
    # since DEFAULT_STREAM is True. This is the only gate that would have
    # caught it, so it asserts the streamed refusal too (fixed 2026-08-28).
    r = client.post("/v1/chat/completions", json=dict(huge, stream=True))
    assert r.status_code == 400, r.text[:200]


def test_prefix_reuse(client):
    # ── 4. the prefix pool was exercised by the two identical turns ──
    stats = client.get("/health").json()["prefix_cache"]
    print(f"prefix cache: {stats}")
    # Assert the PREMISE, not just the field: "stats is non-empty" passes
    # even when the pool is disabled, which is how the first run of this
    # gate reported green while prefix reuse was off entirely.
    if stats == "disabled":
        # Expected only with an explicit KV_CACHE_MODE="fp16": the bare
        # DynamicCache has no to_cpu/to_device/resident_bytes. Since row 3
        # (2026-09-11) "auto" enables the pool on every loader.
        print("  (pool disabled -- expected only with KV_CACHE_MODE='fp16', todo row 3)")
    else:
        assert "hits=0" not in stats, stats


def _run_dir():
    """The directory this process's server wrote to."""
    from pathlib import Path
    latest = Path("logs") / "latest"
    assert latest.is_symlink(), "logs/latest was not created"
    return latest.resolve()


def test_env_json_is_written_and_clean(client):
    data = json.loads((_run_dir() / "env.json").read_text())
    assert data["collect_errors"] == [], data["collect_errors"]
    assert data["model"]["forward"], "the forward name was not tagged at dispatch"


def test_requests_jsonl_has_one_row_per_request(client):
    """Every request this module issued -- successes and the two oversized
    refusals alike. The refusals are the ones the engine cannot see."""
    rows = [json.loads(l) for l in
            (_run_dir() / "requests.jsonl").read_text().splitlines()]
    assert len(rows) >= 4, f"only {len(rows)} rows"
    assert any(r["finish"] == "refused" for r in rows), \
        "the oversized-prompt refusals wrote no row"
    assert all("decode" in r for r in rows)


def test_the_token_summary_reports_nvme_bytes_consistent_with_pinning(client):
    """The assertion that catches T5 silently reading zero on a streamed dense
    model: expert bytes are counted on the MoE path only, and dense reads
    nothing through it. But "nonzero" is not a universal expectation -- a
    checkpoint small enough to fully pin every transformer layer in VRAM at
    load (Llama-3.2-1B on this box: 3.00 GB against an 11.06 GiB pinning
    budget) legitimately streams NOTHING from NVMe on any token, and a gate
    that forced nonzero there would be asserting something false (found
    2026-09-04, task 10: every gb_tok was 0.00 on that checkpoint, and it was
    the correct report, not a broken counter).

    So this derives its expectation from what THIS run's own startup output
    says it did, rather than hardcoding a per-checkpoint expectation (CLAUDE.md
    5): the "[Vaayu-Pin] N fully-pinned (X transformer + Y shared), Z
    non-expert-pinned transformer" summary line, cross-checked against
    env.json's derived `num_layers`, says whether every transformer layer was
    pinned (no streaming possible, ever) or not (some layers -- MoE experts,
    MoE non-expert tensors, or plain dense layers past the pinning budget --
    stream on every token, so the counter must show it).
    """
    import re
    text = (_run_dir() / "server.log").read_text(errors="replace")
    values = [float(m) for m in re.findall(r"gb_tok=([0-9.]+)", text)]
    assert values, "no token summary carried a gb_tok= field"

    pin_match = re.search(
        r"\[Vaayu-Pin\] \d+ fully-pinned \((\d+) transformer \+ \d+ shared\), "
        r"(\d+) non-expert-pinned transformer", text)
    assert pin_match, "no [Vaayu-Pin] summary line -- can't derive the expectation"
    pinned_transformer = int(pin_match.group(1))
    non_expert_pinned_transformer = int(pin_match.group(2))

    num_layers = json.loads((_run_dir() / "env.json").read_text())["model"]["num_layers"]
    fully_pinned = (pinned_transformer == num_layers
                    and non_expert_pinned_transformer == 0)

    if fully_pinned:
        print(f"[gate] every one of {num_layers} transformer layers is "
              f"VRAM-pinned -- this run cannot stream, so gb_tok must be zero.")
        assert max(values) == 0.0, \
            (f"every transformer layer is pinned (no NVMe read is possible "
             f"during a token), but gb_tok was nonzero: {values[:8]}")
    else:
        print(f"[gate] only {pinned_transformer}/{num_layers} transformer "
              f"layers are fully pinned ({non_expert_pinned_transformer} "
              f"non-expert-pinned) -- this run streams, so gb_tok must be "
              f"nonzero somewhere.")
        assert max(values) > 0.0, \
            (f"this run does not fully pin its transformer layers, so it "
             f"must read from NVMe on some token, but every gb_tok was "
             f"zero: {values[:8]}")


def test_no_absolute_paths_leaked_into_the_bundle(client):
    """`env.json` ONLY. It is the one artifact that promises scrubbed paths.
    `requests.jsonl` writes raw and its `error` block embeds a full
    `traceback.format_exception()`, whose frames name absolute repo paths --
    asserting on it here passed only because no request on any checkpoint
    happened to error, and would have failed the moment one did. README says
    the file is unfiltered; this gate must not contradict it."""
    text = (_run_dir() / "env.json").read_text(errors="replace")
    assert "/home/" not in text, "env.json leaked an absolute path"
