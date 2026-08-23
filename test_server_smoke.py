"""On-device gate: the server against a real checkpoint.

    SERVER_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
      systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
      ./venv/bin/python test_server_smoke.py

Defaults to the small dense checkpoint so a run costs seconds, not minutes.
Point SERVER_CKPT at an EXL3 MoE checkpoint to exercise that path.

ONE model load for the whole file: freed pinned host blocks are cached rather
than returned, and a second load in the same process is OOM-killed under a 22 GB
cgroup cap (todo.md, "Never load a model twice in one pytest process").
"""
import json
import os
import sys

from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app

CKPT = os.environ.get("SERVER_CKPT", "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct")
PROMPT = "Write a Python function that reverses a string. Reply with code only."

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    # delete_original=False: the server default is True, and running this gate
    # against a not-yet-split checkpoint DELETED its 2.3 GB source file (measured
    # 2026-08-23 on Llama-3.2-1B). Worse, the loader then no longer recognised
    # the directory as a local checkpoint and fell through to a HuggingFace
    # download. A gate may not consume the thing it is testing.
    cfg = ServerConfig(model_path=CKPT, served_model_name="vaayu-smoke",
                       delete_original=False)
    with TestClient(create_app(cfg)) as client:
        h = client.get("/health").json()
        print(json.dumps(h, indent=2))
        check("health is ready", h["status"] == "ready")
        # The derived window, NOT a literal: only that it is a real number and
        # bigger than the CPU fallback this server warns about.
        check("max_seq_len is derived, not the CPU fallback", h["max_seq_len"] > 512,
              f"max_seq_len={h['max_seq_len']}")

        # REQUIRE_SESSION_ID has been on since the session-id commit, and every
        # request without one is a 400. This gate sent none, so from that commit
        # it failed at the first call -- and its oversized-prompt check went on
        # "passing" on the wrong 400. The id also keys the prefix cache, which
        # step 4 below asserts on.
        body = {"model": "vaayu-smoke", "session_id": "smoke",
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 48, "temperature": 0.0}

        # ── 1. non-streaming ──
        # stream=False must be EXPLICIT: DEFAULT_STREAM is True, so a body that
        # omits the key gets SSE back and this step's r.json() fails on it --
        # which is what this gate did from the day that default landed.
        r = client.post("/v1/chat/completions", json=dict(body, stream=False))
        check("non-stream returns 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            b = r.json()
            text = b["choices"][0]["message"].get("content") or ""
            print(f"--- non-stream ({b['usage']['completion_tokens']} tokens, "
                  f"finish_reason={b['choices'][0]['finish_reason']}) ---\n{text}\n---")
            check("non-stream produced tokens", b["usage"]["completion_tokens"] > 0)
            check("finish_reason is a real value",
                  b["choices"][0]["finish_reason"] in {"stop", "length", "tool_calls"})

        # ── 2. streaming, and prefix reuse on the second turn ──
        # Ask for usage: the stream's completion_tokens and finish_reason were
        # unchecked here while B2 was live -- the streamed path is the DEFAULT
        # one, and this gate watched only that text came out of it.
        with client.stream("POST", "/v1/chat/completions",
                           json=dict(body, stream=True,
                                     stream_options={"include_usage": True})) as s:
            check("stream returns 200", s.status_code == 200)
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
        check("stream terminated with [DONE]", saw_done)
        check("stream produced text", len(streamed) > 0)
        check("stream reported a finish_reason",
              stream_finish in {"stop", "length", "tool_calls"}, str(stream_finish))
        # B2: this count came from re-tokenizing the decoded text, so it did not
        # match the ids the model generated. It is now the real id count, and
        # `cap` is the only bound it may not exceed.
        check("stream reported real completion_tokens",
              stream_usage is not None and 0 < stream_usage["completion_tokens"] <= body["max_tokens"],
              str(stream_usage))
        # A run that used every admissible token is truncated, and must say so.
        if stream_usage and stream_usage["completion_tokens"] == body["max_tokens"]:
            check("a stream that used the whole cap reports length",
                  stream_finish == "length", str(stream_finish))

        # ── 3. an oversized prompt is refused, not truncated ──
        huge = {"model": "vaayu-smoke", "session_id": "smoke-huge",
                "messages": [{"role": "user", "content": "x " * (h["max_seq_len"] + 1000)}]}
        # stream=False for the same reason step 1 needs it. The STREAMING form of
        # this refusal is Bug B8: prepare() runs inside the generator, so
        # ContextTooLong becomes a 200 carrying an "internal_error" SSE event --
        # confirmed here on 2026-08-23. Add the streaming assertion when B8 lands.
        r = client.post("/v1/chat/completions", json=dict(huge, stream=False))
        check("an oversized prompt is a 400", r.status_code == 400, r.text[:200])

        # ── 4. the prefix pool was exercised by the two identical turns ──
        stats = client.get("/health").json()["prefix_cache"]
        print(f"prefix cache: {stats}")
        # Assert the PREMISE, not just the field: "stats is non-empty" passes
        # even when the pool is disabled, which is how the first run of this
        # gate reported green while prefix reuse was off entirely.
        if stats == "disabled":
            # Expected only for non-EXL3 checkpoints -- todo OPTIONAL row 3, the
            # bare DynamicCache has no to_cpu/to_device/resident_bytes.
            print("  (pool disabled -- expected on a non-EXL3 checkpoint, todo row 3)")
        else:
            check("prefix reuse fired across the two identical turns",
                  "hits=0" not in stats, stats)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
