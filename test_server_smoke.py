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
    cfg = ServerConfig(model_path=CKPT, served_model_name="vaayu-smoke")
    with TestClient(create_app(cfg)) as client:
        h = client.get("/health").json()
        print(json.dumps(h, indent=2))
        check("health is ready", h["status"] == "ready")
        # The derived window, NOT a literal: only that it is a real number and
        # bigger than the CPU fallback this server warns about.
        check("max_seq_len is derived, not the CPU fallback", h["max_seq_len"] > 512,
              f"max_seq_len={h['max_seq_len']}")

        body = {"model": "vaayu-smoke",
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 48, "temperature": 0.0}

        # ── 1. non-streaming ──
        r = client.post("/v1/chat/completions", json=body)
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
        with client.stream("POST", "/v1/chat/completions",
                           json=dict(body, stream=True)) as s:
            check("stream returns 200", s.status_code == 200)
            chunks, saw_done = [], False
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
                piece = ev["choices"][0]["delta"].get("content")
                if piece:
                    chunks.append(piece)
        streamed = "".join(chunks)
        print(f"--- streamed ({len(chunks)} deltas) ---\n{streamed}\n---")
        check("stream terminated with [DONE]", saw_done)
        check("stream produced text", len(streamed) > 0)

        # ── 3. an oversized prompt is refused, not truncated ──
        huge = {"model": "vaayu-smoke",
                "messages": [{"role": "user", "content": "x " * (h["max_seq_len"] + 1000)}]}
        r = client.post("/v1/chat/completions", json=huge)
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
