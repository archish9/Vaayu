# Talking to Vaayu from Rudra — integration guide & what we fixed

**Date:** 2026-10-28 · **Audience:** the Rudra client (and anyone driving Vaayu's server over the wire).
Self-contained: you do not need the Vaayu source to use this.

Vaayu is an OpenAI-wire inference server. Rudra talks to it exactly as it would talk to any
OpenAI `/v1/chat/completions` endpoint. This doc covers (1) the two connection failures you hit and
what actually caused them, (2) the fix we shipped, and (3) the rules for talking to Vaayu so the
remaining failure modes do not bite.

---

## 1. What went wrong in your two runs (and what did NOT)

Both attempts died in the planner's first streaming call. We debugged run
`logs/run-2026-10-28T10-46-53Z` (attempt 2, MiniMax-M2.7) line by line.

**Vaayu was not broken.** The server streamed **649 tokens steadily** — every token took 0.4–0.8 s,
disk wait `0.0 s`, no stall, no 500, no traceback — right up to the moment the connection died at
**354 s**, mid-decode. The server then logged `finish=cancelled`: *the client (or the tunnel) went
away*, and Starlette tore the stream down. That is a **downstream disconnect**, not a server fault.

Two real root causes, both outside Vaayu's code:

1. **The model is slow for this job.** MiniMax-M2.7 is a 144 GB disk-streamed MoE; it decodes at
   **~1.83 tok/s** on this box, with a **26 s** cold time-to-first-token. A planner response needs
   far more than 649 tokens, so the call runs for many minutes.
2. **The transport cannot hold a multi-minute stream.** An ngrok *free* tunnel (and most proxies /
   load balancers) drop a long-lived streaming connection — on inactivity, or on total duration.

One detail that looks like a bug but is not: the wire reports `finish_reason="length"`, while
Vaayu's own `requests.jsonl` logs `finish="cancelled"` for the same request. That is deliberate —
the OpenAI wire vocabulary has no word for "cancelled", so a cut-off stream is reported as
`"length"`, which already means *"this answer is incomplete, do not trust it as final."* See §4.

---

## 2. What we fixed: SSE keepalive during the silent prefill window

The one real Vaayu-side gap the investigation found: between the opening `role` chunk and the first
generated token — i.e. during **prefill and any queue wait** — the stream was **completely silent**.
That is **26 s on a cold MiniMax load, and minutes on a large prompt** (prefill time grows with
prompt length). A proxy/tunnel with a 30–60 s idle timeout drops the connection *before the first
token ever arrives*.

**Fix (shipped):** the server now sends a periodic **`: keepalive`** SSE comment while a stream is
stalled, which resets the idle timer. New server knob:

| Knob | Default | Meaning |
|---|---|---|
| `SSE_KEEPALIVE_S` | `15` | Seconds of silence before a `: keepalive` comment is sent. `0` disables it. |

- It fires **only while output is stalled.** Once tokens flow (every ~0.5 s even on the slowest
  model), none are sent — zero noise on a healthy stream.
- It costs ~13 bytes per interval.
- **It does NOT save a stream dropped by a *total-duration* limit** (bytes are actively flowing
  then). That failure — which is what killed your attempt 2, mid-decode at 354 s — is addressed by
  model choice and client timeouts, not keepalive. See §3 and §5.

### What Rudra must do about it

**Your SSE parser must ignore comment lines** — any line beginning with `:`. This is standard SSE
(the EventSource spec ignores them), and most HTTP/SSE clients already do. If Rudra hand-parses the
stream, make sure a `: keepalive` line is skipped and never treated as a data frame or JSON.

---

## 3. How to connect

- **Base URL:** `http://<host>:9999` (default `127.0.0.1:9999`; set `HOST="0.0.0.0"` on the server to
  expose it on the LAN). OpenAI-compatible.
- **Endpoints:** `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.
- **Auth:** none by default (it is a local-first engine). Do not send Vaayu an API key it did not ask
  for.
- **One request at a time.** The server serves a single generation at a time; extra concurrent
  requests queue (up to `QUEUE_DEPTH`, default 8) and get a **503 with `Retry-After`** past that.
  Concurrent *fan-out* (N>1 batching) is **not** wired into the default server, so firing many
  parallel calls does not run them in parallel — they serialize. `/v1/models` and `/health` advertise
  a fan-out hint (`helps`, `recommended_client_concurrency`); honor it, and when in doubt keep Rudra's
  calls **sequential**.

### Required: `session_id` on every request

The server runs with `REQUIRE_SESSION_ID = True`. **Every request must carry a session identifier**,
or it is refused with **400**. Send either:

```jsonc
{ "model": "vaayu", "session_id": "rudra-<task-or-conversation-id>", "messages": [...] }
// or OpenAI's field, which Vaayu accepts as a session:
{ "model": "vaayu", "user": "rudra-<task-or-conversation-id>", "messages": [...] }
```

Use a **stable id per conversation/task**, not a random id per call. This is not red tape — it drives
the prefix cache (§6). Two agents that share a system prompt but send *no* id collide on one cache
entry and evict each other, collapsing reuse to zero.

### Streaming

- Send `"stream": true`. (The server defaults to streaming even if you omit it, but be explicit.)
  Non-streaming sends **nothing until the last token** — on a slow model that guarantees a timeout.
- Read the SSE stream incrementally; ignore `:` comment lines (§2).

---

## 4. Reading `finish_reason` correctly

| `finish_reason` | Meaning |
|---|---|
| `stop` | Normal completion (EOS or a stop string). Trust the answer. |
| `tool_calls` | The turn is a tool call; `delta.tool_calls` / message `tool_calls` carry it. |
| `length` | **Incomplete — do NOT trust as final.** Either the token cap was hit **or** the stream was cancelled/disconnected. The wire cannot distinguish the two; the server log (`requests.jsonl` `finish` field) can. |

So treat a streamed turn that ends in `length` as truncated: retry it, raise `max_tokens`/context, or
switch to a faster model — never hand a `length` result to the next planner step as if it were
complete.

A **mid-stream disconnect cannot be resumed** — there is no replay. Restart the call.

---

## 5. Timeouts and transport (this is what actually dropped you)

- **Set a generous per-read (inactivity) timeout**, comfortably **greater than `SSE_KEEPALIVE_S`**
  (default 15 s) — e.g. **60 s**. With keepalive on, the server now refreshes the connection every
  15 s during prefill, so an inactivity timeout of 60 s will not fire on a long prefill.
- **Do not set a tight total/overall timeout.** A legitimate long answer on a slow model can run
  many minutes. If your HTTP client has a single overall timeout, make it very large or disable it,
  and rely on the inactivity timeout instead.
- **The tunnel is the weak link.** An ngrok *free* tunnel will drop long streams regardless of what
  Vaayu does. Options, best first:
  1. Run Rudra on the **same box / LAN** as Vaayu — no tunnel at all.
  2. Use a **durable tunnel / reverse proxy** configured for long-lived streaming (large idle and
     read timeouts, buffering OFF — Vaayu already sends `X-Accel-Buffering: no`).
  3. If you must use a free tunnel, expect drops on multi-minute generations and keep responses short.
- **Client-side handling gap you flagged (Rudra's call):** a mid-stream `RemoteProtocolError`
  (peer dropped) currently crashes with a raw traceback, while a mid-stream 500 gives a clean error
  panel. Both are "the stream died" — give the transport drop the same graceful panel.

---

## 6. Make it fast: prefix cache + model choice

- **Prefix cache (free, huge):** Vaayu reuses the KV of the shared prefix across turns — measured
  **136–377× on time-to-first-token** for the agentic pattern of resending mostly-unchanged context.
  To get it: **reuse the same `session_id`**, and **keep the stable part of the prompt first** (system
  prompt + unchanged files at the top, the turn's new content at the bottom). Changing early tokens
  throws the reuse away.
- **Model choice is the biggest lever for your workload.** MiniMax-M2.7 at ~1.83 tok/s is the wrong
  model for remote, fan-out-heavy agentic loops — a single planner call outlives most tunnels.
  Faster targets on this box:
  - **Qwen3-Next-80B-A3B (int4)** — the fastest and most perf-tuned path here.
  - **gpt-oss-20b** — VRAM-resident experts, ~40–55 ms/token at short-to-mid context.

  Either finishes a planner call in a fraction of MiniMax's wall time, which keeps the whole call
  comfortably inside any reasonable tunnel window. Pick the model on the server (`MODEL_PATH`); it is
  one model per server process.

---

## 7. Error shapes to expect

| Status / event | Cause | Rudra action |
|---|---|---|
| `400` invalid_request_error | missing `session_id`, oversized prompt, unsupported tool template | fix the request; do not retry unchanged |
| `503` server_busy (+ `Retry-After`) | queue full (`QUEUE_DEPTH` reached) | back off and retry after the header's seconds |
| `500` | real server error | surface it; safe to retry once |
| SSE `event: error` mid-stream | error after headers were already sent | treat as a failed turn; restart the call |
| connection dropped mid-stream | client/tunnel disconnect (not resumable) | restart the call; see §5 |

---

## 8. Quick checklist for Rudra

- [ ] Always send a **stable `session_id`** (or `user`) per conversation. Stable prefix on top.
- [ ] `"stream": true`; **ignore `:` comment lines** in the SSE parser.
- [ ] Per-read (inactivity) timeout **≥ 60 s** (> `SSE_KEEPALIVE_S`); no tight total timeout.
- [ ] Treat `finish_reason == "length"` as **incomplete**, not final.
- [ ] Keep calls **sequential** unless `/v1/models` says fan-out helps.
- [ ] Prefer a **local/LAN** connection or a durable tunnel over a free tunnel.
- [ ] For agentic loops, serve a **fast model** (Qwen3-Next / gpt-oss), not disk-streamed MiniMax.
- [ ] Give a transport drop the **same graceful error panel** as a 500.
