# Design — Vaayu as an inference server (OpenAI-compatible), v1

**Date:** 2026-08-16
**Status:** approved in brainstorm, not yet planned or implemented
**Supersedes for this topic:** nothing. `exllamav3-server.py` at the repo root is a
reference sample from another engine, not a Vaayu component; it is read for its API
shape and explicitly **not** copied (see §9.1).

---

## 1. Purpose

Vaayu today generates tokens only from test scripts (`test_generate.py`,
`test_multiturn.py`, …), each of which hardcodes its own prompt and reads
`MAX_NEW_TOKENS` from the environment. This design turns Vaayu into a long-running
server that an autonomous coding agent can drive over HTTP, the way it would drive
vLLM: one model path in, OpenAI-shaped requests over the wire, real code out.

Two problems are in scope:

1. **No server surface.** No process boundary, no HTTP API, no streaming to a client,
   no tool-calling, no session concept. Everything the engine can do is reachable only
   from Python in-process.
2. **`MAX_NEW_TOKENS` truncates output.** A fixed cap of 512 cuts code mid-function.

Problem 2's proposed fix — "stream the tokens" — does **not** solve it. See §4.

## 2. Constraints this design was built against

These are measured, on this box, and recorded in `plans/settled_and_method_notes.md`.
They are the reason the design looks the way it does.

| constraint | value | source |
|---|---|---|
| context, all three big checkpoints | `max_seq_len = 8192` | todo row 9, verified on-device 4/4 |
| decode, Qwen3-235B (MoE) | 0.562 tok/s | row 11 |
| decode, MiniMax-M2.7 (MoE) | ~0.56 tok/s | row 11 |
| decode, Llama-3.3-70B (dense) | 0.155 tok/s | row 12 |
| decode, 70B + 1B drafter | 0.864 tok/s (5.57×) | row 12 |
| TTFT cold @ S=6000 | 61.5 s (Qwen) / 97.8 s (70B) | row 11 |
| TTFT on a prefix-cache hit | 136–377× better | Plan A1 / Plan I |
| concurrency | **N ≥ 2 INFEASIBLE at long context** | settled, hardware wall |
| usable VRAM | 15.47 GiB | hardware constant |

**Consequence, stated plainly:** with every existing lever switched on, the ceiling on
current checkpoints is ~0.86 tok/s at 8192 context, single stream. That is usable for an
unattended loop-engineering agent and not usable for anything interactive. No server
design changes this — it is the engine.

**Target chosen for v1:** a **model-agnostic server** (CLAUDE.md §5 requires this
regardless) whose *default* deployment is a **VRAM-resident** coder model, so the thing is
actually interactive. The streamed big-model path stays first-class and is selected by
`--model`. `vaayu_base.py:3698` already skips the disk load for a pinned layer, so a
fully-pinned model pays no streaming cost — the fast path exists.

**Resident weights budget** (derived, not a literal):

```
15.47 GiB usable
−  ~5    GiB  sdpa reserve — row 8's flash-parity floor, measured 4.79–6.20 GiB
               across the three big checkpoints (KV + activations + score buffer + logits)
= ~10.5  GiB  for weights
```

→ a **14B-class coder at 4–4.5 bpw** (≈8 GB, leaving ~7 GiB for reserve + KV → 16–32k
context at 20–40 tok/s) is the safe target. A 30B-A3B MoE at ~2.6 bpw (≈11 GB) is the
aggressive one. **Nothing of that size is on disk today** — see §10.

## 3. Architecture

New package `vaayu/server/`. Nothing is added to `vaayu_base.py`, which is already
4564 lines (CLAUDE.md §3).

| module | responsibility | pure |
|---|---|---|
| `server/app.py` | FastAPI routes, lifespan, SSE plumbing | no |
| `server/schemas.py` | pydantic request/response models (OpenAI subset + Vaayu extras) | yes |
| `server/engine.py` | `EngineHandle` — owns the single `VaayuBaseModel`, the queue and the worker; maps a validated request onto `generate` / `stream_generate` kwargs | no |
| `server/chat.py` | `tokenizer.apply_chat_template(messages, tools=…)` → token ids | mostly |
| `server/tools.py` | tool-call format registry: detect the model's call syntax, parse output into OpenAI `tool_calls` | yes |
| `server/stopping.py` | resolve `max_tokens` against derived context; build stop criteria; decide `finish_reason` | yes |
| `server/__main__.py` | `python -m vaayu.server --model PATH --port 8000` | no |

The pure/impure split is the testing strategy: `stopping.py`, `tools.py` and
`schemas.py` unit-test with **no GPU and no checkpoint**, so they join the existing
892-test suite. Only `engine.py` / `app.py` need an on-device gate.

### 3.1 Lifecycle

One model, loaded once in the FastAPI lifespan from `--model` / `VAAYU_MODEL_PATH`.
**No hot-swap.** Vaayu's load costs minutes (pinning, slab arena, NVMe warmup,
expert-cache sizing); swapping a model is a process restart. `/v1/models` reports the one
loaded model and its derived `max_seq_len`.

### 3.2 Concurrency

`asyncio.Queue(maxsize=8)` → a single worker task → `run_in_executor` for the blocking
generate call. No two threads ever touch the model. Depth exceeded → `503` with
`Retry-After`. Queue position is emitted as a leading SSE comment so a waiting client can
distinguish "queued" from "hung".

`batched_generate` is deliberately **not** wired in v1: it is static-batch v1, and N ≥ 2 at
long context is a settled hardware refusal. It stays green and unwired.

### 3.3 Endpoints

```
POST /v1/chat/completions      stream and non-stream
GET  /v1/models                the one loaded model + derived max_seq_len
GET  /health                   VRAM, queue depth, PrefixCacheStore.stats(), model id
```

`/health` surfacing `PrefixCacheStore.stats()` is not decoration: prefix-cache hit rate is
the difference between ~1 s and ~98 s TTFT, and there is currently no way to observe it
from outside the process.

### 3.4 Request fields

OpenAI subset: `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`,
`stop`, `stream`, `stream_options`, `tools`, `tool_choice`, `seed`.

Vaayu extras, all optional (OpenAI clients ignore unknown fields, so compatibility
holds): `session_id`, `kv_cache_mode`, `drafter_model_path`. No engine-specific field is
ever required.

## 4. `max_tokens` and stopping — the truncation fix

**Streaming does not fix truncation.** Streaming is *delivery*; truncation is *stopping
policy*. Streaming 512 tokens still yields 512 tokens and still cuts mid-function — you
merely watch it happen. Streaming ships in v1 for TTFT visibility, early abort and
mid-stream tool-call detection, not as the fix.

The fix is that generation stops when the **model** is done, and when it is cut off the
client is **told**.

In `server/stopping.py`, pure, no literals:

```
resolve_max_tokens(requested, prompt_len, max_seq_len) -> int
    budget = max_seq_len - prompt_len       # max_seq_len is DERIVED at load
    if budget <= 0: raise ContextTooLong
    return budget if requested is None else min(requested, budget)
```

| condition | `finish_reason` |
|---|---|
| EOS token id | `stop` |
| stop string matched | `stop` |
| tool-call close marker | `tool_calls` |
| cap reached | `length` |
| `prompt_len >= max_seq_len` | 400, never generates |

`max_tokens` **omitted is the normal case**; the default is the whole remaining window.
No `512` anywhere in the server. Stop strings go through HF
`generate(stop_strings=…, tokenizer=…)`.

**Why the honest `finish_reason` matters more than the cap.** An agent that receives
`"length"` knows the code is incomplete and can resend to continue. An agent that
receives silently-truncated code writes it to a file.

**Interaction with admission.** Defaulting the cap to the full window means every request
admits against `predict_peak_vram(max_seq_len)`. Row 9 verified exactly that length passes
on-device 4/4, so this is the honest cap and not an optimistic one. It does make todo row
4's 2× KV ring-doubling overshoot the common case rather than the rare one (§9.3).

**CPU path.** `vaayu_base.py:1648` falls back to `max_seq_len = 512` when there is no
`VramPlan` — i.e. CPU / no-CUDA, which CLAUDE.md §5 requires keep working. The server
would silently inherit a 512-token window there. It must **warn at startup** when
`max_seq_len == 512` and no plan exists, naming the cause.

## 5. Prefix cache wiring

The server passes `session_id=None` by default, so `PrefixCacheStore.lookup` auto-prefix-
matches by longest common prefix across the pool. OpenAI clients resend the full history
every turn, so turn *n+1* LCP-matches turn *n*'s stored entry. This reaches the
136–377× TTFT path with **zero client changes**. An optional `session_id` extra field
gives explicit control.

Two engine changes belong to this spec, both server-specific failures that a single-user
test script never hits:

1. **An explicit-id miss must fall back to auto-match.** `prefix_cache.py:71` gates
   auto-match on `session_id is None`, so naming a session and missing it yields a full
   prefill with no fuzzy fallback — strictly worse than not naming it at all.
2. **A client disconnect must not drain the pool.** `lookup()` *pops* the entry and
   `store()` re-inserts it only after `generate()` returns. An SSE client that aborts
   mid-stream — routine for an agent — kills the request, and the entry is gone
   **permanently**. Fix: `try/finally` re-insert on the abort path.

## 6. Tool calling

- **Render.** `tokenizer.apply_chat_template(messages, tools=req.tools,
  add_generation_prompt=True)`. A template with no tool support → **400 with a named
  reason**; `tools` is never silently dropped (CLAUDE.md §5: warn, never silently
  degrade).
- **Parse.** A registry in `server/tools.py`, selected by probing the chat template and
  config for call markers — **detection, not model-name matching** (§5). v1 covers the
  Hermes/Qwen `<tool_call>{json}</tool_call>` family plus a generic fenced-JSON fallback.
- **Stop.** The close marker is registered as a stop string, so decode halts at the call
  instead of hallucinating the tool's result.
- **Stream.** The tool-call region is buffered and emitted as one complete
  `delta.tool_calls`. Partial-JSON streaming breaks clients and is not worth it in v1.
- **MCP is out of scope for the engine.** The MCP client belongs to the agent. Vaayu
  speaks `tools` / `tool_calls` and nothing more.

## 7. Error mapping

| cause | response |
|---|---|
| `VramPlanError` from `_assert_context_fits` | 400, carrying `_too_big_message` verbatim |
| `prompt_len >= max_seq_len` | 400 |
| queue depth exceeded | 503 + `Retry-After` |
| background generate raises | SSE `error` event then close; 500 non-stream |

`streaming.py` already catches a background-thread exception, pushes the stop signal and
re-raises after `join()`, so the server inherits a correct failure path rather than a hang.

## 8. Testing

- **Pure, no GPU — join the 892-test suite.** `resolve_max_tokens` parametrized over all
  four on-disk geometries' derived `max_seq_len` (§5: "test both models"); the
  `finish_reason` table; every tool parser; schema validation.
- **FastAPI `TestClient` against a fake engine.** Routes, SSE framing, 503 at depth, and
  the disconnect-restores-prefix-entry case from §5.
- **New on-device gate `test_server_smoke.py`**, at the repo root beside the other two
  root-level gates. One real load, then a non-stream request, a stream request and a
  tool-call request. Run under
  `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`.
- **Must not regress.** Full suite 892 passed / 3 skipped; dense smoke gate PASS with
  perplexity 8.78.

## 9. Named risks

### 9.1 Do not copy `exllamav3-server.py`'s model swapping
It unloads and reloads per request. Vaayu's load takes minutes. One model, loaded at
boot, swapped only by restart.

### 9.2 The speculative path is sampling-blind
`speculative_generate` honours `temperature` only; `top_p`, `top_k` and stop strings fall
into `**kwargs` and are silently ignored, and it stops only on EOS or the counter. If a
drafter is active and a request asks for `top_p`, the server must **warn**, not pretend.

### 9.3 Reserve policy versus a resident model
Row 8's reserve floors were tuned for streamed models. On Llama-3.2-1B the clamp drove
`pinning_budget → 0` at a measured −19 % decode. A resident coder model may hit the same
policy and be refused the pinning it obviously deserves. Needs an on-device check during
implementation.

### 9.4 KV ring doubling
todo row 4 (parked): at `max_seq_len` the EXL3 ring-buffer cache allocates capacity for
2× that length. §4's default makes this the common case.

### 9.5 `batched_generate` stays unwired
Deliberate. Queue, not batch. Revisit only if per-cell VRAM drops enough for N = 2.

## 10. Out of scope for v1

- MCP client (belongs to the agent).
- Continuous batching / paged attention.
- Multi-model hosting and hot-swap.
- `/v1/completions` and FIM.
- The Anthropic Messages wire format.
- **Acquiring the resident coder checkpoint.** A ~14B-class coder at 4–4.5 bpw EXL3 must
  be downloaded for the "usable now" target. This is a parallel action item and **not** a
  design dependency: §5 requires the server be model-agnostic, so it is built and tested
  against the checkpoints already on disk.
