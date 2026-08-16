# 🌐 Vaayu as a Server — Complete Reference

Vaayu runs as a long-lived, OpenAI-compatible HTTP server. Any client that already
speaks the OpenAI API — a coding agent, Cline, Aider, LiteLLM, the `openai` SDK —
works against it with no adapter.

Every number on this page was measured on the reference box — 16 GB VRAM, 32 GB RAM,
NVMe — with same-day A/B arms. Where a figure is a refutation rather than a win, it says so.

---

## Contents

1. [Starting the server](#1-starting-the-server)
2. [Endpoints](#2-endpoints)
3. [CLI parameter reference](#3-cli-parameter-reference)
4. [Request body reference](#4-request-body-reference)
5. [curl examples](#5-curl-examples)
6. [**Integrating a coding agent — the must-knows**](#6-integrating-a-coding-agent--the-must-knows)
7. [Making it faster](#7-making-it-faster)
8. [Limits, and what is deliberately not here](#8-limits-and-what-is-deliberately-not-here)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Starting the server

```bash
./venv/bin/python -m vaayu.server --model /path/to/checkpoint --port 8000
```

or set the path in the environment:

```bash
VAAYU_MODEL_PATH=/path/to/checkpoint ./venv/bin/python -m vaayu.server
```

The load is **not** instant — pinning, slab arena, NVMe warmup and expert-cache
sizing all happen at boot. Expect seconds for a small dense checkpoint and minutes
for a large MoE one. The server is ready when it prints:

```
[Vaayu-Server] loaded in 15.8s; derived max_seq_len=8192
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**One model per process.** There is no hot-swap endpoint and there will not be one:
a Vaayu load costs minutes, so swapping a model is a restart, not a request. Run one
server per model and route in front of them if you need several.

---

## 2. Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | generation, streaming and non-streaming |
| `GET` | `/v1/models` | the one loaded model and its **derived** `max_seq_len` |
| `GET` | `/health` | readiness, VRAM, queue depth, **prefix-cache stats** |

`/health` is not decoration. It is the only way to see whether prefix reuse is
working from outside the process, and prefix reuse is the difference between ~1 s
and ~98 s time-to-first-token:

```json
{
  "status": "ready",
  "model": "vaayu",
  "max_seq_len": 8192,
  "queue_depth": 8,
  "queued": 0,
  "prefix_cache": "1hit/1miss/0evict hit_rate=50% saved=52/104 tok (50%)",
  "vram_free_gib": 14.07,
  "vram_total_gib": 15.47
}
```

---

## 3. CLI parameter reference

### Serving

| flag | default | what it does |
|---|---|---|
| `--model` | `$VAAYU_MODEL_PATH` | checkpoint directory. **Required.** |
| `--served-model-name` | the checkpoint path | the `id` reported by `/v1/models` and echoed in responses. Set it to something short and stable so clients can hardcode it |
| `--host` | `127.0.0.1` | bind address. `0.0.0.0` to expose on the LAN |
| `--port` | `8000` | |
| `--device` | `cuda:0` | |
| `--queue-depth` | `8` | waiting requests allowed before `503`. **One runs at a time regardless** |

### Speed

These are the knobs that change throughput. Defaults are already the fast ones —
they differ deliberately from `vaayu_base`'s library defaults, because a server
holds one model for its whole life and should fill the box.

| flag | default | what it does |
|---|---|---|
| `--max-pinned-layers` | **`-1`** | `-1` auto-fills the VRAM budget with transformer layers. `0` pins shared layers only, so **every transformer layer streams from NVMe on every token**. Measured cost of getting this wrong: **13× on Llama-3.2-1B, 2.0× on MiniMax-M2.7, 1.27× on Llama-3.3-70B** (§7) |
| `--no-prefix-cache` | *(off — cache is ON)* | disables cross-turn KV reuse. Do not use this for an agent workload; reuse is the largest measured lever here (136–377× TTFT) |
| `--prefix-compress N` | `None` | packs pooled KV at N bits (**9** is the tested floor) so the same RAM budget holds ~2× the sessions. Matters when serving several agent sessions — see §6 |
| `--drafter-model-path` | `None` | speculative decoding with a small drafter. Measured **5.57×** on streamed dense. Refused at load on both MoE targets |
| `--use-prompt-lookup` | off | n-gram speculative decoding, **zero VRAM**. Measured **1.94×** on streamed dense; strongest on code, where identifiers and boilerplate repeat |
| `--draft-tokens` | `5` | candidates per speculative round. `5–8` with a drafter, `10–15` with prompt lookup |
| `--prefill-chunk-size` | `512` | tokens per prefill chunk. 512 is measured best; smaller costs TTFT (+24.7 % MoE, +213 % dense at 128) and buys nothing |
| `--kv-cache-mode` | `auto` | `auto` picks bf16 for EXL3. `fp8` is **1.5× larger**, not smaller — it keeps a dequantized mirror |
| `--attn-implementation` | `sdpa` | `flash_attention_2` derives a larger context at the same reserve. Every shipped `max_seq_len` figure was verified on **sdpa** |
| `--ram-cache-budget-gb` | *derived from total RAM* | GB of **pageable** RAM for the dense layer cache. `0` disables it. **No effect on MoE** — see §7.5 |
| `--enable-kv-persist` | off | persist prefix KV to disk so reuse survives a restart |

> **`--drafter-model-path` requires a vocabulary match.** The drafter and target must
> have identical vocab sizes; the server refuses the pairing at load otherwise, before
> any weights are read. Llama-3.2-1B drafts for Llama-3.3-70B (both 128256). It does
> **not** draft for MiniMax (200064) — that mismatch used to surface as a CUDA
> `CUBLAS_STATUS_INTERNAL_ERROR` several frames away.

---

## 4. Request body reference

`POST /v1/chat/completions`. Unknown OpenAI fields (`logit_bias`, `presence_penalty`,
`user`, …) are **ignored, not rejected**, so off-the-shelf clients do not 422.

| field | type | default | notes |
|---|---|---|---|
| `model` | string | — | required; matched loosely, this server has one model |
| `messages` | array | — | required, non-empty. Standard `role`/`content`, plus `tool_calls` / `tool_call_id` |
| `max_tokens` | int \| null | **null** | **omit it.** See below |
| `temperature` | 0.0–2.0 | `0.0` | `0.0` = greedy, no sampling |
| `top_p` | 0.0–1.0 | `1.0` | only applied when `temperature > 0` |
| `top_k` | int ≥ 0 | `0` | only applied when `temperature > 0` |
| `stop` | string \| array | `[]` | a string is normalised to a one-element list |
| `stream` | bool | `false` | SSE when true |
| `stream_options.include_usage` | bool | `false` | adds `usage` to the final chunk |
| `tools` | array | `null` | OpenAI function-tool definitions |
| `tool_choice` | any | `null` | accepted and passed through the template |
| `seed` | int | `null` | |
| `n` | int | `1` | **must be 1.** `n>1` is a 422 with a named reason — concurrency is 1 on this hardware |
| `session_id` | string | `null` | **Vaayu extra.** Pins this turn to a prefix-cache session. Omit it and auto-prefix-match picks the best match itself |
| `kv_cache_mode` | string | `null` | Vaayu extra |
| `drafter_model_path` | string | `null` | Vaayu extra |

### `max_tokens` — omit it

This is the single most important difference from a fixed-`MAX_NEW_TOKENS` script.

- **Omitted (the normal case):** the cap becomes `max_seq_len − prompt_len`, derived
  at load. Generation stops when the model emits EOS or a stop string — that is,
  when the model is **done**.
- **Given:** clamped to that same remaining window.
- **Prompt already fills the window:** `400`, with a message naming both numbers and
  how many tokens to cut.

`finish_reason` tells you what happened, and an agent must act on it:

| value | meaning | what the agent should do |
|---|---|---|
| `stop` | EOS or a stop string | use the output |
| `length` | the cap was reached — **output is truncated** | resend the turn to continue; do **not** write it to a file |
| `tool_calls` | the model called a tool | dispatch the calls, append results, send the next turn |

---

## 5. curl examples

### Basic completion

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [{"role": "user", "content": "write a binary search in Python"}]
}'
```

`max_tokens` omitted → the model generates a complete function and stops itself.

### Streaming

```bash
curl -sN localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [{"role": "user", "content": "write a quicksort"}],
  "stream": true,
  "stream_options": {"include_usage": true}
}'
```

```
: vaayu queue position 0

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"```python"}}]}
…
data: {"id":"chatcmpl-…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{…}}

data: [DONE]
```

The leading `: vaayu queue position N` is an SSE **comment** — every client ignores
it, and it lets a waiting caller tell "queued behind N" from "hung".

### Tool calling

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [{"role": "user", "content": "what is in src/main.py?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read a file from the repository",
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  }]
}'
```

```json
{"choices": [{
  "message": {"role": "assistant",
              "tool_calls": [{"id": "call_…", "type": "function",
                              "function": {"name": "read_file",
                                           "arguments": "{\"path\": \"src/main.py\"}"}}]},
  "finish_reason": "tool_calls"}]}
```

Then send the result back as a `tool` message and continue:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [
    {"role": "user", "content": "what is in src/main.py?"},
    {"role": "assistant", "tool_calls": [{"id": "call_abc", "type": "function",
       "function": {"name": "read_file", "arguments": "{\"path\":\"src/main.py\"}"}}]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "print(\"hello\")"}
  ]
}'
```

### Pinning a session for prefix reuse

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "session_id": "agent-7",
  "messages": [{"role": "user", "content": "…"}]
}'
```

### Watching the cache and the queue

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/models
```

### With the `openai` Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

r = client.chat.completions.create(
    model="vaayu",
    messages=[{"role": "user", "content": "write a binary search"}],
)                                        # no max_tokens -- that is the point
print(r.choices[0].finish_reason)        # act on this
print(r.choices[0].message.content)
```

---

## 6. Integrating a coding agent — the must-knows

Seven things. The first three are the ones that bite.

### 6.1 Omit `max_tokens`, and branch on `finish_reason`

Most agent frameworks default `max_tokens` to some number like 1024 or 4096. **Turn
that off.** Vaayu derives the cap from the real remaining context window; a client-side
number can only make it smaller.

Then treat `finish_reason == "length"` as a first-class case. It means the code you
received is **incomplete** — the correct response is to continue the turn, never to
write it to disk.

### 6.2 Resend the full history every turn — that is the fast path

Prefix caching reuses the KV of the longest matching prefix and prefills only the
divergent tail. Measured 136–377× on TTFT. OpenAI clients already resend full history,
so **the default behaviour is the optimal one** and there is nothing to configure.

Do not try to "optimise" by trimming or summarising history mid-conversation — that
breaks the prefix and forces a full prefill. If you must trim, trim from the **end**,
never the beginning.

Pass `session_id` when you have several independent agent streams: it makes the match
exact instead of a scan. An unrecognised `session_id` still falls back to auto-match,
so it can only help.

### 6.3 One request at a time — design your fan-out around it

The engine is not thread-safe and N≥2 at long context is a hardware refusal on 16 GB,
not a missing feature. Requests queue (FIFO, `--queue-depth`, default 8) and get `503`
+ `Retry-After` beyond that.

If your agent issues parallel sub-agent calls, they will **serialise**. Either accept
that, or cap your own concurrency to 1 and save the round trips. Handle `503` with a
retry — do not fail the tool call.

### 6.4 Context is the binding constraint, not speed

Check `/v1/models` at startup and budget against the real number:

| checkpoint | derived `max_seq_len` (sdpa) |
|---|---|
| Qwen3-235B | 8192 |
| MiniMax-M2.7 | 8192 |
| Llama-3.3-70B | 8192 |
| Llama-3.2-1B | 90639 |

8192 is **tight** for a tool-using agent: system prompt plus a few MCP tool schemas can
be 1.5–3k before any code is read. Keep tool schemas terse, and read files in slices.
`--attn-implementation flash_attention_2` derives more context at the same reserve.

### 6.5 MCP belongs in your agent, not here

Vaayu speaks OpenAI `tools` / `tool_calls`. It has no MCP client and will not grow one.
Your agent connects to MCP servers, converts their tool definitions into OpenAI
function schemas, and passes them as `tools`. Results come back as `tool` messages.

### 6.6 Tool-call support is a property of the checkpoint

The call format is detected by **probing the chat template**, never by model name. If
the template has no tool support, a request carrying `tools` is a `400` with a named
reason — deliberately, rather than silently dropping the tools and leaving your agent
waiting for a call the model was never told about.

Check before you rely on it:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vaayu","messages":[{"role":"user","content":"hi"}],
       "tools":[{"type":"function","function":{"name":"t","parameters":{}}}]}'
```

A `400` mentioning tool support means: pick a checkpoint whose template renders tools.

### 6.7 Streaming gives you early-abort, and buffers tool calls

Content arrives as OpenAI deltas. A **tool call is buffered and emitted once, complete** —
never as partial JSON, because partial tool-call JSON breaks clients that parse each
delta. Do not expect incremental `function.arguments` fragments.

Aborting a stream is safe: the prefix-cache entry is restored on the abort path rather
than being lost.

---

## 7. Making it faster

Measured on this box (16 GB VRAM, 32 GB RAM), same-day A/B.

### 7.1 Pin the layers — the big one, and it is now the default

`--max-pinned-layers -1` fills the VRAM budget with transformer layers, which then never
touch the NVMe again. At `0` the budget sits idle and every layer streams per token.

| checkpoint | `--max-pinned-layers 0` | `-1` (default) | gain | what actually pinned |
|---|---|---|---|---|
| Llama-3.2-1B | 6.40 tok/s | **83.27 tok/s** | **13.0×** | 16 of 16 layers — fully resident |
| MiniMax-M2.7 (MoE) | 0.760 tok/s | **1.537 tok/s** | **2.02×** | non-expert parts of all 62 layers |
| Llama-3.3-70B (dense) | 0.131 tok/s | **0.167 tok/s** | **1.27×** | 1 of 80 layers |

**The gain scales with the fraction of the model that fits**, which is why it ranges from
13× to 1.27×. Llama-3.3-70B's layers are ~460 MB each against a ~2.6 GB pinning budget, so
only one stays resident and 79 still stream per token — that model is disk-bound no matter
what this flag says. The 1B fits entirely and never touches the NVMe again.

On MoE, `-1` pins the *non-expert* parts of every layer (attention, norms, router
gates) and leaves the experts to the expert cache — the right split, automatically.

**You get this for free; it is the default.** It is documented because the cost of
overriding it to `0` is large and non-obvious.

### 7.2 Use a checkpoint that fits in VRAM

The structural answer. A model whose weights fit entirely in the ~10.5 GiB weights
budget pins completely and never streams — that is what the 1B's 83 tok/s is. A
14B-class coder at 4–4.5 bpw EXL3 (~8 GB) is the target shape: interactive speed, and
a much larger context because its KV is small.

Vaayu is model-agnostic; this needs no code change, only a checkpoint.

### 7.3 Speculative decoding — dense only

| mode | flag | measured |
|---|---|---|
| drafter | `--drafter-model-path /path/to/Llama-3.2-1B-Instruct` | **5.57×** on Llama-3.3-70B (86 % acceptance, byte-identical output) |
| prompt lookup | `--use-prompt-lookup --draft-tokens 15` | **1.94×** on the same, zero VRAM |

Both are **refused at load on MoE** — a 6-position verify pass needs 53.4 GB (Qwen) /
26.4 GB (MiniMax) of expert working set against a 9.1 GB budget. That is a wall, not a
tuning problem.

> The speculative loop honours `temperature` only. `top_p`, `top_k` and `stop` are
> ignored on that path — the server logs a warning and puts it in `warnings` on the
> response rather than pretending.

### 7.4 Keep the prefix pool big enough

Reuse does not degrade gracefully. Measured: once the pool is over capacity, reuse
collapses to **zero**, not proportionally, because each recomputed session's re-store
cascade-evicts the ones before it. TTFT went from 9.8 s back to 352 s at one session
past capacity.

If you serve several agent sessions, add `--prefix-compress 9`: pooled KV packs to
~0.55×, so the same RAM holds roughly twice the sessions, at ~0.7 s of TTFT on the hit
path. The first budget-pressure eviction prints a warning naming the capacity — watch
for it.

### 7.5 RAM layer caching — and why it is dense-only

Layers that do not fit in VRAM are held in **pageable host RAM** rather than re-read
from NVMe on every token. It is on by default and it works: on Llama-3.3-70B it holds
**33 of 80 layers (16.1 GB)** and scores **264 hits / 248 misses** during decode.

The budget is derived from **total** system RAM minus a stated hardware reserve
(`RAM_CACHE_RESERVE`, 16 GiB on the reference box), *then* clamped by what is actually
free. That matters more than it sounds. The old form keyed off momentary free RAM, so
two identical runs of the same command gave:

| | budget | layers cached | decode |
|---|---|---|---|
| unlucky boot | 9.7 GB | 18 / 80 | 0.167 tok/s |
| lucky boot | 16.4 GB | 33 / 80 | 0.230 tok/s |

**A 38 % throughput swing decided by what else was running at boot.** The derived form
now reproduces the good configuration exactly — two consecutive runs: 16.2 GB, 33 layers,
264/248 hits, **0.232 tok/s** both times. The startup line names which bound applied:

```
[Vaayu-Cache] RAM layer cache budget 16.2 GB (bound by total RAM; total-derived 16.2 GB, live-derived 22.7 GB)
```

If it says `bound by available RAM`, something else on the box is eating your throughput.

**Why MoE gets none of this.** It is not an oversight, and it is not that MoE has no RAM
budget — it has a large one. The two caches use *different kinds of memory*:

| | memory | sized by | on this box |
|---|---|---|---|
| dense `LayerRAMCache` | **pageable** | total RAM − reserve | 16.2 GB |
| MoE `ExpertRAMCache` | **CUDA-pinned / page-locked** | `_EXPERT_CACHE_OOM_CEILING` | 8.5 GiB (9.1 GB) |

Pinned pages cannot be swapped, must be found contiguously by the kernel, and the
allocator rounds a request up to the next power of two — a 9.13 GB arena is charged
16 GiB. That, not a policy decision, is where the 8.5 GiB ceiling comes from, and it is
why raising it is an OOM experiment rather than a config change. MoE's non-expert
weights are already pinned in **VRAM** (all 62 layers on MiniMax-M2.7), so a pageable
layer cache would have nothing left to hold.

> **Measured 2026-08-16, and the answer was not what the working-set figures suggested.**
> Both MoE targets are **disk-bound on expert reads**: Qwen3-235B pulls **4.99 GB/token**
> from NVMe (422 expert reads), MiniMax-M2.7 **2.62 GB/token** (295) — roughly **72–74 % of
> decode time** at ~5.5 GB/s. And the two models sit in different regimes despite an
> identical 9.12 GB budget and near-identical share of their expert populations:
>
> | model | cache share | measured hit rate | locality |
> |---|---|---|---|
> | MiniMax-M2.7 | 6.5 % | **6.6 %** | none — rate equals share |
> | Qwen3-235B | 6.4 % | **44 %** | strong — 6.9× its share |
>
> So a pageable L2 tier behind the pinned cache is promising **on Qwen** and close to
> worthless on MiniMax. `working_set_bytes` was a misleading guide: it counts ONE decode
> step's experts, while the union across steps is the whole ~141 GB population.

### 7.6 What will not help

Do not spend time here; each is measured and settled.

| idea | verdict |
|---|---|
| batching / higher concurrency | N≥2 at long context is a hardware wall (peak 14.17–15.57 GiB of 15.47) |
| smaller `--prefill-chunk-size` | costs TTFT (+24.7 % MoE, +213 % dense at 128), buys nothing measurable |
| expert pinning | refuted: eviction share is linear in N and always 2–4× below the pin's cost |
| TurboQuant KV in the live cache | failed its quality gate 2–7× on both models at 4 bits |
| speculative decoding on MoE | refused by the working-set gate; EAGLE-3 would not lift it |
| `fp8` KV to save VRAM | `fp8` is **1.5× larger** than bf16 here — it keeps a dequantized mirror |

---

## 8. Limits, and what is deliberately not here

- **One model per process**, no hot-swap.
- **One request in flight**, FIFO queue, `n=1` only.
- **No continuous batching / paged attention.** `batched_generate` exists and is
  deliberately unwired: it is static-batch v1 and would admit nothing at these context
  lengths.
- **No `/v1/completions`, no FIM**, no Anthropic Messages wire format.
- **No MCP client** — that is the agent's job.
- **No auth, no TLS, no rate limiting.** Bind to `127.0.0.1` (the default) or put a
  reverse proxy in front of it. Do not expose this directly to a network you do not
  control.
- **Prefix caching is disabled for non-EXL3 checkpoints.** The bare `DynamicCache`
  returned for `kv_cache_mode="fp16"` has no `to_cpu`/`to_device`/`resident_bytes`, so
  the pool is disabled at load with a warning. `/health` will report
  `"prefix_cache": "disabled"`. Every EXL3 checkpoint is unaffected.

---

## 9. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `"prefix_cache": "disabled"` in `/health` | non-EXL3 checkpoint | use an EXL3 build; the fp16 adapter is unbuilt |
| generation is ~10× slower than expected | `--max-pinned-layers 0` | drop the flag; the default `-1` is what you want |
| `400` naming a token count | prompt exceeds the derived window | shorten by the number the message gives |
| `400` mentioning tool support | the checkpoint's template cannot render `tools` | use a tool-capable checkpoint |
| `503` + `Retry-After` | queue full | retry; raise `--queue-depth`; reduce client concurrency |
| `422` on `n` | `n>1` requested | this server returns one choice |
| `warnings` on the response | speculative decoding is ignoring your sampling params | drop `--drafter-model-path`/`--use-prompt-lookup`, or drop `top_p`/`top_k`/`stop` |
| a drafter pairing is refused at load | drafter/target vocab sizes differ | pair within one model family |
| server exits on startup, no VRAM plan warning | loaded on CPU | check `--device` |

### The on-device gate

```bash
SERVER_CKPT=/path/to/checkpoint \
  systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  ./venv/bin/python test_server_smoke.py
```

Real load, non-streaming, streaming, an oversized-prompt refusal, and a check that
prefix reuse actually **fired**. `ALL PASS` or a named failure list.
