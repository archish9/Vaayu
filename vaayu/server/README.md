# Vaayu Server — configuration reference

The server takes **no command-line arguments**. Every value lives in
[`config.py`](config.py); edit it and start:

```bash
./venv/bin/python -m vaayu.server
```

`MODEL_PATH` is the only required value. With it empty the server exits and tells you
which file to edit.

**Full HTTP reference** — endpoints, request fields, curl examples, agent-integration
checklist: [`documentation/server.md`](../../documentation/server.md).

---

## Where to put your values

| file | tracked by git | purpose |
|---|---|---|
| `config.py` | yes | the documented defaults, shared by everyone |
| `config_local.py` | **no** (gitignored) | your machine's values — anything defined here shadows `config.py` |

`config.py` imports `config_local.py` last, so a local file only needs the lines it
changes:

```python
# vaayu/server/config_local.py
MODEL_PATH = "/home/you/models/Qwen3-235B-exl3"
PORT = 8001
```

Keeping `MODEL_PATH` there means your checkpoint paths never land in a diff.

---

## Serving

| constant | default | what it does |
|---|---|---|
| `MODEL_PATH` | `""` | checkpoint directory. **Required.** |
| `SERVED_MODEL_NAME` | `None` | the `id` reported by `/v1/models` and echoed in responses. `None` uses the checkpoint path — set something short and stable so clients can hardcode it |
| `HOST` | `"127.0.0.1"` | bind address. `"0.0.0.0"` to expose on the LAN |
| `PORT` | `8000` | |
| `DEVICE` | `"cuda:0"` | |
| `QUEUE_DEPTH` | `8` | waiting requests allowed before `503`. **One runs at a time regardless** |
| `REQUIRE_EXCLUSIVE_GPU` | **`True`** | refuse to start while another process holds GPU memory, naming each holder and the command that frees it. Budgets are derived once at load, so sharing is a permanent per-token tax — up to **13×** and invisible in the output. `False` warns and loads anyway. Compute processes only; your desktop is never listed |
| `DEFAULT_STREAM` | **`True`** | what `stream` means when a request omits it. **Not** OpenAI's `false`: a non-streaming response sends nothing until the last token, and an uncapped one on a slow checkpoint outlives every client. An explicit `"stream": false` still wins |
| `ENABLE_THINKING` | **`False`** | let the model emit a reasoning block before the answer. Off: the trace costs the same tok/s as the answer. Templates with their own flag are told; one that hard-codes an open `<think>` (MiniMax-M2.7) gets the block closed for it — detected from the rendered generation prompt, never from a model name. Per-request override: `"enable_thinking": true` |

## Speed

Defaults are already the fast ones, and they differ deliberately from `vaayu_base`'s
library defaults: a server holds one model for its whole life and should fill the box.

| constant | default | what it does |
|---|---|---|
| `MAX_PINNED_LAYERS` | **`-1`** | `-1` auto-fills the VRAM budget with transformer layers. `0` pins the shared layers only, so **every transformer layer streams from NVMe on every token** while the pinning budget sits idle. Measured cost of getting this wrong: **13× on Llama-3.2-1B, 2.0× on MiniMax-M2.7, 1.27× on Llama-3.3-70B**. `N` pins the first `N` |
| `ENABLE_PREFIX_CACHE` | **`True`** | cross-turn KV reuse. Leave it on for an agent workload — it is the largest measured lever here (**136–377× TTFT**) |
| `PREFILL_CHUNK_SIZE` | `512` | tokens per prefill chunk. 512 is measured best; smaller costs TTFT (**+24.7 % MoE, +213 % dense at 128**) and buys nothing |
| `PREFIX_COMPRESS` | `None` | pack pooled KV at N bits (**9** is the tested floor) so the same RAM budget holds ~2× the sessions. Worth it when serving several agent sessions: reuse collapses to **zero, not proportionally**, once the pool is over capacity |
| `PREFIX_SESSIONS` | `3` | concurrent sessions the prefix pool must hold before the MoE expert cache claims the rest. Lower = faster decode, fewer cached sessions |
| `KV_CACHE_MODE` | `"auto"` | `auto` picks bf16 for EXL3. `"fp8"` is **1.5× larger**, not smaller — it keeps a dequantized mirror |
| `ATTN_IMPLEMENTATION` | `None` | `None` uses **sdpa**, which every shipped `max_seq_len` figure was verified on. `"flash_attention_2"` derives a larger context at the same reserve |
| `RAM_CACHE_BUDGET_GB` | `None` *(derived from total RAM)* | GB of **pageable** RAM for the dense layer cache. `None` derives it from TOTAL RAM so throughput is reproducible across restarts — keying off momentary free RAM swung Llama-3.3-70B decode **38 %** run to run. `0` disables it. **No effect on MoE**, whose experts live in the pinned expert RAM cache |
| `ENABLE_KV_PERSIST` | `False` | persist prefix KV to disk so reuse survives a restart |

## Speculative decoding

Opt-in, and **refused at load on both MoE targets** by the Fix #9 gate — these are
streamed-dense wins.

| constant | default | what it does |
|---|---|---|
| `DRAFTER_MODEL_PATH` | `None` | small drafter model. Measured **5.57×** on streamed dense |
| `USE_PROMPT_LOOKUP` | `False` | n-gram speculative decoding, **zero VRAM**. Measured **1.94×** on streamed dense; strongest on code, where identifiers and boilerplate repeat |
| `DRAFT_TOKENS` | `5` | candidates per speculative round. `5–8` with a drafter, `10–15` with prompt lookup |

> **`DRAFTER_MODEL_PATH` requires a vocabulary match.** Drafter and target must have
> identical vocab sizes; the server refuses the pairing at load otherwise, before any
> weights are read. Llama-3.2-1B drafts for Llama-3.3-70B (both 128256). It does **not**
> draft for MiniMax (200064) — that mismatch used to surface as a CUDA
> `CUBLAS_STATUS_INTERNAL_ERROR` several frames away.

---

## One model per process

There is no hot-swap endpoint and there will not be one: a load costs minutes (pinning,
slab arena, NVMe warmup, expert-cache sizing), so swapping a model is a **restart**, not
a request. Run one server per model and route in front of them if you need several.

The server is ready when it prints:

```
[Vaayu-Server] loaded in 15.8s; derived max_seq_len=8192
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Which knob for which workload

[`documentation/TUNING.md`](../../documentation/TUNING.md) says what to turn on for
which workload — and which knobs do nothing today, so you don't waste time on them.
`tune_advisor.py` derives the settings for your model and box and prints them to copy.
