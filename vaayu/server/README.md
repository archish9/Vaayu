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
| `PORT` | `9999` | |
| `DEVICE` | `"cuda:0"` | |
| `QUEUE_DEPTH` | `8` | waiting requests allowed before `503`. **One runs at a time regardless** |
| `LOG_DIR` | `"logs"` | `<LOG_DIR>/server.log` gets **everything** the process emits — Python prints, uvicorn, and the `[vaayu_rust]` io_uring lines (captured at fd level, since Rust writes them straight to fd 2). `<LOG_DIR>/sessions/<session_id>.log` gets one line per request lifecycle event |
| `CAPTURE_STDIO` | `True` | `False` leaves the console untouched and writes only the per-session files |
| `LOG_ROTATE_MB` | `50` | `server.log` rolls to `server.log.1` past this. The io_uring trace is verbose — 230 MB in 30 minutes was measured on a MoE run |
| `DELETE_ORIGINAL` | **`True`** | after Vaayu's split copy of a checkpoint is verified complete (every layer has its `.safetensors` **and** its `.done` marker), delete the source shards it was split from — **39–66 GB per checkpoint** on the reference box. A server holds one checkpoint for its whole life, so the duplicate is pure waste. **Irreversible**: recovery is a re-download, and the removal follows symlinks, so a shard linked into the shared HuggingFace cache takes the cached blob with it. Set `False` if another framework reads the same checkpoint. The server prints a line naming this before every load |
| `REQUIRE_SESSION_ID` | **`True`** | refuse a request naming neither `session_id` nor `user`. An unnamed request still gets reuse, but it keys on a hash of the prompt's first 64 tokens — so two agents sharing a system prompt land on one pool entry and evict each other, and reuse collapses to **zero**, not gradually. `False` allows unnamed requests |
| `REQUIRE_EXCLUSIVE_GPU` | **`True`** | refuse to start while another process holds GPU memory **on `DEVICE`**, naming each holder and the command that frees it. Budgets are derived once at load, so sharing is a permanent per-token tax — up to **13×** and invisible in the output. `False` warns and loads anyway. Compute processes only; your desktop is never listed |
| `EXCLUSIVE_GPU_MIN_MIB` | **`256`** | how much a holder must hold before it is worth refusing over. A bare CUDA context is 128 MiB, so `0` refuses the boot over any process that merely imports torch. Smaller holders are still reported |
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

## Reading the logs

```
logs/
  server.log                  # everything, rotated at LOG_ROTATE_MB
  sessions/agent-7.log        # one line per lifecycle event
  sessions/_unnamed.log       # requests that sent no session_id
```

```
2026-08-22T16:57:11 REQ    prompt=3 cap=16 stream=False thinking=on
2026-08-22T16:57:11 DONE   tokens=4 finish=stop prefix_hit=False wall=0.2s decode=14.8tok/s
```

Streaming requests add a `TTFT at=…` line between the two. `prefix_hit` is the one to
watch across turns of the same session: `False` on turn 2 means the prefix cache is not
being reused and TTFT is paying full prefill.

Filenames are sanitized — `session_id` arrives in an HTTP body, so it is untrusted input
that becomes a path. A hostile id lands inside `sessions/` with a hash suffix
(`../../etc/pwned` → `____etc_pwned-bf73a33346.log`), never outside it.

---

## One model per process

There is no hot-swap endpoint and there will not be one: a load costs minutes (pinning,
slab arena, NVMe warmup, expert-cache sizing), so swapping a model is a **restart**, not
a request. Run one server per model and route in front of them if you need several.

The server is ready when it prints:

```
[Vaayu-Server] loaded in 15.8s; derived max_seq_len=8192
INFO:     Uvicorn running on http://127.0.0.1:9999
```

---

## Which knob for which workload

[`documentation/TUNING.md`](../../documentation/TUNING.md) says what to turn on for
which workload — and which knobs do nothing today, so you don't waste time on them.
`tune_advisor.py` derives the settings for your model and box and prints them to copy.
