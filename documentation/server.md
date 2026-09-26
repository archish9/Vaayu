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
3. [Configuration reference](#3-configuration-reference)
4. [Request body reference](#4-request-body-reference)
5. [curl examples](#5-curl-examples)
6. [**Integrating a coding agent — the must-knows**](#6-integrating-a-coding-agent--the-must-knows)
7. [Making it faster](#7-making-it-faster)
8. [Limits, and what is deliberately not here](#8-limits-and-what-is-deliberately-not-here)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Starting the server

The server takes **no command-line arguments**. Set `MODEL_PATH` in
`vaayu/server/config.py` — or in `vaayu/server/config_local.py`, which is gitignored and
shadows it — then:

```bash
./venv/bin/python -m vaayu.server
```

With `MODEL_PATH` empty the server exits and names the file to edit.

The load is **not** instant — pinning, slab arena, NVMe warmup and expert-cache
sizing all happen at boot. Expect seconds for a small dense checkpoint and minutes
for a large MoE one. The server is ready when it prints:

```
[Vaayu-Server] loaded in 15.8s; derived max_seq_len=8192
INFO:     Uvicorn running on http://127.0.0.1:9999
```

**Stopping it.** `Ctrl-C` or `SIGTERM` stops generation immediately and the process
exits within about a second, even mid-generation — measured 0.46 s with ~200 s of tokens
still to run. That is not free behaviour: uvicorn's graceful shutdown waits for in-flight
requests *before* it runs the app's shutdown hook, and with no `timeout_graceful_shutdown`
it waits for the whole cap, so Vaayu cancels from the signal handler itself and chains to
uvicorn's. Anything the server held — GPU memory, pinned RAM, the io_uring reader — is
released when the process exits, and the disk KV writer is flushed on the way out.

**One model per process.** There is no hot-swap endpoint and there will not be one:
a Vaayu load costs minutes, so swapping a model is a restart, not a request. Run one
server per model and route in front of them if you need several.

---

## 2. Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | generation, streaming and non-streaming |
| `GET` | `/v1/models` | the one loaded model and its **derived** `max_seq_len` |
| `GET` | `/health` | readiness, VRAM, queue depth, **prefix-cache stats**, **batch formation** |

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
  "batch": {
    "max_batch_size": 1,
    "batching_enabled": false,
    "batches_run": 0,
    "gate1_refused": 0,
    "gate2_refused": 0,
    "formation_attempts": 0
  },
  "vram_free_gib": 14.07,
  "vram_total_gib": 15.47
}
```

`prefix_cache` gains ` borrow=N` once any request has been served by COPYING
another session's prefix instead of reusing its own. Read it against `saved`: a
high `hit_rate` with a low `saved` and a climbing `borrow` means your sessions
share a system prompt and little else, so each one is still re-prefilling its own
context every turn — give each agent a stable `session_id` and keep it across
turns.

The `batch` block answers "is `MAX_BATCH_SIZE` doing anything?" — the one
question a batching knob cannot answer from throughput alone. `batching_enabled`
is false whenever the model did not load on `sdpa`, whatever `MAX_BATCH_SIZE`
says. `formation_attempts` counts the times two or more requests were in flight
together at all; if it stays 0, the server never had the offered load to batch,
which is a different problem from being refused. When it does climb and
`batches_run` does not, the two refusal counters say which gate said no: gate 1
is the scheduler's own VRAM pre-check narrowing the batch before the engine sees
it (it prints nothing, so this counter is the only way to see it), gate 2 is
`batched_generate` refusing at the real width and falling back to serial.

---

## 3. Configuration reference

Every value below is a constant in [`vaayu/server/config.py`](../vaayu/server/config.py).
There are no flags: edit the file (or shadow a value in the gitignored
`vaayu/server/config_local.py`) and restart. The same table lives next to the code in
[`vaayu/server/README.md`](../vaayu/server/README.md).

### Serving

| constant | default | what it does |
|---|---|---|
| `MODEL_PATH` | `""` | checkpoint directory. **Required.** |
| `SERVED_MODEL_NAME` | `None` (the checkpoint path) | the `id` reported by `/v1/models` and echoed in responses. Set it to something short and stable so clients can hardcode it |
| `HOST` | `"127.0.0.1"` | bind address. `0.0.0.0` to expose on the LAN |
| `PORT` | `9999` | |
| `DEVICE` | `"cuda:0"` | |
| `QUEUE_DEPTH` | `8` | waiting requests allowed before `503`. **One runs at a time regardless** |
| `DEFAULT_STREAM` | **`True`** | what `stream` means when the request omits it. Not OpenAI's `false` — a non-streaming response sends nothing until the last token |
| `ENABLE_THINKING` | **`False`** | suppress the model's reasoning block. Detected from the rendered generation prompt: a template with its own flag is told, one that hard-codes `<think>` has it closed. Per-request override available |
| `LOG_DIR` | `"logs"` | `server.log` (everything, fd-level so the `[vaayu_rust]` lines are included) plus `sessions/<session_id>.log`, one line per request lifecycle event |
| `CAPTURE_STDIO` | `True` | `False` writes only the per-session files and leaves the console alone |
| `LOG_ROTATE_MB` | `50` | when `server.log` rolls to `server.log.1` |
| `DELETE_ORIGINAL` | **`True`** | delete the ORIGINAL shards once the split copy is verified complete, reclaiming **39–66 GB per checkpoint**. **Irreversible** — recovery is a re-download, and it follows symlinks into the shared HF cache. `False` keeps them for other tools |
| `REQUIRE_SESSION_ID` | **`True`** | refuse a request that names neither `session_id` nor `user`. Unnamed requests still get reuse, but they key on a hash of the prompt's first 64 tokens, so agents sharing a system prompt collide on one pool entry and reuse collapses to zero. `False` allows them |
| `REQUIRE_EXCLUSIVE_GPU` | **`True`** | refuse to start while another process holds GPU memory **on `DEVICE`**, naming the holders and the commands that free them. Budgets are derived once at load, so sharing is a permanent per-token tax — up to **13×**, and invisible in the output. `False` warns and loads anyway. Counts compute processes only; the desktop is never included |
| `EXCLUSIVE_GPU_MIN_MIB` | **`256`** | how much a holder must hold before it is worth refusing over. A bare CUDA context is 128 MiB, so `0` refuses the boot over any process that merely imports torch. Anything smaller is still reported, just not treated as a blocker |

### Logs

One directory per server process:

```
logs/
  run-2026-09-03T17-03-11Z/
    env.json          # the machine, the effective config, the derived model
    server.log        # every line the process emitted, timestamped, session-tagged
    requests.jsonl    # one JSON row per request, appended when it ends
    sessions/
      agent-7.log     # only the lines emitted while agent-7 was generating
  latest -> run-2026-09-03T17-03-11Z
```

`LOG_DIR` names the ROOT; runs live beneath it and `logs/latest` always points
at the newest, so `tail -f logs/latest/server.log` works across restarts. Old
run directories are never deleted automatically — they are yours to prune.

**Retention:** `sessions/<id>.log` carries the whole process stream for that
session, not just lifecycle events, so it rotates at the same `LOG_ROTATE_MB`
boundary as `server.log` (rolling to `<id>.log.1`) rather than growing
unbounded — this matters most for `sessions/_unnamed.log`, where every
request from a client that sends neither `session_id` nor `user` lands.

`server.log` captures at the file-descriptor level, so it includes the
`vaayu_rust` io_uring lines that a Python-level logger cannot see. Every line
carries an ISO8601 UTC timestamp and a `session=` field (`-` while the engine
is idle).

**Reporting a problem:** zip `logs/run-<timestamp>/` and attach it. That
directory is self-contained — `env.json` records the commit, the library
versions, the CPU/RAM/kernel, the GPU and its free VRAM, the NVMe devices with
their I/O scheduler and APST state, which of Vaayu's four forwards your model
selected, the derived MoE geometry, every budget the VRAM plan solved, your
effective configuration (including anything set in the gitignored
`config_local.py`) and every `VAAYU_*` variable.

Everything under `budgets` is what the engine RESOLVED, not what you asked
for — `budgets.kv_cache_mode` reads `bf16` or `fp16` where you configured
`"auto"`, because `"auto"` is a request and the KV dtype is decided at load
time from the loader strategy. The value you set is still there, unchanged,
under `config`.

Absolute paths are redacted before they are written: a checkpoint path is
recorded as its directory name plus a hash, never in full, so no username
leaves your machine.

**`requests.jsonl`** is the per-request companion to `server.log`: one JSON
object, appended when the request ends — whether it completed, was refused
before generation, raised, or was abandoned mid-stream by the client. A
request never ends without a row, which is the point of the file — a `server.log`
traceback used to be the only trace of a failed request, with nothing that said
which request it belonged to.

Each row carries the prompt and completion token counts (`prompt_tokens` and
`tokens`), wall time and tok/s, the prefix-cache outcome, the resolved `finish`
value, and a `decode` object with the request's own disk traffic and cache
telemetry. `finish` is one of:

| `finish` | when |
|---|---|
| `stop` | EOS or a stop string |
| `length` | the token cap was hit — **output is truncated** |
| `tool_calls` | the model called a tool |
| `cancelled` | the client disconnected before the stream finished |
| `error` | the request raised |
| `refused` | rejected before generation started — an oversized prompt, a missing `session_id`, a full queue, a request the model cannot render (no chat template, or `tools` it cannot take) |

The dividing line between the last two is the **status code, not the moment**:
`refused` is a 400 or 503 the client — or, for a missing chat template, the operator — can fix (the cases named above);
anything that raised and answers 500 is `error`, including a failure inside the
chat template, which is rejected before generation starts but is not the
client's to fix. Both values are **path-independent** — the same failure gets
the same word whether the request was streaming or not.

`cancelled` is a `requests.jsonl`-only value: a client that has already
disconnected never receives an OpenAI-vocabulary `finish_reason` for the
request it abandoned, so this row is the only place the outcome is recorded.
For a **non-streaming** cancellation the row and `server.log` deliberately
disagree: the log's `DONE` line prints the wire `finish_reason`, which maps a
cancellation to `length` because the OpenAI vocabulary has no other word for it,
while the row says `cancelled`. The row is the richer record; when the two
files differ on a cancelled request, this is why.

An `error`/`refused` row also carries an `error` object (`type`, `message`,
`traceback`). A `refused` row has the **same keys** as every other row — the
fields that are unknowable before generation starts (`prompt_tokens`, `cap`,
`prefix_hit`, `ttft_s`, `decode_tok_s`, `decode`) are `null` rather than absent,
so a script can index any field on any row without checking which kind of row it
is.

Every row also carries `batch`: `null` for a request run on its own, or
`{"n": <int>, "members": [<session_key>, ...]}` when the dispatcher (P10)
ran it together with `n - 1` other requests as one batch.

Inside `decode`: `bytes_read_gb` and `gb_per_token` are this request's own NVMe
traffic **when `batch` is `null`** — the difference between the persister's
read-byte counter at the end of the request and at its start (see the
`bytes=`/`gb_tok=` fields below for what that counter does and does not
count). `mean_expert_stall_s` is the same kind of diff over the engine's
expert-disk-stall total.

**Under a batch (`batch.n` > 1), `decode` is the WHOLE BATCH's counters, not
this row's own**, and is flagged `"shared": true` so nothing sums N rows into
a traffic figure that never happened on the wire. `tokens` and `ttft_s` stay
genuinely per-stream, but `wall_s` and `decode_tok_s` are also batch-wide: a
stream that retires early is still charged the batch's full wall time and its
own `decode_tok_s` understates. `prefix_hit` is batch-wide too — it reads
`true` on every row in the batch if any one member hit the prefix cache.

`expert_misses`, `expert_ram_hit_rate` and `expert_vram_hit_rate` are **not**
diffs — they are the engine's counters read once, at the end of the request.
That is deliberate: the engine resets those specific counters at the
prefill→decode boundary, inside each request, so a decode pass's hit rate is
not polluted by prefill's near-100%-miss cost. The point-in-time reading is
therefore the correct per-request figure, not a shortcut. When a request never
reached a decode pass — it failed during prefill, or produced no tokens — that
reset never fired, and these three fields come back `null` rather than the
*previous* request's numbers dressed up as this one's.

`peak_rss_gb` and `peak_vram_gb_process` are high-water marks, not this
request's own peak. `peak_vram_gb_process` in particular is a **process**
figure: CUDA's peak-memory counter is process-global, and Vaayu deliberately
never resets it, because resetting it would corrupt the VRAM calibration
measurements that read the same counter elsewhere. Expect it to only ever hold
steady or climb across a run — a rising number here is not a leak, it is the
model's true ceiling being reached once and remembered.

**The `bytes=`/`gb_tok=` fields on `server.log`'s `[VAAYU-IO] token summary`
line.** These count bytes **requested of the NVMe drive** for that one token —
counted where the read is submitted, not where it completes, so the figure is
what was asked of the device rather than a confirmed-delivered byte count. The
length counted is whichever one the submitting call has to hand, and they are
not uniform: the O_DIRECT paths count the sector-aligned length they actually
ask the kernel for, the preadv fallback counts the unaligned request length, and
the non-pinned `safetensors` load counts the materialised tensor bytes. Read it
as an account of disk traffic, not a block-exact one. They are decimal GB (`gb_tok`), matching every
other GB figure this engine prints.

Anything a cache serves — the expert RAM cache, the expert VRAM cache, the
non-expert tensor cache — contributes zero to this pair, because the counting
happens inside the persister's own read functions, which a cache hit never
reaches. Do not confuse `gb_tok` with `expert_disk` on the same line:
`expert_disk` is a **stall duration** (seconds spent waiting on those reads),
not a byte count.

### Speed

These are the knobs that change throughput. Defaults are already the fast ones —
they differ deliberately from `vaayu_base`'s library defaults, because a server
holds one model for its whole life and should fill the box.

| constant | default | what it does |
|---|---|---|
| `MAX_PINNED_LAYERS` | **`-1`** | `-1` auto-fills the VRAM budget with transformer layers. `0` pins shared layers only, so **every transformer layer streams from NVMe on every token**. Measured cost of getting this wrong: **13× on Llama-3.2-1B, 2.0× on MiniMax-M2.7, 1.27× on Llama-3.3-70B** (§7) |
| `ENABLE_PREFIX_CACHE` | **`True`** | cross-turn KV reuse. Leave it on for an agent workload — reuse is the largest measured lever here (136–377× TTFT). **Hybrid / linear-attention models (`qwen3_next` and similar) get cross-turn reuse too, since 42-B:** the pooled entry carries the recurrent state along with the KV, and a turn that continues a previous one (resends its ids and appends) is served from it. A turn that diverges mid-context instead takes a full prefill — there is no partial prefix of a recurrent state to lend — and is counted separately as `recurrent_partial` in the prefix stats line. Measured **6.24×** on round-2 TTFT at 8k tokens on Qwen3-Next |
| `PREFIX_COMPRESS` | `None` | packs pooled KV at N bits (**9**, both the recommended setting and the quantizer's ceiling) so the same RAM budget holds ~1.6× the sessions. 9 bits plus a mixed-precision outlier split is what clears the quality bar on both MoE forwards — [TUNING.md §2](TUNING.md). Matters when serving several agent sessions — see §6 |
| `DRAFTER_MODEL_PATH` | `None` | speculative decoding with a small drafter. Measured **5.57×** on streamed dense. Refused on both MoE targets **before the drafter is loaded** — see below. **Refused, with a WARNING naming the reason, on a model whose layers keep recurrent state** (hybrid / linear-attention families such as `qwen3_next`): a rejected draft cannot be rolled back out of a recurrent state, and before this refusal Qwen3-Next emitted garbage from the first drafting round. `VAAYU_SPEC_FORCE` does not lift it. |
| `USE_PROMPT_LOOKUP` | `"auto"` | n-gram speculative decoding, **zero VRAM**. `"auto"` resolves at load, where MoE-ness is known: **ON for dense** (measured **1.94×** streamed, strongest on code, where identifiers and boilerplate repeat), **OFF for MoE**. `True`/`False` opt a model in or out and always win. **On MoE, MEASURE BEFORE OPTING IN — it depends on what the model emits, not on which model it is.** Qwen3-235B's gate refuses to draft at all (`off_union`), so it gains nothing. MiniMax-M2.7 on one 128-token code prompt (2026-09-07): at server defaults acceptance is 0.14 vs a 0.6259 break-even so the gate latches `off_acceptance` after ~1 call; with thinking on the gate said `run` and still lost 12.9 % total / 16.0 % decode. An earlier arm on a different prompt measured 1.307×. **Since 2026-09-11 the gate runs its own A/B.** After calibration, its trial alternates 4 drafting rounds with 4 suppressed ones, for at least 4 pairs and at least 48 drafted tokens (a trial that drafts slowly runs more pairs). It admits speculation only if acceptance clears the expert-byte break-even AND the drafting rounds cost less per token than their suppressed partners, averaged over the pairs. It reports `bar_used` (`"pooled"`; `"wall"` / `"bytes"` only when too few pairs were timed, which falls back to the earlier single-window rule), `pooled_ratio` (above 1 means drafting was dearer per token), `pairs_used`, `alpha`, `wall_breakeven` and `base_from`. Measured on MiniMax-M2.7: the arm where the earlier single-window gate said `run` and lost **−19.7 %** is now refused, 3 of 3 replicates (pooled ratio 1.33–1.36; forcing speculation on it measured **−18.6 %**), and the arm where speculation wins is still admitted and worth **+15.4 %**. **It is still a claim about the start of a call, not a promise:** the refusal above rests on the wider window — its first four pairs alone averaged 0.99 — and the whole trial runs in the first ~40–60 rounds, which are the cheapest part of a call. **Time your own workload both ways.** When speculation *is* worth running, sizing the VRAM expert-cache prefix on the verify width is worth a further +4.16 %. **Refused, with a WARNING naming the reason, on a model whose layers keep recurrent state** (hybrid / linear-attention families such as `qwen3_next`): a rejected draft cannot be rolled back out of a recurrent state, and before this refusal Qwen3-Next emitted garbage from the first drafting round. `VAAYU_SPEC_FORCE` does not lift it. |
| `DRAFT_TOKENS` | `5` | candidates per speculative round. `5–8` with a drafter, `10–15` with prompt lookup |
| `PREFILL_CHUNK_SIZE` | `512` | tokens per prefill chunk. 512 is measured best; smaller costs TTFT (+24.7 % MoE, +213 % dense at 128) and buys nothing. On a hybrid (linear-attention) model the recurrent layers carry their state across chunks (gated-delta-net families such as `qwen3_next`; before 2026-09-18 they restarted at every chunk). A hybrid whose mixer Vaayu cannot continue refuses a multi-chunk prefill with an error; set this to `0` for such a model. |
| `KV_CACHE_MODE` | `"auto"` | `auto` picks bf16 for EXL3. `fp8` is **1.5× larger**, not smaller — it keeps a dequantized mirror |
| `ATTN_IMPLEMENTATION` | `None` | **prefers `flash_attention_2`** when `flash_attn` imports, then `sdpa`, then `eager` — it does **not** mean sdpa. Every shipped `max_seq_len` figure was verified on **sdpa**; `"flash_attention_2"` derives a larger context at the same reserve, and costs **−2.3 % MiniMax / −4.1 % Qwen** end-to-end against sdpa. `MAX_BATCH_SIZE` below is **inert** unless this resolves to `sdpa` |
| `MAX_BATCH_SIZE` | `1` | requests served together as one batch. A **ceiling, never a target** — max width is model- and context-dependent and is **not 3**. Measured 2026-09-08 at ~4.15k-token prompts, **engine-level (`batched_generate` called directly, not through the server) and with the prefix cache OFF** — no server-level A/B has been run, and the server defaults `ENABLE_PREFIX_CACHE = True`: `2` is **+1.606×** on MiniMax-M2.7 (TTFT 317.2 → 223.3 s, peak VRAM *lower* batched); `3` OOMs MiniMax and is **slower** than 2 on Qwen3-235B. Qwen admits N=1 only today, so this buys it nothing. **Inert unless `ATTN_IMPLEMENTATION` resolves to `sdpa`** |
| `BATCH_WAIT_MS` | `50` | milliseconds a forming batch waits for company before running what it has. **Without it `MAX_BATCH_SIZE` never batches two concurrent clients at all** — a request wakes the dispatcher the instant it is queued, so the first of two simultaneous clients runs alone while the second is still being tokenized (measured on Llama-3.2-1B through the real server: 2 clients, warm, `batch=None` on both requests, 3 runs of 3). It also un-sticks two **steady** clients, which otherwise alternate one-at-a-time forever and never meet in the queue. Cost of getting it wrong is one-sided and bounded: a lone request at an idle server is delayed by this much, once — 0.02 % of MiniMax-M2.7's ~220 s TTFT at 4.15k tokens, ~4 % of a Llama-3.2-1B request. A batch that fills stops waiting immediately. `0` restores the pre-2026-09-09 behaviour. **Inert at `MAX_BATCH_SIZE = 1`.** Does **not** help a client whose turnaround exceeds the window — joining a *running* batch is continuous batching, which Vaayu does not do |
| `RAM_CACHE_BUDGET_GB` | `None` *(derived from total RAM)* | GB of **pageable** RAM for the dense layer cache. `0` disables it. **No effect on MoE** — see §7.5 |
| `ENABLE_KV_PERSIST` | `False` | persist prefix KV to disk so reuse survives a restart |
| `KV_PERSIST_DIR` | `None` *(`~/.cache/vaayu/kv`)* | where those entries are written. Until 2026-09-10 the feature above could be turned **on but not aimed**. The write budget is **derived as half the free space of whatever filesystem this names**, so this choice sets the cache's *size* as well as its location — a small or shared partition silently caps how much reuse survives a restart |
| `KV_PERSIST_SYNC` | `False` | `fsync` each persisted entry before returning. `False` leaves the write in the page cache: a clean shutdown keeps it, a power loss or panic can leave a **torn entry** read back at the next boot. `True` costs a disk flush on the prefix-store path. Inert while `ENABLE_KV_PERSIST` is `False` |
| `AUTO_PREFIX_MATCH` | `True` | let a request naming **no session** still find a prefix entry, by matching the longest shared prompt prefix instead of an exact key. This is the behaviour `REQUIRE_SESSION_ID` above describes; the server could not turn it off before 2026-09-10. `False` gives an unnamed request **no reuse at all** rather than approximate reuse — on this workload the larger loss. Changes nothing for a request that names its session |
| `MAX_SEQ_LEN` | `None` *(derived)* | context to plan VRAM for. `None` derives the largest context whose predicted peak fits the reserve floor — the shipped path, and the one every measured figure here was taken at. The one reason to set it: a **shorter** context hands the difference back to the expert cache, the decode bottleneck on both MoE targets. Both failure directions are loud, not silent — above the box's `S_max` the load **refuses** (`VramPlanError`, naming `S_max`); between the derived point and `S_max` it loads while the larger KV reserve eats the pinning budget, and at `0` the engine warns that every layer now streams from disk on every token |
| `ENABLE_EXPERT_USAGE` | `False` | record which experts are routed and which are evicted, for a triage bundle. **MoE-only by construction** — on a dense checkpoint the engine warns that no tracker will be built and does nothing (there are no experts to count, not a dense second-class default). Off because its cost is **unmeasured**: it puts `track_evicted_keys` on the expert-cache path, which decode hits on every layer visit. Measure your own decode both ways before leaving it on |
| `DECODE_TOP_K` | `None` | decode-time expert budget. `None` is off (lossless); an int is a fixed clamp; **`"auto"` measures this checkpoint once at first load — a fixed k if one fits `DECODE_TOP_K_BUDGET`, otherwise a per-token routing-mass cutoff — applies the fastest in-budget setting and caches it per checkpoint.** The only knob here that changes the model's output: greedy ids diverge by design. The first load of each checkpoint is slower (≈6.5 min Qwen3-Next-80B, ≈21 min Qwen3-235B, ≈26 min MiniMax-M2.7) and no endpoint answers meanwhile; later loads apply the cached pick instantly. `/health` → `decode_top_k` shows the active pick and its measured cost. The safe value is per model and not derivable from geometry, so do not guess an int — see [TUNING.md §5c](TUNING.md) |
| `DECODE_TOP_K_BUDGET` | `0.035` | the perplexity cost `"auto"` may spend, as a fraction. At 3.5 %: Qwen3-Next-80B → fixed k=8, **+23.1 % tok/s** for +3.43 %; MiniMax-M2.7 → cutoff p=0.90, **+8.4 %** for +2.25 %. At 2 %: Qwen3-Next → k=9, +17.0 % for +1.47 %. Changing it re-measures on the next load; ignored for an explicit int |
| `KEEP_AWAKE` | `"auto"` | hold the CPU package out of deep idle while a generation runs: a `SCHED_IDLE` busy loop on each detected **E-core**, up for the whole generation and down when it returns. **Measured through the server (row 27): +23.1 % decode tok/s on Qwen3-235B** (0.9229 → 1.1363 tok/s, n=3, greedy ids identical), TTFT −22.3 %; MiniMax-M2.7 +1.3 %, Llama-3.3-70B +2.9 %, Llama-3.2-1B unchanged. `"auto"` = every detected E-core, an int = that many, `False` = off. Costs one E-core's power per spinner, never engine time. A non-hybrid CPU gets no spinner and a printed warning |
| `PIN_ENGINE_THREADS` | `True` | pin the engine's threads to the **P-cores** for the duration of a generation and restore their masks after. Composes with `KEEP_AWAKE` (in-process: −7.6 % time per token on its own, −21.5 % together). Threads started inside a generation inherit the mask, which is how it reaches the io_uring readers. Skipped automatically on a CPU with no P/E split |

> **Not every engine parameter is a server knob, and that is a decision, not an
> oversight.** Nine of `VaayuBaseModel.__init__`'s parameters are deliberately not
> exposed — `dtype`, `prefetching`, `layer_names_dict`, `tokenizer_class`,
> `layer_shards_saving_path`, `profiling_mode`, `compression`, `use_torch_compile`,
> `hf_token`. They are loader plumbing, dev overrides for a checkpoint the loader
> misreads (detection is the supported path), dev instrumentation, two paths
> `__init__` already overrides in places, and HF auth, which belongs in the
> environment. The list is enforced: `tests/test_engine_param_reachability.py` fails
> the suite when a new engine parameter appears with **neither** a knob **nor** a
> written-down decline.


> **`DRAFTER_MODEL_PATH` requires a vocabulary match.** The drafter and target must
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
| `stream` | bool | *server default* (`DEFAULT_STREAM`, **`true`**) | SSE when true. Omitting it means "the server decides" — this server streams, unlike OpenAI's `false` default, because nothing is sent until the last token otherwise. An explicit `false` still wins |
| `stream_options.include_usage` | bool | `false` | adds `usage` to the final chunk |
| `tools` | array | `null` | OpenAI function-tool definitions |
| `tool_choice` | any | `null` | accepted and passed through the template |
| `seed` | int | `null` | |
| `n` | int | `1` | **must be 1.** `n>1` is a 422 with a named reason — the server does not implement multiple completions per request (unrelated to `MAX_BATCH_SIZE`, which batches separate *requests*, not multiple completions of one) |
| `session_id` | string | — | **Vaayu extra, and REQUIRED** (`REQUIRE_SESSION_ID`, default on). Pins this turn to a prefix-cache session. A request naming no session is a `400`; OpenAI's `user` satisfies it |
| `enable_thinking` | bool | *server default* (`ENABLE_THINKING`, **`false`**) | **Vaayu extra.** Let the model reason before answering. Suppressed by default; set `true` per request to get it back |
| `user` | string | — | OpenAI's end-user tag. Used as `session_id` when that is absent, so an OpenAI-native client satisfies the requirement with no Vaayu-specific field. `session_id` wins when both are sent |
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

> **Omit it *with* `stream: true`.** A non-streaming response sends nothing until the
> last token, so an uncapped one can sit silent for as long as the cap takes — on a
> reasoning model at ~1.5 tok/s, an 8k window is over an hour, and every HTTP client
> gives up long before that. The server logs a WARNING naming the number of tokens the
> request may run for. Streaming has no such problem: bytes flow from the first token.

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
curl -s localhost:9999/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [{"role": "user", "content": "write a binary search in Python"}]
}'
```

`max_tokens` omitted → the model generates a complete function and stops itself.

### Streaming

```bash
curl -sN localhost:9999/v1/chat/completions \
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
curl -s localhost:9999/v1/chat/completions \
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
curl -s localhost:9999/v1/chat/completions \
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

`session_id` is **required** — this is the shape every request takes:

```bash
curl -isN localhost:9999/v1/chat/completions \
  -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "session_id": "agent-7",
  "messages": [{"role": "user", "content": "…"}]
}'
```

Without one the server answers `400` and says why. An OpenAI-native client that has no
`session_id` can send `"user": "agent-7"` instead — it is used as the session.

The session comes back three ways, so a client can confirm what it was served under:
the `x-vaayu-session-id` header, `session_id` in a non-streaming body, and `session_id`
on **every** streaming chunk.

Every completion — streaming and not — answers with the session it actually keyed on:

```
x-vaayu-session-id: agent-7
```

`(auto)` there means **no usable id arrived** and the pool fell back to scanning. Unknown
request fields are ignored rather than rejected (off-the-shelf clients depend on that), so
a misspelled `sessionId` is silently dropped — this header is the only way to see it from
outside the process.

### Watching the cache and the queue

```bash
curl -s localhost:9999/health
curl -s localhost:9999/v1/models
```

### With the `openai` Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9999/v1", api_key="not-used")

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

`session_id` is required (see §4). Give each independent agent stream its own: it makes
the match exact instead of a scan. An unrecognised `session_id` still falls back to auto-match,
so it can only help. Check `x-vaayu-session-id` on the response once when wiring a client
up: `(auto)` means the id never arrived.

An OpenAI-native client that has no `session_id` can use the standard `user` field
instead — Vaayu keys on it when `session_id` is absent. It is a coarser key by nature:
`user` names a **person**, not a conversation, so two concurrent streams from one user
share a pool entry and evict each other. For fan-out, give each stream its own
`session_id`.

### 6.3 Concurrency is a knob, not a constant — design your fan-out around what's configured

The server serves **N concurrent requests together as one batch** when
`MAX_BATCH_SIZE >= 2` *and* the loaded checkpoint runs on `sdpa`
(`ATTN_IMPLEMENTATION` resolves to it — see §3). Otherwise it serialises exactly
as before: one generation at a time, FIFO, `QUEUE_DEPTH` deep (default 8), `503`
+ `Retry-After` past that.

Batching is a **ceiling, never a target** — the real width a batch reaches also
depends on VRAM admission at the actual context length, and can still fall back
to serial per-request if that admission refuses at the real width. See
`MAX_BATCH_SIZE` in §3's config table for the measured numbers and their
caveats before you plan capacity around it — the one number on record so far is
engine-level, one model, one width, with the prefix cache off (below).

If your agent issues parallel sub-agent calls and `MAX_BATCH_SIZE` is left at its
default (`1`), they will still **serialise** — accept that, or cap your own
concurrency to 1 and save the round trips. With `MAX_BATCH_SIZE >= 2` on an
`sdpa` load, up to that many will run together instead. Handle `503` with a
retry — do not fail the tool call.

**Requests batch only if they are waiting at the same moment**, and that window
is `BATCH_WAIT_MS` wide (default 50 ms). Two consequences for a client:

* Fan out in a **burst**. Requests issued together land together. Issuing them
  one-per-completed-tool-call spreads them past the window and each one runs
  alone, even at `MAX_BATCH_SIZE = 4`.
* A request already **running** cannot be joined. There is no continuous
  batching here: a batch is fixed at formation and runs to the end. So a
  long-running request does not "absorb" a later arrival — that arrival waits
  for the next formation.

### 6.4 Context is the binding constraint, not speed

Check `/v1/models` at startup and budget against the real number:

| checkpoint | derived `max_seq_len` (sdpa) |
|---|---|
| Qwen3-235B | 8192 |
| MiniMax-M2.7 | 8192 |
| Llama-3.3-70B | 8192 |
| Llama-3.2-1B | 90639 |
| Qwen3-Next-80B int4 | 65536 (flash) |

8192 is **tight** for a tool-using agent: system prompt plus a few MCP tool schemas can
be 1.5–3k before any code is read. Keep tool schemas terse, and read files in slices.
`ATTN_IMPLEMENTATION = "flash_attention_2"` derives more context at the same reserve.

A **hybrid-attention** checkpoint derives far more than its size suggests, because only some
of its layers store KV at all: Qwen3-Next-80B has 12 attention layers among 48, so a token
costs a quarter of what the layer count implies. The engine counts those layers off the model
itself — there is no knob, and nothing to set.

### 6.5 MCP belongs in your agent, not here

Vaayu speaks OpenAI `tools` / `tool_calls`. It has no MCP client and will not grow one.
Your agent connects to MCP servers, converts their tool definitions into OpenAI
function schemas, and passes them as `tools`. Results come back as `tool` messages.

### 6.6 Tool-call support is a property of the checkpoint

The call format is detected by **probing the chat template**, never by model name. If
the template has no tool support, a request carrying `tools` is a `400` with a named
reason — deliberately, rather than silently dropping the tools and leaving your agent
waiting for a call the model was never told about. A model that renders its own prompts
instead of shipping a template (DeepSeek-V4 today) refuses every request carrying `tools`
the same way, because its tool-call format is not parsed yet.

Check before you rely on it:

```bash
curl -s localhost:9999/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vaayu","messages":[{"role":"user","content":"hi"}],
       "tools":[{"type":"function","function":{"name":"t","parameters":{}}}]}'
```

A `400` mentioning tool support means: pick a checkpoint whose template renders tools.

### 6.7 Abandoning a request stops it, and streaming buffers tool calls

Content arrives as OpenAI deltas. A **tool call is buffered and emitted once, complete** —
never as partial JSON, because partial tool-call JSON breaks clients that parse each
delta. Do not expect incremental `function.arguments` fragments.

**Closing the connection stops generation** — client timeout, `Ctrl-C`, a cancelled task —
on the streaming *and* non-streaming paths, within about a second. The slot frees for the
next request and the server logs how many tokens it got through.

That matters more here than on a normal server: at `MAX_BATCH_SIZE = 1` (the default),
concurrency is 1, so a generation that outlives its client holds the GPU for the rest of
its cap while everything behind it queues. With `MAX_BATCH_SIZE >= 2` a batch slot is
held instead of the whole GPU, but the principle is the same — an abandoned client still
occupies capacity it is not using. Aborting is also safe for the cache: the prefix entry
is restored on the abort path rather than lost, and the partial KV a cancelled turn
leaves in the pool is real KV the next turn can reuse.

---

## 7. Making it faster

Measured on this box (16 GB VRAM, 32 GB RAM), same-day A/B.

### 7.0 What the defaults are worth, end to end

Measured on Qwen3-235B, 2 concurrent sessions x 3 rounds, ~4.6k-token shared prefix,
24 decode tokens per turn — an agent-shaped workload. Same box, same day. The only
difference between the arms is configuration:

| | server's first working config | shipped defaults | |
|---|---|---|---|
| layer pinning | 0 transformer layers | auto-fill | |
| prefix cache | **off** | on | |
| expert ceiling | 8.5 GiB literal | derived | |
| **total wall, 6 turns** | **477.0 s** | **198.1 s** | **2.41x** |
| end-to-end throughput | 0.302 tok/s | **0.727 tok/s** | 2.41x |
| cold turn | 77.8 s | 41.3 s | 1.88x |
| warm turn | 80.3 s | **28.9 s** | **2.78x** |
| expert cache hit | 46.7 % | 61.3 % | |
| prefix reuse | none | 83 % of prompt tokens skipped | |

In the old configuration warm turns were *slower* than cold ones (80.3 s against
77.8 s): with no prefix cache every turn is a full prefill, so an agent resending its
history was punished for doing the right thing.

**None of this made the engine itself faster except the expert ceiling.** The other
three were the server failing to use capabilities the engine already had. They are
documented below because each is easy to switch back off by accident.

### 7.1 Pin the layers — the big one, and it is now the default

`MAX_PINNED_LAYERS = -1` fills the VRAM budget with transformer layers, which then never
touch the NVMe again. At `0` the budget sits idle and every layer streams per token.

| checkpoint | `MAX_PINNED_LAYERS = 0` | `-1` (default) | gain | what actually pinned |
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

| mode | setting | measured |
|---|---|---|
| drafter | `DRAFTER_MODEL_PATH = "/path/to/Llama-3.2-1B-Instruct"` | **5.57×** on Llama-3.3-70B (86 % acceptance, byte-identical output) |
| prompt lookup | `USE_PROMPT_LOOKUP = True`, `DRAFT_TOKENS = 15` | **1.94×** on the same, zero VRAM |

`USE_PROMPT_LOOKUP` defaults to `"auto"`, which is ON for dense and OFF for MoE. MoE is
excluded on evidence, not caution: prompt lookup measures **1.285× on MiniMax-M2.7** but
**0.836× — 16 % slower — on Qwen3-235B**, and the runtime gate's own TRIAL phase costs a
fixed **0.932×** on short generations. Defaulting it on would hand one actively-tested model
a win and the other a loss. Set `USE_PROMPT_LOOKUP = True` to opt a MoE model in anyway.

Both are **refused at load on MoE** — a 6-position verify pass needs 53.4 GB (Qwen) /
26.4 GB (MiniMax) of expert working set against a 9.1 GB budget. That is a wall, not a
tuning problem.

> The speculative loop is **greedy at every decision point** — `temperature`
> included — and has no stop-string machinery. So it is dispatched **per request**:
> a request asking for `temperature`, `top_p`, `top_k` or `stop` takes the ordinary
> path and gets them. Nothing is lost; that request simply does not speculate. The
> server names the control that sent it there, in the log and in `warnings` on the
> response.

### 7.4 Keep the prefix pool big enough

Reuse does not degrade gracefully. Measured: once the pool is over capacity, reuse
collapses to **zero**, not proportionally, because each recomputed session's re-store
cascade-evicts the ones before it. TTFT went from 9.8 s back to 352 s at one session
past capacity.

If you serve several agent sessions, set `PREFIX_COMPRESS = 9`: pooled KV packs to
~0.61×, so the same RAM holds roughly 1.6× the sessions, at ~0.7 s of TTFT on the hit
path. The first budget-pressure eviction prints a warning naming the capacity — watch
for it.

9 is the highest the quantizer goes. Packing at 9 bits *uniformly* fails the quality bar
on the fused-EXL3 forward against real decode queries, so the engine keeps each layer's 8
highest-variance channels exact and packs the rest — both models then clear the bar with
margin, at 1.211 bytes/element instead of 1.156. It is automatic, with no knob. See
[TUNING.md §2](TUNING.md).

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
| batching beyond `MAX_BATCH_SIZE`'s measured ceiling | not a blanket "won't help": N=2 is **+1.606×** on MiniMax-M2.7 at ~4.15k-token prompts (2026-09-08, gated behind `MAX_BATCH_SIZE` and `sdpa`, **engine-level with the prefix cache OFF** — no server-level A/B run yet) — but N=3 OOMs MiniMax and is slower than N=2 on Qwen3-235B, which still admits N=1 only. See `MAX_BATCH_SIZE` in §3's config table |
| smaller `PREFILL_CHUNK_SIZE` | costs TTFT (+24.7 % MoE, +213 % dense at 128), buys nothing measurable |
| expert pinning | refuted: eviction share is linear in N and always 2–4× below the pin's cost |
| TurboQuant KV in the live cache | failed its quality gate 2–7× on both models at 4 bits |
| speculative decoding on MoE | refused by the working-set gate, before the drafter loads. The gate's bound is 24–51 % (Qwen) / 14–35 % (MiniMax) pessimistic — measured 2026-09-05 — but the real union still buys at most ~1.3× fewer expert routes per token, best case, against a drafter that takes 3.09 of 6.3 GB of expert VRAM cache. EAGLE-3 would not lift it |
| `fp8` KV to save VRAM | `fp8` is **1.5× larger** than bf16 here — it keeps a dequantized mirror |

---

## 8. Limits, and what is deliberately not here

- **One model per process**, no hot-swap.
- **One GENERATION runs at a time**, FIFO queue, `n=1` only — but since P10 that
  generation may be a batch of `MAX_BATCH_SIZE` requests (default `1`, today's
  behaviour) dispatched together through `batched_generate` rather than one
  request. See `MAX_BATCH_SIZE` in §3.
- **No continuous batching / paged attention** — batching here is static per
  round, not token-level continuous batching or paged KV.
- **No `/v1/completions`, no FIM**, no Anthropic Messages wire format.
- **No MCP client** — that is the agent's job.
- **No auth, no TLS, no rate limiting.** Bind to `127.0.0.1` (the default) or put a
  reverse proxy in front of it. Do not expose this directly to a network you do not
  control.
- **Prefix caching is disabled under `KV_CACHE_MODE = "fp16"`.** The bare `DynamicCache`
  that mode returns has no `to_cpu`/`to_device`/`resident_bytes`, so the pool is disabled
  at load with a warning and `/health` reports `"prefix_cache": "disabled"`. The default
  `"auto"` resolves to `"bf16"` on every checkpoint, EXL3 or not, and keeps it on —
  `"bf16"` is the ring buffer holding K/V in the model's own dtype, not a cast.

---

## 9. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `"prefix_cache": "disabled"` in `/health` | `KV_CACHE_MODE = "fp16"` | set it back to `"auto"` (or `"bf16"`) |
| generation is ~10× slower than expected | `MAX_PINNED_LAYERS = 0` | set it back to `-1`, the default |
| `400` naming a token count | prompt exceeds the derived window | shorten by the number the message gives |
| `400` mentioning tool support | the checkpoint's template cannot render `tools` | use a tool-capable checkpoint |
| `400` naming `encoding/encoding_dsv4.py`; a startup WARNING naming the same file | a DeepSeek-V4 download without its `encoding/` folder, or with an encoder that cannot load (the WARNING gives the error). The model ships no chat template; its prompt format lives in that file | download the full repository, `encoding/` included. If the WARNING names a missing module, install it and restart |
| `503` + `Retry-After` | queue full | retry; raise `QUEUE_DEPTH`; reduce client concurrency |
| a non-streaming request never returns; the client times out with no status | uncapped generation on a slow model — nothing is sent until the last token | set `max_tokens`, or `stream: true`. The startup log carries a WARNING naming the token count |
| `422` on `n` | `n>1` requested | this server returns one choice |
| `warnings` on the response naming a sampling param | that request skipped speculation, because the speculative loop would drop the param | nothing to fix — the param was honoured. To speculate instead, send `temperature: 0` and no `top_p`/`top_k`/`stop` |
| a drafter pairing is refused at load | drafter/target vocab sizes differ | pair within one model family |
| startup WARNING: `N persistent buffer(s) have no tensor in this checkpoint` | the converted checkpoint dropped tensors the model was trained with (MiniMax-M2.7's EXL3 build: its MoE routing bias) | run `scripts/fetch_missing_buffers.py --repo <original repo> --checkpoint <model dir>` and restart — see [missing_buffers.md](missing_buffers.md) |
| server exits on startup, no VRAM plan warning | loaded on CPU | check `DEVICE` |
| killed the server, `nvidia-smi` still shows the VRAM | another process holds it — Vaayu's share goes with the process | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` names the real holder |
| `refusing to load -- the GPU is not exclusive` | something else holds GPU memory | run the commands the message prints, then start again. A container needs `docker stop`, not `kill` — `restart=unless-stopped` respawns it |

### Diagnosing without starting the server

```bash
./venv/bin/python -m vaayu.server.diagnose
```

When the server will not start at all, there is no `logs/run-*/env.json` to
read — this prints the same environment snapshot in seconds, without loading
any weights, then flags the settings worth checking on this box:

- **NVMe APST is active.** Minor, reported for completeness: measured with no
  decode or TTFT cost, because decode keeps the drive's idle gaps shorter than
  APST's sleep timeout.
- **Another process holds GPU memory.** Budgets are sized once at load, so
  this is subtracted from the expert cache for the whole run even if the
  process exits later.
- **The NVMe I/O scheduler is not `none`.** Reads are already O_DIRECT +
  io_uring; a scheduler in front of them adds latency without reordering
  anything useful.
- **A drafter is configured against an MoE checkpoint.** Speculative decoding
  is refused on MoE targets. Since 2026-09-05 the refusal happens *before* the
  drafter is loaded, so the load is no longer paid for and discarded; the
  setting is still doing nothing and is worth removing. This check reads the
  checkpoint's own
  `config.json` for an expert count, since `diagnose` never loads weights and
  so can't ask the loaded model whether it is MoE.
- **`DEFAULT_STREAM` is `False`.** A client that sends no `max_tokens` then
  gets nothing until the whole answer is generated.

A box with nothing to flag prints `nothing to flag on this box.` rather than a
banner — a check that always has something to say stops being useful.

### Is speculative decoding actually running?

Two places answer it without reading a startup line that has already scrolled
away: `GET /v1/models` carries a `speculative` object, and `env.json` in the run
directory carries the same one.

```json
"speculative": {
  "requested": "<drafter path, \"prompt_lookup\", or null>",
  "draft_tokens": 5,
  "active": false,
  "refused_because": "speculative verify working set 53.4 GB ... exceeds ..."
}
```

`active` is the only field that says whether speculation is running;
`requested` says what was asked for. On an MoE target the two normally
disagree, and `refused_because` says why with the numbers the decision used.

There are three gates behind that field, and any of them may write it:

- **before anything is built**, on whether the model can be checked at all: a
  MoE checkpoint loaded on a non-CUDA compute device has no expert geometry, no
  expert cache and no routing history, so the two byte gates below and the
  runtime gate all skip themselves. Speculation there would run checked by
  nothing, for the life of the process, and report nothing — so it is refused
  instead. Load the model on CUDA to speculate on it, or set
  `VAAYU_SPEC_FORCE=1` to run it unchecked deliberately. Dense models and MoE
  on CUDA are unaffected;
- **before the drafter loads**, against an *upper bound* of the expert cache
  budget — this is the one that avoids loading a 3 GB drafter only to discard
  it (measured on Qwen3-235B: **3.85 s** of load, every start);
- **after the load**, against the real budget, unchanged from before.

The bound can only over-state the budget, so the early gate never refuses a
configuration the later one would accept. That is also why a refusal that names
"the largest draft_tokens that could fit" says *could*: the value fits the
bound, and the post-load gate re-tests it against the real budget.

A third gate runs at generation time on MoE models, because the first two can
only see bytes. It decodes normally for the first few steps while watching the
model's real expert routing, admits speculation only if a verify pass fits the
expert cache, then runs a trial that alternates drafting and non-drafting
rounds: it measures acceptance over ~48 drafted tokens against a break-even
derived from that same routing, and compares the seconds per token of the
drafting rounds with the rounds next to them that did not draft. Below the bar,
or slower per token, it disables speculation for the rest of the process and
records why in `calibration`.

Measured 2026-09-05, re-run against the final code (the first measurements
were taken against a gate that rebuilt itself every request, so they did not
describe what ships): on MiniMax-M2.7 with thinking suppressed the gate
self-admits — no measurement hatch — on a real measured union of 12.45 GB
against a 14.37 GB budget, sees 1.00 acceptance against a 0.61 break-even,
and keeps a **1.307×** win (1.7795 → 2.3253 tok/s), ids byte-identical to
plain decode.

On Qwen3-235B, whose answers open with prose, the same shape of
configuration measures **0.836×** (16 % slower) with no gate at all. With
the gate it correctly detects the low acceptance — 0.351 against a 0.528
break-even, derived from that run's own routing — and latches off. **That
is not free.** Finding out cost 37 drafting rounds, and the gated run lands at
**0.932×** (1.080 → 1.006 tok/s), not parity.

**Read that 6.8 % as a fixed cost divided by a short benchmark, not as a rate.**
The gate pays for exploration exactly once per process — the decision latches,
and an `off` decision removes the speculator entirely, so every later request
runs the ordinary path. Only rounds where the proposer actually drafted are
paid for: calibration steps, and trial rounds where no n-gram matched, verify a
single position and cost precisely what plain decode costs (27 of the 64 rounds
above were free for that reason). The whole exploration is worth

    (drafted tokens) x (break-even acceptance - measured acceptance)

token-equivalents of decode, at most `48 x break-even` when nothing is accepted:
**25.3 tokens on Qwen3-235B, 29.3 on MiniMax-M2.7**. Both numbers come from the
model's own routing — there is no per-model constant behind them — and both are
already in the `calibration` object below, so you can compute your own load's
figure from what the server reports.

Divided over a process that generates `T` tokens, that is `6.8 % x 128/T`:
**under 1 % once the process has produced ~870 tokens, under 0.1 % past ~8 700.**
A long-lived server, or an agentic session that reuses one process, is far past
both. The 6.8 % is what it costs a 128-token benchmark, and quoting it as the
cost of running the gate is wrong. Where it does bite is a process that starts,
answers once, and exits — there, a configuration you already know does not pay
is still better switched off by hand.

The exploration is deliberately not made shorter. The latch is permanent, so a
wrong decision is permanent in both directions — forfeiting a measured 1.307×,
or keeping a measured 0.836× — and deciding on fewer drafted tokens makes both
likelier. Trading a bounded one-time cost for an unbounded permanent one is the
wrong way round.

### Turning it on

```python
# vaayu/server/config_local.py
USE_PROMPT_LOOKUP = True     # zero VRAM; the runtime gate decides whether it runs
DRAFT_TOKENS      = 3        # measured best on MiniMax-M2.7
```

Nothing else. `ENABLE_THINKING` already defaults to `False`, and that default is
load-bearing here rather than cosmetic: both MoE templates end their assistant
turn with a hard-coded `<think>` and ignore the `enable_thinking` flag, so
`render_prompt` closes the block for them. A request that sends
`"enable_thinking": true` gets a prose answer, prose does not match n-grams, and
the gate will correctly latch off.

Three conditions have to hold together, and they are not independent:

| condition | why | if it does not hold |
|---|---|---|
| MiniMax-M2.7, not Qwen3-235B | Qwen opens its answers with prose | the gate latches off, once, after paying ≤25 token-equivalents of decode (6.8 % of a 128-token run; under 1 % of a ~900-token process) |
| thinking suppressed | the copy phase never starts inside a think block | same — the gate latches off |
| copy-heavy work (re-emitting a file with an edit) | PLD drafts from n-grams already in the context | no matches, no speedup, near-zero overhead |

Confirm what actually happened with `GET /v1/models` or `logs/latest/env.json`:
the `speculative.calibration.decision` field reads `run`, `off_acceptance`,
`off_union`, or `off_no_routing` (that last one means the gate saw no routing
evidence at all — a bug worth reporting, not a normal outcome).

A weighted drafter (`DRAFTER_MODEL_PATH`) remains refused on MoE: it costs
3.09 GB of VRAM the expert cache needs, before the acceptance question is even
reached. This section is about prompt lookup only.

### The on-device gate

`tests/test_server_smoke.py` -- restored 2026-09-04 (was deleted 2026-09-03,
commit `7d806e7`, along with the other on-device gate scripts; that gap is why
this one is committed rather than left as a scratchpad script). Skipped unless
`SERVER_CKPT` is set:

```
SERVER_CKPT=<checkpoint dir> \
  systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s
```

`-s` is required, not optional. Several of these assertions read `server.log`,
and pytest's default capture rebinds `sys.stdout` — so the engine's `print()`
output never reaches fd 1, where `capture_stdio`'s pump tees it from. Without
`-s` the log holds only the native fd-1 writes (`[vaayu_rust]` lines) and the
byte-counter check fails as if the counter were broken.

The module now refuses to run in that state rather than mis-reporting it: a
module-scoped guard reads pytest's capture mode and, unless it is `no` (what
`-s` sets), **fails** every test in the file with a message naming `-s`, before
any checkpoint is loaded. It fails rather than skips deliberately — a skipped
gate is a gate nobody notices is not running. Only `-s` passes the guard;
`--capture=tee-sys` does not, despite the name — measured through a live pump,
it hides Python prints from fd 1 exactly like `fd` and `sys` do.

It runs a real load, non-streaming, streaming (checking `finish_reason` and
`completion_tokens` on both), an oversized-prompt refusal on both paths, a
check that prefix reuse actually **fired**, `env.json` is clean, `requests.jsonl`
has one row per request including the refusals, no absolute path leaked into
either file, and that the token summary's `gb_tok=` figures are consistent with
what the run's own `[Vaayu-Pin]` output says it did -- nonzero if some layers
stream, zero if every transformer layer was VRAM-pinned (a checkpoint small
enough to fully pin, e.g. Llama-3.2-1B on a 16 GB box, legitimately reads
nothing from NVMe per token; that is not a bug). It does **not** run a
teacher-forced perplexity check -- see `todo.md`'s on-device gates section for
that gate's status. See `todo.md`'s on-device gates section for the full
command per checkpoint and expected run time.
