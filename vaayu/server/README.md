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

**Prompt format.** Every model is prompted through its tokenizer's own `chat_template`, except a
checkpoint that ships none on purpose: **DeepSeek-V4** is prompted by DeepSeek's own
`encoding/encoding_dsv4.py`, read in place from the checkpoint — download the **full** repository,
`encoding/` included, or every chat request is refused with a message naming the missing file.
Nothing is written into the model directory for this. A request carrying `tools` is refused for
DeepSeek-V4 (its tool-call format is not parsed yet).

## Speed

Defaults are already the fast ones, and they differ deliberately from `vaayu_base`'s
library defaults: a server holds one model for its whole life and should fill the box.

| constant | default | what it does |
|---|---|---|
| `MAX_PINNED_LAYERS` | **`-1`** | `-1` auto-fills the VRAM budget with transformer layers. `0` pins the shared layers only, so **every transformer layer streams from NVMe on every token** while the pinning budget sits idle. Measured cost of getting this wrong: **13× on Llama-3.2-1B, 2.0× on MiniMax-M2.7, 1.27× on Llama-3.3-70B**. `N` pins the first `N` |
| `ENABLE_PREFIX_CACHE` | **`True`** | cross-turn KV reuse. Leave it on for an agent workload — it is the largest measured lever here (**136–377× TTFT**). A model whose KV cache cannot be cropped (hybrid / linear-attention models, DeepSeek-V4) reuses only a turn that **appends** to the previous one — see documentation/server.md §6.2. On DeepSeek-V4, reuse needs thinking off (`ENABLE_THINKING = False`, the default, and no per-request `"enable_thinking": true`): with thinking on, the checkpoint's own prompt encoder re-renders the previous assistant turn without its reasoning, so the new prompt no longer contains the stored entry and every turn prefills in full; `ENABLE_PREFIX_CACHE = False` then avoids parking a session for nothing. |
| `PREFILL_CHUNK_SIZE` | `512` | tokens per prefill chunk. 512 is measured best; smaller costs TTFT (**+24.7 % MoE, +213 % dense at 128**) and buys nothing. On a hybrid (linear-attention) model the recurrent layers carry their state across chunks (gated-delta-net families such as `qwen3_next`; before 2026-09-18 they restarted at every chunk). A hybrid whose mixer Vaayu cannot continue refuses a multi-chunk prefill with an error; set this to `0` for such a model. `0` also makes a model with windowed layers (gpt-oss-20b) store them at full length (no window-length saving). |
| `PREFIX_COMPRESS` | `None` | pack pooled KV at N bits (**9**, which is both the recommended setting and the quantizer's ceiling) so the same RAM budget holds ~1.6× the sessions. 9 bits *uniformly* fails the quality bar on the fused-EXL3 forward, so the engine keeps each layer's 8 highest-variance channels exact — 1.211 B/elem, both models under the bar, no knob (`documentation/TUNING.md` §2). Worth it when serving several agent sessions: reuse collapses to **zero, not proportionally**, once the pool is over capacity. Ignored, with a line at load saying so, on a model whose KV cache is servable at one length only (hybrid / linear-attention models, DeepSeek-V4) and on a model with windowed attention layers (gpt-oss-20b), whose live cache keeps those layers at window length. |
| `PREFIX_SESSIONS` | `3` | concurrent sessions the prefix pool must hold before the MoE expert cache claims the rest. Lower = faster decode, fewer cached sessions. Measured at 1: **+5.7 %** for one session at 2.7k context, **−8.9 %** for two sessions at 7.4k (row 31) — see `documentation/TUNING.md` §3b. Reserves the same way on a hybrid model (e.g. Qwen3-Next) since 42-B — the pool holds the recurrent state alongside the KV, ~4.83 GB reserved at this setting, not nothing |
| `KV_CACHE_MODE` | `"auto"` | `auto` picks bf16 on every checkpoint (the ring buffer, in the model's own dtype). `"fp16"` disables the prefix cache but skips the ring buffer's doubling, which can cost up to ~2× derived context (Llama-3.2-1B on flash: 65536 under `auto`, 91106 under `"fp16"`). `"fp8"` is **1.5× larger**, not smaller — it keeps a dequantized mirror |
| `ATTN_IMPLEMENTATION` | `None` | `None` **prefers `flash_attention_2`** when `flash_attn` imports, then **sdpa**, then eager — it does **not** mean sdpa. Every shipped `max_seq_len` figure was verified on sdpa; `"flash_attention_2"` derives a larger context at the same reserve. Batching (`MAX_BATCH_SIZE` below) needs `"sdpa"` |
| `MAX_BATCH_SIZE` | `1` | Requests served together as one batch. A **ceiling, never a target** — max width is model- and context-dependent and is **not 3**. Measured 2026-09-08 at ~4.15k-token prompts, **engine-level (`batched_generate` called directly, not through the server) and with the prefix cache OFF** — no server-level A/B has been run, and the server defaults `ENABLE_PREFIX_CACHE = True`: `2` is **+1.606x** on MiniMax-M2.7 (TTFT 317.2 → 223.3 s, peak VRAM *lower* batched); `3` OOMs MiniMax and is **slower** than 2 on Qwen3-235B. Qwen admits N=1 only today, so this buys it nothing. **Inert unless `ATTN_IMPLEMENTATION = "sdpa"`** |
| `BATCH_WAIT_MS` | `50` | milliseconds a forming batch waits for company before running what it has. **Without it `MAX_BATCH_SIZE` never batches two concurrent clients at all** — a request wakes the dispatcher the instant it is queued, so the first of two simultaneous clients runs alone while the second is still being tokenized (measured on Llama-3.2-1B through the real server: 2 clients, warm, `batch=None` on both requests, 3 runs of 3). It also un-sticks two **steady** clients, which otherwise alternate one-at-a-time forever and never meet in the queue. Cost of getting it wrong is one-sided and bounded: a lone request at an idle server is delayed by this much, once — 0.02 % of MiniMax-M2.7's ~220 s TTFT at 4.15k tokens, ~4 % of a Llama-3.2-1B request. A batch that fills stops waiting immediately. `0` restores the pre-2026-09-09 behaviour. **Inert at `MAX_BATCH_SIZE = 1`.** Does **not** help a client whose turnaround exceeds the window — joining a *running* batch is continuous batching, which Vaayu does not do |
| `RAM_CACHE_BUDGET_GB` | `None` *(derived from total RAM)* | GB of **pageable** RAM for the dense layer cache. `None` derives it from TOTAL RAM so throughput is reproducible across restarts — keying off momentary free RAM swung Llama-3.3-70B decode **38 %** run to run. `0` disables it. **No effect on MoE**, whose experts live in the pinned expert RAM cache. On a dense model it also decides buffered vs O_DIRECT reads: the load charges this cache against free RAM before asking whether the page cache can hold 40 % of the checkpoint. Uncharged (before 2026-10-07), Llama-3.3-70B went buffered and decoded **21 %** slower (2.48 vs 1.96 s/token); `0` brings that old choice back |
| `ENABLE_KV_PERSIST` | `False` | persist prefix KV to disk so reuse survives a restart. Ignored, with a line at load saying so, on a model whose KV cache is servable at one length only (hybrid / linear-attention models, DeepSeek-V4) and on a model with windowed attention layers (gpt-oss-20b), whose live cache keeps those layers at window length. |
| `KV_PERSIST_DIR` | `None` *(`~/.cache/vaayu/kv`)* | where those entries are written. Until 2026-09-10 the feature above could be turned **on but not aimed**. The write budget is **derived as half the free space of whatever filesystem this names**, so this choice sets the cache's *size* as well as its location — a small or shared partition silently caps how much reuse survives a restart |
| `KV_PERSIST_SYNC` | `False` | `fsync` each persisted entry before returning. `False` leaves the write in the page cache: a clean shutdown keeps it, a power loss or panic can leave a **torn entry** read back at the next boot. `True` costs a disk flush on the prefix-store path. Inert while `ENABLE_KV_PERSIST` is `False` |
| `AUTO_PREFIX_MATCH` | `True` | let a request naming **no session** still find a prefix entry, by matching the longest shared prompt prefix instead of an exact key. This is the behaviour `REQUIRE_SESSION_ID` above describes; the server could not turn it off before 2026-09-10. `False` gives an unnamed request **no reuse at all** rather than approximate reuse — on this workload the larger loss. Changes nothing for a request that names its session |
| `MAX_SEQ_LEN` | `None` *(derived)* | context to plan VRAM for. `None` derives the largest context whose predicted peak fits the reserve floor — the shipped path, and the one every measured figure here was taken at. The one reason to set it: a **shorter** context hands the difference back to the expert cache, the decode bottleneck on both MoE targets. Both failure directions are loud, not silent — above the box's `S_max` the load **refuses** (`VramPlanError`, naming `S_max`); between the derived point and `S_max` it loads while the larger KV reserve eats the pinning budget, and at `0` the engine warns that every layer now streams from disk on every token. On a model with windowed attention layers (gpt-oss-20b) that runs an elastic expert tier, `None` derives a LARGE context (since S50, 2026-10-13: the full-length-parity pricing is dropped so the context grows instead of protecting a static tier) — `None` → **~39,420 tokens** (+19 % over the pre-S50 33,213) with a small static base (~27 experts) and the elastic tier lending the rest up for shorter requests (8k decode ~45 ms/token, logits bit-identical to stock). The derived context carries a 0.30 GiB admission margin below the raw planner maximum so the full window is serviceable, not just advertised. Since 2026-10-10 the VRAM tier on such a model is sized to the request that is running — **no knob**, with `max_tokens` set or omitted — so those tier sizes are what the tier holds while a request runs at the full window, not at 8k: with `None`, 8k decode is 44.4 ms/token (56.9 with the tier held static in the same runs), and with `40442` it is 45.0 (84.0 static) — a longer plan now costs only the requests that run near its full window (two at 40,374 + 68 tokens: 120 ms/token on both arms, with 30–70 MB of the card left). A request that needs more of the reserve than the last one empties the tier before its prefill (see `documentation/TUNING.md`, "The expert VRAM tier follows the running request") |
| `KEEP_AWAKE` | `"auto"` | hold the CPU package out of deep idle while a generation runs: a `SCHED_IDLE` busy loop on each detected **E-core**, up for the whole generation (prefill included) and down when it returns. This box idles in `C2_ACPI` (127 µs declared exit latency) with the CPU ~7.5 % busy, and Qwen's fused-EXL3 forward *sleeps* on ~800 read completions per token — it pays that wake latency on nearly every one. Measured **in-process** (row 26 arms, Qwen3-235B, greedy ids identical): off **1.147–1.190 s/token**, one spinner **0.970 (−15.5 %)**, all eight **0.901 (−21.5 %, = +27 % tok/s)**. MiniMax-M2.7 gains **−2.6 %** — its synchronous copies already spin-wait inside `cudaMemcpy`. `"auto"` = every detected E-core, an int = that many, `False` = off. Cost of leaving it on: one E-core's power per spinner (~1 MB RSS each), never engine time — `SCHED_IDLE` runs only when the CPU would otherwise idle. **A non-hybrid CPU gets no spinner and a printed warning**: every core a spinner could take there is an SMT sibling of one the engine wants, and that arrangement SLOWED MiniMax **27 %**. **Measured through the server (row 27): +23.1 % decode tok/s on Qwen3-235B** (0.9229 → 1.1363 tok/s, n=3, greedy ids identical), TTFT −22.3 %; MiniMax-M2.7 +1.3 %, Llama-3.3-70B +2.9 %, Llama-3.2-1B unchanged. The figures above are in-process |
| `PIN_ENGINE_THREADS` | `True` | pin the engine's threads to the **P-cores** for the duration of a generation, and restore their masks after. Composes with `KEEP_AWAKE`: **−7.6 %** on its own, **−21.5 %** together. Threads started inside a generation inherit the mask, which is how it reaches the Rust io_uring readers. A thread the kernel refuses to move is printed by name — **0 of 66 were refused here**. Skipped automatically on a CPU with no P/E split |
| `DECODE_TOP_K` | `None` | route this many experts per token at **decode** instead of the checkpoint's own `num_experts_per_tok`. Prefill is never truncated (it routes ~every expert anyway — 224.23/layer measured). **This is the only LOSSY knob in this file: it changes the model's output by design, so greedy ids diverge and that is not a bug.** `None` = the checkpoint's value; **`"auto"` measures this checkpoint once at first load (a fixed k if one fits the budget, otherwise a per-token routing-mass cutoff), applies the fastest setting within `DECODE_TOP_K_BUDGET` and caches it per checkpoint** — the first load of each checkpoint is slower (≈6.5 min Qwen3-Next, ≈21 min Qwen3-235B, ≈26 min MiniMax) and no endpoint answers meanwhile; `/health`'s `decode_top_k` shows the active pick and its measured cost; a value above the checkpoint's own top_k is clamped down with a warning, and an active clamp WARNS at load that output is degraded by design — it is the only knob here whose wrong value fails silently. **MEASURED COST OF GETTING IT WRONG — and the right value is per-MODEL, so measure your own checkpoint with `scripts/decode_top_k_ppl.py` before setting it.** Qwen3-Next-80B int4 (top-10 of 512): `9` → **+17.0 % tok/s for +1.47 % perplexity** (the measured sweet spot), `8` → +29.5 % for +3.43 %, `6` → +67.3 % for +8.97 %. Qwen3-235B EXL3 (top-8 of 128): `7` → **+3.40 %** perplexity, `6` → +5.33 % (tolerance only; throughput not measured). MiniMax-M2.7 (top-8 of 256): as a fixed int, leave it off — its mildest step `7` costs **+6.34 %** perplexity, and `6` costs **+19.30 %** for only +26.0 % tok/s; **`"auto"`** picks the routing-mass cutoff p=0.90 instead (**+8.4 %** tok/s for +2.25 %). Perplexity leads visible damage by a wide margin: at k=4 Qwen3-Next still wrote a correct Dijkstra while MiniMax emitted degenerate repetition, so do not certify a value by reading one generation |
| `DECODE_TOP_K_BUDGET` | `0.035` | the perplexity cost `DECODE_TOP_K="auto"` may spend, as a fraction (0.035 = 3.5 %). **Measured picks at 3.5 %:** Qwen3-Next-80B int4 → fixed k=8, **+23.1 % tok/s** for +3.43 %; MiniMax-M2.7 → routing-mass cutoff p=0.90, **+8.4 %** for +2.25 % (no fixed k fits its budget). At 2 %: Qwen3-Next → k=9, +17.0 % for +1.47 %. The budget is a PERPLEXITY budget: perplexity leads visible damage by a wide margin. Changing it re-measures on the next load (the cache is keyed on it). Ignored, with a note, for an explicit int |
| `ENABLE_EXPERT_USAGE` | `False` | record which experts are routed and which are evicted, for a triage bundle. **MoE-only by construction** — on a dense checkpoint the engine warns that no tracker will be built and does nothing (there are no experts to count, not a dense second-class default). Off because its cost is **unmeasured**: it puts `track_evicted_keys` on the expert-cache path, which decode hits on every layer visit. Measure your own decode both ways before leaving it on |

> **Not every engine parameter is a server knob, and that is a decision, not an
> oversight.** Nine of `VaayuBaseModel.__init__`'s parameters are deliberately not
> exposed — `dtype`, `prefetching`, `layer_names_dict`, `tokenizer_class`,
> `layer_shards_saving_path`, `profiling_mode`, `compression`, `use_torch_compile`,
> `hf_token`. They are loader plumbing, dev overrides for a checkpoint the loader
> misreads (detection is the supported path), dev instrumentation, a lossy option
> that only works on an unquantized dense checkpoint (`compression`: `__init__`
> refuses it with a warning on a quantized, MoE or originals-deleted one, which is
> every production model this server is tested with; only the small unquantized
> Llama-3.2-1B smoke checkpoint keeps it), a flag `__init__` refuses
> with a warning because it crashes on CUDA (`use_torch_compile`: its cudagraphs
> are replayed once per layer with no step boundary; without them it measured
> 24–35 % slower on Qwen3-Next), and HF auth, which belongs in the environment. The list is
> enforced: `tests/test_engine_param_reachability.py` fails the suite when a new
> engine parameter appears with **neither** a knob **nor** a written-down decline.


## Speculative decoding

Opt-in. A **weighted drafter** is still refused at load on a MoE model whose verify
pass cannot fit the expert cache — it has VRAM and load time to lose. **Prompt lookup
has neither and is not refused**: it is admitted at load and decided at runtime by the
adaptive gate, from that model's own routing and acceptance.

| constant | default | what it does |
|---|---|---|
| `DRAFTER_MODEL_PATH` | `None` | small drafter model. Measured **5.57×** on streamed dense. **Refused, with a WARNING naming the reason, on a model whose layers keep recurrent state** (hybrid / linear-attention families such as `qwen3_next`): a rejected draft cannot be rolled back out of a recurrent state, and before this refusal Qwen3-Next emitted garbage from the first drafting round. `VAAYU_SPEC_FORCE` does not lift it. |
| `USE_PROMPT_LOOKUP` | `"auto"` | n-gram speculative decoding, **zero VRAM**. `"auto"` resolves at load, where MoE-ness is known: **ON for dense** (measured **1.94×** streamed, strongest on code, where identifiers and boilerplate repeat), **OFF for MoE**. `True`/`False` always win. **Refused, with a WARNING naming the reason, on a model whose layers keep recurrent state** (hybrid / linear-attention families such as `qwen3_next`): a rejected draft cannot be rolled back out of a recurrent state, and before this refusal Qwen3-Next emitted garbage from the first drafting round. `VAAYU_SPEC_FORCE` does not lift it. |
| `DRAFT_TOKENS` | `5` | candidates per speculative round. `5–8` with a drafter, `10–15` with prompt lookup |

> **Measure before opting a MoE model in — it is workload-dependent, not a property
> of the model.** Qwen3-235B's gate **refuses to draft at all** (`off_union`: observed
> verify union 19.31 GB against a ~17 GB budget, 0 drafting rounds), so the switch buys
> it nothing. MiniMax-M2.7 was measured both ways on one 128-token code prompt
> (2026-09-07): at **server defaults** (`ENABLE_THINKING = False`) acceptance is **0.14**
> against a 0.6259 break-even, so the gate latches `off_acceptance` after roughly one
> call — costing ~10 % on that call and nothing after it; with **thinking on** acceptance
> is 0.6875 and the gate says `run`, yet it still **loses 12.9 % total / 16.0 % decode**.
> An earlier arm on a different prompt measured **1.307×**.
>
> **A `run` decision is not a promise of a speedup.** The gate's break-even is
> denominated in *expert bytes*, so it can admit a configuration that loses wall-clock.
> Time your own workload both ways before leaving it on.
>
> Watch which one you got in `logs/latest/server.log`:
> `[Vaayu-Spec] ... DISABLING speculative decoding for this MoE model on measured
> evidence: {...}` is a refusal, and `/v1/models` reports the same under
> `speculative_status.calibration`.

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

`True` is not enough on its own: it is true for a one-token match. `requests.jsonl`
carries `prefix_match_tokens` beside `prompt_tokens` — a turn that matched 44 of
7,517 tokens is paying nearly a full prefill and still logs `prefix_hit=True`.

At boot, `[Vaayu] WARNING: N persistent buffer(s) have no tensor in this checkpoint` means the
checkpoint dropped part of the model — MiniMax-M2.7's EXL3 build ships no MoE routing bias, so it
routes with zeros. There is no knob: fetch the missing tensors with
`scripts/fetch_missing_buffers.py --repo <original repo> --checkpoint <model dir>` and restart; the
file it writes is applied on load (`Loaded N missing buffer(s)`), and deleting it undoes it. Why it
matters and what it measured: [documentation/missing_buffers.md](../../documentation/missing_buffers.md).

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
