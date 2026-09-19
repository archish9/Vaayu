# Vaayu Tuning Guide — what to set, and when

Every knob below is off or conservative by default. This page says which ones to
turn on for which workload, and — just as important — which ones do nothing today
so you don't waste time on them.

**Don't want to read this?** Run the advisor. It derives everything from your
model's config and your machine, and prints the settings to copy:

```bash
source venv/bin/activate
K_SESSIONS=3 TOKENS_PER_SESSION=8000 python tune_advisor.py
```

Load the model once first and read the real pool budget off the startup line
`[Vaayu] cross-turn prefix cache ENABLED: X.XX GB budget`, then pass it as
`POOL_BUDGET_GB=X.XX`. Without it the advisor guesses from free RAM before load,
which overstates the budget several-fold and under-recommends compression.

---


> **Running the server?** These knobs are all set in `vaayu/server/config.py` — the server
> takes no CLI flags. See [**server.md §3**](server.md#3-configuration-reference) and
> [**vaayu/server/README.md**](../vaayu/server/README.md).
> The server's defaults differ from the library's on purpose: `max_pinned_layers` is `-1`
> (not `0`) and `enable_prefix_cache` is on (not off), because a server holds one model for
> its whole life and should fill the box. Getting the first one wrong costs **13x** on
> Llama-3.2-1B, **2.0x** on MiniMax-M2.7 and **1.27x** on Llama-3.3-70B, measured — the gain
> scales with how much of the model fits in the pinning budget.
>
> The dense **RAM** layer cache is likewise derived from TOTAL RAM rather than momentary
> free RAM, so a server's throughput is reproducible: the old form swung Llama-3.3-70B
> between 0.167 and 0.232 tok/s (**38 %**) purely on what else was running at boot.
> Override with `RAM_CACHE_BUDGET_GB`. MoE is unaffected — its experts live in the
> **pinned** expert RAM cache, a different and scarcer resource.

## The 30-second version

```python
model = AutoModel.from_pretrained(
    MODEL_PATH,
    enable_prefix_cache=True,     # yes, almost always
    enable_kv_persist=True,       # yes, if turns should survive a restart
    prefix_compress=None,         # 9 only when the pool is over budget — see below
    kv_cache_mode="auto",         # leave alone
    max_pinned_layers=10,         # leave alone
    prefill_chunk_size=512,       # leave alone
)
```

The only decision that usually matters is `prefix_compress`.

---

## 1. `enable_prefix_cache` — turn it on

**Default:** `False`. **Set it to `True`** if consecutive requests share a prefix
(same system prompt, mostly-unchanged files) — i.e. any agent loop.

This is the single largest lever in the engine. It reuses the previous turn's KV
and prefills only the diverged tail. Measured **TTFT 91 s → 9.6 s** at 2k-token
sessions, and **332 s → 9.8 s** at 8k.

Those two figures are the *over-budget* case — sessions that would otherwise be
recomputed from scratch. An ordinary single-session second turn on a **streamed**
MoE gains less, because the reused prefill still routes ~every expert per layer:
measured on Qwen3-235B, a full hit is worth **−6.6 % of TTFT at 2.7k tokens** and
**−53 % at 7.4k**. The benefit grows with context; it is never negative. Keep it on.

It does nothing if your requests are unrelated to each other; the cache cannot hit
what it has never seen.

## 2. `prefix_compress` — the one real decision

**Default:** `None` (off). **Set it to `9`** when your concurrent sessions no
longer fit the RAM pool.

Pooled KV is TurboQuant-packed on the way into the pool and rebuilt on the way
out — once per turn, never per decoded token. Decode throughput is unaffected
(measured Qwen 0.86 → 0.86, MiniMax 1.53 → 1.56 tok/s).

**When it pays.** Only when **≥2 sessions** together exceed the pool budget:

| situation | setting | why |
|---|---|---|
| 1 session, any length | `None` | the pool always keeps one entry regardless of budget — already cached |
| sessions fit uncompressed | `None` | pure cost (~0.45 s/turn at 2k), no benefit |
| **sessions don't fit, but fit at 9 bits** | **`9`** | **the case it exists for — measured 34× TTFT** |
| don't fit even compressed | `9`, plus reduce load | compression widens the range ~1.65×, it does not remove the wall |

**The failure mode is a cliff, not a slope.** A pool one session short of what you
need evicts each session before its revisit, so reuse collapses to **zero**, not to
a fraction. Measured: bf16 scored 0/4 and 0/3 hits in the two over-budget tests,
never a partial score. This is why the crossover matters more than it looks.

**Use 9, not 8.** At 8 bits both models fail the quality bar against real decode queries
(Qwen 0.0934, MiniMax 0.0546, against 0.05). Anything below 9 prints a warning. 9 is also
the highest the quantizer goes — `bits_total=10` does not exist — so there is no setting
above this one.

**How 9 bits is made to hold up.** The original gate (2026-08-02) could not use real
queries: the persisted KV format stores keys and values but not queries, so the bar was
scored with a K-row query *proxy*. A 2026-09-06 re-run with **real** post-RoPE decode
queries at a ~4.16k-token context found that proxy is 2.5–13.7× optimistic on
Qwen3-235B, where uniform 9-bit packing **failed**. It is not a long-context effect: it
failed at every context length from 256 to 4096.

The engine therefore does not pack uniformly. It keeps each layer's **8 highest-variance
channels** in the cache's own dtype and packs only the remaining 120 — a mixed-precision
outlier split, on by default whenever `prefix_compress` is set:

| model | forward | uniform 9-bit | **what ships: 9 bits + 8 exact channels** |
|---|---|---|---|
| Qwen3-235B | fused EXL3 | 0.0683 — **over the 0.05 bar** | **0.0281 — under it** |
| MiniMax-M2.7 | module-list | 0.0428 — under | **0.0263 — under** |
| bytes/element | | 1.156 (0.578×) | **1.211 (0.605×)** |

That costs about 11 % of the saving and is what makes `prefix_compress=9` usable on a
fused-EXL3 model at all. There is no knob for it: the channel count is a swept parameter
whose sweep is non-monotonic, and at head_dim 128 the neighbouring values cost *more than
bf16* because a 124-wide residual cannot be bit-packed.

One thing this still does not say: **nobody has shown that failing this bar degrades
generated text.** The bar is a deliberately strict proxy for quality, and a 256-token
coherence check passed even at the far worse 4-bit setting.

**What it does not do:** shrink VRAM for the *in-flight* session. The live cache is
always bf16. If a single session's KV won't fit in VRAM, this cannot help — that
needs attention over compressed keys, which is not built.

Full measurements are summarised above; the raw arms live in this repo's working notes
(local working notes, not tracked in the repo).

## 3. `enable_kv_persist` — on, unless disk is precious

**Default:** `False`. Writes the KV to `~/.cache/vaayu/kv` so a session survives a
process restart. Measured **TTFT 27.6 s → 3.0 s** on a restored session. Costs disk
(~130 MB per 700-token Qwen session) and a little CPU. `kv_persist_sync=False`
(the default) is the faster, production-shaped choice.

Not compatible with `kv_cache_mode="turboquant"` — the writer refuses that layout
and says so. It works fine with `prefix_compress`.

## 3b. `prefix_sessions` — leave it at 3 unless you run ONE agent at short context

**Default:** `3`. It is not a limit on how many sessions you may open — it is how many
full-length sessions the engine reserves RAM for **before** the MoE expert cache claims
what is left:

    expert cache ceiling = available RAM − 3.5 GiB − (prefix_sessions × max_seq_len × KV bytes/token)

So lowering it hands RAM to the expert cache, which is the decode bottleneck on a streamed
MoE. On Qwen3-235B the default reservation is **4.74 GB** of a ~20.7 GB budget; at
`prefix_sessions=1` that falls to 1.58 GB and the expert cache grows 12.26 → 15.42 GB.

**It reserves nothing on a model whose prefix cache cannot run.** A hybrid model with
recurrent (linear-attention) layers, such as Qwen3-Next, or any model loaded with
`kv_cache_mode="fp16"`, prints `cross-turn prefix cache DISABLED` at load. The expert
cache then gets that RAM, and `prefix_sessions` has no effect. Before 2026-09-19 the
sessions were reserved anyway: on Qwen3-Next that was 4.83 GB of expert cache held for
nothing, and returning it measured **+17.2 % decode tok/s**.

**Measured on Qwen3-235B, through the server, prefix cache on** (`prefix_sessions=1` vs the
default 3):

| workload | effect of `prefix_sessions=1` |
|---|---|
| one session, 2.7k-token context, two turns | **+5.7 %** wall (decode +6.1 %, TTFT −9.2 %) |
| four sessions, 2.7k-token context | +1.9 % |
| two sessions, **7.4k-token context** | **−8.9 %** |

The last row is the one to take seriously, and it is why the default stays at 3. Cross-turn
reuse does **not** degrade gracefully: one session past the pool's capacity it goes to
**zero**, because each recomputed session's re-store cascade-evicts the ones before it. At
7.4k tokens the second turn's prefix match fell from **7422 of 7517 tokens to 44**, and TTFT
went from 25.0 s back to 51.3 s — which costs far more than the extra expert cache pays.

**Set `prefix_sessions=1` only if all of these hold:** one agent session at a time, contexts
of a few thousand tokens, and no need for a second session's turn-2 to be fast. Then it is
worth about +6 %.

**How to tell you got it wrong** — the engine says so itself, at load and at run time:

    [Vaayu] cross-turn prefix cache ENABLED: 2.03 GB budget (free RAM 5.79 GB - 3.76 GB reserve)
    [Vaayu] WARNING: prefix pool over capacity — evicting under budget pressure. Budget 2.04 GB holds 3 sessions

If that WARNING appears, your reuse is gone, not reduced. **Do not trust `prefix_hit` in
`requests.jsonl` for this** — it is `True` even for a 1-token match, so it reads as a HIT in
exactly the runs where reuse has collapsed. Read the `[Vaayu-Prefix] HIT … match_len=X/Y`
line instead: what matters is X, not that it says HIT.

## 3c. `max_seq_len` — lowering it does NOT buy VRAM

**Default:** `None` (derive). It is tempting to ask for a shorter context to free VRAM. It
does not work: the VRAM reserve is `max(predicted peak, 3.5 GiB floor)` and on both on-device
MoE models it already sits **on the floor** at the derived 8192, so requesting 4096 returns
**zero** VRAM and does not lift the `Restricting VRAM cache to the first 65/94 MoE layers`
warning.

What a shorter context *does* do is shrink the same RAM reservation `prefix_sessions` scales —
and only as the product. `prefix_sessions=1 × 8192` and `prefix_sessions=2 × 4096` are
**literally the same configuration**. So if you want more expert cache, use §3b's table and
decide how many sessions you are giving up; halving the context is just another way of paying
the same price, and it also refuses prompts you may actually want to send.

## 4. `kv_cache_mode` — leave on `"auto"`

`auto` selects bf16 (2.000 bytes/element, lossless) for EXL3 models. The
alternatives:

| value | verdict |
|---|---|
| `"auto"` | **use this** |
| `"bf16"` | same thing, explicitly |
| `"fp8"` | **worse than bf16** — stores an fp8 copy *and* a full-size dequantized mirror, 3.016 B/elem, i.e. ~1.5× more VRAM than no quantization at all |
| `"turboquant"` | **do not use.** Failed its quality gate by 2–7× at the 4-bit default, and reconstructs the whole prefix every decode step: +0.28 s/token at 8k, **+1.49 s/token at 32k**. Kept only as the substrate for `prefix_compress`. |

## 5. Knobs that currently do nothing

Recorded so nobody spends an afternoon on them:

- **`VAAYU_CACHE_BIAS`** — proven inert. `cache_aware_topk` lives in a router
  branch no model reaches under transformers 5.5.4; a call-count probe measured
  **0 calls on both on-device models**. The engine warns when you set it.
- **Batched decode** — `plan_batch` collapses to N=1 on a 16 GB card. Built,
  correct, never batches.
- **Expert pinning** — never built. Measurement refuted it: eviction share is
  linear in N and always 2–4× below the pin's cost, on both models.
- **`use_torch_compile`** — shipped but not validated against the streamed-expert
  MoE path. Experiment at your own risk; not a recommendation.
- **`VAAYU_LOOKAHEAD`** — deliberately disabled; raises on use.

## 5b. `keep_awake` / `pin_engine_threads` — leave both on

Streamed decode sleeps on disk reads — roughly 800 read completions per token on
Qwen3-235B — and every one of them wakes a CPU out of a deep idle state. On this
box that state is `C2_ACPI`, whose declared exit latency is **127 µs**, and decode
leaves the CPU only ~7.5 % busy, so the package keeps falling back into it.

Vaayu holds the package shallow for the duration of a generation: a `SCHED_IDLE`
busy loop on each detected E-core (`keep_awake`), plus P-core placement for the
engine's own threads (`pin_engine_threads`). `SCHED_IDLE` runs only when a CPU
would otherwise be idle, so the loop never takes time from the engine — it costs
power, not throughput.

Measured through the server at its default configuration, one request at a time,
greedy, identical outputs in every arm:

| model | off | on | decode |
|---|---|---|---|
| Qwen3-235B (fused-EXL3 MoE) | 0.9229 tok/s | **1.1363** | **+23.1 %** |
| MiniMax-M2.7 (module-list MoE) | 1.7405 | 1.7625 | +1.3 % |
| Llama-3.3-70B (streamed dense) | 0.2745 | 0.2824 | +2.9 % |
| Llama-3.2-1B (fully VRAM-pinned) | 111.7 | 111.2 | none — it reads no disk |

Directly visible in the kernel's own counters over one Qwen request: C2 residency
1.142 s → 0.492 s, wakes 1.74 M → 1.22 M.

**Turn `keep_awake` off only to save power**, and expect to pay up to ~23 % of
decode for it on a heavily streamed MoE. Set it to an integer to run fewer
spinners: one E-core buys +3.6 % where all eight buy +12.4 % (prefix cache off),
so the win is the package leaving deep idle, not the individual core.

**On a CPU with no P/E split Vaayu declines and prints a warning** rather than
guessing: every core a spinner could take there is an SMT sibling of one the
engine wants, and that arrangement measured **27 % slower** on MiniMax.

### Root-only alternatives (they cover the whole machine, not just Vaayu)

Neither is needed when `keep_awake` is on, and both require privileges Vaayu does
not have or want:

- **PM QoS** — `sudo sh -c 'exec 3<>/dev/cpu_dma_latency; echo -n -e "\x00\x00\x00\x00" >&3'`
  holds the whole system at a 0 µs wake-latency target for as long as that file
  descriptor stays open. `/dev/cpu_dma_latency` is mode 0600 on this box.
- **Disable the deep state** — `sudo cpupower idle-set -d 2` disables `C2_ACPI`
  (check the numbering with `cpupower idle-info` first). Persists until re-enabled.

Both raise idle power across the machine. The spinner is the scoped version of the
same idea: it is active only while a generation is running.

## 5c. `decode_top_k` — set `"auto"` on an MoE model, unless output must be exact

MoE models route each token to several experts. Routing to fewer of them at decode is a large speed
lever — and the only **lossy** knob Vaayu has: output changes, greedy ids diverge. How much quality it
costs is a property of the model, and it is **not predictable** from expert count or top-k (a 10 % cut
cost Qwen3-Next 1.47 % perplexity; a 12.5 % cut cost MiniMax-M2.7 6.34 %). So do not guess a number.

**Set `DECODE_TOP_K = "auto"`.** On the first load of a checkpoint Vaayu measures it on a built-in code text and a prose passage: it tries fewer experts one step at a time and keeps the fastest setting whose perplexity cost stays within `DECODE_TOP_K_BUDGET` (default `0.035` = 3.5 %) on **both** texts. If no fixed expert count fits, it tries a per-token cutoff instead, which drops an expert only on tokens where it carries little weight. The result is cached next to the checkpoint's split files, so every later load applies it instantly.

| model | pick at 3.5 % | decode tok/s | perplexity cost (worse of code and prose) | one-time first load |
|---|---|---|---|---|
| Qwen3-Next-80B int4 | fixed k = 8 of 10 | **+23.1 %** | +3.43 % | ≈ 6.5 min |
| MiniMax-M2.7 | cutoff p = 0.90 | **+8.4 %** | +2.25 % | ≈ 26 min |
| Qwen3-235B | fixed k = 7 of 8 | **+11.2 %** (one sample) | +3.40 % | ≈ 21 min |

These picks were measured by `"auto"` itself on 2026-09-14; your checkpoint's pick is the one its load log prints. **While the first load measures, the server is not reachable** — like a long load.
Watch `server.log` for one `DECODE_TOP_K=auto:` line per trial and a final WARNING naming the pick and its
cost (or a line saying no setting fits the budget, when none does); afterwards `/health` → `decode_top_k` reports it.

- **Tighter quality needs:** lower `DECODE_TOP_K_BUDGET` (at 0.02 Qwen3-Next picks k = 9, +17.0 % for
  +1.47 %). A changed budget re-measures on the next load.
- **Output must match the model exactly** (evaluation, reproducibility): leave `DECODE_TOP_K = None`.
- **The budget is a perplexity budget.** Perplexity moves long before text visibly degrades; a short
  read of one generation is not a quality check.
- **To force a re-measure:** delete `decode_top_k_auto.json` from the checkpoint's `splitted_model/` directory.
- Dense models ignore the setting (with a note in the log).

## 6. Knobs to leave alone unless you have a reason

- **`max_pinned_layers=10`** — how many transformer layers stay resident in VRAM.
  Raising it competes with the expert cache and the KV cache for the same 16 GB.
- **`prefill_chunk_size=512`** — chunked prefill granularity. Measured TTFT
  differences between 128 and 512 were within noise at ~440-token prompts.
- **Speculative decoding** (`drafter_model_path`, `use_prompt_lookup`) — prompt
  lookup only helps when output copies the prompt. For novel code generation the
  n-gram hit rate is ~0, so it is off in the benchmarks.
- **`enable_expert_usage`** — diagnostics only, for expert-routing investigations.

## 7. Working out your own numbers

The advisor does this for you, but the arithmetic is simple enough to check:

```
KV bytes per token = layers × 2 × kv_heads × head_dim × bytes_per_element
    bf16                = 2.000 B/elem
    prefix_compress=9   = (head_dim-8)×9/8/head_dim + (4 + 2×8)/head_dim
                        ≈ 1.211 B/elem  (0.605×)   [8 channels kept exact]

sessions that fit = pool_budget / (tokens_per_session × bytes_per_token)
```

Worked example, MiniMax (62 layers, 8 KV heads, head_dim 128) with a 4.2 GB pool:

| | bf16 | 9-bit |
|---|---|---|
| per 8k-token session | 1,937 MiB | 1,173 MiB |
| sessions that fit | **2** | **3** |
| total token ceiling | 17.8k | 29.4k |

So three 8k sessions need compression; two do not.

**The pool budget is not free RAM.** It is computed at model-load time *after* the
expert RAM cache has taken its share. On the development box, 23 GB free before
load became **3.5–5.4 GB** of actual pool across six runs. Always read the real
figure from the startup line rather than estimating.

## 8. Harnesses

- **`tune_advisor.py`** — what to set, for your model and workload. No weights, no
  GPU, runs in seconds.
- **`test_generate.py`** — single-turn code-generation benchmark. Reports TTFT and
  decode throughput. Deliberately does **not** set `prefix_compress`: one session,
  one turn, nothing to reuse.
- **`test_multiturn.py`** — K interleaved sessions × 2 rounds, and the harness that
  proves prefix caching works. It **asserts** the final-round hit count and exits
  non-zero if the budget failed to keep sessions resident, so it doubles as a gate.
  Raise `K_SESSIONS`, or the system-prompt size, to push the pool over budget and
  watch the cliff.
