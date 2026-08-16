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


> **Running the server?** These knobs are all reachable as CLI flags on
> `python -m vaayu.server` — see [**server.md §3**](server.md#3-cli-parameter-reference).
> The server's defaults differ from the library's on purpose: `max_pinned_layers` is `-1`
> (not `0`) and `enable_prefix_cache` is on (not off), because a server holds one model for
> its whole life and should fill the box. Getting the first one wrong costs **13x** on
> Llama-3.2-1B and **2.0x** on MiniMax-M2.7, measured.

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
| don't fit even compressed | `9`, plus reduce load | compression widens the range ~1.7×, it does not remove the wall |

**The failure mode is a cliff, not a slope.** A pool one session short of what you
need evicts each session before its revisit, so reuse collapses to **zero**, not to
a fraction. Measured: bf16 scored 0/4 and 0/3 hits in the two over-budget tests,
never a partial score. This is why the crossover matters more than it looks.

**Use 9, not 8.** At 8 bits MiniMax's worst layer measured 0.0612 attention-score
relative error against a 0.05 bar. 9 bits passed both models (0.0214 / 0.0372).
Anything below 9 prints a warning.

**What it does not do:** shrink VRAM for the *in-flight* session. The live cache is
always bf16. If a single session's KV won't fit in VRAM, this cannot help — that
needs attention over compressed keys, which is not built.

Full measurements: the compress-on-evict RESULTS doc under `docs/superpowers/plans/`
(local working notes, not tracked in the repo).

## 3. `enable_kv_persist` — on, unless disk is precious

**Default:** `False`. Writes the KV to `~/.cache/vaayu/kv` so a session survives a
process restart. Measured **TTFT 27.6 s → 3.0 s** on a restored session. Costs disk
(~130 MB per 700-token Qwen session) and a little CPU. `kv_persist_sync=False`
(the default) is the faster, production-shaped choice.

Not compatible with `kv_cache_mode="turboquant"` — the writer refuses that layout
and says so. It works fine with `prefix_compress`.

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
    prefix_compress=9   = 9/8 + 4/head_dim  ≈ 1.156 B/elem  (0.578×)

sessions that fit = pool_budget / (tokens_per_session × bytes_per_token)
```

Worked example, MiniMax (62 layers, 8 KV heads, head_dim 128) with a 4.2 GB pool:

| | bf16 | 9-bit |
|---|---|---|
| per 8k-token session | 1,937 MiB | 1,120 MiB |
| sessions that fit | **2** | **3** |
| total token ceiling | 17.8k | 30.7k |

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
