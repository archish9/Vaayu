# Plan I — Compress-on-evict KV: RESULTS

**Date:** 2026-08-02 · **Plan:** `2026-08-02-compress-on-evict-kv.md` ·
**Branch:** `plan-i-compress-on-evict` · **Suite:** 500 → 519 green

## VERDICT

**Works, and does what Plan H could not.** Compressing pooled KV at the eviction boundary costs
**zero** decode throughput on both models, nearly halves the bytes a cached session occupies, and
where the pool is budget-bound it turns a total prefix-cache collapse into a full set of hits.
Measured twice: **91 s → 9.6 s (9.5×)** at K=4 with a capped budget, and **332 s → 9.8 s (34×)** at
K=3 × 7.7k tokens with the real budget. The default bit-width was raised from 8 to **9** on the
gate's own evidence.

**The benefit is real but conditional.** It appears only when ≥2 concurrent sessions exceed the pool
budget — for MiniMax, roughly 20k–35k combined resident tokens. Below that the pool already fits and
compression is pure cost (~0.45 s/turn); above it, both modes thrash. At the harness default (K=2–3
× 2k tokens) there is no benefit at all, which is why it ships off by default. The realistic hit is
**a few concurrent sessions each carrying a large re-sent context** — which is the agent shape this
engine exists for.

---

## 1. Decode must be untouched — **PASS**

The defining property: compression happens once per turn on eviction, never on the decode path. If
decode moves at all, something is compressing in the hot loop.

`test_generate.py`, 256 output tokens, fresh session id per run:

| model | `prefix_compress=None` | `=8` | floor | verdict |
|---|---|---|---|---|
| Qwen (fused EXL3) | 0.86 tok/s | **0.86 tok/s** | 0.82 | unchanged |
| MiniMax (module-list) | 1.53 tok/s | **1.56 tok/s** | 1.47 | unchanged |
| Qwen TTFT | 27.26 s | 27.37 s | — | flat |
| MiniMax TTFT | 31.02 s | 30.87 s | — | flat |

**Contrast with Plan H**, which compressed the live cache: MiniMax decode was 1.45 / 1.40 against
the same 1.53 baseline — 5–8% below floor — and the benchmarked cost grew to **+1.49 s/token at
32k**. Moving compression off the decode path removes that entirely, on both forwards.

## 2. Pool capacity — **PASS, and the failure mode is a cliff**

`test_multiturn.py`, MiniMax, K=4 sessions × 2 rounds.

**Scaffolding disclosure:** the pool budget was capped to **1050 MiB** for this measurement. At the
real budget (~5 GB) the bf16 crossover sits near 11 sessions, which is ~2.4 h of prefill to
demonstrate; capping moves the crossover to K=4 without changing the mechanism under test. Per-session
bytes and eviction behaviour are unaffected by the cap.

| | bf16 | compressed (8b) |
|---|---|---|
| final-round HITs | **0/4** | **4/4** |
| evictions | 4 | 0 |
| bytes per session | 497.2 MiB | **256.4 MiB** (0.516×) |
| round-2 TTFT | ~91 s | **~9.6 s** |
| harness exit | 1 (its own assertion) | 0 |

**The result is sharper than predicted.** The expectation was 2/4 hits for bf16 — instead it scored
**zero**. With a pool that holds fewer sessions than K, round-robin access evicts each session
before it is revisited, so reuse does not degrade gracefully, it **collapses**. Being one session
short of budget costs everything, not a proportional slice. That makes doubling capacity worth
more than a linear reading suggests.

Extrapolated to the real ~5 GB budget and these session sizes: **~10 → ~20 concurrent sessions**
before the cliff.

## 3. Long context — **MEASURED, 34× TTFT, no budget cap**

Run at the operator's request before merging, since §2's demonstration needed a capped budget.
`test_multiturn.py`, MiniMax, **K=3 sessions × 7,751 tokens = 23.3k combined**, `N_MODULES=160`,
`MAX_LENGTH=16384`, **real (uncapped) pool budget**, 9 bits.

| | bf16 | compressed (9b) |
|---|---|---|
| pool budget this run | 3.54 GB | 4.18 GB |
| bytes per session | ~1,877 MiB (derived) | **1,093.9 MiB** (measured, 0.583×) |
| round-0 evictions | **2** | **0** |
| final-round HITs | **0/3** | **3/3** |
| final-round evictions | 3 | 0 |
| round-2 TTFT | **~332 s** (full re-prefill) | **~9.8 s** |
| tokens saved | 0 / 23,250 (0%) | 23,190 / 23,250 (**100%**) |
| harness exit | 1 | 0 |

**332 s → 9.8 s, a 34× TTFT reduction**, and the largest effect measured anywhere in Plans H or I.
It is larger than §2's 9.5× for the obvious reason: a cold prefill at 7.7k tokens costs far more
than at 2k, so preserving the HIT is worth proportionally more.

**Budget confound, disclosed:** the two runs drew different pool budgets (3.54 vs 4.18 GB) because
the budget is computed from free RAM at load time. This does not affect the conclusion — bf16 needs
`3 × 1,877 MiB = 5.50 GB` for three resident sessions, which exceeds **both** figures, so bf16
would have evicted even with the compressed run's larger budget. Compressed needed 3.20 GB and fit
in either.

This is the configuration the feature exists for, and it is a realistic agent shape: **a few
concurrent sessions, each carrying a large re-sent context.** Not many tiny sessions, and not one
enormous one (§7).

## 4. Per-turn compression cost — acceptable, and it is the whole cost

Qwen geometry (94 layers, 4 KV heads, d=128), 9 bits, measured on GPU:

| context | compress | decompress | total per turn |
|---|---|---|---|
| 2k | 0.39 s | 0.07 s | **0.45 s** |
| 8k | 0.48 s | 0.26 s | **0.73 s** |
| 32k | 2.16 s | 1.39 s | **3.55 s** |

Set against what a prefix HIT avoids: a cold round-2 prefill measured **91 s** at 2k in §2. The
trade is ~0.45 s to save ~81 s — two orders of magnitude. Even at 32k, 3.55 s against a cold
prefill that would be far larger than 91 s stays overwhelmingly favourable.

Compare with Plan H, where the same arithmetic ran per **token** rather than per turn.

## 5. Restored-prefix quality — **8 bits FAILED, 9 bits PASSED; default changed**

Bar (same as Plan H's, pre-stated): attention-score relative error < 0.05. Method: real K/V read
back from bf16 `.vaayu_kv` files, pushed through the shipping `vaayu.kv_compress` path.

| model | bits | K relL2 | score (mean) | score (max) | gate |
|---|---|---|---|---|---|
| Qwen | 8 | 0.0244 | 0.0203 | 0.0480 | pass (thin) |
| MiniMax | 8 | 0.0246 | 0.0132 | **0.0612** | **FAIL** |
| Qwen | 9 | 0.0157* | 0.0080* | **0.0372** | PASS |
| MiniMax | 9 | 0.0157 | 0.0080 | **0.0214** | PASS |

<sub>*Qwen 9-bit means not separately itemised; its worst layer is the figure that matters.</sub>

At 8 bits the worst layer on MiniMax exceeds the bar and Qwen clears it by only 0.002. **Both
models' outliers are early layers** (layer 1 on MiniMax, layers 1 and 70 on Qwen) — the massive
early-layer activations the paper's outlier-channel split exists to handle.

**Action taken:** `_DEFAULT_BITS` raised 8 → 9, and the sub-threshold warning now fires below 9.
Cost of the change: 1.156 B/elem instead of 1.031, i.e. **0.578× bf16 rather than 0.516×**. Buying
the gate back for 12% of the saving is the right trade — a pooled prefix is what the model attends
to on every subsequent turn.

## 6. The ordering trap, live — **PASS**

With `prefix_compress=8` **and** `enable_kv_persist=True` on Qwen, the written `.vaayu_kv` has:

```
sig=bf16   layers=94  heads=4  head_dim=128  nrec=694  rec_bytes=192516
```

`192516 = 4 + 94 × (2 × 4 × 128 × 2)` exactly — real bf16 values, arithmetic proof that no packed
index reached disk. The file is structurally identical to the uncompressed run's (same
`rec_bytes`, size differing only by session-id length). The Task 3 reordering holds under real
conditions; without it this file would have been packed indices stamped as bf16.

## 7. What this does not do

- **No VRAM saving for the in-flight session.** The live cache stays bf16 by design. Longer context
  on one card still needs attention against compressed keys (fold `Π` into the query).
- **No benefit at small scale.** At K=2–3 × 2k tokens the pool holds everything in both modes
  (0 evictions), so compression is pure cost. §3 shows the benefit arrives once sessions get large;
  §2's small-session demonstration needed a capped budget to surface at all.
- **Plan H's mode is not retired.** `kv_cache_mode="turboquant"` remains, opt-in and documented as
  unsuitable for long context.

## 8. Recommendation

1. **Merge, keep default off.** `prefix_compress=None` remains the default; nothing changes for
   existing runs.
2. **Turn it on when combined resident tokens approach the pool budget** — measured for MiniMax at
   ~20k (bf16 wall) against ~35k at 9 bits. In practice: **3 sessions at ~8k tokens each already
   qualifies** (§3, 34×), as does fan-out past ~8–10 short sessions. Below that it is inert and
   costs ~0.45 s/turn; the transition is a cliff, not a slope (§2).
3. **Use 9 bits** (now the default). Do not use 8 — it fails MiniMax's worst layer.
4. §3 is now measured on device, uncapped — the strongest result in this plan.
5. **The outlier split remains the real follow-up.** Both models' failures at 8 bits are early-layer
   outliers, which is exactly what that technique addresses — and it would buy back the 12% that
   moving to 9 bits cost.

## Files

- `vaayu/kv_compress.py`, `tests/test_kv_compress.py` (13 tests)
- `vaayu/prefix_cache.py`, `tests/test_prefix_cache.py` (5 added)
- `vaayu/vaayu_base.py` — `_build_prefix_store`, `_persist_and_store`, `prefix_compress=`
- Evidence: `~/vaayu-evidence/2026-08-02-plan-i/` — `k4-off.log`, `k4-c8.log`,
  `restored_quality.py` (CPU-only, re-runnable from any bf16 `.vaayu_kv`)
