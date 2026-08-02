# Plan H — TurboQuant KV cache: RESULTS

**Date:** 2026-08-02 · **Plan:** `2026-08-02-turboquant-kv.md` (revision 2) ·
**Branch:** `plan-h-turboquant-kv` · **Suite:** 457 → 500 green

## VERDICT

**The implementation is complete and hits its size target exactly — 0.5312 bytes/element measured on
device (−73.4% vs bf16), 3.5× more concurrent sessions resident in the prefix pool. It FAILS the
pre-stated quality gate at the plan's default 4 bits/channel, by 2–7×, on both models, and it
regresses decode by 5–8% on MiniMax (below the 1.47 tok/s floor) while leaving Qwen unaffected. Do
not ship it as a default. Do not ship it at 4 bits at all.**

The size and fan-out wins are real and reproducible. They are bought with a **23% relative L2
perturbation of every stored key and value**, which is inherent to uniform 4-bit TurboQuant and not
a defect in this implementation — the measured distortion matches the paper's own anchors to three
decimals. The gate is failed on quality, not on engineering.

---

## The pre-stated gate

Registered **before any number was looked at** (the plan requires this, and greedy decode on this
stack is not bit-reproducible, so a text diff proves nothing either way):

> attention-score relative error `‖softmax(qK̃ᵀ/√d) − softmax(qKᵀ/√d)‖ / ‖softmax(qKᵀ/√d)‖ < 0.05`

## 1. Quality — FAIL on both models

Method: read **real** K/V from the bf16 `.vaayu_kv` files a real prefill produced (lossless, so
they are ground truth), push them through `vaayu.turboquant`, measure per layer. CPU-only, no model
load needed. 7 layers sampled per model (first two, quartiles, last two).

| model | forward | K relL2 | V relL2 | score err (mean) | score err (max) | gate |
|---|---|---|---|---|---|---|
| Qwen3-235B, 94L/4H/D128 | fused EXL3 | 0.2296 | 0.2314 | 0.0919 | 0.2411 | **FAIL** |
| MiniMax-M2.7, 62L/8H/D128 | module-list | 0.2294 | 0.2298 | 0.1780 | 0.3499 | **FAIL** |

Both means exceed the 0.05 bar; both maxima exceed it by 5–7×.

**The quantizer is not broken.** K relL2 of 0.2296 on real keys matches the 0.2297 predicted from
synthetic unit vectors to four decimals, which is itself the paper's own MSE anchor (0.03 at
b_mse=3 → √0.03 = 0.173) times the QJL variance factor. The rotation is doing its job: real keys,
after rotation, behave exactly like random unit vectors. Uniform 4-bit TurboQuant simply has 23%
error, and softmax amplifies it.

**Query proxy caveat, stated plainly.** The `.vaayu_kv` format stores K and V; queries are not
persisted, so two proxies were used: `q` = the layer's last stored K row (correlated, peaked
softmax), and `q` ~ Gaussian scaled to the mean K row norm (uncorrelated, flat softmax). The table
reports the K-proxy. The Gaussian proxy is far worse (mean 0.638 Qwen / 0.508 MiniMax). The true
figure lies between them; **both exceed the bar by a wide margin, so the verdict does not depend on
which proxy is right.** A definitive gate would capture real queries with an attention hook —
worth doing only if someone wants to revisit the bit-width question.

### Where the gate *would* pass

Same real K, sweeping bit-width and the QJL variant (`b_mse` = MSE bits; B/elem includes signs and
norms):

| b_mse | B/elem | K relL2 | Qwen score max | MiniMax score max |
|---|---|---|---|---|
| 2 | 0.406 | 0.425 | 0.489 | 0.623 |
| 3 (**plan default**) | 0.531 | 0.230 | 0.241 | 0.350 |
| 4 | 0.656 | 0.120 | 0.146 | 0.194 |
| 5 | 0.781 | 0.065 | 0.063 | 0.303 |
| 6 | 0.906 | 0.039 | 0.128 | 0.059 |
| 7 | 1.031 | 0.024 | **0.035 PASS** | **0.043 PASS** |
| 8 | 1.156 | 0.016 | 0.018 PASS | 0.053 |

Smallest configuration that passes on both models: **`bits_total=8` (b_mse=7), 1.031 B/elem** — a
**1.94× win over bf16, not the 3.77× the plan targeted.** The tail figures are noisy (one query row
per layer, hence the non-monotonic b_mse=8 row on MiniMax); the means are monotone and tell the
same story.

**MSE-only, at equal storage, is not better.** Dropping QJL and spending the bit on the MSE stage
gives lower L2 (0.0965 vs 0.1203 at ~0.52 B/elem) but *never* passes the score gate on MiniMax at
any bit-width up to 8. The QJL stage's unbiasedness is doing real work for attention even though it
costs L2 — which is the paper's claim, now measured here rather than inherited.

## 2. Size — matches the prediction exactly

| | bf16 (`auto`) | TurboQuant 4-bit | ratio |
|---|---|---|---|
| Qwen, one 694-token session | 127.4 MiB | **33.8 MiB** | 0.266 |
| MiniMax, one 690-token session | 167.1 MiB | **44.4 MiB** | 0.266 |
| MiniMax, per 2054-token session (fan-out harness) | 497.2 MiB | **132.1 MiB** | 0.266 |
| bytes/element, every case | 2.0000 | **0.5312** | −73.4% |

0.5312 is the plan's 0.531 prediction to three decimals: `(3 index bits + 1 sign bit)/8 +
4 norm bytes/128 channels`. The indices are genuinely bit-packed — one index per byte would have
read 1.16.

## 3. Fan-out — the thing this feature exists for

`test_multiturn.py`, MiniMax, K=3 sessions × 2 rounds:

| | bf16 | TurboQuant |
|---|---|---|
| final-round HITs | 3/3 | 3/3 |
| evictions | 0 | 0 |
| bytes per session | 497.2 MiB | 132.1 MiB |
| pool budget | 4.872 GiB | 4.564 GiB |
| **sessions that fit** | **10** | **35** |
| round-1 cold TTFT (mean) | 99.79 s | 99.53 s |
| round-2 HIT TTFT (mean) | 9.47 s | 9.69 s |

**Honest reading, as the plan demanded:** the pool recorded **zero evictions in both modes**, so
the measured fan-out benefit today is **nil** — the budget is not binding at K=3, exactly as the
plan predicted. The win is headroom: the crossover where sessions start being evicted moves from
**10 to 35 concurrent sessions**. On this 16 GB box, with this agent workload, that ceiling is not
currently reached.

## 4. Speed — Qwen unaffected, **MiniMax decode regresses below its floor**

`test_generate.py`, 256 output tokens, one generate per row.

| model | chunk | mode | prompt | TTFT | decode | floor |
|---|---|---|---|---|---|---|
| Qwen | 512 (off) | bf16 | 439 | 27.45 s | 0.81 | 0.82 |
| Qwen | 512 (off) | **tq4** | 439 | 30.52 s | **0.84** | |
| Qwen | 128 (4 chunks) | bf16 | 439 | 27.11 s | 0.87 | |
| Qwen | 128 (4 chunks) | **tq4** | 439 | 27.39 s | **0.83** | |
| MiniMax | 512 (off) | bf16 | 435 | 30.18 s | 1.53 | 1.47 |
| MiniMax | 512 (off) | **tq4** | 435 | 30.49 s | **1.45** | |
| MiniMax | 512 (off) | **tq4** (repeat) | 435 | 30.61 s | **1.40** | |

**Qwen: no regression** (0.81/0.87 bf16 vs 0.84/0.83 tq — same band across four runs).

**MiniMax: a real 5–8% regression, below the 1.47 tok/s floor**, reproduced across two runs
(1.45, 1.40 against 1.53). The asymmetry has a mechanical explanation: reconstruct-on-read costs
two `[S,d]×[d,d]` matmuls per KV head per layer, so the work scales with `layers × kv_heads` —
MiniMax is 62×8 = 496 head-layers against Qwen's 94×4 = 376 (+32%), and it pays that on a
0.65 s/token budget where Qwen has 1.2 s. **The two models sit on different forwards (module-list
vs fused EXL3), and this is a case where one is not evidence for the other** — CLAUDE.md §0's point,
confirmed again.

## 5. Chunked-prefill TTFT — no quadratic penalty at this length

The plan flagged reconstruct-on-read as quadratic-ish across chunked prefill (`Σ chunks ≈
S²/(2·chunk)`), and the default configuration would not have tested it: the 439-token prompt is
**below** the 512-token chunk size, so chunking never engaged. Re-run at `PREFILL_CHUNK_SIZE=128`
(4 chunks):

| | bf16 | tq4 | delta |
|---|---|---|---|
| TTFT, 4 chunks | 27.11 s | 27.39 s | **+1.0%** |
| TTFT, 1 chunk | 27.45 s | 30.52 s | +11.2% |

With chunking genuinely on, the penalty is **+1%** — smaller than the unchunked delta, which places
that +11% inside run-to-run variation rather than reconstruct cost. **The quadratic risk is not
refuted, only unexercised:** at 439 tokens over 4 chunks the S² term is negligible. It would need a
multi-thousand-token prompt to appear, which this harness (`MAX_LENGTH=2048`, ~440-token prompts)
does not produce. Anyone raising the context length should re-measure TTFT before assuming it holds.

## 6. Coherence

Both Qwen continuations are fluent, on-task, and of comparable quality — the TurboQuant one is
arguably the tidier plan. **This proves less than it appears to.** 256 tokens of planning prose is
an insensitive probe; the numerical gate exists precisely because coherent-looking output can hide
a 23% perturbation of every key. Coherence passed; the gate did not; the gate is the one that was
designed to be informative.

## 7. What the plan said we could not claim, confirmed

The plan put the **mixed-precision outlier split** out of scope and warned that the paper's
quality-neutral 3.5-bit LongBench result therefore cannot be claimed. That warning is now backed by
measurement: uniform `bits_total=4` — which spends *more* bits than 3.5 — misses a 5%
attention-score bar by 2–7×. The paper's fractional bit-widths are not a rounding detail; the
outlier split is doing the quality work. **Any future attempt at sub-1 B/elem KV on this stack
should start there, not at a higher uniform bit-width.**

## 8. Corrections to the plan (revision 2)

Two of revision 2's specifications were mathematically wrong and were corrected in code and pinned
by tests:

1. **Codebook `dim`-scaling ratio was inverted.** Post-rotation coordinates are ≈`N(0, 1/d)`, so
   centroids scale `1/√d`: `make_codebook(3,128)` is `√(64/128)×` the `dim=64` codebook, not `√2×`.
   The plan's own prose (`the centroids scale with 1/√d`) implies the correct direction; only the
   test assertion was inverted. Guard: `test_codebook_scales_with_one_over_sqrt_dim`.
2. **"QJL strictly helps [L2]" is false as specced.** With the `d×d` sketch the estimator's own
   variance contributes `γ²(π/2 − 1/d)`, so reconstruction error is **~1.57× the MSE stage alone**
   (measured 1.56×). Beating the MSE stage on distortion would need a sketch with `m > 1.57d` rows.
   QJL buys **unbiased inner products**, not lower distortion — and §1's sweep shows that
   unbiasedness is what actually matters for attention. The false assertion was replaced by two
   tests: one pinning the 1.57 factor, one proving unbiasedness directly.

## 9. Recommendation

1. **Do not flip `auto`.** Not at 4 bits, and not at 8 bits either without a real-query quality
   gate. `auto` stays bf16.
2. **Keep the mode, ship it dark, at its current default.** It is opt-in
   (`kv_cache_mode="turboquant"`), fully tested, cannot corrupt persisted KV (the writer refuses
   it), and correctly accounted for in both VRAM budget sites. It costs nothing to leave in place
   and is the substrate for any follow-up.
3. **Document the honest operating point.** If someone needs the VRAM, `turboquant:8` at
   1.031 B/elem passes the gate on both models for a 1.94× saving — a real but unspectacular win
   over bf16, and one that should be re-measured with real queries first. Note it does **not**
   fix the MiniMax decode regression, which is per-token reconstruction work and independent of
   bit-width.
4. **The follow-up that matters is the outlier split, not more bits.** §7.
5. **Folding the rotation into the query is now worth something after all** — the plan's named
   perf follow-up, which §4 turns from speculative into targeted. `⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩` turns the
   per-token `O(S·d²)` reconstruction into `O(d²)`, which is exactly the term that costs MiniMax
   its 5–8%. It needs hooking attention itself (the `Cache` interface obliges us to return real
   tensors), so it is a spec, not a patch — and only worth writing if the quality question in §7
   is solved first. A fast cache that is 23% wrong is not worth speeding up.

## Files

- `vaayu/turboquant.py`, `tests/test_turboquant.py` (43 tests)
- `vaayu/vaayu_base.py` — `TurboQuantKVCache`, `_create_kv_cache`, `_get_vram_pinning_budget`
- `vaayu/batch.py`, `vaayu/kv_cache_manager.py`, `vaayu/kv_persist.py`
- Evidence logs are operator-local and untracked: **`~/vaayu-evidence/2026-08-02-plan-h/`**
  (9 gzipped run logs, 516 MB → 27 MB, plus `SUMMARY.md` with the distilled figures and the two
  CPU-only analysis scripts `quality_gate.py` / `sweep.py`, which re-run the quality gate and the
  bits sweep from the persisted bf16 `.vaayu_kv` files with no GPU).
