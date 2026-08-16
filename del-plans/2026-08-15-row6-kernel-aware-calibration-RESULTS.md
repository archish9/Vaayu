# Row 6/7 — Kernel-aware VRAM calibration, and what the honest data did to the form

**Status: rows 6 and 7 done as ONE change. The ±10 % gate does NOT pass, deliberately and
visibly.** The form is materially better than it was — but the honest whole-prefill data exposed a
second defect that no amount of re-pointing fixes, and that defect is written up here and carried
as row 6a rather than papered over.

## 1. Provenance

- Date: 2026-08-15. Box `a`, RTX 5060 Ti, 16.61 GB VRAM (decimal), driver 595.71.05,
  torch `2.11.0+cu130`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- sdpa arms: `scripts/probe_vram_curve.py` with `PROBE_ATTN=sdpa` (the probe asserts
  `model._active_attn_impl` matches, so a silent fallback cannot corrupt a point), one arm per
  subprocess under `systemd-run --user --scope -p MemoryMax=<22G; 28G for the 70B>
  -p MemorySwapMax=0`, strictly sequential, each checkpoint's first touch discarded. Raw logs
  `/tmp/row6/*_sdpa.log` (ephemeral); every point transcribed into
  `scripts/fit_vram_calibration.py` is `status=ok`.
- flash arms: transcribed from `plans/2026-08-15-vram-peak-RESULTS.md` §3 — the honest
  whole-prefill data row 1 produced.
- **S lists were passed EXPLICITLY, not derived by `sweep_points`.** Caught mid-run: `sweep_points`
  takes its ceiling from `model.max_seq_len`, which `predict_peak_vram` itself decides, so a
  derived grid moves under the form being calibrated. The first arms were discarded and relaunched
  with explicit lists (`512 513 1024 1025 2048 2049 4096 4097 8192 8193` for the three
  8192-ceiling geometries; out to `65537` on Llama-3.2-1B).

## 2. What shipped

**Row 7 — the score term is now kernel-keyed.** `MATERIALIZES_SCORES = False` (a global bool,
measured against flash_attention_2 only) became `SCORE_ELEM_BYTES = {"flash_attention_2": 0.0,
"sdpa": 10.9}` plus `score_elem_bytes(impl)`, and `VramShape.materializes_scores: bool` became
`score_elem_bytes: float`. The call site reads `model._active_attn_impl` (`init_model` runs at
`vaayu_base.py:1421`, before the plan block at ~1505 — verified, not assumed). Any kernel with no
measured coefficient — `eager`, BetterTransformer, `None` — takes the LARGEST measured value, never
0: `eager` materializes scores by construction, and under-reserving is the dangerous direction.

`10.9` is the mean of the four geometries' own paired flash-vs-sdpa slope estimates, measured on
identical S lists: Llama-3.2-1B **10.35**, Qwen3-235B **10.52**, Llama-3.3-70B **11.33**,
MiniMax-M2.7 **11.42** B/elem. It is fitted from measured slopes and **not** tuned to minimise gate
failures — the 10 % spread means no single value is right for all four, and choosing one to make
the gate greener would hide exactly that.

**Row 6 — `MEASURED` is keyed by kernel first**, `{impl: {model: [(S, alloc_before, alloc_peak)]}}`,
flash and sdpa both kept: sdpa is what `batched_generate` hard-requires and must be budgeted for,
flash is what ordinary single-stream `generate()` actually runs. The STALE last-chunk caveat is
gone. Excluded points live in `NOISY_FIRST_POINTS` with a written reason and are printed below the
gate line — never silently dropped.

**The intercept changed too, and this was not in row 6's brief.** See §4.

## 3. Paired flash-vs-sdpa data (transient GB = `alloc_peak − alloc_before`)

| geometry | S | flash | sdpa | ratio |
|---|---|---|---|---|
| llama-3.2-1b | 4096 | 0.9264 | 1.2295 | 1.33 |
| llama-3.2-1b | 16384 | 1.3291 | 3.7209 | 2.80 |
| llama-3.2-1b | 65537 | 2.9397 | 13.6866 | 4.66 |
| llama-3.3-70b | 2048 | 1.5989 | 2.2952 | 1.44 |
| llama-3.3-70b | 8192 | 3.6122 | 6.5881 | 1.82 |
| qwen3-235b | 2048 | 1.8788 | 2.5730 | 1.37 |
| qwen3-235b | 8192 | 3.0948 | 6.0553 | 1.96 |
| minimax-m2.7 | 2048 | 1.1504 | 1.1711 | 1.02 |
| minimax-m2.7 | 8192 | 2.7484 | 4.4939 | 1.64 |

Row 10's conclusion reproduces on the denser grid: the gap grows with S on every geometry, and the
implied per-element cost lands in the same 10.3–11.4 band across four architecturally unrelated
models.

## 4. The second defect: the trailing-chunk sawtooth was an artefact

Re-pointing alone left the form under-predicting **flash** by 12–97 % on both dense geometries —
where the score term is genuinely zero, so the kernel work cannot be the cause. The residual has
structure:

`transient − KV(capacity)`, flash arm:

| geometry | at a FULL trailing chunk | at a 1-token trailing chunk |
|---|---|---|
| llama-3.2-1b | 0.7922 G | **0.7922 G** (identical) |
| llama-3.3-70b | 0.928 G | 0.714 G |
| minimax-m2.7 | 0.62–0.67 G | 0.082–0.089 G |
| qwen3-235b | ≈ residency (1.514 G) | ≈ residency |

Llama-3.2-1B's measured transient at S=4096 and S=4097 is **byte-identical** (`alloc_peak`
2013892608 vs 2013893120, 512 bytes apart), and `transient − KV` is a flat **0.7922 G** at every S
from 4096 to 106987. There is no sawtooth in that geometry at all.

The mechanism is simple once stated: a **whole-prefill** peak is the max over chunks, and a full
chunk always precedes a short trailing one — so the trailing chunk cannot lower the peak. The old
sawtooth was fitted to **last-chunk-only** peaks, the exact measurement defect row 1 removed.

And 0.7922 decomposes exactly: **0.5253 G of tied fp16 lm_head weight + 0.2627 G of full-chunk fp32
logits = 0.788 G**, 0.5 % from the measurement, on five independent points.

So `_peak_intercept_bytes` now charges `head_bytes + N · min(S, chunk) · vocab · 4`. `head_bytes` is
a **file stat** on the checkpoint's own head shard — the embedding shard when
`config.tie_word_embeddings` is set, since a tied checkpoint ships an empty 16-byte `lm_head`
shard — mirroring the `_layer_slack` stat already in the same block, so no model number is typed
(CLAUDE.md §5). Measured: 1B 0.5253 G (tied), 70B 0.7883 G, Qwen 0.4671 G, MiniMax 0.4614 G.

**Consequence:** `predict_peak_vram` is now monotone non-decreasing in S. `solve_max_seq_len`'s
exhaustive "every S' ≤ S" check is therefore redundant — and was **kept**, because the sawtooth has
already come and gone once, and the check is what makes its return safe rather than silent.

## 5. Gate outcome — FAIL, 44/77 outside ±10 %, worst +273 %

Every failure but two tiny ones is **over**-prediction, the safe direction for a reserve. The two
under-predictions are Llama-3.2-1B flash at −2.2 % and −0.5 %, both inside the bar.

Where the form is now good (flash, operating range):

| geometry | points inside ±10 % | best/worst inside |
|---|---|---|
| llama-3.2-1b | S ≥ 4096, all 7 | −2.2 % … +0.3 % |
| qwen3-235b | all 7 non-excluded | −0.1 % … +6.2 % |
| llama-3.3-70b | S ≥ 2048, 4 of 5 | +3.4 % … +9.8 % |
| minimax-m2.7 | S = 8192 only | +7.4 % |

Before this change the same 1B flash points read **−12 % … −97 %**. That is the size of the win,
and it is in the operating range rather than at the noisy small-S end.

Where it still fails:

1. **Small S (512–1025), every geometry.** Over-predicted by 15–63 %. The intercept charges a full
   head + full chunk of logits against a transient that is barely larger than the head itself.
2. **MiniMax-M2.7 at short trailing chunks** — +70 % at S=2049, +231 % at S=513. Its real
   whole-prefill transient genuinely collapses when the trailing chunk is short (0.3415 G at S=513
   against 0.8690 G at S=512), i.e. **MiniMax's module-list forward really does sawtooth even in
   whole-prefill data, and the 1B really does not.** One intercept rule cannot be right for both.
3. **sdpa, mid-range S.** The score term is added to the intercept, and the two do not fully
   coexist — the score buffer peaks inside attention, the head+logits peak in the lm_head. Charging
   `max()` instead was tested and is worse in the dangerous direction (−38 % on Qwen at S=4096,
   −25 % on three others), so the safe `sum` was kept. The overshoot decays with S: on Llama-3.2-1B
   sdpa, +44 % at S=2048 → +25.6 % at 8192 → +10.8 % at 65536.

## 6. What this costs, and what it unblocks

The reserve grows: every geometry now carries `head_bytes` (0.46–0.79 GB) it did not before, plus
the sdpa score term whenever a model is loaded on sdpa. Derived `max_seq_len` therefore shrinks —
which is precisely the tradeoff **row 8** exists to decide (reserve vs expert-cache) and **row 9**
exists to re-derive and re-verify on-device. Both are now unblocked with a form that is at least
correct in sign.

## 6b. On-device verification (this change touches the load path)

Two bugs of exactly this shape shipped GREEN inside this same `__init__` block before (todo row 2),
so a real load was run on every forward after the edit. All four `rc=0`:

| arm | kernel | result |
|---|---|---|
| `test_batched_dense_smoke.py` (Llama-3.3-70B) | sdpa (forced) | pass, teacher-forced **perplexity 8.78** — identical to the documented baseline |
| Qwen3-235B load + forward (fused-EXL3 MoE) | flash_attention_2 | pass |
| MiniMax-M2.7 load + forward (module-list MoE) | flash_attention_2 | pass |
| Llama-3.2-1B load + forward | sdpa (forced) | pass |

Full suite: **865 passed, 3 skipped** (was 860/3 before this change; net +5 tests).

Derived `max_seq_len` after the change, measured at those loads:

| geometry | kernel | derived now | shipped before |
|---|---|---|---|
| qwen3-235b | flash | 8192 | 8192 (unchanged) |
| minimax-m2.7 | flash | 8192 | 8192 (unchanged) |
| llama-3.3-70b | sdpa | **3821** | 4373 (−13 %) |
| llama-3.2-1b | sdpa | **14052** | 106987 (−87 %) |

The two MoE geometries are untouched on flash — their intercept was already dominated by the
resident-expert term. The dense drops are the score term doing its job: 14052 is where the 1B's
predicted sdpa peak reaches the 3.50 GiB reserve floor, and its *measured* sdpa transient at
S=16384 is 3.72 GB, so the old 106987 was admitting contexts `batched_generate` could not serve —
exactly the failure row 10 predicted. Whether the reserve floor or the context should give is
**row 8's decision**, made with these numbers.

## 7. Row 6a — the residual that is left

Three geometries need a different intercept than the one term charges, and nothing derivable
explains the difference:

- **llama-3.3-70b** wants 0.928 G at a full chunk / 0.714 G at a short one, against 1.051 charged.
- **minimax-m2.7** wants 0.62–0.67 / 0.085, against 0.871 charged.
- **llama-3.2-1b** wants 0.792 flat and gets 0.788 — the one that fits.

Both hypotheses considered in row 6's brief are now testable but untested: a fixed unmodelled
residency term, or a per-forward difference in *when* the peak occurs. The decisive experiment is
the one row 6 already named — hold S fixed at one KV-capacity plateau and vary only the trailing
chunk length — plus per-chunk allocation instrumentation to find which chunk actually sets the
peak in the module-list forward. That is a GPU round and its own row.
