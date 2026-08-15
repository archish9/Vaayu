# NEXT-4 — the ~1.39 GB of fused-EXL3 prefill residency that `predict_peak_vram` does not model

**Status:** **DONE 2026-08-15 — implemented and the ±10 % gate PASSES** (0/26 outside, worst
7.3 %, zero fitted constants). Full verdict in `plans/settled_and_method_notes.md`. Do not re-run
this brief. · **Cost:** ~1 h to CONFIRM the mechanism (no GPU for the
arithmetic, one short probe to verify), then a design decision · **Sequence:** the only remaining
row that is real, actionable and unstarted.

> **This document is self-contained.** Mechanism, measured data, file/symbol names, the tradeoff
> that could sink the change, and the stop rule are all below. You do not need `todo.md` or any
> `docs/` file (that directory is gitignored and may not exist on a fresh clone). The 26 measured
> points live in `scripts/fit_vram_calibration.py`, which is tracked.

---

## 0. Read this first: the change is NOT urgent, and that shapes how much to spend

At the operating point both MoE models actually run, the true transient already fits the reserve:

| model | derived `max_seq_len` | reserve | true transient there |
|---|---|---|---|
| Qwen3-235B | 8192 | 3.50 GiB | ~2.76 GiB (KV 1.58 + 1.39 residency) |
| MiniMax-M2.7 | 8192 | 3.50 GiB | ~2.56 GiB |

The 3.5 GiB reserve floor (`_VRAM_RESERVE_FLOOR_BYTES`) and the KV doubling cliff both pin
`max_seq_len` to 8192 for both models **regardless of this error**, so fixing it changes nothing
operational today. It becomes live if the reserve floor changes, the hardware changes, or another
fused-EXL3 checkpoint lands with different expert geometry. **Do not let this grow into a
campaign.** If §4's confirmation fails, park it.

---

## 1. What is measured (do not re-measure)

Measured transient (`alloc_peak − alloc_before`) minus the two DERIVED terms `KV(at capacity)` and
`last-chunk logits`, with **zero fitted constants**, over 26 points on four geometries
(2026-08-15, torch 2.11.0+cu130):

| model | forward | residual mean | spread | verdict |
|---|---|---|---|---|
| llama-3.2-1b | dense non-EXL3 | −0.083 G | 0.129 G | explained |
| llama-3.3-70b | dense EXL3 streamed | −0.087 G | 0.129 G | explained |
| minimax-m2.7 | MoE module-list | +0.048 G | 0.142 G | explained |
| qwen3-235b | **MoE fused-EXL3** | **+1.106 G** | 0.536 G | **unexplained** |

Three of four forwards are predicted to within ±0.13 GB by KV + logits alone, with no constants.

**Qwen's peak does not contain the last-chunk logits at all.** Subtracting them leaves the residual
spread at **0.536 G**; leaving them in leaves **0.140 G** — 4× flatter, mean **1.386 G**. On that
forward a roughly constant residency dominates the logits, so the logits never SET the peak.

---

## 2. THE LEAD — confirm this before modelling anything

**1.386 GB / 11.8 MB per Qwen expert = 117.5 experts.** A full Qwen layer is 128 experts = 1.51 GB.
MiniMax's residual is 0.048 GB = 5.4 of its 8.9 MB experts, nowhere near its 256-expert layer —
which is exactly the asymmetry to explain, since only the fused forward shows the term.

The fused forward has a mechanism shaped like that: **chunk-major retention**. In
`_make_fused_sparse_forward_exl3` (`vaayu/moe_patch.py`), `_cm_gpu` / `vaayu_ref._chunk_major_gpu`
holds each expert's GPU tensors across chunks of a prefill, bounded by
`_expert_vram_cache.budget_bytes`, and is dropped on the last call. The module-list forward has no
equivalent.

**Task 1 — DONE 2026-08-15. CONFIRMED.** Instrumented `_chunk_major_gpu_bytes` at every expert
construction during multi-chunk prefills:

| S | peak `_chunk_major_gpu_bytes` | measured residual | delta |
|---|---|---|---|
| 5000 | **1.514 GB** | 1.314 GB | +0.201 GB |
| 7000 | **1.514 GB** | 1.428 GB | +0.086 GB |
| 12288 | **1.514 GB** | 1.366 GB | +0.149 GB |

**1.514 GB is constant in S, and 1.514 GB / 11.8 MB = exactly 128 experts = one full Qwen layer.**
Retention is reset per layer (`_chunk_major_gpu = None` on the last call), so the engine holds at
most one layer's routed experts on the GPU at a time. The expert VRAM cache budget (9.31 GB here)
is nowhere near binding.

The term slightly EXCEEDS the residual (+0.09 to +0.20 GB) because the retention peak and the
global allocation peak need not coincide in time. That is the SAFE direction for a reserve, and it
is 2.9 % of the 3.005 GB transient at S=7000 — comfortably inside the ±10 % gate.

**The term, derived, no literals:**

```
fused-EXL3 residency = num_experts * expert_bytes        # both from _moe_geometry
module-list / dense  = 0                                  # no chunk-major retention exists
```

**Do NOT clamp it to the expert VRAM cache budget, even though the runtime bound is
`min(retention, vram_cache_budget)`.** That clamp would make the term depend on
`expert_vram_budget_for(plan, free) = free - plan.reserve_bytes`, while the reserve depends on the
term — a circular definition. The unclamped form is conservative (it can only over-reserve) and on
this box the clamp is not binding by 6×. If a model ever appears where one layer's experts exceed
the cache budget, break the cycle deliberately rather than by accident.

**Worst case is what should be modelled:** `num_experts`, not the count actually routed. A full
512-token chunk routes essentially all of them, and over-reserving is the safe error for a value
whose job is "every request up to this length is safe".

---

## 3. Settled — do NOT re-derive

- **Calibrating `act_coeff`/`alloc_factor` cannot fix this** (refuted 2026-08-15, evidence in
  `scripts/fit_vram_calibration.py`). A joint fit over all four geometries gives 13/26 outside
  ±10 %, worst 44.2 %, and per-model fits want mutually incompatible constants — `alloc_factor`
  0.69–1.44, `act_coeff` −1.34 to +3.51, **negative for two models**, which is the signature of a
  form error absorbed by a free constant.
- **`act_coeff × S` is not observable.** Llama-3.3-70B's peak is *identical* at S=12288 and
  S=16384 — 8538230272 vs 8538263040 bytes, 33 KB apart — where the term predicts +0.13 GB.
  Llama-3.2-1B's entire growth out to S=49152 is accounted for by KV alone.
- **`alloc_factor` wants ~1.0**, per §1's table.
- **The KV cliff is real and already modelled** (`kv_capacity_tokens`): EXL3 checkpoints resolve
  `kv_cache_mode="auto"` to a doubling ring buffer, `capacity(S) = chunk · 2^ceil(log2(ceil(S/chunk)))`.
- **`MATERIALIZES_SCORES = False`** on this box — the scores term contributes nothing.
- **`predict_peak_vram` is NOT monotone in S**, and `solve_max_seq_len`'s two-level search depends
  on that. Any new term must preserve the two properties the proof needs: non-decreasing at chunk
  boundaries, and flat-or-climbing within a period. **A term that violates this silently breaks the
  "every S ≤ s_max is safe" guarantee** — that was a real latent bug once already.

---

## 4. Shape of the change (only after Task 1 confirms)

1. **Drop the activation term** (`act_coeff × S`) — unobservable, and two geometries fit it negative.
2. **Pin `alloc_factor` at ~1.0**, with the measured justification in a comment.
3. **Add a residency term for the fused-EXL3 forward only**, derived from `_moe_geometry`, never a
   literal (CLAUDE.md §5 — no `1.39`, no `128`, no `11.8` in the code).
4. **Suppress the logits term on that forward** — it is dominated, and subtracting it is what
   inflates the residual spread 4×. Note this makes the model forward-dependent, so `VramShape`
   needs to carry which forward it is, the same way it already carries `kv_capacity_buckets`.

---

## 5. The tradeoff that could sink it — decide deliberately

`plan_vram` sets `reserve = max(predict_peak_vram(shape, chosen), reserve_floor)`, and
`expert_vram_budget_for` gives the expert VRAM cache `free − reserve`. **So making the prediction
MORE accurate makes the reserve BIGGER and the expert VRAM cache SMALLER.**

That is a decode-throughput cost paid for a refusal-accuracy gain, and this repo has already priced
that trade twice: audit F8 measured the expert VRAM cache re-saturating in ~4 tokens (cold 37.2 %
vs warm 36.8 % hit rate), and Plan K measured giving back 1.22 GB moving Qwen's hit rate 43 %→41 %
with decode +0.4 % — i.e. unlocked nothing. So the expected cost is small, **but it must be
measured, not assumed**: same-box same-day alternating A/B, and this box cannot resolve <~4 %.

At today's operating point the reserve is floor-bound at 3.50 GiB and the true transient is
~2.76 GiB, so a correct model would **not** raise the reserve for Qwen at all. Confirm that before
worrying about the trade.

---

## 6. Stop rule

- **Task 1 fails to explain the 1.386 GB** → STOP. Record what it is not, and park. A term that is
  not derived from geometry is a per-model literal, which §5 forbids outright.
- **The term would break `solve_max_seq_len`'s monotonicity properties** → STOP and redesign; do
  not weaken the search.
- **Never tune a constant to make a gate pass.** Four rounds have held that line; do not be the
  fifth that breaks it.

---

## 7. Verification

- `vaayu/vram_plan.py` stays **PURE** — no torch, no CUDA, no I/O. ~60 tests depend on it.
- `./venv/bin/python -m pytest tests/ -q` → baseline **829 passed, 3 skipped**.
- Re-run `scripts/fit_vram_calibration.py`: the new form must bring all four geometries inside
  ±10 % *without* per-model constants. That script already prints the per-model residual table.
- On-device, per `plans/settled_and_method_notes.md`'s gate list: the dense smoke gate
  (perplexity **8.78** on Llama-3.3-70B, unchanged), and `MOE_CKPT=<ckpt> pytest
  tests/test_moe_accumulator_determinism.py` on both MoE models.
- Environment: `python` is NOT on PATH — use `./venv/bin/python`; scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`. One GPU arm per subprocess under
  `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`; **never load a model twice in
  one process** (measured: second load OOM-killed, exit 137).

## 8. Checkpoints on disk (`/home/a/code/ai-ml/models/`)

| model | dir | forward | expert size | measured at load |
|---|---|---|---|---|
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | MoE fused-EXL3 | 11.8 MB | 128 experts, top_k 8, moe_intermediate 1536, hidden 4096 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | MoE module-list | 8.9 MB | 256 experts |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | dense EXL3 streamed | — | — |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | dense non-EXL3 | — | the fast control, seconds per run |
