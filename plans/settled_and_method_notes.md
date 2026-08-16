# ARCHIVE — settled findings and method notes

Moved out of `todo.md` 2026-08-15 so that file is a pure pending list. Nothing here is a task.
Each line cost a full plan or probe to establish; re-measuring is wasted GPU.

---

# DONE — the fix sequence that closed (rows 1–6, closed 2026-08-09 .. 2026-08-15)

| # | Item | Verdict |
|---|---|---|
| 1 | Strengthen the dense smoke test (**D2**) | DONE 2026-08-09. Gate is teacher-forced perplexity, not token count. Caught D1 immediately. |
| 2 | Fix the dense path (**D1**) | DONE 2026-08-09. Tied `lm_head` was never weighted. Left behind **D4** (splitter still emits the empty shard) — now H7. |
| 3 | Make VRAM exhaustion fail honestly (**D3**) | DONE 2026-08-09. `_precompute_rope` raises with the OOM chained instead of returning an overloaded `None`. |
| 4 | F3 — prefix cache in `batched_generate` | Phase 1 + Task 6 DONE 2026-08-09; measured on all three forwards (4.96x / 2.15x / 1.76x). Phase 2 refuted. |
| 4b | MiniMax non-identical output under prefix reuse | DONE 2026-08-09. Not prefix reuse: bf16 expert accumulator + nondeterministic arrival order. fp32 accumulator in all three forwards. Left one open bug (fused-EXL3 residual). |
| 5 | Plan K — batched-decode admission reclaim | REFUTED 2026-08-09. `plan_batch` deferred ZERO requests in every cell of both models at N<=5 x 7.7k. Tasks 1-2 ship default-off. |
| 6 | F4 + the 7.7k->8.2k VRAM ceiling | DONE 2026-08-14, merged to `main` 2026-08-15 (`5205e84`..`4e873e8`). All 10 tasks shipped, suite **795 passed / 2 skipped / 1 xfailed**, five on-device gates PASSED. Left calibration (pending row 1) + three pending items. |

---

# DONE 2026-08-15 — moved out of `todo.md` when they closed

## Calibration gate (was todo item 1) — REFUTED

**Brief:** `plans/next_1_vram_calibration.md` (self-contained, 247 lines).
**Cost:** ~1–2 h, **NO GPU runs needed.**

### VERDICT 2026-08-15 — calibration REFUTED as the fix. The FORM is wrong, not the constants.

**Do not tune `DEFAULT_ACT_COEFF` / `DEFAULT_ALLOC_FACTOR`. They cannot close the ±10 % gate.**
Evidence and data: `scripts/fit_vram_calibration.py` (new, tracked, GPU-free, exits 1).
`scripts/probe_vram_curve.py` gained a per-point `alloc_before_bytes` column plus
`driver_resident=` / `alloc_resident=` in its header — per-point because a MoE model's expert VRAM
cache keeps growing across the S loop. Suite unchanged: **795 passed / 2 skipped / 1 xfailed**
(nothing under `vaayu/` was touched).

**What made this decidable:** four measured-baseline probe arms, 2026-08-15, ~60 min GPU, one
subprocess each under `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`, all
`status=ok` with no OOM even at S=16384. 26 points. Overlapping S reproduce the 2026-08-14 arm to
within **0.063 GB** worst (MiniMax 0.001 GB). With `transient = peak − alloc_before` the unknown
per-model offset is measured rather than estimated — and that is what exposed the form error, since
every earlier round had a free offset absorbing exactly this structure.

**The decomposition, with ZERO fitted constants** — measured transient minus `KV(at capacity)` and
`last-chunk logits`:

| model | forward | residual mean | spread | verdict |
|---|---|---|---|---|
| llama-3.2-1b | dense non-EXL3 | **−0.083 G** | 0.129 G | explained |
| llama-3.3-70b | dense EXL3 streamed | **−0.087 G** | 0.129 G | explained |
| minimax-m2.7 | MoE module-list | **+0.048 G** | 0.142 G | explained |
| qwen3-235b | **MoE fused-EXL3** | **+1.106 G** | 0.536 G | **unexplained** |

Three of the four on-device forwards are already predicted to within ±0.13 GB **by KV + logits
alone, with no constants at all.**

**Three reasons the two constants cannot fix the fourth:**
1. **`act_coeff × S` is not observable.** Llama-3.3-70B's peak is *identical* at S=12288 and
   S=16384 — 8538230272 vs 8538263040 bytes, a 33 KB difference — where the term predicts +0.13 GB.
   Llama-3.2-1B's entire growth out to S=49152 is accounted for by KV alone.
2. **`alloc_factor` wants ~1.0**, per the table above.
3. **The real gap is one forward's unmodelled residency**, and a global multiplier cannot express
   a term that is ~0 on three geometries and ~1.39 GB on the fourth.

Fitting both constants jointly across all four therefore **FAILS: 13/26 outside ±10 %, worst
44.2 %**, and the per-model fits are mutually incompatible — `alloc_factor` 0.69–1.44, `act_coeff`
−1.34 to +3.51, **negative for two models**, which is physically meaningless and is the signature
of a form error being absorbed by a free constant.

**Bonus finding — item 6 is CONFIRMED and localised.** Qwen's peak does not contain the last-chunk
logits at all: subtracting them leaves residual spread **0.536 G**, leaving them in leaves
**0.140 G** — 4× flatter. On the fused-EXL3 forward a larger, roughly constant residency (~1.39 GB)
dominates the logits, so the logits never SET the peak. Qwen's router geometry, for whoever models
it: 128 experts, top_k 8, `moe_intermediate_size` 1536, hidden 4096.

**What this row becomes.** Calibration is closed — refuted, per the rule that a refutation closes
its row. The live question is now a FORM change, which needs its own brief: drop the activation
term, keep `alloc_factor` at ~1.0, and add an expert-staging residency term for the fused-EXL3
forward derived from `_moe_geometry` (never a literal, CLAUDE.md §5). **That is a design change to
`predict_peak_vram`, not a calibration, so it needs its own approval before anyone writes it.**

**Item 4 was closed separately on 2026-08-15** — not by calibration, which could never have done
it, but by charging the request against the reserve. See item 4.

### What is wrong

`vaayu/vram_plan.py` — **PURE** module (no `torch`, no CUDA, no I/O, never imports
`vaayu_base`; ~60 tests run GPU-free because of it — **do not break that**). Two constants are
still placeholders:

```python
DEFAULT_ACT_COEFF = 4.0        # placeholder
DEFAULT_ALLOC_FACTOR = 1.18    # placeholder
```

Three fitting rounds against 17 real GPU-measured points all failed a **±10 %** gate:

| round | model change | points outside ±10 % | worst error |
|---|---|---|---|
| Task 5 | KV charged at `S` (naive linear) | 13/17 | **+41.2 %** |
| Task 5b | KV charged at bucketed capacity | 11/17 | **−30.2 %** |
| Task 5c | + trailing-chunk logits intercept | **10/17** | **−28.0 %** |

**No round tuned a constant to force a pass. Keep that line.**

### The model being fitted

| term | expression | notes |
|---|---|---|
| KV | `kv_per_token × kv_capacity_tokens(S, ...) × N` | charged at **allocated capacity**, not `S` |
| activations | `N × S × hidden × elem_bytes × act_coeff` | `act_coeff` uncalibrated |
| attn scores | `N × n_heads × chunk × elem × 2` if `materializes_scores` | **zero on this box** (measured) |
| logits | `N × trailing_chunk × vocab × (elem_bytes + 4)` | last chunk only; trailing = `S mod chunk` |

…all multiplied by `alloc_factor` (second uncalibrated constant).

### THE HYPOTHESIS TO TEST (this is the whole task)

**The probe measured a weights-INCLUSIVE figure against a weights-FREE model.**
`scripts/probe_vram_curve.py` records `torch.cuda.max_memory_allocated()` — absolute, including
resident weights, pinned layers, expert VRAM cache. `predict_peak_vram` has **no weights term**.
So all three fits compared *weights + transients* against *transients*: a **constant offset per
model**, not a slope error or missing mechanism.

Evidence already in hand (recomputed 2026-08-15, **do not re-run**) — MiniMax-M2.7, header
`total=16.62GB free_after_load=13.85GB` ⇒ ~2.77 GB resident after load:

| S | measured peak alloc | peak − resident | KV at bucket capacity |
|---|---|---|---|
| 5000 | 4.29 GB | 1.52 GB | 2.08 GB (cap 8192) |
| 6000 | 4.26 GB | 1.49 GB | 2.08 GB (cap 8192) |
| 7000 | 4.24 GB | 1.47 GB | 2.08 GB (cap 8192) |
| 7700 | 3.89 GB | 1.12 GB | 2.08 GB (cap 8192) |
| 8200 | 5.96 GB | 3.19 GB | 4.16 GB (cap 16384) |

Decisive, allocator-to-allocator: measured jump 7700→8200 = **2.06 GB** vs predicted KV-capacity
jump 8192→16384 = **2.08 GB** — a **1 % match on the step**. The mechanism is right; the residual
is flat at 1.47–1.52 GB across S=5000/6000/7000 — the shape of a missing weights term.

### Steps

1. **Locate the baseline for every logged point (no GPU).** Probe prints
   `[probe] total=16.62GB free_after_load=13.85GB` per run. Logs:
   `/tmp/vramcurve_NeuroSenko--MiniMax-M2.7-exl3.log`,
   `/tmp/vramcurve_Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw.log`,
   `/tmp/vramcurve_llama70b.log`, `/tmp/vramcurve_Llama-3.2-1B-Instruct.log`, plus `*_noexpand.log`
   allocator-control arms for the two MoE models. **`grep -a`, never `cat`** (MiniMax's is ~437 MB).
   **If the logs are gone:** do NOT re-run the sweep. `free_after_load` is driver-level while
   `max_memory_allocated` is allocator-level, so add ONE cheap probe line — record
   `torch.cuda.memory_allocated()` right after load, before the first prefill — and re-run **only
   Llama-3.2-1B** (seconds) to validate the method.
2. **Re-fit `act_coeff` and `alloc_factor` against `measured − baseline`, jointly across all four
   geometries** — never per-model; they are properties of the FORWARD and the ALLOCATOR
   (CLAUDE.md §5). Re-run the ±10 % gate arithmetically against the same 17 points.
3. **Decide honestly.**
   - **Pass** → write the constants into `vaayu/vram_plan.py` with a comment recording the date,
     torch version (**2.11.0+cu130**), that they were fitted across all four geometries, and that
     the fit is baseline-subtracted. Then check whether
     `tests/test_vram_plan.py::test_batch_width_at_the_real_reserve_floor_xfails` **XPASSes** —
     that is the designed signal that item 4 is closed.
   - **Fail** → **STOP and park it.** Record the residual pattern in the brief. Do not add a fifth
     term, do not tune. Next hypothesis is item 6 (the probe instrument), which is measurement
     work, not modelling work.
4. **Optional cleaner fix:** give `VramShape` a `resident_bytes` field from
   `torch.cuda.memory_allocated()` at load so `predict_peak_vram` returns an ABSOLUTE peak.
   **Interacts with a deliberate decision:** `_assert_context_fits` compares `need > free_vram`
   alone because `need` is a transient and `mem_get_info()`'s free figure already excludes this
   process's allocations. Make `need` absolute and that predicate must change too, or it silently
   double-counts. The current framing is coherent — change only if absolute measurably fits better.

### Why it is capped at ~1 h of effort

The reserve is clamped to `_VRAM_RESERVE_FLOOR_BYTES` (3.5 GB, `vaayu/vaayu_base.py`), so
calibration does **not** affect reserve, pinning, expert cache, or decode throughput. It affects
exactly two things: `s_max` (the honest-refusal ceiling) and `_assert_context_fits`. Uncalibrated
= occasional ugly CUDA OOM instead of a clean `VramPlanError` — pre-F4 behaviour, a diagnostics
issue.

### Do NOT re-derive (settled)

- The KV cliff mechanism: `kv_cache_mode="auto"` on any EXL3 checkpoint → `FP8DynamicCache(
  quantize=False)` (`_create_kv_cache`), whose `_ensure_capacity` doubles: `_GROWTH = 2`,
  `new_cap = max(needed, max(1, cap) * _GROWTH)`. Closed form, already implemented as
  `kv_capacity_tokens`: `capacity(S) = chunk · 2^ceil(log2(ceil(S/chunk)))`. Non-EXL3 Llama-1B
  control shows **no cliff** — format-dependent.
- `MATERIALIZES_SCORES = False` on torch 2.11.0+cu130 → the scores term contributes nothing.
- `predict_peak_vram` is **NOT monotone in S**; `solve_max_seq_len`'s **two-level search** handles
  it. Do not "simplify" it back to a single binary search — that was a real latent correctness bug.
- `expandable_segments:True` is already the default (`vaayu/__init__.py`, via `setdefault`). The
  informative control arm is `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.

---

## N>=2 admission gap (was todo item 4) — FIXED

**The earlier diagnosis in this file was wrong, and so was the xfail's reason string.** This was
never a calibration gap. Even an exact `predict_peak_vram` returns ~7.5 GiB for MiniMax at N=3,
S=8192 — which genuinely fits a bare 15.47 GiB card. A bigger prediction could never have produced
the refusal.

**Real cause:** `expert_vram_budget_for(plan, free_vram_now)` = `free_vram_now − plan.reserve_bytes`,
so the expert VRAM cache is entitled to *everything above the reserve* — and it fills **lazily,
during the very prefill being admitted**. `_assert_context_fits` compared against live
`mem_get_info()`, so the answer depended on cache warmth:

| cache state | free VRAM seen | N=3 need (MiniMax, S=8192) | outcome |
|---|---|---|---|
| warm | ≈ reserve, 3.5 GiB | 7.67 GiB | refused — correct |
| **cold** | ≈ 13 GiB | 7.67 GiB | **admitted** → cache expands into the same bytes → OOM |

**Fix:** new pure `admission_headroom(free_vram, plan, expert_cache_active)` in `vaayu/vram_plan.py`
returns `min(free_vram, plan.reserve_bytes)` when a VRAM expert cache exists, and `free_vram`
untouched otherwise (dense checkpoints have no such cache — clamping there would refuse contexts
that fit). The reserve is exactly the right budget: `plan_vram` sizes it as
`max(predict_peak_vram(shape, max_seq_len), reserve_floor)`, i.e. one N=1 request at
`max_seq_len`. A fan-out of N multiplies the request but not the reserve, so N≥2 at long context is
refused — matching Plan K's on-device result.

`_too_big_message` gained an optional `headroom` so the refusal does not read as a contradiction
(D3's lesson): it now names the budget, the real free figure, and the expert cache as what took the
difference. Dense refusals are byte-identical to before, pinned by a test. It also stops offering
`s_max` unqualified at N>1 — `solve_max_seq_len` solves it at N=1, so printing it bare while
refusing a wide batch pointed the reader at a length that does not fit either.

**Deliberately NOT reclaim-aware** — Plan K's reclaim runs *after* this check and ships default-off
because it was refuted; even reserve + reclaim (1.22 GB Qwen / 2.09 GB MiniMax) stays under what
N≥2 needs. Recorded in the function's docstring: if reclaim ever moves ahead of admission, that is
the function that must learn about it.

**What this buys:** a clean `VramPlanError` instead of a mid-prefill CUDA OOM. It does **not**
unlock N≥2 — that remains a hardware wall on this box.

Tests: `test_headroom_is_capped_by_the_reserve_when_an_expert_cache_is_active`,
`test_headroom_is_the_whole_free_vram_without_an_expert_cache`,
`test_headroom_never_exceeds_the_vram_that_is_actually_free`,
`test_batch_width_at_the_real_reserve_floor_is_refused` (was a non-strict xfail; now a real test
with the correct premise), `test_the_admission_predicate_charges_a_wide_batch_against_the_reserve`
(drives the real `VaayuBaseModel._assert_context_fits`), plus two on message honesty.
Suite: **803 passed / 2 skipped / 0 xfailed** (was 795 / 2 / 1).

---

## Hygiene backlog H1–H7 (was todo row 1) — CLOSED 2026-08-15

All seven items resolved. Suite went 803 → 824 passed / 2 skipped / 0 xfailed.

| item | what it was | resolution |
|---|---|---|
| **H1** | `_create_kv_cache` fell through to fp8 on an unrecognised `kv_cache_mode` | `fp8` is now an explicit branch; anything else raises `VramPlanError` naming the value and the known modes, worded to match `kv_elem_bytes` so planner and cache agree. `tests/test_kv_cache_mode_dispatch.py`. |
| **H2** | `SlabArena`'s O_DIRECT alignment premise asserted in prose, enforced nowhere | `compute_slab_bytes` now `_align_up`s its result, and `SlabArena.__init__` raises on a misaligned `slab_bytes`. `tests/test_slab_alignment.py` builds the case that bites — 300 contiguous 1-byte tensors make `packed_span` (16-aligned, 4800 B) exceed `raw_aligned_length`, so the old code returned a slab size off every sector boundary. One existing test used a toy `slab_bytes=64` and was moved to `_ALIGN`. |
| **H3** | `KVCacheManager` unreachable but maintained | **Split, because the module was not uniformly dead.** Deleted `KVCacheManager` and its INT8 helpers (no other caller) and `tests/test_kv_cache_manager_fp8.py`. KEPT `_extract_kv_as_bf16` — `tests/test_turboquant.py` uses it as an independent oracle for the TurboQuant packed-indices corruption case — and moved it to `vaayu/kv_extract.py`, since a file named `kv_cache_manager.py` with no manager in it is the same misleading-name defect H4 is about. Its one surviving test moved into `test_turboquant.py` beside the other two. |
| **H4** | `_expert_history` write-only, docstring claims a prefetch loop reads it | **Docstring fixed, write KEPT** — deviating from "remove the write", with reason: `_expert_history` is not write-only in the tests. Eight-plus assertions across `tests/test_moe_patch.py` use it as the observation hook for which experts a forward actually routed to, on all three forwards. Removing it would cost those tests their only view of routing to save one dict assignment per layer. The docstring now says plainly that there is no production reader and that the prefetch mechanism never existed. |
| **H5.1** | no `chunk_count` / `charged_bytes` accessor | Both added as properties; `vaayu_base.py` stops reaching into `_chunks`. |
| **H5.2** | stale "`arena_slab_count` IS that gate in production" | Corrected — Plan L took it off the arena path, which calls `unclamped_slab_count`. **Noted while there: `arena_slab_count` now has NO production caller at all.** Left in place deliberately (CLAUDE.md §3: mention unrelated dead code, do not delete it). |
| **H5.3** | no test for an allocation failure on an ESCALATED retry | Added. It passes on unchanged code — a coverage gap, not a bug — and its premise asserts (`calls[1] > calls[0]`) prove it really reaches the doubled request rather than passing on the first-request path. |
| **H5.4** | zero-chunks error recomputed `chunk_decomposition()` for a count | Decomposition computed once into `_planned_chunks`. |
| **H5.5** | pinned→pageable fallback could leave the arena `None` silently | Now prints `slab_arena_exhausted_warning` — the helper the sibling path already used. This was the fourth silent-degradation defect on that branch. |
| **H6/F13** | `_decode_stats_reset` latched per PROCESS, not per turn | New pure `decode_stats_reset(...)` returns `(reset_now, latched)`; a prefill pass re-arms the latch, so every turn's first decode clears prefill's ~100 %-miss pollution instead of only the first turn's. `tests/test_decode_stats_reset.py`. |
| **H6 (other two)** | `matched_tokens` ≤1-token over-report; `hit_rate` not comparable across lookup modes | Both were already recorded as deliberate non-fixes, documented in `stats()` and pinned by tests. No change. |
| **H7** | D4 empty `lm_head` shard; `speculative_generate` missing `enqueue_truncate` | Both fixed 2026-08-15 — see their own entries above. |

---

## Fused-EXL3 / module-list decode order-nondeterminism (was todo row 2) — FIXED 2026-08-15

**The bug was NOT where the ledger said.** Three claims corrected by measurement:

1. **The EXL3 kernel is bitwise deterministic.** Built a `BC_LinearEXL3` from one real Qwen expert
   read off `splitted_model/model.layers.9.safetensors` — no model load — and called `run_alloc` on
   identical fp16 input at T = 1, 8, 512, 2048, 7700: five repeats on the same object, a rebuilt
   object, and a call interleaved with a second linear sharing the same `g_tensor_cache` `xh`
   scratch. **All identical, max diff 0.000e+00.** "Split-K/atomics in the fused kernel" is refuted,
   and so is "cost of chasing it is likely a third-party kernel change".
2. **Prefill-only forwards are reproducible** — 3 runs, identical logits, identical weight multiset,
   identical call order, 20991 BC constructions every run. `_moe_prefill_pass` suppresses cache
   puts during prompt ingestion, so nothing is retained and arrival order never varies. (A first
   probe that CLEARED caches between runs also showed no divergence — that null was the harness
   removing the mechanism, not evidence.)
3. **Decode DOES flip greedy tokens**, which the row denied. Three `do_sample=False` runs:
   `[68, 377, 291, 83, 349, 11]` vs `[68, 377, 291, 220, 15, 15]` twice — two distinct sequences,
   diverging at the 4th new token, with BC construction counts 25380 / 24570 / 24696 showing cache
   warmth changing the order.

**Root cause:** both completion-order forwards accumulated expert outputs in ARRIVAL order.
Execution order varies (VRAM-resident experts first, the rest in disk completion order via
`_iter_expert_loads`' `FIRST_COMPLETED`), each token routes to `top_k` experts so contributions
overlap the same rows, and floating-point addition is not associative at any width.

**The fp32 accumulator was not the fix it was recorded as.** Its comment claimed fp32 made "the
cast result identical whatever the order". False: fp32's error is normally far below the bf16
output's own precision, but a value on a bf16 rounding boundary tips regardless, and across 94
layers something always does. Width narrows the window; only a fixed summation order closes it.
Both comments now say so.

**Fix:** `accumulate_expert_outputs(final_hidden, contributions)` buffers `(expert_idx, token_idx,
hidden)` during the loop and replays `index_add_` **sorted by expert id** afterwards. Execution
stays in completion order, so the disk overlap — the entire point of `_iter_expert_loads` — is
untouched. Wired into the two affected forwards. `_make_fused_sparse_forward` (non-EXL3) iterates
`for eidx in unique_experts`, a fixed list, so it was never affected and is unchanged.

**Verification:**
- Qwen3-235B, same probe that failed: **1 distinct sequence of 3** (was 2), while BC construction
  counts still vary (25848 / 24714 / 24714) — order still varies, output no longer depends on it.
- MiniMax-M2.7 (module-list): 3 runs identical. This is a **regression check, not a before/after** —
  MiniMax was already reported deterministic, so it proves the change did not break that path.
- Decode A/B, alternating arms against one loaded model, control = same buffer replayed unsorted:
  **−0.29 %** (0.2032 vs 0.2038 tok/s), far inside this box's ~4 % resolution. Peak VRAM
  **identical at 12.531 GB** both arms. (Absolute tok/s here includes prefill and is an A/B figure
  only — not comparable to the 0.83 decode floor.)
- `tests/test_expert_accumulation_order.py`: 5 CPU-only tests, including an asserted premise that
  the fixture really is order-sensitive (1e8 + 1.0 rounds away, (1e8 + −1e8) + 1.0 does not).

**Memory:** buffers `tokens × top_k × H` of fp16 — 64 KB at decode, ~32 MB for a 512-token prefill
chunk. Applied to both phases deliberately: prefill uses the same completion-order iterator, so its
determinism was empirical (one 512-token chunk, nothing retained), not structural.

---

## F3 broadcast prefix reuse on MoE (was todo row 5) — RE-SCOPED, gate now passes 2026-08-15

**Token-identity under prefix reuse is UNACHIEVABLE on an MoE model. The old gate asserted it, so
it was asserting a bug that cannot be fixed.** Established by three probes, not by argument.

**It is not nondeterminism.** Three COLD batched rounds are bitwise identical to each other and to
the gate's cold arm — the engine IS a deterministic function of its input (the arrival-order
accumulation bug that used to break that was fixed the same day). This is *path-dependence*: two
routes equal in exact arithmetic, unequal in floating point.

**It is not the mask, the positions, or a lossy pool.** First decode step, cold vs warm:

| row | tail | pads | mask equal | positions equal | max\|Δlogit\| | relative |
|---|---|---|---|---|---|---|
| 0 | 7 | 0 | **yes** | **yes** | **1.125** | 4.9 % |
| 1 | 4 | 3 | no | no | 1.281 | 9.0 % |
| 2 | 3 | 4 | no | no | 2.371 | 17.1 % |

Padded rows encode differently — cold puts pads at the FRONT (`[0,0,0,1×15]`), warm puts them
INSIDE (`[1×10,0,0,0,1×5]`) — but both are semantically valid and the REAL tokens get identical
positions. Row 0 settles it: identical mask, identical positions, `store()` on its lossless
`compress_bits is None` branch, and still 1.125 of logit difference. (An interim reading that "only
padded rows diverge" was the argmax fooling the tokens — every row differs numerically.)

**Root cause: call-shape rounding amplified by discontinuous top-k routing.** Cold prefills all 17
tokens in one pass; warm prefills a 7-token tail against a 10-token restored prefix. A 7-query GEMM
tiles differently from a 17-query one, so layer 0's attention output shifts ~1 ULP — and on an MoE
that flips a top-k selection. Measured at the first decode step on MiniMax-M2.7: **53 of 62 layers
routed a different expert set**, first divergence at **layer 1** (expert 137 → 139), one layer even
differing in COUNT (14 vs 15).

**Both MoE models drift; the old note's "Qwen is token-identical" was prompt luck.** Drift onset by
row (None = never diverged over 8 tokens): MiniMax `{0: None, 1: 4, 2: 6}`, Qwen `{0: 3, 1: None,
2: 1}` — Qwen's row 2 diverges at the SECOND token. Dense Llama-3.3-70B is token-identical, as it
has no discontinuous routing to amplify the rounding.

**Re-scoped gate** (`tests/test_batch_prefix_equivalence.py`, renamed
`test_broadcast_reuse_preserves_behaviour`), branching on `model.is_moe`:
- **dense** — token-identical, exactly as before. Verified: Llama-3.3-70B **passes** (123 s).
- **MoE** — (1) round 2 HITs, the unchanged premise; (2) every row's FIRST token is unchanged by
  reuse; (3) two warm rounds are token-identical to each other. Verified: MiniMax **passes** (94 s),
  Qwen **passes** (142 s). Drift onset is PRINTED, not asserted.

A broken mask or an unpopulated pool yields garbage from token 1 and still fails (2). **Known flake
risk, deliberately accepted:** (2) leans on the argmax margin, and Qwen's row 2 held by exactly one
position. If it ever flakes, that is a finding about how close reuse runs to the edge — do not
suppress it.

---

## F1 — expert-cache / prefix-pool RAM arbitration (was todo row 2) — CLOSED AS OBSOLETE 2026-08-15

**Premise refuted by measurement; nothing built.** F1 assumed the expert RAM cache took its share
first and left the prefix pool starved, with nothing arbitrating between them. Measured across
three loads per model, separate processes, `VAAYU_NO_WARMUP=1`, on an otherwise idle box — every
number the task needed was ALREADY printed at load, so no instrumentation was added:

| model | expert RAM cache | slab arena charged | free RAM at load | prefix pool | spread over 3 loads | KV/session @ 8192 | K_max |
|---|---|---|---|---|---|---|---|
| Qwen3-235B | 9.1 GB | 9.24 GB | 15.71–15.78 GB | **11.95–12.02 GB** | 0.07 GB | 1.58 GB | **7** |
| MiniMax-M2.7 | 9.1 GB | 9.21 GB | 15.46–15.52 GB | **11.71–11.77 GB** | 0.06 GB | 2.08 GB | **5** |

**The two budgets do not overlap, so there is nothing to arbitrate.** The slab arena is allocated
FIRST and its ~9.2 GB is already gone from the free-RAM figure the pool then sizes itself against
(`free RAM − 3.76 GB reserve`). The arithmetic closes by construction; F1's "two budgets, each
plausible alone, jointly overcommitting" describes the VRAM situation that `plan_vram` fixed, not
this one.

**Capacity vastly exceeds need.** Realistic concurrency here is **sequential turns of ONE stream** —
N≥2 at long context is separately settled as infeasible on this box — against a measured K_max of
5–7 sessions. Even allowing the cliff (exceeding pool capacity costs the ENTIRE win, not a share:
K=7 against a pool holding 5–6 gave 0 hits, Plan J Task 4), there is 5–7× headroom over a need of 1.

**On the old 2.36 GB reading** that motivated F1: today's pools are stable to within 0.07 GB across
six loads. That earlier scarcity was environmental — other processes holding RAM at load — not the
expert cache crowding the pool out. Per the brief's own decision rule, the remedy for environmental
variance would be a FLOOR on the pool, not an arbitration mechanism, and no measurement today shows
one is needed. Re-open only if a load is observed where the pool falls below one session's KV.

---

## `predict_peak_vram` form change (was todo row 2) — DONE 2026-08-15, gate PASSES

**The ±10 % accuracy gate passes for the first time: 0/26 points outside, worst 7.3 %, with ZERO
fitted constants.** Prior best was 10/17 outside at worst −28.0 %, across four rounds of trying to
calibrate two constants that turned out to be the wrong lever.

**The form now:** `peak = KV(at allocated capacity) + max(last-chunk logits, resident experts)`.

Four changes, each measured rather than fitted:

1. **The activation term is GONE.** `act_coeff × S` was unobservable — Llama-3.3-70B's measured
   peak is IDENTICAL at S=12288 and S=16384 (33 KB apart) where it predicted +0.13 GB. `act_coeff`
   and `DEFAULT_ACT_COEFF` are deleted, not defaulted to zero.
2. **`alloc_factor` = 1.0.** KV + logits alone explains three of four forwards to ±0.13 GB.
3. **Resident-experts term**, non-zero only for fused-EXL3, derived as
   `num_experts × expert_bytes` from `_moe_geometry`. Confirmed by instrumenting
   `_chunk_major_gpu_bytes`: a flat **1.514 GB = 128 Qwen experts = one full layer**, constant
   across S=5000..12288, reset per layer. The caller DETECTS the layout (`expert_style == "fused"`
   and EXL3) rather than assuming it — charging MiniMax's 256 experts would over-reserve 2.28 GB.
4. **Logits charged at 4 bytes/element, not `elem_bytes + 4`.** Only the fp32 upcast is resident at
   the peak, not the fp16 copy too. Measured at a full trailing chunk on THREE geometries: the
   residual is 0.2627 GB on llama-3.3-70b (S=12288 AND 16384) and llama-3.2-1b (S=16384, 32768,
   49152); `trailing × vocab × 4` = 0.2627 GB exactly, `× 6` = 0.3940 GB. **Five points, four
   decimals.** This was found BY the gate: with ×6 it failed 4/26, all on llama-1b.

`max(logits, residency)` rather than their sum: retention lives inside the layer loop and is
dropped on its last call; the logits materialize afterwards in the lm_head, so the two never
coexist.

**Effect on derived `max_seq_len`:** Qwen **8192 → 8192**, MiniMax **8192 → 8192** (both pinned by
the reserve floor and the KV cliff, exactly as the brief predicted), Llama-3.3-70B **4373 → 8192**,
Llama-3.2-1B **56828 → 106987**. The dense models gain context because an unobservable term was
inflating their predicted peak.

**Verification:** suite **836 passed / 3 skipped**; `scripts/fit_vram_calibration.py` (now a form
CHECK, not a fit) exits 0; dense smoke perplexity **8.78**, unchanged; MoE accumulator gate passes
on Qwen and MiniMax.

**Known residual, not chased:** MiniMax sits ~0.079 GB above even the ×6 logits figure at a full
trailing chunk — its module-list forward holds something small this does not model. It leaves that
point under-predicted by 5.8 %, inside the gate. That is the thing to chase if the bar tightens.

**Two bugs I introduced and caught on-device, both invisible to the suite:** referencing
`detect_moe_blocks` and then `EXL3LoaderStrategy` before their in-function imports made them
unbound locals, and **both MoE models failed to load** while 836 tests stayed green. Nothing
unit-tests the load path. Run a real load after touching `__init__`.

---

# SETTLED — do not re-open without new evidence

Not tasks. Each cost a full plan or probe to establish; re-measuring them is wasted GPU.

- **F4 + the 7.7k→8.2k VRAM ceiling: the framing question is ANSWERED and the plan's hypothesis
  REFUTED. It is a real CLIFF, not "a ceiling grazed by a linear term" (settled 2026-08-14,
  F4 plan Tasks 5/5b/5c, verified on-device by Task 10).** Root cause, traced in code, not just
  measured: `kv_cache_mode="auto"` on ANY EXL3 checkpoint resolves to `FP8DynamicCache(quantize=
  False)` (`vaayu_base.py`'s `_create_kv_cache`), a ring-buffer KV cache whose `_ensure_capacity`
  grows capacity by **geometric doubling** — `_GROWTH = 2`, `new_cap = max(needed, max(1, cap) *
  _GROWTH)`. At `prefill_chunk_size=512`, capacity locks at 8192 tokens for every S in [4609, 8192]
  and jumps to 16384 the instant S exceeds 8192 by even one token — exactly the 7.7k→8.2k boundary
  the plan set out to characterize. Closed form: `capacity(S) = chunk · 2^ceil(log2(ceil(S/chunk)))`.
  Confirmed within 7% of `kv_bytes_per_token × 8192` on THREE independent geometries: Qwen fused-MoE
  (−6.8%), MiniMax module-list-MoE (−0.8%), Llama-70B dense (−6.4%) — all EXL3. **The non-EXL3
  Llama-3.2-1B control (plain `transformers.DynamicCache`, no ring buffer) shows NO cliff over the
  same S range**, proving the mechanism is format-dependent (EXL3's KV cache implementation), not a
  property of context length or of any one model. `predict_peak_vram` now models this as a derived
  bucket term (`kv_capacity_tokens`, parameterized by `kv_growth_factor`/`kv_capacity_buckets`, both
  read off the live cache class per CLAUDE.md §5 — never a literal). Verified on-device (Task 10,
  2026-08-14): derived `max_seq_len` per model — Qwen3-235B 8192, MiniMax-M2.7 8192, Llama-3.3-70B
  4373, Llama-3.2-1B 56828, all against the old flat 512 — and the honest-refusal gate (Gate 3) fires
  a clean `VramPlanError` at `s_max+512` with no OOM. Full gate results, raw logs, and the derived
  numbers per model: `docs/superpowers/plans/2026-08-09-vram-plan-RESULTS.md` (Task 10 section).
- **`solve_max_seq_len`'s single-level binary search was UNSOUND, because `predict_peak_vram` is NOT
  monotone in S once the KV-cliff/trailing-chunk terms are both modeled (fixed 2026-08-14, F4 plan
  Task 5c).** The trailing-chunk intercept resets DOWN at every ordinary chunk boundary (the last
  chunk's logits shrink from `chunk` tokens' worth back to 1) while the linear activation term only
  grows by one token and the KV term is flat within a capacity bucket — so peak VRAM can go DOWN as
  S goes up (proven concretely: Qwen at `S=5·chunk` vs `S=5·chunk+1`, `predict(S) > predict(S+1)`). A
  single-level binary search over a non-monotone function can silently return an S for which some
  SMALLER S′ does not actually fit — dangerous for a value whose entire purpose is "every request up
  to this length is safe." **Fixed with a two-level search**: first an exponential+binary search over
  whole chunk PERIODS against `predict_peak_vram(shape, k·chunk)` (provably monotone in the period
  count k — boundary values are non-decreasing since trailing chunk is constant there, KV never
  decreases, and the slope term is non-negative), then a bounded search within the one remaining
  partial period (also monotone, since KV is flat inside a period and trailing chunk climbs
  monotonically). Every S ≤ the chosen answer is now provably safe, not merely the answer itself.
  Pinned by `test_solve_max_seq_len_stays_safe_for_every_smaller_context` (exhaustive on a tiny
  synthetic shape) and `test_solve_max_seq_len_can_land_exactly_on_a_chunk_boundary` (the real Qwen
  S=8192 KV-doubling boundary). This was a latent correctness bug independent of the fit gate's
  pass/fail — it would have been wrong even if `act_coeff`/`alloc_factor` had been calibrated.
- **The OOM handler SILENTLY HALVED THE KV CACHE mid-forward (fixed 2026-08-09). This was F3's unreproduced mask/KV bug.**
  `vaayu_base.py:3563-3571`, inside the layer loop's `except torch.cuda.OutOfMemoryError`:
  `_oom_target = _oom_seq_len // 2` then `_trim_kv_cache_front(...)` → `crop_front`, then the layer
  retries. Nothing tells the CALLER, so `batched_generate`'s `pad_mask` still describes the full
  sequence and the next forward mismatches.
  **Two independent incidents, both landing on exact `// 2`:**
  | incident | mask | KV | delta |
  |---|---|---|---|
  | F3 Task 6, unreproduced (2026-08-09 am) | 1127 | 1690 | 563 = **1127 // 2** |
  | Plan K, N=3 @ 7 700 (2026-08-09 pm) | 7 770 | 3 885 = **7 770 // 2** | — |
  This is why it never reproduced: it fires only when a layer actually OOMs, which depends on expert
  cache warmth and free VRAM — the same config crashes or does not run to run.
  **It is worse than a shape bug.** `crop_front` drops the OLDEST tokens, so every surviving token's
  position shifts relative to both the attention mask and the RoPE positions already baked into the
  cached keys. A run that "recovers" and completes has silently answered from a truncated,
  position-shifted context. The per-layer variant is nastier still: the OOM hits at layer i, so
  layers 0..i-1 hold the pre-crop length and layer i onward hold post-crop + the retried chunk —
  ragged `_seq_len`, which is exactly what `check_batch_kv_invariant`'s per-layer arm now catches.
  **Confirmed by log, not inferred** — MiniMax N=3 @ 7 700, pool off:
  ```
  [Vaayu] OOM on model.layers.35 (attempt 1/2). Clearing cache and retrying...
  [Vaayu] Truncating KV cache: 7770 -> 3885 tokens
  RuntimeError: The expanded size of the tensor (3886) must match the existing size (7770)
  ```
  **FIXED:** deleted the truncation. The `empty_cache()` + `gc.collect()` retry is
  non-destructive and can stay; silently shrinking the user's context to survive is the degradation
  CLAUDE.md §5 forbids, and D3 already settled that a VRAM wall must raise with the real levers
  named rather than fabricate a "recovery". **Consequence for measurement:** configs that used to
  "recover" and finish now raise honestly, so a sweep cell at high N × long context reports an OOM
  where it previously reported a number. Any throughput figure taken at the VRAM edge before this
  date may have been measured on a silently truncated context.
- **Plan K is REFUTED: batch admission was never the limiter (2026-08-09).** `plan_batch` deferred
  **zero** requests in **every cell of both models**, at N up to 5 and ~7 700-token streams — the
  exact context F2 was written about. `reclaimable_vram_budget` and the once-per-batch reclaim both
  work (freed 1.22 GB Qwen / 2.09 GB MiniMax when asked) and unlock nothing. F2's "N=1 today vs N=5
  if counted" compared per-stream KV against the admission budget; **KV admission is not what binds.**
  Total VRAM residency is — prefill activations + KV + expert cache + (pool on) restored prefix KV,
  against 15.47 GiB. Peak allocation was 14.17-15.57 GB in every cell that ran.
  **Reclaim's measured cost is nil and so is its benefit:** MiniMax 0.713 → 0.716 tok/s (+0.4 %),
  Qwen's four N=1 arms span 0.454-0.488 with no ordering by reclaim. Giving back 1.22 GB moved the
  expert-VRAM hit rate 43 % → 41 %, consistent with audit F8's "the cache re-saturates in ~4 tokens".
  **Keep both default-off rather than deleting them** — the mechanism is sound and tested; only its
  premise failed, and that premise could return if batched prefill activations come down.
  RESULTS: `docs/superpowers/plans/2026-08-03-batch-admission-reclaim-RESULTS.md` (docs/ is gitignored).
- **N≥2 at ~7 700 tokens is INFEASIBLE on this 16 GB box, both MoE models**, with or without reclaim
  or the prefix pool. Only N=1 runs. 14 of MiniMax's 16 cells died of VRAM exhaustion, and Qwen's
  cheapest N=2 cell did too. **§0 names parallel fan-out as a first-class lever; at agentic context
  lengths on this hardware it is currently unreachable**, and admission control is not what stands in
  the way. **Limit on this claim:** Qwen N=3 and N=5 were never executed — the sweep was stopped at 21
  of 32 cells once the remainder could not change the verdict. Re-measure rather than extrapolate if
  per-cell VRAM drops.
- **Cross-turn prefix reuse beats every batching lever by two orders of magnitude, and works where
  batching does not.** Same box, same day, N=1 @ ~7 700 tokens: Qwen cold 396.4 s → warm **2.4-2.9 s
  (136-165×)**; MiniMax ~380 s → **~1.0 s (377×)**, `match_len=7767/7768`. For §0's agentic loop —
  which re-sends a largely-unchanged context every iteration — **this is the lever, and batching is
  not** at this context length on this hardware.
- **The batched store-back cloned the KV row in VRAM, so the prefix cache could not store at long
  context (fixed 2026-08-09).** `slice_cache_row` did `t[row:row + 1].clone()` — a second full copy
  of the KV cache on the SOURCE device — immediately after a long prefill, when VRAM is at its peak,
  and `store()` moved it to the CPU anyway. At 7 700 tokens on MiniMax it died allocating **16 MiB
  with 19.56 MiB free**. The store-back deliberately swallows exceptions (it must not replace an
  exception already unwinding through that `finally`), so the pool was simply never populated and
  the failure printed one warning nobody was reading.
  **Measured, same box same day, MiniMax N=1 @ 7 768 tokens:**
  | arm | round 1 (cold) | round 2 (warm) | ratio |
  |---|---|---|---|
  | before fix | 380.9 s | 352.8 s | **1.08×** (no hit at all) |
  | after fix | ~380 s | **~1.0 s** | **377×**, `match_len=7767/7768` |
  Slicing straight to host fixes it: `t[row:row + 1]` is a contiguous view, so the device slice
  costs nothing and only the host copy allocates.
  **Two lessons:** (a) **the prefix cache was broken exactly where it is most valuable** — short
  contexts fit, so every earlier measurement passed; (b) **a swallowed exception plus a plausible
  number is worse than a crash.** The pool-on arm of a 16-cell sweep would have read as "prefix
  reuse does not help at long context", contradicting F3's measured 2.15× with nobody able to say
  why. Check that a HIT actually happened before believing an arm that depends on one.
- **The MoE forwards accumulated experts in bf16, and expert ARRIVAL ORDER is nondeterministic — so
  the engine was not a deterministic function of its input (fixed 2026-08-09).** Experts reach
  `index_add_` in an order that varies run to run: VRAM-resident experts run first (residency
  depends on cache warmth) and the rest arrive in **disk COMPLETION order** — `_iter_expert_loads`
  waits `FIRST_COMPLETED` and iterates the resulting set. `final_hidden` was
  `torch.zeros_like(hidden_states)`, i.e. **bf16**, whose ~2⁻⁸ rounding makes addition
  non-associative at the 1-ULP scale. Measured on MiniMax, two forwards over identical ids: layer 0
  bitwise identical, **layer 1 off by one bf16 ULP (0.00195)**, compounding to **37.0 by layer 62**
  and max|Δlogit| **2.29** — **three distinct outputs from four identical greedy runs**
  (`temperature=0.0`). Fix was an fp32 accumulator cast back at the return, in **all three** forwards.
  After: bitwise identical, 4/4 **on MiniMax** — but see the 2026-08-15 entry above: fp32 only
  NARROWED the window. Qwen still flipped greedy tokens until the summation order itself was
  fixed. Do not cite this entry as evidence that accumulator width removes order dependence. Decode cost is **nil** — alternating same-day A/B, MiniMax
  0.7833 vs 0.7812 tok/s (+0.27%), Qwen 0.4459 vs 0.4430 (+0.66%), both far inside this box's ~4%
  resolution. Dense was **bitwise identical throughout**, before and after: it has no such
  accumulation, which is what made it the control that localised this.
  **Two lessons worth more than the fix:**
  (a) **A perf-driven ordering optimisation silently became a correctness bug.** Consuming experts
  in completion order is exactly right for overlapping disk reads, and it is what made the output
  order-dependent. Anywhere output is accumulated in the order work completes, the accumulator's
  precision is load-bearing.
  (b) **Every MoE measurement before this date was taken on a nondeterministic engine.** Any
  historical A/B under ~1 % on MiniMax, and any single-run quality/perplexity figure, carries this
  as unmodelled variance.
- **F3 Phase 1 ships; F3 Phase 2 is refuted at the measured tail sizes.** Shared-prefix broadcast in
  `batched_generate` cuts TTFT on a 3-stream fan-out over a ~1104-token shared prefix, with a control
  arm (pool OFF) that moves 1.00–1.01× — so the win is prefix reuse, not page cache or allocator
  warmth. **(Its "token-identical on Llama-3.3-70B-exl3 and Qwen3-235B" claim was later shown to be
  prompt luck for Qwen — see the 2026-08-15 re-scope entry above.)** **The speedup spans 2.8×
  across models and must never be quoted bare:** Llama-3.3-70B-exl3
  dense **4.96×** (n=3), MiniMax-M2.7 module-list **2.15×** (n=2), Qwen3-235B fused-EXL3 **1.76×**
  (n=1 usable). Coverage is 0.981–0.983 on all three because it is a property of the prompts, not the
  model; the divergent tails are 1.7 % of prefill, so Phase 2 — which needs a cache-class change,
  `_seq_len` being per-LAYER — would be paying for 1.7 %. Break-even is roughly a **500-token mean
  tail**. Re-measure before reopening; do not re-scope Phase 2 to find a win.
- **The FIRST run of a checkpoint in a session is not a valid baseline, even same-box same-day
  same-process.** Qwen's repeat 0 cold arm took **165.7 s** against 29.1–29.2 s in the other three
  runs — a 5.7× first-touch NVMe outlier that inflated its ratio to 10.20×. The warm arm was stable
  across both repeats, so the BASELINE was contaminated, not the treatment; averaging the repeats
  would have baked a disk-warming artifact into a headline. Discard or repeat any checkpoint's first
  touch.
- **A completion count is not an equivalence check.** F3's "4/4 succeeded" on MiniMax meant four runs
  did not crash; the harness timed TTFT and never compared tokens. The token-equivalence gate, run
  later, FAILED on that same model. Whenever a perf harness and a correctness gate are separate,
  a green perf run says nothing about correctness.
- **No expert pin is viable.** Eviction share is linear in N and always 2–4× below the pin's share of
  the cache, on both models. (Plan F, whole curve measured)
- **The expert VRAM cache is NOT emptied every prefill, and preserving it would not help.** 0 wipes
  across 4 turns; a **cold** cache reaches 37.2 % hit rate vs a fully warm one's 36.8 % — it
  re-saturates in ~4 tokens. **This kills the whole preserve / pre-warm / pin family.** (audit F8)
- **`VAAYU_CACHE_BIAS` is unreachable.** `cache_aware_topk` / `_build_resident_mask` exist in **1 of
  3** MoE forwards (fused-EXL3) and are absent from module-list and non-EXL3-fused entirely. Any
  re-targeting spec must cover all three forwards or say which it excludes.
- **TurboQuant KV at 4 bits fails its quality gate** by 2–7× on both models; 8 bits fails on MiniMax;
  **9 bits passes both** and is the shipped `_DEFAULT_BITS`. Uniform-bit TurboQuant on the *live* KV
  cache is the wrong tier — moving the identical code to the eviction boundary took the per-token
  cost to zero (Plan I).
- **Exceeding prefix-pool capacity costs the ENTIRE win, not a share.** At K=7 against a pool holding
  5-6 the result was **0 hits, not 5** — round 1's re-stores cascade-evict round 0's entries.
  **Capacity is a cliff, not a slope; provisioning the pool to exactly the expected concurrency is
  provisioning it to fail.** (Plan J Task 4)
- **The prefix pool budget is NOT a constant of this box.** `free RAM at load − 3.76 GB reserve`, and
  free RAM at load varies: 11.49 / 12.16 / 12.87 / 13.70 / 13.76 GB across five MiniMax loads, and
  Qwen measured **4.96 GB, then 2.36 GB** on different days. Quote `K_max = floor(pool / per-session
  KV)` and the pool it was measured at — never a bare integer.
- **The expert RAM cache hit rate is NOT uniformly low, and "5-7 %" is a MiniMax number.** Same box,
  same day: **MiniMax 6 %, Qwen 44-47 %**. Generalising MiniMax's figure across geometries is what
  let Plan J ship an unconditional clamp that cost Qwen 49 % of decode. **Never generalise a
  cache-effectiveness number across the two models.**
- **PyTorch's caching host allocator charges the NEXT POWER OF TWO for every pinned allocation, and
  charges an exact power of two exactly.** Measured torch 2.11.0+cu130: 128 MiB → 128, **129 → 256**,
  1024 → 1024. Freed pinned blocks are **cached, not returned to the OS** — `del` frees nothing at OS
  level and `torch.cuda.empty_cache()` is device memory. `torch._C._host_emptyCache()` DOES return
  them (measured −8.59 GB where `del` gave +0.00). (Plan J §0, Plan L E4)
- **Chunking a pinned arena costs no bandwidth.** 101–103 % of the single-tensor baseline at every k
  from 2 to 36, both at 256 MiB burst and real slab granularity; allocation time flat in k. **k is
  free.** (Plan L E4)
- **MiniMax runs the module-list forward, not the EXL3 fused one.** Every MiniMax perf number on
  record is a module-list number.
- **Layer offload to meta destroys BUFFERS, and a buffer the checkpoint cannot restore is gone for
  good (fixed 2026-08-09).** `free_from_gpu` in BOTH loader strategies ended with
  `layer_module.to("meta")`. Parameters SHOULD go to meta — that is layer streaming, and the next
  per-layer load restores them. Buffers are the trap: `MiniMaxM2SparseMoeBlock` registers
  `e_score_correction_bias` via `register_buffer(torch.zeros(...))`, the checkpoint has **no tensor
  by that name** (layer 3: 3093 keys, only `block_sparse_moe.gate.weight`), and the only thing that
  ever materialized it was the load-time `named_buffers()` pass. Each layer's router therefore
  worked on the FIRST forward pass and died on every one after. **Latent for as long as the
  module-list forward used a hardcoded softmax; it surfaced the moment that forward switched to the
  model's own `route_tokens_to_experts` (sigmoid + bias).** A "works once then fails" symptom on a
  streamed layer means look at what the offload destroys that the reload cannot rebuild.
- **Decode floors, same-box same-day 2026-08-08: MiniMax 1.45 tok/s, Qwen 0.83 tok/s.** These are
  BASELINE numbers from `main`, and they sit *below* the older 1.48 / 0.85 figures on unchanged code.
  Always re-measure the baseline the same day; never gate against a historical number.
- **`_full_position_embeddings = None` was an OVERLOADED sentinel; failure now raises (D3, fixed
  2026-08-09).** `None` legitimately means "this architecture exposes no top-level `rotary_emb`, so
  each decoder layer computes RoPE itself" — `get_pos_emb_args`' empty-dict path exists for exactly
  those models. The old handler reused that same `None` to mean "precompute failed", printing
  "RoPE will be computed per-layer" for a mechanism that has never existed. Since every modern
  decoder layer takes `position_embeddings` as a REQUIRED argument, a VRAM OOM surfaced as
  `...DecoderLayer.forward() missing 1 required positional argument` ~30 frames away. Now
  `_precompute_rope` returns `None` only for the no-`rotary_emb` case and raises otherwise, with
  the OOM chained via `from` and the real levers named (`prefill_chunk_size`, `max_seq_len`,
  expert VRAM budget). **A diagnostic must not be able to raise:** the first draft of that error
  interpolated `position_ids.shape[1]`, which would have re-created D3 inside D3's own fix — there
  is now a test pinning it.
- **D1 was a tied `lm_head`, never an attention/RoPE/mask bug (fixed 2026-08-09).** A checkpoint with
  `tie_word_embeddings: true` stores no `lm_head.weight`; the splitter still writes a valid EMPTY
  16-byte `{}` shard plus its `.done` marker, so the split verifies complete and the load SUCCEEDS
  with zero tensors. The pin phase measured `size = 0`, injected nothing, and still added `lm_head`
  to `pinned_layer_names` — **resident while meta-initialized** — and the forward skips pinned
  layers, so nothing ever weighted it. Measured, all same box same day:
  | arm | ppl | note |
  |---|---|---|
  | arm | forward | ppl | note |
  |---|---|---|---|
  | Llama-3.2-1B (tied), before | dense | 128365.41 | vocab-uniform baseline 128256 — zero signal |
  | Llama-3.2-1B (tied), after | dense | **11.94** | |
  | Llama-3.3-70B-exl3 (untied) | dense | 8.78 → 8.78 | unaffected both sides of the fix |
  | MiniMax-M2.7 | MoE module-list | 8.73 / 8.66 | shared pin phase, no regression |
  | Qwen3-235B (94 layers) | MoE fused-EXL3 | **6.71** | arena + prefill-stager path |
  **All four rows of `CLAUDE.md` §0's dispatch table that have an on-device model were measured.**
  Every untied checkpoint pinned a real `lm_head` (70B 0.79 GB, MiniMax 0.46, Qwen 0.47); the tied
  branch stays silent for them.
  **A perplexity at the vocab-uniform baseline means an unweighted output layer — go to weight
  loading, not to attention.** Three attn impls failing identically was the same tell, misread.

---

# METHOD NOTES — these cost real time to learn

- **A measurement that only lands in a doc is not a deliverable.** Decide what the engine will DO
  with a number before spending GPU hours on it.
- **Derive the sweep instead of walking it.** Computing per-session KV from `config.json` takes 30
  seconds and can predict a cliff before any run launches.
- **n=1 decode comparisons cannot resolve <~4 % on this box.** Repeat, alternating arms.
- **Run logs are 300 MB – 1.3 GB.** Always `grep -a`, never `cat`, and never grep one while another
  run is streaming experts off the same NVMe.
- **NEVER run multiple pinned-allocation arms in one process, and cap every probe's cgroup.** Freed
  pinned blocks are cached, not returned, so arms accumulate: one probe hit a **28.3 GB peak** and
  `systemd-oomd` killed the whole terminal cgroup — it kills by *scope*, so `oom_score_adj` is
  useless. Pattern that works: **one arm per subprocess**, each under
  `systemd-run --user --scope -p MemoryMax=<charge+2GB> -p MemorySwapMax=0`, plus a driver ceiling of
  `min(40 % total RAM, available − 4 GB)`.
- **A test whose fixture premise is not asserted can silently stop testing anything.** Plan L's
  `slab_index_of` mutant passed **84/84** because every fixture collapsed to a single chunk,
  non-deterministically (2 chunks 39/40 runs standalone, 1 under pytest). Assert the premise
  (`assert len(arena._chunk_first) > 1, "..."`), the way `assert tails > 0` already does.
- **Extract load-time decisions into pure functions so they can be tested without a model load.**
  `arena_backing_is_sufficient`, `degraded_arena_decision`, `slab_arena_exhausted_warning` exist for
  this reason. Two Plan L defects lived in code "verified" by a hand-run harness that **copied** the
  logic instead of executing it.
- **Silent degradation hides on paths that only run once something has already gone wrong.** All
  three Plan L review defects were of this shape and none was visible on a healthy box.
- **When a fix changes nothing, check whether your code EXECUTED before forming a second
  hypothesis.** D1's first fix was correct code at the wrong site: `load_layer_to_cpu` already had a
  tied-weights fallback, but its comment read "lm_head.safetensors might not be generated" — it
  anticipated a MISSING file, and an empty-but-valid one does not raise. Patching it changed
  nothing because **pinned layers never reach that function at all**. The tells were that the
  re-run was byte-identical AND *neither new print appeared*; `[Vaayu-Pin] pinned lm_head: 0.00 GB`
  was the log line that named the real path. A second hypothesis about *why the fix was wrong*
  would have been wasted — the fix was never run.
- **An existing fallback is not proof the case is handled.** Check what TRIGGERS it. D1's fallback
  keyed on an exception; the failure mode was a success returning `{}`.
- **`size == 0` is not the emptiness test for a layer — `not tensors` is.** `model.norm`
  legitimately measures 0.00 GB, so a byte-size check silently lumps a real layer in with an
  unloaded one.


---

## Context-limit verification (todo row 1, 2026-08-15) — RAN, FAILED


Ran, on-device, one arm per subprocess under `systemd-run --user --scope -p MemoryMax=22G/28G
-p MemorySwapMax=0`:

| arm | expected | measured |
|---|---|---|
| Llama-3.3-70B, total == 8192 | completes | **PASS** — prefill peak 9.39 GB, no OOM |
| Llama-3.3-70B, total == 8193 | `VramPlanError` | **FAIL** — no exception, request ran to completion |
| Llama-3.2-1B, total == 106987 | completes | **FAIL** — CUDA OOM at ~70,656 tokens (34 % short) |

So the dangerous point (8192, the KV doubling cliff) is safe, and the row's real finding is the
one below: the cap is not a cap, and the 1B's limit is 1.5× past what fits. Both trace to a single
measurement defect, now row 2.

---


The defect this exposed is now the scheduled row in `todo.md` (`predict_peak_vram` calibrated against a last-chunk peak).

---

## VRAM peak measurement (todo row 1) — CLOSED 2026-08-15

**Original writeup (moved verbatim from `todo.md`):**

`vaayu_base.py:3334` called `torch.cuda.reset_peak_memory_stats()` after EVERY prefill chunk
(inside the chunk loop, right after the `[Vaayu] Chunk [...] peak VRAM` print). Any
`max_memory_allocated()` read taken after a chunked-prefill forward therefore reported the peak of
the LAST CHUNK ONLY, not of the prefill.

`scripts/probe_vram_curve.py:52` measured exactly that way, so every point in
`scripts/fit_vram_calibration.py`'s `MEASURED` table was a last-chunk peak. That is why the form
`KV(capacity) + last-chunk logits` fit with zero fitted constants and why the ±10 % gate passed
0/26: both sides described the same under-measured quantity. Verified — llama-3.2-1b at S=32768,
`MEASURED` said 2.409 GB, an independent post-`generate` read said 2.426 GB, and the in-loop chunk
log for the same run said 8.13 GB.

Measured true slope of prefill peak (least-squares over the in-loop chunk series):

| model | measured | `predict_peak_vram`'s KV term | ratio |
|---|---|---|---|
| Llama-3.2-1B | 0.2027 MB/tok (n=120, to S=32768) and 0.2025 (n=275, to S=70656) | 0.0328 MB/tok | 6.2× |
| Llama-3.3-70B | 0.7149 MB/tok (n=32, to S=8191) | 0.328 MB/tok | 2.2× |

The excess is score/mask-shaped — several copies of `n_heads · chunk · elem_bytes` per token
(5.2× that quantity on the 1B, 5.9× on the 70B) — i.e. `MATERIALIZES_SCORES = False` and the
2026-08-15 removal of the S-linear activation term are both consequences of the blind spot, not
findings about the hardware. Do not re-fit anything until the measurement is fixed.

Design spec: `plans/2026-08-15-vram-peak-measurement-design.md` (approved for Phase 1: drop the
in-loop reset, re-probe all four geometries, change no prediction). Two decisions taken with it:
the fix is deleting the reset (no engine-side peak state), and `max_seq_len` is a derived DEFAULT,
not a hard cap — so 8193 running to completion is CORRECT and this row's original "must raise
`VramPlanError`" expectation is retired as wrong, not filed as a bug. **The ±10 % gate is EXPECTED
to go red after Phase 1; do not tune the form to re-green it.**

Settled 2026-08-15 alongside this: `_assert_context_fits` admitting 8193 is not a defect —
`max_seq_len` is a derived default, and admission belongs to the headroom predicate. The
predicate's PREDICTION was the only thing wrong.

*[Editorial note, added when this was archived — not part of the original quote above: the "true"
slope table just above was later found, during the RESULTS.md re-probe, to have been read from an
sdpa-forced session, not the flash_attention_2 this task's arms and the old `MEASURED` table both
ran under. That is the CLOSING VERDICT below's second bullet, not a correction to the quote.]*

**CLOSING VERDICT (2026-08-15, Phase 1 re-probe complete):** All four geometries re-probed with
the reset removed, one arm per subprocess, `status=ok` throughout (no OOM anywhere, including
Llama-3.2-1B out to its full derived `max_seq_len=106987`). Full data, per-point signed error
against the unmodified `predict_peak_vram`, and before/after comparison against the old `MEASURED`
table: **`plans/2026-08-15-vram-peak-RESULTS.md`**.

Headline findings:
- The two dense geometries moved, as expected (the legacy per-chunk loop held the reset); the two
  MoE geometries did not move, as expected (the layer-major branch returns before that loop) — a
  ~31% apparent drop in the closest Qwen3-235B/MiniMax-M2.7 overlap points was checked directly and
  is fully explained by a known `kv_capacity_tokens` doubling boundary sitting between the old
  table's S=8200 and this task's S=8192, not a behavioural change.
- **Unexpected finding, found only after a coordinator cross-check:** every arm in both the old
  `MEASURED` table and this task's re-probe ran under `flash_attention_2`, not `sdpa` —
  `scripts/probe_vram_curve.py` passes no `attn_implementation`, so Vaayu's own candidate order
  picks flash first. But `batched_generate` (the real batched-decode path) hard-requires `sdpa`,
  and `MATERIALIZES_SCORES=False` was measured for sdpa specifically. Direct comparison,
  Llama-3.2-1B S=8192: flash_attention_2 1.0974 GB vs sdpa (forced) 2.0967 GB, 1.91×. This also
  explains the pre-fix "6.2×/2.2×" slopes above: those were read from an sdpa-forced session, not
  a form error at all. **This reopens as `todo.md` item 10** — Phase 2 cannot re-point `MEASURED`
  or re-decide the form meaningfully until it is resolved.
- The ±10 % gate (`scripts/fit_vram_calibration.py`) was run and its output captured as the
  "before" state in the RESULTS file. It still PASSES (0/26 outside, self-consistent against the
  untouched `MEASURED` table) — deliberately not fixed by this task. `MEASURED`, `predict_peak_vram`,
  `MATERIALIZES_SCORES`, and `DEFAULT_ALLOC_FACTOR` are byte-for-byte unchanged.
- Three on-device gates re-run and PASS: dense smoke (perplexity 8.78, unchanged),
  `test_moe_accumulator_determinism.py`, `test_batch_prefix_equivalence.py` (both MiniMax-M2.7).

Phase 2 (form, `MATERIALIZES_SCORES`, budget policy, `max_seq_len` re-derivation) is now `todo.md`
items 6-9, gated behind item 10 (the attention-kernel question).

## Attention-kernel calibration question (todo row 10) — CLOSED 2026-08-15

**Question:** which kernel should the VRAM calibration describe — `flash_attention_2` (what the
probe picks by default) or `sdpa` (what `batched_generate` hard-requires)? And does the
flash-vs-sdpa gap grow with S enough to explain the pre-fix "6.2×/2.2×" slopes above, or is
something else going on?

**Method:** `scripts/probe_vram_curve.py` gained an opt-in `PROBE_ATTN` env var (default behaviour
unchanged), then paired flash/sdpa sweeps were run on all four on-device geometries — 1B out to
S=98304, the other three to S=8192 — one arm per subprocess, same caps as the row-1 re-probe.

**Verdict — sdpa, kernel-aware. Full data and derivation: `plans/2026-08-15-row10-attention-kernel-RESULTS.md`.**

- The flash-vs-sdpa ratio **grows with S on all four geometries** (1B: 1.00×→4.66×; 70B:
  1.00×→1.82×; Qwen3-235B: 1.00×→1.97×; MiniMax: 1.00×→1.64×), never flat, never shrinks.
- Reaches **3.774× at S=32768 on the 1B**, matching the ~3.7× the "gap explains 6.2×" hypothesis
  needed — confirmed as the asymptotic ratio of two fitted lines (sdpa slope 0.2028 MB/tok / flash
  slope 0.0328 MB/tok ≈ 6.2×).
- sdpa's own slopes **reproduce the pre-fix hand-measured figures above almost exactly** (1B: 0.2028
  vs 0.2027 MB/tok, three independent segments; 70B: 0.6984 vs 0.7149 MB/tok) — those old readings
  were sdpa readings all along, not a form error.
- `MATERIALIZES_SCORES = False` **is contradicted by the sdpa data specifically**: the
  flash-to-sdpa slope gap implies a per-element byte cost clustered at 10.4–11.4 bytes across all
  four architecturally-unrelated geometries — consistent with a real `n_heads · chunk · S`-shaped
  score buffer sdpa's masked call materializes and flash's kernel does not. The constant is right
  for flash, wrong for sdpa.
- **Recommendation:** calibrate `MEASURED` against sdpa (what `batched_generate` needs), and make
  `MATERIALIZES_SCORES` kernel-aware (gate on `model._active_attn_impl`) rather than a single global
  value — flash and sdpa need different constants, not one constant picked for both. Single-stream
  `generate()` (which may run flash) stays out of scope, same reason as before: nothing today gates
  that path on `predict_peak_vram`.

Rows 6-9 are unblocked and re-scoped onto this verdict (`todo.md`).

---

## Budget policy: reserve vs expert cache (todo row 8) — CLOSED 2026-08-16

**Question:** rows 6/7 made `predict_peak_vram` kernel-aware, which CUT the derived context on every
sdpa load (Llama-3.3-70B 4373→3821, Llama-3.2-1B 106987→14052, both MoE geometries unchanged at 8192
on flash). Row 8's decision: grow the reserve at the expert VRAM cache's expense, or accept the
shorter contexts?

**Decision — grow it, but only on kernels that materialize scores, and by a DERIVED amount.**
`reserve_floor_bytes` (`vaayu/vram_plan.py`), called from `vaayu_base.__init__` before `plan_vram`:

- Gated on `shape.score_elem_bytes > 0`, not on a kernel name — so `eager` and every unmeasured
  kernel land on the raised side automatically, matching row 7's conservative coefficient rule.
  `flash_attention_2` keeps the 3.5 GiB base floor exactly, and pays nothing.
- The raised value is *flash parity*: the floor that derives the same context flash would have
  derived at the base floor, `predict_peak_vram(shape, _largest_s_within_budget(flash_shape, base))`.
  A flat 6.25 GiB literal was rejected — it is the max of what these four checkpoints happen to need
  for 8192, i.e. a model number in disguise (CLAUDE.md §5), and would under-serve any other geometry.
- Clamped only by physics (`free − min_resident − layer_slack`), never below `base_floor`.

**Derived floors and what they cost** (checkpoint configs, chunk 512, N=1, KV auto; free 14.2 GiB):

| geometry | flash floor / S | sdpa floor / S | reserve delta | pinning 10.10 GiB → |
|---|---|---|---|---|
| Qwen3-235B | 3.50 / 8192 | 5.60 / 8192 | +2.10 GiB | 8.00 GiB |
| MiniMax-M2.7 | 3.50 / 8192 | 4.79 / 8192 | +1.29 GiB | 8.81 GiB |
| Llama-3.3-70B | 3.50 / 8192 | 6.20 / 8192 | +2.70 GiB | 7.40 GiB |
| Llama-3.2-1B | 3.50 / 90639 | clamped 13.60 / 65363 | +10.10 GiB | 0.00 GiB |

The three EXL3 targets recover their full 8192 under sdpa. The MoE cost lands on the expert VRAM
cache, already priced cheap: Plan K gave 1.22 GB back for a 43%→41% hit rate and +0.4% decode, and
Audit F8 measured the cache re-saturating in ~4 tokens.

**Llama-3.2-1B hits the clamp, and the price was MEASURED before accepting it** — 1B/sdpa/greedy,
64 tokens, first touch discarded, same box same day:

| arm | reserve | pinning | max_seq_len | decode |
|---|---|---|---|---|
| base floor (control) | 3.50 GiB | 10.24–11.29 GiB | 14052 | 6.38–6.45 tok/s |
| **physics clamp (shipped)** | 13.74 GiB | **0.00 GiB** | 66074 | **5.19 tok/s (−19 %)** |
| bounded clamp (rejected) | 11.32 GiB | 3.50 GiB | 53755 | 6.50 tok/s (no regression) |

**Ruling: context over the 19 %.** The `pinning_budget=0` WARNING is EXPECTED on a 1B sdpa load, not
a bug. The bounded variant (`... − base_floor`) is written down here because it is the cheap way back
if the 1B ever needs its decode speed — e.g. as row 12's drafter, where −19 % on the draft model
directly attacks the point of speculative decoding.

**Method note — a framing error caught only by running it.** The clamp question was first put with
"nothing else competes for VRAM on a small dense model, so the cost is warning noise". That was
wrong: pinning competes, and the real load showed pinning going 11.29 GiB → 0. A budget claim about
a path with no unit test is not knowledge until a real load prints it (`todo.md`'s gates table says
the same thing about `__init__`).
