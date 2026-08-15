# TODO — Vaayu pending work

**Everything in this file is PENDING.** Nothing finished lives here. Closed rows, settled
verdicts and the method notes moved to **`plans/settled_and_method_notes.md`** (tracked in git)
on 2026-08-15; older history is in `git log -- todo.md` and `docs/superpowers/plans/*-RESULTS.md`
(`docs/` is **gitignored** — may not exist on a fresh clone; `plans/` is tracked).

Protocol: `CLAUDE.md` §6 — update this as work happens, never after.
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Every item below carries the detail needed to fix it: file, symbol, what is wrong, what
"done" means, and the stop rule.** The `plans/next_*.md` briefs hold the long-form background
for the three scheduled rows and are self-contained on their own.

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q` → baseline **795 passed, 2 skipped,
  1 xfailed**.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  ~5.5 GB/s NVMe), each commented as such.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
- **A plan ending in refutation CLOSES its row.** Record the verdict in the archive and move on;
  do not re-scope a plan to find a win.

## Checkpoints on disk (`/home/a/code/ai-ml/models/`)

| model | dir | L / kv_heads / head_dim / hidden / n_heads / vocab | KV bf16 | forward | shipped `max_seq_len` |
|---|---|---|---|---|---|
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 4373 |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 56828 |

(KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
constants and decimal `MB_DEC`/`GB_DEC` for these — use the right family.)

---

# ▶ PENDING SEQUENCE — work top to bottom

| # | Item | Status | Cost | Why here |
|---|---|---|---|---|
| **1** | ~~Close the `predict_peak_vram` calibration gate~~ | **REFUTED 2026-08-15** | done, ~1 h GPU | The form is wrong, not the constants. Verdict below; the successor work is item 6. |
| **2** | F1 — RAM budget arbitration | `[!]` | ~1 h measure, then decide | Blocked on its own premise; expected outcome is "close as obsolete". |
| **3** | Hygiene backlog H1–H7 | `[ ]` | minutes–1 h each | Nothing blocks anything; good filler. |
| **4** | N≥2 admission gap (xfail pinned) | `[ ]` | — | **Downstream of #1.** Do not chase separately. |
| **5** | Fused-EXL3 expert GEMM nondeterminism | `[bug]` | likely third-party kernel | Open bug; cheap first probe defined below. |
| **6** | Fused-EXL3 prefill peak: ~1.39 G unmodelled residency | `[ ]` | needs a form change | **CONFIRMED 2026-08-15** and localised to one forward. Now the live VRAM item. |
| **7** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **8** | F3 Phase 1 loose end (MiniMax equivalence) | `[~]` | — | Phase 1 shipped; Phase 2 refuted. See #5. |
| **9** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; premise needed first. |

> ### ▶▶ NEXT ACTION: item 2 or 3 — item 1 is closed (refuted 2026-08-15)
> Calibration cannot close the gate; the successor is item 6, which needs a design brief and
> approval before any code. Items 2 and 3 are unblocked and independent.

---

# 1 · [~] Close the `predict_peak_vram` calibration gate

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

**Item 4 stays parked and is NOT closed by this.** The xfail
`test_batch_width_at_the_real_reserve_floor_xfails` will not flip, because the model still
under-predicts the fused-EXL3 forward by ~1.1 GB at N=1 — the admit-too-much direction, and the
error grows with N.

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

# 2 · [!] F1 — expert-cache / prefix-pool RAM arbitration

**Brief:** `plans/next_2_f1_ram_arbitration.md`. **Blocked on its own premise.** ~1 h to measure.
Independent of item 1 — either can go first.

### What is wrong

In `vaayu/vaayu_base.py`, two RAM budgets are computed independently and **nothing arbitrates**:
`compute_expert_budget(...)` runs first and takes its share; `compute_prefix_cache_budget(...)`
runs ~250 lines later and gets the remainder.

**F1's premise — "the prefix pool is starved because the expert cache ate the RAM first" — may no
longer hold.** Plan L took Qwen's measured pool **2.36 → 10.53 GB**. Measure before building.

### Constraints on the measurement

- **The pool budget is not a constant of this box:** `pool = free RAM at load − 3.76 GB reserve`,
  and free RAM at load varies — MiniMax across five loads **11.49 / 12.16 / 12.87 / 13.70 /
  13.76 GB**; Qwen **4.96**, then **2.36**, then **10.53** after Plan L. **Never quote a bare pool
  size or a bare K** — quote `K_max = floor(pool / per-session KV)` and the pool it was measured at.
- **Capacity is a cliff, not a slope.** At K=7 against a pool holding 5–6: **0 hits, not 5** —
  round 1's re-stores cascade-evict round 0's. Sizing the pool to exactly expected concurrency is
  sizing it to fail.
- **Realistic concurrency is sequential turns of ONE stream**, not wide fan-out: N≥2 at ~7700
  tokens is infeasible on this box for both MoE models (VRAM binds, not RAM).
- **The prize is still large if it binds:** cross-turn prefix reuse is Qwen 396.4 s → 2.4–2.9 s
  (136–165×), MiniMax ~380 s → ~1.0 s (377×).

### Steps

1. Instrument, don't guess: at load, print for both models — free RAM at load, expert RAM cache
   budget actually taken, prefix pool budget actually granted, derived `K_max`.
2. Repeat across several loads per model (free RAM at load is the variable that moves).
3. Apply the decision rule:

| finding | action |
|---|---|
| Pool comfortably exceeds realistic `K_max` need, across repeated loads | **Close F1 as obsolete** (expected outcome). Record measured pools in the archive. |
| Pool binds sometimes (varies with free RAM at load) | The defect is *variance*, not *allocation order* → consider a pool floor, not arbitration. |
| Pool binds consistently | Only now design arbitration — with headroom above `K_max`, per the cliff. |

**Write the numbers into this file with the pool they were measured at.** A measurement that only
lands in a doc is not a deliverable.

---

# 3 · [ ] Hygiene backlog — H1–H7, any order, nothing blocks

**Brief:** `plans/next_3_hygiene_backlog.md`.

- **[ ] H1 — `_create_kv_cache` silently falls through to fp8 on an unknown mode.**
  `vaayu/vaayu_base.py`. A typo in `kv_cache_mode` yields the fp8 cache instead of an error — the
  silent degradation CLAUDE.md §5 forbids. Precedent: `vram_plan.py`'s `kv_elem_bytes` **already
  raises** `VramPlanError` naming the offending value and the known modes, so the engine is
  inconsistent — planner refuses a typo, cache accepts it. **Done:** unknown mode raises with
  value + valid modes in the message, plus a test. (2026-08-02, Plan H audit)
- **[ ] H2 — `slab_bytes` is not provably `_ALIGN`-aligned, so `SlabArena`'s O_DIRECT premise is
  unenforced.** `compute_slab_bytes` → `expert_span_bytes` → `_group_layout` (`slab_span =
  max(raw_aligned_length, packed_span)`). `SlabArena`'s docstring asserts slab_bytes is
  `_ALIGN`-aligned so every slab start stays sector-aligned for O_DIRECT, and **nothing validates
  it** — `packed_span` only guarantees `_TENSOR_ALIGN` (16 B), not `_ALIGN` (4096). Narrow edge
  case, never observed on either target; the unchecked *assertion* is the defect. **Done:**
  constructor validation or `_align_up` on the computed slab size, plus a test. (2026-08-08, Plan L)
- **[ ] H3 — F6: `vaayu/kv_cache_manager.py` is unreachable but still maintained.**
  `KVCacheManager` is constructed **only** in `tests/test_kv_cache_manager_fp8.py`; production
  superseded it with `PrefixCacheStore`. Its `_extract_kv_as_bf16` is reachable only from
  `save_snapshot` and was hardened by **two separate plan tasks** (Plan E Task 2, Plan H Task 3) —
  a tax every future KV-format change pays again. **Done:** delete the module + its tests, OR mark
  it a reference implementation and exempt it from cache-mode work. Pick one, write it down.
- **[ ] H4 — F7: `vaayu_ref._expert_history` is write-only and its docstring lies.** Written by all
  three MoE forwards in `vaayu/moe_patch.py`, read by no production code, yet
  `patch_moe_blocks`' docstring claims *"The forward() loop reads `_expert_history` after each
  layer and submits speculative prefetch"*. Runtime cost ≈ 0; **the cost is the false docstring.**
  **Done:** remove the write and the claim (cheap, honest), or implement the prefetch.
- **[ ] H5 — Plan L review backlog (five small items, triaged non-blocking 2026-08-08).**
  1. `SlabArena` exposes no `chunk_count` / `charged_bytes` accessor, so `vaayu_base.py` reaches
     into `_chunks` and computes `sum(c.numel() ...)` inline — one property fixes both call sites.
  2. `tests/test_slab_arena.py` still claims "`arena_slab_count` IS that gate in production" —
     stale since Plan L removed it from the arena path.
  3. No test exercises an allocation failure on an **escalated retry**, only on a chunk's first
     request.
  4. The zero-chunks error message recomputes `chunk_decomposition()` purely to get a count.
  5. The **pre-existing** first-construction `except RuntimeError` can still leave the arena `None`
     with no message — Plan L fixed only the instance it introduced.
  **When touching these:** a test whose fixture premise is not asserted can silently stop testing
  anything — Plan L's `slab_index_of` mutant passed **84/84** because every fixture collapsed to a
  single chunk, non-deterministically. Assert the premise: `assert len(arena._chunk_first) > 1`.
- **[ ] H6 — telemetry (correctness unaffected).**
  - **F13 — `_decode_stats_reset` latches for the process, not the turn.** Set `True` on the first
    decode pass in `vaayu/vaayu_base.py` and never cleared, so the reset that exists to exclude
    prefill's ~100 %-miss pollution runs **once per model lifetime**; from turn 2 on, cumulative
    expert-cache hit rate is diluted by every new prefill. (2026-08-03)
  - **`matched_tokens` over-reports by ≤1 token per hit.** `PrefixCacheStore.lookup()` records the
    raw match length; the `generate` override clamps it down by one so the model has a token to
    run. Fires only when the new prompt is a complete prefix of the cached one — ≤0.05 % at ~2000
    -token prefixes. **Documented in `stats()`, deliberately NOT fixed** (moving the clamp into the
    store would couple the cache to the caller's execution needs). Listed so nobody rediscovers it.
  - **Prefix `hit_rate` is not comparable across lookup modes.** On a named session an entry
    sharing ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the same
    case counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing, documented in
    `stats()`, pinned by tests. Revisit only if a mixed-mode workload needs one comparable number.
- **[ ] H7 — small correctness items carried from earlier work.**
  - **D4 — the splitter writes an EMPTY `lm_head.safetensors` for tied-weight checkpoints and marks
    it `.done`.** `vaayu/utils.py`: with no `lm_head.weight` in the source (`tie_word_embeddings:
    true`), the split emits a valid 16-byte `{}` shard and `all(found_layers.values())` reports the
    split **complete** — an incomplete split that looks verified. Two candidate fixes: write the
    tied copy at split time (duplicated embedding on disk), or omit the shard so the existing
    absent-file paths fire. **Not urgent:** consumer-side fallbacks make already-split models load
    correctly, and both MoE targets plus the 70B dense reference are untied. (2026-08-09)
  - **`speculative_generate` skips `enqueue_truncate` on a prefix HIT**, so `enable_kv_persist` can
    corrupt that session's on-disk KV. `generate()` calls
    `self._kv_writer.enqueue_truncate(session_id, match_len)` on a HIT; the `speculative_generate`
    hook does not. `DiskWriter.enqueue_append` packs `range(st["nrec"], length)` from its OWN
    tracked `nrec`, not `match_len`. Scenario: turn 1 persists [0,7) (`nrec=7`); turn 2 HITs at
    `match_len=5`, ends at length 9; append starts at 7 and writes only [7,9), leaving **[5,7)
    holding turn 1's stale bytes** — corrupted, undetectable until restore. **Dormant today** (no
    on-device target combines a speculator with `enable_kv_persist=True`). Fix is the two-line
    mirror of `generate()`. (2026-08-09)
  - **The cross-turn prefix cache only works for EXL3 models**; DISABLED with a warning otherwise.
    `PrefixCacheStore` calls `to_cpu()` / `to_device()` / `resident_bytes()` on every cache it
    stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
    `transformers.cache_utils.DynamicCache` returned for `kv_cache_mode="fp16"` — what `"auto"`
    resolves to for every **non-EXL3** model — has none (verified transformers 5.5.4: layer-object
    based, `layers[i].keys`/`.values`, no `key_cache` attribute). `vaayu_base.py` now probes at load
    and disables the pool with a WARNING instead of failing per-turn — the per-turn failure was the
    dangerous one, since `lookup()` POPS before `store()` puts back, so each swallowed failure
    **permanently drained the pool**. To actually support fp16: adapter giving `DynamicCache` the
    three methods, or route fp16 through `FP8DynamicCache(quantize=False)`. Not urgent — all
    on-device targets are EXL3. (2026-08-09)

---

# 4 · [ ] `_assert_context_fits` can still ADMIT a batch that then OOMs

**DOWNSTREAM OF ITEM 1 — do not chase separately.** Fixing calibration is what closes it.

`predict_peak_vram`'s −28 % worst-case error was only ever validated at N=1. At MiniMax-M2.7's
REAL floor-derived operating point (`max_seq_len=8192`, `reserve_floor=_VRAM_RESERVE_FLOOR_BYTES`)
the model predicts **N=3 FITS** (9.55 GiB < 15.47 GiB usable) — directly contradicting Plan K's
real GPU measurement that N≥2 at ~7700 tokens is infeasible on this box for both MoE models. The
refusal predicate correctly catches the extreme, obviously-oversized case (F4 Task 10 Gate 3); it
does NOT yet catch a moderately-oversized batch width at the real operating context.

Pinned by a committed, non-strict `xfail`:
`tests/test_vram_plan.py::test_batch_width_at_the_real_reserve_floor_xfails` — it **XPASSes the
moment calibration is fixed**, so this is a visible signal, not a silent gap.

Also: `test_batch_width_is_rechecked_against_the_plan` passes only because its own `_plan()` helper
omits `reserve_floor`, leaving `s_max` unclamped at 32768 — an operating point production never
reaches; that test's premise does not exercise the real admission path.

**If item 1's re-fit fails and gets parked, this gap is parked with it** — known and bounded, not
silent. (2026-08-14, F4 Task 8 review + Task 10 gate 3)

---

# 5 · [bug] The fused-EXL3 expert GEMM is still nondeterministic after the fp32 accumulator fix

Qwen3-235B, two forwards over identical input ids: layers 0–8 bitwise identical (pinned, so they
bypass `sparse_forward`); **layer 9 — the first STREAMED MoE layer — diverges by 3.05e-05**, one
fp16 ULP at that magnitude, growing to 4.5 by layer 94 and max|Δlogit| ≈ 0.25–0.28. Greedy argmax
has not flipped in any run so far, so it is not currently changing tokens — but it is the same
amplify-through-depth mechanism that DID flip MiniMax's.

**Ruled out:** the accumulator (fp32 now) and the caches (measured with the expert VRAM cache
disabled, and again with both VRAM and RAM caches disabled — nondeterministic in every arm).
Since layer 9's inputs are bitwise identical, the differing quantity is the expert GEMM's own
output.

**Suspect:** split-K / atomic accumulation inside the fused kernel path, not Vaayu's Python. The
fused-EXL3 path casts `current_state` to **fp16** and calls the EXL3 kernel through
`_make_bc_linear_exl3`; the module-list path calls `ExLlamaV3Linear` and is now bitwise
deterministic.

**Cheap first step:** call the kernel twice on identical inputs in isolation to confirm it is the
source. **Cost of a real fix is likely a third-party kernel change.** (found 2026-08-09)

---

# 6 · [ ] Probe/instrument mismatch on the MoE prefill-peak probe

**CONTINGENT on item 1 failing — do not start it first.**

The trailing-chunk intercept fix matches MiniMax's measured peak-vs-S closely but NOT Qwen's,
whose measured peak is nearly flat within a KV-capacity bucket instead of showing the predicted
sawtooth. Likely cause: the dense legacy per-chunk prefill loop calls
`torch.cuda.reset_peak_memory_stats()` after every chunk (so `max_memory_allocated()` genuinely
reflects only the last chunk), but MoE's layer-major prefill has **no equivalent per-chunk reset**
— so for MoE the probe may be measuring expert-streaming residency across the WHOLE prefill, not
last-chunk structure.

**If true, adding more terms to `predict_peak_vram` is the wrong move; re-instrumenting the probe
per-phase is.** Item 1's baseline subtraction is cheaper and may close the gate without this.
(2026-08-14, F4 Task 5c)

---

# 7 · [ ] 2× KV overshoot, ~0.83 GB at Qwen's derived operating context — PARKED

Real, quantified, recoverable, and **deliberately not scheduled.** The EXL3 ring-buffer KV cache's
geometric-doubling capacity means that at Qwen's derived `max_seq_len=8192` the cache allocates
capacity for **16384** tokens, wasting ~0.83 GB on a box already grazing 15.47 GiB. Pre-sizing the
cache to the derived `max_seq_len` at load would recover it — allocator surgery, needs its own
decode A/B.

**Why parked:** this repo has already priced that VRAM. Audit F8 measured the expert VRAM cache
re-saturating in ~4 tokens (cold 37.2 % vs warm 36.8 % hit rate), and Plan K measured giving back a
LARGER 1.22 GB moving Qwen's hit rate 43 %→41 % with decode +0.4 % — i.e. unlocked nothing. 0.83 GB
is less than that, and well short of what N≥2 needs (peak was 14.17–15.57 GB of 15.47 GiB).

**Re-open only if a measurement shows per-cell VRAM has dropped enough for N=2 to be within
reach** — then this 0.83 GB becomes part of a real budget rather than a rounding error.
(2026-08-14, F4 Task 5 ruling R12)

---

# 8 · [~] F3 — cross-turn prefix cache in the batched paths

Phase 1 (shared-prefix broadcast) landed 2026-08-09 on `feat/f3-batched-prefix-cache`:
`batched_generate` computes the batch's longest common prefix, looks it up ONCE, broadcasts that
cache to N rows, prefills only the divergent tails, and stores exactly one entry back (from the
longest-prompt row, inside the `finally` so an abandoned generator cannot lose the popped entry).
`speculative_generate` is wired in too. Token-identical on Llama-3.3-70B-exl3 and Qwen3-235B;
**NOT on MiniMax — that is item 5.** Task 6's measurement is DONE (coverage + TTFT A/B, all three
forwards).

**Phase 2 (ragged per-stream merge) is REFUTED** at the measured tail sizes and is NOT built.
Reopening needs a workload with ~500-token mean tails AND a cache-class change, since `_seq_len`
is per-LAYER and cannot represent per-row sequence lengths.

Plan / results (gitignored, may be absent):
`docs/superpowers/plans/2026-08-09-batched-prefix-cache-plan.md`, `…-RESULTS.md`.

**Open work on this row = item 5 only.**

---

# 9 · [ ] DEFERRED — written up, no scheduled slot

- **Plan B Tasks 8, 9 — Phase 2 live expert re-pin.** The Phase 1 pin it builds on is refuted (no
  viable pin N: eviction share is linear in N and always 2–4× below the pin's share of the cache,
  both models), so this needs a **new premise** before it means anything.
  `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- **Plan H follow-up — real-query quality gate.** Plan I's gate used K-row and Gaussian query
  proxies because `.vaayu_kv` stores no queries. Both missed the bar widely so the verdict holds,
  but certifying a specific bit-width needs real queries captured with an attention hook.
- **Plan H follow-up — fold `Π` into the query** (`⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`): turns per-token
  reconstruction from `O(S·d²)` into `O(d²)`, exactly what costs MiniMax its 5–8 % decode. Needs
  hooking attention itself. **Only worth writing after the quality question is settled** — a fast
  cache that is 23 % wrong is not worth speeding up.
- **Mixed-precision outlier split (TurboQuant).** The *only* way the paper reaches fractional
  bit-widths, so the uniform implementation cannot claim its 3.5-bit LongBench result. Outliers are
  early layers on both models — exactly what the split addresses.
