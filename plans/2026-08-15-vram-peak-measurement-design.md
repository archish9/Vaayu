# Design — restore the VRAM peak measurement, then re-measure

Date: 2026-08-15 · Status: approved for Phase 1 · Supersedes nothing; corrects the
premise of the 2026-08-15 `predict_peak_vram` form change.

`docs/` is gitignored on this box, so this spec lives in `plans/` (tracked) with the
other briefs.

## 1. Problem

`vaayu/vaayu_base.py:3334` calls `torch.cuda.reset_peak_memory_stats()` **inside the
chunked-prefill loop**, once per chunk, immediately after printing
`[Vaayu] Chunk [a:b] done — peak VRAM: N GB`.

`torch.cuda.max_memory_allocated()` is one process-global high-water mark. Resetting it
mid-forward means any caller that resets, runs a prefill, and then reads the counter —
the standard measurement — observes **only the final chunk**, not the prefill.

`scripts/probe_vram_curve.py:52,68` measures exactly that way. Therefore **every point in
`scripts/fit_vram_calibration.py`'s `MEASURED` table is a last-chunk peak**, and the
"±10 % gate, 0/26 outside, worst 7.3 %, zero fitted constants" result compares a
last-chunk model against a last-chunk measurement. It is self-consistent and wrong.

### Evidence (measured 2026-08-15, this box)

Llama-3.2-1B at S=32768, one run, three readings of the same quantity:

| reading | value |
|---|---|
| `MEASURED["llama-3.2-1b"]` (last-chunk peak) | 2.409 GB |
| independent post-`generate` read (also last-chunk) | 2.426 GB |
| in-loop chunk log, final chunk of that run | **8.13 GB** |

True prefill-peak slope, least-squares over the in-loop chunk series:

| model | measured slope | `predict_peak_vram` KV term | ratio |
|---|---|---|---|
| Llama-3.2-1B | 0.2027 MB/tok (n=120, to S=32768); 0.2025 (n=275, to S=70656) | 0.0328 MB/tok | 6.2× |
| Llama-3.3-70B | 0.7149 MB/tok (n=32, to S=8191) | 0.328 MB/tok | 2.2× |

Consequences already observed on-device:

- Llama-3.2-1B's derived `max_seq_len=106987` **OOMs at ~70,656 tokens** — 34 % short.
  `max_seq_len`'s stated contract ("every request up to this length fits") does not hold.
- Llama-3.3-70B at exactly 8192 **passes** (prefill peak 9.39 GB): the KV-doubling-cliff
  boundary that motivated the check is safe. It survives on ample free VRAM, not on the
  reserve — the transient there is ~6.5 GB against a 3.50 GiB reserve.
- `MATERIALIZES_SCORES = False` and the 2026-08-15 removal of the S-linear activation term
  are both conclusions drawn from the truncated measurement. Neither is refuted by this
  spec; both are **unsupported** until re-measured.

## 2. Decisions taken (2026-08-15, with the user)

1. **Measurement fix = drop the reset.** Delete the in-loop
   `reset_peak_memory_stats()`; do not add engine-side peak state. The per-chunk print
   becomes a running high-water mark, which is how every existing log already reads.
2. **`max_seq_len` is a derived DEFAULT, not a hard cap.** Admission stays with the
   headroom predicate alone. Consequently the "8193 must raise `VramPlanError`"
   expectation recorded in `todo.md` row 1 is **retired as a wrong expectation**, not
   filed as a bug: 8193 running to completion is correct behaviour. What is wrong is
   that the predicate's prediction is too small, which is this spec's subject.
3. **Measure before deciding the budget policy.** Whether the corrected form should cost
   context length or expert-cache VRAM is deferred to Phase 2, with data.

## 3. Scope — Phase 1 only

Phase 1 delivers a correct measurement and the data it produces. It changes **no**
prediction, no budget, no derived limit.

### 3.1 `vaayu/vaayu_base.py`

Delete line 3334 (`torch.cuda.reset_peak_memory_stats()`). Amend the surrounding comment
to state that (a) the printed figure is a running high-water mark across the prefill, and
(b) the process-global counter is deliberately left alone because callers own it —
resetting it here silently truncated every external measurement of a chunked prefill.

No other behaviour changes. The print itself stays.

### 3.2 `scripts/probe_vram_curve.py`

- Replace the docstring's claim that "peak-vs-S is NOT observable inside one call" — that
  belief is what produced the bad table. State what is now true: with the engine no longer
  resetting mid-forward, `reset → forward → max_memory_allocated()` observes the whole
  prefill.
- Keep the existing per-point `alloc_before` baseline and the driver-vs-allocator header;
  both are correct and were not implicated.
- Widen the default S sweep so each arm crosses at least one prefill-chunk boundary and
  one KV capacity-doubling boundary (the two places the current form is piecewise).

### 3.3 Regression test — `tests/test_vram_peak_measurement.py` (new)

`tests/` has no GPU, so the test asserts the **invariant that broke**, not a byte count:
drive a multi-chunk prefill with `torch.cuda` monkeypatched to record calls, and assert
`reset_peak_memory_stats` is called **zero** times by the engine.

A test that asserted a peak value would need a GPU and would not have caught this — the
defect is "the engine touches shared global state", which is exactly what this asserts.

Per the gates table's lesson ("a gate that cannot fail is not a gate"), the test must be
checked by restoring the deleted line underneath it and confirming it fails.

### 3.4 Re-probe — the actual deliverable

All four geometries: Qwen3-235B, MiniMax-M2.7, Llama-3.3-70B, Llama-3.2-1B (paths in
`todo.md`'s checkpoint table). Method, per the standing environment rules:

- one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0` (28 G for the 70B — 22 G is no longer enough);
- never attach py-spy (corrupts `vaayu_rust`'s io_uring state);
- discard each checkpoint's first touch; redirect stdout to a file; `grep -a` the logs;
- record `status` per point, including OOM, rather than stopping the sweep silently.

Output: a RESULTS write-up in `plans/` carrying, per geometry, the measured transient
curve, the fitted slope, `predict_peak_vram`'s prediction at the same points, and the
signed relative error. That table is the input Phase 2 needs and the artifact that lets
anyone re-derive the verdict without a GPU.

## 4. Phase 2 — gated, NOT in scope here

Recorded so it cannot be silently dropped (CLAUDE.md §6). Each item needs Phase 1's data
before it can be specified:

- **Form.** Preferred direction: restore a *derived* attention term. The measured surplus
  over KV is 5.2× `n_heads · chunk · elem_bytes` per token on the 1B and 5.9× on the 70B;
  if the re-probe shows a stable multiplier across four geometries, it is a countable set
  of tensors (mask, scores, softmax, output copies) and stays derived rather than fitted
  (§5). The alternative — a per-geometry fitted envelope — reintroduces exactly the free
  parameter this module's own comments warn absorbs form errors, and is the fallback only
  if the multiplier is not stable. Self-calibration at load is rejected on cost: a
  calibration prefill on the streamed 70B runs ~40 s per chunk.
- **`MATERIALIZES_SCORES`.** Re-decide on the corrected data; today's `False` was measured
  through the truncated view.
- **Budget policy.** If the corrected form says today's 8192 exceeds the reserve, choose
  explicitly between growing the reserve at the expert cache's expense (Audit F8: cache
  re-saturates in ~4 tokens; Plan K: giving back 1.22 GB cost 0.4 % decode) and accepting
  shorter derived contexts. User decision, with numbers.
- **Re-derive `max_seq_len`** on all four geometries and re-run an on-device arm at each
  new limit — the 1B's is ~1.5× past what fits today, and only an on-device arm caught it.

## 5. Success criteria

Phase 1 is done when all of the following hold, with evidence:

1. `grep -n reset_peak_memory_stats vaayu/` returns nothing.
2. `tests/test_vram_peak_measurement.py` passes, and **fails** when the deleted line is
   restored (checked, not assumed).
3. Full suite still `795 passed, 2 skipped, 1 xfailed` (baseline; +1 for the new test).
4. The three on-device gates still pass: dense smoke (ppl 8.78 on Llama-3.3-70B), MoE
   accumulation order (both MoE models), prefix-reuse behaviour.
5. A re-probe RESULTS file exists in `plans/` with all four geometries, each point
   labelled ok/OOM, and the signed error against `predict_peak_vram` at every point.
6. `todo.md` reflects the outcome: Phase 1 closed, Phase 2 rows opened with the data.

Explicit non-criteria: the ±10 % gate is **expected to fail** after Phase 1 and that is
the correct result — it is currently passing against a truncated measurement. Do not
touch `predict_peak_vram` to make it pass; that is Phase 2's decision.

## 6. Risks

- **The gate goes red and stays red between phases.** Accepted and stated above; a red
  gate that describes reality beats a green one that does not.
- **Re-probing the 70B and the two MoE models is slow** (streamed weights, ~40 s/chunk on
  the 70B). Budget a long sweep; keep points few and chosen to straddle the two
  boundaries rather than to fill a curve.
- **A probe arm may OOM at large S.** That is a data point, not a failure: record
  `status=OOM` with the S that produced it and continue with the next arm.
- **Log volume.** These runs produce 300 MB–1.3 GB logs; always `grep -a`, never `cat`.
