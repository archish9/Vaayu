# Results: Bottleneck Fixes Plan (2026-07-12)

**Plan:** `docs/superpowers/plans/2026-07-11-bottleneck-fixes.md`
**Status:** 12/12 code tasks complete, all reviewed, all on `main`. First real-hardware run
(2026-07-12) surfaced a crash in Task 9's one-copy GPU rebuild — root-caused, fixed, and
RAM-regression-corrected without a GPU (see "Real-hardware crash + fix" below). Re-verification
on real hardware is the next required step.
**Execution:** subagent-driven-development (fresh implementer + independent reviewer per task,
opus for the highest-risk/architecturally-complex tasks — 8, 11, the real-hardware crash fix and
its RAM-efficiency redesign, and their re-reviews — sonnet for integration-risk tasks touching the
active EXL3 hot path, haiku for mechanical/independent tasks).

---

## Real-hardware crash + fix (2026-07-12, post-plan)

User ran `python3 test_generate.py` (Qwen3-235B) and hit `torch.AcceleratorError: CUDA error:
misaligned address` on the very first prefill MoE layer, inside the EXL3 GEMV kernel's weight read.

**Root cause (found via static analysis + hand-computed offsets against the real checkpoint's
`vaayu_expert_offsets.json` — no GPU needed to reach this conclusion):** `load_expert_into_slab`
(Task 8) returned zero-copy views into a CPU slab at file-derived byte offsets guaranteed only
itemsize-aligned, not aligned to any stronger boundary. This was always true but harmless
everywhere else (`load_expert_sparse` clones; every per-component `.to(device)` H2D copy always
allocates a fresh, naturally-aligned GPU destination on cross-device copy). Task 9's one-copy GPU
rebuild (`views_from_packed_copy`) is the ONE path that reslices sub-views inside a single shared
GPU buffer without reallocating — the first path where a CPU-side offset propagates directly into
a GPU tensor's `data_ptr()`. EXL3's GEMV kernel (`exl3_gemv_kernel.cuh:142-143`, confirmed in the
installed venv) does a raw `uint4*` (16-byte) vectorized load on the weight buffer — exactly what
faults on a non-16-aligned pointer.

**Fix, two commits:**
- `5bc5ea8` — `load_expert_into_slab` individually repacks each tensor onto a 16-byte
  (`_TENSOR_ALIGN`) boundary after the raw read. `expert_span_bytes`/`compute_slab_bytes` share
  the same `_group_layout` arithmetic as the reader (avoiding a repeat of the `ef24551`
  sizing/reading-drift class of bug). New adversarial regression test
  (`test_load_expert_into_slab_views_are_tensor_aligned`) with deliberately non-16-aligned file
  offsets and an odd byte-length tensor. Opus review: High confidence this resolves the crash —
  but flagged an Important side effect: keeping BOTH the raw read region and the repacked region
  co-resident in every persistent slab roughly doubled per-expert slab size, halving decode RAM
  cache capacity and roughly doubling per-staged-expert prefill H2D volume (a real perf giveback
  against Tasks 7-9's central goal, though not a new OOM/safety risk since the arena still respects
  its pre-set budget).
- `a6aa875` — redesign (human chose "fix now" over "ship and measure"): the raw read now lands in
  a TRANSIENT buffer drawn from `_EXPERT_BUF_POOL` (the pinned scratch pool `load_expert_sparse`
  already used, bounded by the same shared `_moe_executor` both loaders were already routed
  through — no new pool, no oversubscription). Only the 16-byte-repacked tensors persist in the
  slab. Since the persistent slab is no longer an O_DIRECT read target, it no longer needs the
  4096-byte sector rounding either — slabs end up marginally SMALLER than even the pre-crash
  baseline, not just back to parity. Opus re-review confirmed, against the real checkpoint: slab
  size went 11.29 MB (pre-crash) → 22.57 MB (post-5bc5ea8, the regression) → 11.28 MB (post-a6aa875,
  fixed) — a clean 2x reversal, both the alignment guarantee and lockstep sizing independently
  re-verified. Three tests whose assertions encoded the now-retired O_DIRECT-rounding invariant
  were updated with hand-recomputed expected values.

**Test suite:** 138/138 throughout both commits (1 new test from `5bc5ea8`, none needed for
`a6aa875`'s pure internal redesign).

**Second real-hardware run (2026-07-12): crash confirmed gone, but decode throughput regressed.**
`total generation time` went 48.4s → 66.2s (TTFT 25.52s→30.32s, decode 1.05→0.67 tok/s), uniformly
across every decode token in the run's telemetry (avg token time ~1.0-1.2s → ~1.56s, `expert_disk`
0.2-0.9s → 0.4-1.2s) — not a few outliers, a systemic per-call regression.

**Root cause (again found via code inspection + the run's telemetry, no GPU needed):**
`load_expert_into_slab` has exactly two callers — `stage_read_expert` (prefill stager) and
`_load_one` (decode's cache-miss path, hit 350-400x/token). `a6aa875`'s scratch-pool repack was
applied to BOTH unconditionally, but only the prefill stager's output is ever consumed via
`views_from_packed_copy` (the path needing 16-byte alignment) — decode's cache-miss path populates
the RAM cache and is always read back later via per-component `.to(device)` copies (Task 7, always
naturally aligned on a fresh cross-device copy), never touching `views_from_packed_copy`. Applying
the repack there anyway added `_EXPERT_BUF_POOL` contention (8 shared buffers, now drawn on by
every decode miss) plus real per-tensor `copy_()` overhead, on a call path hit hundreds of times
per token.

**Fix (`f05efeb`):** `load_expert_into_slab` gains a required `tensor_align: bool`. `True` (prefill
staging) keeps the scratch-pool repack unchanged. `False` (decode's cache-miss path) restores the
exact zero-copy behavior that existed before this whole alignment-fix arc started — raw read lands
directly in the persistent slab, no pool, no extra copies. `_group_layout`/`expert_span_bytes` now
size every group by `slab_span = max(raw_aligned_length, packed_span)` so the same shared arena
stays safely sized for either mode. Opus review (third pass over this exact code) independently
re-verified all three invariants hold simultaneously: the crash fix (16-byte alignment for
`tensor_align=True`), the RAM fix (sizing stayed at 11.29 MB, `max()` adds only ~7KB/expert vs the
theoretical worst case, confirmed against the real checkpoint), and the regression fix (`False`
branch provably never touches the pool). Two Minor test-hardening findings (a test whose name
overclaimed what it verified; a partly-circular assertion) were fixed in `c6bbbb2` — the new pool-
usage test would have caught the original unconditional-repack regression.

**Test suite:** 140/140 (138 → 139 `f05efeb` → 140 `c6bbbb2`).

**Status:** Ready to merge per three independent Opus reviews. **Needs a fourth real-hardware run**
to confirm decode throughput actually recovers to the ~1.05 tok/s / TTFT ≤~26s range — everything
above was verified by code inspection, arithmetic, and the run's own telemetry, not by directly
observing the fix's effect on new CUDA execution, since no GPU is available in this environment.

## Overview

Thirteen-task plan fixing the nine bottlenecks from the 2026-07-11 whole-project review: prefill
H2D transfer overhead (pageable staging → pinned slabs → one-copy GPU rebuild), decode
micro-overheads (per-expert dispatch H2D pairs, EXL3 object churn on VRAM hits), chunked-prefill
re-reading the whole model per chunk, the KV-snapshot FP8 downgrade, the PCIe idle-downtrain false
positive, and a measured/gated experiment on decode lookahead routing. Baseline (from the prior
plan's real-hardware run, Qwen3-235B, 439-token prompt): TTFT 25.52s, decode 1.05 tok/s, prefill
`compute=18.9s`. Target models: Qwen3-235B (91 MoE layers × 128 experts) and MiniMax-M2.5 (62 × 256)
— both must work untouched (CLAUDE.md §5).

## Tasks completed

| Task | Summary | Commits | Fix rounds |
|---|---|---|---|
| 1 | Remove per-token inference prints; cache `position_ids` allocation | `3cc14c9` | 0 |
| 2 | Decode-path `clean_memory(cuda=False)` — keeps CUDA caching allocator warm | `1f3795c` | 0 |
| 3 | Telemetry ring cap (`MAX_TOKENS=4096`); NVMe warmup buffer derived from real file sizes | `ec36eff` | 0 |
| 4 | PCIe check measures H2D bandwidth under load instead of idle link-state probing | `f180bbb` | 0 |
| 5 | Permutation-based expert dispatch (EXL3 forward) — one CPU argsort + one H2D per layer | `8c1d48e` | 0 |
| 6 | Legacy sparse-forward fixes — persistent executor reuse, vectorized module-list dispatch | `abc492e` | 0 |
| 7 | VRAM cache stores `BC_LinearEXL3` objects — skip rebuild on cache hits | `30277f4` | 0 |
| 8 | Prefill stager stages into pinned arena slabs (zero-clone) | `177c0c7` + fix `91821c7` | 1 (Important/plan-mandated: worker-publish vs. cleanup dict-mutation race, early-abort only — human approved fix) |
| 9 | One-copy GPU rebuild for staged prefill experts | `c6f0b6b` + fix `f0a65c9` | 1 (2 Important: test coverage gap for multi-byte dtype offsets, missing uint8 defensive assertion) |
| 10 | KV snapshot restores model's own cache class (FP8 preserved) | `aed229b` | 0 (2 reviewer findings overridden by controller — see Adjudicated Findings below) |
| 11 | Layer-major chunked prefill — each expert read once per prefill for MoE | `261893c` + fix `2bf86cf` | 1 (Important/plan-mandated: stale `_chunk_major_gpu` retention dict on exception path — human approved fix) |
| 12 | EXPERIMENTAL lookahead routing predictor (Phase A measurement, Phase B gated behind `VAAYU_LOOKAHEAD=1`) | `3b6ef01` + fix `cd9e462` | 1 (Important: silent exception swallowing undermined the experiment's own diagnostic purpose — fixed directly, matches established pattern from Tasks 8/11) |

**Test suite:** 121 (pre-plan baseline) → **137 passed, 0 failed**, clean tree, throughout. No regressions.

---

## Real-hardware findings caught during review (without a GPU)

Two of the three findings that required human adjudication were caught by hand-tracing code during
review, not by a real-hardware run — because none was available this session. Both were real,
narrow-window correctness risks that only manifest on an early-abort/exception path:

- **Task 8:** the prefill stager's background worker thread publishes each layer's staged slabs via
  two unlocked dict writes; `forward()`'s cleanup `finally` iterated that same dict unlocked. A
  reviewer proved (not just asserted) that on early-abort this could raise `RuntimeError: dictionary
  changed size during iteration`, masking the original exception AND permanently leaking that
  layer's slab indices (next `forward()` rebinds to a fresh dict, discarding the stale one). Fixed by
  rebinding the dict reference before iterating (atomic under the GIL) instead of iterating the live
  dict then clearing it.
- **Task 11:** the chunk-major H2D retention dict (`_chunk_major_gpu`) had a per-layer teardown but
  no exception-path sweep. A reviewer traced that a mid-chunk exception could leave a stale dict
  (keyed only by `eidx`, no `layer_idx`) that a LATER `forward()` call's VRAM-cache miss could
  misread as belonging to a different layer — silently wrong expert weights, not a crash. Fixed by
  adding the same reset to `_chunked_prefill`'s existing `finally` block (already resetting
  `_chunk_major_lens`/`_chunk_major_n`).

Both fixes were verified in a second review round to genuinely close the window without introducing
new races, using CPython GIL-semantics reasoning and direct code tracing — but neither could be
exercised end-to-end without real hardware and a real early-abort scenario. **These are exactly the
kind of subtle, low-probability, high-consequence bugs the plan's own USER CHECKPOINTs exist to
catch; they should be watched for during the real-hardware runs below**, though by design they only
fire on a rare abort path, not the happy path this plan's targets measure.

## Adjudicated findings (controller override, no fix applied)

Two reviewer findings on Task 10 were assessed and rejected rather than fixed:

1. **"Add try/except around `model._create_kv_cache()`"** — rejected. This is an internal factory
   call (object construction), not a system boundary; CLAUDE.md explicitly forbids speculative error
   handling for scenarios that can't happen. Worse, swallowing exceptions here would risk silently
   reintroducing the exact bug this task fixes (silent FP8→uncompressed KV degradation) by turning a
   loud crash into a masked one.
2. **"Inline import in test function is unconventional"** — rejected. Verified via grep that this
   matches the pre-existing codebase convention (`test_prefill_stager_slabs.py`,
   `test_moe_streaming_perf.py` already do this).

---

## Static sanity checks

```
$ venv/bin/python -m pytest tests/ -q
137 passed, 16 warnings in 3.41s

$ git status --short
(clean)

$ grep -rn "11\.28\|8\.02\|== 91\|== 94\|== 128\b\|== 256\b\|1700 \* 1024" vaayu/ --include="*.py" | grep -v "test\|#"
(no output — no hardcoded model numbers, old 1700MB warmup literal confirmed gone)

$ grep -n "inference start\|inference end" vaayu/vaayu_base.py
(no output — dead per-token prints confirmed removed)
```

---

## Real-hardware verification — OUTSTANDING (USER ACTION REQUIRED)

**No GPU was available in this execution environment at any point in this session.** Every
performance number in the plan's targets (TTFT ≤ ~18s, prefill `compute` ≤ ~12s, decode ≥ 1.05 tok/s)
is a **projection from the code changes**, not a measurement. The following real-hardware checks from
the plan are all still needed, in order:

1. **Baseline run** (`python test_generate.py`, Qwen3-235B, 439-token prompt) — compare against:

   | Signal | Prior baseline | Plan target | Regression trigger |
   |---|---|---|---|
   | TTFT | 25.52s | ≤ ~18s | > 25.5s |
   | prefill `compute` | 18.9s | ≤ ~12s | — |
   | decode | 1.05 tok/s | ≥ 1.05 tok/s (Task 7 may push ~1.1-1.2) | < 1.0 tok/s |
   | `lookahead_acc=` | (new) | printed, any value | missing line |
   | PCIe line at startup | — | `✅ [Vaayu] PCIe measured: ~18 GB/s ...` | ⚠️ warning on this box |
   | output | — | coherent | gibberish (slab/offset bug — Tasks 8/9's silent-corruption risk) |
   | after run | — | `journalctl -k \| grep -i oom` empty | any OOM-kill |

2. **Long-prompt parity check (Task 11, required)** — duplicate the test prompt to ~700-900 tokens
   (> `prefill_chunk_size=512`), run once normally and once with `prefill_chunk_size=0` forced in
   `test_generate.py`. Greedy decoding is deterministic — **generated tokens must be byte-identical**
   between the two runs. Any divergence means the chunk-major past-length/KV bookkeeping (Task 11)
   has a bug; revert commit `261893c` (and its fix `2bf86cf`) and report back.

3. **Task 12 checkpoint** — run normally, read `lookahead_acc=` on the token summaries, confirm decode
   speed is unchanged vs. the Task 9 baseline (the extra router GEMM should be noise — if decode
   slows >5%, revert `3b6ef01`+`cd9e462`). If accuracy ≥ ~60%, re-run with `VAAYU_LOOKAHEAD=1` and
   compare hit rate; keep Phase B only if it improves ≥5 points without eviction thrash.

4. **MiniMax-M2.5 spot check** — still outstanding from the prior plan too. Expect: no Fix #7
   restriction warning, stager gate active at seq_len ≥ 64, slab staging exercises the per-expert
   pageable-fallback path (256 experts/layer × 3 in-flight layers ≈ 768-of-770-slab pressure).

Any regression trigger firing → `git log --oneline` identifies the task commits (each independently
revertable, in the order landed above); revert the offending one and update this doc.

---

## Final whole-branch review (opus, 0723c49..860d332)

**Ready to merge: Yes** (pending the real-hardware checklist above). No Critical/Important findings.
The review's primary focus was cross-task composition — six tasks (5, 7, 8, 9, 11, 12) modified
overlapping regions of the same `sparse_forward` closure sequentially; the review traced the CURRENT
merged state (not just isolated diffs) to confirm they compose rather than silently interact. All
four named risks (Task 7 VRAM-cache vs. Task 11 chunk-major retention fighting over ownership;
Task 8's now-multi-call slab lifecycle vs. Task 9's per-chunk one-copy rebuild; Task 5's dispatch
locals possibly going stale across chunk-calls; Task 12's lookahead accounting under chunk-major)
resolved cleanly, primarily because the prefill/decode gate (`_vram` forced `None` during prefill,
`_chunk_major_gpu`/`_moe_prefill_pass` both false during decode) keeps the dangerous state machines
mutually exclusive rather than interleaved — verified by reading the live merged function, not the
diffs in isolation.

## Open items for final review (not blocking, tracked for the whole-branch review)

- Task 5 reviewer noted `non_blocking=True` on the unpinned CPU `_pair` tensor is a latent no-op
  (falls back to synchronous copy) — cosmetic, doesn't affect correctness, not fixed.
- Task 6 reviewer noted a stale comment ("Fallback covers unit tests whose mock vaayu_ref has no
  executor") that no longer matches the actual test fixture (which does have a real executor) — not
  fixed, purely a comment-accuracy nit.
- Task 7 reviewer noted non-atomic `"bc"` tuple construction vs. entry assignment under a
  mid-construction exception — self-healing (next hit rebuilds), not fixed. Reconfirmed by the final
  review.
- Task 9 test coverage for multi-byte dtype offsets and the uint8 defensive assertion were both
  fixed in review (see table above) — no open item remains from that task.
- **No unit test exercises the full stack simultaneously** (chunk-major + slab-staged + GPU
  retention + BC reuse all at once) — `tests/test_layer_major_prefill.py` covers the retention
  lifecycle with an empty `_prefill_staged_slabs`, so the specific interaction of a slab-backed
  `_gpu_blob` being retained across chunks while its CPU slab is released is verified by code
  inspection only, not by a test. Final reviewer recommends adding this test but rates it Minor/
  non-blocking since the real-hardware long-prompt parity check below would catch a functional break.
- **Latent multi-turn VRAM pressure** (Task 7 × Task 11): a second generation's chunk-major prefill
  allocates fresh per-layer retained GPU tensors on top of whatever VRAM-cache entries the first
  generation's decode left resident (cache is never cleared between `generate()` calls). Bounded to
  ~one layer's worth (~1.4GB for Qwen) since retention frees per-layer, not the full budget — the
  single-prompt `test_generate.py` target never exercises multi-turn, so this is theoretical for the
  plan's stated scope. Worth a comment or hardware note if multi-turn usage is added later.
- **Pre-existing stale docstring** (`vaayu/moe_patch.py`, not introduced by this plan): a comment
  still cites "~88 MB per expert" cost model, contradicting the actual measured 11.28MB (see
  `moe_perf_rootcause` memory) this whole line of work is built on. Out of scope for surgical-change
  rules since it predates this branch, but actively misleading — flagged for a future cleanup pass.
- MiniMax-M2.5 spot check outstanding (see Real-hardware verification above) — same open item
  carried forward from the prior plan's RESULTS doc.
