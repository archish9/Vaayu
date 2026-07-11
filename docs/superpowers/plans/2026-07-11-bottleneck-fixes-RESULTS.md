# Results: Bottleneck Fixes Plan (2026-07-12)

**Plan:** `docs/superpowers/plans/2026-07-11-bottleneck-fixes.md`
**Status:** 12/12 code tasks complete, all reviewed, all on `main`. Real-hardware verification
(Task 13 Steps 3-4) is OUTSTANDING — no GPU was available in this execution environment.
**Execution:** subagent-driven-development (fresh implementer + independent reviewer per task,
opus for the highest-risk/architecturally-complex tasks — 8, 11, and their re-reviews — sonnet for
integration-risk tasks touching the active EXL3 hot path, haiku for mechanical/independent tasks).

---

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
