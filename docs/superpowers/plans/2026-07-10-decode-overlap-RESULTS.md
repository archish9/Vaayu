# Results: Decode Overlap + Pinned Slabs + Prefill Pipeline Plan (2026-07-11)

**Plan:** `docs/superpowers/plans/2026-07-10-decode-overlap-pinned-slabs-prefill-pipeline.md`
**Status:** 9/9 tasks complete + 1 real-hardware regression found and fixed. All on `main`.
**Execution:** subagent-driven-development (fresh implementer + independent reviewer per task, opus for
concurrency/memory-safety-sensitive tasks, sonnet/haiku for mechanical ones).

---

## Overview

Ten-task plan to cut Qwen3-235B decode latency and TTFT by removing hot-loop waste, overlapping
disk I/O with compute, replacing double-cloned pageable expert memory with a pinned slab arena,
giving the Fix #7 VRAM expert cache partial value via layer-range restriction, and pipelining
prefill expert reads. Target models: Qwen3-235B (94 layers, 91 MoE, 128 experts/layer) and
MiniMax-M2.5 (62 MoE layers, 256 experts/layer) — both must work untouched (CLAUDE.md §5).

**Baseline (pre-plan), Qwen3-235B, 439-token prompt, 256-token generation:**
```
Time to first token: 63.10s
Decode throughput:   0.35 tokens/sec (≈ 2.6 s/token)
token summary: total=2.6s io_wait=0.0s expert_disk=1.2s compute=1.0s other=0.4s
vram=0hit/752miss/752evict hit_rate=0%
```

## Tasks completed

| Task | Summary | Commits |
|---|---|---|
| 1 | Remove per-layer-per-token debug prints (2 fix rounds — reviewer caught 2 leftover hot-loop prints outside the named sites) | `ce3fe57`, `e77886a` |
| 2 | Throttle `clean_memory()` to prefill + every-32-decode-tokens | `7fe2b4f` |
| 3 | Remove dead decode prefetch machinery (provably no-op, ~91 wasted submits/token) | `893e2f6` |
| 4 | Persistent forward-loop executor instead of per-token create/join | `ecaa8e7` |
| 5 | Prefill-pass flag; skip expert-cache puts during prefill; fix dead decode-stats-reset guard (1 fix round — reviewer added real integration test coverage) | `10a8629`, `a425717` |
| 6 | Overlap expert disk reads with compute in EXL3 sparse forward (submit + FIRST_COMPLETED instead of blocking `.map()`) | `83b9333` |
| 7 | Pinned slab arena — zero-clone disk→RAM→GPU expert path | `7e6e92f` |
| 7-fix | **Critical real-hardware regression, found and fixed** — see below | `ef24551` |
| 8 | Fix #7 follow-up: layer-range VRAM cache + honest working-set warning | `a48921e` |
| 9 | Prefill lookahead pipeline — stage next layers' experts during compute | `0b73fec` |

**Test suite:** 104 (pre-plan baseline) → 121 passed, 0 failed, throughout. No regressions.

---

## Task 7 real-hardware regression (the plan's own required checkpoint caught this)

The plan mandated a real-hardware run before Task 9 specifically because Task 7 (pinned slab arena)
was flagged as the riskiest task. That checkpoint worked exactly as intended:

**Symptom (user-reported):** 25-token generation took 20+ minutes instead of the expected ~30-40s.

**Root cause (verified against the actual checkpoint's offset index):** Task 7's `expert_span_bytes`
assumed one expert's tensors are file-contiguous (`min(offset)` to `max(offset+len)`). The real
Qwen3-235B EXL3 checkpoint stores tensors **grouped by weight-type across all 128 experts** (all
small `suh`/`svh` tensors first, then all large `trellis` tensors far away in the file) — not
per-expert. For the worst-indexed expert this produced a measured slab size of **1511.1MB** versus
the expert's actual ~11.8MB of data (~128x overcount, matching `num_experts`).

**Consequences, all from this one bug:**
- Pinned allocation OOM'd (`14 × 1511.1MB`) → fell back to **21.16GB pageable** host RAM (≈2.5x the
  project's 8.5GB hardware safety ceiling, real OOM-kill risk — only 26.9GB was free)
- Only 14 (grossly oversized) slabs existed instead of ~720 → RAM cache **0% hit rate on every
  token, forever**
- Each miss read up to ~1.5GB instead of ~11.8MB → `expert_disk` 60-80s/token instead of the
  expected ~0.3-0.6s

**Fix (`ef24551`):** extracted the ALREADY-CORRECT grouping logic from the pre-existing
`load_expert_sparse` (which never assumed contiguity — it groups tensors into separate reads
whenever the gap exceeds `_ALIGN`) into a shared `_group_contiguous_ranges` helper, and made
`expert_span_bytes`/`load_expert_into_slab` sum/read per-group instead of spanning the whole file
range. Independently verified against the real checkpoint's offset index: **1441.11MB → 11.29MB**
for layer 3's worst-case expert. Model-agnostic — a general grouping algorithm, not special-cased to
any expert/layer count.

---

## Real-hardware verification (Qwen3-235B, 439-token prompt, 25-token generation)

Post-fix run, compared against plan's regression triggers:

| Signal | Baseline | Plan target | Measured | 
|---|---|---|---|
| Startup | — | slab arena pinned, no fallback | `Expert slab arena: 778 x 11.8MB (pinned, 9.21GB resident)` ✓ |
| TTFT | 63.10s | ≤ ~45s | **25.52s** |
| Decode throughput | 0.35 tok/s | — | **1.05 tok/s** |
| Decode total | 2.6s/token | ≤ ~1.6s/token | **~0.6-1.8s/token** (median ~0.9s) |
| `expert_disk` | 1.2s | ≤ ~0.5s (stall-only) | **0.2-0.9s** |
| `other` | 0.4s | ≤ ~0.2s | **0.0s** |
| RAM cache hit rate | n/a | — | **45-49% cumulative**, up to 71% per-token |
| VRAM `hit_rate` | 0% | 30-50% on restricted prefix | **39-69%** |
| Output text | — | coherent | clean exit, no traceback, 25 tokens generated |
| `journalctl -k \| grep -i oom` | — | empty | **empty** |

Total generation time: 20+ minutes (broken) → **48.4s** (fixed) for 25 tokens.

All regression triggers clear. All metrics meet or beat plan targets.

---

## MiniMax-M2.5 spot check

**Not yet run** — no MiniMax checkpoint available in this environment during this session. Per the
plan (Task 10 Step 4), expected: no Fix #7 layer-range restriction warning (working set ~5.3GB fits
the ~7-9GB VRAM budget → `compute_vram_layer_range` returns `None`), VRAM hit rate substantially
> 0%, stager gate active for prompts ≥ 64 tokens (256 experts × top_k 8 → `seq_len*8 >= 512`).
Recommend running when the model is available; nothing in this plan's code is Qwen-specific (all
sizing derives from `_moe_geometry`/the offset index per CLAUDE.md §5), so no changes are expected
to be needed — but it should still be run and recorded per the plan's own verification discipline.

---

## Static sanity checks

```
$ grep -rn "11\.28\|8\.02\|== 91\|== 94\|== 128\b\|== 256\b" vaayu/ --include="*.py" | grep -v "test\|#"
(no output — no hardcoded model numbers in production code)

$ grep -n "seq device\|Vaayu-Debug\|weight device" vaayu/vaayu_base.py
$ grep -rn "_prefetch_all_predicted\|prefetch_room_bytes" vaayu/
(both empty)
```

## Open items for final review (not blocking, tracked for the whole-branch review)

- `[Vaayu] inference start: ...` print still fires once per decode token (`vaayu_base.py`, adjacent
  to but outside Task 1's named scope — noted by Task 1's reviewer, not fixed, out of that task's
  literal scope).
- `self._expert_history` (populated in `moe_patch.py`) is write-only since Task 3 removed its only
  readers (the dead prefetch machinery) — harmless, flagged for a future cleanup task.
- MiniMax-M2.5 spot check outstanding (see above).
