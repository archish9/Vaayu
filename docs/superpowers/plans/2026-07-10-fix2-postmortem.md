# Fix #2 Postmortem: Bulk-Sequential Expert Reads — Reverted

**Date:** 2026-07-10
**Status:** Implemented, code-reviewed (clean), real-hardware-verified, **found to
regress performance, and reverted.**
**Verdict:** The plan's core performance premise — that one large sequential
buffered read beats many small parallel `O_DIRECT` reads for the same bytes —
does not hold on this hardware/kernel/io_uring configuration. The
*implementation* was correct (extensively verified); the *hypothesis* was wrong.

---

## 1. What Fix #2 attempted

Plan: `docs/superpowers/plans/2026-07-09-fix2-sequential-prefill.md`

**Goal:** cut Time-To-First-Token (TTFT), measured at **89s** in the original
root-cause report, by reading MoE experts during prefill as a few large
sequential NVMe reads per layer instead of thousands of small per-expert
scatter reads.

**Rationale given in the plan:** during prefill of a long prompt, routing
selects nearly every expert in every layer (`P(expert unused) ≈ e^-27 ≈ 0`),
so effectively the whole model is read from disk once. The existing code reads
this as ~12,000 individual `load_expert_sparse` calls (one per (layer,
expert)). The plan's premise: a Gen4 NVMe does ~5-6 GB/s on large sequential
reads, but the existing scattered-read pattern was assumed to be far below
that, so replacing "N small reads" with "1 big sequential read per layer"
should approach the drive's true sequential bandwidth.

**Implementation** (3 commits, now reverted — see §5):
1. `load_experts_bulk(file_path, experts_metas, use_direct_io=None)` in
   `vaayu/persist/safetensor_model_persister.py`: computes the
   `[min_offset, max_end)` byte span covering every requested expert, reads
   that whole span in **one** `io_uring_read_into` call at queue depth 32,
   using an **unpinned, buffered** (non-`O_DIRECT`) buffer, then slices each
   expert's tensors out of the resulting buffer.
2. A test-coverage strengthening commit (added after code review flagged a
   coverage gap — see §2).
3. Wiring into `moe_patch.py`'s active EXL3 `sparse_forward`: when a layer's
   miss set is `≥ 50%` of that layer's expert count, use the bulk reader;
   otherwise keep the existing per-expert parallel path (used for decode's
   small miss sets).

## 2. Implementation correctness — thoroughly verified

Before real-hardware testing, this fix went through the project's standard
subagent-driven review process: two implementer/reviewer cycles per task, plus
a final whole-branch review, at increasing scrutiny given the risk profile
(silent weight corruption is possible if a byte-reconstruction bug exists).

- **Offset arithmetic** was independently hand-traced by two different
  reviewers using two different, adversarial synthetic scenarios each (gaps
  between experts, multiple tensor keys per expert, non-contiguous/scrambled
  ordering, non-U8 dtypes) — both proved `local = meta.file_offset -
  span_start` is correct for the fully general case, not just the trivial
  test case originally shipped.
- **Test-coverage gap found and fixed:** the first review found the original
  test only covered a trivial contiguous/single-key/U8/ascending-order case. A
  follow-up commit added real coverage for gaps, multi-key experts, F16/F32
  dtypes, and scrambled iteration order — re-reviewed and confirmed the new
  tests would actually fail on a reintroduced bug (not just pass regardless).
- **Cross-task integration** (does `load_experts_bulk`'s return shape exactly
  match what the wiring code and `expert_cache.put()` expect?) was flagged as
  unverifiable from either task's isolated diff and explicitly resolved in the
  final whole-branch review by reading both together.
- **Dormant code path correctly avoided:** `moe_patch.py` contains a second,
  unused `sparse_forward` variant (non-EXL3 legacy path) with its own
  `if misses:` block. Two independent reviews confirmed via `grep` that only
  the active EXL3 path was touched.
- **No crash/corruption risk identified.** Final review's explicit conclusion:
  *"the one property with real correctness stakes — that `load_experts_bulk`
  returns exactly what `load_expert_sparse` would for the same experts — is
  provably true by hand-traced byte arithmetic and a shared reconstruction
  helper."*

**Conclusion: the code was correct.** The problem was performance, not
correctness.

## 3. Real-hardware verification result

Ran `test_generate.py` (Qwen3-235B-A22B, EXL3 5.0bpw, 439-token prompt) with
Fix #2 live:

```
Time to first token: 112.95s
Decode throughput: 0.31 tokens/sec
```

The original root-cause report's baseline (pre-any-fix, different cache
budget/config) was **89s** TTFT. The plan's own success criterion was "TTFT
drops meaningfully from 89s (target ~50-60s)." Instead, TTFT was **higher**
than the pre-fix baseline.

The first token's telemetry line (Fix #6 instrumentation) showed the
prefill step spending **71.5s of 85.1s total** in `expert_disk` — i.e. disk
I/O, not compute, exactly as expected — but the absolute number did not
improve.

Decode-phase numbers (0.31 tok/s, 64% hit rate on the tail) were in line with
or better than prior fixes' verification runs, so this was not a general
regression — it was specific to the prefill/bulk-read path.

## 4. Root cause: direct A/B benchmark

Rather than infer from the full pipeline's noisy timing, `load_experts_bulk`
and `load_expert_sparse` were benchmarked directly against the real checkpoint
on disk, isolating the read path from all other prefill overhead (router,
GPU compute, cache bookkeeping).

**Methodology:** two different, previously-untouched layers (to guarantee a
genuinely cold page cache for both — a same-layer sequential comparison would
have let the second measurement hit page cache warmed by the first, invalidating
the result). Full 128-expert layer (~1.5 GB) read cold from disk for each.

```
per-expert COLD (serial, one call per expert): 128 experts, 1514.3 MB in 0.527s -> 2.87 GB/s
bulk COLD      (one sequential span read):     128 experts, 1514.3 MB in 0.735s -> 2.06 GB/s
```

**Bulk was ~28% *slower* than per-expert, even in serial form** — and
production's per-expert path is *not* serial, it's parallelized across
`vaayu_ref._moe_executor`'s worker threads, so the real-world gap in favor of
per-expert is almost certainly larger than this measurement shows.

### Why the premise failed

Two compounding factors, both traceable to specific code:

1. **Queue-depth parallelism, not sequentiality, is what wins on this
   NVMe/io_uring setup.** The per-expert path issues reads via
   `vaayu_ref._moe_executor.map(...)` — multiple worker threads, each calling
   `io_uring_read_into(..., 8, use_direct_io)` (queue depth 8) concurrently.
   During a full-layer prefill load with ~8 workers × qd=8, up to ~64 requests
   can be in flight simultaneously across independent io_uring submissions.
   `load_experts_bulk` issues exactly **one** synchronous call at qd=32 from a
   single thread — one submission, no cross-call concurrency. On this
   hardware, aggregate parallel small-read throughput beat single large-read
   throughput.

2. **Buffered I/O vs `O_DIRECT`.** The plan's own design (see plan §"Task 1
   Step 3" comment) deliberately made `load_experts_bulk` use a fresh
   unpinned buffer with **buffered** I/O (`use_direct_io` hardcoded to
   `False` in the `io_uring_read_into` call), reasoning that "bandwidth here
   is dominated by sequential throughput, not pinned-DMA latency." In
   production, the per-expert path (when the pinned buffer pool is
   initialized, as it is during real inference) uses `O_DIRECT` + pinned
   memory, bypassing the page cache and kernel readahead machinery entirely.
   The assumed "5-6 GB/s Gen4 sequential" figure is an `O_DIRECT` raw-device
   number; buffered reads through the page cache do not necessarily reach it,
   especially for reads much larger than typical readahead windows.

Both factors point the same direction: this specific combination of hardware,
kernel io_uring implementation, and the two I/O modes compared favors **many
concurrent small `O_DIRECT` reads** over **one large buffered sequential
read** — the opposite of the plan's assumption.

## 5. Decision: reverted

All three commits were cleanly reverted via `git revert` (preserves history,
does not rewrite it):

| Reverted commit | Revert commit | Subject |
|---|---|---|
| `321a441` | `cbd964e` | feat: use bulk sequential read for large expert miss sets |
| `ef6d94b` | `a585db0` | test: cover gaps, multi-key experts, non-U8 dtype, scrambled order |
| `3e14912` | `89038b6` | feat: add load_experts_bulk sequential reader |

Post-revert: full test suite back to 95/95 passing (the pre-Fix-#2 count).
`vaayu/moe_patch.py` and `vaayu/persist/safetensor_model_persister.py` are
byte-identical to their state before Fix #2 (base commit `9533fe9`, Fix #3's
HEAD).

The original implementation remains in git history (commits `3e14912`,
`ef6d94b`, `321a441`) if a future attempt wants to build on the verified-correct
`load_experts_bulk` code with a different I/O strategy (see §6).

## 6. Recommendations for a future attempt

If TTFT is revisited, the fix needs a different mechanism, not just a
threshold/tuning change on the current one:

1. **Match the parallelism, not just the sequentiality.** Split each layer's
   bulk span into N sub-ranges and issue them as N concurrent `io_uring`
   submissions (matching or exceeding the per-expert path's effective
   concurrency), rather than one single synchronous call. This keeps the
   "read contiguous ranges instead of thousands of scattered tiny ones"
   benefit (fewer, larger I/Os) while restoring the queue-depth parallelism
   that this benchmark shows is the actual performance driver.
2. **Try `O_DIRECT` for the bulk path**, using a pinned (or sector-aligned)
   buffer instead of `torch.empty` + buffered read, to remove the page-cache
   variable from the comparison entirely.
3. **Re-benchmark with the exact production concurrency** (the real
   `_moe_executor` pool size and qd=8 pattern) as the control, not a serial
   loop — this postmortem's serial per-expert number (2.87 GB/s) is already a
   *conservative* baseline for what the real code achieves; the true gap is
   probably larger.
4. Consider whether TTFT is better attacked from a different angle entirely
   (e.g. increasing per-expert `_moe_executor` worker count / queue depth
   further, since that axis is what's actually winning here) rather than
   bulk sequential reads.
