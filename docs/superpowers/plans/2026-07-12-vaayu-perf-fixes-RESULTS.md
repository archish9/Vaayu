# Vaayu Perf Fixes + Advanced Features — Results

Plan: `2026-07-12-vaayu-perf-fixes-and-features.md`. Executed via
superpowers:subagent-driven-development, task-by-task, on-device A/B on Qwen3-235B
(RTX 5060 Ti) — no MiniMax checkpoint available this session, per plan's own
"Qwen only for on-device A/B" note. Base commit `4935259`, final commit `22b5a3c`.

## Landed

| Task | What | Commit(s) | On-device result |
|---|---|---|---|
| 1 (CR-4) | `compute_moe_geometry`: max-over-sampled-layer + warn on non-uniform | `9744e97` | n/a (pure hygiene) |
| 2 (CR-1) | Gate always-on lookahead probe behind `VAAYU_LOOKAHEAD=1` | `1ab1b1d` | decode 0.92→0.97 tok/s |
| 3 (B1) | Prefill zero-copy staging + GPU-side 16-byte alignment | `e45adee`, `c5014cd` | TTFT 29.8-30.4s→27.4-27.5s (real), no crash, coherent. Gap to plan's ~19-20s hoped target **root-caused and closed 2026-07-18**: disk-bandwidth-bound (~5.5GB/s), not a bug — see below |
| 4 (CR-3) | Stream-scoped event replaces per-layer device sync | `dddaefc` | correctness clean, TTFT flat vs Task 3 (sub-1%, noise) |
| 5 (CR-2) | Phase B prefetch dedicated executor + zero-clone | `384a15c`, `41b6573` | **`VAAYU_LOOKAHEAD=1` hard-blocked** — 2 real fix rounds landed (harden default path) but a 3rd race (suspected stager-pool worker-vs-worker) not found; flag raises `RuntimeError` at init |
| 6 (#4) | Cache-aware routing, ships dark | `7c9c468`, `79ddc65` | feature-off byte-identical (proven + on-device confirmed); perplexity/miss sweep complete (4/4 points, 2026-07-18, decode-mode script fix) — **no bias value helps**, `VAAYU_CACHE_BIAS` stays `0.0` permanently |
| 7 | NVMe controller warm-up pre-read | `2ee52b2` | non-blocking confirmed; also fixed a pre-existing unflagged dense/MoE double-warmup bug |
| 8 | io_uring `REGISTER_BUFFERS` | — | **gate-skipped**: read path confirmed I/O-bound (NVMe 59-69% util, ~4GB/s, CPU ~4% avg) |
| final review fixes | Hardened `VAAYU_LOOKAHEAD` guard against post-init env mutation; stale comment | `22b5a3c` | 169/169 tests |

## Outstanding follow-ups (non-blocking, documented)

- **Task 5**: root cause of `VAAYU_LOOKAHEAD=1`'s crash not found — suspected worker-vs-worker race inside `_stager_executor`'s own multiple workers (never audited; only main-thread-vs-stager was). Diagnostic suggestion in `.superpowers/sdd/task-5-report.md`: force `max_workers=1` and see if the crash signature/timing changes. Still open as of 2026-07-18.
- ~~**Task 3**: ~7-10s gap between the plan's hoped ~19-20s TTFT and the delivered 27.3s~~ —
  **RESOLVED 2026-07-18**. Root-caused with new env-gated instrumentation (`VAAYU_OVERLAP_TRACE=1`,
  `vaayu_base.py::_prefill_stage_worker` + `moe_patch.py::sparse_forward`) instead of a profiler
  (py-spy was tried first and found to corrupt this process — see below). Per-layer: disk read
  (128 experts, ~1.44GB) averages 261ms ≈ 5.5GB/s real NVMe throughput; MoE-FFN compute is only
  ~10ms; attention+router+dispatch ~97ms; the depth-2 stager genuinely does prefetch the next
  layer while the consumer is still finishing the current one (confirmed via real timestamps, not
  assumed) — **the pipeline already achieves close to its best possible overlap.** It's a rate
  mismatch (read 261ms > per-layer compute+overhead 107ms), not a depth or GIL problem: one
  layer's compute can never hide one layer's read when prefill is strictly sequential. Two earlier
  hypotheses were tested and killed first: pinned-buffer-pool-cap-starves-io_uring-QD32 (8→16 made
  TTFT *worse*, reverted) and NVMe APST micro-sleep (disabled, zero effect). **Nothing left to fix
  — ~24-25s is the real hardware floor for this drive/prompt-size**, not a bug. Detail:
  `moe_perf_rootcause` memory.
- **py-spy / ptrace profilers are unsafe against a live Vaayu process** (found 2026-07-18,
  incidental to the Task 3 investigation): attaching `py-spy dump` corrupted `vaayu_rust`'s
  io_uring completion-queue state (`RuntimeError: incomplete: N/M CQEs`, 100+ occurrences per
  profiled run vs. 0 across 6 unprofiled runs) and once crashed a run outright. **Do not use
  ptrace-based samplers against this codebase again** — use env-gated `print()`/timestamp
  instrumentation instead (see `VAAYU_OVERLAP_TRACE` above). A real, separate, independent bug was
  found via that one crash and fixed regardless: `stream_generate()` (`vaayu/streaming.py`) used
  to hang forever if the background `generate()` thread raised any exception (the consumer blocks
  on `TextIteratorStreamer`'s unbounded queue, which only unblocks via `generate()`'s own
  normal-completion path). Now catches, unblocks the streamer, and re-raises after `join()`.
  Covered by `tests/test_streaming.py` (2 tests). Detail: `io_uring_cqe_overflow_bug` memory.
- ~~**Task 6**: perplexity sweep at `VAAYU_CACHE_BIAS` = 0.3/0.5/0.8 not run~~ — **RESOLVED
  2026-07-18**. The original `scripts/cache_bias_ppl.py` scored PPL via a single-shot
  `model(ids)` forward, which is itself a prefill pass — `_build_resident_mask()` gates
  cache-aware routing off during prefill, so the PPL metric was bias-blind at every value
  (identical `ppl=7.631` for 0.0/0.3/0.5/0.8, no fault of the routing code). Fixed to score PPL
  over a real teacher-forced decode loop (48-token prefix, 48 scored decode steps, so
  `_moe_prefill_pass=False` on every scored step). Full 4-point sweep re-run on-device (Qwen3-235B):
  miss/token flat at 448.0–448.6 across the whole bias range (noise, ~0.13% spread, no trend) —
  cache-aware routing gives **zero measured miss reduction** at any tested bias, because the
  RAM cache is capacity-bound (working set > 9.13GB budget across 94 layers), not
  selection-bound — biasing *which* experts get picked can't fix a cache too small to hold the
  working set. `VAAYU_CACHE_BIAS` stays `0.0` (off) **permanently**, not pending. Detail:
  `.superpowers/sdd/task-6-report.md`.
- **MiniMax-M2.5 on-device spot check**: not run, no checkpoint on this box. All code is architecturally model-agnostic (CLAUDE.md §5) — every sizing/geometry test is parametrized over Qwen-like and MiniMax-like synthetic geometries — but unexercised on real MiniMax hardware.
- **Task 7**: dense re-scope (`_warmup_nvme()` → `not self.is_moe`-gated) verified by code inspection only, no dense checkpoint available to run on-device.

## Full-suite status

171/171 passing as of 2026-07-18 (169 at final plan commit + 2 new tests for the `stream_generate`
hang fix, `tests/test_streaming.py`). Full ledger with per-task review detail: `.superpowers/sdd/progress.md`.
