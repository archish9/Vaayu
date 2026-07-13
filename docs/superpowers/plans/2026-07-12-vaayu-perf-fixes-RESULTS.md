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
| 3 (B1) | Prefill zero-copy staging + GPU-side 16-byte alignment | `e45adee`, `c5014cd` | TTFT 29.8-30.4s→27.4-27.5s (real, below plan's ~19-20s hoped target — backlog item), no crash, coherent |
| 4 (CR-3) | Stream-scoped event replaces per-layer device sync | `dddaefc` | correctness clean, TTFT flat vs Task 3 (sub-1%, noise) |
| 5 (CR-2) | Phase B prefetch dedicated executor + zero-clone | `384a15c`, `41b6573` | **`VAAYU_LOOKAHEAD=1` hard-blocked** — 2 real fix rounds landed (harden default path) but a 3rd race (suspected stager-pool worker-vs-worker) not found; flag raises `RuntimeError` at init |
| 6 (#4) | Cache-aware routing, ships dark | `7c9c468`, `79ddc65` | feature-off byte-identical (proven + on-device confirmed); perplexity sweep 1/4 points (bias=0.0: ppl=7.631) |
| 7 | NVMe controller warm-up pre-read | `2ee52b2` | non-blocking confirmed; also fixed a pre-existing unflagged dense/MoE double-warmup bug |
| 8 | io_uring `REGISTER_BUFFERS` | — | **gate-skipped**: read path confirmed I/O-bound (NVMe 59-69% util, ~4GB/s, CPU ~4% avg) |
| final review fixes | Hardened `VAAYU_LOOKAHEAD` guard against post-init env mutation; stale comment | `22b5a3c` | 169/169 tests |

## Outstanding follow-ups (non-blocking, documented)

- **Task 5**: root cause of `VAAYU_LOOKAHEAD=1`'s crash not found — suspected worker-vs-worker race inside `_stager_executor`'s own multiple workers (never audited; only main-thread-vs-stager was). Diagnostic suggestion in `.superpowers/sdd/task-5-report.md`: force `max_workers=1` and see if the crash signature/timing changes.
- **Task 6**: perplexity sweep at `VAAYU_CACHE_BIAS` = 0.3/0.5/0.8 not run (deliberately stopped after the 0.0 baseline point to avoid further on-device time for a non-blocking recommendation-only sweep). `scripts/cache_bias_ppl.py` is fixed and ready to re-run.
- **MiniMax-M2.5 on-device spot check**: not run, no checkpoint on this box. All code is architecturally model-agnostic (CLAUDE.md §5) — every sizing/geometry test is parametrized over Qwen-like and MiniMax-like synthetic geometries — but unexercised on real MiniMax hardware.
- **Task 7**: dense re-scope (`_warmup_nvme()` → `not self.is_moe`-gated) verified by code inspection only, no dense checkpoint available to run on-device.

## Full-suite status

169/169 passing at final commit. Full ledger with per-task review detail: `.superpowers/sdd/progress.md`.
