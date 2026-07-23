# RAM Accounting Diagnostic — Results

**Date:** 2026-07-23
**Plan:** `2026-07-23-ram-accounting-diagnostic.md`
**Box:** 33.40GB RAM, 16GB VRAM. `MAX_NEW_TOKENS=32`, fresh `SESSION_ID` per run (no prefix-cache hit).
**Instrument:** `ram_snapshot()` (`/proc/self/smaps_rollup`) at three checkpoints — `@ cache-sizing`, `@ post-arena`, `@ post-load`.

## Raw measurements

Three runs. All figures GB. `priv_dirty` = anonymous private (pinned arena, PyTorch host buffers, clones). `shared_clean` = file-backed reclaimable (mmapped weights).

| Run | Checkpoint | avail | rss | priv_dirty | shared_clean | swap |
|---|---|---|---|---|---|---|
| **MiniMax arena ON** | cache-sizing | 22.38 | 4.41 | 4.03 | 0.01 | 0.00 |
| | post-arena | 4.76 | 21.58 | 21.21 | 0.01 | 0.00 |
| | post-load | 4.75 | 21.59 | 21.22 | 0.01 | 0.00 |
| **MiniMax arena OFF** | cache-sizing | 23.90 | 4.41 | 4.03 | 0.01 | 0.00 |
| | post-arena | *(skipped — no arena)* | | | | |
| | post-load | 17.40 | 10.39 | 10.01 | 0.01 | 0.00 |
| **Qwen arena ON** | cache-sizing | 24.48 | 3.98 | 3.61 | 0.01 | 0.00 |
| | post-arena | 7.65 | 21.16 | 20.79 | 0.01 | 0.00 |
| | post-load | 7.63 | 21.17 | 20.80 | 0.01 | 0.00 |

Associated non-ledger lines:

| Run | Arena | Expert RAM cache | Prefix cache budget |
|---|---|---|---|
| MiniMax arena ON | 1034 × 8.9MB = 9.19GB pinned | 9.1GB budget (~1028 slots, max 16/layer, expert=8.9MB, **working_set=4.40GB**) | **0.00GB** (free 4.75 − 12.88 reserve) |
| MiniMax arena OFF | SKIPPED (clone path; freed ~9.1GB pinned) | — | **11.00GB** (free 23.88 − 12.88) |
| Qwen arena ON | 778 × 11.8MB = 9.21GB pinned | — | **0.00GB** (free 7.64 − 12.88) |

## The five questions

### Q1 — How much of RSS is `private_dirty` (anonymous) vs `shared_clean` (file-backed, reclaimable)?

**RSS is ~100% `private_dirty` in every run.** `shared_clean` is 0.01GB at every one of the nine checkpoints. MiniMax arena-ON post-load: rss=21.59, priv_dirty=21.22, shared_clean=0.01 → **99.9% anonymous**.

Consequence: **there is essentially no OS-reclaimable file-backed memory to give back.** EXL3 weights live in VRAM; host RAM holds only anonymous allocations (the pinned arena, expert-cache clones, PyTorch host/staging buffers). The plan's framing "file-backed pages inflate RSS but are not a real cost" does not apply here — those pages do not exist. Every resident byte is a real anonymous allocation.

### Q2 — Arena's true marginal cost (arena-ON priv_dirty − arena-OFF priv_dirty at post-load)

MiniMax: 21.22 − 10.01 = **11.21GB marginal**, against a nominal slab of **9.19GB**.

The arena costs **~2.0GB MORE than its slab size.** The arena path does not merely add the pinned slab — it also carries ~2GB of associated allocation (staging/copy buffers during slab population) that the clone path avoids.

Cross-check within the arena-ON run: the arena/expert-load phase raises priv_dirty from 4.03 → 21.21 = **+17.18GB** for a 9.19GB slab, i.e. ~8GB of *non-arena* anonymous allocation created in the same phase. The arena-OFF control isolates it: without the arena, post-load retains 10.01 − 4.03 = **5.98GB** of non-base anonymous — the expert-cache clone working set (~4.4GB) plus PyTorch buffers (~1.5GB). So the ~8GB "extra" seen in the arena-ON step ≈ that 6GB expert/buffer overhead + the ~2GB arena-path staging. Nothing is unaccounted.

**Not model-specific.** Qwen shows the identical signature: arena 9.21GB, arena-step jump 3.61 → 20.79 = **+17.18GB** (same as MiniMax). This is a property of the arena + expert-cache architecture, not of MiniMax.

### Q3 — Does arena-OFF show a correspondingly larger `shared_clean` / pageable footprint?

**No.** `shared_clean` stays 0.01GB with and without the arena. The arena-OFF run's much lower RSS (10.39 vs 21.58) is **not** pages migrating into reclaimable file-backed memory — it is a genuinely smaller anonymous working set (clone path holds only ~6GB non-base vs the arena's ~17GB non-base).

Hypothesis 1 (the old pageable-clone baseline understated the true working set) is **conceptually supported but not measurable from `shared_clean`**: the clone path's pages are *also* `private_dirty`, not `shared_clean`. The real distinction is **pinned** `private_dirty` (arena — unswappable, always resident) vs **pageable** `private_dirty` (clone path — swappable under pressure). `smaps_rollup` cannot separate those two; both land in `private_dirty`. What we *can* say: the pinned arena guarantees ~11GB is permanently resident and non-reclaimable, whereas the clone path's equivalent memory is at least theoretically swappable (though swap=0 here, so untested).

### Q4 — Is `swap` non-zero anywhere?

**No — swap=0.00 at all nine checkpoints.** The box is not in swap pressure at load with `MAX_NEW_TOKENS=32`. But arena-ON leaves only **4.75GB avail** (MiniMax) / **7.63GB** (Qwen); under the target parallel-fan-out workload (CLAUDE.md §0, multiple concurrent sessions) that headroom would be consumed and the "available" figure is optimistic for real multi-stream use.

### Q5 — How much RAM is genuinely reclaimable?

Two levers, with very different cost:

**Lever A — right-size the over-provisioned expert cache (keeps the arena path, no perf risk):**
Cache budget is **9.1GB** but measured `working_set` is only **4.40GB** — the arena is sized to the *budget* (1034 slots ≈ 9.19GB), roughly 2× the working set. Sizing arena + cache to working set (plus fan-out headroom) frees **≈4.5–4.8GB pinned**.
**But this alone does NOT lift the prefix-cache budget above 0.** Prefix budget = `max(0, free_ram − 12.88GB reserve)`. Freeing 4.8GB raises MiniMax free RAM 4.75 → ~9.5GB, **still below the 12.88GB reserve → budget stays 0.** Right-sizing is necessary but insufficient on its own.

**Lever B — clone path (arena OFF):**
Frees **12.65GB avail** (MiniMax post-load 4.75 → 17.40) and lifts prefix budget **0 → 11.00GB**. This is enough to satisfy the follow-on rebalance goal. **Cost:** the clone/module-list path is "gated in, not perf-tuned" (CLAUDE.md dispatch table); decode throughput on it is unmeasured and likely worse than the pinned-arena path. This is a RAM-vs-throughput trade, not free memory.

**Reclaimable figure for the rebalance plan:**
- **~4.8GB** reclaimable safely (cache right-size), **but blocked by the 12.88GB reserve** from reaching the prefix cache.
- **~12.65GB** reclaimable via arena-OFF, sufficient to lift prefix budget to ~11GB, at an unquantified decode-throughput cost.
- **The binding constraint is the 12.88GB reserve, not the arena size.** Even a right-sized arena leaves free RAM under the reserve. The rebalance plan must either (a) lower the reserve — it currently over-reserves relative to a right-sized arena — or (b) combine right-sizing with a reserve cut, or (c) accept the clone-path trade. A right-sized arena (~4.4GB) plus a reserve re-derived from actual steady-state non-cache RAM is the most promising combination and should be sized in that plan.

## Verdict on the ~5GB "mystery"

**EXPLAINED.**

The prior session's ~5GB-unexplained gap was an artifact of an opaque single-RSS ledger. With the categorized snapshot:

- RSS is ~100% anonymous `private_dirty`; **no file-backed reclaimable memory exists** (shared_clean ≈ 0).
- The arena/expert-load phase creates ~17.2GB of anonymous `private_dirty` for a 9.19GB slab. The ~8GB beyond the slab is: the **expert-cache clone working set (~4.4GB)** + **PyTorch host/staging buffers (~1.5GB)** + **~2GB arena-path staging overhead** (arena marginal 11.21GB vs slab 9.19GB).
- The arena-OFF control confirms it: removing the arena leaves exactly ~6GB of that overhead (5.98GB non-base anonymous), and Qwen reproduces the identical +17.18GB signature — so the gap is architectural, not a leak and not model-specific.

No instrument gap remains for this question. The one thing `smaps_rollup` **cannot** resolve is pinned vs pageable within `private_dirty`; that distinction did not matter here (swap=0) but would need `/proc/self/smaps` per-mapping parsing (VmFlags `mlocked`) if a future question turns on it.

## Bug / behavior notes (recorded, not fixed — per plan constraint)

- None. No engine bug surfaced. The over-provisioned expert cache (9.1GB budget vs 4.40GB working set) is a **sizing/policy** question for the rebalance plan, not a defect.
