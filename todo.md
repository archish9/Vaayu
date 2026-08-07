# TODO — Vaayu task ledger

Single source of truth for what is done and what is pending. **Both the user and Claude
add/update here, as work happens.** Protocol: `CLAUDE.md` §6.

Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

_Last updated: 2026-08-02 (Plan J on-device gates). Restructured 2026-08-03: completed plans (A, C, D, E, F, G, H, I) moved to the
archive at the bottom as one-line verdicts; only unfinished work appears above it. Full task-level
detail for those plans is in their RESULTS docs under `docs/superpowers/plans/` and in this file's
git history._

---

## ▶ START HERE — fresh session, 2026-08-05

**Plan J is CLOSED.** Code, tests and all four parked review bugs are on `main`; tree clean apart
from the untracked `command` scratch file; suite **576 passed / 20 warnings**. No worktree open,
nothing running.

**What Plan J delivered (all measured same-box A/B against `main`, not predicted):**

| | MiniMax | Qwen |
|---|---|---|
| prefix pool | 4.69 → **13.4 GB** | 4.96 GB — **unchanged** |
| round-2 TTFT, K=3 × 7 751 tok | 338.4 → **9.6 s (35×)**, uncompressed | n/a — pool unchanged |
| decode | 1.483 → 1.48 (**unchanged**) | 0.85 → **0.86**, 47 % hit, clamp skipped |
| `post-arena rss` | 21.59 → **12.91 GB** | 21.08 GB — **unchanged** |
| fan-out capacity | **K_max = floor(pool / 1.968 GB)**; first failing K = **7** | unchanged |

**The honest headline: J delivers on MiniMax and NOTHING on Qwen.** Qwen's 8.90 GB working set does
not fit an 8 GiB clamped cache and there is no allocation size between 8 and 16 GiB, so Task 5
correctly skips the clamp for it. Recovering Qwen's share is exactly what **Plan L** exists for.

**Two results from the K sweep that outlive Plan J:**
- **Never quote a bare K.** The pool budget is `free RAM at load − 3.76 GB reserve` and measured
  **11.49 / 12.16 / 12.87 / 13.70 / 13.76 GB across five loads.** Capacity is the rule, not the
  integer; the same box cliffs at a different K on a different day.
- **Overshooting capacity costs EVERYTHING, not a share.** K=7 gave **0** hits, not the 5 the pool
  could still hold — round 1's re-stores cascade-evict. 9.8 s → 352 s. Provisioning the pool to
  exactly the expected session count is provisioning it to fail.

**Also landed 2026-08-05, after J closed** — turning J's measurements into engine behaviour, no GPU:
prefix-pool capacity WARNING (`a7ef4b7`), `tune_advisor.py` units bug fixed (`3108342`) and pinned
by a regression guard (`138eef1`, `2b35573`). See "Applied findings" below. Suite **602**.

### Do this next — Plan L, evidence phase

**Run E4 and E1 first, in that order, and stop if either fails.** They are cheap fresh-process
allocation probes — minutes each, no model load, no GPU decode — and either one can refute the
whole plan before a line of design is written. E2/E3 are code reading. Only then write the plan.

**Do not start from Plan J's §3 Option-B sketch.** It is a starting point, not a spec, and its
central claim (that chunking preserves pinned H2D) is exactly what E4 exists to test.

Full detail, probe designs, gates and the already-measured target numbers are in the **Plan L**
section below. Read that section, not this summary.

### Where the evidence lives

`~/vaayu-evidence/2026-08-03-pinned-roundup/` — every log behind the numbers above:
`gen-{minimax,qwen}-after.log` (branch, pre-Task-5), `gen-{minimax,qwen}-MAIN-baseline.log`
(main, same box), `multiturn-K3-MAIN-baseline.log` (0/3 HIT, exit 1),
`multiturn-K3-BRANCH.log` (3/3 HIT, exit 0), `var-{main,branch}-{1,2,3}.log` (the variance runs).
Note the multiturn logs are 300-550 MB each — always `grep`, never `cat`.

### Method notes that cost real time to learn

- **Always A/B against a same-box baseline run the same day.** A cross-session decode floor sent one
  investigation down a wrong path for an hour. A baseline load costs ~12 min.
- **n=1 decode comparisons cannot resolve <~4 % on this box.** Repeat, alternating arms.
- **A VRAM OOM does not present as an OOM** — it surfaces as
  `MiniMaxM2DecoderLayer.forward() missing 1 required positional argument: 'position_embeddings'`,
  ~30 frames away. See the open bug below before debugging any such TypeError.

---

## Execution sequence — read this first

**Plan J is closed; L is the live row.** Execute strictly top to bottom. Each row's plan doc is
self-contained; do not start a row until the one above it is `[x]` and its RESULTS doc exists.

| # | Plan | Prio | Status | Blocked by | Why here |
|---|---|---|---|---|---|
| **1** | **J — pinned-allocation round-up** | P1 | `[x]` **CLOSED 2026-08-05** | — | Code, tests, both fix waves and all four parked bugs are on `main`; suite **576**. **Delivers on MiniMax only** — pool 4.69 → 13.4 GB, round-2 TTFT 338 → 9.6 s (**35×**, uncompressed, no quantization). Qwen gets **nothing** (working set does not fit the clamped cache; Task 5 correctly skips the clamp). Fan-out deliverable: **K_max = floor(pool / per-session KV)**, first failing K = 7 uncompressed, and 9-bit compression clears it (7/7). RESULTS: `2026-08-03-pinned-allocation-roundup-RESULTS.md`. |
| **2** | **L — chunked slab-arena backing (Option B)** | P1 | `[~]` **evidence DONE, plan WRITTEN — execute it** | — (J's RESULTS landed 2026-08-05) | **The only way Qwen gets Plan J's win.** J proved the effect is real and worth 35× TTFT; it also proved the single-tensor clamp cannot deliver it to a geometry whose working set sits between 8 and 16 GiB. Chunked backing keeps every slab *and* drops resident RAM. See its own section below — **gather evidence (E1-E5) before writing the plan.** |
| **3** | **K — batched-decode admission reclaim** | P2 | `[ ]` ready | nothing — J is closed | J changed free RAM, not free VRAM, so K was never *blocked* — it waited so its sweep would run against a settled memory layout. That layout is now settled. |
| **4** | **F1 — RAM budget arbitration** | P3 | `[!]` blocked | **L** (J satisfied) | ~8 GiB of the arena's footprint was allocator round-up, so rebalancing the expert-cache/prefix-pool split before J would have measured an imaginary scarcity. **The answer now differs per model** — MiniMax's pool is 13.4 GB and almost certainly no longer binding, Qwen's is unchanged at 4.96 GB and still is. Re-open only after L settles Qwen's footprint. No plan written yet. |

**Dependency rules that outlive this queue:**

- **J → L** — L exists only because J measured the effect and found the single-allocation clamp
  cannot deliver it to every geometry. Do not start L from theory; start from J's RESULTS.
- **J, L → F1** — never rebalance a RAM split while a consumer's real footprint is unknown. J fixed
  that for MiniMax; Qwen's stays unknown until L.
- **K's measurement is biased by F3** (the prefix cache is not wired into `batched_generate`), so K
  must report decode throughput separately from TTFT. See K's plan §8c. Fixing F3 first is *not*
  required and is a bigger job — see "Open bugs".
- **Historical, both satisfied:** E → H (KV double-storage before TurboQuant), F → G (the `top_n`
  sweep retired G's pin). Both closed 2026-08-02.

**If a plan ends in refutation, that closes the row.** Record the verdict under "Settled" and move
to the next row — do not re-scope the plan to find a win. Plans F, G and H all ended in refutation;
that is the system working.

---

## Plan J — pinned-allocation round-up  **(CLOSED 2026-08-05)**

`docs/superpowers/plans/2026-08-03-pinned-allocation-roundup.md` — self-contained, no prior context
needed. From the 2026-08-03 misplaced-feature audit, finding F0.

**Finding:** `torch.empty(..., pin_memory=True)` is served by PyTorch's caching host allocator,
which **rounds every pinned allocation up to the next power of two**. Measured in isolation:
5.00 GiB → 8.09, 8.50 → 16.00, **9.21 → 16.09 (the expert slab arena exactly as shipped)**; but
9 × 1 GiB separate tensors → 9.09 GiB (ratio 1.011). This explains the 17.18 GB rss rise that
appears identically on both models across the arena window and was never attributed.

**Waste ≈ 7.6 GiB on a 32 GB box** (arena 6.79 + `_PINNED_BUFS[0]` 0.76 + smaller), resident and
unaddressable. **Number it moves:** cross-turn prefix pool budget — predicted ~12.5 GB (bar ≥ 11),
**measured 4.69 → 13.38 GB on MiniMax**, with `post-arena rss` down 21.59 → 12.91 GB. Plan I had
measured that pool coming up ~1.4 GB short at a cost of 332 s of round-2 TTFT; J closes that gap
and the round-2 TTFT is now **9.6 s**.

**Plan rewritten to v2 (2026-08-03)** after a pre-flight scan found three defects in v1. Corrected
there, and these corrections supersede the numbers above: the clamp costs **6.7 %** of slabs, not
13 % (v1 confused GB with GiB — the arena is 9.20 GB = 8.571 GiB); **Task 2 frees ~0.12 GiB, not
0.79** (rounding a request up to the block already charged makes dead pages addressable, it does
not return them — the only real saving is dropping the `+100 MB` pad); and Task 2's chunked-buffer
option (b) is **infeasible**, not deferred (tensors are zero-copy views spanning the whole file, so
one would straddle a chunk). Allocator rule re-measured directly: next pow2, and an exact pow2 is
charged exactly.

- [x] **Task 1** — clamp the arena backing store to the power of two already being paid for
  (`3037810`). MiniMax 1031 → 965 slabs, Qwen 779 → 727. Option B (chunked backing) still deferred;
  only revisit if Task 4 shows the 6.7 % mattered.
  - [x] Tests: the clamp is a no-op iff nominal is already an exact power of two (v1's stated
    "no-op on a box that was not wasting anything" — the two conditions are the same set), plus
    tightness, and the `+ get_expert_pool_size()` staging slabs survive
  - [x] Clamp announced in the arena log line with slabs lost AND RAM freed (CLAUDE.md §5)
- [x] **Task 2** — `_PINNED_BUFS` alloc_size: round the request UP and use the whole block
  (`b4344f6`). ~0.12 GiB freed; the `+100 MB` pad is gone. Review also caught a latent
  off-by-`prefix_pad` in the reuse test, fixed in `f49dadd`: `_PINNED_SIZES < total_size` ignored
  the alignment padding the io_uring read actually writes, an unbounded raw-pointer write that
  power-of-two sizing moved onto a reachable boundary.
- [x] **Final whole-branch review + fix wave** (`07f929c`) — 2 Important. (a) the Task 1 budget trim
  set the cache budget to the WHOLE arena, driving the loader's staging reserve from
  `get_expert_pool_size()` slabs to **zero** (a plan defect, since corrected in the plan doc);
  (b) the clamp fired on pageable arenas too, where pow2 charging does not apply and the trade is
  strictly negative. Suite 519 → **559**, warnings flat at 20. Dense smoke ran on real GPU + model.
- [x] **Task 5** — **floor the clamp at the model's working set** (`d40bdff`, `cd7fdb1`). ADDED
  after Task 4's decode gate failed. Rule: clamp only when the clamped **cache** (excluding staging
  slabs) still covers `_moe_geometry["working_set_bytes"]`; skip when clamping would break a working
  set that currently fits; clamp anyway when it fits neither way (thrashes regardless, so take the
  RAM). `working_set_bytes=0` default is a true no-op. Both boundary edges pinned by test. Review
  caught the skip-note keying on `_pin` (intent) rather than the arena's actual pin state, so a
  pageable retry would have falsely reported a working-set skip — fixed. Suite **564**, warnings 20.
- [x] **Task 3** — freed RAM reaches the prefix pool. **PASSED BOTH MODELS**, measured:
  | | MiniMax | Qwen |
  |---|---|---|
  | arena | 1034 → **966** × 8.9MB, 9.19 → 8.59 GB | 778 → **725** × 11.8MB, 9.21 → 8.59 GB |
  | `post-arena avail` | 8.46 → **17.15 GB** | 7.43 → **16.91 GB** |
  | `post-arena rss` | 21.59 → **12.91 GB** | → **12.51 GB** |
  | **prefix pool** | 4.69 → **13.38 GB** | → **13.14 GB** |
  | H2D | **19.8 GB/s pinned** | **19.8 GB/s pinned** |
  0 OOM in both. *(These are pre-Task-5 numbers: with Task 5, Qwen no longer clamps, so Qwen's
  column reverts to main's values. Re-measure Qwen for the RESULTS doc.)*
- [x] **Task 4** — on-device gate + RESULTS. **COMPLETE 2026-08-05.** All same-box A/B against `main`:
  - **Pool gate, MiniMax: PASSES.** `test_multiturn.py` K=3 × 7 751 tok, Plan I's exact config.
    main **0/3 HIT, exit 1**; branch **3/3 HIT, exit 0**, round-2 TTFT **338.4 → 9.6 s (35×)**,
    peak rss 26.46 → 20.72 GB, 0 OOM. **Achieved with `PREFIX_COMPRESS=None`** — Plan I needed
    9-bit compression for its 34× because a 3.54 GB pool could not hold 3 bf16 sessions; J's pool
    holds them uncompressed, so the same win arrives with **no quantization and no quality question**.
  - [x] **Decode, Qwen: FAILED pre-Task-5, CONFIRMED FIXED by Task 5.** main **0.85 tok/s / 47 %
    expert-cache hit**; branch pre-Task-5 **0.43 tok/s / 0 % hit** (−49 %) with the engine's own
    `WARNING: working set 8.90GB exceeds cache budget 8.49GB — decode will thrash`. Cause: the
    clamp cut the budget 9.13 → 8.49 GB across Qwen's 8.90 GB working set. Task 5 skips the clamp
    for this geometry. **Confirming run on merged `main` (2026-08-05,
    `gen-qwen-task5-confirm.log`): decode 0.86 tok/s, expert_cache 90307hit/101453miss = 47 %,
    NO thrash warning, 0 OOM, `PCIe measured: 19.9 GB/s pinned H2D`.** The arena logs the skip
    explicitly: `778 x 11.8MB (pinned, 9.21GB resident) — clamp skipped: clamping to 725 slabs
    would cut the cache below the 8.90GB working set`. **Prefix pool 4.96 GB — unchanged**, which
    is the measured form of "J delivers nothing on Qwen". Two riders:
    - The cosmetic trim-line bug (first Pending bug below) is **confirmed live on this path**:
      `budget trimmed 9.13 -> 9.12 GB to match the clamped slab arena` printed on an arena that
      was explicitly NOT clamped, two lines under the skip notice.
    - TTFT was **27.5 s cold** here vs 4.21 s in the main baseline. Not a regression: every
      `.vaayu_kv` on disk was rejected with `different model/quant/schema` (Plan E's bf16 disk
      layout changed the schema), so the restore that made the baseline 4.21 s could not fire.
      Irrelevant to this gate, but do not quote the two TTFTs against each other.
  - [x] **Decode, MiniMax: PASSES — the −2.7 % was noise.** A single pair (main 1.48 / branch 1.44)
    suggested a regression; alternating repeat runs refuted it:
    | | run 1 | run 2 | run 3 | mean |
    |---|---|---|---|---|
    | main | 1.47 | 1.50 | 1.48 | **1.483** |
    | branch | 1.51 | 1.45 | *(cut short)* | **1.48** |
    Branch run 1 (1.51) beats every main run and the arms overlap completely, so 1.44 was the low
    end of ordinary variance. Expert-cache hit rate is stably **6 % → 4 %** — the cache really does
    shrink, it just does not move decode on this geometry. **Lesson worth keeping: n=1 decode
    comparisons on this box cannot resolve differences below ~4 %.**
  - [~] **K-cliff sweep** (MiniMax, Plan I shape: `MAX_LENGTH=16384`, `range(160)` → 7 751 tok/session,
    `PREFIX_COMPRESS=None`, no `max_seq_len`). **Sweep is size-targeted, not walked one K at a time.**
    Per-session pooled KV derived from config (62 layers × 2 × 8 kv-heads × 128 head_dim × 2 B bf16)
    = **0.254 MB/tok = 1.968 GB per 7 751-tok session**; measured peak_rss grows **2.03 GB/session**,
    confirming it. Pool holds `floor(budget / 1.968)`. The model reproduces both known points
    (main pool 4.69 GB holds 2 → K=3 fails; merged pool ~13.7 GB holds 6), so K=5 is skipped as a
    predictable pass — K=6 (expect pass) and K=7 (expect fail) pin the cliff. Back-fill K=5 only if
    K=6 fails.
    | K | pool | HIT | round-2 TTFT | peak_rss | exit |
    |---|---|---|---|---|---|
    | 3 | 13.76 GB | 3/3 | 338 → **9.6 s** | 20.72 GB | 0 |
    | 4 | 13.70 GB | 4/4 | 340 → **9.6 s** | 22.78 GB | 0 |
    | 6 | 12.16 GB | 6/6 | 350 → **9.8 s** | 25.31 GB | 0 |
    | **7** | 11.49 GB | **0/7** | 352 → **352 s (no benefit)** | 26.99 GB | **1** |
    **First failing K = 7.** 0 OOM at every K including the failing one.
    Two things the RESULTS doc must carry, both more useful than the bare integer:
    - **The pool budget is not constant — 11.49 / 12.16 / 13.70 / 13.76 GB across four loads.**
      It is `free RAM at load − 3.76 GB reserve`, and free RAM at load varies. So the deliverable is
      the capacity rule **K_max = floor(pool_budget / per-session KV)**, not "6". At the K=7 load's
      11.49 GB pool even 5 sessions was the true capacity; at 13.76 GB it is 6. A box with less free
      RAM at load cliffs earlier, and the same box cliffs at different K on different days.
    - **The failure is total, not partial: 0 hits, not the 5 the pool could still hold.** Round 0
      already logged 2 evictions; in round 1 each recomputed session re-stores and cascade-evicts
      the rest, so every one of the 7 misses and round-2 TTFT stays at the cold 352 s. This is
      Plan I's **"capacity is a cliff, not a slope"** reproduced exactly — one session past capacity
      does not cost 1/7th of the win, it costs all of it (9.8 s → 352 s, **36×**).
  - [x] **9-bit compressed repeat at K=7 — PASSES**, clearing the cliff it had just hit:
    | K | compress | pool | HIT | round-2 TTFT | peak rss | exit |
    |---|---|---|---|---|---|---|
    | 7 | none | 11.49 GB | **0/7** | 352 → 352 s | 26.99 GB | 1 |
    | 7 | **9-bit** | 12.87 GB | **7/7** | 354 → **10.4 s** | **22.57 GB** | 0 |
    Per-session pooled KV 2.03 → **~1.12 GB** (0.55× bf16, matching the ~0.56× 9 bits predicts),
    peak rss −4.42 GB at equal K, cost **~0.7 s of TTFT** unpacking at restore. **K=6-compressed
    was deliberately skipped** (user call): K=7 carries the result since it is the K that failed
    uncompressed, and K=6 would have been a ~50 min confirmation of an implied pass. **The
    compressed cliff was therefore NOT located** — above 7, predicted near 11 by arithmetic only.
    Label it as arithmetic wherever quoted.
  - [x] **RESULTS doc written**: `docs/superpowers/plans/2026-08-03-pinned-allocation-roundup-RESULTS.md`
    (docs/ is gitignored, so it lives on disk only, as every prior RESULTS doc does).
  - [x] Zero-OOM sweep across all five runs of this session: clean, no host OOM-kill. The 296
    `out of memory` hits in `multiturn-K3-nocompress.log` are the **Aug-2 aborted run** at ~8.2k
    context (the documented VRAM cliff), not part of this sweep.
  - [x] Harness scaffolding reverted (`git checkout test_multiturn.py`).
  Runbook: `docs/superpowers/plans/2026-08-03-pinned-roundup-RUNBOOK.md`.

**Pending bugs / parked findings from Plan J review — ALL FOUR FIXED 2026-08-05.**
Fixed while K=7 was running; none of them blocked the on-device gates, and none touch the
measurement path (the running process had already imported the old code).
- [x] `vaayu_base.py` trim line printed "to match the **clamped** slab arena" on an arena that was
  not clamped. **Confirmed live** on the Qwen confirming run, two lines under `clamp skipped: ...`.
  Reworded to "to match the slab arena's cache share" — the trim fires whenever the arena's cache
  share is under the requested budget, which includes the unclamped case where the staging addend
  alone accounts for the gap. **No test: pure output string.**
- [x] Degenerate branch: trimmed budget 0 → expert RAM cache fully disabled **silently**. Now emits
  a WARNING naming the arena's slab count and the staging reserve that consumed it (CLAUDE.md §5).
  `put_slab` already handled a 0 budget safely — what was missing was telling the operator.
  Boundary pinned by test (`== 0` at and below `pool_size` slabs, one live slab immediately above);
  **the print itself is not asserted.** Still unreachable on any box that loads these models.
- [x] Tests asserted against test-local mirrors of the call-site expressions. **Fixed properly, as
  the finding prescribed**: extracted `arena_slab_count(..., pin=)` (the pageable/clamp gate + the
  working-set floor) and `cache_budget_for_arena(n_slabs, slab_bytes, pool_size)` (the staging
  reserve) into `vaayu_base.py`, and the tests now call those directly. Mirrors `_trimmed_budget`
  and `_n_slabs_for` are gone, so a regression in the production expressions now fails the suite.
  Two tests added for the pageable path ignoring the working-set floor and the pinned path honouring
  it — the interaction the old mirror could not express.
- [x] `load_non_expert_tensors` used `alloc_size = aligned_length + 4 MiB`, a non-power-of-two
  written into the same shared `_PINNED_SIZES` slots Task 2 sizes by powers of two. Now
  `next_pow2_at_least(aligned_length)`. **No slack added on top, deliberately**: unlike the expert
  path's `total_size`, `aligned_length` is already `_align_up(read_len + prefix_pad)` — exactly the
  byte count io_uring writes — so `pinned_alloc_size`'s `2 * _ALIGN` O_DIRECT slack would be wrong
  here, not merely wasteful. Never a larger charge than the old pad, and strictly smaller when
  `aligned_length` is an exact power of two (the pad used to push it a whole block up). Verified
  the other `_PINNED_SIZES` consumer — the io_uring buffer registration at
  `safetensor_model_persister.py:318-319` — still receives the true allocated size from both sites.

  Verification: `tests/test_slab_arena.py` + `tests/test_pinned_staging_alloc.py` **74 passed**.
  Full-suite re-run deferred until K=7 lands — a pinned-memory test competing for RAM would move
  the prefix-pool budget that run is measuring.

## Applied findings — turning Plan J's measurements into engine behaviour

Recorded because the measurements alone were not the deliverable. Plan J's Task 4 defined its
output as "record the first failing K"; **a number in a RESULTS doc is not a deliverable for an
inference engine**, and executing that definition faithfully instead of challenging it left ~40 %
of four GPU-hours sitting in a document. Both items below were built afterwards, with no GPU.

- [x] **Prefix-pool capacity WARNING** (`vaayu/prefix_cache.py`, `_warn_over_capacity`).
  `compute_expert_budget` warns when the expert working set exceeds its budget ("decode will
  thrash") — that warning is what caught the Qwen regression. **The prefix pool had no equivalent**:
  it evicted in silence, hit rate went to 0, TTFT went 9.8 s → 352 s, and nothing was printed. The
  silent degradation CLAUDE.md §5 forbids, in the subsystem holding the engine's largest TTFT lever.
  Fires on the first budget-pressure eviction in `store()` (unambiguous — that loop only runs over
  budget), latched to once per store lifetime (the agentic workload re-stores every session every
  turn; per-eviction would emit hundreds of lines). Every figure derived from the pool's own state,
  no model constant. Names the remedy, and does NOT recommend compression when already compressed.
  10 tests, parametrized over both geometries. **Deliberately not auto-configured**: the engine does
  not know K or tokens/session at load time — sessions arrive later — so runtime detection is the
  only correct place, and `K=7` is not a constant to hardcode (pool varies 11.49–13.76 GB same box
  same day; Qwen is 4.96 GB).
- [x] **`tune_advisor.py` unit bug** (`3108342`). It instructs you to paste the pool budget off the
  engine's startup line, which prints `/1e9` (decimal GB), then multiplied by `1024**3` — inflating
  it 7.4 %. Enough to cross a capacity boundary: at K=6 / 11.49 GB pool it reported room for 6 and
  recommended `prefix_compress=None`, where only 5 fit — the exact config that thrashes to 0 hits.
  **Found by validating the advisor against the measurements**, which had never been done; now
  reproduces all four measured points (K=3/4/6 → `None`, all PASSED; K=7 → `9`, and 7 failed
  uncompressed / passed compressed). Its "3.5–5.4 GB across six runs" guidance was pre-Plan-J and
  understated MiniMax ~2.5×.
- [x] **Regression guard on `tune_advisor.py`** — `tests/test_tune_advisor.py`, 16 tests.
  Extracted `pool_capacity()` and `recommend_compress_bits()` from the print-entangled `main()`
  (the same extract-don't-mirror move that fixed parked finding 3), so the tests exercise the
  production path; `main()` now calls both, and its end-to-end output is unchanged. The guard pins
  **all four on-device points from Task 4** (K=3/13.76, K=4/13.70, K=6/12.16 → `None`, all of which
  PASSED uncompressed; K=7/11.49 → `9`, which FAILED uncompressed and PASSED compressed), the
  `GB == 1e9` convention, per-session size derived from geometry rather than pasted, and capacity
  across every measured budget including Qwen's 4.96 GB. **Verified it fails under the bug it
  guards**: reinstating `GB = 1024 ** 3` fails 5 of the 16. Suite 586 → **602**, warnings flat.

## Plan L — chunked slab-arena backing (Option B)  **(P1 — evidence, THEN plan)**

**No plan doc yet, and do not write one from this section.** Gather the evidence below first; the
plan must be self-contained and built on measurements, the way J's v2 was. J's §3 already sketches
Option B — that sketch is a starting point, not a spec, and one of its claims is already known
shaky (see E4).

**Why it exists:** Plan J proved the pinned-allocation round-up is real and that removing it is
worth **35× round-2 TTFT**. It also proved the single-tensor power-of-two clamp **cannot deliver
that to Qwen**: Qwen's 8.90 GB working set does not fit an 8 GiB clamped cache, and there is no
allocation size between 8 and 16 GiB — the allocator charges the next power of two, so anything
above 8 GiB costs 16. Clamping Qwen cost **49 % of decode** (0.85 → 0.43 tok/s, expert RAM cache
47 % → 0 % hit). Plan J Task 5 therefore skips the clamp for Qwen, correctly, and Qwen keeps the
~8.6 GB of dead resident RAM.

**The idea:** allocate the arena as *k* power-of-two-sized pinned tensors instead of one. Measured
in the audit: 9 × 1 GiB costs 9.09 GiB resident (ratio **1.011**) where one 9.21 GB request costs
16 GiB. That keeps **every** slab *and* drops resident RAM — both models win, and Task 5's
working-set floor would stop firing for Qwen because the cache never shrinks.

**Prize, if it works:** Qwen's prefix pool **4.96 → ~13.1 GB** (measured under the clamp), i.e.
**3 → 8 concurrent sessions**, plus its 47 % expert-cache hit rate preserved. This is the larger of
the two models and the one whose expert cache actually works well.
**Honest caveat: Qwen's multiturn TTFT has never been measured.** MiniMax's 35× comes from
`test_multiturn.py` runs; Qwen has only ever been run single-turn through `test_generate.py`. The
pool and session-count gains are on record, the *TTFT multiplier for Qwen is an extrapolation* —
say so in the plan, and measure it in L's own gate rather than inheriting MiniMax's number.

**The risk, stated up front:** `SlabArena.slab_index_of()` currently does pointer arithmetic against
a single `_backing` (`off = tensor.data_ptr() - self._backing.data_ptr()`). Chunking makes it a
search over per-chunk `(base_ptr, first_slab_idx, n_slabs)` ranges — a change to a function **on the
eviction path**, whose failure mode is returning a wrong slab index and freeing a slab another
expert is still reading. That is silent memory corruption, not a crash. This is the single reason
Option B was deferred out of J, and it has not become less true.

### What L delivers, in already-measured numbers

**Task 3 ran Qwen WITH the clamp, before Task 5 disabled it — so the RAM outcome L is chasing is
not a prediction, it is on record.** What that run also showed is the cost L must avoid:

| Qwen, 7 751-tok sessions | today (`main`) | with the RAM freed (measured, clamped) |
|---|---|---|
| `post-arena rss` | 21.08 GB | **12.51 GB** |
| prefix pool | 4.96 GB | **13.14 GB** |
| concurrent sessions | **3** | **8** |
| expert-cache hit | **47 %** | 0 % ← what the clamp destroyed |
| decode | **0.86 tok/s** | 0.43 ← −49 % |

Per-session pooled KV, Qwen: 94 layers × 2 × 4 kv-heads × 128 head_dim × 2 B = 0.193 MB/tok =
**1.492 GB/session** at 7 751 tok. (Qwen is the SMALLER per-session geometry despite more layers —
half the KV heads dominates. MiniMax is 1.968 GB.)

**So L's deliverable is precisely: that same RAM, at zero expert-cache cost.** 3 → 8 concurrent
sessions on the larger model, plus its 47 % expert-cache hit intact. Secondary: MiniMax recovers
the 6.7 % of slabs it currently forfeits to the clamp.

**Why L and not F1.** F1 (rebalance the expert-cache/prefix-pool split) is zero-sum, and Qwen's
expert cache is the *effective* one at 47 % — trading it away for pool is trading a known good for
a hoped-for one. L creates ~8.6 GB out of allocator waste. That is why F1 sits behind L.

### Code surfaces — already read, do not re-derive

`SlabArena`, `vaayu/persist/expert_cache.py`. **Only TWO methods touch `_backing`:**

- `tensor(idx)` → `self._backing[idx*slab_bytes:(idx+1)*slab_bytes]`, a zero-copy view. The loader
  takes `.data_ptr()` off this and hands it to io_uring with a length.
- `slab_index_of(tensor)` → `off = tensor.data_ptr() - self._backing.data_ptr()`, then
  `off // slab_bytes`. **This is the eviction-path function and the corruption risk**: a wrong
  index frees a slab another expert is still reading, silently.

`acquire()` / `release()` / `record_read()` / `_pending_read` are **index-based and chunk-agnostic** —
they need no change. That materially shrinks L's blast radius versus what J's §3 assumed.

**The design tension, found by reading the code — resolve it in the plan, don't rediscover it.**
A chunk must be a power of two (or the round-up returns) AND a slab must not straddle a chunk
boundary (or `tensor()` hands io_uring a span crossing two allocations). `slab_bytes` is
`_ALIGN`-aligned but NOT a power of two (8.9 MB MiniMax / 11.8 MB Qwen), so a pow2 chunk is
generally not a whole multiple of it. **Resolution: keep chunks pow2 and simply leave each chunk's
tail remainder unused** — waste is `chunk_bytes % slab_bytes` per chunk, bounded by
`k × slab_bytes` (≈106 MB at k=9, Qwen). Small, but E1 must MEASURE it rather than assume it.

### Evidence to gather BEFORE writing the plan

**Order: E4 → E1 → E2 → E3 → E5.** E4 and E1 are the kill switches and cost minutes; run them
before anything else. Both need the box quiet (they allocate 9-16 GB pinned) — no model loaded,
no other run in flight.

- [x] **E4 — PASSED 2026-08-07. Chunking does NOT cost pinned H2D.** The likeliest refutation
  did not refute. Probe: `~/vaayu-evidence/2026-08-07-chunked-arena/probe_e4_chunked_h2d.py`,
  log `e4.log`. Fresh process per arm, no model load. Baseline = single 8 GiB pinned tensor
  (19.6 GB/s burst / 19.5 slab).
  | arm | k | charged | burst | vs base | slab-granularity | vs base | worst single chunk |
  |---|---|---|---|---|---|---|---|
  | mixed pow2 | 2 | 9.66 GB | 19.7 | 101% | 19.8 | 102% | 19.5 |
  | uniform 4GiB | 3 | 12.88 GB | 20.0 | 102% | 19.9 | 102% | 19.4 |
  | uniform 2GiB | 5 | 10.74 GB | 20.0 | 102% | 19.9 | 102% | 19.3 |
  | uniform 1GiB | 9 | 9.66 GB | 20.0 | 102% | 20.0 | 103% | 19.4 |
  | uniform 512MiB | 18 | 9.66 GB | 20.0 | 102% | 19.9 | 102% | 19.2 |
  | uniform 256MiB | 36 | 9.66 GB | 19.9 | 102% | 20.0 | 102% | 19.2 |
  **Every chunk `is_pinned()` True; worst of 111 chunks = 19.2 GB/s against a 15.0 GB/s floor.**
  Measured at BOTH granularities: 256 MiB bursts (comparable to the engine's own
  `PCIe measured:` line) and real `slab_bytes` copies (11.84 MB, the per-expert access pattern).
  **Allocation cost is flat in k** — 1.5 s at k=1 vs 1.7 s at k=36, so no `cudaHostAlloc`
  lock-down penalty either. **k is free**: whatever k a future geometry derives costs nothing,
  so the design need not minimise it.
- [x] **NEW FACT, not previously in this ledger: `torch._C._host_emptyCache()` DOES return pinned
  memory to the OS.** Measured in E4's free probe on a k=2 / 9.66 GB arena: after `del`,
  rss **+0.00 GB** (confirming the Settled "cached, not returned" rule); after
  `torch._C._host_emptyCache()`, rss **−8.59 GB**. So the Settled entry is right about `del` but
  incomplete — torch 2.11.0+cu130 exposes an explicit release. Potentially useful to Vaayu
  itself (arena teardown / retry paths). Not yet wired anywhere.
- [x] **E1 — PASSED 2026-08-07, but only with a better decomposition. Chunking recovers
  7.93 GB resident while keeping EVERY slab.** Probe:
  `~/vaayu-evidence/2026-08-07-chunked-arena/probe_e1_chunked_residency.py`, log `e1.log`;
  decomposition comparison in `recursive_check.py`. Fresh process per arm, no model load.
  **The winning shape is `[8 GiB, 512 MiB, 16 MiB]`, k=3, IDENTICAL on both models:**
  | | MiniMax | Qwen |
  |---|---|---|
  | today, single tensor | 9.13 GB nominal → **17.18 GB charged** | same |
  | recursive k=3 | **9.14 GB charged, 9.245 GB resident** | same bytes |
  | usable slabs | **1027, exactly the unclamped need** | **771, exactly the need** |
  | tail waste | 15 MB | 14 MB |
  | **resident saving** | **−7.93 GB** | **−7.93 GB** |
  All chunks `is_pinned()`. E4 already priced k=3 at **20.0 GB/s, 102 % of baseline** — so this
  costs nothing in bandwidth.
  **Two things the first run got wrong, recorded so the plan does not inherit them:**
  - **The pre-stated bar was set against CHARGED bytes and missed a flat +0.107 GB of residency
    overhead** (page tables + allocator bookkeeping for pinned pages). It is constant across every
    config and both models, so it discriminates between nothing — but it made the first pass report
    1/12. **The bar was NOT softened**; the decomposition was fixed instead.
  - **A one-halving greedy is not good enough.** It jumped 512 MiB → 1 GiB at the tail and charged
    **9.66 GB for 815 slabs when 771 were needed** — 0.47 GB wasted by the helper, not by chunking.
    Recursing on the REMAINDER lands on 9.14 GB with slabs exactly equal to need.
    **The plan must specify the recursive decomposition, with a floor (64 MiB used here) so k stays
    small.** Uniform-chunk configs are all strictly worse and are not candidates.
  **Expert-size sensitivity (arithmetic, no allocation): k stays 2-3 from 8.89 MB to 1 GB slabs**,
  because the arena's byte size is bounded by the RAM budget, not by model size. A larger model does
  not raise k on this box; a larger-RAM box raises it as log2 of the arena (~25 GB arena → k≈4).
  Bounded near 5 for any plausible machine, and E4 measured k=36 at full bandwidth anyway.
- [x] **E2 — DONE 2026-08-07. Blast radius confirmed at TWO methods; no consumer assumes arena
  contiguity.** Verified by grep over `vaayu/`, `vaayu_rust/` and `tests/`, not trusted from the
  earlier pre-read.
  **The only base-pointer arithmetic against the arena is inside `SlabArena` itself:**
  `tensor(idx)` (`expert_cache.py:64`) and `slab_index_of(tensor)` (`:82-86`). Everything else is
  index-based (`acquire`/`release`/`record_read`/`_pending_read`) or slab-relative.
  **Why the loader is safe:** every io_uring destination is `slab[slab_off : slab_off +
  raw_aligned_length]` where `slab = arena.tensor(idx)`, and `load_expert_into_slab` raises
  `RuntimeError` if `slab_off + slab_span > slab.nbytes`. **Spans are slab-relative and hard-bounded
  by one slab**, so the loader can never hand io_uring a range crossing a chunk boundary. Same for
  `views_from_packed_copy` (`persister:934-937`) and the `_used` computation
  (`moe_patch:1628-1629`) — both offset from the slab, not the arena.
  **The biggest unknown came back clean: the arena is NEVER registered with io_uring.**
  `reregister_buffers` (`persister:317`) registers `_PINNED_BUFS[0]/[1]` — the staging pool — only.
  Nothing depends on arena contiguity at the kernel level.
  `arena.tensor()` production call sites: `moe_patch.py:1269, 1399, 1627`, `vaayu_base.py:926`.
  **CORRECTION to this ledger: `slab_index_of` is NOT on the eviction path.** Eviction uses the
  `slab_idx` already stored in the entry (`_account_evict`, `expert_cache.py:308-315`).
  `slab_index_of` has exactly **ONE production caller** — `moe_patch.py:1658`, the `record_read`
  H2D read-guard in the module-list forward — plus one test. Its failure mode is still real
  (a wrong index attaches the read-guard event to the wrong slab, so the correct slab can be reused
  while an H2D read is in flight → torn read), but the surface is one call site, not a hot path.
- [x] **E3 — answered by E2, no probe needed.** The premise ("it is on the eviction path",
  "MiniMax ~76 k/run, Qwen ~190 k/run") was wrong: eviction never calls it. At k=3 a linear scan
  over three `(base_ptr, first_idx, n_slabs)` ranges is free at the one call site that exists.
  No `bisect` needed. Revisit only if a future geometry derives a large k — and E4 measured k=36 at
  102 % bandwidth, so a large k is viable if it ever arrives.
- [x] **NEW REQUIREMENT found by E2, not in the original plan sketch: chunk bases must be
  `_ALIGN`(4096)-aligned.** `slab_bytes` is `_ALIGN`-aligned so slab starts stay sector-aligned for
  O_DIRECT — but that only holds relative to the chunk base, so each chunk base must be aligned too
  or O_DIRECT reads break. **Measured: pinned allocations come back 2 MiB-aligned** (8 GiB, 512 MiB
  and 16 MiB chunks all `data_ptr() % 2MiB == 0`, torch 2.11.0+cu130), so the property holds with
  huge margin. **The plan must ASSERT it at construction rather than assume it.**
  Also: `SlabArena`'s docstring claim *"Allocation is ONE backing tensor (a single cudaHostAlloc,
  not thousands)"* must be updated.
- [x] **E5 — DECIDED 2026-08-07 (user call): keep the successful chunks if they cover the working
  set; otherwise free them and fall back pageable.** Chosen over three alternatives because it
  reuses **Task 5's already-tested working-set floor** rather than inventing a new rule.
  ```
  chunks = []
  for nbytes in decomposition:            # largest first: [8 GiB, 512 MiB, 16 MiB]
      try:    chunks.append(pinned(nbytes))
      except RuntimeError:  break         # keep what we have
  n = sum(c // slab_bytes for c in chunks)
  if cache_budget_for_arena(n, slab_bytes, pool_size) >= working_set_bytes:
      WARN(...); use(chunks)              # degraded but still pinned, 19.8 GB/s
  else:
      free(chunks); WARN(...); fall_back_pageable()   # today's path, ~7.1 GB/s
  ```
  **Why this shape:** the decomposition is largest-first, so *which* chunk failed carries most of
  the information — chunk 0 is ~88 % of capacity (its failure covers nothing and correctly falls
  back), while a tail-chunk failure keeps ~99 % of capacity **and** pinned bandwidth. Falling back
  to pageable costs **19.8 → ~7.1 GB/s on the whole expert path**, a far bigger loss than a slightly
  smaller cache, so discarding 8.5 GiB of secured pinned memory over a 16 MiB shortfall would be
  the wrong trade.
  **Compare against the CACHE share, not the raw slab count** — `cache_budget_for_arena(n,
  slab_bytes, pool_size)`, excluding the staging reserve, exactly as Task 5 does.
  Rejected: (a) free-all-and-fall-back — simplest but pays the 2.8× cliff for a 0.2 % shortfall;
  (b) retry-remainder-with-smaller-chunks — recovers most capacity and E4 proved small chunks are
  free, but adds a retry loop and termination floor for a case that may never fire (**keep as a
  documented future option if partial failure is ever observed**); (c) keep-whatever-succeeded
  unconditionally — reintroduces exactly the Qwen thrash Task 5 exists to prevent (0.85 → 0.43
  tok/s, 47 % → 0 % hit).
  Both branches WARN per CLAUDE.md §5, and the cache budget is re-derived from the ACTUAL slab
  count rather than the requested one.

**PLAN WRITTEN 2026-08-07:** `docs/superpowers/plans/2026-08-07-chunked-slab-arena-plan.md`
— 7 tasks, TDD, self-contained. Tasks 1-6 are code+tests with no GPU; Task 7 is the
on-device gate. Execute with `superpowers:subagent-driven-development`.

**EVIDENCE PHASE COMPLETE 2026-08-07 — E1, E2, E3, E4, E5 all closed. Plan L is NOT refuted.
Write the plan next.** Headline: the winning design is a **recursive** power-of-two decomposition,
`[8 GiB, 512 MiB, 16 MiB]` (k=3, identical on both models), giving **9.245 GB resident vs today's
17.18 GB (−7.93 GB)** with **every slab kept** and **102 % of baseline H2D**. Blast radius is two
`SlabArena` methods and one `slab_index_of` caller.

### Gates the plan must pre-state (do not soften them later)

- **Qwen decode ≥ 0.85 tok/s** (same-box main baseline, measured 2026-08-02 — not the older 0.82
  floor) **and expert-cache hit rate ≥ 47 %.** Losing the cache is exactly what J's clamp did.
- **MiniMax decode ≥ 1.48** (same-box baseline) — L must not regress what J already delivers.
- **H2D still reports `pinned` at ~19.8 GB/s on both models.**
- **Qwen prefix pool ≥ 11 GB**, the same bar J used. Confirmed reachable: 13.14 GB measured under
  the clamp.
- **Qwen decode measured on the SAME DAY as the branch run** (`test_generate.py`), not quoted from
  2026-08-02. Same-box, same-day A/B is not optional — see Method notes.
- **Zero OOM, suite green (baseline 602 as of 2026-08-05), and a test that proves `slab_index_of`
  returns the right index for a slab in every chunk, including the first and last slab of each** —
  plus the boundary cases the single-`_backing` version could not have: a pointer in a chunk's
  unused tail remainder must return `None`, not a bogus index, and a pointer below the first chunk
  or above the last must return `None`.
- **A Qwen multiturn run** (`test_multiturn.py`, MiniMax's harness pointed at Qwen) to measure the
  TTFT win directly rather than inheriting MiniMax's 35×. At 1.492 GB/session the K=3 shape needs
  4.5 GB — under today's 4.96 GB pool, so **main will pass K=3** and the gate must use a higher K
  (K=4 needs 5.97 GB > 4.96, so K=4 is the discriminating shape: main fails, L should pass).

**A refutation is an acceptable outcome.** If E1-E4 show chunking costs pinning, or the corruption
risk cannot be tested down, record the verdict and close the row — the same way F, G and H closed.
**If the row closes, go straight to Plan K** (row 3): it is unblocked, self-contained, and has a
measured 1.28× aggregate sitting in it.

### Method notes — these cost real time to learn, do not relearn them

- **Always A/B against a same-box baseline run the same day.** A cross-session decode floor sent one
  investigation down a wrong path for an hour. A baseline load costs ~12 min.
- **n=1 decode comparisons cannot resolve <~4 % on this box.** Repeat, alternating arms.
- **Derive the sweep instead of walking it.** Computing per-session KV from `config.json` (30
  seconds) turned a five-run sweep into three and predicted the cliff before any of it ran. One
  redundant run in Plan J cost ~35 min of GPU purely because that arithmetic came after the launch.
- **A VRAM OOM does not present as an OOM** — it surfaces as
  `MiniMaxM2DecoderLayer.forward() missing 1 required positional argument: 'position_embeddings'`,
  ~30 frames away, because both RoPE precompute paths OOM and the "graceful" fallback they
  advertise does not exist. See the open bug below before debugging any such TypeError.
- **Run logs are 300 MB - 1.3 GB.** Always `grep -a`, never `cat`, and never grep them while
  another run is streaming experts off the same NVMe.
- **NEVER run multiple pinned-allocation arms in ONE process, and cap every probe's cgroup.**
  Freed pinned blocks are cached, not returned (Settled), so `del` frees nothing at the OS level
  and `torch.cuda.empty_cache()` is device memory. E4's first version ran 8 arms sequentially in
  one process; they accumulated to a **28.3 GB peak** and **systemd-oomd killed the whole terminal
  cgroup scope — 7 processes, four times**, because oomd kills by *scope*, not by process
  (`oom_score_adj` is therefore useless). The working pattern, now in the probe: **one arm per
  subprocess** (process exit is the only reliable release), each spawned under
  `systemd-run --user --scope -p MemoryMax=<charge+2GB> -p MemorySwapMax=0` so a runaway arm dies
  alone, plus a driver-side ceiling of `min(40% total RAM, available − 4 GB)`. Verified by
  deliberately over-allocating against the cap: the arm died, the terminal lived.
  Applies to **any** future pinned-memory probe on this box.
- **A measurement that only lands in a doc is not a deliverable.** Plan J's Task 4 defined its
  output as "record the first failing K"; the number sat inert until it was turned into a runtime
  warning and an advisor fix — and *using* it immediately exposed a 7.4 % bug in the tool meant to
  apply it. **Decide what the engine will DO with a number before spending GPU hours on it.**

### Evidence lives here

`~/vaayu-evidence/2026-08-03-pinned-roundup/` — Plan J's full set, including
`gen-qwen-task5-confirm.log` (the clamp-skip confirming run) and
`multiturn-K{3,4,6,7}-nocompress.log` + `multiturn-K7-compress9.log` (the whole cliff sweep).
Put Plan L's probes in `~/vaayu-evidence/<date>-chunked-arena/`.

## Plan K — batched-decode admission reclaim  **(P2 — do third)**

`docs/superpowers/plans/2026-08-03-batch-admission-reclaim.md` — self-contained. Audit finding F2.

**Finding:** `free_vram_budget()` charges a batch against free VRAM while **5.82 GB (Qwen) /
8.7 GB (MiniMax)** sits in a pure LRU that already has an eviction entry point
(`ExpertVRAMCache.evict_until_free`). At 7 751-token streams that is **N=1 today vs N=5 if counted**.
N=3 measured **1.28× aggregate** (0.631 → 0.809 tok/s, Qwen), and the win is amortised non-expert
weight movement, not expert sharing (union ratio 0.82).

**The eviction cost has never been priced. That is the experiment, and a refutation is an
acceptable outcome** — Plans F, G and H all ended in refutation and cost far more to reach it.

- [ ] **Task 1** — `reclaimable_vram_budget(model, reclaim=False)`; floor = one token's working set
  (derived), so a batch can never be admitted onto a zero-resident cache
- [ ] **Task 2** — `batched_generate(reclaim=)` performs the reclaim it was admitted under, **once
  at the batch boundary, never per decode step** (pinned by test — Plan H's mistake in a new costume)
- [ ] **Task 3** — on-device: (a) does `batched_generate` run at all on MiniMax's module-list
  forward? Never exercised. (b) sweep N × reclaim at ~7.7k context, both models. Measure decode
  separately from TTFT (the batch arm gets no prefix HITs — see plan §8c, and F3 below)
- [ ] **Task 4** — RESULTS + verdict + the expert-VRAM hit-rate-vs-reclaim cost curve; correct the
  stale "always defers to N=1" claim in `test_batched_decode.py` and `.superpowers/sdd/task-7-report.md`

---

## Deferred — written up, deliberately not being built

- [!] **F1 — expert-cache / prefix-pool RAM arbitration is decided by line order.**
  **This is row 3 of the execution sequence** — it is the only deferred item with a place in the
  queue; everything else below has no scheduled slot.
  `compute_expert_budget` runs at `vaayu_base.py:1331`, sees 25.24 GB free, takes 9.19 GB.
  `compute_prefix_cache_budget` runs ~250 lines later at `:1585` and gets the remainder, 3.88 GB.
  Nothing arbitrates. Measured payoff per side: expert RAM cache **5–7 % hit rate** vs prefix pool
  **a cliff worth up to 34× TTFT**. **BLOCKED on Plan J** — ~8 GiB of the arena's footprint is
  allocator round-up, so rebalancing now would measure an imaginary scarcity. Re-open only if the
  pool is still binding after J. (audit F1, 2026-08-03)
- [ ] **Plan B Tasks 8, 9 — Phase 2 live expert re-pin.** Untouched. The Phase 1 pin it builds on is
  refuted (see "Settled" below), so this needs a new premise before it means anything.
  `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- [ ] **Plan H follow-up — real-query quality gate.** Plan I's gate used K-row and Gaussian query
  proxies because `.vaayu_kv` stores no queries. Both missed the bar widely so the verdict holds,
  but certifying a specific bit-width needs real queries captured with an attention hook.
- [ ] **Plan H follow-up — fold `Π` into the query** (`⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`): turns per-token
  reconstruction from `O(S·d²)` into `O(d²)`, which is exactly what costs MiniMax its 5–8 % decode.
  Needs hooking attention itself. **Only worth writing after the quality question is settled** — a
  fast cache that is 23 % wrong is not worth speeding up.
- [ ] **Mixed-precision outlier split (TurboQuant).** The *only* way the paper reaches fractional
  bit-widths, so the uniform implementation cannot claim its 3.5-bit LongBench result. Outliers are
  early layers on both models — exactly what the split addresses.

---

## Settled — do not re-open without new evidence

Recorded so the same ground is not re-measured. Each cost a full plan or probe to establish.

- **The prefix pool budget is NOT a constant of this box.** It is `free RAM at load − 3.76 GB
  reserve`, and free RAM at load varies with whatever ran before: measured **11.49 / 12.16 / 12.87 /
  13.70 / 13.76 GB across five loads of the same model on the same day.** Never quote a session
  capacity as a bare integer — quote **K_max = floor(pool_budget / per-session KV)** and the pool it
  was measured at. (Plan J Task 4, 2026-08-05)
- **Exceeding prefix-pool capacity costs the ENTIRE win, not a proportional share.** At K=7 against
  a pool holding 5-6, the result was **0 hits — not 5** — because round 1's re-stores cascade-evict
  the entries round 0 left. Round-2 TTFT stayed at the cold 352 s vs 9.8 s at K=6 (**36×**). This is
  Plan I's "capacity is a cliff, not a slope" reproduced on a different mechanism (pool budget
  rather than compression). **Provisioning the pool to exactly the expected concurrency is
  provisioning it to fail.** 9-bit compression clears the cliff at the same K (7/7, TTFT 10.4 s,
  peak rss −4.4 GB) for ~0.7 s of unpack on the hit path. (Plan J Task 4, 2026-08-05)
- **No expert pin is viable.** Eviction share is linear in N and always 2–4× below the pin's share
  of the cache, on both models (Plan F, 2026-08-02, whole curve measured).
- **The expert VRAM cache is NOT emptied every prefill, and preserving it would not help anyway.**
  0 wipes across 4 turns (Plan A's reserve bump took prefill clear of the 2 GB threshold), and a
  **cold** cache reaches **37.2 %** hit rate against a fully warm one's **36.8 %** — it re-saturates
  in ~4 tokens. **This kills the whole preserve / pre-warm / pin family, not just Plan F's pin.**
  (audit F8, 2026-08-03)
- **`VAAYU_CACHE_BIAS` is unreachable**, and more narrowly than recorded: `cache_aware_topk` /
  `_build_resident_mask` exist in **1 of the 3** MoE forwards (fused-EXL3) and are absent from
  module-list and non-EXL3-fused entirely — so even reverting the upstream transformers router
  refactor would leave the bias unreachable for MiniMax **by construction**. A warning ships
  (`df9a52e`). Any re-targeting spec must cover all three forwards or say which it excludes.
- **`plan_batch` does not "always defer to N=1".** That claim in `test_batched_decode.py` and
  `.superpowers/sdd/task-7-report.md` is stale (pre-Plan A) — it admitted 3/3 on a warm box at short
  prompts. It is true only at agentic context lengths, which is what Plan K acts on.
- **TurboQuant KV at 4 bits fails its quality gate** by 2–7× on both models; 8 bits fails on MiniMax
  (0.0612 vs a 0.05 bar); **9 bits passes both**, which is the shipped `_DEFAULT_BITS`.
- **Uniform-bit TurboQuant on the live KV cache is the wrong tier** — Plan I moved the identical code
  to the eviction boundary and the per-token cost went to zero.
- **MiniMax runs the module-list forward, not the EXL3 fused one.** Every MiniMax perf number on
  record is a module-list number. `CLAUDE.md` §0's dispatch table was fixed 2026-08-02.
- **Decode floors, for every future gate: MiniMax 1.47 tok/s, Qwen 0.82 tok/s.**
  **Re-confirmed same-box 2026-08-02 on `main`: MiniMax 1.48, Qwen 0.85.** Prefer these — they were
  measured on the same box, same day, against the branch they gate. A cross-session floor misled a
  full debugging cycle this session; a same-day A/B baseline costs ~12 min and removes the doubt.
- **PyTorch's caching host allocator charges the NEXT POWER OF TWO for every pinned allocation, and
  charges an exact power of two exactly.** Measured directly 2026-08-02, torch 2.11.0+cu130, fresh
  processes: 64 MiB → 64, 128 → 128, **129 → 256**, 257 → 512, 1024 → 1024. Consequences: a backing
  store a hair over a power of two costs nearly double and the surplus is never addressable; and
  freed pinned blocks are cached for reuse, not returned to the OS. Any future pinned allocation
  must be sized with this in mind. (Plan J §0)
- **The expert RAM cache hit rate is NOT uniformly low, and "5-7 %" is a MiniMax number.**
  Same-box, same day: **MiniMax 6 %, Qwen 47 %** (89 645 hit / 102 867 miss). Plan J's §2 argued the
  cache had "not much to lose" by generalising MiniMax's figure to both models, and that reasoning
  is what let an unconditional clamp ship — it cost Qwen 49 % of decode. **Never generalise a
  cache-effectiveness number across the two geometries.**
- **Audit F8 does NOT license shrinking the expert RAM cache.** F8 measured the expert **VRAM**
  cache re-saturating in ~4 tokens (cold 37.2 % vs warm 36.8 %). Plan J cited it as evidence that
  losing 6.7 % of **RAM** cache slabs would not matter. Different cache, different question, and the
  RAM answer was the opposite for Qwen. F8's own claim stands as written; **its scope does not.**

---

## Open bugs / questions (found in review, not yet fixed)

- [ ] **F3 — the cross-turn prefix cache is wired into `generate()` only.** `batched_generate`
  (`vaayu_base.py:3553`) and `speculative_generate` (`:3448`) never call `PrefixCacheStore.lookup()`
  or `_persist_and_store()`. CLAUDE.md §0's two shaping facts — parallel fan-out and heavy context
  reuse — therefore cannot compose: a fan-out batch pays a full cold prefill per stream even when
  every stream shares the system prefix. **Not a positional fix** (N per-session caches must merge
  into one left-padded batch cache = new mechanism), so it was not promoted. It also biases Plan K's
  measurement — see that plan's §8c. (found 2026-08-03, audit F3)
- [ ] **F4 — `max_seq_len` defaults to 512 and no harness sets it**, so
  `_get_vram_pinning_budget()` reserves KV for 512 tokens against real contexts of 2 048–7 751.
  Measured (Qwen): KV 0.193 MB/token → reserved 0.092 GB, actual 0.367 GB at 2k (**−0.28 GB**) and
  1.390 GB at 7.7k (**−1.30 GB**). Pinning therefore claims VRAM the KV cache later needs — the
  context-length sensitivity `_VRAM_RESERVE_BYTES`' own comment warns about, attributed there to the
  blanket reserve rather than to this input, and plausibly part of why Plan A needed +1.5 GB of
  blanket reserve. Fix is parameter plumbing, not a move. (found 2026-08-03, audit F4)
  **Attempted promotion to "FATAL" on 2026-08-02 was WRONG — retracted the same day.** A crash at
  K=3 × ~8.2k was blamed on F4 before the cause was isolated; setting `max_seq_len` did not fix it,
  and the controlled A/B then showed Plan I's documented shape (K=3 × 7 751) runs fine on BOTH main
  and the Plan J branch **without** `max_seq_len` being set by any harness. F4 is a real VRAM
  accounting bug, but it is not what breaks these runs, and no harness needs the workaround.
  What the A/B did establish: **VRAM headroom at 7.7k is only ~1 GB** (allocated 12.24-12.33 GB,
  reserved peaks 14.54-14.58 GB of 15.47 GiB), and the crash reproduces at ~8.2k. So there is a
  real context-length cliff between 7.7k and 8.2k on this box, owned by the
  `_VRAM_RESERVE_BYTES` / `compute_vram_expert_budget` pair (the expert VRAM cache takes
  `free_vram_at_load − 3.5 GB` = 8.7-8.8 GB before any KV exists, and never evicts: `0evict`
  throughout). **Characterize that cliff before blaming any single input for it.**
- [ ] **VRAM exhaustion surfaces as a bogus `TypeError`, and the "graceful" RoPE fallback is a lie.**
  When VRAM is exhausted, both RoPE precompute paths in `vaayu_base.py:2917-2941` raise
  `CUDA out of memory`, the handler sets `_full_position_embeddings = None` and prints
  **"RoPE will be computed per-layer"** — but no such per-layer path exists. `get_pos_emb_args`
  (`:2685-2689`) then returns `{}` and the layer call dies with
  `MiniMaxM2DecoderLayer.forward() missing 1 required positional argument: 'position_embeddings'`,
  ~30 frames from the real cause. Every modern decoder layer takes `position_embeddings` as a
  required arg, so the fallback cannot work for any of them. Two things wrong: the message promises
  a mechanism that does not exist, and a VRAM OOM is laundered into a signature error that sends the
  reader to the wrong subsystem entirely. Cost me a full debugging cycle. Fix: either re-raise the
  OOM (honest failure, CLAUDE.md §5) or implement the per-layer path the message claims.
  (found 2026-08-02, Plan J Task 4 gate)
- [ ] **F6 — `vaayu/kv_cache_manager.py` (`KVCacheManager`) is unreachable but still maintained.**
  Constructed only in `tests/test_kv_cache_manager_fp8.py`; superseded by `PrefixCacheStore` (A1).
  Its helper `_extract_kv_as_bf16` is reachable only from `save_snapshot`, and was hardened by
  **Plan E Task 2** and again by **Plan H Task 3** — two plan tasks spent making an unreachable
  function correct, a tax every future KV-format change will pay again. Delete the module + tests,
  or mark it a reference implementation and exempt it from cache-mode work. (found 2026-08-03, F6)
- [ ] **F7 — `vaayu_ref._expert_history` is write-only.** Written by all three MoE forwards
  (`moe_patch.py:1039`, `:1174`, `:1786`), read by no production code — the consumer went with the
  dead prefetch path. `patch_moe_blocks`' docstring still claims *"The forward() loop reads
  _expert_history after each layer and submits speculative prefetch"*, pointing the next reader at a
  mechanism that does not exist. Runtime cost ≈ 0; the cost is the false docstring.
  (found 2026-08-03, audit F7)
- [ ] **F13 — `_decode_stats_reset` latches for the process, not the turn.** Set True on the first
  decode pass (`vaayu_base.py:2946`) and never cleared, so the reset that exists to exclude prefill's
  ~100 %-miss pollution runs once per model lifetime. From turn 2 on, cumulative expert-cache hit
  rate is diluted by every new prefill. **Telemetry only.** (found 2026-08-03, audit F13)
- [ ] **`_create_kv_cache` falls through to fp8 on an unrecognised mode string.** A typo in
  `kv_cache_mode` silently yields the fp8 cache rather than an error — the silent degradation
  CLAUDE.md §5 forbids. Plan H Task 3 inserted the turboquant branch *before* the fall-through, so
  this entry is now only about the fall-through itself: **should an unknown mode raise?** Still open,
  still a separate call. (found 2026-08-02, Plan H audit)
- [ ] **`matched_tokens` over-reports by ≤1 token per hit.** `PrefixCacheStore.lookup()` records the
  raw match length; the `generate` override then clamps it down by one position so the model has a
  token to run. Fires only when the new prompt is a complete prefix of the cached one — ≤0.05 % at
  the ~2000-token prefixes this targets, and exactly zero in the K=2/K=3 gate runs. Documented in the
  `stats()` docstring, NOT fixed: moving the clamp into the store would couple the cache to the
  caller's execution needs to correct a rounding error. **Telemetry precision only.**
  (found 2026-08-01, Plan C design)
- [ ] **Prefix `hit_rate` is not comparable across lookup modes.** On a named session, an entry that
  shares ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the same case
  counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing semantics — documented in
  `stats()` and pinned by tests in `aada3cd`, NOT changed. Worth revisiting if a mixed-mode workload
  ever needs one comparable number. (found 2026-08-01, Plan C final review)

---

## Done — archive

Verdicts only. Task-level detail is in each plan's RESULTS doc (`docs/superpowers/plans/`) and in
this file's git history before 2026-08-03.

**2026-08-03**
- [x] **Misplaced-feature audit** — register:
  `docs/superpowers/specs/2026-08-03-misplaced-feature-audit-RESULTS.md` (14 findings, F0–F14).
  Found, proved and ranked; fixed nothing, per the brief. **Two of the three starting candidates did
  not reproduce**, and the largest finding (F0, → Plan J) was not on the candidate list — it came
  from chasing the one number the load-time RAM ledger could not explain. Promoted: F0 → Plan J,
  F2 → Plan K. Refutations and corrections are under "Settled" above; non-promoted findings F1, F3,
  F4, F6, F7, F13 are open items above. Evidence: `~/vaayu-evidence/2026-08-03-audit/` (2 gzipped
  run logs + 3 re-runnable probe scripts).

**2026-08-02**
- [x] **Plan I — compress-on-evict KV.** MERGED (`82a217f`), suite 519. Decode untouched (Qwen
  0.86 → 0.86, MiniMax 1.53 → 1.56) — the property Plan H failed. Pool capacity K=4: bf16 0/4 HIT →
  compressed 4/4, round-2 TTFT 91 s → 9.6 s. **Long context K=3 × 7 751 tok: 332 s → 9.8 s (34×) —
  the largest effect measured anywhere.** `_DEFAULT_BITS` 8 → 9 after 8 failed MiniMax's quality
  gate. Honest limit: benefit is conditional on ≥2 sessions exceeding the pool budget; reuse
  collapses to zero, not proportionally — **capacity is a cliff, not a slope.**
- [x] **Plan H — TurboQuant KV.** Branch `plan-h-turboquant-kv`. Size target hit exactly (0.5312
  B/elem, 0.266× bf16) but the pre-stated quality gate **FAILED by 2–7 % on both models** at 4 bits
  and MiniMax decode regressed 5–8 %. Ships dark and opt-in; `auto` unchanged. The 0.05 bar first
  passes at `bits_total=8` / 1.031 B/elem (1.94× over bf16, not 3.77×).
- [x] **Plan G — expert pin + routing locality.** CLOSED, nothing built. Pin dropped on Plan F's
  evidence; the bias measured as unreachable code on both models, now warned about (`df9a52e`).
- [x] **Plan F — expert-eviction `top_n` sweep.** DONE. **Verdict: no N is viable.**
- [x] **Plan E — KV double-storage fix.** FULLY CLOSED. KV 3.016 → **2.000 B/elem** (−33.7 %) on
  both geometries, `auto` flipped to bf16, and the bf16 layout serializes to disk (restore gate:
  TTFT 27.6 s → 3.0 s).
- [x] **`delete_original` never fired for an already-split model** — FIXED `c170474`. Executed on
  both models: **273 GB reclaimed**, disk 89 % → 57 %; both still load and generate.
- [x] **`remove_real_and_linked_file` unlinked the same file twice** — FIXED `72c7b08`
  (`os.path.realpath()` returns `str`, every caller passes a `Path`). 3 tests.
- [x] **Stale `QUANT_SIG_BF16` comment in `vaayu/kv_persist.py`** — FIXED in Plan H Task 4.

**2026-08-01**
- [x] **Plan A — VRAM decode-headroom fix** (`045f387` `86a6ca5`). `_VRAM_RESERVE_BYTES` 2 → 3.5 GB;
  OOM closed at K=2 (4/4) and K=3 (6/6), decode unregressed. Attribution: the reserve bump alone
  fixes it; `expandable_segments` is optional insurance.
- [x] **Plan C — prefix-cache hit telemetry.** `PrefixCacheStore` counters + `stats()`;
  `test_multiturn.py` is now a **gate** (exit ≠ 0 on a bad HIT count).
- [x] **Plan D — expert usage histogram, measurement only.** Answered the pin question: hot-but-
  evicted experts exist, but only **19 %** of budget-pressure evictions land on each layer's top-10,
  and the cache runs saturated. Fed Plan F.
- [x] **VRAM prefill OOM chain, all 4 steps** — RAM budget rebalance → VRAM diagnostic (verdict:
  decode-boundary, not prefill) → Plan A → round-2 HIT measurement (TTFT ~90-101 s → ~9-10 s at
  both K=2 and K=3, kernel OOM 0).
- [x] Dead code `PrefixCacheStore.drop()` removed; VRAM-cache-disabled message promoted to a
  `WARNING`; stale `_PREFILL_VRAM_HEADROOM` cross-reference fixed (`9aa2c61`); line-number citations
  replaced with symbol names.

**Earlier**
- [x] KV prefix cache A1 + KV disk persistence (landed, ships dark)
- [x] Batched decode static v1 (landed)
- [x] MiniMax-M2.7 bring-up (4 root causes fixed)
