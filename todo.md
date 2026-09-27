# TODO — Vaayu pending work

**Everything with a `[ ]`, `[~]`, `[!]` or `[bug]` in this file is PENDING. Closed rows are not kept here.**

**Moved 2026-09-26 to [`todo-backlog.md`](todo-backlog.md) (user request): 19 OPEN items with no user value
today** — rows **23, 32, 35, 36, 39, 5, 6a-b**; row 49's step 3 and MiniMax-prefill-H2D items; **47-m2, 47-m5-b,
G3, G6**; and from STILL OPEN / S1: the resident-size-checkpoint note, `_finish` stop-string identity, the
`expert_sim` anchor gap, the `test_slab_arena` flake, row 44's F3 remnant and S1-L3. **Nothing was closed or
deleted.** Each keeps its full text and marker there, plus why it moved and when to re-open it. A pointer line
marks each old spot here, so row numbers still resolve. **Pick work from THIS file; look in the backlog only to
re-open something.**

**Pruned 2026-09-23 (this prune):** every `[x]` row was removed — the closed top-level rows
**2, 3, 13, 14, 18, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 41, 42, 42-B, 43, 44, 45, 46**, the closed
`[x]` C1 row in OPEN, the closed sub-plans **A1–A4** and gaps **G0/G1/G7** inside row 47, row 49's closed
steps (0, 1, 2A, the B re-rank, the ring-pool and read-shape arms), and every closed line in the
navigation tables (START HERE, PRIORITY, IMPLEMENTATION SEQUENCE, §0's OPTIONAL table). **A cross-reference
to any of those row numbers now resolves only in `git log -p -- todo.md`.** Nothing still open was dropped:
row 33's surviving `[bug]` (the `test_slab_arena` full-suite failure) and row 44's **F3 remnant** moved to
**STILL OPEN**; row 46's surviving yardstick and two probe traps are in **SETTLED**; rows 47's live
**47-m1…m6** and **47-m5-b** stay in row 47; row 49's live drive-idle lever and step 3 stay in row 49.
**SETTLED, §0's vacancy notes and the reference sections (Environment, Checkpoints, On-device gates) were
kept in full — they are verdicts and reference, not tasks.**

**Pruned 2026-09-10:** every `[x]` row was removed again — the whole closed OPEN block (L1
T1-T10 and its fix waves, the `env.json`/`logs/`/`requests.jsonl` bug rows, S1/S1d/S1e/S1e-bis/S1f,
G1, V1), the closed CARRIED-FORWARD block (dense speculative decoding and its two bug rows, S1d's
TRIAL cost), and rows **6a, 10, 11, 15, 16, 17, 19, 20, 21, 22** plus row 5's three closed
sub-items. **Nothing that was still open was dropped:** C1 is now a top-level row in OPEN, row 6a's
surviving `[ ]` is row **6a-b**, and row 15's open `[bug]` and its known-open minors moved to
STILL OPEN. The verdicts of the pruned rows survive in §0's vacancy notes and in SETTLED.

**Pruned 2026-09-03:** every `[x]` row was removed — all 16 B-rows (the 2026-08-23 server-review
bugs), the whole FIX SEQUENCE table, and every closed R/M/C/P perf row. Two things survive that
prune and you should not go looking for the originals: the **SETTLED** index below, which is the
anti-regression record of what was REFUTED and must not be rebuilt, and the reference sections
(Environment, Checkpoints, On-device gates), which were never tasks.

**Where the removed text went.** Full pre-prune text: `docs/2026-09-23-todo-archive-preprune.md`,
`docs/2026-09-10-todo-archive-preprune.md` and `docs/2026-09-03-todo-archive-preprune.md` (**gitignored — this disk only**). The durable copy
is `git log -p -- todo.md`; the commit before each prune holds it in full. Older closed detail was
moved out on 2026-08-23 (commit `f3869b3`).

Protocol: `CLAUDE.md` §6 (ledger) and §7 (all planning/investigation/spec work lives in `docs/`).
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Reading this cold, in a fresh session, is the intended way to use it.** Read in this order —
**§0 first: it is the build ORDER and the perf / not-perf split; the sections after it hold the
detail each entry points to.**

0. **START HERE** — the whole open queue on one screen, in pick order, added 2026-09-10. If you read
   nothing else, read that table and §0.
1. **OPEN** — EMPTY as of 2026-09-10 (C1, its last row, was FIXED that day and pruned 2026-09-23).
2. **STILL OPEN, carried forward** — live items left behind by rows that closed.
3. **SETTLED — DO NOT RE-OPEN** — read this BEFORE proposing any perf work. It is the list of
   things already built and refuted, with the number that killed each one.
4. **Environment** — how to run anything here. `python` is not on PATH; every on-device arm needs
   its own `systemd-run` scope. Skipping this section is how arms get thrown away.
5. **On-device gates** — what each gate can actually catch, and which ones a change obliges you to run.
6. **OPTIONAL** — real work, no slot. Never ahead of the rows above.
7. **`todo-backlog.md`** (separate file) — open items with no user value today, each with a re-open condition.

**Read §0 below first: it is the ORDER. Everything after it is the DETAIL.**

---

# ▶ START HERE — the whole open queue, re-prioritised 2026-09-11

---

# ▶▶▶ THE SEQUENCE — S7…S10, set 2026-09-26 at the USER's request ("do all these in fresh sessions"). One per fresh session, in order. ◀◀◀

**HOW A FRESH SESSION PICKS — no judgement needed:** take the **first row below that is not ✅**. Read its
handoff in full, re-verify its evidence against HEAD, mark its `todo.md` row `[~]`, and work it. When
it closes: mark ✅ in this table with a one-line verdict, `[x]` its row below, update the suite floor if
it moved, and point the PICK THIS NEXT line at the next row. **Do not reorder without a reason written
here.** **After S10:** re-prioritise with the user (the long tail is listed under "After S1…S6" in the
old table below).

**Ordering rule** (the same as S1…S6): *(a)* helps the model that serves the agent (Qwen3-Next), *(b)*
the biggest multiplier a user can reach at N=1 (CLAUDE.md §8), *(c)* cheap, and last *(d)* a bug no
server user can reach. All four were surfaced by S6's decomposition (2026-09-26).

| # | item | why here | cost | handoff |
|---|---|---|---|---|
| **S7** ✅ | ~~**Qwen3-Next per-expert dispatch** — ~66 ms of the 0.112 s decode token (57 %), 0.138 ms × 480 slots; a VRAM HIT still dequantizes the packed int4 expert to ~6.3 MB of dense bf16 before three M=1 `F.linear` calls | (a): the production model's largest term; no row covers it, SETTLED refutes nothing~~ **CLOSED 2026-09-27, A WIN: fused packed-expert decode kernel, decode −48.4 % / −48.9 % (1.94×), ids identical, D6 PASS, gates green. See the S7 row below** | ~1 day measure; build unknown | [`…S7-qwen3next-expert-dispatch-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S7-qwen3next-expert-dispatch-HANDOFF.md) |
| **S8** ✅ | ~~**gpt-oss-20b at 0.17 tok/s on a model that FITS the GPU** — stock peaks at 13.79 GB VRAM; Vaayu streams 13.28 GB/token, and `compute` 5.2 s of 5.8 s includes a per-visit MXFP4 swizzle. Split the 5.8 s, then the user picks: pin it whole / cache the converted layer / A6~~ **CLOSED 2026-09-27, A WIN: routed fused-3D delivery, decode 9.61×/9.69× (two replicates), 0.23→2.27 tok/s, logits bit-identical to stock, gates green. See the S8 row below.** | (b): ~100× projected (→ ~20 tok/s); it already serves | hours to measure; route cost varies | [`…S8-gptoss-streamed-transform-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S8-gptoss-streamed-transform-HANDOFF.md) |
| **S9** | **DeepSeek `wo_a` folded on the CPU every token** — 1.42 s of 3.06 s (46 %); fold on the GPU instead, which should be bit-identical (= S6 lead **L-a**) | (c): cheap, ~1.9× on DeepSeek, but a fixture model | hours + one ~10 min A/B | [`…S9-deepseek-wo_a-fold-on-gpu-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S9-deepseek-wo_a-fold-on-gpu-HANDOFF.md) |
| **S10** | **refuse `use_torch_compile=True`** (warn-and-disable) instead of crashing, and correct `config.py:328`'s false reason (= **row 48**, fix (b)) | (d): a real bug, but the server declines the knob, so no server user reaches it | hours | [`…S10-refuse-torch-compile-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S10-refuse-torch-compile-HANDOFF.md) |

**Durable rows** (`docs/` is gitignored — the conclusion lives HERE):
- [x] **S7 — Qwen3-Next per-expert dispatch.** **CLOSED 2026-09-27, A WIN — decode median 100.4–100.5 → 51.3–51.9 ms = −48.37 % / −48.85 % (two ABBA replicates, sd ≤ 0.6 ms), ≈ 9.95 → 19.4 tok/s (1.94×); greedy ids ON == OFF 64/64 both; prefill flat (±0.8 %); D6 band PASS (code +0.09 %, prose −0.10 %); accumulator gate Qwen3-Next 8/8 with the kernel live + MiniMax 8/8; server smoke 9/9 with the kernel live; branch suite `2998 passed, 0 failed, 0 errors, 20 skipped`. **After the final review's fix pass (`df8ec7d`): A/B r3 −48.66 %, ids identical, prefill flat, accumulator Qwen3-Next 8/8 + server smoke 9/9 (kernel live), suite `3001 passed, 0 failed, 0 errors, 20 skipped`.** New `vaayu/packed_expert_kernels.py` (two Triton kernels over the packed int4 blobs, pointer table, per-geometry verdict vs the stock arithmetic, installer); ONE branch in `_make_fused_sparse_forward` for one-token decode; miss bookkeeping shared via `_settle_miss_upload`. §8: no knob, plumbed in `__init__` after `patch_moe_blocks` (test-pinned), reachable at N=1. No fused-EXL3 arm exists. Final whole-branch review (Opus): with fixes → I1 + two re-graded minors fixed and re-verified on device. MERGED to `main` 2026-09-27 (`9e6f81d`); merged suite `3001 passed, 0 failed, 0 errors, 20 skipped` (212.27 s).** Was: **STARTED 2026-09-27: handoff re-verified against HEAD — §2 numbers reproduce exactly from r36recheck1.json; line refs drift ≤2 (call `:2349`, F.linear `:2397-2399`, accumulate `:2417`); a 2nd per-layer sync the handoff omits: `unique().tolist()` `:2219`; NOEXP stubs the WHOLE block, so the pool also holds the shared expert + per-layer overhead.** Evidence (r36recheck1, pre-S3): U 0.1123 s, NOEXP
      0.0368 s, bytes ~9 ms, so dispatch ≈ 0.138 ms/slot ≈ 66 ms/token. Per slot (`moe_patch.py:2329-2413`):
      gather, `expert_weights_from_entry` (compiled int4 dequant, 31.8 µs even on a VRAM hit), 3 `F.linear`,
      scale, append; plus 48 blocking `expert_indices.cpu()` syncs per token (`:2317`). Steps: re-baseline at
      HEAD (NOEXP arm) → U-level decode-only census → price L1 (fused w4a16 MoE decode kernel reading the
      packed blob), L2 (dense cache for hot experts), L3 (host trim); L4 = CUDA graphs stays PARKED (row 36)
      → brainstorm → user approval → build. **Stop if the pool is < 25 % of T at HEAD, or the best lever's
      ceiling is < 5 % of the real token.** Done = ABBA ≥ 2 % (else revert), greedy ids identical, D6 band,
      expert-id accumulation order kept, gates (suite floor, Qwen3-Next + MiniMax accumulator, Qwen3-Next
      server smoke). §8: no knob (S3 precedent), reachable at N=1.
      - [x] **Step 0 — re-baseline at HEAD `cab066a` (2026-09-27, `row26_decode.py -s7base1`, BALLAST 25):**
            U1 0.1110 / U2 0.1109 s (9.01 tok/s), NOEXP 0.0333; pool 77.7 ms = 70 %; minus bytes (80.5 VRAM
            misses × 1.77 MB @ 19.7 GB/s = 7.2 ms + 1.5 ms stall) → **dispatch 68.9 ms = 62.1 % of T_U =
            143.6 µs/slot. Stop rule 0 does not fire.** Greedy ids W/U1/U2 identical.
      - [x] **Step 1 — attribution: HOST-BOUND (2026-09-27, `plans/probes/s7_dispatch.py -r1`, one load).**
            U_HOST (unprofiled, inert: 0.1089 = U1): MoE blocks 78.6 ms host/token = 1,637 µs/layer —
            hit dequant 17.6 ms (**44.1 µs host per call for a 7.8 µs kernel** — torch.compile call
            overhead), miss path 9.8 (122 µs/call), CPU argsort `compute_expert_dispatch` 3.8 (80 µs/layer),
            shared expert 2.9, router 2.2, accumulate 2.0, miss dispatch 1.1, block exclusive (per-slot
            gather / 3 `F.linear` / silu / mul / weight index+mul + per-layer syncs) 39.2. U_SYNC: MoE issue
            76.0 ms vs **drain 3.8 ms/token (79 µs/block)** — the GPU trails the host by < 80 µs. CUPTI
            (8 decode fwds, 0 unmatched incl. Triton): 39.6 ms device per 108.9 ms token; MoE 6,733
            launches (~14/slot), **27.5 ms device busy vs 78.6 ms host** → GPU idle ~65 % of MoE time.
            **Per slot ≈ 135 µs host vs ≈ 33 µs device** (dequant 7.8 + 3 `gemvx` 15.2 + elementwise ~9 +
            acc 1). **L2 REJECTED on arithmetic:** dense entries are 3.56× the packed ones (VRAM tier ≈4,210
            → ≈1,184 experts) to save only the 17.6 ms hit dequant; L1 removes that AND the GEMV/elementwise
            host cost at no capacity loss. **L3-via-`bmm` batching is risky:** cuBLAS may pick a kernel per
            batch size, so an expert's output could depend on how many OTHER experts hit — cache state —
            breaking the row 44 F6 determinism invariant; a Triton kernel with a fixed per-expert
            reduction does not. Trap: an unprofiled arm AFTER a CUPTI arm reads +16 % (profiler residue) —
            put CUPTI last. RESULTS `docs/2026-09-27-S7-dispatch-census-RESULTS.md`.
      - [x] **Step 2 — priced (2026-09-27, `plans/probes/s7_microbench.py` / `s7_kvariants.py` /
            `s7_numcheck.py`; one real layer's 10 experts, M=1, ABBA).** Per layer, synced: **stock 995–1,021 µs**
            (×48 ≈ 48.6 ms), stock-with-dense-weights 518–532 (dequant ≈ 490 µs/layer), L1 first cut 179.
            Kernel variants (layer 10, do_bench, 17.69 MB packed/layer): 3-D-reshape tiles 149–159 µs (~111
            GB/s — the reshape layout was the limiter, not `cvt`); **V3 = exact 2-D (static nibble loop,
            magic-number int→float, integer RNE to bf16) 75.3 µs (235 GB/s)**; V4 scale-factored 66.1 µs (not
            stock-equal). **V3 over all 48 layers × 8 draws: `h` 99.98 % / output 99.87 % bit-equal to stock**
            (cuBLAS `gemvx` summation order is not replicable — one-ulp flips; fp64 error identical).
            Expert-id accumulation order kept exactly (down kernel sums fp32 in table order); an expert's
            result is independent of which others share the launch → F6 invariant holds by construction.
            **Price: L1 ≈ 100–110 µs/layer vs ~1,010 → ~43 ms/token ≈ 39 % of T_U on VRAM hits alone.**
            L3 (host trim, no kernel) ≤ ~18 ms — dominated. Stop rule (< 5 %) does not fire. Open design
            questions: the miss path (80.5 slots/token, H2D of packed blobs 7.8 ms on the compute stream),
            M > 1 (prefill chunks, spec verify), the per-layer syncs.
      - [x] **Step 3 — design (brainstorming) → spec → user approval → plan → build.** Design approved in
            brainstorming 2026-09-27 (approach **A "L1 exact"**, user-chosen over B = scale-factored V4 and C =
            host trim only); spec `docs/superpowers/specs/2026-09-27-S7-packed-expert-kernels-design.md`
            (**approved by the user 2026-09-27**; §9 = four refinements found while planning: behavioural SiLU check, per-geometry verdict with stock fallback, one shared `_settle_miss_upload`, loud refusal instead of dropping an expert). Plan `docs/superpowers/plans/2026-09-27-S7-packed-expert-kernels.md` (6 tasks: kernel module, miss-path helpers, forward branch, installer + engine wiring, docs + suite, on-device verdict) — **approved 2026-09-27, executing NATIVE (inline) on branch `s7-packed-expert-kernels`**.
            - [x] T1 kernel module  - [x] T2 miss-path helpers  - [x] T3 forward branch
            - [x] T4 installer + engine wiring  - [x] T5 docs + suite  - [x] T6 on-device verdict
            **Rulings made while building (durable copy; ledger was `.superpowers/`, gitignored):**
            (1) the spec's output-level "≥ 99 % bit-equal to stock" bar failed only on the MiniMax-like
            geometry (98.73 %, max 6 ulp) — fp32 dot-product order noise amplified by cancellation in the
            sum over experts, not a missed rounding point (h 99.967 %, fp64 error 4.514e-3 vs stock
            4.512e-3) — replaced by per-stage bars (gate/up ≥ 99 %, down ≥ 99 %; measured ≥ 99.95 % /
            ≥ 99.98 %; the down stage with its rounding removed = 0 %) + an EXACT table-order sum test +
            output accuracy (relL2 ≤ 1e-3, fp64 ≤ 1.25× stock); (2) that exact test caught LLVM contracting
            `out += o * w` into an FMA (skips stock's fp32 product rounding) → `libdevice.mul_rn`; (3) the
            unbound-locals lint gate: kernel launches carry the gated_delta_kernels-style disable, the
            kernel branch BINDS the per-slot loop's dispatch names (no suppression in the core forward).
            **Final whole-branch review (fresh Opus reviewer): "with fixes", 0 Critical.** Fixed, each
            RED→GREEN: **I1** — the branch assumed every expert of every layer is a fusable blob (the
            installer reads ONE expert): an unpacked expert was silently DROPPED, mixed-width roles RAISED,
            a dense VRAM entry crashed; now `packed_experts_decode` runs any call that is not all blobs of
            one geometry through `stock_experts_decode`, and misses build the stock loop's own entry
            (supersedes spec §9.4's raise); **M1→Important** — the bf16 integer rounding turned NaN into ±0
            (masking faults stock propagates) → NaN passed through (+4.4 µs/layer gate_up, ≈ 0.2
            ms/token); **M5→Important** — this row's text (install site, Step 3 marker, rulings not durable).
            **Deferred minors:** SiLU uses approximate exp/div (libdevice exp/div_rn would sit closer to
            stock; D6 PASS, ids identical as is); the evict-mid-call test cannot prove blob lifetime and the
            `_TableRing` docstring credits per-slot device buffers where single-stream ordering protects;
            `gemm_hit`/`construct_hit` stay 0 on the kernel branch (`vram_split` reads "hits are free"); the
            installer checks SiLU before packedness; Triton specialises `n_pairs` (a `DECODE_TOP_K` in another
            class JIT-compiles at decode). Not built: load-time seeding of every layer's geometry (a model
            whose layers differ pays that geometry's JIT + verdict at its first decode token).
            Suite (branch, 2026-09-27, FOREGROUND): `3001 passed, 0 failed, 0 errors, 20 skipped` (219.32 s, after the review fix pass) = floor 2955 + 46 new S7 tests. Floor moved at the merge (`9e6f81d`). Durable summary: new `vaayu/packed_expert_kernels.py` — two Triton
            kernels (V3: gate_up grid pairs×I/BN, down grid H/BN looping pairs in expert-id order into the
            fp32 accumulator), pointer table per routed PAIR sorted by (expert id, slot) via an event-guarded
            pinned ring; installed in `VaayuBaseModel.__init__` right after `patch_moe_blocks` (NOT beside S3 in `init_model`: that runs before patching, and the installer reads each block's `forward_name`) only on CUDA + triton + a fused non-EXL3 packed,
            blob-fusable MoE block, self-checked on ONE REAL expert read at load (relL2 ≤ 1e-2), else
            WARNING + stock. ONE forward branch in `_make_fused_sparse_forward`, taken iff installed AND M=1
            AND not prefill; **misses go through the kernel too** (the accumulator gate compares logits
            across different VRAM residency, so hit/miss must not change an expert's math). Split
            `_expert_dense_weights` into upload+blob / dequant. No knob; private `_packed_moe_kernel` is the
            A/B toggle. Out (banked): removing the 2 per-layer syncs, M>1 kernel, hit/miss-read overlap.
            Accepted cost: disk stall no longer hidden by the slow loop. Unit tests over Qwen3-Next-like
            (2048/512 int4 g32 top-10), MiniMax-like (3072/1536 int4 g128 top-8) + int8. Acceptance = §4.
- [x] **S8 — gpt-oss-20b: attribute the 5.8 s, then the user picks a route.** **CLOSED 2026-09-27, A WIN:
      routed fused-3D delivery, decode 9.61×/9.69× faster than whole-layer (two replicates), logits
      bit-identical to stock, gates green — see T6 below for every number. STARTED 2026-09-27: handoff
      re-verified against HEAD `678bdb9` — every A3 number it cites reproduces from
      `docs/2026-09-20-A3-gptoss-RESULTS.md` (0.17 tok/s, 5.8 s, `gb_tok` 13.28, compute 5.2 / io_wait 0.3,
      stock peak 13.79 GB); line refs drifted with S7: the forward is `moe_patch.py:2108` (pinned short-circuit
      `:2151`), `_populate_fused_experts` `:2046`, `_release_fused_experts` `:2089`; `reserve_floor_bytes`
      still `vram_plan.py:538`. 47-m5-b's backlog re-open trigger is this row — re-check it if the cache sizing
      moves. **The "CLAUDE.md is stale" warning below (and the handoff's) is itself stale: CLAUDE.md §0/§5 now
      put gpt-oss on the whole-layer row, running since 2026-09-21.** It runs the FOURTH forward
      `_make_whole_layer_sparse_forward` (`moe_patch.py:2090`), which is NOT in CLAUDE.md's dispatch table.
      **CLAUDE.md §0/§5 still say "NOT YET RUN / G1"; that is stale since A3 (2026-09-21).** Evidence (A3):
      decode 0.17 tok/s = 5.8 s/token, 13.28 GB/token (the whole model), `compute` 5.2 / `io_wait` 0.3 s,
      eager attention, a Hub kernel fetched at run time; stock transformers peaks at **13.79 GB of 16.31**.
      That forward already short-circuits a PINNED layer. Steps: load log (why is nothing pinned — the eager
      floor?) → split read / conversions (`_populate_fused_experts`) / `original_forward` / release → price
      **R-A pin whole** (costs context), **R-B cache the converted layer**, **R-C A6 per-expert MXFP4
      streaming** (~3 days) → the USER picks. Traps: row 40 `[bug]` goes LIVE (experts 10.17 GB < arena);
      R-C must not reach the twin cast. Done = logits bit-identical to stock (A3's sha bar), server 9/9,
      gates.
      - [x] **Steps 0–2 — attributed and priced (2026-09-27, HEAD `678bdb9`, no engine change; RESULTS
            `docs/2026-09-27-S8-gptoss-attribution-RESULTS.md`; probes `plans/probes/s8_{split,stock,origfwd,price}.py`).**
            **The handoff's premise is WRONG: the MXFP4 swizzle is 5.7 ms/token (0.2 % with getconv+assign).**
            The token is BYTES — the whole-layer read runs inside the layer's forward, never reaches
            `_expert_disk_s`, so A3's "`compute=5.2s`, most of the token is NOT the disk" was a timer artifact.
            Re-baseline (198-tok server-rendered prompt, `BALLAST_GB=25`, ids identical W/U1/S/U2): **4.44–4.59
            s/token ≈ 0.22 tok/s**, 13.12 GB/token. Split per token: io_uring read 2.553 s (10.17 GB @ 3.98
            GB/s) · host clone pinned→pageable 0.624 · pageable H2D 0.675 (15.1 GB/s) · swizzle+conv 0.009 ·
            `original_forward` 0.251 (10.5 ms/layer) · MoE total 4.112 (90 %) · rest ≈ 0.46 (embed + head 1.16
            GB each streamed per token, 12 attn layers streamed). **Stock transformers, same prompt, same greedy
            ids: 28.0 ms/token = 35.7 tok/s, MoE 0.64 ms/layer, peak 13.88 GB → Vaayu is 162× slower.**
            `original_forward` 10.5 vs 0.64 ms/layer is GPU/host IDLE, not the kernel: same tensors
            back-to-back 0.58–0.94 ms, populate+call without disk 0.72–1.09 (no per-new-tensor cost), after
            0.25 s idle 3.8–4.2 (the rest to 10.5 unexplained, 5.5 % of the token).
            **Step 0 — why nothing is pinned:** eager (flash: sinks Hopper-only; sdpa: refused by transformers)
            → the row-8 eager floor (`reserve_floor_bytes`) buys flash-parity context (33,206) and lands on its
            clamp: reserve 13.86 GiB, pin budget 0.63 GiB; embed + head skipped. That ruling was priced on
            Llama-1B at −19 %; here it costs ~160×. **Also: the planner never pins a MoE layer WHOLE** (Phase 2
            adds MoE layers to `non_expert_pinned_layer_names` only), so the whole-layer forward's
            `pinned_layer_names` short-circuit (`moe_patch.py:2151`) is unreachable. **Arena SKIPPED** on
            whole-layer-only models (`vaayu_base.py:2800`) → **row 40 is NOT live today** (handoff wrong); it goes
            live under R-C (experts 10.17 GB < the 12.81 GB budget printed).
            **Routes:** **R-A pin whole** — engine's own planner curve (eager, chunk 512): all attn + 24 MoE layers
            → 6,639 ctx; +head → 4,061 (below the 3.5 GiB base floor); everything (stock) → **998**; each MoE layer
            (0.395 GiB) ≈ −0.16 s/token for ≈ −1.0–1.2k context. Needs a context-vs-speed knob (§8(1)).
            **R-B cache converted layer — DEAD, ceiling 8.7 ms/token.** **R-C per-expert (A6) — premise MEASURED:
            random bytes in every non-routed expert's blocks/scales/bias → layer-5 block output bit-identical
            (`torch.equal`, max|diff| 0); layout `StridedLayout`, each expert a contiguous dim-0 slab → NO new
            kernel: fill only routed rows of a `[num_experts,…]` buffer, run stock `original_forward`.** 96 slots
            × 13.24 MB = 1.27 GB/token; all experts fit the RAM budget → after warm-up ≤ ~65 ms pinned H2D/token
            at today's 33k context; projected several → ~10 tok/s with the embed/head stream fixed too. **R-D
            (new) whole layers in pinned RAM** — ~1 tok/s, same RAM as R-C, dominated. Other levers on the
            table: CPU gather for the embedding (streams 1.16 GB/token to read ONE row), smaller prefill chunk
            (eager score term), 47-m4 (halves KV).
      - [x] **Step 3 — route: USER PICKED R-C (per-expert streaming, stock kernel over an `[E,…]` buffer)
            2026-09-27**, over R-A (pin whole + context knob), R-C-then-R-A, and parking S8. Row 40 is fixed in
            scope. The non-expert stream (embed/head/attn, ≈ 0.46 s) is a separate follow-up. **Design
            approved in brainstorming 2026-09-27 (approach A, user-chosen over B = pre-route with own `topk` —
            a tie reads a stale row — and C = compact `[k,…]` buffer — rewrites the kernel's private
            `RoutingData`).** Spec `docs/superpowers/specs/2026-09-27-S8-routed-fused3d-design.md` — **approved by
            the user 2026-09-27.** Plan `docs/superpowers/plans/2026-09-27-S8-routed-fused3d.md` — 7 tasks (T0 branch `s8-routed-fused3d`; T1 pure helpers; T2 experts-call wrapper in the whole-layer forward; T3 load-time self-check; T4 engine wiring + row-40 cap + env.json `forward_delivery`; T5 user docs + suite (expect `3045 passed`); T6 on-device verdict, stop if < 3×) — **Approved; execution SUBAGENT-DRIVEN for all 7 tasks incl. the on-device T6 (user, 2026-09-27; GPU jobs one at a time, FOREGROUND).** Durable summary: wrap the experts SUBMODULE's call; routed set =
            `nonzero(routing_data.expt_hist)` (the kernel's own set, exact by construction); fetch through the
            shared `_dispatch_expert_misses` / `_iter_expert_loads` RAM-arena path; copy rows into ONE persistent
            `[E,…]` device buffer per geometry (423.6 MB = today's transient, inside `layer_slack`, planner
            unchanged); run the quantizer's conversions AFTER the upload, then the stock forward; raise on any
            routed expert not uploaded or an unrecognised call. Enabled per geometry by a load-time self-check
            (real layer, random bytes in non-routed rows at M=1 and M=max(2, E//(2·top_k)), `torch.equal`,
            vacuous = FAIL) that runs before arena sizing; FAIL → WARNING, whole-layer, no arena. Row 40 fixed for
            ALL models: arena slabs capped at indexed experts + withheld (gpt-oss ≈ 10.17 GB, not 12.81).
            Delivery tag `routed|whole` beside `__vaayu_forward__`; private `_routed_fused3d` A/B toggle; §8 no
            knob. Acceptance: logits sha == stock at A3's 4-tok and 256-tok refs, ids == stock ×64; ABBA decode,
            **stop if < 3×**; prefill ≤ 5 % regression; gpt-oss server 9/9; Qwen3-Next + MiniMax accumulator 8/8
            with unchanged arena lines; suite floor 3001 + new.
            **T0–T5 done, branch `s8-routed-fused3d`:** T0 branch; T1 `8d6306c` (routed-set adapter,
            geometry key, expert buffers, enabling predicate); T2 `4fa0d57` (routed delivery through the
            whole-layer forward's experts call); T3 `0877e75` (load-time self-check) + fix round `8593195`
            (corrupt non-routed rows with full-range random bytes, not finite values); T4 `2b34fa5`
            (self-check wiring at load, row-40 arena cap, `forward_delivery` in env.json); T5 (this row —
            user docs + full suite + ledger). **Suite (FOREGROUND, full):**
            `3046 passed, 0 failed, 0 errors, 20 skipped`. New files: `vaayu/routed_fused3d.py`,
            `tests/routed_fused3d_fakes.py`, `tests/test_routed_fused3d_helpers.py`,
            `tests/test_routed_fused3d_forward.py`, `tests/test_routed_fused3d_selfcheck.py`. **§8:** no
            knob — `_routed_fused3d` is a private A/B attribute, not a setting; plumbed at load in
            `VaayuBaseModel.__init__`, `from_pretrained` needs nothing new; reachable at N=1 through the
            server; nothing banked (the non-expert stream is filed as its own follow-up row below; 47-m5-b
            re-checked in `todo-backlog.md`, staying there with a dated note).
      - [x] **T6 — on-device verdict, 2026-09-27. A WIN.** RESULTS
            `docs/2026-09-27-S8-routed-fused3d-RESULTS.md`. **Step 1 (4-tok):** delivery ON, arena capped
            1258→786 slabs (768 indexed + 18 withheld), `VERDICT BIT-IDENTICAL`. **Step 2 (256-tok):** first
            pass showed `DIFFERENT` against the committed sha `6668afdd...` — investigated, not a stop: that
            fixture tokenizes `vaayu/moe_patch.py` itself as the prompt, and S7/S8 edited that file since the
            2026-09-21 reference was captured, so the *input* differed (`ids_tail` totally different), not the
            model's output. Regenerated stock fresh at current HEAD: sha `2e4aad53...` on **both** sides,
            `ids_tail match: True`. **The committed `6668afdd...` reference is now stale** — flag for the
            user; `longctx_stock_256.json` on disk holds the refreshed value. **A/B (`s8_ab.py`, new probe,
            `ARMS=W,R1,X1,X2,R2`, `NEW_TOK=24`, `BALLAST_GB=25`), two replicates:** ids identical across all 5
            arms both times; decode whole/routed = **9.61× (r1) / 9.69× (r2)** — clears the 3× stop rule by
            ~3.2×; routed decode 2.26-2.29 tok/s vs whole-layer 0.234-0.240 tok/s (matches the pre-S8 baseline
            4.44-4.59 s/token within noise); prefill routed/whole = 0.243-0.279 **WARM** — the R arms follow a
            warm-up arm (W) that filled the RAM cache, so that ratio is a warm-cache figure; the **cold first
            request is at parity**: W arm (routed, cold) 3.27-3.43 s vs whole-layer 3.51-4.27 s — under the
            1.05 flag either way. Per-arm table, copied from `ab-r1.log` / `ab-r2.log` (`s8ab`
            lines; prompt 198 tokens, `NEW_TOK=24`):

            | arm | r1 prefill (s) | r1 decode median (s) | r2 prefill (s) | r2 decode median (s) |
            |---|---:|---:|---:|---:|
            | W (routed, warm-up, discarded) | 3.43 | 0.4824 | 3.27 | 0.4676 |
            | R1 (routed) | 1.21 | 0.4366 | 1.01 | 0.4395 |
            | X1 (whole-layer) | 3.51 | 4.1831 | 3.93 | 4.2529 |
            | X2 (whole-layer) | 4.27 | 4.2484 | 4.23 | 4.2988 |
            | R2 (routed) | 0.96 | 0.4404 | 0.97 | 0.4433 |

            **One probe-only deviation** (not engine code, `plans/`
            gitignored): running the brief's `s8_ab.py` verbatim crashed arm 1 with
            `RuntimeError: The size of tensor a (128) must match the size of tensor b (199) at
            non-singleton dimension 3` (`eager_attention_forward`) —
            `enable_prefix_cache=False` makes `generate()` skip Vaayu's own `_create_kv_cache()`, so HF
            auto-selects a cache that truncates gpt-oss's sliding layers to 128 tokens while Vaayu's mask
            code assumes full length; **confirmed on `main` too (678bdb9, pre-S8)** via a worktree repro
            (`plans/probes/s8_out/repro-main-prefixoff2.log`) —
            pre-existing, not server-reachable at the default `ENABLE_PREFIX_CACHE=True`. Worked around by
            passing `model._create_kv_cache()` explicitly per arm (documented inline in `s8_ab.py`). **This
            is the same code path as the open `[bug]` row below at "`kv_cache_mode` is INERT ... whenever
            prefix caching is OFF" (`vaayu_base.py:5684-5688`) — see that row's 2026-09-27 S8 T6 sub-bullet
            for the sharpened finding.**
            **Step 5 (64-tok vs stock): `identical over 64: False`** (first diff at index 50: stock token
            `558`, vaayu token `350`) — investigated,
            not a stop (no stop clause on this step): isolated by comparing routed vs whole-layer **on the
            same branch, same cache**, at 64 tokens — **identical to each other at every position**, so the
            stock divergence is a pre-existing Vaayu-vs-stock KV-cache numerics artifact over long
            autoregressive decode (near-tied-logit flip, plausible for code-gen whitespace tokens), not
            caused by routed delivery. **Tracked as its own open row below** ("gpt-oss greedy decode
            diverges from stock ... at token 50"). **Gates:** gpt-oss server `9 passed`; Qwen3-Next accumulator
            `8 passed`, no `capped`/`Routed fused-3D` line, arena `13293` slabs; MiniMax accumulator
            `8 passed`, no `capped`/`Routed fused-3D` line, arena `2629` slabs. **Arena vs `main`**
            (`/tmp/claude-1000/s8-main` worktree, removed after): MiniMax `2636→2629` (7 slabs, 0.27%, RAM
            variance); Qwen3-Next `13668→13293` (375 slabs, 2.7%) — bigger than "a few," flagged, but no
            `capped`/`Routed fused-3D` line fired in any of the 4 runs (main×2, branch×2), so the S8 cap code
            never engaged for either model; it is free RAM at load: `avail` at cache-sizing was 27.46 GB
            (branch) vs 28.12 GB (main) with RSS 3.10 vs 3.09 GB (`gate-qwen.log` vs `gate-qwen-main.log`),
            and the 0.66 GB gap = 375 slabs × 1.77 MB — the arena tracked what the box had free, not S8. §8 all four: no knob (private
            `_routed_fused3d`); plumbed at load; reachable at N=1 (server gate is the proof); nothing banked.
            **Three findings recorded, not fixed:** the stale 256-tok fixture (RESULTS doc only, no
            row — a fixture problem, not a code bug); `enable_prefix_cache=False` + gpt-oss + >128 tokens
            crashes (now a dated sub-bullet on the open `kv_cache_mode` INERT row, cross-referenced
            above); Vaayu-vs-stock 64-tok decode drift at token ~50 (now its own open `[bug]` row below,
            routed==whole so not S8-caused). **PICK THIS NEXT → S9.**
      - [x] **Final-review fix wave, 2026-09-27** (review 678bdb9..77cbfc3: ready to merge WITH FIXES, no
            Critical). Commits: `466d1f8` I1+I2 — `run_routed_self_checks` now, only when a check ran,
            `gc.collect()` + `torch.cuda.empty_cache()` + new `release_fused_staging_buffer()` (persister; frees
            the 512 MiB dedicated pinned `_FUSED_BUF`); `a96b92d` I3 — `tests/test_routed_fused3d_arena.py`
            (CPU `SlabArena(pin=False)`: decode miss→hit with eviction, declined insert, prefill staging, a load
            failure mid-visit) + the prefill test now proves the routed branch ran; `562e7e5` M5 — WARNING when
            a second routed geometry's buffer is allocated (not charged to the planner); `315620c` M1-M3 —
            docstrings + `documentation/moe_expert_aware_streaming.md` §3.4b; this ledger commit (M4 + the
            two reviewer ledger items above). **Re-verified on device (FOREGROUND, one job at a time):**
            gpt-oss server gate `9 passed`; the false `WARNING: Expert VRAM cache disabled` is GONE —
            `Expert VRAM cache: 0.4 GB (~31 experts resident on GPU)` (main, same pin layout: 0.5 GB, ~36-37;
            the ~6-expert ≈ 80 MB residual is not attributed — hypothesis, unmeasured: CUDA kernel modules /
            cuBLAS workspace the check's forwards load early, outside the caching allocator);
            `free fused-layer staging 512.0MB` logged; RSS at cache-sizing 4.59 → **3.79 GB** (main 3.80-3.82),
            post-load 15.07 → **14.19 GB** (−0.88 GB). 4-tok sha `7bfd3bb0…` `VERDICT BIT-IDENTICAL`; 256-tok sha
            `2e4aad53…` == refreshed stock, `ids_tail match: True`; A/B r3 **8.92×** (routed 0.4414/0.4429 s,
            whole 3.934/3.954 s — the whole arms ran faster than r1/r2, routed unchanged; ids identical across
            all 5 arms; prefill cold W 3.26 s vs whole 4.03/3.97 s). Suite (FOREGROUND, full):
            `3058 passed, 0 failed, 0 errors, 20 skipped` (+12). **§8:** no knob (load-time internals and a
            warning); nothing new to plumb; reachable at N=1 through the server (the gate is the proof);
            nothing banked. Logs `plans/probes/s8_out/*-fix.log`, `ab-r3.log`.
      - [ ] **gpt-oss non-expert stream** — embed (1.16 GB/token to read one row), `lm_head` 1.16 GB/token,
            12 attention layers 0.64 GB/token, ≈ 0.46 s/token; CPU gather for the embedding, pinned-RAM
            head/attention.
- [bug] **gpt-oss greedy decode diverges from stock transformers at token 50** (found by S8 T6,
      2026-09-27, `plans/probes/s8_out/ab-r64.json`, `ab-x64.json`, `stock-r64.json`). 198-token
      server-rendered prompt, `NEW_TOK=64`, greedy: Vaayu's ids match stock through position 49, then
      diverge at index 50 (**stock token `558`, Vaayu token `350`**) and every position after. **Isolated
      as independent of routed delivery**: re-ran with `ARMS=X1` (whole-layer) at the same `NEW_TOK=64` and
      diffed against the `R1` (routed) run — **identical to each other at all 64 positions**, including 50
      — so both delivery paths hit the same divergence; it is **pre-existing**, not caused by S8's routed
      fused-3D work. **Single-forward logits are bit-identical to stock** at both 4 and 256 tokens (S8 T6
      Steps 1-2, sha match both), so the divergence is specific to **multi-step decode through the KV-cache
      path**, not the forward pass itself. **Cause not established** — likely candidates are a near-tied
      logit flip (plausible in code generation, where whitespace/formatting tokens can be genuinely close)
      or a numerics difference between Vaayu's `_create_kv_cache()` (`FP8DynamicCache(quantize=False)`,
      the bf16 ring buffer) and stock's native `DynamicCache` accumulating over many steps. **First step
      when picked up:** repeat the 64-token comparison at a few different `kv_cache_mode` settings and with
      per-step logit-diff instrumentation to localise where the drift starts, rather than only where
      argmax first flips.
- S9 = the S6 row's **L-a** `[ ]` (not duplicated here). Fix: fold `wo_a` on the device
      (`vaayu_deepseek_v4.py:344` folds it on the CPU today). **Stop if the device fold is not bit-identical
      to the CPU fold.** Re-baseline first: 0.3267 predates S5-M's arena. Rejected: a RAM cache of the folded
      bf16 (2.35 GB against S5-M-H's 3.78 GB margin).
- S10 = **row 48** `[bug]`, fix (b) (not duplicated here). The guard lives at `vaayu_base.py:2180-2189` and
      the compiled call at `:5332-5339` — row 48's own line numbers are stale. Fix the declined-reason text at
      `config.py:328` and `test_engine_param_reachability.py:65`. It also closes the duplicate `[bug]` bullet
      near row 28b.

---

# ▶▶▶ ~~THE SEQUENCE — S1…S6~~ — **COMPLETE 2026-09-26; history, not the queue — the live queue is S7…S10 above.** Set 2026-09-25; S5-M/S5-W/S5-P inserted before S6 by the USER 2026-09-26 ("fix the bugs first"); S5-M-H inserted before S5-W by the USER 2026-09-26. One per fresh session, in order. ◀◀◀

**Each S-item has its own self-contained HANDOFF in `docs/superpowers/specs/` — gitignored, so THIS
row is the durable copy of why it exists and what "done" means.** Read the handoff first; it carries
the file:line evidence, the gates, the stop rules and the environment. **Do not reorder without a
reason written here.** The ordering rule used: *(a)* does it help the model that actually serves the
agent (Qwen3-Next, 9.06 tok/s), *(b)* is it cheap, *(c)* does it protect something already built.

**Why NOT DeepSeek-V4 first, stated once so it is not re-litigated:** it decodes at **0.3267 tok/s**,
**27.7× slower than Qwen3-Next**. A 500-token function takes ~25 minutes. It is a generality fixture —
the proof that Vaayu runs a 147 G two-quant-format fp4 MoE on 16 GB VRAM — not a production model. So
its two remaining gaps (S4, S5) sit BELOW the production-path work, and that is deliberate.

| # | item | why here | cost | handoff |
|---|---|---|---|---|
| **S1** ✅ | ~~**41.7 pageable HtoD memcpys per decode token** (Qwen3-Next)~~ **CLOSED 2026-09-25, NULL — really 84/token, and removing them LOSES 1.39 %. See the S1 row below** | cheapest thing on the board with a real win on the production model; pageable copies are SYNCHRONOUS, so the prize is host stall, not the 0.98 ms of device time | ~1 day, mostly measurement | [`…S1-pageable-h2d-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S1-pageable-h2d-HANDOFF.md) |
| **S2** ✅ | ~~**chunk-major byte counter survives an aborted expert injection**~~ **CLOSED 2026-09-25, REFUTED — the reset really is outside the `finally`, in all THREE forwards, but `_chunked_prefill`'s own `finally` already clears both fields. See the S2 row below** | it was on MiniMax's production prefill path — but the harm needed the state to outlive the abort, and it cannot | hours | [`…S2-chunk-major-counter-leak-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S2-chunk-major-counter-leak-HANDOFF.md) |
| **S3** ✅ | ~~**Qwen3-Next `GatedDeltaNet` elementwise fusion** — **STARTED 2026-09-25. STEP 1 DONE: census re-derived DECODE-ONLY and the handoff's does NOT reproduce — row 36's CUPTI arm had the prefill forward in its window (S1's flaw again), and its "653 µs/layer" was the whole decoder layer. Real: 4,586 launches / 15.9 ms device per `serial` token (not 7,702 / 27.70), so `serial` is ~55 % HOST gap — row 36's "GPU-bound" verdict used the contaminated number. GDN non-matmul = 63 launches/mixer (rule 29, norm 12, conv 4, body 18) = 49 % of `serial`'s launches; mixer host wall 438.8 µs = 15.80 ms/token on the real forward. CEILING MEASURED (ABBA, one load): dropping every GDN non-matmul op cuts `serial` 35.5 → 24.35 ms = −11.15 ms = 10.3 % of T_unprof 0.1085 s — ABOVE the handoff's 6.9 %. The three stock seams (`recurrent_gated_delta_rule`, `norm`, `causal_conv1d_update`) hold 45 of the 63. Detail `docs/2026-09-25-S3-census-rederive-RESULTS.md`. STEP 2 DONE (bake-off, user-approved incl. a scratch `fla` 0.5.2 install): one real mixer's decode call, ABBA x4 — stock 400.2 µs/66 kernels, narrow `torch.compile` 327.0 (−18 %), **Vaayu Triton prototype at the 3 seams 196.6 µs/24 kernels (−51 %)**, `fla` 391.1 (−2 %: it cuts launches but gives the time back in Python wrappers — REJECTED for decode). All closer to fp64 than stock. Two upstream traps: HF's `conv_states` is CHANNELS-LAST after any ≥4-token prefill (`fla`'s conv kernel silently corrupts it — kernels must be stride-aware), and the cache keeps the recurrent state in bf16. Banked lead: `fla`'s chunk kernel is ~10× the torch prefill fallback (0.36 → 0.03 s per 2k prompt, ~2 % TTFT). DESIGN APPROVED 2026-09-25 (scope A, user-chosen over B = body rewrite and C = `fla` prefill chunk, both banked): new `vaayu/gated_delta_kernels.py` — three stride-aware Triton kernels (rule with fp32 l2norm, conv update, gated norm) installed PER SEAM by surface detection (norm patched in place, streaming-safe), behavioural self-check at install falls back to stock with a WARNING, CUDA + triton only (CPU keeps stock), one call beside `install_state_continuation`; reaches qwen3_next/qwen3_5/qwen3_5_moe (3 seams), olmo_hybrid (2), MiniMax/dense/DeepSeek (0). Acceptance: conv state bit-exact, fused ≤ 1.25× stock's fp64 error over 2 geometries, D6 6-draw ppl band, ABBA real-token win ≥ 2 % (else revert), on-device win ≥ half the microbench's −204 µs/mixer. §8: no knob, reachable at N=1. Spec `docs/superpowers/specs/2026-09-25-S3-gated-delta-kernels-design.md`. Next: user reviews spec → implementation plan**~~ **CLOSED 2026-09-25, A WIN — real decode token −3.6 to −5.6 %, quality PASS. See the S3 row below** | the real perf build: **4,353 elementwise launches @1.8 µs = 7.80 ms/token (~6.9 %)** + 653 µs/layer CPU across 36 of 48 layers. Replaces row 36, which is PARKED on evidence | ~1 week | [`…S3-gateddeltanet-fusion-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S3-gateddeltanet-fusion-HANDOFF.md) |
| **S4** ✅ | ~~**restore the 46 DeepSeek-V4 tests** the shard deletion broke~~ **CLOSED 2026-09-26, FIXED — all 46 green, suite `2928 passed, 0 failed, 0 errors, 16 skipped`. One resolver in `tests/deepseek/parity.py` reads either layout; no `vaayu/` change. See the S4 row below** | the only coverage proving that checkpoint's numerics, on a model the engine now runs | ~1 day | [`…S4-parity-tests-split-shards-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S4-parity-tests-split-shards-HANDOFF.md) |
| **S5** ✅ | ~~**DeepSeek-V4 ships no `chat_template`** → the server 400s on every completion~~ **CLOSED 2026-09-26 — a PLAIN DOWNLOAD now loads and serves, with nothing written into the model dir. `vaayu_deepseek_v4.py` registers `DeepseekV4Config`/`DeepseekV4ForCausalLM` with transformers at import (no `auto_map`, no copied `.py`); a new `VaayuBaseModel.get_prompt_renderer()` hook (None default, 41 lines of shared code) lets the server render with DeepSeek's OWN `encoding/encoding_dsv4.py`, read in place. PROOF on a clean-download view (symlinks + upstream `config.json`): `test_non_stream` PASSED with a correct 16-token completion, and the real model dir was byte+mtime identical before/after. **NOT 9/9: the 2nd request OOM-killed at the 29 GB cap — the S5-M `[bug]` row below, pre-existing, not fixed here (filed as expert-RAM sizing; measured 2026-09-26 as glibc malloc retention on the clone path, and FIXED).** See the S5 row below** | the ONLY thing between that model and §8(3) reachability; fifth because nobody will serve a 0.33 tok/s model | ~1 day | [`…S5-chat-template-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S5-chat-template-HANDOFF.md) · spec [`…S5-portable-deepseek-v4-design.md`](docs/superpowers/specs/2026-09-26-S5-portable-deepseek-v4-design.md) |
| **S5-M** ✅ | ~~**DeepSeek-V4 through the server is OOM-killed on its 2nd request**~~ **CLOSED 2026-09-26, FIXED, MERGED (`2d78f71`) — not a reference leak: glibc malloc kept the freed expert clones (live tensors flat, `malloc_trim(0)` handed back 5.5 GB). Non-EXL3 module-list was denied the slab arena ("never measured"), so DeepSeek cached through the double-clone path. `6596c49` gives it the arena on CUDA. Through the probe's monkeypatched predicate (same answer for DeepSeek): RSS flat over 128 decode tokens, greedy ids identical over 48. On the committed code: server smoke 9/9 (RSS 26.67 GB at the last token). MiniMax never leaked (it already had the arena). See the S5-M row below** | S5 made DeepSeek serving reachable, and this capped it at ONE request per server process | days, GPU, systematic debugging | [`…S5-M-deepseek-server-oom-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-M-deepseek-server-oom-HANDOFF.md) · RESULTS `docs/2026-09-26-S5-M-deepseek-server-oom-RESULTS.md` |
| **S5-M-H** ✅ | ~~**DeepSeek-V4's RAM headroom on the arena path is unmeasured for LONG prompts**~~ **CLOSED 2026-09-26, PASS (measurement, no change), MERGED (`bc69193`) — one server process served 4,019 then 8,018 prompt tokens, both 200 (TTFT 166.9 s / 296.6 s). cgroup peak 27.36 GB = 3.78 GB margin to the 31.14 GB cap; the peak is the first request's warm-up (set inside A's first 512-token chunk), and no host-RAM column grows with prompt length at ≤ 8k (malloc free ≤ 0.47 GB, no clone churn). Host `MemAvailable` min 1.63 GB — not a FAIL; opened as the unsequenced **S5-M-HR** row. See the S5-M-H row below** — was: S5-M pinned the arena up front at host `available` − 3.5 GiB (23.0–23.5 GB); post-arena `available` 3.40 GB, RSS 26.67–28.28 GB vs the 31.1 GB cap, and every run used a prompt under ~40 tokens. A MEASUREMENT: one server process, a ~4k then a ~8k request, per-second RSS/HWM/pinned/malloc timeline + cgroup peak. PASS = both complete with ≥ ~1 GB margin; FAIL = root-cause, then STOP and report before any (derived) fix. **Starts with the S5-R check (§0)** | S5-M made DeepSeek serve past one request — this checks it survives the prefill-heavy workload Vaayu exists for, before more S5 work builds on it | ~1 hour, GPU, mostly waiting | [`…S5-M-H-long-prompt-headroom-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-M-H-long-prompt-headroom-HANDOFF.md) |
| **S5-W** ✅ | ~~**Vaayu still writes bytecode into a DeepSeek model dir**~~ **CLOSED 2026-09-26, FIXED — `no_bytecode()` holds `sys.dont_write_bytecode` around both `exec_module`s (the runtime `kernels()` and the parity harness); 4 new tests (RED→GREEN + a mutation-verified restore test). S5-R done the same session (user-authorised agent restore), so the proof is on the REAL dir and non-vacuous: DeepSeek files, a library load + generate and a server smoke 9/9 all left it byte+mtime identical with no `__pycache__`. Suite `2952 passed, 0 failed, 0 errors, 20 skipped`. Review: ready to merge, 5 minors all fixed. MERGED (`41b7488`). See the S5-W row below** — was: **Vaayu still writes bytecode into a DeepSeek model dir** — `kernels()` `exec_module`s the checkpoint's own `inference/kernel.py` whenever the attention tile needs no refit (bigger-smem GPUs; not this sm_120 box) → `inference/__pycache__/kernel.cpython-312.pyc`; the test harness writes `model.cpython-312.pyc` the same way. **Starts with the S5-R precondition check (§0)** | finishes S5's user-set promise: nothing written into a user's download | hours | [`…S5-W-model-dir-writes-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-W-model-dir-writes-HANDOFF.md) |
| **S5-P** ✅ | ~~**S5's deferred minors**~~ **CLOSED 2026-09-26, FIXED — an unloadable encoder now WARNs + 400s instead of stopping the load; the hook answers None before `init_model`; `documentation/server.md` has the §9 row, the §6.6 sentence, and §3's `refused` list corrected (user-approved). Suite `2955 passed, 0 failed, 0 errors, 20 skipped`. MERGED (`5fbb5c6`). See the S5-P row below** — was: an encoder that fails to IMPORT crashes the model load instead of refusing with a 400; `get_prompt_renderer()` raises AttributeError before `init_model`; `documentation/server.md` has no troubleshooting row / §6.6 note for the new 400 | cheap hardening of code S5 just shipped, and the user docs for its refusal | hours, no GPU | [`…S5-P-renderer-polish-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-P-renderer-polish-HANDOFF.md) |
| **S6** ✅ | ~~**row 28's reach-projection METHOD does not hold**~~ **CLOSED 2026-09-26, METHOD CORRECTED — one MISSING term, not a mis-sized one: row 28 never checked whether the non-expert weights fit the VRAM pin budget. On DeepSeek, 35 of 43 layers streamed 5.496 GB/token of non-expert weights through `inject_to_gpu` = `gpu_load` 2.20 s of the 3.06 s token (72 %), and 1.42 s of that is the CPU re-running `dequantize_at_load(wo_a)` on every streamed layer, every token. The handoff's three explanations did NOT reproduce: the expert bytes were HALVED by the caches (1.786 of 3.449 GB), not inflated by re-reads; row 28's 54 % miss assumption was right (51.8 %); and "2.7 s of 3.9–4.0" was the Uspin diagnostic arm, not GPU work. Corrected method cross-predicts DeepSeek 0.440 tok/s (+35 %; old 5.3–7.0×), Llama-70B +9 %, MiniMax −17 %, spec-decoding bound −4.8 %; Qwen3-Next misses 0.43× cross-forward (its dispatch constant has no donor) — the stated blind spot. gpt-oss-20b (already serving at 0.17 tok/s since A3): regime R like DeepSeek — whole model streamed per token + a per-visit MXFP4 swizzle; A6's per-expert streaming could take it to ~20 tok/s dispatch-bound. See the S6 row below** — was: it under-predicted DeepSeek's cost **5.4–6.4×** and got the binding resource wrong (GPU-bound, not NVMe-bound). Every future "can this checkpoint reach?" call uses it | ~half a day, no new runs needed | [`…S6-reach-projection-method-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S6-reach-projection-method-HANDOFF.md) |

**Suite floor: `3001 passed, 0 failed, 0 errors, 20 skipped` (212.27 s, FOREGROUND; set 2026-09-27 by S7's merge `9e6f81d`, +46 tests) — was `2955 passed, 0 failed, 0 errors, 20 skipped` (216.68 s,
FOREGROUND; set 2026-09-26 by S5-P's +3 tests) — was `2952 passed, 0 failed, 0 errors, 20 skipped` (209.57 s,
FOREGROUND; set 2026-09-26 by S5-W's +4 tests and S5-R's restore, which turned the 4 DeepSeek
stale-copy guards from passes into SKIPS — any other skip count means the DeepSeek folder changed) — was
`2952 passed, 0 failed, 0 errors, 16 skipped` (216.47 s), raised from 2759 by S1-L1's 17 new tests, from 2776 by S2's 1, from 2777 by S3's 97,
from 2874 by +5 from S3's final-review fix wave (2026-09-25), from 2879 by S4's 46 restored + 3 new,
from 2928 by S5's net +23, and from 2951 by S5-M's net +1 (2026-09-26; +2 at `6596c49`, −1 when its review
wave dropped a no-op parametrization).** **The suite is GREEN for the first time since `4bc70b1`: any failure is new.**
Wall time
rose ~16 s (192 → 208 s) because the 46 restored tests now read real weights off the split and run
DeepSeek's reference modules on the GPU. On-device gates run FOREGROUND under `systemd-run --user --scope -p MemoryMax=29G
-p MemorySwapMax=0`, one GPU job at a time; long scripts run as a DETACHED systemd unit. **No
fused-EXL3 arm exists** (no model on disk since Qwen3-235B was retired 2026-09-20) and **dense has no
committed gate** (`test_batched_dense_smoke.py` deleted 2026-09-03) — say both, claim neither.

**After S1…S6**, the remaining open work is the long tail already on the board: the `47-m*` minors,
the G3–G6 generality gaps, the `_inject_expert_weights` warn-and-skip row, the fused forward's twin
cast (`[!]`, blocked on G1), and the order-dependent `test_slab_arena` failure that did not reproduce
on 2026-09-25. Re-prioritise then, not now.

---

## ▶▶ PICK THIS NEXT: **S9** — DeepSeek `wo_a` folded on the CPU every token, handoff [`…S9-deepseek-wo_a-fold-on-gpu-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S9-deepseek-wo_a-fold-on-gpu-HANDOFF.md). ◀◀ (S8 CLOSED 2026-09-27, A WIN: gpt-oss routed fused-3D delivery, decode 9.61×/9.69×.) Was: **S8** — gpt-oss-20b at 0.17 tok/s on a model that fits the GPU, handoff [`…S8-gptoss-streamed-transform-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S8-gptoss-streamed-transform-HANDOFF.md). (S7 CLOSED 2026-09-27, A WIN: Qwen3-Next decode 1.94×.) Was: **S7** — Qwen3-Next per-expert dispatch, handoff [`…S7-qwen3next-expert-dispatch-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S7-qwen3next-expert-dispatch-HANDOFF.md). (Rule: the first row of the S7…S10 table at the top that is not ✅. S1…S6 COMPLETE 2026-09-26. S6's leads L-b/L-c/L-d stay unsequenced; L-a = S9.) Was: **S6** (user, 2026-09-26). (S5-P closed 2026-09-26, FIXED — the S5 bug-fix run is complete; S5-M closed 2026-09-26, FIXED; S5-M-H closed 2026-09-26, PASS — no prefill reserve needed up to 8k; S5-W closed 2026-09-26, FIXED; S5-R closed 2026-09-26 — this box's DeepSeek folder now matches a plain download)

**Set by the user 2026-09-26: fix S5's bugs before S6.** Each has a self-contained handoff (links in
the sequence table and in the S5 rows below). ~~Every session in this run starts with the S5-R check~~ —
**S5-R is DONE (2026-09-26); no session needs the check any more.** The DeepSeek folder is a plain download:
no `auto_map`, no copied `.py`, no `__pycache__`. Keep it that way — nothing may be written into it. **Order rationale:**
- S5-M first: it is the one that makes DeepSeek unusable (one request per server process), and it may
  reach MiniMax.
- S5-M-H next (user, 2026-09-26): S5-M's fix pins ~23 GB up front, and only short prompts were ever
  measured on it — confirm the long-prompt headroom before anything else builds on DeepSeek serving.
- S5-W after that: cheap, and it finishes S5's no-write promise.
- S5-P last: pure polish.

**S1, S1-L1, S2 and S3 are all closed 2026-09-25** — S1 a measured null, S1-L1 a WIN (+14.4 to +24.5 % prefill TTFT on Qwen3-Next, +17.5 to +18.9 % on MiniMax), S2 a REFUTATION, and **S3 a WIN** (real decode token −3.6 % to −5.6 % across four replicates (r1/r2 with their cold first arm excluded: −3.57 % / −3.68 %; r3/r4 with a warm-up arm: −4.32 % / −5.58 %), `serial` −4.3 ms, quality PASS, prefill flat, greedy ids identical — detail in the rows below). **S4 closed 2026-09-26, FIXED** (the 46 DeepSeek-V4 tests are green — row below). **The sequence resumes at S5** — DeepSeek-V4 ships no `chat_template`, handoff [`…S5-chat-template-HANDOFF.md`](docs/superpowers/specs/2026-09-25-S5-chat-template-HANDOFF.md) — then **S6**. **The suite floor is now `2928 passed, 0 failed, 0 errors, 16 skipped`** (208.49 s; +46 restored and +3 new from S4, 2026-09-26) — green, so any failure is new.

**One live lead S1-L1 did NOT chase, still open in the S1 row below: S1-L2** (96.5 pageable DtoH per decode forward). **S1-L3** (`fla` is not installed) is now answered in part by S3: `fla` was measured at the three seams and REJECTED for decode (−2 %, Python wrappers eat the launch savings), and its prefill chunk kernel is banked as **S3-C** (0.36 → 0.03 s per 2k prompt, ~2 % TTFT) rather than installed.

---

## S6 — `[x]` the reach-projection method, corrected and back-tested. **CLOSED 2026-09-26. No engine code changed, no on-device run. Suite not re-run: nothing under `vaayu/` or `tests/` was touched (floor stays `2955 passed, 0 failed, 0 errors, 20 skipped`).**

**Root cause — one MISSING term, not a mis-sized one.** Row 28 (2026-09-12) and row 28b step 1 both
projected DeepSeek-V4 as if every non-expert weight were VRAM-resident. Row 28 wrote the assumption down
(`docs/2026-09-12-row28-…-VERDICT.md` §4.3: "dense weights are 8.24 GiB … and those have to be
VRAM-resident"). **They were not.** `attn='eager'` raises the planner's reserve floor to 8.03 GiB to buy
back flash-parity context (`reserve_floor_bytes`, `vram_plan.py:538`), leaving a **3.33 GiB pin budget**.
Embed, head and 8 layers used it; **35 of 43 layers streamed their non-expert weights, 5.496 GB/token**.
The fallback RAM tier is an LRU of `_ne_budget` = 0.5 GB, and on a 5.5 GB cyclic scan it hits 0 %, so
every byte comes from the drive every token. Measured U1 (3.061 s): **`gpu_load` 2.20 s (72 %)** —
`inject_to_gpu` of those layers, NOT GPU work (`vaayu_base.py:5236-5241`, `:5275-5280`). **1.42 s of it
(46 % of the token) is the CPU re-running `dequantize_at_load(wo_a)`** (40.6 ms per layer, microbenched
2026-09-26 on layer 10), because a streamed layer is re-loaded every token. The rest moves ~6.67 GB at
8.6 GB/s. Compute + expert work is 0.68 s; `other` 0.21 s is unattributed.
**A4 check 2 had the residency finding BEFORE the run** ("0 non-expert-pinned") but concluded "RAM-cached
re-reads, not per-token NVMe streaming". That conclusion was wrong: the 0.5 GB LRU never hits.

**The handoff's three explanations did NOT reproduce** (raw data re-read: `decode-deepseek.json`, 384
forwards; `/tmp/a57c-row26.log`):
- **(1) "~2× bytes from re-reads" — WRONG.** Expert bytes from disk were **1.786 GB/token = 0.52× the
  3.449 GB working set**: the caches halve them. The extra **5.496 GB is non-expert**, constant to ±1 MB
  on every token of every arm, and it equals layers 8–42's summed non-expert bytes (5.494 GB).
- **(2) "thrashing, ~20 % hit" — WRONG.** 20 % is the RAM tier's hit rate among VRAM misses. The combined
  hit rate is 48 %, and row 28's assumed 54 % miss was right (measured 51.8 %).
- **(3) "2.7 s of a 3.9–4.0 s token was GPU work" — WRONG.** Those figures are the **Uspin** diagnostic arm.
  U1 is 2.20 of 3.06 s, and that is the non-expert inject, not per-expert H2D/dequant: all per-expert
  work is 0.58 s (19 %).
- Also stale: the §3 row-36 census (superseded by S1: 84 HtoD/token; S3: 4,586 launches / 15.9 ms per
  `serial`). MiniMax **1.794** is a 2026-09-03 server figure, and the same checkpoint's U arm measures
  2.081 (09-23). Llama-70B **0.155** is from 2026-08-16, and it measures **0.3044 on 2026-09-21**.

**THE METHOD (durable copy):**
```
T_token = T_fixed + T_ne + T_slot + T_h2d + T_disk            (s / decode token, N=1)
T_fixed = L · c_layer                                          resident non-expert work
T_ne    = N_stream · (b'/R_inject + x)                         NEW — non-expert OUTSIDE the pin budget
        + N_disk · max(0, b/R_read − (b'/R_inject + x + c_layer))   (prefetch is one layer ahead)
T_slot  = S · c_slot(forward)          T_h2d = S · (1−h_v) · e / R_h2d          T_disk = φ · S · m · e / R_busy
```
Steps: **(1) residency FIRST.** Non-expert bytes (headers) against the `pinning budget` a ~5 s load
prints (`[Vaayu-VRAM]` / `[Vaayu-Pin]`). The budget depends on the kernel as well as on geometry:
10.57 / 9.09 / 10.78 GiB on Qwen3-Next / MiniMax / Llama-70B at the 3.50 GiB floor, 3.33 on DeepSeek
under `eager`. A streamed layer is read from the DRIVE every token unless its RAM tier holds the whole
streamed set. **(2) transforms.** Every load-time weight transform in the model code
(`DEQUANTIZE_AT_LOAD`, casts, repacks) runs again per token on every streamed layer; CPU-microbench it.
**(3) routing.** At 8–16 % per-layer cache coverage, `m` ≈ 50–55 % and `h_v` ≈ 28–37 %, on three
checkpoints, DeepSeek out-of-sample. No rule exists above that (Qwen3-Next: 30–47 % coverage — its tiers overlap — `m` 2–3 %).
If the whole expert set fits the RAM tier, `m` → 0. **(4) name the binding term:** **R**
residency-bound (DeepSeek, dense 70B), **D** expert-byte-bound (MiniMax), **G** dispatch-bound
(Qwen3-Next).
Constants: hardware `R_busy` 6.14, `R_read` 6.2, `R_pcie` 20.1 GB/s. Per forward: `c_layer` 0.77–0.84 ms
(three checkpoints); `c_slot` **0.138** fused non-EXL3 / **0.438** module-list EXL3 / **0.588**
module-list fp4 ms; `R_inject` 15.7 (EXL3 path) / 8.6 (Standard path, DeepSeek) GB/s; `R_h2d`
(experts) 14.8 arena / 8.1 clone path; `φ` 0.23 MiniMax / 0.50 DeepSeek.

**Back-test** (tolerance: binding term right AND within **1.5×** cross-predicted, ±10 % in-sample. 1.5×
because it is 3.5–4.7× tighter than the error being fixed, the borrowed software constants themselves
span 1.8–4.3× between forwards, and engine drift on an unchanged checkpoint is the same size: MiniMax
+20 % in 12 days, Llama ×1.96 in 36):

| figure | cross-predicted (constants from elsewhere) | in-sample | binds | verdict |
|---|---|---|---|---|
| DeepSeek 0.3267 | **0.440 (+35 %)**; with `T_ne` dropped = row 28's error: 2.36 (+623 %) | 0 % (by construction) | `T_ne` | PASS |
| MiniMax 1.731 (1.794 server) | **1.440 (−17 % / −20 %)**; with DeepSeek's clone-path `R_h2d`: 1.07 (−38 %) | +6 % | expert terms | PASS |
| Qwen3-Next 8.91 (9.06) | **3.83 (−57 %)** — `c_slot` borrowed from module-list; no donor exists | −4 % | `T_slot` | **MISS 0.43× cross-forward — the stated blind spot** |
| Llama-70B 0.3044 (09-21) | **0.332 (+9 %)**, `R_inject` = `R_pcie` | +2 % | `T_ne` | PASS |
| Llama-70B 0.155 (08-16) | not retrodictable — no decomposition survives; the same checkpoint is 1.96× faster on today's engine | — | — | MISS, stated |
| spec decoding 5.57× | ≤ tokens/round = **5.30× (−4.8 %)**; a verify pass streams `T_ne` once | — | `T_ne` | PASS |

**Blind spot:** in the dispatch-bound regime, `c_slot` is software with no hardware-derived substitute.
A new checkpoint on an existing forward inherits that forward's `c_slot`; a new forward needs one
profiled arm first. **N>1 is not modelled** (MiniMax's 2.94–3.28 at N=3) — not reachable through the
server (§8).

**gpt-oss-20b, one line (CORRECTED same day — the first version said "cannot load, G1", which is
STALE: A3 closed 2026-09-21 and it serves at 0.17 tok/s):** TODAY it is **regime R, the same failure
as DeepSeek**. Whole-layer delivery streams the entire model every token (13.28 GB/token; embed and
head do not fit the pin budget). The byte floor is 13.28 / 6.2 ≈ 2.1 s → **≤ 0.47 tok/s**; measured
**5.8 s** (`compute` 5.2 s, `io_wait` 0.3 s). The ~3.7 s gap is the method's `x` term: the per-visit
`Mxfp4Deserialize` swizzle, GPU-side and not yet separated (A6's first measurement). So the method names
the regime and the binding term, but it gives only an upper bound until `x` is measured. **What A6 could
buy:** per-expert streaming + resident non-expert weights puts it dispatch-bound at ~0.049 s/token
≈ 20 tok/s (fused forward's `c_slot`; whole expert set fits the RAM tier), pending the MXFP4 dequant
cost per slot.

**Leads recorded, NOT chased (stop rule) — unsequenced, re-prioritise with the user:**
- [ ] **L-a — SEQUENCED AS S9 (2026-09-26). DeepSeek re-dequantizes `wo_a` on the CPU every token: 1.42 s/token (46 %).** Keeping the
      folded bf16 (35 × 67 MB ≈ 2.35 GB RAM), or folding on the GPU, removes it. Ceiling ≈ 0.6 tok/s.
- [ ] **L-b — DeepSeek's reserve floor may be over-charged.** 8.03 GiB comes from charging `eager` a
      materialized-score term. DSA attends to `index_topk` selected keys, so whether its real peak
      materializes S×S scores is UNVERIFIED. If it does not, up to 4.53 GiB of pin budget ≈ 30 more
      resident layers ≈ −1.9 s/token, traded against context length. Needs a VRAM measurement.
- [ ] **L-c — the planner's VRAM priority is unpriced.** DeepSeek's expert VRAM tier (3.5 GB) hits 34.9 %,
      while the non-expert layers it streams are used on 100 % of tokens.
- [ ] **L-d — DeepSeek's token is now 72 % per-FORWARD bytes**, like dense, so its built-in MTP draft would
      amortise that share (row 37's re-gate).
- Note: the 0.3267 predates S5-M (arena on CUDA), so today's DeepSeek decode is unmeasured.

**§8:** (1) no knob — no engine behaviour changed. (2) nothing to plumb; the method is the gitignored
probe `plans/probes/s6_reach_method.py` (no GPU, reproduces every number above). (3) all figures are N=1
decode, the server's only mode. (4) n/a.
Detail: `docs/2026-09-26-S6-reach-projection-method-RESULTS.md` (gitignored).

---

## S5 — `[x]` DeepSeek-V4 works from a plain download: in-process registration + the checkpoint's own prompt encoder. **CLOSED 2026-09-26. Suite `2951 passed, 0 failed, 0 errors, 16 skipped` (209.15 s). MERGED to `main` 2026-09-26 (`5b215b5`) after the final whole-branch review (Opus: with fixes → the encoder bytecode write, fixed); merged suite 2951 passed in 210.01 s.**

**Re-scoped by the user 2026-09-26:** "no write into the model directory; a user downloads the model and it
works with Vaayu". Found while re-verifying: a plain download did not even LOAD — this box ran on three
hand-placed edits (`auto_map` in `config.json`, two copied `.py` files), and no Vaayu doc ever told users to
make them.

**Built (branch `s5-portable-deepseek-v4`):**
- `1545e4e` — registration at import of `vaayu/vaayu_deepseek_v4.py`.
- `467e74a` — `get_prompt_renderer()` hook. `render_prompt(renderer=)` tokenizes with
  `add_special_tokens=False`; `tools` → named 400; `FileNotFoundError` → `NoChatTemplate` naming the file;
  the Jinja path is byte-identical.
- `2010e03` — `vaayu/model_code/deepseek_v4/prompt.py` loads the checkpoint's own `encoding/encoding_dsv4.py`
  in place (`enable_thinking` → `"thinking"`/`"chat"`). A MISSING encoder, or one without
  `encode_messages`, → load WARNING + 400 naming it. One that fails to IMPORT stops the model load with its
  own traceback (deferred minor).
- Final-review fix — the encoder is compiled and executed, NOT imported. An import had cached bytecode as
  `encoding/__pycache__/*.pyc` INSIDE the model dir, so the no-write promise was false on a fresh download.
  Pinned by `test_loading_the_encoder_writes_nothing_into_the_checkpoint`.
- `7115af5` — a fourth copy-drift test the spec missed, now a guard.
- Every copy-drift test is now a stale-copy guard: it skips when there is no copy.

**Proof:**
- Suite after the final-review fix: **2951 passed, 0 failed, 16 skipped** (209.15 s).
- DeepSeek's own 4 encoder goldens reproduce through `render()`.
- Clean-download view (scratchpad symlinks + the upstream `config.json`, no `auto_map`, no copies):
  - server `test_health` + **`test_non_stream` PASSED** (correct `reverse_string` code, 16 tokens,
    `finish_reason=stop`);
  - the real model dir's 133 paths were **identical** (size + mtime) before and after, and the HF
    remote-code cache gained nothing. **CORRECTION (final review):** that check was vacuous for the encoder:
    the agent's OWN design-phase probe (2026-09-26 09:28:31, before any S5 commit) had already written
    `encoding/__pycache__/encoding_dsv4.cpython-312.pyc` into this box's model dir, so the run found it
    current. The loader no longer writes it (fix above; a scratch copy of the real encoder stays clean). The
    stray pyc is removed by S5-R;
  - all DeepSeek test files on the view: **359 passed, 4 skipped** (the 4 guards), 0 failed.
- Qwen3-Next server smoke **9/9** (21.70 s).
- MoE gates: Qwen **8/8** (40.08 s), MiniMax **8/8** (74.95 s).
- No fused-EXL3 arm exists; dense has no committed gate.

**NOT met:** smoke 9/9 on DeepSeek. The 2nd request was OOM-killed at the 29 GB cap → **S5-M** below. The
real-dir DeepSeek smoke was skipped for the same reason.

**§8:**
1. No knob.
2. Plumbed at `engine.py:401`.
3. REACHABLE at N=1 (it was 0 % before) — but for ONE request per server lifetime at the 29 GB cap, until
   S5-M.
4. n/a.

Detail: `docs/2026-09-26-S5-portable-deepseek-v4-RESULTS.md` (gitignored).

- [x] **S5-R (USER, then agent) — restore this box's DeepSeek folder so it matches a user's download.**
  **CLOSED 2026-09-26 in the S5-W session — the user authorised the agent to run the restore. Merged with
  S5-W (`41b7488`).** Targets
  inspected first (`config.json` = upstream + `auto_map` + a re-indent; both `.py` copies byte-identical to
  git; `__pycache__` held only 3 pycs); the `auto_map` config saved to
  `docs/evidence/2026-09-26-S5-R-config.json.with-auto_map`; then step 1's command, verbatim. After: all five
  gone, 0 `auto_map`, 0 pycs, 133 → 125 paths. The optional items were then removed at the user's
  request, after checking neither was needed: `.goutputstream-LZUPV3` (917 MB of raw data dated
  2026-09-20, an interrupted GNOME write — no valid safetensors header, not open, referenced by
  nothing) and the DeepSeek entry in `~/.cache/huggingface/modules/transformers_modules/` (the old
  `auto_map` remote-code copies; the MiniMax entry kept). Re-checked after: DeepSeek files 364
  passed / 4 skipped, library load + generate OK, the cache entry not re-created, 124 paths.
  Step 2: DeepSeek files on the real dir **362 passed, exactly 4 guard skips, 0 failed** (39.70 s), no
  `__pycache__` gained; the deferred `configuration_deepseek_v4.py` docstring applied verbatim. Step 3: S5-W's
  real-dir proof done (S5-W row). Original text:
  1. The user runs:
     `cd /home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731 && mv config.json.orig-2026-09-23 config.json && rm configuration_deepseek_v4.py modeling_deepseek_v4.py && rm -r encoding/__pycache__ inference/__pycache__`
     (both `__pycache__` dirs are bytecode left by agent and test runs, not upstream files;
     `inference/__pycache__` was added 2026-09-26 with S5-W)
     (optional: remove `~/.cache/huggingface/modules/transformers_modules/deepseek_hyphen_ai_hyphen__hyphen_DeepSeek_hyphen_V4_hyphen_Flash_hyphen_0731`
     and the 917 MB `.goutputstream-LZUPV3`).
  2. The agent then re-runs `tests/test_deepseek_v4_*.py` against the real dir (expect 4 guard skips, 0
     failed), confirms `encoding/` gains no `__pycache__` (makes the no-write check non-vacuous), and applies the DEFERRED docstring update to `configuration_deepseek_v4.py`. It was reverted in
     S5 because the guard correctly flagged this box's copy as stale; the new text is in the S5 plan, Task 1
     Step 7.
  3. **Added 2026-09-26 by S5-W:** S5-W edits `modeling_deepseek_v4.py`, so once it is merged the three
     modeling-copy guards FAIL on the un-restored dir (the copy is `beaa65a`'s bytes). The restore turns
     them into skips — step 2's "4 guard skips, 0 failed" already expects that. Then run S5-W's real-dir
     no-write proof (S5-W row, handoff §3.4) as part of this follow-up.

- [x] **S5-M — DeepSeek-V4 through the server is OOM-killed on its SECOND request at the 29 GB cap:
  ~~a LEAK of ~0.19 GB per decode token outside the expert cache~~ — as filed; MEASURED as glibc malloc
  retention, not a leak (below)** (found by S5, 2026-09-26). It is
  PRE-EXISTING: before S5 no DeepSeek completion ever reached generation. **Sequence: FIRST of the S5
  bug-fix run.** Handoff: [`…S5-M-deepseek-server-oom-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-M-deepseek-server-oom-HANDOFF.md).
  - **STARTED 2026-09-26. ROOT CAUSE MEASURED — NOT a reference leak: glibc malloc retaining freed
    expert clones.** Library probe (`plans/probes/s5m_rss_probe.py`, clean view, expert cache capped
    at 8 GB so the plateau comes early; log `docs/evidence/2026-09-26-S5-M-ds-probe1-malloc.log`):
    - live CPU tensors (gc census) FLAT at 3,704 storages / 8.48 GB = cache 7.99 + non-expert LRU
      0.48, at tokens 8/16/24/32/40/48 → suspects 1, 3, 4, 5 REFUTED;
    - pinned host allocator FLAT at 1.61 GB → suspect 2 REFUTED;
    - `mallinfo2` free-in-arena grew 2.52 → 5.51 GB (tokens 5→31) with in-use flat at ~9.5 GB — RSS
      growth IS glibc free space; it saturated near 5.4 GB (~0.68× an 8 GB cache). At the server's
      21.9 GB budget that overhead never saturates before the 29 GB cap;
    - the engine's own every-32-pass `malloc_trim(0)` dropped RSS **18.06 → 12.56 GB** at pass 32, and
      the probe's at token 48 **17.08 → 12.54 GB**; RSS then refills ~1 GB/token. The server smoke died
      at pass 24, before any trim ran.
    - **Why only DeepSeek:** `moe_supports_slab_arena` (`vaayu_base.py:789`) gives EXL3 module-list
      (MiniMax) the pinned slab arena but NOT non-EXL3 module-list (DeepSeek) — "never measured" — so
      DeepSeek caches through `ExpertRAMCache.put()`: `load_expert_sparse` clones each component, then
      `put()` clones AGAIN, on executor threads; evictions free them and glibc keeps the pages. (Measured:
      free space summed over ALL arenas, and `m_mmap` flat at ~0.87 GB while ~8 GB of 4 MiB clones
      lived — so they came from the heap, not per-allocation mmap. Which arena held them: not measured.)
  - **MiniMax does NOT leak (checked first, per the handoff):** 128 decode tokens, RSS flat 20.87-20.99
    GB, `m_free` ≤ 0.05 GB, `malloc_trim` at token 96 freed 0.02 GB
    (`docs/evidence/2026-09-26-S5-M-minimax-probe.log`). Not a production-path bug.
  - [x] Hypothesis test: DeepSeek on the arena (predicate monkeypatched in the probe, no engine edit),
    server-default sizing (arena 1,754 × 13.4 MB pinned, 23.49 GB): **RSS 27.88-28.28 GB over 128
    decode tokens** (28.20 at token 1, peak 28.28 at token 2, 27.88 at 128), flat while the cache filled to 23.01 GB and churned; malloc free ≤ 0.26 GB;
    `malloc_trim` at token 96 freed 0.01 GB. **Greedy ids identical to the clone path over 48 tokens.**
    ~3.0 s/token on both paths, but not an A/B (budgets differ) — no speed claim.
    (`docs/evidence/2026-09-26-S5-M-ds-probe2-arena.log`, `…-probe3-clone-ids.log`)
  - [x] **Fix `6596c49` (branch `s5m-deepseek-arena`):** `moe_supports_slab_arena` non-EXL3 →
    `cuda and style in ("fused", "module_list")`. CPU compute still gets no arena. Gate test rewritten
    (a CUDA case + a no-CUDA case): RED 2 failed before, GREEN after. (The review wave `c775585` dropped
    a 43/62 layer-count parametrization: the predicate reads only the style, so it was coverage in name
    only.)
    Suite **2953 passed, 0 failed, 0 errors, 16 skipped (215.31 s) FOREGROUND**; after the review wave
    (`c775585`) **2952 passed, 0 failed, 0 errors, 16 skipped (216.47 s)**.
  - [x] **Gates, all on `6596c49`:** DeepSeek clean-view server smoke **9/9** (113.19 s; arena 1,721 ×
    13.4 MB pinned; RSS 27.07 → **26.67 GB at the last token**, flat over 32 passes / 2 requests —
    before: 5.50 → 29.75 GB and OOM); MoE gates Qwen **8/8** (43.62 s), MiniMax **8/8** (74.45 s);
    server smokes Qwen **9/9** (22.25 s), MiniMax **9/9** (52.36 s). The ≥128-token check is §4's probe
    run (its monkeypatched predicate answers exactly what the commit does for DeepSeek). No fused-EXL3
    arm exists and dense has no committed gate; neither is claimed, and the change reaches neither.
  - **§8:** (1) no knob — the arena is derived from layout + device, and the only thing a knob could
    offer is the clone path, which OOMs this model; (2) nothing to plumb — the predicate runs in
    `init_model` (`vaayu_base.py:2718`), proven reachable by the smoke's `Expert slab arena` line;
    (3) REACHABLE at N=1 — every DeepSeek server load on CUDA, and DeepSeek is no longer limited to one
    request per server process; (4) n/a.
  - Detail: `docs/2026-09-26-S5-M-deepseek-server-oom-RESULTS.md` (gitignored). **MERGED to `main` 2026-09-26
    (`2d78f71`) after the whole-branch review (ready to merge, no Critical; its Important → S5-M-H, its
    minors fixed in `c775585` / `9d1b6f4`); merged suite 2952 passed, 0 failed, 16 skipped in 209.40 s.**
  - [bug] **S5-M-C — the CLONE path itself still fragments glibc arenas** (found by S5-M, NOT fixed by
    it). Any cache with `arena is None` stores `load_expert_sparse`'s clones cloned a SECOND time by
    `ExpertRAMCache.put()` (`persist/expert_cache.py:577`), on executor threads; evictions free them
    and glibc keeps the pages. Measured on DeepSeek at an 8 GB cache: +5.4 GB of RSS held as malloc
    free space (~0.68× the cache). After S5-M no on-device model reaches it on CUDA, but CPU compute
    (every layout — `moe_supports_slab_arena` answers False without CUDA) and the arena-exhausted
    fallback still do — and so does a MIXED or UNKNOWN expert layout on CUDA (the predicate answers
    False for both). Candidates, unmeasured: set glibc's `M_MMAP_THRESHOLD` (`mallopt` or
    `MALLOC_MMAP_THRESHOLD_`) at or below the expert component size, DERIVED not literal — it turns off
    the dynamic threshold, so each clone gets its own mmap and returns to the OS on free, which fits
    probe 1's flat `m_mmap`; drop `put()`'s redundant second clone (halves the churn, does not end it);
    or a pageable `SlabArena` for arena-less layouts. Unblocks when a CPU-compute model is run here,
    or the fallback is seen in a log.
  - [x] **S5-M-H — DeepSeek's RAM headroom on the arena path is unmeasured for LONG prompts** (**Sequence:
    FIRST of the remaining S5 run**, set by the user 2026-09-26.) **CLOSED 2026-09-26, PASS — a
    measurement, no engine change** (branch `s5mh-long-prompt-headroom`). **MERGED to `main` 2026-09-26
    (`bc69193`) with the S5-M-HR row; todo.md-only, so no suite or gate run.**
    - **Pre-work:** S5-R check = all five artefacts present → S5-R stays `[!]`, so the run used the
      clean-download view. Handoff anchors re-verified against `3e9f4c9`. One handoff figure was wrong:
      code tokenizes at **3.85** chars/token with DeepSeek's `tokenizer.json`, not 3.3. The probe cut
      the prompts by the tokenizer instead.
    - **Run:** one server process (`create_app` + `TestClient`), detached unit `MemoryMax=29G`,
      `MemorySwapMax=0`, per-second in-process sampler plus an external host `MemAvailable` sampler.
      Arena 1,652 × 13.4 MB = 22.11 GB pinned (ceiling 21.86 GB from `available` 25.62); derived
      `max_seq_len` 16,766.
    - **Both requests returned 200:**

      | request | prompt_tokens | TTFT | per 512-token chunk | decode |
      |---|---|---|---|---|
      | A | 4,019 | 166.9 s | 20.2–21.4 s | 2.85 s/tok |
      | B | 8,018 | 296.6 s | 16.6–19.1 s | 2.81 s/tok |

      B's chunks found ~16 % of their experts in the RAM cache A's decode filled (A's: 0 %); coincident
      with B's faster chunks, not isolated as the cause.
    - **Headroom: cgroup peak 27.36 GB** (in-process `memory.peak` = `MemoryPeak` 27,360,894,976 B =
      journal `25.4G`) → **3.78 GB margin** to the 31.14 GB cap. PASS needs ≥ 1 GB. Not borderline
      (the repeat rule is < 2 GB), so one draw stands.
    - **The peak is the FIRST request's warm-up, not prompt length.** It was set 9 s into A's first
      512-token chunk, with `RssAnon` 0.97 → 2.75 GB. Nothing later exceeded it: 23 more chunks, 2× the
      prompt, and both decodes. Peak RSS was 27.10 (A prefill) vs 27.10 (B prefill), `VmHWM` 27.24.
    - **Hypotheses:**
      - **(1)** The legacy per-chunk loop is what runs; its cost is time, not RAM.
      - **(2)** Staging-fallback clone churn was NOT seen. malloc free stayed 0.14–0.47 GB over all 24
        prefill chunks, 17 of them before the run's only engine `malloc_trim` (`vaayu_base.py:5451`,
        every 32 passes). S5-M's clone path went 2.52 → 5.51 GB.
      - **(3)** No per-prompt-token host cost is measurable at ≤ 8k: `RssAnon` peak 2.96 → 3.09 GB for
        +3,999 tokens, inside the decode jitter.
      - **(4)** Pinned stays flat at 23.74 GB (arena + 1.63 GB).
    - **→ No prefill reserve is needed in `derive_expert_ceiling` up to 8k. Nothing built.**
    - **Observation, NOT a FAIL** (outside this item's question; the user opened **S5-M-HR** for it, below): the
      HOST-level margin is thinner than the cgroup one.
      - host `MemAvailable` min **1.63 GB**;
      - `SwapFree` 4.72 → 4.57 GB, i.e. ~0.15 GB of OTHER processes swapped out (the unit used 0 swap).
      - Cause: the 3.5 GiB reserve left after the arena is ~2.1 GB consumed by this process's own
        prompt-independent warm-up. Because the ceiling tracks `available`, that ~1.6 GB host margin
        holds at any load-time `available`.
    - **§8:** n/a — no change; the measurement ran through the server at N=1, its only mode.
    - **Scope:** DeepSeek only. No suite or gate run (no code change). No fused-EXL3 arm and no dense
      gate; neither claimed. The real model dir had 0 files newer than the launch. Not measured: > 8k
      and N>1.
    - Detail: `docs/2026-09-26-S5-M-H-long-prompt-headroom-RESULTS.md`; evidence
      `docs/evidence/2026-09-26-S5-M-H-{long-prompt.log,host-meminfo.log,probe.py}` (all gitignored).

    **Original row text:** Handoff:
    [`…S5-M-H-long-prompt-headroom-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-M-H-long-prompt-headroom-HANDOFF.md).) (review of
    S5-M, 2026-09-26). The arena is pinned at load at host `available` − 3.5 GiB
    (`derive_expert_ceiling`, `vaayu_base.py:355`; no prefix reserve, since DeepSeek's cache has no
    `to_cpu`). Measured: post-arena host `available` 3.40 GB; RSS 26.67-28.28 GB against the 29 GiB
    (31.1 GB) cap, i.e. ~2.8-4.4 GB of headroom. EVERY run so far used a prompt under ~40 tokens;
    Vaayu's workload is prefill-heavy. **Closes when:** one long-prompt DeepSeek server request (4-8k
    tokens) completes with RSS recorded per token. If it OOMs, the fix is a MEASURED prefill reserve in
    the derivation — never a blind cut to the ceiling (handoff §6).
  - [ ] **S5-M-HR — the HOST RAM left after the expert arena is ~1.6 GB, not the 3.5 GiB the reserve
    names** (opened by the user 2026-09-26 from S5-M-H's observation. **Not in THE SEQUENCE**; the user
    places it.)
    - **Measured (S5-M-H, DeepSeek-V4, server path):**
      - `derive_expert_ceiling` (`vaayu_base.py:355`) pins the arena at host `available` −
        `_EXPERT_CEILING_RESERVE` (3.5 GiB = 3.76 GB, `:352`). DeepSeek has no prefix pool, so that is
        the whole reserve.
      - The process's own FIRST-REQUEST warm-up then spends ~2.1 GB of it: `RssAnon` 0.97 → ~3.0 GB,
        inside request A's first 512-token chunk. It did not grow with 2× the prompt.
      - Host `MemAvailable` min **1.63 GB** (1.63–2.10 through both requests); `SwapFree` 4.72 → 4.57
        GB, so the kernel swapped ~0.15 GB of OTHER processes out. The unit was capped at 0 swap and
        used 0.
      - The cgroup margin was 3.78 GB (peak 27.36 of 31.14). The 29 GiB cap is the harness's; a real
        server runs uncapped and lives on the host figure.
    - **Why it holds at any load:** the ceiling tracks `available`, so the post-warm-up host margin is
      ~reserve − warm-up ≈ 1.6 GB whatever `available` reads at load. It is thin, not negative: nothing
      OOMed.
    - **Unknown (measure before any fix):**
      - what the ~2.1 GB warm-up is (not decomposed; the non-expert LRU is ≤ 0.5 GB of it);
      - whether it has the same size on Qwen3-Next and MiniMax;
      - on models WITH a prefix pool, whether the warm-up eats into the pool's reserved sessions or the
        OS's share.
    - **Fix shape, if one is warranted:** DERIVE a warm-up term into the reserve from a measured
      post-first-request growth, with a test over two geometries (CLAUDE.md §5). Or WARN at load when the
      post-warm-up margin is under a stated floor. **Never** a blind raise of `_EXPERT_CEILING_RESERVE`
      or cut to the ceiling.
    - **Closes when:** the warm-up is decomposed and measured on ≥ 2 models, and either the reserve
      derives it or the ~1.6 GB margin is written down as accepted. **§8 on a fix:** no new knob
      expected; the derivation already runs in every server load.
    - Evidence: `docs/2026-09-26-S5-M-H-long-prompt-headroom-RESULTS.md` §5,
      `docs/evidence/2026-09-26-S5-M-H-host-meminfo.log`.
  - **Evidence** (clean-view server smoke, detached unit `MemoryMax=29G`; log kept at
    `docs/evidence/2026-09-26-S5-M-clean-view-smoke-oom.log`):
    - `test_health` and `test_non_stream` PASSED;
    - the 2nd request (`test_stream`) was OOM-killed at `memory peak 29.0G`, 95 s in.
  - **Sizing at load:** `Expert cache ceiling 21.90 GB (derived: available 25.66 GB, reserving 0 prefix
    session(s))` on top of `rss=3.00GB`.
  - **Per-token growth** (from the `token summary` lines' `usage=`/`rss=`):

    | token | rss − cache |
    |---|---|
    | 1 | 5.50 GB |
    | 16 | 8.41 GB |
    | 24 | **9.90 GB** |

    The cache PLATEAUS at 19.85/21.90 GB from token 16 (the `max 38/layer` cap binds first), yet RSS
    rises +1.49 GB over tokens 16→24 with the cache flat. That is a leak or unbounded accumulation of
    ~0.18–0.19 GB/token (as read at filing; measured cause: glibc malloc retention, above). At that rate no budget survives a long answer.
  - **Secondary:** `derive_expert_ceiling` (`vaayu_base.py:355`) uses host `available` minus 3.5 GiB, and
    reserves nothing when the prefix cache is off. (Tracked as **S5-M-H** above — closed 2026-09-26, PASS: no prefill reserve needed up to 8k.)
  - **Ranked suspects (as filed — ALL FIVE REFUTED 2026-09-26 by the measurements above):**
    1. evicted module-list CLONES still referenced;
    2. pinned-host caching-allocator growth from streaming non-expert layers 8–42 every token;
    3. the non-expert `LRUTensorCache` overrunning its 0.5 GB budget;
    4. DeepSeek CPU-side cache/ContextVar state;
    5. Python-level accumulation.
  - **Check MiniMax early:** same module-list forward. If it leaks too, this is production — tell the
    user and re-rank.
  - **Done means:**
    - DeepSeek clean-view server smoke 9/9;
    - ≥128-token decode with `rss − cache` flat after the plateau;
    - if shared code changed: MoE gates plus the MiniMax long-decode check;
    - suite green;
    - §8.
  - Do NOT "fix" it by lowering the ceiling.

- [x] **S5-W — Vaayu still writes bytecode into a DeepSeek model dir** (found 2026-09-26, while writing
  S5's follow-ups). **Sequence: SECOND.** Handoff:
  [`…S5-W-model-dir-writes-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-W-model-dir-writes-HANDOFF.md); its §0 is the S5-R check.
  - **STARTED 2026-09-26 (branch `s5w-model-dir-writes`, off `beaa65a`).** S5-R check: **all five
    artefacts present** (both `__pycache__` pycs still dated 09-23) → S5-R stays `[!]`; on-device runs
    use a clean-download view. Handoff re-verified against `beaa65a`: `kernels()` `:149`, spec `:189`,
    `exec_module` `:192`, parity `:239-243` all hold; `grep exec_module|spec_from_file_location vaayu/`
    = that one site only; suite floor still 2952/0/0/16.
  - [x] RED: no-refit `kernels()` writes no `__pycache__` beside a stub `kernel.py` — failed on
    `['__pycache__', 'kernel.cpython-312.pyc', 'kernel.py']`; its `__file__` assert passed, so the stub
    took the no-refit branch on this CUDA box
  - [x] RED: `parity.reference_modules(ckpt=<stub>)` writes no `__pycache__` into `inference/` (a stub
    checkpoint in `tmp_path` makes this RED-able without S5-R — the handoff expected it needed one) —
    failed on `['__pycache__', 'kernel.py', 'model.py']`
  - [x] GREEN: one `no_bytecode()` context manager in `modeling_deepseek_v4.py` around both
    `exec_module` calls (parity imports it) — 2 passed. ~~Both tests also pin the flag is restored.~~
    **Overstated (review):** they only checked `is False` from a `False` start, so a `finally` restoring the
    constant `False` survived. Fixed in the review wave below.
  - **Found while doing it:** editing `modeling_deepseek_v4.py` at all makes its THREE stale-copy guards
    (`test_deepseek_v4_{linear,moe,model}.py`) FAIL on this box's real dir — its hand-placed copy is
    byte-identical to `beaa65a` and S5-R has not removed it. The guards are right (on the real dir that
    copy, not the fix, is what `auto_map` loads). Same trap that made S5 revert its docstring edit.
    Unavoidable here: `kernels()` lives in that file. Gates therefore run twice (real dir + clean view).
  - [x] `tests/test_deepseek_v4_*.py` on a fresh clean view: **362 passed, 4 skipped (the 4 guards), 0
    failed** (32.89 s); every TileLang/GPU golden ran → **the JIT works under the flag** (handoff §6 stop
    rule not hit). The view has REAL `inference/`+`encoding/` dirs of per-file symlinks (S5's symlinked the
    dirs, so a write would have hit the real dir). **Non-vacuous:** with the fix stashed, the same run on a
    fresh view wrote `inference/__pycache__/model.cpython-312.pyc` into it; with the fix, nothing.
  - [x] full suite FOREGROUND, 29G scope — **real dir: 2951 passed, 3 failed, 0 errors, 16 skipped
    (210.32 s)** = floor 2952 + 2 new, the 3 failures EXACTLY the modeling-copy guards above; **clean
    view: 2950 passed, 0 failed, 0 errors, 20 skipped (208.54 s)** = 16 + the 4 guards skipping, which is
    what the real dir gives once S5-R deletes the copies.
  - [x] extra: DeepSeek server smoke on the clean view **9/9** (114.44 s); view, real dir (133 paths,
    size+mtime) and the HF modules cache unchanged. On this sm_120 GPU `kernels()` always refits into a
    tempdir, so NO on-device run here exercises the checkpoint's own `kernel.py` branch — the unit test
    (a stub takes it on any box) is the only proof of that branch. Real-dir diff vacuous for the pycs.
  - [x] on-device no-write proof on the REAL dir (handoff §3.4) — ~~blocked on S5-R~~ **DONE after S5-R, same
    session, and NON-vacuous (no `__pycache__` existed first).** `find $M -printf '%p %s %T@'` identical
    (125 paths), no `__pycache__`, HF modules cache unchanged, across: the DeepSeek test files (real
    `inference/model.py` imported in place by the harness); a library load + 8-token greedy generate (load
    8.4 s, `return s[::-1]`); the DeepSeek server smoke on the real dir **9/9** (114.01 s); and both suites
    below.
  - [x] **Whole-branch review** (fresh reviewer, `beaa65a..6d5e10b`): **ready to merge**, 0 Critical, 0
    Important, 5 Minor — ALL fixed: (1) new `test_no_bytecode_puts_the_flag_back_as_it_found_it_even_when_the_import_fails`
    (start flag False/True, `ImportError` inside) — mutation-verified to kill hard-coded `False`, hard-coded
    `True` and a missing `finally`; (2) `no_bytecode()` docstring now says it covers the file's whole
    first-import graph (tilelang: ~0 on a pip-compiled venv, ~+0.4 s/start uncompiled) and is not
    thread-safe (every caller is on the one-request forward path); (3) the parity test asserts the
    import ran (`module.__file__`); (4) that test no longer leaks a `/tmp/a5_kernel_*` (272 → 272);
    (5) `modeling_deepseek_v4.py`'s module docstring lost the same stale "copied verbatim" claim.
  - [x] **Suites on the restored real dir, FOREGROUND:** before the review wave **2950 passed, 0 failed, 0
    errors, 20 skipped** (217.81 s, as the clean view predicted); final code **2952 passed, 0 failed, 0
    errors, 20 skipped** (209.57 s) = the NEW FLOOR.
  - **§8:** (1) no knob — nothing to tune, the only visible effect is a clean model dir; (2) nothing to
    plumb — `kernels()` is on the server's DeepSeek load path (smoke: `DeepSeek-V4 kernels found at
    …/inference`); (3) REACHABLE at N=1 on every DeepSeek load that imports the kernels — on a GPU whose
    shared memory fits DeepSeek's own tile the download no longer gains `inference/__pycache__/kernel.*.pyc`
    (this box's GPU was already clean via the refit tempdir); a dir still carrying `auto_map` + a copy
    (this box's real dir only) runs the stale copy instead; (4) n/a.
  - **MERGED to `main` 2026-09-26 (`41b7488`)**, with S5-R's docstring commit, after the review — the
    order the user chose (review → restore → proof → merge). The merged tree is byte-identical to the
    branch tip the final suite ran on (`git diff --quiet`), so that suite is main's.
  - Minor, pre-existing, unsequenced: `parity.reference_modules()` leaks one `/tmp/a5_kernel_*` dir per
    REAL call (250 at session start, 272 by the end). The new stub test no longer adds to it.
  - Detail: `docs/2026-09-26-S5-W-model-dir-writes-RESULTS.md`; evidence
    `docs/evidence/2026-09-26-S5-W-{viewA-prefix,viewB-postfix,suite-real,suite-view,smoke-viewB,s5r-real-dsfiles,s5w-library-load,s5w-smoke-real,suite-restored,suite-final}.log`,
    `…-{real-before,real-pre-restore,real-final-before}.txt`, `…-make_view.sh`, `…-s5w_library_load.py` (all gitignored).
  - **Runtime:** `modeling_deepseek_v4.kernels()` (`:149`, `exec_module` at `:192`) imports the
    checkpoint's own `inference/kernel.py` whenever `_fit_sparse_attn` leaves the text unchanged (GPUs
    with enough shared memory). That writes `inference/__pycache__/kernel.cpython-312.pyc` into the
    user's download. On this sm_120 box the tile IS refitted into a tempdir, so it is invisible here.
  - **Tests:** `tests/deepseek/parity.py:239-243` writes `inference/__pycache__/model.cpython-312.pyc`
    the same way.
  - **Recommended fix:** `sys.dont_write_bytecode = True` around `exec_module`, restored in `finally`.
    NOT S5's `exec(compile())`: TileLang's JIT calls `inspect.getsource`.
  - **Test first:** the no-refit branch writes no `__pycache__` beside a stub `kernel.py`.
  - **Proof:** the non-vacuous no-write check on the real dir, which needs S5-R.
  - **OUT of scope:** the by-design `splitted_model/` cache, `DELETE_ORIGINAL`, and
    `vaayu_expert_offsets.json`. Moving the split out of the model dir would need a server knob
    (`layer_shards_saving_path` is library-only); that is the user's decision.

- [x] **S5-P — S5's deferred minors (final review, 2026-09-26).** **Sequence: THIRD. CLOSED 2026-09-26, FIXED — A and B reproduced exactly as the handoff states (A: `ModuleNotFoundError` and `SyntaxError` both escaped `load_prompt_renderer` out of `prompt.py:59`; B: bare `AttributeError` at `vaayu_deepseek_v4.py:437`), 3 new tests RED → GREEN, suite `2955 passed, 0 failed, 0 errors, 20 skipped` (216.68 s, FOREGROUND, capped) = floor + 3, skip count unchanged. MERGED to `main` 2026-09-26 (`5fbb5c6`; merge tree identical to the tested branch tip `b48e95a`).** Handoff:
  [`…S5-P-renderer-polish-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S5-P-renderer-polish-HANDOFF.md).
  - **(A)** An encoder that fails to IMPORT (SyntaxError, or a missing third-party import) crashes
    `init_model` → route it through `_refuse` (WARNING + 400) like a missing file.
    `vaayu/model_code/deepseek_v4/prompt.py:57-59`; keep the `exec(compile())` form.
  - **(B)** `get_prompt_renderer()` raises AttributeError before `init_model` → class attribute
    `_prompt_renderer = None` (`vaayu/vaayu_deepseek_v4.py:434`).
  - **(C)** `documentation/server.md`: a §9 troubleshooting row for the 400 naming
    `encoding/encoding_dsv4.py`, plus one sentence in §6.6 that a model-rendered prompt refuses `tools`.
  - TDD for A and B. No GPU. Suite floor `2952 passed, 0 failed, 0 errors, 20 skipped` (S5-W + S5-R,
    2026-09-26; the handoff's §5 is updated to match).
  - **Done:** (A) the `exec(compile(...))` is wrapped in `except Exception` → `_refuse` naming the path
    and the error type; the no-bytecode form is kept (`test_loading_the_encoder_writes_nothing_…` still
    green). (B) `_prompt_renderer = None` as a class attribute, so the hook answers the base's None and
    `render_prompt` falls to the ordinary "ships no chat_template" 400. (C) §9 row (covers the missing
    AND the unloadable encoder, since (A) now makes both a 400 naming the file) + one §6.6 sentence.
    Tests: `test_an_encoder_that_cannot_load_is_refused_not_a_crashed_load` ×2 (import, SyntaxError),
    `test_the_renderer_is_none_before_init_model`, in `tests/test_deepseek_v4_prompt.py`.
  - **One deviation from the handoff, user-approved 2026-09-26:** its §3 said `documentation/server.md`'s
    `refused`/`error` paragraph "needs no change". Re-verified, it was INCOMPLETE — it said `refused` is
    "the three cases named above" (oversized prompt, missing `session_id`, full queue), but the code
    files `TemplateLacksToolSupport` and `NoChatTemplate` as `refused` too (`engine.py:618-632`,
    `app.py:961-965`). Both predate S5. Fixed with a two-line edit (the table cell + "the client — or,
    for a missing chat template, the operator — can fix").
  - **§8:** (1) no knob. (2)/(3) reachable at N=1 through `/v1/chat/completions` on DeepSeek-V4 —
    `init_model` always runs under the server, so (A) is what a user with a broken encoder now sees
    (server starts, WARNING, 400) instead of a server that never starts. (B) is library/tooling-only;
    unreachable through the server by construction. (4) n/a.
  - **No on-device gates:** `prompt.py` and `vaayu_deepseek_v4.py` are DeepSeek-only and change no
    behaviour on Qwen3-Next, MiniMax or dense; the rest is docs. No fused-EXL3 arm exists; dense has no
    committed gate — neither was claimed.

## S4 — `[x]` restore the 46 DeepSeek-V4 tests the original-shard deletion broke. **CLOSED 2026-09-26, FIXED — 46/46 green, suite `2928 passed, 0 failed, 0 errors, 16 skipped` in 208.49 s FOREGROUND. Test-only change, no `vaayu/` edit. MERGED to `main` 2026-09-26 (`906675c`).**

**Root cause (re-verified 2026-09-26, not taken from the handoff):** all 46 died with
`FileNotFoundError: …/model-000NN-of-00048.safetensors` at fixture time — every reader resolved a key
through `model.safetensors.index.json` (still on disk) to an original shard `4bc70b1` deleted. **The
handoff named ONE reader; there were SEVEN**: `tests/deepseek/parity.py:96` (23 tests),
`test_deepseek_v4_attention.py:467` (16), `test_deepseek_v4_linear.py:324` (2),
`test_deepseek_v4_block.py:64` (2 errors), `test_deepseek_v4_checkpoint.py:171` (1),
`test_deepseek_v4_model.py:376` (1), `:568` (1) — four of them copy-pasted raw readers.

**Fix (Route A, with a correction):** one resolver in `tests/deepseek/parity.py` — `locate(key)` returns
the ORIGINAL shard while it exists, else the `splitted_model/` file holding
`remap_checkpoint_key(key)` (split index read off the 46 headers, ~0.7 s, cached — the split's file
naming is not re-derived); `read_checkpoint_tensors` / `raw_module_weights` on top; all seven sites
call it. **Why BOTH layouts, not the handoff's "point the harness at the split":** the server's default
`delete_original=True` (`app.py:115`) leaves split-only (this box), the library default keeps the
originals, and two tests (`test_the_real_checkpoint_measures_32`, `model.py`'s group-size helper, now
`_fp4_group_size_from_checkpoint`) exist precisely because "the split may not exist yet". A split-only
resolver would have re-broken a fresh download. **Route B (commit fixtures) not taken:** the goldens —
the OUTPUTS — are already committed; what went missing is WEIGHTS, GiB per MoE layer.

**Evidence:** the 7 files 44 failed + 2 errors → **162 passed** (113 + 46 + 3 new) in 34 s; every
bit-identity golden passes (`test_a_golden_regenerates_bit_identically`,
`test_the_moe_is_bit_identical_to_the_reference_…`), so the split reproduces goldens recorded off the
originals. **3 new CPU tests** in `test_deepseek_v4_parity.py` prove the layout this box cannot: fake
checkpoints, fresh-download vs split-only, three remap shapes (`layers.*`→`model.layers.*`,
`head.weight`→`lm_head.weight`, `hc_head_*`→`model.norm.hc_head_*`), and a missing key names itself.
`plans/probes/a5_5_attn_gate.py` imports `parity` and is fixed by the same change; the other probes
that read originals (`a57b_split_dryrun.py`, `a57b_verify_split.py`, `row28_dsv4_projection.py`,
`task6f_detect_probe.py`) are one-shot historical scripts, left alone.

**Handoff corrections banked:** its §4 gate (2805 passed) predates S3 — the real target was 2925 + the
3 new = 2928; its §6 "no GPU needed" is wrong — the golden tests build DeepSeek's reference modules on
CUDA.

**§8:** (1) no knob — test harness only, nothing a user reaches; (2) n/a; (3) n/a — no engine
behaviour changed; (4) n/a. **No gates beyond the suite:** no `vaayu/` line changed, so no on-device
arm is owed.

Detail: `docs/2026-09-26-S4-split-shard-tests-RESULTS.md` (gitignored — this row is the durable copy).

## S3 — `[x]` fused Triton kernels on the gated-delta mixers' three decode seams. **CLOSED 2026-09-25, A WIN — real decode token −3.6 % to −5.6 % across four replicates (r1/r2 with their cold first arm excluded: −3.57 % / −3.68 %; r3/r4 with a warm-up arm: −4.32 % / −5.58 %), `serial` −4.3 ms, prefill flat, greedy ids identical, D6 quality band PASS. MERGED to `main` 2026-09-26 (`87a7f4d`) after the final whole-branch review (Opus: ready with fixes → I1 real-geometry self-check, I2 non-current-device guard, M1-M5, all fixed and re-reviewed).**

**Why (measured 2026-09-25, HEAD `18ef01c`, detail `docs/2026-09-25-S3-census-rederive-RESULTS.md`):**
- **The row-36 spike's census was wrong.** Prefill sat inside its CUPTI window, and its "653 µs/layer" was the whole decoder layer, not the mixer. Decode-only, `serial` is **4,586 launches / 15.9 ms device against a ~35.5 ms wall**, so it is HOST-bound.
- **The GDN mixers:** 63 non-matmul launches each; mixer host wall **438.8 µs = 15.80 ms/token**.
- **Ceiling (ABBA):** 35.5 → 24.35 ms of `serial` = **−11.15 ms = 10.3 % of T_unprof 0.1085 s**.
- **Bake-off (one real mixer, ABBA ×4):**

  | mechanism | µs per mixer call | kernels | vs stock |
  |---|---|---|---|
  | stock | 400.2 | 66 | — |
  | `torch.compile` | 327.0 | — | −18 % |
  | `fla` 0.5.2 | 391.1 | — | −2 % (Python wrappers eat the saving) |
  | **hand-written Triton** | **196.6** | **24** | **−51 %** |

- **Scope A, user-chosen.** Banked, not built:
  - **S3-B:** the 18 forward-body ops (price against what remains of the 11.15 ms ceiling after this build — `serial` fell 4.3 ms of it);
  - **S3-C:** `fla`'s prefill chunk kernel (0.36 → 0.03 s per 2k prompt, ~2 % TTFT; a dependency decision).

**Spec** `docs/superpowers/specs/2026-09-25-S3-gated-delta-kernels-design.md`; **plan** `docs/superpowers/plans/2026-09-25-S3-gated-delta-kernels.md` (both gitignored — the lines below are the durable copy).

**Done means:**
- real-token ABBA ≤ −2 % (else revert);
- `serial` ≤ −3.66 ms (half the microbench; else find out why before claiming);
- D6 6-draw ppl band: fused mean − stock mean ≤ pooled sd (one retry with bf16 l2norm, else revert);
- conv state bit-exact and fused ≤ 1.25× stock's fp64 error on 2 geometries;
- suite floor → **2879 passed, 44 failed, 2 errors, 16 skipped** (+97, +5 from the final-review fix wave), same failing SET;
- both MoE accumulator gates green, MiniMax and Llama-1B reporting 0 fused seams;
- §8: no knob, reachable at N=1.

- [x] **T1** (`2bb8165`) branch `s3-gated-delta-kernels`; `vaayu/gated_delta_kernels.py` with `gated_delta_seams()` (surface detection); 4 CPU tests (all-3 seams on Qwen3-Next and qwen3_5; rule+norm only for a synthetic conv-less mixer; real olmo_hybrid = rule only; 0 on a dense model despite its top-level `norm`)
- [x] **T2** (`106a968`, worst ratio 0.737/1.25) rule kernel `fused_recurrent_rule` — 39 CUDA tests (2 geometries × B/T × h0 None/bf16/fp32 × bf16/fp16 vs stock vs fp64; strided inputs; no-final-state)
- [x] **T3** (`0504f61`, state bit-exact) conv kernel `fused_conv_update` + `_conv_seam` — 19 tests (state bit-exact on channels-last AND contiguous, B 1/2, bf16/fp16; bias; prefix-cache round trip; T>1 delegates to stock)
- [x] **T4** (`61bf9dd`) gated-norm kernel `fused_gated_norm` + in-place `_patch_norm` — 18 tests (weight bf16/fp32 × x bf16/fp16 × rows; weight replaced after install; gateless call → stock)
- [x] **T5** (`3d3f4fa`, 97 passed) `install_gated_delta_kernels(hf_model, device, dtype)` with per-seam on-device self-check → WARNING + stock on failure — 17 tests (CPU/no-triton install nothing; idempotent; bad/raising kernel and different-math norm stay stock; meta weights; full mixer both families × both caches × prefill 2/12; row-43 chunked-prefill coexistence)
- [x] **T6** (`a07d0cc` pylint-gate fix + `68b6448`; suite 2874/44/2/16 in 192 s, same failing SET) one call in `vaayu_base.py` `init_model` beside `install_state_continuation`; full suite FOREGROUND = 2874/44/2/16, failing SET verified by composition
- [x] **T7** DONE — Qwen3-Next gate 8 passed + `rule=36 conv=36 norm=36`; MiniMax 8 passed, 0 fused; Llama-1B 1 passed, 0 fused; no self-check WARNING; server reaches it (app.py:400 → init_model); §8(2) confirmed by grepping the server's `from_pretrained` path
- [x] **T8** DONE — real token −5.58 % / −4.32 % (r3/r4, warm-up arms; r1/r2 cold-arm artifact, ruling R9), `serial` −4.36 / −4.26 ms, prefill flat at 135/2048, greedy ids identical fused vs stock; `plans/probes/s3_ab.py` — in-process ABBA (seams toggled via `_MARK`; deviation from the spec's cross-process ABBA, S1's method): N/U decode + P prefill at 135/2048, two replicates; apply the stop rules
- [x] **T9** DONE — VERDICT PASS: code 1.8933 → 1.8945 (+0.06 %, pooled sd 0.0026), prose 8.5756 → 8.5613 (−0.17 %, pooled sd 0.0360); no retry needed; `plans/probes/s3_quality.py` — D6 band at the server's untruncated default (null perturbation on EVERY `truncate_top_k` call, not only truncated ones as `d6_band.py` did), 6 draws × stock/fused × 2 texts, 256 scored tokens; PASS/FAIL (+ the one bf16-l2norm retry)
- [x] **T10** RESULTS section appended (`docs/2026-09-25-S3-census-rederive-RESULTS.md`), this row → `[x]`, suite floor → 2879 (+5 from the final-review fix wave), PICK THIS NEXT → S4, S3-B/S3-C banked below, row 36 note added; finishing the branch is the controller's call after final review, not this task's

**Verdict (durable):**
- **A/B** (in-process ABBA, `plans/probes/s3_ab.py`, Qwen3-Next, four replicates total): real decode
  token **−3.6 % to −5.6 % across four replicates** (r1/r2 with their cold first arm excluded: −3.57 %
  / −3.68 %; r3/r4 with a discarded warm arm leading every phase per ruling R9: r3 97.159→102.902 ms =
  **−5.58 %**, r4 97.887→102.310 ms = **−4.32 %**, OFF ~9.72–9.77 → ON ~10.22–10.29 tok/s); `serial` (NOEXP) r3
  25.664→30.018 ms = **−4.355 ms (−14.51 %)**, r4 25.632→29.889 ms = **−4.258 ms (−14.24 %)**; prefill
  flat at 135/2048 tokens (inside arm sd both replicates); greedy ids IDENTICAL fused vs stock over 64
  tokens, all four replicates.
- **Quality** (D6, `plans/probes/s3_quality.py`, 6 null draws × stock/fused × 2 texts, 256 scored
  tokens): code +0.06 % (pooled sd 0.0026), prose −0.17 % (pooled sd 0.0360), both inside pooled sd →
  **PASS**, no retry needed.
- **Gates** (T7, FOREGROUND under `systemd-run --user --scope`): Qwen3-Next MoE accumulator (29G) 8
  passed, `rule=36 conv=36 norm=36`, no self-check WARNING; MiniMax-M2.7 MoE accumulator (29G) 8
  passed, 0 fused seams; Llama-3.2-1B prefix-equivalence (22G) 1 passed, 0 fused seams. No fused-EXL3
  arm exists (no model on disk since 2026-09-20); dense has no committed gate (the Llama-1B prefix
  gate is a sanity check, not the deleted `test_batched_dense_smoke.py`).
- **Suite** (T6): 2874 passed, 44 failed, 2 errors, 16 skipped in 192.11 s, FOREGROUND — floor +97
  (was 2777), failing SET unchanged by composition.
- **§8:** (1) no knob — faster, closer to fp64, self-checks back to stock with a WARNING; (2) plumbed
  by construction, `app.py:400` → `vaayu_base.py:2263 init_model()` → `vaayu_base.py:3520
  install_gated_delta_kernels`, no server change; (3) reachable at N=1, a per-token decode effect in
  the server's own one-request-at-a-time mode; (4) n/a.
- **R5:** the repo's pylint used-before-assignment gate flagged the three kernel launches (bound under
  `if triton is not None:`); fixed with the gate's own prescribed inline
  `# pylint: disable=possibly-used-before-assignment` (`a07d0cc`).
- **R9:** r1/r2's first-reported U +0.78 % / +0.93 % was an ordering artifact — the U phase's first
  `on` arm ran with a cold expert RAM cache and no warm-up; re-run with a discarded warm arm leading
  every phase gives r3/r4 above.

**Leads banked:**
- [ ] **S3-B** — the 18 forward-body ops per mixer (gating, `repeat_interleave`, cat, reshape copies);
  reaching them means replacing the stock forward body — price against what remains of the 11.15 ms
  ceiling after this build (`serial` fell 4.3 ms of it).
- [ ] **S3-C** — `fla`'s prefill chunk kernel, ~10× the torch chunk fallback (0.36 → 0.03 s per 2k
  prompt, ~2 % TTFT); a new-dependency decision.

Row 36's "GPU-bound" verdict rested on a prefill-contaminated census (`serial` is really ~55 %
HOST-bound); it stays PARKED for its other two reasons (`use_torch_compile` crashes; 48 blocking D2H
routing syncs per token).
HF's cache `conv_states` goes channels-last after any prefill ≥ the conv width and `fla`'s
conv-update kernel silently corrupts it on that layout; every kernel installed here is stride-aware
instead, and the cache keeps the recurrent state in bf16 regardless.
Method lesson: in an in-process multi-phase A/B, lead EVERY phase with a discarded warm arm — a
phase change (NOEXP → real forward) leaves the expert RAM cache cold and can flip the sign, as it did
in r1/r2.

## S2 — `[x]` chunk-major byte counter survives an aborted expert injection. **CLOSED 2026-09-25, REFUTED: the shape is real in all three MoE forwards, the harm is unreachable.**

**No production code changed. One test added.** Detail:
`docs/2026-09-25-S2-chunk-major-counter-leak-RESULTS.md` (gitignored).

**The claim** (A5.7c review finding M2, carried into the S2 handoff): the `if _cm_last_call:` reset
sits outside the forward's `finally`, so an abort leaves `_chunk_major_gpu_bytes` stale **for the
process**, pre-charging every later prefill's budget check and silently shrinking row 45's retention
win (2k 2.75× / 8k 5.47×).

**The shape is real, and worse than reported — it is THREE sites, not one** (HEAD `38aba5c`; +12 from
the handoff's line numbers after S1-L1 landed): `moe_patch.py:1995-1997` (module-list, MiniMax),
`:2431-2433` (fused non-EXL3, Qwen3-Next), `:3196-3200` (fused-EXL3). All three reset outside their
own `finally`.

**The harm is not.** `vaayu_base.py:4746-4757` — `_chunked_prefill`'s layer-major `finally` — already
clears `_chunk_major_gpu`, `_chunk_major_gpu_bytes`, `_chunk_major_n` and `_chunk_major_lens` on ANY
exception, and its comment says exactly why ("Defensive: if an exception unwound mid-layer,
moe_patch.py's last-chunk teardown never ran"). **Retention can only be live inside that `try`:**
it arms on `_chunk_major_n > 1`, and `_chunk_major_n` has exactly ONE production assignment,
`vaayu_base.py:4741`, inside it. Nothing swallows the exception in between either — the only catch on
that path, `moe_patch.py:1330-1335`, **re-raises** `PackedInjectionError` by name.

**Measured:** the real `_chunked_prefill` bound to a stub whose `forward()` sets
`_chunk_major_gpu={...}` / `_chunk_major_gpu_bytes=12_345_678` and raises → exception propagates and
all four fields come back `None`/`0`. The handoff's "red before the fix" test is **green before any
fix**. That is the refutation.

**What WAS missing: coverage.** Both forwards' comments cite this `finally` by name
(`moe_patch.py:1741-1742`, `:2259-2260`), but no test ran it — `test_modulelist_chunk_retention.py:143`
and `test_fused_nonexl3_chunk_retention.py:115` hand-simulate it
(`ref._chunk_major_gpu = None  # what _chunked_prefill's finally does`), so deleting the guard left
both green. **Added** `tests/test_layer_major_prefill.py::test_aborted_layer_major_prefill_clears_chunk_major_retention`
(+55 lines). **Mutation-proved:** deleting the guard's two lines turns it RED
("retained GPU experts survived the abort"), rest of the file green.

**Decision, argued because the handoff asked:** the reset lines were **NOT moved**. Resetting on abort
when `_cm_last_call` is False is moot — the layer-major path is ONE forward carrying every chunk, so an
exception ends the whole prefill and there is no next chunk to mistrust. Moving them buys nothing
measurable and reshapes the hottest loop in the engine (CLAUDE.md §3; the handoff's own §6 stop rule).

**Gates:** full suite FOREGROUND **2777 passed, 44 failed, 2 errors, 16 skipped in 191.07 s** — floor
+1, SET verified by composition (attention 21, parity 8, moe 8, linear 4, model 2, checkpoint 1, block
2 errors). **The two on-device MoE accumulator gates were NOT run and are not owed** — the handoff
required them for a `moe_patch.py` edit, and there is none; the diff is one test file. No fused-EXL3
arm exists, dense has no committed gate. **§8: no knob** — no behaviour change to configure; the guard
itself is on the server's own N=1 prefill path, so the invariant is live for users.

**Lesson, reusable:** M2 was found by reading `moe_patch.py` and never checking the one caller that
arms the mechanism. Same class as row 13 — replay the real path, not a model of it.

## S1-L1 — `[x]` prefill's pageable HtoD. **CLOSED 2026-09-25, A WIN: prefill TTFT +14.4 % to +24.5 % on Qwen3-Next, +17.5 % to +18.9 % on MiniMax, greedy ids bit-identical, decode flat.**

**Shipped, wired and reachable at N=1** (the fix, its A/B and §8 are at the end of this row).

**It is no longer a lead. One call site is 99.85 % of the copies and 99.5 % of the bytes, the copies
are FULLY SYNCHRONOUS (they drain the stream despite `non_blocking=True`, because the source is
pageable), and the host-side prize is measured at 89.1 µs per expert load.** Detail:
`docs/2026-09-25-S1-L1-prefill-pageable-h2d-RESULTS.md` (gitignored); probes
`plans/probes/s1l1_attrib.py` (python `.to`/`.cuda`/`.copy_` interception by call site, cross-checked
against CUPTI in the same forward) and `plans/probes/s1l1_copycost.py` (pageable-vs-pinned pricing at
the real shapes); raw `plans/probes/s1_out/s1l1-*.json`.

**Attribution — one 135-token prefill forward, Qwen3-Next, cold expert RAM cache.** The python
counter and CUPTI agree EXACTLY (56,628 = 56,628), so the python table *is* the attribution; no
trace parsing was needed.

| site | n | bytes each | GB | what it is |
|---|---|---|---|---|
| **`moe_patch.py:600` `_expert_dense_weights`** | **56,544 (99.85 %)** | 524,288 + 65,536 | **16.676 (99.5 %)** | 9,424 expert loads × 3 roles × {`weight_packed`, `weight_scale`} |
| `moe_patch.py:2310` `sparse_forward` | 48 | 21,600 | 0.001 | `_pair_gpu` — **this is S1-b, already refuted; do not re-chase** |
| `modeling_qwen3_next.py:426` `torch_chunk_gated_delta_rule` | 36 | 2,097,152 | 0.075 | prefill's sibling of S1-a's `:470` (decode). 0.4 % of the bytes |

**Root cause, in two lines of shipped code.** `moe_patch.py:2532` — on `_moe_prefill_pass` the shared
miss loader returns `load_expert_sparse(...)`, deliberately bypassing the slab arena ("transient load,
never cached"). `safetensor_model_persister.py:907` — `load_expert_sparse` reads into a PINNED pool
buffer and then **`tensor.clone()`s every tensor out into pageable RAM**, because the pool buffer must
go straight back to the pool. So prefill pays a 1.77 MB host memcpy per expert (**16.7 GB per
prefill**) and then a pageable, stream-draining H2D of the clone. Decode escapes both via the arena
(row 30-P, +10.62 %). The module-list forward is on the SAME path and says so at
`moe_patch.py:1875-1878` ("a prefill clone … is in no slab -> None -> synchronous").

**The count is cache-state-dependent; the per-expert cost is not.** 56,628 pageable copies cold
(every expert a miss) vs S1's **27,084** on a warm arm — a warm RAM-cache hit copies from a PINNED
slab view and is already async. Prefill itself never warms the cache (it does not insert), so only
decode does: **a first request is ~100 % pageable, a steady-state one ~48 %.**

**Priced (`s1l1_copycost.py`, 300 experts/arm, ABBA, real shapes, dequant+3 GEMMs queued behind each
load so a blocking copy pays its drain):** pageable **202.8 µs host / 202.9 µs device** — host equals
device to 0.05 %, i.e. **fully serialising** — vs pinned **113.6 µs host / 159.8 µs device** (host
runs ahead, so it is genuinely async). **Delta 89.1 µs of host stall per expert load**, arms cleanly
separated (pageable 205.5/200.1, pinned 113.5/113.8).

| prompt | expert loads | prefill (unprofiled) | H2D bytes | host prize @89.1 µs | share |
|---|---|---|---|---|---|
| 135 tok, 1 chunk | 9,424 | **7.18–8.80 s** | 16.75 GB | 0.840 s | **11.7 %** cold, ~5.6 % steady-state |
| 2,048 tok, 4 chunks | 16,276 | **16.57–16.98 s** | 28.89 GB | 1.451 s | **8.8 %** cold, ~4.2 % steady-state |

Both are far above the handoff's ~1 %-of-prefill stop rule and above the 0.36 % real-arm resolution
floor, and the clone at `:907` (16.7 GB of host memcpy) is deleted by the same fix on top of that.
**Prefill is N=1 and is what the server runs, so unlike P8 (§8(3)) this is a real user win.**

**The cheap fix was REFUTED before it was built, and this is the S1-b lesson repeating.** Cloning
into PINNED RAM instead of pageable (a 3-line change in `load_expert_sparse`, every lifetime
unchanged) measures **−10.0 µs/expert — SLOWER than the code already there**: the pinned clone costs
**109.1 µs** against the pageable clone's **34.0 µs** (the CachingHostAllocator's `copy_` into a
page-locked block is 3.2× a plain clone), which eats the whole async-upload gain. Do not rebuild it.
Copying with NO clone, straight out of pinned memory, is **+67.4 µs/expert** in the same probe
(`plans/probes/s1l1_pinnedclone.py`, arms A/B/C/C/B/A).

**What SHIPPED (the handoff's own preference #1 — "put it in the arena, not pin at the copy").**
`SlabArena` withholds slabs from its own free list for TRANSIENT prefill staging
(`reserve_staging`/`acquire_staging`/`release_staging`, `vaayu/persist/expert_cache.py`); prefill's
branch of the SHARED miss loader reads each expert into one with `load_expert_into_slab` — decode's
zero-clone path — instead of `load_expert_sparse` (`_stage_prefill_expert`, `moe_patch.py`). A staging
slab IS an arena slab, so `slab_index_of` keeps answering and **all three forwards' existing
`record_read` guard and the module-list forward's `pinned_source=` flag light up with no per-forward
special case**; each forward returns the slab as its uploads issue and sweeps the layer's leftovers in
the `finally` it already had.

- **`acquire_staging` BLOCKS** (5 s backstop, `_STAGING_WAIT_BACKSTOP_S`, the spirit of
  `_LOAD_WAIT_BACKSTOP_S`). This is the design decision the win depends on: completed-but-unconsumed
  loads hold their slabs and a layer routes 196-339 experts against 15-16 slabs, so an immediate
  fallback would have left ~92 % of experts pageable. Blocking bounds read-ahead to the staging depth,
  which is also what the drive wants (P8: past ~4 concurrent reads throughput FALLS). **Measured
  coverage 100 %** (7,525/7,525 and 11,466/11,466 expert loads staged, zero clones).
- **No deadlock:** `_iter_expert_loads` consumes in COMPLETION order, so the consumer always has
  something to consume and release while workers wait. **No permanent leak:** the worker registers its
  slab BEFORE checking whether its layer has torn down, so either the layer's sweep releases it or the
  worker does (both idempotent), and the next layer's dispatch releases any other layer's leftovers.
- **Sizing (§5):** `prefill_staging_slabs(pool_size, slab_bytes)` = twice the loader width, capped by a
  **128 MiB pinned-RAM allowance (a HARDWARE number)** — Qwen 16 × 1.79 MB = 28 MB, MiniMax
  15 × 8.89 MB = 133 MB. The arena is requested with these ON TOP of the loader's landing zone and the
  cache is charged for what is left, so **the expert RAM cache loses nothing.** Reservation failure
  WARNS at load and prefill keeps the clone path.

**A/B — ABBA, arms alternating inside ONE model load, prefill forward wall clock, unprofiled, prefix
cache OFF, arena byte-identical in both arms (`plans/probes/s1l1_ab.py`):**

| model / forward | prompt | OFF (s) | ON (s) | delta | arm sd |
|---|---|---|---|---|---|
| **Qwen3-Next-80B int4**, fused non-EXL3 | 135 | 4.649 / 4.668 | 3.846 / 3.882 | **+17.06 %** | 0.55 % |
| | 2,048 | 10.814 / 10.833 | 9.285 / 9.238 | **+14.43 %** | 0.31 % |
| | 135, 3 arms/side | 4.228 / 4.293 / 4.250 | 3.102 / 3.338 / 3.201 | **+24.51 %** | 2.78 % |
| **MiniMax-M2.7 EXL3**, module-list | 135 | 20.272 / 20.336 | 16.468 / 16.452 | **+18.93 %** | 0.23 % |
| | 2,048 | 33.686 / 33.966 | 27.765 / 28.028 | **+17.53 %** | 0.59 % |

Every ON arm is separated from every OFF arm with no overlap. **Greedy ids IDENTICAL across all arms,
both models, both lengths** (plus a CPU bit-identity test in the suite). **Decode flat:** five paired
comparisons read −5.17 / −1.74 / −0.87 / +0.07 / +1.93 %, no consistent sign, each inside its
within-state arm spread (the −5.17 % is one ON arm at 353.6 ms against its twin at 308.3 ms); and
mechanically decode cannot move — staging is reached only under `_moe_prefill_pass`, cache capacity is
unchanged by construction, and the release helper early-returns on an empty dict. The realised win
(195 µs/expert at 135 tok) EXCEEDS the 89 µs priced, because the fix also deletes the 1.77 MB host
clone per expert and reads O_DIRECT into page-locked memory.

**§8, all four:** (1) **no knob — stated, not assumed**: the depth derives from
`get_expert_pool_size()` and a hardware allowance, the fallback is automatic and warns, so nothing was
added to `vaayu/server/config.py`, `vaayu/server/README.md` or `documentation/` (which documents
knobs, and there is no new one); (2) **plumbed** — armed at load in `vaayu_base.py` for every MoE model
that gets an arena, and the A/B loaded through `AutoModel.from_pretrained` with the server's own
config values, the path `app.py` uses; (3) **REACHABLE, and that is the point — prefill at N=1, which
`engine.py` runs for every request.** A user's TTFT falls 14-25 % on the production model; this is not
a P8-style N>1 effect; (4) n/a.

**Gates run:** full suite FOREGROUND **2776 passed, 44 failed, 2 errors, 16 skipped in 193.44 s** —
2759 + 17 new tests, failing SET verified member-by-member (attention 21, parity 8, moe 8, linear 4,
model 2, block 2 errors, checkpoint 1, all the S4 shard-deletion class). Both MoE accumulator gates
FOREGROUND at `MemoryMax=29G`: **Qwen 8 passed / 33.58 s, MiniMax 8 passed / 76.16 s**. New tests
`tests/test_prefill_staging_slabs.py` (17). **No fused-EXL3 arm exists** (no model on disk) so that
forward's release call is covered by unit tests only, and **dense has no committed gate** — said,
neither claimed.

---

## S1 — `[x]` pageable HtoD memcpys per decode token. **CLOSED 2026-09-25 as a measured NULL.**

**Verdict: the copies are real and now fully priced, and removing them does not pay. The one fix
that needs no copied upstream code was BUILT, unit-tested, MEASURED at −7.06 % on `serial` /
−1.39 % on the real forward, and REVERTED. S1-b was refuted before it was built. Nothing in
`vaayu/` changed — the tree is byte-identical to `8642a50`.** Full detail:
`docs/2026-09-25-S1-pageable-h2d-RESULTS.md` (gitignored); probes `plans/probes/s1_h2d_attrib.py`,
`s1_parse_trace.py`, `s1_ab.py`; raw `plans/probes/s1_out/`.

**The spike's 41.7/token was wrong twice, and BOTH errors understated it.** Row 36 profiled
**prefill and decode in one window and divided by the decode-forward count**, on the **NOEXP** arm.
Bracketing each forward in a `VAAYU_FWD_<i>` annotation and starting the profiler *inside*
`model.forward` gives **36.00/decode-forward on NOEXP** and **84.00 on the real forward**.

**Exactly two call sites**, both per-layer, neither disk:

| # | site | n/fwd | bytes | host µs each | host µs/fwd | stream drain |
|---|---|---|---|---|---|---|
| S1-a | `modeling_qwen3_next.py:470`, `core_attn_out = torch.zeros(...).to(value)` in `torch_recurrent_gated_delta_rule` | 36 = the `GatedDeltaNet` layers | 16,384 | 20.76 | 747 | **YES** — `cudaStreamSynchronize` in every one |
| S1-b | `vaayu/moe_patch.py:2310`, `_pair_gpu = _pair.to(device, non_blocking=True)` | 48 = `num_moe_layers` | 160 = `2 × top_k × 8` | 8.48 | 407 | no |

Device time for all 84 is **37.9 µs/fwd**; host time **1,154 µs/fwd — 30× the device time**. The
spike's "0.98 ms" was the wrong quantity to quote.

**S1-b — REFUTED, never built.** Idle-stream microbenchmark, 5,000 calls, the real 160-byte `_pair`:
status quo **pageable `.to(non_blocking=True)` = 3.14 µs**, pre-pinned **3.28 µs**,
`.pin_memory().to(...)` **5.80 µs**. **Every available fix is slower than the code already there** —
`non_blocking=True` already skips the stream sync (confirmed in the trace) and 160 bytes is too
small for pinning to repay itself. The 8.48 µs above is CUPTI overhead, not a user cost.

**S1-a — BUILT, MEASURED, REVERTED.** Isolated cost: `torch.zeros(...).to("cuda")` **8.41 µs** vs
`torch.zeros(..., device="cuda")` **3.22 µs** (with 8 GEMMs queued: **159.06 vs 17.56 µs** — the
blocking copy drains the queue). Ceiling therefore **187–630 µs/token = 0.17–0.6 % of a 110 ms
token**. Built as `install_device_local_factories` in `recurrent_state.py`, wrapping the
`chunk_/recurrent_gated_delta_rule` module attributes (`modeling_qwen3_next.py:553-554`) in
`with torch.device(value.device):` — detection by attribute surface, no copied upstream code, and it
**worked**: on-device count 36 → **0** per forward, outputs bit-identical, 8 new tests green on both
families. A/B in ONE model load, toggling in place, 8 alternating `serial` arms + 4 real arms,
64 tokens each, medians:

| phase | fix ON | fix OFF | delta | within-state arm sd = resolution floor |
|---|---|---|---|---|
| `serial` (NOEXP) | 41.096 ms | 38.384 ms | **−2.711 ms = −7.06 %** | 0.457 ms = 1.19 % |
| real forward | 127.458 ms | 125.710 ms | **−1.748 ms = −1.39 %** | 0.458 ms = 0.36 % |

Perfectly separated (ON 40.885/41.776/40.930/40.793, OFF 38.336/38.559/38.373/38.270).
**Correction 2026-09-25, same day: "TTFT flat" was WRONG and is withdrawn — TTFT was never tested.**
The A/B's prefill column is the forward the probe profiles with CUPTI to count memcpys, so ~53 s of
it is profiler overhead; a real unprofiled 135-token prefill is **4.6-4.9 s** (0.16 s on NOEXP).
The verdict is unchanged — the decode regression settles it on its own.
**Root cause of the regression, confirmed directly: `torch.device` used as a CONTEXT MANAGER pushes a
`DeviceContext` torch-function mode, so every torch op inside pays python dispatch — +0.906 µs/op
measured (100 ops: 323.63 → 414.25 µs).** ~80–110 ops/layer × 36 layers ≈ 2.7 ms, which is the
regression to the millisecond.

**What is left and why it is not worth doing:** only a Vaayu-owned COPY of upstream's two torch
fallbacks with `device=` on the zeros. It buys the 0.17–0.6 % ceiling — **below this box's 0.36 %
real-arm floor and far below its ~3 % positional drift (rows 33, 44)** — for the cost of carrying
someone else's numerics code across transformers upgrades. **Stop rule fires.**

**§8, all four:** (1) no knob — nothing shipped, and a bit-identical change would need none;
(2) n/a; (3) it WOULD have been reachable — both sites are per-layer in the N=1 decode path the
server actually runs, not an N>1 effect, and that still was not enough; (4) n/a — what would revive
this is not a knob but `fla` (see the S1-L3 lead).

**Gates:** full suite FOREGROUND **2759 passed, 44 failed, 2 errors, 16 skipped in 192.44 s** — the
floor, with the failing SET verified member-by-member (attention 21, parity 8, moe 8, linear 4,
model 2, block 2 errors, checkpoint 1). **Both MoE accumulator gates NOT RUN and not owed: no file
under `vaayu/` changed.** No fused-EXL3 arm exists and dense has no committed gate; neither claimed.

### Leads S1 recorded and deliberately did NOT chase

- **[x] S1-L1 — ~27,084 pageable HtoD in ONE 135-token PREFILL forward. CLOSED 2026-09-25 as a WIN — see the S1-L1 row above, which supersedes this bullet** (one site, `moe_patch.py:600`, 99.85 % of the copies; 89.1 µs of host stall per expert load; prefill TTFT +14.4 to +24.5 % once they came out of pinned staging slabs). Handoff `docs/superpowers/specs/2026-09-25-S1-L1-prefill-pageable-h2d-HANDOFF.md`. Counted in the S1 A/B
  (fix off; 27,048 with S1-a's 36 removed). Decode's expert loads are NOT pageable — the slab
  arena's async path works there (row 30-P) — so this is **prefill-only**. Prefill is TTFT, this
  workload's top lever (CLAUDE.md §0), and the number is ~320× the entire decode-side finding.
  **This is the larger fish; it is a lead, not yet a diagnosis** — nobody has attributed those 27k
  to a call site the way §2 above attributed the 84. **Calibration: that prefill really costs
  4.6-4.9 s** (unprofiled, 135 tokens; 0.16 s on NOEXP). 27,084 copies at S1-b's measured 3.14 µs
  floor for a *non-draining* pageable copy is ~85 ms = ~1.8 % of it; if they drain the stream the way
  S1-a's did, far more. **The count alone establishes nothing — S1 is the cautionary tale.** Attribute
  first, at a realistic agentic prompt (2k-8k, several chunks), not at 135 tokens.
- **[ ] S1-L2 — 96.5 `Memcpy DtoH (Device -> Pageable)` per real decode forward.** Never counted
  before: the row 36 spike's top-25 only ever showed the 25.4 *pinned* DtoH. More numerous than the
  84 HtoD and fully synchronous. Belongs with S3's routing-sync item; S1's handoff §5 forbade
  chasing DtoH inside S1.
- *(→ moved to `todo-backlog.md` 2026-09-26: S1-L3 — `fla` / `causal_conv1d` not installed. Re-open when: with S3-C.)*

---

**A5.7c CLOSED 2026-09-25 and is merged (`6870dab`), so the old "PICK THIS NEXT: A5.7c" pointer further down is history, not an instruction. The >512-token prefill bug it surfaced is FIXED and merged too (`876865f`).**

**One line, so nobody reads fifty before finding it: A5 step 1 (the kernel gate) PASSED 6/6 on
2026-09-23; step 2 was decomposed the same day into A5.1…A5.7, each with its own document and
durable row; A5.1–A5.4 ALL CLOSED the same day — `AutoConfig` resolves `deepseek_v4`, the parity
harness has 15 committed goldens from DeepSeek's OWN modules, `DeepseekV4Linear` reproduces all
three `linear_*` goldens BIT-IDENTICALLY, and the MoE block lands on Vaayu's module-list forward
with **zero engine changes** (both MoE gates green), bit-identical on `gate_*`, `expert` and both
`moe_*` goldens. **A5.5 IS CLOSED TOO — its step-0 gate passed 3/3 on
2026-09-23 and the port itself landed 2026-09-24 in 17 commits (`e983e60..6d5beb5`, merged to
main). A5.6 IS CLOSED TOO — `AutoModelForCausalLM.from_config` now builds the real 43-layer,
11,008-expert skeleton under `init_empty_weights` in 1.44 s, both new goldens hold (`hc_head`
bit-identical, `head` rel-L2 0.00166 + exact argmax). The step where A5 was expected to die did
not kill it. A5.7a IS CLOSED TOO (2026-09-24) — seven commits teach the streamed path the
architecture end to end (cache, layer signature, `hc_head`, the id scope, a new `run_embed` hook,
`auto_model` dispatch); DeepSeek suite 289 passed, full repo 2711 passed/16 skipped, both on-device
gates green. **START AT A5.7b**, DESIGNED 2026-09-24 and `[~]` in flight; A5.7a's durable record, including
everything A5.7b MUST carry, is row 28b.** **Correction carried from
A5.3: the goldens store the INPUT and the OUTPUT, not the weights, so a golden-reproducing test DOES
need the 157 G checkpoint and a GPU; "neither" holds only for the structural half (shapes, contracts,
import rules), which is 15 of A5.3's 19 tests.** Everything below is PENDING work only — the closed
rows were pruned 2026-09-23 and live in `git log -p -- todo.md`.

---

**ROW 32 IS OUT OF THIS TABLE — `[!]` BLOCKED 2026-09-23, by the user, on cost.** It was order 1 that
morning; the session verified its route (a full HF sweep: the build on disk ships `2.0…8.0bpw` as
BRANCHES of the SAME repo, so route 1 is live and route 2 is dead) and then the user **declined the
download**. **Row 32's entire cost IS that download — there is nothing else in it to build — so it
cannot proceed and is not pickable.** Do not re-open it by re-deriving the projection; it unblocks
only on the user authorising an ~81.6 GiB pull. Its verified branch sizes and two handoff
corrections are banked in row 32 itself. **NO DOWNLOAD is a standing constraint until the user says
otherwise: every item below runs on checkpoints already on this disk.**

## ~~▶ PICK THIS NEXT: **A5.7c** (row 28b)~~ — **HISTORY, NOT AN INSTRUCTION. A5.7c CLOSED and MERGED 2026-09-25 (`6870dab`); the >512-token prefill bug it surfaced is fixed and merged too (`876865f`). The live queue is the S1…S6 sequence at the top of this file.** The text below is kept for the reasoning it records, not to be acted on.

**A5.7b is `[~]` PARTIALLY DONE, branch `a5.7b-load-first-token`, 14 commits, NOT merged, suite
green at 2778/16.** DeepSeek-V4-Flash is **SPLIT (46 shards, 146 G, originals intact), LOADS in
4.0 s, and runs inside Vaayu's MoE streaming forward** — 11,008 experts indexed, all 43 layers
patched. It does not yet emit a token: ONE defect remains and it is in `vaayu/moe_patch.py`, the
expert-streaming injector, which is the module-list forward **MiniMax-M2.7 runs in production**.
A5.7b's own stop rule ("if it grows into the expert cache, STOP and re-plan") fired, so the
remainder is **A5.7c**, to be RE-PLANNED in a fresh session from
[`…-A5.7c-expert-injection-HANDOFF.md`](docs/superpowers/specs/2026-09-25-A5.7c-expert-injection-HANDOFF.md).

**A5.7's §0 gate was DECIDED 2026-09-24: TEACH THE ENGINE, and A5.7 was SPLIT into A5.7a (engine)
and A5.7b (load + first token), both approved by the user that day.** Unstreamed is dead on
arithmetic and must not be re-litigated: the checkpoint is **157 G** against **16,311 MiB VRAM +
31 GB RAM = ~47 GB** of maximum resident capacity, **3.3× over**. **The brief over-estimated the
shared-infra cost — four of the five deltas land on hooks `VaayuBaseModel` ALREADY EXPOSES**
(`_create_kv_cache`, the four `*_args` hooks, `run_norm`, and a `forward` wrapper for
`input_ids_scope`); only the embed-time HC expand needs a NEW hook, `run_embed(layer, seq)`, two
lines with an identity default at `vaayu_base.py:5182`. **That one hook is the whole shared-infra
surface, and it is why A5.7a is separable.** Re-cost: **A5.7a ≈ 1 day, A5.7b ≈ 1–2 days** (mostly
wall-clock on the irreversible split) — the brief's 2–3 days survives, redistributed.

**A5.1–A5.7a ARE ALL CLOSED (A5.1–A5.4 on 2026-09-23, A5.5/A5.6/A5.7a on 2026-09-24).**
**Items (6) and (7) below are now CLOSED by A5.7a** — the streamed path calls this forward through
`VaayuDeepseekV4`'s hooks, and `_create_kv_cache` now builds a real `DeepseekV4Cache`; they are kept
below for the historical record of the problem, not as open work. **Item (2)'s severity was
claimed corrected by A5.7a, but that claim was ITSELF WRONG** — the pre-existing generic surface
check (`vaayu_base.py:3755`) does NOT run for this model; A5.7a's own hard refusal in
`VaayuDeepseekV4.__init__` raises first, making `:3755` dead code on this path, so the real outcome
is a hard load failure, not an announced downgrade — see row 28b's A5.7a bullet, carried item (b),
for the full correction (found and fixed in the final whole-branch review, 2026-09-24). **A5.7a
also found a NEW, still-open vector into the same class of risk: the
`_start_pos` contiguity guard is bypassed on the streamed path, and speculative decoding is the one
residual path into it that today's refusals do not cover — see row 28b's A5.7a bullet, carried item
(a). A5.7b must add that check when it wires generation.**
**SEVEN THINGS A5.7 MUST CARRY, from A5.5 and A5.6, all in row 28b (two added by final review,
2026-09-24 — see below):** (1) attention attends a
contiguous `kv_view = cat([history, chunk, compressed])`, NOT the reference's in-place ring —
**do not "simplify" it back**, a 512-token chunk overruns the 128-slot window; (2) the open
`[bug]` — `batch.cache_holds_recurrent_state` (`batch.py:500`), `prefix_cache.py:112`/`:364`,
`kv_persist`, and `Cache.max_cache_len`/`max_batch_size`/`batch_repeat_interleave`/
`batch_select_indices` do not understand this cache, so **this model must run with prefix caching
and KV persistence OFF until that is fixed**, and A5.7 cannot claim reachability before it is;
(3) **hard precondition**: the loader MUST set `config.fp4_group_size` (measured = 32, off the
shard headers) before `from_config`, or the skeleton raises A5.4's named `ValueError`; (4) a
pending bug in A5.5 code: all three `.get("weight_block_size")` sites in `modeling_deepseek_v4.py`
(lines 493, 1166, 1236) pass `None` through to `DeepseekV4Linear` and raise an unnamed `TypeError`
instead of naming the missing field, when a config's `checkpoint_quantization_config` omits it;
(5) A5.6's deferred minors — see row 28b's A5.6 bullet for the full list.
(6) **The streamed path does not call this forward at all.** `VaayuBaseModel.forward`
(`vaayu_base.py`) is a layer-major loop, not a call into `DeepseekV4Model.forward` /
`DeepseekV4ForCausalLM.forward`: it calls the embedding directly (`batch[j] = layer(seq)`,
`:5182`), then each decoder layer as
`layer(seq, use_cache=…, **get_past_key_value_args(…), **get_pos_emb_args(…),
**get_attention_mask_args(…), **get_position_ids_args(…))` (`:5208-5215`), then `run_norm` and
`run_lm_head`. None of that (a) can call
`DeepseekV4DecoderLayer.forward(hidden_states, past_key_values, start_pos)` — `start_pos` is
missing and `use_cache`/`position_embeddings`/`attention_mask` are unexpected kwargs it doesn't
take; (b) expands the residual to `[b, s, hc, d]` at embed time; (c) calls `hc_head`; or (d)
enters `input_ids_scope` — so the three hash-routed layers would raise out of
`current_input_ids()`. **Closing this is a both-paths change under CLAUDE.md §0** (it needs a
`VaayuDeepseekV4` subclass over the existing `get_pos_emb_args`/`get_past_key_value_args`/
`get_attention_mask_args`/`run_norm` hooks, plus something for the embed-time HC expand and
`input_ids_scope` lifetime — neither has a hook today — or a non-streamed path), which obliges
the Qwen3-Next and MiniMax on-device gates, not just this model.
(7) **Nothing constructs a `DeepseekV4Cache`.** `_create_kv_cache` (`vaayu_base.py:3789`) only
returns `DynamicCache`/`FP8DynamicCache`/`TurboQuantKVCache`, and `GenerationMixin` manufactures a
`DynamicCache` when the caller supplies none — either route hits A5.6's `TypeError` guard on the
first token. Distinct from the existing `[bug]` above (item 2), which is about existing
cache-management code *mistreating* an already-constructed `DeepseekV4Cache`; this is that one is
never built in the first place.
**A5.5 left
`DeepseekV4Cache`/`DeepseekV4CacheLayer`, and A5.6 left `DeepseekV4ForCausalLM` + the `auto_map`
entry proving `AutoModelForCausalLM.from_config` builds the real skeleton in 1.44 s under
`from_config` — but that is skeleton construction only, not a working forward; items (6) and (7)
above are what is actually left, and the row's prior "purely weights + index + first token" framing
undersold both.** A5.2 left `tests/deepseek/parity.py` + 15 committed
goldens in `tests/deepseek/goldens/` (1.1 MB); the recorder is `python -m tests.deepseek.parity`.
A5.3 + A5.4 left `vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py` (`DeepseekV4Linear`,
`Gate`, `Expert`, `MoE`, `input_ids_scope`) — **A5.5 adds attention to that same file, reads
`KERNEL_DIR` before any kernel call, and A5.6's model forward runs its layers inside
`input_ids_scope` so the three hash layers don't raise.** A5.1 left two things every later
sub-plan uses:
`tests/fixtures/deepseek_v4_config.json` (a committed copy of the shipped config — **no 157 G
checkpoint and no GPU needed to test this path**) and
`vaayu/model_code/deepseek_v4/configuration_deepseek_v4.py`, the repo-tracked SOURCE OF TRUTH that
is copied into the checkpoint dir. **Put A5.x modeling code in that package and copy it the same
way** — a drift test enforces the copy.

**A5 step 2 is DESIGNED as of 2026-09-23 — seven sub-plans, each with a self-contained document and
its own durable row under row 28b.** A fresh session needs nothing but the row it is picking.
**Work them IN ORDER: each module only has a golden reference once its dependencies are correct.**

| order | item | kind | what to open FIRST | cost |
|---|---|---|---|---|
| ~~1~~ | ~~**A5.1** config + `auto_map`~~ | **CLOSED 2026-09-23** | [`…A5.1-config-RESULTS.md`](docs/2026-09-23-A5.1-config-RESULTS.md) | done |
| ~~2~~ | ~~**A5.2** parity harness~~ | **CLOSED 2026-09-23** | [`…A5.2-parity-harness-RESULTS.md`](docs/2026-09-23-A5.2-parity-harness-RESULTS.md) | done |
| ~~3~~ | ~~**A5.3** dequant + `Linear`~~ | **CLOSED 2026-09-23** | [`…A5.3-dequant-linear-RESULTS.md`](docs/2026-09-23-A5.3-dequant-linear-RESULTS.md) | done |
| ~~4~~ | ~~**A5.4** MoE~~ | **CLOSED 2026-09-23** | [`…A5.4-moe-RESULTS.md`](docs/2026-09-23-A5.4-moe-RESULTS.md) | done |
| ~~5~~ | ~~**A5.5** attention~~ | **CLOSED 2026-09-24** — 17 commits, suite 2656/16, both MoE gates green | [`…A5.5-attention-RESULTS.md`](docs/2026-09-24-A5.5-attention-RESULTS.md) | done |
| ~~6~~ | ~~**A5.6** block + model~~ | **CLOSED 2026-09-24** — 5 commits, suite 2686/16, both MoE gates green | [`…A5.6-block-model-RESULTS.md`](docs/2026-09-24-A5.6-block-model-RESULTS.md) | done |
| 7 | ~~**A5.7a** engine change~~ | **CLOSED 2026-09-24** — 7 commits, DeepSeek suite 289 passed, full repo 2711 passed/16 skipped, both MoE gates green | [`…A5.7a-streamed-path-RESULTS.md`](docs/2026-09-24-A5.7a-streamed-path-RESULTS.md) | done |
| ~~8~~ | ~~**A5.7b** load + index + split~~ | **MOSTLY CLOSED 2026-09-25** — 14 commits on `a5.7b-load-first-token` (unmerged), suite 2778/16, split written and verified, model loads in 4.0 s and reaches the streaming forward. Stopped on its own stop rule. | [`…A5.7b-load-first-token-design.md`](docs/superpowers/specs/2026-09-24-A5.7b-load-first-token-design.md) | done |
| **9** | **A5.7c** streamed fp4 expert injection → first token | **RE-PLAN FROM SCRATCH — shared-infra change** | [`…-A5.7c-expert-injection-HANDOFF.md`](docs/superpowers/specs/2026-09-25-A5.7c-expert-injection-HANDOFF.md) | **unknown until planned**; the code change looks small, the gate burden is the cost |

**A5.1–A5.4 are full designs; A5.5–A5.7 are briefs with start gates and stop rules**, promoted to
plans by the session that reaches them (their inputs need measuring first — a stale design is worse
than a brief, because the next session implements it instead of questioning it). **`docs/` is
gitignored: row 28b's A5.x bullets carry every number and stop rule and are the durable copy.**

~~**A5.5 is where A5 dies if it dies**~~ — **IT DID NOT. Step-0 gate PASSED 2026-09-23, 3/3, in
~1.5 h.** `sparse_attn` at `block=16` is correct at the real working point (topk 640, rel 0.00205),
the module runs at real length with the Indexer saturating `index_topk`, and attention costs
**8.4–9.9 % of the 3.2124 GiB/token disk floor** (48.08 ms against 486–575 ms). **Gate (3) is
cancelled — its precondition did not fire.** And the tile was the wrong worry: `sparse_attn` is
**10 %** of attention, behind the **Indexer (26 %)**, the **Compressor (18.5 %)** and **`wo_a`'s
einsum (15 %)**. Full numbers in row 28b's A5.5 bullets.

**A5 / row 28b IS IN FLIGHT `[~]` — ITS STEP-1 GATE PASSED 2026-09-23, 6/6.** The abandon branch
(*kernels will not build on sm_120 → decide Triton-or-abandon, ~2 days not weeks*) **did NOT fire**:
all six of DeepSeek's TileLang kernels build and validate here, both GEMMs at rel err ~0.0027, suite
unchanged at 2422/16. **The next step is A5 step 2 — HF modeling code ported from the checkpoint's
own `inference/model.py`, with `auto_map`, experts as `nn.ModuleList`.** Three things a fresh session
must carry (full detail in row 28b, which is the durable copy):
**(1) `apache-tvm-ffi==0.1.9` is an EXACT pin now in `requirements.txt`, and `import tilelang` is NOT
a valid test of it** — 0.1.10 imports fine and dies at kernel COMPILE, 0.1.12+ dies at import.
**(2) `sparse_attn` must run at `block=16`** — the shipped `block=64` needs 141,312 B of dynamic
shared memory against this card's 101,376 B ceiling; `block=32` misses by only 3,072 B, so trimming
3 KiB is the first optimisation if step 5 shows the ¼-size tile hurts. **(3) A5 is REACH, not speed:**
3.2124 GiB/token of routed experts caps decode at **~1.75–2.06 tok/s** — MiniMax-class, ~5× slower
than Qwen3-Next. **[S6 2026-09-26: this projection is REFUTED — measured 0.3267 tok/s. It omitted the non-expert weights that do not fit the VRAM pin budget (35 of 43 layers streamed, 72 % of the token). Use the S6 method, not this figure.]** **No download was needed; the pip install was authorised by the user 2026-09-23.**

**With 46 closed, the queue below is down to ONE row, and it is the weeks-long REACH item.**

| order | item | what to open FIRST | cost | why this order |
|---|---|---|---|---|
| **1** | **A5 — the DeepSeek-V4 port** (row 28b) | brief: [`…-A5-deepseek-port-BRIEF.md`](docs/2026-09-20-arch-A5-deepseek-port-BRIEF.md) | **weeks**, abandon decided at step 1 | **The only row left in this table now that 46 is closed (2026-09-23).** Head of the REACH track since A4 closed, and the user's mandate fixes that order. It was ranked last only on cost: it is kernel-gated on sm_120 and the model still does not load. It inherits A4's two carried checks. No download — the checkpoint is on disk, complete (48/48 shards, 157 G). |

**The SPEED slot is empty, and with row 32 blocked there is now NO large speed lever pickable at
all.** Row 49's two successors were both spent or refuted on 2026-09-21: the ring-pool optimum is
n=2 (shipped, +1.89 % decode) and the read-shape arm was built and REVERTED (−0.40 % decode despite
removing 66.7 % of the reads). **Everything in the table above is correctness, hygiene or reach —
none of it makes Vaayu faster on a target run here.** A new speed idea has to cut bytes per token
or raise delivered GB/s (rows 31-39's decomposition), and must not need a download.

**Also open, smaller, the row carries them:** **47-m6 — now the LAST A3 loose end**, since 47-m7
closed: A3's fp32 rotary table (`cac352e`) changed numerics for EVERY model and only gpt-oss's side
was measured. Then 47-m4 (sliding-window KV, 0.813 GB), **47-m5-b** (new, from 47-m5), 47-m1/m2/m3,
row 40 `[bug]`, row 48 `[bug]`. (47-m5 and 47-m7 are CLOSED.)

**Added 2026-09-12: rows 31–39**, from a user-requested sweep of *"what else can speed Vaayu up"*, priced against a
measured decomposition of one decode token. Detail:
[`docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md)
(**gitignored — these rows are the durable copy**). **The one fact that reorders everything: on both actively-tested
models 70–83 % of a decode token is the NVMe being busy** (Qwen 0.772 s of 0.930; MiniMax 0.396 of 0.562), 8–9 % is
serial non-expert work and 9–20 % is exposed host time. So **any lever that only touches host-side time is capped at
+9 % Qwen / +20 % MiniMax before it starts** — which is independently why row 26 measured the CPU-AVX2 hybrid's
realistic ceiling at +6.8 % / +22.0 %, and why further Rust is dead (SETTLED: GIL 0.11–0.72 %, GC 0.26–0.41 % of
decode). **Everything that can pay more than that cuts bytes per token or raises delivered GB/s.**

**Model set re-set by the user 2026-09-20: Qwen3-235B is RETIRED AND DELETED — do not test it, do not put it in a gate;
the accumulator gate is now ×2, not ×3, and the fused-EXL3 forward has NO on-device model. DeepSeek-V4-Flash is
COMPLETE on disk (48/48 shards, 157 G, since 2026-09-20) and still cannot load — `deepseek_v4` is unknown to
transformers 5.5.4 and the repo ships no `auto_map`. That is row 28b / A5.** `CLAUDE.md` §0/§5 updated
the same day; `test_generate.py`'s hardcoded default repointed from the deleted checkpoint to Qwen3-Next.

| # | row | what it is | start gate | stop rule | cost | brief |
|---|---|---|---|---|---|---|
| ~~1~~ | ~~**36**~~ | **PARKED 2026-09-20 — gate fired, then two spikes killed every lever.** `serial` is **74.8 % GPU-bound**; its matmul half (12.59 ms) is at **80–86 % of the RTX 5060 Ti's 448 GB/s**, a floor; `use_torch_compile` **CRASHES** (cudagraph-trees step boundary, reproduced 2×); without cudagraphs it is **24–35 % SLOWER** (46 dynamo graphs — broken by the engine's own `time.perf_counter` instrumentation and by `aten.bincount`). Elementwise term still real but **not ~1 day**. | — | — | done, ~1 day | [`…-SPIKE.md`](docs/2026-09-20-row36-serial-split-SPIKE.md) · [`…-torchcompile-AB-RESULTS.md`](docs/2026-09-20-row36-torchcompile-AB-RESULTS.md) |
| **1 (NEXT — SPEED)** | **49** | **STEP 2A SHIPPED 2026-09-20 (+6.90 % decode); B RE-RANKED AND REFUTED 2026-09-21 (<= +6 %).** The probe that killed B also priced its successors on the same run: **the drive is busy 75.7-76.4 % of a MiniMax decode token at 6.03-6.05 GB/s, idling 1.57 ms in EVERY layer** — **BOTH SUCCESSORS ARE NOW SPENT OR IN DOUBT (2026-09-21).** (a) The ring-pool / read-shape arm ran and the rate hypothesis was BACKWARDS: `VAAYU_RING_POOL_N=4` REGRESSES -2.07 %, the optimum is **n=2, shipped at +1.89 % decode / +1.85 % 2k prefill** — and the read-shape half was then BUILT AND REVERTED, because removing **66.7 % of the reads** and raising delivered bandwidth **6.07 -> 6.50 GB/s** bought **-0.40 % decode**. That is a direct experiment refuting the **<= +12.8 %** rate lever. (b) The per-layer gap's **>= +31 %** came from the same "token collapses onto disk-busy time" arithmetic the read-shape arm just falsified (busy fell 5.4 pp, token did not move, and the >= 1 ms gap COUNT was unchanged) — do not quote it as measured. Its built form, `VAAYU_LOOKAHEAD`, is **+2.5 % on MiniMax at N=1** (the -3.7 % is the N=3 arm the server never runs) and costs **+18.0 % bytes/token**, which alone caps the token floor near **+11 %**. Original text: **MiniMax-M2.7's expert H2D is SYNCHRONOUS on purpose** (`moe_patch.py:1110`/`:1119`, `non_blocking=False`), so it cannot hide behind the disk read the way the fused forward's does. Row 26 measured it at **0.180 s of a 0.596 s token — 30.2 %, the largest removable phase on that model**, and ORDERED this priced before any CPU path. Never done. **Bigger than the CPU hybrid's whole 0.114 s prize, and needs no CPU kernel or new on-disk format.** | open now — row 26's precondition 1 (after row 27) was met 2026-09-12 | **step 1: `h2d_pct` < ~15 % of `T_prof` → close the row AND the CPU-hybrid half with the number** | step 0 ~1 h · step 1 ~1 h · step 2 1–3 days | [`…-row49-minimax-h2d-BRIEF.md`](docs/2026-09-20-row49-minimax-h2d-BRIEF.md) · row **49** below |
| **1b (REACH track — user mandate, NON-NEGOTIABLE 2026-09-20)** | **47 = A1→A2→A3, then A4→A5** | **Vaayu must serve different architectures.** Decomposed into 5 plans. ~~**A1** guards~~ · ~~**A2** fused-3D indexing~~ · ~~**A3** gpt-oss-20b end to end~~ **(all three CLOSED 2026-09-21; gpt-oss generates through the server, bit-identical to stock)** · ~~**A4** shared-expert pinning~~ **(CLOSED 2026-09-23, premise REFUTED, zero code — already pinned, 48/48 layers, 4.83 GB)** · **A5** DeepSeek-V4 port (NEXT). | **A5 open now** (A1-A4 closed), no dependency | A5 has an abandon gate at its first task; A3's PASSED (packed MXFP4 runs here) | A1 ~1d · A2 ~1d · A3 ~3–5d · A4 **closed in ½ day, no build** · A5 weeks | [`PROGRAM`](docs/2026-09-20-architecture-support-PROGRAM.md) · row **47** below |
| ~~2~~ | **35** | ~~Per-model measured gate for `VAAYU_LOOKAHEAD` instead of the hard block.~~ **BLOCKED 2026-09-20 — DO NOT PICK.** Its +11.7 % was measured on **Qwen3-235B, deleted 2026-09-20**, so its stop rule (*reproduce +11.7 % on Qwen at N=1 through the server*) is **unrunnable**. Its own kill condition (*dies if S2 lands*) **fired 2026-09-13**, and Qwen3-Next's disk is now 0.42 % of a token. The only still-streaming model, MiniMax, is the **−3.7 %** arm — **but that is its N=3 arm, and the server runs N=1, where the same post-P4 measurement reads +2.5 %** (`check_lookahead_flag_safe`). **CLOSABLE ON ARITHMETIC 2026-09-21:** at +18.0 % bytes/token against a drive busy 76 % of a MiniMax decode token, the token floor is ~+11 % — **below this row's own ≥ +15 % re-open bar**, so the bar cannot be met by this predictor. | **blocked** | re-opens only on a measured **≥ +15 % on MiniMax at N=1** on a current build | — | row **35** below |
| **3** | **32** | Lower-bpw EXL3 build (5.01 → ~3.0 bpw) of a model already on disk. | open | **×1.50 projected, a download and ZERO code**; under ~3.25 bpw Qwen's working set fits its VRAM budget, and it is the only thing that re-opens row **37** | a download | row **32** below |
| **3** | 44's **F3 remnant** | pinned-H2D half only (~0.6 s of an 18 s 2k prefill). **Not recommended** — listed so nobody re-derives it. | — | — | ~1 day | **STILL OPEN** (row 44 closed and pruned 2026-09-23) |

**After those, on the user's call:** row **40** `[bug]` (arena never capped at the model's
expert bytes — latent, 0 measured impact on this box's models; shared sizing → both accumulator gates); row **48**
`[bug]` (`use_torch_compile=True` crashes); row **36** (PARKED — its gate FIRED at **32.34 / 32.73 %**, then two
spikes killed every lever); and the blocked pair **32** (download declined) and **35**. Row **34** closed
2026-09-21 on its own stop rule (read depth is already 2–4).

## PRIORITY — worth working, in this order

| # | one line | why it is worth it | state | cost | brief (gitignored) |
|---|---|---|---|---|---|
| **32 · 35** (**34** closed 2026-09-21: depth already 2–4) | **The streaming-regime ladder — fallback if S2 stalls.** **32** run a lower-bpw EXL3 build of the models already here (5.01 → 3.0 bpw). **34** find out why the drive delivers 6.3 GB/s when it plateaus at 7.1. **35** give `VAAYU_LOOKAHEAD` a per-model measured gate instead of a hard block. | 32 = **×1.50 projected, a download and zero code**, and below ~3.25 bpw Qwen's working set drops under its VRAM cache budget so the 62/94-layer restriction disappears entirely. 34 = ×1.10, probe first. 35 = **+11.7 % Qwen measured**, gate machinery already built. **All three only pay while experts stream from disk — if S2 lands they mostly die.** | `[ ]` **priority 4** | 32: a download · 34: ½ day probe · 35: ~2 days | §2.2 / §2.4 / §2.5 · rows **32**, **34**, **35** below |
| **28b** | **Port DeepSeek-V4-Flash-0731** (step S4) — kernels first on sm_120, then HF modeling code + `auto_map` the MiniMax way, key remap, persist, measure. | Projected **~2.5–2.7× Qwen's decode** from 2.58×-lighter tokens. **Weeks**, mostly kernels and numerics. Its disk keys already satisfy the streaming index — the hard part is the model. | `[ ]` **priority 6** | weeks | [`…-row28b-IMPL-deepseek-v4-port.md`](docs/2026-09-12-row28b-IMPL-deepseek-v4-port.md) · row **28b** below |

## CONDITIONAL — real levers with a stated gate, added 2026-09-12. Not optional, not yet startable.

| # | one line | gate that opens it | why it is worth it then | state |
|---|---|---|---|---|
| **36** | **CUDA graphs / fused per-layer dispatch. PARKED 2026-09-25 — the gate FIRED and the build was REFUSED anyway, on three documents from 2026-09-20 that must not be re-derived. (a) `…row36-gate-recheck-RESULTS.md`: strict `serial / T_unprof` = **32.34 / 32.73 %** vs the ≥ 30 % rule, two replicates within 1.2 % — it fires, and `serial` itself never moved (0.0363 → 0.0360 → 0.0368 s/tok across three builds; the share rose only because the token shrank). (b) `…row36-serial-split-SPIKE.md`: **the build's own stop rule then fired against it — `serial` is GPU-bound, 74.8 % GPU kernel time vs only 25.2 % host gap against a ≥ 40 % host-gap rule.** Host time is 9.34 ms = **8.3 % of the token**, and a whole-step capture is impossible regardless: the MoE path does a blocking D2H routing sync **48× per token**, so only non-expert segments are capturable. (c) `…row36-torchcompile-AB-RESULTS.md`: the cheap version does not pay twice over — `use_torch_compile=True` as shipped (`mode="reduce-overhead"` = compile **plus** CUDA graphs) **CRASHES**, reproduced twice; with graphs off (`mode="default"`) it is correct but **24–35 % SLOWER**. **The re-scoped successor is S3 in the sequence at the top of this file** — the 4,353 elementwise launches (7.80 ms/token) in `Qwen3NextGatedDeltaNet`, which attacks device time AND host gap at once where graphs attack only the smaller one.** | **a resident model measures `serial / T` ≥ 30 %** — **MEASURED 2026-09-13 (S3-pre): 23.0 / 23.4 % strict (NOEXP), 32.5 / 32.8 % wide (`other`/T). Straddles the rule; does NOT fire strictly.** **Re-read 2026-09-18 on the 30-P build (S3 re-gate runs): 26.4 / 26.7 % strict, 35.9 / 35.9 % wide — `serial` unchanged at ~0.036 s, the share rose only because the token shrank; still does NOT fire strictly (derived: fires once `T_unprof` ≤ ~0.120 s).** **RE-CHECK IS NOW THE NEXT TASK ON THE BOARD (2026-09-20): row 44 + row 30's RAM-ceiling fix took Qwen3-Next decode to 9.06 tok/s = `T_unprof` 0.1104 s, UNDER the trigger, so the strict share is ≈ 32.6 % if `serial` still sits at ~0.036 s — but row 44 rewrote the very dispatch the NOEXP arm brackets (F5 replaced a per-expert `torch.where` one-hot mask, ~960 tiny H2D copies/token), so `serial` may have moved either way. MEASURE IT, do not estimate — the row says so. Brief: [`…-row36-gate-recheck-BRIEF.md`](docs/2026-09-20-row36-gate-recheck-BRIEF.md).** | `serial` is 0.074 s of Qwen's 0.930 s today — 8 %, invisible. Row 29 projected it becomes the LARGEST phase once experts are resident (38 ms of Qwen3-30B-int4's projected 49 ms). Nothing owns it today. | `[x]` PARKED |
| **37** | **Re-gate speculative decoding on MoE.** The gate refuses Qwen with `off_union` — observed verify union **19.31 GB against a ~17 GB budget**. That is a **byte** bound. | row 30's S2, or row 32's lower-bpw build, lands | At 3.0 bpw Qwen's union falls to ~11.6 GB and fits. The break-even acceptance falls too: with experts resident, extra drafted positions cost ~0 extra disk bytes, which is the dominant term today. Machinery is BUILT (`speculative/adaptive_gate.py`, row 13's `even-4` trial, P9's +4.16 %). **Prefer a checkpoint that ships an MTP module** — a free drafter, no vocab mismatch, no second load. | `[ ]` **Gate MET 2026-09-13 (S2 landed). CLOSED FOR HYBRIDS 2026-09-19 (row 42 (b): verify 2.6–4.8× t_dec, drafter needs p ≥ 0.87). Open only for row 32's route.** |
| **38** | **Mixed-bpw by expert hotness** — hot experts high bpw, cold tail 2–3 bpw. | **trace arithmetic clears a 20 % byte reduction** | NOT the refuted pinning idea: Plan F/B-ph2 refuted **residency** by hotness; this is **precision**, which pays on every disk read whether or not the expert is ever cached. Plan F's "share is linear in N" is the argument FOR it — the tail holds most of the bytes. Likely loses to row 32 at a fraction of the effort; do the arithmetic first. | `[ ]` |
| **39** | **Expert slimming** — structured row / N:M prune of the intermediate dim, re-pack, stream fewer bytes per expert. | rows 31–35 measured | Research, not engineering; quality risk real and unbounded. Priced honestly: a 30 % row prune is a 30 % byte cut — roughly what downloading a 3.5 bpw build gets for free in an afternoon. | `[ ]` |

## NOT worth working now — and why (re-open only on the stated condition)

| # | one line | why not | re-open if | state | brief (gitignored) |
|---|---|---|---|---|---|
| **40** | `[bug]` The expert slab arena is never capped at the model's own expert bytes: a fused non-EXL3 MoE smaller than its arena pins RAM it can never fill (since 30-P). | Latent: **0 measured impact on every model on disk** — Qwen3-Next's experts are 43.5 GB against an 18.54 GB server-sizing arena, and both EXL3 models' experts exceed their arenas too. | a fused non-EXL3 MoE whose experts total less than its derived arena is going to be loaded — e.g. Qwen3-30B-A3B int4 (rows 29/30's smoke rung, NOT on disk): **~15–16 GB of experts, derived** | `[bug]` | row **40** below (no brief) |
| **6a-b** | `head_bytes` is probably not a transient; the 70B's 0.714 G floor is covered BY LUCK. | A GPU round that buys admission headroom, never tok/s — the reserve is floor-bound on flash. The current form errs SAFE. | someone edits `predict_peak_vram` (then this is a trap to read first) | `[ ]` | [`…-row6a-b-head-bytes-BRIEF.md`](docs/2026-09-10-row6a-b-head-bytes-BRIEF.md) |
| **5** | Does failing the 0.05 rel-L2 compression bar degrade TEXT? | `PREFIX_COMPRESS = None` is the server default, so it only affects users who opt in. | compression becomes a default, or a user reports degraded prefix-reused turns | `[ ]` | [`…-row5-quality-certification-BRIEF.md`](docs/2026-09-10-row5-quality-certification-BRIEF.md) |
| **23** | Prefill stager re-runs once per batched chunk (fused-EXL3). | `MAX_BATCH_SIZE = 1` → 0 % to users (§8). P10 banked batching as not recommended. | batching is enabled in the server | `[ ]` | [`…-row23-batched-chunk-stager-BRIEF.md`](docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md) |
| **4** | 2× KV overshoot, ~0.83 GB at Qwen's 8192. **Since row 3 (2026-09-11) non-EXL3 models pay it too, as CONTEXT: 1B flash `max_seq_len` 91106 → 65536.** | PARKED — a larger 1.22 GB giveback measured decode +0.4 %. | the condition in its brief (per-cell VRAM drops enough for N=2), **or a non-EXL3 user needs the lost context** | `[ ]` parked | [`…-row4-kv-overshoot-PARKED-BRIEF.md`](docs/2026-09-10-row4-kv-overshoot-PARKED-BRIEF.md) |
| carried | P10 batching minors (STILL OPEN). **One is NOT batching-gated:** the sentinel-only cancel path is reachable at N=1 — a job QUEUED behind the running one, cancelled by `cancel_all()` at SIGTERM. | Coverage only; no defect known on any of them. | a shutdown/cancel bug is reported | `[ ]` / `[bug]` | [`…-carried-p10-batching-minors-BRIEF.md`](docs/2026-09-10-carried-p10-batching-minors-BRIEF.md) |
| carried | `_finish` stop-string identity · `expert_sim.py` anchor gap · row **8** `tensor_align` | Reporting / a diagnostic tool / dead code — none changes the engine. | a client needs the stop string / someone quotes `expert_sim` small-budget absolutes | `[ ]` | see STILL OPEN |
| carried | No resident-size checkpoint on disk. | A download, not code — but still the single largest usability change for the agent (streamed MoE runs ~0.56–0.86 tok/s). | — (do it whenever) | `[ ]` | [`…-carried-resident-checkpoint-BRIEF.md`](docs/2026-09-10-carried-resident-checkpoint-BRIEF.md) |

**Every open item now has a self-contained cold-pickup brief** (written 2026-09-10). They live in
`docs/`, which is **GITIGNORED** — so the brief is the detail, and the row in this file is the
durable copy of the conclusion (CLAUDE.md §7). If a brief is missing after a fresh clone, the row
still carries the numbers, the traps and the stop rule.

Carried-forward items have briefs too:
[`…-carried-finish-stop-string-BRIEF.md`](docs/2026-09-10-carried-finish-stop-string-BRIEF.md) ·
[`…-carried-expert-sim-anchor-gap-BRIEF.md`](docs/2026-09-10-carried-expert-sim-anchor-gap-BRIEF.md) ·
[`…-carried-resident-checkpoint-BRIEF.md`](docs/2026-09-10-carried-resident-checkpoint-BRIEF.md) ·
[`…-carried-p10-batching-minors-BRIEF.md`](docs/2026-09-10-carried-p10-batching-minors-BRIEF.md)

**Also live, carried out of closed rows (see STILL OPEN):** no resident-size checkpoint on disk
(a download, not code, and NOT a blocker); `_finish` cannot report WHICH stop string fired (needs a
new premise); `expert_sim.py`'s anchor gap is unexplained (do not quote its small-budget absolutes);
and a batch of `MAX_BATCH_SIZE = 1`-gated minors from P10.

**Standing hardware tax, warned at every boot, not a code row:** `rustdesk` permanently holds 0.14 GiB
of GPU. ~~NVMe APST~~ — **measured 2026-09-11 (row 25): NULL on Qwen and MiniMax at N=1**, not a tax on the dominant cost
(the drive's idle gaps are < 100 ms, APST's timeout). The boot prints it as info, not a warning; row 25 is closed and pruned.

**Before proposing ANY perf work, read SETTLED.** It is the list of things already built and
refuted, with the number that killed each one.

---

# ▶▶ IMPLEMENTATION SEQUENCE — added 2026-09-12, the ORDER to build in

**This section is the build order. The PRIORITY table above is the priority; this is the sequence, with the gate that
lets each step start.** Every step names a stop rule. Do not start a step whose gate has not reported.

| # | step | row | cost | gate that lets it start | stop rule |
|---|---|---|---|---|---|
| **T4** ← **LIVE BRANCH** | **The residency-regime levers — ONLY if S2 lands.** **S2 LANDED 2026-09-13 and the disk is now 1.7 % of a decode token here (`gb_tok=0.030`, 96.5 % RAM-cache hit) against 70–83 % on the EXL3 models — the branch condition is MET.** Take S2b's three levers first; 36's own gate below is still unmeasured on a resident model. **36** CUDA graphs / fused per-layer dispatch — **its gate is now MEASURED (S3-pre, 2026-09-13): strict `serial`=NOEXP is 23.0 / 23.4 %, BELOW the 30 % rule; the wide reading (`other`/T, all host time no expert placement removes) is 32.5 / 32.8 %, above it. The two readings straddle the rule; the 9.4 pp between them is the MoE loop's own unbracketed host work, which is exactly what fused dispatch targets.** **37** re-gate speculative decoding, whose refusal (`off_union`, 19.31 GB vs ~17 GB) is a BYTE bound that S2 dissolves. | **36 · 37** | 36: ~1 week · 37: ~2 days | **36:** a resident model measures `serial / T` ≥ 30 %. **37:** S2 or row 32 landed | 36: `serial / T` under 30 % → it is still invisible, close it · 37: gate still refuses on the resident model → close it |
| **S4** | **Port DeepSeek-V4-Flash.** TileLang/Triton kernels → HF modeling code + `auto_map` → key remap → **an engine change so the streamed path can run it** → persist → measure. (A5.6 found `VaayuBaseModel.forward` never calls this model's forward — see A5.7's §0. That step was not in this sequence when it was written.) | **28b** | **weeks** | S2 done (it proves the non-EXL3 MoE path end to end, cheaply) | kernels will not build/validate on sm_120 → abandon at step 1, ~2 days lost |
| **S5** | *optional, after S4:* the 10.12 GiB **MTP draft module** already in that checkpoint — speculative decoding with no second model. **Qwen3-Next ships one too** — if S2 lands, this arrives with it and pairs with T4's row 37. | **28b** §3 step 6 | ? | S4 measured (or S2, for Qwen3-Next's own MTP) | — |
| **T5** | **Research tier. Never ahead of anything above.** **38** mixed-bpw by expert hotness — *do the trace arithmetic, not the repack*; **39** expert slimming. | **38 · 39** | weeks | 38: trace arithmetic clears a 20 % byte cut. 39: rows 31–35 measured | 38: under 20 % → row 32 beats it at a fraction of the effort, close it |

**Reference for S2 and S4:** [`docs/2026-09-12-IMPL-adding-new-architectures-PLAYBOOK.md`](docs/2026-09-12-IMPL-adding-new-architectures-PLAYBOOK.md)
— the five contracts (C1 transformers builds it · C2 quantizer dispatches · C3 module tree exposes experts+router ·
C4 **disk keys carry per-expert tensors** · C5 HF forward contract), the tier decision tree, the format cheat-sheet,
and Vaayu's seven known generality gaps.

**Two facts that decide more than anything else in this sequence:**

1. **C3 and C4 are different questions and they disagree for modern MoE checkpoints.** `ExpertOffsetIndex` streams
   only what matches **`\.experts\.(\d+)\.`** on disk; transformers 5.5.4 builds `Qwen3MoeExperts` /
   `Qwen3NextExperts` / `GptOssExperts` as **fused 3D `nn.Parameter`s** in memory. GPTQ/AWQ int4 checkpoints are
   per-expert on disk (good for C4, risky for C2); gpt-oss MXFP4 is one 3D tensor per layer (**C4 finds ZERO experts
   → no streaming at all**). A C4 failure is silent — it looks like a model with no experts.
2. **Bytes per token decides speed, not the quant lineage.** `moe_layers × top_k × 3 × hidden × moe_intermediate ×
   bpw/8`. Never take an 8-bit build over a 4-bit one (row 28: the fp8 build of the *same* model costs ~43 % of its
   decode). Never assume more NVMe concurrency helps (row P8: the drive plateaus at ~7.1 GB/s at 2 readers and
   **slows past ~4**).

**What is NOT in this sequence, and why:** the CPU-AVX2 hybrid is capped at **+21 %** on this box in the best case
(row 29), because the GPU link (18.7 GB/s measured) is only 2.5× slower than 6 P-cores (46.6 GB/s). **S2's model swap
is worth ~21× on its own** (0.85 → ~18 tok/s projected). S2 is the value; S3 is the user's stated goal and rides on
S2's measurement. If S2 lands and S3's gate fails, **S2 was still the biggest win on this board.**

**The third fact, added 2026-09-12, and it is why T3 and T4 are mutually exclusive branches rather than a list.**
70–83 % of a decode token is the NVMe being busy *because the experts are streamed*. **S2 exists to leave that
regime.** So T3's rows (32/34/35) buy a factor on a cost S2 is trying to delete, and T4's rows (36/37) buy nothing
until S2 has deleted it. **Branch on S2's verdict; do not work both ladders.** The two rows that are safe to do
regardless are **T1** (it re-sizes a cache every regime has) and **T2** (`top_k` cuts bytes when streaming and cuts
GEMM + H2D when resident). Physics primer and the arithmetic behind every projection:
[`docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md) §1.

---

# §0 · EXECUTION ORDER — what to build, in sequence (added 2026-09-06)

**This section is authoritative for ORDER and for the perf/not-perf split. It is NOT a second copy
of any row: each entry points at the row that owns the detail, and that row stays where it is.**
Classified by one question — *can this make Vaayu faster on a target actually run here?* Rewrite
this section whenever a row opens or closes; do not let it drift.

## HIGH PRIORITY — can make Vaayu faster. Build in this order.

**P1–P4 and P6 are all closed as of 2026-09-06** (P1 refuted, P2 shipped, P3 was already built, P4
accepted-and-documented, **P6 SHIPPED at +11.6–12.1 % on MiniMax-M2.7**). The vacancy bullets below
say what happened to each, and they are anti-regression notes, not history for its own sake.

**P10 CLOSED 2026-09-09** (measured, banked, not recommended — see its row). **P5 CLOSED
2026-09-09 too — root cause found and fixed, its perf premise refuted, 0 % on tok/s; its one
survivor (`head_bytes` is probably not a transient) is carried as OPTIONAL row **6a-b** and needs
a GPU round.** **No P-row is open. Row 21 CLOSED 2026-09-09** — both its STATED halves refuted,
its real cause found and FIXED (gate 2 priced the layer-major forward for a batch that runs the
per-chunk loop: over-charging 21-26 % at T10's ~4.2k contexts, UNDER-charging up to 1.02 GiB
elsewhere). **Do not re-open it to "let gate 2 see reclaimable VRAM"** — that half is dead, and
the server passes `reclaim=False` anyway. **Its A/B produced the finding that matters and it is
now row 22: batching is worth 1.58x on the module-list forward and 2.04-2.36x SLOWER on the
fused-EXL3 one, at the same context, both interleave directions.** **Row 22 CLOSED 2026-09-09
by an 8-arm sweep: the variable is CHUNK COUNT, not length (boundary `chunks = N`, crossed both
ways at FIXED length), the agent is the PREFILL STAGER re-arming once per `forward()`
(fused-EXL3 only — MiniMax stages ZERO times), and the fused forward WINS on decode at every
length while losing only on prefill — so the `expert_style` formation rule row 22 sketched must
NOT be built. It also refuted raising `prefill_chunk_size` as a mitigation: that costs serial
decode 2.46x on the N=1 path the server runs today. Its unfixed half is now row **23**.** **Row 19 CLOSED 2026-09-09 —
audited, and NO recorded drafter/speculative number was taken under `sdpa`; every one ran flash,
so nothing was owed a re-measurement and CLAUDE.md §0's 5.57× stands as to kernel. It also
established that row 16 can never reach a DRAFTER: `StandardDrafterDecoder` loads via stock
transformers, with no `pinning_budget` at all.** **Row 22 CLOSED 2026-09-09; the live work is row **23**, its unfixed half — and row 23
is `MAX_BATCH_SIZE = 1`-gated, so it is worth 0 % to users today. Weigh it against the reachable
single-stream rows (OPTIONAL **13**) before spending a GPU round on it.** Two NEW rows came out of P10's measurement and are the live work
now: ~~**row 20 `[bug]`**~~ — **FIXED 2026-09-09: the padding was a red herring, the batched
path computed RoPE without `rope_scaling`; both MoE checkpoints were never affected, and
`MAX_BATCH_SIZE` stays 1 on row 15's performance grounds, not on corruption** — and **row 21**,
the gate-2 ceiling that admits batching about once per process, **which row 20 no longer
blocks.** **Order: P10 first, and it is now MEASURED rather than argued.** Its refuting
measurement ran 2026-09-07 and the row PASSED at **1.48-1.78x end-to-end** at realistic agentic
context — the largest effect on this board by an order of magnitude, and mostly a PREFILL win, not
the decode win P6/P8 banked. It also found its own prerequisite: `_assert_context_fits` refuses
every N>1 at that context, so the server wiring must be preceded by fixing gate 2 or it ships
inert (**DONE — P10a, 2026-09-07**). Its second blocker, row **17**, is **FIXED 2026-09-08**, so
P10 is UNBLOCKED and its remaining work is the server wiring plus the flash@N=1 vs sdpa@N=2 A/B
that can still refute it. P5 is a sweep that may end in "not derivable — close the row". **P8 SHIPPED 2026-09-07 at +3.21 % on
Qwen3-235B — read its vacancy note below before proposing anything that touches the fused
forward's expert dispatch.**
**P9 was the cheapest, was built first, and SHIPPED at +4.16 % on 2026-09-07 — see its vacancy note
below before proposing anything that re-sizes the VRAM prefix for a verify pass.**
**P7 was the cheapest, was built first, and was REFUTED on 2026-09-06 — see the vacancy note
below before proposing anything that re-sizes the VRAM prefix.**

**`P7`/`P8`/`P9` here are §0 SLOTS in this file's own sequence.** They are NOT the
HANDOFF-2026-08-30 `P7`/`P8` (that series closed 2026-09-03), NOT the `p7-*`/`p8-*` probe files in
`plans/probes/`, and NOT the stale "row P7" that `CLAUDE.md` §0 still cites for dense speculative
decoding (that row was pruned 2026-09-03; its content lives in git history).

**No P-row is open, so this section has no table.** The closed slots' verdicts are the vacancy
notes below — they are anti-regression records, not history for its own sake. The two most recent,
**P10** (server batching) and **P5** (which chunk sets the peak), close the list; their rows were
pruned 2026-09-10 and their detail lives in `docs/` and in `git log -p -- todo.md`.

**Warnings that belong to this order, not to a later surprise:**
- **The P10 slot is VACANT — MEASURED 2026-09-09, BANKED, NOT RECOMMENDED, and its row is pruned.**
  12 arms (MiniMax + Qwen, both interleave directions) plus a dense smoke, through the real server.
  Batching is worth **1.442×** on a two-turn MiniMax session with the prefix cache ON (the server's
  default), **1.250×** with it off, and **nothing** on the warm steady-state turn (+0.9 % / +1.3 %,
  inside noise) — the formation policy correctly prefers prefix reuse and declines with BOTH gates
  at 0. **Qwen gets nothing and regresses nothing** (202.50 → 201.12 s, 0 batches). Three structural
  facts bound it: batching is admitted about **once per process**, the whole win is the **cold
  turn**, and **the prefix cache is worth 4.44× where batching's best is 1.44×**. `MAX_BATCH_SIZE =
  1` stands on THESE performance findings — no longer on corruption, since the padded-batch bug was
  FIXED 2026-09-09 (the batched path computed RoPE without `rope_scaling`; neither MoE checkpoint
  was ever affected). Four bugs found and FIXED on the way, THREE of which bite users who never
  enable batching: formation starvation (`94c8e1a`), the prefix-hit chunked-prefill OOM (`266ec80`),
  cross-session prefix cannibalisation (`efeb343`, **2.77×** on its own arm), the batched store-back
  (`845608e`). Suite 1838 → **1865**.
  [`docs/2026-09-09-P10-server-batching-RESULTS.md`](docs/2026-09-09-P10-server-batching-RESULTS.md).
  Its live residue is OPTIONAL row **23**, and row 23 is `MAX_BATCH_SIZE = 1`-gated, so it is worth
  0 % to users today.
- **The P5 slot is VACANT — CLOSED 2026-09-09; the root cause was WHICH CHUNK SETS THE PEAK and the
  row's own perf premise is REFUTED.** `predict_peak_vram` charged ONE chunk's terms against the
  FINAL KV capacity; a prefill's peak is the **max over chunks**, and which chunk that is differs
  between the per-chunk loop and the layer-major MoE branch (which skips `lm_head` for all but the
  last chunk, `vaayu_base.py:4581`). Fixed; gate **44/77 → 39/77** outside ±10 %, worst **273.1 % →
  152.7 %**, never above the old envelope, no new under-prediction, suite 1870 → 1875. **But the
  RESERVE DOES NOT MOVE on any geometry or kernel** — flash reserves are floor-bound at 3.5 GiB and
  `chosen` lands on a KV doubling, where a trailing chunk is full. **The expert cache does not grow
  and there is no decode win.** The saving lands on ADMISSION instead: MiniMax at 7700 tokens is
  charged **2.7488 → 2.3821 GiB (−13.3 %)**, Qwen zero (residency dominates).
  [`docs/2026-09-09-P5-which-chunk-sets-the-peak-RESULTS.md`](docs/2026-09-09-P5-which-chunk-sets-the-peak-RESULTS.md).
  Its one survivor is OPTIONAL row **6a-b** (`head_bytes` is probably not a transient), which needs
  a GPU round. See SETTLED for the two structural facts that outlive the fix.
- **The P1, P2 and P3 slots are VACANT.** P1 was measured 2026-09-06 and **REFUTED** (zero disk reads
  for GPU-resident weights, on 5922 decode visits in each of two arms — see SETTLED). **P2 was
  BUILT and shipped 2026-09-06** — the mixed-precision KV outlier split, Qwen 0.0683 FAIL → 0.0281
  PASS at 1.211 B/elem, closed under OPTIONAL row 5 with
  `docs/2026-09-06-P2-outlier-split-RESULTS.md`.
  **P3 was ALREADY BUILT when this section first listed it — closed 2026-09-06 with no new code.**
  This section was written 2026-09-06 quoting the S1f row's *"Not built, not designed"*, which S1f
  wrote on 2026-09-05 **before** S1d landed later the same day; S1d's own row — closed 2026-09-05, pruned
  2026-09-10, in `git log -p -- todo.md` — is the authority and it says BUILT, WIRED and MEASURED. Verified in code, not taken on the row's word:
  `vaayu/speculative/adaptive_gate.py` (3 commits, `2811d6f` → `800467d` → `87b36e8`) sizes the
  verify pass from `_expert_history`'s OBSERVED per-layer union, `_build_spec_gate()`
  (`vaayu_base.py:4794`) wires it MoE-only and caches it across requests, and both static
  zero-overlap checks (`vaayu_base.py:2134`, `:2406`) are now scoped to `drafter_model_path` so a
  PLD proposer is admitted on observation instead. `tests/test_adaptive_spec_gate.py` +
  `..._wiring.py`: **39 passed**. **P3's own headline configuration was measured post-build** — a
  4-position verify IS `draft_tokens=3`, and the S1d MiniMax-M2.7 arm ran exactly it:
  `decision="run"`, `observed_union_gb=12.45` against `budget_gb=14.37`, **1.7795 → 2.3253 tok/s
  = 1.307×**, ids byte-identical. **Do not re-derive or rebuild union-based sizing; it exists.**
  The one piece deliberately NOT covered: a weighted drafter (`drafter_model_path`) on MoE still
  uses the zero-overlap bound — refuted a fortiori by S1e-bis (3.09 GB of a 6.3 GB expert VRAM
  cache), so it costs nothing measured today. Numbers here are SLOTS in an order, not stable
  ids; P5 keeps its number so references elsewhere in this file stay valid, and new rows take
  the next free number rather than renumbering. **This §0 `P6` is not the HANDOFF-2026-08-30 `P6`**
  (that one closed REFUTED 2026-09-03).
- **The P6 slot is VACANT — BUILT, MEASURED and SHIPPED 2026-09-06.** `batched_generate` now sizes
  the VRAM cacheable prefix on the batch's routing union (`batch_union_per_layer` =
  `min(num_experts, N*top_k)`, derived) instead of one stream's `top_k`, and restores the load-time
  set in its `finally`. **MiniMax-M2.7 N=3: 2.940/2.910 → 3.280/3.265 tok/s = +11.6–12.1 %**, hit
  rate 2.35 % → 48.4 %, evictions 63 676 → 17 539, prefix ALL 62 → 34/62. **Qwen3-235B N=3: NO
  effect** (1.795/1.782 → 1.791/1.805, inside the baseline's own 0.7 % spread) although its hit
  rate went 0.04 % → 56.9 % and evictions 38 574 → 5 431, prefix 43 → 14 of 94. Greedy ids
  byte-identical in every arm. **The asymmetry is measured, not reasoned:** experts reaching
  `_dispatch_expert_misses` per decode layer-visit fell 16.52 → 12.50 (−24 %) on the module-list
  forward, which filters VRAM-resident experts BEFORE dispatch, and stayed at 14.38 on the fused
  forward, which does not. **RAM misses/visit were unchanged on both (7.27→7.23, 6.48→6.47), so
  bytes read moved only −0.33 % / −0.05 % — the win is dispatch-side work, NOT I/O. Never quote P6
  as a byte reduction.** `docs/2026-09-06-P6-batch-union-vram-prefix-RESULTS.md`. The bound is
  ~1.7× pessimistic against the measured union and was deliberately NOT tightened; whether a longer
  prefix pays is unmeasured (row **10**, closed as P8 below and pruned 2026-09-10).
- **The P7 slot is VACANT — BUILT, MEASURED and REFUTED 2026-09-06, and the code was REVERTED.**
  Sizing the batch VRAM prefix on the OBSERVED routing union instead of P6's zero-overlap bound
  measured **−10.85 / −11.54 %** on MiniMax-M2.7 at N=3 (2.913/2.891 against P6's 3.275/3.261) —
  it hands back P6's entire win, landing exactly on the pre-P6 number. Hit rate 48.4 % → **2.38 %**,
  evictions 17 533 → **61 278**, greedy ids identical in every arm. **The row's PREMISE failed, not
  the implementation.** A whole-decode routing trace shows the per-layer union RISES 10.47 → 20.55
  across 64 decode steps (whole-decode mean 16.91 — the brief's 16.51 was RIGHT), so an 8-step
  window under-counts steady state by 2.0×; and more decisively, **`compute_vram_layer_range` sets
  capacity/layer = the charged load, so P6's 1.17× over-charge IS the reuse headroom, not slack.**
  The union over k=2 consecutive steps is 26.38, already above the bound of 24 — P6's capacity of
  24.34 sits between the k=1 union (20.55) and the k=2 union (26.38), which is precisely the 48.4 %
  hit rate. A forced-charge sweep puts the optimum at P6's bound: 24 → 3.257 tok/s, 21 → 3.268
  (+0.34 %, inside noise), 18 → 3.141 (−3.54 %), ALL 62 → 2.902 (−10.9 %). **Never quote the
  bound's pessimism as slack, and do not re-open this to "measure the union better" — a
  perfectly-measured union is the wrong quantity.** Row **11** (pruned 2026-09-10) ·
  `docs/2026-09-06-P7-observed-union-prefix-RESULTS.md`. It left ONE live question, which is NOT
  this row: nothing derives the ~1.2× headroom the sizing depends on, so a geometry with a
  different bound/union ratio has no reason to land right. Open that on its own evidence.
- **The P9 slot is VACANT — BUILT, MEASURED and SHIPPED 2026-09-07 at +4.16 %.**
  `speculative_generate` now sizes `_vram_cacheable_layers` on the verify width
  (`batch_union_per_layer(geometry, draft_tokens + 1)` = the same
  `min(num_experts, positions * top_k)` bound Fix #9 uses for admission) and restores the load-time
  set in a new `finally`. **Trap #1 did NOT refute it:** `compute_prefill_pass` sets
  `verify_cap = draft_tokens + 1`, so a verify pass is explicitly DECODE and both MoE forwards do
  consult the VRAM cache during it. **MiniMax-M2.7, `draft_tokens=3`, prompt-lookup, 4 interleaved
  FIRST-CUT arms: 1.2044/1.2048/1.2053/1.2065 → 1.2542/1.2592/1.2590/1.2568 tok/s**, mean
  **1.2053 → 1.2573 = +4.32 %** (worst pairing +3.95 %, best +4.55 %; baseline spread 0.18 %); the
  SHIPPED version measured +4.16 % on the same baseline (see the Qwen paragraph below). Prefix **ALL 62 →
  30/62**, decode hit rate **5.4 % → 37.6 %**, evictions **74 961 → 23 721 (−68 %)**, cache lookups
  79 221 → 38 070. Greedy ids byte-identical in every arm, and the gate reported the SAME state in
  all four (`observed_union_gb=12.73`, `measured_acceptance=0.6875`, `decision="run"`) — the
  treatment moved the prefix and nothing else. **The win is dispatch-side work, not I/O** (the
  module-list forward's pre-filter, P6's mechanism); never quote P9 as a byte reduction. Smaller
  than P6's +11.6 % because this is single-stream speculation, not N=3 batched decode — the drafter's
  own wall-clock is time the cache cannot touch. Sized ONCE per call
  rather than per gate phase (the gate latches per PROCESS, `CALIBRATE_STEPS=8`); batching and
  speculation still cannot combine, and if that changes the width is `N * (draft_tokens + 1)`, not
  the max. **The bound was deliberately NOT tightened — P7 already refuted that family.**
  **THE QWEN ARM CAUGHT A REGRESSION, and the fix for it is part of what shipped — read this before
  touching the sizing again.** Qwen's gate latches `off_union` after calibration (observed union
  19.31 GB against a 16.79–17.09 GB budget) with `drafting_rounds: 0`, so it NEVER drafts and every
  round of that call is 1-wide. Sizing unconditionally for 4 positions charged it anyway for the
  whole call: **65/94 layers → 16/94 and 0.9740 → 0.9423 tok/s, −3.25 %**, every replicate below
  every baseline replicate. **Note the trap: the losing arm's hit rate was HIGHER (71.6 % vs
  43.1 %) — 16/94 leaves 78 layers uncached. A hit rate is not a speedup.** The premise that failed
  was "the narrow phase is a handful of steps once per process": true when the gate reaches RUN,
  FALSE when it refuses, because `off` latches and the call finishes on the suppressed-draft path
  1-wide to the end. **Fix: size only when `gate is None or gate.should_draft()`**, read once at the
  top of the call (the sizing moved inside the `try` to see the gate). Post-fix on device: Qwen back
  to **65/94 at 0.9671**, inside its 0.9693–0.9786 baseline band; MiniMax's warm call now unsized
  (ALL 62, 1.2051, matching the p9=0 warm arms) and its timed calls still **30/62 at 1.2573/1.2536 =
  +4.16 %**. MiniMax forgoes the gain on the FIRST call of a process — bounded, one-time, and in the
  safe direction. **Qwen's speculative steady state remains unmeasurable here** (its gate refuses
  before a single draft), so P6's fused-forward finding is untested by this row.
  `docs/2026-09-07-P9-spec-verify-prefix-RESULTS.md`; harness `plans/probes/p9_spec.py` +
  `p9_ab.sh`; tests `tests/test_spec_vram_prefix.py` (+15).
- **The P8 slot is VACANT — MEASURED, BUILT and SHIPPED 2026-09-07 at +3.21 % on Qwen3-235B, and
  the row's own premise was REFUTED on the way.** The measurement first: dispatch-side work for
  VRAM-resident experts is **0.021 % of decode** (the WHOLE `_dispatch_expert_misses` call is
  0.632 %), so **P6's +11.6 % on MiniMax was never dispatch-side work** and **P6 §2 is wrong where it
  says the module-list filter removes "slab acquire, H2D, per-expert attach/detach" from dispatch** —
  dispatch does none of those for a resident expert on either forward, and the H2D + QMatrix skip
  both forwards get happens in the COMPUTE loop. What the probe found instead: **1246 routed experts
  were VRAM-resident AND a RAM miss = 14.74 GB = 2.45 % of decode bytes read for weights already on
  the GPU** (P1 measured 0 against its 0.1 %-hit cache). Cause, derived: on a prefix layer RAM's
  per-layer cap is charged over ALL MoE layers (15 on Qwen) while VRAM's budget is concentrated on
  the prefix (25.5), so **RAM evicts first** — which is also why the RAM cache returns a **0.04 %**
  hit rate to non-resident experts there (56.4 % off-prefix). **This ANSWERS P1's objection instead
  of ignoring it**, and the shipped fix keeps the RAM refresh anyway. Shipped: A/B 1.7909 → 1.8485
  tok/s **+3.21 %**, bytes −2.44 %, greedy ids byte-identical, MiniMax unchanged. **+3.21 % exceeds
  the ≤1.8 % the byte saving alone predicts and the excess is NOT attributed — never quote P8 as a
  byte-to-time conversion.** Row **10** (pruned 2026-09-10) ·
  `docs/2026-09-07-P8-fused-predispatch-filter-RESULTS.md`. It leaves ONE live question, which is
  NOT this row: the same measurement shows the RAM expert cache is **strictly dominated by the VRAM
  cache on every prefix layer** (0.04 % useful hits). Whether prefix layers should keep a RAM cache
  at all is unmeasured — open it on its own evidence.
- **The P4 slot is VACANT too — CLOSED 2026-09-06, "accept and document", its own second option.**
  The exploration cost is **one-time per process** and worth **≤ `TRIAL_DRAFTED_TOKENS` ×
  `breakeven_acceptance` token-equivalents of decode** — 25.3 on Qwen3-235B, 29.3 on
  MiniMax-M2.7 — so the 6.8 % is an artifact of dividing a fixed cost by a 128-token benchmark,
  not a property of the engine. Full derivation, the corrected cost model, and why bounding TRIAL
  was REFUSED: the P4 row in **STILL OPEN**. **Do not re-open it to "shorten TRIAL"** — that
  trades a bounded one-time cost for a permanent wrong latch.
- **P4's investigation filed one new item — the CONTENT-BLIND latch — and it was MEASURED and
  REFUTED the same day. It never took a P-slot.** See SETTLED. Do not re-file it from S1e-bis's
  0.286-vs-0.927 pair: that pair is two PROMPTS, not two regimes inside one answer, and the
  difference is exactly what the measurement killed.
- **P3 HAS a tok/s number now — 1.307× on MiniMax-M2.7, and 0.932× on Qwen3-235B.** The warning
  this bullet used to carry (*expert bytes and round arithmetic only; do not quote a speedup
  without an on-device A/B*) was discharged by S1d's three on-device arms. The Qwen arm is what
  opened P4 — the gate correctly refused Qwen and still cost 6.8 %, because TRIAL's exploration is
  paid before the refusal latches — and P4 closed on the finding that the cost is fixed and
  one-time (see two bullets above).

## OPTIONAL — cannot make Vaayu faster on a target run here. Never ahead of the HIGH-PRIORITY rows.

Correctness, capability, tooling and hardware. Real work, and some of it matters — none of it moves
tok/s or TTFT on Qwen3-235B, MiniMax-M2.7 or the 70B dense today.

| item | what it is | why it is not perf | where |
|---|---|---|---|
| **6a-b** `[ ]` | `head_bytes` is probably not a transient at all | Correctness of the peak form, and §8-bounded the same way P5 was: the reserve is floor-bound on flash, so tightening it further buys admission headroom, not tok/s. Needs a GPU round (the 70B's 0.714 G floor). | row **6a-b** |
| `_finish` stop-string identity | Cannot report WHICH stop string fired | Reporting only, and no client reads it. Needs a new premise (matching stop strings ourselves). | STILL OPEN |
| `expert_sim.py` anchor gap | 31.9 % simulated vs 43.9 % live | A diagnostic tool's accuracy, not the engine. The obvious hypothesis was built and REJECTED. | STILL OPEN |
| No resident-size checkpoint | Needs a ~14B coder at 4–4.5 bpw EXL3 | A download, not code. The row says it is **not a blocker**. | STILL OPEN |
| fp16 prefix cache | `DynamicCache` lacks the three methods the pool needs | **Would** be a large win (prefix reuse is 136–377× TTFT) — but only for a **non-EXL3** model, and none is run here. Same class of gap as the `_trim_kv_cache` bug fixed 2026-09-06. | OPTIONAL row **3** |
| 2× KV overshoot | ~0.83 GB recoverable at Qwen's 8192 | **Would** free real VRAM — **parked by decision**, not by lack of value. Re-open only on new evidence. | OPTIONAL row **4** |
| Does failing the compression bar degrade TEXT? | Certify the proxy against a real quality measure | Quality certification, deliberately unscheduled. Costs speed if anything. | OPTIONAL row **5** |
| `tensor_align=True` | Production-dead code | Docstrings fixed 2026-09-03; branch deliberately kept. Effectively closed. | OPTIONAL row **8** |

**S1e-bis is UNBLOCKED as of 2026-09-06** — it was blocked only "until P3 lands", and P3 was
already built (see the vacancy note above). Its `[!]` is cleared and its `[bug]` half is answered:
S1e-bis predicted PLD on MoE at **1.26–1.28× in expert bytes** with the think block closed, and
S1d's on-device arm then measured **1.307× in tok/s** on MiniMax-M2.7 — the same mechanism, now in
the unit that matters. What survives of the row is its fourth consequence, which is a standing rule
and not a task: **thinking-mode state is a first-class variable in any speculation measurement
here**, because `enable_thinking=False` is a no-op on both checkpoints.

---

**▶ HISTORICAL CONTEXT, STILL TRACKED: [`HANDOFF-2026-08-30.md`](HANDOFF-2026-08-30.md)** (repo
root, in git). It carries the 2026-08-30 MoE disk-path work with each row's file, symbol,
done-condition and stop rule, plus the preserved probe harness, the seven measurement traps, and
the numbers not to re-derive. **Every P-row it lists is now closed** (P6 last, REFUTED 2026-09-03) —
read it for method and for the traps, not for work.

**`speed-up-vaayu.md`** (repo root) was the whole Rust/perf argument — **deleted 2026-09-03**
(`7d806e7`), not restored — the same commit deleted both on-device gate scripts (see the gates table below). `plans/` and `docs/` are GITIGNORED, so neither
survives a fresh clone; `plans/settled_and_method_notes.md` is the older archive, on this disk only.

**Suite baseline: 2686 passed, 16 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
2026-09-24 after A5.6 (+30: its two new test files). The 2656 / 16 was verified
2026-09-24 after A5.5 (+63: its new test file). The 2593 / 16 was verified
2026-09-23 after A5.4 (+24: its new test file). The 2569 / 16 was verified
2026-09-23 after A5.3 (+19: its new test file; the 2550 / 16 it starts from was re-measured on the
same HEAD that morning, before any A5.3 code, and MATCHED A5.2's figure). The 1979 / 15 was verified
2026-09-11 after row 18 (+3 skipped: its new on-device gate, skipped without `HIDDEN_CKPT`; the
1979 / 12 was re-measured against a stashed HEAD the same day and MATCHED). The 1979 / 12 was verified
2026-09-11 after row 3's `auto` → `bf16` (1972 + 7). The 1972 was verified
2026-09-11 after row 25's APST live-knob fix (1968 + 4). The 1968 was verified
2026-09-11 after row 13's `even-4` window (1950 + 18; the 1950 was re-measured against a stashed
HEAD the same day and MATCHED). The 1950 was verified
2026-09-10 after row 13's timed spec bar (1906 + 44). The 1906 was verified the same day after
row 14's config reachability work (1893 + 13). **The 1893 was re-measured against
a stashed HEAD the same day and MATCHED the figure written here** — the second time in ten that it
was not stale. The 1893 was verified
2026-09-10 after C1's no-geometry refusal (1887 + 6). The 1887 was verified the same day after
row 2's lint gate (1886 + 1), the 1886 re-measured against a stashed HEAD the same day and
MATCHING the figure written here — the first time in nine that it was not stale.
**The suite now costs 124 s -> 164 s (+40 s, +32 %)**: `tests/test_no_unbound_locals.py` shells
out to pylint, and in-venv astroid infers the real `torch`/`transformers` (the same run is 3.3 s
under `uvx`, where they are absent). Findings are IDENTICAL either way. The 1886 was verified
2026-09-09 after the dispatcher-error row-ownership fix (1882 + 4). The 1882 was verified the
same day after row 21's batched-prefill form (1875 + 7). The 1875 was verified the same day
after P5's max-over-chunks form (1870 + 5), the 1870 re-measured against a stashed HEAD
the same day rather than taken from this line. The 1870 was itself verified
2026-09-09 after row 20's rope_scaling fix (1865 + 5). The 1758 written here before that was
stale by 112 — it predates T10's four fixes and row 20's. It was itself verified
2026-09-08 after row 17's sdpa causal-mask fix (+10). **The 1715 written here before that was stale
by 33** — a stashed-HEAD re-measure the same day returned 1748, exactly as the warning at the end of
this paragraph says to expect. The 1715 was verified
2026-09-07 after P9 shipped (the 1700 measured 2026-09-06 + 15 spec-prefix tests; that 1700 was
1685 + 6 batch-union sizing + 9 batch-prefix wiring tests; the 1685
was measured the same day after P4's closure: 1684 + 1 suppressed-round-width test; the 1684 was measured the
same day after the P2 outlier split: 1678 + 6 split tests; the 1678 was measured the same day
after the dense prompt-lookup default: 1647 + 7 bypass + 11 non-streaming + 4 trim + 9 auto; the 1647 was the two dense-telemetry cache-hit fixes' 1639 + 4 + 4; the 1639 was the stray-`logs/run-*` fix's 1638 + 1, the 1638 itself measured against a stashed
HEAD the same day — the 1570 written here before that was stale, as this note warns). **This figure
has gone stale EIGHT times now.** Re-measure against a stashed HEAD before quoting it; do not trust the
number written here.

---

# ▶▶ OPEN — EMPTY as of 2026-09-10

**NO `[bug]` row is live in this section. C1 was FIXED 2026-09-10** — its row was pruned
2026-09-23 (verdict in §0's OPTIONAL table and `git log -p -- todo.md`). Everything else this section carried closed
between 2026-09-04 and 2026-09-06 (L1 T1–T10 and its fix waves, the `env.json`/`logs/`/
`requests.jsonl` rows, S1/S1d/S1e-bis/S1f, G1, V1) and was pruned 2026-09-10 — verdicts in SETTLED
and §0's vacancy notes, full text in `git log -p -- todo.md`. The next work comes from **STILL
OPEN** or from **OPTIONAL** below.

Opened by a `/systematic-debugging` pass on a "the server is too slow" report. **The report was
NOT a bug** — both models ran inside their own recorded benchmark bands (MiniMax 1.794 tok/s
against a 1.798–1.811 band; Qwen 0.937 tok/s against 1.083–1.122, the gap explained by an 809-token
run vs the bench's 64). Verdict, evidence and the two reading traps:
**[`docs/2026-09-03-slow-generation-verdict.md`](docs/2026-09-03-slow-generation-verdict.md)**.
**`docs/` is GITIGNORED** (`.gitignore:28`), so every file it names is on this disk only and does
not survive a fresh clone; this file is the tracked one, and the durable copy of anything moved out
of it is `git log -p -- todo.md`. That is the standing convention — see `CLAUDE.md` §7.

**Applied on the spot 2026-09-03, no row needed:** `DEFAULT_STREAM = True` in
`vaayu/server/config_local.py`. It was `False`, and with a client sending no `max_tokens` that meant
**864 s of silence** for an 809-token answer — most of the reported "slowness" was this, not tok/s.
The server already warns about the combination; nobody was reading the warning.

**Still unfixed on this box, warned at every boot, not a code row:** `rustdesk` holds 0.14 GiB of GPU,
below the 256 MiB refusal threshold but permanent, because budgets are sized once at load. **NVMe APST
is no longer on this list — measured NULL on Qwen and MiniMax at N=1 (2026-09-11, row 25, closed and pruned)**,
and the script that was meant to disable it could not do so without a reboot (fixed).

---

# ▶ STILL OPEN — carried forward from rows that closed

Live items left behind by rows that closed and were pruned. **All of them are OPTIONAL by §0's
test — none can make Vaayu faster.** No §0 P-row is open.

- **[bug] `_detect_moe` decides `is_moe` from a FIELD-NAME LIST and an architecture substring, and
  disagrees with `ExpertOffsetIndex` about the same question** (found by A5.7b Task 6f, 2026-09-25,
  on the first streamed run of DeepSeek-V4). `_detect_moe` (`vaayu_base.py:3582`) tests
  `num_local_experts` / `num_experts` / an arch-string match. DeepSeek-V4 declares
  **`n_routed_experts`** — which `ExpertOffsetIndex._EXPERT_COUNT_FIELDS` ALREADY knows about — so
  `is_moe` came back **False**, `patch_moe_blocks` and `_moe_key_remapper` never ran, the model fell
  through to its own DENSE forward, and the engine loaded all 256 experts of layer 0 at once instead
  of streaming any of them. **`detect_moe_blocks` is healthy** — a probe on the real meta skeleton
  found all 43 module-list blocks; only the `is_moe` gate in front of it is wrong.
  **This is CLAUDE.md §5 in its own words:** "Detection over assumption. Find MoE blocks, router
  names, and expert tensor roles by probing the module tree / checkpoint keys (as `moe_patch.py`
  already does), not by matching an architecture string." `_detect_moe` matches one.
  **FOURTH occurrence of this defect class** — a guard or index asking a question the load path
  answers differently (A1 Task 4, 47-m5, this row's own `mtp.*`/`detect_moe_blocks` layer-0
  collision, now this). Row 47's whole premise is that the NEXT architecture should be cheap; this
  gate silently un-MoEs any model that names its expert count differently, with no warning anywhere.
  **Not fixed generally in A5.7b (its Ruling 12):** A5.7b overrode `_detect_moe` in
  `VaayuDeepseekV4` so the blast radius stays zero. The general fix is to ask the module tree (or
  reuse `ExpertOffsetIndex`'s own field list) instead of a hand-maintained name list — small, but it
  is shared infra under §0 and obliges both on-device gates.
  **Cheap first check when picked up:** does gpt-oss-20b's config name its expert count in a field
  this gate knows? If not, row 47's G1 work has the same hole waiting.

- **[bug] `kv_cache_mode` is INERT for EVERY model whenever prefix caching is OFF** (found by
  A5.7b Task 6d, 2026-09-25, while porting DeepSeek-V4). **Measured, not inferred.**
  `VaayuBaseModel.generate` seeds `kwargs["past_key_values"] = self._create_kv_cache()` **only on
  the prefix-cache path** (`vaayu_base.py:5562`, `:5825`). When `self._prefix_cache is None` it
  calls `super().generate(*args, **kwargs)` bare (`:5550`), and transformers'
  `_prepare_cache_for_generation` then manufactures a stock `DynamicCache` — gated only by a
  hardcoded class-name exclusion list (`reformer`, `minimax`, `xlnet`, `olmohybrid`, `rwkv`,
  `xlstm`, plus a `minimaxm2` override; `transformers/generation/utils.py:1783-1898`) that names
  **neither Llama, nor Qwen3-Next, nor DeepSeek-V4**. `prepare_inputs_for_generation` (`:4407`)
  then forwards that object unchanged — **its comment claims it normalises the cache to `None`;
  the code does not** — so `forward`'s `if use_cache and past_key_values is None:
  past_key_values = self._create_kv_cache()` (`:4942`) never fires.
  **CONFIRMED ON DEVICE:** a real CUDA load of **Llama-3.2-1B-Instruct**, unmodified repo code,
  `enable_prefix_cache=False`, put a plain `transformers.cache_utils.DynamicCache` into `forward`
  where `kv_cache_mode="auto"` should have produced `FP8DynamicCache`. Independently re-read and
  confirmed by a second agent.
  **Scope, stated carefully — do NOT overclaim:** `auto` resolves to bf16, i.e.
  `FP8DynamicCache(quantize=False)` at 2.0 B/elem, and a stock `DynamicCache` is also 2 B/elem, so
  the DEFAULT path's byte cost may be close. The clear casualties are explicit **`fp8`** and
  **`turboquant`** modes, which are silently not applied — Plan H/I's work included. **The measured
  cost is UNKNOWN; nobody should quote a number until one is taken.**
  **Not fixed in A5.7b, deliberately (its Ruling 9):** the fix is in `vaayu_base.py`, shared infra
  under §0 serving Qwen3-Next, MiniMax, dense Llama and gpt-oss, so it needs its own plan and both
  on-device gates. A5.7b fixed only `VaayuDeepseekV4.generate` in the subclass (`a2d7280`), which is
  why DeepSeek-V4 is unaffected by this row.
  **First step when picked up:** measure what it actually costs on MiniMax at `kv_cache_mode="fp8"`
  with prefix caching off — if the answer is ~0, this is a docs-and-comment fix, not a perf row.
  **2026-09-27, S8 T6 (`todo.md`'s S8 row above), sharper consequence found:** on gpt-oss-20b, the
  SAME code path (`generate()` bypassing `self._create_kv_cache()` when `self._prefix_cache is
  None`, `vaayu_base.py:5684-5688`) is not just a silently-wrong dtype — with
  `enable_prefix_cache=False` and a prompt/generation crossing gpt-oss's 128-token sliding window,
  it **crashes outright**: `RuntimeError: The size of tensor a (128) must match the size of tensor
  b (199) at non-singleton dimension 3` (`eager_attention_forward`,
  `transformers/models/gpt_oss/modeling_gpt_oss.py:259`) — HF's own `generate()` auto-selects a
  cache that truncates gpt-oss's sliding layers to 128 tokens, which Vaayu's mask-building code
  (written only for Vaayu's own always-full-length `_create_kv_cache()`) cannot handle.
  **Reproduces on `main`** (678bdb9, pre-S8) as well as on `s8-routed-fused3d`, so it predates S8
  and is not caused by routed delivery — `plans/probes/s8_out/repro-main-prefixoff2.log`.
  **Server-reachable only if a user sets `ENABLE_PREFIX_CACHE=False`** (the default is `True`,
  `server/config.py:100`, which takes the other `generate()` branch and always passes
  `_create_kv_cache()`). **Probe workaround** (not an engine fix): pass
  `past_key_values=model._create_kv_cache()` explicitly into `stream_generate(...)`/`generate(...)`
  per call — `plans/probes/s8_ab.py`. The underlying fix is still the one described above: make
  `generate()` pass its own cache (or refuse the config) even when `pc is None`.

- **[bug] `prefix_hit` in `requests.jsonl` cannot detect a reuse COLLAPSE** (found by row 31,
  2026-09-12). It is `True` for a 1-token match, so an arm whose turn-2 match had fallen from
  **7422/7517 tokens to 44** still logged `prefix_hit: true` on every request — row 31's fan-out
  arms read as "4/4 HITs" in the exact configurations where reuse was provably gone, and only the
  engine's `[Vaayu-Prefix] HIT … match_len=X/Y` stdout line told the truth. **Anyone tuning the
  prefix pool from `requests.jsonl` today is reading a metric that cannot fail.** The fix is small
  and known: the engine already computes `match_len` at the point it logs it — carry it (and the
  prompt length) into the request row so the ratio is visible. Not fixed on the spot: it is a
  reporting change, and row 31 did not need it once the stdout line was read. **Reachable at N=1,
  affects every user who looks at the logs.**
- **[ ] `documentation/TUNING.md` §1's prefix-cache headline describes a different case than it
  reads as** (row 31, 2026-09-12). `TTFT 91 s → 9.6 s at 2k-token sessions` and `332 s → 9.8 s at
  8k` are the OVER-BUDGET recompute-cascade case (Plan I, K=3 × 7.7k). Ordinary single-session
  reuse on a streamed MoE is worth far less, because the reused prefill still routes ~every expert
  per layer: measured on Qwen3-235B, **−6.6 % of TTFT at 2.7k tokens and −53 % at 7.4k**. A
  qualifying paragraph with those two numbers was added beneath it; the original figures were left
  standing because they are correct for the case they describe. Anyone re-measuring prefix reuse
  should state which of the two regimes they are in.
- *(→ moved to `todo-backlog.md` 2026-09-26: No resident-size checkpoint on disk. Re-open when: a ~14B-class resident checkpoint is wanted; S8 (gpt-oss-20b, which fits the GPU whole) may answer it first.)*
- *(→ moved to `todo-backlog.md` 2026-09-26: `_finish` cannot report WHICH stop string fired. Re-open when: a client needs it.)*
- *(→ moved to `todo-backlog.md` 2026-09-26: `vaayu/expert_sim.py`'s anchor gap. Re-open when: before anyone quotes its small-budget absolutes.)*
- *(→ moved to `todo-backlog.md` 2026-09-26: `test_slab_arena` order-dependent full-suite failure. Re-open when: the next full-suite failure of this test — record the run order.)*
- *(→ moved to `todo-backlog.md` 2026-09-26: Row 44's F3 remnant (pinned-H2D half of prefill). Re-open when: Qwen3-Next prefill turns disk- or H2D-bound again.)*
- **[bug] Vaayu assumes the CURRENT CUDA device in ~35 places** (no-device `torch.cuda.synchronize()`
  / `mem_get_info()` / `torch.cuda.Stream()`, and Triton launches) — a model on `cuda:N≠current` is
  unsupported; S3's installer now refuses and warns (found by S3's final review 2026-09-25).
  Single-GPU box today; unblock = pass the device or `torch.cuda.set_device` at load.

**Each carried item now has its own cold-pickup brief** (gitignored, written 2026-09-10):
[`…-carried-resident-checkpoint-BRIEF.md`](docs/2026-09-10-carried-resident-checkpoint-BRIEF.md) ·
[`…-carried-finish-stop-string-BRIEF.md`](docs/2026-09-10-carried-finish-stop-string-BRIEF.md) ·
[`…-carried-expert-sim-anchor-gap-BRIEF.md`](docs/2026-09-10-carried-expert-sim-anchor-gap-BRIEF.md) ·
[`…-carried-p10-batching-minors-BRIEF.md`](docs/2026-09-10-carried-p10-batching-minors-BRIEF.md)

## Carried out of row 15 (P10 server batching, closed 2026-09-09, pruned 2026-09-10)

**Everything below is on the batching path, which is `MAX_BATCH_SIZE = 1`-gated — worth 0 % to
users today (CLAUDE.md §8). Recorded so it is not lost, not because it is scheduled.**

- **[bug] No test for the sentinel-only cancel path.** `_sse`'s sentinel-with-no-owner
  branch (`owed=(None, None, "cancelled")`) has code but no coverage: it is reached when a
  job's `CancelToken` is set while queued WITHOUT its ASGI task being cancelled — e.g.
  `cancel_all()` at SIGTERM. `test_a_disconnect_while_queued_still_writes_a_row`
  exercises the neighbouring `except BaseException` handler instead.

Known-open minors (none blocking):

* **Gate 1 and gate 2 are not byte-identical arithmetic, deliberately.** After the B1 fix,
  gate 1 (`_admit` → `plan_batch(..., giveback=True)`) prices `free − safety + giveback`
  uncapped, while gate 2 (`admission_headroom`) imposes a reserve-based ceiling
  (`min(free + reusable_pool, reserve + giveback)`) that gate 1 does not. The fix only
  ever makes gate 1 MORE generous, never stricter, so gate 2 stays the real backstop and
  the B1 collapse cannot recur — but the two are not the same expression. **Any batching
  run must report `gate1_refused` and `gate2_refused` separately** (both counters exist for
  this reason); a run where gate 1 refuses and gate 2 would not is the shape to watch.
* `_stopping` is never reset, so `start()` after `stop()` yields a scheduler that refuses
  everything — consistent with the no-resurrection rule, but undocumented.
* `test_the_queue_refuses_past_its_depth` no longer pins WHICH refusal it got; a `match=`
  on the message would make that explicit.
* `_BatchStream` feed-after-retired is exercised only incidentally, never asserted.
* `build_generate_kwargs` materialises and discards an N-token prompt tensor per job.
* `plan_batch` SUCCEEDING and deferring a member is covered by zero tests (the test stub
  has no `.config`, so every test takes the exception path).
* **Found by the final whole-branch review (2026-09-09), B4:** `app.py`'s lifespan now
  awaits `app.state.scheduler.stop()` BEFORE `_teardown`'s `cancel_all()`, so a
  PROGRAMMATIC shutdown — an embedded caller, or a `with TestClient(...)` block exiting
  mid-generation — waits for the current run to finish NATURALLY (up to its full token
  cap) where the old pre-P10 code reached `cancel_all()` immediately with no wait in
  between. Signal-driven shutdown (real SIGINT/SIGTERM) is unaffected: the signal handler
  calls `cancel_all()` first, independent of this coroutine. Already flagged in a code
  comment at `vaayu/server/app.py`'s lifespan `finally` block (the "NOTE on ordering"
  above `await app.state.scheduler.stop()`). Not fixed — deciding whether `stop()` itself
  should cancel first is Task 7's contract to change, not a whole-branch-review fix.
* `engine.slot()` is now dead production code — nothing but tests calls it since T8 moved
  admission onto `BatchScheduler.submit()`. Left in place per CLAUDE.md §3 (don't remove
  pre-existing code not asked about).

---

# ▶ SETTLED — DO NOT RE-OPEN

**Removing the pageable host-to-device copies from the decode token cannot pay on this box —
measured 2026-09-25 (S1; the change was built, A/B'd and reverted).** Two sites, both per-layer:
36 × 16 KB from `torch.zeros(...).to(value)` in transformers' `torch_recurrent_gated_delta_rule`,
and 48 × 160 B from `_pair.to(device, non_blocking=True)` at `moe_patch.py:2310`. **The real count
is 84/token, not the spike's 41.7** (that figure divided prefill's copies into decode, on the NOEXP
arm). **Do not rebuild either:**
- **Pinning `_pair` is SLOWER than leaving it pageable** — 3.14 µs status quo, 3.28 pre-pinned,
  5.80 via `.pin_memory()`, idle stream, 5,000 calls. `non_blocking=True` already skips the stream
  sync; 160 bytes never repays a pin.
- **Wrapping the delta rules in `with torch.device(...)` costs −7.06 % on `serial` and −1.39 % on the
  real forward.** `torch.device` as a context manager pushes a `DeviceContext` torch-function mode,
  so EVERY torch op inside pays **+0.906 µs** of python dispatch; at ~80–110 ops × 36 layers that is
  ~2.7 ms against a 187–630 µs prize. **This trap applies to any future use of `torch.device` or
  `torch.set_default_device` around a hot loop — it is not specific to this mixer.**
- The only untried delivery is copying upstream's two torch fallbacks into Vaayu. Its ceiling is
  **0.17–0.6 % of the token, below this box's 0.36 % resolution floor.** Not worth the maintenance.

Still open from S1, as leads and NOT refuted: **S1-L1** (~27,084 pageable HtoD in one prefill
forward — the larger fish), **S1-L2** (96.5 pageable DtoH per decode forward), **S1-L3** (`fla` is
not installed; installing it subsumes S1-a and much of S3).


**Read this before proposing any perf work.** Each line is something that was BUILT or MEASURED and
then refuted. A refutation closes its row; it is not re-scoped. Full text in
`docs/2026-09-03-todo-archive-preprune.md` and `git log -p -- todo.md`.

**Sizing the prefix reservation to "what the pool actually holds" cannot pay on this box — measured
2026-09-23 (row 46, two on-device spikes, no engine line changed).**
- **The premise — "the reservation over-reserves 8× at 8k, 32× at 2k" — is REFUTED, and it came from
  reading ONE model's numbers onto another.** Real stored sessions were measured by prefilling and
  reading the pool entry's own `nbytes` (the exact figure the budget is spent against), two lengths
  per model so `bytes(n) = a·n + b` is determined: **MiniMax 248.0 KiB/token** (fixed term 0.25 MB),
  **Qwen3-Next 24.0 KiB/token + 40.13 MB** fixed (the 36 linear layers' recurrent state).
  **The engine's accounting is EXACT on both** — `derive_expert_ceiling` is fed
  `kv_bearing_layer_count(self)` (`vaayu_base.py:2691`), not `config.num_hidden_layers`, so it
  already charges 12-of-48 layers on the hybrid. **There was no accounting bug on this path**, and
  47-m4's "counts every layer" concern does NOT reach it. The handoff's "a stored 8k session holds
  0.20 GB" was Qwen3-Next's 2k figure (0.201 GB) applied to MiniMax.
- **What the reservation actually is: one full-context session, and `max_seq_len` is the VRAM plan's
  own cap (`vaayu_base.py:2446`) — not an arbitrary ceiling.** MiniMax `max_seq_len` **8192**,
  reserved 2.080 GB/session, a real 8000-token session **2.0319 GB = 1.02×**; three of them leave
  **0.14 GB** to recover. Qwen3-Next `max_seq_len` **65536**, reserved 1.610 GB, a real 8000-token
  session 0.2367 GB = 6.80×; **4.12 GB** genuinely recoverable there.
- **THE PRIZE AND THE RECOVERABLE BYTES ARE ANTI-CORRELATED, and that is structural.** Returning the
  whole reservation measured **+13.36 % decode on MiniMax** (ABBA `off,base,base,off`, 4 processes ×
  2 in-process replicates, `PREFIX_SESSIONS=0` vs `=3`: **2.3467 vs 2.0701 tok/s**, within-arm spread
  1.4 %, greedy ids bit-identical across all 8 arms; mechanism: RAM-cache hit **20.0 → 35.7 %**,
  expert disk reads **−19.6 %**, bytes/token **2.19 → 1.76 GB**, stall **43.1 → 36.2 %** of decode)
  — and **−2.56 % on Qwen3-Next** (row 42-B). MiniMax's context is capped at 8192 *because that is
  all the VRAM allows*, so a full agent session IS 8192 tokens and the bytes are not idle.
- **All three candidate mechanisms die on the same number** (reserve to a measured typical length;
  grow the reservation dynamically; shrink the pool instead of the arena). Each is a way of guessing
  a session length below `max_seq_len`; on MiniMax the guess must be ~4× short to pay, the arena is
  **pinned at load and cannot shrink**, and a wrong guess buys row 31's collapse at full price
  (−8.9 %, turn-2 match 7422/7517 → 44/7517, TTFT 25.0 → 51.3 s).
- **Do NOT re-open as "measure the typical length better", "grow the arena later", or "just cut
  `PREFIX_SESSIONS`".** The first two are the refuted mechanisms; the third is row 31, already swept
  and closed with "defaults STAY". **§8: no new knob — `PREFIX_SESSIONS` and `MAX_SEQ_LEN` already
  span the trade.**
- **WHAT SURVIVES, and it is worth carrying:** the A/B prices MiniMax's RAM cache at roughly
  **+2 % decode per GB returned to the expert arena** (6.3 GB → 13.36 %), on a model whose hit rate
  is only 20 %. That is a measured yardstick for any FUTURE MiniMax RAM lever — Qwen3-Next's
  near-dead 0.42 %-disk regime does NOT generalise to it. Two probe traps also survive:
  `plans/probes/row26_decode.py:36` defaults `MODEL=qwen`, a **deleted** checkpoint; and `BALLAST_GB`
  can trim the arena below both arms' ceilings, silently making a ceiling A/B inert — run the
  1-token binding check (`NEW_TOK=1 ARMS=W`) first.
  [`docs/2026-09-23-row46-step1-minimax-AB-RESULTS.md`](docs/2026-09-23-row46-step1-minimax-AB-RESULTS.md),
  [`docs/2026-09-23-row46-step2-session-bytes-RESULTS.md`](docs/2026-09-23-row46-step2-session-bytes-RESULTS.md)
  (**gitignored — row 46 itself is the durable copy**).

**Re-evaluating a latched speculation decision cannot pay at its own re-entry price — measured
2026-09-10 (row 24, offline replay of row 13's six alternating MiniMax-M2.7 traces).**
- **"The decision latches while acceptance and cost move inside one call, so re-evaluate" is
  REFUTED ON PRICE.** Both halves of the premise are real and neither is worth the fee.
  Perfect-foresight, ZERO-cost re-evaluation is worth **≤ +6.28 %** over a one-shot gate that
  decides correctly (six arms: +0.00 / +0.00 / +0.49 / +1.15 / +3.53 / +6.28 %), and it breaks even
  at **0.5–5.9 token-equivalents** of re-entry against a real price of **~29.3**
  (`TRIAL_DRAFTED_TOKENS × breakeven`). **5×–60× too expensive**, and a re-entry made five times
  cheaper still loses on 3 of the 4 arms with anything to win. The cadence sweep's own optimum is
  its SPARSEST setting — the gradient points back at the latch. **Do not re-open this as
  "re-evaluate more cheaply", and never propose a re-calibrating gate without pricing it against
  that ~29.3 first.** What survives is a DIFFERENT defect on row 13: the decision window's WIDTH.
  One-shot verdicts are right at only 62–100 % of possible decision points and pooling **m ≥ 3**
  paired blocks was right on 6/6 — while deciding LATER would have been wrong on two arms instead
  of one, because wrong verdicts cluster at BOTH ends of a call.
  [`docs/2026-09-10-row24-relatch-replay-RESULTS.md`](docs/2026-09-10-row24-relatch-replay-RESULTS.md).

**Tightening the VRAM transient reserve cannot pay on this box — measured 2026-09-09 (§0 P5).**
- **"The intercept over-charges, so the expert cache is smaller than it needs to be" is REFUTED as
  a PERF premise.** The over-charge was real (up to +42 % at a partial trailing chunk) and it was
  FIXED — but the reserve did not move on any of the four geometries, on either kernel. Two
  structural facts, and they outlive the fix: **(1) on flash the reserve is
  `max(predict(chosen), reserve_floor)` and `reserve_floor` is the 3.5 GiB base floor, which every
  EXL3 geometry's predicted peak sits BELOW** (MiniMax 2.75 GiB at 8192), so the peak form is
  invisible there; **(2) `chosen` lands on a KV capacity doubling, always a multiple of
  `prefill_chunk_size`**, i.e. exactly where the trailing chunk is FULL. **Any future idea whose
  payoff is "predict a smaller peak → bigger expert cache" is worth ZERO until one of those two
  changes — price it against the FLOOR, not against the prediction.** What a smaller predicted peak
  does buy is ADMISSION headroom (`_assert_context_fits` evaluates at the requested length, whose
  trailing chunk is usually partial): MiniMax −13.3 % at 7700 tokens, Qwen zero.
  `docs/2026-09-09-P5-which-chunk-sets-the-peak-RESULTS.md`.
- **Do not conclude "no sawtooth" from Llama-3.2-1B again.** It is the ONE geometry whose logits
  term outgrows its per-chunk KV, so its peak sits on an early full chunk and looks flat; and its
  `head_bytes` is EXACTLY 2 × its full-chunk fp32 logits, so it cannot arbitrate the intercept's
  composition either. Two 2026-08-15 conclusions rest on that geometry and both were wrong-shaped.

**The VRAM expert cache's prefix sizing — the observed-union idea, killed 2026-09-06.**
- **Sizing the batch VRAM prefix on the OBSERVED routing union instead of P6's zero-overlap bound —
  BUILT, MEASURED, REFUTED, and REVERTED 2026-09-06.** MiniMax-M2.7 N=3: **−10.85 / −11.54 %**
  (2.913/2.891 against P6's 3.275/3.261 — exactly the pre-P6 number), hit rate 48.4 % → 2.38 %,
  evictions 17 533 → 61 278, greedy ids identical in every arm. Suite returned to 1700 passed / 12
  skipped after the revert. `docs/2026-09-06-P7-observed-union-prefix-RESULTS.md`, harness
  `plans/probes/p7_ab.sh` + `p7_sweep.sh` + `p7_union_analysis.py` (**gitignored — this row is the
  durable copy**). **NOTE the name collision: this is §0's `P7` SLOT. It is NOT the regime-latch
  item below, whose write-up is also called `...-P7-...` for unrelated reasons.**
  **What is settled, and it is a fact about the sizing contract, not about one model:**
  `compute_vram_layer_range` sets capacity/layer = the load it is charged, so charging the TRUE
  per-layer union leaves capacity ≈ union and an entry cannot survive until its own reuse.
  **P6's bound over-charges MiniMax's steady-state union by 1.17× and THAT IS THE REUSE HEADROOM,
  not slack.** Measured: the per-layer union RISES 10.47 → 20.55 across 64 decode steps (whole-decode
  mean 16.91), the union over k=2 consecutive steps is 26.38 — already above the bound of 24 — and
  P6's capacity of 24.34 sits between the two, which is precisely why it hits 48.4 %. A forced-charge
  sweep puts the optimum at the bound: 24 → 3.257 tok/s, 21 → 3.268 (+0.34 %, inside noise), 18 →
  3.141 (−3.54 %), ALL 62 layers → 2.902 (−10.9 %).
  **Do not re-open with "a longer calibration window", "measure the union better", or "charge the
  whole-decode mean" — a perfectly-measured union is the WRONG QUANTITY.** Also settled by the same
  run, and worth keeping: `_expert_history` is written PRE-filter by all three MoE forwards
  (`moe_patch.py:1020`, `:1335`, `:1713`; `_to_load` at `:1109` is the separate filtered variable),
  and it is written during PREFILL too, where a chunk routes **224.23 experts/layer** against a
  decode union of 9–21 — anything sampling it for decode statistics must sample inside the decode
  loop.
  **NOT settled, and deliberately left open as a different question:** nothing derives the ~1.2×
  headroom the sizing silently depends on, so a geometry whose bound/union ratio differs has no
  reason to land in the right place.

**Speculation on MoE — the regime-latch idea, killed 2026-09-06.**
- **Re-entering speculation on a "code regime" — MEASURED and REFUTED 2026-09-06, nothing built.**
  Full write-up `docs/2026-09-06-P7-regime-latch-RESULTS.md`, probe
  `docs/probes/p7_regime_latch.py`, analysis `docs/probes/p7_analyse.py`, trace
  `docs/probes/p7-qwen-regime.json` (**all gitignored — this row is the durable copy**).
  **The claim:** `AdaptiveSpecGate` latches once and holds for the process, while S1e-bis had Qwen
  at 0.286 accepted/drafted in prose and 0.927 inside a code block — so a gate calibrating during a
  prose preamble would forfeit the code regime, which is the agentic workload (CLAUDE.md §0).
  **The bar, fixed before the run:** in-code PLD saves 20.6 % of the tokens it covers and one TRIAL
  re-entry costs `TRIAL_DRAFTED_TOKENS × breakeven` = 25.3 token-equivalents on Qwen (P4's closure),
  so a switch pays only on a code stretch past **~123 tokens**.
  **Method:** S1e's lossless offline replay of `PromptLookupDecoder.draft` over one real greedy run
  (Qwen3-235B, `MAX_NEW=900`, `draft_tokens=1`, own systemd scope, 900 tokens in 734.0 s), keeping
  every round's `(position, drafted, accepted)` instead of summing them away. Nothing built, no gate
  relaxed.
  **Three findings, each independently fatal:**
  1. **The transition does not happen. 900 tokens, ZERO fenced code blocks** — Qwen wrote prose the
     whole way and hit the cap still deliberating. That is now **four Qwen arms (192/192/160/900)
     with no spontaneous code**. The only arm that reached code, `edit_nothink_forced`, had the
     code's opening lines **pre-seeded into the assistant turn**; in `edit_forced` Qwen closed the
     injected fence and went back to prose. **S1e-bis's 0.927 needs the model handed the start of
     the code — it is a prompt property, not a phase every answer passes through.**
  2. **The latch was CORRECT.** Whole-answer acceptance **0.4170** against a 0.5277 break-even;
     speculating throughout would have cost `679 × 1.5277` = 1037 token-equivalents for 900 tokens
     = **1.153/token, a 15.3 % loss**. That reproduces the on-device forced arm's **0.836× (16 %
     slower)** from a separate run — a cross-check that the replay accounting is right. The
     simulated gate latched `off_acceptance` at token 83.
  3. **The excursions are decoys, and they are 3.2× too small.** Acceptance swings 0.12–0.92, but
     from **prose SELF-REPETITION**, not code. Best above-break-even stretch: 99 tokens at a/d
     0.661 → **7.9 token-equivalents of gain against the 25.3 cost**. Nothing clears the bar at any
     window (15/25/40 rounds). Five such stretches in 900 prose tokens.
  **The ceiling, if the untested branch had gone the other way** — recorded so nobody re-opens this
  hoping for 1.26×: with code starting at token 1500 and running ~300 tokens, the ideal is
  `1500 + 237.9 + 25.3` = 1763.2 against today's 1800 — **2.0 %, not 1.26×**, because the trick
  covers only the answer's last ~17 % and you pay to switch back on. And 2.0 % assumes a detector
  firing exactly once: **two false triggers (50.6) exceed the entire 36.8 gain**, and finding 3 says
  the decoys are there to trigger on. The only false-positive-free detector is a literal ``` fence
  match — a markdown assumption baked into the engine, helping only models that fence (CLAUDE.md §5).
  **The `MAX_NEW=1800` follow-up was considered and deliberately NOT run:** it could only show the
  code phase exists, never that the idea works, and finding 3 already prices the detector at a loss.
  **Re-open only with BOTH** a code stretch past ~123 tokens AND a detector shown not to fire on the
  five decoys recorded in the RESULTS file. **The permanent latch is the right design.**

**Rust rewrites — all dead ends.**
- **R2 `io_uring_read_many`** — REFUTED 2026-08-30. Reads are bimodal: exactly ONE bulk group per
  expert (11.80 MB Qwen / 8.85 MB MiniMax) plus 1–2 tiny scale/bias groups carrying 0.3–0.4 % of
  bytes. In BYTES there is one read per miss, already dispatched one-future-each across the workers.
  Best case was ~1.02×.
- **R3 expert-cache miss path into Rust** — REFUTED 2026-08-30 by M1's GIL probe. A 40× switch-interval
  swing moves decode **0.11 % / 0.72 %**. There is no GIL contention to recover.
- **R4 = P6 prefill repack memcpys into Rust** — REFUTED 2026-09-03, nothing built. Its target
  (`load_expert_into_slab(tensor_align=True)`) has had **zero production callers since 2026-07-13**;
  the row was written six weeks later off a stale docstring. `gpu_repack_aligned` is CUDA D2D, not
  reachable from Rust, and Qwen-only. The real residual is 4.7 % of a PCIe-bound prefill.
- **R1 hot-path `eprintln!`** — perf REFUTED (inside the noise floor) but the change was KEPT: log
  volume 18 MB → 364 KB per 71 tokens.
- **Also dead, from M1:** the garbage collector is 0.26 % / 0.41 % of decode. Rewriting the decode
  loop, replacing PyTorch, and raising the ring `queue_depth` were all covered in
  `speed-up-vaayu.md` §4 (file deleted 2026-09-03, `7d806e7`; the argument
  is not reproduced here). MoE decode is 72–74 % disk-bound and the rest is CUDA; streamed dense
  already reads at ~5.7 GB/s, i.e. at the wall.

**Expert caching and prefetch.**
- **P1 / OPTIONAL row 7 — hoisting the VRAM-resident filter above the fused-EXL3 dispatch** —
  REFUTED 2026-09-06, nothing built. The premise was that a VRAM-resident expert can miss the RAM
  cache and issue an 11.83 MB read for weights already on the GPU. It **cannot**: measured **0** on
  **every one of 5922 decode layer-visits** on Qwen, at N=1 and again at N=3, while the intersection
  itself is large (3.61 of 8 routed experts on the cacheable layers at N=1). VRAM ⊆ RAM, and the
  redundant lookup is *what keeps it that way* — it refreshes the RAM LRU for the hottest experts,
  so hoisting it would make them the first evictions and CREATE the read it was meant to remove.
  Residual saving if hoisted anyway: 155 lookups/token × **0.665 µs** = 103 µs against 890 ms/token,
  **0.012 %**. `docs/2026-09-06-P1-vram-resident-dispatch-RESULTS.md`, probe
  `plans/probes/p1_trace.py`.
- **Expert pinning (Plan F top_n sweep)** — REFUTED 2026-08-02. No viable pin N on either model:
  share is linear in N and 2–4× below the cost line.
- **Plan B Phase 2 (Tasks 8–9) live expert re-pin** — REFUTED DIRECTLY 2026-09-06, nothing built.
  A *realizable* adaptive pin (top-P/layer from the previous window — literally what Phase 2
  proposes) is worth **−0.082 to +0.011 hit-rate points** at every non-degenerate budget on both
  models, while the clairvoyant bound is worth **+0.004 to +0.330**. The value is all in a
  predictor recency does not reach. Do NOT re-open on "a better predictor" — the oracle bound IS
  that. **Trap:** at Qwen's ~8.5 GB budget `max_per_layer == top_k`, the LRU thrashes to a 0.0000
  hit rate, and any pin looks like a +0.27 win on that artifact alone. See row 5.
- **Fold `Π` into the query (Plan H perf follow-up)** — CLOSED 2026-09-06, target gone. Plan I moved
  compression off the decode path, removing the 5–8 % MiniMax regression it existed to fix.
- **P3 tiny-read store** — BUILT and REFUTED 2026-08-30. Holding scale/bias groups resident removes
  100 % of the tiny reads and returns byte-identical output, and costs **−2.7 % at N=1, −11.3 % /
  −14.5 % at N=3** — every store-off arm beat every store-on arm. **Occupancy is not contention,
  and read COUNT is a resource.** Do not rebuild the store.
- **`VAAYU_CACHE_BIAS` / `cache_aware_topk`** — dead code; never executes under transformers 5.5.4
  (0 calls, both models).
- **`VAAYU_LOOKAHEAD=1` — still hard-blocked, and the block is CORRECT.** Its two old grounds are
  both false now (P2 wired the module-list forward; P1/P2 measured fan-out on both models), so the
  guard stands on a NEW reason: post-P4 it is **Qwen N=1 +11.7 %** against **MiniMax N=3 −3.7 %**.
  Defaulting it ON ships a win to one actively-tested model and a loss to the other at the fan-out
  this engine exists for — the asymmetry `CLAUDE.md` §5 forbids. P4 exhausted admission as a lever
  (a 12-slot/layer reservation removes 99.96 % of prefetch-caused evictions and makes BOTH models
  worse); what is left is prediction QUALITY (0.766 vs 0.902). **Do not "turn it on to go faster".**
- **Speculative decoding on MoE** — refused at load on both targets by the Fix-#9 gate. See the
  live row `S1` in OPEN: the gate's 53.4 GB figure is a zero-overlap upper bound and is worth
  measuring, but the most likely outcome is that it is right for a better reason.

**Measurement rules that cost real time to learn.**
- **P5 control drift — does NOT reproduce (2026-09-02); the ~20 % unfalsifiability rule is LIFTED.**
  Seven arms, both models, 0.06–0.18 %, greedy ids byte-identical in every pair. **New rule:
  `BALLAST_GB` is an arm variable, not hygiene** — 25 → 27 GB is +9.4 % tok/s. Never compare across
  targets.
- **P8 — the ~5.12 GB/s ceiling was the ring pool, and it is FIXED (2026-09-03).** The pool was
  sized to the CALLER pool (8) instead of to the drive; 8 concurrent O_DIRECT readers is past this
  NVMe's knee. Pool → 3: decode **+4.8 % Qwen / +9.7 % MiniMax**, per-read latency −14.1 % / −18.7 %,
  no overlap; dense unmoved. See `CLAUDE.md` §5 for the corrected bandwidth ladder.
- **C1 — `~5.5 GB/s` is NOT a hardware constant.** It was a measured-path number promoted to one
  and then used as a denominator. Never quote a single-stream rate as the drive's limit; always say
  which concurrency a bandwidth figure was taken at, and never assume more concurrency helps.
- **A quality proxy validated on one forward does not transfer to the other (2026-09-06).** Plan H/I
  set the `prefix_compress` 9-bit floor with a K-row query proxy. Against REAL decode queries that
  proxy is **2.5–13.7× optimistic on Qwen** (which then FAILS the 0.05 bar at 0.0649) and only ~2× on
  MiniMax (which passes). 9 bits is also the quantizer's CEILING — `make_codebook` raises above
  `b_mse=8` — so there is no extra bit to spend. See row 5.
- **The token-summary buckets are a partition by construction** and `expert_disk` is a STALL, not
  read time. Both traps are written up in
  [`docs/2026-09-03-slow-generation-verdict.md`](docs/2026-09-03-slow-generation-verdict.md).
- **None of the 16 B-row bugs was caught by `tests/`** — the suite was green while every one of them
  was live. That is why each fix shipped with a test verified to FAIL against the code it replaced.

**KV compression — the mixed-precision outlier split, SHIPPED 2026-09-06 (§0 P2). Two things about
it must not be "restored", both locked by test.**
- **Channels are picked ONCE PER LAYER, shared by every head.** Per-(layer, head) selection was
  measured and **FAILS MiniMax at m=8 (0.0725 — worse than not splitting at all, 0.0426)** while
  looking better than per-layer on Qwen at m=16. A granularity validated on one forward does not
  transfer to the other (CLAUDE.md §5, again).
- **`m` is not a knob, and m=4 is worse than useless.** `pack_codes` needs `(head_dim − m) % 8 == 0`,
  so at head_dim 128 m=2 and m=4 cost **2.031 B/elem — above bf16's 2.000**; `_quantize_split` snaps
  `m` up to the nearest bit-packable width. m=16 costs 1.266 and buys nothing (Qwen 0.0396, *worse*
  than m=8's 0.0274).
- What shipped: 8 highest-variance channels kept in the cache dtype, the remaining 120 packed at
  9 bits, on by default whenever `prefix_compress` is set. **Qwen3-235B 0.0683 FAIL → 0.0281 PASS**,
  **MiniMax-M2.7 0.0428 → 0.0263**, at **1.211 B/elem, 0.605× bf16**. Pool capacity 5 → **9**
  sessions, so the advisor's claim is **~1.65×, not 2×**. `PREFIX_COMPRESS` still defaults to
  `None`: this changed what `9` MEANS, not whether it is on.
  `docs/2026-09-06-P2-outlier-split-RESULTS.md`.
- **The stated mechanism was WRONG and is corrected here:** Plan H §7 and Plan I §8.5 justify the
  split by "both models' outliers are early layers". Measured, the layers that FAIL are the
  **diffuse-attention** ones (Qwen L0 top-1 mass 0.058), and the two highest-kurtosis layers
  (Qwen L47/L93) do **not** fail. **Still NOT established:** that failing the bar degrades generated
  text (OPTIONAL row 5), and anything at all about V — the gate scores attention over K, and the
  split is applied to V by the same code with no V number behind it.

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q`. The current baseline is the one stated once
  at the top of this file — do not duplicate it here; it went stale four times that way.
- The server needs `fastapi` and `uvicorn` (in `requirements.txt` from 2026-08-16). `httpx` and
  `pydantic` were already present; `fastapi.testclient` needs `httpx`.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  NVMe bandwidth), each commented as such.
- **NVMe is concurrency-dependent, and MORE CONCURRENCY IS NOT MONOTONICALLY BETTER
  (corrected 2026-09-03, row P8; supersedes C1's ladder).** Samsung SSD 990 PRO 1TB, **spec
  7.45 GB/s**. Measured **directly through the expert path** (`plans/probes/p8_samepath.py`, real
  expert byte ranges, O_DIRECT, registered fds, pinned buffers, wall-clock — no delivery model):
  **1 reader → 6.0–6.4 GB/s, 2 → ~7.1, and ~7.1 IS THE PLATEAU.** It holds to 16 concurrent callers
  as long as no more than ~4 reads reach the device at once; past that it **costs** throughput,
  falling to **~5.15 GB/s at 8**. Reproduced on both models, at every ring queue depth
  (qd=1…32), and for scattered and file-adjacent reads alike.
  **C1's `1 → 3.3, 2 → 5.3, 3 → 6.5, 4 → 6.9, still rising` is SUPERSEDED**: it was inferred from
  overlapping in-path reads under a uniform-delivery assumption (its own k≥5 buckets exceed spec),
  and it is wrong in both directions — it understates k=1 by ~2× and implies a rise that reverses.
  C1's headline survives: 5.5 GB/s is not the drive. The **~5.5 GB/s** this file used before
  2026-08-30 was what the expert path achieved at 2–3 reads in flight.
  **Every "% of the drive" written before 2026-08-30 is against the wrong denominator** — decode's
  3.70–3.83 GB/s is 50–51 % of the drive, not the 71–73 % on record. Archived `plans/` documents
  keep their original wording; this file and `CLAUDE.md` carry the correction.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **CPU / RAM / PCIe, measured 2026-09-11 (row 26, `plans/probes/row26_physics.py`) — hardware constants:** i5-14600K,
  CPUs 0–11 = 6 P-cores × 2 HT (5.3 GHz), CPUs 12–19 = 8 E-cores (4.0 GHz); avx2 + avx_vnni, no avx512. **Host RAM read
  65.6 GB/s** (6 P-threads; 63.3 @14, 60.5 @20). **Pinned H2D 19.7 GB/s** (256 MB; one 11.8 MB expert 634 µs) — NOT
  Gen5 x8's ~32 GB/s theory; D2H 16.9, pageable H2D 15.3. Hidden-state GPU→CPU→GPU round trip ~18 µs. CPU batch-1
  expert GEMV is memory-bound; 6 P-core threads beat 14/20. Idle states POLL / C1_ACPI (1 µs) / **C2_ACPI (127 µs)**,
  governor `powersave`, EPP `balance_performance`; `/dev/cpu_dma_latency` is root-only. **An idle package costs Qwen
  decode ~21 % (row 27)** — a probe that spins (e.g. `profiling_mode`'s `synchronize()`) runs FASTER than the real thing.
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
- **`BALLAST_GB` is an ARM VARIABLE — state it, hold it fixed across a comparison, never compare
  across targets (P5, 2026-09-02).** It sets the expert slab arena, and the arena sets throughput:
  25 → 27 GB grew the arena 1469 → 1622 slabs (+10.4 %), cut bytes/token 8.3 %, and raised N=3
  aggregate **+9.4 %** — larger than most effects these probes exist to detect. The ballast pins
  avail BEFORE the load; the arena is sized DURING it, so identical arms still land on arenas a few
  slabs apart (harmless: 0.06 % on throughput).
- **N=3 A/Bs are trustworthy again (P5, 2026-09-02).** Replicates hold to **0.06–0.18 %** on both
  MoE models. The old "treat any N=3 A/B inside ~20 % as unfalsifiable" rule is LIFTED. Still check
  greedy ids match across arms before believing any throughput delta (trap #6).
- **A plan ending in refutation CLOSES its row.** Record the verdict in the archive and move on;
  do not re-scope a plan to find a win.

## Checkpoints on disk (`/home/a/code/ai-ml/models/`)

| model | dir | L / kv_heads / head_dim / hidden / n_heads / vocab | KV bf16 | forward | shipped `max_seq_len` |
|---|---|---|---|---|---|
| ~~Qwen3-235B~~ ⛔ **RETIRED + DELETED 2026-09-20 (user request) — do not test, do not gate** | ~~`Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`~~ | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 8192 (sdpa) |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 71981 (sdpa) |
| **DeepSeek-V4-Flash-0731** ✅ **SPLIT + LOADS as of 2026-09-25 (A5.7b); originals DELETED after a byte-for-byte verify, MTP's 3 shards moved to `models/deepseek-v4-mtp-shards/`. Loads in 3.7 s; does NOT yet emit a token — A5.7c.** | `deepseek-ai--DeepSeek-V4-Flash-0731` | 43 / 1 / 512 / 4096 / 64 / 129280 · 256 experts, top-6, **fp4 4.2500 bpw**, 12.750 MiB/expert, 155.42 GiB (137.06 experts + 8.24 dense + 10.12 MTP draft) | n/a | none — `deepseek_v4` unknown to transformers 5.5.4, no HF remote code in repo, TileLang kernels absent (row 28) | n/a |

(`max_seq_len` is KERNEL-DEPENDENT — the sdpa score term is charged from 2026-08-15 (rows 6/7), and
the sdpa reserve floor is raised to flash parity from 2026-08-16 (archived row 8). Every figure above is
sdpa, read off a real load and verified on-device at that exact length, 4/4 (archived row 9,
`plans/2026-08-16-row9-sdpa-limits-RESULTS.md`); all four geometries also derive these or more on flash.
KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
constants and decimal `MB_DEC`/`GB_DEC` for these — use the right family.)

---

## On-device gates — how to run them, and what each one can actually catch

`pytest tests/` runs NONE of these: they need a checkpoint, and two of them live at the repo ROOT,
not under `tests/`. Verified 2026-08-15 against the day's committed work.

| gate | invoke | ~time | catches |
|---|---|---|---|
| dense smoke (CLAUDE.md §0's named guard) | **DELETED** 2026-09-03 (`7d806e7`) — `test_batched_dense_smoke.py` no longer exists. No committed replacement | -- | *was:* dense path broken by MoE work. Teacher-forced perplexity, **8.78** on Llama-3.3-70B — unchanged as of its last real run |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. **×2 since 2026-09-20 (Qwen3-Next + MiniMax); the Qwen3-235B arm is gone with its checkpoint** |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity (8 tokens — a long generation CAN flip at an exact bf16 tie, row 3). MoE: first token + warm reproducibility. Runs on any checkpoint under `auto` since row 3 (2026-09-11); **Llama-3.2-1B passes in ~7 s**. Before row 3 a non-EXL3 checkpoint failed its premise assert |
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (**`-s` is enforced** — without it the module fails every test with a message naming the flag, before loading anything; G1) | 1–8 min | server broken end-to-end: load, non-stream, SSE stream incl. finish_reason and completion_tokens, oversized-prompt refusal, prefix reuse firing, env.json clean, requests.jsonl complete, gb_tok consistent with the run's own pinning (nonzero if it streams, zero if every layer is VRAM-pinned) |
| hidden states (row 18) | `HIDDEN_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_output_hidden_states.py -q -s` | 6 s (1B) / 30 s (MiniMax) | `output_hidden_states` broken: count `num_hidden_layers + 1`, shapes, a shifted or pre-norm entry (values vs stock HF — non-quantized checkpoints only; skips on EXL3), `output_attentions` refused by name. Seen to fail: RED on the unfixed forward, and on a shifted-layer mutant |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`** — except
**Qwen3-Next, which ALSO needs 29G as of 2026-09-23 (A5.4): the accumulator gate was OOM-killed
(exit 137) at 22G, after its 6 fixture tests and during the on-device load, and passes at 29G in
49.22 s** — and except
**MiniMax-M2.7, which needs 29G: at 22G its LOAD is OOM-killed (exit 137) at the pinning stage, measured twice
2026-09-11 (row 18); every recent MiniMax arm already runs at 29–30G** — and except
the dense smoke gate directly after another checkpoint's run, which needs **26G**: freed pinned host
blocks are cached rather than returned, and 22G was OOM-killed (exit 137) on 2026-08-16.

**The server gate had been dead for two commits — repaired 2026-08-23.** It failed at its FIRST
call and had done so since `b124320`, so nothing it claims to cover was actually covered in between.
Three independent breakages, none of them in the engine:

- `REQUIRE_SESSION_ID` went on and the gate sent no `session_id`, so every request was a **400**.
  Its oversized-prompt check went on "passing" — on the wrong 400.
- `DEFAULT_STREAM` went to `True`, so the gate's "non-streaming" step got SSE back and died in
  `r.json()`. Both steps now set `stream` explicitly rather than relying on the default.
- The gate ran with the server's `delete_original=True` default and **destroyed the checkpoint it
  was testing** (B14). It now passes `delete_original=False`: a gate may not consume its own input.

It also never checked the streamed `finish_reason` or `completion_tokens` — the exact blind spot B2
lived in, on the DEFAULT path. It does now. **The lesson is the one below, again: a gate nobody has
seen fail is not known to work.** Run it after any change to config defaults, not just to the engine.

**Two lessons from strengthening these, both of which cost a run to learn:**

- **A gate that cannot fail is not a gate.** The accumulator gate's original on-device arm ran two
  PREFILL forwards — a configuration where the defect provably cannot appear, since
  `_moe_prefill_pass` suppresses cache puts so arrival order never varies. It passed identically
  before and after the bug it existed to catch. Its replacement asserts on DECODE, and asserts its
  own premise (`_expert_vram_cache.hits > 0`) so it can never silently degrade back into that blind
  spot. **Both versions were checked by reverting the fix underneath them** — the first draft
  compared generated TOKENS and still passed with the bug restored, because an argmax flip is
  probabilistic; comparing LOGITS bitwise fails immediately (`max|d|=5.156e-01`). Assert on the
  continuous quantity, not on the discrete one that usually follows it.
- **Never load a model twice in one pytest process.** Both on-device arms called
  `from_pretrained`, and the second load was OOM-killed (exit 137) under a 22 GB cgroup cap —
  freed pinned host blocks are cached, not returned. A module-scoped fixture shares one load and
  halves the wall time.

# OPTIONAL — real work, no slot. Never ahead of OPEN or STILL OPEN.

**This heading predates §0 and is now a LOCATION, not a priority.** It used to say two rows here
(6a = P5, 10 = P8) were §0's top perf work; **both closed — P8 SHIPPED 2026-09-07, P5 CLOSED
2026-09-09 — and as of 2026-09-09 NO §0 P-row is open at all**, so every `[ ]` in this table is
OPTIONAL by §0's own test. §0 is still the order. Read the `§0` column below before picking anything
from this table.

**Build ORDER, set by the user 2026-09-11 — supersedes the 2026-09-10 order (14 → 13 → 24, then 6a-b / 3 / 5; rows 14, 13 and 24 are all closed): row 3 → row 25's MiniMax arm → row 18.** **Row 3 CLOSED 2026-09-11; row 25 CLOSED 2026-09-11 (MiniMax arm NULL) — next is row 18.** Everything else is not worth working now; the reasons and re-open conditions are in START HERE. The 2026-09-10 paragraph is in `git log -p -- todo.md`.

| # | Item | §0 | Status | Cost | Why it sits here |
|---|---|---|---|---|---|
| **6a-b** | `head_bytes` is probably not a transient at all | OPTIONAL | `[ ]` | GPU round | Split out of row 6a on its own evidence when that row closed 2026-09-09. §8-bounded the same way P5 was: the reserve is floor-bound on flash, so tightening it buys admission headroom, not tok/s. See below. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | OPTIONAL (parked) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) — **TRIAGED + 2 probes run 2026-09-06** | **P2** (the outlier split) | `[ ]` **3 of 4 closed 2026-09-06; 1 quality-certification item left** | CPU sweep | Two closed on new measurement (Plan B Phase 2 refuted directly; fold-`Pi` target removed by Plan I). Real-query gate RUN: **9 bits FAILS on Qwen (0.0649 vs 0.05 bar)** and 9 is the quantizer's ceiling. Outlier split promoted with a measured operating point. See below. |
| **8** | `tensor_align=True` is production-dead code | OPTIONAL | `[~]` **docstrings FIXED 2026-09-03; branch deliberately kept** | done / rest not scheduled | Opened 2026-09-03 by P6, docs half closed the same day. Nothing calls it since `c5014cd` (2026-07-13), and the stale docstring is what produced row R4/P6 and cost that row its premise. **FOUR false claims corrected** in `safetensor_model_persister.py` — `load_expert_into_slab` (mode advertised as "the prefill stager, via stage_read_expert"), the `tensor_align=False` bullet (claimed prefill never routes through `views_from_packed_copy`; it always does now), `_group_layout`, and **`views_from_packed_copy`, the dangerous one — it promised its output was `_TENSOR_ALIGN`-aligned "by construction", which stopped being true at B1; production sources are raw, so its views are copy-SOURCE-safe only and `gpu_repack_aligned` is what makes them kernel-safe.** The BRANCH stays: `_group_layout`/`expert_span_bytes` size every slab `max(raw_aligned_length, packed_span)` for it, and 5 test sites exercise it — deleting it is a sizing change, not a cleanup, and nothing needs it. Docstrings only, no behaviour touched; suite 1322/3. |

---

# 28b · [~] Port DeepSeek-V4-Flash-0731 so Vaayu can run it (added 2026-09-12) — sequence step S4 — **= row 47's A5. STEP 1 (the kernel gate) PASSED 2026-09-23, 6/6 on sm_120 — the abandon branch did NOT fire. Step 2 (HF modeling code) is next.**

**Source:** row 28's verdict. **Durable copy of the plan summary is this row.** Detail:
[`docs/2026-09-12-row28b-IMPL-deepseek-v4-port.md`](docs/2026-09-12-row28b-IMPL-deepseek-v4-port.md).

**Prize (projection, row 28, ±20 %):** 258 routed slots/token × 12.750 MiB = **3.212 GiB/token vs Qwen3-235B's 8.285**
→ **2.1–2.3 tok/s vs Qwen's measured 0.85 (~2.5–2.7×)**, with the CPU doing nothing. Already on disk, 155.42 GiB.
**NOT a CPU-hybrid enabler** — row 28 priced the hybrid on it at **+4.2 %**, the weakest of every candidate.

**Four blockers, each verified 2026-09-12, each independently fatal:** (1) transformers 5.5.4 has no `deepseek_v4`
(`AutoConfig` raises; Vaayu dies at `vaayu_base.py:1913`); (2) no `auto_map`, no `modeling_*.py` in the repo — what
ships is DeepSeek's standalone torchrun reference impl in `inference/`; (3) `tilelang==0.1.8` +
`fast_hadamard_transform` absent, needed for `fp4_gemm` / `fp8_gemm` / `sparse_attn` / `hc_split_sinkhorn`, none of
which exists in transformers; (4) ~155 GiB more disk for the `splitted_model/` copy (190 GB free → ~34 GB spare).

**The template is MiniMax-M2.7** — also unknown to transformers, runs here anyway because its repo ships
`modeling_minimax_m2.py`. **Write HF modeling code, drop it in the checkpoint dir with `auto_map`. Do not upstream.**

**STEP 1 IS DONE AND IT PASSED — 2026-09-23, ~1 h, no engine code.** `6/6` TileLang kernels build
AND validate on sm_120 / torch 2.11+cu130 / py3.12. Probe `plans/probes/a5_kernel_gate.py` (shapes
derived from `config.json`, one subprocess per kernel); suite after: **2422 passed / 16 skipped**,
the unchanged baseline. Detail:
[`…-A5-step1-kernel-gate-RESULTS.md`](docs/2026-09-23-row47-A5-step1-kernel-gate-RESULTS.md)
(gitignored — this row is the durable copy).

- **`act_quant`, `fp4_act_quant`, `hc_split_sinkhorn` run as shipped.** Both GEMMs validate against
  an exact-dequant reference (kernel vs the MATH, not vs the unquantised input, which would only
  re-measure fp4/fp8 rounding): **`fp8_gemm` rel 0.002753**, **`fp4_gemm` rel 0.002744**.
- **`apache-tvm-ffi==0.1.9` IS A LOAD-BEARING EXACT PIN, and `import tilelang` IS NOT A VALID TEST
  OF IT.** tilelang 0.1.8 declares `>=0.1.2,~=0.1.0`, so pip takes **0.1.14** → dies at import
  (`AttributeError: attribute '__dict__' of 'type' objects is not writable`,
  `tvm_ffi/registry.py:410`). **0.1.10 imports FINE and then dies when any kernel is COMPILED**
  (`'_NestedLoopCheckVisitor' object has no attribute '_inst'`, bundled
  `tvm/runtime/support.py:159`) — a bisect using import as its test stops there and reports a false
  pass, and the gate then "fails" six kernels for a reason unrelated to sm_120. **0.1.9 builds all
  six.** Both pins are in `requirements.txt` with this table. **`nvcc` is NOT needed** — tilelang
  compiles through nvrtc, already present for torch. Neither package can move this venv's torch
  (tilelang does not pin torch on Linux; verified by dry-run before and by version check after).
- **The ONE real sm_120 limit is `sparse_attn`'s tile — and it is TUNABLE, not an unsupported
  instruction.** `num_stages` has NO effect on the demand; `block` (KV positions/tile) is the only
  lever, against this card's **101,376 B** opt-in dynamic-smem ceiling:
  **block=64 (shipped) → 141,312 B (1.394× over) · block=32 → 104,448 B (over by only 3,072 B) ·
  block=16, stages=1 → FITS, runs, output finite.** So DeepSeek's sparse attention runs here at
  **¼ the shipped KV tile**; the throughput cost is **UNMEASURED** and belongs to step 5 — do not
  assume it is small, ¼ the tile is 4× the iterations over `index_topk=512`. **First optimisation
  if step 5 shows it hurts: recover 3 KiB of shared allocation and block=32 doubles the tile back.**
- **CORRECTION to the A5 brief — `fp4 4.2500 bpw` in the checkpoint table is RIGHT, not wrong.**
  Per expert matrix: weight `[2048,2048]` I8 = 4,194,304 B + scale `[2048,128]` F8_E8M0 = 262,144 B
  → ×3 = 13,369,344 B = **12.750 MiB exactly** over 25,165,824 params = **4.25 bits/param exactly**
  (4 for the fp4 weight + 8 bits per 32-param block = 0.25). The brief is right about the
  *container* — the dtype is `I8` holding two fp4 nibbles, and `quantization_config` names only the
  fp8 half of the checkpoint — but the bpw figure stands.
- **NEW NUMBER FOR A4's INHERITED CHECK 2:** shared experts are **24 MiB/layer** (3 × `[2048,4096]`
  F8_E4M3) → **~1.01 GiB across 43 layers**, previously unmeasured. That is what the VRAM pin
  budget has to cover.
- **Decode ceiling re-derived, and it prices A5 as REACH not speed:** 43 × top-6 = 258 routed
  slots/token × 12.750 MiB = **3.2124 GiB/token** → **~2.06 tok/s** at the NVMe plateau (7.1 GB/s,
  2 readers), **~1.75 tok/s** at the 1-reader 6.0–6.4 GB/s in-path rate. Consistent with row 28's
  2.1–2.3. That lands near MiniMax, ~5× slower than Qwen3-Next. **[S6 2026-09-26: this projection is REFUTED — measured 0.3267 tok/s. It omitted the non-expert weights that do not fit the VRAM pin budget (35 of 43 layers streamed, 72 % of the token). Use the S6 method, not this figure.]**
- **NOT established by this gate:** anything in `model.py` (only `kernel.py` was exercised) — the
  MLA/DSA modeling code, key remap, dual dequant and MLA KV sizing are all step 2+.
  `fast_hadamard_transform` is still absent but `kernel.py` never imports it: a step-2 concern.

## A5 STEP 2 — decomposed into A5.1 … A5.7 (designed 2026-09-23, user-directed)

**Step 2 ("HF modeling code") was too large for one spec — it is six subsystems, three of which the
A5 brief does not mention (hash routing, hyper-connections, DSpark/MTP).** Decomposed into seven
sub-plans, each with a SELF-CONTAINED document under `docs/superpowers/specs/` and its own row
below. **Work them IN ORDER: each module only has a golden reference if its dependencies are
already correct.** Implement one per fresh session.

**A5.1–A5.4 are full DESIGNS** (their inputs are knowable today). **A5.5–A5.7 are BRIEFS with start
gates and stop rules** — their inputs need measurement, and a stale design is worse than a brief
because the next session implements it instead of questioning it. Each is promoted to a plan by the
session that reaches it. Same ratchet A5 itself went through.

**Three decisions taken by the user 2026-09-23, do not re-litigate:**
1. **Verification is PER-MODULE golden parity, not end-to-end logits.** Row 28b's "verify against
   `inference/generate.py`" is **unrunnable and always will be** — the reference has no streaming
   and the checkpoint is **155.42 GiB against ~45.3 GiB** of VRAM+RAM (**3.4×** the box). But
   `world_size = 1` is the reference's default with **no `dist` init**, so ONE module runs here
   (measured 2026-09-23: 102.01-130.53 MiB for a layer's attention by compress ratio, 76.5 MiB for
   six experts). **A5.2 built that harness and it WORKS — see its row below.**
2. **Hash routing stashes `input_ids` on the model** — zero engine changes.
3. **MTP is OUT and excluded explicitly** (A5.7).

| # | sub-plan | kind | delivers | cost | doc |
|---|---|---|---|---|---|
| **A5.1** `[x]` | config + `auto_map` | **DONE 2026-09-23** | `AutoConfig` resolves `deepseek_v4`; next failure is the model class | ~½ d | [`…A5.1…`](docs/superpowers/specs/2026-09-23-A5.1-config-skeleton-design.md) |
| **A5.2** `[x]` | **parity harness** | **DONE 2026-09-23** | 15 goldens from the reference's own modules | done | [`…A5.2-RESULTS…`](docs/2026-09-23-A5.2-parity-harness-RESULTS.md) |
| **A5.3** `[x]` | dequant + `Linear` | **DONE 2026-09-23** | all three `linear_*` goldens bit-identical; 19 tests | done | [`…A5.3-RESULTS…`](docs/2026-09-23-A5.3-dequant-linear-RESULTS.md) |
| **A5.4** `[x]` | MoE | **DONE 2026-09-23** | zero engine changes, both `moe_*` goldens bit-identical inside; 24 tests | done | [`…A5.4-RESULTS…`](docs/2026-09-23-A5.4-moe-RESULTS.md) |
| **A5.5** `[x]` | **attention** | **DONE 2026-09-24** | MLA + sparse index on a contiguous `kv_view`; 3/4 goldens bit-identical, chunked prefill works where the reference RAISES | done | [`…A5.5-RESULTS…`](docs/2026-09-24-A5.5-attention-RESULTS.md) |
| **A5.6** `[x]` | block + model | **DONE 2026-09-24** | `from_config` builds the real skeleton in 1.44 s; `hc_head` bit-identical, `head` rel-L2 0.00166 + exact argmax; 30 new tests | done | [`…A5.6-RESULTS…`](docs/2026-09-24-A5.6-block-model-RESULTS.md) |
| **A5.7a** `[x]` | **engine: teach the streamed path** | **DONE 2026-09-24** | `run_embed` hook (identity default) + `VaayuDeepseekV4` subclass + `auto_model` dispatch + 2 refusals + the `weight_block_size` bug fixed; DeepSeek suite 289 passed, repo 2711/16, both MoE gates green; 25 new tests. **No checkpoint, no split, no token.** | done | [`…A5.7a-RESULTS…`](docs/2026-09-24-A5.7a-streamed-path-RESULTS.md) |
| **A5.7b** `[ ]` | load + first token | **BRIEF's remainder** | `fp4_group_size` into the loader, key remap (prefix + 2 renames + `mtp.*` skip), MTP index skip (11,008 not 11,776), split/persist, A4's two checks, first generation, row 26 probe | **~1–2 d**, mostly wall-clock | [`…A5.7…BRIEF`](docs/superpowers/specs/2026-09-23-A5.7-load-first-token-BRIEF.md) |

~~**≈3 weeks. A5.5 is where this dies if it dies**~~ — **IT DID NOT.** A5.5 closed 2026-09-24:
chunked prefill equals the whole prompt inside a measured null floor, and the acceptance gate was
mutation-verified to FAIL at 45–74× floor with the ring semantics restored. A5.6 and A5.7 remain,
and both have a golden reference or a Vaayu precedent.

### Durable summaries — each row survives its `docs/` file going missing (§7)

- [x] **A5.1 — config + `auto_map`. DONE 2026-09-23.**
      `AutoConfig.from_pretrained(CKPT, trust_remote_code=True)` now returns a `DeepseekV4Config`
      (43 layers) on this box; **the next failure moved downstream to `AutoModelForCausalLM`** —
      `ValueError: Unrecognized configuration class ... DeepseekV4Config`, which is A5.6/A5.7.
      Suite **2528 passed / 16 skipped** (2422 baseline + 106 new). Evidence:
      `tests/test_deepseek_v4_config.py`, `docs/2026-09-23-A5.1-config-RESULTS.md`.
      **What shipped:** `vaayu/model_code/deepseek_v4/configuration_deepseek_v4.py` is the
      SOURCE OF TRUTH (repo-tracked; the checkpoint dir is untracked and dies with the disk); a
      byte copy sits in the checkpoint, and a test asserts the two never drift.
      `tests/fixtures/deepseek_v4_config.json` is a committed copy of the shipped config — it is
      both the mandated backup and what lets A5.2–A5.7 test this path with **no 157 G checkpoint
      and no GPU**. Second backup in the checkpoint: `config.json.orig-2026-09-23`.
      **`auto_map` carries `AutoConfig` ONLY** — a dangling `AutoModelForCausalLM` pointing at a
      file A5.6 has not written yet turns the legible ValueError above into an ImportError out of
      HF's module cache. A5.6/A5.7 adds the second entry in the commit that creates the file.
      **Every field defaults to `None`, none is invented** (a parametrized test asserts the
      dataclass default of all 47 own fields), and every one of the 52 shipped keys round-trips
      field by field. `ModelArgs` NOT ported.
      **THREE FACTS FOUND BY RE-VERIFICATION, each durable:**
      **(1) The fp8-quantizer risk is VAAYU'S OWN, not transformers'.** `vaayu_base.py:3394` reads
      `config.quantization_config` and hands any non-exl3 `quant_method` to `AutoHfQuantizer` → a
      `FineGrainedFP8HfQuantizer` replacing every Linear. So the config **renames** the block to
      `checkpoint_quantization_config`: `getattr(config, "quantization_config", None)` is now
      `None` and no quantizer attaches. **Renamed, NOT dropped** — `fmt`/`scale_fmt` are the only
      source for the reference `ModelArgs`' `dtype="fp8"` / `scale_fmt="ue8m0"` (A5.2 needs them).
      **(2) `num_nextn_predict_layers: 1` IN THE SHIPPED CONFIG IS WRONG.** The reference's own
      `inference/config.json` says `n_mtp_layers: 3`, the disk holds `mtp.0/1/2`, A5.7's 768 extra
      experts are 3×256, and `compress_ratios` has **46 entries for 43 layers** (= 43 + 3).
      **A5.7 must not derive the MTP skip from that field.**
      **(3) transformers 5.5.4 rewrites two keys on load, and the tests pin both:** `torch_dtype`
      → `config.dtype` as a real `torch.dtype` (`configuration_utils.py:239`), and `rope_scaling`
      → `config.rope_parameters`, standardized — `rope_theta` folded IN and `rope_type: "yarn"`
      added (`modeling_rope_utils.py:706`). A5.5/A5.6 read `rope_parameters`, not `rope_scaling`.
      A1's `expected_expert_count()` **confirmed 256** against the real shipped config (third key
      `n_routed_experts`), not a hand-written dict.
      **§8:** (1) **no knob** — config resolution is automatic and there is nothing to configure;
      (2) nothing to plumb, `vaayu_base.py:2131` already calls `AutoConfig`; (3) **reachable but
      inert** — the server gets past 2131 on this checkpoint and dies at the model class instead;
      (4) it unblocks A5.2–A5.7 and ships **zero user-visible capability**. Banked, said so.
- [x] **A5.2 — the parity harness. DONE 2026-09-23. THE PREMISE HOLDS: DeepSeek's own reference
      modules RUN on this box with real weights.** `tests/deepseek/parity.py` + **15 committed
      goldens** in `tests/deepseek/goldens/` (**1.1 MB**). **NARROWED BY A5.3 (2026-09-23): the
      goldens hold the INPUT and the OUTPUT, not the weights, so a test that REPRODUCES one still
      needs the 157 G checkpoint and a GPU. What needs neither is the structural half — shapes,
      contracts, import rules — which was 15 of A5.3's 19 tests.** Recorder: `python -m tests.deepseek.parity [name…]`. Suite **2550
      passed / 16 skipped** (2528 baseline + 22 new). Evidence:
      `tests/test_deepseek_v4_parity.py`, `docs/2026-09-23-A5.2-parity-harness-RESULTS.md`.
      **Goldens:** `linear_fp4/fp8/bf16`, `expert`, `gate_hash`, `gate_score`, `moe_hash`,
      `moe_score`, `attn_{prefill,decode,chunked}_r128`, `attn_{prefill,decode,chunked}_r4`,
      `block`. **Attention is recorded on TWO compress geometries, not one** — `compress_ratios` is
      2×0, 21×4, 20×128 over 43 layers and only `ratio == 4` carries an `Indexer`; layers are found
      by searching `compress_ratios`, never numbered. Shapes are small (seqlen 8 / 4 tokens)
      because these are committed fixtures — **correctness at real length and `topk=512` is still
      A5.5's step-0 gate, NOT discharged here**.
      **`attn_chunked` — the answer, and the most valuable thing A5.2 bought: the reference CANNOT
      do chunked prefill, and it RAISES rather than corrupting.**
      `seqlen=512 at start_pos=8 raised RuntimeError: The expanded size of the tensor (1) must
      match the existing size (512) at non-singleton dimension 0` (`[1, 512]` vs `[512, 512]`;
      the r4 layer reads `[1, 256]`). `model.py:535` is
      `kv_cache[:bsz, start_pos % win] = kv.squeeze(1)` — the ring-buffer branch assumes
      `seqlen == 1` and Vaayu chunks at 512. Raising is the GOOD outcome: A5.5 cannot inherit the
      bug by accident, but **A5.5 must WRITE a chunked path, not port one.** Row-43 class.
      **SIX FINDINGS THE DESIGN DID NOT HAVE, each a silent failure if missed:**
      **(1) Module globals are load-bearing and fail SILENTLY.** `linear()` reads `scale_fmt`/
      `scale_dtype`, `Linear.__init__` reads `default_dtype` — all set ONLY in
      `Transformer.__init__` (`model.py:881-886`), which a bare module never runs. Leave them and
      `default_dtype` stays bf16, so an F8_E4M3 disk tensor loads **cast without its scale**, no
      exception. **(2) `torch.set_default_device("cuda")` is NOT optional** (`generate.py:91`):
      `get_window_topk_idxs` uses a bare `torch.arange`, so `sparse_attn` otherwise gets CPU
      indices and raises. Scope it and clear both `lru_cache`s on exit. **(3) The reference must
      run under `torch.inference_mode()`** — `Transformer.forward` is decorated with it and leans
      on it: `fp4_act_quant` writes in place over the view `fast_hadamard_transform` returns, which
      autograd forbids. Hits the `r4` path only. **(4) Two VALUE transforms are mandatory at load;
      the key RENAMES are not** — this checkpoint already ships the reference's own key names, but
      `wo_a.weight` is F8_E4M3 + per-128×128 scale on disk and **bf16 in the module** (load it raw
      and `load_state_dict` casts FP8 into bf16 WITHOUT the scale, silently), and routed experts are
      `I8` on disk / `float4_e2m1fn_x2` in the module. **(5) `fast_hadamard_transform`'s PyPI sdist
      is BROKEN** — ships without `csrc/`, fails at `cc1plus: fatal error:
      csrc/fast_hadamard_transform.cpp: No such file`. From **git** it builds and runs on sm_120
      (`TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation
      "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"`). It is **OPTIONAL, not a
      Vaayu requirement** (§5): the harness carries an exact torch Sylvester fallback and
      **replaces the reference's own `rotate_activation` binding**, since that function imports the
      extension directly. With the extension present the reroute is **bit-identical** — verified by
      re-recording both `r4` goldens. **(6) The config → `ModelArgs` mapping is VERIFIED, not
      assumed:** `inference/config.json` IS a `ModelArgs` JSON (`generate.py:81`), so all 38 of its
      keys are derived and match **value for value**, with exactly one disagreement —
      `num_nextn_predict_layers: 1` vs `n_mtp_layers: 3`, which turns A5.1's finding (2) into an
      enforced test. `norm_eps`/`hc_eps` are derived from the shipped config rather than taking
      `ModelArgs` defaults (§5).
      **MEASURED module costs** (shard headers, superseding the design's estimates): `Attention`
      **102.01 MiB** (ratio 0) / **110.26** (128) / **130.53** (4) — not "~119"; routed `Expert`
      12.750 MiB; `shared_experts` 24.00 MiB; `Gate` 2.00 MiB + `tid2eid` **5.92 MiB** (I64).
      **Traps that survived into the code:** `Attention.__init__` sizes `kv_cache` AND the RoPE
      table from `max_seq_len` (harness uses 4096, not the shipped 1,048,576); `sparse_attn` needs
      **`block=16, num_stages=1`** — the tile is a LOCAL, so the harness rewrites `kernel.py` to a
      temp file, and tilelang's `inspect.getsource` means it must be a REAL file; `model.py` is
      imported under an explicit name with `kernel` shadowed only for the import.
      **§8:** (1) **no knob** — test infrastructure; (2) nothing to plumb; (3) **NOT reachable by
      users at all, by design** — it does not touch `vaayu/`; (4) it unblocks A5.3–A5.7 and ships
      **zero user-visible capability**. Banked, said so.
- [x] **A5.3 — dequant + `Linear`. DONE 2026-09-23. ALL THREE `linear_*` GOLDENS REPRODUCE
      BIT-IDENTICALLY (`torch.equal`), not within the design's 0.0027 tolerance** — same kernels,
      same arguments, so any difference would have meant a different dispatch. Delivered
      `vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py` (`DeepseekV4Linear`,
      `dequantize_block_scaled`, `dequantize_at_load`, lazy `kernels()`), copied verbatim into the
      checkpoint dir with a drift test, + `tests/test_deepseek_v4_linear.py` (**19 tests**). Suite
      **2569 passed / 16 skipped** (2550 baseline + 19). Evidence:
      `docs/2026-09-23-A5.3-dequant-linear-RESULTS.md`.
      **THE MANDATORY DIVERGENCE HELD AND IS NOW GATED: read `self.scale`, never
      `self.weight.scale`.** The reference aliases scale onto the Parameter (`model.py:145`);
      `_inject_expert_weights` (`moe_patch.py:1225`) rebinds each key with
      `setattr(submod, base, nn.Parameter(t))`, so `weight` becomes a fresh object with no
      `.scale`. **Mutation-checked: restoring the reference idiom fails BOTH rebind tests with
      `AttributeError: 'Parameter' object has no attribute 'scale'`, including the real fp4
      golden.** Three more mutants each killed their gate (drop the default-dtype scope → 2 fail;
      transpose the dequant block dims → 3; hardcode `scale_fmt = None` → 2).
      **CONFIRMED ON REAL WEIGHTS: both keys bind, so streaming a dual-tensor quantized expert
      needs ZERO engine changes** — the fp4 golden is asserted through a weight rebound exactly the
      way the engine rebinds a streamed expert. Biggest de-risking fact for A5.4, now measured.
      **FOUR FINDINGS THE DESIGN DID NOT HAVE:**
      **(1) Both GEMM wrappers allocate their output with `torch.get_default_dtype()`**
      (`kernel.py:268`, `:533`) while the kernels are compiled to write **bf16**. The reference
      survives on `generate.py:77`'s bf16 process default; **Vaayu's default is float32**, so the
      forward scopes the default dtype to the activation's around the two kernel calls. Same class
      as A5.2 finding (1) — a module global nobody sets.
      **(2) `KERNEL_DIR` is unavoidable and A5.7 MUST set it.** transformers copies remote code
      into its own module cache WITHOUT the siblings, so a `__file__`-relative
      `inference/kernel.py` resolves only for the copy sitting beside the shards. The module
      exposes `modeling_deepseek_v4.KERNEL_DIR`; point it at the checkpoint's `inference/`.
      **(3) The fp4 group size is in NO config field** — an fp4 `DeepseekV4Linear` built without
      `fp4_group_size` RAISES rather than defaulting to this checkpoint's 32 (§5). Callers derive
      it as `in_features // scale.shape[-1]`.
      **(4) A5.2's goldens store the INPUT and the OUTPUT, not the weights** — so reproducing one
      needs the 157 G checkpoint AND a GPU. "Neither" holds only for the structural half (shapes,
      the rebinding contract, the lazy import, the dequant rule): 15 of the 19 tests. The other 4
      carry `needs_checkpoint` / `needs_gpu` and skip elsewhere.
      `dequantize_at_load` reproduces A5.2's harness transform on the real `wo_a` **bit-for-bit**;
      that test is what keeps the two copies from drifting.
      **§8:** (1) **no knob** — modeling code for one architecture, nothing to tune; `KERNEL_DIR`
      is load-time wiring for A5.7, not a server knob; (2) **not plumbed by design** — nothing
      outside its tests imports it yet; (3) **NOT reachable** — `deepseek_v4` still has no model
      class, so the checkpoint still cannot load; (4) unblocked by A5.4 → A5.5 → A5.6 → A5.7.
      Banked, said so.
- [x] **A5.4 — MoE, 94.3 % of the bytes. DONE 2026-09-23. "ZERO ENGINE CHANGES" HELD AND IS NOW
      MEASURED:** the diff touches `vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py` and one
      new test file — **nothing under `vaayu/` proper** — and **both MoE gates are green**
      (Qwen3-Next 8 passed, MiniMax-M2.7 8 passed). Delivered `DeepseekV4Gate` / `DeepseekV4Expert`
      / `DeepseekV4MoE` / `input_ids_scope`, + `tests/test_deepseek_v4_moe.py` (**24 tests**).
      Suite **2593 passed / 16 skipped**. Evidence: `docs/2026-09-23-A5.4-moe-RESULTS.md`.
      **PARITY — bit-identical wherever the arithmetic is shared:** `gate_score`, `gate_hash` and
      `expert` all `torch.equal`, and **both `moe_*` goldens are `torch.equal` when the routing
      weight is applied INSIDE**. Through Vaayu's convention (weight OUTSIDE, `moe_patch.py:1710`)
      the same block reads **max rel 0.02653 / 0.03125, cos 0.99958 / 0.99944** — and the
      bit-identical row is what proves that gap is the **fp8 activation rounding** (`w2` quantizes
      its input, so scaling before lands in different fp8 buckets than scaling after) and NOT a
      scoring slip, a mis-selected expert or a dropped shared expert. Do not "fix" it.
      **The reference's names are kept and are load-bearing:** `detect_moe_blocks` finds **43
      `module_list` blocks, `router_attr == "gate"`, `experts_attr == "experts"`** on a
      meta-device tree at the shipped geometry (43 × 256, ~4.6 s, no weights, no GPU); renaming
      `experts` kills 8 tests. `_shared_expert_output` (`moe_patch.py:536`) binds the **ungated**
      branch **on the real tree, by running** — A4 Check 1 discharged.
      **A4 CHECK 2, MEASURED OFF THE SHARDS (was unmeasured): shared experts are 1.0079 GiB over
      the 43 body layers, 24.001 MiB/layer, always resident.** MTP's three modules add 72.00 MiB
      (A5.7 excludes them); the 43 gates are 103.79 MiB including the hash layers' `tid2eid`.
      **HASH ROUTING IS A `ContextVar`, NOT A MODEL ATTRIBUTE.** Same user decision (stash the ids,
      no engine change), less coupling: `input_ids_scope(ids)` / `current_input_ids()` in the
      modeling file, so the MoE needs no back-reference and two generations cannot read each
      other's ids. **A5.6's model forward MUST run its layers inside that scope** — a hash layer
      with nothing stashed RAISES naming it, rather than falling back to score routing (silently
      wrong on 3 of 43 layers).
      **Mutants each killed their gate:** softmax instead of sqrtsoftplus → 8 fail; weights from
      the BIASED scores (the MiniMax bug) → 4; hash layers routed by score → 6; symmetric
      `swiglu_limit` clamp on `gate` → 1 (**the `expert` golden is the ONLY test covering the
      asymmetry**); `experts` renamed → 8.
      **RECORDED, NOT BUILT:** hash-layer routing is knowable from the token id BEFORE any compute,
      so those layers' experts could be prefetched a layer ahead — needs the engine change this
      plan avoided, so it belongs to a later row.
      **§8:** (1) **no knob** — routing, scoring and the swiglu limit all come from the config;
      (2) **not plumbed by design**; (3) **NOT reachable** — no model class yet, the checkpoint
      still cannot load; (4) unblocked by A5.5 → A5.6 → A5.7. Banked, said so.
- [x] **A5.5 — attention. STEP-0 GATE PASSED 2026-09-23, ALL THREE QUESTIONS. A5 DOES NOT DIE
      HERE.** MLA (one 512-dim latent shared by 64 heads), grouped low-rank O, circular 128-window
      KV, `Compressor`, and on `ratio == 4` layers a learned `Indexer` over `index_topk 512`.
      `compress_ratios` over 43 layers: **2 at 0, 21 at 4, 20 at 128 — confirmed on the real tree.**
      **KV IS TINY — G4 is an API problem, not a VRAM one: 50.99 MB/sequence at 8k = 0.0062
      MB/token, 41× SMALLER than MiniMax's 0.254** (recomputed from the config, holds). Do not spend
      a week on `vram_plan`. **THE PLAN IS NOW WRITABLE; the numbers below are its inputs.**
      Evidence: `docs/2026-09-23-A5.5-step0-gate-RESULTS.md`, probe
      `plans/probes/a5_5_attn_gate.py` (both gitignored — this row is the durable copy). ~1.5 h.
      **GATE RESULT (1) CORRECT.** `sparse_attn` at `block=16`, at the REAL working point
      (heads 64, head_dim 512, **topk 640 = window 128 + index_topk 512**, kv_slots 2176, m = 1 and
      m = 2048): **rel 0.002049 decode / 0.001955 prefill**, finite — same magnitude as step 1's
      GEMM validations (0.00275). **`block` does not change the math:** 16/32/64 agree to within
      6e-5 at a head_dim where all three fit.
      **THE BRIEF'S GATE (1) WAS UNRUNNABLE AS WRITTEN AND IS NOW CORRECTED.** It said to check
      against A5.2's `attn_prefill` golden. That golden is **seqlen 8**, so `index_topk` never binds
      (window alone is 128), **and it was RECORDED at `block=16`** (`parity.SPARSE_ATTN_BLOCK`) —
      reproducing it is CIRCULAR for the tile question. The gate instead compares the kernel to an
      independent write-out of `sparse_attn_kernel`'s own arithmetic (`kernel.py:294-348`). Two
      details there fail SILENTLY if a port drops them: **`attn_sink` enters the DENOMINATOR only**
      (`:343`, mass not value) and **the second GEMM consumes the probabilities cast to BF16**
      (`acc_s_cast`, `:338` — doing it in fp32 makes the kernel look wrong by ~1e-3).
      **GATE RESULT (2) COSTS 8.4–9.9 % OF THE DISK FLOOR, a 10× margin.** Per decode token over all
      43 layers: **`Attention.forward` 48.08 ms, of which `sparse_attn` is 4.65 ms**, against the
      **3.2124 GiB/token expert stream = 485.8–574.9 ms** at row P8's 6.0–7.1 GB/s. (The 3.2124
      GiB/token is **re-derived independently here from the shard headers** and reproduces row 28b's
      figure exactly.) Prefill of 2048 tokens, all layers: **932.8 ms = 0.455 ms/token.** Decode is
      timed at `start_pos = CONTEXT-1`, not at the prefill boundary — a ratio-128 layer's selection
      keeps growing as `start_pos // ratio` long after ratio-4's has saturated, and timing at the
      boundary under-charges 20 of 43 layers (192 vs 144 topk).
      **GATE (3) NOT RUN AND SHOULD NOT BE — its precondition did not fire.** The ¼ tile can cost at
      most what `sparse_attn` costs: **9.7 % of attention, 0.8–1.0 % of a token.** A perfect
      `block=64` recovers under 1 %. **Banked lever, nobody had written it down: the HEAD AXIS IS
      FREE.** The kernel pads heads to 16 and grids over `(m, b)`, so the fixed `q_shared`/`o_shared`
      pair scales with heads per launch — **4 launches at h=16 run the SHIPPED `block=64` tile on
      sm_120 (86,016 B ≤ 101,376)**, no allocation shaving. The ladder as a validated formula
      (reproduces step 1's three points exactly): **`h·d·2 + 2048 + block·d·2 + h·block·2`**.
      **THE FINDING THAT REORDERS A5.5: THE TILE WAS THE WRONG THING TO WORRY ABOUT.** Decode split
      by stage (96.1 % accounted, of 45.35 ms): **`indexer` 11.83 ms / 26.1 %** (21 layers only) ·
      **`compressor` 8.41 ms / 18.5 %** · **`wo_a` einsum 6.87 ms / 15.1 %** · `sparse_attn` 4.59 ms
      / 10.1 % · `wq_a`+`q_norm` 3.35 / 7.4 · `wkv`+`kv_norm` 3.27 / 7.2 · `wq_b` 2.18 / 4.8 ·
      `wo_b` 1.99 / 4.4 · `act_quant` 1.08 / 2.4. Per-layer forwards: **ratio 0 = 0.538 ms, ratio 4
      = 1.353, ratio 128 = 0.793.** **`wo_a` is bandwidth-bound and the reference itself flags the
      fix** (`model.py:544`): A5.2/A5.3 dequantise it to bf16 at load, so the einsum reads a 67 MB
      bf16 weight at ~420 GB/s against this card's 448 GB/s — an fp8 einsum roughly halves it.
      **Honest limit:** stages are timed by repeating ONE layer, so weights stay hot in this card's
      32 MB L2. It does not carry the result — 43 layers hold ~5.03 GiB of attention weight, a
      **~12 ms/token floor at 448 GB/s**, and the measured 45–48 ms is already 4× that.
      **STILL OPEN FOR THE PLAN (the gate did not touch these):** ring-buffer KV vs HF `Cache` (and
      42-B's snapshot); **chunked prefill — A5.2 proved the reference RAISES, so A5.5 must WRITE
      that path, not port one**; **in-place rotary, applied again INVERTED to the attention output**
      (`model.py:539`); `act_quant(..., 64, inplace=True)` FP8 QAT sim (**group 64, not 128**, and
      not in `config.json`); `attn_sink`; two rope variants (ratio-0 disables YaRN, base
      `rope_theta`).
      **§8:** (1) **no knob** — a gate measures, it configures nothing; (2) nothing to plumb, the
      probe does not touch `vaayu/`; (3) **NOT reachable** — `deepseek_v4` still has no model class;
      (4) it unblocks A5.5's plan and ships **zero user-visible capability**. Banked, said so.
      **DESIGNED 2026-09-23, SAME DAY — `docs/superpowers/specs/2026-09-23-A5.5-attention-design.md`
      (gitignored; these bullets are the durable copy). TWO USER DECISIONS, both widening the
      brief:**
      **(D1) A5.5 OWNS THE KV CACHE CLASS, not just its shape** (the brief gave the migration to
      A5.6). Writing attention against module buffers and migrating later is not a refactor, it is
      a second module. A5.6 keeps only the plumbing (`past_key_values`, `cache_position`,
      `attention_mask`, `position_ids`).
      **(D2) A5.5 WRITES THE MULTI-TOKEN PATH.** Refusing chunked prefill (the
      `recurrent_state.continuation_refusal` / `PREFILL_CHUNK_SIZE=0` precedent) was costed and
      REJECTED: it forfeits Vaayu's VRAM model, row 45's retention win and 42-B's partial prefix
      reuse, and does not scale past ~8k (the Indexer's score matrix alone is ~2.1 GB of transient
      at 8k).
      **THE REFERENCE BREAKS AT FOUR SITES, NOT ONE.** A5.2 proved site 1 raises; the other three
      were read off the source: (1) `Attention.forward` `model.py:535` one-slot ring write; (2)
      **`Compressor.forward`'s else-branch `:351-367`** — `kv.squeeze(1)`, `state[…start_pos %
      ratio]`, `should_compress = (start_pos+1) % ratio == 0`; (3) `get_window_topk_idxs` `:262-266`
      returns a `[window]` vector, not `[seqlen, window]`; (4) `get_compress_topk_idxs` `:276-277`
      and **`Indexer`'s mask `:430-436`** mask causally only under `start_pos == 0`.
      **CACHE DESIGN: `DeepseekV4CacheLayer(CacheLayerMixin)` per layer + `DeepseekV4Cache(Cache)`.**
      transformers 5.5.4's `CacheLayerMixin` has FIVE abstract methods (`lazy_initialization`,
      `update`, `get_mask_sizes`, `get_seq_length`, `get_max_cache_shape`). **`values` IS `keys`** —
      MLA's one 512-dim latent is both operands of `sparse_attn`. **That aliasing breaks three
      INHERITED methods**, so the layer overrides all three: `offload`/`prefetch` move keys and
      values independently (splitting the alias) and `reorder_cache` index_selects both —
      **`reorder_cache` RAISES** (beam search is unreachable in Vaayu's server; a half-reordered
      ring is worse than a refusal). **`get_mask_sizes` returns `(ring capacity, 0)` and says so in
      its docstring: this model builds NO HF mask** — `sparse_attn` masks by writing `-1` into
      `topk_idxs` and causality lives in the index builders.
      **SIZING CORRECTION — THE BRIEF'S KV NUMBER IS THE RING ALONE.** "50.99 MB at 8k = 0.0062
      MB/token, 41× smaller than MiniMax" counts only `keys`. Derived from the shipped config at
      8k: **ring 50.99 MB + Indexer rings 11.01 + Compressor states 12.21 = 74.20 MB/sequence =
      0.0091 MB/token, 28.0× smaller than MiniMax's 0.254** (not 41×). The conclusion is unchanged
      — KV is tiny, G4 is an API problem not a VRAM one — but **`vram_plan` must get the TOTAL, not
      the ring.**
      **`crop()` REFUSES, AND THAT IS A CORRECTNESS PROPERTY.** The window holds absolute positions
      `[L-win, L-1]` at slots `pos % win`; cropping to `keep` needs `[keep-win, keep-1]` and the
      `L-keep` positions below `L-win` were OVERWRITTEN. Exact only at `keep == L` — 42-B's
      "servable at exactly ONE length", reached by a different mechanism.
      **[bug] THE TRAP THIS CREATES — REQUIRED BEFORE A5.7 CAN CLAIM REACHABILITY.** With `values`
      aliased, the cache LOOKS like an ordinary KV cache to `batch.cache_holds_recurrent_state`
      (`batch.py:500`), which detects state by `conv_states`/`update_conv_state` and returns
      **False** — so `prefix_cache.py:112` (lookup), `prefix_cache.py:364` (abort-path restore) and
      `kv_persist` would all treat it as croppable. Extending that check is a **both-paths change
      under CLAUDE.md §0** and the user scoped it OUT of A5.5. **Until it is paid, this model must
      not be served with prefix caching or KV persistence enabled.**
      **ACCEPTANCE (§5.1) IS CHUNK EQUIVALENCE, and A5.5 has no model, so the object under test is
      ONE `Attention` module on real weights, one per ratio (0/4/128):** reference whole-prompt
      forward vs our port fed the same prompt in 512-token slices, rel-L2 on the final chunk's
      outputs **against a null-perturbation floor measured in the same run**, at TWO chunk counts.
      Row 43's precedent (0.314 vs a 0.051 floor) and D6's bit-sensitivity finding are why a fixed
      epsilon is not evidence. **Widening the tolerance to fit the result is the row-43 failure
      re-run — STOP and raise instead.** The two `attn_chunked` goldens store NO output (their
      content is the reference's RuntimeError + `prefill_x`; the chunk input is reproducible as
      `seeded_input(1, 512, dim, seed=1)`), so they become the test that our port does NOT raise.
      The four `attn_{prefill,decode}_{r4,r128}` goldens carry real outputs: reproduce them and
      **state which — `torch.equal`, or a measured divergence naming the op** (A5.3/A5.4 both
      landed bit-identical where the arithmetic was shared).
      **PERF IS OUT OF SCOPE and becomes its own row**, with the gate's breakdown as its opening
      position: `indexer` 26.1 % · `compressor` 18.5 % · `wo_a` einsum 15.1 % · `sparse_attn`
      10.1 %.
      **PLANNED 2026-09-23, SAME DAY — 7 TDD tasks,
      `docs/superpowers/plans/2026-09-23-A5.5-attention.md` (gitignored; this row is durable).**
      **ONE SPEC AMENDMENT, FOUND WHILE WRITING THE PLAN'S REAL CODE AND MATERIAL: A 512-TOKEN
      CHUNK OVERRUNS A 128-SLOT WINDOW.** Generalising the four `seqlen == 1` sites is necessary
      but NOT sufficient — the reference attends the ring IN PLACE, and writing 512 new positions
      into a `sliding_window = 128` ring destroys the history the chunk's own early queries need
      (a query at absolute 512 must see 385…512; after the write the ring holds 896…1023). **So
      attention attends a CONTIGUOUS `kv_view` assembled per forward: `cat([history(h), chunk(L),
      compressed(C)])` with `h = min(start_pos, window_size - 1)` and absolute position `a` at
      index `a - start_pos + h`.** It reduces EXACTLY to the reference at both supported shapes —
      `start_pos == 0` → `h = 0`, index == absolute; `seqlen == 1` → the same `window_size`
      positions in the same oldest-to-newest order the ring walk produces, so the online softmax
      accumulates identically and **bit-identity with `attn_decode_*` is expected, not hoped for**.
      The ring stays in the reference's `pos % window_size` layout as the CARRY between chunks.
      **A5.6 MUST NOT "SIMPLIFY" THIS BACK TO THE RING.**
      **Task order:** 1 rotary/RMSNorm/two rope variants (`precompute_freqs_cis` takes an explicit
      device, killing A5.2 finding 2's `set_default_device` trap at the source) · 2 index builders
      for any seqlen at any start_pos, each asserted to REDUCE to the reference · 3 the cache ·
      4 `Compressor` (**flushes every block a chunk completes** — the reference flushes at most
      one, and a miss leaves a hole nothing fills) · 5 `Indexer` (**per-query causal bound**; the
      reference masks only at `start_pos == 0`) · 6 `Attention` + the four goldens · 7 chunk
      equivalence + 5 mutants + both MoE gates + RESULTS.
      **Two literals with NO config source, named rather than hidden (§5):** `_ACT_QUANT_GROUP =
      64` (`model.py:512` — NOT `weight_block_size`'s 128) and `_FP4_BLOCK_SIZE = 32`
      (`model.py:18`).
      **Task 4 (`Compressor`) DONE 2026-09-24, commits `bb1c3ef` + a fix round (see report).**
      The parity test (`test_compressor_chunked_equals_whole_prompt`, both ratios 4 and 128) found
      ONE real bug, fixed: the overlap "carry" (previous block's raw kv/score, needed by the next
      block's overlap window) can't be produced by shifting `kv_state` the way the reference's
      decode branch does (`model.py:359-360`) — that shift only works because the decode branch
      fills the current window one position at a time; this port pools a whole block from a local
      tensor and never touches `kv_state` while doing it, so the shift copied stale/zero data.
      Fixed by writing the carry explicitly from the local tensor whenever a block flushes.
      **A second finding was NOT a bug and is NOT worked around in production code:** `wkv`/`wgate`'s
      plain `F.linear` is not invariant to the batch (M) dimension on this GPU (RTX 5060 Ti, torch
      2.11+cu130) — the same position's projection differs by a few ULPs depending on how many other
      positions ride the same GEMM call. This is expected cuBLAS behaviour (shape-dependent
      tile/split-k selection, confirmed by ruling out TF32, cuBLAS-vs-cuBLASLt, and determinism
      flags) — no BLAS guarantees bit-invariance across shapes, and `wkv`/`wgate` stay plain fp32
      `F.linear` exactly as the reference declares them. **The chunked-vs-whole comparison is
      instead gated the way spec §5.1 (Task 7's methodology) actually specifies: rel-L2 against a
      null-perturbation floor measured in the same run** (perturb the input by one ULP, re-run the
      whole-prompt path against itself), not `torch.equal`. Measured on device: ratio 4 chunked-vs-
      whole rel-L2 0.000375 vs a 0.0216 null floor (58x inside it); ratio 128 exactly 0 vs a 0.0219
      floor. Chunk-boundary drift for the full attention stack remains Task 7's job to gate, on the
      same null-floor methodology. Full DeepSeek-V4 suite green (214 passed) after refreshing the
      checkpoint's copy. Report: `.superpowers/sdd/2026-09-23-A5.5-attention/task-4-report.md`.
      **The first gate design FAILED ITS OWN MUTATION CHECK and was tightened, on device, before
      being trusted (2026-09-24).** Reinstating the overlap-carry bug above and running the test
      with the original `rel-L2 <= 3 * floor` threshold did NOT fail: mutant rel-L2 0.0608655 =
      2.82x the floor, but the 3x threshold (0.0648) sat just outside that, so `ratio=4` PASSED
      with the bug present (`2 passed`, no failure) — a gate with zero margin on the wrong side of
      the thing it exists to catch. The statistic (whole-tensor rel-L2) was never the problem; the
      multiplier was. **Threshold corrected to exactly the floor** (`rel-L2 <= max(floor, 1e-6)`,
      not a multiple — "chunking must perturb the output no more than a last-bit input change
      does" is a property, not a tuned constant). Re-run under the corrected threshold: `ratio=4`
      now FAILS on the same mutation (0.0608655 vs threshold 0.0216088), correct code still passes
      at 0.017x the floor — a ~160x separation, all margin either side. **`ratio=128` gives ZERO
      coverage of the carry bug** (it never enters the `overlap` branch the bug lives in, so its
      rel-L2 is 0 with or without the bug) — noted in the test's own docstring so "2 passed" is
      never misread as two independent confirmations of the carry path.
      **Task 6 (`DeepseekV4Attention`) DONE 2026-09-24, CLOSED — fix round 1 landed, suite green.**
      `DeepseekV4Attention` assembles `kv_view = cat([history, chunk, compressed])` exactly as
      designed. **One brief bug found and fixed against the reference (`model.py:530`/`:537`):
      the brief's Step 3 code called `self.compressor` only in the non-indexer branch; the
      reference calls its OWN Attention-level `Compressor` (head_dim-wide, filling `state.keys`'
      compressed half) UNCONDITIONALLY whenever `compress_ratio` is set, regardless of whether an
      `Indexer` also runs — the Indexer's internal `Compressor` (index_head_dim-wide,
      `state.index_keys`) is a separate object for scoring only. Fixed: `self.compressor(...)` now
      runs before either branch of the topk-index selection.** Also fixed the brief's
      golden-loading line itself: `modeling.dequantize_at_load(parity.load_module_weights(...))`
      raises `KeyError: 'wo_a.scale'` because `parity.load_module_weights` already pre-dequantizes
      `wo_a` for the REFERENCE module and drops the scale key — added a local
      `_raw_module_weights` (the same helper `test_deepseek_v4_linear.py` already uses) and load
      through `dequantize_at_load(_raw_module_weights(...))` instead, matching the production
      rule's own contract.
      **3 of 4 `attn_*` goldens are `torch.equal` bit-identical** (`attn_prefill_r128`,
      `attn_decode_r4`, `attn_decode_r128`), and both `attn_chunked_*` run to completion with a
      finite, correctly-shaped output (the reference's own RuntimeError golden).
      **`attn_prefill_r4` is NOT `torch.equal`, and is now GATED rather than left failing.** Root
      cause: `sparse_attn_kernel` tiles its online-softmax reduction in `block=64`-wide chunks;
      `window_topk_idxs` always returns `window_size=128` columns (decision #3 — intentional,
      needed so ONE function serves prefill/chunk/decode at any `start_pos`/`seqlen`), so at this
      golden's prefill (2 compressed blocks appended) the combined width is 130 → 3 tiles, with
      the 2 real compressed indices landing in the THIRD tile. The reference's own prefill branch
      SHRINKS its window index to `min(seqlen, window_size)` (8, here) so its combined width is 10
      → 1 tile, with the same compressed indices in the SAME tile as the window ones. An
      online-softmax merge across tiles (`scores_max`/`sum_exp` rescaled by `exp(prev_max -
      new_max)`) is exact in real arithmetic but NOT bit-identical to a single-pass reduction over
      the same values — classic floating-point reassociation, confirmed algebraically (an
      ALL-masked extra tile, as in `attn_prefill_r128`'s blocks=0 case, IS an exact no-op:
      multiply-by-1.0 and add-0.0 are both exact in IEEE754).
      **Ruling (coordinator, fix round 1): `window_topk_idxs` keeps its fixed `window_size` width
      — NOT re-branched for `start_pos == 0` — because that fixed-width contract is what lets one
      code path serve prefill, chunk and decode; re-adding the reference's shrink would be the
      design regressing to satisfy a fixture.** Instead, `attn_prefill_r4` is gated the way Task 4
      gates chunk equivalence: a same-run one-ULP null-perturbation floor, threshold exactly the
      floor (not a multiple). **Measured: rel-L2 vs golden 0.00423459, null floor 0.0488187 — the
      divergence sits at 8.7 % of the floor, 11.5× inside it.** The other three goldens stay on
      strict `torch.equal`. **Root cause pinned by a dedicated test, not just narrative:**
      `test_attn_prefill_r4_divergence_is_tile_width_not_masking` replays the forward with
      `window_topk_idxs`' output manually shrunk to the reference's own prefill width and asserts
      `torch.equal` against the golden — **it reproduces the golden bit-for-bit**, confirming tile
      width is the only source of divergence (a masking/indexing bug would not have healed by
      narrowing an unrelated tensor's width). Full suite: `31 passed` in
      `test_deepseek_v4_attention.py` alone, `224 passed` across all `test_deepseek_v4_*.py`
      (attention/config/kernels/linear/moe/parity), zero failures. Checkpoint copy unchanged this
      round (`diff -q` clean — no modeling-file edits were needed, only tests). Report:
      `.superpowers/sdd/2026-09-23-A5.5-attention/task-6-report.md` (see "Fix round 1").
      **Task 7 (acceptance gate) DONE 2026-09-24, A5.5 CLOSED.** The acceptance test
      (`test_chunked_prefill_equals_whole_prompt`) feeds one `DeepseekV4Attention` module per
      ratio (0/4/128) 1024 tokens whole vs. in slices at two chunk counts, against a same-run
      null-perturbation floor, threshold exactly the floor (Task 4/6's methodology): **6/6 passed
      honestly — ratio 0 and 128 land at rel-L2 exactly 0 (floors 0.0117058 / 0.0125609); ratio 4
      (the layer with an Indexer + overlap Compressor) lands at 0.0049874/0.00539576 against
      floors 0.0117829, 2.2-2.4x inside.** Two verbatim-brief bugs fixed before it would even run
      (both already-known failure modes from Task 6): `parity.load_module_weights` drops
      `wo_a.scale` (use `_raw_module_weights`, as the golden test already does), and the brief's
      test signature omitted the `checkpoint_kernels` fixture (`KERNEL_DIR` unset →
      `FileNotFoundError`).
      **STEP 2B, THE LOAD-BEARING CHECK, PASSED WITH LARGE MARGIN — THIS GATE CAN SEE ITS OWN
      DEFECT.** Mutated `view = torch.cat([history, kv], dim=1)` → `view = kv` (the reference's
      ring semantics, exactly what `kv_view` exists to prevent): **all 6 parametrizations FAILED,
      at 44.9x-73.8x the floor** (e.g. ratio 4/[512,512]: rel-L2 0.624875 vs floor 0.0117829, 53x
      over) — nowhere near the 1x boundary on either side. Reverted; `git diff` on the modeling
      file clean.
      **Five required mutants, each turned at least one test red, each reverted clean:** (1) drop
      the inverted output rotary → 5 failed (4 goldens + `attn_prefill_r4_divergence`); (2)
      `_ACT_QUANT_GROUP = 128` not 64 → 15 failed (4 goldens, `attn_prefill_r4_divergence`,
      `attention_does_not_raise` x2, all 6 `chunked_prefill` parametrizations); (3) swap rope
      variants (compressed dict for `compress_ratio == 0`) → 1 failed
      (`test_rope_settings_splits_the_two_variants_off_the_config`, structural, no GPU); (4)
      `DeepseekV4Cache.crop` returns instead of raising → 1 failed
      (`test_crop_refuses_below_full_length`); (5) reference ring semantics (`view = kv`, same as
      2b) → 8 failed (2 goldens + all 6 `chunked_prefill` parametrizations). No mutant stayed
      green; no new gate had to be written.
      **The design spec (§5.4) named SIX mutants; five shipped, and it is a coverage gap, not a
      shortfall — say why:** the other two, drop `attn_sink` and cast the second GEMM's
      probabilities to bf16, are arithmetic that lives entirely inside the checkpoint's own
      `inference/kernel.py` (`sparse_attn_kernel`), not in anything this port wrote — there is no
      line in `modeling_deepseek_v4.py` to mutate for either. Mutant (5) above (`view = kv`) was
      substituted in their place; it is not one of the spec's original six, it is A5.5's own
      load-bearing detail (the `kv_view` amendment, §0 of this row) getting the same mutation
      discipline.
      **Two one-line deferred findings folded in and BOTH VERIFIED TO ACTUALLY FAIL when
      violated, not just added:** `test_indexer_matches_the_reference_at_prefill` gained a runtime
      assertion `seqlen // 4 > config.index_topk` (derived, not literal) guarding the vacuous
      `blocks == index_topk` length the docstring only warned about — verified by setting
      `seqlen = 2048` and watching `assert (2048 // 4) > 512` fail before touching the GPU, then
      reverting to 4096.
      `test_attn_prefill_r4_divergence_is_tile_width_not_masking` (which duplicates `forward`'s
      body rather than calling it, so it pinned a COPY) gained a drift guard: the duplicated body
      is now a `replay(shrink: bool)` closure, and `replay(shrink=False)` (a plain unrolling of
      `forward` with no behavioural change) is asserted `torch.equal` against
      `layer(hidden_states, cache, 0)`'s own real output — verified by mutating real `forward`'s
      `apply_rotary_emb(..., True)` → `..., False)` (a change the copy does not mirror) and
      watching the new assertion fail; reverted.
      **Full suite 2652 passed / 16 skipped** (above the expected 2593/16 floor — the difference
      is A5.5's own added tests), **199.37 s**. **Both on-device MoE gates verified green** (this
      plan touched nothing under `vaayu/` proper, so this was a claim to check, not assume):
      **Qwen3-Next 8 passed (35.69 s), MiniMax-M2.7 8 passed (73.76 s).** The fused-EXL3 forward
      still has no on-device model (Qwen3-235B retired 2026-09-20) — no coverage claim made there.
      **Two things carried forward for A5.6/A5.7, restated here because `docs/` is gitignored:**
      (1) **`[bug]`, required before A5.7 can claim reachability — WIDENED by the final review,
      same family, more sites than croppability alone:**
      `batch.cache_holds_recurrent_state` (`batch.py:500`), `prefix_cache.py:112`,
      `prefix_cache.py:364` and `kv_persist` all detect recurrent state by
      `conv_states`/`update_conv_state` and would treat this cache as an ordinary croppable KV
      cache — it is not. **Until fixed, this model must not be served with prefix caching or KV
      persistence enabled.** The final review found the base `transformers.Cache`'s
      `max_cache_len`, `max_batch_size` (properties that read `layer.max_cache_len` /
      `layer.max_batch_size` for every layer) and `batch_repeat_interleave` /
      `batch_select_indices` (which call the same-named method per layer) all read/call
      attributes `DeepseekV4CacheLayer` does not define — a loud `AttributeError`, not a silent
      mistreat-as-croppable like the four above. **Nothing under `vaayu/` calls any of these four
      today (grepped clean)**, so this is not a new bug, just not yet a live one — A5.7 needs the
      full list before it can claim this cache is reachable through the generic `Cache` surface.
      (2) The `kv_view` amendment (chunk overruns the 128-slot ring →
      contiguous `cat([history, chunk, compressed])` per forward) is load-bearing, proven by
      Step 2b/mutant 5 above — **A5.6 must not "simplify" it back to the ring.**
      **§8:** (1) **no knob** — this task is a test suite + docs, no user-facing behaviour; (2)
      N/A, nothing importable is produced; (3) **NOT reachable** — no `deepseek_v4` model class
      yet; (4) unblocked by A5.6 → A5.7, with the two carried items above. Banked, said so.
      Detail: `docs/2026-09-24-A5.5-attention-RESULTS.md` (gitignored; this bullet is the durable
      copy). Report: `.superpowers/sdd/2026-09-23-A5.5-attention/task-7-report.md`.
      **FIX ROUND 1 (2026-09-24): the coordinator caught what two independent reviews had flagged
      from opposite ends — every original chunk split (`[512,512]`, `[256]x4`) has boundaries
      divisible by both compress_ratios, so `held = start_pos % ratio` was always 0 and the
      Compressor's `if held:` partial-block-resume prepend (exactly where Task 4's only real bug
      lived) was never entered.** Added a third `chunks` parametrization, `[511, 513]` (511 % 4 ==
      3, 511 % 128 == 127 -- genuinely off both boundaries), same gate, same threshold. **Passes on
      correct code:** ratio 4 rel-L2 0.00498729 vs floor 0.0117829 (2.36x inside), ratio 0/128
      exact 0 -- in line with the aligned splits. Full file: **40 passed**. **Mutation check:
      reinstated Task 4's exact original `_hold` bug** (the shift `kv_state[:bsz, :offset] =
      kv_state[:bsz, offset:]` restored, the explicit carry write removed from `forward`) --
      **new split FAILS at 0.0476728 vs floor 0.0117829 (4.05x over).** Outcome 1 (expected): gap
      closed and proven. Reverted; modeling file `git diff` clean.
      **Side-finding, not asked for:** the two PRE-EXISTING aligned splits also fail under this
      same mutation (3.85x and 5.14x over the floor) -- the overlap-carry READ
      (`first >= ratio`) fires on any call after the first regardless of alignment, so it was
      already reachable; only the NEW split's `if held:` prepend branch was genuinely uncovered
      before. Both observations consistent with the coordinator's finding.
      **FIX ROUND 2 (2026-09-24), the final whole-branch review's six small items, all closed in
      one pass:** (1) **closed the `compress_ratio == 0` coverage gap** — no test anywhere had
      compared a ratio-0 layer against the reference (goldens exist only at r4/r128, and the
      chunk-equivalence gate at ratio 0 runs OUR module against itself). Added
      `test_attention_matches_the_reference_at_a_ratio_0_layer` in `test_deepseek_v4_attention.py`,
      same pattern as `test_indexer_matches_the_reference_at_prefill`
      (`parity.reference_modules`/`reference_args`/`build_attention`, real weights, seqlen 300,
      `start_pos == 0`). **`torch.equal` — bit-identical, first try, no divergence.** Full suite
      234 passed (233 baseline + this one), 33.07 s. (2) Named the `compress_ratio == 4`
      discriminator: new module constant `_INDEXER_COMPRESS_RATIO = 4` beside `_ACT_QUANT_GROUP`/
      `_FP4_BLOCK_SIZE`, used at all four sites (`modeling_deepseek_v4.py`, the `coff` calc, the
      Indexer-construction guard in both `DeepseekV4CacheLayer` and `DeepseekV4Attention`, and
      `DeepseekV4Compressor.overlap`). No behaviour change. (3) Fixed the `kernels()` tempdir leak
      (`modeling_deepseek_v4.py:172` pre-fix) — `tempfile.mkdtemp` per load was never cleaned up;
      now `atexit.register(shutil.rmtree, ...)` on the same directory, so it is removed at process
      exit instead of never. The rewrite still happens at the same point (tilelang's JIT is LAZY —
      `inspect.getsource` fires on first kernel call, not at `kernels()` return — so the directory
      must outlive the function; it cannot be cleaned up synchronously inside it). (4) Corrected
      the `_have_hadamard_extension`/`rotate_activation` docstrings: selection is by importability
      only, not device — with the extension installed, a CPU tensor reaches it and raises
      `RuntimeError: Expected x.is_cuda()` rather than falling back. Harmless today (the Indexer
      path is CUDA-only) but the old wording overclaimed. (5) **Corrected the mutant count's
      record** — see the "Five required mutants" bullet above, now annotated: the spec (§5.4)
      named six, `attn_sink` and the bf16-probability-cast GEMM are arithmetic inside the
      checkpoint's own `inference/kernel.py`, not in ported code, so neither has a line to mutate;
      mutant (5) (`view = kv`) was substituted and is A5.5's own finding, not one of the spec's
      six. (6) **Widened the carried `[bug]` row** above (A5.7 reachability) — the final review
      found `Cache.max_cache_len`/`max_batch_size`/`batch_repeat_interleave`/`batch_select_indices`
      all read or call per-layer attributes `DeepseekV4CacheLayer` doesn't define (loud
      `AttributeError`, distinct from the four existing sites' silent mistreat-as-croppable);
      nothing under `vaayu/` calls any of the four today, so not a new bug, just now a complete
      list for A5.7. **Checkpoint copy refreshed and `diff -q` clean before commit** (items 2-4
      touched the modeling file). Full `tests/test_deepseek_v4_*.py`: **234 passed**, 33.07 s.
- [x] **A5.6 — block + model. CLOSED 2026-09-24.** 5 commits (`dce2657..9a5e056`). **SUCCESS
      CRITERION MET: `AutoModelForCausalLM.from_config` builds the real 43-layer, 11,008-expert
      skeleton under `init_empty_weights` in 1.44 s** (budget ~5 s) — this checkpoint has been
      unloadable since it landed on disk 2026-09-20; this is the first time `AutoModelForCausalLM`
      has resolved its class and built the full module tree. `fp4_group_size` derived = **32** off
      the shard headers (never a literal). **Goldens:** `hc_head` bit-identical (`torch.equal`,
      373.1 KiB fixture); `head` rel-L2 **0.0016627010190859437** vs gate `< 5e-3`, **exact
      argmax** — the bf16-head stop rule did NOT fire, the 0.986 GiB saving stands. **Block**: not
      bit-identical with the MoE weight applied outside (rel **0.00725** vs A5.4's own `< 5e-2`
      bound) but **bit-identical (`torch.equal`) with the weight applied inside**, isolating the
      whole gap to A5.4's documented weight-placement choice — nothing in HC, norms or attention.
      **Suite: DeepSeek 264 passed, 32.18 s (234 baseline + 30); full repo 2686 passed / 16
      skipped, 205.80 s** (= A5.5's 2656/16 close + these 30 — no drift elsewhere). **Both
      on-device MoE gates CHECKED (A5.6 touches nothing under `vaayu/`) and green: Qwen3-Next 8
      passed 35.07 s, MiniMax-M2.7 8 passed 73.92 s.** No fused-EXL3 coverage claim (no model on
      disk since 2026-09-20). **§8: (1) no knob — reached via `auto_map`; (2) N/A; (3) NOT
      reachable — `from_config` succeeds, `from_pretrained` does not; (4) unblocked by A5.7.**
      Detail: `docs/2026-09-24-A5.6-block-model-RESULTS.md` (gitignored; this bullet is the
      durable copy). Design: `docs/superpowers/specs/2026-09-24-A5.6-block-model-design.md`.
      **Hyper-connections are not in the A5 brief at
      all:** every block keeps **`hc_mult=4` copies of the residual** (`[b,s,4,d]`) and runs
      `hc_pre`/`hc_post` twice per block around a 20-iteration Sinkhorn
      (`kernels().hc_split_sinkhorn`, validated by A5 step 1).
      **STOP RULE DISCHARGED BEFORE BUILDING: HC IS contained inside `Block`** — `model.py:705` is
      `x = self.ffn(x, input_ids)` and `x` left `hc_pre` as `[b,s,d]`, so **A5.4's
      zero-engine-changes result holds**. A spy test keeps it true; if anything outside the layer
      ever sees `[b,s,hc,d]`, STOP and re-plan.
      **`hc_*` sizing re-derived, not inherited:** `mix_hc = (2+hc_mult)*hc_mult = 24`,
      `hc_dim = hc_mult*dim = 16384`; per layer `2×[24,16384] + 2×[24] + 2×[3]` fp32 =
      **3,145,944 B = 3.146 MB (3.00 MiB)**, over 43 layers **135.3 MB (129.0 MiB)** always-resident
      (feeds A4 Check 2 with A5.4's ~1.01 GiB). **They are F32 ON DISK** — not a load-time cast — so
      they are allocated with an explicit `dtype=torch.float32`, never via the default dtype, or
      `from_config(dtype=bf16)` makes a bf16 skeleton and accelerate casts the checkpoint's fp32
      DOWN to match it.
      **TWO BRIEF CORRECTIONS, both found by re-verification 2026-09-24:** (1) **`hc_head` is NOT on
      `ParallelHead`** — it is a method on `Block` (`model.py:709`) that `Transformer.forward` calls
      on whatever `layer` its loop left bound (`model.py:921`), passing the MODEL's params. It reads
      `self` only for `norm_eps`/`hc_eps`/`hc_mult`, all per-block copies of the same config fields,
      so it moves to `DeepseekV4Model` — arithmetically identical, and its golden proves it rather
      than asserting it. (2) **its params are `hc_head_fn [4,16384]`, `hc_head_base [4]`,
      `hc_head_scale [1]`** (`hc_mult` rows, 262,164 B) — **NOT the `[24,16384]` the brief says**.
      Arithmetic is a **plain sigmoid, NO Sinkhorn, no `post`/`comb`** — same-looking code,
      different shapes; that is the copy-paste hazard and why it gets its own golden.
      **DECISIONS (user, 2026-09-24).** (a) **`lm_head` stays bf16** — the disk dtype — matmul
      accumulating fp32 on the tensor cores, output cast to fp32 for sampling. The reference keeps
      it fp32 (`model.py:727`); **that would cost 1.973 GiB instead of 0.986 GiB**, against a Check-2
      budget already holding ~1.01 GiB shared experts + 129 MiB `hc_*` + ~5.33 GiB attention on a
      15.47 GiB card. Head golden is therefore rel-L2 **plus `argmax` equality**; **STOP RULE: if
      argmax ever disagrees, escalate to a vocab-chunked fp32 head — do not build it first.**
      (b) **A5.6 ships through `DeepseekV4ForCausalLM` + the `auto_map` entry**, proving
      `AutoModelForCausalLM.from_config` under `init_empty_weights()` — skeleton construction only.
      **CORRECTED by final review, 2026-09-24: A5.7 is NOT "purely weights + index + first token"
      from here** — the streamed path never calls this forward and constructs no `DeepseekV4Cache`;
      see the "SEVEN THINGS A5.7 MUST CARRY" items (6)–(7) near the top of this file for the full
      finding and the `vaayu_base.py` line numbers. (c) **Two goldens + structural tests**, no composed multi-layer golden —
      it would cover layers 0–1 only (both ratio 0, both hash-routed) for ~9.8 GiB to record and
      ~9.8 GiB to test.
      **NAMING — HF convention outside, reference names inside** (`model.embed_tokens`,
      `model.layers.N.{attn,ffn,attn_norm,ffn_norm,hc_*}`, `model.norm`, `model.hc_head_*`,
      `lm_head`). Forced, not taste: `loader_strategy.py:673` globs `model.layers.0.safetensors` to
      decide if a checkpoint is already split, `utils.py:359` defaults the streaming order to those
      names, and MiniMax uses them. `detect_moe_blocks` does not care that the MoE's parent is
      `ffn`: it matches the `nn.ModuleList` + router sibling and takes `layer_idx` from the first
      `\.?(\d+)\.`.
      **CORRECTION FOR A5.7 — the remap is NOT "~a `model.` prefix" (A5.4's claim, now checked
      against the index).** Disk keys are FLAT: `embed.weight`, `layers.N.*`, `norm.weight`,
      `head.weight`, `hc_head_*`, `mtp.N.*` (72,317 tensors). The remap is prefix **plus two renames
      plus a skip**: `embed.`→`model.embed_tokens.`, `head.weight`→`lm_head.weight`, `mtp.*`
      dropped. `head`→`lm_head` cannot be avoided by naming — `lm_head` sits outside `model.` by HF
      convention and `utils.py:359` expects it there. `config.json` already declares
      `"architectures": ["DeepseekV4ForCausalLM"]`, so the class name is fixed by the checkpoint.
      **Also owns:** the **`start_pos` → HF `Cache`** migration — the MODEL becomes HF-shaped
      (`cache_position` → one `start_pos`, passed down) while A5.5's attention signature does not
      churn — plus a **contiguity guard** (a `cache_position` gap/reorder RAISES; the ring, the
      Compressor's flushes and `window_topk_idxs` all assume `arange(start_pos, start_pos+seqlen)`,
      and a silent mis-index here is the same defect class A5.7 already calls out as the third time
      this program has paid for it); the `input_ids_scope` lifetime (whole layer loop, released
      after; **`inputs_embeds` without `input_ids` is REFUSED** — hash routing has no fallback);
      `tie_word_embeddings: false`; and `precompute_freqs_cis`'s `@lru_cache(2)` holding **exactly
      two** rope variants.
      **`attn_implementation`: `DeepseekV4PreTrainedModel` overrides
      `_check_and_adjust_attn_implementation` to return `"eager"` unconditionally** (its return is
      stored at `modeling_utils.py:1255`), `_supports_*` left `False`. This model has no HF
      attention backend — `sparse_attn` is its only kernel. Honest, and it saves Vaayu two
      thrown-away constructions of an 11,008-expert skeleton before its loop reaches `eager`.
      **DELIBERATELY NOT PORTED:** `Transformer.forward`'s `main_hiddens` collection at
      `dspark_target_layer_ids` (**non-empty in the shipped config: `[40,41,42]`** — live reference
      code, but consumed only by `forward_spec`, which A5 excludes) and its `sample()` call
      (that is `GenerationMixin`'s job). Recorded so a later reader does not read the omission as a
      miss.
      **§8, answered in advance:** (1) **no knob** — the classes are reached through `auto_map`;
      (2) N/A; (3) **NOT reachable** — the checkpoint still cannot load, `from_config` succeeds but
      `from_pretrained` does not; (4) unblocked by A5.7. Banked, said so.
      **BLOCKER FOUND WHILE WRITING THE PLAN 2026-09-24, and it changes A5.7 too:**
      `DeepseekV4MoE(config, layer_idx)` **RAISES** on the shipped config —
      `ValueError: fp4_group_size is in no config field -- read it off the shipped scale shape`.
      A5.4 made it an explicit keyword on purpose (knowable only from the checkpoint), but **A5.6
      builds 43 layers inside `from_config` under `init_empty_weights()`, where there is no
      checkpoint**, and Vaayu's call site is fixed and off-limits
      (`AutoModelForCausalLM.from_config(self.config, dtype=…, attn_implementation=…,
      trust_remote_code=True)`, `vaayu_base.py:3361` — editing it is a both-paths change under §0).
      **The config object is the only channel, and it already is one:** `DeepseekV4Config` gains
      `fp4_group_size: int | None = None` (precedent: `checkpoint_quantization_config`, a field the
      shipped `config.json` also does not name), and the decoder layer passes it through. **A5.4's
      keyword API is NOT touched** — its tests keep passing the value explicitly.
      **MEASURED so nobody re-derives it: `fp4_group_size = 32`** — routed `w1.weight [2048,2048] I8`
      (packed 2/byte → in_features 4096) against `w1.scale [2048,128] F8_E8M0`; 32 on all 40 of a
      random sample from the checkpoint's **33,024** routed expert weight tensors. **Still not a
      literal in code** — read off the shard HEADERS (`get_slice(...).get_shape()`, no tensor data).
      **HARD PRECONDITION FOR A5.7: the loader MUST set `config.fp4_group_size` before
      `from_config`, or the skeleton raises.** That is the correct failure — it names what is
      missing instead of inventing a group size — and A5.6 ships a test for each side of it.
      **Verified on-device 2026-09-24 while designing, so the plan does not gamble on either:**
      `with torch.device("meta"): reference.Block(0, args)` constructs (carrying
      `norm_eps=1e-06`, `hc_eps=1e-06`, `hc_mult=4`, `hc_sinkhorn_iters=20`), which is what makes the
      `hc_head` golden cost 262 KB instead of a 3.3 GiB layer; and `tests/test_deepseek_v4_*.py` is
      at **234 passed, 27.64 s** — A5.6 may only add. A5.6 touches nothing under `vaayu/`, so the two
      on-device MoE gates are a claim to CHECK, not assume — and **there is no `tests/gates/`
      directory**: the gate is `MOE_CKPT=<ckpt> pytest tests/test_moe_accumulator_determinism.py -q`,
      one arm per `systemd-run` scope at **29G** (22G exit-137s both models), in the FOREGROUND.
      Also measured: a full 43-layer meta skeleton is **~5 s** (0.11 s per MoE layer), which is why
      the structural tests stub the layer and only the `from_config` test builds the true geometry.
      **PLAN:** `docs/superpowers/plans/2026-09-24-A5.6-block-model.md` — 6 tasks, TDD, one commit
      each (gitignored; this bullet is the durable copy).
      **THREE BRIEF/PLAN CORRECTIONS PROVEN DURING THE WORK (all folded into the text above, restated
      here as the durable summary):** (1) `hc_head` lives on `Block`, not `ParallelHead` — a method
      reading only per-block copies of model-level config fields, moved to `DeepseekV4Model`,
      golden-proven not asserted; (2) its parameters are `[hc_mult, hc_mult*hidden_size]` =
      `[4, 16384]`, NOT the brief's `[24, 16384]` (that shape is `mix_hc`, used by `hc_pre`/`hc_post`,
      not `hc_head`'s plain sigmoid); (3) A5.7's key remap is NOT "~a `model.` prefix" — disk keys are
      flat (`embed.weight`, `layers.N.*`, `norm.weight`, `head.weight`, `hc_head_*`, `mtp.N.*`,
      72,317 tensors); the real remap is a `model.` prefix **plus two renames**
      (`embed.`→`model.embed_tokens.`, `head.weight`→`lm_head.weight`) **plus a skip** (`mtp.*`
      dropped).
      **NEW PENDING BUG for A5.7 — A5.5 code, out of A5.6's scope:** all three
      `.get("weight_block_size")` sites in `modeling_deepseek_v4.py` (current lines **493, 1166,
      1236**) pass `None` through to `DeepseekV4Linear` when `checkpoint_quantization_config` omits
      the field; `DeepseekV4Linear` unpacks it as `out_block, in_block = weight_block_size` and
      raises `TypeError: cannot unpack non-iterable NoneType object` instead of a message naming
      what's missing. Found investigating a Task 3 test failure; the fix that would have hidden it
      (a hardcoded `_WEIGHT_BLOCK_SIZE = (128, 128)` fallback) was REJECTED and reverted — `128` is
      on CLAUDE.md §5's forbidden-literal list, and unlike `_ACT_QUANT_GROUP`/`_FP4_BLOCK_SIZE`
      (which have no config source anywhere) this field DOES have one and the real checkpoint
      carries it, so inventing a default here repeats the exact mistake the `fp4_group_size` ruling
      above refused to make. The three tests that surfaced it were fixed to use `stub_layers`
      instead of constructing a real `DeepseekV4Attention`.
      **DEFERRED MINORS, collected for final review / A5.7 triage (each is low-stakes, none is a
      correctness bug in shipped behavior):** `test_deepseek_v4_block.py:119-151`'s `loaded_block`
      fixture hand-saves/restores `KERNEL_DIR`/`_KERNELS` because `monkeypatch` is function-scoped —
      needs a comment so a future reader doesn't "fix" it back and break the sharing;
      `test_the_residual_stream_is_hc_mult_copies_wide` (`test_deepseek_v4_model.py:179-199`)
      asserts only `.shape` — **checked on-device (final review, 2026-09-24) with `hc_mult=4`:
      `expand` (stride `(15,5,0,1)`) is numerically indistinguishable from `repeat` through
      `hc_pre`/`hc_post` (`hc_post` never writes into `residual`; `hc_pre`'s `flatten(2)` gives
      `torch.equal` results either way), so `.shape` is correct and sufficient given that — the
      docstring previously claimed `.repeat()` proves real storage vs. `.expand()`'s view, which is
      false and has been corrected** (the production code still uses `repeat`, matching the
      reference, as a fidelity choice, not a correctness one); `_start_pos`'s
      `cache_position.numel() != seqlen` branch has
      no test; no `get_output_embeddings`/`set_output_embeddings` on `DeepseekV4ForCausalLM`
      (harmless while `tie_word_embeddings` is `False`); `test_the_head_tracks_the_references_fp32_logits`
      exercises a standalone RMSNorm+Linear rather than the production `lm_head` (partially covered
      by Task 5's dtype assertion); the `labels`/`loss_function` branch is unexercised; **no
      parameter in this file is covered by `_init_weights`** (dispatches only on
      `nn.Linear`/`nn.Embedding`/norm-named modules) — `hc_head_*` and per-layer `hc_*` are bare
      `nn.Parameter(torch.empty(...))` (measured `hc_head_fn` |max| 5.47e+36), and so is every other
      weight in the file: `DeepseekV4Linear._parameter`, `DeepseekV4Gate.weight`/`tid2eid`/`bias`,
      and every expert weight — none reached by HF's `_init_weights`, and the reference does the
      same. `from_config` on a real
      device holds inf/NaN/garbage there until a checkpoint loads over it; harmless in Vaayu's own path
      (meta-only construction, always overwritten by real weights) but a surprise to any other
      caller — this is what caused (and was root-caused via) an order-dependent test flake, fixed
      test-side with a `seeded_parameters()` helper, not a production `_init_weights` override
      (would be dead code under `init_empty_weights`, and one covering only `hc_*` would be
      arbitrary given every other weight in the file is equally uninitialized); 16 pre-existing pytest warnings (14x
      `torch.jit.script_method` deprecation, 2x `SwigPyPacked`/`SwigPyObject`) are environment noise
      untouched by this branch.
- [x] **A5.7a — teach the streamed path DeepSeek-V4. ENGINE ONLY. CLOSED 2026-09-24.** Seven
      commits, one per task: `da21ec6` (weight_block_size guard), `974b3ee` (run_embed hook),
      `fd4fe40` (decoder layer use_cache), `7de88ec` (VaayuDeepseekV4 cache + compute hooks),
      `87d8bcb` (id scope + three refusals), `a5d6bde` (prefix-cache / kv_persist refusal),
      `9fdfbcc` (auto_model dispatch). RESULTS:
      [`…A5.7a-streamed-path-RESULTS.md`](docs/2026-09-24-A5.7a-streamed-path-RESULTS.md). Design:
      `docs/superpowers/specs/2026-09-24-A5.7a-streamed-path-design.md` (gitignored — this bullet is
      the durable copy).
      **§0 DECIDED: TEACH THE ENGINE.** Unstreamed is dead on arithmetic — **157 G** against
      **16,311 MiB VRAM + 31 GB RAM = ~47 GB** maximum resident, **3.3× over**. No `device_map` and
      no Vaayu setting changes that. Do not re-litigate the branch.
      **THE FINDING THAT RE-COST THE SUB-PLAN: four of the five deltas land on hooks
      `VaayuBaseModel` ALREADY HAS.** (1) cache → override `_create_kv_cache` (five call sites,
      `:2586/:3780/:4917/:5536/:5799`, one override covers all) returning
      `DeepseekV4Cache(config, max_seq_len=self.max_seq_len, ...)` — `max_seq_len` is already a
      DERIVED attribute (`:2446` off the VRAM plan, `:2474` fallback 512), so **no new knob**, and
      `kv_cache_mode` is overridden LOUDLY because this cache has no fp8/bf16/TurboQuant variant.
      (2) layer signature → `get_pos_emb_args` returns `{'start_pos': len_p}` (`start_pos` IS the
      past length), `get_past_key_value_args` is inherited unchanged, `get_attention_mask_args` and
      `get_position_ids_args` return `{}` (this model builds no HF mask; `sparse_attn` masks by
      reading `-1` out of `topk_idxs`). (3) `hc_head` before the norm → `run_norm` override. (4)
      `input_ids_scope` → subclass `forward` wraps `super().forward()`; `input_ids` is arg 1.
      **(5) IS THE ONLY SHARED EDIT: a new `run_embed(layer, seq)` hook**, default `return
      layer(seq)`, with `vaayu_base.py:5182` becoming `batch[j] = self.run_embed(layer, seq)`. The
      subclass expands `[b,s,d]`→`[b,s,hc,d]` with `.repeat` (not `.expand`, A5.6's fidelity
      choice) and reads `hc_mult` **off the config, never the literal 4**. The rejected alternative
      was a `dim()==3` guard inside `DeepseekV4DecoderLayer.forward` — zero engine edit, but it
      makes the layer's input contract conditional and hides model state in a shape test.
      **`use_cache` is hardcoded into the kwargs dict at `:5222` and NO hook can remove it**, so the
      modeling file accepts `use_cache=None` explicitly — **NOT `**kwargs`**, which would swallow a
      future engine kwarg silently (A1 Task 4 / 47-m5 / the MTP index, three times now).
      **NEW FINDING, from the design's own self-review — `_chunked_prefill` HAS TWO BRANCHES and
      only one is safe.** The **chunk-major** branch (`vaayu_base.py:4674`) calls
      `self.forward(input_ids=chunks, ...)` where `chunks` is a **Python LIST of 1-D tensors**; the
      scope wrapper would stash a list and `current_input_ids()` would hand a list to the
      hash-routed layers, so the wrapper REFUSES a non-tensor `input_ids`, naming this branch.
      **CORRECTED BY THE FINAL WHOLE-BRANCH REVIEW, 2026-09-24 (commit `f4854f4`) — THIS BULLET
      PREVIOUSLY CLAIMED THE BRANCH "IS NOT TAKEN FOR THIS MODEL", AND THAT WAS FALSE.** The model
      REPORTS eager (`_check_and_adjust_attn_implementation` returns `"eager"` for any request,
      without raising), but that is NOT what the engine records: `init_model` stores the
      **REQUESTED** candidate (`vaayu_base.py:3373`), and because the model accepts any request,
      `from_config` succeeds on the FIRST candidate — `"flash_attention_2"` wherever `flash_attn`
      is installed, as it is here. So `_active_attn_impl` was `"flash_attention_2"`, the gate at
      `:4666` was **OPEN**, and every prompt over `prefill_chunk_size` would have RAISED out of the
      wrapper's own TypeError. The tripwire test passed for the wrong reason: it asserted the
      METHOD's return value, while the gate reads `_active_attn_impl`, which that method never
      sets. **FIXED IN THE SUBCLASS, no shared-infra edit:** `__init__` now does
      `kwargs.setdefault("attn_implementation", "eager")`, which collapses `init_model`'s candidate
      list to one (`:3321`) and shuts the gate for real. `setdefault`, so an explicit caller choice
      still wins. With that in place DeepSeek-V4 does take the legacy per-chunk loop at `:4707`,
      which passes a 2-D tensor slice.
      **TTFT CONSEQUENCE FOR A5.7b, and it is structural, not a bug:** the chunk-major branch IS
      row 45's layer-major prefill (every expert read ONCE per prefill instead of once per chunk —
      2.75× at 2k, 5.47× at 8k on the module-list forward). Being eager-only, **DeepSeek-V4 cannot
      use it: a 4k prompt at `PREFILL_CHUNK_SIZE=512` re-reads its routed expert set 8×.** Widening
      that gate is shared infra well past `run_embed`, so A5.7a's stop rule fires and it is NOT done
      here — **candidate follow-up, and the reason A5.7b's measured TTFT must NOT be compared
      against MiniMax's post-row-45 figure.**
      **TWO REFUSALS THE SUBCLASS SHIPS.** (a) **`batch > 1` refuses** — `_INPUT_IDS` is a
      `ContextVar`, one per REQUEST, not per batch element, but the streamed loop runs
      `for j, seq in enumerate(batch)` INSIDE one scope, so at `j>0` the hash-routed layers would
      route sequence `j`'s tokens on sequence 0's ids, **silently**, producing plausible logits from
      the wrong experts. The server is N=1 so it cannot bite today; guard it before someone wires
      batching. (b) **prefix cache and `kv_persist` refuse** — the open `[bug]`:
      `batch.cache_holds_recurrent_state` (`batch.py:500`), `prefix_cache.py:112`/`:364`, and
      `Cache.max_cache_len`/`max_batch_size`/`batch_repeat_interleave`/`batch_select_indices` do not
      understand `DeepseekV4CacheLayer`. **A5.7b cannot claim reachability while this stands.**
      **ALSO FOLDED IN (A5.5 inheritance, same file): the `weight_block_size` `None` bug — and the
      OBVIOUS FIX IS A REGRESSION, corrected while writing the plan 2026-09-24.** Raising at the
      three `.get("weight_block_size")` call sites (`modeling_deepseek_v4.py:493`, `:1166`, `:1236`
      — line numbers re-verified exact) **would refuse a VALID checkpoint**: the unpack
      `out_block, in_block = weight_block_size` is reached ONLY inside
      `DeepseekV4Linear.__init__`'s `dtype == torch.float8_e4m3fn` branch (`:288`), and
      `_expert_dtypes` returns **bf16** whenever `checkpoint_quantization_config["fmt"] != "e4m3"`,
      where `None` is legitimate and never unpacked. That is the file's own documented contract
      (*"a bf16 checkpoint of this architecture builds bf16 experts and never touches a kernel"*)
      and CLAUDE.md §5's mandate. **The guard therefore goes in ONE place — inside the fp8 branch —
      mirroring the `fp4_group_size` error at `:280-284`; the three call sites are untouched.**
      **WHERE IT LIVES:** `vaayu/vaayu_deepseek_v4.py`, following `vaayu_qwen.py` (58 lines) /
      `vaayu_chatglm.py` (53), reached by `auto_model.get_module_class` branching on
      `config.architectures[0]` — the checkpoint already declares `"DeepseekV4ForCausalLM"`.
      **TESTING:** 15 tests on A5.1's committed `tests/fixtures/deepseek_v4_config.json` with
      stubbed layers (**no 157 G checkpoint, no GPU**; a real 43-layer meta skeleton is ~5 s, and
      the cache test pins a SMALL `max_seq_len` because the cache is preallocated across 43 layers).
      Floors A5.7a may only add to: **DeepSeek 264 passed**, full repo **2686 passed / 16 skipped**.
      **GATES OBLIGED** because delta 5 edits `vaayu/`: Qwen3-Next **and** MiniMax,
      `MOE_CKPT=<ckpt> pytest tests/test_moe_accumulator_determinism.py -q`, one `systemd-run` scope
      each at **29G**, **FOREGROUND**. **No fused-EXL3 claim is available** (no model on disk since
      2026-09-20) and dense has no committed gate (`test_batched_dense_smoke.py` deleted
      2026-09-03) — the identity default is the argument dense is safe, and it must be STATED.
      **STOP RULE: if the `vaayu/` edit grows past `run_embed` and its identity default — into
      shared staging, the prefix cache or the expert cache — STOP and re-plan.** That two-line
      surface is the entire reason A5.7a is separable.
      **§8: (1) NO KNOB** — the subclass is selected by `auto_model.py` from the checkpoint's own
      `architectures` field; inventing a knob would let a user pick the wrong forward. **(2) plumbed**
      — `auto_model.get_module_class`, with a test. **(3) NOT REACHABLE — banked, and said so:**
      `from_pretrained` still fails (no `fp4_group_size`, no remap, index still finds 11,776), and
      prefix caching / KV persistence are refused for this model. N=1 being the server's only mode is
      **necessary but not sufficient** — the brief's §0 correction still holds at A5.7a's close.
      **(4) unblocked by A5.7b.**
      **PLAN:** `docs/superpowers/plans/2026-09-24-A5.7a-streamed-path.md` — **8 tasks, TDD, one
      commit each** (gitignored; this bullet is the durable copy). Task order: 1 the
      `weight_block_size` guard · 2 the `run_embed` hook · 3 the decoder layer's `use_cache` · 4 the
      subclass's cache + compute hooks · 5 the `forward` wrapper and its three refusals · 6 the
      prefix-cache / `kv_persist` refusal · 7 `auto_model` dispatch + the eager-gate tripwire · 8
      full suite + both on-device gates + close. **25 new tests, none needing the checkpoint or a
      GPU** (one of the design's planned 27 was deleted during Task 5 review as redundant with
      existing `tests/test_deepseek_v4_model.py` coverage — floors: DeepSeek 264 → **289** [not
      291], repo 2686/16 → **2711/16** [not 2713/16]).
      **Two facts measured while designing, so A5.7b does not re-derive them: 230 G free on the
      models volume** (a split needs ~157 G — it fits, 73 G slack) and `DELETE_ORIGINAL = True`
      (`vaayu/server/config.py:62`) confirmed, which is what makes the first split irreversible under
      the standing NO-DOWNLOAD constraint. **A5.7a touches the checkpoint ZERO times.**

      **CLOSED 2026-09-24 — MEASURED, the durable copy (full detail in the RESULTS doc above):**
      **DeepSeek suite: `289 passed`, zero failed, 32.33s.** **Full repo suite: `2711 passed, 16
      skipped`, zero failed, 202.36s.** Both exactly match the floors above; nothing outside
      `tests/test_deepseek_v4_*` changed state. **Both on-device gates green, foreground,
      `MemoryMax=29G`:** Qwen3-Next-80B-A3B-AWQ-4bit **8 passed, 34.38s (wall 38.33s)** — A5.6's
      figure was 35.07s, consistent; MiniMax-M2.7-exl3 **8 passed, 72.72s (wall 76.80s)** — A5.6's
      figure was 73.92s, consistent. **No fused-EXL3 claim** (no model on disk since Qwen3-235B was
      retired 2026-09-20). **Dense has no committed gate** (`test_batched_dense_smoke.py` deleted
      2026-09-03); the argument dense is unaffected is `run_embed`'s identity default
      (`return layer(seq)`, `vaayu_base.py:4628`) — every model that doesn't override the hook gets
      back exactly what it got before the hook existed. Stated, not assumed.
      **§8, the durable answers: (1) NO KNOB** — the subclass is selected by `auto_model.py` off the
      checkpoint's own `architectures` field; a knob would let a user pick the wrong forward.
      **(2) PLUMBED** — `auto_model.get_module_class`, `elif "DeepseekV4" in arch: return
      VaayuDeepseekV4, {}`, with a Task 7 test confirming real-call reachability against the actual
      on-disk `config.json`. **(3) NOT REACHABLE — banked, said so plainly.** `from_pretrained`
      still fails on this checkpoint for three independent A5.7b reasons: `config.fp4_group_size`
      unset before `from_config`, the key remap unwritten, and the expert offset index finding
      11,776 entries (regex `\.experts\.(\d+)\.`) against the real 11,008 (`mtp.0/1/2.ffn.experts.*`
      = 768 extra = 3×256, colliding with layer 0 under `detect_moe_blocks` unless skipped).
      Separately, prefix caching and KV persistence are refused for this model outright. N=1 being
      the server's only mode is necessary but not sufficient here. **(4) unblocked by A5.7b.**

      **FIVE THINGS CARRIED FORWARD TO A5.7b (from the Tasks 1–7 reviews, not the brief):**
      **(a) The contiguity guard is bypassed on the streamed path.**
      `DeepseekV4Model._start_pos` validates `cache_position == arange(start_pos, start_pos+seqlen)`
      and raises otherwise — this cache's sliding window indexes by `pos % window_size` and its
      Compressor flushes on block boundaries counted from `start_pos`. The streamed path never
      calls `_start_pos`: `get_pos_emb_args` returns `{"start_pos": len_p}` with no check.
      Deliberately NOT fixed here — every other model's hook is equally unvalidated, and today's
      vectors are closed (prefix caching / KV persistence refused, batch>1 refused). **The residual
      vector is SPECULATIVE DECODING** (`drafter_model_path`, `draft_tokens`), which those refusals
      do **not** cover. A5.7b must add this check when it wires generation.
      **UPDATE from the final whole-branch review (2026-09-24): speculative decoding is confirmed
      NOT refused for this model** — `cache_holds_recurrent_state` (`vaayu/batch.py:500`, the
      `getattr(cache, "layers", None) or ()` loop at `:515`) finds no `update_conv_state` /
      `conv_states` on `DeepseekV4Cache` / `DeepseekV4CacheLayer`, so `spec_recurrent_refusal`
      returns `None` and speculative decoding is allowed to start. **But it fails LOUDLY, not
      silently**: the first rejected draft reaches `_trim_kv_cache` (`vaayu_base.py:5461`) →
      `DeepseekV4Cache.crop` → the sliding-window/Compressor state is not what `crop` expects →
      `RuntimeError`. Also note `vaayu_base.py:2586` calls `self._create_kv_cache()` purely as a
      probe to feed `spec_recurrent_refusal`, which for this subclass allocates a full preallocated
      `DeepseekV4Cache` (43 layers) and immediately discards it — wasted, not wrong.
      **(b) The open `[bug]`'s severity is lower than previously stated.** Measured:
      `_PREFIX_STORE_METHODS = ("to_cpu", "to_device", "resident_bytes")`, and neither
      `DeepseekV4Cache` nor `DeepseekV4CacheLayer` has any of them.
      **CORRECTED by the final whole-branch review (2026-09-24): this does NOT mean the pre-existing
      `_refuse_unsupported_prefix_cache` (`vaayu_base.py:3755`) auto-disables prefix caching with a
      warning for this model.** Task 6's hard refusal in `VaayuDeepseekV4.__init__` runs FIRST (it
      raises `NotImplementedError` before `super().__init__` is ever reached), so `:3755` is DEAD
      CODE on this path. The actual outcome of requesting `enable_prefix_cache=True` on this model is
      a hard load failure, not an announced downgrade. **Add prominently: the server's DEFAULT config
      cannot load this model** — `vaayu/server/config.py:100` sets `ENABLE_PREFIX_CACHE = True`, and
      `vaayu/server/app.py:406` passes `enable_prefix_cache=config.enable_prefix_cache` to
      `from_pretrained` unconditionally — so **A5.7b must decide** between (i) defaulting that knob
      off for this architecture, or (ii) downgrading Task 6's refusal to warn-and-disable, which is
      the engine's own precedent (`:3755`'s behavior for models it does understand). Do **NOT** change
      the refusal's behaviour as part of this correction; it is approved and correct as shipped — this
      is a note for A5.7b, not a reopening of A5.7a.
      **(c) DeepSeek-V4 cannot use the layer-major chunked prefill.** That branch (row 45's win) is
      gated on flash/sdpa.
      **CORRECTED by the final whole-branch review (2026-09-24): "this model forces eager
      unconditionally" was FALSE as a description of what the engine does.** The model's own
      `_check_and_adjust_attn_implementation` reports `"eager"` unconditionally, but `init_model`
      records the REQUESTED candidate, not that report (`vaayu_base.py:3373`) — and because the
      method accepts any request without raising, the first candidate tried wins. On a box with
      `flash_attn` installed the candidate list is `["flash_attention_2", "sdpa", "eager"]` and
      nothing refuses `"flash_attention_2"` for this architecture, so **the chunk-major gate
      (`:4666`) was open**, and every prompt over `PREFILL_CHUNK_SIZE` would have hit this class's
      own `TypeError` refusal for a list `input_ids`. Fixed in the commit that lands alongside this
      correction (`fix(A5.7a): force eager attn_implementation in VaayuDeepseekV4 to keep the
      chunk-major gate shut`): `VaayuDeepseekV4.__init__` now sets
      `kwargs.setdefault("attn_implementation", "eager")` before `super().__init__`, which collapses
      `init_model`'s candidate list to one (`vaayu_base.py:3321-3322`) and forces the requested and
      resolved value to agree. **With that fix in place, the underlying TTFT consequence stands and
      is real**: a 4k prompt at `PREFILL_CHUNK_SIZE=512` re-reads its routed expert set 8×.
      Structural, not a bug. **A5.7b's measured TTFT must NOT be compared against MiniMax's
      post-row-45 figure.**
      **(d) A positional bypass exists in theory.** The construction-time refusal reads
      `enable_prefix_cache`/`enable_kv_persist` from `kwargs`, and they are positional slots 21 and
      24 of `VaayuBaseModel.__init__`. Every construction path in the repo passes keyword-only
      (grepped clean) — unreachable today, but defended by convention, not code.
      **(e) Deferred minors:** two committed citations of `vaayu_base.py:5209` are stale.
      **CORRECTED by the final whole-branch review (2026-09-24): the file names above were wrong.**
      Verified by grep against the current tree: the kwargs dict this cites is at
      `vaayu_base.py:5222` (not `:5209`), and the two stale citations of the old line number are in
      `vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py` (~`:1430`) and
      `tests/test_deepseek_v4_engine.py` (~`:87`) — **`vaayu/vaayu_deepseek_v4.py` carries none.** The
      `run_embed` source-text test pins that the hook call exists but not WHERE; the Task 7 eager-gate
      tripwire is a source-text substring match, brittle to harmless reformatting (confirmed
      today-correct by the reviewer; not a live defect).

- [x] **A5.7b — load, index, split. RAN 2026-09-24/25; STOPPED ON ITS OWN STOP RULE. CLOSED
      2026-09-25 when its remainder A5.7c closed — nothing of A5.7b is still pending.** (Was `[~]`;
      corrected 2026-09-25.) **WHAT LANDED — branch `a5.7b-load-first-token`, 14 commits off `1995382`, NOT
      MERGED, full suite green at `2778 passed, 16 skipped`:** `cda7f3b` key_remap hook on
      `split_and_save_layers` + `get_checkpoint_key_remap` base hook (identity default) · `e553137`
      DeepSeek's disk-key→module-tree mapping with `mtp.*` dropped · `6dfd041` `fp4_group_size`
      derived from the SPLIT shards before `from_config` · `a3a5a17` prefix cache / KV persist
      downgraded to the engine's generic warn-and-disable, drafting refused by name, `start_pos`
      contiguity check · `0d895a5` docstring · `eaba376` **G2 homeless-key guard** + `hc_head_*`
      routed into the `model.norm` shard · `b21dd49` guard moved ahead of BOTH delete sites ·
      `01dbad6` prefix-cache refusal answered by class detection, not a probe cache · `4995b0e`
      **`F8_E8M0` mapped and the silent float32 dtype fallback replaced by a raise at all five
      reader sites** · `0816094` `KERNEL_DIR` pointed at the checkpoint's `inference/` ·
      `a2d7280` a real `DeepseekV4Cache` seeded before `generate()` delegates · `ab2e8cb` test
      hygiene · `64e7036` all 4 borrowed symbols resolved off the RUNNING module (the two-module
      split) · `9616c04` `_detect_moe` override, fp4 reinterpret + `dequantize_at_load`, injection
      failures now RAISE.
      **ON DISK AND IRREPLACEABLE — DO NOT RE-SPLIT. THE ORIGINALS ARE GONE AS OF 2026-09-25:**
      `deepseek-ai--DeepSeek-V4-Flash-0731/splitted_model/` = **46 shards + 46 `.done`, 146 G**,
      written in 186.4 s. `model.norm.safetensors` carries `hc_head_{base,fn,scale}`.
      **The split was BYTE-VERIFIED against the originals before they were deleted**
      (`plans/probes/a57b_verify_split.py`, gitignored): all **67,612 non-mtp tensors compared via
      `.view(torch.uint8)` through safetensors' OWN reader on both sides — 0 mismatches, 88 s.**
      Then **45 of the 48 originals were deleted, reclaiming 145.3 GiB** (free 85 G -> **231 G**),
      and the split was re-verified to load without them: **3.7 s, `is_moe=True`, experts=256,
      top_k=6, fp4_group_size=32, attn=eager**.
      **THE 3 PURE-MTP SHARDS WERE MOVED, NOT DELETED**, to
      **`/home/a/code/ai-ml/models/deepseek-v4-mtp-shards/`** (10.1 GiB: shards 46/47/48 plus a
      README and an `mtp_weight_map.json` slice of the original index) — so **row 28b step 6, the
      free speculative drafter, is STILL POSSIBLE** under the no-download constraint. That directory
      is **NOT a loadable checkpoint** (no config, no tokenizer, no complete shard set).
      **Why they were moved OUT rather than left in place — a trap worth remembering:**
      `assert_checkpoint_complete` returns early ONLY when **zero** `*.safetensors` remain
      (`checkpoint_integrity.py:28`). Leaving 3 of 48 would fail both its shard-numbering check
      (missing 1-45) and its index cross-check, and **break every future load of that directory**.
      Deleting all and deleting none are the only safe in-place states.
      **MEASURED ON DEVICE (run 7):** zero injection warnings (was 769); **11,008 expert entries
      across 43 MoE layers**; every layer patched `router=gate, experts=experts, n=256, top_k=6`;
      Expert VRAM cache 5.2 GB (~386 experts), RAM cache 23.4 GB (~1749 slots, expert=13.4 MB,
      working_set=3.45 GB); slab arena SKIPPED (module_list does not read it); **LOADED in 4.0 s**;
      `fp4_group_size=32`; `attn impl: eager`; prefix cache warn-and-disabled from an
      `enable_prefix_cache=True` call, i.e. the server's default.
      **A4 CHECK 2 IS ANSWERED, and it is a finding:** the pin summary reads `3 fully-pinned
      (0 transformer + 3 shared), 0 non-expert-pinned transformer` against `pinning budget=3.34 GiB`
      and an 8.24 GiB dense footprint — **NOT ONE transformer layer was pinned**, stronger than the
      design's predicted "budget exhausted partway".
      **BOTH ON-DEVICE MoE GATES GREEN, run 2026-09-25 at the branch tip before the merge**
      (they were owed from the start: this branch changes `vaayu/utils.py`, `vaayu/vaayu_base.py`
      and — the one that matters — `vaayu/persist/safetensor_model_persister.py`, whose dtype
      resolver Task 6b replaced at all five reader sites, THREE of them the expert readers
      Qwen3-Next and MiniMax stream through). Foreground, `MemoryMax=29G`:
      **Qwen3-Next-80B-A3B-AWQ-4bit 8 passed, 35.35 s** (A5.7a's figure was 34.38 s) ·
      **MiniMax-M2.7-exl3 8 passed, 73.22 s** (A5.7a's was 72.72 s). Both within a second of
      baseline. **No fused-EXL3 arm exists** (no model on disk since 2026-09-20) and **dense has no
      committed gate** (`test_batched_dense_smoke.py` deleted 2026-09-03) — stated, not assumed.
      **STILL NOT DONE: no token, no server gate, no row-28b step-5 measurement.** All three are
      A5.7c's.
      (Original A5.7b row text follows.) **DESIGNED 2026-09-24 —
      [`…-A5.7b-load-first-token-design.md`](docs/superpowers/specs/2026-09-24-A5.7b-load-first-token-design.md)
      · PLANNED the same day, 6 tasks:
      [`…-A5.7b-load-first-token.md`](docs/superpowers/plans/2026-09-24-A5.7b-load-first-token.md)
      (both gitignored; the bullets under "DESIGN, the durable copy" at the end of this row carry every
      decision and number). Task order: 1 `key_remap` in the splitter + hook · 2 DeepSeek's mapping ·
      3 `fp4_group_size` off the SPLIT shards · 4 prefix-cache downgrade + drafting refusal +
      `start_pos` contiguity check · 5 dry-run, split at `delete_original=False`, index = 11,008 ·
      6 first token, server gate, row-26 measurement, gates, close. **Suite floors as the plan leaves
      them: repo 2744/16, DeepSeek subset 314.** **ROW 28b's STEP-3 VERIFY IS WRONG AS WRITTEN:**
      it expects `ExpertOffsetIndex` to find `43 × 256 = 11,008`, but the regex
      `\.experts\.(\d+)\.` finds **11,776** — the extra **768 experts = 9.56 GiB** are
      `mtp.0/1/2.ffn.experts.*`. MTP is OUT, so **the index must skip `mtp.*` and find 11,008**.
      **DO NOT derive that count from `num_nextn_predict_layers`: it says 1 and the disk holds 3**
      (A5.1, 2026-09-23 — the reference's own `inference/config.json` says `n_mtp_layers: 3`, and
      `compress_ratios` carries 46 entries for 43 layers = 43 + 3). 768 = **3**×256.
      **It is a correctness hazard, not just waste:** `detect_moe_blocks` takes the layer index from
      the **first `.N.` in the path**, and `mtp.0.ffn` contains `.0.` → detected as **layer 0**,
      colliding with the real one. Same defect class as A1 Task 4 and 47-m5 — an index asking a
      question the load path answers differently, **third time this program has paid for it**.
      Also: key remap — **CONFIRMED by A5.6, not "~a `model.` prefix"**: it is that prefix PLUS
      `embed`→`model.embed_tokens`, `head`→`lm_head`, and a `mtp.*` skip (full detail in the A5.6
      bullet above); splitter runs
      A1's guards **before** `DELETE_ORIGINAL` deletes the only copy — **dry-run it**; A4 Check 1
      (pinning half) and Check 2 (**~1.01 GiB shared + ~135 MB hc + ~5.33 GiB attention** vs the pin
      budget; Phase 2 **`break`s** on miss at `vaayu_base.py:3956` — count the layers); then row 28b
      step 5's measurement. **§8: answer all four.** **Expect ~1.75–2.06 tok/s** (3.2124 GiB/token
      ÷ the NVMe plateau) **[S6 2026-09-26: this projection is REFUTED — measured 0.3267 tok/s. It omitted the non-expert weights that do not fit the VRAM pin budget (35 of 43 layers streamed, 72 % of the token). Use the S6 method, not this figure.]** — **A5 buys REACH, not speed; ~5× slower than Qwen3-Next. Do not present
      it as a perf win.**

      **DESIGN, the durable copy (2026-09-24). Re-verification of this row against the disk first —
      three citations here DRIFTED:** the test floors are **+2** on the A5.7a RESULTS doc —
      **DeepSeek 291 passed, full repo 2713 passed / 16 skipped** (`f4854f4`, the eager fix, added
      two tests after that doc was written; both re-run 2026-09-24, and these are A5.7b's floors);
      the `weight_block_size` sites are `modeling_deepseek_v4.py` **500 / 1173 / 1243**, not
      493/1166/1236; Phase 2's budget `break` is **~`vaayu_base.py:3977`** — `:3956` is the
      gate-pinning loop. Everything else in this row HELD, including 11,776 = 11,008 + 768 and the
      `mtp.0.ffn` → layer-0 collision.
      **Measured from the shard headers, so nothing below is carried:** total **155.42 GiB**;
      `mtp.*` is **10.117 GiB** (the brief's 9.56 GiB is 768 × 12.75 MiB, the experts ONLY);
      **split output 145.3 GiB against 230 GiB free — both copies fit, ~85 GiB slack**; per layer
      3264 MiB routed (256 × 12.75 MiB) + **136.9 MiB non-expert**; **dense footprint 8.24 GiB**
      (6.27 layers + 0.986 embed + 0.986 head), which reproduces this row's 8.24 independently.
      **THREE DECISIONS (user, 2026-09-24):** **(1) the first split runs `delete_original=False`** —
      under NO-DOWNLOAD it is the only undo there is, and it fits. **(2) The key remap happens
      BEFORE the split**, through a new `VaayuBaseModel.get_checkpoint_key_remap()` hook returning
      `None` by default (identity, same shape as A5.7a's `run_embed`), threaded
      `__init__` → `find_or_create_local_splitted_path` → `split_and_save_layers(key_remap=…)`.
      **The load-time alternative is DEAD, and here is why, so nobody re-derives it:** the splitter
      matches keys by SUBSTRING (`if layer in k`), so disk-named `layer_names` pulls **248
      layer-internal `*norm` keys into the `norm` shard and 3 `mtp` heads into the `head` shard`**,
      and the embed-shape probe at `vaayu_base.py:2141` breaks. Remapping first also makes the
      `mtp` skip and the 11,008 count fall out for free, with **no `mtp` filter anywhere else**.
      **(3) Prefix cache / KV persistence downgrade to warn-and-disable** — delete the hard refusal
      in `VaayuDeepseekV4.__init__` (KEEP the `attn_implementation` `setdefault("eager")`, which is
      load-bearing) and let the pre-existing generic path run (`_refuse_unsupported_prefix_cache`,
      `vaayu_base.py:3755`; `kv_persist` already prints `ignored` at `:3097`). Reason: the server's
      DEFAULT config sets `ENABLE_PREFIX_CACHE = True` and `app.py:406` passes it unconditionally,
      so the hard refusal made the default config unable to load this model at all.
      **TWO BLOCKERS FOUND IN RE-VERIFICATION, not in the brief:** (a)
      `_fp4_group_size_from_headers` reads the ORIGINAL index + shards, and
      `delete_original_shards` keeps the index but DELETES the shards — so the loader must derive
      the group size (**32**) from the SPLIT shards; the helper moves out of
      `tests/test_deepseek_v4_model.py` into `vaayu/model_code/deepseek_v4/` and the test imports
      it. (b) The substring-match problem in decision 2 above.
      **Two things SIZED rather than assumed:** the whole 43-layer `DeepseekV4Cache` is **70.8 MiB
      at `max_seq_len=8192`** (43.9 at 4096, 124.5 at 16384), so A5.7a's "wasted, not wrong" probe
      allocation needs no gating; and A4 Check 2 should EXPECT the pin budget exhausted partway —
      6.27 GiB of layers + 1.97 GiB embed/head against a budget near 4.8 GB, with the remainder
      falling back to `LRUTensorCache(_ne_budget)` (RAM re-reads, NOT per-token streaming).
      **Also carried in from A5.7a item (a): speculative decoding is NOT refused today** and dies
      in `_trim_kv_cache` (`vaayu_base.py:5428`) on the first rejected draft — A5.7b refuses
      `drafter_model_path` by name and adds the `_start_pos` contiguity assert to
      `get_pos_emb_args`.
      **Gates obliged (shared infra → both paths, CLAUDE.md §0):** full suite **2713/16**, DeepSeek
      **291**, BOTH MoE accumulator gates foreground at `MemoryMax=29G`. **No fused-EXL3 arm exists**
      (no model on disk since 2026-09-20) and **dense has no committed gate** — the argument dense
      is unaffected is the hook's `None` default, stated not assumed.

- [x] **A5.7c — streamed fp4 expert injection, then the first token. DESIGNED 2026-09-25, CLOSED 2026-09-25.**
      **READ FIRST:** [`…-A5.7c-expert-injection-design.md`](docs/superpowers/specs/2026-09-25-A5.7c-expert-injection-design.md),
      **PLANNED 2026-09-25 — 7 tasks in
      [`…-A5.7c-expert-injection.md`](docs/superpowers/plans/2026-09-25-A5.7c-expert-injection.md)**
      (1 injector + tests, 2 allowlist single home, 3 suite + both gates, 4 first token, 5 server
      smoke, 6 row-26 measurement, 7 bank + close). Branch `a5.7c-expert-injection` off `main`.
      Its input was [`…-A5.7c-expert-injection-HANDOFF.md`](docs/superpowers/specs/2026-09-25-A5.7c-expert-injection-HANDOFF.md)
      (both gitignored — this row is the durable copy). **The handoff's "branch NOT merged" is
      STALE: A5.7b IS merged (`4842250` + docs `4bc70b1`, `c89ea1e` on `main`), and per `4bc70b1`
      the 48 original shards are DELETED after a byte-for-byte verify — 46 split shards, 231 G free,
      MTP's 3 shards in `models/deepseek-v4-mtp-shards/`. A5.7c branches off `main`.**
      **THE ONE DEFECT:** `moe_patch.py:1208` (`t = tensor.to(device, dtype=dtype, non_blocking=True)`)
      casts every streamed expert tensor to the running dtype, so a packed-fp4 expert weight reaches
      `DeepseekV4Linear.forward` as **int8**, takes its `weight.dtype not in _QUANTIZED` branch
      (`modeling_deepseek_v4.py:314`) and dies in `F.linear` with
      `mat1 and mat2 shapes cannot be multiplied (1x4096 and 2048x2048)` — the packed [2048, 2048]
      weight against a [1, 4096] activation. It must be **reinterpreted** (`.view(target_dtype)`),
      never cast; a cast corrupts the packed bytes even when it does not raise (A5.7b Task 6f, Q2).
      **WHY IT IS A RE-PLAN, NOT A FIX:** `moe_patch.py` is the expert-streaming path, and its
      module-list forward is the one **MiniMax-M2.7 runs in production**. A5.7b fixed the identical
      defect for the NON-expert path inside the subclass (`9616c04`), but experts do not travel that
      path and `moe_patch.py` has **no subclass seam** — module-level functions over `block_info` /
      `vaayu_ref`. A5.7b's plan said "if it grows into the expert cache, STOP and re-plan"; it did.
      **SECOND DEFECT, SAME LINE — FOUND AND MEASURED WHILE DESIGNING 2026-09-25, and fixing only
      the fp4 weight buys a crash-free WRONG token:** each fp4 weight's companion `scale` is
      declared `_SCALE_DTYPE = torch.float8_e8m0fnu` (`modeling_deepseek_v4.py:55`) and `forward`
      reads the quantization format back off it (`scale_fmt = "ue8m0" if scale.dtype == _SCALE_DTYPE
      else None`, then `gemm(..., scale, scale.dtype)`). Line 1208 casts it to bf16 → `scale_fmt`
      collapses to `None` and TileLang's `fp4_gemm` gets a bf16 scale. **`e8m0 → bf16` SUCCEEDS
      SILENTLY (probed).** **Why the non-expert path never needed this:** it injects via
      `set_module_tensor_to_device`, which **coerces the value back to the declared parameter dtype**
      (probed: bf16 into an e4m3/e8m0 param keeps the declared dtype, and bf16 round-trips both
      losslessly) — so `4995b0e`'s "F8_E8M0 casts without incident" is TRUE there. `moe_patch` uses
      `setattr(submod, base, nn.Parameter(t))` — **no coercion**, the param just becomes bf16.
      **THE TWO DESIGN QUESTIONS, ANSWERED (design §2):** (1) **target dtype = `getattr(submod,
      base, None).dtype`**, read off the live expert module in the loop — one attr read + one tuple
      test per streamed tensor (~6/expert), NOT `self.model.get_parameter()`'s string walk, and no
      per-layer cache. Safe because the dtype is a **fixpoint** of the cache/detach cycle: probed,
      `.to("meta")` preserves dtype, and re-binding a param to a `.view(fp4)` tensor then detaching
      keeps fp4. `ExLlamaV3Linear` (no `weight` attr) and a `None` scale both fall to the old path,
      so MiniMax is untouched. (2) **shared edit in `moe_patch`, NOT a subclass hook** — the rule is
      a torch invariant (a packed dtype can only be reinterpreted, never cast), not model policy, so
      gpt-oss's MXFP4 needs it with no subclass written; a hook costs an indirection per tensor in
      the decode hot path; and the branch's blast radius is ZERO for any model declaring none of
      these dtypes, which both gates prove rather than argue. `NO_CAST_INJECTION_DTYPES =
      (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu)` at `moe_patch` module level;
      `vaayu_deepseek_v4.py:27`'s `_PACKED_INJECTION_DTYPES` (referenced nowhere else) imports it.
      **fp4 + e8m0 ONLY** (user's call): `float8_e4m3fn` is excluded because DeepSeek's e4m3 tensors
      are PINNED, not streamed, so that arm would ship with no test reaching it.
      **(3) FAILURE MODE, decided:** the NEW branch raises naming layer/expert/submod/base/both
      dtypes; the existing cast branch keeps its `warn` + `return None` verbatim — no behaviour
      change for MiniMax/Qwen, no silent path in new code.
      **§8, four answers (this line's original claim — "reachable, proven by the server smoke below" —
      is STALE as of Task 5; corrected here, restated in full at the row's close below):** (1) no
      knob — STATED, not assumed; a correctness invariant with nothing to tune. (2) not applicable —
      no knob, nothing to plumb. (3) claimed going in as "reachable in the server's only mode (N=1
      decode), proven by the server smoke below" — **Task 5 REFUTES that for the completion path: see
      below, the missing-`chat_template` `[bug]` blocks every `/v1/chat/completions` request before
      the injection fix is ever exercised.** (4) said out loud at the row's close, with the three
      banked rows immediately below it.
      **MUST NOT CHANGE:** any tensor whose target dtype is not in the allowlist must inject
      byte-identically to today — explicit allowlist, NOT a general "dtypes differ → reinterpret"
      branch. Unit test asserts that byte-identity directly.
      **GATES, NON-NEGOTIABLE:** full suite (floor **2778 passed, 16 skipped**) AND **both** MoE
      accumulator gates foreground at `MemoryMax=29G` — Qwen3-Next (~35 s) and **MiniMax-M2.7
      (~73 s), which shares this exact forward and is the whole point**. No fused-EXL3 arm exists;
      dense has no committed gate — say both rather than claiming coverage.
      **THEN, still A5.7c:** first token · `SERVER_CKPT=… tests/test_server_smoke.py -q -s` (the only
      proof of §8(3)) · row 28b step 5 via `plans/probes/row26_decode.py`, which takes **no
      `--ckpt`** — add a `"deepseek"` entry to its hardcoded `MODELS` dict and run
      `MODEL=deepseek BALLAST_GB=25` (it hardcodes `delete_original=False`, verified, so it cannot
      eat the checkpoint). **Expect ~1.75–2.06 tok/s — REACH, not speed, ~5× slower than
      Qwen3-Next. **[S6 2026-09-26: this projection is REFUTED — measured 0.3267 tok/s. It omitted the non-expert weights that do not fit the VRAM pin budget (35 of 43 layers streamed, 72 % of the token). Use the S6 method, not this figure.]** TTFT is NOT comparable to MiniMax's post-row-45 figure** (eager keeps the
      chunk-major branch shut, so a 4k prompt re-reads its expert set 8×).
      **BANKED OUT OF SCOPE, as their own rows before this branch closes:** (a) the **fused
      forward's twin cast** — `_make_fused_sparse_forward`'s injector casts the same way and
      gpt-oss's MXFP4 experts will hit it, but gpt-oss cannot load (G1: expert index matches 0/459
      tensors), so a fix there is untestable code on Qwen3-Next's perf-tuned path (user's call
      2026-09-25); (b) **`warn` → `raise`** in `_inject_expert_weights` — corrected 2026-09-25
      (final-review M3): that function has exactly one call site, the module-list forward, so this is
      a behaviour change on MiniMax-M2.7 only, not "two production forwards" as first written here;
      (c) **A4 check 2's finding** (0 non-expert-pinned transformer layers,
      budget 3.34 GiB vs an 8.24 GiB dense footprint).
      **TASK 3 GATES, RUN 2026-09-25.** Full suite `./venv/bin/python -m pytest -q` →
      **44 failed, 2748 passed, 16 skipped, 2 errors in 189.71s** — BELOW this row's own corrected
      floor of "2786 passed, 8 failed, 16 skipped" (the `[bug]` row directly below). Investigated,
      not papered over: all 46 non-passing tests (44 failed + 2 errors) are the SAME
      `FileNotFoundError` from A5.7b's original-shard deletion (`4bc70b1`) — each traced to
      `tests/deepseek/parity.py:96` or an equivalent `needs_checkpoint` fixture reading a now-deleted
      `model-000NN-of-00048.safetensors`, dying before any engine code runs. **The `[bug]` row's
      count of 8 UNDERCOUNTS the true blast radius, which is 46**, spanning
      `test_deepseek_v4_attention.py` (21), `test_deepseek_v4_checkpoint.py` (1),
      `test_deepseek_v4_linear.py` (4), `test_deepseek_v4_model.py` (2), `test_deepseek_v4_moe.py`
      (8, the originally-tracked list), `test_deepseek_v4_parity.py` (8), and 2 setup ERRORs in
      `test_deepseek_v4_block.py`. None touch `moe_patch.py`'s injection logic or Tasks 1-2's
      allowlist. Needs a follow-up correction to the `[bug]` row's count; not fixed here — no
      checkpoint restore/re-split/delete, per this task's scope. **Both on-device MoE accumulator
      gates ARE green, unaffected by the above:** Qwen3-Next-80B-A3B int4 AWQ (fused non-EXL3
      forward), `tests/test_moe_accumulator_determinism.py` → **8 passed in 36.82s**; MiniMax-M2.7
      EXL3 (module-list forward, the one this change edits) → **8 passed in 73.40s**. No fused-EXL3
      arm exists (no model on disk since Qwen3-235B was retired 2026-09-20) and dense has no
      committed gate (`test_batched_dense_smoke.py` deleted 2026-09-03). Neither is claimed as
      coverage.
      **TASK 4, RUN 2026-09-25 — DeepSeek-V4-Flash emits its first token.** `plans/probes/a57b_first_token.py`
      launched as a detached `systemd-run --user` unit (`a57c-first-token`, `MemoryMax=29G`), log
      `/tmp/a57c-first-token.log`. Zero `Expert weight inject failed` warnings, zero
      `PackedInjectionError` (both greps: 0) — Tasks 1-2's fix holds on the real checkpoint. **Two
      NEW bugs found and fixed, both confined to `vaayu/vaayu_deepseek_v4.py`, zero blast radius on
      other models:** (1) `moe_patch`'s module-list forward (the same one MiniMax runs) always
      returns `(hidden_states, router_logits)`; DeepSeek's own reference decoder-layer code (loaded
      via `trust_remote_code`, never touched by Vaayu) does `x = self.ffn(x)` expecting a single
      tensor and feeds it straight into `hc_post`'s `x.unsqueeze(-2)` → `AttributeError: 'tuple'
      object has no attribute 'unsqueeze'`. Fixed in `VaayuDeepseekV4.__init__`, after
      `super().__init__()`: wraps each `_moe_blocks[i].moe_module.forward` to unwrap a tuple return
      down to its first element (tolerates the pinned-layer branch's plain-tensor return too).
      (2) `get_past_key_value_args` read `past_key_values.get_seq_length()` (defaults to
      `layer_idx=0`); `DeepseekV4Cache` gives every layer an INDEPENDENT `cumulative_length`, and the
      layer-major loop calls this hook once per decoder layer within one token step, so once layer 0
      ran and bumped its own count, layer 1's kwargs saw a "moved" cache and raised `RuntimeError:
      start_pos 0 disagrees with the cache's 5 tokens` on literally the first prefill, every time.
      Fixed by reading `get_seq_length(self.config.num_hidden_layers - 1)` (the last layer, the only
      index that stays at the pre-step value for every layer's kwargs within the step). Both fixes
      covered by new tests in `tests/test_deepseek_v4_engine.py`
      (`test_construction_unwraps_a_tuple_returning_moe_forward`,
      `test_construction_leaves_a_plain_tensor_returning_moe_forward_alone`,
      `test_get_past_key_value_args_reads_the_last_layer_not_the_default_first`); full file
      58 passed (was 55). **Result: prompt `"The capital of France is"` → decoded
      `' Paris. The capital of Spain is Madrid'` — coherent, factually correct continuation, not
      garbage.** Load 3.7s; 8 tokens in 31.8s = **0.252 tok/s** (eager attention re-reads the routed
      expert set several times per step, as expected — reach, not speed, per this row's own note
      above). `attn impl: eager`, `fp4_group_size: 32`. Log: `/tmp/a57c-first-token.log` (gitignored,
      not preserved — this row is the durable copy).
      **TASK 5, RUN 2026-09-25 — server smoke, §8(3).** `SERVER_CKPT=.../deepseek-ai--DeepSeek-V4-Flash-0731`
      (checkpoint ROOT, matching `a57b_first_token.py`'s `CKPT`, NOT `splitted_model/`), foreground
      `systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 env PYTHONPATH=$PWD SERVER_CKPT=…
      ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s`. **3 failed, 6 passed, 17 warnings
      in 8.47s; load 4.8s** (`[Vaayu-Server] loaded in 4.8s; derived max_seq_len=16766`). **PASSED:**
      `test_health` (status ready, max_seq_len=16766, decode_top_k present), `test_oversized_prompt_refused`,
      `test_prefix_reuse` (reported "disabled" — expected, same `DeepseekV4Cache` no-`to_cpu`/`to_device`/
      `resident_bytes` gap as row 28b), `test_env_json_is_written_and_clean`,
      `test_requests_jsonl_has_one_row_per_request`, `test_no_absolute_paths_leaked_into_the_bundle`.
      **FAILED:** `test_non_stream`, `test_stream`, `test_the_token_summary_reports_nvme_bytes_consistent_with_pinning`
      (the third is downstream of the first two — no completion ever ran, so no `gb_tok=` line existed
      to assert on). **Root cause, verbatim (`tests/test_server_smoke.py:131`/`:148`, `assert 400 == 200`):**
      `{"error":{"message":"this checkpoint's tokenizer ships no chat_template, so /v1/chat/completions
      cannot render messages for it. Use a chat/instruct checkpoint, or send an already-formatted single
      user message..."}}`. **NOT caused by A5.7c's change** — verified `tokenizer_config.json` carries no
      `chat_template` key, and the checkpoint's own `README.md` states it outright: "This release does
      not include a Jinja-format chat template. Instead, we provide a dedicated `encoding` folder..." —
      DeepSeek ships a custom Python encode/decode scheme instead of a Jinja template, and
      `vaayu/server/chat.py:render_prompt` (pre-existing, untouched by this branch) hard-requires
      `tokenizer.chat_template`. A checkpoint/server-default gap, not a `moe_patch.py` /
      `vaayu_deepseek_v4.py` defect — filed as its own `[bug]` row below, not scope creep into this one.
      **So §8(3) is NOT proven for this checkpoint through `/v1/chat/completions`, short prompt or long:**
      load, health, refusal-plumbing, and logging all work, but the actual completion path — the
      reachability proof this task exists to produce — never runs, because every request 400s at
      template-render time before length is even checked. That is EARLIER than the `prefill_chunk_size`
      `[bug]` two rows below (>512 tokens): this run never got far enough to reach it either way.
      **AND THERE IS NO SECOND ROUTE TO TRY: the server exposes exactly three endpoints —
      `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (`vaayu/server/app.py:534/578/592`).
      No raw `/v1/completions`, so no prompt-string path bypasses the template.** Confirmed by the
      controller, 2026-09-25. A third fix option therefore exists beside the two on that row: have the
      model subclass supply the template/renderer, which keeps the blast radius at one model. Full
      log: `/tmp/a57c-server-smoke.log` (gitignored, not preserved — this row is the durable copy).

      **THEN, still A5.7c (Task 6, 2026-09-25): row 28b step 5, the decode measurement.** `plans/probes/
      row26_decode.py` (untracked, `plans/` is gitignored) got a `"deepseek"` `MODELS` entry pointing at
      `/home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731`; `delete_original=False` was
      confirmed hardcoded before running, so the probe cannot consume the checkpoint it measures. **The
      probe's own `render_prompt` call also hit the no-`chat_template` gap above** (`NoChatTemplate`) —
      worked around IN THE PROBE ONLY, for the `deepseek` arm alone, with a raw `tokenizer(...)` encode
      of the same system+task text (no chat markup), exactly the pattern Task 4's `a57b_first_token.py`
      already used on this checkpoint. No `vaayu/` file touched. Prompt came to 127 tokens, under the
      512-token `prefill_chunk_size` default, so the `[bug]` two rows below did NOT fire and
      `prefill_chunk_size` was left at its default (NOT set to 0).
      **Arm: `MODEL=deepseek BALLAST_GB=15 NEW_TOK=64 ARMS=W,U1,P,U2,Usi,Uspin`** (systemd unit
      `a57c-row26`, `MemoryMax=29G`). **`BALLAST_GB` CHANGED FROM THE BRIEF'S 25 TO 15**: the first run
      at `BALLAST_GB=25` was OOM-killed by the cgroup (`status=9/KILL`, "killed by the OOM killer") after
      completing arm P, with `rss=29.75GB` logged one token before the kill — this model's RAM footprint
      under that squeeze does not fit a 29G ceiling. Re-run once at `BALLAST_GB=15` per the brief's
      standing rule, and it is labelled here as the changed arm, not a quiet substitution.
      **Result (the 15 GB-ballast run, wall ≈22m16s, 11:04:31→11:26:47):** all 6 arms completed, greedy
      ids identical across every arm (deterministic). Decode rate: **W 0.3335, U1 0.3267 (the TRUE
      unprofiled figure), P 0.3195, U2 0.3203, Usi 0.3186, Uspin 0.2537 tok/s** (Uspin is the
      SCHED_IDLE-spinner diagnostic arm, not the headline number). VRAM peak `max_allocated 7.15 GiB /
      max_reserved 7.27 GiB`, device used 7.92 of 15.47 GiB; VRAM expert-cache hit_rate 35% overall
      (usage 3.52/3.53 GB). Raw output: `plans/probes/row26_out/decode-deepseek.json` (gitignored).
      **Verdict against the ~1.75–2.06 tok/s prediction: REFUTED, not confirmed — measured ~0.32–0.33
      tok/s is 5.4–6.4× SLOWER than the prediction's low end**, not the ~5× (vs Qwen3-Next) the row
      called REACH-not-speed and accepted going in — this is slower even than that acceptance covered.
      **[S6 2026-09-26: the explanation in this paragraph is WRONG — see the S6 row. The ~2× bytes are
      non-expert weights of 35 streamed layers (5.496 GB/token), not expert re-reads, and the caches
      HALVED the expert stream. `gpu_load` is their inject — mostly CPU `dequantize_at_load(wo_a)` —
      not GPU work. The "2.7 of 3.9–4.0 s" figures are the Uspin arm; U1 is 2.20 of 3.06 s.]**
      **One-sentence explanation from the log's own per-token `[VAAYU-IO] token summary` lines:** bytes
      moved per token sit at **~7.0–7.8 GB, roughly 2× the 3.2124 GiB/token projection**, with the
      expert RAM cache pinned at **~20% hit_rate** (evict count tracking miss count 1:1 — a thrashing
      cache, not a warming one) under the `BALLAST_GB=15`-derived 9.13 GB budget, AND the breakdown is
      **GPU-load bound, not NVMe-stall bound** (`gpu_load` ~2.7 s of a ~3.9–4.0 s token vs `expert_disk`
      only ~0.2–0.3 s) — so the shortfall traces to roughly double the projected bytes plus a GPU-side
      (H2D/dequant) cost the NVMe-plateau projection never modelled, not to a disk-bandwidth shortfall.
      **This also means the row 28b projection's method (bytes/token ÷ delivered NVMe GB/s) does not
      hold for this model as measured** — a follow-up would need to explain the 2× byte inflation and
      the GPU-load dominance before re-projecting, and that is not built here (measurement only, per
      this task's scope).

      **A5.7c CLOSED 2026-09-25 — summary for a reader who only has this row.** Full suite:
      **2748 passed, 44 failed, 2 errors, 16 skipped, 189.71s** at the Task-3 gate, and
      **2756 passed, 44 failed, 2 errors, 16 skipped, 191.88s at the BRANCH TIP** (`2aea6dc`) — the
      +8 are Task 4's 3 subclass tests and the final-review fix wave's 5, and the 46 non-passing are
      the same set throughout; all 46 non-passing are the
      pre-existing shard-deletion breakage (its own `[bug]` row below), and 2748 + 46 = 2794 =
      the 2778 pre-deletion floor + A5.7c's 16 new tests, so nothing regressed. Both on-device MoE
      accumulator gates green, foreground at `MemoryMax=29G`: **Qwen3-Next-80B-A3B int4 AWQ (fused
      non-EXL3 forward) 8 passed / 36.82s; MiniMax-M2.7 EXL3 (module-list forward, the one this
      change edits) 8 passed / 73.40s.** First token: prompt `"The capital of France is"` →
      `" Paris. The capital of Spain is Madrid"`, coherent and factually correct; load 3.7s, 8 tokens
      in 31.8s; zero `Expert weight inject failed`, zero `PackedInjectionError` — the injection fix
      was clean on the first on-device run. Server smoke: **3 failed, 6 passed, 8.47s** — load,
      health, oversized-prompt refusal, prefix-cache reporting, `env.json`, `requests.jsonl` and the
      path-leak check all pass; the two completion tests and the token-summary check fail purely on
      the checkpoint's missing `chat_template` (its own `[bug]` row below), not on anything this
      branch touched. Decode (row 28b step 5): **U1 0.3267 tok/s**, arm `MODEL=deepseek
      BALLAST_GB=15 NEW_TOK=64` (changed from 25 after an OOM kill), 127-token prompt (chunking
      stayed enabled), wall 22m16s — the plan's ~1.75–2.06 tok/s prediction is **REFUTED**, 5.4–6.4×
      slower than its low end. **No fused-EXL3 arm exists** (no model on disk since Qwen3-235B was
      retired 2026-09-20) and **dense has no committed gate** (`test_batched_dense_smoke.py` deleted
      2026-09-03) — neither is claimed as coverage. **§8, final answer, all four:** (1) no knob — a
      correctness invariant, nothing to tune. (2) not applicable — no knob to plumb. (3) **NOT
      reachable through the server today** — the missing-`chat_template` `[bug]` row below is what
      unblocks it; library-reachable (Task 4's first token) and server-BANKED, stated in those terms,
      not claimed as shipped. (4) said out loud, right here, and as the three dedicated rows
      immediately below: the fused forward's twin cast, `_inject_expert_weights`'s warn-and-skip, and
      A4 check 2's finding (also closed below, in the A4 checks section).

- [!] **The fused forward's twin cast — BLOCKED 2026-09-25, decided out of scope by the user.**
      **[S6 2026-09-26: the blocker named below is STALE. gpt-oss has loaded and served since A3
      (2026-09-21), through a FOURTH forward, `_make_whole_layer_sparse_forward`, which never calls
      `_expert_dense_weights`. So the cast is still unreachable by gpt-oss, but the thing that would
      make it reachable is A6 (per-expert MXFP4 streaming), not G1.]**
      `_make_fused_sparse_forward`'s own expert injection casts streamed tensors to the running dtype
      exactly as `_inject_expert_weights` did before A5.7c — same defect class, a different forward,
      and a different FUNCTION: the fused cast lives in `_expert_dense_weights`
      (`moe_patch.py:625-627`, called at `:2327`), not in `_inject_expert_weights` at all — the two
      never shared code. gpt-oss-20b's MXFP4 experts (fused-3D `mlp.experts.gate_up_proj_blocks`, U8
      `[32, 5760, 90, 16]`) would hit it. **Not fixed, and here is why:** gpt-oss cannot load at all
      today (row 47 / gap G1 — the expert-offset index matches 0 of its 459 tensors and does not
      raise), so a fix on this forward would be untestable code sitting on `_make_fused_sparse_forward`
      — Qwen3-Next's perf-tuned, most-tested path (CLAUDE.md §0). **Unblocked by G1**: once gpt-oss
      loads, this becomes testable; until then, leave this forward alone for this reason.

- [ ] **`_inject_expert_weights`'s warn-and-skip stays a silent drop, deliberately, out of A5.7c's
      scope.** Every non-packed injection failure still logs a warning and `return None`s, silently
      dropping that expert from the token — unchanged, verbatim, from before A5.7c. A5.7c added
      `PackedInjectionError` only for its own new no-cast branch (`NO_CAST_INJECTION_DTYPES`),
      deliberately leaving this shared path alone. **Corrected 2026-09-25 (final-review M3): this
      does NOT touch both forwards.** `_inject_expert_weights` has exactly ONE call site
      (`moe_patch.py:1858`), inside `_make_modulelist_sparse_forward` — the module-list forward
      (MiniMax-M2.7, production). The fused forward (Qwen3-Next, production) never calls it; its own
      cast is `_expert_dense_weights` (`moe_patch.py:625-627`), untouched by A5.7c. So Qwen3-Next's
      green accumulator gate is a non-regression result on a forward this branch does not touch, not
      both-paths coverage of it — the change is narrower and safer than this row previously claimed.
      **Why it is not a quick follow-up regardless:** turning `warn` + `return None` into a raise on
      the module-list forward still needs that on-device MoE accumulator gate green AND a deliberate
      decision about raising mid-decode (today a dropped expert degrades output silently; a raise
      would kill the request instead). Not started.

- [x] **CLOSED 2026-09-26 by S6 — METHOD CORRECTED; see the S6 row at the top. THIS ROW'S "why" IS
      WRONG AND MUST NOT BE RE-COPIED:** the ~2× bytes are **non-expert weights** (35 of 43 layers were
      outside the 3.33 GiB pin budget: 5.496 GB/token read from the drive), not expert re-reads — the
      caches HALVED the expert stream to 1.786 GB. The ~20 % is the RAM tier's hit rate among VRAM
      misses (combined 48 %; row 28's 54 % miss assumption held at 51.8 %). `gpu_load` is the inject
      of those streamed layers, 65 % of it CPU `dequantize_at_load(wo_a)`, not GPU work. And "3.9–4.0 s"
      is the Uspin arm; U1 is 3.06 s. Original text:
      **Row 28's reach-projection METHOD does not hold on DeepSeek-V4-Flash, measured 2026-09-25
      (A5.7c Task 6) — a correction to the METHOD, not just this one model's number.** Row 28
      projected decode from `bytes/token ÷ delivered NVMe GB/s` = **3.2124 GiB/token** for this
      checkpoint, giving the ~1.75–2.06 tok/s prediction row 28b step 5 REFUTED (measured **U1
      0.3267 tok/s**, 5.4–6.4× slower). **Measured breakdown of why, from the per-token `[VAAYU-IO]
      token summary` lines:** actual bytes moved per token are **~7.0–7.8 GB, roughly 2× the
      3.2124 GiB/token projection**; the expert RAM cache is **thrashing, not warming** (~20% hit
      rate, evict count tracking miss count 1:1, under a 9.13 GiB budget at `BALLAST_GB=15`); and the
      per-token cost is **GPU-load bound (~2.7s of a ~3.9–4.0s token), NOT NVMe-stall bound (~0.2–0.3s,
      `expert_disk`)**. **What the method is missing:** it has no term for GPU-side H2D-plus-dequant
      work per token — it treats the NVMe transfer as the whole cost, so the delivered-bandwidth
      denominator is not the binding constraint here. **Why this matters beyond one model:** row 28's
      method is the one every future "can this checkpoint reach the hardware?" projection uses; as
      measured, it under-predicts cost by 5–6×. **Not built here** — explaining the 2× byte inflation
      and adding a GPU-load term to the method is follow-up work, out of A5.7c's scope (measurement
      only). Source: `plans/probes/row26_decode.py` output,
      `plans/probes/row26_out/decode-deepseek.json` (gitignored — this row is the durable copy).

- [ ] **Four deferred minors from A5.7c's final whole-branch review (2026-09-25). Recorded here
      because `docs/` is gitignored and the review document did not survive the workspace cleanup.**
      **(M1)** `NO_CAST_INJECTION_DTYPES` (`moe_patch.py`) is OPT-IN, so the next packed dtype silently
      re-enters the cast path — gpt-oss's MXFP4 is already on disk. Consider refusing a target dtype
      that has neither `finfo` nor `iinfo` instead of allowlisting one at a time.
      **(M2)** a `PackedInjectionError` escaping `_load_inject_run` skips `_arena.record_read` and the
      `_chunk_major_gpu`/`_chunk_major_gpu_bytes` reset; the latter SURVIVES the abort and shrinks every
      later retention budget in that process. Pre-existing for ANY exception escaping that function, and
      it cannot tear a live tensor — but it is a real leak on the abort path.
      **(M3)** `vaayu_deepseek_v4.py`'s `_wrapped` is keyword-only, hence caller-reachable.
      **(M4)** no test drives a `PackedInjectionError` through the REAL forward to assert propagation,
      the expert landing back on meta, and the pin being released — the unit tests call the injector
      directly. **All four were triaged as correctly deferred by the reviewer; none blocks anything.**

- [x] **FIXED 2026-09-25 — DeepSeek-V4 prefills prompts longer than `prefill_chunk_size`.**
      Fix option (1) from this row, the subclass-only one: the contiguity check moved out of
      `get_pos_emb_args` into `get_attention_mask_args`, the ONLY hook the layer-major loop tells
      which layer it is about to run (`vaayu_base.py:5241`, `layer=layer`), and it now reads **that
      layer's** `get_seq_length(layer_idx)` instead of any fixed index. `get_past_key_value_args`
      stops reading a length at all and just stashes the cache; `layer=None` skips the check rather
      than guessing an index. `DeepseekV4DecoderLayer.layer_idx` supplies the index
      (`modeling_deepseek_v4.py:1364`). **Subclass-only — no shared file touched, so blast radius is
      one model.**
      **ON-DEVICE GATE, 2026-09-25** (`plans/probes/a57d_chunked_prefill.py`, gitignored; detached
      unit, `MemoryMax=29G`, log `/tmp/a57d-chunked-prefill.log`): load 3.5 s,
      `prefill_chunk_size=512`, `attn=eager`. **SHORT arm 53 tokens / 1 chunk → OK in 25.0 s
      (0.160 tok/s), `'2.5x'`. LONG arm 729 tokens / 2 chunks → OK in 44.8 s (0.089 tok/s),
      `'</think>Vaayu streams'`.** Exit 0, zero tracebacks, zero contiguity raises; the only warning
      is the known prefix-cache warn-and-disable. The LONG arm is the case that could not run at all.
      **Tests:** 6 in `tests/test_deepseek_v4_engine.py`, each with a mirrored negative so none can
      pass with the check absent — a pre-write layer must REJECT a post-write layer's legitimate
      `start_pos` and vice versa, and a layer midway through chunked prefill validates against its
      own partial count. 309 passed across the engine/cache/config/moe-patch files. **The full suite
      was NOT re-run after this change** (the user declined the 3-minute run); the change touches one
      subclass file only.
      **Why `len_p` itself is sound, checked while fixing:** `vaayu_base.py:4844`'s
      `_past_len_for_pos` uses the base's `get_past_key_values_cache_seq_len`, which does default to
      layer 0 — but it is read ONCE per `forward` entry, when every layer's count is equal, and
      chunk-major derives `_past_lens` from that single `past0` (`compute_chunk_past_lens`,
      `:1044`). So the layer-0 default is correct there and needs no override.
      (Original row follows.) **FOUND 2026-09-25 by A5.7c's Task 4 review; NOT a regression — the old
      code had the same defect at a different index.** `VaayuDeepseekV4.get_past_key_value_args`
      derives the cache's past length from ONE fixed layer index, but `DeepseekV4Cache` gives every
      layer an independent `cumulative_length`, and the hook is called once per **(layer, chunk)**
      pair (`vaayu_base.py:5222-5241`, `_past_lens` from `compute_chunk_past_lens` at `:1044`).
      **No fixed index can be right:** for the layer currently executing chunk *j*, the correct past
      is `past0 + sum(len(chunks) < j)`; layers already done read `past0 + total`, layers not yet
      started read `past0`. So `max()` and `min()` over layers are both wrong too. Before A5.7c the
      hook read `get_seq_length()` (layer 0 by default) and was self-consistent only at layer 0 by
      coincidence; `05a9cf6` moved it to the LAST layer, which fixes the single-chunk case it was
      raising on (decode and prefill ≤ 512 — that is what unblocked the first token) and leaves the
      multi-chunk case broken: **layer 0 raises `RuntimeError` on its SECOND chunk**, before
      execution ever reaches the "safe" last layer.
      **Reachability, measured:** `tests/test_server_smoke.py`'s real prompt is ~16 tokens
      (`PROMPT`, `:34`) and its oversized prompt is refused with 400 before any forward (`:186`), so
      the §8(3) smoke does NOT cover this. Every prompt over 512 tokens does — i.e. **the normal case
      for the agentic workload Vaayu exists to serve.**
      **THREE FIX OPTIONS, in preference order:** (1) **subclass-only, and the pattern already
      exists** — `05a9cf6` wraps each patched `moe_module.forward`; wrap each decoder layer's
      `forward` the same way to record the executing layer index, and have the hook read THAT
      layer's `cumulative_length`. Blast radius stays zero for other models. (2) pass the layer into
      `get_past_key_value_args` — correct but a shared-infra signature change touching every model's
      hook, so it carries the both-paths gate burden. (3) move the contiguity check into
      `DeepseekV4Cache` itself, which is the only object that knows its own per-layer counters.
      **STOPGAP available today:** `prefill_chunk_size=0` disables chunking, at the cost of a
      single-shot prefill's VRAM on a 4096-hidden, 43-layer model — untested for this checkpoint.
      **Why A5.7c did not fix it:** its stop rule is that it must not grow into shared
      infrastructure, and options (2) and (3) are exactly that; option (1) is new design with no
      chunked-prefill test coverage to land it against. Parked with this row, not silently.

- [bug] **(SAME ISSUE AS ROW 48 — sequenced as S10.) `use_torch_compile=True` CRASHES on Qwen3-Next, and `vaayu/server/config.py:328`'s stated
      reason for declining the knob is FACTUALLY WRONG. Found 2026-09-20 by row 36's re-scope A/B
      (`docs/2026-09-20-row36-torchcompile-AB-RESULTS.md`); recorded as its own row 2026-09-25 because
      it was buried in a parked row's document.**
      **The crash:** with the shipped `mode="reduce-overhead"` (torch.compile **plus** CUDA graphs,
      `vaayu_base.py:5064`), `effective=True` at load and the run dies with
      `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a
      subsequent run`, inside transformers' `modeling_qwen3_next.py` resume block. **Reproduced twice.**
      **The wrong reason:** `config.py:328` declines the knob saying *"`__init__` overrides these in
      places; they are not live paths"* — but the only `__init__` guard (`vaayu_base.py:2118`) disables
      it for NON-CUDA devices and leaves it ON for CUDA, so it IS a live path. **The right reason to
      decline it is that it crashes** — fix the comment, whatever else happens.
      **Also measured:** with the CUDA-graph half removed (`mode="default"`) it is CORRECT but
      **24–35 % SLOWER** on Qwen3-Next, so there is no win hiding behind the crash.
      **Scope:** a comment fix plus either a named refusal at load or a real fix of the graph
      interaction. The knob defaults `False`, so nothing is broken for users today — this is a trap
      waiting for whoever flips it, plus a false statement in the config that would mislead them.

- [x] **DeepSeek-V4-Flash-0731 ships no `chat_template`, so `/v1/chat/completions` 400s on every
      request. FOUND 2026-09-25 by A5.7c Task 5's server smoke; NOT caused by A5.7c. FIXED 2026-09-26 by
      S5 without writing into the model dir: the model class hands the server DeepSeek's own
      `encoding/encoding_dsv4.py` as a prompt renderer (see the S5 row).**
      `tokenizer_config.json` has no `chat_template` key, and the checkpoint's own `README.md` says so
      by design: "This release does not include a Jinja-format chat template. Instead, we provide a
      dedicated `encoding` folder with Python scripts... to encode messages in OpenAI-compatible format
      into input strings for the model." `vaayu/server/chat.py:render_prompt` (pre-existing, unedited
      by A5.7c) hard-requires `tokenizer.chat_template` and raises `NoChatTemplate` → 400 otherwise; this
      is a correct, working guard, not a defect in it. **Effect on §8(3):** `tests/test_server_smoke.py`'s
      `test_non_stream` and `test_stream` both fail (`assert 400 == 200`), and
      `test_the_token_summary_reports_nvme_bytes_consistent_with_pinning` fails downstream of them (no
      completion ever ran, so no `gb_tok=` line to assert on) — 3 of 9 checks, see the A5.7c row's Task 5
      entry for the full pass/fail list. Every other server-smoke check (load, health, oversized-prompt
      refusal, prefix-cache reporting, env.json, requests.jsonl, path-leak check) passes on this
      checkpoint. **Not fixed here** — no production code touched, per A5.7c Task 5's scope (verification
      only). **What unblocks it:** either (1) build a Jinja `chat_template` from the checkpoint's own
      `encoding/` Python scripts and ship it in `tokenizer_config.json` (checkpoint-side, not a Vaayu
      code change — and a write into the user's model directory, so it is their call, not an agent's),
      or (2) teach `vaayu/server/chat.py` a non-Jinja rendering path that calls the
      checkpoint's `encoding/` module directly — a real server feature, not a one-line fix, and its own
      row if picked up.

- [x] **A5.7b's original-shard deletion broke 46 committed DeepSeek-V4 tests across 6 files
      (44 failed + 2 errors). FOUND 2026-09-25 while gating A5.7c; NOT caused by A5.7c.
      FIXED 2026-09-26 by S4 — all 46 green, suite `2928 passed, 0 failed, 0 errors, 16 skipped`;
      one resolver in `tests/deepseek/parity.py` reads the originals while they exist, else the split
      via `remap_checkpoint_key`. Seven reader sites, not one — see the S4 row.** (Title said
      "8 parity tests" until 2026-09-25 — that was the scoped first look at one file; the full suite
      put it at 46. Per file: attention 21, parity 8, moe 8, linear 4, model 2, block 2 errors,
      checkpoint 1. Re-confirmed unchanged by the 2026-09-25 post-fix suite: same 46, same files.) `tests/deepseek/parity.py:96` reads the
      goldens straight out of the ORIGINAL shards via `model.safetensors.index.json`'s `weight_map`,
      and `4bc70b1` deleted all 48 originals after a byte-for-byte verify. Every affected test now
      dies at FIXTURE time with
      `FileNotFoundError: .../deepseek-ai--DeepSeek-V4-Flash-0731/model-00005-of-00048.safetensors`,
      before any engine code runs. **BLAST RADIUS CORRECTED 2026-09-25 after the full suite ran:
      it is 46 tests across 6 files, not 8 — 44 failed + 2 errors.** Per file: `attention` 21,
      `parity` 8, `moe` 8, `linear` 4, `model` 2, `block` 2 errors, `checkpoint` 1. Every one of the
      46 was confirmed by hand to read an ORIGINAL shard (`model-00002/00004/00005/00045-of-00048`)
      and to die before engine code runs; none touches `moe_patch.py`. Arithmetic closes: 2748
      passed + 46 = 2794 = the 2778 pre-deletion floor + A5.7c's 16 new tests, so nothing regressed.
      **`model-00046/00047/00048` are NOT missing — they were MOVED to
      `models/deepseek-v4-mtp-shards/` with an `mtp_weight_map.json`; only 00001-00045 are gone.**
      **The 8 in `test_deepseek_v4_moe.py`:** `test_the_gate_goldens_reproduce_through_the_split_gate`
      [gate_score, gate_hash], `test_the_expert_golden_reproduces_with_the_weight_inside`,
      `test_the_moe_goldens_reproduce_with_the_weight_applied_outside` [moe_score, moe_hash],
      `test_the_moe_is_bit_identical_to_the_reference_when_the_weight_goes_inside` [moe_score,
      moe_hash], `test_the_shared_expert_helper_runs_on_the_real_module_tree` — all in
      `tests/test_deepseek_v4_moe.py`, 8 failed / 16 passed in 4.91 s.
      **CONSEQUENCE, and it is why this row exists rather than a note: the suite floor of "2778
      passed, 16 skipped" is STALE for every row after `4bc70b1`.** A5.7c gates against **2786
      passed, 8 failed, 16 skipped** with these 8 named, so a NEW failure is still caught.
      **Superseded by the measured suite, same day: A5.7c's gate is `2748 passed, 44 failed, 2
      errors, 16 skipped` in 189.71 s, all 46 non-passing in the class above.**
      **Fix direction:** point `tests/deepseek/parity.py` at `splitted_model/` and apply the same
      disk-key → module-tree remap the loader uses (`cda7f3b`/`e553137`), or commit a small golden
      fixture so the parity tests stop needing 157 G of checkpoint at all. **Real coverage loss
      meanwhile:** DeepSeek-V4's gate/expert/MoE bit-identity goldens do not run.

**Two checks CARRIED OUT OF A4 (row 47, closed 2026-09-23 with its premise refuted) — they belong to
A5. BOTH CLOSED 2026-09-25, on the A5.7b/A5.7c on-device load:**
- [x] **check 1: DeepSeek's shared experts DO land in the pinned non-expert set — discharged by
      inspection AND NOW BY A LIVE LOAD, not inspection alone.** A4's key-shape argument
      (`\.experts\.(\d+)\.` does not match `...shared_expert.*`, so shared experts are pinned, not
      streamed) held on paper; A5.7c's Task 4 first-token run then loaded the real checkpoint with
      the shared experts inside the pinned non-expert set — the same load check 2's pin summary
      below is measured from.
- [x] **check 2 — CLOSED AS A FINDING, not a pending check: the pinning budget does NOT cover
      DeepSeek's non-expert set.** Measured on the A5.7b/A5.7c load: pin summary `3 fully-pinned
      (0 transformer + 3 shared), 0 non-expert-pinned transformer` against `pinning budget=3.34 GiB`
      and an **8.24 GiB dense footprint** — **not one transformer layer was pinned**, stronger than
      the design's predicted "budget exhausted partway" (which expected some layers to fit before the
      budget ran out). **Why it matters: the pinning is INCIDENTAL to the budget, not structural.**
      Phase 2 `break`s on budget miss (`vaayu_base.py:3956`/`:3977`); Qwen3-Next's ~3.55 GB fits
      inside 4.83 GB today, DeepSeek's does not. Un-pinned layers fall back to
      `LRUTensorCache(_ne_budget)` (`vaayu_base.py:2710`, byte-budgeted, derived) — RAM-cached
      re-reads for the WHOLE transformer stack, **not** per-token NVMe streaming for it.
      **CORRECTED 2026-09-26 (S6): the last sentence is WRONG as measured.** `_ne_budget` was **0.5 GB**
      against a 5.5 GB per-token cyclic scan, so the LRU evicts every layer before it comes round and
      NEVER hits. The row-28b step-5 run read **5.496 GB/token of non-expert weights from the drive**
      (8 layers pinned, 35 streamed), and their inject was **72 % of the token**. This finding was the
      residency evidence the projection needed, and it went unused.

**Steps (each with its own verify):** 1 **kernels first, standalone, on sm_120** (2–5 days; **stop rule: if TileLang
will not build or validate, decide Triton-or-abandon HERE**, before any modeling code) → 2 HF modeling code, MP
machinery stripped, **experts as `nn.ModuleList`** so it lands on the module-list forward MiniMax already exercises
(1–2 weeks; verify logits against `inference/generate.py` BEFORE measuring) → 3 key remap + persist (verify
`ExpertOffsetIndex` finds **43 × 256 = 11,008** derived entries) → 4 first on-device generation → **5 DONE
2026-09-25 (A5.7c Task 6) — see the A5.7c bullet above for the figure and the caveats; run
`row26_decode.py`, compare against row 28's prediction (serial 0.035–0.070 s, device-busy 0.305 s, T 0.439–0.474 s)** **[S6 2026-09-26: this projection is REFUTED — measured 0.3267 tok/s. It omitted the non-expert weights that do not fit the VRAM pin budget (35 of 43 layers streamed, 72 % of the token). Use the S6 method, not this figure.]**
→ 6 *optional* the **10.12 GiB MTP draft module** already in the checkpoint (free speculative drafter; dense spec
measured 5.57× in **row 12** — NOT "row P7", which is a different, REFUTED item; see §0's warning
and this file's `P7`/`P8`/`P9` note — but **the MoE mechanism differs — do not assume it
transfers**). **Why it differs, so nobody re-derives it:** dense spec wins because verifying k
drafted tokens costs ONE pass over the weights, the same bytes as a single decode step. MoE decode
reads only the ROUTED experts (here top-6 of 256, 3.2124 GiB/token), so verifying k tokens needs the
UNION of the experts those k tokens route to; if routing locality is weak that union grows ~linearly
with k and the win vanishes. **Three measurements on this box already point that way:**
`VAAYU_LOOKAHEAD` costs **+18.0 % bytes/token** (token floor capped near +11 %); the P-series `P7`
slot — sizing the batch VRAM prefix on the observed routing union — was built and **REFUTED at
−10.85 / −11.54 %**; and Plan F's top_n sweep found expert share **linear in N**, 2–4× below the
cost line, which is the signature of the weak locality MTP would need to be strong.
**ALSO BLOCKED TODAY, independent of any of that:** A5.7b refuses `drafter_model_path` for this
model, because rejecting a draft rewinds the KV cache through `_trim_kv_cache` ->
`DeepseekV4Cache.crop`, which cannot rewind a preallocated sliding window plus its compressed half.
MTP needs that built first — it is a project, not a flag.
**The weights survive:** the 3 pure-MTP shards were moved to
`/home/a/code/ai-ml/models/deepseek-v4-mtp-shards/` (10.1 GiB) on 2026-09-25 rather than deleted, so
this step is still POSSIBLE — but it is an unproven option, not a banked win.

**Good news that is not obvious:** its disk keys are `layers.3.ffn.experts.7.w1.weight` — they already match
`ExpertOffsetIndex`'s only rule (`\.experts\.(\d+)\.`), so streaming works once the model loads. **The hard part is
the model, not the streaming.** Watch: dense footprint **8.24 GiB** (vs Qwen's ~5.8) must be VRAM-resident or the
per-token H2D (~0.31 s) swamps everything.

**Check before starting:** a newer transformers may have added `deepseek_v4`; that deletes steps 1–2.

**§8:** (1) no new knob — the model path is already a parameter; state it in the commit. (2) server must be able to
load it — grep `app.py` / `engine.py` for the `trust_remote_code` path. (3) reachable — per-token N=1, the server's
only mode. (4) not built; step 1 unblocks step 2, step 4 unblocks the measurement.

---

# ▶ ROWS 31-39 — added 2026-09-12 from the "what else can speed Vaayu up" sweep

**Durable copy of the conclusions is these rows.** Detail, arithmetic and the physics primer:
[`docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md)
(**gitignored — it will not survive a fresh clone; these rows must stand alone**).

**The primer in one table — every row below is sized against it.** Measured, in-process, greedy ids identical across
arms (`docs/2026-09-11-row26-cpu-hybrid-ceiling-RESULTS.md` §5.3; `serial` from the `NOEXP` stubbed-expert arm):

| term of one decode token | Qwen3-235B (keep-awake) | MiniMax-M2.7 (keep-awake) |
|---|---|---|
| `serial` non-expert | 0.074 s (8 %) | 0.052 s (9 %) |
| **`device-busy` (the NVMe)** | **0.772 s (83 %)** | **0.396 s (70 %)** |
| `exposed` host-side | 0.085 s (9 %) | 0.114 s (20 %) |
| **T** | **0.930 s** | **0.562 s** |

`device_busy ≈ bytes_per_token / delivered_GB_s`, and `bytes = moe_layers × top_k × expert_bytes × (1 − hit_rate)`,
`expert_bytes ∝ bpw`. Qwen: 4.84 GB/token at 6.3 GB/s delivered; MiniMax 2.38 GB at 6.0. Drive plateau is **7.1 GB/s
at 2 concurrent readers and falls to 5.15 at 8** (row P8). **Project a lever as
`T_new = serial + bytes_new/delivered_new + exposed`, then measure it.**

**Consequence, and it is the reason these rows exist:** anything touching only `exposed` is capped at **+9 % Qwen /
+20 % MiniMax** before it starts — which is independently why row 26 put the CPU-AVX2 hybrid's realistic ceiling at
+6.8 % / +22.0 %, and why more Rust is dead (SETTLED: GIL 0.11–0.72 %, GC 0.26–0.41 % of decode). **Everything that
can pay more cuts bytes per token or raises delivered GB/s.**

---

# 32 · → **MOVED to `todo-backlog.md` 2026-09-26** — run a lower-bpw EXL3 build of the models on disk. Why: BLOCKED 2026-09-23: the user declined the ~81.6 GiB download that is the row's entire cost. Re-open when: the user authorises the download.

---

# 35 · → **MOVED to `todo-backlog.md` 2026-09-26** — a per-model measured gate for `VAAYU_LOOKAHEAD`. Why: DEAD 2026-09-20: its prize model (Qwen3-235B) was deleted and its mechanism died when S2 landed. Re-open when: a served model where lookahead measures ≥ +15 % at N=1.

---

# 36 · → **MOVED to `todo-backlog.md` 2026-09-26** — CUDA graphs / fused per-layer dispatch. Why: PARKED 2026-09-20: every lever killed — `use_torch_compile` crashes (row 48 / S10), 48 blocking D2H routing syncs per token, and compile without cudagraphs measured 24–35 % SLOWER. Re-open when: new evidence that a graph-capturable dispatch exists — S7 (Qwen3-Next per-expert dispatch) may produce it.

---

# 37 · [ ] Re-gate speculative decoding on MoE — its refusal is a BYTE bound — CONDITIONAL (added 2026-09-12)

**2026-09-19: CLOSED FOR HYBRIDS by row 42 (b).** On Qwen3-Next a cold verify of w = k+1 tokens costs 2.6–4.8× a
decode token (bytes grow with w on offloaded MoE), prompt lookup loses on real edits and novel text, and any drafter
(MTP included) needs per-token acceptance ≥ 0.87 at k=5 for +10 %. Re-open for hybrids only on a measured MTP
acceptance ≥ 0.87 or cold verify ≤ ~2× t_dec. The row stays open for row 32's lower-bpw EXL3 route.

**2026-09-18: the gate below is MET — row 30's S2 landed 2026-09-13.** On Qwen3-Next, the only model it opened, speculation first needs row **41** (a rejected draft may corrupt the recurrent state) and row **42** (a state snapshot/restore to make rejection correct). Start there, not here.

The gate refuses Qwen today with `off_union`: **observed verify union 19.31 GB against a ~17 GB budget, 0 drafting
rounds.** That is a **byte** bound, and every byte lever in this block shrinks it — at 3.0 bpw Qwen's union falls to
**~11.6 GB** and lands inside the budget. **The break-even acceptance falls too:** with experts resident, drafting extra
positions costs ~0 extra disk bytes, which is the dominant term in today's break-even.

**The machinery is already built** — `vaayu/speculative/adaptive_gate.py` sizes the verify pass from `_expert_history`'s
observed per-layer union, `_build_spec_gate()` (`vaayu_base.py:4794`) wires it MoE-only and caches it across requests,
row 13's `even-4` pooled paired trial decides, and P9's verify-prefix sizing added +4.16 % on top of MiniMax's 1.307×.
**Do not re-derive or rebuild union-based sizing; it exists.**

**Free drafter, and it decides the model choice:** MTP / multi-token-prediction modules ship **inside** several
candidate checkpoints (DeepSeek-V4-Flash carries a 10.12 GiB one — row 28b §3 step 6; **Qwen3-Next ships one too, and
the on-disk AWQ checkpoint KEEPS it: 4,625 `mtp.*` keys in `model.safetensors.index.json`, an MTP layer with its own
experts — verified 2026-09-12**; check the quantizer's `ignore` list in `config.json` to see whether it was left
unquantized). A drafter that arrives in the checkpoint has no vocab-mismatch risk and no second load. **When picking a
model, prefer one with an MTP module.**

**Gate: row 30's S2 or row 32 has landed. Stop rule: the gate still refuses on the resident model → close it.**

**SETTLED trap:** the **latching** and **regime-detection** questions stay REFUTED — row 24 (re-evaluating a latched
decision is worth ≤ +6.28 % against a ~29.3 re-entry price) and the P7 regime-latch row (the code regime does not
spontaneously arrive: 900 Qwen tokens, **zero** fenced code blocks; the latch was correct; the excursions are prose
self-repetition and 3.2× too small). **Nothing here re-opens either.** What changes is the **byte premise underneath the
refusal**, which neither row touched.

**§8:** (1) knobs exist (`USE_PROMPT_LOOKUP`, `DRAFT_TOKENS`, `DRAFTER_MODEL_PATH`). (2) plumbed. (3) reachable at N=1.
(4) blocked on the gate above.

---

# 38 · [ ] Mixed-bpw by expert hotness — do the ARITHMETIC, not the repack — CONDITIONAL (added 2026-09-12)

Store hot experts at high bpw and the long cold tail at 2–3 bpw. `vaayu/expert_usage.py` already records routing and
eviction (`ENABLE_EXPERT_USAGE`, off by default, **cost unmeasured** — it puts `track_evicted_keys` on a path decode
hits every layer visit). Plan D established hot-but-evicted experts genuinely exist.

**Why this is not the refuted idea it resembles.** Plan F refuted **pinning** by hotness (no viable pin N on either
model; share linear in N, 2–4× below the cost line) and Plan B Phase 2 refuted an adaptive re-pin directly (−0.082 to
+0.011 hit-rate points against a clairvoyant bound of +0.004 to +0.330). **Both are about RESIDENCY — which experts sit
in a cache. This is about PRECISION:** a cold expert at 2.5 bpw is cheaper *every time it is read from disk*, cached or
not. **Plan F's "share is linear in N" is in fact the argument FOR this shape** — if the top N hold only a proportional
share, the tail holds most of the bytes, and the tail is what gets cheapened.

**Gate: from an `expert_usage` trace compute `Σ reads_i × bpw_i` under a candidate split vs flat bpw. Under a 20 % byte
reduction → row 32 beats it at a fraction of the effort. Close it.** That is the likely outcome. **Start with the
arithmetic, not the repack.**

**§8:** (1) a knob for the split point. (2)/(3) reachable at N=1. (4) blocked on the arithmetic.

---

# 39 · → **MOVED to `todo-backlog.md` 2026-09-26** — expert slimming. Why: research tier with no build path; its own gate was "rows 31–35 measured". Re-open when: the user wants research work, or a calibration/re-pack path exists.

---

# 49 · [~] MiniMax-M2.7's expert H2D is SYNCHRONOUS — price the GPU-side attack row 26 ordered, THEN decide the CPU hybrid (added 2026-09-20, re-opened out of row 26 at the user's request) — **STEPS 0, 1, 2A AND THE B RE-RANK DONE: H1 TRUE on decode, `h2d` was 31.5 %, async upload SHIPPED at +6.90 % decode (2026-09-20); B REFUTED 2026-09-21 (<= +6 % against a 76 %-busy drive) — ring pool 3 -> 2 SHIPPED 2026-09-21 (+1.89 % decode / +1.85 % 2k prefill). READ-SHAPE arm BUILT AND REVERTED 2026-09-21 on its stop rule: reads/token -66.7 % and delivered rate +6.9 % bought -0.40 % decode, which REFUTES the rate lever by direct experiment and puts the idle lever's >= +31 % in doubt. Left: the drive-idle gap and step 3 (the CPU-hybrid decision).**

**Brief:** [`…-row49-minimax-h2d-BRIEF.md`](docs/2026-09-20-row49-minimax-h2d-BRIEF.md) (gitignored;
this row is the durable copy).

**Why this is legitimate to re-open: row 26 did NOT refute the hybrid, it DEFERRED it behind two
preconditions.** Verbatim: *"the bytes-neutral hybrid … is **not killed** … it **earns a design
brief, not code** — but only AFTER row 27 … and a cheaper GPU-side attack on the same slice … must be
priced **first**."* Row 27 closed 2026-09-12 (precondition 1 ✅). **The GPU-side pricing was never
done (precondition 2 ❌) — this row is it.** Row 28's later *"close the CPU-hybrid idea"* answered a
different question (whether the non-EXL3 DeepSeek-V4 checkpoint *rescued* it; it did not) and never
re-refuted MiniMax's +22.0 %.

**THE TARGET — and it is bigger than the hybrid's whole prize.** The module-list forward copies
expert bytes with **`non_blocking=False`, deliberately** (`vaayu/moe_patch.py:1110` and `:1119` — row
26 cited `:694`, the code has moved; the comment says async there leaks host RAM). So MiniMax's H2D
**cannot hide behind the disk read the way the fused forward's does.** Row 26's own profiled split:
**`h2d` = 0.180 s of a 0.596 s token — 30.2 %, the largest removable phase on this model** (disk 0.091
/ 15.3 %, gemm 0.153 / 25.7 %, construct 0.008 / 1.3 %). Cross-check: 2.38 GB/token at the measured
19.7 GB/s pinned H2D = 0.121 s. Row 45 independently measured h2d at **59.8 % of the wall** in
MiniMax *prefill*, same forward, same cause. **The CPU hybrid's entire prize was 0.114 s
(0.562 → 0.448 s floor) — smaller than this slice — and this one needs no CPU kernel, no new on-disk
format, no AVX2.**

**H1, the lead hypothesis, verify in code FIRST:** the `non_blocking=False` comment's reason is that
the tensors are *"not guaranteed page-locked"* — **that may be stale.** Row **30-P** (2026-09-18)
shipped a pinned RAM tier and recorded *"EXL3 models (Qwen3-235B, MiniMax) are unchanged — they
already had it."* If those bytes already arrive pinned, the safety reason is gone and the fix is far
smaller than "an explicit buffer-lifetime mechanism".

**Plan:** step 0 code read (~1 h, no GPU) answer H1 · step 1 **re-measure the MiniMax decode split on
the current build** (~1 h GPU, `row26_decode.py MODEL=minimax ARMS=W,U1,P,U2,NOEXP`, 2 replicates) —
nothing has re-split this model's decode since rows 27/30-P/44/45 · step 2 the overlap fix, 1–3 days
depending on H1, **correctness gate = prompt-end logits BIT-IDENTICAL + greedy ids identical + a
host-RAM watermark** (the original comment's fear was a leak, so the test must be able to see one) ·
step 3 re-price the CPU hybrid only then.

**STATUS 2026-09-23 — steps 0, 1, 2A and the B re-rank are DONE and their `[x]` bullets were pruned;
this is what a live bullet below refers to** (full text: `git log -p -- todo.md`). **Step 0:** H1 is
**TRUE on decode**, FALSE on prefill (pageable there). **Step 1:** `h2d` measured **31.5 %** of
`T_prof` — the ~15 % stop rule did NOT fire. **Step 2A:** async (`non_blocking=True`) expert upload
**SHIPPED 2026-09-20, +6.90 % decode**, logits bit-identical, no host-RAM growth. **B re-ranked and
REFUTED 2026-09-21** (≤ +6 % against a drive already busy 75.7–76.4 % of a MiniMax token at
6.03–6.05 GB/s, idling 1.57 ms per layer). **Ring pool `_NVME_READ_CONCURRENCY` 3 → 2 SHIPPED
2026-09-21** (+1.89 % decode, +1.85 % 2k prefill; `VAAYU_RING_POOL_N=4` REGRESSES −2.07 %).
**The read-shape arm was BUILT AND REVERTED 2026-09-21 on its own stop rule:** −66.7 % reads/token and
delivered rate 6.07 → 6.50 GB/s bought **−0.40 % decode** — a direct experiment refuting the rate lever.

- [ ] **The last lever, and its headline number is NOW IN DOUBT: fill the 1.57 ms/layer drive-idle
  gap.** The **>= +31 %** came from "token -> 0.399 s at the same 6.03 GB/s", i.e. from assuming the
  token collapses onto disk-busy time. **The read-shape arm above tested exactly that assumption and
  it FAILED: busy fell 26 ms/token and the token moved 2 ms the WRONG WAY, while the >= 1 ms
  per-layer gaps stayed identical in number.** Prefetching *useful* bytes is a different experiment
  from supplying the same bytes faster, so this is not refuted — but do not quote +31 % as if it
  were measured. **Also note the N=1 arm is POSITIVE, not the -3.7 % this row used to cite:**
  `VAAYU_LOOKAHEAD` on MiniMax measured **+2.5 % at N=1** post-P4 and **-3.7 % at N=3**
  (`check_lookahead_flag_safe`), and the server runs N=1. Cheap way in is still to re-gate that
  built machinery rather than scope a new build — but price it against its **+18.0 % bytes/token**
  first: against a 76 %-busy drive that alone puts the token floor near +11 %, under row 35's own
  >= +15 % re-open bar.
- *(→ moved to `todo-backlog.md` 2026-09-26: Row 49 step 3 — re-price the CPU hybrid. Re-open when: row 49 step 2 lands AND a CPU expert kernel exists.)*
- *(→ moved to `todo-backlog.md` 2026-09-26: Row 49 — MiniMax PREFILL's H2D is pageable. Re-open when: verify with one MiniMax prefill census; if the pageable copies are gone, close it.)*

**STOP RULE (step 1): `h2d_pct` < ~15 % of `T_prof` → the slice shrank under rows 27/30-P/44/45;
close this row with the number, and close the CPU-hybrid half too — its prize was smaller than this
slice to begin with.**

**WHAT IS LEFT OF THE CPU HYBRID, stated so nobody re-derives it:** its **+6.8 % Qwen3-235B arm is
UNVERIFIABLE FOREVER** (model deleted 2026-09-20, fused-EXL3 forward has no on-device model); it is
**dead on Qwen3-Next** (disk is 0.42 / 0.32 % of a decode token, row 36's spike — no device-busy
shadow to hide CPU work in); so its prize is **MiniMax-only**, a §5 asymmetry question before it is
an engineering one. **+22.0 % is an arithmetic CEILING, never executed** (0.562 → 0.461 s computed
from row 26 step-1 physics: CPU int4 expert 214 / 163 µs, RAM read 65.6 GB/s). And **EXL3 has no CPU
kernel** — row 26 says so — so collecting it on MiniMax means re-quantising the checkpoint to a
CPU-friendly int4 format **plus** kernels **plus** loader **plus** dispatch. **Weeks.**
**Recommended default: close the CPU-hybrid half for good once step 2 lands.**

**§8:** (1) step 2 probably needs no knob — a correct overlap is strictly better; any safety escape
hatch lands in `vaayu/server/config.py` in the SAME commit. (2) n/a, forward-internal. (3) **REACHABLE
— an N=1 per-decode-token effect on a model the server loads**, unlike P8's N=3 win that delivered
0 % to users. (4) not blocked.

---

# 48 · [bug] `use_torch_compile=True` CRASHES on CUDA — and `config.py` declines it for the wrong reason (added 2026-09-20, found by row 36's A/B) — **SEQUENCED AS S10 (2026-09-26), fix (b); handoff [`…S10-refuse-torch-compile-HANDOFF.md`](docs/superpowers/specs/2026-09-26-S10-refuse-torch-compile-HANDOFF.md). Line numbers below are stale: guard `vaayu_base.py:2180-2189`, compiled call `:5332-5339`.**

**Reproduced in two independent processes on Qwen3-Next**, both during the first generation:
`RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
run`, raised at `transformers/models/qwen3_next/modeling_qwen3_next.py:866`.

**Cause (structural, model-independent):** `_compiled_layer_fn` is built with
**`mode="reduce-overhead"`, which is torch.compile PLUS inductor's cudagraph trees**
(`vaayu/vaayu_base.py:5064-5068`). Cudagraph trees assume one model invocation per step and require
`torch.compiler.cudagraph_mark_step_begin()` at each boundary. Vaayu drives **48+ layers through that
one compiled callable per token** with no step boundary, so a graph's output is read after a later
replay overwrote it. The Qwen3-Next line number is incidental; the mismatch is not.

**Second, separate defect — a doc/test claim that is wrong.** `vaayu/server/config.py:328` puts
`use_torch_compile` on `tests/test_engine_param_reachability.py`'s decline list with the reason
*"`__init__` overrides these in places; they are not live paths"*. Measured: **`effective=True` after
load on CUDA** — the only `__init__` guard (`vaayu_base.py:2118`) disables it for non-CUDA devices
and leaves it ON for CUDA. It IS a live path, reachable by any library caller. The correct reason to
decline it is that it crashes.

**Not fixed here** — row 36's A/B was a measurement spike, nothing in `vaayu/` was touched. Two
candidate fixes, unpriced: (a) change the mode off `reduce-overhead` — but measured **24–35 % SLOWER**
that way, so that is a correctness fix, not a perf one; (b) refuse `use_torch_compile=True` at load
the way RC7 refuses the prefix cache on recurrent caches, and correct `config.py:328`'s reason.
**(b) is the cheap honest one.** Detail:
[`…-row36-torchcompile-AB-RESULTS.md`](docs/2026-09-20-row36-torchcompile-AB-RESULTS.md).

**Not measured:** MiniMax-M2.7 and the dense 70B. The crash site is in Qwen3-Next modeling code, but
the step-boundary mismatch is not model-specific.

---

# 40 · [bug] The expert slab arena is never capped at the model's own expert bytes (added 2026-09-18, split out of row 30)

**Split out of row 30 on 2026-09-18 at the user's request.** The 30-P final review found it, and it was filed as a
bullet under row 30. It is not part of row 30's goal (S1–S3), and its own text asked for a separate row. The bullet
below is moved verbatim.

- [bug] **Latent (found by the 30-P final review, 2026-09-18): the expert slab arena is sized to the RAM budget, never capped at the model's own expert bytes.** Where: `compute_expert_budget` (`vaayu/vaayu_base.py` ~:379-423: `expert_budget = min(ceiling, max(1 GB, available − overhead))`) and the arena sizing (`unclamped_slab_count(_expert_budget, _slab_bytes, pool)`, ~:448 and ~:2651-2696) — nothing caps the slab count at (indexed experts × slab_bytes) + pool. Why it matters now: pre-existing on the EXL3 forwards; since 30-P the gate also allocates the PINNED arena for every fused non-EXL3 CUDA model, which before used a pageable clone cache that only grew as far as its expert population. A fused non-EXL3 MoE whose experts total less than the derived budget would pin (unswappable) RAM it can never fill. Measured impact today: none on the box's models — Qwen3-Next's experts are 48 layers × 512 × 1,769,520 B ≈ 43.5 GB vs an 18.54 GB server-sizing arena; the EXL3 models' experts also exceed their arenas. Fix sketch: cap the arena (and the budget it trims) at the offset index's expert count × slab_bytes + pool, derived (CLAUDE.md §5), with a test parametrized over Qwen-like and MiniMax-like geometries plus a small-model geometry. Touches shared sizing → its own row, all three accumulator gates.

**2026-09-26 (S5-M): the gate now ALSO pins the arena for every non-EXL3 MODULE-LIST model on CUDA**
(`6596c49`), so one more layout is exposed. Still nothing on disk triggers it: DeepSeek-V4's routed experts
are ~147 GB against a ~23 GB arena.

**Status 2026-09-18: open, latent, not started. Nothing on disk triggers it.** Checkpoints under
`/home/a/code/ai-ml/models/`: the only fused non-EXL3 MoE is Qwen3-Next (43.5 GB of experts > its arena). Qwen3-235B and
MiniMax-M2.7 are EXL3 and also exceed their arenas. The dense models build no arena.

**What makes it real — derived, not measured:** Qwen3-30B-A3B int4, the smoke rung rows 29/30 named but did not
download. The per-expert byte formula, `3 × (hidden × inter / 2 + hidden × inter / group × 2) + 48`, reproduces
Qwen3-Next's logged 1,769,520 B exactly (2048 / 512 / g32). With Qwen3-30B-A3B's published config (hidden 2048,
`moe_intermediate_size` 768, 128 experts × 48 layers) it gives **~16.3 GB of experts at g32, ~15.0 GB at g128**. That is
below the ~18.5 GB arena this box derives for Qwen3-Next at server sizing, and a smaller model's lighter non-expert
weights leave more RAM for the arena, not less. So loading it would likely pin **~2–3.5 GB or more of unswappable RAM the
arena can never fill.** Check this with a real load's `Expert slab arena:` line before quoting it.

**Stop / done rule:** done when the arena (and the RAM budget it trims) is capped at the offset index's derived expert
bytes + pool. The test is parametrized over Qwen-like, MiniMax-like and small-model geometries (CLAUDE.md §5). All three
accumulator gates must stay green (Qwen3-Next, Qwen3-235B, MiniMax-M2.7) with **unchanged arena lines** on all three,
because none of them is under-populated. **§8 (to answer at build time):** no knob is expected, since the cap is
derived like the arena. It is reachable at N=1, since the arena is allocated at load in every mode. But it delivers
0 % to any user until a model smaller than its arena is served.

---

# 47 · [ ] ARCHITECTURE-SUPPORT PROGRAM — the row that makes Vaayu serve different architectures (added 2026-09-20; decomposed into 5 plans the same day at the user's NON-NEGOTIABLE direction)

**User mandate 2026-09-20: "Vaayu MUST serve different architectures."** The work was decomposed
into **five plans** because it spans four independent subsystems — checkpoint validation, expert
indexing, MoE dispatch, model porting — plus two models that share almost nothing. One plan would
stall entirely whenever any single piece stalled, which is the failure mode §6 exists to prevent.
Program doc: [`…-architecture-support-PROGRAM.md`](docs/2026-09-20-architecture-support-PROGRAM.md).

**THE ORDER TO IMPLEMENT — this is the answer to "when do I work on which item":**

| # | plan | delivers | depends on | gate | cost | doc |
|---|---|---|---|---|---|---|
| ~~**A1–A4**~~ | **ALL CLOSED — A1 guards + A2 fused-3D indexing + A3 gpt-oss-20b end to end (2026-09-21), A4 shared-expert pinning (2026-09-23, premise REFUTED, zero code — they were already pinned, 48/48 layers, 4.83 GB).** gpt-oss-20b generates through the server, bit-identical to stock transformers at 4 and 256 tokens, at **0.17 tok/s / 13.28 GB per token** (whole-layer delivery — that cost is what A6 exists to cut). Full task-by-task text: `git log -p -- todo.md` (pruned 2026-09-23). | — | fired/passed | spent | [`PROGRAM`](docs/2026-09-20-architecture-support-PROGRAM.md) |
| **A5** `[~]` | **DeepSeek-V4-Flash port — step 2 DECOMPOSED into A5.1…A5.7 2026-09-23; PICK A5.1** | the 157 GB I8/fp8 MoE loads and runs | ~~A4~~ **(closed)** | ~~kernels must build on sm_120~~ **GATE PASSED 2026-09-23: 6/6 build+validate. `sparse_attn` needs block=16 (shipped 64 exceeds sm_120's 101,376 B smem); `apache-tvm-ffi==0.1.9` is an exact pin. Step 2 next.** | **weeks** | brief: [`…-A5-deepseek-port-BRIEF.md`](docs/2026-09-20-arch-A5-deepseek-port-BRIEF.md) |
| A6 | per-expert MXFP4 streaming | 1.27 GB working set instead of 10.17 GB; drops the Hub kernel dependency. **Perf, not reach** | A3 | A3's tok/s exists to beat | ~3 days | plan when A3 lands |
| A7 | G3 `compute_device` parity · G5 NVMe re-measure | removes two latent generality traps | — | none | ~1 day | plan when picked |

**Three follow-up rows, opened from A1's final review (§6 — they must not vanish):**
- [ ] **47-m1 — `_detect_moe` does not recognize `n_routed_experts`** (`vaayu_base.py:3483-3489`
      matches only `num_local_experts` / `num_experts` / a `Moe|MoE|Mixtral` arch string). A
      checkpoint whose only expert field is `n_routed_experts` — **the DeepSeek shape, which
      `_EXPERT_COUNT_FIELDS` explicitly claims** — gets `is_moe=False`, so the index is never built
      and G1 never runs. Same for `device="cpu"` (`server/config.py:27` is user-settable). NOT
      silent-wrong output: with `is_moe=False` Vaayu streams whole layers and the experts come
      along. A coverage-claim gap, not a correctness hole. Fix: add `n_routed_experts`, or gate on
      `expected_expert_count() is not None`. **Bites row 28b/A5 directly.**
- *(→ moved to `todo-backlog.md` 2026-09-26: 47-m2 — `expert_read_buffer_bytes` builds the index config-less. Re-open when: the load order changes.)*
- [ ] **47-m4 — sliding-window layers are STORED and BUDGETED at full length.** Split out of A3 task 5
      (2026-09-21) because it is a subsystem, not a sizing line. Two places: (1) Vaayu's own KV cache
      (`FP8DynamicCache`, `vaayu_base.py:1382`) keeps its ring buffers in `key_cache`/`value_cache` and
      never consults the per-layer cache classes it constructs — measured: `cache.layers[0]` IS a
      `DynamicSlidingWindowLayer` and holds NOTHING, while `key_cache[0]` holds all 512 tokens of a
      512-token prefill; (2) `kv_bytes_per_token` (`vram_plan.py:63`) multiplies `num_hidden_layers` by the
      full per-token cost with no per-layer awareness. **Measured waste on gpt-oss-20b: 12 of 24 layers are
      windowed at 128, so 24,576 of 49,152 B/token — exactly HALF — is stored and reserved but unreadable.
      At the derived `max_seq_len` of 33,206 that is 1.632 GB held against 0.819 GB needed: 0.813 GB, 49.8 %,
      on a 16 GB card. It also halves the `max_seq_len` Vaayu derives for such a model.** Correctness is NOT
      affected (A3 task 5 fixed the mask; the extra KV is simply masked out). Fix touches the cache's
      per-layer capacity, `get_seq_length`/`_seq_len` bookkeeping, the prefix cache's snapshot/crop and the
      KV persist format — hence its own row. **Second, smaller gap found with it: when the mask is None
      (flash, and sdpa at `q_len == kv_len`) the window is NOT applied by Vaayu at all.** Flash is fine —
      the attention module passes its own `sliding_window` to the kernel — but **sdpa with an implicit
      causal mask would silently ignore the window**; no on-device model hits it today (gpt-oss refuses
      sdpa), and it must be closed before a windowed model runs on sdpa.
- [ ] **47-m6 — A3's fp32 rotary table changed numerics for EVERY model, and only gpt-oss's side of it was
      measured.** Opened 2026-09-21 by the change that closed A3 task 5. `_precompute_rope` used to build the
      rotary module in `running_dtype`, so `inv_freq` — a table of FREQUENCIES — was bf16, and the phase error
      grows with the position it multiplies: **cos/sin 0.0039 off stock at position 3, 0.51 off (38 % of full
      scale) at position 255.** It is now fp32, which is what stock transformers does for every model, and that
      is what made gpt-oss-20b bit-identical to stock. **For Qwen3-Next, MiniMax-M2.7, Llama-3.3-70B and
      Llama-3.2-1B the change is real but UNMEASURED.** The gates that ran (accumulator determinism ×2, 1B
      batch/prefix equivalence, suite 2416) prove determinism and self-consistency — **none of them measures
      quality**, so "green" here is not evidence either way. Expected direction: better, and more so the longer
      the context, since the error being removed grows with position. **The work: teacher-forced perplexity
      before vs after (`git stash` the one-line dtype change, or check out `cac352e^`), on Qwen3-Next AND
      MiniMax, at a context long enough for the effect to exist (≥ 4k — at 256 tokens the cos error was already
      0.51, so 4k is not exotic), SEVERAL draws per arm** (one decode-ppl draw is ONE draw: a null bit change
      moves it sd 0.5-0.7 pp, D6). Also record decode tok/s: the fp32 table costs a little more compute where
      cos/sin are built, once per forward, and that should be ~0 but has not been checked. **Stop rule: if ppl
      is unchanged within noise on both models, close with the number — the fix is still right, it just did not
      matter below 33k context. If it improved, that is a quality win nothing else on this board delivers. If it
      REGRESSED, stop and investigate: that would mean something downstream depended on the bf16 table.**
- *(→ moved to `todo-backlog.md` 2026-09-26: 47-m5-b — a whole-layer model advertises caches it never fills. Re-open when: S8 (gpt-oss) touches the whole-layer forward's cache sizing.)*
- [ ] **47-m3 — shard-name arithmetic assumes 5 digits and 1-based numbering.**
      `checkpoint_integrity.py` uses `_SHARD_RE` with `\d{5}` and `set(range(1, total + 1))`.
      **`gpt-oss-20b` on disk is `model-00000-of-00002` … `model-00002-of-00002` — 0-based, THREE
      files against total=2** — which yields no raise today (correct), but a missing `00000` is
      invisible and `len(present) > total` is silently ignored. Also: a PARTIAL `delete_original`
      (an `OSError` at `utils.py:276` is caught and logged) leaves shards plus the retained index,
      and the `weight_map` check would then raise on a model that had been loading fine.

**A5 is a BRIEF, not a plan, deliberately** — a TDD plan for kernel work of unproven
feasibility would be placeholders. It carries its scope, gate and abandon rule and becomes a plan
the day its gate passes. (A4 was the same, and closed without ever needing to become one.)

**Sequence in one line: ~~A1~~ → ~~A2~~ → ~~A3~~ → ~~A4~~ → **A5 (next)**. A6 now has a baseline to
beat (0.17 tok/s, 13.28 GB/token) and A7 stays opportunistic.**

**A3 does NOT make DeepSeek cheaper.** gpt-oss is MXFP4 / fused-3D / sliding-window and natively
supported by transformers; DeepSeek is I8+fp8 / per-expert / MLA+sparse-index and unsupported, needing
modeling code plus sm_120 kernels. The only shared piece is A1. **Do not price A5 as "cheaper because
A3 is done."**

**Design spec behind A1–A3:**
[`…-row47-G1-fused3d-gptoss-design.md`](docs/superpowers/specs/2026-09-20-row47-G1-fused3d-gptoss-design.md).
Approach decided: **whole-layer streaming through the stock MXFP4 kernel (A)**, because
`Mxfp4GptOssExperts.forward` calls `matmul_ogs` over the whole 3D parameter and routes INSIDE the
kernel — per-expert slices do not compose with it. Per-expert streaming is A6, priced against A3.

**The gap list below is the source material the plans were cut from. G0, G1 and G7 closed with A1–A4
(pruned 2026-09-23); G2 is closed; G6 is closed as a perf idea (rows 26/28) and is reach-only.**

**This is the row that broadens REACH, not speed. It does not rank against the perf queue (49 · 36 · 35 · 32)
on the same axis — those buy tok/s, this buys "the next checkpoint loads at all".** Source of the gap list:
[`…-IMPL-adding-new-architectures-PLAYBOOK.md`](docs/2026-09-12-IMPL-adding-new-architectures-PLAYBOOK.md) §6,
which recorded them 2026-09-12 with the note *"None is currently a row; each is a candidate one."* This is that row.
Row **28b** is the DeepSeek port — ONE architecture. This row is what makes the one after it cheap.

**MODEL SET RE-SET BY THE USER 2026-09-20.** On disk now: **Qwen3-Next-80B int4** (fused non-EXL3, hybrid attn),
**MiniMax-M2.7 EXL3** (module-list), **Llama-3.3-70B EXL3 4.5bpw** (dense), **Llama-3.2-1B** (small dense),
**DeepSeek-V4-Flash-0731** (does not load). **Qwen3-235B is RETIRED AND DELETED — do not test it, do not put it in
a gate.** Consequence to state plainly: **the fused-EXL3 forward now has NO on-device model.** Its perf history
stands; nothing re-tests it. `CLAUDE.md` §0/§5 were updated the same day.

**DeepSeek-V4-Flash — the board's record of it was STALE; these are measured off the bytes on disk 2026-09-20:**
- **COMPLETE as of 2026-09-20 (re-checked): 48/48 shards, `model.safetensors.index.json` present, 72,317 tensors,
  every index ref resolves on disk, 157 G.** (It was 46/48 earlier the same day — verify before trusting.)
- **TWO QUANT FORMATS IN ONE CHECKPOINT, and `config.json` tells you only one of them.** `quantization_config` says
  `{quant_method: fp8, fmt: e4m3, scale_fmt: ue8m0, weight_block_size: [128,128]}`, but the tensor dtypes say:
  **routed experts are `I8` (int8) weights + `F8_E8M0` scales** — that is the 147.2 GB bulk — while **shared experts
  and attention are `F8_E4M3` + `F8_E8M0`**, norms are BF16, `attn_sink` is F32, and `ffn.gate.tid2eid` is I64.
  **Dequant must handle both, and rows 28/28b's "fp4 4.2500 bpw" is simply wrong — re-derive every byte number.**
- **Geometry, measured off the index, not the config:** 43 MoE layers (0–42, **every** layer is MoE — no dense prefix),
  256 routed experts + **1 shared expert** per layer, 6 tensors per expert (`w{1,2,3}.{weight,scale}`),
  **one expert = 12.750 MiB** (that figure in row 28 was right), **all routed experts = 147.2 GB**.
- **Attention is MLA + sparse-index:** `q_lora_rank 1024`, `num_key_value_heads 1`, `num_attention_heads 64`,
  `index_topk 512`, keys `attn.wq_a/wq_b/wkv/q_norm/kv_norm/attn_sink/indexer.*/compressor.*`. G4 bites hard here —
  this is not a sliding window, it is low-rank KV plus a learned index.
- **Its expert keys DO satisfy the offset index** — `layers.4.ffn.experts.0.w1.weight` matches `\.experts\.(\d+)\.`.
  So gap G1 does NOT block this model. Note the prefix is `layers.N.ffn.…`, not HF's `model.layers.N.mlp.…` → key remap.
- **It has `shared_experts`** (`layers.N.ffn.shared_experts.w{1,2,3}`) → G7 bites. They are always active, so they
  belong PINNED, not streamed; nothing does that today.
- **It has sparse-index attention** (`attn.indexer`, `attn.compressor`, `attn_sink`) → G4 bites.
- Still `model_type: deepseek_v4` with **no `auto_map`**, so transformers 5.5.4 still cannot build it. **NEW since
  2026-09-12: the repo now ships `inference/model.py`, `inference/kernel.py`, `inference/convert.py`** — DeepSeek's own
  reference implementation. Row 28b assumed this had to be written from scratch; it can now be PORTED. Re-cost 28b.

**The gaps still open (G0, G1 and G7 were closed by A1/A2/A3/A4 and pruned 2026-09-23 — `git log -p -- todo.md`):**

- [x] **G2 — ~~`_make_fused_sparse_forward` (non-EXL3) has never executed on device~~ CLOSED by row 30's S2
      (2026-09-13).** Qwen3-Next runs it daily and 30-P/44 perf-tuned it. Kept here so the playbook's stale line is
      not re-copied into a plan.
- *(→ moved to `todo-backlog.md` 2026-09-26: G3 — only one MoE forward takes `compute_device`. Re-open when: a CPU-compute feature is wired into any MoE forward.)*
- [ ] **G4 — uniform-attention assumption.** KV sizing, the prefix cache and derived `max_seq_len` assume every layer
      attends the same way. **Partly paid down already:** row 30 RC4/RC6 and row 42-B handled hybrid linear/full for
      Qwen3-Next. **Still unhandled: sliding-window and sparse-index attention (DeepSeek-V4).** Check before trusting
      any TTFT or context number on such a model.
- [ ] **G5 — slab scale.** `SlabArena` (`vaayu/persist/expert_cache.py:25`) was exercised at ~1,000–1,250 slabs of
      8–12 MiB. Qwen3-Next is already 24,576 slabs of ~1.6 MiB and works, so this is partly disproven — but
      `_NVME_READ_CONCURRENCY` (`safetensor_model_persister.py:181`, **2 since row 49, 2026-09-21**) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
- *(→ moved to `todo-backlog.md` 2026-09-26: G6 — no CPU expert compute exists. Re-open when: CPU-only serving becomes a goal.)*

**WHAT ACTUALLY HAS TO BE IMPLEMENTED FOR DeepSeek-V4-Flash — and row 47 ALONE DOES NOT DO IT.**
Row 47 removes silent failures and makes the NEXT checkpoint cheap. Making THIS one run is row **28b**, and the
split is:

| # | work | row | why |
|---|---|---|---|
| 1 | **G0** incomplete-checkpoint guard · **G1-minimum** raise instead of finding zero experts | **47** | ~1 day, independent of everything, and it is what stops a silent wrong answer later |
| 2 | **G7** shared experts pinned, not streamed (this model has exactly 1/layer, always active) | **47** | 28b needs it; nothing does it today |
| 3 | **G4** KV sizing / `max_seq_len` for **MLA + sparse-index** attention (`q_lora_rank 1024`, `kv_heads 1`, `index_topk 512`) | **47** | 28b needs it; hybrid linear/full was paid down by rows 30/42-B, this is NOT that case |
| 4 | **Dequant for BOTH formats** — I8+E8M0 block-scaled routed experts AND F8_E4M3+E8M0 shared/attention | **28b** | the bulk of the bytes is int8; do not assume the config's fp8 |
| 5 | **HF modeling code + `auto_map`**, ported from the repo's own `inference/model.py` (MiniMax is the precedent) | **28b** | `deepseek_v4` is unknown to transformers 5.5.4 and there is no remote code |
| 6 | **Key remap** `layers.N.ffn.experts.M.w{1,2,3}` → the module tree the forward expects | **28b** | keys are DeepSeek's own, not HF's `model.layers.N.mlp.…` |
| 7 | **Kernels on sm_120** — MLA and the DSA indexer/compressor | **28b** | **the hard part, and the abandon gate: if these will not build or validate, stop at step 1** (~2 days lost, not weeks) |

**Do NOT start at 28b.** Do 1–3 inside row 47 first: each is small, each is needed by 28b anyway, and G1-minimum is
the one that converts "finds zero experts and proceeds" into an error. **G1's full fused-3D splitter is NOT needed for
DeepSeek** — its keys already match `.experts.N.`. **But a fused-3D checkpoint HAS now arrived: `gpt-oss-20b`,
2026-09-20**, so the splitter has a real driving case and its own brief + plan prompt
([`…-row47-G1-gptoss-BRIEF.md`](docs/2026-09-20-row47-G1-gptoss-BRIEF.md)). Build the raise as the floor and the
splitter as the goal.

**G5 is NOT a blocker here (re-checked 2026-09-20):** 12.750 MiB/expert sits inside the 8–12 MiB band the slab arena
was tuned at, and Qwen3-Next already runs 24,576 slabs against this model's ~11,008. Only `_NVME_READ_CONCURRENCY`
remains unverified, and at 12.75 MiB reads it is close to its original tuning point.

**Suggested order:** G0 + G1-minimum first (both turn silent wrongness into a message, both are small), then G7 and G4
if and only if DeepSeek-V4 is the next target, then G3, then G5's re-measure. G2 is already closed; G6 is closed as a
perf idea and is reach-only.

**§8:** (1) knob — probably none; a new loader strategy may need a format knob, and if so it lands in
`vaayu/server/config.py` in the SAME commit with the measured cost of getting it wrong, plus `vaayu/server/README.md`.
(2)/(3) the model path is already a server parameter, so reach work is reachable at N=1 by construction. (4) nothing
is banked — a model that loads, loads for users.

**§5:** every fix here is a generality fix; none may hardcode a model number. G1's splitter in particular must derive
expert count and byte size from the checkpoint, never from a config literal.

---

# 23 · → **MOVED to `todo-backlog.md` 2026-09-26** — the prefill stager re-runs once per BATCHED chunk. Why: N>1 only: the server runs one request at a time (`MAX_BATCH_SIZE = 1`), so this is worth 0 % to users (CLAUDE.md §8). Re-open when: `batched_generate` is wired into the server.

---

# 6a-b · → **MOVED to `todo-backlog.md` 2026-09-26** — `head_bytes` is probably not a transient. Why: buys batch-admission headroom, never tok/s — N>1 only. Re-open when: batching reaches the server.

---

# 4 · [ ] 2× KV overshoot, ~0.83 GB at Qwen's derived operating context — PARKED

**Cold-pickup brief — read it BEFORE re-opening this row:** [`docs/2026-09-10-row4-kv-overshoot-PARKED-BRIEF.md`](docs/2026-09-10-row4-kv-overshoot-PARKED-BRIEF.md) (gitignored). It carries the single condition that reverses the park.

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

# 5 · → **MOVED to `todo-backlog.md` 2026-09-26** — quality certification of prefix compression. Why: affects only users who opt into `PREFIX_COMPRESS` (server default `None`). Re-open when: compression becomes a default, or a user opts in.

---
