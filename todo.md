# TODO — Vaayu pending work

**Everything with a `[ ]`, `[~]` or `[!]` in this file is PENDING. Closed rows are not kept here.**

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

**Where the removed text went.** Full pre-prune text: `docs/2026-09-10-todo-archive-preprune.md`
and `docs/2026-09-03-todo-archive-preprune.md` (**gitignored — this disk only**). The durable copy
is `git log -p -- todo.md`; the commit before each prune holds it in full. Older closed detail was
moved out on 2026-08-23 (commit `f3869b3`).

Protocol: `CLAUDE.md` §6 (ledger) and §7 (all planning/investigation/spec work lives in `docs/`).
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Reading this cold, in a fresh session, is the intended way to use it.** Read in this order —
**§0 first: it is the build ORDER and the perf / not-perf split; the sections after it hold the
detail each entry points to.**

0. **START HERE** — the whole open queue on one screen, in pick order, added 2026-09-10. If you read
   nothing else, read that table and §0.
1. **OPEN** — EMPTY as of 2026-09-10 (C1, its last row, was FIXED that day).
2. **STILL OPEN, carried forward** — live items left behind by rows that closed.
3. **SETTLED — DO NOT RE-OPEN** — read this BEFORE proposing any perf work. It is the list of
   things already built and refuted, with the number that killed each one.
4. **Environment** — how to run anything here. `python` is not on PATH; every on-device arm needs
   its own `systemd-run` scope. Skipping this section is how arms get thrown away.
5. **On-device gates** — what each gate can actually catch, and which ones a change obliges you to run.
6. **OPTIONAL** — real work, no slot. Never ahead of the rows above.

**Read §0 below first: it is the ORDER. Everything after it is the DETAIL.**

---

# ▶ START HERE — the whole open queue, re-prioritised 2026-09-11

## ▶▶ PICK THIS NEXT (re-set 2026-09-23: A4 closed, the user DECLINED row 32's download, 47-m7 AND 47-m5 were picked up and CLOSED, and row 46 was RE-VERIFIED ON DEVICE so it starts cold) ◀◀

**A fresh session starts here and needs nothing else: each item below has a SELF-CONTAINED handoff
document that carries its own start gate, stop rule, numbers and commands.**

**A4 CLOSED 2026-09-23 — premise REFUTED, zero code.** Shared experts were never streamed: the
routed regex `\.experts\.(\d+)\.` does not match `...mlp.shared_expert.*`, so they land in the
non-expert set and Phase-2 pinning puts them in VRAM — **48 of 48 Qwen3-Next layers, 4.83 GB
reserved, ~302 MB of shared expert (6.2956 MB × 48, BF16, un-quantized by AWQ)**. The compute path
was verified to BIND on the real module tree, which byte accounting alone could not show. Detail in
the A4 bullets under row 47. **A5 is now the head of the REACH track — but it is weeks and
kernel-gated, so the cheaper rows in the table below lead it.**

**ROW 32 IS OUT OF THIS TABLE — `[!]` BLOCKED 2026-09-23, by the user, on cost.** It was order 1 that
morning; the session verified its route (a full HF sweep: the build on disk ships `2.0…8.0bpw` as
BRANCHES of the SAME repo, so route 1 is live and route 2 is dead) and then the user **declined the
download**. **Row 32's entire cost IS that download — there is nothing else in it to build — so it
cannot proceed and is not pickable.** Do not re-open it by re-deriving the projection; it unblocks
only on the user authorising an ~81.6 GiB pull. Its verified branch sizes and two handoff
corrections are banked in row 32 itself. **NO DOWNLOAD is a standing constraint until the user says
otherwise: every item below runs on checkpoints already on this disk.**

**47-m7 CLOSED 2026-09-23 — INERT, byte-identical, zero code.** It was order 1 that afternoon and it is done:
A3's FP8-compat gating does not change Qwen3-Next. Both arms' `prefill_sha256` and `decode_logits_sha256` matched
exactly, all 32 greedy ids matched, and the negative control fired (the ungated arm logged the patch, the gated arm
logged nothing). The mechanism is now known: the compressed-tensors quantizer never reads `modules_to_not_convert`.
**The live consequence for the next session: A3's ONLY remaining unverified behaviour change on Qwen3-Next is the
fp32 rotary table (`cac352e`) — that is 47-m6, still open, and it is now the last A3 loose end.** Numbers in the
47-m7 bullet under row 47.

**47-m5 CLOSED 2026-09-23 — BUILT, MEASURED, SHIPPED.** It was order 1 and it is done: gpt-oss-20b's load RSS
went **19.20 → 3.81 GB (−15.39 GB)** with the output BIT-IDENTICAL, and the win is reachable through the server at
N=1 on defaults (the gpt-oss server gate logs the arena SKIPPED and `rss=3.78GB`). The guard and the dispatch were
asking different questions; the whole-layer condition is now written once, in `moe_patch.dispatches_whole_layer`,
and both sites call it. Qwen3-Next and MiniMax keep their arenas, verified — that was the dangerous direction.
Gates: suite 2422/16, accumulator 8+8, 1B prefix 1, server 9+9. It left **47-m5-b** behind (a whole-layer model
still ADVERTISES caches it never fills — misleading telemetry, zero bytes). Numbers in the 47-m5 bullet under row 47.

**ROW 46 IS READY TO START COLD — re-verified on device 2026-09-23, and its old handoff was stale in
four places.** A fresh session opens the 2026-09-23 handoff and begins at step 1; the start gate is
already passed (MiniMax's 6.24 GB confirmed live under server knobs). **The four corrections, in case
`docs/` is gone: (1) Qwen3-Next reserves 4.83 GB TODAY, not 0 — 42-B's RC7 lift has applied; (2) gpt-oss
is a third payer at 4.89 GB but the lever is INERT there since 47-m5, so do not spend an arm on it;
(3) `derive_expert_ceiling` is `vaayu_base.py:355`, called `:2703`; (4) the accumulator gate loads with
`enable_prefix_cache=False`, so an arm taken through its config measures ZERO reservation.** The old
checklist also said to measure on Qwen3-235B, which was deleted 2026-09-20 — corrected to Qwen3-Next.

| order | item | what to open FIRST | cost | why this order |
|---|---|---|---|---|
| **1** | **Row 46 — size the prefix reservation** | [`…-2026-09-23-row46-prefix-reservation-HANDOFF.md`](docs/2026-09-23-row46-prefix-reservation-HANDOFF.md) — **RE-VERIFIED ON DEVICE 2026-09-23; the 2026-09-21 handoff is SUPERSEDED, do not read it** | ½ day to measure, 1-2 days to build | **START GATE IS ALREADY PASSED: MiniMax's 6.24 GB reservation was confirmed LIVE under server knobs on 2026-09-23** (ceiling 16.11 of 26.11 GB), and it is the ONLY live target — Qwen3-Next's mechanism is measured away (disk 0.42 % of a token, −2.56 % measured by 42-B) and gpt-oss's 4.89 GB is INERT since 47-m5 (no arena, no cache). Magnitude measured: +43 % cache slots (1814 → 2594). **STEP 1 IS DONE 2026-09-23 AND IT FIRED: +13.36 % decode (ABBA, 8 reps, ids identical) — the stop rule did NOT close it. But the winning arm zeroes the prefix pool (5.90 → 0.00 GB budget), so 13.36 % is the UPPER BOUND, not the answer; step 2 (the design) is what remains.** A4 found a second reason to care: Phase-2 pinning `break`s on budget miss, so shrinking this budget is what would un-pin the always-active shared experts. |
| **2** | **A5 — the DeepSeek-V4 port** (row 28b) | brief: [`…-A5-deepseek-port-BRIEF.md`](docs/2026-09-20-arch-A5-deepseek-port-BRIEF.md) | **weeks**, abandon decided at step 1 | Head of the REACH track now that A4 is closed, and the user's mandate fixes that order. Ranked last only on cost: it is kernel-gated on sm_120 and the model still does not load. It inherits A4's two carried checks. No download — the checkpoint is on disk, complete (48/48 shards, 157 G). |

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


**Nothing is in flight. No §0 P-row is open, so nothing here makes Vaayu faster on Qwen3-235B, MiniMax-M2.7
or the 70B today — every row is OPTIONAL by §0's test.** The order below was set by the user on 2026-09-11
after a triage of every open row: **three are worth working, in this order; the rest are not, for the
reason given.** Pick from the PRIORITY table first. The sections after this one are the detail.
**All three PRIORITY rows are CLOSED as of 2026-09-11 (3, 25, then 18).** Row **26** was added the same day at the
user's request (a measurement, no engine code) and **CLOSED the same day** — and it opened row **27**, which is
**the first open row since 2026-09-10 that measurably makes Vaayu faster: +27 % decode tok/s on Qwen3-235B** (keep one
CPU core awake during decode). Row 27 **CLOSED 2026-09-12: +23.1 % decode tok/s on Qwen3-235B through the server, shipped ON** — the first measured user-facing speed-up since 2026-09-10. Beyond it, what remains is the NOT-worth-working
table — pick from it only on its stated re-open condition.

**Added 2026-09-12: row 28**, the user's follow-up to 26 (*re-run it on the non-EXL3 model that arrived*). **CLOSED the same
day: the checkpoint (`deepseek-ai--DeepSeek-V4-Flash-0731`) DOES NOT LOAD** — `deepseek_v4` is unknown to transformers 5.5.4
and the repo ships no HF remote code — and the arithmetic says a non-EXL3 model makes the CPU-AVX2 hybrid **worse**
(+4.5–5.0 % vs row 26's +6.8 / +22.0 %), so **the CPU-hybrid idea is recommended closed**. It did find a large
port-gated prize (~2.5–2.7× Qwen's decode from 2.58×-lighter tokens). **Row 27 still outranks everything.**

**Added 2026-09-12: rows 31–39**, from a user-requested sweep of *"what else can speed Vaayu up"*, priced against a
measured decomposition of one decode token. Detail:
[`docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md)
(**gitignored — these rows are the durable copy**). **The one fact that reorders everything: on both actively-tested
models 70–83 % of a decode token is the NVMe being busy** (Qwen 0.772 s of 0.930; MiniMax 0.396 of 0.562), 8–9 % is
serial non-expert work and 9–20 % is exposed host time. So **any lever that only touches host-side time is capped at
+9 % Qwen / +20 % MiniMax before it starts** — which is independently why row 26 measured the CPU-AVX2 hybrid's
realistic ceiling at +6.8 % / +22.0 %, and why further Rust is dead (SETTLED: GIL 0.11–0.72 %, GC 0.26–0.41 % of
decode). **Everything that can pay more than that cuts bytes per token or raises delivered GB/s.**

**Also 2026-09-12, and it moves the build order:** `cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` is **ALREADY ON
DISK AND COMPLETE** (46 GB, 10/10 shards + index + config), and the unusable DeepSeek-V4-Flash checkpoint has been
**deleted** → **299 GB free**. Row 30's S2 therefore needs **no download** and is unblocked right now; every "+ a
download" cost estimate on row 30 is stale.

**2026-09-13 — T2 / row 33 IS DONE, and it is the second-largest single win on this board.** `DECODE_TOP_K`
lands as the engine's first LOSSY knob, reachable at N=1, **shipped OFF**. On Qwen3-Next the trade is good:
**k=9 → +17.0 % tok/s for +1.47 % perplexity**, k=8 → +29.5 % for +3.43 %, k=6 → +67.3 % for +8.97 %. **On
MiniMax there is NO viable k** — its mildest step (8→7) already costs +6.34 %, and k=6 costs +19.30 % for only
+26.0 %. The default stays `None` because the safe k is a property of the MODEL and is **not predictable from
geometry** (a 10 % slot cut costs Qwen3-Next 1.47 %; a 12.5 % cut costs MiniMax 6.34 %), so no `auto` rule can be
derived. **S3-pre's projection was low by exactly 2× on both Qwen3-Next arms** — it held the VRAM cache hit rate
constant, and the hit rate actually RISES with the clamp (83 → 90 %). Row **33**.

**2026-09-14 — `DECODE_TOP_K="auto"` merged; six row-33 follow-ups open (K1, D4, D3, D1, D2, D5)**, each with a
self-contained brief linked from its row. Their suggested order among themselves is in the table headed
"Row-33 follow-ups — suggested order" (search that phrase); it does not rank them against the rows here.

**2026-09-13 — S2 IS DONE, AND IT IS THE LARGEST MEASURED WIN ON THIS BOARD.** `Qwen3-Next-80B-A3B-Instruct` int4
runs: **4.0648 tok/s** warm at N=1 on the server's own knobs (replicate spread 1.30 %, greedy ids identical across
arms), and **the server gate passes on it (9 passed)** — so it is **shipped, not banked**. That is **4.8×**
Qwen3-235B-EXL3's raw 0.85 tok/s and **3.6×** row 27's served 1.1363. It is the **first non-EXL3 MoE ever executed
on this box**, and it took **seven** root causes, three of which are bugs in code every model uses:
**RC5** the splitter silently dropped tensors whenever a layer spanned non-adjacent shards (24356 of 24576 experts,
`.done` marker stamped anyway), **RC6** `get_seq_length()` returned 0 on a hybrid model so every decode step re-ran
the whole sequence (**9.5×** when fixed), **RC7** the prefix cache tried to lend recurrent state and crashed inside
transformers (now refused, loudly). Detail: [`…-row30-S2-load-path-RESULTS.md`](docs/2026-09-13-row30-S2-load-path-RESULTS.md);
the durable copy is row **30**.

**And it settles the T3/T4 branch with a number.** The drive is now **1.7 %** of a decode token on this model
(`gb_tok=0.030`, 96.5 % RAM-cache hit) against **70–83 %** on Qwen3-235B / MiniMax-M2.7. **So T3's streaming ladder
(rows 32 · 34 · 35) is dead on this model and T4 is the live branch** — but take **S2b** first: phase 3 measured
three levers worth more than anything in T4, and the phase-4 gate passes for a reason that is NOT the one S3 was
priced on — **CORRECTED 2026-09-13: `h2d` is BANDWIDTH-bound (14.52 GB/s as shipped, 95 % of this box's
pageable line rate), S3's PCIe premise is restored, and lever (i) is REFUTED at ≤ 5.7 % of a token.** What
remains of S2b is (ii), landed, and (iii), the VRAM expert cache — the only lever that removes the
0.849 GB/token transfer rather than making it faster. **S3-pre closed the branch question on
2026-09-13: neither S3's gate nor row 36's fires cleanly (15.1 / 15.2 % addressable and 23.0 / 23.4 %
strict), S3 is recommended CLOSED because the VRAM cache's only fill site is the very miss branch it
removes, and the next work is T2 / row 33 — `top_k` 10→8 is +14.7 %, 10→6 is +34.2 %, derived from
measurement on this model.** **(iii) IS BUILT, MEASURED AND SHIPPED (2026-09-13):
`4.1584 → 6.4702 tok/s, +55.6 %`** at N=1 on the server's own knobs, against a **same-day baseline arm on the
stashed tree**, greedy ids identical across every arm, VRAM cache 83 % hit, suite 2081 passed and every
on-device gate green (server gate on this checkpoint 9 passed; Qwen3-235B, MiniMax-M2.7 and Llama-3.2-1B
regression gates all pass). The entry layout is the **PACKED BLOB** (`flat_packed`/`flat_scale`
pre-concatenated), chosen by replay and confirmed by the build's own stop-rule measurement —
`construct_flat` **31.8 µs/expert** measured in place against a 58 µs abandon rule (and against the spike's
assumed 17.6, which is exactly the correction that step existed to make). **Peak VRAM 11.66 GiB, i.e.
3.17 GiB free against the 3.50 GiB reserve floor — tight, recorded, no OOM.** Detail:
[`…-row30-S2b-iii-vram-cache-RESULTS.md`](docs/2026-09-13-row30-S2b-iii-vram-cache-RESULTS.md).

**2026-09-18 — S3-fill scoping (row 30) found a measured lever that needs no CPU kernel: row 30-P, a PINNED RAM tier
for the non-EXL3 fused forward, +10.62 % decode tok/s on Qwen3-Next, greedy ids identical** (probe, not yet built; ~1–2 days).
The same scoping PARKED S3 behind it: a second VRAM fill site exists (fill-behind), but S3's increment on top of 30-P is
−0.2 … +7 % and needs a native AVX2 kernel. Detail in row 30's S3-fill / 30-P bullets.

**30-P DONE 2026-09-18 on branch `row30p-pinned-ram-tier`: +11.69 % decode tok/s measured
(6.6329 → 7.4084, ABBA, greedy ids identical × 12), beating the probe's own +10.62 %. **MERGED to
`main` 2026-09-18 (fast-forward `c1799bc → 8475123`) — the server gets it: `python -m vaayu.server` loads with
`DEVICE="cuda:0"`, so a fused non-EXL3 model (Qwen3-Next) now allocates the pinned arena through `create_app` (server gate:
`10380 x 1.8MB`, pinned 18.54 GB, 9 passed). EXL3 models (Qwen3-235B, MiniMax) are unchanged — they already had it.** See row 30-P below and
[`docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md`](docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md)
(gitignored; this row is the durable copy).** D6 CLOSED the same day (no code). Standing user
rule from the same day: a non-EXL3 model must never execute an exllamav3 API (true today: 0
calls measured on this build, `docs/probes/p30p-gate-noexl3.log`).

**S3 re-gate DONE 2026-09-18 — S3 CLOSED (row 30).** On the 30-P build (`main` `108d67f`), two replicates of
S3-pre's own arms (`plans/probes/row26_decode.py MODEL=qwen3next BALLAST_GB=25 NEW_TOK=64 ARMS=W,U1,P,U2,NOEXP`,
tags `-s3regate` / `-s3regate2`) measure gate 1, in-run `(construct_miss + gemm_miss)/T_prof`, at **6.96 / 6.99 %
against the ≥ 10 % rule — a clean fail on both** (most generous basis, same numerator over `T_unprof`: 8.59 / 8.67 %;
on the pre-30-P runs it was already 7.06 / 7.19 %). Gate 2 (native CPU kernel) was not run, by the rule. Detail:
[`…-row30-S3-regate-RESULTS.md`](docs/2026-09-18-row30-S3-regate-RESULTS.md) (gitignored; row 30's re-gate bullet
is the durable copy).

**▶ NEXT — the hybrid-model sequence, set by the user 2026-09-18. Work it in this order, in a fresh session; one
self-contained brief covers all three: [`…-hybrid-model-next-steps-BRIEF.md`](docs/2026-09-18-hybrid-model-next-steps-BRIEF.md) (gitignored — the rows are the
durable copies).** Why these: Qwen3-Next is the fastest model on this box (7.34 tok/s at N=1, ~6.5× Qwen3-235B as
served). Its 36 linear-attention layers keep a recurrent state that Vaayu's cache code cannot crop, which already cost
it the prefix cache (RC7). Order rule: correctness first, cheap and no-GPU next, research last.
0. ~~**Row 43**~~ **DONE 2026-09-18 — chunked prefill restarted Qwen3-Next's 36 linear-attention layers at every
   512-token chunk (DEFAULT path; logits rel-L2 0.314 vs a 0.051 floor). FIXED by exact state continuation: 0.041.**
1. ~~**Row 41**~~ **DONE 2026-09-18 — CONFIRMED (speculation on a hybrid emitted garbage from the first drafting round);
   now REFUSED at load on any recurrent cache.** Original text: speculative decoding may corrupt a hybrid model's recurrent state on a rejected draft.
   Confirm with one short GPU run; if confirmed, refuse speculation on recurrent caches, as RC7 does for the prefix cache.
2. ~~**Row 30's OPTIONAL RAM-ceiling observation — price it.**~~ **DONE 2026-09-19 — FIRED (+17.2 % measured un-reserving it) and FIXED: refusal now decided before sizing; +15.1 % decode tok/s on Qwen3-Next at default knobs, server tier 17.26 → 22.20 GB. Qwen3-Next is now ~7.84 tok/s at probe sizing.** Original text: ~4.8 GB held back for prefix sessions this model refuses.
   CPU replay, ~1 h. Close if < +2 % (upper bound: `disk_load` is 8.4 % of `T_prof`).
3. ~~**Row 42**~~ **DONE 2026-09-19 — (a) prefix cache GO: round-2 TTFT 8.6–21× at an 8k context (269 s → 13–35 s),
   snapshot 40.1 MB, restore exact (bit-identical); (b) speculation NO-GO: cold verify 2.6–4.8× a decode token, prompt
   lookup loses on real edits and novel text, any drafter needs p ≥ 0.87 → row 37 closed for hybrids. SIDE FINDING:
   Qwen3-Next prefill is 33 ms/token (2k → 69 s) because the non-EXL3 fused forward re-reads experts per chunk → row
   44.** Original text: SCOPING spike, recurrent-state snapshot/restore; measurement only; go/no-go per use.

**▶ ROW 45 IS CLOSED (2026-09-19). The module-list forward re-read its experts on every prefill chunk, exactly as row 44
predicted by code reading: 3.659× at 2k, 14.216× at 8k. Retention shipped — MiniMax-M2.7 2k TTFT 107.07 → 38.91 s
(2.75×), 8k 430.36 → 78.70 s (5.47×), prompt-end logits bit-identical, decode +0.30 % (null), N=1 on server defaults.
The next step is row 42-B.**

**▶ ROW 44 IS CLOSED (2026-09-19). It shipped F1+F2+F6+F5 and changed the board: Qwen3-Next 2k TTFT 83.51 →
18.13 s (4.61×), 8k 335.26 → 37.20 s (9.01×), decode 7.95 → 9.06 tok/s (+14 %), all at N=1 on server defaults,
logits bit-identical, every gate green.** Commits `85478ef` `c289945` `4d2fa99` `736c4c6`.

**▶ NEXT — the order to work in, updated 2026-09-19 after row 45 closed. Each step is startable cold: its
brief is self-contained and its gate says whether to continue.**

**START HERE IN A FRESH SESSION: row 42-B.** Row 45 is done and shipped. 42-B needs no setup from this
session — read [`…-row42B-hybrid-prefix-cache-BRIEF.md`](docs/2026-09-19-row42B-hybrid-prefix-cache-BRIEF.md)
(or row 42-B's own section below, which carries the same numbers, traps and stop rule and survives a fresh
clone), then run its ~20 min re-measure FIRST. Do not write code before N* is recomputed from measured
numbers — the stop rule can close the row.

**UPDATE 2026-09-20 (LATEST) — THE ROW 36 GATE RE-CHECK IS DONE AND **IT FIRED**: strict `serial / T_unprof` =
**32.34 % / 32.73 %** against its ≥ 30 % rule, two replicates on `main` `da118ab`, spread 1.2 %. **`serial` did NOT
move (0.0363 → 0.0360 → 0.0368 s, flat across three builds, row 44's F5 notwithstanding); the share rose because the
token shrank to 0.1141 s.** **So row 36 — CUDA graphs / fused per-layer dispatch — IS THE NEXT BUILD (~1 week), and
nothing else on this board owns `serial`. Write a plan before any code (CLAUDE.md §4/§7): this measurement says the
phase is LARGE, it does not say what removes it.** The same run also settled row 46's open question: `disk_load`
collapsed to **0.42 / 0.32 %** of `T_prof` (was 8.41 / 8.46 %) with a 96.6 / 96.7 % RAM-cache hit, so **the expert RAM
cache is no longer the binding constraint on Qwen3-Next and every "give the cache more RAM" lever is near-dead there**
(MiniMax unmeasured — do not carry it across). S3's gate 1 re-read 8.15 / 8.25 %, still under 10 % — S3 stays closed.
[`…-row36-gate-recheck-RESULTS.md`](docs/2026-09-20-row36-gate-recheck-RESULTS.md).

**Superseded, kept for the trail — UPDATE 2026-09-20 (after 42-B merged) — START A FRESH SESSION AT THE ROW 36 GATE RE-CHECK, NOT AT 46.**
Cold-pickup brief, self-contained:
[`…-row36-gate-recheck-BRIEF.md`](docs/2026-09-20-row36-gate-recheck-BRIEF.md) (gitignored; rows 36 and 46 are the
durable copies). **It is ~30 min and no code**, and it may promote a ~1-week lever nobody owns: row 36 fires at
`serial / T_unprof ≥ 30 %`, was 26.4 / 26.7 % on the 30-P build with `serial` FLAT at ~0.036 s across two reads, and
was derived to fire once `T_unprof` ≤ ~0.120 s. **Row 44 (+ row 30's RAM-ceiling fix) then took Qwen3-Next decode to
9.06 tok/s = 0.1104 s — under the trigger — which puts the strict share at ≈ 32.6 % if `serial` has not moved.**
It has never been measured post-row-44, and row 44 rewrote exactly the dispatch the NOEXP arm brackets, so it could
have gone either way. **Measure it; do not assume it.** The same run also answers whether the expert RAM cache is
still the binding constraint here, which is what re-priced row 46. Tree clean at `8c5ae12`, suite 2336/16.

**TWO TRACKS from 2026-09-20, and they do not compete — one buys speed, the other buys REACH:**

* **SPEED: 36 (if its gate fires) → 35 (+11.7 % measured, machinery built) → 32 (×1.50 projected, a download and zero
  code; also the only thing that re-opens 37) → 46 (re-priced to ~2.6 %) → 34 (×1.10, probe first, dead on Qwen3-Next).**
* **REACH: row 47, five plans in a fixed order — `A1 → A2 → A3`, then `A4 → A5`. A1, A2, A3 AND A4 ARE ALL
  CLOSED (A1-A3 2026-09-21, A4 2026-09-23); A5 IS THE NEXT ONE TO PICK UP.** A1 (guards) shipped both raises and made them fire on the real
  load path; A2 (fused-3D indexing) shipped unit-tested plus four green on-device gates; **A3 took gpt-oss-20b
  END TO END — it generates through the server (server gate 9 passed) and its logits are BIT-IDENTICAL to stock
  transformers at 4 and 256 tokens.** The abandon gate passed (packed MXFP4 runs here, no bf16 dequant), and
  getting there took SEVEN defects, five of them in code every model runs — see A3's rows below. Cost of the
  reach: **0.17 tok/s, 13.28 GB/token**, because whole-layer delivery moves every expert per token; A6 exists to
  cut that and now has a baseline. **A4 (shared experts) CLOSED 2026-09-23 with its premise REFUTED and zero
  code — shared experts were already pinned, and the compute path was verified to bind on the real module tree;
  it handed A5 two explicit checks instead of a mechanism.** A5 is the DeepSeek-V4 port, weeks,
  gated on sm_120 kernels at step 1. Row **28b** IS A5.
  Program: [`…-architecture-support-PROGRAM.md`](docs/2026-09-20-architecture-support-PROGRAM.md).

**Model set re-set by the user 2026-09-20: Qwen3-235B is RETIRED AND DELETED — do not test it, do not put it in a gate;
the accumulator gate is now ×2, not ×3, and the fused-EXL3 forward has NO on-device model. DeepSeek-V4-Flash is on disk
but INCOMPLETE (46 of 48 shards, no `model.safetensors.index.json`) and still cannot load.** `CLAUDE.md` §0/§5 updated
the same day; `test_generate.py`'s hardcoded default repointed from the deleted checkpoint to Qwen3-Next.

**Superseded, kept for the trail — ROW 42-B IS CLOSED AND SHIPPED**, and it
re-priced row 46 DOWN to ~2.6 % (see its row). 42-B measured: turn-2 TTFT 30.599 → 4.902 s (**6.24×**), decode
9.7364 → 9.4868 tok/s (−2.56 %), generated ids sha256 identical across all four ABBA arms, suite 2334/16,
every on-device gate green, **N\* = 9,512 generated tokens/turn against a ~500 stop rule**. Branch
`row42b-hybrid-prefix-cache`, commits `6cac130..13f804e`.

**UPDATE 2026-09-20 (earlier, kept for the trail) — the re-measure IS DONE and the stop rule did NOT fire, so the next step is the BUILD PLAN,
not the probe.** Measured on HEAD `5b9d898`: MISS 2k **16.45 s** / 8k **35.05 s** (row 42's 269.3 s 8k miss is
7.7× gone), HIT flat in P at **3.92 / 3.97 s** for d=64 → **N\* = 1862 / 1635 / 1318 at 8k** and **751 / 576 /
300 at 2k** for d = 64 / 256 / 1024; only the 2k/d=1024 corner is under the ~500 close threshold. The build
plan's first job is the RESERVATION: `derive_expert_ceiling` takes 4.83 GB for 3 full-`max_seq_len` sessions
while an 8k session holds 0.20 GB — an 8× over-reservation that IS the −13 % decode tax.
[`…-row42B-reprice-RESULTS.md`](docs/2026-09-20-row42B-reprice-RESULTS.md).

| # | row | what it is | start gate | stop rule | cost | brief |
|---|---|---|---|---|---|---|
| ~~1~~ | ~~**45**~~ | **DONE 2026-09-19 — it fired (3.659× at 2k, 14.216× at 8k) and the retention shipped: 2k TTFT 107.07 → 38.91 s (2.75×), 8k 430.36 → 78.70 s (5.47×), logits bit-identical, decode null, every gate green.** Original text: **MEASUREMENT FIRST, no code.** Does the module-list forward (MiniMax-M2.7) re-read its experts per prefill chunk, the way the fused one did? It is the ONLY one of row 44's four levers that this forward still lacks — F5's dispatch and F6's overlap are already in it, and F1 never applied (EXL3 pools were always 8). Nothing amortises it either: the prefill stager is EXL3-**and-fused**, so MiniMax never gets one. | none — start here | `routed_sum / routed_union` at 2k **< 1.5×** → close the row with the number | ~½ day to measure, ~1–2 days if it fires | [`…-row45-modulelist-chunk-retention-BRIEF.md`](docs/2026-09-19-row45-modulelist-chunk-retention-BRIEF.md) |
| ~~1~~ | ~~**42-B**~~ | **DONE 2026-09-20 — shipped: round-2 TTFT 6.24× (30.60 → 4.90 s), decode −2.56 %, ids bit-identical, N\* = 9,512 vs a ~500 stop rule, all gates green.** Original text: | **RE-MEASURE, THEN PLAN, THEN BUILD** the hybrid cross-turn prefix cache. Row 42 proved the mechanism (40.1 MB snapshot, exact restore) but priced it against a 269 s 8k miss that is now **37.2 s**. | **open now — row 45 is closed** | recompute `N* = (T_miss − T_hit) / (1/7.87 − 1/9.06)`; **N\* below ~500 generated tokens/turn → close 42-B** | re-measure ~20 min · plan ½ day · build ~2–3 days | [`…-row42B-hybrid-prefix-cache-BRIEF.md`](docs/2026-09-19-row42B-hybrid-prefix-cache-BRIEF.md) |
| ~~1~~ | ~~**36 gate re-check**~~ | **DONE 2026-09-20 — IT FIRED: strict 32.34 / 32.73 % vs ≥ 30 % (`main` `da118ab`, 2 replicates, spread 1.2 %). `serial` FLAT at 0.0368 / 0.0374 s across three builds; the share rose because the token shrank to 0.1141 s. → **row 36 is now the next BUILD, ~1 week, plan first.** Same run: `disk_load` 8.41 → 0.42 %, RAM hit 96.6 % → row 46's mechanism is near-dead on Qwen3-Next; S3 gate 1 = 8.15 / 8.25 %, still closed.** | — | — | done, ~30 min | [`…-row36-gate-recheck-RESULTS.md`](docs/2026-09-20-row36-gate-recheck-RESULTS.md) |
| ~~1~~ | ~~**36**~~ | **PARKED 2026-09-20 — gate fired, then two spikes killed every lever.** `serial` is **74.8 % GPU-bound**; its matmul half (12.59 ms) is at **80–86 % of the RTX 5060 Ti's 448 GB/s**, a floor; `use_torch_compile` **CRASHES** (cudagraph-trees step boundary, reproduced 2×); without cudagraphs it is **24–35 % SLOWER** (46 dynamo graphs — broken by the engine's own `time.perf_counter` instrumentation and by `aten.bincount`). Elementwise term still real but **not ~1 day**. | — | — | done, ~1 day | [`…-SPIKE.md`](docs/2026-09-20-row36-serial-split-SPIKE.md) · [`…-torchcompile-AB-RESULTS.md`](docs/2026-09-20-row36-torchcompile-AB-RESULTS.md) |
| **1 (NEXT — SPEED)** | **49** | **STEP 2A SHIPPED 2026-09-20 (+6.90 % decode); B RE-RANKED AND REFUTED 2026-09-21 (<= +6 %).** The probe that killed B also priced its successors on the same run: **the drive is busy 75.7-76.4 % of a MiniMax decode token at 6.03-6.05 GB/s, idling 1.57 ms in EVERY layer** — **BOTH SUCCESSORS ARE NOW SPENT OR IN DOUBT (2026-09-21).** (a) The ring-pool / read-shape arm ran and the rate hypothesis was BACKWARDS: `VAAYU_RING_POOL_N=4` REGRESSES -2.07 %, the optimum is **n=2, shipped at +1.89 % decode / +1.85 % 2k prefill** — and the read-shape half was then BUILT AND REVERTED, because removing **66.7 % of the reads** and raising delivered bandwidth **6.07 -> 6.50 GB/s** bought **-0.40 % decode**. That is a direct experiment refuting the **<= +12.8 %** rate lever. (b) The per-layer gap's **>= +31 %** came from the same "token collapses onto disk-busy time" arithmetic the read-shape arm just falsified (busy fell 5.4 pp, token did not move, and the >= 1 ms gap COUNT was unchanged) — do not quote it as measured. Its built form, `VAAYU_LOOKAHEAD`, is **+2.5 % on MiniMax at N=1** (the -3.7 % is the N=3 arm the server never runs) and costs **+18.0 % bytes/token**, which alone caps the token floor near **+11 %**. Original text: **MiniMax-M2.7's expert H2D is SYNCHRONOUS on purpose** (`moe_patch.py:1110`/`:1119`, `non_blocking=False`), so it cannot hide behind the disk read the way the fused forward's does. Row 26 measured it at **0.180 s of a 0.596 s token — 30.2 %, the largest removable phase on that model**, and ORDERED this priced before any CPU path. Never done. **Bigger than the CPU hybrid's whole 0.114 s prize, and needs no CPU kernel or new on-disk format.** | open now — row 26's precondition 1 (after row 27) was met 2026-09-12 | **step 1: `h2d_pct` < ~15 % of `T_prof` → close the row AND the CPU-hybrid half with the number** | step 0 ~1 h · step 1 ~1 h · step 2 1–3 days | [`…-row49-minimax-h2d-BRIEF.md`](docs/2026-09-20-row49-minimax-h2d-BRIEF.md) · row **49** below |
| **1b (REACH track — user mandate, NON-NEGOTIABLE 2026-09-20)** | **47 = A1→A2→A3, then A4→A5** | **Vaayu must serve different architectures.** Decomposed into 5 plans. ~~**A1** guards~~ · ~~**A2** fused-3D indexing~~ · ~~**A3** gpt-oss-20b end to end~~ **(all three CLOSED 2026-09-21; gpt-oss generates through the server, bit-identical to stock)** · ~~**A4** shared-expert pinning~~ **(CLOSED 2026-09-23, premise REFUTED, zero code — already pinned, 48/48 layers, 4.83 GB)** · **A5** DeepSeek-V4 port (NEXT). | **A5 open now** (A1-A4 closed), no dependency | A5 has an abandon gate at its first task; A3's PASSED (packed MXFP4 runs here) | A1 ~1d · A2 ~1d · A3 ~3–5d · A4 **closed in ½ day, no build** · A5 weeks | [`PROGRAM`](docs/2026-09-20-architecture-support-PROGRAM.md) · row **47** below |
| ~~2~~ | **35** | ~~Per-model measured gate for `VAAYU_LOOKAHEAD` instead of the hard block.~~ **BLOCKED 2026-09-20 — DO NOT PICK.** Its +11.7 % was measured on **Qwen3-235B, deleted 2026-09-20**, so its stop rule (*reproduce +11.7 % on Qwen at N=1 through the server*) is **unrunnable**. Its own kill condition (*dies if S2 lands*) **fired 2026-09-13**, and Qwen3-Next's disk is now 0.42 % of a token. The only still-streaming model, MiniMax, is the **−3.7 %** arm — **but that is its N=3 arm, and the server runs N=1, where the same post-P4 measurement reads +2.5 %** (`check_lookahead_flag_safe`). **CLOSABLE ON ARITHMETIC 2026-09-21:** at +18.0 % bytes/token against a drive busy 76 % of a MiniMax decode token, the token floor is ~+11 % — **below this row's own ≥ +15 % re-open bar**, so the bar cannot be met by this predictor. | **blocked** | re-opens only on a measured **≥ +15 % on MiniMax at N=1** on a current build | — | row **35** below |
| **3** | **32** | Lower-bpw EXL3 build (5.01 → ~3.0 bpw) of a model already on disk. | open | **×1.50 projected, a download and ZERO code**; under ~3.25 bpw Qwen's working set fits its VRAM budget, and it is the only thing that re-opens row **37** | a download | row **32** below |
| **4** | **46** | **Size the prefix reservation to what the pool holds.** Split out of 42-B 2026-09-20. `derive_expert_ceiling` takes 3 × `MAX_SEQ_LEN` × kv/token — **4.74 GB on Qwen3-235B, 6.24 GB on MiniMax, live today** — while a stored 8k session holds 0.20 GB (8× over; 32× at 2k). ~~Row 30 measured returning 4.83 GB as +15.09 % decode~~ — **RE-PRICED 2026-09-20 by 42-B's own A/B: returning 4.83 GB (ceiling 13.78 → 18.77 GB) moved decode only −2.56 %, so the prize is ~2.6 %, not ~15 %. Row 30's number predates row 44.** | open (needs no hybrid) | row 31's 7.4k two-session arm must not collapse (it measured −8.9 % / match 7422→44 for the naive `PREFIX_SESSIONS=1` version) | price offline ½ day · build ~1–2 days | row **46** below |
| **3** | 44's **F3 remnant** | pinned-H2D half only (~0.6 s of an 18 s 2k prefill). **Not recommended** — listed so nobody re-derives it. | — | — | ~1 day | row **44** below |

**Why 45 went first (settled — it is closed):** 45 was the same defect class row 44 had just fixed, on the second
actively-tested model, and cheap to answer (one instrumented prefill). It fired at 3.659× and shipped a 2.75× / 5.47×
TTFT win. 42-B is the bigger feature but its whole case has to be re-measured first, and its cost — the prefix-session
reservation returning ~4.83 GB to the KV side — now bites harder against a 37 s miss than it did against a 269 s one.

**After those, on the user's call:** row **40** `[bug]` (arena never capped at the model's
expert bytes — latent, 0 measured impact on this box's models; shared sizing → all three accumulator gates); row
**36** (CONDITIONAL — strict `serial/T` now **26.4 / 26.7 %**, derived to fire once `T_unprof` ≤ ~0.120 s); PRIORITY
rows **32 · 34 · 35** and **28b**.

## PRIORITY — worth working, in this order

| # | one line | why it is worth it | state | cost | brief (gitignored) |
|---|---|---|---|---|---|
| ~~**3**~~ | **CLOSED 2026-09-11 — prefix cache now ON for every non-EXL3 model.** `auto` resolves to `bf16` (the ring buffer, model's own dtype) on every loader. | Done: 1B prefix gate + server gate pass, turn-2 TTFT ~4.5× at 2.8k tokens, decode flat on flash. **Cost: derived `max_seq_len` drops up to ~2× (1B flash 91106 → 65536)** — row 4's doubling overshoot, now paid by non-EXL3 too; `KV_CACHE_MODE="fp16"` is the escape hatch. | `[x]` | done | [`…-row3-auto-bf16-prefix-cache-RESULTS.md`](docs/2026-09-11-row3-auto-bf16-prefix-cache-RESULTS.md) · row **3** below |
| ~~**25**~~ | **CLOSED 2026-09-11 — APST is NULL on both MoE targets.** MiniMax idle-sampler arm: decode 0 gaps ≥ 100 ms; cold prefill 3 mid-generation gaps ≥ 100 ms + a 80–100 ms band (same band Qwen has). | Done: bound ≤ 0.15 % of cold TTFT (≤ 20 wakes × ≤ 15 ms, the kernel's primary-state latency cap), 0 on decode; no full A/B — even a 1 % worst case is inside the A/B's replicate spread. The 70B dense stays unmeasured (not a blocker). | `[x]` | done | [`…-row25-apst-ab-RESULTS.md`](docs/2026-09-11-row25-apst-ab-RESULTS.md) (MiniMax addendum) · row **25** below |
| ~~**18**~~ | **CLOSED 2026-09-11 — `output_hidden_states=True` fixed; `output_attentions=True` now raises `NotImplementedError`.** Root cause confirmed by instrumentation: 19 sublists + 19 bare Tensors (outer len 38 = 2 × `len(self.layers)`), from `6253fd1` switching the init to sublists without removing the flat `+=`. | Done: returns transformers 5.x's `num_hidden_layers + 1` entries (post-norm last); matches stock HF to ≤ 0.0184 per-token rel-L2 vs ≥ 0.459 to a neighbour layer; new on-device gate `tests/test_output_hidden_states.py` fails a shifted-layer mutant. 0 % on throughput (diagnostic only). | `[x]` | done | [`…-row18-output-hidden-states-RESULTS.md`](docs/2026-09-11-row18-output-hidden-states-RESULTS.md) · row **18** below |
| ~~**26**~~ | **CLOSED 2026-09-11 — 8-bit + AVX2 proposal KILLED; bytes-neutral CPU hybrid has a real mechanism but a modest ceiling (+6.8 % Qwen / +22.0 % MiniMax over row 27's baseline) → design brief only, after row 27.** 8-bit's disk floor alone is slower than today on both (−9.6 % / −15.5 % tok/s). | Done: measured floor = serial (stubbed-expert arm, 0.074 / 0.052 s) + device-busy (0.772 / 0.396 s). The brief's formulas implied times 21 % / 9 % BELOW that floor. Found row 27 on the way. | `[x]` | done | [`…-row26-cpu-hybrid-ceiling-RESULTS.md`](docs/2026-09-11-row26-cpu-hybrid-ceiling-RESULTS.md) · row **26** below |
| ~~**28**~~ | **CLOSED 2026-09-12 — the new non-EXL3 model does NOT rescue the CPU-AVX2 hybrid; it undercuts it. `deepseek-ai--DeepSeek-V4-Flash-0731` (155.42 GiB) is fp4 **4.2500 bpw**, not 8-bit, and Vaayu CANNOT LOAD IT** (transformers 5.5.4 has no `deepseek_v4`; repo ships no HF remote code; TileLang fp4/sparse-attn/Sinkhorn kernels absent). Hybrid projected **+4.5–5.0 %** over row 27's baseline vs row 26's +6.8 % Qwen / +22.0 % MiniMax → **close the CPU-hybrid idea.** | Real prize found instead: **258 slots/token × 12.750 MiB = 3.212 GiB/token vs Qwen's 8.285** → projected **2.1–2.3 tok/s (~2.5–2.7× Qwen's measured 0.85)**, from architecture (top-6 of 256 over 43 layers), not from the CPU. Unblocked only by an architecture PORT (weeks). Its fp8 sibling build would cost ~43 % of that — row 26's bytes rule, reproduced non-EXL3. | `[x]` | done | [`…-row28-nonexl3-dsv4-cpu-hybrid-VERDICT.md`](docs/2026-09-12-row28-nonexl3-dsv4-cpu-hybrid-VERDICT.md) · row **28** below |
| ~~**27**~~ | **CLOSED 2026-09-12 — BUILT, MEASURED THROUGH THE SERVER, SHIPPED ON. +23.1 % decode tok/s on Qwen3-235B at the server's real default** (0.9229 → 1.1363 tok/s, n=3, ids identical, 0.5 % replicate spread); TTFT 1.680 → 1.306 s (−22.3 %). Cache-off sweep: spinAll +12.4 %, pin alone +0.9 %, both +14.1 % decode / +10.1 % served. No regression: MiniMax-M2.7 +1.3 %, Llama-3.3-70B dense +2.9 %, Llama-3.2-1B null (VRAM-pinned, reads no disk — the control). Mechanism confirmed in the kernel's counters: C2_ACPI residency 1.142 → 0.492 s, wakes 1.74 M → 1.22 M. | Knobs `KEEP_AWAKE="auto"` + `PIN_ENGINE_THREADS=True`, plumbed and documented; reachable at N=1. Stop rule (≥ +10 % on Qwen) PASSED. | `[x]` | done | [`…-row27-keepawake-RESULTS.md`](docs/2026-09-12-row27-keepawake-RESULTS.md) · row **27** below |
| ~~**29**~~ | **Pick a non-EXL3 MoE Vaayu can LOAD, then measure the hybrid on it.** EXL3 has no CPU kernel, so non-EXL3 is a PREREQUISITE for CPU+GPU decode — row 28's +4.2 % was DeepSeek-V4-Flash's disk-streamed regime, not a verdict on non-EXL3. Best hybrid case on this box is **+21 %** (Qwen3-Next-80B int4); gpt-oss-120b actually **LOSES −4.6 %**. | The far bigger number is the download itself: **gpt-oss-120b ≈ 11 tok/s (13× today, zero new kernels)**, GLM-4.5-Air ≈ 7.3 (9×), Qwen3-Next-80B ≈ 18 (21×) vs Qwen3-235B EXL3's measured 0.85. Step 2 would be the FIRST non-EXL3 MoE ever run on device. Only int4-g128 has a torch CPU kernel (row 26: 214 µs/expert); MXFP4 and fp8 have none. | `[x]` decision taken 2026-09-12: **Qwen3-Next-80B-A3B-Instruct int4**, smoke rung Qwen3-30B-A3B int4 → row **30** | 1 day + a download | [`…-row29-hybrid-model-selection-BRIEF.md`](docs/2026-09-12-row29-hybrid-model-selection-BRIEF.md) · row **29** below |
| **31** | **Stop spending the expert RAM cache on context nobody is using.** `derive_expert_ceiling` = `available − 3.5 GiB − (PREFIX_SESSIONS × MAX_SEQ_LEN × kv_bytes/token)`, so the KV reservation is subtracted from the expert cache. Both terms are ALREADY server knobs. Sweep them. | **Zero code, zero download, today.** The reservation costs Qwen **4.74 GB** and MiniMax **6.24 GB** of a 15.2 GB budget at the defaults (3 × 8192). An on-device sweep already on record (`vaayu_base.py:286-291`) measured **−25 % warm wall** from 8.5 → 18 GiB of ceiling. Two-sided: prefix reuse is worth 4.44× and collapses to ZERO one session past capacity (Plan I), so this is a single-agent-vs-fan-out trade, not a free win. Price it offline first with `vaayu/expert_sim.py`. | `[x]` **CLOSED 2026-09-12 — defaults STAY** | done | [`…-speedup-options-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md) §2.1 · row **31** below |
| **30** | **S3 CLOSED 2026-09-18 by its re-gate on the 30-P build: gate 1 `(construct_miss + gemm_miss)/T_prof` = 6.96 / 6.99 % < 10 %; every S-step of this row is now closed.** — **S2b (iii) BUILT AND SHIPPED 2026-09-13 — the VRAM expert cache now reaches the non-EXL3 fused forward: `4.1584 → 6.4702 tok/s, +55.6 %` at N=1 on server knobs (same-day baseline arm, greedy ids identical, 83 % VRAM hit). Stop rule was ≥ +20 %. Peak VRAM 11.66 GiB — 3.17 GiB free against a 3.50 GiB reserve floor, tight, recorded. RC4(c) DONE 2026-09-13 (max_seq_len 16384 -> 65536, 0 % tok/s). **S3-pre DONE 2026-09-13: NEITHER gate fires cleanly and the gate had been read the wrong way — `construct`/`gemm` run on BOTH sides of a VRAM hit, so the literal reading counts work S3 cannot take. ADDRESSABLE `(h2d+construct_miss+gemm_miss)/T` = 15.1 / 15.2 % (literal 20.8 / 21.1 %); row 36's `serial/T` = 23.0 / 23.4 % strict, 32.5 / 32.8 % wide. S3 RECOMMENDED CLOSED — the VRAM cache's only fill site IS the miss branch S3 removes, so its 15 % is self-defeating (that mechanism objection was later dissolved by S3-fill's fill-behind, 2026-09-18; S3 then closed on its re-gate NUMBER). NEXT WAS T2 / row 33 (`top_k` 10→8 = +14.7 %, →6 = +34.2 %, derived from measurement).** — S2 CLOSED 2026-09-13 — `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` RUNS: `4.0648 tok/s` warm at N=1 on server knobs, server gate green (9 passed), the FIRST non-EXL3 MoE ever executed on this box.** Seven root causes fixed (RC1 dequant · RC3 shared expert · RC4(a)(b) hybrid KV · **RC5 the splitter dropped tensors** · **RC6 `get_seq_length()`=0 on hybrid, 9.5×** · **RC7 prefix cache refused on recurrent state**). ~~Remaining: S3 only~~ — **S3 CLOSED 2026-09-18** (re-gate, above); 30-P DONE 2026-09-18 (+11.69 %). **S2b CLOSED 2026-09-16: (iv) re-priced at 0.33 % of `T_prof` and DROPPED** (`docs/2026-09-16-row30-S2b-iv-reprice-RESULTS.md`); T2 / row 33 closed 2026-09-13. | **7.6× Qwen3-235B-EXL3's raw 0.85 / 5.7× row 27's served 1.1363** (was 4.8× / 3.6× at S2's close) — the largest measured win on this board. NOT row 29's projected 18.1; re-derive from the measurement. **Phase-4 gate, re-read correctly 2026-09-13 (S3-pre): 15.1 / 15.2 % addressable, ON the 15 % rule — and S3 is recommended closed on mechanism, not on the number.** Disk fell to **1.7 %** of a token, so **T3 is dead here**; T4's row 36 straddles its own rule (23.4 % strict / 32.8 % wide), so **T2 / row 33 goes first**. | `[x]` **CLOSED 2026-09-19 — last item (RAM-ceiling observation) FIXED: +15.1 % decode tok/s on Qwen3-Next.** (was: priority 2 — S1, S2, S2b, RC4(c), S3-pre, 30-P and S3 all DONE; its `[bug]` arena cap is row 40) | **NO download** | [`…-row30-S2b-iii-vram-cache-RESULTS.md`](docs/2026-09-13-row30-S2b-iii-vram-cache-RESULTS.md) · [`…-S2-load-path-RESULTS.md`](docs/2026-09-13-row30-S2-load-path-RESULTS.md) · row **30** below |
| **33** | **Truncate `top_k` at decode.** `bytes_read ∝ top_k`, and `top_k` enters the engine at exactly one place (`vaayu_base.py:3233`), so clamping it there shrinks routed slots AND every derived budget coherently, on all three forwards. **Lossy — the user authorised lossy 2026-09-12.** | Projected **×1.26 Qwen / ×1.21 MiniMax at k=8→6**, ×1.71 / ×1.54 at k=4, holding hit rate constant (conservative). **Regime-independent** — it pays streamed and resident. Two traps: contract-routed models (MiniMax's `route_tokens_to_experts`) take NO `k` and read the module's own `top_k`, so the clamp must be written into each MoE module by detection or the model keeps selecting 8 against budgets sized for 6 (renormalisation already exists on every path — corrected 2026-09-12); and prefill routes ~every expert anyway (224.23/layer) so decode-only is the defensible scope. | `[x]` **CLOSED 2026-09-13 — shipped OFF; Qwen3-Next k=9 = +17.0 % for +1.47 % ppl, MiniMax has no viable k** | ~1 day + a ppl sweep | [`…-NEW-LEVERS-HANDOFF.md`](docs/2026-09-12-speedup-options-NEW-LEVERS-HANDOFF.md) §2.3 · row **33** below |
| **30-P** | **DONE 2026-09-18 — pinned RAM tier (slab arena) for the non-EXL3 fused forward.** Replaced the forward's inline pageable-clone loader with `_dispatch_expert_misses`; the arena gate now accepts fused non-EXL3 on CUDA. | **+11.69 % decode tok/s measured on branch `row30p-pinned-ram-tier` on Qwen3-Next (6.6329 → 7.4084, ABBA, greedy ids identical × 12 × 64 tokens)** — beats the probe's own +10.62 %. No knob, reachable at N=1 (measured there), bit-identical, VRAM hit/peak unchanged (83 %, Δ ≈ 0.03 GiB). Server-sizing arena: 10380 × 1.8MB, 18.54 GB pinned. It is also the hard prerequisite of any S3 fill-behind — unblocked now. | `[x]` **DONE — merged to `main` 2026-09-18 (`8475123`); the S3 re-gate it unblocked ran the same day and CLOSED S3** | ~1–2 days | [`…/plans/2026-09-18-row30P-pinned-ram-tier-nonexl3-fused.md`](docs/superpowers/plans/2026-09-18-row30P-pinned-ram-tier-nonexl3-fused.md) · spec [`…-row30-S3-fill-site-SCOPING.md`](docs/2026-09-18-row30-S3-fill-site-SCOPING.md) · [`…-row30P-pinned-ram-tier-RESULTS.md`](docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md) · row **30** below |
| **41** | **`[bug]` SUSPECTED (code reading 2026-09-18, not reproduced): speculative decoding on a hybrid model rolls back only the KV layers.** `_trim_kv_cache` → `crop()` (`vaayu_base.py:5171` / `:1544`) shortens KV lengths; the 36 linear-attention layers' recurrent state keeps the rejected draft tokens → silently wrong text. | Correctness. Reachable only by opt-in: `USE_PROMPT_LOOKUP=True` (server knob) with Qwen3-Next. `"auto"` = off on MoE, so the default is safe. Fix if confirmed is small: refuse at load via `cache_holds_recurrent_state`, as RC7 does. | `[x]` **CLOSED 2026-09-18 — confirmed, refused at load; its root cause also broke chunked prefill (row 43, FIXED)** | ~½ day | [`…-hybrid-model-next-steps-BRIEF.md`](docs/2026-09-18-hybrid-model-next-steps-BRIEF.md) item 1 · row **41** below |
| **42** | **SCOPING spike: recurrent-state snapshot/restore for hybrid models.** Would unblock the cross-turn prefix cache (RC7 refused it) AND speculative decoding (row 37) on Qwen3-Next. | Qwen3-Next is the fastest model here. Prefix reuse is the agent workload's top TTFT lever (CLAUDE.md §0). Row 37's gate is MET (S2, 2026-09-13), and the checkpoint ships a free MTP drafter. One snapshot ~38–75 MB (derived). | `[x]` **CLOSED 2026-09-19 — (a) GO (8.6–21× round-2 TTFT at 8k) → row 42-B; (b) NO-GO (verify 2.6–4.8× t_dec, p ≥ 0.87 needed) → row 37 closed for hybrids; found row 44** | done | [`…-row42-hybrid-state-snapshot-SCOPING.md`](docs/2026-09-19-row42-hybrid-state-snapshot-SCOPING.md) · row **42** below |
| **44** | **Qwen3-Next prefill is 33 ms/token: the non-EXL3 fused forward re-reads its experts on every 512-token prefill chunk, pageable, never cached.** A 2048-token prefill read **96.1 GB (2.19× the model's experts, 53,938 loads) at ~1.13 GB/s**; RAM-tier hits do not change the rate (row 41: 53–66 % hits, same ~31 ms/token) → per-expert loop cost, not NVMe. The `_chunk_major_gpu` retention exists only in the EXL3 fused forward. | TTFT on the fastest model here: **2k → 69 s, 8k → 269 s, 16k ≈ 540 s (derived)** — never measured before row 42. Derived ceiling ~5–10 s for 2k (load each expert once, RAM-tier hits, NVMe at ~7 GB/s). Every Qwen3-Next request, N=1, default config. Also shrinks the hit path of 42-B. | `[x]` **CLOSED 2026-09-19 — F1+F2+F6+F5 shipped: 2k TTFT 83.51 → 18.13 s (4.61×), 8k 335.26 → 37.20 s (9.01×), decode +14 % (7.95 → 9.06 tok/s), logits bit-identical, all gates green.** Four causes: the loader pool was 1 worker on every non-EXL3 ckpt (512 MB buffer guess); the forward retained nothing across chunk calls (3.09× at 2k, 10.9× at 8k); it waited for every read before computing anything; and it was the last forward dispatching by int64 one-hot mask with per-expert `torch.where` (~960 tiny H2D copies per decode token). F4 (coalesced reads) REVERTED on its stop rule; F3 reduced to ~0.6 s | ~1–2 days | [`…-row42-…-SCOPING.md`](docs/2026-09-19-row42-hybrid-state-snapshot-SCOPING.md) §4 · row **44** below |
| **45** | **Does the module-list forward (MiniMax-M2.7) re-read its experts per prefill chunk too?** Found by row 44 by code reading, UNMEASURED. Row 44's F1 does not apply (MiniMax's pool was always 8) and F2 fixed the other forward. | At 512-token chunks MiniMax routes ~all 256 of a layer's experts (4096 slots, top-8), so the factor should be ~the chunk count (4× at 2k, 16× at 8k) on a disk-bound prefill reading **141 GB of experts, 2.28 GB per layer**. **Nothing amortises it: the prefill stager is EXL3-AND-fused, so this model never gets one** (the earlier "the stager may cover it" guess is refuted). Only F2 is missing — F5's dispatch and F6's overlap are already in this forward. **Measure first** (loads vs per-layer union, `row44_prefill_split.py MODEL=…`); gate < 1.5× → close. | `[x]` **CLOSED 2026-09-19 — MEASURED, BUILT, MEASURED, ALL GATES GREEN.** Gate FIRED: `routed_sum/routed_union` = **3.659× at 2k, 14.216× at 8k** (55,871 / 220,473 loads vs unions of 15,268 / 15,509; 495.86 / 1956.70 GB read; stop rule was < 1.5×). **The cost was NOT the disk** — h2d was 59.8 % of the wall, exposed `disk_load` 1.7 %; the NVMe ran at 4.63 GB/s fully hidden by F6, and the RAM cache is empty all prefill by design (`moe_patch.py:2098`). **RETENTION SHIPPED: 2k TTFT 107.07 → 38.91 s (2.75×), 8k 430.36 → 78.70 s (5.47×)**, loads land exactly on the per-layer union, bytes 495.86 → 135.50 / 1956.70 → 137.64 GB, **prompt-end logits BIT-IDENTICAL at both lengths**, decode **+0.30 %** (null, inside spread), +1.48 GiB peak VRAM against an 8.8 GB budget. §8: no knob, nothing to plumb, **reachable at N=1 on server defaults — every MiniMax prefill over one chunk**. Suite 2318 / 16 skipped; gates accumulator MiniMax 8, Qwen3-235B 8, Qwen3-Next 8, server Qwen3-Next 9, 1B prefix 1. brief [`…-row45-…-BRIEF.md`](docs/2026-09-19-row45-modulelist-chunk-retention-BRIEF.md) · [`…-RESULTS.md`](docs/2026-09-19-row45-modulelist-chunk-retention-RESULTS.md) ||
| **42-B** | **Build the hybrid-model prefix cache** (write the build plan first, then build). Snapshot the recurrent state at the prompt end (40.1 MB, 2.5 ms D2H); append-only turns need no snapshot; lift RC7 for hybrids. | Row 42 measured 8.6–21× round-2 TTFT at 8k **against a 269 s miss that row 44 cut to 37.2 s**, so the case must be re-measured on HEAD (both sides moved; the HIT side is mostly a prefill of the diverged tail and got faster too). **Cost is unchanged: the prefix-session reservation returns 4.83 GB → ~−13 % decode** (9.06 → ~7.87), so the break-even falls from ~13,300 generated tokens/turn to **~1,450–1,960** — it now LOSES on long-output turns. Decide the reservation sizing in the plan, not just RC7's lift. | `[x]` **CLOSED 2026-09-20 — round-2 TTFT 30.60 → 4.90 s (6.24×), decode −2.56 %, ids bit-identical, N\* = 9,512 vs a ~500 stop rule, all gates green.** | re-measure 20 min · plan ½ day · build ~2–3 days | [`…-row42B-…-BRIEF.md`](docs/2026-09-19-row42B-hybrid-prefix-cache-BRIEF.md) · row **42-B** below |
| **32 · 34 · 35** | **The streaming-regime ladder — fallback if S2 stalls.** **32** run a lower-bpw EXL3 build of the models already here (5.01 → 3.0 bpw). **34** find out why the drive delivers 6.3 GB/s when it plateaus at 7.1. **35** give `VAAYU_LOOKAHEAD` a per-model measured gate instead of a hard block. | 32 = **×1.50 projected, a download and zero code**, and below ~3.25 bpw Qwen's working set drops under its VRAM cache budget so the 62/94-layer restriction disappears entirely. 34 = ×1.10, probe first. 35 = **+11.7 % Qwen measured**, gate machinery already built. **All three only pay while experts stream from disk — if S2 lands they mostly die.** | `[ ]` **priority 4** | 32: a download · 34: ½ day probe · 35: ~2 days | §2.2 / §2.4 / §2.5 · rows **32**, **34**, **35** below |
| **28b** | **Port DeepSeek-V4-Flash-0731** (step S4) — kernels first on sm_120, then HF modeling code + `auto_map` the MiniMax way, key remap, persist, measure. | Projected **~2.5–2.7× Qwen's decode** from 2.58×-lighter tokens. **Weeks**, mostly kernels and numerics. Its disk keys already satisfy the streaming index — the hard part is the model. | `[ ]` **priority 6** | weeks | [`…-row28b-IMPL-deepseek-v4-port.md`](docs/2026-09-12-row28b-IMPL-deepseek-v4-port.md) · row **28b** below |

## CONDITIONAL — real levers with a stated gate, added 2026-09-12. Not optional, not yet startable.

| # | one line | gate that opens it | why it is worth it then | state |
|---|---|---|---|---|
| **36** | **CUDA graphs / fused per-layer dispatch.** | **a resident model measures `serial / T` ≥ 30 %** — **MEASURED 2026-09-13 (S3-pre): 23.0 / 23.4 % strict (NOEXP), 32.5 / 32.8 % wide (`other`/T). Straddles the rule; does NOT fire strictly.** **Re-read 2026-09-18 on the 30-P build (S3 re-gate runs): 26.4 / 26.7 % strict, 35.9 / 35.9 % wide — `serial` unchanged at ~0.036 s, the share rose only because the token shrank; still does NOT fire strictly (derived: fires once `T_unprof` ≤ ~0.120 s).** **RE-CHECK IS NOW THE NEXT TASK ON THE BOARD (2026-09-20): row 44 + row 30's RAM-ceiling fix took Qwen3-Next decode to 9.06 tok/s = `T_unprof` 0.1104 s, UNDER the trigger, so the strict share is ≈ 32.6 % if `serial` still sits at ~0.036 s — but row 44 rewrote the very dispatch the NOEXP arm brackets (F5 replaced a per-expert `torch.where` one-hot mask, ~960 tiny H2D copies/token), so `serial` may have moved either way. MEASURE IT, do not estimate — the row says so. Brief: [`…-row36-gate-recheck-BRIEF.md`](docs/2026-09-20-row36-gate-recheck-BRIEF.md).** | `serial` is 0.074 s of Qwen's 0.930 s today — 8 %, invisible. Row 29 projected it becomes the LARGEST phase once experts are resident (38 ms of Qwen3-30B-int4's projected 49 ms). Nothing owns it today. | `[ ]` |
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

**Closed 2026-09-11 (kept here until the next prune):** row **18** (`output_hidden_states` fixed, `output_attentions`
refused — [`…-row18-output-hidden-states-RESULTS.md`](docs/2026-09-11-row18-output-hidden-states-RESULTS.md)), row **3** (`auto` → `bf16` for non-EXL3, prefix cache ON —
[`…-row3-auto-bf16-prefix-cache-RESULTS.md`](docs/2026-09-11-row3-auto-bf16-prefix-cache-RESULTS.md)), row **13** (`even-4` built, done condition met,
0 % on the default config — [`…-row13-even4-RESULTS.md`](docs/2026-09-11-row13-even4-RESULTS.md)) and
row **25** (APST NULL on Qwen by A/B and on MiniMax by idle-gap bound, remedy fix `c256e18` —
[`…-row25-apst-ab-RESULTS.md`](docs/2026-09-11-row25-apst-ab-RESULTS.md)).

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
(the drive's idle gaps are < 100 ms, APST's timeout). The boot prints it as info, not a warning; see row 25.

**Before proposing ANY perf work, read SETTLED.** It is the list of things already built and
refuted, with the number that killed each one.

---

# ▶▶ IMPLEMENTATION SEQUENCE — added 2026-09-12, the ORDER to build in

**This section is the build order. The PRIORITY table above is the priority; this is the sequence, with the gate that
lets each step start.** Every step names a stop rule. Do not start a step whose gate has not reported.

| # | step | row | cost | gate that lets it start | stop rule |
|---|---|---|---|---|---|
| ~~**S0**~~ **DONE** | **Keep one CPU core awake during decode.** `SCHED_IDLE` spinner on a detected E-core + P-core affinity for engine threads. Knob + server A/B + MiniMax/70B/prefill checks. | **27** | ~1 day | ~~none — start here~~ **CLOSED 2026-09-12, +23.1 % through the server, shipped ON.** Applied to every model. | A/B through the server shows < 5 % → close it |
| ~~**S1**~~ **DONE** | **Load probe, no download.** For each candidate 4-bit repo pull only `config.json` + `model.safetensors.index.json`; count keys matching `\.experts\.(\d+)\.`; derive bytes/expert and bpw; check `quant_method` resolves in transformers 5.5.4. | **30** §3 | **< 1 hour** | none | all candidates are fused-3D-on-disk → change model, do **not** start writing a splitter |
| ~~**T1**~~ **DONE** | **Sweep `PREFIX_SESSIONS` × `MAX_SEQ_LEN`** — **CLOSED 2026-09-12: both defaults STAY.** The +5.7 % a single session at 2.7k context wins is an **−8.9 %** loss at 7.4k with 2 sessions, where reuse collapses 7422/7517 → 44/7517 tokens. `MAX_SEQ_LEN` pays ONCE, not twice (the VRAM reserve is floor-bound). | **31** | done | — | fired in substance: the point that wins the two-turn session loses the workload the server exists for |
| ~~**S2**~~ **DONE** | **CLOSED 2026-09-13 — the non-EXL3 int4 MoE LOADS, STREAMS AND IS MEASURED: `4.0648 tok/s` warm at N=1 on server knobs (spread 1.30 %, greedy ids identical across arms), and the server gate passes on it (9 passed) — 4.8× Qwen3-235B-EXL3's raw 0.85, 3.6× row 27's served 1.1363.** Seven root causes, all fixed: RC1 packed-int4 dequant · RC3 shared expert · RC4(a)(b) hybrid-attention KV cache · **RC5 the splitter silently dropped tensors on a non-layer-major shard index** (24356 of 24576 experts) · **RC6 `get_seq_length()` returned 0 on hybrid so every decode step re-ran the whole sequence** (9.5×) · **RC7 the prefix cache cannot lend recurrent state — refused.** RC4(c) CLOSED 2026-09-13. | **30** §4–6 | done | — | fired and PASSED: output coherent, ids stable |
| ~~**S2b**~~ **DONE — all four levers answered.** **(ii) 2026-09-13: `4.0648 → 4.3459 tok/s`, +6.92 %, bit-identical.** **(iii) BUILT AND SHIPPED 2026-09-13: `4.1584 → 6.4702 tok/s`, +55.6 %** at N=1 on server knobs. (i) **REFUTED** — the H2D is bandwidth-bound at 95 % of line rate, worth ≤ 5.7 % (see §5g). **(iv) RE-PRICED AND DROPPED 2026-09-16: both remainders together are 0.33 % of `T_prof` (~1.0 % on a generous in-run scaling) against a ~12 % claim** — (iii) already deleted the hit-side cats, the slab arena is PINNED so (i)'s remainder is 3.0 not 13 µs/expert, and the row's "contiguous blob at cache-fill time" prerequisite was a CLONE the zero-clone slab arena cannot afford. Detail in row **30**. *Original text: (i) coalesce the per-expert H2D (4320 tiny `.to()` per token, ~34 % of `T_prof`) · (ii) land RC1's banked `torch.compile` dequant (17.9 vs 117.5 µs/expert, **6.5× measured**, on ~25 %) · (iii) wire the VRAM expert cache into the non-EXL3 fused forward (`V=0.0`, **7.79 GB reserved and idle**).* | **30** | ~2–4 days | S2's measurement (**reported**) | all three bounded by `FLOOR ×6.30` over U1 (~25.6 tok/s); a lever that measures under a third of its share → drop it |
| ~~**T2**~~ **DONE** | **Truncate `top_k` at decode.** One clamp at `vaayu_base.py:3233` PLUS the same clamp written into each MoE module's own `top_k` by detection (contract-routed models ignore the engine's `k`; renormalisation already exists everywhere), `DECODE_TOP_K` knob, ppl sweep at k ∈ {8,6,5,4} on both MoE models. **Regime-independent** — do it whether or not S2 lands. | **33** | ~1 day + a ppl sweep | none — and S3-pre (2026-09-13) makes it the recommendation: neither S3's nor row 36's gate fires cleanly, this model routes **top-10 of 512**, per-slot phases are **63.8 % of the token**, and k=8 → **+14.7 %** / k=6 → **+34.2 %** derived from measurement | k=6 costs more quality than row 32's 3.0 bpw build at the same speed-up → prefer 32 and close |
| ~~**S3**~~ `[x]` **CLOSED 2026-09-18 by its re-gate on the 30-P build (measurement only, no engine code).** Gate 1, in-run `(construct_miss + gemm_miss)/T_prof`, = **6.96 / 6.99 %** against **≥ 10 %** — two replicates, clean fail, no straddle (generous basis over `T_unprof` 8.59 / 8.67 %; pre-30-P 7.06 / 7.19 %, so it never passed). Gate 2 (native AVX2 kernel ≤ 60 µs/expert) NOT run, by the rule. ADDRESSABLE incl. h2d fell 15.1 / 15.2 → 11.8 / 11.8 % (h2d 186.8 → 101.9 µs/miss-slot, the pinned DMA). Increment on top of 30-P re-priced **−0.2 … +5.9 %** (was −0.2 … +7.1). **Re-open only if a future build measures gate 1 ≥ 10 %, then gate 2 as well.** Detail in row 30's re-gate bullet. (history: KEPT OPEN 2026-09-14 by the user; PARKED behind row 30-P and UNBLOCKED by it 2026-09-18) | **CPU+GPU hybrid.** `EXPERT_COMPUTE_DEVICE` knob, int4 CPU kernel on the RAM/disk tier, 6 P-core threads, A/B vs the **row-27 keep-awake** baseline. **S3-pre re-measured it 2026-09-13 and recommends CLOSING it** — not blocked any more, just not worth the week. Awaiting the user's call; the number technically clears 15 %. | **30** §7 | **S3-pre: hours** · S3: ~1 week | **S3-pre REPORTED 2026-09-13 (two replicates): ADDRESSABLE `(h2d+construct_miss+gemm_miss)/T` = 15.1 / 15.2 %, ON the 15 % rule** (literal `(h2d+gemm)/T` = 20.8 / 21.1 %, but that counts VRAM-hit work S3 cannot take). **The mechanism, not the number, is the reason to stop: the VRAM cache has exactly ONE fill site — the miss branch, after the H2D — so the 83.4 % hit rate exists BECAUSE misses pay it. S3 removes the only thing that fills the cache.** **T2 (row 33, 2026-09-14) makes it SMALLER: the decode clamp lifts the VRAM hit rate 83 % -> 84 % (k=9) -> 90 % (k=6), and misses are the only thing S3 can address.** **The user KEPT THIS ROW OPEN on 2026-09-14 knowing all of the above** — the row costs nothing while unstarted; what is deferred is the ~1 week, not the idea. **What would actually revive it: a SECOND fill site for the VRAM expert cache**, so the cache survives misses moving to CPU — that dissolves the mechanism objection and nobody has scoped it. Gain ~9 % for ~1 week | measured gain < half the projection, or negative under CPU/IO contention → **close the hybrid idea** |
| ~~**T3**~~ **DEAD on this model** | **The streaming-regime ladder — ONLY if S2 stalls or is abandoned.** In order: **32** download a ≤ 3.25 bpw EXL3 build of Qwen3-235B (×1.50, zero code, and it lifts the 62/94-layer VRAM-cache restriction) → **34** probe in-flight read depth, build only if ≥ 60 % of busy time sits at depth 1 (×1.10) → **35** per-model measured gate for `VAAYU_LOOKAHEAD` (+11.7 % Qwen). | **32 · 34 · 35** | a download · ½ day · ~2 days | **S2 stalled.** If S2 lands, these three mostly die — they only pay while experts stream from disk | 34: depth already ≥ 2 → R2's verdict extends, close it · 35: cannot reproduce +11.7 % through the server with the trial cost included → close it |
| **T4** ← **LIVE BRANCH** | **The residency-regime levers — ONLY if S2 lands.** **S2 LANDED 2026-09-13 and the disk is now 1.7 % of a decode token here (`gb_tok=0.030`, 96.5 % RAM-cache hit) against 70–83 % on the EXL3 models — the branch condition is MET.** Take S2b's three levers first; 36's own gate below is still unmeasured on a resident model. **36** CUDA graphs / fused per-layer dispatch — **its gate is now MEASURED (S3-pre, 2026-09-13): strict `serial`=NOEXP is 23.0 / 23.4 %, BELOW the 30 % rule; the wide reading (`other`/T, all host time no expert placement removes) is 32.5 / 32.8 %, above it. The two readings straddle the rule; the 9.4 pp between them is the MoE loop's own unbracketed host work, which is exactly what fused dispatch targets.** **37** re-gate speculative decoding, whose refusal (`off_union`, 19.31 GB vs ~17 GB) is a BYTE bound that S2 dissolves. | **36 · 37** | 36: ~1 week · 37: ~2 days | **36:** a resident model measures `serial / T` ≥ 30 %. **37:** S2 or row 32 landed | 36: `serial / T` under 30 % → it is still invisible, close it · 37: gate still refuses on the resident model → close it |
| **S4** | **Port DeepSeek-V4-Flash.** TileLang/Triton kernels → HF modeling code + `auto_map` → key remap → persist → measure. | **28b** | **weeks** | S2 done (it proves the non-EXL3 MoE path end to end, cheaply) | kernels will not build/validate on sm_120 → abandon at step 1, ~2 days lost |
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
| **13** `[x]` | The spec gate's bar is measured in an UNREPRESENTATIVE window and evaluated once | **CLOSED 2026-09-11 — `even-4` built and validated on device; see row 13's CLOSED block.** **BUILT + MEASURED 2026-09-10. The timed bar exists, is a strict tightening, and changed NO decision on any of 4 on-device arms — banked, not shipped-with-effect. The design's Done condition REFUTED: the losing arm is still admitted (−19.7 % on its own A/B), because CALIBRATE+TRIAL sit in the cheapest stretch of a call.** Still worth **0 %** on the default config — `USE_PROMPT_LOOKUP` is OFF for MoE. **RE-AIMED 2026-09-10 when row 24 closed REFUTED: the path is a WIDER, POOLED one-shot window, not re-evaluation and not a later sample. PRICED the same day and it PAYS: `even-4` — TRIAL becomes 4 alternating pairs of 4 drafting + 4 suppressed rounds, verdict `run` iff acceptance > byte bar AND MEAN of the per-pair s/token ratios < 1 — decides **6/6** against today's 5/6 for **+16 suppressed rounds** (40 rounds vs 24) and the SAME 48 drafted tokens, worth **+1.3–2.5 %** mean over six arms against a **+1.9 %** ceiling for perfect verdicts at today's price. **DO NOT route the pooled ratio through the shipped `alpha`/wall-bar form — measured 4/6, worse than today.** Row 24's “m ≥ 3, median” is wrong twice: the estimator must be a MEAN, and charged faithfully for CALIBRATE the safe width is **m ≥ 4** (m=3 is 5/6). Re-running whole TRIALs (`full-m`) NEVER pays at any call length.** | row **13** · [`docs/2026-09-10-row13-timed-bar-RESULTS.md`](docs/2026-09-10-row13-timed-bar-RESULTS.md) · [`…-DESIGN.md`](docs/2026-09-10-row13-timed-bar-DESIGN.md) · [`…-SPIKE.md`](docs/2026-09-10-row13-timed-trial-SPIKE.md) · [`…-window-price-RESULTS.md`](docs/2026-09-10-row13-window-width-price-RESULTS.md) · [`…-even4-BUILD-BRIEF.md`](docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md) |
| ~~**24**~~ | **CLOSED 2026-09-10 — REFUTED ON PRICE** (≤ +6.28 % with perfect foresight at zero cost; break-even 0.5–5.9 token-equivalents against a real ~29.3). Its live residue is row 13's remainder, restated as WINDOW WIDTH | Same §8 bucket as 13 — 0 % on the default config. **Now the head of this queue and no longer a side observation: row 13's four on-device arms measured the decision window at 0.510-0.532 s/token against 0.637-0.793 for the whole call, and that gap is exactly why a correct, timed, tightened bar still admits a −19.7 % configuration.** | row **24** |
| ~~**A**~~ | **CLOSED 2026-09-10 — 19 of 28 reachable, 9 declined in code, drift test wired. Suite 1893 → 1906** | Was: capability and correctness, not speed. `ENABLE_KV_PERSIST` could be turned on but not aimed. **All 5 new knobs are worth 0 % until an operator sets one — no default moved.** | OPTIONAL row **14** |
| ~~**C1**~~ | **CLOSED 2026-09-10 — construction refused, Option A, suite 1887 → 1893** | Was: real correctness hole. CUDA is what runs here, so the fix is worth **0 %** on every target run here. | OPEN section |
| `_finish` stop-string identity | Cannot report WHICH stop string fired | Reporting only, and no client reads it. Needs a new premise (matching stop strings ourselves). | STILL OPEN |
| `expert_sim.py` anchor gap | 31.9 % simulated vs 43.9 % live | A diagnostic tool's accuracy, not the engine. The obvious hypothesis was built and REJECTED. | STILL OPEN |
| No resident-size checkpoint | Needs a ~14B coder at 4–4.5 bpw EXL3 | A download, not code. The row says it is **not a blocker**. | STILL OPEN |
| ~~Static unbound-local check~~ | **CLOSED 2026-09-10 — `pylint E0606` wired as `tests/test_no_unbound_locals.py`** | Was: reliability, two such bugs shipped GREEN and broke both MoE loads. Prevents crashes, not slowness. | OPTIONAL row **2** |
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

**Suite baseline: 1979 passed, 15 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
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

**NO `[bug]` row is live in this section. C1 was FIXED 2026-09-10** — see its row below, which
keeps the verdict and is pruned at the next prune. Everything else this section carried closed
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

- **[x] C1 · a MoE model loaded on a non-CUDA compute device ran a PLD or drafter speculator
  with NO gate at all — FIXED 2026-09-10, Option A (refuse construction), CPU-only, no GPU round.**
  Pre-existing, not introduced by S1d. Full brief (the defect, the trap in the obvious fix, the
  four things not to break):
  **[`docs/issue-C1-cpu-moe-ungated-speculation.md`](docs/issue-C1-cpu-moe-ungated-speculation.md)**
  (**gitignored — this disk only**; everything load-bearing is restated here).
  **The defect, re-verified in the code before fixing (line numbers had drifted ~+230 from the
  brief):** `_expert_offset_index` is built only under `if device.startswith("cuda") and
  self.is_moe:` (`vaayu_base.py:2027`), `_moe_geometry` only when that index exists (`:2036`), and
  `_expert_cache` only under the matching `running_device` guard (`:2335`) — while `self.is_moe`
  is read off the module tree and is device-independent. So on CPU-compute MoE all three gates
  skipped themselves (S1a `:2325`, Fix #9 `:2585`, `_build_spec_gate()`), the generation loop reads
  a missing gate as "no restriction", and speculation ran silently and unmeasured for the life of
  the process — the `CLAUDE.md` §5 violation, independent of the perf question.
  **Fix:** new pure helper `spec_no_geometry_refusal(is_moe, geometry, device)`
  (`vaayu_base.py:201`, beside `spec_preload_refusal`) returns a reason whenever `is_moe` and
  `_moe_geometry is None`; wired as the FIRST `_pre` in the speculator-construction block
  (`:2305`), so it reuses the existing refusal machinery — one `[Vaayu-Spec] WARNING`, one
  `speculative_status["refused_because"]`, and the existing `VAAYU_SPEC_FORCE=1` hatch still
  overrides it (deliberate: that is how a CPU-compute MoE workload gets measured if one appears).
  It covers **prompt lookup as well as a weighted drafter** — the two static gates' PLD exemption
  is an argument about WHICH gate decides, and on this path there is no gate to hand it to; the
  exemption itself is untouched. The S1a bound below it needed no new guard: it already requires
  `_moe_geometry is not None`, which is exactly when this refusal returns None (a `_pre is None`
  guard was written, found to be dead by that implication, and removed). One adjacent message
  fixed because the fix made it reachable for PLD, which has no
  drafter: the refusal line now says "no prompt-lookup proposer is built" in that case.
  **`compute_moe_geometry` never returns None**, so `is_moe and geometry is None` is exactly
  "MoE not on CUDA" — dense (any device) and CUDA MoE are provably untouched, which is the
  regression that mattered (dense drafter 5.57×, MiniMax PLD 1.307×).
  **Evidence:** `tests/test_spec_no_geometry_gate.py` (+6, written first and verified to fail on
  ImportError against the old code): CPU-MoE refused, message names cause and lever, both on-device
  MoE geometries NOT refused, dense not refused on either device. Full suite **1887 → 1893 passed,
  12 skipped**; `tests/test_no_unbound_locals.py` green.
  **§8, all four:** (1) **no knob needed, stated deliberately** — this is a refusal, not a feature,
  and a knob to disable a safety gate is what `VAAYU_SPEC_FORCE=1` already is; (2) plumbed — it is
  in `__init__`, the path every server load takes, not behind a config flag; (3) **reachable, and
  worth 0 %** — the server's `DEVICE` defaults to `"cuda:0"` (`server/config.py:27`), so the
  refusal never fires on the default configuration and moves no tok/s on any target run here; it
  changes behaviour only for someone who sets a non-CUDA `DEVICE` with a MoE checkpoint, who
  previously got a silent unchecked run and now gets a stated refusal; (4) documented in
  `documentation/server.md` ("Is speculative decoding actually running?" — "two gates" → "three
  gates", with the new one first and the CUDA/`VAAYU_SPEC_FORCE` levers named).
  **NOT done, deliberately (Option B in the brief):** making the gate WORK on CPU — a CPU-safe
  geometry, a budget premise that means something with no expert cache, and forward patching so
  `_expert_history` is written. That is a change to the CPU MoE path, nothing tests it, no
  CPU-compute MoE checkpoint exists on this box, and no workload asks for it. **Re-open only with
  such a workload.** **Also NOT done:** a load-path wiring test. The refusal lives inside
  `__init__`, which needs a real checkpoint, and no CPU-compute MoE one exists here — so the
  predicate is unit-tested and the wiring is inspection-verified, the same standard the two
  adjacent gates (S1a, Fix #9) have always had. A CPU-MoE checkpoint would be the thing that
  changes that.

**Applied on the spot 2026-09-03, no row needed:** `DEFAULT_STREAM = True` in
`vaayu/server/config_local.py`. It was `False`, and with a client sending no `max_tokens` that meant
**864 s of silence** for an 809-token answer — most of the reported "slowness" was this, not tok/s.
The server already warns about the combination; nobody was reading the warning.

**Still unfixed on this box, warned at every boot, not a code row:** `rustdesk` holds 0.14 GiB of GPU,
below the 256 MiB refusal threshold but permanent, because budgets are sized once at load. **NVMe APST
is no longer on this list — row 25 measured it NULL on Qwen and MiniMax at N=1 (2026-09-11)**, and found the
script that was meant to disable it could not do so without a reboot (fixed).

---

# ▶ STILL OPEN — carried forward from rows that closed

Live items left behind by rows that closed and were pruned. **All of them are OPTIONAL by §0's
test — none can make Vaayu faster.** No §0 P-row is open.

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
- **[ ] No resident-size checkpoint on disk.** The server's "usable now" target needs a ~14B-class
  coder at 4–4.5 bpw EXL3 (~8 GB weights, against a derived ~10.5 GiB budget). Nothing of that size
  is on this box — only the two huge MoEs, the 70B dense, and the 1B/1.5B tinies. Until then the
  server runs the streamed big models at their measured ceiling (~0.56–0.86 tok/s MoE at 8192,
  single stream). **Not a blocker:** the server is model-agnostic and does not wait on this.
- **[ ] `_finish` cannot report WHICH stop string fired.** HF trims a matched stop string before
  returning, so the ids cannot be inspected for one; `_finish` infers `hit_stop_string` from
  `produced < cap`. The dangerous half closed with B5 (2026-08-23) — a cancelled run no longer
  satisfies that inference. What remains is only the stop string's IDENTITY, which no client acts
  on. **Needs a new premise before it means anything.** Fixing it means not trusting HF's trim and
  matching stop strings ourselves — a behaviour change, not a patch.
- **[ ] `vaayu/expert_sim.py`'s anchor gap is unexplained.** 31.9 % simulated against 43.9 % live at
  Qwen's 9.12 GB budget. The obvious hypothesis — that it fails to model the engine's eviction
  pinning — was implemented and **REJECTED** (byte-identical result). The gap is confined to the
  small-budget regime (at 16 GB it predicted 20.3 % for MiniMax against a measured 21.3 %).
  **Do not quote its small-budget absolutes.**

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

**Read this before proposing any perf work.** Each line is something that was BUILT or MEASURED and
then refuted. A refutation closes its row; it is not re-scoped. Full text in
`docs/2026-09-03-todo-archive-preprune.md` and `git log -p -- todo.md`.

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
| **DeepSeek-V4-Flash-0731** ⛔ **COMPLETE ON DISK 2026-09-20 (48/48, 157 G) — STILL CANNOT LOAD** | `deepseek-ai--DeepSeek-V4-Flash-0731` | 43 / 1 / 512 / 4096 / 64 / 129280 · 256 experts, top-6, **fp4 4.2500 bpw**, 12.750 MiB/expert, 155.42 GiB (137.06 experts + 8.24 dense + 10.12 MTP draft) | n/a | none — `deepseek_v4` unknown to transformers 5.5.4, no HF remote code in repo, TileLang kernels absent (row 28) | n/a |

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
| **2** | Static check for unbound locals / import order | OPTIONAL | `[x]` **CLOSED 2026-09-10 — wired, and PROVEN to fail on the real shape** | ~30 min, CPU only | Two such bugs shipped GREEN and broke both MoE loads. **`ruff` does NOT catch the class — `pylint E0606` does.** `vaayu/` is clean today under it (2 false positives). The row narrows to wiring pylint + suppressing those two. See below. |
| **13** | `[bug]` The spec gate's bar is measured in an UNREPRESENTATIVE window and evaluated once (was: it is in expert bytes) | OPTIONAL | `[x]` **CLOSED 2026-09-11 — `even-4` built, done condition met on device (see row 13).** Was: `[~]` **WINDOW WIDTH PRICED 2026-09-10 — the remainder is now a DECIDED, COSTED build (`even-4`), awaiting a go/no-go.** | window fix: ~half a day + 1 on-device A/B | **Build order: row 14 FIRST, then this, then row 24.** Row 14 is CPU-only and fixes a feature that IS exposed today (`ENABLE_KV_PERSIST` can be turned on but not aimed); this row is worth **0 %** unless a user explicitly opts a MoE checkpoint into prompt lookup. **Cold-executable build brief for the remainder: [`docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md`](docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md)** — read it before touching code; it carries the trap that makes the obvious implementation WORSE than today. Evidence: [`…-window-width-price-RESULTS.md`](docs/2026-09-10-row13-window-width-price-RESULTS.md), [`…-timed-trial-SPIKE.md`](docs/2026-09-10-row13-timed-trial-SPIKE.md), [`…-timed-bar-DESIGN.md`](docs/2026-09-10-row13-timed-bar-DESIGN.md) (the SHIPPED half). | Opened 2026-09-07 by the P9 end-to-end arm. MiniMax-M2.7, thinking ON: gate measured acceptance 0.6875 against a 0.6259 break-even and decided `run` — and speculation still cost **−12.9 % total / −16.0 % decode**. **Self-contained brief: [`docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md`](docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md).** See below. |
| **24** | `[x]` **CLOSED 2026-09-10 — REFUTED ON PRICE.** Re-evaluation is worth ≤ **+6.28 %** with perfect foresight at ZERO cost, and breaks even at **0.5–5.9** token-equivalents against a real re-entry price of **~29.3** | OPTIONAL | `[x]` **closed by an offline replay spike — CPU only, no GPU round, nothing shipped** | none spent | Opened 2026-09-10 by row 13's spike. MiniMax-M2.7 leaving its reasoning preamble for code reproduction: blocks 0-9 lost, blocks 10-14 ran **1.21-1.28x**. A TRIAL sampled in the preamble latches `off` and forfeits the win; one sampled in the code stretch latches `run` and keeps it through the next preamble. **NOT the SETTLED regime-latch item** (that was refuted on Qwen, which never emitted code at all). See below. |
| **6a-b** | `head_bytes` is probably not a transient at all | OPTIONAL | `[ ]` | GPU round | Split out of row 6a on its own evidence when that row closed 2026-09-09. §8-bounded the same way P5 was: the reserve is floor-bound on flash, so tightening it buys admission headroom, not tok/s. See below. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | OPTIONAL | `[x]` 2026-09-11 | cache swap (option b) | CLOSED: `auto` → `bf16` on every loader. See row 3. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | OPTIONAL (parked) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) — **TRIAGED + 2 probes run 2026-09-06** | **P2** (the outlier split) | `[ ]` **3 of 4 closed 2026-09-06; 1 quality-certification item left** | CPU sweep | Two closed on new measurement (Plan B Phase 2 refuted directly; fold-`Pi` target removed by Plan I). Real-query gate RUN: **9 bits FAILS on Qwen (0.0649 vs 0.05 bar)** and 9 is the quantizer's ceiling. Outlier split promoted with a measured operating point. See below. |
| **8** | `tensor_align=True` is production-dead code | OPTIONAL | `[~]` **docstrings FIXED 2026-09-03; branch deliberately kept** | done / rest not scheduled | Opened 2026-09-03 by P6, docs half closed the same day. Nothing calls it since `c5014cd` (2026-07-13), and the stale docstring is what produced row R4/P6 and cost that row its premise. **FOUR false claims corrected** in `safetensor_model_persister.py` — `load_expert_into_slab` (mode advertised as "the prefill stager, via stage_read_expert"), the `tensor_align=False` bullet (claimed prefill never routes through `views_from_packed_copy`; it always does now), `_group_layout`, and **`views_from_packed_copy`, the dangerous one — it promised its output was `_TENSOR_ALIGN`-aligned "by construction", which stopped being true at B1; production sources are raw, so its views are copy-SOURCE-safe only and `gpu_repack_aligned` is what makes them kernel-safe.** The BRANCH stays: `_group_layout`/`expert_span_bytes` size every slab `max(raw_aligned_length, packed_span)` for it, and 5 test sites exercise it — deleting it is a sizing change, not a cleanup, and nothing needs it. Docstrings only, no behaviour touched; suite 1322/3. |

---

# 13 · [x] The spec gate can say `run` on a configuration that loses wall-clock — CLOSED 2026-09-11: `even-4` BUILT, DONE CONDITION MET ON DEVICE

**CLOSED 2026-09-11 (GO given that day).** Full write-up
[`docs/2026-09-11-row13-even4-RESULTS.md`](docs/2026-09-11-row13-even4-RESULTS.md) (gitignored —
everything load-bearing is restated here).

**Built.** `adaptive_gate.py`: `TRIAL_LEG_ROUNDS = 4`, `TRIAL_PAIRS = 4`, `TRIAL_ROUND_CAP` 64 → 128
(same 64 drafting-eligible rounds). TRIAL alternates [D4 S4] pairs; ends on a pair boundary at
≥ 4 pairs AND ≥ 48 drafted (a slow proposer runs more pairs), or the cap. With ≥ 4 fully-timed pairs:
`run` iff acceptance > byte bar AND **mean** pair s/token ratio < 1 (`bar_used="pooled"`); fewer →
the old single-window rule, `bar_used` says which. `status()` + `pooled_ratio`, `pairs_used`; every
old key unchanged. **Plus a fix the brief said was unnecessary:** `vaayu_base.py`'s per-CALL VRAM
prefix read (P9) used `should_draft()`, which now alternates inside TRIAL — a call opening on a
suppressed leg would have run its drafting legs under the load-time prefix. Now reads
`gate.phase in ("trial", "run")`; test RED before, both geometries. Tests +18. **Suite: stashed
HEAD 1950 / 12 skipped (matched), with the change 1968 / 12.**

**The brief's price did NOT reproduce.** Its "today 5/6 → 6/6, +1.3–2.5 %" scored a RATIO-ONLY model
of the shipped gate. Replayed through the REAL `AdaptiveSpecGate` (HEAD and new) on the same six
traces at byte bars 0.6125 / 0.6259 / 0.6462: **HEAD is 6/6** (it refuses `winner-p0-a` on
acceptance, as on-device arm 3 did on 2026-09-10), even-4 changes no verdict, and costs **−1.03 /
−0.73 / −0.47 %** at T = 200 / 400 / 2000 on the probe's own pricing. **Lesson: a probe that models
the shipped rule must replay the shipped CLASS.** The traces cannot contain the real failure (their
in-window loser acceptance is 0.42–0.50, under the byte bar), so the validation was re-aimed at the
arm the real gate actually got wrong.

**On device, same day, BALLAST_GB=25, d=3, WARM 96 + TIMED 200:**

| arm | gate | acceptance / byte bar | pooled ratio (pairs) | timed tok/s |
|---|---|---|---|---|
| MiniMax loser THINK=1 ×3 | **off_acceptance** ×3 | 0.6078 / 0.6305 | 1.359 / 1.352 / 1.333 (5) | 1.5615 / 1.5049 / 1.5092 |
| — forced (no gate) | — | — | — | **1.2710 → speculation −18.6 %** |
| MiniMax winner THINK=0 | **run** | 0.8542 / 0.6125 | 0.836 (4) | **1.7724** |
| — off | — | — | — | **1.5363 → +15.4 %** |
| MiniMax winner THINK=1 | off_acceptance | 0.3750 / 0.6462 | 1.550 (7) | 1.5047 |
| Qwen3-235B winner THINK=0 | off_union, 0 drafts | — | — | 0.9786 |

**How it flips, and how thin:** in the wider window EVERY rule refuses the loser — acceptance 0.6078 is
under the byte bar alone (HEAD's 16-consecutive-round window measured 0.6875 and admitted). The
improvement is the window's WIDTH, the brief's thesis, not the decision rule. The pooled ratio agrees
but is **not independently decisive: its first 4 pairs average 0.99 on all 3 replicates**; it refuses
through pair 5 (every draft rejected), which exists only because 48 drafted tokens were not reached in
4 pairs. Greedy text is deterministic, so 3/3 is three timings of ONE trajectory.

**§8:** no knob (a window is not a feature); nothing to plumb (inside `AdaptiveSpecGate`); REACHABLE
but **worth 0 % on the default config** — MoE-only gate, `USE_PROMPT_LOOKUP = "auto"` OFF for MoE.
Unblocked by an operator opting a MoE checkpoint in. Docs updated in the same change:
`documentation/server.md` (config row + runtime-gate paragraph), `vaayu/server/config.py` comment.
**Do NOT re-open as "make the ratio decisive"** — on the one losing workload measured, four pairs sit
at 0.99; tuning to that arm is tuning to noise.

---

*Pre-close history of this row follows (kept until the next prune).*


**STATUS 2026-09-10: the timed bar is BUILT, TESTED and MEASURED on device — and the DESIGN's own
Done condition is REFUTED. The row does NOT close.** The defect is no longer *"the bar is in
expert bytes"* (it is not, any more): it is **"the bar is measured in an unrepresentative window
and evaluated once"**. Full write-up
[`docs/2026-09-10-row13-timed-bar-RESULTS.md`](docs/2026-09-10-row13-timed-bar-RESULTS.md)
(gitignored — everything load-bearing is restated here).

**What shipped.** `wall_breakeven_acceptance(alpha, d) = ((1+d)*alpha - 1)/d` in
`vaayu/speculative/adaptive_gate.py`; `observe()`/`record_round()` take an optional `elapsed_s`
(and `record_round` a `draft_overhead_s`); `alpha = median(wide)/(median(base)*(1+d))`, medians
never means; the bar is `max(byte_breakeven, wall_breakeven)` so **bytes stays NECESSARY**;
`status()` gains `alpha`, `wall_breakeven`, `bar_used`, `base_from`, which reach `/v1/models` and
`env.json` unchanged. The loop marks one `perf_counter()` per round at the point both gate call
sites already fire from, and times `speculator.draft()` separately. **The gate still reads no
clock itself.** Tests 24 -> 74 in `tests/test_adaptive_spec_gate.py`, +4 wiring.

**THE FOUR ARMS (real gate, no probe subclass, one `systemd-run` scope each, BALLAST_GB=25,
WARM=96 + TIMED 200):**

| arm | required | **got** | bar_used | alpha | wall bar | byte bar | acceptance | A/B |
|---|---|---|---|---|---|---|---|---|
| MiniMax loser THINK=1 | `off_acceptance` | **`run`** ✗ | wall | 0.7349 | 0.6466 | 0.6305 | 0.6875 | **1.2604 spec-on vs 1.5690 off = −19.7 %** |
| MiniMax winner THINK=0 | `run` | **`run`** ✓ | wall | 0.7787 | 0.7050 | 0.6125 | 0.8125 | **1.7681 vs 1.5450 = +14.4 %** |
| MiniMax winner THINK=1 | either, stated | `off_acceptance` | wall | 0.7750 | 0.7000 | 0.6462 | 0.5417 | bytes would refuse it too |
| Qwen3-235B THINK=0 | `off_union` | **`off_union`** ✓ | — | None | — | 0.4036 | None | 0 drafting rounds |

**The timed bar changed NO decision on any arm.** Bytes alone gives run / run / off_acceptance /
off_union — the same four. It raises the bar (0.6305 -> 0.6466, 0.6125 -> 0.7050) and becomes the
binding test wherever it runs, but it moves no outcome. **§8: BANKED, not shipped-with-effect**,
on top of already being worth 0 % on the default config (`USE_PROMPT_LOOKUP` is OFF for MoE).
**No knob** — a bar is not a feature.

**WHY ARM 1 STILL FAILS — the finding that supersedes this row's premise, and it is row 24's
evidence.** To refuse arm 1 the bar needs `alpha >= (1 + d*a)/(1+d)` = **0.7656**; measured
in-window **0.7349–0.7599**. The gate is not mis-measuring — **it measures a window that does not
represent the call it decides for**: TRIAL's drafting rounds run at **0.510 s/token** and its k=0
rounds at **0.532**, while the whole timed call runs at **0.793 s/token speculating and 0.637
not**. Both slow as context grows and **the speculating arm slows more** (the spike's trap 3,
`alpha` 0.749 -> 0.832 over 130 rounds, whose consequence was never drawn). CALIBRATE and TRIAL
sit in the cheapest, most speculation-favourable stretch of the call, where speculation on the
loser prompt genuinely IS marginally profitable; the loss accrues afterwards. **A bar evaluated
ONCE, early, cannot be right for the whole call however well it is measured.**

**TWO DESIGN CORRECTIONS MADE ON DEVICE — do not re-derive them:**
1. **CALIBRATE is a COLD baseline.** The design divided TRIAL's wide median by CALIBRATE's. Run as
   written that gave **alpha 0.6976 -> wall bar 0.5969, UNDER the byte bar**, so `bar_used` stayed
   `"bytes"` and the fix was inert on the very arm it was built for. CALIBRATE's rounds are the
   FIRST rounds of a call, right after the prefill; TRIAL's run warm. **Prefer TRIAL's own k=0
   rounds** (1-wide decodes in TRIAL's regime), CALIBRATE as fallback, reported in `base_from`.
   alpha 0.6976 -> 0.7599. **This REVERSES the design's trap 4 for the BASE list**; k=0 rounds
   still never enter the WIDE median.
2. **The proposer's own search does not belong in the baseline** — speculation OFF runs no search
   at all. The loop times `draft()` and the gate subtracts it from k=0 rounds only. **Numerically
   negligible on MiniMax** (two otherwise-identical arms differ more from run-to-run variance in
   the ~8-sample k=0 median), kept because it is correct and free.

**WHAT IS LEFT ON THIS ROW — RE-AIMED, THEN PRICED, 2026-09-10. It is now a GO/NO-GO on a
designed, costed build, not an investigation.** Making the measurement representative does NOT mean
re-evaluating: **row 24 is REFUTED ON PRICE** (perfect-foresight re-evaluation is worth ≤ +6.28 % at
ZERO cost and breaks even at 0.5–5.9 token-equivalents against a real ~29.3 — 5×–60× too expensive).
What row 24's replay found instead is that the defect is the decision window's **WIDTH**, and that
width is now PRICED and it PAYS.

**Cold-executable build brief (read this first if you are picking the row up):**
[`docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md`](docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md).
Evidence: [`docs/2026-09-10-row13-window-width-price-RESULTS.md`](docs/2026-09-10-row13-window-width-price-RESULTS.md).
Both gitignored — everything load-bearing is restated here. CPU only, no GPU round, no engine edit
so far: probe `plans/probes/p13_window_price.py` (throwaway, gitignored) replays the SAME six
alternating traces at ROUND granularity.

**BUILD THIS — `even-4`.** CALIBRATE unchanged at 8 suppressed rounds; TRIAL becomes **four
alternating pairs of 4 drafting + 4 suppressed rounds**; verdict pools the four pairs. **40 rounds
against today's 24** (`8 + 4×(4+4)` vs `8 + 16`), the **same 48 drafted tokens**
(`m × r_draft × d = 4 × 4 × 3` = `TRIAL_DRAFTED_TOKENS`, so the acceptance estimate the byte bar
consumes is NOT loosened), marginal cost **+16 suppressed rounds**. Decides **6/6** where today
decides 5/6. Worth **+1.28 % / +1.92 % / +2.45 %** mean over the six arms at T = 200 / 400 / 2000
tokens; beats shipped from **T ≥ 98 tokens**.

**THE DECISION RULE — and the trap in the obvious implementation.** Per pair *j*:
`ratio(j) = (drafting-leg s/token) / (suppressed-leg s/token)`; then
**`run` iff `acceptance > byte_breakeven` AND `mean(ratio) < 1`.** The byte bar stays the NECESSARY
condition (free, sound, and what refuses Qwen at zero cost). **DO NOT route the pooled ratio through
the shipped `_measured_alpha()` / wall-bar form** — measured on the same traces, that route scores
**3/6–6/6 and 4/6 at the recommended legs, WORSE than today's 5/6**, while `bytes AND pooled ratio`
is **6/6 at all sixteen leg geometries** at m=4. `alpha` is a per-position cost from 4-sample
round-time medians; the s/token ratio already contains realized acceptance, which is the quantity
the row is about. **This is P7's trap again: a perfectly measured intermediate is not the quantity
you need.**

**Four findings that must not be re-derived:**
1. **The prize is bounded at +1.9 % and lives on ONE arm of six.** A gate deciding perfectly at
   today's window price is worth +1.60 / +1.78 / +1.91 % at T=200/400/2000. The whole gap is
   `winner-p0-a` (THINK=1), where the shipped gate says `on` and the call wants `off`. `even-4`
   fixes exactly that arm (**+13.18 %** at T=400); the other five move **−2.29 % to +3.46 %**.
   **Do not quote "+2 % everywhere"**, and watch `winner-p1-t0` (−2.29 %) on device — a correctly
   decided `on` arm paying the 16 extra suppressed rounds.
2. **Row 24's "m ≥ 3, median" is wrong twice.** (a) The ESTIMATOR must be a **MEAN** of the per-pair
   ratios: a median over 3–4 samples flips when one pair moves. (b) The WIDTH must be **m ≥ 4**:
   row 24 measured without charging CALIBRATE, which shifts which rounds the legs land on; charged
   faithfully, m=3 is **5/6** whenever `r_draft ≤ 4`, and m=4 is 6/6 at every leg geometry.
   **Width alone is not enough either** — at m=1 nothing beats 4/6.
3. **CALIBRATE is paid by every candidate and must NOT enter the pooled baseline.** Its 8 suppressed
   rounds must run before any draft (the union/byte refusal is what stops Qwen paying for a single
   draft), and its rounds are cold — using them moved alpha 0.7801 → 0.6976 on device and put the
   wall bar UNDER the byte bar, making the timed fix inert.
4. **"m paired blocks" read LITERALLY is refuted.** Re-running whole shipped-length TRIALs
   (`full-m`) **never** beats shipped at any call length — −14.94 % at T=200, still −1.72 % at
   T=2000. And **do not shrink to m=3** to save rounds: 5/6, and negative at every length.

**Do NOT re-open this row as "measure alpha better"** — a perfectly measured in-window alpha is the
wrong quantity, exactly as P7's perfectly-measured union was. And **do NOT re-open it as "decide
later"**: wrong verdicts cluster at BOTH ends of a call, so a gate that sampled later would have
been wrong on two arms instead of one (row 24 §3).

**On-device validation, if built:** the arm that matters is **`winner-p0-a`, THINK=1** — the gate
must say `off` and the A/B must show `off` faster — while **`winner-p0-t0` (THINK=0, the server's
regime) must still be admitted** (row 13 arm 2 measured it a genuine +14.4 %). Harness kept:
`plans/probes/p13_validate.py` + `p13v_ab.sh` (gitignored), which carry both reference prompts.
**Do not assert cross-arm greedy id identity** — the arms differ in which rounds are 4-wide.

**§8, decided in advance:** **no knob** (a bar and its window are not features); it lands inside
`AdaptiveSpecGate`, which `_build_spec_gate()` (`vaayu_base.py:5080`) already constructs and caches,
so there is nothing new to wire; and it is **REACHABLE but worth 0 %** on the default configuration
— the gate is MoE-only and `USE_PROMPT_LOOKUP = "auto"` is OFF for MoE
(`vaayu/server/config.py:211`). **What unblocks it: an operator opting a MoE checkpoint into prompt
lookup, or a measured case for flipping that default. Neither exists.** If built,
`documentation/server.md`'s "Is speculative decoding actually running?" section changes in the same
commit.

**STOP RULE / the alternative to building:** if the leg schedule cannot hold 48 drafted tokens
without raising `TRIAL_ROUND_CAP` past what a short call finishes, or if the on-device arm does not
flip `winner-p0-a` to `off`, **close the row** by documenting that `run` is a window-local timed
claim (`documentation/` already says this as of `5aa9aab`). Closing is a legitimate outcome here:
the build is worth 0 % on the default config, and this queue has items that are not.

Where the pooled finding comes from, for a cold pickup:
[`docs/2026-09-10-row24-relatch-replay-RESULTS.md`](docs/2026-09-10-row24-relatch-replay-RESULTS.md)
§3 and its probe `plans/probes/p24_replay.py` (gitignored, throwaway), whose loaders
`p13_window_price.py` imports rather than re-deriving. Both replay row 13's OWN six alternating
traces (`plans/probes/p13-minimax-*.json`), so the accuracy claim can be re-checked with nothing new
measured. Harness kept: `plans/probes/p13_validate.py` + `p13v_ab.sh` (gitignored), which carries
both reference prompts and the winner that arm 2 proves is a genuine +14.4 %.

**Self-contained brief: [`docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md`](docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md)**
— written to be picked up COLD in a fresh session: the defect, the exact call sites, all 8 arms,
five open design questions, the stop rule, and six traps. **`docs/` is gitignored, so that file is
on this disk only**; everything load-bearing is restated below and the durable copy of this row is
`git log -p -- todo.md`.

**Opened 2026-09-07 by the P9 end-to-end arm** (`plans/probes/p9_e2e.py` + `p9_e2e_ab.sh`,
8 arms, MiniMax-M2.7, one 128-token code prompt, `draft_tokens=3`, interleaved, BALLAST_GB=25).
Results: `docs/2026-09-07-P9-e2e-prompt-lookup-RESULTS.md` (gitignored — numbers restated here).

**`AdaptiveSpecGate` decides on EXPERT BYTES, and nothing checks the clock.**
`spec_breakeven_acceptance(union_bytes_1, union_bytes_k, draft_tokens)`
(`vaayu/speculative/adaptive_gate.py:61`) asks whether a k-wide verify moves fewer expert bytes
per token than k separate decodes. It models neither the drafter's own cost nor the dispatch-side
work a wider pass creates — so a `run` decision is not a claim about tok/s, and was measured
not to be one:

| regime | acceptance | break-even | gate | total tok/s | decode tok/s |
|---|---|---|---|---|---|
| thinking ON | 0.6875 | 0.6259 | **`run`** | 1.4290 → **1.2443 (−12.9 %)** | 1.8890 → **1.5869 (−16.0 %)** |
| thinking OFF (server default) | **0.14** | 0.6259 | `off_acceptance` | 1.4665 → n/a (refused after ~1 call) | 1.9518 → n/a |

Both regimes reproduce across 2 replicates. **The gate caught the second case and missed the
first.** In the first it admitted a configuration that costs 16 % of decode throughput, and it
would keep it for the life of the process — `run` latches exactly as `off` does.

**Two things this row is NOT.** It is not P9: P9 sizes the VRAM prefix correctly *within* the
speculative path and is worth +4.16 % when that path runs (measured, unaffected). It is not "PLD
is bad on MoE": an earlier arm on a different prompt measured **1.307×**, so the answer is
workload-dependent — which is the point, since a byte-only break-even cannot see that.

**Also note the sign flip nobody predicted:** acceptance was HIGHER with thinking ON (0.6875) than
OFF (0.14). Reasoning prose repeats n-grams; freshly-generated code does not. Any future
speculation measurement here must state its thinking-mode state (the S1e-bis rule) — and must not
assume code-shaped output is the favourable case for prompt lookup.

**Candidate fix, not designed:** give TRIAL a wall-clock arm — time N drafting rounds against N
suppressed-draft rounds in the same call and latch on the ratio, instead of (or as well as) the
byte break-even. The brief lists the five questions to answer first; the two that decide the shape
are **what unit is timed** (seconds per ACCEPTED token, not per round — rounds yield a variable
number of tokens at ~1.5 tok/s) and **whether the byte test stays as a necessary condition**
(recommended: it is cheap, sound, and it is what refuses Qwen at zero cost).
**One thing the Done condition needs that this row does not have:** a reference workload where PLD
genuinely WINS, to prove a fix does not simply refuse everything. The 1.307× prompt was never
archived — constructing one is part of the work.
**Stop rule:** if a wall-clock TRIAL cannot be made cheap enough to run per process, say so and
close the row by documenting that `run` is a bytes claim only — the docs already say this as of
`5aa9aab`.

## SPIKE DONE 2026-09-10 — feasible, and the fix is CHEAPER than this row assumed

**The fix is now DESIGNED and ready to execute cold in a fresh session — BOUNDED, one class plus two existing call sites: [`docs/2026-09-10-row13-timed-bar-DESIGN.md`](docs/2026-09-10-row13-timed-bar-DESIGN.md)** (gitignored; §4 below restates every load-bearing number it depends on). **Take it AFTER row 14** — see the build order under the OPTIONAL heading.

**Full write-up: [`docs/2026-09-10-row13-timed-trial-SPIKE.md`](docs/2026-09-10-row13-timed-trial-SPIKE.md)**
(**gitignored — this disk only**; everything load-bearing is restated here). Probe
`plans/probes/p13_timed_trial.py` + `p13_report.py` + `p13_ab.sh`, **throwaway, also gitignored**.
6 arms, MiniMax-M2.7, `draft_tokens=3`, NTOK=200, WARM=96 discarded, prefix cache OFF,
BALLAST_GB=25, one `systemd-run` scope each. **No engine code was changed.**

**Method, and why it cost nothing to build:** a gate subclass that never latches, flips
`should_draft()` every 8 rounds, and timestamps each round from inside `record_round`. The loop
already asks `should_draft()` per round (`vaayu_base.py:5301`), a False answer is an ordinary
1-wide decode that still yields a token, and `record_round` already fires on EVERY round including
the suppressed ones (k=0). Block 0 excluded from aggregates.

**1. The signal is decisive; noise is not the obstacle.** Suppressed rounds are a very stable
reference — **0.4386–0.5838 s/tok across 90 blocks** — while drafting swings 0.357–1.074.
* **loser** (p9_e2e's code prompt, THINK=1): drafting faster in **0/15** and **2/16** paired
  blocks, pooled ratio **0.734 / 0.753** → a timed gate REFUSES it. Correct.
* **winner** (constructed, THINK=0 = the server's regime): **8/9** and **8/10** pairs, pooled
  **+22.8 % / +17.4 %**, acceptance to 1.000 → a timed gate ADMITS it. Correct.
* **This row's missing reference winner now EXISTS** (trap 3 discharged): reproduce a module given
  in the prompt three times verbatim with only the class name changed. Archived in the spike doc
  and the probe.

**2. The measured cost model, and the defect in arithmetic.** PLD drafts 0 or 3, never partial. A
k=3 round costs **3.124× / 2.900×** a plain decode round; a k=0 round costs 1.02–1.06×. **So a
4-position verify is ~3× a 1-position decode — ~0.73–0.78× PER POSITION, near-linear in positions
and nothing like the expert-byte ratio the gate models.** With that per-position factor `alpha`,
the honest bar is `a* = ((1+d)*alpha − 1)/d`. **Measured `a*` = 0.633–0.710 in EVERY arm against
the gate's byte break-even of 0.6259** — the byte bar is systematically too permissive, and P9's
admitted-then-lost arm measured **0.6875, inside exactly that gap.**

**3. The confound this row did not know about is DEAD.** `should_draft()` is False throughout
CALIBRATE, so P9's verify-width prefix sizing (`vaayu_base.py:5199`) NEVER fires on a process's
first call — TRIAL always measures under the load-time prefix while the RUN phase it predicts runs
under the verify-width one. Measured both ways: **`alpha` moves <1 %** (0.7801/0.7821,
0.7255/0.7250, 0.7372/0.7438) although absolute speed moved several percent. **TRIAL's measurement
is valid for RUN.**

**4. RECOMMENDED FIX — no new phase, zero extra tokens.** The gate already runs both arms:
CALIBRATE is `max(8, d+1)` 1-wide rounds (`base`), TRIAL is k-wide rounds. Feed both call sites a
wall-clock delta, take medians, `alpha = median_k / (base * (1 + draft_tokens))`, and have
`_decide_acceptance` compare against `max(byte_breakeven, ((1+d)*alpha − 1)/d)` — **keeping the
byte test as the necessary condition**, since it is free, sound, and what refuses Qwen at zero
cost. Fall back to bytes alone when the timing sample is too small or too dispersed, and SAY so in
`status()`. **This costs two timestamps per round**, where this row's original candidate
(block-alternating TRIAL) roughly doubles TRIAL's length. Validation set is the 6-arm table above.
**§8: worth 0 % on the server's default configuration** — `USE_PROMPT_LOOKUP = "auto"` is OFF for
MoE (`config.py:211`) and the gate is MoE-only, so this only ever reaches a user who explicitly
opts a MoE checkpoint in. **No knob needed** (a bar is not a feature).

**5. Two things NOT folded in.** (a) The latch — filed as row **24**. (b) **Cross-arm greedy id
identity is not assertable when the arms differ in WHICH rounds are 4-wide** (loser diverged at
index 169, winner at 84); a 4-wide verify is not bitwise a 1-wide decode. Compare ids only between
arms with matching per-round widths. Also observed: `alpha` drifts up within a call (0.749 → 0.832
over 130 rounds) as context grows — small across the adjacent CALIBRATE→TRIAL, but a
re-calibrating gate would have to re-measure rather than latch it.

# 24 · [x] Speculation decisions latch for the process — CLOSED 2026-09-10, REFUTED ON PRICE

**CLOSED 2026-09-10 by an offline replay spike. CPU only, no GPU round, no engine edit, nothing
shipped.** Re-evaluation cannot pay at this row's own re-entry price, and the sweep's optimum
points back at the latch. Full write-up
[`docs/2026-09-10-row24-relatch-replay-RESULTS.md`](docs/2026-09-10-row24-relatch-replay-RESULTS.md)
(**gitignored — everything load-bearing is restated here**); probe `plans/probes/p24_replay.py`
(throwaway, gitignored), replaying the six block-alternating traces row 13's spike archived
(`plans/probes/p13-minimax-*.json`) — both arms of the A/B interleaved through one call.

**THE REFUTATION.** Perfect-foresight, ZERO-cost re-evaluation is worth **+0.00 % / +0.49 % /
+1.15 % / +3.53 % / +6.28 % / +0.00 %** over a one-shot gate that decides correctly (6 MiniMax-M2.7
arms). It breaks even at a re-entry price of **0.5 / 1.0 / 2.2 / 5.9 token-equivalents** on the four
arms that have anything to win (two never switch at any price) — against the real price of
**~29.3** (`TRIAL_DRAFTED_TOKENS × breakeven`, the P4/row-13 figure). **5×–60× too expensive.** At
the real price the best causal policy lands **−15.5 % to −35.5 %** against the best static arm on
every arm, and the cadence sweep (re-trial every 2/3/4 blocks) is best at its SPARSEST setting —
the gradient says re-trial less, and the limit of that is the latch. **Even a re-entry made five
times cheaper still loses on 3 of the 4.** Do not re-scope this into "re-evaluate more cheaply".

**WHAT THE SAME TRACES SAY IS WORTH MONEY — it is row 13's remainder, restated.** The single
decision's WINDOW WIDTH, not its repetition. Deciding at the wrong single block costs **−10.9 % to
−27.0 %** (one-shot verdicts are correct at only **62–100 %** of the possible decision points), so
getting the one decision right is worth up to **+37 %** against re-evaluation's +6.28 % ceiling.
**"Decide later" is NOT the fix and that is the trap:** wrong verdicts cluster at BOTH ends — the
head (block 2) and the late code stretch (blocks 11-14, where acceptance rises to 0.83-0.88 and a
local sample says `on` while the call favours `off`) — so a later sample would be wrong on two arms
instead of one. **Pooling is:** the median `c_draft/c_supp` over the first **m ≥ 3** paired blocks
is correct on **6/6** arms; m=1 and m=2 are each wrong, on DIFFERENT arms. The cost of pooling (a
longer TRIAL, roughly linear in m) is NOT modelled — this establishes only that the accuracy is
there to buy.

**VALIDITY ANCHOR:** the replay reproduces both known on-device A/Bs from an independent trace
family — sim `always_on/always_off` **0.7373/0.7325** (loser THINK=1) against row 13 arm 1's
measured **0.8033**, and **1.1227/1.2504** (winner THINK=0) against arm 2's **1.1444**. Same sign,
same order, sim slightly more pessimistic. **It is a CEILING analysis, not a predictor:** blocks are
realized under alternation, so an oracle below the cost line refutes (what happened) while the
converse would only have been permissive.

**§8, all four:** (1) no knob — nothing was built; (2) N/A; (3) N/A, and the standing bound on this
whole area holds: the gate is MoE-only and `USE_PROMPT_LOOKUP` is OFF for MoE
(`vaayu/server/config.py:211`), so ANY outcome here — a win included — is worth **0 %** on the
server's default configuration; (4) what would unblock it: an operator opting a MoE checkpoint into
prompt lookup, or a measured case for flipping that default. Neither exists.

**LIMITS:** MiniMax-M2.7 only (Qwen refuses on bytes before a single draft, so it has no
alternating trace and no re-evaluation question), 2 prompts, 200-token calls, 8-round block
granularity — and a finer cadence pays the re-entry price MORE often, so the refutation is in the
safe direction. Thinking-mode state stated per arm (S1e-bis rule): 4 arms THINK=1, 2 THINK=0.

---

**Original row, kept for the premise it was opened on. Opened 2026-09-10 by row 13's spike** (`docs/2026-09-10-row13-timed-trial-SPIKE.md`,
gitignored — this row is the durable copy). Not measured on its own evidence yet; the number below
is a side observation of a run aimed at something else, and that is exactly its status.

**STRENGTHENED 2026-09-10 by row 13's four on-device arms — this row is no longer only about
acceptance drifting.** The DECISION WINDOW itself is unrepresentative: CALIBRATE and TRIAL are the
first ~40 rounds of a call, and on MiniMax-M2.7 they run at **0.510 s/token (drafting) / 0.532
(k=0)** while the whole 200-token call runs at **0.793 speculating and 0.637 not** — both arms slow
as context grows and the speculating one slows more. That is why a correctly measured, strictly
tightened, TIMED bar still admitted a configuration its own A/B measured at **−19.7 %**: in the
window it looked at, speculation really was marginally profitable. **So the case for re-evaluation
no longer rests on acceptance alone; the cost side moves too, and in the same direction.** Any
proposal still has to clear `TRIAL_DRAFTED_TOKENS × breakeven` ≈ 29.3 token-equivalents on MiniMax.
See row **13** and `docs/2026-09-10-row13-timed-bar-RESULTS.md` §4.3.

`AdaptiveSpecGate`'s decision latches for the PROCESS — `run` as permanently as `off`. Measured on
MiniMax-M2.7, THINK=1, one 200-token call, alternating 8-round blocks: acceptance went
**0.048 (blocks 4–8, reasoning preamble) → 0.875 (blocks 12–14, reproducing code)**, and the
per-block paired ratio with it, **0.43–0.51 → 1.21–1.28**. So a TRIAL sampled during the preamble
latches `off` and forfeits the win; one sampled in the code stretch latches `run` and keeps it
through the next preamble. **This is row 13's brief question 5, and it is orthogonal to row 13's
fix** — a timed bar is a better bar, still evaluated once.

**NOT the SETTLED "regime latch" item.** That one was refuted on **Qwen3-235B**, whose four arms
(192/192/160/900 tokens) emitted **zero** fenced code blocks, so its transition never happened at
all. This is MiniMax, the transition is measured, and the bar a re-entry must clear
(`TRIAL_DRAFTED_TOKENS × breakeven` ≈ 29.3 token-equivalents on MiniMax) is unchanged and still
applies — **any proposal here must price against it before proposing re-calibration.**

**§8: worth 0 % on the server's default configuration**, same as row 13 — `USE_PROMPT_LOOKUP` is
OFF for MoE by default and this gate is MoE-only. Weigh it accordingly.

---

# 14 · [x] The server reached 14 of the engine's 28 load parameters — CLOSED 2026-09-10, now 19 + 9 declined

**Self-contained brief: [`docs/2026-09-07-A-server-config-reachability-BRIEF.md`](docs/2026-09-07-A-server-config-reachability-BRIEF.md)**
(gitignored, this disk only — the numbers and the stop rule are restated here so the row survives
without it). Opened 2026-09-07 by the CLAUDE.md §8 audit. **Capability/correctness, NOT perf.**

`app.py:360` passes **14** kwargs to `AutoModel.from_pretrained`; `VaayuBaseModel.__init__`
(`vaayu_base.py:1617`) declares **28**. **Every user reaches Vaayu only through the server (§8), so
an unreachable parameter is a feature that does not exist.** NOT reachable: `dtype`, `max_seq_len`,
`layer_shards_saving_path`, `profiling_mode`, `compression`, `hf_token`, `prefetching`,
`use_torch_compile`, `layer_names_dict`, `tokenizer_class`, `auto_prefix_match`, `kv_persist_dir`,
`kv_persist_sync`, `enable_expert_usage`.

* **Tier 1, a shipped feature is half-wired:** `ENABLE_KV_PERSIST` IS exposed but `kv_persist_dir`
  (`vaayu_base.py:2694`, defaults to `~/.cache/vaayu/kv`) and `kv_persist_sync` (`:2700`) are not —
  **it can be turned on but not aimed, and never fsyncs.** Fix these first.
* **Tier 2, needs a DECISION not a default:** `max_seq_len` (only way to ask for a shorter context
  and buy VRAM back for the expert cache — but must clamp to the derived value, never above),
  `auto_prefix_match` (`config.py`'s `REQUIRE_SESSION_ID` comment already describes behaviour the
  server cannot turn off), `enable_expert_usage` (diagnostic telemetry, MoE-only, ships dark —
  measure its decode cost before changing its default).
* **Tier 3, deliberately NOT exposed, and §8 requires that be STATED:** the remaining nine are
  library/dev-facing loader plumbing, HF auth and profiling.
* **The durable part:** a test that reads `inspect.signature(VaayuBaseModel.__init__)` and asserts
  it equals `SERVER_EXPOSED | SERVER_DELIBERATELY_NOT_EXPOSED`, **so a new engine parameter breaks
  the suite until someone classifies it** — §8's decision, forced when it is cheapest. Assert the
  plumbing too: a knob `from_pretrained` never receives is decoration.

**Done means** Tier 1 exposed + documented in `vaayu/server/README.md`, a recorded decision per
Tier 2 parameter (declining is a valid deliverable), Tier 3 written down, the drift test proven to
fail on an unclassified parameter, suite green. **No on-device arm needed** unless you propose
changing `enable_expert_usage`'s default. **This row cannot be refuted, only reduced: closing with
2 knobs and 12 written-down declines is a success.**

## CLOSED 2026-09-10 — 5 knobs, 9 written-down declines, 1 drift test. CPU only, no default moved.

**19 of 28 now reachable; the other 9 are declined IN CODE with a reason each.** Every new knob
defaults to exactly what the engine already defaulted to, so **no behaviour changed and no
on-device arm was owed** (brief §5). Suite **1893 → 1906 passed, 12 skipped** — and the 1893 was
**re-measured against a stashed HEAD**, not quoted (it matched; second time in ten).

**Tier 1 (the half-wired feature), both exposed:** `KV_PERSIST_DIR = None`, `KV_PERSIST_SYNC =
False`. `ENABLE_KV_PERSIST` could be turned on but **not aimed and never fsynced**. The doc for
`KV_PERSIST_DIR` carries the trap the brief flagged: the write budget is DERIVED as
`shutil.disk_usage(dir).free // 2` (`vaayu_base.py:2857-2861`), so the directory choice sets the
cache's SIZE as well as its location. That budget stays derived — **making it a knob is a NEW row
with its own evidence**, per the brief.

**Tier 2 — all three EXPOSED (the user's call, 2026-09-10), all at their existing defaults:**
* `MAX_SEQ_LEN = None`. **Zero new logic, and the brief's "must clamp, never above" was already
  the engine's contract** — `plan_vram` raises `VramPlanError` above `s_max`
  (`vram_plan.py:697`) and warns at `pinning_budget=0`. The real trap is documented instead:
  between the derived operating point and `s_max` the load SUCCEEDS while the larger KV reserve
  eats the pinning budget. Both failure directions are loud; neither degrades quietly (§5).
* `AUTO_PREFIX_MATCH = True`. Closes the gap where `REQUIRE_SESSION_ID`'s own comment described
  behaviour the server could not turn off. Documented as: `False` gives an unnamed request NO
  reuse rather than approximate reuse, which on this workload is the larger loss.
* `ENABLE_EXPERT_USAGE = False`. **Default deliberately NOT moved**, which is why no cost
  measurement was owed. Documented MoE-only *by construction* (`vaayu_base.py:2785` warns and
  no-ops on dense — no experts to count, not a dense-second-class default, §5), and that flipping
  it puts `track_evicted_keys` on the decode path at an **unmeasured** cost.

**Tier 3 — 9 declines written down twice**, in `config.py` (a commented block) and in both docs:
`dtype`, `prefetching`, `layer_names_dict`, `tokenizer_class`, `layer_shards_saving_path`,
`profiling_mode`, `compression`, `use_torch_compile`, `hf_token`.

**The durable part — `tests/test_engine_param_reachability.py` (+13).** Reads
`inspect.signature(VaayuBaseModel.__init__)` and asserts it equals
`SERVER_EXPOSED | SERVER_DELIBERATELY_NOT_EXPOSED` (minus the positional checkpoint path), each
decline carrying a non-empty reason. **PROVEN both ways, not asserted:** written first and RED on
the right 5 names; then a dummy `dummy_drift_probe=None` parameter was added to
`VaayuBaseModel.__init__` and the test failed naming it, and `vaayu_base.py` was restored
byte-clean (`git diff --stat` empty). It also asserts the plumbing — `from_pretrained` really
receives every exposed name — so a knob that is decoration fails too.
**What did NOT need building:** the `config.py → ServerConfig → from_pretrained` chain was
ALREADY gated in both directions by `test_server_config_module.py:56/94` and
`test_server_config_passthrough.py:49`; only the engine-signature end was unguarded. The 5 new
knobs fall into those existing gates automatically.

**§8, all four:** (1) **it IS the knobs** — 5 of them, in `config.py`, each with the cost of
getting it wrong; (2) plumbed — `__main__.py` maps all 5, `app.py`'s `from_pretrained` passes all
5, and the drift test fails if either stops; (3) **reachable — every one is live on the default
server path**, which is the whole point of the row, though **all 5 are worth 0 % until an operator
sets one**: no default moved; (4) documented in `vaayu/server/README.md` AND
`documentation/server.md` (5 table rows + the Tier-3 note).

**Files:** `vaayu/server/config.py`, `app.py`, `__main__.py`, `vaayu/server/README.md`,
`documentation/server.md`, `tests/test_engine_param_reachability.py` (new).
[`docs/2026-09-10-A-server-config-reachability-RESULTS.md`](docs/2026-09-10-A-server-config-reachability-RESULTS.md)
(**gitignored — this disk only**; everything load-bearing is restated above).

---

# 25 · [x] NVMe APST A/B — CLOSED 2026-09-11: NULL on Qwen (A/B) and MiniMax (idle-gap bound); the documented fix was inert (FIXED)

Opened 2026-09-11. **Question:** does disabling APST move Qwen3-235B's cold TTFT or decode tok/s through
the real server (N=1, server-default config, flash)? Qwen is 47 % disk-stall; 47 % is NOT APST's cost —
APST only bites when the drive idles past the kernel's timeouts (`apst_primary_timeout_ms=100` for states
with total latency ≤ 15000 µs, `apst_secondary_timeout_ms=2000` for ≤ 100000 µs, this box).

**Found before any arm ran — `scripts/vaayu_optimize_os.sh` cannot disable APST without a reboot.** It
writes `/sys/module/nvme_core/parameters/default_ps_max_latency_us`, a plain `module_param` read only in
controller init (verified in `drivers/nvme/host/core.c`: no `param_ops` set hook). The live path is
`/sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us` → `nvme_set_latency_tolerance` →
`nvme_configure_apst` (0 = APST off). `vaayu/auto_model.py:31-40`'s detector checks the module param
FIRST, so after the script it prints "APST: optimized (disabled)" against a drive that still has APST on.
Its second candidate `/sys/class/nvme/<ctrl>/power/ps_max_latency_us` does not exist on this kernel
(6.17). **Fix is owed whatever the A/B says** (script + detector), because the boot warning tells users to
run it.

- [x] Arms: warm-up (discarded, first touch) → on-1 → off-1 → off-2 → on-2 (ABBA), `BALLAST_GB=25` fixed,
  prompt ~4179 tok, 2 turns, 128 tokens, temperature 0; each arm asserted the live tolerance at start and end.
- [x] **Verdict: NULL on Qwen3-235B at N=1.** Means off vs on: turn-1 decode 0.7620 vs 0.7557 tok/s (+0.8 %),
  turn-2 decode 0.6362 vs 0.6398 (−0.6 %), turn-1 TTFT 28.30 vs 28.44 s, turn-2 TTFT 25.95 vs 25.58 s. Replicate
  spread ~1 %; signs disagree between turns. Same work in every arm (text identical, device reads 1604.1–1605.1 GB,
  misses within 0.1 %). **Mechanism (100 Hz `/sys/block/nvme0n1/stat` sampler):** the drive is idle ~35 % of the
  time, but in gaps < 20 ms; only 2–5 gaps ≥ 100 ms per ~361 s arm (≤ 1 s total), identical on and off — APST's
  100 ms primary timeout almost never fires mid-generation. Bound ≤ ~0.1 %. **So APST is NOT a tax on the
  dominant cost; "Qwen is 47 % disk-stall" was never evidence that it was.**
  [`docs/2026-09-11-row25-apst-ab-RESULTS.md`](docs/2026-09-11-row25-apst-ab-RESULTS.md).
- [x] **MiniMax-M2.7 — NULL, 2026-09-11, no full A/B.** Warm-up + `mm-on-1`, APST on, `BALLAST_GB=25`, same
  harness (`plans/probes/apst_mm_chain.sh`, windows by `apst_windows.py`). Decode (2 × 70 s): **0 gaps ≥ 100 ms**.
  Cold prefill (205 s): **3 mid-generation gaps ≥ 100 ms** (0.10–0.11 s) and 19 candidates ≥ 80 ms (the sampler
  quantises to ~10 ms); prefix-hit prefill (14.6 s): 1 candidate. Longer gaps (0.6–1.2 s) are request
  boundaries. The 50–100 ms prefill band is NOT MiniMax-specific — Qwen's cold prefill has it denser.
  **Bound:** the 100 ms primary timeout only enters states with ≤ 15 ms total latency
  (`apst_primary_latency_tol_us=15000`; deeper states need 2 s idle, never reached mid-generation) → 20 × 15 ms
  = **0.30 s per 205 s cold turn, ≤ 0.15 %**, 0 on decode. Charging the full 100 ms tolerance instead gives
  1.0 %, already inside the Qwen A/B's ~1 % replicate spread — an A/B could not resolve it, so none was run.
  Arms agree (t1 TTFT 204.6/205.2 s, t2 decode 1.494/1.500 tok/s, text identical). `setup_and_speed_guide.md`
  now cites both targets; `envinfo.py`'s "APST is the live hardware tax" docstring corrected.
  **70B dense: unmeasured, not a blocker** (continuous ~37 GB/token reads — fewer gaps expected, not measured).
- [x] Toned down 2026-09-11 (user's call): the boot line is now `[Vaayu] … APST: active … no measured
  decode/TTFT cost; optional to disable` — info, NOT `[Vaayu Warning]`, so it no longer lands in env.json's
  `warnings` (`storage.apst_active` still records it structurally). `diagnose` still lists it (its tests pin
  that) but says "Minor". `documentation/server.md`, `setup_and_speed_guide.md` and `README.md` lost the
  "50–500 µs jitter over 80+ layers" claim; the guide now scopes the measurement to Qwen3-235B.
- [x] Fix `vaayu_optimize_os.sh` and the detectors to use the live knob — **done 2026-09-11.** Live-verified by
  toggling the knob (module param left at 100000 → check says disabled; back → warns); test fails 4/4 on the old read. One helper,
  `vaayu.auto_model.nvme_apst_tolerance(block_dev)`, reads `/sys/block/<dev>/device/power/pm_qos_latency_tolerance_us`
  (`device` resolves to the controller); both the boot check and `server/envinfo.py`'s `storage.apst_*` (→
  `diagnose`) call it, so they cannot drift apart again. The script writes that file; the setup guide's verify
  command reads it. **Also fixed on the way:** the boot check's `nvme.split("n")[0]` returned `''` for `nvme0n1`,
  so its second candidate path was `/sys/class/nvme//power/…` and its warning named no drive. envinfo hardcoded
  `nvme0`; it now reports APST active if ANY NVMe controller allows it. Test: `tests/test_nvme_apst.py`.
- §8: no server knob needed (OS setting, measured ~free). Plumbed/reachable: the boot info line and `diagnose`
  both read the live knob since `c256e18`, and the script they point at now takes effect without a reboot.

---

# 18 · [x] `output_hidden_states=True` crashes the forward — CLOSED 2026-09-11: FIXED; `output_attentions` refused

**Verdict (2026-09-11):** [`docs/2026-09-11-row18-output-hidden-states-RESULTS.md`](docs/2026-09-11-row18-output-hidden-states-RESULTS.md) (gitignored).

- [x] **Root cause CONFIRMED by instrumentation** (the brief's hypothesis). On Llama-3.2-1B the list before
  `[0:-2]` held **38 = 2 × `len(self.layers)`** elements: 19 per-layer sublists (embedding and lm_head
  EMPTY) then 19 bare Tensors from the flat `+=`. Origin: the AirLLM init `[] * n` is `[]` (flat scheme
  only); `6253fd1` (2026-05-25) made it real sublists without removing the `+=`. **Never worked in either form.**
- [x] **Fix:** flat `+=` deleted; the sublist scheme kept (decoder records its INPUT, norm its OUTPUT =
  transformers 5.x's convention, checked in `utils/output_capturing.py`); embedding and lm_head sublists
  dropped **by name**, never by emptiness. Returns `num_hidden_layers + 1` entries, post-norm last.
- [x] **`output_attentions=True` DECLARED OUT OF SCOPE — raises `NotImplementedError` at the top of `forward()`.**
  It had the same double structure, and could never work anyway: transformers 5.5.4's decoder layer returns
  a bare Tensor (weights never leave the attention module) and sdpa/flash never compute weights. Its dead
  accumulation code is removed.
- [x] **Gate:** new on-device `tests/test_output_hidden_states.py` (`HIDDEN_CKPT=`). RED on HEAD (3 errors, the
  exact `ValueError`); fixed 1B **3 passed** — count, shapes, finite, no duplicates, refusal, and values vs
  stock HF fp32: per-token rel-L2 **≤ 0.0184** against **≥ 0.459** to a neighbour layer (TOL 0.05, the test
  asserts TOL < every gap as its premise). **A shifted-layer mutant FAILS it (1.697).** MiniMax-M2.7 **2 passed,
  1 skipped** (value arm skips on EXL3). Qwen not run — shared loop, one dense + one MoE geometry covers it.
- [x] **Trap found:** whole-tensor rel-L2 CANNOT tell adjacent layers apart — the BOS token's massive activation
  (norm ~790 vs ~5) put neighbours 0.009 apart, under bf16 rounding (0.019). Leave position 0 out.
- [x] **Comment fixed:** `predict_peak_vram`'s note (~`vaayu_base.py:2187`) called the MoE layer-major → per-chunk
  fallback "unreachable" *because* of this crash. Now reachable via `output_hidden_states=True` only — a long
  diagnostic prompt on a MoE load may OOM (logits under-reserve + one chunk's hidden states, uncharged).
- **Known, not changed:** a chunked prefill returns only the LAST chunk's hidden states, as it does its logits
  (keeping all would retain `(L+1) × S × hidden` on GPU — 4.5 GB for the 1B at 65k).
- **§8:** no knob; nothing to plumb; not reachable through the server, by design — 0 % on throughput.
- Suite: stashed HEAD 1979 passed / 12 skipped → **1979 passed / 15 skipped**.

*Original row text follows.*

**Cold-pickup brief, with a ROOT-CAUSE diagnosis this row did not have:** [`docs/2026-09-10-row18-output-hidden-states-BRIEF.md`](docs/2026-09-10-row18-output-hidden-states-BRIEF.md) (gitignored). Two accumulation schemes collide — `vaayu_base.py:4397` allocates one empty sublist per layer, `:4667`/`:4680` fill sublists, but `:4766` does `all_hidden_states += (torch.cat(batch, 0),)`, which EXTENDS the outer list with a bare Tensor. `:4836` then cats a still-empty sublist. The `[0:-2]` at `:4834` is the hand-tuned compensation. **Line numbers below drifted: the crash is now `:4836`, not `:4594`.**

Found 2026-09-07 (incidentally, by a diagnostic probe that asked for hidden states). **Pre-existing
and untouched by the `fix/gate2-postreclaim-headroom` branch** — `git diff 879f63d..HEAD --
vaayu/vaayu_base.py` does not mention `all_hidden_states`.

```
File "/home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py", line 4594, in forward
    all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)
ValueError: torch.cat(): expected a non-empty list of Tensors
```

Repro: any `model.forward(input_ids=..., use_cache=False, return_dict=True,
output_hidden_states=True)`. Seen on Llama-3.2-1B under sdpa; the accumulation list for at least one
index is empty when the concat runs, so the entry is never appended for some layer class (the
`norm`/`lm_head`/decoder branches each append under different conditions — read `:4425-4437`).

**Severity: low, and say so.** Nothing in the engine, the server or any gate asks for hidden states;
this only bites a diagnostic. It is filed because a debugging session that reaches for hidden states
loses a tool exactly when it needs one — which is how it was found.

---

# 26 · [x] CPU-hybrid (AVX2) expert compute — CLOSED 2026-09-11: 8-bit KILLED; bytes-neutral hybrid +6.8 % Qwen / +22.0 % MiniMax over row 27 → design brief only, after row 27 — **ITS UNMET PRECONDITION IS NOW ROW 49 (2026-09-20)**

**2026-09-20 — READ THIS BEFORE CITING ANY NUMBER IN THIS ROW.** The hybrid half was **deferred, not
refuted** ("not killed … earns a design brief, not code"), behind two preconditions: *after row 27*
(met 2026-09-12) and *a cheaper GPU-side attack on the same slice must be priced FIRST* (**never done
— it is now row 49**). Row 28's later "close the CPU-hybrid idea" answered whether the non-EXL3
DeepSeek-V4 checkpoint rescued it, not whether MiniMax's +22.0 % stands. **Both measured arms have
since decayed:** the **+6.8 % Qwen3-235B arm is UNVERIFIABLE FOREVER** (model deleted 2026-09-20), and
the mechanism is **dead on Qwen3-Next** (disk 0.42 / 0.32 % of a decode token, row 36's spike). The
**+22.0 % is an arithmetic ceiling, never executed**, on a model whose EXL3 format **has no CPU
kernel**. See row **49**.

**Cold-pickup brief:** [`docs/2026-09-11-row26-cpu-hybrid-ceiling-BRIEF.md`](docs/2026-09-11-row26-cpu-hybrid-ceiling-BRIEF.md)
(gitignored — this row is the durable copy). **MEASUREMENT ONLY: no engine code in this row.**

**Question (user, 2026-09-11):** use a non-EXL3 8-bit 200B+ MoE and add CPU AVX2 compute so the mostly-idle CPU also
works on tokens? **Arithmetic verdict, UNMEASURED: NO as proposed.** (1) Decode is disk-bound — 3.70–3.83 GB/s off a
~7.1 GB/s drive, Qwen ~47 % disk-stall; idle CPU = I/O-bound box, the GPU waits too. (2) 8-bit is ~1.6× the bytes of
Qwen's EXL3 5.0 bpw (139 GB → ~220–240 GB) on the binding resource; estimated 0.56–0.86 → ~0.3–0.5 tok/s. (3) CPU
compute cannot remove a disk read. (4) The one real mechanism (KTransformers-style): compute RAM-resident experts on the
CPU so they skip PCIe (RTX 5060 Ti on **Gen5 x8**, ~32 GB/s peak vs host RAM ~50–90 GB/s, type unknown). It can remove
only **H2D + EXL3 construct**, and EXL3 has no CPU kernel, so it forces a CPU-friendly format on disk.

**Box:** i5-14600K, 6P+8E / 20 threads, **avx2 + avx_vnni, no avx512**; 32 GB RAM; 344 GB disk free.

- [x] **Step 1 — physics micro-benches** (no model): host RAM read GB/s and pinned H2D GB/s; CPU expert GEMV per model
  (shapes DERIVED from config / `_moe_geometry`) in bf16, fp32 and int8 weight-only, at 6 / 14 / 20 threads.
  **Done 2026-09-11 (`plans/probes/row26_physics.py`):** RAM read 65.6 GB/s (6 P-threads; 63.3 @14, 60.5 @20);
  pinned H2D **19.7 GB/s** (not Gen5 x8's ~32 theory), D2H 16.9, one Qwen expert H2D 634 µs; hidden GPU→CPU→GPU
  round trip ~18 µs. CPU expert at batch 1, 6 P-threads best: Qwen int8 348 µs / int4-g128 214 µs / bf16 697 µs /
  fp32 1132 µs; MiniMax int8 282 / int4 163 µs. int8 with fp32 activations is 3.7× slower than bf16 activations.
- [x] **Step 2 — Qwen3-235B decode**: one process, `MemoryMax=29G`, `BALLAST_GB=25` held in-process and freed after load,
  speculation OFF, greedy, warm-up discarded → unprofiled arm → `model.profiling_mode = True` → identical profiled arm
  (verify the toggle works; else a second process). Greedy ids must match across arms.
  **Done 2026-09-11:** U1 1.167 / P 1.018 / U2 1.173 s/token, greedy ids identical in all arms. **ANOMALY: the profiled
  arm is 13 % FASTER** — the brief's trap 1 (profiling only slows) is false on the fused-EXL3 forward. Same bytes
  (4.84 GB/token) and tiers (V 208.5 / R 135 / M 409 of 752); only the stall differs (0.548 vs 0.355 s). Diagnostic
  re-runs (`-run2`, `-run3`) DONE: mechanism = CPU idle-state wake latency + thread placement → row **27**.
- [x] **Step 3 — MiniMax-M2.7**, identical (module-list forward — a Qwen result does not transfer).
  **Done 2026-09-11:** U1 0.578 / P 0.595 / U2 0.580 s/token (profiling +3 %, as expected — no anomaly on
  module-list, whose H2D is synchronous). Ids identical U1/P/U2. Tiers V 186 / R 42 / M 268 of 496; 2.38 GB/token.
  Diagnostic `Usi` null; `Uspin` (spinners on every CPU) −27 % — confounded by SMT sharing, not a valid C-state test.
- [x] **Step 4 — compute per model:** shares `D` (disk_load) / `H` (h2d) / `C` (construct) / `G` (gemm), tiers V/R/M.
  Optimistic ceiling `1 / (1 − (H + C) / T_prof)`; realistic estimate with Step 1's CPU rate added back; 8-bit penalty
  `D × (8 / current_bpw − 1)`, `current_bpw` derived.
  **Done 2026-09-11 — the brief's formulas are UNPHYSICAL here.** Profiled split (run 2): Qwen D 0.371 / **H 0.329** /
  C 0.013 / G 0.062 of T_prof 1.007; MiniMax D 0.091 / **H 0.180** / C 0.008 / G 0.153 of 0.596. Formula (a) gives
  +51 % / +46 % but implies 0.666 / 0.408 s, **21 % / 9 % below the measured floor** (the copies were hiding disk time).
  Replaced by a measured floor: `NOEXP` arm (experts stubbed, router + routing sync kept) = serial non-expert time
  **0.074 s (Qwen) / 0.052 s (MiniMax)**; **device-busy** (union of in-flight raw read intervals) 0.772 / 0.396 s under
  keep-awake. Floor 0.845 / 0.448 s. Realistic hybrid (bytes-neutral ~5 bpw at int4's 47 GB/s, CPU work hides inside
  device-busy, per-layer tail = last expert's CPU compute + 18 µs hidden round trip): **0.871 / 0.461 s = +6.8 % /
  +22.0 % over keep-awake** (+35 % / +25 % over today). 8-bit: bytes ×1.595 (bpw 5.014 / 5.016 derived); brief's test
  `H + C ≤ D × 0.595` does NOT fire (D is exposed stall), corrected test (device-busy × 0.595 = 0.460 / 0.235 s > H + C)
  does; 8-bit floor 1.305 / 0.683 s vs today 1.180 / 0.577 → **−9.6 % / −15.5 % tok/s** before cache-hit loss.
- [x] **Verdict + RESULTS doc**, box bandwidths added to Environment.
  [`docs/2026-09-11-row26-cpu-hybrid-ceiling-RESULTS.md`](docs/2026-09-11-row26-cpu-hybrid-ceiling-RESULTS.md).
  **VERDICT:** (1) the proposal as asked (8-bit non-EXL3 + AVX2) is **killed**. (2) The bytes-neutral hybrid
  (KTransformers-style) is **not killed** — CPU int4 expert 214 µs vs 634 µs to copy it over PCIe (measured 19.7 GB/s,
  not Gen5 x8's 32) — and by the brief's rule (realistic ≥ 15 % on one model, no loss on the other) it **earns a design
  brief, not code** — but only AFTER row 27, measured against row 27's baseline (a hybrid keeps the CPU busy, so an A/B vs
  today would credit it with row 27's win). Its value is then mostly MiniMax, and a cheaper GPU-side attack on the same
  slice (MiniMax module-list GEMV ~0.31 ms/expert vs ~0.08 ms on the fused path) must be priced first. **Second GPU-side lead on the same slice (added 2026-09-12):** that forward's H2D is **synchronous** — `moe_patch.py:694` passes `non_blocking=False` **deliberately** (the comment says async there leaks host RAM), so it cannot hide behind disk the way the fused forward's does. Not a flag flip: a fix needs an explicit buffer-lifetime mechanism. Price it before building any CPU path to win the same 0.114 s. Turn-level
  hybrid gain over keep-awake: ×1.05–1.06 Qwen, ×1.05–1.21 MiniMax (prefill cannot be helped). **The 8-bit half is
  closed by refutation; the hybrid half is closed with its number and handed to a design-brief decision.**
  **Probe trap, new:** on the fused-EXL3 forward `profiling_mode=True` is 13–15 % FASTER than unprofiled (its
  `synchronize()` spin-waits keep a core awake) — never read a profiled Qwen arm as an upper bound on true time.

**Traps:** profiling adds `torch.cuda.synchronize()` and serialises overlap → profiled phases are UPPER bounds on what
removing them saves, true tok/s is the unprofiled arm · MiniMax load dies at `MemoryMax=22G`, use 29G · exclude the
prefill forward · no py-spy · no literal model numbers.

**Decision rule (proposed; user may override):** KILL if the optimistic ceiling is **< +15 %** decode tok/s on both
models (a real hybrid is kernels + a new format + loader + dispatch; shipped wins here were +3–12 %). KILL the 8-bit
variant if `H + C ≤ D × (8 / current_bpw − 1)` on both — **expected**. Only a REALISTIC ≥ +15 % on one model with no
loss on the other earns a design brief (still not code). Report the decode win as a share of a whole agent turn: a CPU
path cannot help prefill, and a cold MiniMax turn spent ~205 s in prefill (row 25). **A refutation CLOSES the row.**

**§8:** no knob; nothing to plumb (`profiling_mode` stays out of the server, the probe is in-process); the hypothetical
feature WOULD be reachable — a per-token N=1 decode effect, the server's mode; not built, unblocked only by this verdict.

---

# 28 · [x] Does a non-EXL3 model change the CPU-AVX2 verdict? — CLOSED 2026-09-12: NO, it makes the hybrid WORSE; and the checkpoint does not load

**Source:** the user's follow-up to row 26 (2026-09-11): *"whole question was if i can run non-EXL3 model on this box and
also use idle CPU to generate token… now i have one in `/home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731`,
so re-run the measurement on non-EXL3."* **Durable copy of the verdict is this row** (docs/ is gitignored).
Detail: [`docs/2026-09-12-row28-nonexl3-dsv4-cpu-hybrid-VERDICT.md`](docs/2026-09-12-row28-nonexl3-dsv4-cpu-hybrid-VERDICT.md).

**MEASUREMENT ONLY — no engine code written, none proposed.**

## The three answers

**1. The measurement cannot be run — the checkpoint does not load.** Three independent blockers, each verified 2026-09-12:
- `AutoConfig.from_pretrained(..., trust_remote_code=True)` raises `ValueError: ... model type deepseek_v4 but Transformers
  does not recognize this architecture`. transformers 5.5.4 carries `deepseek_v2/v3/vl/vl_hybrid`, no `v4`. Vaayu hits this
  at `vaayu/vaayu_base.py:1913`, essentially the first thing `from_pretrained` does.
- `trust_remote_code` has nothing to load: **no `auto_map` in `config.json`, no `modeling_*.py` at repo root.** (Contrast
  MiniMax-M2.7, which is also unknown to transformers but runs here *because* its repo ships `modeling_minimax_m2.py`.)
  What the repo ships instead is `inference/` — DeepSeek's standalone torchrun/model-parallel reference impl, not HF.
- Tensor names are private (`layers.3.ffn.experts.7.w1.weight`, `layers.3.attn.wq_b.weight`, `layers.3.hc_ffn_fn`) with no
  `model.` prefix, and the math needs `tilelang==0.1.8` + `fast_hadamard_transform` (`fp4_gemm`, `fp8_gemm`, `sparse_attn`,
  `hc_split_sinkhorn`) — **both absent from the venv**, no transformers equivalent. torch 2.11 does have `float4_e2m1fn_x2`
  / `float8_e4m3fn` / `float8_e8m0fnu` and the GPU is sm_120, so a port is possible — it is just a **new architecture port,
  weeks**, not a quant-format switch. (Plus ~155 GiB for the `splitted_model/` persist copy; 190 GB free.)

**2. Non-EXL3 makes the CPU-AVX2 hybrid WORSE, not better — close the idea.** Projected on this model with row 26's
measured box constants: **+4.5 … +5.0 %** decode tok/s over row 27's keep-awake baseline, against row 26's **+6.8 % (Qwen)
/ +22.0 % (MiniMax)**. The mechanism is still real and still the same size (CPU computes one expert in **287 µs** vs
**715 µs** to copy it over PCIe) — but a hybrid can only remove *exposed* host-side time, and this model's 2.58×-lighter
token shrinks that term while its own CPU tail (43 layers × ~305 µs) does not shrink at all. Cost remains months
(CPU kernels, a new dual-kernel on-disk format + its quality validation, loader, dispatch). **Row 28 removes the last open
justification for the hybrid, which was "maybe a non-EXL3 model changes the picture".**

**3. The premise "non-EXL3 ⇒ 8-bit ⇒ more bytes" does not describe this checkpoint — and the real prize is elsewhere.**
Derived from its own safetensors headers (nothing typed): 43 layers × 256 experts, top-6, **13,369,344 B/expert = 12.750 MiB
= exactly 4.2500 bpw fp4** (group-32, e8m0 scales) — *fewer* bytes per weight than EXL3 5.014. Per token it moves
**258 slots × 12.750 MiB = 3.212 GiB**, vs **Qwen 8.285** and **MiniMax 4.100**.

| | Qwen3-235B (measured) | DSV4 (projected, serial band 1×–2×) |
|---|---|---|
| floor = serial + device-busy | 0.845 s | **0.340 – 0.375 s** |
| today-equivalent | **1.180 s = 0.85 tok/s** | **0.439 – 0.474 s = 2.11 – 2.28 tok/s** |
| + row 27 keep-awake | 0.930 s = 1.08 tok/s | 0.370 – 0.405 s = 2.47 – 2.70 tok/s |
| + CPU hybrid on top | 0.871 s = 1.15 tok/s | 0.353 – 0.388 s = 2.58 – 2.83 tok/s |

**≈ 2.5–2.7× Qwen's decode, from architecture (top-6 of 256 over 43 layers, ~6.5 B routed active params vs ~22 B), with the
CPU doing nothing.** Miss fraction assumed 54 % because both measured models hit 54.4 / 54.1 % at the same per-layer cache
coverage (Qwen 12.8 %, MiniMax 14.0 %, DSV4 11.7–12.3 %); sensitivity 50 → 70 % gives 2.16 – 3.17 tok/s, so the conclusion
does not depend on it. Unblocked only by the port.

**Bonus, flagged not priced:** the checkpoint carries a **10.12 GiB MTP/DSpark draft module** (`num_nextn_predict_layers=1`)
— built-in speculative decoding, no second checkpoint to download. Dense spec decoding measured 5.57× (row P7) but the MoE
mechanism differs; **do not assume it carries over** — measure after a port, if there ever is one.

**Row 26's bytes rule, reproduced on a non-EXL3 checkpoint:** DeepSeek publishes an fp8 expert build of this same model
(`inference/README.md`: drop `"expert_dtype": "fp4"`, `--expert-dtype fp8`). Ratio 8/4.250 = **1.882×** → device-busy
0.305 → 0.574 s, floor **0.61 s = 1.64 tok/s** vs 2.67–2.95 at fp4. **Picking the fp8 build of the very same model costs
~43 % of its decode speed.** The rule is about bytes per token, never about quant lineage.

## Ranking after this row

1. **Row 27 (keep-awake)** — measured +27 % on Qwen, ~1 day, model-independent, reaches users at N=1. Still priority 1;
   nothing here outranks it.
2. **Port DeepSeek-V4-Flash** (weeks: HF modeling + config with `auto_map` the MiniMax way, key remap → module-list forward,
   fp4/fp8/sparse-attn/Sinkhorn kernels via TileLang or Triton 3.6.0, ~155 GiB persist copy). Largest number on the board.
3. **CPU-AVX2 hybrid — recommend CLOSED.** +4.5–5.0 % here, +6.8 / +22.0 % on the EXL3 pair, months of work.

**§8:** (1) no knob — no engine behaviour changed, none proposed. (2) nothing to plumb; the projection is a gitignored
probe, `plans/probes/row28_dsv4_projection.py` (no GPU, no model load, ~20 s). (3) reachable? N/A today — the checkpoint
does not load; a ported model's decode effect WOULD be a per-token N=1 effect, the server's only mode. (4) not reachable
today: the hybrid is unblocked by nothing (close it); the 2.5× is unblocked by the port; row 27 by its own server A/B.

**Stop rule honoured:** the question is answered with numbers and the row is closed. It is NOT re-scoped into a port plan.

---

# 27 · [x] Keep one CPU core awake during decode — the idle package's wake latency costs Qwen3-235B ~21 % — PRIORITY 1 (added 2026-09-11)

**Source:** found by row 26 (see its RESULTS §7). **Durable copy of the evidence is this row** (docs/ is gitignored).

**Observation:** on Qwen's fused-EXL3 forward, `profiling_mode=True` (which only ADDS `synchronize()` calls) decoded
13–15 % faster than unprofiled arms on both sides of it, in two processes, same tokens / bytes / tiers. Token-by-token the
gap tracked disk reads (r = 0.81, ~0.37 ms per read).

**Tests (Qwen, same process, greedy ids identical in every arm; s/token, 63 decode forwards):**

| arm | run 2 | run 3 | reading |
|---|---|---|---|
| unprofiled U1 / U2 | 1.171 / 1.190 | 1.156 / 1.138 | baseline |
| GIL switch interval 5 → 0.5 ms (`Usi`) | 1.185 | — | GIL starvation REJECTED |
| ONE `SCHED_IDLE` spinner on an E-core (`UspinE1`) | — | **0.970 (−15.5 %)** | package-level idle: C2 residency barely moved (0.74 → 0.70) |
| all threads pinned to P-cores, no spinner (`UP`) | — | **1.060 (−7.6 %)** | thread placement |
| spinners on all 8 E-cores (`UspinE`) | **0.930** | **0.901 (−21.5 %)** | both: 1 − 0.845 × 0.924 = 21.9 % predicted |

Raw Rust read latency 2.9–3.1 ms/call in EVERY arm (the drive is not slower); what shrinks is the host side of each read:
loader time minus raw reads 1.22 → 0.34 ms/read, slab-acquire wait 69 → 20 ms/token, device-busy gaps 0.80 → 0.74 s.
Box idles in `C2_ACPI` (declared exit latency 127 µs), governor `powersave` / EPP `balance_performance`, CPU 7.5 % busy
during decode. **Why only Qwen (most likely — inferred, not instrumented):** its forward issues async copies then SLEEPS waiting on
reads; MiniMax's synchronous copies block inside `cudaMemcpy`, which under CUDA's default scheduling spin-waits, so
MiniMax gains only −2.6 % (0.577 → 0.562). Spinners on ALL 20 CPUs
(`Uspin`) slowed MiniMax 27 % — SMT siblings shared with engine threads; never put a spinner on a P-core sibling.

- [x] Design the keep-awake: a GIL-free busy thread (or subprocess) at `SCHED_IDLE` on ONE detected E-core (lowest
  `cpuinfo_max_freq`; never a typed CPU id — CLAUDE.md §5; on a non-hybrid CPU pick the least-loaded core), started at
  generation begin, stopped at end; optional P-core affinity for the engine's own threads on hybrid CPUs. Measure 1 vs
  2 spinners and the power cost.
- [x] §8: knob in `vaayu/server/config.py` + `README.md` with the measured cost of getting it wrong (−21 % tok/s on Qwen
  off; one E-core's power on). Plumbed through `app.py`/`engine.py`. Reachable at N=1 — the server's mode.
- [x] Same-day A/B THROUGH THE SERVER on Qwen and MiniMax (ballast fixed, ids identical); check prefill TTFT (unmeasured
  — prefill is read-heavy too and may gain) and the 70B dense.
- [x] Root-only alternatives for users go in `documentation/TUNING.md` as documentation: PM QoS via
  `/dev/cpu_dma_latency` (mode 0600 on this box) or disabling `C2_ACPI` (`cpupower idle-set`).

**CLOSED 2026-09-12 — built, measured through the server, shipped ON. Stop rule PASSED.**
[`docs/2026-09-12-row27-keepawake-RESULTS.md`](docs/2026-09-12-row27-keepawake-RESULTS.md) (gitignored; the numbers
below are the durable copy).

**Result — through the server, N=1, its real default config (prefix cache ON), greedy, ids identical in every arm:**

| model | forward | off | keep-awake + pin | delta | TTFT |
|---|---|---|---|---|---|
| **Qwen3-235B** | fused-EXL3 MoE | 0.9229 tok/s | **1.1363** | **+23.1 %** (n=3, spread 0.5 %) | 1.680 → 1.306 s (−22.3 %) |
| MiniMax-M2.7 | module-list MoE | 1.7405 | 1.7625 | +1.3 % | 0.839 → 0.798 s |
| Llama-3.3-70B | streamed dense | 0.2745 | 0.2824 | +2.9 % | 3.454 → 3.321 s |
| Llama-3.2-1B | dense, VRAM-pinned | 111.7 | 111.2 | **null** — the control: it reads no disk | — |

**Qwen sweep, prefix cache OFF (every arm pays a full prefill), served / decode-excluding-prefill:**
off 0.7952 / 1.1484 · spin1 +2.7 / +3.6 % · **spinAll +8.8 / +12.4 %** · pin alone +0.6 / +0.9 % ·
**both +10.1 / +14.1 %**. The win is the PACKAGE leaving deep idle, not the core: one spinner buys a quarter of what
eight do. **P-core pinning is worth far less through the server (+0.9 %) than in-process (row 26: −7.6 % wall)** — it
still composes (+12.4 → +14.1 %), so it ships on, but the spinners are the lever.
**Mechanism, measured directly** (per-CPU `cpuidle` sums over one Qwen request): C2_ACPI residency **1.142 → 0.492 s**,
wakes **1.74 M → 1.22 M**.

**Built:** `vaayu/keepawake.py` + `tests/test_keepawake.py` (23) + `tests/test_keepawake_wiring.py` (6), TDD.
E-cores DETECTED off `cpuinfo_max_freq` (CLAUDE.md §5), and a spin CPU can never be an SMT sibling of an engine CPU
**by construction** — row 26's `Uspin` −27 % MiniMax confound is now unreachable. Spinner = `/bin/sh` arithmetic loop
at `SCHED_IDLE` (99.5 % of one E-core measured), ~1 MB RSS, no `chrt`/`taskset` dependency. Wired into `generate`,
`batched_generate` AND `speculative_generate`. Knobs `KEEP_AWAKE="auto"` / `PIN_ENGINE_THREADS=True` in
`vaayu/server/config.py` → `__main__.py` → `ServerConfig` → `from_pretrained`, documented in `vaayu/server/README.md`
and `documentation/TUNING.md` §5b (with the root-only alternatives: `/dev/cpu_dma_latency`, `cpupower idle-set`).

**Four findings worth keeping (each has a test that fails without its fix):**
1. **`speculative_generate` is a SEPARATE entry point** — `choose_speculative_target` hands it to `stream_generate`
   whenever a speculator is loaded, so wiring only `generate` misses every speculating request (and
   `USE_PROMPT_LOOKUP="auto"` turns speculation ON for dense by default).
2. **Orphaned spinners.** SIGKILL runs no `atexit`, and this box OOM-kills loads (exit 137 on three gates), so a
   spinner would burn an E-core until reboot. It now watches its owner's pid with `kill -0` and exits by itself.
3. **Sticky P-core placement.** Restoring only the SAVED threads left every thread born inside a generation pinned for
   the life of the process — half the effect applied silently forever, and in an A/B every arm after the first pinned
   one contaminated. Exit now returns new threads to the entering thread's mask too.
4. **Keep the loops under their PUBLIC names.** A wrapper + `_private` rename broke **46 tests**: stub engines across
   the suite borrow these loops as unbound methods. `keepawake_for(model)` no-ops for a partial object.

**Measurement traps this run hit** (all in `plans/probes/row27_server_ab.py`): `fastapi.testclient` delivers a whole
SSE response at once, so client-side timing read **536,072 tok/s** — take the numbers from the server's own
`requests.jsonl`; the server's `capture_stdio=True` swallows a probe's own prints into `logs/<run>/server.log`; and
without a discarded warm-up plus alternating arm order, warm-up alone reads as **+78 %** for whichever arm ran later.

**Not measured:** package POWER. `/sys/class/powercap/intel-rapl:0/energy_uj` is root-only on this box, so the cost of
leaving it on is stated qualitatively (one E-core per spinner at `SCHED_IDLE` for the duration of a generation) rather
than in watts. Non-hybrid CPUs decline with a WARNING (unmeasured hardware; `allow_non_hybrid` reserves a whole
physical core for an opt-in).

**Stop rule:** if the server A/B does not reproduce ≥ +10 % decode tok/s on Qwen, close as refuted-through-the-server
and record why the in-process effect did not transfer.

---

# 29 · [x] CLOSED 2026-09-12 (decision taken → row 30) — Pick a non-EXL3 MoE Vaayu can actually LOAD, then measure the CPU+GPU hybrid on it (added 2026-09-12)

**Source:** the user, pushing back on row 28: *"EXL3 use custom CUDA driver that only run with GPU, and because of this
limitation i want to use non-EXL3 model and use CPU and GPU both… i am not letting this idea go."* **Correct — and rows
26/28 agree**: EXL3 has no CPU kernel, so a non-EXL3 checkpoint is a **prerequisite** for the hybrid, not an optimisation.
Row 28's "+4.2 %" was a number for **DeepSeek-V4-Flash specifically** (a 147 GB disk-streamed model — the regime where the
hybrid is structurally weakest), not a statement about non-EXL3 formats. This row prices the regimes and names the download.
**Durable copy of the numbers is this row.** Detail: [`docs/2026-09-12-row29-hybrid-model-selection-BRIEF.md`](docs/2026-09-12-row29-hybrid-model-selection-BRIEF.md).

**DECISION + 1-DAY MEASUREMENT. No engine code until step 3 below reports.**

## The constraint that decides everything: which expert formats have a CPU kernel (checked 2026-09-12)

| format | CPU kernel | GPU | verdict |
|---|---|---|---|
| **EXL3** | **none** | yes | impossible for a hybrid — the user's point, confirmed |
| **int4 group-128** | **`torch._weight_int4pack_mm_for_cpu`, measured row 26 at 214 µs/expert (46.9 GB/s)** | yes | **the path.** Needs a one-time repack GPTQ/AWQ → torch CPU layout |
| int8 weight-only | `torch._weight_int8pack_mm`, 348 µs | yes | 8 bpw doubles footprint → out of the good regime (rows 26/28) |
| bf16 / fp32 | yes, 697 / 1132 µs | yes | zero kernel work, but 16/32 bpw — small models only |
| **MXFP4** (gpt-oss) | **none in torch** | yes — transformers `mxfp4`; `kernels` 0.13.0 + triton 3.6.0 ARE installed | **loads today**, CPU path needs an AVX2 e2m1 kernel |
| fp8 e4m3 block-128 | none in torch | yes (Vaayu already patches FineGrainedFP8, `vaayu_base.py:2964`) | GPU-only today |

**Installed backends: bitsandbytes only.** `gptqmodel`/`auto_gptq`/`autoawq`/`compressed_tensors` are ABSENT — a GPTQ or
AWQ checkpoint needs one `pip install` first.

## Projected decode, all candidates (model calibrated on both measured models; tiers within 0.01–0.03, timing +7 % Qwen / −16 % MiniMax — treat ±20 %)

| candidate | footprint | GB/tok | vram% | disk% | **T_gpu → tok/s** | **hybrid** |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B int4 | 15.4 GB | 0.96 | 97 | 0 | 0.049 → **20.5** | +14.8 % |
| **Qwen3-Next-80B-A3B int4** | 41.1 GB | 0.80 | 91 | 3 | 0.055 → **18.1** | **+21.1 % (best in set)** |
| gpt-oss-120b MXFP4 | 60.9 GB | 1.90 | 71 | 18 | 0.090 → **11.2** | **−4.6 % (LOSES)** |
| GLM-4.5-Air int4 | 52.9 GB | 3.31 | 67 | 15 | 0.136 → **7.3** | +7.7 % |
| Qwen3-235B-A22B int4 | 120.6 GB | 7.54 | 32 | 52 | 0.800 → 1.25 | +7.8 % |
| DeepSeek-V4-Flash fp4 *(won't load, row 28)* | 147.2 GB | 3.45 | 27 | 59 | 0.398 → 2.5 | +4.2 % |
| **Qwen3-235B EXL3 — today** | 135.8 GB | 8.90 | 28 | 54 | **0.85 MEASURED** | n/a |

## Two conclusions

**1. The hybrid has a regime, and nobody has tested it in that regime.** Footprint ≈ 2× the ~20 GB cache is its best case
(**+21 %**); ≫ cache is its worst (+4–8 %, rows 26/28); and at 3–5× it can go **negative** (gpt-oss-120b −4.6 %, because its
per-layer CPU tail 36 × 301 µs = 10.8 ms exceeds the GPU arm's 6 ms host tail). **But the honest ceiling on THIS box is
+21 %**, because the GPU link (18.7 GB/s measured) is only 2.5× slower than 6 P-cores (46.6 GB/s) — replacing a copy with a
compute saves a 2.5× factor on a term that is rarely the bottleneck. KTransformers-class wins need many more cores and far
more RAM bandwidth per core.

**2. The far bigger number is the model choice itself — and it is a DOWNLOAD, not code.** Same engine, model whose experts
fit the box: **gpt-oss-120b ≈ 11 tok/s (13× today, zero new kernels)**, GLM-4.5-Air ≈ 7.3 (9×), Qwen3-Next-80B ≈ 18 (21×).
This is exactly the carried row already on this board ("no resident-size checkpoint on disk… the single largest usability
change for the agent"). The hybrid adds ~20 % **on top of** that, at best.

**3. What breaks next in that regime:** for every resident candidate the **serial per-layer term (0.8 ms × layers) becomes
the largest phase** — 38 ms of Qwen3-30B's 49 ms. That is Vaayu's own per-layer Python/launch overhead, not bytes. **In the
resident regime the next perf lever is CUDA graphs / fused per-layer dispatch, not AVX2.**

## Order of work

1. `pip install gptqmodel` (or `autoawq`) — nothing GPTQ/AWQ loads without it.
2. **Download `Qwen3-30B-A3B-Instruct-2507` int4 (~17 GB)** — `JunHowie/…-GPTQ-Int4`, `cyankiwi/…-AWQ-4bit`, or
   `stelterlab/…-AWQ`. `model_type: qwen3_moe`, the same family Vaayu already runs at 235B. Persist, generate.
   **This would be the first non-EXL3 MoE ever run on this box** (§0 dispatch table: fused non-EXL3 = "gated in, not
   perf-tuned", on-device today: none) — expect to find load-path bugs, which is why it is 17 GB and not 63.
3. **Run `plans/probes/row26_decode.py` on it unchanged** → real `serial` / `device_busy` / `h2d` / `gemm` / tier split.
   One afternoon; replaces every projection above with a measurement and confirms or kills the regime story.
4. Only then: write the int4 repack (GPTQ/AWQ packing → torch CPU layout), flag it on one forward's RAM tier, A/B.
   **Stop rule: if measured H2D + gemm + tail is under ~15 % of the token, the ceiling is not there — close it.**
5. **Row 27 (keep-awake, +27 % MEASURED, ~1 day) comes before all of it** and applies to every model here.

**Do NOT** download an 8-bit build (rows 26/28: bytes per token is the whole game) or another EXL3 model (no CPU kernel).
**Disk:** 190 GB free, Vaayu needs its own `splitted_model/` copy → budget 2× the download. Deleting the unusable
DeepSeek-V4-Flash checkpoint returns 156 GB.

**§8:** (1) knob — a hybrid would need `expert_compute_device` in `vaayu/server/config.py`; a model choice needs none.
(2) nothing plumbed, nothing built; the picker is a gitignored probe, `plans/probes/row29_hybrid_model_picker.py`.
(3) reachable — every effect here is a per-token N=1 decode effect, the server's only mode. (4) not reachable today: the
hybrid is unblocked by step 3's measurement, the 9–21× by steps 1–2 (a load path never exercised on device).

---

# 28b · [ ] Port DeepSeek-V4-Flash-0731 so Vaayu can run it (added 2026-09-12) — sequence step S4

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

**Steps (each with its own verify):** 1 **kernels first, standalone, on sm_120** (2–5 days; **stop rule: if TileLang
will not build or validate, decide Triton-or-abandon HERE**, before any modeling code) → 2 HF modeling code, MP
machinery stripped, **experts as `nn.ModuleList`** so it lands on the module-list forward MiniMax already exercises
(1–2 weeks; verify logits against `inference/generate.py` BEFORE measuring) → 3 key remap + persist (verify
`ExpertOffsetIndex` finds **43 × 256 = 11,008** derived entries) → 4 first on-device generation → 5 run
`row26_decode.py`, compare against row 28's prediction (serial 0.035–0.070 s, device-busy 0.305 s, T 0.439–0.474 s)
→ 6 *optional* the **10.12 GiB MTP draft module** already in the checkpoint (free speculative drafter; dense spec
measured 5.57× in row P7 but **the MoE mechanism differs — do not assume it transfers**).

**Good news that is not obvious:** its disk keys are `layers.3.ffn.experts.7.w1.weight` — they already match
`ExpertOffsetIndex`'s only rule (`\.experts\.(\d+)\.`), so streaming works once the model loads. **The hard part is
the model, not the streaming.** Watch: dense footprint **8.24 GiB** (vs Qwen's ~5.8) must be VRAM-resident or the
per-token H2D (~0.31 s) swamps everything.

**Check before starting:** a newer transformers may have added `deepseek_v4`; that deletes steps 1–2.

**§8:** (1) no new knob — the model path is already a parameter; state it in the commit. (2) server must be able to
load it — grep `app.py` / `engine.py` for the `trust_remote_code` path. (3) reachable — per-token N=1, the server's
only mode. (4) not built; step 1 unblocks step 2, step 4 unblocks the measurement.

---

# 30 · [x] Run a non-EXL3 int4 MoE, then build CPU+GPU hybrid decode (added 2026-09-12) — sequence steps S1–S3 — CLOSED 2026-09-19 (last item: RAM-ceiling fix, +15.1 %)

**Status 2026-09-19: ROW CLOSED — its last item, the RAM-ceiling observation, FIRED and is FIXED (+15.1 % decode tok/s on Qwen3-Next, bullet below).** (was, 2026-09-18: S1, S2, S2b, 30-P and S3 are all CLOSED — S3 by its re-gate (bullet below: gate 1 6.96 / 6.99 % < 10 %). The row stays `[~]` ONLY for the `[ ]` OPTIONAL RAM-ceiling observation below (found by the 30-P dry-run). The `[bug]` uncapped slab arena moved to its own row **40** on 2026-09-18.** (was, 2026-09-16: S1, S2 and S2b closed, S3 the only step left.)

- [x] **S3-fill CLOSED 2026-09-18 — a second fill site EXISTS (fill-behind), but it needs a PINNED RAM tier, and that tier alone is the win: +10.62 % measured (row 30-P below).** Spike, no engine code. Doc (self-contained, re-verifiable): [`docs/2026-09-18-row30-S3-fill-site-SCOPING.md`](docs/2026-09-18-row30-S3-fill-site-SCOPING.md) (gitignored — this bullet is the durable copy). Findings, all measured on Qwen3-Next int4 (non-EXL3 fused forward):
  - **S3 as specified → 0 % VRAM hits**: nothing refills the cache, so ALL 480 routes/token go to the CPU, not the 79.7 misses S3-pre priced (replay of the S2b(iii) traces through the real `ExpertVRAMCache`, `plans/probes/s3fill_replay.py`).
  - **Fill-behind** (CPU computes the miss AND it is uploaded + `put()` off the critical path) keeps today's hit trajectory EXACTLY (83.3 % / 88.0 % warm, identical `put()` sequence), with ≥ 1 decode token of slack between upload and first reuse (min gap 1 forward; upload drains in ~0.1 ms). **The S3-pre mechanism objection is dissolved.** Other sites fail: prefill-only fill sustains 42–48 %; prefill seeding adds −1.2…+0.5 pp; admit-on-second-touch −0.4…−12 pp (unstable). 42–48 % of today's fills are dead on arrival.
  - **Prerequisite: the RAM tier on this model is PAGEABLE** (`moe_supports_slab_arena` requires `is_exl3`, `vaayu_base.py:786`; log `Expert slab arena: SKIPPED`). Issuing one expert's upload from pageable RAM holds the host **157 µs**; from pinned **32–40 µs** (`plans/probes/s3fill_bench.py`, real bytes). **Correction: S2b(iv)'s RESULTS/probe claim "the slab arena is PINNED" is FALSE for this model** — its drop verdict stands, but it concealed the pageable→pinned lever.
  - **The CPU half is the hard half**: torch `_weight_int4pack_mm_for_cpu` = **47.6 µs/expert** (6 P-cores, rel-L2 5.2e-3 vs GPU — NOT bit-identical) but needs a repack: `_convert_weight_to_int4pack_for_cpu` alone **308.9 µs** (> a whole GPU miss, ~297 µs) → on-demand repack DEAD; dual-format RAM (both layouts) halves the tier → RAM hit 96 % → 83–88 %, disk reads ×2.8–3.6 (+8.6…+13.7 ms/token, replayed through the real `ExpertRAMCache`) → DEAD; engine's eager dequant on CPU 1054–1896 µs → DEAD. **Only a new native AVX2 kernel over the compressed-tensors layout works; `vaayu_rust` has no SIMD code.**
  - **S3's increment on top of 30-P prices at −0.2 % … +7 %** (serial CPU work, microbench basis → fully overlapped, in-run basis), ~1–2 weeks, plus a perplexity-gated A/B (greedy ids cannot match; D6 shows one decode ppl has sd ≈ 0.6–0.7 pp).
  - **Verdict: S3 PARKED behind 30-P, not closed** → **re-gated 2026-09-18: CLOSED** (re-gate bullet below). Re-gate after 30-P ships: re-run S3-pre's split on the 30-P build; revive only if in-run `(construct_miss + gemm_miss)/T_prof ≥ 10 %` AND a native CPU kernel prototype on the checkpoint layout measures ≤ 60 µs/expert at 5 P-core threads. Otherwise close S3.
- [x] **30-P — DONE 2026-09-18 on branch `row30p-pinned-ram-tier` (built + measured on branch `row30p-pinned-ram-tier`; **merged to `main` 2026-09-18, fast-forward `c1799bc → 8475123`**). Pinned RAM tier (slab arena) for the non-EXL3 fused forward.**
  **Headline: mean 6.6329 → 7.4084 tok/s = +11.69 %** (ABBA, N=1, server-shaped knobs, Qwen3-Next; stop rule was ≥ +5 %, and this beats the standalone probe's own +10.62 %). min(new)/max(base) +8.18 %; pairwise A1→B1 +8.85 %, B2→A2 +14.53 % (both halves clear +5 % independently — no ABBA-position drift masking the effect). **Greedy ids identical across all 12 generations × 64 tokens.** VRAM hit 83 % on every arm (0 pp drift); VRAM peak Δ ≈ 0.03 GiB (alloc and reserved). RAM-tier comparability: base pageable 13.132/13.070 GB vs new pinned arena 12.96/13.21 GB — same tier size, no confound. W arm (first-touch check): base 3.83 → new 4.97 tok/s, **+29.9 %** — the probe's predicted −9 % first-touch penalty does not appear; the arena is pinned once at load instead of per-insert. Load time +2.3 s (one-time, `cudaHostAlloc` of the arena). **Server-sizing arena: 10380 × 1.8MB, 18.54 GB pinned**, post-arena avail 8.57 GB — the number a user's box actually pins.
  **§8: no knob (arena derived at load, as for the EXL3 forwards); N/A plumbing; reachable at N=1 — measured there; reaches the server since the merge (`8475123`): `vaayu/server/__main__.py` passes `device=config.DEVICE` (`"cuda:0"`) → `app.py` `from_pretrained(device=…)` → the arena gate fires; the server gate through `create_app` logged the pinned arena (`10380 x 1.8MB`, 18.54 GB) and passed 9/9. Benefit applies only when the server serves a fused non-EXL3 model (today: Qwen3-Next) — set `MODEL_PATH` to it; the EXL3 models already had the arena.**
  **Task 2 ruling on `cuda=`:** the LIVE guard against an arena on a CPU-compute non-EXL3 load is the pre-existing outer `if self.running_device.startswith("cuda") and self.is_moe:` at `vaayu_base.py:2571` (CPU MoE takes the `elif self.is_moe:` branch and never builds an arena) — `moe_supports_slab_arena(..., cuda=...)` is always `True` at its only call site today, so `cuda=` is correct but redundant defense-in-depth, load-bearing only if that outer guard is ever loosened.
  **User-rule evidence:** `docs/probes/p30p-gate-noexl3.log` — 3 passed, zero `exllama` string in the log, mutation check (Task 3) turns the CPU tests red when `import exllamav3` is injected.
  **Surprises:** the build beat the probe (+11.69 % vs +10.62 %) and its W arm is faster than base rather than ~9 % slower as the probe predicted; the per-miss saving's extra unexplained third (spec §6, ~56–64 µs/miss beyond the microbench projection) remains unexplained — not investigated by this build.
  Full RESULTS doc (self-contained, all numbers sourced from task reports + `docs/probes/p30p-*.log`):
  [`docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md`](docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md) (gitignored — this bullet is the durable copy).
  **Build plan (self-contained, code verified by dry-run — start here):** [`docs/superpowers/plans/2026-09-18-row30P-pinned-ram-tier-nonexl3-fused.md`](docs/superpowers/plans/2026-09-18-row30P-pinned-ram-tier-nonexl3-fused.md) (gitignored; this row is the durable summary). Probe `plans/probes/s3fill_pinned_ab.py` (one-line monkeypatch: `ExpertRAMCache.put` stores `v.pin_memory()` instead of `v.clone().detach()`; everything else unchanged) through `plans/probes/row26_decode.py`, `MODEL=qwen3next BALLAST_GB=25 NEW_TOK=64 ARMS=W,U1,U2`, 29G scope, ABBA: base U **6.6006/6.6247 · 6.7092/6.7038**, pinned U **7.3075/7.3085 · 7.4435/7.4068** → mean **6.6596 → 7.3666 (+10.62 %)**, slowest pinned beats fastest base by 8.92 %, **greedy ids identical across all 12 generations**, VRAM hit 83 % and peak 11.71 GiB unchanged, RSS +0.2 GB. Saved 14.41 ms/token = 180.8 µs/miss — ~56–64 µs/miss MORE than the microbench issue-cost projection (+6.4–6.9 %); "pageable copy is a hidden sync point" tested and REFUTED (`s3fill_pageable_sync.py`: 35.9 µs behind 20 ms of queued GPU work); the extra third is unexplained. Probe's discarded W arm is ~9 % slower (per-insert `cudaHostAlloc`); a real build pins the arena once at load. **Build scope (~1–2 days):** move `_make_fused_sparse_forward`'s inline `_load_one` (`moe_patch.py:1826`) onto `_dispatch_expert_misses` (zero-clone slabs, `SlabArena.record_read`/`acquire` DMA safety), let `moe_supports_slab_arena` accept the non-EXL3 fused layout, check the ~18.4 GB pinned arena fits the 32 GB box; keep the fused-EXL3 and module-list forwards unchanged (shared gate — run all three accumulator gates + server gate on Qwen3-Next). **Stop rule:** < +5 % through the same ABBA harness → do not ship. **§8:** no knob (arena is derived), nothing to plumb, reachable at N=1 (measured there), not reachable at scoping time (pre-build). Priority vs other rows: user's call.
  - [x] Task 0 — branch `row30p-pinned-ram-tier`, baseline worktree `/home/a/code/ai-ml/vaayu-30p-base` at the pre-30-P HEAD `c1799bc`, baseline suite **2243 passed / 15 skipped / 0 failed** (171 s)
  - [x] Task 1 (`d03b93d`, suite 2252/15/0, review clean) — `_make_fused_sparse_forward` loads through `_dispatch_expert_misses` (`vram_resident=set(_vram_hold)`), waits all reads before the id-ordered loop, arms `arena.record_read` + unpins after each miss upload, unpins the rest in `finally`; new `tests/test_fused_nonexl3_slab_arena.py` (9 tests: 3 RED before — `7 experts cloned into pageable RAM`, `7 uploads, 0 read events`, helper never called); update `test_only_the_fused_forward_passes_vram_resident` → both fused forwards pass `vram_resident=`
  - [x] Task 2 (`e2b9de6`, suite 2259/15/0, review clean; note: `cuda=` is always True at its call site today — the outer `running_device.startswith("cuda") and is_moe` guard at `vaayu_base.py:2571` is what keeps CPU MoE off the arena) — `moe_supports_slab_arena(styles, is_exl3, cuda=False)`: EXL3 answers unchanged; fused + non-EXL3 + CUDA → True; non-EXL3 module-list and CPU → False; call site passes `cuda=str(self.running_device).startswith("cuda")`; `tests/test_slab_arena_gate.py` truth table
  - [x] Task 3 (`5df2209`, suite 2261/16/0 = dry-run count; mutation `import exllamav3` → 2 failed; review clean) — `tests/test_non_exl3_never_touches_exllamav3.py`: CPU test with `sys.modules["exllamav3"] = None` (arena and clone paths) + on-device gate (`NOEXL3_CKPT=`, counts every exllamav3 call on every thread → must be 0, module never imported). **User rule 2026-09-18: a non-EXL3 model never executes an exllamav3 API.**
  - [x] Task 4 (2026-09-18, all green: noexl3 gate 3 passed — arena `9665 x 1.8MB` pinned 17.27 GB, avail 8.65 GB; accumulator gate 8/8/8 passed on Qwen3-Next / Qwen3-235B / MiniMax-M2.7, EXL3 arenas still pinned `1969 x 11.8MB` 23.37 GB / `2588 x 8.9MB` 23.04 GB; server gate 9 passed, **server-sizing arena `10380 x 1.8MB` pinned 18.54 GB, avail 8.57 GB**) — on device: arena line pinned (dry-run: `10192 x 1.8MB … pinned, 18.21GB`, post-arena avail 9.05 GB), accumulator gate on Qwen3-Next + Qwen3-235B + MiniMax (8 passed each), server gate on Qwen3-Next (9 passed)
  - [x] Task 5 (2026-09-18, **STOP RULE PASSED: 6.6329 → 7.4084 tok/s = +11.69 %**, slowest new beats fastest base by 8.18 %, pairwise +8.85 % / +14.53 %, greedy ids identical 12 × 64 tokens, VRAM hit 83 % all arms, peak Δ ≈ 0.03 GiB; W arm +29.9 % (no first-touch penalty); load +2.3 s) — ABBA `row26_decode.py` (base worktree vs build via `VAAYU_ROOT`), **stop rule ≥ +5 %**, greedy ids identical across all arms, VRAM hit/peak unchanged; record W-arm and load-time deltas
  - [x] Task 6 — RESULTS doc ([`docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md`](docs/2026-09-18-row30P-pinned-ram-tier-RESULTS.md)), §8 answers, `todo.md` closure; START HERE now points at the **S3 re-gate**
  Dry-run 2026-09-18 (throwaway worktree, deleted): plan code verbatim → full suite **2261 passed / 16 skipped / 0 failed**; on-device gate **3 passed** with the pinned arena allocated at server sizing. **What unblocks S3 afterwards:** 30-P shipped → re-run S3-pre's split on the 30-P build → revive S3 only if in-run `(construct_miss + gemm_miss)/T_prof ≥ 10 %` AND a native CPU kernel prototype ≤ 60 µs/expert.
- [x] **S3 re-gate DONE 2026-09-18 — S3 CLOSED. Gate 1 fails on both replicates: in-run `(construct_miss + gemm_miss)/T_prof` = 6.96 % / 6.99 % against ≥ 10 %.** Measurement only, no engine code; branch `row30-s3-regate` (todo.md only). Doc: [`docs/2026-09-18-row30-S3-regate-RESULTS.md`](docs/2026-09-18-row30-S3-regate-RESULTS.md) (gitignored — this bullet is the durable copy).
  - **How:** S3-pre's exact arms on `main` `108d67f` (30-P = `8475123`): `systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 env PYTHONPATH=… MODEL=qwen3next BALLAST_GB=25 NEW_TOK=64 ARMS=W,U1,P,U2,NOEXP TAG=-s3regate[2] ./venv/bin/python plans/probes/row26_decode.py`, one GPU job at a time, GPU idle before each; then `plans/probes/s3pre_gate.py -s3regate[2]`. Gate 1 isn't printed by `s3pre_gate.py`: it is `(construct miss s + gemm miss s)/T_prof` off its output, or `docs/probes/s3regate_table.py` (same JSONs, same arithmetic, reproduces S3-pre's printed numbers exactly). Both logs: `vaayu imported from /home/a/code/ai-ml/Vaayu/vaayu/__init__.py`, arena **pinned** (`7236` / `7240 x 1.8MB`, 12.93 / 12.94 GB). Denominator `T_prof`, exactly as S3-pre (profiled arm, `profiling_mode` syncs; U arms already run row 27's keep-awake, the engine default).
  - **Gate 1, all four runs:** S3-pre (pre-30-P) **7.06 / 7.19 %** → re-gate **6.96 / 6.99 %**. Generous basis (profiled miss terms over `T_unprof`, overstates the share): 8.35 / 8.57 → **8.59 / 8.67 %** — still < 10 %. 30-P cut the miss work 13.2 → 11.7 ms/token (−11 %) and `T_prof` 0.1848 → 0.1686 s (−8.8 %) together, so the ratio barely moved. **S3 never passed this gate, before or after 30-P.** Gate 2 (native kernel) NOT run, by the rule.
  - **Every term, S3-pre → re-gate (cross-day context, not an A/B; the verdict itself is same-build, same-day):** U tok/s 6.45 → 7.34 (`T_unprof` 0.1550 → 0.1362 s); `T_prof` 0.1848 → 0.1686; profiling overhead ×1.19 → ×1.24 (pinned H2D overlaps the host unprofiled, serialized profiled). **`h2d` 186.8 → 101.9 µs/miss-slot (−45 %, 17.4 GB/s vs 19.7 pinned line; the expected fall)**, 8.1 → 4.8 % of `T_prof`. `construct` miss 108.1 → 94.8 µs (unexplained, pure GPU work between syncs; hit unchanged 59.0 → 58.5). `gemm` hit/miss 48.7/58.6 → 45.5/52.0 µs. `disk_load` 20.3 → 14.2 ms/token (30-P's zero-clone loader) despite disk reads/token 14.8 → 16.6. `other` (unbracketed host loop) **unchanged 60.6 ms** → 32.8 → 35.9 % share. Slots 400.3+79.7 → 400.0+80.0; VRAM hit 83.39 → 83.33 % (VRAM cache budget 7.602 → 7.535 GB, ~38 experts fewer, cause not probed); RAM hit 96.5 → 96.1 % (pinned arena 12.92 GB < the 13.31 GB pageable budget). ADDRESSABLE `(h2d+construct_miss+gemm_miss)/T_prof` 15.1 / 15.2 → **11.8 / 11.8 %**; LITERAL 20.8 / 21.1 → 18.1 / 18.1 %. **Greedy ids identical across all four runs on every arm** (incl. NOEXP), pre- vs post-30-P.
  - **Row 36 off the same runs (context only):** strict `serial/T_unprof` (NOEXP 0.036 s, unchanged) 23.0 / 23.4 → **26.4 / 26.7 %**; wide `other/T_prof` 32.5 / 32.8 → 35.9 / 35.9 %. Strict still below 30 %; derived to fire once `T_unprof` ≤ ~0.120 s (~8.3 tok/s).
  - **What closing gives up (spec §7 method, re-gate numbers, derived):** per miss GPU 118.6 µs unprofiled vs CPU 55.6 µs serial / ~18 µs overlapped → **−0.2 % (microbench) … +3.7 % (serial) … +5.9 % (fully overlapped, upper bound)** of the 30-P token — for ~1–2 weeks, a new native AVX2 kernel (`vaayu_rust` has no SIMD), P-core contention with the pinned host loop, and a perplexity-gated A/B.
  - **§8:** no knob (nothing built); nothing plumbed; S3 WOULD be reachable at N=1 (80 VRAM misses per single-stream token) but it is unbuilt and now closed → 0 % to users, nothing banked. **Re-open only if** a future build/model measures gate 1 ≥ 10 % (e.g. a far lower VRAM hit rate), and then gate 2 must pass too.
- [x] **DONE 2026-09-19 — FIXED on branch `row30-ram-ceiling-refused-prefix`: +15.1 % decode tok/s on Qwen3-Next at default knobs (fix arms 7.837 vs shipped 6.809, ids identical); server tier 17.26 → 22.20 GB.** RESULTS: [`…-row30-ram-ceiling-refused-prefix-RESULTS.md`](docs/2026-09-19-row30-ram-ceiling-refused-prefix-RESULTS.md) (gitignored). ~~→ Sequence step 2 (set 2026-09-18): price this with [`…-hybrid-model-next-steps-BRIEF.md`](docs/2026-09-18-hybrid-model-next-steps-BRIEF.md) item 2 (CPU replay; close if < +2 %).** **Observation (2026-09-18, found by the 30-P dry-run; OPTIONAL, not in 30-P): the expert RAM ceiling reserves prefix-session KV on a model whose prefix cache is REFUSED.** Qwen3-Next log: `Expert cache ceiling 18.20 GB (… reserving 3 prefix session(s) of 1.61 GB each)` next to `cross-turn prefix cache DISABLED — this model has recurrent (linear-attention) layers` (RC7). ~4.8 GB of RAM held back for sessions that cannot exist (`derive_expert_ceiling`, `vaayu_base.py` ~:2630, is called with `prefix_enabled=bool(enable_prefix_cache)` before RC7's refusal is known). Value unmeasured: RAM hit is already ~96 % on this model, so a bigger tier may buy little. Price it with `plans/probes/s3fill_ram_replay.py` at +4.8 GB before touching code.
  - **PRICED 2026-09-19 — FIRES, build the fix.** Step 1: the ceiling BINDS the arena (server gate `p30p-gate-noexl3.log:200-204`: ceiling 17.26 GB = arena 17.26 GB of slabs), so the 3 × 1.61 GB reservation comes straight out of the RAM tier. Step 2: trace replay (`plans/probes/row30_ram_ceiling_replay.py`) saves 4.4–5.7 reads/token at server sizing, but the traces are 63/127 tokens and their footprint (10.7/12.1 GB) fits either budget → lower bound only. **On-device ABBA, zero code** (`row26_decode.py MODEL=qwen3next BALLAST_GB=26.5 NEW_TOK=128 ARMS=W,U1,U2`, U = `PREFIX_SESSIONS=0` = exactly the fix's sizing, S = shipped 3; logs `docs/probes/row30-ramceil-{U1,S1,S2,U2}.log`): **U 7.98 vs S 6.81 tok/s = +17.2 %** (worst U 7.87 > best S 7.07), reads/token 9.0 vs 27.4, RAM hit 98.1 vs 94.1 %, greedy ids identical in all 8 timed arms. RAM tier 19.5–19.6 vs 14.3–14.9 GB (probe sizing sits ~2.4 GB below the server's). Measured ~1.17 ms wall per saved read. Projected at server sizing (17.26 → 22.09 GB): **~+9–10 %** (replay's saved-reads ratio between the two sizings × the measured per-read cost). Post-arena avail in U arms 3.73–4.11 GB, no OOM — the same state any MoE loaded with `ENABLE_PREFIX_CACHE=False` is in today.
  - **FIX 2026-09-19:** RC7's refusal block moved, unchanged, into `VaayuBaseModel._refuse_unsupported_prefix_cache()`, called BEFORE cache sizing (ahead of `_detect_moe()`, still before the dense/MoE split → model-agnostic); store build stays after the arena. Ceiling log line now prints the sessions ACTUALLY reserved (it printed `prefix_sessions` even when reserving 0 — it lied on every `enable_prefix_cache=False` load, e.g. the accumulator gate). Tests RED-first in `tests/test_hybrid_attention_cache.py` (refusal behaviour ×3 + source-order guard). **Verified:** suite 2292 passed / 16 skipped; fix arms F1/F2 (default `PREFIX_SESSIONS=3`) 7.837 tok/s = **+15.09 % vs S**, −1.78 % vs U (inside spread), reads 9.23/token, greedy ids identical across all 12 timed arms; server gate Qwen3-Next 9 passed, tier 17.26 → **22.20 GB**, post-arena avail 3.76 GB; server gate Qwen3-235B (prefix ON) 9 passed, still reserves 3 sessions (unchanged); accumulator Qwen3-Next / Qwen3-235B / MiniMax 8/8/8; Llama-3.2-1B prefix gate 1 passed (prefix cache still ENABLED). **§8:** no knob (existing `ENABLE_PREFIX_CACHE`/`PREFIX_SESSIONS` now mean what they say); plumbed (`app.py:406,420`); reachable at N=1 (sizing at load). User docs: `documentation/TUNING.md` §3b, `vaayu/server/README.md` `PREFIX_SESSIONS` row. Server-sizing speed-up is PROJECTED ~+9–10 %, not measured (the probe cannot reach the server's available RAM). Caveat: pinned tier → post-arena avail ~3.4–4.6 GB on refused models (= the 3.5 GiB hardware reserve); no OOM anywhere. Row 42 interacts: a hybrid prefix cache would bring the reservation back automatically.
- → **Moved 2026-09-18 to its own row: row 40 below.** the `[bug]` found by the 30-P final review (the expert slab arena is never capped at the model's own expert bytes). Text moved verbatim.

## S2 STARTED 2026-09-13 — FOUR ROOT CAUSES MEASURED BEFORE ANY CODE (probes, no full load)

Probes (gitignored, re-derive in ~1 min each): `plans/probes/row30_s2_skeleton.py` (meta-device skeleton +
quantizer + key diff), `row30_s2_unpack.py` (one real expert's tensors), `row30_s2_sizes.py` (byte geometry
off the shard headers). `pip install compressed-tensors` **done** (0.18.0) — must land in `requirements.txt`
in the commit that makes the model load.

**Geometry, measured off the real shards (all derived, no literals in code):** expert **1,769,520 B =
1.6875 MiB** exactly (uniform across all 25,088 expert/layer pairs), routed **0.849 GB/token** (48 × 10 ×
1.6875 MiB) — 5.2× lighter than Qwen3-235B-EXL3's 4.43. Per-layer NON-expert weights **62.9 MB** (12 full-attn
layers) / **75.8 MB** (36 linear-attn layers), **3.485 GB over 48 layers**, plus embed 0.622 + lm_head 0.622 =
**4.73 GB that must sit in VRAM as non-expert pins** (fits 15.47 GiB usable with room for the expert VRAM
cache). Total 48.2 GB excluding the **0.986 GB unused `mtp.*` draft module** (49 expert-carrying "layers" on
disk = 48 + mtp; `Qwen3NextForCausalLM` never builds it). One expert = 3 projs × {`weight_packed` I32,
`weight_scale` **BF16** (not F16), `weight_shape` I64}; gate/up `[512, 2048]`, down `[2048, 512]`, g32 sym.

**RC1 — the packed int4 experts have no dequantiser anywhere in Vaayu.** `_classify_expert_tensors`
(`moe_patch.py:424`) returns the FIRST tensor whose key contains the role segment, so for this checkpoint it
hands `_make_fused_sparse_forward` whichever of `weight_packed`/`weight_scale`/`weight_shape` iterates first,
and `F.linear` gets an int32 pack. **This, not the HF binder, is S2's real first task.**

**RC2 — the binder gap is real but must be BYPASSED, not fixed** (this CORRECTS the row's earlier
"that binding is S2's first task"). Measured: `CompressedTensorsHfQuantizer` resolves and
`preprocess_model` runs clean — and does **nothing**: `Applying quantization config: 0it`,
`Compressing model: 0it`, 2 params added (both rotary buffers). Reason: every non-expert Linear is in the
checkpoint's own `ignore` list and the routed experts are **not `nn.Linear`** — 5.5.4 builds
`Qwen3NextExperts` with fused 3D `nn.Parameter`s (`gate_up_proj [512, 1024, 2048]`,
`down_proj [512, 2048, 512]`, bf16 = **3 GiB/layer if ever materialised**). Key diff: **225,809 disk keys with
no model target, 98 model keys with no disk tensor** (the 49 × 2 fused params). Vaayu streams experts from its
own per-layer shards and never reads those params, so the fix is to keep them on meta forever — binding them
would cost 144 GiB of bf16. **C1 PASS** (skeleton builds, sdpa), **C3 fused / non-EXL3** →
`_make_fused_sparse_forward`, the branch nothing has ever run, as predicted.

**RC3 — the shared expert is silently DROPPED.** `Qwen3NextSparseMoeBlock.forward` returns
`routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)`; `patch_moe_blocks` (`moe_patch.py:842`) replaces
the whole **block** forward and every one of the three Vaayu forwards returns the routed part only. Neither
on-device model has a shared expert (Qwen3-235B and MiniMax-M2.7 both lack one), so no gate covers this. It is
**~63 MB/layer of weights that would just not be applied** — output would be plausible-looking and wrong.
Affects every shared-expert MoE family (Qwen2/Qwen3-Next, DeepSeek, GLM).

**RC4 — the KV cache is built without `config=`, and this model is HYBRID attention.** `layer_types` is
36 × `linear_attention` + 12 × `full_attention` (`full_attention_interval: 4`). `DynamicCache()` /
`FP8DynamicCache(quantize=False)` (`vaayu_base.py:3400,3407`) pass no config, so 5.5.4 lazily replicates
`DynamicLayer` for all 48; the linear-attention layers need `LinearAttentionLayer`
(`conv_states`/`recurrent_states`, read at `modeling_qwen3_next.py:609`). Second half of the same problem:
Vaayu's VRAM/KV planner charges `num_hidden_layers` KV layers, but only **12 of 48** hold KV here, so
`max_seq_len` is derived ~4× too small. Risk #4 of this row, now confirmed rather than suspected.

**Order this implies (unchanged stop rule: garbage output → fix numerics before ANY perf work):**
RC1 (dequant, unit-testable against `compressed_tensors`' own reference, no GPU) → RC3 (shared expert,
detected not hardcoded) → RC4 (cache + planner) → persist 46 GB → stream → `row26_decode.py`.

### S2 progress 2026-09-13 — RC1, RC3 and RC4(a)(b) FIXED and green; RC4(c) left on purpose

Full results: [`docs/2026-09-13-row30-S2-load-path-RESULTS.md`](docs/2026-09-13-row30-S2-load-path-RESULTS.md).
Suite **2025 passed, 15 skipped** after both fixes. **No perf claim — nothing has run end to end yet.**

- `[x]` **RC1 fixed.** `vaayu/moe_patch.py`: `_group_packed_tensors_by_weight`, `unpack_int32_to_int8`,
  `dequantize_packed_weight`, `_expert_dense_weights` (one call site in `_make_fused_sparse_forward`).
  Bit width derived from packed-vs-original element count, group size from the scale's last dim — no
  config read, no literal; asymmetric/act-order raises `NotImplementedError` by name. **Dequant runs on
  the COMPUTE DEVICE at use time, deliberately:** dequantizing at persist time (what this row's IMPL doc
  floated as path B) would store bf16 and multiply the disk read AND the H2D by 4×, deleting the reason
  to run a 4-bit model. New gate `tests/test_packed_expert_dequant.py` (13) holds the unpack to
  `compressed_tensors`' own reference **bit-for-bit** across int4-g32 / int4-g128 / int8-g128, so the
  decode path never imports that package; verified additionally on a REAL expert (exact match on all
  three projections, CPU == CUDA). `compressed-tensors>=0.18.0` added to `requirements.txt`.
  **Measured dequant cost, and it is LAUNCH-bound, not bandwidth-bound** (trivial kernel = 2.7 µs on this
  GPU): as landed **117.5 µs/expert = 56 ms/token**; no-int8-hop 93.1; one call per expert 78.5;
  **`torch.compile` of that 17.9 µs = 8.6 ms/token**. Shipped the simplest. **Do not optimise before
  phase 3 says dequant is visible** against the 0.849 GB/token read — the three alternatives are measured
  and banked here.
- `[x]` **RC3 fixed.** `_shared_expert_output` in `vaayu/moe_patch.py`, wired into **all three** forwards
  (a feature wired into one silently no-ops in the others — CLAUDE.md §0). Found by name beside the
  router: `_gate`-suffixed sibling ⇒ sigmoid-gated (Qwen2/Qwen3-Next), absent ⇒ added ungated (DeepSeek).
  New gate `tests/test_shared_expert.py` (4), **including the no-shared-expert case as the regression
  guard for Qwen3-235B / MiniMax-M2.7**.
- `[x]` **RC4 (a)+(b) FIXED — the KV cache was wrong for hybrid attention. Two distinct failures, both
  reproduced first** (`plans/probes/row30_s2_cache_hybrid.py`); this was what blocked the first load:
  **(a)** `_create_kv_cache` (`vaayu_base.py:3400,3407`) passes no `config=`, so 5.5.4 replicates
  `DynamicLayer` for all 48 layers and the gated-delta-net layers hit
  `AttributeError: 'DynamicLayer' object has no attribute 'conv_states'`. With `config=` the split is
  correct (36 `LinearAttentionLayer` + 12 `DynamicLayer`).
  **(b)** the DEFAULT ring buffer cannot address non-contiguous layer indices — `FP8DynamicCache`'s
  per-layer buffers are positional appends and the first layer to `update()` on this model is layer 3:
  `IndexError: list index out of range` at `vaayu_base.py:1365`. Config alone does not fix this; the
  storage must become index-safe.
  **Fix (a):** `FP8DynamicCache` / `TurboQuantKVCache` take and forward `config=`, and `_create_kv_cache`
  passes it through the new `cache_config_if_non_uniform()`. **That gate matters and cost a second pass to
  find:** passing the config unconditionally MATERIALIZES `cache.layers`, flipping `len(cache)` from 0 to
  `num_hidden_layers` and `is_initialized` False→True before any token is stored — observable state
  changing for the four working models in exchange for nothing, and **2037 tests passed with that version**.
  Config now goes over only when the layer classes actually differ (hybrid, or a sliding window). **Fix (b):** both
  `_ensure_capacity*` address their per-layer lists BY `layer_idx`, padding with **zero-capacity
  placeholder tensors** — zero-capacity, not `None`, so every existing slice-based consumer (`crop`,
  `crop_front`, `copy_prefix`, `resident_bytes`) keeps working with no per-site guard. New gate
  `tests/test_hybrid_attention_cache.py` (12), **including the uniform-model case for all three cache
  modes** — checked, not assumed: none of the four checkpoints on disk has `layer_types` or a sliding
  window, so all four are byte-identical to before.
- `[x]` **RC4 (c) — CLOSED 2026-09-13, see its own entry below** (`max_seq_len` 16384 -> 65536,
  every budget unchanged, 0 % tok/s). *Text as it stood when this was deferred:* Only **12 of 48**
  layers hold KV, so `kv_bytes_per_token`'s
  `config.num_hidden_layers` charge (`vaayu_base.py:2189`, `batch.py:45`) over-reserves ~4× here and the
  derived `max_seq_len` comes out that much smaller. It errs SAFE, it does not block the load, and it sits
  in the middle of the most load-bearing tested code here — **fix it after the first successful load, with
  a measured VRAM peak, not before.** Separately, the prefix cache needs a DECISION not a port: conv /
  recurrent state is not sliceable by token position, so `copy_prefix` on a hybrid model must be refused
  loudly or redesigned.
- `[x]` **RC5 FIXED — the splitter silently DROPPED tensors, and the completeness check called it good.**
  Found by running the persist, not by reading code. `split_and_save_layers` (`vaayu/utils.py`) loaded, per
  layer, only `max(shard number)` and only when it differed from the shard already held — which assumes the
  index is **layer-major and monotonic**. This checkpoint is neither: one layer spans up to 4 NON-ADJACENT
  shards and the numbers do not rise with the layer index (observed load order `1,4,7,3,9,6,2,8,5`). The
  layer was written anyway, minus the tensors from shards never loaded, **with its `.done` marker stamped**,
  so `found_layers` reported the split complete. **Measured: 24,356 of 24,576 experts, layers 14 and 24
  short, 11 distinct expert byte sizes (partial experts), 41.57 GB instead of 48.24.** Fix: group the index
  by layer once, then read each layer's keys out of whichever files hold them — safetensors **one tensor at
  a time via `safe_open`**, so the resident set is ONE LAYER instead of a growing pile of whole shards (the
  `single_file_loaded` preload is gone, orphaned by the change). `.bin` keeps whole-file loads, same
  memory profile as before, same bug fixed. Cost **+8 % wall** (64.1 → 69.2 s) for correctness. New gate
  `tests/test_split_shard_spanning.py` (3) — 2 RED on the old code, and the third asserts `mtp.layers.*` is
  NOT swept into `model.layers.*`. **Affects every model Vaayu splits**, not just this one.
- `[x]` **Persist DONE and asserted** (`plans/probes/row30_s2_persist.py`, gitignored): 51 shards,
  **48.24 GB**, index finds **24,576 = 48 × 512** experts over 48 layers, **0 `mtp` groups** (shard 10 is
  100 % `mtp` and is never opened), and **one single distinct expert size, 1,769,520 B = 1.6875 MiB**,
  matching the header geometry above exactly. Suite **2041 passed, 15 skipped** (baseline 2038 + the 3 new).
- `[x]` **FIRST SUCCESSFUL RUN — the first non-EXL3 MoE ever executed on this box.** Loads in **4.1 s**,
  greedy output coherent (`Here's a Python function that reverses a string:\n\n```python\ndef
  reverse_string(s):` …), geometry all DERIVED and exact: expert 1,769,520 B, 48 MoE blocks, top_k 10,
  512 experts, working set 0.849 GB, `expert_style="fused"` / `is_exl3=None` → **`_make_fused_sparse_forward`,
  the branch nothing had ever run** (CLAUDE.md §0's "on-device today: none" row), flash_attention_2,
  derived `max_seq_len` 16384, peak VRAM 4.48 GiB. RC1's dequant and RC3's shared expert are exercised for
  real. **No perf claim yet — this is a probe, not `row26_decode.py`.**
- `[x]` **RC6 FIXED — `get_seq_length()` reported 0 on a hybrid model, so EVERY decode step re-ran the whole
  sequence.** Found by running it. `FP8DynamicCache` (what `auto` resolves to) overrides
  `Cache.get_seq_length` and returned `_seq_len[0]`; layer 0 here is linear-attention and never calls
  `update()`, so it stays 0 for the whole run. transformers' generation loop then never trimmed `input_ids`.
  **A/B that settled it: stock `DynamicCache(config=cfg)` → 1, Vaayu's ring buffer → 0** on the same
  update. Second-order damage: `seq_len > 1` every step ⇒ `compute_prefill_pass` ⇒ `_moe_prefill_pass` ⇒
  the fused forward SUPPRESSES `expert_cache.put()`, so the RAM cache sat at **0.00 GB / 0 % hit / 0 evict
  forever**. Measured before: **7.49 GB read per token** against a 0.849 GB working set (**8.8×**),
  **4205 expert misses/token** against 480 routed, **0.201 tok/s**. After: **480 exactly** (48 × 10),
  gb_tok 0.05–0.32, hit rate to 95 %, **1.906 tok/s** — **9.5×**, with the first 16 greedy ids IDENTICAL
  to the broken run. Fix: `_kv_layer_indices()` / `kv_layer_lengths()` derived from which buffers have
  capacity (no config read, no layer-type names), and `get_seq_length()` falls through to the first
  KV-bearing layer for the default index — the same rule `Cache.get_seq_length` already uses.
  `check_batch_kv_invariant`'s `min(lens) != max(lens)` had the same flaw (a permanent 0 on every
  linear-attn layer is an absence, not a disagreement) and is fixed with it. 8 new tests in
  `tests/test_hybrid_attention_cache.py` incl. the stock-vs-Vaayu A/B, the uniform-model regression guard,
  and the invariant still catching a REAL disagreement. Suite **2049 passed, 15 skipped**.
- **Perf note, not a bug:** `moe_supports_slab_arena` requires `is_exl3`, so this model gets **NO pinned
  slab arena** (`arena_slabs: null`) and the non-EXL3 fused forward clones into PAGEABLE RAM — pageable H2D
  is 15.3 GB/s vs 19.7 pinned (row 26). The gate is CORRECT for the code as written: that forward reaches
  `expert_cache.put()`, not `_dispatch_expert_misses`. Worth ~? — price it in phase 3, do not assume.
- `[x]` **RC7 FIXED (refused) — the prefix cache cannot lend a hybrid model its state.** The SECOND
  generation in one process died: `[Vaayu-Prefix] HIT match_len=134/135` → `copy_prefix` rebuilds the cache
  as `type(self)(quantize=...)` **with no config** → `layers == []` →
  `IndexError: list index out of range` at `cache_utils.py:959` (`update_conv_state`). Passing the config
  through only moves it: `copy_prefix` copies the K/V ring buffers and NOTHING else, and conv/recurrent
  state is not sliceable by token position — a fixed-size summary has no "first N tokens" — so the hit
  would be silently WRONG instead of loudly broken. **This is the decision RC4 (c) said was owed; it is
  taken: refuse.** `cache_holds_recurrent_state` (detected off the layer objects' `update_conv_state` /
  `conv_states`, not a config field, not a name) makes `cache_supports_prefix_store` False → the existing
  refuse-at-load path prints one WARNING and turns the prefix cache off for that model; `copy_prefix`
  raises the same explanation so no other caller reaches the IndexError. **A bug inside the fix, caught by
  reading the run:** a hybrid cache fails BOTH checks and the surface check ran first, printing
  *"FP8DynamicCache … does not implement to_cpu/to_device/resident_bytes"* — true of nothing. Both reasons
  now live in one pure `prefix_cache_refusal_reason(cache, kv_cache_mode)`, recurrent first, with a test
  asserting the hybrid reason says "recurrent" and never mentions `to_cpu`.
- `[x]` **PHASE 3 MEASURED — `row26_decode.py`, `MODEL=qwen3next`, server knobs, 29G scope.
  `T_unprof = 0.2444 / 0.2476 s → 4.0648 tok/s`, replicate spread 1.30 %, greedy ids identical across
  W/U1/P/U2.** Cold W 2.897, profiled P 2.942 (overhead ×1.382), NOEXP serial floor **0.0348 s/token**
  (28.71 tok/s). Derived bpw 4.500, 480 routed slots/token, 0 pinned MoE layers.
  **Qwen3-235B-EXL3 measures 0.85 raw / 1.1363 through the server (row 27) → this is 4.8× / 3.6×.**
  NOT row 29's projected 18.1 — re-derive from this, never quote the projection.
- `[x]` **THE REGIME CHANGED — this is what S2 existed to do, and it settles the T3/T4 branch.**
  `tiers U1: V=0.0 R=463.1 M=16.9 gb_tok=0.030 stall=0.0000s`; disk floor at the 7.1 GB/s plateau =
  **0.0042 s = 1.7 % of the token**, against **70–83 %** on Qwen3-235B / MiniMax-M2.7. **96.5 % of routed
  slots come from the RAM cache.** → **T3 (rows 32 · 34 · 35) buys a factor on a cost that no longer exists
  on this model; T4 (rows 36 · 37) is the live branch.** Also demotes the 24,576-slab / `_NVME_READ_CONCURRENCY`
  watch **with a number**: it can move at most 1.7 % here.
- `[x]` **THE PHASE-4 GATE PASSES — `h2d + gemm = 40.7 %` of `T_prof`, 65.8 % with `construct`, against a
  15 % stop rule.** Phases s/token (share of T_p 0.3400): **h2d 0.1155 (34.0 %)** · **construct — the int4
  dequant — 0.0854 (25.1 %)** · other 0.0631 (18.6 %) · disk_load 0.0239 (7.0 %) · scatter 0.0232 (6.8 %) ·
  **gemm 0.0228 (6.7 %)** · route 0.0047 · mask_sync 0.0013 · attach/detach 0.
  **Two honesty notes.** (1) Profiling costs ×1.382 and `h2d` carries the most `synchronize()` points, so
  its share is an UPPER BOUND; scaled to `T_unprof`: h2d ≈ 0.0875 (35.8 %), construct ≈ 0.0647 (26.5 %),
  gemm ≈ 0.0173 (7.1 %). (2) **`h2d` is NOT bandwidth** — the token moves 0.030 GB, which is 2 ms at
  pageable 15.3 GB/s, not 115 — **THIS WAS WRONG, see the correction below.** The gate is satisfied and S3
  is NOT closed.
- `[x]` **CORRECTED 2026-09-13 by direct measurement (`plans/probes/row30_s2b_h2d.py`), and the correction
  reverses the conclusion.** The earlier text here read *"`h2d` is NOT bandwidth — the token moves 0.030 GB,
  which is 2 ms at pageable 15.3 GB/s, not 115. It is 4320 tiny `.to(device)` calls per token:
  launch-bound … S3 was priced on H2D being a PCIe cost a CPU-resident expert avoids. It is not."*
  **Both halves are wrong.** `gb_tok=0.030` is what is read from DISK; what crosses PCIe is the ROUTED
  WORKING SET, **0.849 GB/token** (`working_set_bytes`, 480 × 1.6875 MiB) — a cache hit still has to copy.
  And the count is **2880**, not 4320: `weight_shape` never crosses (6 of 9 tensors per expert).
  Measured at the real geometry, 200 reps: **as shipped (pageable, 6 copies) 121.9 µs/expert = 58.5 ms/token
  at 14.52 GB/s — already 95 % of this box's pageable H2D line rate (15.3 GB/s, row 26).**
  Coalescing to one prebuilt byte blob: 108.9 µs (52.3 ms/token, 16.25 GB/s). Pinned, as shipped: 95.7
  (45.9 ms, 18.50). Pinned + blob: 92.7 (44.5 ms, 19.09). Building the blob per call is a NET LOSS (123.9
  pageable / 115.9 pinned). **So `h2d` is BANDWIDTH-bound, S3's PCIe premise is RESTORED, and lever (i) is
  REFUTED: coalescing is worth 2.5 % of the token, pinning 5.2 %, both together 5.7 % — not the ~34 % the
  profile share implied.** The only lever that removes these bytes rather than moving them faster is the
  VRAM expert cache, (iii).
- `[x]` **S3-pre DONE 2026-09-13 — BOTH GATES MEASURED, NEITHER FIRES CLEANLY, AND THE GATE HAD
  BEEN READ THE WRONG WAY.** Two replicates, `row26_decode.py MODEL=qwen3next ARMS=W,U1,P,U2,NOEXP`,
  29G scope, server knobs, greedy ids identical across W/U1/P/U2. `U1 6.3687 / 6.4678 tok/s`,
  VRAM cache **83.4 %** hit (25218/5022), RAM 96.1–96.5 %, `gb_tok` 0.026–0.029, stall 0.0000 s,
  VRAM peak 11.70 GiB / 12.32 of 15.47 used.
  **THE READING CORRECTION:** `h2d` is miss-only, but `construct` and `gemm` are recorded on BOTH
  branches — a packed-blob hit still owes the dequant, and the GEMM runs on the GPU either way. The
  literal `(h2d+gemm)/T` therefore counts work S3 CANNOT take: a VRAM-resident expert is already
  where the GPU wants it, and a CPU kernel would have to copy it back. The split is now MEASURED
  (`construct_hit`/`gemm_hit`/`slots_hit`/`slots_miss` in `_moe_profile`, gated on `profiling_mode`)
  and the proportionality assumption would have been wrong: **a miss constructs at 108 µs/slot
  against a hit's 59**; `h2d` is 187 µs/miss-slot; 480 slots/token = 400.3 hit + 79.7 miss.
  **GATE (a) S3 — literal 20.8 / 21.1 %, ADDRESSABLE `(h2d+construct_miss+gemm_miss)/T` =
  15.1 / 15.2 %, i.e. ON the 15 % rule** (the row projected ~14.5 %); realistic gain share × 0.6 =
  **9.0–9.1 %** before CPU/IO contention.
  **THE MECHANISM, NOT THE NUMBER, IS THE REASON TO STOP: the VRAM cache is filled by exactly ONE
  code path and it is the path S3 removes.** `_make_fused_sparse_forward` has a single `_vram.put()`
  site, inside the miss branch, after the H2D is paid; there is no prefill stager on this forward.
  So the 83.4 % hit rate exists BECAUSE misses pay the H2D and get cached — divert those 79.7
  slots/token to a CPU kernel and nothing refills the cache. S3 and S2b (iii) are mutually exclusive
  per slot, so S3's 15.1 % is not a stable share to spend. **RECOMMENDED: CLOSE the CPU+GPU hybrid**
  (~9 % for ~1 week, against S3's own "gain < half the projection → close" rule, in direct conflict
  with the shipped win). Recommendation, not a unilateral close — the number does clear 15 %.
  **GATE (b) row 36 — strict `serial`=NOEXP arm is 23.0 / 23.4 %, BELOW its 30 % rule; the wide
  reading (all host time no expert placement removes, `other`/T) is 32.5 / 32.8 %, above it.** NOEXP
  is 0.0363 s/token and did not move between replicates or across builds (0.0348/0.0343 pre-(iii)).
  The 9.4 pp between the readings is the MoE loop's own unbracketed host work (`torch.where` per
  expert over a CPU-side mask, dict lookups, the ~480-slot Python loop) — precisely what "fused
  per-layer dispatch" targets. Row 29's prediction half-landed: `serial` IS the largest single block
  (32.8 %) but has not reached the rule as literally written.
  **NEXT WORK = T2 / row 33 (`top_k` truncation), not S3 and not yet row 36.** This model routes
  **top-10 of 512** (more headroom than Qwen3-235B's 8) and per-slot phases are **63.8 % of the
  token**. Derived from these measurements, hit rate held constant (conservative — fewer slots means
  the VRAM cache covers more of the working set): **k=8 → ~7.40 tok/s (+14.7 %), k=6 → ~8.66
  (+34.2 %)**. ~1 day + a ppl sweep, lossy authorised 2026-09-12.
  **§8:** diagnostic only — no knob (`profiling_mode` is a model attribute the server never sets),
  nothing to plumb, **0 % to users**, gating asserted by test. Suite **2090 passed, 15 skipped**.
  Detail: [`…-row30-S3-pre-phase-gate-RESULTS.md`](docs/2026-09-13-row30-S3-pre-phase-gate-RESULTS.md).
- `[x]` **Phase profile added to the non-EXL3 fused forward** — it was the only one of the three recording
  nothing into `_moe_profile`, so the gate would have read ZERO and closed the hybrid idea on an artefact.
  Same keys and bracketing as the other two; `h2d`/`construct` split inside `_expert_dense_weights(timing=)`
  because the copy and the dequant run back to back there. Fully gated on `profiling_mode`.
- `[x]` **S2b (ii) LANDED AND MEASURED — one fused dequant per expert instead of three: `4.0648 →
  4.3459 tok/s`, +6.92 %, greedy ids IDENTICAL, output bit-identical.** Replicate spread 0.55 % (baseline
  1.30 %), so the delta is ~12× the spread. `construct` **0.0854 → 0.0444 s/token** (25.1 % → 16.3 % of
  `T_prof`), a 48 % cut. Server gate green on the checkpoint afterwards (9 passed). Mechanism: the cost is
  kernel LAUNCHES, not arithmetic — one real expert, 300 reps: three per-linear calls **114.3 µs**, one
  fused call **75.7**, the fused call under `torch.compile` **17.6** (a trivial kernel is 2.7). `torch.compile`
  is applied only on CUDA (that is where launch overhead exists) and falls back to eager on a dynamo failure.
  **Bit-exactness is the whole point and is asserted, not assumed:** these weights feed `F.linear` directly,
  so `tests/test_packed_expert_dequant.py` holds the fused path to `torch.equal` against the per-role path on
  int4-g32 / int4-g128 / int8-g128, plus a fallback when roles disagree on bit width or group size, the
  `NotImplementedError` for asymmetric/act-order restated on the fused path, a guard that the fused path
  ACTUALLY fuses (0 per-role calls — the silent-no-op failure mode), and the compile-fallback branch (which
  shipped with a `NameError` in it until that test was written). Suite **2064 passed, 15 skipped**.
  **Why +6.9 % and not the microbench's 6.5×:** production does two on-device `torch.cat`s per expert that
  the probe's prebuilt flat tensors skipped. Note also that `h2d` fell 0.1155 → 0.0850 in the PROFILED arm —
  that is a MEASUREMENT artefact (one `synchronize()` bracket per expert instead of three; overhead ×1.382 →
  ×1.181), not a gain.
- `[x]` **S2b (iii) — DONE 2026-09-13: THE VRAM EXPERT CACHE IS WIRED INTO THE NON-EXL3 FUSED FORWARD.
  `4.1584 → 6.4702 tok/s, +55.6 %` (same-day baseline arm, ids identical, VRAM cache 83 % hit, peak
  11.66 GiB). Everything below is the reasoning that produced it; the build's own record is the numbered
  plan further down.** Measured before the build: `V=0.0`, `vram=0hit/0miss/0evict`,
  **7.79 GB of VRAM expert-cache budget unused**, while peak VRAM is only **4.48 GiB of 15.47**. Verified in
  source, not inferred: `_make_fused_sparse_forward` contains ZERO references to `_expert_vram_cache`,
  `_dispatch_expert_misses`, the slab arena or the prefill stager — the other two forwards reach the VRAM
  cache through that shared helper, this one never did. **Prize: a VRAM-resident expert skips BOTH `h2d`
  (0.0850 s/token) AND `construct` (0.0444) — together ~0.103 s of a ~0.230 s token, ~45 %.** Nothing else
  left touches more than ~12 %.
  **THE DESIGN QUESTION, and it must be MEASURED, not assumed: store PACKED or DEQUANTIZED?**
  packed 1.6875 MiB/expert → ~4600 experts in 7.79 GB = **19 %** of the 24,576 population, skips `h2d` only;
  dequantized bf16 ~6.29 MB/expert → ~1240 = **5 %**, skips `h2d` AND `construct`. Packed holds **3.7×**
  more; dequantized saves ~1.75× more per hit. The answer turns entirely on routing skew — and the steepness
  is real (the RAM cache reaches **96.5 % hit at ~32 % coverage**).
  **DO NOT price this with `vaayu/expert_sim.py`.** Its anchor gap is a STILL-OPEN item in this file —
  31.9 % simulated against 43.9 % live at a **9.12 GB** budget — and it is confined to the SMALL-BUDGET
  regime, which is exactly where 7.79 GB sits. Its curve SHAPE is usable; this decision needs ABSOLUTES at
  two very different capacities, which is the half it gets wrong. **Instead: record the routed-expert
  sequence from one real decode run and replay it through the ACTUAL `ExpertVRAMCache` class at both element
  sizes** — row 13's lesson (2026-09-11), that a brief priced against a ratio-only MODEL of a gate got 5/6
  where the real class got 6/6. Replay evidence through HEAD code before building.
  **Watch:** the 7.79 GB is *reserved* but nothing is allocated today, so actually using it will move peak
  VRAM for the first time — check it against the 3.50 GiB reserve floor and the KV plan, and re-run the
  server gate.
  - `[x]` **THE LAYOUT QUESTION IS ANSWERED BY REPLAY, AND THE ANSWER IS NEITHER OF THE TWO OPTIONS ABOVE:
    store the PACKED BLOB — `flat_packed`/`flat_scale` already concatenated.** Two ordered routed-expert
    traces (code prompt, 63 tok; prose prompt, 127 tok — greedy ids identical to their warm-up arms) replayed
    through the REAL `ExpertVRAMCache`, scored on a warm window, at the shipped budget (7.52 / 7.55 GB):
    **packed +28.1 % / +32.7 %** (80.9 / 89.3 % hit, 4248 resident = 17.3 % coverage) · **dequant bf16
    +33.2 % / +37.6 %** (57.4 / 62.2 % hit, 1194 = 4.9 %) · **packed-blob +47.4 % / +56.1 %**
    (4.3459 → **6.41 / 6.78 tok/s**). Packed-blob is packed's bytes, so packed's coverage and hit rate, but a
    hit also skips the two on-device `torch.cat`s `dequantize_expert_weights` does per call
    (`moe_patch.py:724-725`) — per hit 182.6 µs vs packed's 121.9 and dequant's 200.2. **It also absorbs most
    of S2b (iv)**, whose whole prerequisite was one contiguous blob per expert. Two-tier (cold packed,
    promote to dequant) peaks at +36.0 / +40.2 % and is REJECTED: beaten outright, and its optimum sits one
    step from a 0 % cliff. **Sensitivity, stated:** the packed-blob figure rests on `construct_flat` = 17.6 µs
    (S2b (ii)'s microbench on prebuilt flat tensors, a different measurement context); it beats dequant for
    any value **< 58 µs** and never loses to plain packed. **The build must measure it directly.**
    [`…-row30-S2b-iii-vram-layout-SPIKE.md`](docs/2026-09-13-row30-S2b-iii-vram-layout-SPIKE.md)
  - `[x]` **THE BUILD — DONE 2026-09-13, +55.6 % (4.1584 → 6.4702 tok/s), shipped. Plan as written 2026-09-13. Everything needed is in this row; the SPIKE
    doc is detail, not the only copy.** Scope: `_make_fused_sparse_forward` (`moe_patch.py:1586`) is the ONLY
    forward without VRAM-cache wiring; the other two reach it and are the pattern to copy
    (fused-EXL3 `:2133-2161` for the gate/hold, `:2295-2303` for the `put()` ownership contract).
    1. `[x]` **MEASURED 2026-09-13 — `construct_flat` = 31.8 µs/expert interleaved (26.9 isolated), the stop
       rule PASSES, the layout is PACKED-BLOB.** `plans/probes/row30_s2b_construct_flat.py` (gitignored),
       64 distinct REAL experts cycled with the three `F.linear` calls the forward runs between dequants, so
       the launch queue and L2 pressure are production's: gemms-only 25.5 µs, +today's construct 81.0
       (construct **55.5**), +blob 57.2 (**construct_flat 31.8**). Blob output asserted `torch.equal` to
       today's fused path, which is itself equal to the per-role path. **Two corrections the build must carry
       forward, both of which shrink the spike's projection:** (a) `construct_flat` is **31.8**, not the
       spike's assumed 17.6 (S2b (ii)'s microbench on prebuilt flats, a different context — exactly the risk
       this step existed to retire); (b) today's construct measured **55.5 µs**, not the 78.3 the spike
       de-scaled from the profiled arm. Per-hit saving is therefore **h2d 121.9 + (55.5 − 31.8) = 145.6 µs**,
       and the honest projection is **+32.5 % (trace A) / +37.2 % (B), i.e. 4.3459 → 5.76 / 5.96 tok/s**, not
       +47.4 / +56.1 %. **Packed-blob still wins and by a real margin**: re-derived at the measured costs the
       crossover to dequant-bf16 sits at `construct_flat` **51.5 (A) / 53.8 (B) µs** against the measured
       31.8, and dequant-bf16 projects +27.0 / +30.0 %. Step 5's ≥ +20 % stop rule is unchanged and still the
       gate.
    2. `[x]` **DONE — `build_expert_blob` / `dequantize_expert_blob` / `expert_weights_from_entry`
       (`vaayu/moe_patch.py`).** `dequantize_expert_weights` is now the first two called in sequence, so the
       cached and uncached paths are THE SAME CODE, not two that must be kept in step. The non-fusable
       fallback and RC1's `NotImplementedError` both live in `build_expert_blob` and are tested there.
    3. `[x]` **DONE — the fused forward reaches the VRAM cache.** `_vram` lookup for every routed expert
       BEFORE the load dispatch, entries HELD for the whole compute loop, gated off during prefill and
       outside `_vram_cacheable_layers`, `put()` after construct. A GPU-resident expert that missed the RAM
       cache now issues NO disk read (P8's lesson, in this forward for the first time). `_expert_dense_weights`
       returns `(w_gate, w_down, w_up, entry)`. **Dense (non-packed) fused checkpoints are cached too** —
       entry = the weights; a packed-only wiring would silently no-op on every other checkpoint, which is the
       failure mode this row exists to stop. Ownership: `put()` False is ignored because these are plain
       torch tensors — unlike the EXL3 forward's BC objects there is no C++ heap, so dropping the reference
       IS the free (stated, not assumed).
    4. `[x]` **DONE — 17 tests, written first, watched fail.** `tests/test_fused_vram_expert_cache.py` (new,
       10) incl. **a hit actually skips the H2D** (the silent-no-op guard), hits change not one bit, no
       re-read off NVMe for a resident expert, the prefill and layer-restriction gates (**both
       mutation-checked — they fail when the gate is deleted**), a refused `put()` still answers correctly,
       and the entry-bytes invariant over BOTH layouts; `tests/test_packed_expert_dequant.py` (+7) holds the
       blob roundtrip `torch.equal` to the fused path on int4-g32 / int4-g128 / int8-g128;
       `tests/test_expert_vram_cache.py` (+1) for `_group_bytes`. **Suite 2081 passed, 15 skipped.**
       `ExpertVRAMCache._group_bytes` now skips any value that is not a dict of tensors (the blob entry's
       derived `geom`) **while keeping the `bc` name check** — generalising it away broke the EXL3 forward's
       accounting and the module-list `bc` lifecycle test caught it.
    5. `[x]` **DONE — +55.6 %, stop rule (≥ +20 %) PASSED. `4.1584 → 6.4702 tok/s`** on `T_unprof`, N=1,
       server knobs, 29G, `BALLAST_GB=25`, greedy ids identical across every arm. Two treated runs
       (U1 6.4702 / 6.4138, U2 6.4473 / 6.3829, spread 0.9 %) against a **same-day baseline arm on the
       stashed tree** (U1 4.1584, U2 4.1714, `vram=0hit/0miss` — the wiring's own control). VRAM cache
       83 % hit over the run (73–95 % per token), `usage=7.57GB/7.57GB`. **The same-day baseline is the
       point:** this row's pre-build figure of 4.3459 was measured earlier the same day and had drifted 4 %;
       quoting it would have inflated the win to +49 %. Spike projected 6.41–6.78 tok/s; measured 6.41–6.47.
       **Against the board: 7.6× Qwen3-235B-EXL3's raw 0.85, 5.7× row 27's served 1.1363.**
    6. `[x]` **DONE, and peak VRAM is TIGHT — recorded, not waved off.** Treated peak `max_allocated`
       **11.66 GiB** (baseline 4.61), device used **12.30 of 15.47 GiB**, i.e. **3.17 GiB free against the
       3.50 GiB reserve floor** — the cache overshoots its intended floor by ~0.33 GiB (allocator reservation
       + context overhead above the budget arithmetic). No OOM, `prefill_evict=0`, and the reactive guard for
       exactly this (`_ensure_prefill_headroom` → `evict_until_free`) is untouched, so a long prefill leans on
       reactive eviction rather than spare headroom. **Re-check if `max_seq_len` or the reserve floor moves.**
       Gates all green on the same tree: server gate on this checkpoint **9 passed**, Qwen3-235B
       `test_moe_accumulator_determinism.py` 8 passed, **MiniMax-M2.7 the same 8 passed** (run because
       `_group_bytes` is shared by all three forwards), Llama-3.2-1B `test_batch_prefix_equivalence.py`
       1 passed.
    7. `[x]` **§8 — (1) NO knob, stated not assumed:** the cache, its `free_vram − reserve` budget and the
       layer-restriction guard already exist and are already derived; the entry layout is derived from the
       tensors (bit width from packed-vs-original element count, group size from the scale's last dim), not
       chosen. **(2)** N/A, no knob. **(3) REACHABLE and PROVEN at N=1**, the server's only mode — server gate
       green on this checkpoint; a user pointing `vaayu/server` at it gets **6.47 tok/s** where they got 4.16.
       **(4)** nothing blocked. Full detail:
       [`…-row30-S2b-iii-vram-cache-RESULTS.md`](docs/2026-09-13-row30-S2b-iii-vram-cache-RESULTS.md).
    **Probes to re-use, not rebuild** (`plans/probes/`, gitignored): `row30_s2b_trace.py` captures an ordered
    routed-expert trace (monkeypatches `_maybe_record_usage`, no engine edit; `PROMPT=A|B`, `TAG=`),
    `row30_s2b_replay.py` replays it through the real cache (`BUDGETS_GB=`, `FRACS=`, `CONSTRUCT_FLAT_US=`).
    Traces already captured: `plans/probes/row30_out/trace-qwen3next{,-b}.json`.
  - `[x]` **VERDICT 2026-09-13 — NOT a bug under the layout that shipped; PINNED BY A TEST instead of
    changed.** `compute_vram_layer_range` (`vaayu_base.py:700`) sizes the working set with
    `geometry["expert_bytes"]`, the CHECKPOINT's expert size, not the cache ENTRY's. **Under packed-blob the
    two are equal** — the entry drops `weight_shape` (72 B/expert, never crosses PCIe) — so the guard is
    correct here and conservative in the safe direction; the 3.56× under-count the spike named belongs to the
    bf16 layout this build REFUTED and did not build. Changing it now would be a parameter no caller can fill
    with anything but today's value (CLAUDE.md §2). **What ships instead is the invariant, tested over BOTH
    entry layouts** (`test_entry_bytes_never_exceed_the_checkpoints_expert_bytes`): an entry may never exceed
    the checkpoint's expert bytes. **The one configuration that would still break it is named:** a dense
    fused MoE whose compute dtype is WIDER than the checkpoint stores (bf16 checkpoint at fp32) — no
    checkpoint on disk does this and no server knob reaches it. That test failing IS the re-open signal.
    **Measured floor, kept because it outlasts the layout choice:** a tier under one token's working set hits
    **0 %** — dequant reads 0.0 % at 1 and 2 GB where packed reads 45.2 % / 61.2 %.
- `[x]` **S2b (iv) — RE-PRICED AND DROPPED 2026-09-16: 0.33 % of `T_prof` (0.40 % of `T_unprof`) for both
  remainders, against a ~12 % claim; stop rule "under a third of its share" FIRED.** Microbench at the real
  geometry, 200 reps, pinned host, no model load (`plans/probes/row30_s2b_iv_bench.py`): a miss costs
  **113.1 µs/expert** as shipped (6 × `.to()` **94.0** + `build_expert_blob`'s 2 cats + geom **16.4** +
  compiled kernel **27.4**; a hit is 27.8). **The cats are the whole construct half and they are 16.4 µs, not
  the 49.1 the in-run hit/miss gap implied.** Removing them needs **no `ExpertRAMCache` change**: copy each
  role straight into a preallocated flat buffer instead of copy-then-cat (bit-identical decode, verified) —
  **108.4 µs total, i.e. 4.7 µs/miss-slot saved** = 0.37 ms/token at 79.7 miss slots = **0.20 % / 0.24 %**.
  Lever (i)'s remainder is **3.0 µs/miss-slot**, not the 2.5 % on record: **the slab arena is PINNED**
  (`vaayu_base.py:2685`), so the pinned microbench row (95.7 → 92.7) applies, not the pageable one
  (121.9 → 108.9). Scaling by the 2.61× in-run inflation (294.9 µs/miss-slot in-run vs 113.1 here) gives a
  generous ceiling of **~1.0 %**. `DECODE_TOP_K` shrinks it further (hit rate 83 → 90 %, misses are the only
  addressable population). **The row's stated prerequisite was also wrong-headed:** a host-side prebuilt blob
  is a CLONE, and the RAM tier is a zero-clone slab arena, so it would have doubled per-expert RAM or forced
  a persist-format change. **Kept, unexplained:** the in-run `construct_miss − construct_hit` gap of 49.1 µs
  is not concatenation work — ~33 µs/miss-slot (≈1.4 % of `T_prof`) does not reproduce off-run and nothing in
  (iv) addresses it; allocator churn against a full VRAM cache is the suspect.
  RESULTS: `docs/2026-09-16-row30-S2b-iv-reprice-RESULTS.md`. §8: nothing shipped — no knob, no code, no
  plumbing; the verdict applies at N=1, the server's only mode. Original text: **the leftover of (i) AND (ii), behind ONE
  prerequisite: store each expert as a single contiguous blob at cache-fill time.** Both remainders need exactly that. With it: h2d's coalesced 2.5 %
  (§5g) becomes collectable without the per-call `cat` that made it a loss, AND the dequant drops its two
  `cat`s to reach the microbench's ~17.6 µs (0.0444 → ~0.018 s ≈ another **10 %** of the token). Together
  **~12 %**. Touches `ExpertRAMCache`'s storage layout, so it is a real change, not a tweak.
- `[x]` **THREE LEVERS THIS MEASUREMENT PUT ON THE TABLE — ALL THREE ANSWERED 2026-09-13: (i) REFUTED at
  ≤ 5.7 % of a token (h2d is bandwidth-bound, not launch-bound), (ii) LANDED at +6.92 %, (iii) BUILT at
  +55.6 %. Original text kept below for the reasoning:**
  **(i) coalesce the per-expert H2D** — 9 separate copies per expert, one per expert or per layer instead;
  attacks the largest phase (~34 %), no new kernel. **(ii) land RC1's banked `torch.compile` dequant** —
  measured **17.9 µs/expert vs the 117.5 shipped, 6.5×**, banked behind "do not optimise before phase 3 says
  dequant is visible". **Phase 3 says 25.1 %. That condition has fired.** **(iii) wire the VRAM expert cache
  into this forward** — `V=0.0`, `vram=0hit/0miss`, while **7.79 GB of VRAM budget sits reserved and idle**;
  verified in source, not inferred: the non-EXL3 fused forward has ZERO references to `_expert_vram_cache`,
  `_dispatch_expert_misses`, the slab arena or the prefill stager. A VRAM-resident hot set deletes h2d AND
  construct for what it holds. **This is the `VAAYU_CACHE_BIAS` pattern (CLAUDE.md §0) in a fourth place.**
  All three are bounded together by `FLOOR[plateau] = T_noexp 0.0348 + disk 0.0042 = 0.0390 s → ×6.30 over
  U1`, i.e. **~25.6 tok/s** if every host-side expert cost vanished.
- `[x]` **RC4 (c) DONE 2026-09-13 — the planner now charges KV only for the layers that hold it.
  Derived `max_seq_len` 16384 -> 65536 (4x) on the int4 checkpoint; server gate green on it
  (9 passed, `derived max_seq_len=65536`). 0 % on throughput, and that is the honest framing:
  this is CONTEXT, not speed.** RC4 (b) had already made the ALLOCATOR right (zero-capacity
  placeholders on non-KV layers); the PLANNER kept multiplying by `config.num_hidden_layers`.
  The checkpoint index says **12 of 48** layers carry `self_attn` and 36 carry `linear_attn`
  (stride 4), so `kv_bytes_per_token` quoted **exactly 4x** what the cache allocates.
  **Fix:** `vaayu/batch.py :: kv_bearing_layer_count(model)` counts layers owning BOTH `k_proj`
  and `v_proj`, read off the **MODULE TREE**, never `config.layer_types` (that field is optional
  and `"full_attention"` is a transformers convention; the projections are what actually put
  bytes in the cache -- CLAUDE.md §5). Errs SAFE both ways it can be wrong: finds nothing (fused
  `qkv_proj`, an unfamiliar tree) -> full depth, i.e. today's behaviour; and clamped to the config
  depth so a second layer stack (this checkpoint's 10.12 GiB MTP module) cannot inflate it.
  Wired at **all three** charging sites -- `VramShape.kv_per_token`, `derive_expert_ceiling`,
  `batch.geo_from_model` -- with a **source-guard test** asserting no site in `vaayu_base.py` /
  `batch.py` / `vram_plan.py` passes `config.num_hidden_layers` into `kv_bytes_per_token`. That is
  the invariant `kv_bytes_for`'s own docstring names: if the reserve and admission quote different
  expressions, admission is lying and the gap surfaces as an OOM, not a deferral.
  **Paired same-day arms** (`plans/probes/row30_rc4c_arm.py`, server knobs, 29G; baseline on the
  stashed tree): `kv_per_token` 98304 -> **24576**, `max_seq_len` 16384 -> **65536**, `s_max`
  131072 -> 262144. **Reserve 3.50 GiB (KV 1.50) and pinning budget 10.57 GiB are UNCHANGED, as
  predicted before the run** -- `plan_vram` picks `chosen` to fit the 3.50 GiB floor and then sets
  `reserve = max(predict_peak(chosen), floor)`, which IS the floor; and `derive_expert_ceiling`'s
  KV term is gated on `prefix_enabled`, which RC7 turned off here. Peak VRAM 9.043 -> 9.055 GiB
  (+0.13 %), **16 greedy ids IDENTICAL**, 1.393 -> 1.384 tok/s (one cold 16-token run, not a perf
  arm). **Uniform-model control, Llama-3.2-1B:** `kv_layers_probed == num_hidden_layers == 16` and
  `kv_per_token` identical at 32768 -> the four shipping checkpoints cannot move.
  **Gates:** suite **2088 passed, 15 skipped** (same-day HEAD baseline **2082**, so +6 = exactly
  the new tests -- the 2081 on record was one stale); server gate on the int4 checkpoint 9 passed;
  Qwen3-235B accumulator 8 passed; MiniMax-M2.7 accumulator 8 passed; Llama-3.2-1B prefix 1 passed.
  **The term this uncovers, measured and DELIBERATELY NOT CHARGED:** the 36 linear-attention layers
  hold per-sequence conv + recurrent state, charged nowhere before or after. From the config and
  `LinearAttentionLayer.lazy_initialization`'s own `zeros_like` contract: conv `8192 x 4 = 32,768`
  elems + recurrent `32 x 128 x 128 = 524,288` elems = **1.0625 MiB/layer bf16 x 36 = 38.25 MiB**,
  i.e. **1.1 % of the 3.50 GiB floor**, which absorbs it. It is genuinely uncharged only if a
  caller supplies a `max_seq_len` big enough that `reserve = predict_peak > floor`. It is
  ARITHMETIC -- no arm isolated those allocations on device. **§8: (1) NO knob needed, stated not
  assumed -- it is a derivation off the tree the load just built, and a knob could only be set
  wrong. (2) N/A. (3) REACHABLE AND PROVEN at N=1, the server's only mode. (4) nothing blocked.**
  Generalises to any hybrid checkpoint (sliding window, linear attention, Mamba-style mixers).
  Detail: [`...-row30-RC4c-kv-layer-charge-RESULTS.md`](docs/2026-09-13-row30-RC4c-kv-layer-charge-RESULTS.md).
- **§8 for what landed — UPDATED 2026-09-13, and it is now SHIPPED, not banked.** (1) **No knob needed**,
  stated not assumed, for every fix: RC1/RC3 are format and architecture detection, RC5 reads the
  checkpoint's own index, RC6 reads which buffers hold KV, RC7 refuses on a detected capability; the
  profile instrumentation rides `profiling_mode`, already on record as NOT an operator control
  (`tests/test_engine_param_reachability.py`). (2) N/A, no knob. (3) **REACHABLE, and PROVEN:**
  `SERVER_CKPT=<the int4 checkpoint> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` →
  **9 passed in 24.9 s**, load 4.2 s, derived `max_seq_len` 16384, prefix refusal printing its correct
  reason. A user can point `vaayu/server` at this model today and get **4.06 tok/s at N=1** — the server's
  only mode. RC5 and RC6 also reach every other model (any non-layer-major index; any hybrid model).
  Regression gates for the models that already ship are green on the same tree:
  `test_moe_accumulator_determinism.py` on Qwen3-235B (8 passed, **29G — it is OOM-killed at 22G directly
  after another checkpoint's run**) and `test_batch_prefix_equivalence.py` on Llama-3.2-1B (1 passed).
  (4) Nothing is blocked; the three levers above are the next work.


**AVX2, stated plainly (asked 2026-09-12): phase 4 does NOT write AVX2 intrinsics.** It calls
`torch.ops.aten._weight_int4pack_mm_for_cpu`, which is what emits AVX2/AVX-VNNI on this box (row 26 measured that
kernel at **214 µs/expert at g128**). "CPU-AVX2 hybrid" in rows 26/28 is the NAME of the idea, not a kernel-writing
commitment. A hand-written AVX2 kernel is only needed for formats torch has no CPU path for — MXFP4 (e2m1) and fp8,
per this row's format table — and **the selected int4-g32 checkpoint avoids that**; what it does owe is a REPACK, since
compressed-tensors' `pack-quantized` layout is not torch's `int4pack` layout.

**NEW INTERACTION, 2026-09-12 — row 27 shipped, so the CPU is no longer idle, and phase 4's bar moved:**
the default now runs **8 `SCHED_IDLE` spinners on the E-cores** plus **engine threads pinned to the P-cores**
(`KEEP_AWAKE="auto"`, `PIN_ENGINE_THREADS=True`). Phase 4 wants **6 P-core threads** for expert GEMV — i.e. the same
cores the engine is now pinned to. This is the CPU/io_uring core contention this row already names as phase 4's most
likely miss, now with a concrete source. Two consequences: (1) the keep-awake baseline this row demands is **measured**
as of today (Qwen3-235B +23.1 % decode through the server), so the hybrid must beat a materially faster baseline than
the one row 29 projected against; (2) a hybrid arm must state what it did with `KEEP_AWAKE`/`PIN_ENGINE_THREADS` —
leaving them at default while the CPU does expert work is a different experiment from turning them off, and the two
must not be compared to each other.

**CHECKPOINT SELECTED BY THE USER, 2026-09-12: `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`.** This is the
model S2/S3 are built against; the sweep below stays as the comparison it was chosen from.

**What it is, measured (not from its model card):** despite the "AWQ" in the name its `quant_method` is
**`compressed-tensors`**, format `pack-quantized`, `group_0: bits=4 strategy=group group_size=32 sym=True
targets=['Linear']`. 512 experts × 48 MoE layers, top_k **10**, **1.769 MB/expert** (1.688 MiB), **45.82 GiB** in 10
shards / 226,472 tensors, **0.85 GB/token** — **5.22× lighter per token than Qwen3-235B-EXL3's measured 4.43**.
C1 PASS (`qwen3_next` → `Qwen3NextConfig`), C4 PASS (225,792 keys match `\.experts\.(\d+)\.`, 512 ids, 48 layers).

**Four consequences of this choice, to carry into S2/S3 — none is a blocker, all are re-pricings:**
1. **`pip install compressed-tensors` is required** (C2 dispatch fails without it: `ImportError: compressed_tensors is
   not installed`). It lands in `requirements.txt` in the commit that makes the model load.
2. **The binder gap is S2's first real problem.** transformers 5.5.4 merges per-expert disk keys into
   `Qwen3NextExperts`'s fused 3D params via `conversion_mapping.py` (`qwen3_next` → `qwen2_moe`), but that converter
   matches **`.weight`**, and this checkpoint stores `weight_packed` / `weight_scale` / `weight_shape`. The
   compressed-tensors quantizer inherits `get_weight_conversions() → []`, so nothing maps the packed expert tensors
   onto the fused parameter. Expect to write that binding; it is the "load-path bugs, not tuning" this row warns about.
3. **S3's CPU kernel is NOT blocked by g32 — verified on device 2026-09-12:**
   `torch.ops.aten._weight_int4pack_mm_for_cpu` accepts `group_size` 32/64/128/256 on torch 2.11.0+cu130 (checked with
   the correct `[K/gs, N, 2]` scale layout; an earlier probe failed only because the scales were shaped wrong). BUT
   row 26's **214 µs/expert was measured at g128**, g32 carries 4× the scales, and compressed-tensors' pack layout is
   not torch's int4pack layout — a repack is needed either way. **Re-measure that number, do not inherit it.**
4. **It is the heaviest of the four candidates**, chosen with that known: 0.85 GB/token vs 0.78 for
   `RedHatAI/…w4a16` and `Intel/…AutoRound` (+9 %), 45.82 GiB vs 40.87 / 39.48 (+12 %) — and vs row 29's assumed
   0.80 GB/token / 41.1 GB, which produced the **18.1 tok/s** projection. More bytes AND a bigger footprint against a
   fixed ~8.5 GB expert-cache ceiling both push the same way, so **expect below 18.1 tok/s and re-derive it from the
   first measured run rather than quoting the projection.**

**Smoke rung unchanged and still recommended first:** `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16`
(15.52 GiB, 0.93 GB/token, int4 g128, same compressed-tensors family) — it hits the same binder problem for a quarter
of the download. If you would rather smoke on the same group size as the target, any 30B compressed-tensors g32 build
does the job equally well.

**S1 CANDIDATE SWEEP, 2026-09-12 — Qwen3-Next-80B-A3B-Instruct int4 located and probed (metadata only, no
download). Probe: `plans/probes/row30_s1_probe.py <repo_id>` (gitignored — it re-derives all of this in ~1 min/repo).**

| repo | format | C2 dispatch | footprint | GB/token |
|---|---|---|---|---|
| `unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit` | bitsandbytes NF4 (blk 64) | **PASS, `validate_environment() → OK`** (bitsandbytes already installed) | 39.23 GiB | **0.78** |
| `RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16` | compressed-tensors int4 **g128** sym | needs `pip install compressed-tensors` | 40.87 GiB | **0.78** |
| `Intel/Qwen3-Next-80B-A3B-Instruct-int4-AutoRound` | auto-round 4-bit **g128** | needs `pip install auto-round>=0.5` | 39.48 GiB | 0.78 |
| `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` (34k dl) | compressed-tensors **g32** | needs `pip install compressed-tensors` | 45.82 GiB | 0.85 |
| smoke rung: `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` | compressed-tensors int4 g128 | same pip | **15.52 GiB** | 0.93 |

All pass **C1** (`qwen3_next` → `Qwen3NextConfig`; 30B: `qwen3_moe`) and **C4** (221k–442k keys matching
`\.experts\.(\d+)\.`, 512 expert ids over 48 layers; 30B: 55k / 128 / 48). **Row 29's projection is confirmed by the
real checkpoints** — it assumed 41.1 GB / 0.80 GB per token, they measure 39–41 GiB / 0.78, i.e. **5.7× lighter per
token than Qwen3-235B-EXL3's measured 4.43 GB/token.**

**CORRECTION to this row's "disk keys and module layout disagree".** Half of that is wrong and it changes S2:
transformers 5.5.4 DOES ship the merge — `transformers/conversion_mapping.py`'s `_MODEL_TO_CONVERSION_PATTERN` maps
`qwen3_next` AND `glm4_moe` → `qwen2_moe`, whose `WeightConverter` folds `mlp.experts.*.gate_proj.weight` +
`up_proj.weight` into the fused 3D `mlp.experts.gate_up_proj` (`MergeModulelist`/`Concatenate`). **But it matches
`.weight`.** compressed-tensors / GPTQ / AutoRound store `weight_packed`+`weight_scale` (or `qweight/scales/qzeros/
g_idx`), and **all three quantizers inherit `get_weight_conversions() → []`** (enumerated: only bitnet, bnb_4bit,
bnb_8bit, finegrained_fp8, fouroversix, fp_quant, metal, mxfp4, quark, sinq, torchao override it). So their packed
expert tensors have **no target to bind to** — that, not the fused layout itself, is S2's real blocker.
**bnb-4bit is the one family where both halves exist** (`Bnb4bitDeserialize` + `.weight`-shaped expert keys).

**Recommendation, two tracks:** (a) to get a non-EXL3 MoE LOADING, take `unsloth/…bnb-4bit` — no new package, complete
native path; caveat: NF4 block-64 is **not** the int4-g128 row 29's CPU kernel was priced on (214 µs/expert), and it
carries 442k tensors (6 per projection) for the streaming index to chew. (b) for the S3 hybrid itself, take
`RedHatAI/…w4a16` (int4-g128, exactly row 29's assumption) and pay one pip install plus the expert-binder work.
**The 30B rung is now OPTIONAL** — the 80B target is already downloaded (see Disk below), so the rung no longer
saves a download; take it only if S2 stalls on numerics and a smaller model would localise the fault faster.
~~Run S2 on the 30B RedHat rung first (15.5 GiB) to hit the binder problem cheaply before a 40 GiB download.~~

**S1 data point, 2026-09-12 — `QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix` REJECTED (metadata only, no download).**
Asked for by the user; probed against the playbook's five contracts.
- **C1 PASS** — `glm4_moe` → `Glm4MoeConfig` in transformers 5.5.4.
- **C2 FAIL, measured** — `quant_method: "gptq_marlin"` → `ValueError: Unknown quantization type`; rewritten to
  `"gptq"` → `NameError: name 'QuantizeConfig' is not defined` (needs **gptqmodel**, which is NOT installed and NOT in
  `requirements.txt`; `auto_gptq` absent too). Its mixed-precision `dynamic` block
  (`+:model[.]layers[.]([0-4]|4[4-6])[.].*: bits 8`) is a GPTQModel extension transformers' `GPTQConfig` does not model.
- **C3 MISMATCH** — repo ships **no remote code, no `auto_map`**, so it rides built-in `glm4_moe`, and 5.5.4 builds
  `Glm4MoeNaiveMoe`: fused 3D `nn.Parameter`s (`gate_up_proj [128, 2816, 4096]`, `down_proj`), **no `nn.Linear` for
  GPTQ's QuantLinear to replace**, and a FUSED `gate_up` where the disk has separate `gate_proj`/`up_proj`.
- **C4 PASS** — 70,656 keys match `\.experts\.(\d+)\.`, 128 expert ids, layers 1–46 (46 = the `num_nextn_predict_layers`
  MTP layer, unused by `Glm4MoeForCausalLM`). Streaming index would be satisfied.
- **Geometry, exact, read from the shard safetensors headers:** 4-bit expert **9.027 MB** (8.609 MiB, vs row 29's
  assumed 8.77 MiB ✓); 8-bit expert (layers 1–4, 44–45) **17.745 MB**; **routed 3.67 GB/token** (top-8 over 45 MoE
  layers) vs row 29's assumed 3.31 (+11 %); footprint **66.02 GiB** vs 52.9 (+25 %); plus an **unquantized fp16 shared
  expert, 34.60 MB/layer × 45 = 1.56 GB**, touched every token. Qwen3-235B measures 4.43 GB/token today, so this build
  is only ~1.2× lighter than what already runs here — **row 29's 7.3 tok/s was priced on a pure-int4 52.9 GB build at
  85 % hit and must be re-derived for this one, not inherited.**
- **Verdict:** stay with Qwen3-Next-80B-A3B int4 (smoke rung Qwen3-30B-A3B int4). The C3 fused-3D problem applies to
  those too — it is this row's known blocker — but they do not also carry GLM's C2 failure. A **pure int4** GLM-4.5-Air
  build (`quant_method` plain `gptq`/`awq`, no `Int8Mix`) would drop the +11 % bytes and one of the two blockers.

**Source:** the user, 2026-09-12, choosing `Qwen3-Next-80B-A3B-Instruct` int4 to test CPU+GPU token generation.
**Durable copy of the plan summary is this row.** Detail:
[`docs/2026-09-12-row30-IMPL-nonexl3-int4-cpu-gpu-hybrid.md`](docs/2026-09-12-row30-IMPL-nonexl3-int4-cpu-gpu-hybrid.md).

**FOUR THINGS THAT ARE NOT TRUE and are easy to assume:**
1. **"Vaayu supports int4 group-128."** It does not. `torch._weight_int4pack_mm_for_cpu` is a **torch** kernel that
   row 26's throwaway probe benchmarked (214 µs/expert, 46.9 GB/s on 6 P-cores). **Vaayu has NO CPU expert compute
   of any kind** — `compute_device` is forced to `"cuda"` for EXL3 (`moe_patch.py:819`).
2. **"Non-EXL3 MoE runs, just untuned."** It has **never run on device** (CLAUDE.md §0: "on-device today: none").
   Expect load-path bugs, not tuning.
3. **"A 4-bit HF checkpoint will just stream."** Two independent layout questions — see below.
4. **"The hybrid is the win."** Best case **+21 %** on this box (row 29). The **model swap is ~21×**. Phases 0–3 are
   the value; do them even if phase 4 never happens.

**The blocker nobody has hit yet — C3 ≠ C4.** `detect_moe_blocks` (`moe_patch.py:552`) reads the **module tree**;
`ExpertOffsetIndex` (`persist/expert_offset_index.py`) reads the **disk keys**, with exactly one rule:
**`\.experts\.(\d+)\.`**. transformers 5.5.4 builds `Qwen3MoeExperts` / `Qwen3NextExperts` / `GptOssExperts` as
**fused 3D `nn.Parameter`s** (`modeling_qwen3_moe.py:215–226`) → `expert_style == "fused"`, `is_exl3 == False` →
**`_make_fused_sparse_forward` (`:1289`), the branch nothing has ever run.** Meanwhile GPTQ/AWQ int4 checkpoints are
**per-expert on disk** (good for streaming, risky for loading: only `finegrained_fp8` implements transformers'
fused-MoE experts abstraction — **there is no GPTQ/AWQ fused-MoE experts class**), and gpt-oss MXFP4 is **one 3D
tensor per layer → the index finds ZERO experts → no streaming at all.**

**Phases.** **0** — `row30_load_probe.py`: config + index only, no download, **< 1 hour**; counts per-expert keys,
derives bytes/expert + bpw, checks the quantizer resolves. **Decides the path: A it loads · B dequant+repack to
torch int4-g128 at persist time (likely, and the best shape — it is also the hybrid's dispatch point, a new
`Int4PackLoaderStrategy` beside `EXL3LoaderStrategy` at `loader_strategy.py:431`) · C split fused 3D at persist time
(pick another repo first).** **1** make it load (`pip install gptqmodel`/`autoawq` — only bitsandbytes is installed;
verify 64 coherent greedy tokens). **2** make it stream (index count == `moe_layers × num_experts`, derived).
**3** run `row26_decode.py` unchanged. **4** hybrid, **only if the gate passes**.

**THE GATE for phase 4:** `(h2d + gemm + tail) / T`. Row 29 projects **~31 % (80B) / ~22 % (30B)**.
**< 15 % → STOP and close the idea.** ≥ 15 % → realistic gain ≈ share × 0.6 (= 1 − 18.7/46.6, the measured
PCIe-vs-CPU ratio). **Baseline must be row 27's keep-awake arm, not today's** — a hybrid keeps the CPU busy and would
otherwise be credited with up to 21 % that one `SCHED_IDLE` spinner delivers alone.

**Smoke rung first: `Qwen3-30B-A3B-Instruct-2507` int4 (~17 GB, `qwen3_moe` — the family Vaayu already runs at
235B).** Repos: `JunHowie/…-GPTQ-Int4`, `cyankiwi/…-AWQ-4bit`, `stelterlab/…-AWQ`. Target repos for the 80B:
`cyankiwi/…-AWQ-4bit`, `dazipe/…-GPTQ-Int4A16`, `RedHatAI/…-quantized.w4a16`, `Intel/…-int4-AutoRound`.

**Row 29's projections this row must beat or refute (±20 %):** 80B — expert 1.59 MiB, footprint 41.1 GB,
0.80 GB/token, vram 91 % / disk 3 %, **T ≈ 0.055 s → 18.1 tok/s**, phases serial 0.038 / devbusy 0.005 / h2d 0.004 /
gemm 0.006 / tail 0.007. 30B — 2.39 MiB, 15.4 GB, 0.96 GB/token, **T ≈ 0.049 s → 20.5 tok/s**.
**Expect the surprise to be that `serial` dominates** (0.038 of 0.055 s = 69 %) — Vaayu's own per-layer Python/launch
overhead. If measurement confirms it, **the next lever on this model is CUDA graphs / fused per-layer dispatch, not
AVX2** — open it as its own row rather than absorbing it here.

**Risks, ranked:** (1) the 4-bit checkpoint may not load in 5.5.4 → phase 0 catches it in under an hour;
(2) `_make_fused_sparse_forward` has never executed; (3) **24,576 slabs** on the 80B (512 experts × 48 layers, ~1.6 MiB
each) vs Qwen's 1,034 of ~11 MiB — untested scale for `SlabArena`, the offset index and io_uring, and
`_NVME_READ_CONCURRENCY = 3` was tuned at ~11 MiB reads (**re-measure with `p8_samepath.py`, do not assume**);
(4) **`qwen3_next` is hybrid linear/full attention** (full attention every 4 layers) and Vaayu's KV cache, prefix
cache and derived `max_seq_len` all assume uniform attention — **unverified**; this risk does not exist on the 30B
rung, which is another reason to do it first; (5) int4 4.25 bpw quality vs EXL3 5.0 unassessed; (6) CPU/io_uring core
contention, the most likely reason phase 4 misses its projection.

**§8:** (1) phases 0–3 no knob; phase 4 needs `EXPERT_COMPUTE_DEVICE` in `vaayu/server/config.py` **in the same
commit**, with its measured cost, plus `vaayu/server/README.md`. (2) phase 4's knob must be passed by `app.py`'s
`from_pretrained(...)` — grep it and `engine.py`. (3) reachable — per-token **N=1**, the server's only mode
(`MAX_BATCH_SIZE = 1`, `config.py:144`). (4) not built; phase 1 unblocks 2–3, phase 3's gate unblocks 4.

**Disk (updated 2026-09-12):** **299 GB free** — DeepSeek-V4-Flash has been deleted, and
`cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` is **already on disk and complete** (46 GB, 10/10 shards + index +
config). Vaayu's own `splitted_model/` copy needs another ~46 GB. **No download stands between S1 and S2.**

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

# 31 · [x] Stop spending the expert RAM cache on context nobody is using — CLOSED 2026-09-12: defaults STAY, the trade is now measured and documented

**VERDICT: `PREFIX_SESSIONS = 3` and `MAX_SEQ_LEN = None` both STAY. No default moved.** The lever is real
and it is NOT free, and the arm that made it look free was the unrepresentative one. Measured on
Qwen3-235B THROUGH THE SERVER, prefix cache on, `MemoryMax=29G`, `BALLAST_GB=25`, 4 two-turn arms + 5
fan-out arms, all `exit=0`, identical text across arms (trap 6):

| workload | `PREFIX_SESSIONS=1` vs default |
|---|---|
| one session, 2.7k context, two turns | **+5.7 %** wall (decode +6.1 %, TTFT −9.2 %) |
| four sessions, 2.7k context | +1.9 % |
| **two sessions, 7.4k context** | **−8.9 %** |

The third row is the workload `CLAUDE.md` §0 describes, and it decides the row: at 7.4k the turn-2 prefix
match collapsed **7422/7517 → 44/7517** and TTFT went **25.0 s → 51.3 s**, which costs far more than the
3.2 GB of extra expert cache pays. Plan I's *collapses to ZERO, not proportionally* reproduced exactly,
with the engine's own `WARNING: prefix pool over capacity … Budget 1.49 GB holds 1 session`.

**Full two-turn table (2.7k context, medians of 2 reps, baseline spread 0.7 % decode / 1.4 % TTFT):**

| arm | expert budget | cap/layer | VRAM layers | t1 tok/s | t2 tok/s | RAM hit | wall |
|---|---|---|---|---|---|---|---|
| `3×None` **default** | 12.26 G | 11 | 65/94 | 0.7558 | 0.6364 | 49.9 % | 143.0 s |
| `3×4096` | 14.49 G | 13 | 65/94 | +2.5 % | +2.8 % | 53.7 % | +2.6 % |
| `1×None` | 15.42 G | 13 | 65/94 | **+6.1 %** | +5.2 % | 53.7 % | **+5.7 %** |
| `1×4096` | 16.16 G | 14 | 65/94 | **+6.7 %** | +5.9 % | 55.5 % | **+6.4 %** |

**Why a prefix hit is worth so little at 2.7k and so much at 7.4k (−6.6 % vs −53 % of TTFT):** the expert
sweep SATURATES at either size (95 new positions × top-8 already exceeds 128 experts/layer), so reuse
only saves the attention/GEMM work, which grows with context while the expert bytes do not. **On a
streamed MoE, a second turn's TTFT is waiting on the expert cache, not the prefix cache.**

**§8:** (1) no new knob — both existed; the measured cost of getting each wrong is now IN
`vaayu/server/config.py` beside the default. (2) plumbed already. (3) reachable at N=1, measured there.
(4) shipped: `documentation/TUNING.md` §3b (`prefix_sessions`) and §3c (`max_seq_len`), plus the
`PREFIX_SESSIONS` row in `vaayu/server/README.md`.

**MiniMax was NOT re-measured on device** — deliberately, because no default moved and the Phase-0
projection puts its gains strictly BELOW Qwen's (+8.2 % vs +13.2 % at `1×None`) against the same
collapse risk, so the conservative verdict covers it. Anyone who wants `PREFIX_SESSIONS=1` shipped for
MiniMax owes the 7.4k fan-out arm on that checkpoint first.

**PHASE 0 DONE 2026-09-12 (offline pricing, no GPU, no load — `plans/probes/row31_price.py`; detail
[`…-row31-prefix-reservation-RESULTS.md`](docs/2026-09-12-row31-prefix-reservation-RESULTS.md)).** Five findings, and
two of them change what this row is:

1. **`MAX_SEQ_LEN=None` IS 8192 on both MoE models** (the boot log derives it: *largest context whose predicted peak
   fits the 3.50 GiB reserve floor, capped at s_max=65536*), so 3 of the 9 grid points are duplicates of `8192`.
2. **The two knobs are ONE knob — the PRODUCT `sessions × max_seq_len × kv_bytes/token`.** Nothing downstream sees the
   factors, so `1 session × 8192` and `2 × 4096` are literally the same arm. The 9-point grid holds **5 distinct cache
   configurations** (Qwen budgets 12.21 / 13.79 / 14.58 / 15.36 / 16.15 GB).
3. **`MAX_SEQ_LEN` does NOT pay twice — this row's VRAM half is REFUTED offline.** The VRAM reserve is
   `max(predict_peak_vram(S), 3.5 GiB floor)` and the log shows it sitting exactly ON the floor at 8192
   (`reserve=3.50 GiB of which KV 1.47 GiB`). Under flash (`score_elem_bytes = 0`, the kernel the server picks) the
   only S-dependent term left is KV capacity, so `reserve(4096) = reserve(8192) = 3.50 GiB` — **halving the context
   returns ZERO VRAM, and Qwen's `Restricting VRAM cache to the first 65/94 MoE layers` does not move.** Same shape as
   P5's SETTLED verdict, on a knob P5 never touched.
4. **Projected decode (simulator's miss-rate RATIO applied to the MEASURED device-busy):** Qwen `2×None` +7.8 %,
   `3×4096` +13.2 %, `1×None` **+13.2 %**, `1×4096` **+18.5 %**; MiniMax +4.0 / +6.0 / **+8.2** / **+10.6 %**. The gain
   is QUANTISED by the derived per-layer cap, not smooth in bytes (Qwen's 14.58 GB and 15.36 GB both derive cap 13).
5. **The `expert_sim` anchor gap (STILL OPEN) now has a candidate cause:** the simulator's absolute bytes/token match
   Qwen to **+2.5 %** but over-state MiniMax by **+57 %**, and the simulator models the **RAM tier only** while the VRAM
   expert tier — *unrestricted* on MiniMax (working set 4.40 GB fits), *restricted to 65/94 layers* on Qwen — serves
   part of the same demand stream. Right direction, right order of magnitude. Not proven; recorded.

**Validated before being trusted:** the pricing script reproduces the boot log's own ceiling (12.21 vs 12.20 GB Qwen,
10.92 vs 10.92 MiniMax), its KV term (1.47 / 1.94 GiB at 8192) and `expert_sim`'s own recorded 44 % Qwen hit rate.

**TRAP found on device, costs an arm if missed:** raising the ceiling raises RSS, so **every arm needs a BIGGER
cgroup, and the same one**. The first attempt ran `MemoryMax=22G` (row 27's figure) and BOTH arms were **OOM-killed
(exit 137)** — the `PREFIX_SESSIONS=1` arm during its LOAD, because its arena is ~3.2 GB larger by construction. The
engine sizes from host `available`, which the cgroup does not constrain. Re-run at a uniform **29G**.

**Also measured on device, and it reframes TTFT for this row:** a turn-2 prefix HIT reused **2732 of 2827 tokens** yet
TTFT stayed **24.4 s** — the 95 remaining positions still route ~every expert per layer (95 × top-8 = 760 slots vs 128
experts → saturation), ≈142 GB at 6.3 GB/s. **Prefix reuse saves attention work, not expert streaming.** So on a
streamed MoE the expert cache, not the prefix cache, is what a second turn's TTFT is waiting on.

`derive_expert_ceiling` (`vaayu/vaayu_base.py:302`) is literally
`available_RAM − 3.5 GiB − (prefix_sessions × max_seq_len × kv_bytes_per_token)`. **The expert RAM cache is paid for
out of the KV reservation, and both terms are ALREADY server knobs** — `PREFIX_SESSIONS` (default 3) and
`MAX_SEQ_LEN` (default `None` = derive the largest context that fits).

**What the default reservation costs**, from this file's own checkpoint table (Qwen 0.193 MB/token decimal,
MiniMax 0.254), against a printed 15.2 GB Qwen budget:

| model | 3 × 8192 (default) | at `PREFIX_SESSIONS=1` | returned to the expert cache |
|---|---|---|---|
| Qwen3-235B | 4.74 GB | 1.58 GB | **+3.16 GB (+21 % of budget)** |
| MiniMax-M2.7 | 6.24 GB | 2.08 GB | **+4.16 GB** |

`MAX_SEQ_LEN` pays **twice**: halving it returns RAM to the ceiling *and* shrinks the VRAM KV reserve, which is what
pushes Qwen into `restricting VRAM cache to the first 62/94 MoE layers` (printed at every load — 32 of 94 MoE layers
have no VRAM expert cache at all).

**Direct on-device evidence a bigger expert cache pays, already in the tree** at `vaayu/vaayu_base.py:286-291`
(Qwen3-235B, 2 sessions × 3 rounds, real context, `MemoryMax=27G`): ceiling **8.5 GiB → warm wall 36.6 s**, 13 GiB →
32.0 s, **18 GiB → 27.3 s = −25 % end to end**, prefix reuse unchanged at 83 % in every arm.

**This is a genuine two-sided trade and the other side is measured too:** prefix reuse is worth **4.44×** where
batching's best was 1.44× (P10), and reuse collapses to **ZERO, not proportionally**, one session past capacity
(Plan I). A single-agent loop plausibly wants `PREFIX_SESSIONS=1`; fan-out (the workload `CLAUDE.md` §0 describes)
wants the sessions. **Do not set it globally on a hunch.**

**Verified 2026-09-12 that the ceiling is what binds, so the reservation returns 1:1:**
`compute_expert_budget` (`vaayu_base.py:327`) takes `min(ceiling, available − ~2.5 GiB)`, and the ceiling is
`available − 3.5 GiB − R`, i.e. smaller by `1 GiB + R` for every `R ≥ 0`. The boot log prints the terms —
`[Vaayu-MoE] Expert cache ceiling X GB (derived: available Y GB, reserving N prefix session(s) of Z GB each)` — and a
later `Expert RAM cache budget trimmed` line means the pinned arena, not the ceiling, bound it; read both before
believing a sweep point.

**Price it for free first:** `vaayu/expert_sim.py` replays a recorded trace against `ExpertRAMCache`'s exact policy
(per-layer cap + global LRU) and returns hit-rate-vs-capacity from ONE instrumented decode run. **Read its curve's
SHAPE, not its absolutes** — its unexplained anchor gap is a STILL-OPEN item in this file.

**Also free, found the same day, unrelated to the cache:** `vm.swappiness` is **60** and this box had **2 GB swapped
out with 17 GB free**; the CPU governor is `powersave` / EPP `balance_performance` (row 26). Row 27's +23 % came from
C-state residency, **not** frequency, so `performance` + EPP `performance` is a *separate, untested* knob that may
compose with it. `/sys/block/nvme0n1/queue/scheduler` is already `none`, so that part of
`scripts/vaayu_optimize_os.sh` is done.

- [x] Sweep `PREFIX_SESSIONS ∈ {1,2,3}` × `MAX_SEQ_LEN ∈ {4096, 8192, None}` — priced offline (9 points → **5 distinct**), 4 run on device
- [x] Confirmed on a two-turn session at BOTH 2.7k and 7.4k tokens, prefix cache ON — and the 7.4k arm is the one that reversed the verdict
- [x] Landed: default UNCHANGED (measured), `documentation/TUNING.md` §3b + §3c, `config.py` comments, `server/README.md`

**Stop rule: best point under +5 % on that two-turn session → document the default and close the row.**

**§8:** (1) no new knob — both exist. (2) plumbed. (3) **reachable at N=1**, the server's only mode. (4) reachable today.

**SETTLED trap:** this is **not** *"predict a smaller VRAM peak → bigger expert cache"* (P5, REFUTED — the flash
reserve is floor-bound at 3.5 GiB and did not move on any geometry or kernel). This changes the **requested context**,
which moves the reservation itself rather than the prediction of it.

---

# 32 · [!] Run a lower-bpw EXL3 build of the models already on disk — a download, zero code (added 2026-09-12) — **RE-ANCHORED 2026-09-21: the table below is Qwen3-235B's and that checkpoint is GONE** — **BLOCKED 2026-09-23: the user declined the download. Row 32's ENTIRE cost is that download; there is nothing else in it to build, so it cannot proceed.**

**Unblocks on: the user authorising an ~81.6 GiB pull.** Route 1 was VERIFIED LIVE 2026-09-23 and is better than the handoff knew — a full HuggingFace sweep found exactly ONE EXL3 build of M2.7 (`NeuroSenko/MiniMax-M2.7-exl3`, already on disk), **but it ships bpw as BRANCHES: `2.0 / 3.0 / 4.0 / 5.0 / 6.0 / 7.0 / 8.0`**, all `quant_method: exl3`, `version 0.0.29`, `head_bits: 6`, `codebook: mcg`. Same repo, same module-list forward, same loader. **Route 2 (local quantize) is therefore DEAD — no original weights are needed.** Measured branch sizes from the tree API: **3.0bpw = 87.6 GB (81.6 GiB), 4.0bpw = 116.1 GB (108 GiB), 5.0bpw = 144.5 GB (134.6 GiB — matching the 135 GiB on disk byte-for-byte)**. **Two handoff corrections banked 2026-09-23:** (a) the disk budget is a PEAK not a steady state — the on-disk 5.0bpw dir holds ONLY `splitted_model/`, no original safetensors (`tests/test_delete_original.py`), so a 3.0bpw pull costs **163 GiB peak, 81.6 GiB steady** against **232 GiB free**, with the 5.0bpw build surviving for the A/B; (b) the handoff's *"there is no committed MoE perplexity baseline"* is **STALE** — row 33 (`todo.md` §33) records MiniMax **ppl 7.2626** at k=8, 256 scored tokens, via `scripts/decode_top_k_ppl.py`, alongside decode **1.8168 tok/s**. That is ONE draw, so D6's several-draws rule still binds, but the anchor and the tooling already exist.

**READ THE HANDOFF FIRST: [`…-row32-lower-bpw-HANDOFF.md`](docs/2026-09-21-row32-lower-bpw-HANDOFF.md)
(gitignored — the corrected numbers are repeated here so this row survives without it).** Two things
invalidated the arithmetic below. (1) **Qwen3-235B, the model it is computed on, was RETIRED AND DELETED
2026-09-20.** (2) **The row's own kill clause fired for the other candidate:** it says *"only pays while
experts stream — if row 30's S2 lands, this row mostly dies"*, and S2 landed 2026-09-13, after which
Qwen3-Next's drive is **0.42 % of a decode token** (RAM hit 96.6 %, 0.030 GB/token). **The one live target
is MiniMax-M2.7**, which is EXL3 **5.0 bpw** (head 6), 135 GB, and whose drive is busy **75.7-76.4 % of a
decode token** at 6.03-6.05 GB/s (row 49, 2026-09-21) against a **~1.82 tok/s** baseline. Re-derived there:
**3.0 bpw → 0.382 s/token ≈ 2.62 tok/s = ×1.44**; 4.0 bpw → ×1.18. **PROJECTION, not a measurement.**
**And the trap:** row 49 refuted a **RATE** lever on this same model (−66.7 % reads and +6.9 % delivered
bandwidth bought **−0.40 %** decode). This is a **BYTE** lever, which is the axis rows 26/28 found
decisive — different thing — but the token did NOT collapse onto the disk-busy line there, so price this
from an A/B, not from the table. Disk: 231 GB free, and a 3.0-bpw build needs ~81 GB plus an equal
`splitted_model/` copy. **Quality gate unchanged and mandatory — and there is still no committed MoE ppl
baseline, so establish one on the CURRENT MiniMax build first** (and take several draws: one decode-ppl
draw is ONE draw, D6).

**The original Qwen3-235B text follows, kept because the method is still right.**


Qwen3-235B is here at **5.01 bpw** (11.8 MB/expert). `bytes_read ∝ bpw`, so by the primer's formula, holding hit rate
constant (**conservative** — smaller experts mean more of them fit the same cache):

| expert bpw | expert_bytes | bytes/token | `device-busy` | T | vs 0.930 |
|---|---|---|---|---|---|
| 5.01 (today) | 11.8 MB | 4.84 GB | 0.772 s | 0.930 s | — |
| 4.0 | 9.4 MB | 3.86 GB | 0.616 s | 0.775 s | ×1.20 |
| **3.0** | **7.1 MB** | **2.90 GB** | **0.463 s** | **0.622 s** | **×1.50** |
| 2.5 | 5.9 MB | 2.42 GB | 0.386 s | 0.545 s | ×1.71 |

**Below ~3.25 bpw a step change fires that the table does NOT include.** The 62/94-layer VRAM-cache restriction lifts
when `top_k × moe_layers × expert_bytes < vram_cache_budget`, i.e. for Qwen when `expert_bytes < 5.9 GB / (8 × 94) =
7.85 MB`. A 3.0 bpw build should therefore land **better** than ×1.50.

**Two routes, cheap one first.** (1) **Download an existing lower-bpw EXL3 build** — same architecture, same
fused-EXL3 forward, same loader, nothing to port. 299 GB free; budget 2× the download for Vaayu's own
`splitted_model/` copy. (2) **Quantize locally** with the repo's `quantize.py` (a wrapper over
`exllamav3.conversion.convert_model`) — **check the prerequisite first: it converts from the ORIGINAL weights, which
are not on this disk** (~470 GB download for a 235B), so route 1 is strictly better unless no published build exists.

**The optimal shape, and only route 2 reaches it:** EXL3 supports per-module bpw. Experts are ~95 % of the bytes and
the least sensitive per byte. **Keep attention / router / `lm_head` high, push only the experts down.** Say this in the
plan even on route 1 — it is why a naive "3.0 bpw everything" build may lose more quality than it needs to.

**Quality gate (mandatory; "lossy is allowed" does not remove it):** teacher-forced perplexity, old build vs new, same
held-out text, plus `tests/test_moe_accumulator_determinism.py` and the server gate. **There is no committed MoE ppl
baseline — establish one on the current build first** or the comparison has no anchor. (The dense number on record is
8.78 on Llama-3.3-70B.)

**Stop rule: under ×1.25 through the server, or a perplexity regression the user will not accept at that speed → keep
the 5.01 build and close.**

**§8:** (1) no knob — `MODEL_PATH`. (2) nothing to plumb. (3) reachable at N=1. (4) reachable when the download lands.

**SETTLED trap:** this is **weight** precision, not KV quantization (Plans E/H/I — KV is 0.193 MB/token and irrelevant
to `device-busy`). It runs the same direction as rows 26/28's rule (*bytes per token decides speed*; the fp8 sibling of
a model cost ~43 % of its fp4 decode). **Only pays while experts stream — if row 30's S2 lands, this row mostly dies.**

---

# 33 · [x] Truncate `top_k` at decode — CLOSED 2026-09-13: BUILT, MEASURED ON BOTH MoE MODELS, SHIPPED OFF

## VERDICT: the knob works and pays well on Qwen3-Next; the safe k is per-MODEL, so the default is `None`.

**Qwen3-Next-80B int4 (top-10 of 512), N=1, server knobs, same-day same-box arms:**

| k | tok/s (mean U1,U2) | vs off | ppl (256 scored) | vs config | speed per unit ppl |
|---|---|---|---|---|---|
| 10 (off) | 6.4491 | — | 8.5824 | — | — |
| **9** | **7.5452** | **+17.0 %** | **8.7089** | **+1.47 %** | **11.6×** |
| **8** | **8.3493** | **+29.5 %** | **8.8765** | **+3.43 %** | **8.6×** |
| 6 | 10.7887 | +67.3 % | 9.3523 | +8.97 % | 7.5× |

**MiniMax-M2.7 (top-8 of 256): NO viable k.** k=7, its mildest possible step, already costs
**+6.34 %** ppl; k=6 costs **+19.30 %** for only **+26.0 %** tok/s (1.35× speed per unit ppl, vs
7.5-11.6× on Qwen3-Next). Baselines 1.8168 → 2.2901 tok/s, ppl 7.2626 → 8.6645.

**S3-pre projected +14.7 % (k=8) and +34.2 % (k=6); measured +29.5 % and +67.3 % — DOUBLE, on
both.** S3-pre derived those holding the VRAM cache hit rate constant and called that
conservative; it was. Hit rate RISES with the clamp (**83 → 84 → 85 → 90 %** at k=10/9/8/6) because
a smaller working set is covered better by the same 7.60 GB, so each removed slot also makes the
remaining ones cheaper. On MiniMax, which is disk-bound, the old `bytes ∝ top_k` projection was
right (×1.21 projected, ×1.260 measured) — the compounding is a residency-regime effect.

**Why `None` and not 9:** the safe k is a property of the MODEL and this is one global knob. It is
also NOT predictable from geometry — a 10 % slot cut costs Qwen3-Next 1.47 % while a 12.5 % cut
costs MiniMax 6.34 %, 4× more for a comparable fraction — so no `auto` rule can be derived from
anything on this box. `vaayu/server/README.md` carries the per-model numbers and tells the user to
measure their own checkpoint with `scripts/decode_top_k_ppl.py`.

**Quality method, and its limit.** Decode-mode teacher-forced ppl, 48-token prefill then 256
single-token scored steps (a full-sequence forward is a PREFILL pass and can never see this
feature). Baselines reproduced to the digit across independent processes. **A 64-token window
disagrees with the 256-token one by up to 19 pp — quote the 256 column.** And the human read is a
WEAK gate here: at k=4 Qwen3-Next still wrote a correct Dijkstra (+22.05 % ppl) while MiniMax
emitted degenerate repetition (+135.15 %). Perplexity leads visible damage by a wide margin on
Qwen3-Next; do not certify a k by reading one generation.

**Regression, knob off (bit-identical BY CONSTRUCTION — `truncate_top_k` returns the SAME tensor
objects when k >= K):** Qwen3-Next 6.4491 (vs S2b (iii)'s same-day 6.4702/6.4138), **Qwen3-235B
fused-EXL3 1.1457 (vs row 27's 1.1363)**, MiniMax 1.8168 — all three forwards, ids identical.
Suite **2090 → 2113 passed**, 15 skipped, 23 new tests.

**Stop rule could NOT be evaluated as written** — it compared k=6 against row 32's 3.0 bpw build,
and row 32 is on the T3 ladder, dead on this model. The bar actually used was a per-model ppl bar
pre-registered before the sweep: ≤2 % → recommend ON, 2-5 % → knob only, >5 % → reject.

Detail: [`…-row33-T2-decode-top-k-RESULTS.md`](docs/2026-09-13-row33-T2-decode-top-k-RESULTS.md).

**Follow-ups 2026-09-14 (user's calls):** (1) **an active clamp now WARNS at load** — `resolve_decode_top_k` prints
`[Vaayu-MoE] WARNING: DECODE_TOP_K=<k> routes fewer experts than the checkpoint's <K>. Output is DEGRADED by design`,
because this is the only server knob whose wrong value fails silently; inert values (None, k == K) stay quiet. 3 new
tests. (2) **Shared-expert hypothesis under test:** the tolerant model (Qwen3-Next) is the only one with a shared expert
(`shared_expert_intermediate_size: 512`), which runs on every token and is never truncated; MiniMax and Qwen3-235B have
none. Detectable via `_shared_expert_output` (`moe_patch.py:475`), so if it holds it is the only candidate basis for a
`DECODE_TOP_K="auto"`. **REFUTED 2026-09-14 by the third point:** Qwen3-235B (no shared expert) costs **+3.40 % at k=7, +5.33 % at k=6** —
prediction was > 5 % at k=7. Matched against MiniMax (same top-8, same cut fractions, also no shared expert: +6.34 % /
+19.30 %) it differs 1.9× / 3.6×, so the shared expert does not explain the gap; the two softmax Qwen models sit on
roughly one curve and **MiniMax is the outlier.** What MiniMax alone has: **tensor truncation is NOT a true clamp there**
(biased `sorted=False` selection vs unbiased weights) — a choice this row made. So part of MiniMax's result may be an
artifact of the design, not the model. **No `auto` signal survives three models; `None` stays.** Qwen3-235B throughput
not measured (projected ~+21 % at k=6 by the byte model, which was accurate on disk-bound MiniMax). k=8 baseline 7.7818
reproduced across a reboot.

- [x] **CLOSED 2026-09-15 (D5): REFUTED — MiniMax is intolerant; tensor truncation stays.** Device, one process,
  256 scored steps (`docs/probes/d5-probe.{log,json}`): trunc k=8 **7.262630** / k=7 **7.723136 (+6.34 %)** / k=6
  **8.664506 (+19.30 %)** — all three reproduce the references; true clamp k=7 **7.858454 (+8.20 %)**, k=6 8.457805
  (+16.46 %). Adopt rule (clamp k=7 ≤ ~4.9 %) FAILS. Root cause: the checkpoint's routing bias is zero, so the router's
  own top-k IS the k largest weights — measured kept-set disagreement **0.00 % (k=7) / 0.66 % (k=6)** of 15,872 rows;
  H2 (bf16 weight ties) refuted too — boundary gap 7.6-8.4 % vs bf16's 0.8 % step. The ±2-3 pp clamp-vs-trunc gap
  flips sign (k=7 +1.86, k=6 −2.84) → order/rounding noise (→ D6). No toggle written (probe monkeypatch), no
  `METHOD_VERSION` bump. §8: no knob, nothing plumbed or adopted. **Scope:** zero-bias checkpoint only — the routing-bias
  `[bug]` row re-opens the question if the upstream bias is wired. RESULTS:
  `docs/2026-09-15-D5-minimax-true-clamp-RESULTS.md`.
  Was: **Separate MiniMax's intolerance from the truncation design.** Build a TRUE clamp for contract-routed owners only
  (set `moe_module.top_k` on decode passes so `route_tokens_to_experts` selects k by its own biased scores), re-measure
  MiniMax ppl at k=7 and k=6 with `scripts/decode_top_k_ppl.py`. **Stop rule:** closes ≥ half the gap to Qwen3-235B
  (k=7 ≤ ~4.9 %) → contract owners switch to the attribute clamp, softmax/router owners keep tensor truncation;
  otherwise MiniMax is intolerant and the design stands, close this. ~½ day + ~15 min device. Reachable at N=1 (§8), no
  new knob. NOT worth doing unless someone wants DECODE_TOP_K on MiniMax — the default is `None` either way.
  **Relevance changed 2026-09-14:** `auto` now picks the cutoff p=0.90 on MiniMax (+8.4 % tok/s, worst +2.25 %); a true clamp that
  lets k=7 fit the budget could beat it. **Brief (D5, self-contained):** [`docs/issue-D5-minimax-true-clamp-experiment.md`](docs/issue-D5-minimax-true-clamp-experiment.md) (gitignored — this row is the durable summary).
  - history: **2026-09-15, Phase 1 (CPU, no code changed) — the brief's premise is FALSE, but the experiment survives
    for a different reason.** (a) **MiniMax's `e_score_correction_bias` is ZERO at runtime:** no tensor by that name
    anywhere (index; `splitted_model/` 65 files, 191,772 keys, 0 hits); a CPU build with Vaayu's own
    `init_empty_weights` + `from_config` + compat patches, then the `vaayu_base.py:3291` buffer move, gives 62 real
    buffers, nonzero count 0. So selection is `topk(sigmoid)`, monotone in the weights — the biased-selection gap the
    brief describes does not exist on this checkpoint. (b) **But bf16 breaks the equivalence:** synthetic 256-expert
    top-8 routing, zero bias — float32 weights give truncation == true clamp (0/20,000 rows differ; ≤ 0.02 % at logit
    scale 4); **bf16 weights differ on 7-10 % of rows at logit scale ≤ 1, 33 % at 2, 99.9 % at 4** (bfloat16 has 7
    mantissa bits, so near-equal top-8 weights tie and truncation drops an arbitrary tied slot). Positive control
    (bias std 0.05) 73-95 %. MiniMax's router returns bf16 weights (D1); Qwen3-235B's softmax fallback and
    Qwen3-Next's router return float32. **H2: MiniMax's outlier k=7 cost is partly bf16 rank loss at the drop
    boundary.** Next: one device probe (scratchpad `d5_probe.py`, monkeypatch only, no engine edit) — trunc k=8/7,
    clamp k=7/6, trunc k=6 in one process, with real set-disagreement and bf16 boundary-tie rates. Stop rule unchanged.
- [x] **`DECODE_TOP_K="auto"` — opt-in automatic k. DESIGN IN PROGRESS 2026-09-14.** The user's concern: on `None`, a
  user running a tolerant model silently loses speed (Qwen3-Next: +17 % for 1.47 %) and has no way to know. **User
  decisions:** (1) **opt-in** — default stays `None` (lossless), keeping `documentation/TUNING.md`'s "off or conservative
  by default" promise; (2) quality budget **3.5 %** perplexity; (3) measured **at first load**, cached per checkpoint
  (under the split dir, `self.checkpoint_path`), `/health` to report a calibrating state meanwhile — which needs the
  lifespan changed, since `/health` is only built once the engine is bound. On today's data a 3.5 % budget gives
  Qwen3-Next k=8 (+29.5 %), Qwen3-235B k=7 (3.40 %, throughput unmeasured), MiniMax 8. **Step 1, approved: a throwaway
  spike** — a per-token routing-mass cutoff (keep the smallest set of experts carrying ≥ p of the routing weight) vs
  fixed k, on Qwen3-Next + MiniMax, ppl vs mean experts/token. **If ONE p is safe (≤ 3.5 %) on both with a real slot
  cut, auto needs no calibration; otherwise auto = per-checkpoint fixed-k calibration.** Spec to
  `docs/superpowers/specs/` after the spike reports, then a plan.
  **SPIKE REPORTED 2026-09-14** ([`…-row33-auto-topp-SPIKE.md`](docs/2026-09-14-row33-auto-topp-SPIKE.md)): a per-token
  routing-mass cutoff at **p=0.90 is within 3.5 % on BOTH** — Qwen3-Next +2.95 % at a 13.4 % slot cut, MiniMax **+1.76 %
  at 11.2 %** (fixed k=7 cost MiniMax +6.34 % for 12.5 %: ~3.6× less damage). **Not a blind default:** MiniMax's cliff
  sits between p=0.90 (+1.76 %) and 0.80 (+17.49 %), so an unmeasured model needs a check — but 2 ppl arms, not a sweep.
  **Neither mechanism dominates:** in budget, Qwen3-Next is best with fixed k=8 (20 % cut), MiniMax only with the cutoff.
  **Speed of the cutoff UNMEASURED** (slot proxy only; fixed k got 2× its proxy from cache compounding, a per-token set
  may not). **SPEED CHECK DONE same day:** Qwen3-Next off 6.9869 → fixed k=8 **8.5383 (+22.2 %)** → cutoff p=0.90
  **7.8861 (+12.9 %)**; MiniMax off 1.8204 → cutoff p=0.90 **2.0161 (+10.75 %)**. The cutoff converts ~0.85× of its
  slot cut into tok/s (fixed k ~1.11×) and keeps the cache-hit compounding. **Split verdict: fixed k is best on
  Qwen3-Next, only the cutoff helps MiniMax.** Next: user picks the auto design.
  **DESIGN APPROVED 2026-09-14 (all 6 sections).** Spec: [`docs/superpowers/specs/2026-09-14-decode-top-k-auto-design.md`](docs/superpowers/specs/2026-09-14-decode-top-k-auto-design.md)
  (gitignored — this is the durable summary). **Mechanisms:** both — fixed k and the routing-mass cutoff. **Pick rule:
  fixed k first** — walk k = K−1, K−2, … while cost ≤ budget; only if none fits, walk p = 0.95, 0.90, … 0.50 (step 0.05)
  at k=K; nothing fits → no clamp. **Measure:** built-in original code text (`CALIB_TEXT`, ≥ 305 tokens), prefill 48 +
  256 teacher-forced decode steps, cost = ppl/baseline − 1, deterministic. **When:** end of `__init__` (after prefix/KV
  setup, before the post-load RAM ledger), blocking, server unreachable meanwhile. **Cache:**
  `<checkpoint_path>/decode_top_k_auto.json`, key = (method_version, text_version, budget, sha256 of config JSON +
  split `*.safetensors` names/sizes), atomic write. **No reload/resize:** caches are RAM-bound, not k-bound. **Cutoff
  mechanism:** dropped slots re-pointed at the row's top-1 expert (by weight) with weight 0 — shapes unchanged, the
  measured +12.9 % / +10.75 % used exactly this. **Engine:** `truncate_top_k(w, i, k, p=None)`, `_decode_limits` → (k,p),
  `_decode_top_p`, `decode_top_k_budget=0.035`, `decode_top_k_status()`. **Server:** `DECODE_TOP_K` None|int|"auto",
  `DECODE_TOP_K_BUDGET=0.035`, `/health` `decode_top_k` field, README + `documentation/TUNING.md`. **Errors never silent:**
  any measurement failure → WARNING, checkpoint's own k, no cache (retry next load); invalid values raise at load.
  **Acceptance:** cold pick ≤ 3.5 % on code text (Qwen3-Next, MiniMax, Qwen3-235B); warm load = cache hit, same pick,
  no trials; same-day throughput gain > 2× replicate spread; server gate with "auto"; knob None = no-op on all three
  forwards. **Out of scope:** changing the default, true dispatch skip, background measuring, N>1. **SPEC APPROVED by the
  user 2026-09-14.** Next: writing-plans → plan tasks added here as checkboxes → branch + one commit (user: hold the
  ledger commit until the plan lands).
  **PLAN WRITTEN 2026-09-14:** [`docs/superpowers/plans/2026-09-14-decode-top-k-auto.md`](docs/superpowers/plans/2026-09-14-decode-top-k-auto.md)
  (gitignored — the checklist below is the durable copy; the plan holds the exact code).
  - [x] **Task 1 — cutoff mode in `truncate_top_k`** (`vaayu/moe_patch.py`): `truncate_top_k(w, i, k, p=None)` keeps the
    smallest set of slots carrying ≥ p of the weight and re-points dropped slots at the row's top-1 expert (found by
    weight) with weight 0; `_decode_limits(ref) -> (k, p)` replaces `_decode_k_for` at the 3 forward call sites.
    - [x] failing tests: cutoff on sorted and `sorted=False` routing, top-1 always kept, identity when off; all three
      routing owners through real forwards with `p=0.01` compute exactly 1 expert, prefill untouched
    - [x] implement; `grep _decode_k_for` returns nothing
    - [x] `tests/test_decode_top_k.py` + `tests/test_moe_patch.py` green; commit
    - done 2026-09-14: commit `edbe4dd`, task review clean (2 minors deferred)
  - [x] **Task 2 — `vaayu/decode_top_k_auto.py` core:** constants (prefill 48, score 256, min 305 tokens, cutoff steps
    95→50), original code `CALIB_TEXT`, `run_ladders` (fixed k first), `fingerprint`, `cache_key`, `read_cache`,
    `write_cache` (atomic temp + `os.replace`).
    - [x] failing tests: picks for Qwen3-Next-, MiniMax- and Qwen3-235B-shaped costs, nothing-in-budget, cost == budget,
      k floor at 1; cache round trip / key-field miss / corrupt / failed write leaves nothing / fingerprint; text length
      in chars and in tokens on local tokenizers
    - [x] implement; green; commit
    - done 2026-09-14: commit `5d93f11`, task review clean (2 minors promoted to Task 3: damaged-cache guard, top-1 guard)
  - [x] **Task 3 — `decode_mode_ppl` + `resolve(model, budget)`:** cache hit → apply; miss → measure, write, apply;
    every failure loud and lossless (`source: "failed"`, no cache written).
    - [x] failing tests (uniform-logits LM; fake MoE model): cold, warm, budget change, cutoff pick, none,
      measurement raises, too-short text, unwritable cache
    - [x] implement; green; commit
    - done 2026-09-14: commit `ce9c770`, task review clean (+ damaged-cache and top-1 guards)
  - [x] **Task 4 — engine wiring** (`vaayu/vaayu_base.py`): `resolve_decode_top_k` accepts `"auto"` and rejects
    anything not None/int; `validate_decode_top_k_budget`; `_detect_moe` records status and notes;
    `_resolve_decode_top_k_auto()` is the LAST statement of `__init__`; `decode_top_k_status()`.
    - [x] failing tests: resolution, invalid values, budget bounds, status, explicit-int budget note, dense no-op,
      hook runs only for auto, hook is the last statement of `__init__`
    - [x] implement; green (no new `__init__` parameter yet, so §8 reachability unaffected); commit
    - done 2026-09-14: commits `fdd6199` + review fix `6ace30c` (review found the dense branch never validated DECODE_TOP_K — now `check_decode_top_k_value` runs on every checkpoint)
  - [x] **Task 5 — server, §8 in one commit:** `decode_top_k_budget` engine parameter (validated); `DECODE_TOP_K`
    comment + `DECODE_TOP_K_BUDGET = 0.035` in `config.py`; `ServerConfig` fields + plumbing in `app.py` and
    `__main__.py`; `/health` `decode_top_k` (hasattr fallback); `EngineHandle.decode_top_k_status()`; README rows.
    - [x] failing tests: reachability `SERVER_EXPOSED`, `/health` field and fallback, forwarder, parameter default,
      knob equals engine default
    - [x] implement; green; commit
    - done 2026-09-14: commit `51a7388`, task review clean; full suite 2181 passed, 15 skipped (2116 before the branch)
  - [x] **Task 6 — user docs:** `documentation/server.md` §3 rows for `DECODE_TOP_K`, `DECODE_TOP_K_BUDGET`,
    `KEEP_AWAKE` (with the server-measured +23.1 %, not the README's stale "pending"), `PIN_ENGINE_THREADS`;
    `documentation/TUNING.md` §5c.
    - [x] failing test `tests/test_docs_speed_knobs.py`; implement; green; commit
    - done 2026-09-14: commit `ea0cdc2`, task review clean
  - [x] **Task 7 — on-device acceptance.** The probe `plans/probes/row26_decode.py` is gitignored and hard-codes the
    project root on `sys.path`, so it gains a `VAAYU_ROOT` override, env-gated `DECODE_TOP_K` kwargs and a
    `vaayu imported from` proof line; the smoke gate gains `SERVER_DECODE_TOP_K`.
    - [x] full CPU suite green
    - [x] pre-feature worktree at the commit before Task 1; every base log must name `/tmp/vaayu-base`
    - [x] Qwen3-Next base / cold auto / warm auto: pick ≤ 3.5 %; warm = cache hit, no trials; gain > 2× replicate spread
    - [x] MiniMax base / cold auto: pick ≤ 3.5 %; gain > 2× replicate spread
    - [x] Qwen3-235B cold auto pick ≤ 3.5 %; knob-None no-op vs base on all three forwards (tok/s within spread, ids identical)
    - [x] server gate on Qwen3-Next with `SERVER_DECODE_TOP_K=auto`; `/health` shows the cached pick
    - [x] remove worktree; RESULTS doc; close this bullet with numbers and the §8 answers; close the server.md `[bug]`; commit
  - **Method-1 acceptance run 2026-09-14 (code calibration text only) — SUPERSEDED:** full CPU suite
    2186 passed, 15 skipped. Baseline worktree `/tmp/vaayu-base` at `1955fbe` (parent of `edbe4dd`),
    every base log names `/tmp/vaayu-base/vaayu/__init__.py` — valid. **Picks (code text only,
    3.5 % budget): Qwen3-Next k=6 (cost +2.26 %, ladder k=9 −0.81 %/k=8 +0.55 %/k=7 +1.53 %/k=6
    +2.26 %/k=5 +3.76 % OVER); MiniMax k=6 (cost +1.19 %, ladder k=7 +1.87 %/k=6 +1.19 %/k=5 +5.11 %
    OVER — fixed k stayed in budget, no cutoff ladder ran); Qwen3-235B k=6 (cost +2.37 %, ladder k=7
    +0.93 %/k=6 +2.37 %/k=5 +5.27 % OVER).** None match the design-doc prose-passage guesses
    (Qwen3-Next k=8, MiniMax p=0.90, Qwen3-235B k=7) — recorded as measured. First-load measurement
    times: Qwen3-Next 261.9s, MiniMax 578.1s, Qwen3-235B 975.4s (all = trial-second sum + baseline
    load, confirmed against the probe's own `loaded in` line). Qwen3-Next warm load: cache hit, same
    pick k=6, `source: cache`, no trial lines. **Throughput (U1/U2 tok/s, mean/spread): Qwen3-Next
    base 6.65345/0.0049 → auto(k=6) 10.2006/0.0324, gain 3.54715 > 2×0.0324=0.0648 PASS, ids
    identical; MiniMax base 1.80265/0.0021 → auto(k=6) 2.2914/0.0018, gain 0.48875 > 2×0.0021=0.0042
    PASS, ids identical.** Knob-`None` no-op: Qwen3-Next off vs base 6.57815 vs 6.65345, diff
    0.0753 ≤ 2×0.2867=0.5734 PASS, ids identical; MiniMax off vs base 1.79485 vs 1.80265, diff
    0.0078 ≤ 2×0.0131=0.0262 PASS, ids identical; **Qwen3-235B off vs base (single-U1 samples, no
    spread) 1.1137 vs 1.1537 tok/s, diff 3.47-3.59 % — FAILS the brief's ≤2 % bound (ids ARE
    identical); a stop-rule trigger, not re-run** — likely single-sample GPU noise on the largest
    model (no U2 replicate to gauge spread, unlike the two models above). Server gate:
    `SERVER_DECODE_TOP_K=auto` on Qwen3-Next, 9 passed, `/health.decode_top_k` =
    `{mode: k, value: 6, cost: 0.0226, source: cache}` (warm from Step 4's cache) — PASS. §8: (1)
    knob — two, `decode_top_k` + `decode_top_k_budget`; (2) plumbed — `vaayu/server/app.py` +
    `vaayu/server/__main__.py`; (3) reachable at N=1 — yes; (4) reachable now — yes, first load of
    each checkpoint is slower (up to 975.4s measured) and the server is unreachable while that
    one-time measurement runs; every later load of the same checkpoint+budget is a cache hit.
    **User decision 2026-09-14: the code text priced truncation ~4x too cheaply (Qwen3-Next top-6 =
    +2.26 % on code but +8.97 % on the row-33 prose passage), so auto will score code AND prose and
    keep the worse cost (plan Tasks 8-9); these picks are not the shipped behaviour.** Full logs:
    `docs/probes/row33-auto-*.log`; full RESULTS (gitignored, labeled "method 1 (code text only),
    superseded"): `docs/2026-09-14-row33-decode-top-k-auto-RESULTS.md`.
    - done 2026-09-14: commit `4471ba1` — method-1 evidence (auto picks superseded by Tasks 8-9); Qwen3-235B knob-off −3.5 % on one sample routed to Task 9's replicated re-check
  - [x] **Task 8 — measure BOTH calibration texts, keep the safer pick (user decision 2026-09-14).** Task 7's first
    on-device arm showed the code text prices truncation ~4× too cheaply: Qwen3-Next baseline ppl **1.90** on it vs 8.58
    on the row-33 prose passage; k=9 measured **−0.81 %** (noise ~±1 % against a 3.5 % budget); the pick **k=6** (+2.26 %
    on code) costs **+8.97 % on prose** — 2.6× the budget. Fix: one ladder whose trial cost is the WORSE of code and
    prose (row-33 passage copied verbatim as `CALIB_PROSE`); `METHOD_VERSION` and `CALIB_TEXT_VERSION` → 2 so method-1
    caches re-measure. On today's numbers Qwen3-Next picks **k=8** (+0.55 % code, +3.43 % prose). First load ~2×
    (one-time, cached). Task 7's `auto` picks are recorded as superseded method-1 evidence.
    - [x] failing tests: worst-text ladder picks k=8 on Qwen3-Next's measured code+prose costs; per-text costs recorded;
      resolve measures both texts; a too-short prose text warns naming it; `CALIB_PROSE` equals the row-33 passage and
      tokenizes to ≥ 305 tokens
    - [x] implement (+ TUNING.md wording); green; review; commit
    - done 2026-09-14: commit `28df444`, task review clean; full suite 2191 passed, 15 skipped
  - [x] **Task 9 — re-run on-device acceptance with method 2:** delete caches; Qwen3-Next cold + warm, MiniMax cold,
    Qwen3-235B cold; throughput vs same-day base arms (re-run the base arms if not the same day as Task 7); server gate
    with `"auto"`; RESULTS doc; close row 33's `auto` bullet
  - **CLOSED 2026-09-14 (method 2):** Step 2b's first (back-to-back) design FAILED: worktree `/tmp/vaayu-base` at
    `1955fbe`; `q235-base3` U1 **1.1575**/U2 **1.1641** tok/s (mean 1.1608, spread 0.0066), `q235-off3` U1
    **1.1182**/U2 **1.1260** (mean 1.1221, spread 0.0078); `|1.1221−1.1608|=0.0387` vs bound
    `max(0.02×1.1608, 2×max(0.0078,0.0066))=0.023216` → **FAIL** (−3.33 %), ids identical to U1 in both. Controller
    ruling: this design was order-confounded (base always ran immediately before off, on the most disk-bound model —
    back-to-back sustained load throttles the second arm) — re-checked with a counterbalanced **ABBA** order
    (off4, base4, base5, off5, same worktree/tree pattern): `q235-off4` U1 1.1560/U2 1.1550 (mean 1.1555, spread
    0.0010); `q235-base4` U1 1.1285/U2 1.1377 (mean 1.1331, spread 0.0092); `q235-base5` U1 1.1360/U2 1.1250 (mean
    1.1305, spread 0.0110); `q235-off5` U1 1.1368/U2 1.1400 (mean 1.1384, spread 0.0032); all four import the
    expected tree, ids identical to U1 in all four. Code effect `(mean_off4+mean_off5)/2 − (mean_base4+mean_base5)/2
    = 1.1470−1.1318 = +0.0151` vs bound `max(0.02×1.1318, 2×0.0110) = 0.0226` → **PASS** (+1.34 %, i.e. off is
    actually *faster*); order effect (positions 3-4 minus 1-2) `= −0.0098` — confirms the original failure was
    positional drift, not a `DECODE_TOP_K=None` code-path regression. Resumed the brief from Step 3. **Picks (method
    2, code+prose worst cost, 3.5 % budget): Qwen3-Next k=8 (code +0.55 %, prose +3.43 %, worst 3.43 %; ladder k=9
    worst +1.47 % in budget / k=8 worst +3.43 % in budget / k=7 worst +4.93 % OVER); MiniMax p=0.90 (code +2.25 %,
    prose +1.76 %, worst 2.25 %; ladder k=7 worst +6.34 % OVER / p=0.95 worst +0.13 % in budget / p=0.90 worst
    +2.25 % in budget / p=0.85 worst +11.45 % OVER — the walk keeps stepping to the most aggressive cut still in
    budget); Qwen3-235B k=7 (code +0.93 %, prose +3.40 %, worst 3.40 %; ladder k=7 worst +3.40 % in budget / k=6
    worst +5.33 % OVER).** None match the design-doc prose guesses exactly (predicted Qwen3-Next k=8 ✓, Qwen3-235B
    k=7 ✓, MiniMax p≈0.90 ✓) — recorded as measured, not chased. First-load measurement times (probe's own
    `loaded in` line, ~2× method 1's since two texts are scored): Qwen3-Next 396.6s, MiniMax 1584.0s, Qwen3-235B
    1555.9s. Qwen3-Next warm load: cache hit, same pick k=8/cost 0.034272, `source: cache`, no trial lines.
    **Throughput (U1/U2 tok/s, mean/spread vs Task 7's same-day base logs): Qwen3-Next base 6.65345/0.0049 →
    auto(k=8) 8.18865/0.0261, diff 1.5352 > 2×0.0261=0.0522 PASS, ids identical; MiniMax base 1.80265/0.0021 →
    auto(p=0.90) 1.95425/0.0027, diff 0.1516 > 2×0.0027=0.0054 PASS, ids identical to U1 for U1/U2 (W≠U1 is the
    documented pre-existing MiniMax warmup quirk, not a failure); Qwen3-235B (U1 only, no inequality per the brief)
    base 1.1537 → auto(k=7) 1.2823, +0.1286 tok/s (+11.15 %), ids identical to U1 (W≠U1 here too — new on this
    model under the clamp, method 1's q235-cold had W=U1; not required to match, noted as a minor observation).**
    Server gate: `SERVER_DECODE_TOP_K=auto` on Qwen3-Next, 9 passed, `/health.decode_top_k` =
    `{mode: k, value: 8, cost: 0.034272, source: cache}` (warm from Step 3's cache) — PASS. §8: (1) knob — two,
    `decode_top_k` + `decode_top_k_budget`; (2) plumbed — `vaayu/server/app.py` + `vaayu/server/__main__.py`; (3)
    reachable at N=1 — yes; (4) reachable now — yes, the first load of each checkpoint is slower (~2× method 1's
    times since two calibration texts are scored: up to 1584.0s measured here) and no endpoint answers while that
    one-time measurement runs; every later load of the same checkpoint+budget is a cache hit. Full logs:
    `docs/probes/row33-auto-q235-{base,off}{3,4,5}.log`, `docs/probes/row33-auto-{q3n-cold2,q3n-warm2,mm-cold2,
    q235-cold2,servergate2}.log`; full RESULTS: `docs/2026-09-14-row33-decode-top-k-auto-RESULTS.md` ("Method 2"
    section).
    - done 2026-09-14: commits `0c78fe7` + `1a225a1`, task review clean; picks Qwen3-Next k=8, MiniMax p=0.90, Qwen3-235B k=7 (all ≤ 3.5 % worst of code/prose); ABBA no-op PASS
**Row-33 follow-ups — suggested order (added 2026-09-14).** Six items below, each with a self-contained brief in
`docs/issue-*.md` (gitignored; the row is the durable summary). This orders them **against each other only** — not against
other open rows, whose priority is the user's call. Rule: cheap and no-GPU first, correctness before performance,
research last.

| # | item | why here | effort | GPU |
|---|---|---|---|---|
| ~~1~~ | ~~**K1**~~ **DONE 2026-09-15** README `KEEP_AWAKE` row still says the server A/B is "pending" | docs-only, removes a false claim users read | ~5 min | no |
| ~~2~~ | ~~**D4**~~ **DONE 2026-09-15** invalid `DECODE_TOP_K` rejected only after the layer split | one line + one test; a typo can cost a full split (~136 GB on Qwen3-235B) | ~15 min | no |
| ~~3~~ | ~~**D3**~~ **DONE 2026-09-15** calibration routes stay in the expert-usage histogram | CPU test only; edits `_resolve_decode_top_k_auto`, which D2 also edits — land D3 first, then D2 on top | ~30 min | no |
| ~~4~~ | ~~**D1**~~ **DONE 2026-09-15** truncation forces router weights to sum 1 | **correctness** for scaling-router families (DeepSeek-V2/V3, GLM-4-MoE, …) even with an explicit int; latent on this box; bumps `METHOD_VERSION` — do it before D2/D5 so their measurements run on the final semantics | ~½ day + ~15 min device | yes |
| ~~5~~ | ~~**D2**~~ **DONE 2026-09-15 — kept, Qwen3-235B first load −15.1 %** calibration runs without keep-awake / thread pinning | performance, unmeasured; needs an ~2 h ABBA first-load A/B on Qwen3-235B; stop rule < 5 % saving → close | ~30 min + ~2 h device | yes |
| ~~6~~ | ~~**D5**~~ **DONE 2026-09-15 — REFUTED, truncation design stands** MiniMax true-clamp experiment | research; only pays if a true clamp lets k=7 beat `auto`'s cutoff p=0.90 on MiniMax; would bump `METHOD_VERSION` again if adopted | ~½ day + ~20 min device | yes |

- [x] **CLOSED 2026-09-15 (D1):** `truncate_top_k` (`vaayu/moe_patch.py`) rescales kept weights to each row's
  ORIGINAL sum on both the `k` and `p` paths, and a sum within 1e-2 of 1 (rounding: bf16 ≤ 2.44e-3, measured) snaps
  to exactly 1.0. So scaled routers (`routed_scaling_factor`: DeepSeek-V2/V3, GLM-4-MoE, …) are no longer shrunk by
  `1 / original_sum`, and unit-sum routers (every model on this box) stay BIT-IDENTICAL. `METHOD_VERSION` 2 → 3.
  Tests: 11 scaled-router cases plus 4 bit-identity cases in `tests/test_decode_top_k.py`, each RED before its step and
  GREEN after; full suite **2228 passed / 15 skipped / 0 failed**. **Device (256-token decode ppl, same day):**
  - The snap reproduces an unmodified HEAD worktree to 6 decimals on all five arms (Qwen3-Next k=10/9/8 8.582392 /
    8.708857 / 8.876525; MiniMax k=8/7 7.262630 / 7.723136).
  - **Round 1, without the snap, FAILED the ±0.0005 stop rule** (Qwen3-Next k=9 +0.66 %, k=8 −0.77 %; MiniMax k=7
    −0.62 %), although on Qwen3-Next's float32 router it changed weights by ≤ 1.8e-7.
  - **A null control (HEAD × (1+2^-23) on the kept weights) moved Qwen3-Next k=9 +0.40 % and k=8 −0.16 %.** This
    perplexity is bit-sensitive, so ±0.0005 is a bit-identity test (→ new row D6).

  §8: no knob; already plumbed; reachable at N=1; correctness-only on this box. **Cost:** the version bump makes the 3
  cached `auto` picks on disk re-measure once on their next `auto` load (~400 s / 1556 s / 1584 s). With the snap they
  should land on the same picks, since the arithmetic is bit-identical. Server default `None`. RESULTS:
  `docs/2026-09-15-D1-truncation-keeps-row-sum-RESULTS.md`. Semantics note: for DeepSeek-V3 / GLM-4-MoE (normalise,
  then scale) the kept sum is the model's own value at the smaller k; for DeepSeek-V2 / `norm_topk_prob: false` it is
  a magnitude-preserving choice, and per-family detection was not built.
  Was: [bug] **DECODE_TOP_K truncation renormalises routing weights to sum 1 — wrong for scaling routers** (final review M3,
  2026-09-14; **scope widened while writing the brief:** the fixed-k int path has it too, since row 33 — on the 12
  transformers families that multiply by `routed_scaling_factor` (DeepSeek-V2/V3, GLM-4-MoE, dots1, Mistral4, …) an
  explicit `DECODE_TOP_K=<int>` silently shrinks every MoE block's output). `truncate_top_k`'s
  cutoff path (`vaayu/moe_patch.py`) rescales kept weights to sum 1 always. For contract-routed models whose top-k weights do
  NOT sum to 1 (`norm_topk_prob=False`, or a scaling factor inside `route_tokens_to_experts`), the very first cutoff trial
  already includes a scale change, so `auto` measures it as damage and lands on `none` — lossless, but `auto` silently cannot
  help that family. Spec §7 says "renormalise", and the fixed-k path did the same before this branch, so the fix is a spec
  change: scale kept weights to preserve the ORIGINAL row sum, on both paths. On-device numbers are unaffected (MiniMax
  p=0.95 cost +0.13 %). Needs a ppl re-check of the fixed-k path if it changes.
  **Brief (D1, self-contained):** [`docs/issue-D1-cutoff-renormalises-non-unit-sum-routers.md`](docs/issue-D1-cutoff-renormalises-non-unit-sum-routers.md) (gitignored — this row is the durable summary). **Latent on this box:** all three on-device models return unit-sum top-k weights (both Qwen3 configs
  `norm_topk_prob: true`; MiniMax divides by the sum, `modeling_minimax_m2.py:117`), so it needs a synthetic test to reproduce.
  - history: **2026-09-15, CPU phase (round 1).** `truncate_top_k` now rescales kept weights to each
    row's ORIGINAL sum on both the `k` and `p` paths (`row_sum` taken once, before either); `METHOD_VERSION` 2 → 3; spec
    §7 amended. Confirmed in transformers source: DeepSeek-V3 / GLM-4-MoE normalise then `× routed_scaling_factor`, so the
    kept sum equals what the model computes at the smaller k (DeepSeek-V2 does not renormalise — there sum-preserving is
    a magnitude choice, not the model's own semantics). New tests in `tests/test_decode_top_k.py`: 11 RED before
    (scaled row sum, k/p/k+p × 3 geometries; drop-nothing cutoff at scale 2.5; module-list forward with a scaled
    contract router at p=0.999), GREEN after; 2 unit-sum guards green both sides. Three files 231 passed; full
    `pytest tests/` **2224 passed / 15 skipped / 0 failed, 172 s** (GPU visible — with `CUDA_VISIBLE_DEVICES=""` three
    GPU-assuming tests fail, and an 8 G scope OOM-kills the run at ~64 %). **Watch on device:** MiniMax's router returns
    bf16 weights, whose row sums miss 1 by 4.9e-4 mean / 2.4e-3 max (measured offline); the old code forced 1, the new
    keeps the rounded sum the untruncated model already uses. **Cost of the bump:** the 3 cached `auto` picks on disk
    re-measure once on their next `auto` load (~400 s / 1556 s / 1584 s); server default is `None`.
  - history: **2026-09-15 13:14 — round-1 device stop rule FIRED, not shipped; resolved by the snap (above).** `decode_top_k_ppl.py`, 256 scored:
    Qwen3-Next k=10 **8.5824** (=), k=9 **8.7662** (ref 8.7089, +0.0573), k=8 **8.8082** (ref 8.8765, −0.0683);
    MiniMax k=8 **7.2626** (=), k=7 **7.6751** (ref 7.7231, −0.0480). Identity arms reproduce to the digit; every
    truncated arm misses by ~100× tolerance, both signs. **But on Qwen3-Next the router returns float32, where old vs
    new `truncate_top_k` differ by ≤ 1.8e-7 with identical indices (measured) — the fix cannot move it; suspect the
    references/harness.** In bf16 (MiniMax) old vs new differ by one bf16 quantum (≤ 7.8e-3 rel), so MiniMax can
    move. Unblocks: same-day arms on an unmodified HEAD worktree (`36c1178`), running. Logs `docs/probes/d1-*.log`.
    **13:18–13:23 — HEAD reproduces ALL FIVE references to the digit (Qwen3-Next 8.5824 / 8.7089 / 8.8765, MiniMax
    7.2626 / 7.7231), so the fix DID move
    it: a ≤ 1.8e-7 float32 weight change shifts 256-token decode ppl by ~0.6 % (bf16 activations + recurrent state).**
    Round 2: `row_sum` within 1e-2 of 1 snaps to exactly 1.0 (bf16 rounding max 2.44e-3, measured) → unit-sum routers
    bit-identical to plain renormalisation, scaled routers still keep their sum.
    `test_unit_sum_rows_that_miss_one_by_rounding_stay_bit_identical[k|p × float32|bf16]` RED before (weights not
    `torch.equal`), GREEN after; three files 235 passed. Running: snap arms (must equal HEAD) + a NULL CONTROL (HEAD
    with ×(1+ulp) on the k path — how far ANY bit-level perturbation moves this ppl; it bears on row 33's 3.5 %
    budget, where k=8 was picked at +3.43 %) + full suite.
- [x] **D6 CLOSED 2026-09-18 by the user — no code change.** Was: `DECODE_TOP_K=auto`'s 3.5 % budget walk decides inside its own measurement noise (started 2026-09-16).
  **Evidence and verdict (the user accepted "close, no code" on 2026-09-18):** RESULTS:
  [`docs/2026-09-16-D6-budget-walk-noise-band-RESULTS.md`](docs/2026-09-16-D6-budget-walk-noise-band-RESULTS.md)
  (gitignored — these lines are the durable copy). Probe `plans/probes/d6_band.py` = `auto`'s own `decode_mode_ppl`
  ops on `CALIB_TEXTS`, kept weights × (1 + n·2^-23), n ∈ {0, ±1, ±2, ±4}, per-token losses kept. It reproduces
  `auto`'s cached baselines BIT FOR BIT (Qwen3-Next code 1.9004663318542685 / prose 8.582392269202229; Qwen3-235B
  7.7817766799255255) and D1's single null exactly; all in-process n=0 controls are identical.
  - **Band at T=256: range 1.41–1.80 pp, sd 0.49–0.68 pp, on both models → stop rule (≤ 0.1 pp) FAILS by ~14–18×.**
  - **Mechanism:** a 1-ulp change decorrelates the teacher-forced trajectory from step 0–5; the effect is diffuse
    (46–78 of 256 steps move > 0.1 nat) and saturated (n=±4 ≈ n=±1).
  - The band narrows ≈ T^-0.5; 0.1 pp needs ~10,800 scored tokens (42×) → not a fix.
  - **Qwen3-Next k=8:** draws +3.43/+3.26/+3.93/+2.13/+3.83/+2.57/+2.87, mean **+3.15**, **flips to k=9 on 2/7**.
  - **Qwen3-235B k=7:** draws +3.40/+2.60/+2.60/+2.04/+2.71/+3.46/+2.83, mean **+2.81**, **0/7 flips**.
  - **Both picks are correct in expectation**; the only exposure is Qwen3-Next's reproducibility on a re-measure
    (the conservative side). A boundary ensemble would cost +60–100 % calibration and was not built. (found 2026-09-15 by
  D1's null control; no brief, so this row is the summary). A single 256-token decode perplexity is bit-sensitive:
  - a semantically null ×(1+2^-23) on the kept weights moved Qwen3-Next k=9 **+0.40 %** and k=8 **−0.16 %**;
  - D1's round-1 change (≤ 1.8e-7 on the weights) moved it +0.66 % / −0.77 %, and MiniMax k=7 −0.62 %.

  It is deterministic: the same code reproduces bit-for-bit, and any bit-pattern change draws a different value from
  that band. **Two on-device picks sit inside it:**
  - Qwen3-Next k=8 at +3.43 %, 0.07 pp under budget. The next-milder step is k=9, +1.47 %.
  - Qwen3-235B k=7 at +3.40 %, 0.10 pp under. Its only milder step is `none`, which would lose its measured
    **+11.15 %** decode.

  MiniMax p=0.90 at +2.25 % is 1.25 pp clear. So an unrelated change that alters bits on the truncated path can flip a
  pick at its next re-measure (any `METHOD_VERSION` / text / fingerprint change). Not a defect in today's picks. Price
  first, no GPU yet: how wide the band really is (several null perturbations, not one), whether it narrows with
  `SCORE_TOKENS`, or a margin / hysteresis on the budget. Stop rule: a band ≤ 0.1 pp at the current 256 tokens → close
  as noise-free enough.
  **Data point 2026-09-15 (D5), and it WIDENS the band:** on MiniMax, a true clamp and tensor truncation keep the same
  experts on 100 % (k=7) / 99.34 % (k=6) of rows with weights equal to one bf16 step — they differ in slot order and
  rounding path only — yet measured +8.20 % vs +6.34 % at k=7 (**+1.86 pp**) and +16.46 % vs +19.30 % at k=6
  (**−2.84 pp**). So between equivalent arms at these damage levels the band is ±2-3 pp, not ±0.8. Caveat: MiniMax is
  bf16-routed and more damaged here than Qwen3-Next/Qwen3-235B at their picks; the band may scale with damage.
- [x] **CLOSED 2026-09-15: a checkpoint missing a persistent buffer is no longer silent; the MiniMax bias is
  fetchable and measured NEUTRAL.** Shipped: `vaayu/missing_buffers.py` + `VaayuBaseModel._apply_missing_buffers()`
  (WARNING for any persistent buffer the owning layer file lacks — fires on MiniMax only, 0 on Qwen3-235B, Qwen3-Next,
  Llama-3.3-70B, Llama-3.2-1B), an auto-applied sidecar `splitted_model/vaayu_missing_buffers.safetensors`,
  `scripts/fetch_missing_buffers.py` (HTTP Range, pinned sha), `documentation/missing_buffers.md`. Tests 14 new;
  full suite **2243 passed / 15 skipped / 0 failed**. Device, MiniMax, `decode_top_k_ppl.py` KS=8 256 tokens: with the
  sidecar `Loaded 62 missing buffer(s)`, no WARNING, ppl **7.0212** (= B1's f32 arm 7.021189); without it the WARNING
  prints and ppl **7.2626** (= reference). B1 verdict NEUTRAL (code mean −2.58 %, one code passage +6.25 %) → bias
  optional. **Box left as found:** sidecar moved to `models/MiniMaxAI--MiniMax-M2.7-e_score_correction_bias/` —
  copy it back into `splitted_model/` to enable. §8: (1) no knob — the sidecar's presence is the switch; (2) runs in
  `VaayuBaseModel.__init__`, the path `app.py`'s `from_pretrained` takes; (3) reachable at N=1 (load time); (4)
  reachable today — verified through `AutoModel.from_pretrained` with the server's config knobs, not through a live
  server process. Not done: reporting the exllamav3 key mismatch upstream (outward-facing, user's call).
  Was: [bug] **MiniMax-M2.7 EXL3 build runs WITHOUT its trained routing bias, silently** (found 2026-09-15 by D5, no brief —
  this row is the summary). `config.json` declares `use_routing_bias: true`, `scoring_func: sigmoid`, and its
  `original_quantization_config.modules_to_not_convert` lists `e_score_correction_bias` (so upstream
  `MiniMaxAI/MiniMax-M2.7` ships it), but `NeuroSenko--MiniMax-M2.7-exl3` has no such tensor; the HF block's
  `register_buffer(torch.zeros)` stays zero on all 62 layers (measured, CPU). exllamav3 looks for
  `block_sparse_moe.gate.e_score_correction_bias` with `optional = True`
  (`exllamav3/architecture/minimax_m2.py:133`), while the HF code registers it on the block — likely why the
  conversion dropped it, and why exllamav3 is silent too. **Effect unmeasured:** every token's routed set may differ
  from the trained model's, at k=8, on every MiniMax number on record. CLAUDE.md §5 says warn, don't silently degrade:
  Vaayu should at least WARN when the config declares a routing bias the checkpoint has no tensor for. Measuring the
  cost needs the 62 × 256 bias values from upstream (a small ranged download — user's call) and a ppl A/B.
  - history: **2026-09-15 — user approved the download; DONE.** Ranged GETs of only the 62 tensors from
    `MiniMaxAI/MiniMax-M2.7` @ `d494266a4affc0d2995ba1fa35c8481cbd84294b` (upstream key
    `model.layers.N.block_sparse_moe.e_score_correction_bias` — the HF buffer's name, NOT exllamav3's
    `...gate.e_score_correction_bias`: key mismatch confirmed). Saved to
    `/home/a/code/ai-ml/models/MiniMaxAI--MiniMax-M2.7-e_score_correction_bias/e_score_correction_bias.safetensors`
    (70,896 B, sha256 `48c9bcfe…2de2cc1`), outside the EXL3 dir so the `auto` fingerprint is untouched. float32,
    15,872/15,872 nonzero, range 1.303-8.860, mean 2.153, **per-layer std median 0.045** — comparable to MiniMax's
    measured decode boundary gap (top-8 sigmoid spread 0.458, k=7/k=8 relative gap 7.6 %), so selection should change.
    **Trap if wired:** `vaayu_base.py:3291` casts buffers to `running_dtype`; bf16's step at ~2.15 is ~0.016, a third
    of the selection-relevant spread. Next: ppl A/B (scratchpad `d5_bias_ab.py`) zero vs float32 vs bf16 bias at k=8,
    prose + code, plus trunc vs clamp k=7 under the float32 bias.
  - history: **2026-09-15 — A/B MEASURED: MIXED, verdict open.** One process, 48 + 256 decode ppl, buffer swapped per
    arm (`docs/probes/d5-bias-ab.{log,json}`). k=8 prose: zero **7.262630** (= reference) → f32 bias **7.021189
    (−3.32 %)**, bf16 bias 7.053263 (−2.88 %). k=8 code (`CALIB_TEXT`): zero **1.775318** → f32 **1.859224 (+4.73 %)**.
    k=7 prose with f32 bias: truncation 7.217688 (**+2.80 %** vs its k=8; zero-bias k=7 was +6.34 %), true clamp
    7.237924 (+3.09 %). So: prose better, code WORSE, both outside the k=8 null band (±0.4 %); bias ≈ halves MiniMax's
    k=7 prose cost (suggestive — 3.5 pp, edge of D5's ±2-3 pp equivalent-arm band); bf16 cast costs only 0.45 pp.
    Unverified: the EXL3 build was likely calibrated on zero-bias routing (exllamav3 runs it biasless), which would
    explain code getting worse. **Do NOT wire the bias yet** — Vaayu's workload is code, the only code number is a
    regression. Next (user's call): (1) load-time WARNING when a config declares a routing bias with no tensor —
    cheap, CPU, correct either way; (2) broader ppl: several code + prose passages, code at k=7, before any sidecar
    loading. RESULTS: `docs/2026-09-15-D5-minimax-true-clamp-RESULTS.md` §4.
  - **PLAN APPROVED 2026-09-15 (user: "add the warning then run, instructions how to add bias, document why it's
    important, do the broader test"):**
    - [x] **B1 broader ppl test** (GPU, scratchpad probe): 4 code + 4 prose passages from real files, zero vs upstream
      f32 bias at k=8, code at k=7 both ways; decision rule written in the RESULTS doc BEFORE the run.
      **DONE 2026-09-15 — verdict NEUTRAL by the pre-registered rule** (`docs/probes/b1-bias-broad.{log,json}`). k=8,
      zero → f32: code Vaayu `expert_cache.py` 32.970153 → 29.764052 (−9.72 %), Rudra `prefetch.py` 5.865248 →
      6.231713 (**+6.25 %**), stdlib `textwrap.py` −1.89 %, `json/decoder.py` −4.96 % → **code mean −2.58 %**; prose
      row-33 −3.32 %, `architecture.md` −0.37 %, `TUNING.md` +2.58 %, Rudra README −3.38 % → mean −1.13 %. Rule: code
      mean ≤ −0.5 % AND no code passage > +1.0 % → FAILS on `prefetch.py` (and §4's calibration code was +4.73 %);
      not ≥ +0.5 % either → neutral: optional, not a default, no throughput A/B. k=7 cost vs own k=8 on code: zero
      +1.96 %, f32 +2.63 % — **the bias does not make `DECODE_TOP_K` cheaper on code** (the one prose passage's
      halving did not generalise).
    - [x] **B2 load-time WARNING** (TDD; DONE 2026-09-15 — device run without sidecar prints it): persistent buffer with no checkpoint tensor → WARNING naming it; false-positive
      check on Qwen3-235B, Qwen3-Next, Llama-3.3-70B, Llama-3.2-1B first. §8 answer in the row.
      - false-positive check DONE (CPU, `from_config` + `state_dict()` keys vs checkpoint keys): MiniMax 62 missing (all
        `e_score_correction_bias`); Qwen3-235B, Qwen3-Next, Llama-3.3-70B, Llama-3.2-1B **0** persistent buffers, 0
        missing. Non-persistent buffers (rotary `inv_freq`) are excluded by the rule.
    - [x] **B3 a working way to add the bias** (DONE 2026-09-15 — real fetch + device load `Loaded 62`, ppl 7.0212) — none exists today (both layer loaders would cast a file tensor to
      `running_dtype` and nothing user-facing writes one). **User chose 2026-09-15: fetch script + auto-load** —
      `scripts/fetch_missing_buffers.py` range-downloads only the missing tensors from the upstream repo the user names
      into a sidecar file in the split dir; Vaayu applies it at load (file dtype kept), no server knob (file presence is
      the switch), and the `auto` fingerprint changes with it. TDD; end-to-end check on MiniMax after B1.
      - progress 2026-09-15: `tests/test_missing_buffers.py` (14) RED (`ModuleNotFoundError: vaayu.missing_buffers`)
        → GREEN 14 passed. New `vaayu/missing_buffers.py` (detect by owning layer file header on a dot boundary,
        sidecar `vaayu_missing_buffers.safetensors` in the split dir keeps file dtype, wrong shape raises, WARNING
        names the fetch command); `VaayuBaseModel._apply_missing_buffers()` runs right after the post-`setup_model`
        `set_layers_from_layer_names()`; `scripts/fetch_missing_buffers.py` (meta skeleton → missing names → HTTP Range
        fetch, pinned sha, `HF_TOKEN` honoured, refuses a server that ignores Range). Split-dir scanners checked safe:
        expert index keeps `.experts.N.` keys only (one rebuild on newer mtime), `moe_key_remap` filters by layer
        prefix, EXL3 detect peeks `model.layers.0`, io_uring looks files up by path (`file_map`). Pending: full suite
        (deferred — B1 holds the RAM), device load check (warning gone, ppl = f32 arm).
      - real fetch DONE 2026-09-15: `scripts/fetch_missing_buffers.py --repo MiniMaxAI/MiniMax-M2.7` found the 62
        missing buffers from the meta skeleton, pinned `d494266a…`, wrote `splitted_model/vaayu_missing_buffers.safetensors`
        (70,896 B); all 62 tensors `torch.equal` to the manual download, float32, source repo + sha in metadata.
        **The sidecar is LIVE on this box:** every MiniMax load from now applies the upstream bias, and the cached
        `DECODE_TOP_K=auto` pick in that split dir re-measures on its next `auto` load. Delete the file to revert.
    - [x] B1 first launch stopped before load (probe's prose start rule found no ≥ 200-char line in TUNING.md);
      rule amended to ≥ 120 before any result, relaunched.
    - [x] **B4 docs** (`documentation/`; DONE 2026-09-15): why the routing bias matters + how to add it, with B1's numbers.
      - progress 2026-09-15: new `documentation/missing_buffers.md` (what a missing persistent buffer is, the WARNING,
        why the MoE routing bias matters, how to fetch, license, undo, `auto` re-measure side effect); linked from the
        root `README.md` docs list, `vaayu/server/README.md` "Reading the logs" (§8: no knob — file presence is the
        switch) and a `documentation/server.md` §9 troubleshooting row. Measured section now carries B1's 9 passages
        and the neutral verdict (bias optional; the one-passage "k=7 costs half" claim removed — it did not hold on code).
- [x] **CLOSED 2026-09-15 (D2): KEPT — Qwen3-235B first load −15.1 % (1514.0 → 1285.4 s, −228.6 s), picks and
  perplexities identical.** `_resolve_decode_top_k_auto` (`vaayu/vaayu_base.py`) runs `resolve(...)` inside
  `with keepawake_for(self):`, so calibration's bare `model(...)` decode forwards get row 27's spinners and P-core
  pinning, as `generate` does. **Device ABBA** (`plans/probes/row26_decode.py`, `BALLAST_GB=25`, cache deleted before
  each arm; fix = working tree, base = `145566d` worktree): calibration seconds fix1 **1283.7** · base1 1508.2 ·
  base2 1506.6 · fix2 **1274.6** → mean −228.2 s (−15.14 %); base replicates within 0.1 %. Every arm picked top-7
  with identical ppl (1.7659/7.7818 · 1.7824/8.0462 · 1.8078/8.1966) — bit-identical, as predicted (affinity only).
  Mechanism checked live: fix1 had 8 `SCHED_IDLE` spinners on E-cores 12–19 and 170/170 threads on CPUs 0–11; base1
  had 0 spinners and 66/66 threads on 0–19; 0 orphan spinners after. Stop rule (≥ 5 % ≈ 78 s) passed ~3×. Docs:
  Qwen3-235B first load ≈26 → ≈21 min in `vaayu/server/config.py`, `vaayu/server/README.md`,
  `documentation/server.md`, `documentation/TUNING.md`; MiniMax (≈26) and Qwen3-Next (≈6.5) not re-measured, left as
  is. §8: no knob (reuses `KEEP_AWAKE` / `PIN_ENGINE_THREADS`, honoured — both set before the hook); plumbed; reachable
  at N=1 (runs at server load); reachable today for `DECODE_TOP_K = "auto"` users. RESULTS:
  `docs/2026-09-15-D2-calibration-keepawake-RESULTS.md`.
  - history: **Phase 1** confirmed in code: `resolve(...)` (`vaayu/vaayu_base.py`
  `_resolve_decode_top_k_auto`) runs bare; `keep_awake` / `pin_engine_threads` are set before the hook, so a wrap
  honours a user's `False`. Prior Qwen3-235B cold load: calibration 584.0 + 516.2 + 449.2 = 1549.4 s of 1555.9 s.
  Plan: RED test → wrap → suites → ABBA first-load A/B (fix = working tree, base = HEAD worktree).
  - history: **CPU phase done.** `resolve(...)` now runs inside `with keepawake_for(self):`; D3's telemetry clear
    stays after it, outside the wrap. `tests/test_decode_top_k.py::test_calibration_runs_inside_keepawake` RED before
    (`assert False is True`), GREEN after. Two files 88 passed; full `pytest tests/ -q` **2229 passed / 15 skipped /
    0 failed, 172 s**. Device ABBA (fix, base, base, fix; cache deleted before each arm) RUNNING, logs
    `docs/probes/d2-{fix1,base1,base2,fix2}.log`, summary `docs/probes/d2-abba-summary.txt`. Stop rule: saving < 5 %
    of first-load time → revert, close as measured null.
  Was: [bug] **DECODE_TOP_K=auto calibration runs without KEEP_AWAKE / thread pinning** (final review M4). `decode_mode_ppl` calls
  `model(...)` directly; `keepawake_for` wraps only `generate` (`vaayu/vaayu_base.py` ~5204). Row 27 measured −21.5 % time per
  token on Qwen3-235B from those two knobs, so the ~26-min first load is probably longer than necessary. Candidate: wrap
  `run_ladders` in `with keepawake_for(model):`. **Unmeasured — needs a first-load A/B before claiming a saving.**
  **Brief (D2, self-contained):** [`docs/issue-D2-calibration-without-keepawake.md`](docs/issue-D2-calibration-without-keepawake.md) (gitignored — this row is the durable summary).
- [x] **CLOSED 2026-09-15 (D3):** `_resolve_decode_top_k_auto` (`vaayu/vaayu_base.py`, right after `resolve` returns)
  now zeroes `_expert_usage.counts` and clears `_expert_vram_cache.evicted_keys` when the status `source` is `measured`
  or `failed` (a failed ladder may have recorded part of a run); `cache` / `config` ran no forward and clear nothing.
  `tests/test_decode_top_k.py::test_calibration_routes_do_not_stay_in_the_usage_telemetry[measured|failed]` RED before
  (`assert 2 == 0`), GREEN after; `test_a_cached_pick_leaves_the_usage_telemetry_alone` guards against over-clearing
  (green both sides, by design). Three files 103 passed; full `pytest tests/ -q` **2210 passed / 15 skipped / 0 failed,
  173 s**. **Reachability, checked:** latent in-repo — the histogram's only reader, `test_generate.py:254-274`, sets
  neither `enable_expert_usage` nor `decode_top_k`, and the server builds the tracker but never prints the report.
  §8: no knob (behind `ENABLE_EXPERT_USAGE`); already plumbed; reachable at N=1; telemetry only, no output or speed
  change. CPU only. D2 edits the same method next.
  Was: [bug] **DECODE_TOP_K=auto calibration is counted in the expert-usage histogram** (final review M5, **scope corrected
  2026-09-14 while writing the brief**). The review's main claim — that `_decode_stats_reset` fires once per model lifetime,
  so calibration pollutes later decode hit rates — is **wrong**: `decode_stats_reset` (`vaayu/vaayu_base.py:426`) re-arms on
  every prefill (the F13 fix), so the first real turn's first decode pass wipes calibration's cache counts. What remains:
  with `ENABLE_EXPERT_USAGE` on (default off), `ExpertUsageTracker` (`vaayu/expert_usage.py`, no reset method) permanently
  counts calibration's decode routes, truncated trials included. Telemetry only — no effect on output or speed.
  **Brief (D3, self-contained):** [`docs/issue-D3-calibration-consumes-decode-telemetry.md`](docs/issue-D3-calibration-consumes-decode-telemetry.md) (gitignored — this row is the durable summary).
- [x] **CLOSED 2026-09-15 (D4):** `check_decode_top_k_value(decode_top_k)` now runs in `VaayuBaseModel.__init__`
  beside the budget check (`vaayu/vaayu_base.py:2013`), before `find_or_create_local_splitted_path`; the two later
  checks (`resolve_decode_top_k`, `_detect_moe`'s dense branch) stay for tests that bypass `__init__`.
  `tests/test_decode_top_k.py::test_an_invalid_decode_top_k_fails_before_the_layer_split` (`"Auto"`, `"8"`, `True`,
  `7.5`; split stubbed to raise) RED before — the stub was reached — GREEN after; file 61 passed. Full suite 2206
  passed / 15 skipped / **1 failed, unrelated** (`test_slab_arena.py::test_load_expert_into_slab_tensor_align_gates_pool_usage`,
  passes alone and alongside this file — order-dependent, recorded as its own `[bug]` row below). §8: no knob; already
  plumbed (`app.py:423`, `__main__.py:44`); reachable at N=1 — a server started with a typo now fails before the split
  instead of after it. CPU only, no behaviour change for a valid value.
  Was: [bug] **Invalid DECODE_TOP_K fails after the layer split, not before** (final review M6). `check_decode_top_k_value` runs in
  `_detect_moe` (~2206), after `find_or_create_local_splitted_path` (~2022); the budget is validated at ~2013. Calling it next
  to the budget check would fail fast. The late timing predates this branch for the int path.
  **Brief (D4, self-contained):** [`docs/issue-D4-decode-top-k-validated-after-split.md`](docs/issue-D4-decode-top-k-validated-after-split.md) (gitignored — this row is the durable summary).
- [bug] **`tests/test_slab_arena.py::test_load_expert_into_slab_tensor_align_gates_pool_usage` fails in the full suite
  only** (found 2026-09-15 during D4's suite run: 1 failed / 2206 passed / 15 skipped, 18 min). It passes alone (58
  passed), with `test_decode_top_k.py` (119 passed) and with `test_ring_pool_sizing.py` (63 passed). The test asserts on
  process-global state — `_EXPERT_BUF_POOL` (`vaayu/persist/safetensor_model_persister.py:155`) via a wrapped
  `get` `call_count` and `qsize() == 1` — so something earlier in suite order leaves that state dirty. **Which assertion
  fails was not captured** (the run's output was cut to its summary). **Refuted:** `init_model_io`'s pool fill
  (`:314-341`, only called by `tests/test_ring_pool_sizing.py`). **Unconfirmed candidate:** a background thread left by
  an earlier test calling `_EXPERT_BUF_POOL.get` (`:748`, `:1004`) would bump `call_count`. Not proven pre-existing on
  HEAD, but D4's change (one validation line in `__init__`) never reaches the pool. Next: re-run the full suite with
  `--tb=long` to a log to get the assertion, then bisect by file. Test hygiene only — no engine behaviour.
  **2026-09-15 (D3's suite run): did NOT reproduce** — `pytest tests/ -q --tb=long`, 2210 passed / 15 skipped / 0 failed
  in **173 s**. D4's run is recorded at 18 min for ~the same test count; that gap would fit D4's run having collected the
  repo-root on-device scripts too (see the bare-`pytest` row below) — **unverified**, its command was not recorded.
- [x] **CLOSED 2026-09-15:** `pytest.ini` at the repo root sets `testpaths = tests`.
  `tests/test_pytest_config.py::test_bare_pytest_collects_only_the_tests_directory` RED before (`pytest.ini missing`),
  GREEN after. Bare `pytest --collect-only` from the root under a 4 G / no-swap scope with CUDA hidden:
  `configfile: pytest.ini`, `testpaths: tests`, 2226 collected, no root module, exit 0. Bare `pytest -q` from the root:
  **2211 passed / 15 skipped / 0 failed, 172 s**. Rootdir unchanged (already the repo root); no test reads it; the only
  conftest (`tests/conftest.py`) defines no hooks. Explicit paths (`pytest tests/test_x.py`, the on-device gates) are
  unaffected — `testpaths` only applies with no args. §8: test tooling, no engine or server change.
  Was: [bug] **A bare `pytest` from the repo root loads real models and can OOM the box** (found 2026-09-15 during D3). There is
  no pytest config (no `pytest.ini` / `pyproject.toml` / `setup.cfg` / `tox.ini`), so `./venv/bin/python -m pytest -q`
  collects the root scripts `test_generate.py` (Qwen3-235B) and `test_multiturn.py` (MiniMax-M2.7), whose module-level
  code loads and generates at import. Observed: run started 11:54:45, `.vaayu_kv` written 11:58:08, kernel `global_oom`
  killed pytest (total-vm 103 GB) at 64 % at 12:11:21. Nothing destructive (`VAAYU_DELETE_ORIGINAL` unset, both scripts
  default it off). The Environment section's `pytest tests/ -q` is correct; the trap is the default. Candidate fix:
  a minimal config with `testpaths = tests`. Not done — outside D3's scope.
- [x] **CLOSED 2026-09-14 (Task 6, commit `ea0cdc2`):** `documentation/server.md` §3 "Speed" table was missing three shipped knobs (found 2026-09-14 during the
  auto spec self-review). That table lists speed knobs by constant name (`MAX_PINNED_LAYERS` … `ENABLE_EXPERT_USAGE`),
  but has no row for **`DECODE_TOP_K`** (row 33, commit `7bc5703` — documented only in `vaayu/server/README.md`),
  **`KEEP_AWAKE`** or **`PIN_ENGINE_THREADS`** (row 27). `DECODE_TOP_K` is closed by the auto work's docs step (spec §8).
  **User decision 2026-09-14: all three are FOLDED INTO the auto work's docs task** (spec §8/§14 updated) —
  one `server.md` edit adds `DECODE_TOP_K`, `DECODE_TOP_K_BUDGET`, `KEEP_AWAKE`, `PIN_ENGINE_THREADS`, the last two
  in `vaayu/server/README.md`'s wording. Task 6 (commit `ea0cdc2`, "docs: DECODE_TOP_K auto in TUNING.md;
  server.md rows for four speed knobs") added all four rows to `documentation/server.md` §3.
- [x] **CLOSED 2026-09-15 (K1):** README sentence replaced with row 27's server figures (+23.1 % Qwen3-235B, 0.9229 →
  1.1363, TTFT −22.3 %; MiniMax +1.3 %, 70B +2.9 %, 1B unchanged); `tests/test_docs_speed_knobs.py::test_readme_keepawake_row_is_not_stale`
  RED before, GREEN after; doc tests 13 passed. §8: docs only — no knob, nothing to plumb, no behaviour change.
  Was: [bug] **`vaayu/server/README.md`'s `KEEP_AWAKE` row carries a STALE claim** (found 2026-09-14 while planning the
  server.md rows): it ends "**The server-level A/B is pending (todo.md row 27); the figures above are in-process**", but
  row 27 CLOSED 2026-09-12 with the server A/B done — **+23.1 % decode tok/s on Qwen3-235B through the server** (0.9229 →
  1.1363, n=3, ids identical). The auto plan's server.md row uses the measured server figure instead of copying the
  stale sentence. **README not fixed** (not in the agreed scope) — replace that sentence with the row-27 server numbers.
  Docs-only, ~5 min.
  Docs-only, ~5 min. **Brief (K1, self-contained):** [`docs/issue-K1-readme-keepawake-stale-claim.md`](docs/issue-K1-readme-keepawake-stale-claim.md) (gitignored — this row is the durable summary).


## BUILT 2026-09-13 (T2). Code landed, suite green, quality measured. Default `DECODE_TOP_K = None`.

**The clamp is on the ROUTED TENSORS, not on each module's `top_k` attribute — and that
correction matters, because this row named TWO routing owners and there are THREE.** The third is
the flagship's: `Qwen3NextTopKRouter.forward` runs its own `torch.topk(..., self.top_k)` and returns
`(logits, weights, indices)`, so the fused non-EXL3 forward **never uses the engine's `top_k` for
selection**. Clamping `self.moe_top_k` alone would have been a **silent no-op on the fastest model on
this box**, with every derived budget sized for a k the model was not using — the `VAAYU_CACHE_BIAS`
failure mode a fifth time. `truncate_top_k(weights, indices, k)` (`vaayu/moe_patch.py`) re-runs the
top-k on the weights routing already returned, renormalises, and returns the SAME tensors when off
(k=None or k>=K) so the feature-off path stays bit-identical. Called in all three forwards after
routing, gated on `not _moe_prefill_pass`.

**The shape trap fired for real:** `compute_expert_dispatch(... .reshape(-1, top_k))` and
`routing_weights.reshape(-1, top_k)` in the module-list forward close over the patch-time `top_k`
and raised `RuntimeError: shape '[-1, 4]' is invalid for input of size 2` on the first live run.
Both now take the width off the tensor.

**Where it is NOT equivalent to a true clamp:** MiniMax selects on `sigmoid + e_score_correction_bias`
with `sorted=False` and weights from the UNBIASED sigmoid, so truncation keeps the k largest
COMBINATION WEIGHTS of its 8 — defensible (drops the smallest contributions) but not what the model
would have selected at k. For the two softmax owners it IS identical.

**Budgets:** `_build_moe_geometry` (extracted from `__init__` to be testable) sizes the geometry at
the decode k — `working_set_bytes` is one decode step by definition. Prefill keeps `self.moe_top_k`.
A speculative verify pass counts as DECODE (`compute_prefill_pass`), so the clamp shrinks
`compute_spec_working_set` too — which makes row 37's `off_union` byte bound smaller for free.

**§8:** knob `DECODE_TOP_K` in `vaayu/server/config.py`, plumbed through `app.py` and `__main__.py`,
reachable at N=1. Adding the engine parameter made `tests/test_engine_param_reachability.py` fail on
its own — the §8 gate doing exactly its job.

**Known minors, NOT fixed (both worth 0 % to users today):** the batch-overlap telemetry
`vaayu_ref._batch_n * top_k` in all three forwards still reports the config k (N>1 only, and
`batched_generate` is not wired into the server); `_lookahead_step(..., top_k, ...)` predicts and
stages at the config k (`VAAYU_LOOKAHEAD` is hard-blocked — row 35).


`bytes_read ∝ top_k`, and `top_k` enters the engine at **exactly one place** — `vaayu/vaayu_base.py:3233`:

```python
self.moe_top_k = getattr(self.config, "num_experts_per_tok",
                         getattr(self.config, "top_k", 2))
```

Everything downstream reads it: `compute_moe_geometry` (working set, cache budgets, per-layer caps) and all three MoE
forwards — two via `route_experts`, the fused non-EXL3 one via an inline copy (`moe_patch.py:1327-1329`). **Clamping at that single assignment shrinks both the routed slots and every derived
budget, coherently, on all three forwards** — which is what `CLAUDE.md` §5 asks for.

| top_k | Qwen bytes/token | Qwen T | vs 0.930 | MiniMax T | vs 0.562 |
|---|---|---|---|---|---|
| 8 (config) | 4.84 GB | 0.930 s | — | 0.562 s | — |
| **6** | 3.63 GB | 0.738 s | **×1.26** | 0.463 s | **×1.21** |
| 4 | 2.42 GB | 0.545 s | ×1.71 | 0.364 s | ×1.54 |

**Two traps, each of which silently produces garbage or a silent no-op:**

1. **Renormalisation is already there; the trap is that the CONTRACT path ignores your `k` (corrected 2026-09-12
   after reading the routing code — this row's first draft said the opposite).** Every softmax path already
   renormalises over the selected experts: `route_experts`'s fallback (`vaayu/moe_patch.py:158`) and
   `_make_fused_sparse_forward`'s inline copy (`:1329`); the module-list (`:1013`) and fused-EXL3 forwards both call
   `route_experts`. **But when the model defines `route_tokens_to_experts` (MiniMax-M2.7 and the 25+ architectures
   transformers standardises), `route_experts` delegates to it and that method takes NO `k`** — it reads the module's
   own `self.top_k = config.num_experts_per_tok` (`modeling_minimax_m2.py:106,115`) and renormalises its sigmoid
   weights itself (`:117`). A clamp on `self.moe_top_k` alone therefore changes the engine's budgets and the
   `.reshape(-1, top_k)` sites (`:1068`, `:1086`) while the model **keeps selecting 8** — a shape mismatch that
   crashes or silently mis-groups indices. **Write the clamp into every MoE module's own attribute too, by
   detection** (`moe_module.top_k` where it exists, plus the config field the geometry was read from), and test
   `indices.shape[-1] == clamped_k` on **both** a softmax-routed and a contract-routed module.
2. **Decide prefill vs decode explicitly.** Prefill routes ~every expert anyway (**224.23 experts/layer per chunk**,
   measured — SETTLED), so truncation buys nearly nothing there and costs prompt quality. **Decode-only is the
   defensible scope; state which you chose.**

**Quality gate:** perplexity at k ∈ {8, 6, 5, 4} on **both** MoE models, plus a real agentic generation read by a
human. **Greedy ids WILL diverge — this is lossy by construction, so do not use id-identity as the correctness check.**
That check is correct for every lossless arm in this repo and will produce a false failure here.

**Stop rule: if k=6 costs more quality than row 32's 3.0 bpw build at the same speed-up, prefer 32 and close this.**
They attack the same product term and **compose multiplicatively**, so the question is which is cheaper per unit of
quality, not which wins.

**§8:** (1) **yes, a knob** — `DECODE_TOP_K = None` in `vaayu/server/config.py` (`None` = the config value, an int =
clamp, **never raise above config**), in the same commit, with the measured quality cost, and in
`vaayu/server/README.md`. (2) plumb through `app.py`'s `from_pretrained(...)`. (3) reachable at N=1. (4) reachable now.

**SETTLED trap — read carefully.** `cache_aware_topk` (`vaayu/moe_patch.py:167`) is **dead code that never executes**
under transformers 5.5.4 (0 calls, both models; Plan G closed). It is a **different idea** — biasing *which* experts
are selected toward cache-resident ones at constant k. **This row keeps selection honest and reduces k.** Do not
resurrect the dead path to implement it, and do not let a review conflate them.

---

# 34 · [x] The drive delivers 6.3 GB/s while it plateaus at 7.1 — probe why (added 2026-09-12) — **CLOSED 2026-09-21 ON ITS OWN STOP RULE: depth is ALREADY 2-4. Its probe was run under row 49 and its second branch fired.**

**VERDICT: `6.0-6.1 GB/s IS the drive's real rate for this access pattern`** — this row's own words for
the branch that fired. **Time-weighted in-flight depth over a real MiniMax decode (row 49's duty probe,
`docs/2026-09-21-row49-B-rerank-RESULTS.md`), restricted to BUSY time and capped at the ring pool
(= device depth): depth 1 = 21.3 %, depth 2 = 20.5 %, depth 3 = 58.1 % -> DEPTH >= 2 FOR 78.6 % OF
BUSY TIME**, against this row's *"≥ 60 % of busy time at depth 1"* trigger. Not close. The
layer-boundary ramp hypothesis is REFUTED as the explanation of the rate: the boundaries do cost, but
as **drive IDLE** (23.6 % of the span, one 1.57 ms gap per layer), not as low occupancy while busy.
R2's coalescing verdict extends, as this row said it would.

**The one thing it got wrong, and it is now fixed: `_NVME_READ_CONCURRENCY = 3` was NOT the optimum.**
This row said *"target sustained 2-4, never more"*; in path the 2-4 zone is not flat and **2 is the
peak** — shipped 2026-09-21 at +1.89 % MiniMax decode / +1.85 % prefill (row 49, `8e8e8d5`).
**The concurrency axis is now exhausted on this box.** What survives as open work is NOT this row's
hypothesis but the two row 49 successors: the **read shape** (3.04 reads per expert, bimodal) and the
**per-layer idle gap** (>= +31 %, needs prediction).

**Its Qwen arm can never be re-run** — Qwen3-235B was deleted 2026-09-20 — and Qwen3-Next's disk is
0.42 % of a decode token, so MiniMax was the only live model for this question anyway.

*Original text below, kept for the trail.*

Qwen moves 4.84 GB in 0.772 s of busy time = **6.3 GB/s**; MiniMax 2.38 GB in 0.396 s = **6.0**. Row P8's plateau is
**7.1 GB/s at 2 concurrent readers**; a single reader delivers **6.0–6.4**.

**That coincidence is the hypothesis: the expert path may spend most of its busy time at depth 1.** There are 94 layer
boundaries per Qwen token and layer L+1 cannot issue a read until layer L has routed, so every boundary plausibly
restarts at one read in flight and ramps.

| | Qwen | MiniMax |
|---|---|---|
| plateau alone (`0.772 × 6.3/7.1`) | 0.685 s → T 0.844 → **×1.10** | 0.335 s → T 0.501 → **×1.12** |
| plateau **+** `exposed` removed (i.e. also a perfect CPU hybrid) | T 0.759 → ×1.23 | T 0.387 → ×1.45 |

**The second row is row 26 §6.2's `×1.231 / ×1.452` line — it is a COMBINED figure. Do not quote it for this row
alone.**

- [ ] **Probe first, no engine change (½ day):** instrument in-flight read depth over wall time during a real decode and
      histogram it **time-weighted, not count-weighted**
- [ ] **≥ 60 % of busy time at depth 1** → mechanism confirmed, build a fix that keeps a second read in flight across
      the layer boundary
- [ ] **depth already 2–4** → 6.3 GB/s is the drive's real rate for this access pattern. **Close the row.**

**SETTLED trap, and it is a near-miss.** R2 (`io_uring_read_many`) was REFUTED: reads are bimodal, there is already
effectively **one read per miss**, dispatched one-future-each, best case ~1.02×. **That refutation is about read COUNT
(coalescing); this row is about read OCCUPANCY over time.** Different quantities; the probe measures the second. If
depth is ≥ 2, R2's verdict simply extends. **And honour P8 in anything built: more concurrency is NOT monotonically
better** — 8 concurrent readers deliver 5.15 GB/s, *worse than today*. `_NVME_READ_CONCURRENCY = 3` came from that
measurement. **Target "sustained 2–4", never "more".**

**§8:** (1) probably no knob — a scheduling property, not a policy; state it either way. (2)/(3) reachable at N=1 if
built. (4) unblocked by the probe. **Only pays while experts stream — dies if row 30's S2 lands.**

---

# 35 · [!] Give `VAAYU_LOOKAHEAD` a per-model measured gate instead of a hard block (added 2026-09-12) — **BLOCKED / ALL BUT DEAD 2026-09-20: its prize model was DELETED and its mechanism died when S2 landed**

**CORRECTED 2026-09-20. Do not pick this row off the queue as "+11.7 %, machinery built" — that
ranking is stale on three counts:**

1. **Its stop rule is UNRUNNABLE.** The rule is verbatim *"cannot reproduce **+11.7 % on Qwen** at
   N=1 THROUGH THE SERVER … → close it. That number is the entire prize."* In this file "Qwen" is
   **Qwen3-235B**, which the user RETIRED AND DELETED on 2026-09-20 (confirmed: no `*Qwen3-235B*`
   directory under `/home/a/code/ai-ml/models/`). **The only model the prize was ever measured on no
   longer exists**, and its forward (fused-EXL3) has no on-device model at all, so the number cannot
   be reproduced or refuted.
2. **Its own kill condition has already fired.** This row says *"Only pays while experts stream —
   **dies if row 30's S2 lands**."* **S2 landed 2026-09-13.** On Qwen3-Next the split spike measured
   `disk_load` at **0.42 / 0.32 % of a decode token** (2026-09-20, row 36) with a 96.6 % RAM-cache
   hit — there is essentially nothing left to prefetch, so lookahead has no mechanism there.
3. **The only remaining model where experts DO stream is the arm that LOST — at N=3.** MiniMax-M2.7
   measured **−3.7 %** there, and the server does not run N=3. **At N=1 the same post-P4 measurement
   reads +2.5 %** (`check_lookahead_flag_safe`, 2026-09-03: OFF 1.811, ON 1.873/1.838).
   **ARITHMETIC ADDED 2026-09-21 (row 49's duty data): the ≥ +15 % bar cannot be met by this
   predictor.** It buys its −15.4 % misses with **+18.0 % bytes/token** at N=1; against a drive
   already busy **76 %** of a MiniMax decode token that puts busy at 0.391 → 0.461 s and the token
   floor at **~+11 %** even with the idle filled perfectly. **Recommend closing this row on that,
   rather than spending a GPU run to re-measure a losing arm.**

**What would re-open it:** a measured ≥ +15 % on MiniMax-M2.7 at **N=1**, on a current build — i.e.
re-measuring the losing arm on the one model that still streams. That is a new measurement, not the
banked +11.7 %. Nothing else revives this row; the +11.7 % should be treated as historical.

**Historical detail below, kept for the trail — the numbers are real, the models are not current.**


Measured post-P4: **Qwen N=1 +11.7 %**, **MiniMax N=3 −3.7 %**. Hard-blocked at `check_lookahead_flag_safe`
(`vaayu/vaayu_base.py:794`) because defaulting it ON ships a win to one actively-tested model and a loss to the other —
the asymmetry `CLAUDE.md` §5 forbids.

**The new angle: §5 forbids DEFAULTING INTO a one-model-wins asymmetry. It does not forbid a per-model gate that
measures and decides.** This repo already ships that pattern twice: `USE_PROMPT_LOOKUP = "auto"` resolves at load where
`is_moe` is known, and `AdaptiveSpecGate` runs an on-line **paired** A/B — row 13's `even-4` trial alternates 4 drafting
and 4 suppressed rounds for ≥ 4 pairs and ≥ 48 drafted tokens and admits only if the drafting rounds were genuinely
cheaper per token than their suppressed partners. Built and tested (`tests/test_adaptive_spec_gate.py`).

**Reuse that harness.** A lookahead gate is cheaper to trial than speculation — no verify pass, so a wrong decision
wastes prefetch bandwidth for a few rounds instead of costing a re-entry. **Price it anyway against row 24's ~29.3
token-equivalent re-entry figure** — every gate proposed in this repo is expected to be priced against that before it
is built — and **latch once per process**, as row 24's gradient recommends.

**Stop rule: cannot reproduce +11.7 % on Qwen at N=1 THROUGH THE SERVER with the trial cost included → close it.**
That number is the entire prize.

**§8:** (1) **yes, a knob** — `LOOKAHEAD = "auto"`, matching `USE_PROMPT_LOOKUP`'s shape. (2) plumb it: the env flag
exists, the server cannot reach it. (3) reachable at N=1 — **note the losing arm was measured at N=3, which the server
does not run**. (4) blocked only by the gate.

**SETTLED trap:** do **not** re-open as *"turn it on to go faster"*, and do **not** attack it via admission — P4
exhausted that lever (a 12-slot/layer reservation removes 99.96 % of prefetch-caused evictions and makes **both** models
worse). What is left is prediction quality (0.766 vs 0.902), and this gate does not need to fix quality, only to decide
per model. **Only pays while experts stream — dies if row 30's S2 lands.**

---

# 36 · [!] CUDA graphs / fused per-layer dispatch — **PARKED 2026-09-20. Its gate fired (32.34 / 32.73 %), then TWO spikes killed every lever: `serial` is 74.8 % GPU-BOUND, its matmul half is at 80–86 % of this card's bandwidth, `use_torch_compile` CRASHES as shipped, and without cudagraphs it is 24–35 % SLOWER.** (added 2026-09-12)

**GATE RE-CHECK DONE 2026-09-20 on `main` `da118ab` — IT FIRES, for the first time in three readings.**
**strict `serial / T_unprof` = 32.34 % / 32.73 %** (two replicates, spread 1.2 %, under the 2 % re-run rule)
against the **≥ 30 %** rule. Numbers, all three readings:

| when | build | `serial` (NOEXP s/tok) | `T_unprof` (s/tok) | strict | fires? |
|---|---|---|---|---|---|
| 2026-09-13 (S3-pre) | pre-30-P | 0.0363 / 0.0363 | 0.1580 / 0.1550 | 22.97 / 23.41 % | no |
| 2026-09-18 (S3 re-gate) | 30-P `108d67f` | 0.0360 / 0.0362 | 0.1367 / 0.1357 | 26.38 / 26.68 % | no |
| **2026-09-20** | **`da118ab`** | **0.0368 / 0.0374** | **0.1138 / 0.1144** | **32.34 / 32.73 %** | **YES** |

**`serial` did NOT move — 0.0363 → 0.0360 → 0.0368, flat across three builds.** That was the open
question: row 44's F5 replaced the per-expert `torch.where` one-hot dispatch (~960 tiny H2D copies per
decode token) *inside* what the NOEXP arm brackets, and it could have fallen. It did not. **F5's own terms
moved and cancelled:** `scatter` 0.0260 → 0.0148 s (−43 %), `mask_sync` 0.0014 → 0.0055 s (+293 %). The
share rose purely because the token shrank (0.1550 → 0.1362 → 0.1141 s, past the derived ~0.120 s trigger).
U-arm tok/s 8.9066 / 8.6726 and 8.7780 / 8.7119; greedy ids identical to U1 on W/U1/P/U2 in both replicates
(NOEXP differs by construction). S3's gate 1 re-read off the same JSONs: **8.15 / 8.25 %**, still under its
≥ 10 % rule — **S3 stays closed.** Wide `other/T_prof` = 41.1 % (was 35.9 %). Detail:
[`…-row36-gate-recheck-RESULTS.md`](docs/2026-09-20-row36-gate-recheck-RESULTS.md) (gitignored; this row is
the durable copy); logs `docs/probes/r36-recheck-{1,2}.log`, JSONs
`plans/probes/row26_out/decode-qwen3next-r36recheck{1,2}.json`.

**SPIKE DONE 2026-09-20 (same day, `0e6736b`) — IT ANSWERED "what removes it" AND THE ANSWER IS "NOT CUDA
GRAPHS". `serial` = 37.04 ms splits 74.8 % GPU kernel time / 25.2 % host gap** (CUPTI; stop rule was
host-gap ≥ 40 %). **7,702 kernel launches per decode token**, 160 per MoE layer. Inside the 27.70 ms of
kernel time: **matmul 12.59 ms** (306 launches) — and that is a **FLOOR**, not a target: the non-expert tier
is 4.83 GB resident, ~4.5 GB is read per token, and this box is an **RTX 5060 Ti at 448 GB/s**, so the floor
is 10.04–10.78 ms and the matmuls are already at **80–86 % of peak memory bandwidth**; **elementwise
7.80 ms across 4,353 launches at 1.8 µs each** (at/below launch latency); topk/reduce/conv 3.11 ms;
**41.7 PAGEABLE HtoD memcpys/token**. Per-module CPU wall: `Qwen3NextGatedDeltaNet` (36 of 48 layers)
**653 µs each = 76.6 %** of a 30.67 ms CPU total; `Qwen3NextAttention` (12) 592 µs each. Graphs remove only
the 9.34 ms host gap = **8.3 % of the full 0.1124 s token**, and capture is broken 48×/token by the MoE
routing D2H sync, so the realistic take is a fraction of that.

**THE RE-SCOPE WAS MEASURED THE SAME DAY AND IT ALSO FAILED (2026-09-20, `34ceb7b`) — ROW 36 IS PARKED.**
Two findings, both on Qwen3-Next, probe `plans/probes/r36_compile_ab.py` (THROWAWAY, no `vaayu/` change):

1. **`use_torch_compile=True` CRASHES — `[bug]`, reproduced in two independent processes.**
   `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`,
   raised at `transformers/models/qwen3_next/modeling_qwen3_next.py:866`. Structural, not incidental:
   `_compiled_layer_fn` uses **`mode="reduce-overhead"` = torch.compile PLUS inductor cudagraph trees**
   (`vaayu_base.py:5064-5068`), which need `torch.compiler.cudagraph_mark_step_begin()` per model
   invocation — and Vaayu drives **48+ layers through that one callable per token** with no step
   boundary. **`effective=True` at load, so the flag IS live on CUDA and `config.py:328`'s stated reason
   for declining it ("not live paths") is FACTUALLY WRONG** — the real reason is that it crashes.
   Also seen: 9 distinct shapes from chunked prefill, and a dynamo recompile-limit hit (`[14/8]`).
2. **Without cudagraphs (`mode="default"`) it is CORRECT but SLOWER.** Within-process ABBA
   (`W,A1,Bw,B1,B2,A2` — one model load, the engine reads `use_torch_compile` per layer call, so no
   cross-process drift): compiled **0.1543 / 0.1539 s** (spread 0.3 %) vs eager **0.1146** (A1; A2 read
   0.1346, a 16 % spread and the ~6 %-dip artefact row 44 documented — every other same-day baseline on
   this model is 0.1113–0.1146). **+23.7 % slower on the A-mean, +34.6 % against A1.** Greedy ids
   **identical across every arm**, so it is correct, just slow. Compile cost 104.4 s.
   **Why it loses: the fusion never happens.** Dynamo 117 frames / 107 ok / **46 unique graphs**, broken
   by **`time.perf_counter` — the engine's OWN per-layer timing instrumentation inside the traced
   region** — and by `aten.bincount.default` (MoE routing, data-dependent shape). The 4,353 tiny
   elementwise launches are re-entered across graph boundaries instead of fused, and guard/dispatch cost
   is added on top.

**So every lever in this row is now closed or floored:** CUDA graphs refuted (8.3 % of a token, capture
broken 48×/token by the routing D2H sync) · matmul 12.59 ms is a bandwidth FLOOR · the flag crashes ·
compile regresses. **The elementwise term is still real but needs the instrumentation and dynamic ops out
of the traced region, or a hand-written GatedDeltaNet kernel — NOT ~1 day.** Re-open only on a measured
plan for that term that does not route through `torch.compile` as the code stands. Detail:
[`…-row36-torchcompile-AB-RESULTS.md`](docs/2026-09-20-row36-torchcompile-AB-RESULTS.md); logs
`docs/probes/r36-abba-{1A,2B,3B}.log`, `docs/probes/r36-fuse-1.log`.

**Superseded by the A/B above — RE-SCOPE, ~1 day not ~1 week, and it is a §8 BANKED-FEATURE case:** `use_torch_compile`
(`vaayu/vaayu_base.py:2113`) already compiles the layer forward through `self._compiled_layer_fn` with
**`mode="reduce-overhead"` — torch.compile PLUS CUDA graphs** (`vaayu_base.py:5064`). It **defaults False**
and **`vaayu/server/config.py` has NO knob for it** (only a comment, `config.py:328`), so no user can reach
it. Next step: (1) A/B `use_torch_compile=True` on Qwen3-Next at N=1, ABBA, U-arm baseline; (2) if it pays,
plumb the knob in the SAME commit with a per-model gate (§5 forbids defaulting into a one-model-wins
asymmetry; `USE_PROMPT_LOOKUP="auto"` and `AdaptiveSpecGate` are the existing patterns); (3) separately,
pin the 41.7 pageable HtoD copies. **Caveats: one draw, one model (Qwen3-Next only — MiniMax and the dense
70B unmeasured), NOEXP arm only, CUPTI inflated the wall 18 % (on the profiled wall the split reads 63/37,
which does not change the verdict).** Probe `plans/probes/r36_serial_split.py` is THROWAWAY; nothing in
`vaayu/` was touched. Detail:
[`…-row36-serial-split-SPIKE.md`](docs/2026-09-20-row36-serial-split-SPIKE.md) (gitignored; this row is the
durable copy); log `docs/probes/r36-split-1.log`, JSON
`plans/probes/row26_out/r36split-qwen3next-r36split1.json`.

**Superseded by the spike above — NEXT STEP: write a plan (CLAUDE.md §4/§7) before any code. This run says the phase is LARGE, not what
removes it.** Baseline any row-36 A/B on the **U arm**, never `P` (row 27), with ABBA ordering (rows 33/44).

**The same run also answered row 46's open question — see that row: `disk_load` collapsed to 0.42 / 0.32 %
of `T_prof` (was 8.41 / 8.46 %), RAM hit 96.6 / 96.7 %, so the expert RAM cache is NO LONGER the binding
constraint on Qwen3-Next.**

**History — the gate had been read twice and never fired — 23.0 / 23.4 % (2026-09-13,
pre-30-P) then 26.4 / 26.7 % (2026-09-18, 30-P) — with **`serial` FLAT at ~0.036 s both times**; the share rose only
because the token shrank, and the row was derived to **fire once `T_unprof` ≤ ~0.120 s (~8.3 tok/s)**. **Row 44 and
row 30's RAM-ceiling fix then took Qwen3-Next decode to 9.06 tok/s — `T_unprof` = 0.1104 s, UNDER the trigger — which
puts the strict share at ≈ 32.6 % if `serial` has not moved.** Row 42-B's own A/B corroborates the token size
(9.4868 tok/s server-default / 9.7364 cache-off → 0.1054 / 0.1027 s → 34–35 %), but **that is `stream_generate`
end-to-end decode, not the U arm the gate is defined on — corroboration, not the gate.**
**It could equally have FALLEN:** row 44's F5 replaced the per-expert `torch.where` one-hot dispatch (~960 tiny H2D
copies per decode token) and that work is inside what the NOEXP arm brackets. Never measured post-row-44.
**Run:** two replicates of `systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 env PYTHONPATH=$PWD
MODEL=qwen3next BALLAST_GB=25 NEW_TOK=64 ARMS=W,U1,P,U2,NOEXP TAG=-r36recheck<N>
./venv/bin/python plans/probes/row26_decode.py`, one GPU job at a time; arithmetic via
`docs/probes/s3regate_table.py`. **strict = NOEXP `serial` / mean(U1,U2) `T_unprof`** — never the `P` arm
(`profiling_mode` runs FASTER than the real thing on fused-EXL3, ×0.85–0.87, row 27).
**Decision: ≥ 30 % → row 36 becomes the next build (~1 week), and nothing else on this board owns `serial`.
26–30 % → record the number here, do NOT build on a near-miss. < 26 % → `serial` FELL, which is itself new
information (it was flat across two builds) — say so here.** Write the number, the date and the build sha into this
row either way. The same run also decomposes the token, which is what re-priced row **46**.

`serial` is **0.074 s of Qwen's 0.930 s today — 8 %, invisible.** Row 29 projected that for a **resident** candidate the
per-layer serial term becomes the **largest** phase: **38 ms of Qwen3-30B-int4's projected 49 ms**. Nothing in this file
owns it.

**Gate: a resident model measures `serial / T` ≥ 30 %.** Below that it is a distraction, and the `NOEXP` arm of
`plans/probes/row26_decode.py` measures `serial` directly — use it, do not estimate.

**Trap:** `profiling_mode` runs **FASTER** than the real thing on fused-EXL3 (×0.85–0.87 on Qwen) because its
`synchronize()` spins and keeps the package awake (row 27). It is not an upper bound and must not be the baseline.

**§8:** (1) probably no knob. (2)/(3) reachable at N=1. (4) blocked on the gate above.

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

# 39 · [ ] Expert slimming — research tier (added 2026-09-12)

Prune the lowest-importance intermediate rows of each expert (structured row or N:M sparsity) from a calibration pass,
re-pack, stream fewer bytes per expert at unchanged architecture. Unlike row 32 it compounds with any bpw.

**Genuine research; quality risk real and unbounded. Priced honestly: a 30 % row prune is a 30 % byte cut — roughly what
downloading a 3.5 bpw build gets for free in an afternoon (row 32).** Recorded because the user asked for the aggressive
end of the space on 2026-09-12. **Do not start while rows 31-35 are unmeasured.**

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

- [x] **step 0 — code read, H1 ANSWERED 2026-09-20: TRUE on the decode path, FALSE on prefill and on
  the arena-exhausted fallback.** Durable summary (detail:
  [`…-row49-step01-RESULTS.md`](docs/2026-09-20-row49-step01-RESULTS.md)):
  **(a) `:1119` is dead for MiniMax experts** — all 47,616×4 expert tensors are `trellis/suh/svh/mcg`,
  every one in `_EXL3_QTENSOR_KEYS`, so 100 % of expert bytes take the `:1110` branch. **The whole
  0.18 s slice is ONE line: `loader_strategy.py:384`'s `t.to(device, non_blocking=non_blocking)`.**
  **(b) The decode source IS page-locked today.** `moe_supports_slab_arena` is True for
  EXL3+module_list (`vaayu_base.py:798`) and `_pin = torch.cuda.is_available()` (`:2732`); both runs
  today logged `Expert slab arena: 1259/1250 x 8.9MB … (pinned)`. RAM-cache hits return the slab
  views themselves (`get()` = `dict(...)`, no clone) and misses read io_uring straight into a slab.
  Checked directly: a view into a pinned chunk reports `is_pinned() == True`; a `.clone()` of one
  reports False. **(c) Measured cross-check: 309.8 experts/token × 8,875,020 B = 2.749 GB in
  0.1821 s = 15.1 GB/s = 77 % of this box's pinned H2D (19.7 GB/s), 2.1× the pageable rate
  (~7.1 GB/s)** — a pageable source would have put `h2d` at ~0.39 s. The residual is ~3,700 blocking
  per-tensor copies/token (588 µs per expert, 12 copies each), not bandwidth. **(d) The pageable
  sources are prefill (`_moe_prefill_pass` → `load_expert_sparse` → `.clone()`,
  `safetensor_model_persister.py:790`) and the arena-exhausted/`arena is None` fallback** — the
  22-27 GB RSS blow-up the comment fears came from exactly there, so step 2 must scope the flag to
  pinned sources, not flip it globally. **(e) The lifetime machinery already exists:**
  `SlabArena.record_read` (`expert_cache.py:230`) + `acquire()`'s event sync, which both fused
  forwards call after their async copies (`moe_patch.py:1989/2561/2582`), and the module-list forward
  already holds its pins until end-of-layer (`moe_patch.py:1746-1751`) — stricter than the fused
  unpin-after-issue. **So step 2 is a flag change + `record_read`, NOT a buffer-pool build.**
- [x] **step 1 — decode split re-measured 2026-09-20 on `4e5a088`, 2 replicates, 29G scope,
  foreground, `ARMS=W,U1,P,U2,NOEXP`, greedy ids identical U1/P/U2.** **`h2d` = 0.1821 / 0.1833 s =
  31.48 % / 31.47 % of `T_prof`** (0.5786 / 0.5825 s) — replicates agree to 0.02 pp. **STOP RULE
  (< ~15 %) NOT TRIGGERED: the slice did not shrink under rows 27/30-P/44/45** (row 26's own JSONs,
  re-run through the same script: 0.1861/31.26 % and 0.1801/30.20 %). Rest of the token:
  `gemm` 0.1366-0.1391 (23.6-23.9 %), `disk_load` exposed stall 0.0970-0.0991 (16.7-17.1 %),
  `other` 0.0961-0.0971 (16.6 %), `scatter` 0.0305-0.0312, `detach` ~0.010, `construct` ~0.008,
  `route` ~0.009, `mask_sync` ~0.006, `attach` 0.0022. **Decode itself is 3.0-3.2 % FASTER than
  2026-09-11** (`T_unprof` 0.5787/0.5773 → 0.5604/0.5624 s; U arms 1.778-1.790 tok/s), the gain
  sitting in `gemm` and `other`, never in H2D. 2.38 GB/token, 267.8 reads/token, RAM hit 13.6 %,
  VRAM 186.2 hit / 309.8 miss per token — all unmoved. By-product for row 36: `serial` (NOEXP) =
  0.0526/0.0543 s = **9.4-9.6 % of `T_unprof` on MiniMax**, nowhere near the ≥ 30 % trigger.
  **Step-2 ceiling as a LOWER bound:** the same 2.749 GB at the measured pinned 19.7 GB/s is
  0.182 → 0.140 s = **−0.042 s = +7.5 % decode** from killing the per-call sync alone; overlap
  against the 0.099 s disk stall and 0.137 s gemm is the upside beyond that (~+17 % if H2D vanished
  entirely). Measure it, do not project it. Logs `docs/probes/r49-split-{1,2}.log`, table
  `docs/probes/r49-split-table.txt`, script `docs/probes/r49_split_table.py` (module-list key set —
  `s3regate_table.py` reads only the fused forwards' `slots_hit`/`*_hit` keys and cannot parse these).
- [x] **step 2A — SHIPPED AND MEASURED 2026-09-20: +6.90 % MiniMax decode, logits BIT-IDENTICAL, no RAM
  leak.** Detail: [`…-row49-step2-RESULTS.md`](docs/2026-09-20-row49-step2-RESULTS.md). **Scope was A only**
  (approved 2026-09-20): derive `non_blocking` per expert from arena-backing, arm the read event. **B (a
  dedicated copy stream) was NOT built** — copies and gemm share the default stream, so A removes the HOST
  stall, not same-stream serialisation. **The change:** new `_copy_async_ok(pinned_source, tensor, dtype)` in
  `vaayu/moe_patch.py` (async iff page-locked AND no dtype change — a cast runs host-side and makes the real
  source a pageable temp); `_inject_expert_weights(..., pinned_source=False)` threaded into the
  `build_exl3_qmatrix` call and the plain loop; `_load_inject_run` derives it with
  `arena.slab_index_of(next(iter(raw.values())))` (pointer arithmetic, no CUDA call) and calls
  `arena.record_read(_rd)` after injection. Prefill clones and the arena-exhausted fallback are in no slab →
  None → stay synchronous, automatically. Plus the stale `non_blocking` paragraph in `build_exl3_qmatrix`'s
  docstring, and one pre-existing test stub's signature. **A/B: ABBA `fix,base,base,fix`, U arms only (row 27's
  trap), base = a git worktree at `4e5a088` proven by each log's import path — fix 0.5224 vs base 0.5584
  s/token = +6.90 % throughput (1.7907 → 1.9143 tok/s)**, within-arm spread 0.13 % / 0.57 %. **Correctness:
  prompt-end logits sha256 `3211568a…40dd18` IDENTICAL in all four runs; greedy ids identical in every arm;
  peak RSS 18.62/19.01 GB (fix) vs 18.94/19.04 GB (base)** at 4 Hz sampling — the leak the original comment
  feared does not happen from a pinned source. **Gates: suite 2348 passed / 16 skipped (baseline 2336/16),
  accumulator Qwen3-Next 8, accumulator MiniMax 8, server smoke Qwen3-Next 9, 1B prefix 1.** Dense has NO gate
  (`test_batched_dense_smoke.py` deleted 2026-09-03) — it does not reach this code by reading, not by
  measurement. **§8: (1) NO KNOB, deliberately — `config.py` untouched, a correct overlap is strictly better;
  (2) n/a, forward-internal; (3) REACHABLE — N=1 per decode token on the forward the server runs for every
  MiniMax request on its own defaults, unlike P8's N=3 win; (4) not blocked.**
- [x] **DONE 2026-09-21 — B RE-RANKED AND REFUTED: do not build it.** Detail:
  [`…-row49-B-rerank-RESULTS.md`](docs/2026-09-21-row49-B-rerank-RESULTS.md). Measurement only, no
  engine code. Two replicates of `row26_decode.py MODEL=minimax ARMS=W,U1,U2 BALLAST_GB=25
  NEW_TOK=64` on `4743325`, 29G scope, foreground, U arms only (the P arm re-serialises A's async
  copy), greedy ids identical; new analysis = the UNION of every expert read's `(t0, t0+dt)`
  interval inside a decode span, which the probe already recorded. **The drive is BUSY 75.7-76.4 %
  of a decode token and delivers 6.03-6.05 GB/s while busy, against a token-average of
  4.56-4.62 GB/s and P8's 7.1 GB/s plateau.** The union counts ring-wait, so **idle >= 23.6 % is a
  FLOOR and the rate headroom is a CEILING.** The stop rule did NOT fire either way (75-90 % straddle
  band); the decomposition resolved it instead. **The idle is ONE GAP PER MoE LAYER: 61.0-61.1 gaps
  >= 1 ms per token against 62 layers, p50 1.57 ms, p90 1.88 ms, holding 110 of the 123 ms.**
  Split: `token 0.5217 = disk busy 0.3988 + idle 0.1229`, and `idle = serial 0.0541 (NOEXP, same
  build) + exposed expert-phase GPU work 0.0688`. **B reaches only the 0.0688, minus `scatter`
  0.0317 and `construct` 0.0082 which no copy stream removes -> ~0.03 s/token = <= +6 %, for a 1-3
  day build.** A's own pass-through confirms the regime: step 2A removed 0.18 s of host block and
  delivered 0.036 s (**20 %**), the rest handed back as exposed stall. **Ranked ceilings from the
  same run: fill the idle >= +30.8/31.2 % (needs PREDICTION) · raise the busy rate to 7.1 GB/s
  <= +12.8 % (deterministic, hours) · B <= +6 % (last) · first two together +53.6 %.** Not claimed:
  per-depth rates — reads are 3.04 per expert and strongly non-uniform (dt p50 0.119 vs p90 5.20 ms),
  so a uniform-bytes depth model returns 8.47 GB/s at depth 2, above the drive's 7.45 spec.
- [x] **MEASURED 2026-09-21 — the ring-pool arm ran, the hypothesis was BACKWARDS, and it found a
  win one step the OTHER way.** Detail:
  [`…-row49-ringpool-AB-RESULTS.md`](docs/2026-09-21-row49-ringpool-AB-RESULTS.md). MiniMax, U arms,
  29G scope, foreground, ABBA (`4,3,3,4` then `2,3,2`), ids identical in every arm, bytes/token and
  reads/token identical across all eight runs. **Pooled s/token: n=1 0.5536 (−5.70 %) · n=2 0.5123
  (+1.89 %) · n=3 0.5220 (SHIPPED) · n=4 0.5328 (−2.07 %).** Both n=4 arms sit above every n=3 arm
  and both n=2 arms below every one, in both orderings; within-arm spread 0.04–0.40 %. **The
  mechanism is the delivered RATE, not the duty cycle: duty is flat at 75.5–76.3 % while the rate
  goes 6.08 → 6.03 → 5.90 GB/s**, and the in-flight histogram shows 3→4 moving span mass out of
  depth-1 (16.1 → 12.9 %) into depth 4–6 — more reads in flight, less bandwidth. n=1 fails
  differently (duty RISES to 78.1 %, rate collapses to 5.50). P5's arena trap ruled out (slabs
  1250–1259, uncorrelated — the slowest run has the largest arena). **This refines P8: its
  standalone probe measured "FLAT at ~7.09 GB/s for 2–4 rings" and the code took 3 from the middle
  of that zone (`safetensor_model_persister.py:181`, `ecb5c2f`); IN PATH the 2–4 zone is not flat,
  it falls monotonically, and never reaches 7.09.** The concurrency axis is now EXHAUSTED — single
  interior optimum, code one step off it.
- [x] **SHIPPED 2026-09-21 — `_NVME_READ_CONCURRENCY = 3 → 2`, one line, all three verification
  paths cleared FIRST (measured with `VAAYU_RING_POOL_N` as the arm variable, so no engine code
  moved until every one reported).** **MiniMax decode +1.89 % · MiniMax 2k prefill +1.85 %
  (39.515 vs 40.260 s, ABBA `2,3,3,2`, every n=2 arm below every n=3 arm) · Qwen3-Next decode NULL
  (0.45 %, arms interleave inside a ~3.5 % between-run spread) · Llama-3.3-70B dense NULL (3.2846 vs
  3.2895 s/token at 13.86 GB/token, inside the n=3 arm's own 1.6 % spread).** **Numerically inert,
  proven: ONE greedy-id sha256 per model across n=1/2/3/4** — Qwen3-Next `144f0e3f883e`, MiniMax
  `7a8c6f3021cc`, dense `963f6991fba0` — and the prefill probe's prompt-end `top` hash is identical
  in all four prefill runs. **Dense has no committed gate, so those dense arms ARE its coverage for
  this change.** Gates: suite **2348 passed / 16 skipped** (identical to baseline), accumulator
  Qwen3-Next 8, accumulator MiniMax 8, **server smoke Qwen3-Next 9 — and its log prints
  `persistent ring pool initialized n=2`, so the change is verified REACHING the server in the
  gate's own output**, 1B prefix 1. Also corrected the constant's comment block, which asserted P8's
  flat-2-4 zone as the reason for 3; and one test (`test_ring_pool_never_exceeds_the_caller_pool`)
  whose `pool=2 → 2` case the new constant made tautological now uses `pool=4`. **§8: no new knob
  (`VAAYU_RING_POOL_N` is the override), nothing to plumb, REACHABLE at N=1 on every request.**
- [x] **READ SHAPE — BUILT, MEASURED, REVERTED 2026-09-21 ON ITS OWN STOP RULE. The mechanism
  worked perfectly and bought NOTHING, which is why the result is worth more than the feature was.**
  Detail: [`…-row49-small-group-preload-RESULTS.md`](docs/2026-09-21-row49-small-group-preload-RESULTS.md).
  **Characterisation first, off the checkpoints' own `vaayu_expert_offsets.json` through
  `_group_contiguous_ranges`' own rule, no GPU:** both MoE checkpoints store each expert as
  **EXACTLY 3 contiguous file groups**, one bandwidth-bound and the rest latency-bound —
  MiniMax 47,616 groups of which 31,744 are < 64 KiB (**66.7 % of reads, 439.0 MB resident,
  0.31 % of expert bytes**; per expert `trellis` 8.85 MB + `suh/svh` 31.7 KB + `mcg` 4.1 KB,
  useful 8,875,020 B vs aligned 8,887,299 B = **amplification 1.0014**), Qwen3-Next 73,728 of
  which 24,576 (33.3 %, 1.2 MB). That confirms the 3.04 figure exactly: 271 loads/token × 3 =
  the 814 rust reads/token the duty probe counted.
  **COALESCING REFUTED BEFORE THE BUILD, do not retry:** the `mcg`→`suh` gap grows linearly with
  expert id — 29.7 KB at expert 0, **4.84 MB at expert 255** — so no fixed gap tolerance merges
  them without reading megabytes of other experts. (Row 44's F4 died on the OTHER forward for a
  different reason: adjacent-id runs averaged 2.2 experts at 55 % density.)
  **BUILT:** `_NVME_SMALL_READ_BYTES = 64 KiB` hardware constant + a resident table `copy_`d into
  exactly the destination the read would have filled, so slab layout / `prefix_pad` / views /
  `slab_span` / arena backing / step 2A's async-H2D decision were unchanged **by construction**;
  17 tests over two geometries, suite 2365 passed / 16 skipped (baseline 2348/16); engine knob
  `SMALL_READ_PRELOAD` plumbed and documented; engine logged `439.0MB resident (3.9 % of the
  expert cache budget)` on every fix arm — the index prediction landed exactly.
  **MEASURED, ABBA `fix,base,base,fix`, U arms only, 29G scope, foreground, one build with the
  knob as the arm variable:** **reads/token 803 → 268 (−66.7 %)**, **delivered-while-busy
  6.07 → 6.50 GB/s (+6.9 %)**, **disk BUSY 75.8-76.5 % → 70.6 % (−5.4 pp)**, bytes/token
  unchanged — **and decode −0.40 %** (pooled 0.51455 base vs 0.51660 fix; arms INTERLEAVE, the
  fastest pair is a base pair and the slowest a fix pair; within-arm spread 0.10-0.33 %,
  between-run spread on base alone 1.0 %). **The 26 ms/token of freed disk busy went straight
  into idle (0.125 → 0.152 s/token); exposed stall did not move.**
  **The gap census says it twice:** idle gaps ≥ 1 ms are UNCHANGED (3835/3840 base vs 3828/3834
  fix, the same ~61/token, one per MoE layer). What vanished is ~98 **sub-millisecond** gaps per
  token (158.8 → 60.9; median gap 0.09 → 2.22 ms). **The small reads were filling the small gaps.
  The structural per-layer gap never belonged to the drive.**
  **THIS REFUTES THE RATE LEVER BY DIRECT EXPERIMENT.** The B re-rank priced 6.03 → 7.1 GB/s at
  ≤ +12.8 %; this moved the rate **39 % of that distance and paid −0.40 %**. That ceiling was
  computed by holding duty fixed while busy shrank — **duty did not hold.**
  **Not claimed:** cross-arm greedy-id identity on device — `row26_decode.py` compares ids only
  WITHIN a run (identical to U1 in every arm of all four) and never emits them. Byte identity is
  proven structurally by the tests instead. A future build of this idea must add that check.
  **Archived, not deleted:** `docs/refuted/2026-09-21-row49-small-group-preload.diff.refuted`
  (the whole implementation) and `docs/refuted/test_small_group_preload.py.refuted` (the 17
  tests). Rebuild only if something makes the token track disk busy again.
  **§8, answered before the measurement:** it had a knob, it was plumbed, it was reachable at
  N=1 on every MiniMax request. **Reachability was never the problem; the effect was.**
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
- [ ] step 3 — re-price the CPU hybrid, only after step 2. The stop rule did NOT close it; its own
  verdict below (MiniMax-only, EXL3 has no CPU kernel, weeks) is unchanged by these numbers.
- [ ] **NOT IN SCOPE, recorded so it is not lost: MiniMax PREFILL's H2D is pageable** (step 0(d);
  row 45 measured h2d at 59.8 % of prefill wall). H1 does not cover it. Either pin the prefill clone
  path or leave it synchronous — decide after step 2, it is a separate question.

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

# 48 · [bug] `use_torch_compile=True` CRASHES on CUDA — and `config.py` declines it for the wrong reason (added 2026-09-20, found by row 36's A/B)

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

# 41 · [x] Speculative decoding corrupted a hybrid model's recurrent state — CLOSED 2026-09-18: CONFIRMED (garbage from the first drafting round), now REFUSED on recurrent caches (added 2026-09-18)

**Status: `[x]` CLOSED 2026-09-18 — CONFIRMED on device, WIDER than filed, FIXED (refusal) the same day.
The same root cause is a separate, DEFAULT-reachable bug in chunked prefill: row **43**.** Results:
[`…-row41-hybrid-multitoken-state-RESULTS.md`](docs/2026-09-18-row41-hybrid-multitoken-state-RESULTS.md) (gitignored).
**Widened by the investigation, same day (was the hypothesis; now measured, see the confirm bullet):** transformers 5.5.4's `Qwen3NextGatedDeltaNet.forward`
(`modeling_qwen3_next.py:603-691`) continues from cached conv/recurrent state ONLY when `seq_len == 1`; any multi-token
forward on a non-empty cache runs `chunk_gated_delta_rule(initial_state=None)` + a zero-history conv and then
OVERWRITES the cached state. If that holds, (a) every speculative verify pass (k+1 tokens) is wrong, not just rejected
rounds, and (b) Vaayu's own chunked prefill (`_chunked_prefill`, layer-major, `PREFILL_CHUNK_SIZE=512` server default)
restarts the 36 linear layers' state at every chunk for any prompt > 512 tokens -- DEFAULT-reachable. Repro in progress. Brief (self-contained, gitignored):
[`…-hybrid-model-next-steps-BRIEF.md`](docs/2026-09-18-hybrid-model-next-steps-BRIEF.md) item 1.

- **Mechanism (file:line at `9c4f14f`):** the speculative loop's Step 5 calls
  `_trim_kv_cache(past_key_values, past_len_before + num_accepted)` (`vaayu/vaayu_base.py:5700`). That calls
  `cache.crop()` (`:5171` → `:1544`), which only lowers the KV ring buffers' `_seq_len`. The linear-attention layers'
  conv/recurrent state (built at `:1361` / `:1406`) is never touched. So on a partially-accepted draft, Qwen3-Next's 36
  recurrent states still include the rejected tokens. Wrong text, no exception, which is exactly what
  `_trim_kv_cache`'s own docstring says must raise instead.
- **Detector exists, unused here:** `cache_holds_recurrent_state` (`vaayu/batch.py:501`). RC7 uses it for the prefix
  cache via `prefix_cache_refusal_reason` (`batch.py:521`, WARNING at `vaayu_base.py:~3010`).
- **Reachable:** only by opt-in. `USE_PROMPT_LOOKUP = "auto"` (`config.py:380`) → off for MoE
  (`resolve_use_prompt_lookup`, `vaayu_base.py:1328`); `USE_PROMPT_LOOKUP = True` + Qwen3-Next reaches it. Check
  `DRAFTER_MODEL_PATH` too. It bites only on rounds that draft AND are partially accepted (MoE's adaptive gate may
  latch drafting off).
- [x] **CONFIRMED 2026-09-18.** CPU, layer level: a 12-token forward on a cache that already holds 12 tokens of state
  equals the same forward on a FRESH cache **bit for bit (0.000)** and differs from the full-sequence forward by 6.12
  (output max 5.68); one-token-at-a-time matches (4.5e-6) — the state is DISCARDED, not merely polluted. GPU, real
  Qwen3-Next through `VaayuBaseModel.forward` (`plans/probes/row41_hybrid_state.py`, `VAAYU_SPEC_FORCE=1`,
  `draft_tokens=5`, `BALLAST_GB=25`): one fully-accepted 5-token verify forward moves the next logits **rel-L2 0.743 vs a
  0.051 numerics floor (14.6×)**, NLL 3.418 → 4.231 over 48 teacher-forced tokens; the real `speculative_generate` on a
  163-token "repeat this code" prompt emits `'def is a is a 1, is a 10 is a 100 …'` where greedy copies the code
  verbatim — **diverges at token 1, the first drafting round** (40 drafting rounds, 8 fully accepted, 32 partial).
  Root cause: transformers 5.5.4 `Qwen3NextGatedDeltaNet.forward` (`modeling_qwen3_next.py:603-691`) uses the cached
  conv/recurrent state only when `seq_len == 1`; a multi-token forward runs `initial_state=None` + a zero-history conv
  and OVERWRITES the cache. The rollback gap the row was filed for is real too (`FP8DynamicCache.crop` touches only
  `_seq_len`; transformers' `LinearAttentionCacheLayerMixin.crop` is `pass`) but is downstream of the reset.
  Original plan text: a CPU check first (`crop()` leaves conv/recurrent tensors unchanged). Then one GPU run: two
  processes, `use_prompt_lookup` False vs True, same repetitive greedy prompt, compare ids and read
  `model.speculative_status`. Ids differ after the first partial acceptance → confirmed. Zero drafting rounds →
  inconclusive; force drafting.
- [x] **FIXED 2026-09-18.** `spec_recurrent_refusal` (`vaayu/batch.py`) runs at load after the `VAAYU_SPEC_FORCE` hatch (so
  it cannot be lifted): `refused_because` set, one WARNING ("…DISABLING speculative decoding for this model"), no
  proposer built. `_trim_kv_cache` raises `TypeError` on a recurrent cache (backstop). Tests in
  `tests/test_hybrid_recurrent_state.py`. Gates: suite 2288 passed / 0 failed; server gate on Qwen3-Next with
  `SERVER_USE_PROMPT_LOOKUP=1` (new gate env) 9 passed and logs the refusal; server gate on Llama-3.2-1B 9 passed with 4
  drafting rounds (dense still drafts); accumulator gates Qwen3-Next / Qwen3-235B / MiniMax 8/8/8. Docs: `server.md`,
  `server/README.md` (`DRAFTER_MODEL_PATH`, `USE_PROMPT_LOOKUP`). §8: no knob; reachable at N=1 (opt-in); a refusal, 0 %.
  Original plan text: refuse speculation at load when the cache holds recurrent state (status
  `refused_because` + one WARNING, mirroring RC7), and make `_trim_kv_cache` raise on a recurrent cache. RED test
  first (fake cache exposing `conv_states`). Gates: full suite; server gate on Qwen3-Next with `USE_PROMPT_LOOKUP=True`
  logs the refusal; Llama-3.2-1B still drafts.
- **§8:** no new knob (existing `USE_PROMPT_LOOKUP` / `DRAFTER_MODEL_PATH`); reachable at N=1; a refusal, 0 % speed.
  The speed-up is row 42's question.
- **If refuted:** close with where the state IS restored. Row 42 then gets simpler. *(Not refuted — confirmed.)*
- **Consequence for row 42:** the exact continuation fix (row 43, option A, BUILT 2026-09-18) makes the verify LOGITS right but still
  cannot roll back a rejected draft, so speculation on hybrids stays refused until row 42's snapshot exists. And row
  42's prefix-cache design (restore a snapshot, prefill the diff) IS a multi-token forward on a non-empty recurrent
  cache — it needs row 43's option A first.

---

# 43 · [x] Chunked prefill restarted every linear-attention layer at each chunk — CLOSED 2026-09-18: FIXED, exact continuation, at the numerics floor (added 2026-09-18)

**Status: `[x]` CLOSED 2026-09-18 — CONFIRMED on device by row 41's investigation and FIXED the same day (user chose
option A). Was: correctness, silent, on the server's default path for every Qwen3-Next prompt > 512 tokens.** Results:
[`…-row41-hybrid-multitoken-state-RESULTS.md`](docs/2026-09-18-row41-hybrid-multitoken-state-RESULTS.md) (gitignored; this row is the durable copy).

- **Mechanism:** same root cause as row 41 (transformers' gated-delta-net continues from cached state only at
  `seq_len == 1`). `VaayuBaseModel._chunked_prefill`'s layer-major path (MoE) runs `layer(seq)` once per chunk against
  one cache, so chunk *j ≥ 1* restarts all 36 linear layers from zero. After prefill they have seen ONLY the last
  chunk; only the 12 full-attention layers carry earlier context — why text stays fluent and S2 / the server gate /
  row 33 never flagged it.
- **Measured** (real Qwen3-Next, 1100-token code prompt → chunks 512/512/76, server knobs): logits at the prompt end
  **rel-L2 0.314 vs REF against a 0.051 numerics floor (6.2×)**; NLL over the next 48 teacher-forced tokens 3.418 →
  3.579 (+0.16 nats); top-1 agreement with REF 95.8 % (floor) → 77.1 %. One draw at 2 chunk boundaries; longer prompts
  lose more context (unmeasured).
- **Reachable:** default. `PREFILL_CHUNK_SIZE = 512` (`server/config.py:132`); every Qwen3-Next request with a prompt
  > 512 tokens — i.e. most agent-workload requests. Same `seq_len == 1` gate in transformers' qwen3_5, qwen3_5_moe,
  bamba, falcon_h1, jamba, granitemoehybrid, olmo_hybrid, zamba — any hybrid Vaayu loads next inherits it.
- **Affected numbers: NONE.** Checked 2026-09-18: no Qwen3-Next log under `docs/probes/` except this row's contains a
  `Chunked prefill` line — row 33's quality arms, S2, 30-P and the S3 re-gate all ran prompts ≤ 512 tokens.
- [x] **FIXED 2026-09-18 — option A.** `vaayu/recurrent_state.py`: `install_state_continuation` wraps each gated-delta-net
  mixer (detected by surface, not name) so a multi-token forward on a cache holding its state runs the STOCK forward
  with the cached last k-1 conv inputs prepended (via `causal_conv1d_fn`, fast path or not) and the cached recurrent
  state as `initial_state`, then rewrites the conv state as the last k inputs (the stock update zero-fills it for a
  < k-token tail). Installed at the end of `init_model` (Qwen3-Next logs `36 linear-attention mixers continue …`).
  `continuation_refusal` makes a multi-chunk prefill on a hybrid whose mixers are NOT all wrapped raise
  `NotImplementedError` naming `PREFILL_CHUNK_SIZE=0` — any other hybrid family fails loudly, never silently.
  27 CPU tests (`tests/test_hybrid_recurrent_state.py`): split == one forward for qwen3_next AND qwen3_5 × both cache
  classes × 3 split shapes incl. a 2-token tail, final state included; RED watched (18 assertion failures on stubs).
  **On device, same probe:** CHUNK rel-L2 **0.314 → 0.041** at 1100 tokens (floor 0.051), **0.020** at 3000 tokens / 6
  chunks (floor 0.041); NLL 3.579 → 3.410 (REF 3.418); top-1 vs REF 77.1 → 95.8 %. VERIFY 0.743 → 0.056. REF/NOISE
  bit-identical to the pre-fix run. Gates as row 41. Docs: `PREFILL_CHUNK_SIZE` rows in `server.md` / `server/README.md`.
- **Design options as weighed (kept for the record):** (A) exact continuation for gated-delta-net mixers:
  pass the cached recurrent state as `initial_state` (the torch kernel supports it; right-padding is state-neutral)
  and prepend the cached conv inputs; keeps the 512-token memory bound; prerequisite for row 42's hybrid prefix cache;
  detected by capability, covers qwen3_next / qwen3_5 / qwen3_5_moe. (B) do not chunk when the cache holds recurrent
  state: trivial and model-agnostic, but drops the prefill memory bound (long prompts may OOM below the derived
  `max_seq_len` 65536). Either way: a model-agnostic guard that raises on a multi-token forward over a non-empty
  recurrent cache unless the mixer is known to continue correctly.
- **Done rule:** re-run `plans/probes/row41_hybrid_state.py`; the CHUNK arm must fall to the NOISE floor (rel-L2 ≈
  0.05) and a CPU test must pin chunked == single-forward on a tiny hybrid. Gates: full suite; server gate on
  Qwen3-Next; Qwen3-235B / MiniMax / Llama-3.2-1B regression gates unchanged (no recurrent state → untouched path).
- **§8 (answered):** no knob (continuation is always right). Plumbed: `init_model`, which every load runs, including
  `create_app`'s. Reachable at N=1 on the default config — delivered to every Qwen3-Next user; 0 % speed, correctness.

---

# 42 · [x] SCOPING spike: recurrent-state snapshot/restore for hybrid models — unblocks prefix cache + speculation (added 2026-09-18)

**Status: `[x]` CLOSED 2026-09-19 — (a) prefix cache GO → row 42-B; (b) speculation NO-GO → row 37 closed for
hybrids; side finding → row 44. No engine code.** Results (gitignored; this row is the durable copy):
[`…-row42-hybrid-state-snapshot-SCOPING.md`](docs/2026-09-19-row42-hybrid-state-snapshot-SCOPING.md). Probe
`plans/probes/row42_snapshot.py` (+ `row42_report.py`, `row42_oracle.py`), logs `docs/probes/row42-*`. Server knobs,
`BALLAST_GB=25`, one `systemd-run` scope (`MemoryMax=29G`), flash-attn-2, HEAD `b13e82d`. Brief: [`…-hybrid-model-next-steps-BRIEF.md`](docs/2026-09-18-hybrid-model-next-steps-BRIEF.md) item 3.

- **Why:** Qwen3-Next (hybrid: 36 linear + 12 full-attention layers) is the fastest model on this box (7.34 tok/s,
  N=1). Two features are unreachable on it because its recurrent state cannot be cropped:
  - **(a) the cross-turn prefix cache**, refused by RC7. CLAUDE.md §0: the top TTFT lever for the agent workload.
    Measured elsewhere: ~10× round-2 TTFT (EXL3), ~4.5× at 2.8k tokens (1B, row 3).
  - **(b) speculative decoding (row 37)**, whose gate is MET (S2, 2026-09-13). The checkpoint carries a free MTP drafter
    (4,625 `mtp.*` keys, ~0.986 GB, never built by transformers 5.5.4).
- **Why it may be cheap (derived from `config.json`, confirm dtype on device):** recurrent state per linear layer =
  32 value heads × 128 × 128 = 524,288 elements; conv state ≈ 24.6k; × 36 layers → **~38 MB/snapshot bf16, ~75 MB
  fp32**. Against ~3.2 GiB VRAM headroom.
- [x] **Snapshot mechanics.** State lives in `cache.layers[i].conv_states` `[1, 8192, 4]` and `.recurrent_states`
  `[1, 32, 128, 128]`, both **bf16**, on 36 of 48 layers (detected by capability). **One snapshot = 40.1 MB.** Clone
  GPU→GPU 0.24 ms, restore 0.21 ms; GPU→pinned 2.5 ms, pinned→GPU 2.1 ms. Restore is EXACT: prefill 1024 → snapshot →
  40 divergent tokens → restore + crop → prefill 128 gives logits **bit-identical (rel-L2 0.0)** to a fresh 1152-token
  prefill (noise floor 0.0124). KV is 24,576 B/token (12 full-attention layers).
- [x] **(a) TTFT, measured (2 reps, order reversed):** MISS(2048) **69.4 s**, MISS(8192) **269.3 s** (33–34 ms/token;
  MISS(16384) ≈ 540 s derived). HIT(P, d) = restore + crop + prefill d, each after 32 decode tokens: at P=8192, d=64 /
  256 / 1024 → 7.76 / 13.04 / 35.24 s = **35.0× / 21.3× / 8.6×** vs MISS(P+d); at P=2048 → 9.5× / 6.3× / 3.2×.
  **Stop rule ≥ 2×: MET → GO.** Append-only turns need no snapshot (end-of-cache state is right when the match reaches
  the cache length); divergent turns need one at the prompt end; cross-session borrow needs one at the shared point
  (512-token chunk boundaries: 40 MB each).
- [x] **(b) Verify cost, measured** with the build's pass classification (`compute_prefill_pass` patched so ≤ 8-wide =
  decode), cold vs cold (verify-first verify vs decode-first decode, same tokens via snapshot/restore, reps 2–5): w=2 /
  3 / 4 / 6 / 8 → **2.66 / 2.63 / 3.44 / 3.89 / 4.78 × t_dec** (142.9 ms). Rollback R1 (restore + replay only the 36
  mixers on captured inputs): 14 ms for 1 token, ~81 ms for ≥ 2; state rel-L2 0.009–0.030 vs a real forward (floor
  unmeasured). **Prompt-lookup acceptance (exact greedy replay):** verbatim-copy edit ×1.30 naive (k=5) / ×1.42 gated /
  ×1.55 oracle; real edit (type hints) ×0.81 / ×1.08 / ×1.35; novel ×0.55 / ×0.93 / ×1.07. **Any drafter (MTP) needs
  per-token acceptance p ≥ 0.874 at k=5 (0.902 at k=4) for +10 %** — MTP unmeasured (needs new modeling code).
  **Stop rule +10 %: NOT MET on realistic output → NO-GO; row 37 closed for hybrids.**
- [x] **RAM/VRAM:** snapshots are negligible (40 MB/session). The real cost of (a) is the **prefix-session reservation
  returning**: 3 × 65536 × 24,576 B = **4.83 GB** off the expert RAM tier → row 30's +15.09 % reversed ≈ **−13 % decode**
  (derived). Break-even ~13,300 generated tokens/turn at 8k/d=256, ~1,950 at 2k/d=1024 → net win for the agent
  workload; the build plan must question the full-length 3-session sizing.
- **Side finding → row 44:** Qwen3-Next prefill reads 2.19× its experts at ~1.13 GB/s (2k tokens: 96.1 GB, 53,938 loads,
  84.7–85.5 s, RAM tier `usage=0.00GB`).
- **Stop rules:** (a) go if the projected round-2 TTFT on a realistic agent turn is ≥ 2× → write a build plan. (b) go
  only if the projected decode gain is ≥ +10 % after verify cost; else close row 37 for hybrids. Both no-go → close
  this row with the numbers.
- **§8 (answered):** scoping only, nothing shipped. (a) no new knob (`ENABLE_PREFIX_CACHE` / `PREFIX_SESSIONS` exist,
  plumbed `app.py:406,420`), reachable at N=1 once built → banked as row 42-B. (b) nothing to ship; refusal (row 41)
  stands.
- **§5:** detect by capability (`cache_holds_recurrent_state`), derive sizes from config, never special-case
  `qwen3_next`.

---

# 44 · [x] CLOSED 2026-09-19 — Qwen3-Next prefill re-read its experts on every prefill chunk, pageable, at ~1.1 GB/s — 33 ms/token TTFT (added 2026-09-19, found by row 42)

**Status: `[x]` DONE 2026-09-19 — F1 + F2 + F6 + F5 SHIPPED (`85478ef`, `c289945`, `4d2fa99`, `736c4c6`):
**2k TTFT 83.51 → 18.13 s (4.61×, 40.8 → 8.9 ms/token), 8k 335.26 → 37.20 s (9.01×, 40.9 → 4.5 ms/token), and
decode 7.95 → 9.06 tok/s (+14 %) as a side effect of F5.** Logits bit-identical at both lengths in every arm;
suite 2311; every on-device gate green. F4 (coalesced reads) built and REVERTED on its stop rule. F3 survives only
as its pinned-H2D half (~0.6 s at 2k) — left open below, not recommended. What remains at 2k is compute: h2d 4.5,
construct 5.2, gemm 4.7, scatter 3.1 (profiled).** Found by row 42's TTFT arms, not a planned
measurement: Qwen3-Next's TTFT had never been recorded (S2 / 30-P / row 33 / S3 all measured decode).
Detail: [`…-row42-hybrid-state-snapshot-SCOPING.md`](docs/2026-09-19-row42-hybrid-state-snapshot-SCOPING.md) §4
(gitignored; this row is the durable copy).

- **Measured (server knobs, `BALLAST_GB=25`):** MISS(2048) 69.4 s, MISS(8192) 269.3 s → **33–34 ms/token**. Engine
  telemetry for one 2048-token prefill: `this_token=0hit/53938miss`, `bytes=96,107,253,760`, 84.7 s cold / 85.5 s
  warm → **~1.13 GB/s**, **2.19×** the model's 24,576 experts, RAM tier `usage=0.00GB/18.10GB` afterwards. 512 tokens:
  25.25 GB, 14,172 loads, 23.7 s. Row 41's logs agree (1100 tok 34.6 s, 3000 tok 87.7 s).
- **Cause (code reading, to be instrumented first):** (1) `_make_fused_sparse_forward` (`moe_patch.py:1712`, the
  non-EXL3 fused forward) has **no chunk-major retention** — `_chunk_major_gpu` (`moe_patch.py:2317–2601`) exists only
  in `_make_fused_sparse_forward_exl3`, so the layer-major prefill (one layer call per 512-token chunk) routes and
  loads each chunk's experts again; (2) under `_moe_prefill_pass` every load is `load_expert_sparse`
  (`_dispatch_expert_misses`, `moe_patch.py:~2056`) — transient, pageable, never cached. RAM-tier HITS are served
  (`expert_cache.get` runs first; the tier was ~empty in row 42's arms), **but hits do not change the rate: row 41's
  arms had 53–66 % RAM hits and ran at the same ~31 ms/token** → the bottleneck is per-expert work in the prefill loop
  (53,938 iterations = 1.58 ms each; telemetry books all 85 s as `compute`), not NVMe.
- **Derived ceiling (unmeasured, disk physics only):** load each expert once per prefill, serve RAM-tier hits, stream
  the rest at the P8 plateau (~7 GB/s) → a 2k prefill in **~5–10 s** vs 69 s — reachable only if the per-expert loop
  cost goes too. Measure before building: instrument where the 85 s goes
  (read vs H2D vs dequant vs per-expert GEMM loop — 53,938 per-expert Python iterations alone may be seconds).
- **§5 / scope:** a fix to ONE forward (non-EXL3 fused). Check whether the module-list forward (MiniMax) has the
  same per-chunk re-read before assuming EXL3 is unaffected; fused-EXL3 already retains.
- **Interaction:** 42-B's hit prefill runs at the same rate (d=1024: 31–34 ms/token; d=64: 7.5 s). This row shrinks
  MISS and HIT together → re-price 42-B after it.
- **§8:** no knob expected; every Qwen3-Next request, N=1, default config — reachable the day it lands.
- [x] Instrument one 2k prefill: split the 85 s into disk read / H2D / dequant / GEMM loop. **DONE 2026-09-19**
  (`plans/probes/row44_prefill_split.py`, logs `docs/probes/row44-prefill-split-{r1,pool8}.log`, server knobs,
  `BALLAST_GB=25`, 29G scope, HEAD `64d868f`; detail [`…-row44-qwen3next-prefill-RESULTS.md`](docs/2026-09-19-row44-qwen3next-prefill-RESULTS.md)).
  2k prefill U / U2 = **86.26 / 84.62 s**, top id 526 every arm. MoE blocks = **99.3 %** of wall; all 48 mixers
  together 0.52 s (the torch gated-delta fallback is NOT the cost). P arm (89.96 s): **disk_load 59.16 s (65.8 %,
  1.10 ms/expert-call)**, h2d 10.35 (pageable, 192 µs), construct 5.67 (105 µs), scatter 3.43, gemm 3.28,
  unbracketed loop ~7.3. 53,938 loads vs a per-layer union across the 4 chunks of **17,445 → 3.09× re-reads**.
  **RC-A (NEW, not the row's hypothesis): the expert loader runs ONE worker on every non-EXL3 checkpoint.**
  `init_model_io` sizes each pinned loader buffer at `(32 if exl3 else 512) MB` (the 512 "unverified") against a
  `max(0.5, avail − 27 GB)` budget that is always 0.5 GB on this box → **1 buffer** (log: `n=1 size=512.0MB`) →
  `_moe_executor` width 1, ring pool 1, stager executor 1: every Qwen3-Next expert read, prefill AND decode, is
  serial. Qwen3-Next's largest expert is ~1.8 MB. **Override test (probe pre-fills 8 × 8 MB buffers, engine
  untouched): U 86.26 → 57.07 s (−34 %), W 55.99, P 61.91, top id 526** — disk_load 59.16 → 31.60 s, every other
  phase unchanged. Disk then runs at only **3.0 GB/s aggregate** (summed in-call 250 s over 8 callers; the Rust read
  releases the GIL) → suspect the per-tensor pageable `clone()` in `load_expert_sparse` (lever F3, unmeasured).
  **RC-B (the row's hypothesis, confirmed): no chunk-major retention in the non-EXL3 fused forward** → 3.09× the
  loads at 2k, and disk + h2d + construct (84 % of P) all scale with it. Row 41's "hits don't change the rate" is
  explained: a hit still pays pageable h2d + construct + loop per chunk call, and misses were still serial.
  **Also found:** the module-list forward (MiniMax) has no cross-chunk retention either (code reading; MiniMax is
  EXL3, pool 8, RC-A does not apply; unmeasured). Dense checkpoints allocate the unused expert loader pool too
  (1 × 512 MB pinned non-EXL3, 8 × 32 MB EXL3) — pre-existing, unmeasured.
- [x] **F1 — derive the loader buffer size from the expert offset index (`compute_slab_bytes`), not the 32/512 MB
  literals (CLAUDE.md §5). DONE 2026-09-19 (uncommitted):** `expert_read_buffer_bytes()` in
  `init_model_io` (dense → old sizing); tests `tests/test_expert_loader_buffer_sizing.py` (RED on the literal),
  suite **2299 passed / 16 skipped**. Qwen3-Next now `pool n=8 × 1.7 MB` (13.6 MB pinned, was 1 × 512 MB), rings 3.
  **Same-day ABBA (F1a → base stashed → F1b): TTFT 2k 83.51 → 55.82 s (−33.2 %), 8k 335.26 → 223.65 s (−33.3 %,
  cold tiers), decode 8.009 → 8.171 tok/s (+2.0 %); top ids identical, decode greedy ids identical 9 arms × 3
  processes.** 8k re-read factor: 214,717 loads vs union 19,696 (10.9×) — F2's target. Logs
  `docs/probes/row44-f1ab-*.log`. Measured −34 % at 2k by override first. **Gates (logs `docs/probes/row44-f1-*.log`):
  server Qwen3-Next 9 passed; accumulator Qwen3-Next / Qwen3-235B / MiniMax 8 / 8 / 8; Llama-3.2-1B prefix 1 passed;
  no offset-index file written into any dense dir.** Derived buffers (`splitted_model/`): Qwen3-235B 11.29 MiB,
  MiniMax 8.48 MiB, Qwen3-Next 1.70 MiB → pool 8 on all three; dense → 0 → old sizing (unchanged). EXL3 pool count
  and ring count do NOT move (8 / 3); their pinned pool shrinks 256 → ~90 / ~68 MB — no throughput A/B run on EXL3
  (nothing on their token path changes). **§8: no knob (derived); plumbed by construction (`create_app` →
  `from_pretrained` → `init_model_io`); reachable at N=1 on the default config, every MoE load. No `documentation/`
  page describes the loader pool → no doc change.** Shared infra (every MoE checkpoint: EXL3 buffers
  shrink 32 → ~11.8 / ~8.9 MB, count stays capped at 8) → all three regression gates. Dense unchanged (no experts
  → old sizing). §8: no knob; `create_app` → `from_pretrained` → `init_model_io`, reachable at N=1 by default.
- [x] **F2 — chunk-major retention for the non-EXL3 fused forward. DONE 2026-09-19 (uncommitted).** Per-layer
  `_chunk_major_gpu` dict + `_chunk_major_owner=(layer_idx, calls_seen)`, active only when `_moe_prefill_pass` and
  `_chunk_major_n > 1`; retained entries are the same object the VRAM cache stores, joined into `_vram_hold` so the
  hit branch serves them and no read is dispatched; bounded by the prefill-idle VRAM cache budget (7.2 GB on
  Qwen3-Next); dropped at the layer's last chunk call. Tests `tests/test_fused_nonexl3_chunk_retention.py` (RED at
  22 loads vs union 8); suite **2306 passed / 16 skipped** (`test_no_unbound_locals` caught `_cm_budget` — initialised,
  not suppressed). **Same-day ABBA (F2a → F1-only stashed → F2b): TTFT 2k 56.16 → 24.84 s (−55.8 %), 8k 225.48 →
  58.11 s (−74.2 %); expert loads 53,938 → 17,445 (2k) and 214,717 → 19,696 (8k) = the per-layer union exactly;
  decode 7.956 → 8.049 tok/s (+1.2 %, F2 idle at decode); prompt-end logits sha256 IDENTICAL across arms at both
  lengths.** Peak VRAM 5.52 GiB / 8.74 GiB free at 8k (prefill-only process). Gates: server Qwen3-Next 9 passed;
  accumulator Qwen3-Next / Qwen3-235B / MiniMax 8/8/8; Llama-3.2-1B prefix 1 passed. Plan:
  [`…/plans/2026-09-19-row44-F2-chunk-retention-nonexl3.md`](docs/superpowers/plans/2026-09-19-row44-F2-chunk-retention-nonexl3.md).
  **§8: no knob; layer-major chunked prefill is the default for MoE on flash/sdpa → every Qwen3-Next request over
  512 tokens gets it at N=1. No `documentation/` page states a Qwen3-Next TTFT number → no doc change.** Projected 2k ~57 → ~27 s (per-load
  phases × 17,445/53,938 = 0.323; gemm/scatter/loop unchanged). Bigger at 8k (16 chunks).
- [ ] **F3 — PROBED 2026-09-19, PREMISE REFUTED AS STATED; a reduced version survives.**
  `plans/probes/row44_f3_clone.py` (real expert ranges, median 1.69 MiB, O_DIRECT, 3 rings; log
  `docs/probes/row44-f3-clone-r1.log`) measured three read paths at k = 1/2/4/8 callers: RAW (io_uring only)
  1.79 / 2.81 / 3.55 / **3.69** GB/s · SPARSE (today: pooled buffer + per-tensor clone) 1.83 / 2.82 / 3.39 /
  **2.48** · SLAB (pinned, zero clone) 1.83 / 2.76 / 3.55 / **3.74**. **The clone is NOT the cap — read SIZE is:**
  a 1.69 MiB read tops out at ~3.7 GB/s with no tensors at all, against P8's ~7.1 GB/s at 11.8 MB. The clone costs
  only at k=8 (−34 %), which is the width prefill dispatches at, and is free at k=4. H2D per expert: pageable
  **147.3 µs** vs pinned **114.3 µs** (−22 %; NOT 30-P's 32–40 µs — different measurement shape).
  **Post-F2 2k profile (`TAG=-f2prof`, P 28.82 s / U 24.93 s): disk_load 9.41 · construct 3.91 (2.03 = retained
  hits re-running the dequant) · h2d 3.33 · scatter 3.21 · gemm 2.91 · loop ~5.9; 36,493 retained hits + 17,445
  misses.** So F3 reduced (slab-backed transient prefill reads + pinned H2D) is worth **~4 s at 2k (−16 %)**, not
  the row's 5–10 s ceiling. Decide against F4/F5 below before building it.
  **REVISED 2026-09-19 after F4: F3 is now the best remaining lever and its blocker is gone.** The objection was
  that uploading from pinned means holding one of 8 pooled buffers until the H2D lands, while the forward waits for
  all ~281 reads of a layer call first. Use the **slab arena** as the transient source instead: ~10,111 pinned
  1.8 MB slabs, all free during prefill (prefill never fills the RAM tier). Acquire a slab per miss, hand the views
  to the forward, release after the layer's compute behind a CUDA event — the fused-EXL3 staged path already does
  exactly this. Then the read path is SLAB not SPARSE (**3.74 vs 2.48 GB/s at k=8**) and the H2D is pinned
  (**114 vs 147 µs/expert**): projected **disk 9.10 → ~6.0 s, h2d 3.33 → ~2.6 s ≈ −3.8 s of 25 s (−15 %)**.
- [x] **F4 — coalesce adjacent expert reads in prefill. BUILT, MEASURED, REVERTED 2026-09-19 on its own stop rule.**
  Buffer sized to a whole number of experts up to a new hardware constant `_NVME_EFFICIENT_READ_BYTES` (12 MiB) →
  Qwen3-Next 11.9 MiB × 8 (95 MB pinned), EXL3 unchanged (spans 11.29 / 8.48 MiB already exceed it);
  `load_experts_sparse` + `plan_expert_read_runs`; prefill dispatched one task per run; futures returned lists of
  `(eidx, raw)`. Tests `tests/test_coalesced_expert_reads.py` (8), suite 2312 passed.
  **TRAP (cost one GPU arm): the first planner bounded a run by the expert's first-to-last SPAN and was a complete
  no-op** — this checkpoint is TYPE-MAJOR (all `weight_shape`, then all `weight_scale`, then all `weight_packed`,
  experts in id order inside each), so one expert's blocks sit ~832 MiB apart while consecutive experts' same-type
  blocks touch. The bound must be each expert's WIDEST BLOCK. Fixed: on the real index a 281-expert layer miss list
  plans 41 runs, **843 → 337 reads**, largest read 1.69 → **10.5 MiB**.
  **Same-day ABBA (F4a → base → F4b), logits identical: 2k 25.16 → 24.10 s (−4.2 %), 8k 58.90 → 59.33 s (+0.7 %,
  NULL), disk_load 9.10 → 8.55 s (−6.0 %), decode −0.3 % (noise).** Stop rule was disk_load < ~7 s → **NOT MET,
  reverted**; suite back to 2306 passed / 16 skipped. Refuted test kept at
  `docs/refuted/test_coalesced_expert_reads.py.refuted`.
  **Why it under-delivered:** at 2k only ~55 % of a layer's experts are routed, so ADJACENT-id runs average ~2.2
  experts (reads grow 1.69 → ~3.7 MiB, not the 11.8 MiB where the drive pays 7.1 GB/s), and disk is only ~35 % of
  the post-F2 prefill. **Do not re-open without a denser routing regime or a bigger read.**
- [x] **F6 (new, found while re-scoping F3) — overlap prefill reads with compute in the non-EXL3 fused forward.
  BUILT, MEASURED, SHIPPED 2026-09-19 (`4d2fa99`).** It was the only one of the three forwards that waited for EVERY
  expert read of a layer call before computing any expert (the ordering argument that justified it was settled years
  ago in the other two by `accumulate_expert_outputs`, which sums in expert-id order after the loop). Now consumes
  through `_iter_expert_loads` and accumulates the same way. **Same-day ABBA: 2k 25.86 → 23.12 s (−10.6 %; −7 % vs
  today's other base arms), 8k 63.10 → 58.35 s (−7.5 %; ~0 vs the other bases), profiled `disk_load` 8.27 → 0.17 s,
  decode neutral, prompt-end logits BIT-IDENTICAL at both lengths.** Read the base arm with care: it measured 2k
  25.86 s / decode 7.49 tok/s against 24.84–25.16 s / 7.95 from the same code hours earlier and sits below BOTH F6
  arms, so it is a dip, not drift — claim −7 to −10 % at 2k. Suite 2308; gates server Qwen3-Next 9, accumulator
  8/8/8, 1B prefix 1. Also fills a telemetry blind spot (this forward booked no `expert_disk` time at all).
  **It re-prices everything left: the prefill is now COMPUTE-bound (`disk_load` 0.17 s), so F3's read half is worth
  ~nothing and only its pinned-H2D half survives (~0.6 s at 2k).**
- [x] **F5 — permutation dispatch in the non-EXL3 fused forward. BUILT, MEASURED, SHIPPED 2026-09-19 (`736c4c6`) —
  the biggest single lever of the row, and it pays at DECODE too.** `compute_expert_dispatch` (already used by the
  other two forwards) replaces the int64 one-hot of `num_experts × top_k × tokens` (21 MB per 512-token chunk) plus
  a `torch.where` and two tiny H2D copies per routed expert with one CPU argsort and ONE H2D per layer; at decode
  that shape cost ~960 tiny copies per token (48 layers × 10 experts × 2). **Same-day ABBA: 2k 23.50 → 18.13 s
  (−22.9 %), 8k 59.58 → 37.20 s (−37.6 %), decode 7.815 → 9.055 tok/s (+15.9 %), prompt-end logits BIT-IDENTICAL at
  both lengths.** Order inside an expert changes (token-major) but cannot change the result — top-k ids are
  distinct, so `index_add_` has no duplicate indices. Profiled: scatter 4.63 → 3.15, mask_sync 0.14 → 0.30; the bulk
  of the win is per-expert Python + launch overhead no bracket covered, which is why the profile under-predicted it.
  Suite 2311; gates server Qwen3-Next 9, accumulator 8/8/8, 1B prefix 1. **§8: no knob; reachable at N=1 on the
  default config, and the decode half lands on every Qwen3-Next token the server emits.**
- [~] Decide the fix → F1 / F2 / F3 above, in that order (F1 is the smallest and already measured).
- [ ] Build + measure TTFT at 2k / 8k (same-day A/B), greedy ids identical; gates: suite, server gate on Qwen3-Next,
  regression gates Qwen3-235B / MiniMax / Llama-3.2-1B.

---

# 45 · [x] CLOSED 2026-09-19 — the module-list forward re-read its experts on every prefill chunk; retention shipped (2k 2.75×, 8k 5.47×)

**Status: CLOSED 2026-09-19. Measured (3.659× at 2k, 14.216× at 8k), built, re-measured (2.75× / 5.47× TTFT,
logits bit-identical, decode null), every gate green.**
Cold-pickup brief (self-contained, gitignored):
[`…-row45-modulelist-chunk-retention-BRIEF.md`](docs/2026-09-19-row45-modulelist-chunk-retention-BRIEF.md).

- **It is exactly ONE gap, verified by code reading 2026-09-19.** Of row 44's four levers, the module-list forward
  (`moe_patch.py:1403-1709`) already has F5's permutation dispatch (`compute_expert_dispatch`, `:1485`) and F6's
  read/compute overlap (`_iter_expert_loads`, `:1671`), and F1 never applied to it (EXL3 pools were always 8).
  **Only F2's chunk-major retention is missing** — no `_chunk_major*` anywhere in that forward.
- **Nothing amortises it:** `moe_supports_prefill_stager` (`vaayu_base.py:802`) is `is_exl3 AND all styles fused`,
  so MiniMax (EXL3 + module-list) never starts a prefill stager. (The earlier note here guessing the stager might
  already cover it is WRONG — that is why this line replaces it.)
- **Predicted, derived from the offset index (NOT measured):** MiniMax is **62 layers × 256 experts × 8.48 MiB =
  141 GB of experts, 2.28 GB per layer**, `num_experts_per_tok = 8`. One 512-token chunk routes 4096 slots over 256
  experts, so ~every expert routes in ~every chunk → re-read factor ≈ chunk count (2k → ~4×, 8k → ~16×). Compare
  Qwen3-Next, where the same defect measured 3.09× / 10.9× and its fix was worth −56 % / −74 % of TTFT.
- **Step 1 (measure):** `plans/probes/row44_prefill_split.py` with `MODEL=<minimax repo root>` (NOT
  `…/splitted_model` — `config.json` is in the repo root; the split dir fails `AutoConfig`),
  `ARMS=W,U,U2,P`, `BALLAST_GB=25`, **29G scope** (MiniMax is OOM-killed at 22G). Read `routed_sum` vs
  `routed_union`. **Gate: < 1.5× at 2k → close the row with the number.**
- **STEP 1 RESULT 2026-09-19 — THE GATE FIRES.** 2k: `routed_sum/routed_union` = **3.659×** (55,871 loads vs a
  15,268 union, **495.86 GB** read); 8k: **14.216×** (220,473 vs 15,509, **1956.70 GB**). Three arms per length,
  wall spread 0.27 % (2k) / 1.2 % (8k), logits identical. Per-chunk union is 225.3 of 256 experts and per-layer
  union 246.3 of 256, so routing IS saturated at 512 tokens as predicted. **But the wall is NOT the disk:**
  h2d was **68.1 s of 113.9 s (59.8 %)** at 2k and 268.9 of 452.1 at 8k, while exposed `disk_load` was 1.9 / 7.7 s —
  the NVMe ran at 4.63 GB/s the whole prefill, fully hidden by F6. The RAM tier absorbs none of it: prefill loads take
  a transient never-cached path (`moe_patch.py:2098`), so `usage=0.00GB/11.04GB` with 55,871 misses.
- **STEP 2 BUILT AND MEASURED 2026-09-19 — `_chunk_major_gpu` retention in `_make_modulelist_sparse_forward`.**
  **2k prefill 107.07 → 38.91 s (2.75×), 8k 430.36 → 78.70 s (5.47×)**, same day, one process per arm, loads landing
  exactly on the per-layer union, bytes 495.86 → 135.50 GB and 1956.70 → 137.64 GB, **prompt-end logit sha256
  BIT-IDENTICAL to the base arm at both lengths**, peak VRAM +1.48 GiB against an 8.8 GB budget with 7.1 GiB free.
  Both beat their own predictions (54.6 / 165 s) because removing the traffic also sped up `gemm` (24.97 → 15.08 at 2k)
  and `scatter`, which the prediction held fixed. Two deviations from F2, both forced by this forward's entry type:
  the attach-failure branch called `_vram.drop()` unconditionally and would have raised on None during prefill (it now
  drops from whichever tier is live and refunds the bytes), and `_detach_expert` still runs on a retained expert
  because the ENTRY owns the GPU tensors and BC objects, not the module. Tests
  `tests/test_modulelist_chunk_retention.py` (7), RED before / green after. **§8: no knob, nothing to plumb,
  reachable at N=1 on server defaults — it fires on every MiniMax prefill longer than one chunk.**
- **[x] VERIFICATION COMPLETE.** Suite **2318 passed / 16 skipped** (2311 baseline + this row's 7). Gates: accumulator
  **MiniMax 8**, **Qwen3-235B 8**, **Qwen3-Next 8**; server gate **Qwen3-Next 9**; **Llama-3.2-1B prefix 1**.
  **Decode is null on MiniMax, measured not argued:** base 1.8246 / 1.8204 tok/s vs treatment 1.8304 / 1.8256
  (`row26_decode.py MODEL=minimax BALLAST_GB=26.5 NEW_TOK=128 ARMS=W,U1,U2`, base arm on the stashed tree, same day)
  = **+0.30 %**, inside each arm's own ~0.25 % spread.
  **Environment note worth keeping:** the Qwen3-Next gates need a 29G scope (22G is OOM-killed, exit 137, directly
  after a MiniMax arm) and at 29G they trip Claude Code's background-shell memory reaper — run them in the
  FOREGROUND, or start the CLI with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
- **Step 2 (only if it fires):** retention is NOT a copy of F2 — entries here are EXL3 `BC_LinearEXL3` objects bound
  onto live modules and detached per expert, so the retention must reuse this forward's own
  `_attach_cached_expert` / VRAM-cache hit path, bounded by the VRAM budget (one layer's union is 2.28 GB — check
  the `Expert VRAM cache:` line on a real load first), dropped at the layer's last chunk call.
- **§8:** no knob expected; reachable at N=1 if it fires. **§5:** a fix to ONE forward; Qwen3-235B (EXL3 + fused)
  is not in scope — it has the stager and the fused-EXL3 retention.

---

# 42-B · [x] CLOSED 2026-09-20 — BUILT, MEASURED, SHIPPED: round-2 TTFT 30.60 → 4.90 s (6.24×) on Qwen3-Next, decode −2.56 %, ids bit-identical, N* = 9,512 tok/turn vs a ~500 stop rule, every gate green

**Status: closed and shipped 2026-09-20. The next row to work is 46.**
Order (as it ran): RE-MEASURE → plan (`docs/superpowers/plans/`) → build. Cold-pickup brief
(self-contained, gitignored): [`…-row42B-hybrid-prefix-cache-BRIEF.md`](docs/2026-09-19-row42B-hybrid-prefix-cache-BRIEF.md).

**Startable cold in a fresh session — verified 2026-09-19 after row 45.** Every code reference this
row depends on was re-checked against HEAD that day: `cache_holds_recurrent_state` (`vaayu/batch.py:501`),
`prefix_cache_refusal_reason` (`vaayu/batch.py:539`), `copy_prefix` (`vaayu/vaayu_base.py:1576`),
`enable_prefix_cache`/`prefix_sessions` plumbed at `vaayu/server/app.py:406,420` and declared at
`vaayu/server/config.py:100,193`, probe `plans/probes/row42_snapshot.py` present.
**Row 45 does NOT move this row's numbers** — it changed only the module-list forward (MiniMax-M2.7),
and Qwen3-Next runs the non-EXL3 fused forward. The 9.06 → ~7.87 tok/s decode figures below are HEAD's.

**Environment, three traps that each cost a run on 2026-09-19 (suite baseline is now 2318 passed / 16
skipped at `58b19b3`):** (1) pass a checkpoint's **repo root**, never its `splitted_model` subdir —
`config.json` is in the root and `AutoConfig` fails on the split dir; (2) **22G is OOM-killed (exit 137)
directly after a MiniMax arm** — use a 29G scope throughout; (3) at 29G, run gates in the **FOREGROUND**
— as background shells they are stopped by Claude Code's background-shell memory reaper, which is not a
test failure (the Qwen3-Next accumulator gate was killed twice, then passed in 41 s in the foreground).

**One methodological warning row 45 paid for:** it projected its own win from the `_moe_profile` split
and **under-predicted by 40 %** (predicted 54.6 s, measured 38.91) because the projection held `gemm`
and `scatter` fixed and they fell ~40 % once the competing traffic went away. A per-phase projection is
a LOWER BOUND. Compute N* from a measured T_hit, never a projected one.

**RE-PRICED 2026-09-19 — row 44 moved both sides of this trade and row 42's headline is STALE.** Row 42 justified
the build with 8.6–21× round-2 TTFT against a **269.3 s** 8k miss; that miss is now **37.2 s measured**
(`row44-f5ab-ttft-F5*`, U8 arm, cold tiers) and decode is **9.06 tok/s**. The HIT side also got faster (it is mostly
a prefill of the diverged tail) and is **unmeasured on HEAD**, so the ratio may be roughly preserved — do not assume
either way, re-run row 42's part 2 (`plans/probes/row42_snapshot.py PREFIXES=2048,8192 DIFFS=64,256,1024
VERIFY_REPS=0`, ~20 min, 29G scope). **What definitely changed is the break-even**, since the cost (the
prefix-session reservation returning 4.83 GB → −13 % decode, row 30 reversed: 9.06 → ~7.87) is unchanged while the
TTFT saving shrank ~7×:

```
N* = (T_miss − T_hit) / (1/7.87 − 1/9.06)     # generated tokens/turn where the cache stops paying
   row 42's numbers (256 s saved)  → N* ≈ 13,300   (wins for essentially every turn)
   T_hit ≈ 4.5 s  (32.7 s saved)   → N* ≈ 1,960
   T_hit ≈ 13 s   (24.2 s saved)   → N* ≈ 1,450
```

**So the plan must decide the reservation, not just lift RC7** (lower `PREFIX_SESSIONS` for hybrids, size to what
the store holds rather than full `MAX_SEQ_LEN`, or make it dynamic). **Stop rule: if the re-measured N\* lands below
~500 generated tokens/turn, close 42-B with the number.**

Row 42 measured the mechanics (40.1 MB bf16 snapshot, exact restore) — design facts the plan inherits (row 42 doc §2):
- Append-only turn (match ≥ stored cache length): no snapshot, no restore — lift RC7 and move the state with the KV.
- Divergent turn: restore the prompt-end snapshot (taken after prefill, before decode; 2.5 ms, 40 MB). Exact resend
  (`_adopt_prefix_hit` crops to `prompt_len − 1`) needs the state at `prompt_len − 1`: prefill all but the last token,
  run the last as a decode step.
- Cross-session borrow needs a snapshot at the shared point: keep the first few 512-token chunk boundaries (40 MB each).
- Code that must change: `prefix_cache_refusal_reason` (RC7), `copy_prefix` (raises), `to_cpu`/`to_device` (KV only),
  `kv_persist` (KV only — refuse or extend for hybrids).
- **Cost to decide in the plan:** the prefix-session reservation returns (3 × 65536 × 24,576 B = 4.83 GB → ~−13 %
  decode, row 30 reversed). Size it to what the store holds, or lower `PREFIX_SESSIONS` for hybrids — measure.
- **§8:** no new knob (`ENABLE_PREFIX_CACHE` / `PREFIX_SESSIONS`), reachable at N=1 on the default config.
- **§5:** state layers by capability; sizes read off the tensors.
- [x] **Re-measure DONE 2026-09-20 — the stop rule does NOT fire; 42-B continues.** HEAD `5b9d898`, Qwen3-Next,
  row 42's probe part 2, 2 reps order-reversed (`docs/probes/row42B-reprice.log`,
  [`…-row42B-reprice-RESULTS.md`](docs/2026-09-20-row42B-reprice-RESULTS.md)). **MISS 2k 16.45 s · 8k 35.05 s**
  (row 42's 8k was 269.3 → **7.7× faster**, as row 44 predicted). **HIT is flat in P** — 3.92 s at 2k vs 3.97 s
  at 8k for d=64 — because the hit path is a restore plus a prefill of the diverged tail only. Honest ratios on
  HEAD: **8.8× at 8k/d=64, 4.2× at 2k/d=64**, down to 2.7× / 1.4× at d=1024 (row 42's 21× is dead).
  **N\* (Δ = 0.016690 s/tok from 9.06 → 7.87): 8k = 1862 / 1635 / 1318 at d = 64 / 256 / 1024; 2k = 751 / 576 /
  300.** Only the 2k/d=1024 corner is under the ~500 close threshold. Restore still EXACT on HEAD after rows
  44/45 (`hit_vs_ref` 0.0 vs a 0.0124 noise floor, argmax same); snapshot 40.11 MB, D2H 2.45 ms, H2D 2.05 ms.
- **THE NUMBER THAT REORDERS THE BUILD (found by the re-measure):** `derive_expert_ceiling`
  (`vaayu/vaayu_base.py:354`, called `:2651`) reserves `PREFIX_SESSIONS × MAX_SEQ_LEN × kv_bytes/token` =
  3 × 65536 × 24,576 B = **4.83 GB**, but a stored **8k** session holds **0.20 GB** and a 2k one 0.05 GB — the
  default **over-reserves 8× at 8k, 32× at 2k**. That over-reservation IS the −13 % decode tax, and it is the
  only reason N\* sits in the low thousands. Size the reservation to what the store holds and the cost drops
  ~8×, moving every N\* the same way. **Caveat: 7.87 tok/s is DERIVED (row 30's +15.09 % reversed), never
  measured with the cache live — row 45 under-predicted itself 40 % on exactly this kind of projection, so the
  build MEASURES the reserved-side decode.**
- [x] **Build plan WRITTEN 2026-09-20:** [`docs/superpowers/plans/2026-09-20-row42B-hybrid-prefix-cache.md`](docs/superpowers/plans/2026-09-20-row42B-hybrid-prefix-cache.md)
  (gitignored — this bullet is the durable copy). **5 TDD tasks. The design turned out SMALLER than this row
  assumed, because two facts in the code collapse it:** (1) `prefix_cache.py:113` —
  `borrowed = (key != session_id and m < len(entry.token_ids))`, so a prompt consuming the WHOLE entry is a
  check-out, never a copy, and the append shape never reaches `copy_prefix` at all; (2) `vaayu_base.py:5381` —
  a stored entry's `token_ids` is length `T` while its cache holds `L = T − 1` fed positions, and HF re-derives
  the prefill split from `get_seq_length()`, so the restore point is `L` automatically. **There is therefore NO
  snapshot object**: the recurrent state IS part of the entry, and `to_cpu`/`to_device`/`resident_bytes` just
  learn to carry `conv_states`/`recurrent_states` with the KV.
  **T1** carry the state (capability-detected, `_recurrent_state_layers` / `_move_recurrent_state`) ·
  **T2** the serving gate in `lookup()` — refuse unless `m == len(entry.token_ids)` **and**
  `min(m, prompt_len−1) ≥ L`, counted as `recurrent_partial_misses` in `stats()` (verified against all five hit
  shapes: continuation and exact resend both restore at `L`; partial divergence, a prompt shorter than the entry,
  and the borrow shape all MISS) · **T3** keep the KV-only side channels refused — `enable_kv_persist` and
  `prefix_compress` both carry K/V only and would restore a state that never saw those tokens (lands BEFORE the
  lift so no commit ordering exposes them) · **T4** lift RC7 (delete the recurrent clause from
  `cache_supports_prefix_store` and `prefix_cache_refusal_reason`; `copy_prefix` keeps raising) · **T5** on-device
  A/B + gates. Adds 16 unit tests, all CPU-only (real `Qwen3NextConfig` layer objects, no checkpoint) → suite
  2318 → **2334**. A/B is `plans/probes/row42b_ab.py`, **one arm per PROCESS** (4 loads in one process hits the
  exit-137 pinned-block trap), ABBA `off,on,on,off`, greedy ids sha256 identical across arms.
  **Reservation sizing is NOT in this plan — it is row 46** (user's call, 2026-09-20).
- [x] **BUILT AND VERIFIED 2026-09-20 — SHIPPED.** Branch `row42b-hybrid-prefix-cache`, 5 commits
  `6cac130..13f804e`, 4 TDD code tasks + on-device measurement. Detail:
  [`…-row42B-hybrid-prefix-cache-RESULTS.md`](docs/2026-09-20-row42B-hybrid-prefix-cache-RESULTS.md).
  **Measured, ABBA `off,on,on,off`, one process per arm, 8192-token prompt + 256 new, GEN=64, server knobs:
  turn-1 TTFT unchanged (32.634 → 32.560 s, −0.23 %, so the A/B is valid), turn-2 TTFT 30.599 → 4.902 s
  (6.24×, 25.70 s saved), turn-2 decode 9.7364 → 9.4868 tok/s (−2.56 %), generated ids sha256 IDENTICAL on
  all four arms** (`9e312308…`, greedy, 64 tokens). **N\* = 25.697 / 0.0027016 = 9,512 generated tokens/turn
  against a ~500 stop rule — it misses by 19×.** Gates: suite **2334 passed / 16 skipped** (was 2318/16),
  accumulator Qwen3-Next 8 / Qwen3-235B 8 / MiniMax 8, server gate Qwen3-Next 9, Llama-3.2-1B prefix 1.
  **§8: no new knob, already plumbed, REACHABLE at N=1 on server defaults, nothing banked.** Still refused and
  re-verified against live code after the lift: `copy_prefix`, speculative decoding + its independent crop
  backstop, `enable_kv_persist`, `prefix_compress`, and the bare `DynamicCache` from `kv_cache_mode="fp16"`.
- [x] **Final whole-branch review found ONE Critical the five per-task reviews structurally could not
  (each saw one commit, and the breaking store site is PRE-EXISTING) — FIXED 2026-09-20, `8618303`.**
  `restore_on_abort` (`prefix_cache.py`) crops a checked-out entry back to the matched length and re-pools
  it when a turn dies mid-flight. `crop()` shortens `_seq_len` ONLY — it never rolls back the recurrent
  state (that is why `_trim_kv_cache` refuses a recurrent cache) — so an aborted hybrid turn banked an
  entry whose KV claimed `keep` positions while its state described more, **and the serving gate could not
  see it, because the gate reads `get_seq_length()`, which IS the KV length.** The next turn would be
  served against a state that already consumed the tokens it was about to re-run: silently wrong text, no
  crash. Now refused with a warning whenever the crop would actually shorten a recurrent cache; the
  no-crop case (abort before any forward completed) still restores. Mutation-tested: with the guard
  disabled the new test fails. **General lesson worth keeping: a guard that validates an invariant using
  the same accessor the breaking writer uses cannot detect that writer.**
- [x] **BANKED LIMITATION (`a3b90dc`): `batched_generate` refuses the prefix cache on a recurrent cache.**
  Lifting RC7 made `pc` non-`None` on hybrids for the first time, which newly broke two batched sites —
  `broadcast_cache_rows` walks the KV buffer lists only (KV goes to batch N, the state stays batch 1), and
  `slice_cache_row_unpadded` raises on a hybrid, so every batched store-back warned "LOST from the pool".
  Both are now skipped with one warning. **Worth ZERO to users today and costs nothing: the server runs
  `MAX_BATCH_SIZE = 1` (§8 #3), so this path is unreachable.** It has no regression test, deliberately —
  **if batching is ever wired in, this needs one, and `peek_match` still over-reports on hybrids
  (`scheduler.py:237` reads it).**

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
| **A1** `[x]` | **Guards — stop failing silently** | an unusable checkpoint RAISES instead of loading with zero experts (G0 + G1-raise). **No model needed, unit tests only, no on-device risk** | — | none — **START HERE** | ~1 day | [`…-A1-guards.md`](docs/superpowers/plans/2026-09-20-arch-A1-guards.md) |
| **A2** `[x]` | **Fused-3D expert indexing** — **CLOSED 2026-09-21** | the index understands one-blob-per-layer checkpoints and derives per-expert byte ranges. **Unit tests only** | A1 | none | ~1 day | [`…-A2-fused3d-index.md`](docs/superpowers/plans/2026-09-20-arch-A2-fused3d-index.md) |
| **A3** `[x]` | **gpt-oss-20b END TO END — CLOSED 2026-09-21** | a 4th architecture family (MXFP4 · fused-3D · sliding-window) generates through the server, gates green | A1, A2 | **Task 1 IS an abandon gate: stock transformers must run MXFP4 on this box** | ~3–5 days | [`…-A3-gptoss-e2e.md`](docs/superpowers/plans/2026-09-20-arch-A3-gptoss-e2e.md) |
| **A4** `[x]` | **Shared-expert pinning (G7) — CLOSED 2026-09-23, PREMISE REFUTED** | nothing to build: always-active experts were **already** pinned, not streamed. Stop rule fired on Q1 | — | stop rule fired; Q2 never run | **½ day, zero code** | [`…-A4-shared-experts-RESULTS.md`](docs/2026-09-23-A4-shared-experts-RESULTS.md) · brief: [`…-A4-shared-experts-BRIEF.md`](docs/2026-09-20-arch-A4-shared-experts-BRIEF.md) |
| **A5** **(NEXT)** | **DeepSeek-V4-Flash port** | the 157 GB I8/fp8 MoE loads and runs | ~~A4~~ **(closed)** | **kernels must build on sm_120 — decided at step 1, abandon costs ~2 days not weeks** | **weeks** | brief: [`…-A5-deepseek-port-BRIEF.md`](docs/2026-09-20-arch-A5-deepseek-port-BRIEF.md) |
| A6 | per-expert MXFP4 streaming | 1.27 GB working set instead of 10.17 GB; drops the Hub kernel dependency. **Perf, not reach** | A3 | A3's tok/s exists to beat | ~3 days | plan when A3 lands |
| A7 | G3 `compute_device` parity · G5 NVMe re-measure | removes two latent generality traps | — | none | ~1 day | plan when picked |


**A1 IS CLOSED — done and verified 2026-09-21, five commits `512f6b7..444afd1` on main.**
Suite **2348 → 2373 passed / 16 skipped** (+25; the plan asserted +15 for its three tasks and got
exactly that, then +5 for a task the plan did not have and +5 for the final-review fix wave).
RESULTS: [`…-row47-A1-guards-RESULTS.md`](docs/2026-09-21-row47-A1-guards-RESULTS.md).
Plan: [`…-A1-guards.md`](docs/superpowers/plans/2026-09-20-arch-A1-guards.md). Per-task reviews,
reviews and the full ruling list are in `.superpowers/sdd/2026-09-20-arch-A1-guards/progress.md`
(gitignored — this row is the durable copy).

- [x] **Task 1 `5cb90cd` — `ExpertOffsetIndex.expected_expert_count()`.** The index had nothing to
      compare an empty result against. The field name differs per family, so the lookup is an
      EXPLICIT ORDERED LIST `("num_experts", "num_local_experts", "n_routed_experts")`, never a
      substring scan — a scan also matches `num_experts_per_tok` (10 on Qwen3-Next against a real
      512) and `shared_expert_intermediate_size` (512). `None` means "no expectation", never zero,
      so dense checkpoints are unaffected.
- [x] **Task 2 `cbe492d` — `build()` RAISES when the config implies experts and the index found
      none.** Layout-agnostic by construction ("config says E, index holds zero"), so it stays
      correct for the next layout nobody has seen. Runs BEFORE `_write_cache()`, so an empty index
      can never be cached and bypass the guard on a later load. Also deleted the class docstring's
      claim that the `.experts.N.` regex "covers … any future architecture" — untrue, and it is
      what made G1 invisible for the life of the file.
- [x] **Task 3 `9407c17` — `vaayu/persist/checkpoint_integrity.py` (G0).** Verifies the shard count
      implied by the `-of-NNNNN` suffix and that every `weight_map` target resolves. Reads NAMES and
      the index only, never a config field or an architecture string; silent on directories that are
      not sharded checkpoints. Verified OK on all NINE directories under `models/` (the six named
      checkpoints plus `venv`, `Qwen2.5-Coder-1.5B` and the MiniMax score-correction dir).
- [x] **Task 4 `6bb438d` — NOT IN THE PLAN. Added 2026-09-21 at the user's direction after
      measurement showed TASKS 1-3 SHIPPED GUARDS THAT COULD NEVER FIRE.** `vaayu_base.py:2073`
      sets `checkpoint_path` from `find_or_create_local_splitted_path` → the `splitted_model/` dir,
      **which carries no `config.json`** — so `expected_expert_count()` returned `None` on every
      production load and the raise was skipped every time. Measured before the fix: Qwen3-Next
      24576 indexed experts against `expected=None`; MiniMax 15872 against `None`. G0 was inert for
      a different reason: the splitted output has neither `-of-NNNNN` names nor an index, and
      nothing ever checked the ORIGINAL shard set. Fix: the index takes the config's directory
      separately from the shards' (`vaayu_base.py:2295` passes `self.model_local_path`);
      `build_or_load()` also asserts on the CACHED branch; G0 moves to `split_and_save_layers`.
      After: Qwen3-Next `expected=512`, MiniMax `expected=256`, both Llamas `None`, none raising.
      **The general lesson, and it is the second time this program has paid for it: three per-task
      reviews all passed, because the defect lived entirely in UNCHANGED code that no single diff
      contained.**
- [x] **Final whole-branch review (opus) → READY TO MERGE**, no Critical, no regression, no
      model-agnostic violation. Two Important, FIXED in one wave (`444afd1`); three Minor filed as
      rows below by ruling rather than fixed, because each touches MoE detection or shard-name
      arithmetic that is CORRECT for every checkpoint on disk today.
  - **Important 1 — the guard fired AFTER the originals were deleted.** `split_and_save_layers`
    deletes source shards at `utils.py:485` under the server's own `DELETE_ORIGINAL = True`
    (`vaayu/server/config.py:62`); the expert guard was ~200 lines later at `vaayu_base.py:2295`.
    Pointing the server at gpt-oss-20b would cost a ~13 GB split AND the only copy of the
    checkpoint before the error arrived — and `remove_real_and_linked_file` follows symlinks, so a
    shard linked into the shared HF cache takes the cached blob with it. Now
    `assert_expert_layout_indexable` asks the same question at split time on the ORIGINAL keys,
    before anything is written or deleted.
  - **Important 2 — G0 never reached its own motivating case.** `find_or_create_local_splitted_path`
    only calls `split_and_save_layers` when an index or model file exists, so "46-of-48 shards with
    NO index" — the exact DeepSeek incident `checkpoint_integrity.py`'s docstring cites — fell
    through to `snapshot_download(<absolute local path>)` and raised `HFValidationError` instead of
    G0's message. Now guarded in the `else:` branch before the fallthrough.

**HEADLINE, stated as measured and no wider.** Pointing Vaayu at `gpt-oss-20b` now RAISES:
`config implies 32 routed experts, but no key in the checkpoint index matches …`, **at split time,
before the split is written and before the original is deleted.** The four working models are
unaffected — Qwen3-Next OK, MiniMax OK, Llama-3.3-70B OK, Llama-3.2-1B SKIP (single-file, no
index.json, so the verification script cannot reach it; the re-reviewer traced that on the REAL path
it does hit the guard, via the single-file branch that builds `index` from the tensor names, and its
config has no expert field so the answer is `None` either way — but that branch was never re-exercised
by a verification run, only by reading).
**A1 does NOT make gpt-oss-20b load.** That is A2 (fused-3D indexing) + A3 (end to end). A1's whole
deliverable is that the failure became VISIBLE and CHEAP instead of silent and destructive.

**§8, re-answered after Task 4 changed the wiring — and ONE ANSWER IS NARROWER THAN THE PLAN's:**
1. **Knob?** None needed, and none added. Both raises are tied to concretely-absent files or keys.
   There is deliberately no escape hatch.
2. **Plumbed?** Yes. `vaayu_base.py:2295` and `utils.py`'s `split_and_save_layers` are the only two
   production seams, and both sit on the single entry path.
3. **Reachable at N=1?** Yes for every model on this box — but **the plan's wording "on every load,
   for every model" is REFUTED and must not be re-copied.** G0 reaches only loads whose source dir
   holds an index or model file; G1 reaches only `cuda` + `is_moe` loads. True for the four working
   models and for gpt-oss on CUDA, which is what matters today. A library caller gets the identical
   guards — `VaayuBaseModel.__init__` is the path the server takes.
4. **Banked?** Nothing.

**A2 IS CLOSED — done and verified 2026-09-21, MERGED to `main` (fast-forward `16b0fbe → b05f13e`):
`9adc3c1` detect · `3beedad` index + versioned cache · `1653356` the split-time guard · `b05f13e`
this ledger entry.** RESULTS: [`…-row47-A2-fused3d-index-RESULTS.md`](docs/2026-09-21-row47-A2-fused3d-index-RESULTS.md).
Suite **2373 → 2393 passed / 16 skipped**. **A2 does NOT make gpt-oss-20b run — that is A3.** It
makes the checkpoint DESCRIBABLE and stops both guards refusing it.

- [x] **Task 1 `9adc3c1` — `_fused_expert_axis`, detection by capability.** Two conditions, both
      required: an `experts.` prefix with NO numeric component, AND a leading dimension equal to
      the config's expert count (A1's `expected_expert_count()`). Shape alone catches an attention
      tensor that leads with the same number; the prefix alone catches per-layer scalars. **The
      per-expert layout provably cannot move: the fused regex matches 0 keys across all four
      checkpoints on this box** — Qwen3-Next 221,847 tensors, MiniMax 191,772, Llama-3.3-70B 2,406,
      DeepSeek-V4 72,317 (measured before any code was written).
- [x] **Task 2 `3beedad` — per-expert byte sub-ranges + whole-tensor metas.** safetensors is
      row-major and the expert axis leads, so expert *i* is ONE contiguous range — **no new reader
      capability and no 13.8 GB persist-time rewrite**; `load_expert_sparse` already reads
      arbitrary ranges. Both views kept: `index` (per-expert, for a forward that streams only the
      routed experts) and `fused_layer_tensors` (whole blob, for a kernel that routes internally —
      **this is the handle A3 needs**, `Mxfp4GptOssExperts.forward` routes inside `matmul_ogs`).
      **Raises** when the tensor does not divide evenly by the expert count. **Measured on the real
      checkpoint: gpt-oss-20b yields 32 expert ids, 144 whole-tensor metas (24 layers × 6 tensors),
      per-expert lengths 8,294,400 / 4,147,200 B, and the ranges tile each fused tensor exactly** —
      the plan's three numbers, hit.
      **Cache decision (plan step 6): EXTEND the format, keep reading the old one.** The layer map
      is now wrapped in `{"__vaayu_index_version__": 2, "index": …, "fused_layer_tensors": …}`;
      **version 1 — the bare map, which is what both caches on disk hold (Qwen3-Next 31 MB /
      24,576 experts, MiniMax 26 MB / 15,872) — is still READ, never invalidated.** The rejected
      alternative (`load_cached()` returns False for fused) would make a fused checkpoint re-scan
      every header on every load forever.
- [x] **Task 3 `1653356` — NOT IN THE PLAN, and it is the one that makes A2 reachable.** The plan
      was written 2026-09-20; **A1's SPLIT-TIME guard `assert_expert_layout_indexable`
      (`utils.py:343`) landed 2026-09-21 in `444afd1`, after it**, and asks the layout question on
      `_EXPERT_RE` alone. So after task 2 the guard and the index DISAGREED — the index reads a
      fused checkpoint, the guard still refused one, **at the one seam that deletes the original
      under the server's `DELETE_ORIGINAL = True`. Measured: gpt-oss-20b raised there, before the
      index ever ran.** The guard now accepts both layouts. It sees KEYS only, never shapes, so its
      fused answer is deliberately the WEAKER one — it cannot confirm the leading axis is the
      expert axis; that exact check still runs in `build()`, which has the shapes. Verified on the
      real checkpoints: gpt-oss PASSES, DeepSeek-V4 / Llama-3.3-70B / Llama-3.2-1B unchanged.
      **Same lesson as A1 task 4, third time in this program: the defect lived in UNCHANGED code no
      single diff contained.**
- [x] **Three pre-existing tests asserted the OLD behaviour of the fused shape (that it RAISES) and
      were moved — all intended, none a regression:** A1's `test_build_raises_when_config_implies_…`,
      `test_guard_reachability::test_config_in_a_separate_directory_is_found`, and
      `test_checkpoint_integrity::…_raises_for_fused_3d` (→ `…_passes_for_fused_3d` plus a new raise
      test). All three now use a layout outside BOTH detectors (`…mlp.expert_weights`), which is
      what those guards were always about. `test_json_cache_written` asserted the v1 on-disk shape
      and now asserts the envelope.
- [x] **On-device gates, same day, this build, each in its own 29G scope, foreground: accumulator
      Qwen3-Next 8 · accumulator MiniMax 8 · server smoke Qwen3-Next 9 · 1B prefix 1.** The server
      gate's log prints `Expert offset index loaded from cache` — **that is the reachability
      evidence: a v1 cache, 24,576 experts, read by the new `load_cached()` on the real checkpoint
      through the server.** Dense has no committed gate; Llama-3.3-70B was covered by the
      split-guard check and by reading, not by a run.

**§8 for A2:** (1) **no knob** — layout detection is not a user choice; (2) nothing new to plumb —
`build()`/`load_cached()` are on every load path and the split guard is on the single split seam;
(3) **REACHABLE at N=1 on every load**, proven by the four gates; (4) **banked: the per-expert metas
for a fused checkpoint are computed and unconsumed** until A3 (whole-layer, uses
`fused_layer_tensors`) or A6 (per-expert streaming, uses `index`). They cost microseconds and exist
so the layout is described once, not twice.

**A3 IS CLOSED — done and verified 2026-09-21, MERGED to `main` (fast-forward `e756e09 → d4e2fd0`,
nine commits `7c17773..d4e2fd0`); the branch `row47-a3-gptoss-e2e` is gone, so `git log` that range
for the diffs. Suite on merged main: 2416 passed / 16 skipped. Task 1 — THE ABANDON GATE —
PASSED: packed MXFP4 RUNS on this box, no dequant fallback.** Stock transformers 5.5.4,
`device_map="cuda:0"`, no Vaayu: `experts class: Mxfp4GptOssExperts`, `quantization_config.dequantize
= False`, loaded in 13.6 s, **peak VRAM 13.79 GB** (of 16.31 GB — it fits whole, so streaming is a
budget win here, not the only way in). `triton_kernels_hub` fetched from the Hub at run time (42
files) and the sm_120 build ran — that fetch is a live NETWORK dependency of this model, and nothing
caches it in the repo. **Reference for Task 4's comparison** (`plans/probes/a3_out/reference.json`,
gitignored — these numbers are the durable copy): prompt `"def fibonacci(n):"` = ids
`[1314, 165916, 2406, 3127]`, last-position logits
`sha256 7bfd3bb02c1a28280a1adaf3b3c608eccbd74d4df961f1901ef4f36f8f855fc1`, top-5 ids
`[59, 538, 793, 622, 220]`, top-5 logits `[16.25, 14.5, 14.4375, 14.3125, 14.0625]`. Probe:
`plans/probes/a3_mxfp4_gate.py`, log `docs/probes/a3-mxfp4-gate.log`.
**One correction to the plan's premise, found by the gate and it shapes Tasks 3/4: the experts
module's MXFP4 weights are NOT `nn.Parameter`s.** `experts.named_parameters()` returns only
`gate_up_proj_bias (32, 5760) float32` and `down_proj_bias (32, 2880) float32`; the `_blocks` /
`_scales` tensors are not in it. A3 task 3's `fused_param_experts` still detects the module (the
biases lead with the expert axis), but task 4 cannot "populate the module's parameters" as written —
whatever the seam sets, it is not a parameter assignment.

**A3 Task 2 — THE SEAM IS DECIDED (2026-09-21).** A **third branch in `_make_sparse_moe_forward`
(`vaayu/moe_patch.py:1326`), tested BEFORE the `is_exl3` split** and selected by capability
(`fused_param_experts(experts_mod)`, Task 3), producing `_make_whole_layer_sparse_forward`: it reads the
layer's fused tensors by the whole-tensor metas A2 recorded (`ExpertOffsetIndex.fused_layer_tensors`),
copies them H2D, hands them to **the quantizer's OWN weight conversions —
`hf_quantizer.get_weight_conversions()`** — calls the block's captured `original_forward` unchanged, and
frees the layer's expert attributes in a `finally`. **Four findings decided it, each verified on device or
in the source, not assumed:** (1) `original_forward` **is already the MXFP4 kernel path** — transformers
rebinds `GptOssMLP.forward` to its own `mlp_forward` inside `preprocess_model`
(`integrations/mxfp4.py:703`), which Vaayu captures at `moe_patch.py:1273`; on the real skeleton, 24 blocks
detect as `style=fused`, `router=router`, `forward=mlp_forward`. (2) **It returns a TUPLE**
`(routed_out, router_logits)` (`modeling_gpt_oss.py:384` unpacks it) while all three existing Vaayu
forwards return a bare tensor — the new branch must return it unexamined. (3) **`_make_fused_sparse_forward`
cannot be bent to serve it**: it computes the FFN ITSELF with `F.linear` ×3 and `experts_mod.act_fn`
(`:1859`, `:2060-2062`), and `Mxfp4GptOssExperts` **has no `act_fn`** — so today this model raises
`AttributeError` on layer 0 rather than computing wrong output. (4) **Populating is not assignment**:
`get_weight_conversions()` returns two `Mxfp4Deserialize` converters for gpt-oss and **`[]` for
Qwen3-Next's `CompressedTensorsHfQuantizer`** (measured), so the seam is quant-agnostic and inert on every
other model here; the converter swizzles and sets a triton tensor + `{proj}_precision_config`, deleting the
`nn.Parameter`. **The expert RAM/VRAM caches are NOT used on this path and that is deliberate** — `matmul_ogs`
consumes the whole fused tensor and routes inside the kernel, so the layer's full 423.6 MB is needed however
few tokens route; per-expert streaming is A6. **REJECTED alternative, recorded so it is not re-derived:**
dropping `is_moe` and streaming the MoE layer through the DENSE whole-layer path. Same bytes/token and less
new code, but it edits `_detect_moe`, `move_layer_to_device` and `free_from_gpu` — shared code the three
working models run — and it LEAKS, because the swizzled triton tensor is neither parameter nor buffer, so
neither `set_module_tensor_to_device(..., "meta")` (`loader_strategy.py:200-204`) nor
`_to_meta_preserving_buffers` frees it (423.6 MB/layer, 10.17 GB/sweep). **`_expert_history` gets EVERY
expert index** from this forward (the spec-decode gate at `vaayu_base.py:5756` sizes a verify pass off it,
and an empty dict trips its `off_no_routing` warning); `_maybe_record_usage` and `DECODE_TOP_K` stay unwired
here — nothing to clamp when the kernel routes. **Costs accepted, to be MEASURED in Task 6 and not tuned:**
10.17 GB/token of expert bytes, one swizzle per projection per visit (unmeasured), and per-chunk re-reads
during chunked prefill (the row 44-F2 / row 45 defect class, arriving by construction). Decision doc:
[`…-A3-seam-DECISION.md`](docs/2026-09-20-A3-seam-DECISION.md) (gitignored — this paragraph is the durable copy).

**A3 Task 3 DONE (2026-09-21) — and it CORRECTED the plan's selector before it could cost the fastest
model on this box.** `fused_param_experts(experts_module)` ships in `vaayu/moe_patch.py` (shape-derived,
`recurse=False`), with `tests/test_fused_param_detection.py` — **6 tests, all passing** (the plan wrote 4).
**The finding: the predicate does NOT discriminate.** Measured on the live module tree
(`plans/probes/a3_qwen_experts_shape.py`), **`Qwen3NextExperts` reports True too** — it carries DIRECT
fused parameters `gate_up_proj [512, 1024, 2048]` and `down_proj [512, 2048, 512]` with `num_experts=512`,
**both before and after `CompressedTensorsHfQuantizer.preprocess_model`** — while its CHECKPOINT stores one
tensor per expert (24,576 per-expert entries in the index). Had the third branch dispatched on the module
predicate the plan specifies, **Qwen3-Next would have been routed off `_make_fused_sparse_forward`** — the
path rows 30/30-P/44/42-B tuned, 4.06 → 7.4 tok/s — onto an unmeasured whole-layer forward. **The
discriminator is the CHECKPOINT layout, which is what A2 already records: dispatch requires
`offset_index.fused_layer_tensors[layer_name]` to be non-empty AND the module predicate.** A regression test
pins the trap (`test_a_per_expert_checkpoints_module_also_reports_fused`), so dispatching on the predicate
alone fails in unit tests rather than on the tuned model.

**A3 Task 4 DONE (2026-09-21) — gpt-oss-20b RUNS under Vaayu, and the streamed MoE path is
BIT-IDENTICAL to stock transformers.** `_make_whole_layer_sparse_forward` ships in
`vaayu/moe_patch.py` with the dispatch of Task 2's decision. **Proof (`plans/probes/a3_moe_block_compare.py`):
the same fixed input through layer 0, 9, 10 and 23's MoE block gives the SAME sha256 in stock and in
Vaayu — `max|diff| = 0` on all four** (layer 0 pinned-router, the rest streamed, so A2's per-layer byte
ranges and the swizzle are proven, not just layer 0's). **Whole-model:** top-5 ids match stock exactly and
in order (`[59, 538, 793, 622, 220]`), logits within **one bf16 ulp** (16.375 vs 16.25 on the top slot);
the residual is NOT the MoE path, NOT the rotary buffer (an fp32-inv_freq variant returns a byte-identical
sha) and NOT the attention implementation (both eager, both bf16 KV) — it is accumulation order, measured
and bounded, not assumed. Prefill 5.76 s for 4 tokens, **peak VRAM 1.66 GB against stock's 13.79 GB** —
the experts genuinely stream. Gates: suite **2393 → 2406 passed / 16 skipped** (+13), Qwen3-Next
accumulator **8 passed**, MiniMax-M2.7 accumulator **8 passed**, Llama-3.2-1B prefix **1 passed**.
**FIVE root causes, and FOUR of them are bugs in code EVERY model uses — the model only exposed them:**
**RC1 (`_read_grouped_ranges`, shared reader) took a NEW staging slot PER GROUP.** The A/B pair is safe
only because one caller holds its slot while the other writes the other one, so cycling both inside one
call overwrites what the other caller holds. gpt-oss's layer files interleave small non-expert tensors
between six ~70 MB expert blobs → one read is a dozen groups → **it clobbered the 1.16 GB embedding
between its read and its injection (absmax 118.5 → 1.36e38), and the model emitted garbage from layer 10
on, deterministically.** Every checkpoint whose non-expert tensors sit contiguously (Qwen3-Next, MiniMax)
is ONE group and never saw it. Fix: one slot per call. **RC2 the skeleton was built at `config.dtype`,
which is None for a checkpoint declaring neither `dtype` nor `torch_dtype` → torch's fp32 default →
accelerate casts every injected bf16 tensor BACK to fp32** (double VRAM, uncharged against the pin budget,
and flash refuses the layer). Now built at `running_dtype`; the other four checkpoints declare bfloat16 and
are unchanged. **RC3 the FP8-only experts workaround ran for EVERY quantizer.** It appends `"experts"` to
`modules_to_not_convert`, so MXFP4 never installed its experts module and the model kept the stock
grouped-GEMM forward — surfacing three frames away as `grouped_mm_experts_forward() got an unexpected
keyword argument 'scatter_idx'`. Now gated on `FineGrainedFP8HfQuantizer`. **RC4 the eager causal mask was
BOOLEAN.** Every `eager_attention_forward` does `attn_weights + attention_mask`, so a bool mask adds
**1.0 to the positions it means to ALLOW and 0.0 to the ones it means to BLOCK** — nothing is masked and
the model attends to its own future. **Latent since forever: all three working models take flash or sdpa.**
Measured 26 % off stock in layer-0 attention; now additive (`torch.finfo(dtype).min`, same past+new
sizing). **RC5 flash attention accepted a model it cannot run:** transformers approves the request (the
architecture supports flash) but the kernel build dies at the first token with *"S aux is currently only
supported for Hopper GPUs"* — attention SINKS are Hopper-only in this build. Vaayu now skips flash when the
module tree carries sinks and the GPU is not Hopper (sm_120 here), landing on eager, which is what stock
picks for this model anyway. **§8: no knob** — every one of these is a correctness fix with no user choice
in it. Probes: `a3_streamed_forward.py`, `a3_moe_block_compare.py`, `a3_layer_trace.py`, `a3_rope_cause.py`.

**A3 Task 5 DONE (2026-09-21) — and the investigation found a CORRECTNESS bug where the plan expected a
sizing question. Result: gpt-oss-20b is now BIT-IDENTICAL to stock transformers, logits sha256 equal at
BOTH 4 tokens (`7bfd3bb0…`) and 256 tokens (`6668afdd…`), every traced tensor `max|diff| = 0`.** The plan
asked whether sliding-window attention is a sizing tweak or a subsystem. **It is both, and the halves split
cleanly:** **(a) CORRECTNESS — FIXED here.** transformers builds TWO masks and hands each layer the one its
type asks for (`modeling_gpt_oss.py:492`, `causal_mask_mapping[config.layer_types[i]]`); **Vaayu's layer
loop passed ONE mask to every layer**, so a windowed layer attended its whole history. Measured: at 256
tokens (window 128) Vaayu's **top-1 token differed from stock** (18 vs 17). `get_attention_mask_args` now
takes the layer and applies transformers' own overlay (*attend iff `kv_idx > q_idx - window`*) for any
layer whose attention module declares a window — capability, not an architecture string — cached per
(window, size, dtype). `tests/test_sliding_window_kv.py`, **9 tests**, parametrized over two window sizes.
**(b) A SECOND, BIGGER CAUSE, found by the same trace and it is NOT gpt-oss-specific — it affects EVERY
model Vaayu has ever run: `_precompute_rope` built the rotary module in `running_dtype`, so `inv_freq` —
a table of FREQUENCIES — was bf16.** A relative error there grows with the position it multiplies:
measured cos/sin **0.0039 off stock at position 3 (one bf16 ulp) and 0.51 off — 38 % of full scale — at
position 255**, which is what moved the top-1 token. Stock keeps that table in fp32 for every model. Now
built in fp32 with cos/sin cast back to the run dtype; **this changes numerics slightly for Qwen3-Next,
MiniMax-M2.7 and both Llamas, in the direction of stock transformers** (and is the reason gpt-oss now
matches bit-for-bit). Gates after both changes: suite **2406 passed / 16 skipped**, Qwen3-Next accumulator
**8**, MiniMax accumulator **8**, Llama-3.2-1B prefix **1**. **§8: no knob** — neither is a user choice.
**(c) THE SIZING HALF IS A SUBSYSTEM → NOT done here, opened as row 47-m4 below with its measured cost**,
exactly the fallback the plan allows.

**A3 Task 6 DONE — A3 IS CLOSED, 2026-09-21. gpt-oss-20b GENERATES THROUGH `vaayu/server`, and
row 47's mandate now holds for a FOURTH architecture family (MXFP4 · fused-3D experts ·
sliding-window attention).** **Server gate on this checkpoint: 9 passed** (472 s). **Generation
(`test_generate.py`, its own code prompt, `MODEL_PATH` pointed at gpt-oss locally and NOT committed):
497 prompt tokens → 254 output, TTFT 33.17 s, decode 0.17 tok/s (5.8 s/token), load 7.6 s, 24.85 min end
to end.** Text READ, not just gated — first characters verbatim: *"analysisWe need to produce a single
Python file async_watcher.py implementing the described classes, context manager, CLI, etc. Must be
production-grade, with type hints, error handling, no external deps. Use asyncio, pathlib, logging,
argparse, signal."* — continuing into a correct design (snapshot dict of `Path → (mtime, size)`, set
difference for created/deleted/modified, debounce on last-event time). Coherent, on-task English. **Two
honest notes on its shape, neither a Vaayu defect:** it is gpt-oss's **`analysis` channel** (the model
speaks the harmony format and this script sends a plain prompt, not its chat template), and the
256-token ceiling ended the run before the `final` channel. **On 0.17 tok/s — REPORTED, NOT TUNED (this
row buys REACH):** the per-token line reads `bytes=13,281,935,360  gb_tok=13.28` — the WHOLE model moves
per token (10.17 GB experts + a 1.10 GB embedding + a 1.10 GB head, neither of which fits the pin budget
here) against Qwen3-Next's 0.030 GB/token — and `compute=5.2s` of a 5.8 s token against `io_wait=0.3s`,
so **most of the token is NOT the disk**; the per-visit MXFP4 swizzle is inside that 5.2 s and separating
the two is A6's first measurement. **A6 (per-expert MXFP4 streaming, 1.27 GB working set instead of
10.17 GB) now has a baseline to beat.** Final gate sweep: suite **2416 passed / 16 skipped** (2393
entering A3, +23 tests), Qwen3-Next accumulator **8**, MiniMax-M2.7 accumulator **8**, Llama-3.2-1B
prefix **1**, gpt-oss server **9**. **§8 for A3: (1) NO KNOB — every change is a correctness fix or a
layout detection, nothing added to `vaayu/server/config.py`; (2) nothing new to plumb — the dispatch runs
on every load; (3) REACHABLE at N=1 through the server's own knobs, which is where it was measured;
(4) banked: nothing from A3 itself — the two costs it declined to pay are rows 47-m4 and 47-m5.**
User-facing documentation updated in the same commit (`documentation/moe_expert_aware_streaming.md`
§3.4b: what a fused-3D checkpoint is, the two dispatch conditions, what whole-layer delivery costs, and
the eager-attention note). RESULTS: [`…-A3-gptoss-RESULTS.md`](docs/2026-09-20-A3-gptoss-RESULTS.md)
(gitignored — this row is the durable copy).

**Three follow-up rows, opened from A1's final review (§6 — they must not vanish):**
- [ ] **47-m1 — `_detect_moe` does not recognize `n_routed_experts`** (`vaayu_base.py:3483-3489`
      matches only `num_local_experts` / `num_experts` / a `Moe|MoE|Mixtral` arch string). A
      checkpoint whose only expert field is `n_routed_experts` — **the DeepSeek shape, which
      `_EXPERT_COUNT_FIELDS` explicitly claims** — gets `is_moe=False`, so the index is never built
      and G1 never runs. Same for `device="cpu"` (`server/config.py:27` is user-settable). NOT
      silent-wrong output: with `is_moe=False` Vaayu streams whole layers and the experts come
      along. A coverage-claim gap, not a correctness hole. Fix: add `n_routed_experts`, or gate on
      `expected_expert_count() is not None`. **Bites row 28b/A5 directly.**
- [ ] **47-m2 — `expert_read_buffer_bytes` (`safetensor_model_persister.py:888`) still builds the
      index config-less**, and it runs FIRST (`vaayu_base.py:2210`, before `:2295`). For a fused-3D
      checkpoint it expects nothing, returns 0, and falls back to the 512 MB pinned-buffer guess.
      Harmless today — the real guard raises later in the same load — but the ordering is undocumented.
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
- [x] **47-m5 — a whole-layer-dispatch model still pays for the expert cache it never touches. CLOSED 2026-09-23 — BUILT, MEASURED, ALL GATES GREEN, REACHABLE THROUGH THE SERVER.**
      **gpt-oss-20b load RSS 19.20 → 3.81 GB (−15.39 GB), available RAM 11.59 → 24.59 GB (+13.00), and the output is BIT-IDENTICAL** — `prefill_sha256 c78ba616…5700d38` and `decode_logits_sha256 c02f5a07…27cd2b1a` matched across the two arms, as did all 32 greedy ids and the forward census (`_make_whole_layer_sparse_forward` ×24). Same box, back to back, 29G scopes, baseline taken with `vaayu/` stashed so it was the same tree minus exactly this change.
      **The row over-counted by two of three: only `SlabArena` allocates.** `ExpertRAMCache` and `ExpertVRAMCache` are empty `OrderedDict`s at construction — they cost a budget NUMBER, not bytes. The arena was the whole cost.
      **ROOT CAUSE: the guard and the dispatch asked different questions.** `_moe_slab_capable` (`vaayu_base.py:2657`) answers from `expert_style`, and a fused-3D checkpoint's style is legitimately `"fused"`, so it said yes — while `_make_sparse_moe_forward` (`moe_patch.py:1364`) went on to install `_make_whole_layer_sparse_forward` using two facts the style predicate cannot see (the index holds `fused_layer_tensors` for the layer AND `fused_param_experts(experts_mod)`), and that forward takes NO `expert_cache` argument at all.
      **FIX: the whole-layer condition is now written ONCE — `moe_patch.dispatches_whole_layer(block_info, offset_index)` — and the dispatch site and the arena guard both call it**, because moe_patch's own docstring warns that a second copy of that table drifts. The guard uses **`all`, not `any`**: a mixed model's other blocks do read the arena, so it is skipped only when EVERY block dispatches whole-layer — strictly a reduction, never a removal from something in use. The SKIPPED message's old *'cached via the clone path instead'* was false here (nothing is cached at all) and whole-layer got its own wording.
      **THE REGRESSION THAT MATTERED, and it is guarded:** `fused_param_experts` returns True for `Qwen3NextExperts` too (`gate_up_proj [512, 1024, 2048]`) while its checkpoint stores one tensor per expert. Keying on the module alone would have taken Qwen3-Next's arena away and sent every miss upload pageable — **row 30-P measured that at −10.62 % decode**. **Arena KEPT and verified on: Qwen3-Next 19.16 GB (probe) / 24.31 GB (accumulator) / 18.87 GB (server), MiniMax 23.14 GB.** Qwen3-Next's shas on this branch are IDENTICAL to 47-m7's same-day measurement on unmodified `main`, so the change is bit-neutral there too.
      **Gates: suite 2422 passed / 16 skipped** (2416 + 6 new tests, parametrized over the gpt-oss-like and Qwen3-Next-like geometries per §5); accumulator Qwen3-Next 8, MiniMax 8; 1B prefix 1; server Qwen3-Next 9; **server gpt-oss 9**. The gpt-oss accumulator gate is not a target — it asserts `_expert_vram_cache.hits > 0` as its own premise and this forward never touches that cache.
      **§8: (1) NO KNOB — derived from the checkpoint layout and the live module tree, and a knob could only let the two sites disagree again, which is the defect. (2) nothing to plumb. (3) REACHABLE, proven directly: the server smoke gate on gpt-oss logs `Expert slab arena: SKIPPED … Freed ~18.5 GB of pinned RAM` and `avail=27.13GB rss=3.78GB` at N=1 on server defaults — NOT banked. (4) n/a.** Note the freed figure moves between arms (15.97 / 18.45 / ~18.5 GB) because the arena is sized from free RAM AT LOAD — quote the arm, never a single number. [`…-47m5-whole-layer-arena-RESULTS.md`](docs/2026-09-23-47m5-whole-layer-arena-RESULTS.md). **Merged to `main` as `41a2108` (merge commit `4ad6138`); branch `m5-whole-layer-arena` deleted.** Original text follows.
      Found by A3's own generation run (2026-09-21). `_make_whole_layer_sparse_forward` reads the layer
      itself and consults NEITHER the expert RAM cache nor the VRAM expert cache, yet the engine sizes and
      allocates both from `_moe_geometry` at load: measured on gpt-oss-20b, **`Expert slab arena: 1388 x
      13.3MB in 6 power-of-two chunks (pinned, 18.45 GB charged)` and a per-token telemetry line reading
      `expert_cache=0hit/0miss/0evict … usage=0.00GB/18.31GB` for every token of a 256-token generation** —
      18.45 GB of PINNED (unswappable) RAM on a 32 GB box, doing nothing, with RSS at 25.87 GB against the
      29 GB gate ceiling. Fix sketch: the engine already runs `detect_moe_blocks` before sizing the arena
      (that is why `patch_moe_blocks` takes `blocks=`), so it can ask the same two questions the dispatch
      asks — index holds `fused_layer_tensors` for the layer AND `fused_param_experts(experts_mod)` — and
      size the arena to zero for those layers. Derived, no model number. Related to row **40** (the arena is
      never capped at the model's own expert bytes) but distinct: this one is about a model whose forward
      never reads the cache at all. Touches shared sizing → all three accumulator gates.
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
- [x] **47-m7 — VERIFY that A3's FP8-compat gating is inert for Qwen3-Next. CLOSED 2026-09-23 — IT IS INERT, BYTE-IDENTICAL, zero code shipped, branch deleted.**
      **The numbers, so this row survives without its doc:** both arms on `main` at **`753b104`**, same box, same day, back to back, each in its own 29G foreground scope; `ungated` = that commit with ONE hunk reverted (the `if isinstance(...)` line deleted, the call dedented), `cac352e`'s fp32 rotary LEFT IN on both so there was exactly one variable. **`prefill_sha256` `c45857cd6fade36c6f4e7187872274b48ae5e0938085daad7668ef74943bd77d` and `decode_logits_sha256` `cf886697a59edd2fee4aba7b14e4105f9a03c8d05ef931b071c930d58827f958` are IDENTICAL across the two arms**, as are all 32 greedy decode ids, `attn_impl=flash_attention_2`, the forward census (`_make_fused_sparse_forward` ×48) and the module tree (48 `experts`, all `Qwen3NextExperts`). **NEGATIVE CONTROL PASSED — the arms really did differ:** the `ungated` log carries `[Vaayu] FP8 compat: set modules_to_not_convert = ['experts']` plus the Blackwell deep-gemm disable, and the `gated` log has ZERO `FP8 compat` lines. **MECHANISM, now verified rather than assumed and SIMPLER than the handoff predicted: `transformers/quantizers/quantizer_compressed_tensors.py` (5.5.4) contains ZERO occurrences of `modules_to_not_convert`** — it delegates to `compressed_tensors.quantization.apply_quantization_config` with the library's own config — **so the patch set a field nobody on this path reads**, and the quantizer replaced no experts module in EITHER arm. The handoff's own reasoning (the fused non-EXL3 forward computes the FFN from streamed bytes and never reads the module's parameters) also holds, but it is not the proximate cause. **Consequence: A3's ONLY behaviour change on Qwen3-Next is the fp32 rotary table (`cac352e`) — that is 47-m6 and it is STILL OPEN.** §8: no knob, nothing to plumb, nothing to reach; no gates run because no code changed. Probe `plans/probes/m7_fp8_gating.py` (a3_streamed_forward.py was UNUSABLE for this — its reference prompt ids are gpt-oss's, max **165916** against Qwen3-Next's **151936** vocab). [`…-47m7-fp8-gating-verification-RESULTS.md`](docs/2026-09-23-47m7-fp8-gating-verification-RESULTS.md). Original text follows. Opened 2026-09-21, by the
      session that made the change, because it was NOT verified there. **FRESH-SESSION HANDOFF, fully
      self-contained: [`…-fp8-gating-verification-HANDOFF.md`](docs/2026-09-21-fp8-gating-verification-HANDOFF.md)**
      (gitignored — this row is the durable copy). A3 gated `_patch_fp8_quantizer_compat`
      (`vaayu_base.py:3184`, call site `:3396`, commit `35c2b8f`) behind `FineGrainedFP8HfQuantizer`; it used
      to run for EVERY quantizer, appending `"experts"` to `modules_to_not_convert`. That is why gpt-oss could
      not run, and gating it is a BEHAVIOUR CHANGE for any non-FP8 quantizer that was silently receiving it.
      **Scope is exactly ONE model** (checked against each `config.json` and `vaayu_base.py:3383`'s exl3
      skip): **Qwen3-Next (`compressed-tensors`)** — MiniMax and Llama-3.3-70B are exl3 and never reach
      `AutoHfQuantizer`, Llama-3.2-1B has no `quantization_config`, and gpt-oss is the intended beneficiary.
      **Why the green gates do not answer it:** suite 2416, both accumulator gates and the 1B prefix gate all
      test determinism and self-consistency — **none asks whether this build matches the PREVIOUS build.**
      **The test is one variable:** revert ONLY the gating hunk (NOT the fp32 rotary change from `cac352e`,
      which moves Qwen3-Next's numerics by design — that is 47-m6) and compare greedy ids + logits sha on
      Qwen3-Next. **Expected inert**, because the fused non-EXL3 forward never reads the experts module's
      parameters — it streams each expert off disk and computes the FFN itself — so module replacement cannot
      reach the arithmetic. **Stop rule: identical ids and sha → close with the shas; different ids → STOP,
      diff the module tree on both builds and record what the quantizer now converts before changing
      anything.** ~30 minutes, ships no code.
- [ ] **47-m5-b — a whole-layer model still ADVERTISES caches it will never fill.** Split out of
      47-m5 2026-09-23 rather than folded in (§3: every changed line traces to the request, and the
      request was the pinned RAM). With the arena correctly skipped, gpt-oss-20b still prints
      `Expert RAM cache: 16.0 GB budget (~1205 expert slots, max 32/layer, ...)` and `Expert VRAM cache:
      0.5 GB (~37 experts resident on GPU)` for a forward that reads neither. **Neither costs BYTES** —
      both are empty `OrderedDict`s at construction, verified while closing 47-m5 — so this is
      misleading telemetry, not a leak, and it is why the row's 'sizes and ALLOCATES both' was an
      over-count. Cheap fix: the same `_moe_whole_layer_only` flag the arena guard now sets can zero
      these budgets or say `n/a (whole-layer dispatch)`. Check first whether anything downstream reads
      `_expert_cache.budget_bytes` as a real number — the speculative-decode gate does
      (`spec_decode_fits`), so zeroing it is NOT obviously free.
- [ ] **47-m3 — shard-name arithmetic assumes 5 digits and 1-based numbering.**
      `checkpoint_integrity.py` uses `_SHARD_RE` with `\d{5}` and `set(range(1, total + 1))`.
      **`gpt-oss-20b` on disk is `model-00000-of-00002` … `model-00002-of-00002` — 0-based, THREE
      files against total=2** — which yields no raise today (correct), but a missing `00000` is
      invisible and `len(present) > total` is silently ignored. Also: a PARTIAL `delete_original`
      (an `OSError` at `utils.py:276` is caught and logged) leaves shards plus the retained index,
      and the `weight_map` check would then raise on a model that had been loading fine.

**A4 IS CLOSED — 2026-09-23, PREMISE REFUTED, NO ENGINE CODE, NO KNOB.** The gap it was written to
close does not exist. Full detail:
[`…-A4-shared-experts-RESULTS.md`](docs/2026-09-23-A4-shared-experts-RESULTS.md) (gitignored —
**these bullets are the durable copy**).

- [x] **Q1 — what does Vaayu do with Qwen3-Next's shared expert today? ANSWER: it is already PINNED
      in VRAM, never streamed.** Three links, each read, none assumed: (1) the routed regex
      `_EXPERT_RE = r'\.experts\.(\d+)\.'` (`expert_offset_index.py:8`) does **not** match
      `model.layers.N.mlp.shared_expert.{gate,up,down}_proj.weight` or `...shared_expert_gate.weight`,
      so the index never claims them; (2) `load_non_expert_tensors`
      (`safetensor_model_persister.py:541`) skips **only** indexed expert keys, so anything unclaimed
      is non-expert by construction; (3) `_init_pinned_layers` Phase 2 (`vaayu_base.py:3940`) pins that
      set, forcing `effective_max_pinned = -1` for MoE (`:3941`), and the server default is
      `MAX_PINNED_LAYERS = -1` (`server/config.py:96`, plumbed `app.py:47`/`:407`) — **so this is the
      mode a real user gets (§8 q3).** Measured on a real load (`docs/probes/r36-recheck-1.log`, `main`
      `da118ab`): **`48 non-expert-pinned transformer, 4.83 GB reserved` — 48 of 48, the loop did not
      break on budget.** Per-layer pins `0.06`/`0.08 GB` match the shard headers exactly (layer 3 full-attn
      **62.93 MB**, layer 4 linear-attn **75.84 MB**), and inside layer 3 sit the four shared-expert
      tensors at **6.2956 MB** total — **~302 MB across 48 layers, BF16, left un-quantized by AWQ**
      (derived from `hidden_size=2048` × `shared_expert_intermediate_size=512` × 3 projections + a
      1-logit gate; no literals, §5).
- [x] **Q2 — is there a measurable cost today? ANSWER: zero, and the stop rule said not to spend the
      GPU run.** Q2 was conditional on Q1 showing per-token streaming. Nothing streams, so there are no
      `read_bytes` to attribute. `plans/probes/row26_decode.py` was never invoked.
- [x] **The check the handoff did NOT ask for, and the load-bearing one: the compute path BINDS on the
      real module tree.** Q1 proves bytes were pinned; it does not prove `_shared_expert_output`
      (`moe_patch.py:536`) finds the module. That helper scans `moe_mod.named_children()` for
      `"shared_expert"` and returns `None` on a miss — **silently dropping the contribution from the
      math**, the exact failure its own docstring names. `tests/test_shared_expert.py` covers both
      shapes on all three forwards but with **`MockSharedExpert`**, so it could not catch a real-tree
      mismatch (A3's lesson 1). Verified directly, meta device, 4 layers, no GPU and no weights:
      `detect_moe_blocks` returns `style: fused`, `moe_module: Qwen3NextSparseMoeBlock`, children
      **`['gate', 'experts', 'shared_expert', 'shared_expert_gate']`** — both visible to the scan, and
      the **gated** branch (`name.endswith("_gate")`) is the one that fires.
- [x] **All three routed forwards call it** — `_make_modulelist_sparse_forward` (`:1856`),
      `_make_fused_sparse_forward` (`:2280`), `_make_fused_sparse_forward_exl3` (`:2946`). **This
      CORRECTS the A4 brief**, which implied `_make_whole_layer_sparse_forward` (gpt-oss) was a coverage
      risk: it is not — that forward delegates to `original_forward`, so the stock block does its own
      shared math.
- [x] **DeepSeek key shapes read off disk:** `layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}`,
      **276 keys, 0 matching the routed regex** (the char before `experts` is `_`, not `.`), so they
      would land in the same non-expert set and `_shared_expert_output`'s **ungated** branch would fire.
      **This is a key-shape argument, NOT an end-to-end result** — DeepSeek-V4 still does not load.
- [ ] **CARRIED TO A5 (row 28b) — check 1: confirm DeepSeek's shared experts actually land in the
      pinned non-expert set** once the model loads. The key-shape argument above, executed.
- [ ] **CARRIED TO A5 (row 28b) — check 2: confirm the pinning budget covers DeepSeek's non-expert
      set.** 43 MoE layers, MLA attention, F8_E4M3 shared experts and attention — unmeasured. **Why it
      matters: the pinning is INCIDENTAL to the budget, not structural.** Phase 2 `break`s on budget
      miss (`vaayu_base.py:3956`); Qwen3-Next's ~3.55 GB fits inside 4.83 GB today, but **row 46 and row
      42-B move exactly that number.** Un-pinned layers fall back to `LRUTensorCache(_ne_budget)`
      (`vaayu_base.py:2710`, byte-budgeted, derived) — RAM-cached re-reads, **not** per-token NVMe
      streaming — and the loop does print `budget exhausted at <layer>`, but that warning says nothing
      about shared experts specifically.
- [x] **§8 — does it need a knob? NO, and stated rather than assumed.** Nothing was built, so there is
      nothing to expose. The behaviour rides on `MAX_PINNED_LAYERS`, which already exists, already
      defaults to `-1`, and is already documented at `vaayu/server/README.md:65`.
- [x] **Gates: none run and none owed** — no engine code, no test, no default moved. The meta-device
      check allocated no weights and touched no GPU.

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

**The gap list below (G0–G7) is the source material the plans were cut from. G2 is already closed;
G6 is closed as a perf idea (rows 26/28) and is reach-only.**

**This is the row that broadens REACH, not speed. It does not rank against the perf queue (36/35/32/46/34)
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

**The seven gaps, in bite-order (G1 first because it is the only one that fails SILENTLY):**

- [x] **G0 — an incomplete or inconsistent checkpoint must fail LOUDLY at persist time.** This checkpoint WAS 46-of-48
      with no index for part of 2026-09-20 — a download that stopped, not an exotic case — and RC5 already shipped a
      `.done` marker over dropped tensors once. Verify shard count against the `-of-NNNNN` suffix AND that every
      `weight_map` ref resolves, before writing anything. Still worth building after this checkpoint completed: the
      next one will not be complete either.
- [x] **G1 — NOW REPRODUCIBLE, AND IT HAS A PLAN PROMPT. `gpt-oss-20b` arrived on disk 2026-09-20 and IS the
      silent failure.** Measured: **`r'\.experts\.(\d+)\.'` matches 0 of its 459 tensors**, because its experts
      are one 3D blob per layer (`model.layers.N.mlp.experts.gate_up_proj_blocks`, U8 `[32, 5760, 90, 16]` — the
      leading 32 IS the expert axis). **transformers 5.5.4 supports `gpt_oss` natively** (`AutoConfig` →
      `GptOssConfig`, `modeling_gpt_oss.GptOssExperts` is a fused-3D forward), so there is **no modeling code and no
      kernel port in scope** — this is Tier 0, "supported, just never run here", and the whole gap is Vaayu's
      persist/streaming path. `detect_moe_blocks` reads the MODULE tree, so it classes this `"fused"` non-EXL3 →
      `_make_fused_sparse_forward`, **the same forward Qwen3-Next runs and rows 30-P/44/42-B tuned** — the compute
      path is exercised, the persist path has never seen a 3D layout. **13.8 GB · 24 layers · 32 experts top-4 ·
      13.24 MB/expert · 423.6 MB/layer · 10.17 GB of experts · top-4 working set ≈ 1.27 GB · MXFP4 (U8 blocks + U8
      scales + BF16 expert BIASES, which neither on-device model has) · alternating sliding/full attention,
      `sliding_window: 128`.** **ABANDON GATE, settle it FIRST: does MXFP4 run on this box, or does the only path
      dequantize to bf16 (~40 GB of experts — does not fit)? If bf16-only, STOP and fall back to the raise.**
      Brief + the fresh-session plan prompt:
      [`…-row47-G1-gptoss-BRIEF.md`](docs/2026-09-20-row47-G1-gptoss-BRIEF.md).
      The original gap text follows.
- [x] **G1 (original) — `ExpertOffsetIndex` understands exactly ONE key scheme** (`vaayu/persist/expert_offset_index.py:8`,
      `_EXPERT_RE = re.compile(r'\.experts\.(\d+)\.')`). **Every fused-3D checkpoint is unsupported SILENTLY** — it
      finds zero experts rather than raising, and the docstring's claim that it "covers … any future architecture"
      is exactly the trap. Fused-3D (`experts.gate_up_proj` shaped `[E, …]`) is what modern transformers MoE ships.
      **Minimum: raise a clear error when the config implies experts and the index finds none. Real fix: a persist-time
      3D splitter.** Do the minimum first — it converts a silent wrong answer into a message.
- [ ] **G2 — ~~`_make_fused_sparse_forward` (non-EXL3) has never executed on device~~ CLOSED by row 30's S2
      (2026-09-13).** Qwen3-Next runs it daily and 30-P/44 perf-tuned it. Kept here so the playbook's stale line is
      not re-copied into a plan.
- [ ] **G3 — only ONE of the three MoE forwards takes `compute_device`** (`_make_fused_sparse_forward`); the other two
      read the device off the loaded modules (`moe_patch.py:819`, `:837`). Any device-placement feature wired into one
      forward silently no-ops in the others — **this is exactly how `VAAYU_CACHE_BIAS` came to do nothing on either
      model** (SETTLED, do not rebuild that).
- [ ] **G4 — uniform-attention assumption.** KV sizing, the prefix cache and derived `max_seq_len` assume every layer
      attends the same way. **Partly paid down already:** row 30 RC4/RC6 and row 42-B handled hybrid linear/full for
      Qwen3-Next. **Still unhandled: sliding-window and sparse-index attention (DeepSeek-V4).** Check before trusting
      any TTFT or context number on such a model.
- [ ] **G5 — slab scale.** `SlabArena` (`vaayu/persist/expert_cache.py:25`) was exercised at ~1,000–1,250 slabs of
      8–12 MiB. Qwen3-Next is already 24,576 slabs of ~1.6 MiB and works, so this is partly disproven — but
      `_NVME_READ_CONCURRENCY = 3` (`safetensor_model_persister.py:181`) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
- [ ] **G6 — no CPU expert compute exists at all.** `compute_device` is forced to `"cuda"` for EXL3; the only CPU mode
      is running a whole non-EXL3 model with `device="cpu"`. A hybrid is new construction, not a flag. **Row 26/28
      already priced the CPU-AVX2 hybrid and RECOMMENDED IT CLOSED** (+6.8 % Qwen / +22.0 % MiniMax ceiling, and a
      non-EXL3 model makes it worse) — so treat G6 as REACH-only, never as a perf lever.
- [x] **G7 — ~~shared experts have no special handling~~ CORRECTED AND CLOSED 2026-09-23 by A4. As written this was
      HALF FALSE, and leaving it would have sent a session to build a mechanism that already exists.** (a) **Compute
      IS handled, deliberately** — `_shared_expert_output` (`moe_patch.py:536`) is called by all three routed
      forwards (`:1856`, `:2280`, `:2946`), handles both the gated (Qwen) and ungated (DeepSeek) shapes, and is
      unit-tested for both in `tests/test_shared_expert.py`; it was verified to BIND on Qwen3-Next's real module tree
      (children `['gate','experts','shared_expert','shared_expert_gate']`), not just against mocks. (b) **Pinning is
      NOT special handling** — shared experts land in VRAM because the routed regex does not match them, so they fall
      into the pinned non-expert set (48/48 Qwen3-Next layers, 4.83 GB). **That is INCIDENTAL and budget-dependent:
      Phase 2 `break`s on budget miss.** The two things still genuinely unknown are DeepSeek-specific and are carried
      as explicit A5 checks in the A4 bullets above, not left here as an open gap.

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

# 46 · [~] The prefix reservation is sized to `MAX_SEQ_LEN`, not to what the pool actually holds — 4.74 GB on Qwen3-235B, 6.24 GB on MiniMax, live today (added 2026-09-20, split out of row 42-B) — **RE-PRICED DOWN 2026-09-20: the prize is ~2.6 %, not ~15 %**

**FRESH-SESSION HANDOFF, RE-VERIFIED 2026-09-23: [`…-2026-09-23-row46-prefix-reservation-HANDOFF.md`](docs/2026-09-23-row46-prefix-reservation-HANDOFF.md)**
(gitignored; this row remains the durable copy). **It SUPERSEDES the 2026-09-21 handoff, which is stale in
four places — do not read that one.** Step 1 is a MEASUREMENT on MiniMax-M2.7 with a stop rule — **under
~3 % and the row closes**, the way Qwen3-Next's measured 2.56 % already closed the case there.

**STEP 1 IS DONE — 2026-09-23 — AND IT FIRES: +13.36 % MiniMax decode. THE STOP RULE DID NOT CLOSE THE
ROW.** ABBA `off,base,base,off`, 4 processes x 2 in-process replicates, `PREFIX_SESSIONS=0` vs `=3` through
`plans/probes/row26_decode.py` (`MODEL=minimax NEW_TOK=64 ARMS=W,U1,U2 BALLAST_GB=28`, 29G scope):
**off 2.3467 vs base 2.0701 tok/s**, within-arm spread 1.39 / 1.38 % — ~10x smaller than the separation, and
ABBA-balanced. **Greedy ids bit-identical across all 8 arms.** Mechanism measured, not assumed: RAM-cache hit
**20.0 -> 35.7 %**, expert disk reads **15525 -> 12484** per 63 decode forwards (**-19.6 %**), bytes/token
**2.19 -> 1.76 GB**, blocking stall **43.1 -> 36.2 %** of decode wall. **Qwen3-Next's -2.56 % does NOT carry**
(0.42 % disk at a 96.6 % hit there vs 20 % here). RESULTS:
[`...-row46-step1-minimax-AB-RESULTS.md`](docs/2026-09-23-row46-step1-minimax-AB-RESULTS.md).

**AND THE SIZING PREMISE IS NOW REFUTED — 2026-09-23, the session-bytes spike.** The row was sized off
"a stored 8k session holds 0.20 GB" against a `max_seq_len` reservation. Measured on device by storing real
sessions and reading the pool entry's own `nbytes` (`plans/probes/row46_session_bytes.py`, two lengths per
model so `bytes(n)=a*n+b` is determined): **MiniMax 248.0 KiB/token** (+0.25 MB fixed), **Qwen3-Next 24.0
KiB/token + 40.13 MB fixed** (the 36 linear layers' recurrent state). **The engine's accounting is EXACT on
both** — `derive_expert_ceiling` is fed `kv_bearing_layer_count(self)` (`vaayu_base.py:2691`), not
`config.num_hidden_layers`, so it already charges 12-of-48 layers on Qwen3-Next. **There is no accounting bug
here to fix**, and 47-m4's "counts every layer" concern does NOT apply to this path.

| | MiniMax | Qwen3-Next |
|---|---|---|
| `max_seq_len` (from the VRAM plan, `:2446`) | **8192** | **65536** |
| reserved / session | 2.080 GB | 1.610 GB |
| real session @ 8000 tok | 2.0319 GB = **1.02x** | 0.2367 GB = **6.80x** |
| real session @ 2048 tok | 0.5203 GB = 4.00x | 0.0905 GB = 17.80x |
| **recoverable, 3 x 8k** | **0.14 GB** | **4.12 GB** |
| decode worth of returning it all | **+13.36 %** (step 1) | **-2.56 %** (42-B) |

**THE PRIZE AND THE RECOVERABLE BYTES ARE ANTI-CORRELATED, and not by accident.** MiniMax's `max_seq_len` is
8192 *because that is all this box's VRAM allows*, so a full-context agent session IS 8192 tokens and the
reservation is exactly one of them. "Size to what the pool holds" can only return bytes there by assuming
sessions stay ~4x shorter than the context the engine advertises -- and the arena is pinned at load and
cannot shrink, so a wrong guess buys row 31's collapse at full price. Qwen3-Next is the mirror image: 4.12 GB
genuinely idle, worth -2.56 %. **The handoff's "0.20 GB per 8k session" was Qwen3-Next's numbers read onto
MiniMax** (0.201 GB is exactly Qwen3-Next's formula 2k figure) -- that single cross-model transfer is what
made the reservation look 8x over everywhere. RESULTS:
[`...-row46-step2-session-bytes-RESULTS.md`](docs/2026-09-23-row46-step2-session-bytes-RESULTS.md).

**§8, settled for the whole row: NO NEW KNOB is needed.** `PREFIX_SESSIONS` (`server/config.py:193`) and
`MAX_SEQ_LEN` (`:128`) already span every trade this row could have added, and row 31 swept them and closed
with "defaults STAY".

**BUT THE ARM THAT WON IS NOT SHIPPABLE, AND THAT IS NOW MEASURED TOO.** `compute_prefix_cache_budget`
(`vaayu_base.py:705`) runs AFTER the arena is pinned and takes what is left minus a 3.76 GB reserve, so the
returned bytes come straight out of the pool: free RAM post-arena **9.66 -> 3.61 GB** and the **prefix pool
budget 5.90 -> 0.00 GB**, which prints the `concurrent sessions collapse to one resident entry` WARNING.
That is row 31's collapse as a MECHANISM rather than an inference. **So +13.36 % is this lever's UPPER BOUND
and the design's job is how much of it survives keeping the pool.** The arithmetic that makes that look
possible: the reservation charges `MAX_SEQ_LEN` 8192 x kv/token = **2.08 GB per session** while a stored 8k
session is believed to hold **0.20 GB** -- **verifying that 0.20 GB figure ON MINIMAX is the next step, because
every candidate design is sized off it.**

**TWO TRAPS FOUND WHILE MEASURING, neither in the handoff:** (a) **`row26_decode.py:36` defaults to
`MODEL=qwen`, the DELETED Qwen3-235B path** -- always pass `MODEL=`; (b) **`BALLAST_GB` can silently make the
arm variable inert**: `compute_expert_budget` takes `min(ceiling, available - overhead)` and the arena then
trims to real free RAM, so at the probe's default ballast the arena lands ~11 GB / ~1250 slabs -- BELOW both
arms' ceilings, and both arms then allocate the same arena. **Run the 1-token binding check first**
(`NEW_TOK=1 ARMS=W`, both arms): at `BALLAST_GB=28` the arena tracked the ceiling 1:1 (14.09 GB / 1593 slabs
vs 20.60 GB / 2325 slabs, **+46 % slots**).

**§8, settled for step 1:** the win is REACHABLE at N=1 on server defaults -- `PREFIX_SESSIONS` is already a
knob (`server/config.py:193`) and `derive_expert_ceiling` runs at load. Not an N>1-only effect. Whether the
FIX needs a new knob is still step 2's to state.

**RE-VERIFIED ON DEVICE 2026-09-23 — five corrections, all measured under SERVER knobs, so the row is
pickable cold without the doc:**
1. **MiniMax-M2.7 reserves 3 × 2.08 = 6.24 GB and it is LIVE** (`Expert cache ceiling 16.11 GB … reserving
   3 prefix session(s) of 2.08 GB each`, of 26.11 GB available). **It is the ONLY live target.**
2. **Qwen3-Next reserves 4.83 GB TODAY** (ceiling 19.14 of 27.73 available). The old handoff said "0 today,
   4.83 GB once 42-B's RC7 lift applies" — **that lift HAS applied.**
3. **gpt-oss-20b is a THIRD payer (3 × 1.63 = 4.89 GB, ceiling 15.95 of 24.61) and on it this lever is
   INERT.** Since 47-m5 a whole-layer model allocates NO slab arena and its forward reads neither cache, so
   the reservation shrinks a budget NUMBER and no bytes. **Do not spend an arm on gpt-oss.**
4. **Line pointers drifted: `derive_expert_ceiling` is `vaayu_base.py:355`, called at `:2703`** (not
   `:354` / `:2651`).
5. **TRAP that silently zeroes an arm: `tests/test_moe_accumulator_determinism.py` loads with
   `enable_prefix_cache=False` (`:70`)**, so MiniMax under that gate prints `reserving 0 prefix session(s)`
   and a ceiling of 23.03 GB — the reservation looks already gone. **An arm MUST load on the server's own
   knobs** (`vaayu/server/config.py`), as `plans/probes/m7_fp8_gating.py` and `row26_decode.py` do.

**The magnitude of the prize, measured today (NOT the A/B — the two arms differ in more than the
reservation):** MiniMax with the reservation ON gets a 1820-slab / 16.22 GB arena and ~1814 cache slots;
with it OFF, 2599 slabs / 23.14 GB and ~2594 slots, same 4.40 GB working set. **+43 % cache slots.**

**Split out of row 42-B on 2026-09-20 at the user's call: 42-B lifts RC7 with the reservation AS-IS,
this row prices the reservation on its own, on the two models that pay it TODAY.** Splitting keeps
42-B's ABBA attributable (one change, one number) and puts this lever where its payoff actually is —
it needs no hybrid at all.

**AND THE MECHANISM IS NOW MEASURED (2026-09-20, row 36's gate re-check, `da118ab`): on Qwen3-Next the
expert RAM cache is NO LONGER the binding constraint, so this row's mechanism barely exists here.**
`disk_load` is **0.0006 / 0.0005 s, i.e. 0.42 / 0.32 % of `T_prof`** (was 0.0142 s, 8.41 / 8.46 % on the
30-P build), bytes/token 0.0260 / 0.0255 GB, RAM-cache hit **96.64 / 96.71 %**, blocking expert stall
0.0011 s. That explains 42-B's ~2.6 % rather than merely restating it, and it **generalises: any lever
whose mechanism is "give the expert RAM cache more RAM" is near-dead on Qwen3-Next.** **Do NOT carry
this across to MiniMax-M2.7** — it still streams experts and its 6.24 GB reservation is unmeasured
post-row-44. (Qwen3-235B's 4.74 GB is now moot: retired and deleted 2026-09-20.) Detail:
[`…-row36-gate-recheck-RESULTS.md`](docs/2026-09-20-row36-gate-recheck-RESULTS.md).

**RE-PRICED 2026-09-20 BY ROW 42-B's OWN A/B — READ THIS BEFORE COSTING THE ROW.** 42-B measured
both sides of this reservation directly on Qwen3-Next: the ON arms reserved 3 × 1.61 = **4.83 GB**
(expert ceiling 13.78 / 14.13 GB) and the OFF arms reserved **0** (ceiling 18.77 / 18.86 GB) — and
the decode difference between them was only **−2.56 %** (9.7364 → 9.4868 tok/s, 2 replicates each,
ABBA). **So handing the expert cache ~5 GB back is worth ~2.6 % on this model today, NOT the
+15.09 % row 30 measured on 2026-09-19.** Row 30's figure predates row 44 (F1+F2+F6+F5, shipped
2026-09-19), which changed how the non-EXL3 fused forward retains and dispatches experts.
**Hypothesis, untested: the expert cache is no longer the binding constraint it was, so extra
ceiling buys much less.** The 2.56 % is measured; the explanation is a guess worth one probe.
**Consequence: this row's prize is ~2.6 % on Qwen3-Next against 1-2 days of work with row 31's
collapse risk attached — price it accordingly, and note that the 4.74 GB (Qwen3-235B) and 6.24 GB
(MiniMax) reservations have never been measured post-row-44 either.** Detail:
[`…-row42B-hybrid-prefix-cache-RESULTS.md`](docs/2026-09-20-row42B-hybrid-prefix-cache-RESULTS.md).

**The number.** `derive_expert_ceiling` (`vaayu/vaayu_base.py:354`, called `:2651`) subtracts
`PREFIX_SESSIONS × MAX_SEQ_LEN × kv_bytes_per_token` from the expert RAM tier. Measured off real load
lines: **Qwen3-235B reserves 3 × 1.58 = 4.74 GB, MiniMax-M2.7 3 × 2.08 = 6.24 GB**, and Qwen3-Next
would reserve 3 × 1.61 = 4.83 GB the day 42-B lifts RC7 (today it reserves **0** — the row-30 fix,
ceiling 18.03 GB). **A stored 8k session holds 0.20 GB and a 2k one 0.05 GB**, so the default
over-reserves **8× at 8k and 32× at 2k**. Row 30 measured returning 4.83 GB of it as **+15.09 %
decode** on Qwen3-Next, so the prize here is of that order on two more models.

**THE TRAP, and it is why this is a row and not a one-liner (row 31, SETTLED — read it first).** Row 31
already refuted *cutting sessions*: `PREFIX_SESSIONS=1` measured **−8.9 %** on the two-session / 7.4k
workload, where the turn-2 match collapsed **7422/7517 → 44/7517** and TTFT went **25.0 → 51.3 s**.
Reuse does not degrade — it goes to ZERO one session past capacity (Plan I). Reserving less per session
fails the SAME way and for a subtler reason: **the expert arena is pinned at load and cannot shrink**,
so an under-reservation that guessed prompt lengths too low buys exactly row 31's collapse the first
time real prompts grow. Any design here must say what happens when the guess is wrong.

- [x] **Step 1 A/B on MiniMax — DONE 2026-09-23, +13.36 %, stop rule NOT met (see the block above).**
- [x] **Verify what a stored session ACTUALLY holds — DONE 2026-09-23 on BOTH models: the 0.20 GB
      figure was Qwen3-Next's, the accounting is exact, and the sizing premise is REFUTED (block above).**
- [ ] Price offline first (`vaayu/expert_sim.py`, `plans/probes/row31_price.py` — row 31's own tooling).
- [ ] Decide the mechanism: reserve to a measured typical length; grow the reservation dynamically; or
      shrink the pool instead of the arena. Each must answer the over-capacity case.
- [ ] Measure on **MiniMax AND Qwen3-Next** (§5 — a fix proved on one model is not done), including **row
      31's 7.4k two-session arm**, which is the arm that kills a naive version. **CORRECTED 2026-09-23: this
      read "Qwen3-235B AND MiniMax" — Qwen3-235B was retired and DELETED 2026-09-20, so that instruction was
      dead and would send a cold session after a checkpoint that is not on disk.** gpt-oss is NOT a target
      (correction 3 above).
- [ ] §8: `PREFIX_SESSIONS` and `MAX_SEQ_LEN` are existing knobs (`server/config.py:193` / `:128`); say
      whether the sizing needs a new one, or state that it does not.
- [ ] Gates: full suite, accumulator on Qwen3-Next AND MiniMax, the 1B prefix gate, the server gate on
      Qwen3-Next, **and the server gate on gpt-oss-20b** — `derive_expert_ceiling` is shared sizing and
      gpt-oss now loads through the server (added 2026-09-23).

---

# 23 · [ ] The prefill stager re-runs once per BATCHED CHUNK — stop it, and the fused forward's 2.36x may go

**Cold-pickup brief:** [`docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md`](docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md) (gitignored).

**Opened 2026-09-09 by row 22, which measured the cost but did NOT fix it. UNMEASURED — two
hypotheses, in cost order.** Row 22 established that `batched_generate`'s own chunk loop turns
one prefill into `c` forwards, and the stager (fused-EXL3 only) re-reads the full expert set on
each, costing `c x E` against serial's `N x E` with **E ~ 27 s on Qwen3-235B**.

* **Cheaper: suppress staging entirely for the batched chunk loop.** The fused forward would
  then behave like the module-list one and pay routed misses only. **MiniMax is the existence
  proof that this configuration wins** — it stages zero times and batches profitably at both
  c=2 (1.72x) and c=9 (1.58x, row 21). Risk: the stager exists to buy TTFT on long prompts, and
  the RAM expert cache (19.3 GB, ~13 of Qwen's 94 layers) cannot retain a full expert set across
  chunks, so per-chunk routed misses may not be much cheaper than staging. **Measure, don't
  assume.**
* **Better, and NOT a one-liner: make the stager span the batched chunk loop.** The machinery
  exists for the layer-major path — `_chunk_major_n`, `_staged_consume_count`, and the
  peek-vs-pop at `moe_patch.py:1778-1799` that keeps a staged layer alive across its chunk
  calls — and `batched_generate` simply never sets `_chunk_major_n`, so `_n_calls` is 1 and
  every chunk pops. **But the stager is armed AND torn down inside `forward()`
  (`vaayu_base.py:4430-4457`), so spanning it means hoisting that out of `forward()`.** That is
  real scope, not a flag.

**Worth zero to users today** (`MAX_BATCH_SIZE = 1`) — same §8 bucket as everything on the
batching path. Weigh it accordingly. `plans/probes/p22_crossover.py` measures either variant
directly; the c=9 Qwen arm (0.425x) is the number to move.

---

# 2 · [x] Static check for unbound locals / import order — CLOSED 2026-09-10: `pylint E0606` wired as `tests/test_no_unbound_locals.py`

Two bugs of exactly this shape shipped GREEN today, inside the same edit: `detect_moe_blocks` and
then `EXL3LoaderStrategy` were referenced in `vaayu_base.__init__` ABOVE their in-function
imports, making them unbound locals. **Both MoE models failed to load with `UnboundLocalError`
while the full 836-test suite passed** — nothing unit-tests the load path, and the first was found
by reading, the second only by running a real load.

**Task:** check whether `ruff` / `pyflakes` flags that class (undefined-name / used-before-assign).
If it does, wire it in — it would have caught both, immediately, for free. If it does not, say so
here and close the row; the fallback is already recorded in the gates table ("run a real load after
touching `__init__`").

**PROBED 2026-09-09 — the question above is ANSWERED, and the answer is SPLIT, so the row narrows
instead of closing.** Neither linter is installed in `./venv` (`No module named ruff` /
`No module named pyflakes`); both probes ran via `uvx`, which is why this cost minutes and not the
`~10 min` the table promised for the whole row.

* **`ruff` does NOT catch the class — do not wire it for this and think the row is discharged.**
  Checked against a synthetic reproduction of the exact shape (`def f(flag): if flag: x = 1` then
  `return x`, plus the empty-loop variant): `ruff check --select ALL` reports 9 findings on that
  file and **none of them is the unbound local** — it flags `ANN001`/`T201` and walks past the bug.
  `--select F821,F811,F841,E402` over `vaayu/` is likewise **"All checks passed!"**, which is clean
  for the wrong reason. F821 is *undefined* name, not *used-before-assign* in a branch.
* **`pylint` DOES catch it, rule `E0606` (`possibly-used-before-assignment`).** On the same
  synthetic file it reports case 1 exactly (`Possibly using variable 'x' before assignment`). It
  misses the empty-`for`-loop variant, which is the rarer shape; the branch shape is the one that
  shipped twice.
* **`vaayu/` is CLEAN under it today, with two FALSE POSITIVES to suppress when wiring it:**
  `pylint --disable=all --enable=E0601,E0606,E0602,E0611 vaayu/` → 9.99/10, only
  `vaayu_base.py:3878` (`get_module_from_name`) and `:3879` (`FP8Linear`). Both are guarded by
  `_fp8_available`, set in the same `try` that imports them (`vaayu_base.py:3865-3871`) — pylint
  cannot see the correlation. **Not bugs; do not "fix" them by hoisting the imports.**

**What is left of this row:** wire `pylint --disable=all --enable=E0601,E0606,E0602` (a dev
dependency, NOT a runtime one) as a check, with those two sites suppressed, and say where it runs.
It would have caught both original bugs. **It finds zero real defects today**, so this is
prevention, not a fix — price it accordingly against the other OPTIONAL rows. The gates-table
fallback ("run a real load after touching `__init__`") stays either way: E0606 would not have
caught the import-ORDER half on its own if the name had been module-level.

**WIRED AND CLOSED 2026-09-10.** It runs **inside `pytest tests/`**, as
`tests/test_no_unbound_locals.py` — a lint gate riding the suite, not a test of those two bugs.
That location was chosen because it is the ONLY harness this repo runs: there is no CI, no
`pyproject.toml` and no pre-commit hook, and the precedent for a meta-check as a test already
exists (`tests/test_docs_ports.py` greps markdown for stale ports). The test shells out to
`[sys.executable, "-m", "pylint", "--disable=all", "--enable=E0601,E0606,E0602", "--score=n",
"vaayu"]` from the repo root and asserts zero findings, printing any it gets verbatim plus the
instruction NOT to silence a false positive by hoisting the import.

* **`pylint>=3.0` is in `requirements.txt`** under a comment marking it dev-only and naming the
  test that enforces it; installed into `./venv` (4.0.8). Nothing in `vaayu/` imports it at runtime.
* **The two false positives are suppressed INLINE at the site**, `vaayu/vaayu_base.py:3876-3880`:
  a trailing `# pylint: disable=possibly-used-before-assignment` on each of the two lines and one
  comment above stating that `_fp8_available` is set in the same `try` that imports both names.
  Behaviour untouched. Suppressing inline (rather than allow-listing in the test) means a bare
  `pylint vaayu/` is clean too.
* **PROVEN TO FAIL — "a gate that cannot fail is not a gate" (this file's own rule).** A synthetic
  `if flag: strategy = "exl3"` / `return strategy` appended to `vaayu/profiler.py` — the exact
  branch shape that shipped twice — turned the test RED at
  `vaayu/profiler.py:45:11: E0606`, and the probe was reverted (`git status` clean on that file).
  The test is not merely green-by-construction.
* **Cost, and it is not free: the suite goes 124 s -> 164 s (+40 s, +32 %).** In-venv astroid
  infers the real `torch`/`transformers`; the identical command under `uvx` (no such packages)
  takes 3.3 s. **Findings are identical either way** — E0606 is intra-function control flow and
  needs no cross-module inference. `--jobs=4` recovers only 10 s for 4x the CPU, so the gate runs
  serial. `--init-hook` trimming `site-packages` was tried and does NOT work: it breaks pylint's
  own config initialisation (`AttributeError: 'Namespace' object has no attribute
  'overgeneral_exceptions'`).
* **It skips LOUDLY if pylint is absent**, with a reason naming the install command and saying the
  gate is not running. Skips stayed at **12** in the closing run, which is how it was verified to
  have actually executed rather than skipped.
* **§8 answer: NO server knob, and that is stated rather than assumed.** A dev-time static check
  reaches no user and nothing in `vaayu/server/` changes. Not plumbed, not reachable, by design.
* **The gates-table fallback stays**, exactly as this row said: E0606 would not have caught the
  import-ORDER half if the name had been module-level, so "run a real load after touching
  `__init__`" is still the rule.

---

# 6a-b · [ ] `head_bytes` is probably not a transient at all

**Cold-pickup brief:** [`docs/2026-09-10-row6a-b-head-bytes-BRIEF.md`](docs/2026-09-10-row6a-b-head-bytes-BRIEF.md) (gitignored; the numbers and traps below are the durable copy).

**Split out of row 6a (§0 P5) when that row closed 2026-09-09** — the root cause there was WHICH
CHUNK SETS THE PEAK, and it is fixed; this is the separate finding it turned up.
`docs/2026-09-09-P5-which-chunk-sets-the-peak-RESULTS.md` (gitignored, this disk only).

Found on the way, NOT fixed, and **do not "just remove it"**: MiniMax's whole measured transient at
S=513 is **0.3415 G against a `head_bytes` of 0.4614 G**, and the 70B's at S=512 is 0.7493 against
0.7883 — **a term cannot exceed the total it is part of**, so the pinned `lm_head` is largely
already inside `alloc_before`. Dropping it fits MiniMax (full-chunk residual **1.52–1.63 ×**
`512·vocab·4` against the **1.5 ×** a coexisting fp16+fp32 logits pair predicts, `run_lm_head` is
`layer(seq).float()`) and the 1B (**3.02 ×** = the 12 B/elem of a per-chunk loop where the previous
chunk's `output` is still bound and `torch.cat(batch, 0)` allocates a third copy). **But the 70B
has a 0.714 G floor at a 1-token trailing chunk that `head_bytes` = 0.788 G is currently covering
BY LUCK** — removing it under-predicts the 70B by ~0.7 GB, the dangerous direction. **And
Llama-3.2-1B cannot arbitrate: its `head_bytes` is EXACTLY 2 × its full-chunk fp32 logits, so
"head + 1 logits" and "3 logits" are the same number** — the 2026-08-15 round's 0.5 %-accurate
decomposition on that geometry proves nothing about the split. Needs the GPU round row 6a did
not: per-allocation instrumentation inside ONE chunk on the 70B, to identify that 0.714 G floor
(streamed layer weights in flight is the standing candidate, `_layer_slack` ≈ 0.49 GB/layer there).
**§8 caveat, in advance:** it would move the reserve only where the reserve is not floor-bound —
on flash it is (`max(predict(chosen), reserve_floor)`, floor 3.5 GiB), so price this as admission
headroom and correctness of the peak form, never as tok/s.

---

# 3 · [x] The cross-turn prefix cache only works for EXL3 models — CLOSED 2026-09-11: `auto` → `bf16` on every loader, prefix cache ON for non-EXL3

**Cold-pickup brief:** [`docs/2026-09-10-row3-fp16-prefix-cache-BRIEF.md`](docs/2026-09-10-row3-fp16-prefix-cache-BRIEF.md) (gitignored).

Carried out of the hygiene backlog (H7's third item) when the rest of that row closed
2026-08-15. Not hygiene — a real capability gap, and the only one left from that group.

`PrefixCacheStore` calls `to_cpu()` / `to_device()` / `resident_bytes()` on every cache it
stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
`transformers.cache_utils.DynamicCache` returned for `kv_cache_mode="fp16"` — what `"auto"`
resolves to for every **non-EXL3** model — has none of the three (verified on transformers 5.5.4:
that class is layer-object based, `layers[i].keys`/`.values`, no `key_cache` attribute at all).

`vaayu_base.py` probes the cache at load and disables the pool with a WARNING rather than failing
per-turn — the per-turn failure was the dangerous one, because `lookup()` POPS an entry before
`store()` puts one back, so each swallowed failure **permanently drained the pool**.

**To actually support fp16:** give `DynamicCache` the three methods via an adapter, or route fp16
through `FP8DynamicCache(quantize=False)`. **Not urgent** — every on-device target is EXL3, and
prefix reuse is this engine's biggest measured lever (136–377× TTFT), so the gap only bites a
model nobody currently runs here. (found 2026-08-09)

**RE-PRICED 2026-09-11 — PRIORITY 1. Cheaper than above, and testable today.**
- **Likely fix: resolve `auto` → `bf16` for non-EXL3 too** (`resolve_kv_cache_mode`, `vaayu_base.py:1231-1233`).
  `bf16` builds `FP8DynamicCache(quantize=False)`, which stores K/V in the model's SOURCE dtype (docstring
  `vaayu_base.py:1280`) — so it is the ring buffer, not a dtype change, and has all three methods the pool
  needs. The load-time warning already tells users `kv_cache_mode='bf16'` enables it. **Verify the source-dtype
  claim on an fp16 load before relying on it.**
- **"No fp16 model runs here" was wrong:** `Llama-3.2-1B-Instruct` is unquantized and on disk.
- **Costs to measure BEFORE deciding:** (1) the ring buffer over-allocates by geometric doubling and the VRAM
  planner charges it (`_kv_cache_is_bucketed`) — the 1B's derived `max_seq_len` (71981 sdpa) may drop; record
  before/after. (2) CPU-compute and non-CUDA paths have never run this cache class. (3) Plan E's "bf16 decodes
  slightly faster" was measured on EXL3 MoE only — re-check decode on the 1B.
- **Done condition:** `PREFIX_CKPT=<1B> pytest tests/test_batch_prefix_equivalence.py` — which today FAILS at its
  own premise assert on a non-EXL3 checkpoint — passes; a two-turn server run on the 1B shows `prefix_hit=true`
  on turn 2; `max_seq_len` and decode tok/s before/after recorded; EXL3 resolution unchanged (both MoE targets
  already resolve to `bf16`, so they must be byte-identical); suite green.
- **§8:** no new knob (`KV_CACHE_MODE` exists); plumbed (`app.py` passes it); reachable at N=1 through the
  server's defaults — this is a fix a user gets on the next start, for any non-EXL3 model.

**CLOSED 2026-09-11 — option (b) SHIPPED.** RESULTS: [`docs/2026-09-11-row3-auto-bf16-prefix-cache-RESULTS.md`](docs/2026-09-11-row3-auto-bf16-prefix-cache-RESULTS.md) (gitignored).
- **Change:** `resolve_kv_cache_mode` returns `"bf16"` for `auto` on every loader (was `"fp16"` off EXL3);
  `batch._resolve_kv_elem_bytes`'s duplicate copy aligned (2 B/elem either way). Explicit `"fp16"` is unchanged
  and still hits the load-time refusal + WARNING — the pool-drain guard stays reachable and tested. EXL3
  resolution unchanged (byte-identical by construction). Option (a), the `DynamicCache` adapter, was rejected:
  it depends on transformers' `layers[i]` internals (the brief's own stop rule).
- **Source-dtype claim VERIFIED:** `bf16` mode stores fp16/bf16/fp32 as given and round-trips `to_cpu`→`to_device`
  bit-exact (new unit test). On this box every CUDA load runs bf16 anyway (`vaayu_base.py:1858`).
- **1B, same box, same day** (library-default `max_pinned_layers=0`, compare arms only):
  flash `max_seq_len` **91106 → 65536 (−28.1 %)**, sdpa **70655 → 65953 (−6.7 %)**; decode on flash
  **6.064 → 6.091 tok/s (flat)**; turn-2 wall at 2819 tokens **1.23 s cold → 0.28 s warm (~4.4×)**. The sdpa
  decode gap (4.155 vs 6.204) is a pinning artifact, not the cache: fp16's larger context took the whole sdpa
  reserve → `pinning_budget=0` → `lm_head` streamed per token.
- **COST, stated:** the planner now charges the ring buffer's geometric doubling for non-EXL3 models, so derived
  context can drop by up to ~2× (it lands on a power of two). That is row **4**'s overshoot, now paid as context.
  Escape hatch: `KV_CACHE_MODE = "fp16"` (old context, no prefix cache).
- **Gates:** `PREFIX_CKPT=<1B>` prefix-equivalence **1 passed** (failed its premise before); `SERVER_CKPT=<1B>`
  server smoke **9 passed**, `prefix cache: 1hit/1miss saved=50/100 tok`. Suite **1972 → 1979**.
- **Anomaly investigated, NOT a bug:** warm-vs-cold greedy ids diverge at token 87 (sdpa) / 172 (flash) at an
  **exact bf16 top-2 tie (gap 0.0)**; cold is deterministic, ring-buffer cold == `DynamicCache` cold bitwise,
  warm is reproducible. The documented dense ~1-ulp path-dependence — the EXL3 70B already has it.
- **§8:** no new knob (`KV_CACHE_MODE` comment + `server/README.md` updated with the cost); plumbed (`app.py`);
  reachable at N=1 through the server's defaults — verified by the server gate.
- **Left open, reporting only:** on a pre-Ampere GPU / CPU, "bf16" mode stores fp16 / fp32 but the boot line and
  `env.json` say `bf16`; `_KV_ELEM_BYTES` under-charges fp32 2× in every mode (pre-existing).

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

# 5 · [ ] DEFERRED write-ups — TRIAGED 2026-09-06, two probes run, one item left

**Cold-pickup brief for the one open item:** [`docs/2026-09-10-row5-quality-certification-BRIEF.md`](docs/2026-09-10-row5-quality-certification-BRIEF.md) (gitignored).

Row 5 was never one row. Full evidence:
**[`docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md`](docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md)**
(**gitignored — this disk only**; the numbers below are the durable copy, per §7).

Method: one on-device capture per model at a realistic ~4.15k-token agentic context (re-sent
source + a codegen task), 64-token greedy decode, `MemoryMax=29G`. One run fed both probes —
real post-RoPE decode queries + stored-layout `K` for 7 quantile-sampled layers, and the ordered
decode routing trace. **Harness validated against Plan I's own published figure**: the reproduced
K-row proxy gives 0.0307 at S=694 against Plan I's 0.0372 for Qwen at 9 bits.

**Three of this row's four items closed 2026-09-06 and were pruned 2026-09-10:** Plan B Tasks 8–9
(Phase 2 live expert re-pin) REFUTED DIRECTLY, the Plan H fold-`Π` follow-up CLOSED with its target
gone, and the real-query quality gate RUN — **9 bits FAILS Qwen (0.0649 against the 0.05 bar) and 9
is the quantizer's ceiling**, with the docs corrected at all five sites. All three are in SETTLED;
the mixed-precision outlier split that came out of the third is SETTLED too. One item is left.

**[ ] STILL OPEN, deliberately unscheduled: does failing this bar degrade generated text?** The bar
is a pre-stated *proxy* for quality (score rel-L2 < 0.05 on the whole `[H,S]` matrix per layer);
Plan H §6 warns coherent output hides large perturbations (a
256-token coherence check passed even at the far worse 4-bit setting). Certifying that direction is
a separate experiment and **nothing here claims it**. Needs a real quality measure (perplexity on a
prefix-reused turn, or a task score), not another score-error number.

---
