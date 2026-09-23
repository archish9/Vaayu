# TODO — Vaayu pending work

**Everything with a `[ ]`, `[~]`, `[!]` or `[bug]` in this file is PENDING. Closed rows are not kept here.**

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

**Read §0 below first: it is the ORDER. Everything after it is the DETAIL.**

---

# ▶ START HERE — the whole open queue, re-prioritised 2026-09-11

## ▶▶ PICK THIS NEXT: **A5.1** — jump to "PICK THIS NEXT: **A5.1**" below for the order and the links ◀◀

**One line, so nobody reads fifty before finding it: A5 step 1 (the kernel gate) PASSED 6/6 on
2026-09-23; step 2 was decomposed the same day into A5.1…A5.7, each with its own document and
durable row. START AT A5.1.** Everything below is PENDING work only — the closed rows were pruned
2026-09-23 and live in `git log -p -- todo.md`.

---

**ROW 32 IS OUT OF THIS TABLE — `[!]` BLOCKED 2026-09-23, by the user, on cost.** It was order 1 that
morning; the session verified its route (a full HF sweep: the build on disk ships `2.0…8.0bpw` as
BRANCHES of the SAME repo, so route 1 is live and route 2 is dead) and then the user **declined the
download**. **Row 32's entire cost IS that download — there is nothing else in it to build — so it
cannot proceed and is not pickable.** Do not re-open it by re-deriving the projection; it unblocks
only on the user authorising an ~81.6 GiB pull. Its verified branch sizes and two handoff
corrections are banked in row 32 itself. **NO DOWNLOAD is a standing constraint until the user says
otherwise: every item below runs on checkpoints already on this disk.**

## ▶ PICK THIS NEXT: **A5.1** (row 28b). Then A5.2 → A5.3 → A5.4 → A5.5 → A5.6 → A5.7, one per session.

**A5 step 2 is DESIGNED as of 2026-09-23 — seven sub-plans, each with a self-contained document and
its own durable row under row 28b.** A fresh session needs nothing but the row it is picking.
**Work them IN ORDER: each module only has a golden reference once its dependencies are correct.**

| order | item | kind | what to open FIRST | cost |
|---|---|---|---|---|
| **1 (NEXT)** | **A5.1** config + `auto_map` | design | [`…A5.1-config-skeleton-design.md`](docs/superpowers/specs/2026-09-23-A5.1-config-skeleton-design.md) | ~½ d |
| 2 | **A5.2** parity harness | design | [`…A5.2-parity-harness-design.md`](docs/superpowers/specs/2026-09-23-A5.2-parity-harness-design.md) | ~½ d |
| 3 | **A5.3** dequant + `Linear` | design | [`…A5.3-dequant-linear-design.md`](docs/superpowers/specs/2026-09-23-A5.3-dequant-linear-design.md) | 1–2 d |
| 4 | **A5.4** MoE | design | [`…A5.4-moe-design.md`](docs/superpowers/specs/2026-09-23-A5.4-moe-design.md) | 2–3 d |
| 5 | **A5.5** attention | **BRIEF + gate** | [`…A5.5-attention-BRIEF.md`](docs/superpowers/specs/2026-09-23-A5.5-attention-BRIEF.md) | ~1 w |
| 6 | **A5.6** block + model | **BRIEF** | [`…A5.6-block-model-BRIEF.md`](docs/superpowers/specs/2026-09-23-A5.6-block-model-BRIEF.md) | 3–5 d |
| 7 | **A5.7** load + first token | **BRIEF** | [`…A5.7-load-first-token-BRIEF.md`](docs/superpowers/specs/2026-09-23-A5.7-load-first-token-BRIEF.md) | 2–3 d |

**A5.1–A5.4 are full designs; A5.5–A5.7 are briefs with start gates and stop rules**, promoted to
plans by the session that reaches them (their inputs need measuring first — a stale design is worse
than a brief, because the next session implements it instead of questioning it). **`docs/` is
gitignored: row 28b's A5.x bullets carry every number and stop rule and are the durable copy.**

**A5.5 is where A5 dies if it dies** — everything else has a golden reference or a Vaayu precedent;
the attention path has neither. Its step-0 gate exists to find that out in ~1 day, not ~3 weeks.

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
than Qwen3-Next. **No download was needed; the pip install was authorised by the user 2026-09-23.**

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
  repo-root on-device scripts too (fixed 2026-09-15 by `pytest.ini`'s `testpaths = tests`) — **unverified**, its command was not recorded.
  **Carried 2026-09-23 out of row 33 (closed, pruned).**
- **[ ] Row 44's F3 remnant — the pinned-H2D half only (~0.6 s of an 18 s 2k prefill). NOT
  recommended; recorded so nobody re-derives it.** After F6 the Qwen3-Next prefill is COMPUTE-bound
  (`disk_load` 0.17 s), so F3's read half is worth ~nothing. Full F3 evidence is in
  `git log -p -- todo.md` at the 2026-09-23 prune. **Carried out of row 44 (closed, pruned).**

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
  2.1–2.3. That lands near MiniMax, ~5× slower than Qwen3-Next.
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
   (~119 MiB for a layer's attention, 76.5 MiB for six experts). A5.2 builds that harness.
2. **Hash routing stashes `input_ids` on the model** — zero engine changes.
3. **MTP is OUT and excluded explicitly** (A5.7).

| # | sub-plan | kind | delivers | cost | doc |
|---|---|---|---|---|---|
| **A5.1** `[ ]` | config + `auto_map` | design | `AutoConfig` resolves `deepseek_v4` | ~½ d | [`…A5.1…`](docs/superpowers/specs/2026-09-23-A5.1-config-skeleton-design.md) |
| **A5.2** `[ ]` | **parity harness** | design | golden values from the reference's own modules | ~½ d | [`…A5.2…`](docs/superpowers/specs/2026-09-23-A5.2-parity-harness-design.md) |
| **A5.3** `[ ]` | dequant + `Linear` | design | both weight formats through the gated kernels | 1–2 d | [`…A5.3…`](docs/superpowers/specs/2026-09-23-A5.3-dequant-linear-design.md) |
| **A5.4** `[ ]` | MoE | design | 94.3 % of the bytes, on the module-list forward | 2–3 d | [`…A5.4…`](docs/superpowers/specs/2026-09-23-A5.4-moe-design.md) |
| **A5.5** `[ ]` | **attention** | **BRIEF + gate** | MLA + sparse index at `block=16` | ~1 w | [`…A5.5…`](docs/superpowers/specs/2026-09-23-A5.5-attention-BRIEF.md) |
| **A5.6** `[ ]` | block + model | **BRIEF** | hyper-connections, HF `Cache` migration | 3–5 d | [`…A5.6…`](docs/superpowers/specs/2026-09-23-A5.6-block-model-BRIEF.md) |
| **A5.7** `[ ]` | load + first token | **BRIEF** | remap, index, first generation, gates | 2–3 d | [`…A5.7…`](docs/superpowers/specs/2026-09-23-A5.7-load-first-token-BRIEF.md) |

**≈3 weeks. A5.5 is where this dies if it dies** — everything else has either a golden reference or
a Vaayu precedent; the attention path has neither.

### Durable summaries — each row survives its `docs/` file going missing (§7)

- [ ] **A5.1 — config + `auto_map`.** transformers 5.5.4 has `deepseek_v2/v3/vl`, **not**
      `deepseek_v4`, and the checkpoint ships no `auto_map` → Vaayu dies at `vaayu_base.py:2131`.
      Template is MiniMax (`configuration_*.py` + `modeling_*.py` + `auto_map`). **Do NOT port
      `ModelArgs`** — it is a second source of truth with its own defaults, and carrying both is how
      they drift. **Back up `config.json` before editing: it is the only copy, under a standing
      NO-DOWNLOAD constraint.** Decide explicitly whether to expose `quantization_config`
      (**default: hide it** — A5.3 does its own dequant; 47-m7 closed on exactly this class of
      surprise). Confirm A1's `expected_expert_count()` returns **256** via its third key
      `n_routed_experts` — A1 Task 4 exists because that lookup silently returned `None`.
      **Verify:** every config field round-trips **field by field, no spot check**.
- [ ] **A5.2 — the parity harness. BUILD IT BEFORE PORTING ANYTHING.** Record goldens from the
      reference's own modules, cache them as **committed `.npz` fixtures** so A5.3–A5.7 need
      **neither the 157 G checkpoint nor a GPU**. Goldens: `linear_fp4/fp8/bf16`, `expert`,
      `gate_hash`, `gate_score`, `moe`, `attn_prefill`, `attn_decode`, **`attn_chunked`**, `block`.
      **`attn_chunked` is the important one:** `model.py:537` does
      `kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`, assuming `seqlen == 1`, and Vaayu chunks
      prefill at 512 — **row-43 class, silent context corruption**. If the reference itself raises
      or corrupts there, **that IS the golden** and it is the most valuable thing A5.2 buys.
      **Traps:** `Attention.__init__` allocates `kv_cache` from `max_seq_len` (use a small one, or
      it sizes for 1,048,576); `sparse_attn` needs the **`block=16`** patch or every attention
      golden dies at launch; import `model.py` under an explicit name (it defines a module called
      `kernel`).
- [ ] **A5.3 — dequant + `Linear`.** Two formats: routed experts **`I8` (fp4 packed 2/byte) +
      `F8_E8M0` per-row/32**; shared+attention **`F8_E4M3` + `F8_E8M0` per-128×128-block**. The
      scale **shapes** differ, not just the dtypes. **THE MANDATORY DIVERGENCE: read `self.scale`,
      never `self.weight.scale`.** The reference aliases scale onto the Parameter
      (`model.py:145`); `_inject_expert_weights` (`moe_patch.py:1225`) rebinds each key with
      `setattr(submod, base, nn.Parameter(t))`, so `weight` becomes a **fresh object with no
      `.scale`** → `AttributeError` on every streamed expert, every token. **Both keys DO bind, so
      streaming a dual-tensor quantized expert needs ZERO engine changes** — the biggest de-risking
      fact for A5.4. `wo_a` is `F8_E4M3` on disk but the module declares **bf16** and uses the raw
      weight in an einsum → **dequantise once at load**. Kernels imported lazily (config must stay
      importable without tilelang). **Verify:** a test that rebinds `weight` to a fresh Parameter
      and asserts forward still works — without it the bug returns the first time someone
      "simplifies" back to the reference idiom.
- [ ] **A5.4 — MoE, 94.3 % of the bytes.** **Keep the reference's names** (`ffn`, `experts`,
      `gate`, `w1/w2/w3`): `detect_moe_blocks` is name-driven on exactly those, so it works with
      **zero engine changes** and A5.7's remap stays near-free. Two required changes:
      `Gate.forward` must return **raw scores** `[N, n_routed_experts]` (the reference returns
      `(weights, indices)`), and `MoE` must add **`route_tokens_to_experts`** — Vaayu's own
      documented contract (`moe_patch.py:155`), where all model-specific scoring lives:
      `sqrtsoftplus` → bias for **selection only** → topk → weights from the **UNBIASED** scores →
      normalise → ×1.5. **Scoring a MoE with the wrong function is what made MiniMax emit
      incoherent text** while loading and streaming perfectly. **Weights inside vs outside the
      expert are EQUIVALENT** (`w2` is bias-free linear, `w` a per-token scalar) — write that down,
      it looks like a bug. Hash layers (0–2) read the stashed `input_ids`. **Shared experts:
      24 MiB/layer → ~1.01 GiB over 43 layers**, A4 Check 1 discharged **by running, not reading**.
      **Verify:** `detect_moe_blocks` finds 43 `module_list` blocks on a **meta-device** tree; the
      MoE gates for Qwen3-Next and MiniMax stay green (prove the "zero engine changes" claim).
- [ ] **A5.5 — attention. BRIEF WITH A GATE; THIS IS WHERE A5 DIES IF IT DIES.** MLA (one 512-dim
      latent shared by 64 heads), grouped low-rank O, circular 128-window KV, `Compressor`, and on
      `ratio == 4` layers a learned `Indexer` over `index_topk 512`. `compress_ratios` over 43
      layers: **2 at 0, 21 at 4, 20 at 128**. **KV IS TINY — G4 is an API problem, not a VRAM one:
      50.99 MB/sequence at 8k = 0.0062 MB/token, 41× SMALLER than MiniMax's 0.254.** Do not spend a
      week on `vram_plan`. **Shared-memory ladder vs sm_120's 101,376 B** (`num_stages` has no
      effect): **block 64 → 141,312 B (1.394× over) · 32 → 104,448 B (over by 3,072 B, 3.0 %) ·
      16 → fits**. **STEP-0 GATE before any modeling code:** (1) correctness at **`topk=512`** and a
      real length against A5.2's golden — **fails ⇒ STOP, the port has no attention**; (2) the cost
      of the ¼ tile — **if attention alone dominates the 3.2124 GiB/token disk floor (~0.49 s),
      escalate before building**; (3) only if (2) hurts, try to recover 3,072 B for `block=32`.
      Open: ring-buffer KV vs HF `Cache` (and 42-B's snapshot); chunked prefill; **in-place rotary,
      applied again INVERTED to the attention output**; `act_quant(..., 64, inplace=True)` FP8 QAT
      sim (**group 64, not 128**, and not in `config.json`); `attn_sink`; two rope variants.
- [ ] **A5.6 — block + model. BRIEF.** **Hyper-connections are not in the A5 brief at all:** every
      block keeps **`hc_mult=4` copies of the residual** (`[b,s,4,d]`) and runs `hc_pre`/`hc_post`
      twice per block around a 20-iteration Sinkhorn. **Contained inside `Block` — the MoE still
      sees `[b,s,d]`, so A5.4's zero-engine-changes result holds. CONFIRM THAT; the whole
      decomposition rests on it, and if it is false, STOP and re-plan.** `hc_*` params are fp32 by
      construction → **~3.1 MB/layer, ~135 MB over 43 layers** of always-resident weight (feeds A4
      Check 2 with A5.4's ~1.01 GiB). The head uses `hc_head`, a **plain sigmoid with NO Sinkhorn**
      — same-looking code, different arithmetic. Also owns the **`start_pos` → HF `Cache`**
      migration (shape decided by A5.5), the `input_ids` stash lifetime, `tie_word_embeddings:
      false`, and `precompute_freqs_cis`'s `@lru_cache(2)` holding **exactly two** rope variants.
- [ ] **A5.7 — load, index, first token. BRIEF.** **ROW 28b's STEP-3 VERIFY IS WRONG AS WRITTEN:**
      it expects `ExpertOffsetIndex` to find `43 × 256 = 11,008`, but the regex
      `\.experts\.(\d+)\.` finds **11,776** — the extra **768 experts = 9.56 GiB** are
      `mtp.0/1/2.ffn.experts.*`. MTP is OUT, so **the index must skip `mtp.*` and find 11,008**.
      **It is a correctness hazard, not just waste:** `detect_moe_blocks` takes the layer index from
      the **first `.N.` in the path**, and `mtp.0.ffn` contains `.0.` → detected as **layer 0**,
      colliding with the real one. Same defect class as A1 Task 4 and 47-m5 — an index asking a
      question the load path answers differently, **third time this program has paid for it**.
      Also: key remap (should be ~a `model.` prefix — **verify, it is A5.4's claim**); splitter runs
      A1's guards **before** `DELETE_ORIGINAL` deletes the only copy — **dry-run it**; A4 Check 1
      (pinning half) and Check 2 (**~1.01 GiB shared + ~135 MB hc + ~5.33 GiB attention** vs the pin
      budget; Phase 2 **`break`s** on miss at `vaayu_base.py:3956` — count the layers); then row 28b
      step 5's measurement. **§8: answer all four.** **Expect ~1.75–2.06 tok/s** (3.2124 GiB/token
      ÷ the NVMe plateau) — **A5 buys REACH, not speed; ~5× slower than Qwen3-Next. Do not present
      it as a perf win.**

**Two checks CARRIED OUT OF A4 (row 47, closed 2026-09-23 with its premise refuted) — they belong to A5:**
- [ ] **check 1: confirm DeepSeek's shared experts actually land in the pinned non-expert set** once
      the model loads. A4's key-shape argument (`\.experts\.(\d+)\.` does not match
      `...shared_expert.*`, so shared experts are pinned, not streamed), executed on this checkpoint.
- [ ] **check 2: confirm the pinning budget covers DeepSeek's non-expert set.** 43 MoE layers, MLA
      attention, F8_E4M3 shared experts and attention — unmeasured. **Why it matters: the pinning is
      INCIDENTAL to the budget, not structural.** Phase 2 `break`s on budget miss
      (`vaayu_base.py:3956`); Qwen3-Next's ~3.55 GB fits inside 4.83 GB today. Un-pinned layers fall
      back to `LRUTensorCache(_ne_budget)` (`vaayu_base.py:2710`, byte-budgeted, derived) — RAM-cached
      re-reads, **not** per-token NVMe streaming — and the loop prints `budget exhausted at <layer>`,
      but that warning says nothing about shared experts specifically.

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
row either way. The same run also decomposes the token, which is what re-priced row **46** (closed 2026-09-23 — verdict in **SETTLED**).

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
      `_NVME_READ_CONCURRENCY` (`safetensor_model_persister.py:181`, **2 since row 49, 2026-09-21**) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
- [ ] **G6 — no CPU expert compute exists at all.** `compute_device` is forced to `"cuda"` for EXL3; the only CPU mode
      is running a whole non-EXL3 model with `device="cpu"`. A hybrid is new construction, not a flag. **Row 26/28
      already priced the CPU-AVX2 hybrid and RECOMMENDED IT CLOSED** (+6.8 % Qwen / +22.0 % MiniMax ceiling, and a
      non-EXL3 model makes it worse) — so treat G6 as REACH-only, never as a perf lever.

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
