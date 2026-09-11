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

**Nothing is in flight. No §0 P-row is open, so nothing here makes Vaayu faster on Qwen3-235B, MiniMax-M2.7
or the 70B today — every row is OPTIONAL by §0's test.** The order below was set by the user on 2026-09-11
after a triage of every open row: **three are worth working, in this order; the rest are not, for the
reason given.** Pick from the PRIORITY table first. The sections after this one are the detail.

## PRIORITY — worth working, in this order

| # | one line | why it is worth it | state | cost | brief (gitignored) |
|---|---|---|---|---|---|
| **3** | **Prefix cache is OFF for every non-EXL3 model.** `auto` resolves to `fp16` → bare `DynamicCache`, which lacks `to_cpu`/`to_device`/`resident_bytes`, so the pool disables itself at load. | **RE-PRICED 2026-09-11: likely a one-line fix, and testable TODAY.** `FP8DynamicCache(quantize=False)` (what `bf16` mode builds) stores K/V in the model's SOURCE dtype (`vaayu_base.py:1280`), so it is safe for fp16 models; resolving `auto` to it for non-EXL3 turns the prefix cache on. `Llama-3.2-1B-Instruct` is unquantized and on disk — the "no fp16 model runs here" line was wrong. Reachable through the server's defaults (`ENABLE_PREFIX_CACHE=True`, `KV_CACHE_MODE="auto"`). The model-agnostic mandate's biggest open gap. | `[ ]` **priority 1** | small + 1B runs | [`…-row3-fp16-prefix-cache-BRIEF.md`](docs/2026-09-10-row3-fp16-prefix-cache-BRIEF.md) · row **3** below |
| **25** | **APST: one MiniMax idle-sampler arm** to close the row. Qwen measured NULL 2026-09-11; MiniMax (module-list forward, no stager — a different read pattern) is unmeasured. | Cheap (~7 min GPU, harness exists), closes a row, and makes the "APST costs nothing" wording in the boot line and docs true for both MoE targets rather than one. Full A/B only if the arm shows material ≥ 100 ms idle. | `[~]` **priority 2** | 1 arm | [`…-row25-apst-ab-RESULTS.md`](docs/2026-09-11-row25-apst-ab-RESULTS.md) · row **25** below |
| **18** | `[bug]` **`output_hidden_states=True` crashes the forward.** Diagnosis re-verified 2026-09-11 against HEAD: `vaayu_base.py:4397` builds one sublist per layer, `:4667`/`:4680` fill them, `:4766` does a flat `+=` of a bare Tensor, `:4836` cats an empty sublist; `:4834`'s `[0:-2]` is the hand-tuned compensation. | Small, testable on the 1B in seconds, and it is a real crash on a public forward argument. Low severity (nothing in the engine, server or gates asks for hidden states) — which is why it is third. | `[bug]` **priority 3** | small | [`…-row18-output-hidden-states-BRIEF.md`](docs/2026-09-10-row18-output-hidden-states-BRIEF.md) · row **18** below |

## NOT worth working now — and why (re-open only on the stated condition)

| # | one line | why not | re-open if | state | brief (gitignored) |
|---|---|---|---|---|---|
| **6a-b** | `head_bytes` is probably not a transient; the 70B's 0.714 G floor is covered BY LUCK. | A GPU round that buys admission headroom, never tok/s — the reserve is floor-bound on flash. The current form errs SAFE. | someone edits `predict_peak_vram` (then this is a trap to read first) | `[ ]` | [`…-row6a-b-head-bytes-BRIEF.md`](docs/2026-09-10-row6a-b-head-bytes-BRIEF.md) |
| **5** | Does failing the 0.05 rel-L2 compression bar degrade TEXT? | `PREFIX_COMPRESS = None` is the server default, so it only affects users who opt in. | compression becomes a default, or a user reports degraded prefix-reused turns | `[ ]` | [`…-row5-quality-certification-BRIEF.md`](docs/2026-09-10-row5-quality-certification-BRIEF.md) |
| **23** | Prefill stager re-runs once per batched chunk (fused-EXL3). | `MAX_BATCH_SIZE = 1` → 0 % to users (§8). P10 banked batching as not recommended. | batching is enabled in the server | `[ ]` | [`…-row23-batched-chunk-stager-BRIEF.md`](docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md) |
| **4** | 2× KV overshoot, ~0.83 GB at Qwen's 8192. | PARKED — a larger 1.22 GB giveback measured decode +0.4 %. | the condition in its brief (per-cell VRAM drops enough for N=2) | `[ ]` parked | [`…-row4-kv-overshoot-PARKED-BRIEF.md`](docs/2026-09-10-row4-kv-overshoot-PARKED-BRIEF.md) |
| carried | P10 batching minors (STILL OPEN). **One is NOT batching-gated:** the sentinel-only cancel path is reachable at N=1 — a job QUEUED behind the running one, cancelled by `cancel_all()` at SIGTERM. | Coverage only; no defect known on any of them. | a shutdown/cancel bug is reported | `[ ]` / `[bug]` | [`…-carried-p10-batching-minors-BRIEF.md`](docs/2026-09-10-carried-p10-batching-minors-BRIEF.md) |
| carried | `_finish` stop-string identity · `expert_sim.py` anchor gap · row **8** `tensor_align` | Reporting / a diagnostic tool / dead code — none changes the engine. | a client needs the stop string / someone quotes `expert_sim` small-budget absolutes | `[ ]` | see STILL OPEN |
| carried | No resident-size checkpoint on disk. | A download, not code — but still the single largest usability change for the agent (streamed MoE runs ~0.56–0.86 tok/s). | — (do it whenever) | `[ ]` | [`…-carried-resident-checkpoint-BRIEF.md`](docs/2026-09-10-carried-resident-checkpoint-BRIEF.md) |

**Closed 2026-09-11 (kept here until the next prune):** row **13** (`even-4` built, done condition met,
0 % on the default config — [`…-row13-even4-RESULTS.md`](docs/2026-09-11-row13-even4-RESULTS.md)) and
row **25**'s Qwen A/B + remedy fix (`c256e18`).

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
of GPU. ~~NVMe APST~~ — **measured 2026-09-11 (row 25): NULL on Qwen at N=1**, not a tax on the dominant cost
(the drive's idle gaps are < 100 ms, APST's timeout). The boot still warns about it; see row 25.

**Before proposing ANY perf work, read SETTLED.** It is the list of things already built and
refuted, with the number that killed each one.

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

**Suite baseline: 1972 passed, 12 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
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
is no longer on this list — row 25 measured it NULL on Qwen at N=1 (2026-09-11)**, and found the
script that was meant to disable it could not do so without a reboot (fixed).

---

# ▶ STILL OPEN — carried forward from rows that closed

Live items left behind by rows that closed and were pruned. **All of them are OPTIONAL by §0's
test — none can make Vaayu faster.** No §0 P-row is open.

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
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 8192 (sdpa) |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 71981 (sdpa) |

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
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. Qwen and MiniMax both pass |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity. MoE: first token + warm reproducibility. **Needs an EXL3 checkpoint** — on a non-EXL3 one it fails at its own premise assert (pool disabled at load, OPTIONAL row 3), with or without any change |
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (**`-s` is enforced** — without it the module fails every test with a message naming the flag, before loading anything; G1) | 1–8 min | server broken end-to-end: load, non-stream, SSE stream incl. finish_reason and completion_tokens, oversized-prompt refusal, prefix reuse firing, env.json clean, requests.jsonl complete, gb_tok consistent with the run's own pinning (nonzero if it streams, zero if every layer is VRAM-pinned) |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`** — except
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

**Build ORDER, set by the user 2026-09-11 — supersedes the 2026-09-10 order (14 → 13 → 24, then 6a-b / 3 / 5; rows 14, 13 and 24 are all closed): row 3 → row 25's MiniMax arm → row 18.** Everything else is not worth working now; the reasons and re-open conditions are in START HERE. The 2026-09-10 paragraph is in `git log -p -- todo.md`.

| # | Item | §0 | Status | Cost | Why it sits here |
|---|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | OPTIONAL | `[x]` **CLOSED 2026-09-10 — wired, and PROVEN to fail on the real shape** | ~30 min, CPU only | Two such bugs shipped GREEN and broke both MoE loads. **`ruff` does NOT catch the class — `pylint E0606` does.** `vaayu/` is clean today under it (2 false positives). The row narrows to wiring pylint + suppressing those two. See below. |
| **13** | `[bug]` The spec gate's bar is measured in an UNREPRESENTATIVE window and evaluated once (was: it is in expert bytes) | OPTIONAL | `[x]` **CLOSED 2026-09-11 — `even-4` built, done condition met on device (see row 13).** Was: `[~]` **WINDOW WIDTH PRICED 2026-09-10 — the remainder is now a DECIDED, COSTED build (`even-4`), awaiting a go/no-go.** | window fix: ~half a day + 1 on-device A/B | **Build order: row 14 FIRST, then this, then row 24.** Row 14 is CPU-only and fixes a feature that IS exposed today (`ENABLE_KV_PERSIST` can be turned on but not aimed); this row is worth **0 %** unless a user explicitly opts a MoE checkpoint into prompt lookup. **Cold-executable build brief for the remainder: [`docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md`](docs/2026-09-10-row13-even4-window-BUILD-BRIEF.md)** — read it before touching code; it carries the trap that makes the obvious implementation WORSE than today. Evidence: [`…-window-width-price-RESULTS.md`](docs/2026-09-10-row13-window-width-price-RESULTS.md), [`…-timed-trial-SPIKE.md`](docs/2026-09-10-row13-timed-trial-SPIKE.md), [`…-timed-bar-DESIGN.md`](docs/2026-09-10-row13-timed-bar-DESIGN.md) (the SHIPPED half). | Opened 2026-09-07 by the P9 end-to-end arm. MiniMax-M2.7, thinking ON: gate measured acceptance 0.6875 against a 0.6259 break-even and decided `run` — and speculation still cost **−12.9 % total / −16.0 % decode**. **Self-contained brief: [`docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md`](docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md).** See below. |
| **24** | `[x]` **CLOSED 2026-09-10 — REFUTED ON PRICE.** Re-evaluation is worth ≤ **+6.28 %** with perfect foresight at ZERO cost, and breaks even at **0.5–5.9** token-equivalents against a real re-entry price of **~29.3** | OPTIONAL | `[x]` **closed by an offline replay spike — CPU only, no GPU round, nothing shipped** | none spent | Opened 2026-09-10 by row 13's spike. MiniMax-M2.7 leaving its reasoning preamble for code reproduction: blocks 0-9 lost, blocks 10-14 ran **1.21-1.28x**. A TRIAL sampled in the preamble latches `off` and forfeits the win; one sampled in the code stretch latches `run` and keeps it through the next preamble. **NOT the SETTLED regime-latch item** (that was refuted on Qwen, which never emitted code at all). See below. |
| **6a-b** | `head_bytes` is probably not a transient at all | OPTIONAL | `[ ]` | GPU round | Split out of row 6a on its own evidence when that row closed 2026-09-09. §8-bounded the same way P5 was: the reserve is floor-bound on flash, so tightening it buys admission headroom, not tok/s. See below. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | OPTIONAL | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. Now has a real consumer — the server's largest lever is off for every fp16 model. |
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

# 25 · [~] NVMe APST A/B — MEASURED NULL on Qwen; the documented fix was inert (FIXED) — MiniMax arm is PRIORITY 2

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
- [ ] MiniMax-M2.7 (module-list forward, no stager — a different read pattern) and the 70B dense are
  UNMEASURED. Cheapest check: ONE arm with the idle sampler (`plans/probes/apst_idle_sampler.py` +
  `apst_idle_report.py`); a full A/B only if it shows material ≥ 100 ms idle.
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
- §8: no server knob (OS setting). The server's boot warning is the user-facing path, and it currently
  points at a fix that does not take effect.

---

# 18 · [bug] `output_hidden_states=True` crashes the forward — PRIORITY 3 (2026-09-11)

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

# 2 · [ ] Static check for unbound locals / import order

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

# 3 · [ ] The cross-turn prefix cache only works for EXL3 models — PRIORITY 1 (2026-09-11)

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
