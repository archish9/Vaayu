# TODO — Vaayu pending work

**Everything with a `[ ]`, `[~]` or `[!]` in this file is PENDING. Closed rows are not kept here.**

**Pruned 2026-09-03:** every `[x]` row was removed — all 16 B-rows (the 2026-08-23 server-review
bugs), the whole FIX SEQUENCE table, and every closed R/M/C/P perf row. Two things survive that
prune and you should not go looking for the originals: the **SETTLED** index below, which is the
anti-regression record of what was REFUTED and must not be rebuilt, and the reference sections
(Environment, Checkpoints, On-device gates), which were never tasks.

**Where the removed text went.** Full pre-prune text: `docs/2026-09-03-todo-archive-preprune.md`
(**gitignored — this disk only**). The durable copy is `git log -p -- todo.md`; the commit before
the prune holds it in full. Older closed detail was moved out on 2026-08-23 (commit `f3869b3`).

Protocol: `CLAUDE.md` §6 (ledger) and §7 (all planning/investigation/spec work lives in `docs/`).
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Reading this cold, in a fresh session, is the intended way to use it.** Read in this order —
**§0 first: it is the build ORDER and the perf / not-perf split; the sections after it hold the
detail each entry points to.**

1. **OPEN** — the two live rows, each linking a self-contained doc in `docs/`.
2. **STILL OPEN, carried forward** — live items left behind by rows that closed.
3. **SETTLED — DO NOT RE-OPEN** — read this BEFORE proposing any perf work. It is the list of
   things already built and refuted, with the number that killed each one.
4. **Environment** — how to run anything here. `python` is not on PATH; every on-device arm needs
   its own `systemd-run` scope. Skipping this section is how arms get thrown away.
5. **On-device gates** — what each gate can actually catch, and which ones a change obliges you to run.
6. **OPTIONAL** — real work, no slot. Never ahead of the rows above.

**Read §0 below first: it is the ORDER. Everything after it is the DETAIL.**

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

**TWO rows are open — P10 and P5**, each with a SELF-CONTAINED brief in `docs/` linked in
the table. **Order: P10 first, and it is now MEASURED rather than argued.** Its refuting
measurement ran 2026-09-07 and the row PASSED at **1.48-1.78x end-to-end** at realistic agentic
context — the largest effect on this board by an order of magnitude, and mostly a PREFILL win, not
the decode win P6/P8 banked. It also found its own prerequisite: `_assert_context_fits` refuses
every N>1 at that context, so the server wiring must be preceded by fixing gate 2 or it ships
inert. P5 is a sweep that may end in "not derivable — close the row". **P8 SHIPPED 2026-09-07 at +3.21 % on
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

| # | item | why it is perf | dependency | where the detail lives |
|---|---|---|---|---|
| **P10** `[!]` | Wire batched decode into the SERVER — **BLOCKED on row 17** | **§4's refuting measurement RAN 2026-09-07 and the row PASSED, far bigger than the brief claimed.** At a realistic ~4.15k-token agentic context, batching is worth **1.48-1.78x END-TO-END** (Qwen N=2 1.98x at 16 tokens, 1.78x projected at 512; MiniMax N=2 1.77x / 1.48x) — not the +3-12 % P6/P8 measured at 690-token prompts, because a long prompt makes PREFILL dominant and prefill is what batching amortizes: **Qwen prefilled TWO 4.18k streams in 214.22 s against 214.25 s for ONE.** **But it also found a prerequisite the brief did not name: `_assert_context_fits` refused EVERY N>1 at that context, so wiring the server alone would have shipped the feature INERT. That prerequisite is DONE (P10a, 2026-09-07): the expert-cache giveback, +18.0 % at MiniMax N=2 through the real gate, banked.** What remains is the server wiring itself — **BLOCKED 2026-09-07 on row 17: `sdpa` + chunked prefill is garbage on MoE, and `batched_generate` hard-asserts sdpa — switching the server's kernel today would corrupt every single-stream request with a prompt over `PREFILL_CHUNK_SIZE`. Row 17 also prices the switch: flash → sdpa costs the expert VRAM cache −15.5 %, so this row's real comparison is flash@N=1 vs sdpa@N=2, not P10a's +18.0 %, and it can still be refuted.** The greedy-ids `[bug]` is SETTLED (benign: one word, MoE routing discontinuity — row 15's identity criterion is unachievable on MoE and is replaced there). A DIFFERENT `[bug]` was found on the way and FIXED: the tied lm_head was never materialized on the prefetch path, so any model whose `pinning_budget` hit 0 emitted noise — row **16**, now `[x]`, and dense batched decode is token-identical N=1 vs N=2 after it. | none | OPTIONAL row **15** · [`docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md`](docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md) · brief [`docs/2026-09-07-B-server-batched-decode-BRIEF.md`](docs/2026-09-07-B-server-batched-decode-BRIEF.md) |
| **P5** | The intercept residual the VRAM calibration cannot explain | One intercept rule over-charges 2 of 3 geometries (70B charged 1.051 against a measured 0.928; MiniMax 0.871 against 0.62–0.67). Over-charging the transient reserve shrinks the expert cache, and cache size sets throughput. | none | OPTIONAL row **6a** · [`docs/2026-09-06-P5-intercept-residual-BRIEF.md`](docs/2026-09-06-P5-intercept-residual-BRIEF.md) |

**Warnings that belong to this order, not to a later surprise:**
- **The P1, P2 and P3 slots are VACANT.** P1 was measured 2026-09-06 and **REFUTED** (zero disk reads
  for GPU-resident weights, on 5922 decode visits in each of two arms — see SETTLED). **P2 was
  BUILT and shipped 2026-09-06** — the mixed-precision KV outlier split, Qwen 0.0683 FAIL → 0.0281
  PASS at 1.211 B/elem, closed under OPTIONAL row 5 with
  `docs/2026-09-06-P2-outlier-split-RESULTS.md`.
  **P3 was ALREADY BUILT when this section first listed it — closed 2026-09-06 with no new code.**
  This section was written 2026-09-06 quoting the S1f row's *"Not built, not designed"*, which S1f
  wrote on 2026-09-05 **before** S1d landed later the same day; the `[x] S1d` row below is the
  authority and it says BUILT, WIRED and MEASURED. Verified in code, not taken on the row's word:
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
  prefix pays is unmeasured (see OPTIONAL row **10**).
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
  perfectly-measured union is the wrong quantity.** OPTIONAL row **11** ·
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
  byte-to-time conversion.** OPTIONAL row **10** ·
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

## OPTIONAL — cannot make Vaayu faster on a target run here. Never ahead of P5/P8.

Correctness, capability, tooling and hardware. Real work, and some of it matters — none of it moves
tok/s or TTFT on Qwen3-235B, MiniMax-M2.7 or the 70B dense today.

| item | what it is | why it is not perf | where |
|---|---|---|---|
| **A** | The server reaches 14 of the engine's 28 load parameters | Capability and correctness, not speed. `ENABLE_KV_PERSIST` is exposed while `kv_persist_dir`/`kv_persist_sync` are not, so the feature can be turned on but not aimed. | OPTIONAL row **14** · [`docs/2026-09-07-A-server-config-reachability-BRIEF.md`](docs/2026-09-07-A-server-config-reachability-BRIEF.md) |
| **C1** `[bug]` | A MoE model on `device="cpu"` builds a speculator with **no gate at all** | Real correctness hole. CUDA is what runs here, so it costs nothing today. | OPEN section |
| `_finish` stop-string identity | Cannot report WHICH stop string fired | Reporting only, and no client reads it. Needs a new premise (matching stop strings ourselves). | STILL OPEN |
| `expert_sim.py` anchor gap | 31.9 % simulated vs 43.9 % live | A diagnostic tool's accuracy, not the engine. The obvious hypothesis was built and REJECTED. | STILL OPEN |
| No resident-size checkpoint | Needs a ~14B coder at 4–4.5 bpw EXL3 | A download, not code. The row says it is **not a blocker**. | STILL OPEN |
| Static unbound-local check | `ruff`/`pyflakes` for used-before-assign | Reliability: two such bugs shipped GREEN and broke both MoE loads. Prevents crashes, not slowness. | OPTIONAL row **2** |
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
(`7d806e7`), not restored, see the bug row below. `plans/` and `docs/` are GITIGNORED, so neither
survives a fresh clone; `plans/settled_and_method_notes.md` is the older archive, on this disk only.

**Suite baseline: 1715 passed, 12 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
2026-09-07 after P9 shipped (the 1700 measured 2026-09-06 + 15 spec-prefix tests; that 1700 was
1685 + 6 batch-union sizing + 9 batch-prefix wiring tests; the 1685
was measured the same day after P4's closure: 1684 + 1 suppressed-round-width test; the 1684 was measured the
same day after the P2 outlier split: 1678 + 6 split tests; the 1678 was measured the same day
after the dense prompt-lookup default: 1647 + 7 bypass + 11 non-streaming + 4 trim + 9 auto; the 1647 was the two dense-telemetry cache-hit fixes' 1639 + 4 + 4; the 1639 was the stray-`logs/run-*` fix's 1638 + 1, the 1638 itself measured against a stashed
HEAD the same day — the 1570 written here before that was stale, as this note warns). **This figure
has gone stale seven times now.** Re-measure against a stashed HEAD before quoting it; do not trust the
number written here.

---

# ▶▶ OPEN — server logging & drafter, opened 2026-09-03

**These are the live rows. Everything below them is either carried-forward work, the
SETTLED anti-regression index, reference material, or the unslotted OPTIONAL list.**
Live as of 2026-09-06: **no task rows in THIS section** (S1, G1 and V1 closed 2026-09-05;
`env.json`'s `kv_cache_mode`, the suite's stray `logs/run-*` dirs, dense `avg_nvme_throughput`
and its stale-disk-timings follow-up closed 2026-09-06).
**ONE `[bug]` row is open, carried under a closed parent in this section — do not read
the line above as "no open bugs", which is what it used to say:**
**C1** (a MoE model loaded with `device="cpu"` runs a speculator with NO gate at all;
pre-existing, deferred). **S1e-bis and S1f both CLOSED 2026-09-06** — each was open only on
"S1d must land first", and S1d had already landed on 2026-09-05, later the same day than the
paragraphs that said otherwise. The next work comes from **STILL OPEN**, or
from **OPTIONAL** below. Each of
**S1**, **G1** and **V1** carries a self-contained document in **`docs/`**, meant to be picked up
cold in a fresh session with no other context. **`docs/` is GITIGNORED** (`.gitignore:28`), so those files are
on this disk only and do not survive a fresh clone; this file is the tracked one, and the durable
copy of anything moved out of it is `git log -p -- todo.md`. That is the standing convention: all
planning, investigation and spec work goes in `docs/` — see `CLAUDE.md` §7.

Opened by a `/systematic-debugging` pass on a "the server is too slow" report. **The report was
NOT a bug** — both models ran inside their own recorded benchmark bands (MiniMax 1.794 tok/s
against a 1.798–1.811 band; Qwen 0.937 tok/s against 1.083–1.122, the gap explained by an 809-token
run vs the bench's 64). Verdict, evidence and the two reading traps:
**[`docs/2026-09-03-slow-generation-verdict.md`](docs/2026-09-03-slow-generation-verdict.md)**.
Read it before either row — it is what both rows cite.

What the pass actually found:

- **[x] L1 T1–T4 · `logs/` is not a self-sufficient triage bundle** — 12-item audit, target layout,
  and a 9-task plan in
  **[`docs/issue-L1-triage-logging.md`](docs/issue-L1-triage-logging.md)**.
  Vaayu is going open-source and the support loop is "user zips `logs/`, maintainer debugs here".
  That loop does not close today: `server.log` has **no timestamps and no run boundaries**, and
  every load-bearing fact in the 2026-09-03 diagnosis came from OUTSIDE `logs/` — above all the
  gitignored `config_local.py`, which is where the user's real problem turned out to live.
  Includes the user's explicit ask, **per-session logs**, which WORKS: `ServerEngine.slot()` runs
  one generation at a time under `self._lock` (`vaayu/server/engine.py:180`), so a current-session
  marker attributes the whole process stream unambiguously. **That assumption dies the day batched
  decode (Feature B, N>1) lands** — the doc says what to do then.
  **Stop rule:** T1–T4 close the loop; T5–T9 are additive. Do not change the token-summary bucket
  semantics while doing it — archived measurements are stated in those terms.
  **Design approved 2026-09-03, T1–T4 only** — decisions recorded as §9 of that doc, and they WIN
  where they differ from §3–§6. The two that are not obvious: the forward name is **tagged at the
  dispatch** (`_make_sparse_moe_forward` sets `fn.__vaayu_forward__`) rather than re-derived in
  `envinfo.py`, because a second copy of the dispatch table drifts and then misreports which of the
  four forwards ran; and `warnings` is **harvested by the `capture_stdio` pump** (bounded ring
  buffer of `WARNING` lines, drained after load) rather than plumbed from a dozen print sites — the
  pump is fd-level, so it also catches the `vaayu_rust` warnings no Python hook can see. Console tee
  stays RAW; `env.json` is fail-soft (`collect_errors`, never blocks a boot); no auto-prune of run
  dirs. Verified by unit tests over all FOUR forwards plus real loads of all THREE checkpoints.
  **Plan: [`docs/superpowers/plans/2026-09-03-L1-triage-logging-T1-T4.md`](docs/superpowers/plans/2026-09-03-L1-triage-logging-T1-T4.md)**
  — 8 tasks, 58 TDD steps, every test and every implementation written out. Task order: run dir →
  pump line-treatment → session marker → WARNING buffer → forward tag in `moe_patch.py` → `envinfo.py`
  → lifespan wiring → docs + the three on-device gates. **Task 8 is the gate; everything before it is
  unproven on real weights.** One recorded deviation from the issue doc: §5 asks for an assertion on
  marker overwrite, §8 forbids logging from raising on the generation path — the plan warns once and
  tags `session=*` instead, which is what §5 itself prescribes for unattributable lines.
  **Task 8 (the gate) run 2026-09-03 — real loads of all three checkpoints**, driven by a script in
  the scratchpad (not committed — the repo's own gate scripts for this were gone, see the bug row
  below), `delete_original=False` on every run. Full detail + side-by-side numbers:
  `.superpowers/sdd/2026-09-03-L1-triage-logging-T1-T4/task-8-report.md`
  (gitignored — this disk only). Verified, per run: exactly one new `logs/run-*` dir with `latest`
  pointing at it; `env.json`'s `forward`/`expert_style` matched the dispatch table row
  (Qwen3-235B `fused`/`_make_fused_sparse_forward_exl3`, MiniMax-M2.7 `module_list`/
  `_make_modulelist_sparse_forward`, Llama-3.3-70B `dense`/`is_moe: false`); Qwen (94 layers/128
  experts/11830272 expert_bytes) and MiniMax (62/256/8875020) reported **different** geometry, so
  no literal crept in (CLAUDE.md §5); every geometry/budget field matched that run's own startup
  print, field by field; `grep -c '/home/a' env.json` was 0 on all three; every `server.log` line
  was timestamped; the single real prefill collapsed to one (dense: two, from an interleaved
  `vaayu_rust` line — still not "hundreds") `running layers` progress line, not tqdm spam;
  `sessions/l1-verify.log` held exactly that session's lines on every run; dense's three MoE budget
  fields were null with a named `collect_errors` entry each (`AttributeError:
  'NoneType' object has no attribute ...`), the only `collect_errors` on any run.
  **The dense teacher-forced-perplexity check (8.78) was NOT re-run — its gate script is deleted
  (see bug row below).** What Task 8 verified on the dense checkpoint instead was a real load
  through the server (health, one chat completion, 200 OK, correct derived geometry) — a weaker
  claim than the ppl gate and should not be read as re-confirming 8.78.
- **[x] L1-B1a · race half FIXED 2026-09-04 — `collected_warnings()` now drains the pump first.**
  `env.json`'s `warnings` array was empty on all three Task 8 on-device runs even though
  `server.log` correctly contained the box's `[Vaayu Warning]  APST (power management) is active...`
  line and, on both MoE loads, a correctly-cased `[Vaayu-MoE] WARNING: VRAM cache budget ...` line.
  Root cause: `env.json`'s lifespan called `collected_warnings()` (`vaayu/server/app.py:326`)
  immediately after the load returned, with no `logs.drain()` call first — unlike
  `set_current_session`/`clear_current_session`, which both drain before touching the marker. A
  warning still in flight through the pipe at that instant was silently absent even though it landed
  correctly in `server.log` moments later. Fix: `drain()` is now called **inside**
  `collected_warnings()` itself (`vaayu/server/logs.py`), not at the `app.py` call site, so a future
  caller cannot forget it and library/test callers are unaffected (`drain()` is a no-op with no pump
  active, and takes no lock `collected_warnings()` holds — `note_if_warning`/`collected_warnings`
  touch no lock at all, so there is no deadlock with `_marker_lock`). New regression test
  `tests/test_logs.py::test_a_just_printed_warning_is_visible_with_no_sleep` reproduces the race
  deterministically (subprocess, live `capture_stdio` pump, prints a WARNING, reads
  `collected_warnings()` immediately with **no sleep**) — confirmed failing before the fix (3/3
  runs), passing after. Full suite 1385 passed, 3 skipped (was 1384 + this one new test).
- **[x] L1-B1b · case mismatch — FIXED 2026-09-04 (I7, final fix wave).** `note_if_warning()`
  (`vaayu/server/logs.py`) filtered on the literal substring `"WARNING"` (all caps), so the six
  title-case `"[Vaayu Warning]"`/`"[Vaayu-MoE] WARNING"`-adjacent sites (`auto_model.py:24,39,59`,
  `moe_key_remap.py:98,110`, `vaayu_base.py:1310,3369`) — including the box's own APST warning —
  were never harvested. Fixed by making the match case-insensitive (`"warning" in formatted.lower()`)
  rather than rewording the print sites, so future sites keep working without plumbing
  (`tests/test_logs.py::test_title_case_warning_is_collected_too`). **Folded in at the same time:**
  the `_WARNING_LIMIT` FIFO used to drop the OLDEST warning past the cap, which is backwards — startup
  warnings (APST, foreign-GPU) are the oldest lines and the ones that matter most; it now drops the
  newest instead (`test_the_warning_buffer_is_bounded` updated to assert that).
- **[x] L1 T5–T9 (+T10) · additive half of the triage-logging work. DONE 2026-09-04,
  branch `l1-t5-t10-triage-additive`.** Spec:
  [`docs/superpowers/specs/2026-09-04-L1-T5-T9-design.md`](docs/superpowers/specs/2026-09-04-L1-T5-T9-design.md)
  (**gitignored — this disk only**; the decisions below are the durable copy).
  **Plan: [`docs/superpowers/plans/2026-09-04-L1-T5-T10-triage-logging-additive.md`](docs/superpowers/plans/2026-09-04-L1-T5-T10-triage-logging-additive.md)**
  — 10 tasks, every test and every implementation written out, a suite-count ladder per task
  (baseline re-measured 2026-09-04: **1397 passed, 3 skipped**, confirmed against HEAD, not quoted).
  Task order: persister byte counter → `bytes=`/`gb_tok=` → `_lt_cache_hit` fix → `reqlog.py` →
  engine rows → route refusal rows → `session_event` warning → `diagnose` → docs → the committed
  gate. **Three deviations from the spec, all recorded in the plan's header:** (a) the spec's
  telemetry-summation for layer bytes is dropped — every layer read funnels through
  `SafetensorModelPersister.load_model`, so ONE more increment on the same counter covers dense,
  and a cache hit then contributes zero by construction rather than by a flag; (b) the
  `_lt_cache_hit` fix is kept but demoted to its own task, since `gb_tok` no longer depends on it
  (what still does is `avg_nvme_throughput`, wrong on both MoE models from token 2); (c) **T6 needs
  a SECOND write site the spec did not name** — a request refused before generation
  (`REQUIRE_SESSION_ID`, `ContextTooLong` on the streaming path, `QueueFull`) never reaches the
  engine, so the engine's `finally` alone does not satisfy the spec's own "a request must never end
  without a row".
  **Six decisions that are not obvious and cost exploration to reach:**
  (1) **`bytes=`/`gb_tok=` mean NVMe bytes, not materialised bytes** — which obliges a fix to
  `_load_non_expert_strategy` (`vaayu_base.py:3234`): it serves from `_non_expert_tensor_cache`
  with no disk read, yet callers still sum the state dict into `_lt_file_bytes` and leave
  `_lt_cache_hit` False, so today's `file_bytes` overcounts on both MoE models. Same line fixes
  `TokenTiming.avg_nvme_throughput`, wrong for the same reason.
  (2) **T5 needs TWO sources.** Expert bytes come from a counter inside the persister's
  `load_expert_sparse`/`load_expert_into_slab`, NOT from `_dispatch_expert_misses` — there are
  eight call sites (`moe_patch.py:336,352,357,1357,1505,1510,1549,1559`, `vaayu_base.py:1267,1274`)
  and dispatch covers four, missing both stagers. Layer bytes are ALREADY measured
  (`LayerTiming.file_bytes`, summable off `telemetry.tokens[-1]` since `end_token()` fires at
  `:4198`, 38 lines above the print) and are the ONLY non-zero source on streamed **dense** — a
  MoE-only counter reports `gb_tok=0` on Llama-3.3-70B, a wrong number that looks like a feature.
  (3) **Both counters are MONOTONIC, never reset per token.** T5 wants a per-token window and T6 a
  per-request one over the same counter; only monotonic serves both. Adds `_expert_disk_total_s`
  beside the per-token `_expert_disk_s` (which IS reset, at `:3812`).
  (4) **T6 reads `torch.cuda.max_memory_allocated()` and NEVER resets peak stats** —
  `vaayu_base.py:3628-3637` and `tests/test_vram_peak_measurement.py`. The field is a process
  high-water mark and is named so.
  (5) **T8 is `python -m vaayu.server.diagnose`, a separate module** — `__main__.py`'s "No flags"
  docstring stands. `envinfo.collect(model=None)` OMITS the `model`/`budgets` sections rather than
  filling `collect_errors`.
  (6) **T10, added: a COMMITTED on-device gate.** Restore `git show 7d806e7^:test_server_smoke.py`
  (155 lines, already correct on `session_id`/explicit `stream`/`delete_original=False`/premise
  assert) as `tests/test_server_smoke.py`, skipped unless `SERVER_CKPT` is set, plus assertions on
  `requests.jsonl`, `env.json` and **`gb_tok` > 0** (the assertion that catches T5 reading zero on
  dense). T1–T4's arm ran from an uncommitted scratchpad because `7d806e7` deleted the gates;
  twice makes it permanent.
  Order: T5 → T6 → T7 → T8 → T9 → T10. Original one-liners
  (full text: `docs/issue-L1-triage-logging.md`, on this disk only):
  T5 byte counters — accumulate bytes at the expert read site into a per-token counter, add
  `bytes=`/`gb_tok=` to the `[VAAYU-IO] token summary` line (`vaayu/vaayu_base.py:4236`).
  T6 `requests.jsonl` incl. the ERROR path — wrap the `BaseException` handlers at
  `vaayu/server/engine.py:371,451` so a failed request still writes a row.
  T7 `session_event` failure must not be silent (audit item 12) — one-time warning naming the path
  and errno on the first `OSError`.
  T8 `python -m vaayu.server --diagnose` — print §4 without loading weights, plus the actionable
  taxes (APST, foreign GPU processes, I/O scheduler, drafter-on-MoE, `DEFAULT_STREAM=False` with no
  `max_tokens`).
  T9 `documentation/server.md` + `README.md` "Reporting a problem" section (server.md's half landed
  as part of Task 8; `README.md`'s does not exist yet).
  Progress: Task 1 (byte counter) `55ea63b`; Task 2 (`bytes=`/`gb_tok=`) `4a5d2c4`; Task 3
  (`_lt_cache_hit` fix) `1a0c299`; Task 4 (`reqlog.py`) `595da20`+`c342086`; Task 5 (engine
  `requests.jsonl` rows incl. errors/disconnects, plus `stream()`'s outer error-capture gap and
  the `cancelled` finish) `8c114a1`+`75e155e`; Task 6 (route-refusal rows) `30048c8`+`fc17d02`; Task 7 (`session_event` failure
  warning) `ccfb719`; Task 8 (`python -m vaayu.server.diagnose`) `a108bbe`+`002a114`; Task 9
  (README/server.md "Reporting a problem") `ba5f278`; **Task 10 (the committed on-device gate)
  `ee72e61`** — all ten landed, nothing silently dropped.

  **T10 close-out.** Restored `tests/test_server_smoke.py` from `7d806e7^`, converted to a pytest
  module skipped unless `SERVER_CKPT` is set. Default suite: **1468 passed, 12 skipped** (was 1467
  passed, 3 skipped; the whole new module — 9 tests — skips cleanly with no `SERVER_CKPT`, +1 from
  a `test_envinfo.py` regression test). Gate command:
  `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (see the
  On-device gates table below for the per-checkpoint invocation and memory cap). The brief's given
  `gb_tok > 0` assertion turned out to assume every dense checkpoint streams; it does not — a
  checkpoint small enough to fully pin (Llama-3.2-1B: 3.00 GB against an 11.06 GiB pinning budget)
  legitimately reads nothing from NVMe per token. Fixed to derive its expectation from the run's own
  `[Vaayu-Pin]` summary line instead of a hardcoded direction. Measured `gb_tok` per checkpoint (all
  four arms 9/9 passed):
  - **Llama-3.2-1B** (fully pinned, 16/16 transformer layers): every value **0.00** — correct, not a
    regression; this checkpoint cannot exercise the streamed-dense byte path on this hardware, so it
    cannot stand in for the 70B arm when verifying this feature.
  - **Qwen3-235B** (0/94 pinned, all non-expert-pinned): **1.65–59.77 GB** across 96 tokens; agrees
    with `expert misses × expert_bytes` (11,830,272 B, from this run's own `env.json`) to **0.34%**
    max / 0.07% mean relative difference.
  - **MiniMax-M2.7** (0/62 pinned, all non-expert-pinned): **1.37–74.56 GB** across 34 tokens; agrees
    with `expert misses × expert_bytes` (8,875,020 B) to **0.37%** max / 0.17% mean.
  - **Llama-3.3-70B** (16/80 pinned, 64 stream — the load-bearing arm): constant **13.86 GB** on all
    96 tokens (dense reads the same fixed set of non-pinned layers every token, so no per-token
    variance is expected). This is the arm that actually proves the byte counter fires on a streamed
    dense model.
  Also fixed, authorized as an in-task deviation: `envinfo.py`'s `budgets.prefix_cache_gb` read
  `model._prefix_cache.budget_bytes` unguarded (crashed into `collect_errors` whenever prefix reuse
  auto-disables, e.g. any non-EXL3 checkpoint) — routed through the same `cache_gb()` null-check
  helper its sibling fields already use; regression test added to `tests/test_envinfo.py`.
- **[x] L1 T5–T10 branch · FINAL whole-branch review fix wave — DONE 2026-09-04.** Ten findings
  (2 Important on the "a request never ends without a row" contract, 1 Important lock-scope, 7
  Minor) plus 4 cheap test-hygiene carries and 1 structural hardening, all in one pass.
  Suite **1482 passed, 12 skipped** (was 1468 + 15 new tests − 1 deleted duplicate).
  Report: `.superpowers/sdd/2026-09-04-L1-T5-T10-triage-logging-additive/final-fix-report.md`
  (**gitignored — this disk only**; the substance is here).
  - **F1** `prepare()` sat OUTSIDE `generate()`'s row-writing try, so any Jinja error from
    `apply_chat_template` (a `tool` message with no matching call, an unexpected role order) ended
    with NO row on both paths; on the streaming path it escaped the route's three-type `except`
    entirely. Moved inside the try (matching `stream()`), route's streaming `except` widened to
    write a refusal row and re-raise. **Consequence handled:** with `prepare()` inside the try, the
    non-streaming route's `ContextTooLong` handler would have written a SECOND row — `generate()`
    now tags those three types `finish="refused"` itself and the route no longer writes one.
  - **F2** the `slot()` admission race in `_sse` (`QueueFull` raised before `engine.stream` is ever
    entered) emitted an SSE `server_busy` and no row. One `_refusal_row` call.
  - **F3** `_sse` `break`s out of `async for … engine.stream(…)`, and Python does not close a
    generator on `break` — the row was written at asyncio-finalizer time, AFTER the slot's lock was
    released, so `wall_s` disagreed with `server.log`'s DONE line and the `after` counter snapshot
    could bill the next request's NVMe reads to this one. Now an explicit `try/finally: await
    gen.aclose()` inside the `async with`. New test drives the production shape (break, then read
    the file with no await in between).
  - **F4** `_expert_disk_total_s` was folded at the TOP of each forward, shifting a request's stall
    window by one pass (a 1-token request reported another request's number), and was never
    initialised in `__init__` (first request of a process reported `null`). Now initialised, and
    folded at the END; the top-of-pass fold stays as a catch-up for a pass that raised.
  - **F5** a non-streaming disconnect recorded `finish:"length"` (the OpenAI wire vocabulary's
    word). `requests.jsonl` is not the wire — the row now says `cancelled`, matching `stream()`
    and `documentation/server.md`. The wire value is unchanged.
  - **F6** refusal rows omitted 7 keys instead of nulling them (`row["prefix_hit"]` was a KeyError
    depending on which kind of row a script hit). Same key set now; server.md corrected —
    the fields are `prompt_tokens` and `tokens`, and the null-key rule is documented.
  - **F7** the `decode_stats_reset` latch is sticky, so a request that raised before any forward
    published the PREVIOUS request's hit rates. `decode_block` now requires `tokens > 0` too.
  - **F8** `test_no_absolute_paths_leaked_into_the_bundle` asserted `"/home/" not in text` for
    `requests.jsonl`, which is NOT scrubbed (`error_block` embeds a full traceback). Scoped to
    `env.json`; README now says the file is unfiltered, as it already said of `server.log`.
  - **F9** server.md claimed the byte counters uniformly use sector-aligned lengths at submission;
    true at 2 of 4 sites (`_load_pinned`'s preadv fallback counts unaligned `total_size`, the
    non-pinned `load_model` branch counts materialised `nbytes`). Wording softened; counting
    unchanged.
  - **F10** `_checkpoint_looks_like_moe` used `isinstance(v, int)`, which is True for `False` — a
    dense config with `"expert_parallel": false` or `"num_experts": 0` tripped the drafter-on-MoE
    tax. Now `and not isinstance(v, bool) and v > 0`.
  - **Hardening:** `reqlog.MONOTONIC_KEYS` + a module-level `diff()` that asserts against it, so a
    future field wired through it by mistake fails loudly instead of being silently
    miscategorised (the exact Critical a mid-branch review already caught once).
  - **Carries:** stale `vaayu_base.py:3628-3637` anchor in `test_reqlog.py` replaced with a
    searchable one; `test_session_event_still_never_raises` now resets `_session_event_warned`;
    `test_docs_reporting.py`'s docstring says what it does (substring checks, not live-code
    assertions); deleted `test_token_summary_bytes.py::test_counter_delta_is_the_token_window`
    (duplicate of `test_read_bytes_counter.py::test_note_read_bytes_accumulates`).
- **[x] `7d806e7` ("remove unwanted files") deleted both on-device gate scripts — docs FIXED**
  **2026-09-04 (I6/M3, final fix wave); scripts stay deleted, deliberately.** Eight files were
  removed on 2026-09-03, before Task 8 of this L1 work ran: `test_server_smoke.py`,
  `test_batched_dense_smoke.py`, `test_batched_decode.py`, `test_kv_persist_coherence.py`,
  `test_quick.py`, `control_single_stream_repro.py`, `exllamav3-server.py` and `speed-up-vaayu.md`.
  That deletion is deliberate and was never this row's to reverse — only the stale documentation
  was. Fixed: `CLAUDE.md` (lines formerly 25, 31, 150) no longer presents `test_batched_dense_smoke.py`
  as a runnable guard, both naming its deletion instead; the **On-device gates** table below now
  marks the `dense smoke` and `HTTP server` rows **DELETED** rather than listing them as commands;
  `documentation/server.md`'s "The on-device gate" section (was the worst instance — tracked,
  shipping, user-facing) says the same; and the two `speed-up-vaayu.md` citations above (was lines
  35, 262) now say the file is gone rather than pointing a reader at nothing. No replacement gate was
  invented anywhere, per the finding's own instruction. Whoever next touches the on-device gates
  should restore equivalent scripts, committed and reviewed — Task 8's scratchpad one-off is still
  not a replacement.
- **[x] `env.json`'s `kv_cache_mode` reported the requested value, not the resolved one — FIXED
  2026-09-06.** `env.json` recorded `"auto"` verbatim, against that file's own premise (every field
  is a DERIVED fact, never the raw request echoed back), so a maintainer reading a mailed bundle
  could not tell which KV dtype actually ran. **Root cause was deeper than the fix sketch
  ("read the resolved mode off the KV cache constructor"): no resolved value existed anywhere to
  read.** `envinfo._budgets_section` read the raw ctor arg `model.kv_cache_mode`
  (`vaayu_base.py:1626`), and the resolution lived INSIDE `_create_kv_cache`, **copied a second
  time** into `_kv_cache_is_bucketed` — neither stored it, and the cache itself is not built until
  first use, long after `env.json` is written.
  **Fix:** one pure module-level `resolve_kv_cache_mode(mode, loader_strategy)` in `vaayu_base.py`,
  called by all three readers (`_create_kv_cache`, `_kv_cache_is_bucketed`, `envinfo`). Same shape as
  the `forward` tag: one source of truth, because a second copy of the branch drifts and then
  misreports. `_kv_cache_is_bucketed` collapses to `resolved != "fp16"`.
  **Also resolved, same class of defect:** `fp8` on a PyTorch without `float8_e4m3fn` silently
  becomes fp16 — the resolver now says `fp16`, and the operator-facing "FP8 KV cache unavailable"
  warning moved with it (keyed on the REQUESTED mode, so a plain `fp16` request never prints it).
  An unrecognised mode is passed through unchanged; `_create_kv_cache` still raises `VramPlanError`
  on it. `documentation/server.md` now states that `budgets.*` is resolved and `config.*` is what
  you asked for — the requested value is NOT lost.
  **Tests: +26** — 3 in `tests/test_envinfo.py` (the bug: auto+EXL3→bf16, auto+non-EXL3→fp16,
  explicit bf16 unchanged; asserts the request survives in `config`), 23 in
  `tests/test_kv_cache_mode_dispatch.py` (resolver output per mode; **the resolved mode builds the
  same cache the requested one does**; `_kv_cache_is_bucketed` agrees with the cache actually built;
  the fp8-fallback path under `monkeypatch.delattr(torch, "float8_e4m3fn")`). The envinfo tests were
  confirmed failing first on their assertion (`+ auto`); the anti-drift tests are new-API, so they
  were verified by INJECTING drift instead — `auto` forced to bf16 → 3 red, `!= "fp16"` flipped to
  `!= "bf16"` → 4 red, both clean after revert.
  **Trap for the next reader:** `tests/test_envinfo.py` defines its own name-only
  `EXL3LoaderStrategy` fake (used to check the reported strategy NAME); the resolution is an
  `isinstance` against the REAL class, so a test that uses the local fake silently measures the
  fp16 branch. Use `Real.__new__(Real)` — `__init__` requires exllamav3 importable.
  Suite **1612 → 1638 passed, 12 skipped** (baseline re-measured against a stashed HEAD the same
  day; the 1570 written above was stale, as it warns). **HTTP on-device gate 9/9 on BOTH sides of
  the resolution**, each in its own `systemd-run` scope (`MemoryMax=22G`, `-s`, `delete_original=False`,
  checkpoint verified intact): Llama-3.2-1B 5.8 s (`StandardSafetensorsStrategy` → `config` `auto`,
  `budgets` **fp16**) and MiniMax-M2.7 55.9 s (`EXL3LoaderStrategy`, module-list forward → `config`
  `auto`, `budgets` **bf16**), the latter cross-checked against that run's own
  `[Vaayu] KV Cache: bf16 ring buffer` log line. `collect_errors` empty on both.
- **[x] the test suite creates real `logs/run-*` directories inside the repo — FIXED 2026-09-06.**
  **Root cause:** `ServerConfig.log_dir` defaults to `"logs"` — a RELATIVE path, so it resolves
  against the CWD, which for a test run is the repo — and `create_app`'s lifespan calls
  `new_run_dir(config.log_dir)` for every `config is not None`, injected engine or not
  (`vaayu/server/app.py:338`). Any test building a bare `ServerConfig` inherits it.
  **This row's own file list was wrong, measured per-file before the fix** (dirs created by
  `./venv/bin/python -m pytest tests/<f>.py`): `test_delete_original` **4**,
  `test_server_config_passthrough` **4** and `test_gc_freeze` **2** — the last two were NOT in this
  row — `test_gpu_audit` **1**, and `test_shutdown` **0**, which this row named but which passes
  `engine=` with `config=None`, so it never reaches the call. **11 per full suite run, not ~9.**
  **Fix:** a new `tests/conftest.py` (the suite had none) with an autouse fixture that monkeypatches
  `vaayu.server.app.new_run_dir` to swap the root for a `tmp_path_factory` dir **only when it equals
  the dataclass field's own default** (read off `dataclasses.fields`, not typed as a literal). An
  explicit `log_dir=` — `test_logs.py`'s read-only root, anything read back — is untouched, and
  `test_envinfo.py:408`'s assertion that the reported `config.log_dir` is still named `logs` keeps
  holding, which a fixture that rewrote `cfg.log_dir` would have broken. It **redirects rather than
  disables**: the run dir is still created, so the run-dir path stays under test.
  **Verified**, both arms same-box same-day, the baseline re-measured against a stashed HEAD rather
  than quoted: HEAD **1638 passed, 12 skipped, 11 stray dirs** → branch **1639, 12, 0**. The +1 is
  one regression test, `test_a_default_config_does_not_write_run_dirs_into_the_repo`
  (`tests/test_logs.py`), confirmed failing first on its assertion (the extra `run-*` path) and
  asserting `run_dir is not None` too, so a future fixture that merely turns logging off cannot
  satisfy it.
  **Collateral, recorded because it is not recoverable:** clearing the accumulated dirs mid-session
  used `rm -rf logs/run-* logs/latest`, which took **all 877** — the real on-device runs
  (2026-09-03 Task 8, 2026-09-05) as well as the test litter. `logs/` is gitignored, so those are
  gone; `logs/server.log` (594 KB, 2026-09-03) and `logs/sessions/` survive. No row depended on
  them — every gate result is written up in this file or in `docs/` — but a gate wanting its raw
  run dir must re-run.
  The `logs.py` module-global → `_Capture` refactor that the L1 fix-wave row pairs with this one is
  still NOT done; it was a refactor then and remains one.
- **[x] a client that disconnects before generation starts leaves NO `requests.jsonl` row —
  FIXED 2026-09-05.** Found 2026-09-04 by the whole-branch re-review; **pre-existing, not
  introduced by the T5–T10 branch**. The contract "a request never ends without a row"
  (`README.md`, `documentation/server.md`) now holds with no known exception.
  **The row's own fix sketch — "widen the `try`" — was not sufficient, and a probe found a THIRD
  window it does not reach.** Measured before the fix (scratchpad probe, one row-count per abandon
  shape): normal stream → 1 row; disconnect at the 1st pre-`try` yield → **0**; at the role-chunk
  yield → **0**; **disconnect while QUEUED → 0**, and that one is already *inside* the `try`.
  **Root cause (deeper than the positional one):** `_sse` delegates row-writing entirely to
  `engine.stream`'s `finally`, which runs only once that generator's *body* has started. The whole
  span before it — the three yields, the wait on `engine.slot`'s lock, `slot` itself — was unowned,
  and exactly one hole in it was plugged, by a typed `except QueueFull`. Every other exit wrote
  nothing. `CancelledError` (the queued case) and `GeneratorExit` are both `BaseException`, so the
  route's `except Exception` missed them, and neither may be answered with a yield.
  **Fix:** the three yields moved inside the `try`, plus an `engine_owns_the_row` flag set at the
  exact statement where `engine.stream` takes ownership (no await between it and the first
  `__anext__`, so the handover is exact) and one `finally` that writes the owed row when the
  engine never took it. Added `except BaseException` classifying `GeneratorExit`/`CancelledError`
  as `finish="cancelled"`. `_refusal_row` gained a `finish=` parameter (default `"refused"`, so
  every existing call site is unchanged) and now emits `"error": null` when passed
  `exc_type=None` — matching `EngineHandle.stream`, which already nulls the error on its own
  `GeneratorExit`, and `documentation/server.md`, which says only `error`/`refused` rows carry an
  error object. The `QueueFull` row moved into the same `finally`; it can only be raised before the
  handover, so its behaviour is unchanged and its shipped test still passes.
  **Verified:** 4 new tests in `tests/test_reqlog_refusals.py` (pre-`try` window parametrized over
  all three yield positions, plus the queued/`CancelledError` case), each confirmed failing against
  the code they replaced with the assertion message, not an error. Suite **1482 → 1486 passed, 12
  skipped** — baseline re-measured against a stashed HEAD the same day, not quoted. Independent
  probe re-run after the fix: all four shapes write exactly one row.
  **HTTP on-device gate: all four arms 9/9**, each in its own `systemd-run` scope, `-s` throughout
  (see the gate-invocation bug below), every checkpoint verified intact afterwards
  (`delete_original=False`): Llama-3.2-1B 5.6 s · Qwen3-235B 110 s · MiniMax-M2.7 55 s ·
  Llama-3.3-70B 405 s at `MemoryMax=26G`. That covers **all four dispatch-table rows** — fused-EXL3,
  module-list, and dense both fully-pinned and streamed. `gb_tok` reproduced T10's 2026-09-04
  figures closely, so the byte counter did not move either: Qwen 1.53–59.77 GB over 96 tokens (was
  1.65–59.77), MiniMax 1.32–74.56 over 34 (was 1.37–74.56), Llama-3.3-70B **constant 13.86 on all
  96** (identical), Llama-3.2-1B all 0.00 (correct — it fully pins).
- **[x] V1 · the `finish` vocabulary in `requests.jsonl` was path-dependent for the same failure — FIXED 2026-09-05** —
  **[`docs/issue-V1-finish-vocabulary-path-dependent.md`](docs/issue-V1-finish-vocabulary-path-dependent.md)**
  (self-contained; pick up cold).
  Introduced 2026-09-04 by the final fix wave (`9aa3a1f`), flagged by its re-review, not blocking
  merge. A template/Jinja failure out of `prepare()` records `finish="refused"` on the STREAMING
  path (the route's `_refusal_row` hardcodes it) but `finish="error"` on the NON-streaming path
  (`generate()`'s `finally`). `documentation/server.md` defines `refused` as "rejected before
  generation started — an oversized prompt, a missing `session_id`, a full queue" and `error` as
  "the request raised", so a 500-producing template failure is currently filed as `refused` on one
  path only. Two shipped tests bake the asymmetry in. **Fix sketch:** let `_refusal_row` take the
  `finish` value rather than hardcoding it, and pass `"error"` from the widened `except Exception`.
  **Half of that sketch was already built** (2026-09-05, by the disconnect fix above, which needed
  it): `_refusal_row` takes `finish=` with default `"refused"`, so no call site changed.
  **V1a/V1b/V1c DONE 2026-09-05, plus BOTH siblings.** The streaming route's bare-`except Exception`
  (`vaayu/server/app.py`, the handler whose comment begins "prepare() raises more than those three")
  now passes `finish="error"`; the `except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate)`
  handler above it keeps the `"refused"` default, which is correct — those three are 400s and
  `generate()` already calls them `refused`. **The dividing line is the STATUS CODE, not the moment:**
  400/503 and client-fixable → `refused`; raised and answers 500 → `error`, template failures
  included, even though they are rejected before generation starts. That rule is now written into
  `documentation/server.md`'s `finish` table, because "rejected before generation started" read as
  covering the template case and is what made the wrong value look right.
  **Sibling 1 FIXED** (was: `stream()`'s fallback `prepare()` left the three refusal types as
  `"error"` while `generate()` tagged them `"refused"`): `stream()`'s outer `except BaseException`
  now mirrors `generate()`'s branch, so the word does not depend on which engine method you called
  either. Unreachable from HTTP (the route always passes `prepared=`), reachable for every other
  caller. **Sibling 2 DOCUMENTED, not changed** (a cancelled NON-streaming request prints
  `finish=length` in `server.log`'s DONE line while its row says `cancelled`): `documentation/server.md`
  now says the two files disagree there and why — the log line carries the WIRE value, and F5
  established that split deliberately, so changing it was the riskier option.
  Tests: `test_a_jinja_error_on_the_streaming_path_writes_a_row` flipped `refused`→`error` (changed,
  not deleted — it asserted the old value on purpose); new
  `test_a_jinja_error_gets_the_same_finish_on_both_paths` drives ONE Jinja failure down both HTTP
  paths and asserts the two rows agree (V1c, the anti-drift test); new
  `test_stream_tags_the_three_refusal_types_refused_like_generate_does` covers sibling 1 with a
  genuinely oversized prompt rather than a monkeypatched raise. All three confirmed failing first,
  each on its assertion message rather than an error. `test_a_prepare_failure_still_writes_a_row_non_streaming`
  is UNCHANGED — it was the one already right, and stays the reference.
  Suite **1489 → 1491 passed, 12 skipped**. HTTP on-device gate re-run (route + engine error paths
  touched): **9/9 on Llama-3.2-1B**, own `systemd-run` scope, `MemoryMax=22G`, `delete_original=False`,
  checkpoint intact.
- **[x] G1 · the committed HTTP gate was RED under its own documented invocation — FIXED 2026-09-05** —
  **[`docs/issue-G1-server-smoke-gate-capture-blind-spot.md`](docs/issue-G1-server-smoke-gate-capture-blind-spot.md)**
  (self-contained; pick up cold).
  Found 2026-09-05 while running the gate the `_sse` disconnect fix obliged; **pre-existing,
  reproduced identically on a stashed HEAD** (`1 failed, 8 passed`), so it is not that fix's.
  `tests/test_server_smoke.py::test_the_token_summary_reports_nvme_bytes_consistent_with_pinning`
  fails at `assert values, "no token summary carried a gb_tok= field"`. **Root cause confirmed:**
  under pytest's default capture, `sys.stdout` is rebound, so Python `print()` never reaches fd 1
  and `capture_stdio`'s fd-level pump never tees it. The run's `server.log` held **4 lines, all
  `[vaayu_rust]`** — those are native fd-1 writes, which is exactly why they survive and no
  `[VAAYU-IO]`/`[Vaayu-*]` line does. Adding `-s` makes the same gate **9/9 on Llama-3.2-1B**.
  This is what T10's pytest conversion cost: the arm it recorded green on 2026-09-04 was a
  standalone script, which had no capture to fight. **Gates table below now carries `-s`.**
  **G1a/G1b FIXED 2026-09-05 — the module no longer depends on the operator remembering the flag.**
  A module-scoped autouse guard (`uncaptured_stdout`, `tests/test_server_smoke.py`) reads
  `pytestconfig.getoption("capture")` and, unless it is `"no"`, calls `pytest.fail(pytrace=False)`
  with a message naming `-s`. **FAIL, not skip** (G1b): a skipped gate is a gate nobody notices is
  not running — the failure mode `7d806e7` already caused once. The `client` fixture takes the guard
  as an explicit argument, so the refusal is ordered BEFORE the load rather than left to autouse
  resolution — verified on-device: without `-s` the run is `9 errors in 3.20s` and never prints a
  `loading` line, with `-s` it is **9/9 on Llama-3.2-1B in 5.74 s** (own `systemd-run` scope,
  `MemoryMax=22G`, `delete_original=False`, checkpoint intact).
  **The allowlist is `{"no"}` alone, and that was MEASURED, not reasoned:** a probe printed through
  a live `capture_stdio` pump under all four modes — only `no` tees a Python `print()`; `fd`, `sys`
  and **`tee-sys`** all hide it (`tee-sys` was the plausible-looking second candidate and it fails,
  so do not widen the check on the name). A native `os.write(1, ...)` tees under all four, which is
  exactly why the broken run's `server.log` looked written rather than empty.
  Three regression tests in `tests/test_server_smoke_guard.py` (the predicate over all four modes;
  a nested pytest run asserting the refusal message and that no load was attempted; and that the
  module still SKIPS cleanly with no `SERVER_CKPT`, so the guard cannot turn the default suite red).
  Suite **1486 → 1489 passed, 12 skipped**. `documentation/server.md`'s "The on-device gate" section
  records the fail-not-skip decision and the `tee-sys` result.
  **Note for whoever adds an assertion here:** the blind spot is a property of every `server.log`
  read in the module, not of the one test that hit it; `env.json` and `requests.jsonl` are written
  to files directly and are unaffected.
- **[x] `avg_nvme_throughput` was inflated by RAM-cache hits on DENSE models — FIXED
  2026-09-06.** The strategy cache (`loader_strategy._cache`, reached through
  `_load_from_strategy` → `load_from_disk`) was the sibling of the one Task 3 fixed and had no
  hit signal, so the layer loop billed a zero-I/O hit as a real read with full `file_bytes`.
  Dormant on MoE (`is_moe` forces that cache's budget to 0), live on dense. Fix: a
  `_cache_get()` helper + per-layer-name `cache_hits` dict on `LoaderStrategy` (both
  implementations), copied into the loop's dict by `_load_from_strategy`; that dict renamed
  `_non_expert_from_cache` → `_layer_from_cache` since it now carries both loaders;
  `LayerTiming.nvme_throughput_gbps` guarded on `cache_hit`. **Same-box A/B on
  Llama-3.3-70B, 4 tokens:** before, 3/83 layers billed as hits, 38.53 GB/token,
  **7.84–8.08 GB/s — above the drive's 7.45 GB/s spec, i.e. impossible**; after, 33/83 hits,
  22.47 GB/token, 3.94–4.29 GB/s, under the ~7.1 GB/s plateau. `gb_tok` and
  `requests.jsonl`'s `bytes_read_gb` were never affected (persister read-site counter).
  Suite **1639 → 1643 passed, 12 skipped**; 4 new tests in `tests/test_telemetry_cache_hit.py`.
  [`docs/2026-09-06-nvme-throughput-cache-hit-dense-RESULTS.md`](docs/2026-09-06-nvme-throughput-cache-hit-dense-RESULTS.md)
- **[x] Stale disk timings on a cache hit — FIXED 2026-09-06** (follow-up to the row above,
  same doc). `get_and_clear_pinned_timings` is a POP and `_prepopulate_ram_cache` reads every
  cacheable layer at LOAD time with nothing to consume those entries, so the first token to hit
  the RAM cache popped a pre-inference read and billed itself its timings. **Measured on
  Llama-3.3-70B: 30 of 33 hit layers on token 0, ~115 ms each = 3.50 s of phantom `io_wait`;
  0/33 after.** Tokens 1+ were always clean — token 0 had already popped the entries, which is
  the signature to recognise if it ever returns. Fix: `_disk_timings_for_layer()` in
  `vaayu_base.py` (replaces five duplicated lines at both prefetch branches) returns zeros plus
  the real wall-clock wait on a hit, and still pops the stale entry so it cannot be attributed
  to a later read. Affected `TokenTiming.total_seconds` and per-token io_wait analysis only;
  `avg_nvme_throughput` already excluded hits. Suite **1643 → 1647 passed, 12 skipped**.
- **[x] L1 final fix wave — 2026-09-04.** A whole-branch review of `l1-triage-logging` (11 commits,
  `f585014`..`a4e2e7a`) returned "merge with fixes"; this row is that pass. Fixed, each with a new
  regression test in `tests/test_logs.py` / `tests/test_envinfo.py` unless noted: **C1** `new_run_dir`
  no longer lets an unwritable `logs/` root prevent a boot — it warns and returns `None`, and
  `app.py` now skips `capture_stdio` when `run_dir is None` (env.json and `log_dir=` were already
  guarded). **C2** the drain barrier: `drain()` now writes the sentinel via `os.write(1, ...)` instead
  of `print()` (survives `sys.stdout` being rebound — Jupyter, `redirect_stdout`, capsys) and
  `_emit()` matches it by substring instead of equality (survives a partial line with no trailing
  newline sharing it); folded in the console-tee guard so the sentinel no longer flashes on the
  operator's terminal. **I1** `_session_sink` is now keyed on `(run_dir, session)`, not `session`
  alone, and is closed in `restore()`. **I2** `_pump_active` is now a counter, not a bool, so a
  nested `capture_stdio()` no longer disables `drain()` for the outer pump's whole remaining life.
  **I4** `_session_write` now rotates at `rotate_mb` like `server.log` does; `documentation/server.md`
  gained a retention sentence. **I5** `envinfo.py`'s `_vaayu_section` now requires `(repo/".git")`
  to exist before trusting `git -C repo`, so a venv nested inside a user's own project reports `null`
  instead of that project's branch/commit. **I7 / L1-B1b** and the FIFO direction — see the row
  above, now closed. **Item 7** `_budgets_section` guards `_expert_cache`/`_expert_vram_cache` being
  `None` (legitimate on a dense model) so a healthy dense report no longer carries three
  `AttributeError`s in `collect_errors`. **I6/M3** stale doc references to the deleted on-device gate
  scripts — see the `7d806e7` row above, now closed. Full suite **1385 → 1397 passed, 3 skipped**:
  every one of the 12 new tests is additive, and each encodes a named invariant from a specific
  finding rather than asserting whatever the code now does (the one changed assertion,
  `test_the_warning_buffer_is_bounded`, was flipped deliberately to match I7's new eviction
  direction and asserts that the OLDEST warnings survive).
  **Recorded, not fixed (explicitly out of scope per the review) — do not build without re-reading
  the review first:**
  - **The `logs.py` module-global consolidation.** `_pump_active`, `_session_sink`, `_warnings`,
    `_current_session`, `_overlap_warned`, `_drain_event` are module globals while the resources they
    describe (pump thread, `run_dir`, sink) are closures inside one `capture_stdio` call — nothing
    owns the pair, which is the ROOT CAUSE of C1/I1/I2 above and of tests having to poke
    `logs._pump_active` by hand. Collapsing them into one `_Capture` object is ~30 lines and turns
    those into structural impossibilities. Worth doing before this is anyone's reference
    implementation — but it is a refactor, not a fix, and pairs naturally with the
    `tests/conftest.py` autouse fixture in the row above (both touch the same capture lifecycle).
  - `_URL_RE` (`envinfo.py`) is greedy (`https?://[^\s<>"]+` would close it cheaply).
  - `drain()` can return `True` on a PREVIOUS timed-out drain's leftover sentinel still sitting in
    the pipe (only reachable once C2's own timeout has already fired once — a numbered token would
    close it).
  - `_host_section` (`envinfo.py`) reads `/proc/meminfo`, which reports HOST ram, not the cgroup
    limit, inside a container.
  - `on_disk_gb` (`envinfo.py`) walks the checkpoint with `rglob` + `stat` per file at EVERY startup
    — expensive on a large checkpoint, and re-walks the same bytes every restart.
  - A refused boot still creates a run directory and repoints `latest` at a run that never served —
    `new_run_dir`/`_point_latest` run before the load can fail.
  - `_session_write` re-`mkdir`s and re-`open`s once per line after a write failure (resets `key` to
    `None` inside the `except OSError`) instead of backing off by remembering the failed session.
- **[x] S1 · a configured drafter was dead weight on every MoE target — S1a+S1b FIXED, S1c
  MEASURED, S1d REFUTED. DONE 2026-09-05.** —
  **[`docs/issue-S1-drafter-dead-weight.md`](docs/issue-S1-drafter-dead-weight.md)**.
  Qwen2.5-Coder-1.5B loaded fully to VRAM (1544M, 3.09 GB) and was then discarded by the Fix-#9 gate
  on every start, on both actively-tested models. **The VRAM WAS correctly released — verified by
  arithmetic on the load logs (12.01 GB vs 12.10 GB); there was no leak, do not go looking for one.**
  - **S1a (D1, the circular gate) FIXED.** New pure helpers in `vaayu/vaayu_base.py`:
    `spec_budget_upper_bound(available_bytes, geometry)` asks the REAL sizing helpers
    (`derive_expert_ceiling` with no prefix reservation, `compute_expert_budget` against RAM
    measured before the load) what they could return, rather than re-deriving their arithmetic;
    `max_draft_tokens_that_fit`; `spec_preload_refusal`. Because the bound can only OVER-state the
    budget, the pre-load gate can never refuse what the post-load gate would accept — a
    parametrized invariant test asserts exactly that over four geometries × four RAM figures ×
    four draft_tokens. The post-load gate is unchanged and still decides against the real budget.
  - **S1b (D3, nothing told the user) FIXED.** `model.speculative_status` is set on EVERY load path
    (`{requested, draft_tokens, active, refused_because}`) and is surfaced by `env.json` (new
    `speculative` section) and `GET /v1/models`. Both gates write the same field, so a reader gets
    one answer in one shape whichever fired.
  - **Message honesty, found on-device:** where a `draft_tokens` value fits the BOUND the refusal
    says it "could fit … the post-load check re-tests it against the real one", never that setting
    it will work — measured on Qwen, the bound (20.6 GB) names `draft_tokens=1` while the real
    budget (15.8 GB) still refuses it. On Qwen at `draft_tokens=5` nothing fits at all, and the
    message names **batched decode** instead of "reduce draft_tokens", which on that geometry is
    advice that cannot work.
  - **On-device, Qwen3-235B, three arms, own scopes, same box same day:** before, `[Vaayu-Drafter]`
    loads for **3.85 s** and is discarded; after, **zero** drafter lines and the refusal names its
    numbers. Expert VRAM cache **6.3 GB in all three arms**; expert RAM budget 15.9 / 15.8 / 15.8 GB
    (jitter tracks measured available RAM, 24.32 vs 24.29 GB) — removing the load does not shrink
    the caches. MiniMax-M2.7 refuses pre-load too, on the other message branch.
  - **S1c (D2, is the 53.4 GB bound real?) MEASURED — full numbers and method:
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md),
    probe `docs/probes/s1c_routing_union.py` (both gitignored; the durable summary is here).**
    No new instrumentation was needed: all three MoE forwards already write
    `_expert_history[layer_idx]`, which has had no production reader since audit F7; a HF
    `StoppingCriteria` samples it synchronously per decode step. 47 decode steps per model.
    **The bound IS too pessimistic — 24–51 % on Qwen, 14–35 % on MiniMax** (measured union at k=6:
    23.4 experts/layer vs the assumed 48 on Qwen; 31.4 vs 48 on MiniMax). Against the real budgets
    that buys exactly one more `draft_tokens` step per model (Qwen 0→1, MiniMax 2→3).
  - **[x] S1d · adaptive runtime gate BUILT, WIRED, and MEASURED on-device, 2026-09-05.**
    Design: [`docs/superpowers/specs/2026-09-05-S1d-adaptive-spec-gate-design.md`](docs/superpowers/specs/2026-09-05-S1d-adaptive-spec-gate-design.md).
    Plan: [`docs/superpowers/plans/2026-09-05-S1d-adaptive-spec-gate.md`](docs/superpowers/plans/2026-09-05-S1d-adaptive-spec-gate.md)
    (both **gitignored — this row is the durable copy**). New module
    `vaayu/speculative/adaptive_gate.py`: pure helpers `observed_union_per_layer`, `union_bytes`,
    `spec_breakeven_acceptance`, plus `AdaptiveSpecGate`, a three-phase state machine
    (CALIBRATE → TRIAL → RUN, or → OFF from either) wired into `speculative_generate` (MoE only,
    via `_build_spec_gate`) and reported in `speculative_status["calibration"]` (`envinfo.py`,
    `/v1/models`).
    **Two refuted alternatives, both measured, neither built:** a static gate that swaps the
    Fix #9 zero-overlap bound for the OBSERVED union still ADMITS Qwen's `draft_tokens=1`
    (13.53 GB fits a 15.8 GB budget) and that configuration measures **0.836×** (16 % slower) —
    bytes-fit is necessary, not sufficient. The obvious analytic escape fails in the OTHER
    direction: the independent-routing estimate `num_experts·(1−(1−top_k/num_experts)^k)` puts
    MiniMax's 4-position verify at **16.80 GB** against a 14.7 GB budget and REFUSES the one
    configuration worth 1.29×. Only an OBSERVED union, from the model's own routing, gets both
    right — hence a runtime gate.
    **Phase constants (module constants, no config knobs — CLAUDE.md §2):** `CALIBRATE_STEPS = 8`
    is a FLOOR, not the calibration length — the gate actually waits `max(8, draft_tokens+1)`
    steps, because `observed_union_per_layer` needs one full window of `draft_tokens+1` snapshots
    to measure anything; below that, `draft_tokens ≥ 8` would leave zero windows and latch
    `off_union` on every load regardless of budget (CLAUDE.md §5: never silently degrade).
    `TRIAL_DRAFTED_TOKENS = 48` — at 48 drafts the binomial SE is ~0.07, so Qwen's measured
    acceptance sits ~3 SE below its bar, decisive without over-sampling. `TRIAL_ROUND_CAP = 64` —
    bounds the case where the proposer rarely drafts at all.
    **Three on-device arms, own systemd scopes, same box same day:**
    - **MiniMax-M2.7, `draft_tokens=3`, PLD, NO `VAAYU_SPEC_FORCE`** — self-admitted on its own
      evidence. **RE-MEASURED against the final code** (`calibration.decision="run"`,
      `observed_union_gb=12.45` vs `budget_gb=14.37`, `breakeven_acceptance=0.6094`,
      `measured_acceptance=1.0`, `drafting_rounds=16`): tok/s **1.7795 (off) → 2.3253 (pld) =
      1.307×**, ids byte-identical. (The pre-fix figures were 1.8135 → 2.2883 = 1.26×, taken
      against a gate that rebuilt itself every request — see the Critical below.)
    - **Qwen3-235B, `draft_tokens=1`, PLD, NO `VAAYU_SPEC_FORCE` — the anti-regression arm.**
      **RE-MEASURED against the final code:** `calibration.decision="off_acceptance"`
      (`observed_union_gb=13.59` vs `budget_gb=15.89`, `breakeven_acceptance=0.5277`,
      `measured_acceptance=0.3514`, `drafting_rounds=37`). tok/s
      **1.080 (off) → 1.0064 (pld) = 0.932×**. Ids byte-identical. (Pre-fix: 1.1152 → 1.0193 =
      0.914×.) **The gate fired and avoided the catastrophic 0.836×/16%-slower regression — but
      the result is 6.8 % slower, not the ~2 % this row originally targeted, and that is
      reported as measured, not massaged.** Root
      cause: `CALIBRATE_STEPS(8) + TRIAL_ROUND_CAP(64)` = up to 72 of the 128 generated tokens pay
      the cost of finding out speculation does not pay, before latching off; that fixed
      exploration cost does not amortize over a 128-token run (it would over a longer one — open
      item below, in ▶ STILL OPEN).
    - **Llama-3.3-70B dense + Llama-3.2-1B drafter:** no `calibration` key in `speculative_status`
      on either arm (`_build_spec_gate` returns `None` when `not self.is_moe`) — dense builds no
      gate, confirmed. tok/s **0.166 (base) → 0.805/0.789/0.788 (spec, 3 calls) ≈ 4.8×** — a little
      under P7's banked 5.47–5.57× band; plausible same-box noise (this box had just run both MoE
      arms), not attributable to any S1d code path (dense never touches `is_moe`-gated code). Ids
      byte-identical to the base arm's prefix on all 3 calls — losslessness holds.
    **Two real bugs found and fixed while measuring — without both, the arms above could not have
    been produced honestly, so both are carried here rather than left implicit:**
    1. The OLD Fix #9 / S1a static bound gates (pre-load and post-load, `vaayu_base.py`) still
       refused to even BUILD a speculator for MoE without `VAAYU_SPEC_FORCE=1`, running before the
       new runtime gate ever got a chance — exactly the admission test the design says CALIBRATE
       now performs instead (design doc §2: "this is the admission test the bound used to
       perform, decided on observation instead of a worst case"). **Fixed** by scoping both static
       checks to `drafter_model_path` only: a real weighted drafter still has load cost worth
       avoiding pre-emptively, but a prompt-lookup proposer has zero load cost, so its admission is
       now decided purely by `AdaptiveSpecGate`.
    2. `speculative_generate`'s wiring called `gate.record_round(k, num_accepted)` with
       `num_accepted` straight from `verify_and_accept()` — which is **1..k+1**, always including
       the bonus/correction token every round guarantees (see `prompt_lookup.py`/`standard.py`
       docstrings). The gate (and its own tests — `record_round(drafted=3, accepted=3)` meaning
       "perfect") expects the count of DRAFTED tokens that matched, **0..k**. This inflated
       `measured_acceptance` by ~1 token/round and is why the FIRST Qwen measurement showed
       `decision="run"` with `measured_acceptance=1.34` — the anti-regression arm this row exists
       to prove would have silently passed with the gate never firing. **Fixed** to
       `gate.record_round(k, max(0, num_accepted - 1))`.
    Both fixes verified against the full suite: **1604 passed, 12 skipped** (was 1570 at S1's
    close; +34 new tests from this branch, all green after both fixes).
    **Fix round 1 (review, 2026-09-05): bug 2's fix had no regression test, and structurally
    could not get one from the existing fakes** — `_FakeSpeculator`/`_CountingSpeculator` in
    `tests/test_adaptive_spec_gate_wiring.py` always draft 0 tokens, so `k` is always 0 and the
    accept-count arithmetic the bug lived in never runs through them, before or after the fix; only
    the on-device Qwen run caught it. **Closed:** a new `_DraftingSpeculator` fake drafts real
    `k > 0` tokens and implements `verify_and_accept` with PromptLookupDecoder's OWN algorithm
    (copied, not hardcoded), so `test_record_round_translates_the_real_accept_contract` drives two
    real rounds through the actual `speculative_generate` loop — a full accept (drafted=3,
    accepted=3, not 4) and a reject-AT-POSITION-0 (drafted=3, accepted=0, not 1 — the exact edge
    `max(0, num_accepted - 1)` exists for) — and asserts the gate's own accumulated
    `measured_acceptance == 0.5` (would read 0.833 if the off-by-one came back). Verified the test
    actually catches the bug: reverting the fix locally fails it (`assert 3 == 6`), reapplying the
    fix passes it. Full suite: **1605 passed, 12 skipped**.
    **Fix round 2 (final whole-branch review, 2026-09-05): 4 findings, all fixed in one wave.**
    1. **Critical — the latch did not survive a request boundary.** `_build_spec_gate()`
       constructed a FRESH `AdaptiveSpecGate` on every `speculative_generate()` call, so two
       back-to-back calls each restarted CALIBRATE at snapshot 0 — a server doing short requests
       would never accumulate enough drafted tokens to reach the trial threshold, and a good
       configuration re-paid its calibration cost every request. **Fixed:** the gate is now
       cached on `self._spec_gate` and reused, rebuilt only if `draft_tokens` or the expert cache
       budget changed; dense still builds no gate. New test
       `test_the_gate_is_shared_across_request_boundaries` drives two real calls and asserts the
       same gate object, with snapshots accumulated across the boundary — fails without the fix
       (`assert <gate A> is <gate B>`).
    2. **`rollback_cache(None, n)` crashes on the suppressed-draft path.** Every calibrate round
       (and every round after a latch) sets `draft_state = None`, but Step 6 called
       `speculator.rollback_cache(draft_state, num_accepted)` unconditionally —
       `StandardDrafterDecoder.rollback_cache` does `draft_state.get(...)` once its cache is warm
       and crashes; only PLD's no-op `rollback_cache` hid this. **Fixed:** guarded with
       `if draft_state is not None:`. Two new tests
       (`test_rollback_cache_is_skipped_when_draft_state_is_none`,
       `..._also_skipped_after_a_mid_call_latch`) use a speculator whose `rollback_cache` raises
       on `None` — both crash without the guard, pass with it.
    3. **No wiring test drove `off_acceptance` end-to-end.** Every existing latch test used
       `budget_bytes=1` (`off_union`); `off_acceptance` — the path real models take — had never
       run through the real loop, which is exactly what hid finding 2. **Fixed:** new
       `_CyclingRoutingFake` (routing cycles through 4 disjoint expert sets, giving a
       non-degenerate breakeven=1.0 instead of the static fakes' 0.0) plus
       `test_off_acceptance_latches_through_the_real_loop` calibrates, drafts at 2/3 acceptance,
       and asserts `self.speculator is None`, `decision == "off_acceptance"`, and the generation
       still completes at the requested length.
    4. **An empty `_expert_history` read as a real budget refusal.** If a forward stops writing
       `_expert_history`, every calibration snapshot is `{}`, `union_k == 0`, and the gate latched
       `off_union` reporting `observed_union_gb: 0.0` — indistinguishable from a genuine
       over-budget refusal (800467d fixed this same ambiguity for the starved-window case; this
       is the other way union_k reads 0). **Fixed:** a distinct `off_no_routing` decision plus a
       named `[Vaayu-Spec] WARNING` citing `_expert_history` (CLAUDE.md §5: warn, never silently
       degrade). New tests: pure-gate
       `test_empty_routing_history_reports_a_distinct_decision_not_off_union` and wiring
       `test_empty_routing_history_latches_distinctly_and_warns` (asserts the decision AND the
       warning text via `capsys`) — both fail (`off_union` instead of `off_no_routing`) without
       the fix.
    **Minors also done this wave:** the loose `s["decision"] in {None, "run", ...}` assert in
    `test_status_reports_the_numbers_it_decided_on` now asserts the fixture's actual decision
    (`None`); a `run` decision's `status["calibration"]` is now refreshed every round instead of
    frozen on first write (now load-bearing since the gate is cached — a `run` gate keeps
    accumulating after this fix); this file's dangling "open item below" reference (S1f arm 2's
    paragraph) is now a real item in ▶ STILL OPEN; the dense arm's "0.83–0.86× band" (an absolute
    tok/s pair mistaken for a ratio) corrected to the real **5.47–5.57×** band from P7's own rows.
    Full suite after all of the above: **1612 passed, 12 skipped** (was 1605 + 7 new tests, 0
    deleted). Every new test verified to fail without its corresponding fix and pass with it.
  - **[bug] C1 · a MoE model loaded with `device="cpu"` runs a PLD or drafter speculator with NO
    gate at all — pre-existing, not introduced by S1d, deferred.** Full brief, self-contained and
    meant to be picked up cold in a fresh session:
    **[`docs/issue-C1-cpu-moe-ungated-speculation.md`](docs/issue-C1-cpu-moe-ungated-speculation.md)**
    (**gitignored — this disk only**; the summary here is the durable copy).
    **Severity is LOWER than this row's original wording ("runs unbounded") implied — corrected
    2026-09-06 after checking the code.** `patch_moe_blocks` is called at `vaayu_base.py:2400`,
    INSIDE the same CUDA block as the expert cache, so on CPU-MoE there is no expert streaming, no
    expert cache, and `_expert_history` is never written. The Fix #9 premise — a verify pass blows
    the expert cache budget and thrashes NVMe — **does not apply on this path at all.** What is
    true is narrower: speculation runs UNMEASURED and SILENTLY (no warning, `calibration` stays
    null), and a `d+1`-wide verify costs more CPU compute than a 1-wide one, so a low acceptance
    rate plausibly hurts more here than on CUDA. The silence is the §5 violation, not thrash.
    **Trap for whoever fixes it:** "just build the gate on CPU too" does not work — the gate
    calibrates on `_expert_history`, which is never written here, so every snapshot is `{}` and the
    gate would always latch `off_no_routing`. That is a refusal disguised as a measurement.
    **Recommended fix (Option A in the brief):** refuse speculator CONSTRUCTION when
    `self.is_moe and _moe_geometry is None` regardless of device, with a `[Vaayu-Spec]` warning and
    a `refused_because` entry. Testable with no checkpoint (the existing fakes already have this
    shape). `_moe_geometry` is only computed when
    `_expert_offset_index` was built, which only happens `if device.startswith("cuda") and
    self.is_moe` (`vaayu_base.py:1797`); `_expert_cache` is only constructed inside the matching
    `if self.running_device.startswith("cuda") and self.is_moe:` block (`vaayu_base.py:2078`). On
    CPU-compute MoE both stay `None`. Consequence, found while reviewing S1d's Task 6: the S1a
    pre-load bound (needs `_moe_geometry`) and the Fix #9 post-load bound (needs both) silently
    skip themselves — they always did, this predates S1d — and now `_build_spec_gate()` (S1d) ALSO
    returns `None` when `geo is None or cache is None`. So `use_prompt_lookup=True` or a
    `drafter_model_path` on a CPU-compute MoE model builds a speculator that is checked by NEITHER
    the old static gates NOR the new runtime one — it just runs, unbounded, for the life of the
    process. **Deferred, not fixed:** CPU-compute MoE is architecturally supported
    (`CLAUDE.md` §0/§5) but not on-device tested today (both actively-tested MoE targets are
    CUDA-only EXL3), and no CPU-compute MoE checkpoint exists on this box to verify a fix against —
    building one unreviewed at the tail of an unrelated branch is exactly the scope creep CLAUDE.md
    §3 warns against. Fix sketch for whoever picks this up: either gate PLD/drafter construction
    itself on `_moe_geometry is not None` for MoE regardless of device (refuse rather than run
    ungated), or compute a CPU-safe `_moe_geometry`/`_expert_cache` pair off the same checkpoint
    metadata without the CUDA-only `ExpertOffsetIndex` path.
  - **[x] S1f · CLOSED 2026-09-06 — its only open item (S1d) was already built; see the end of this row. tok/s A/B MEASURED 2026-09-05 — MiniMax-M2.7 **1.285× / 1.293×** faster with PLD
    (1.8166/1.8090 → 2.3337/2.3398 tok/s, two arm pairs), **Qwen3-235B 0.836× — 16 % SLOWER**
    (1.1076 → 0.9261). **Generated ids byte-identical in every arm**, so both are real and neither
    is a quality trade; replicate spread 0.26–0.42 %. Predicted-vs-measured agrees (MiniMax bytes
    predicted 1.28×, measured 1.285×), so the two measurements describe the same mechanism.
    **The split is what the model EMITS, not the forward:** MiniMax with thinking closed copies the
    file (0/43 rounds without an n-gram, 3.35 tokens per 4-wide verify); Qwen writes prose first, and
    every n-gram miss still pays a 2-wide verify for one token — that is the 16 % loss.
    **Trap that cost a run:** `speculative_generate` REPLACES `model.generate()` — calling
    `generate()` with a speculator attached silently runs ordinary decoding and reports a flat A/B.
    **NOT a licence to default PLD on for MoE** (a win on one target, a loss on the other, and only
    with thinking suppressed — the user's quality call). Both arms needed **`VAAYU_SPEC_FORCE=1`**, a
    measurement hatch added the same day (off by default, warns loudly, NOT a shipping mechanism:
    with it set a genuinely oversized verify pass thrashes the cache, which is what the gate
    prevents). **Open item CLOSED — it was already built the same day, later.** This
    paragraph's "Not built, not designed" was written before S1d landed, and §0 then quoted it and
    re-listed the work as P3; both are corrected as of 2026-09-06. S1d done properly — sizing on a
    union the engine DERIVES from `_expert_history`, not a per-model table (`CLAUDE.md` §5) — is
    `vaayu/speculative/adaptive_gate.py`, wired at `vaayu_base.py:4794`. **Read the `[x] S1d` row
    above, not this sentence.**
    Numbers: the S1f section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **[x] S1e-bis · the S1e verdict below was WRONG — it measured with thinking ON. CLOSED 2026-09-06**
    (its `[!]` was "blocked until P3 lands", and P3 was already built; see §0's vacancy note).
    `enable_thinking=False` is a **no-op on both checkpoints**: verified by rendering the same
    messages with and without it (byte-identical), and both chat templates END the assistant turn
    with a hard-coded `<think>\n`. Every S1e arm therefore generated inside an open think block, and
    the "forced code fence" arm appended its fence inside that block — so Qwen closing the fence and
    reverting to prose was a PROMPT ARTIFACT, not a model property. Closing the block explicitly
    (`</think>\n\n`) flips the result: **MiniMax-M2.7 goes straight to code and accepts 0.907 of
    drafted tokens (break-even 0.635), 0/43 rounds with no proposal, 3.43 GB/token vs plain decode's
    4.40 — 1.28× BETTER.** Qwen still writes prose first (0.286, below its 0.520 break-even) but
    inside the code block accepts **0.927** → 7.05 GB/token vs 8.90 — **1.26× better**.
    **Consequences:** (a) PLD on MoE is live again for the copy-heavy regime, which IS the agent
    workload; (b) **S1d LANDED 2026-09-06 — this clause is discharged.** It read
    "S1d is unblocked and now blocking": the configurations that measure well were refused by the
    zero-overlap bound (MiniMax 4-position verify: 17.61 GB by the bound vs **12.78 GB measured**,
    against a 14.7 GB budget). `AdaptiveSpecGate` now sizes that pass on the observed union and
    measured it at **12.45 GB against a 14.37 GB budget, decision `run`** — the same configuration,
    admitted; (c) **a tok/s number NOW exists** — this clause read "no tok/s number
    exists yet ... do not quote a speedup without an on-device A/B", and S1d's arms are that A/B:
    **MiniMax-M2.7 1.7795 → 2.3253 = 1.307×**, ids byte-identical, against this row's 1.28×
    bytes prediction. Qwen3-235B is **0.932×** — the gate refuses it correctly, and TRIAL's
    exploration cost (what opened §0 P4) closed 2026-09-06 as a fixed one-time charge of
    ≤25.3 token-equivalents on Qwen, not a rate.
    (d) thinking-mode state is now a first-class variable in any speculation measurement here.
    Numbers: the S1e-bis section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **S1e · SUPERSEDED by S1e-bis above — read that first. Measured with thinking ON.** PLD (zero-VRAM
    proposer) acceptance on both MoE targets, by lossless offline replay of the engine's own
    `PromptLookupDecoder.draft` over real greedy runs — no gate relaxed, nothing built. Best
    accepted/drafted: **Qwen 0.306, MiniMax 0.347**, against break-evens of **0.520** and **0.635**.
    Per-round byte accounting (a round that drafts nothing costs a 1-wide decode, so it is billed at
    `union(1)`): every arm is WORSE than plain decode — **Qwen 1.11–1.16×, MiniMax 1.34–1.48×**.
    Root cause is recorded, not inferred: both checkpoints are reasoning models that reason in PROSE
    with no `<think>` tag, and a forced code-fence arm showed Qwen closing the fence and returning
    to prose. A standard drafter is refuted a fortiori — same bar, plus 3.09 GB of expert VRAM.
    **Dense is untouched: P7's 5.57× stands, because a dense verify pass reads the same weights at
    any width — the property MoE lacks.** Numbers: the S1e section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **S1d REFUTED — do not build it.** The payoff ceiling is the union ratio: a verify pass reads
    **0.760×** the routes of 2 decode steps at Qwen's newly-admissible k=2, **0.726×** of 4 at
    MiniMax's k=4. That is the BEST case, at 100 % acceptance and ignoring cost. At acceptance 0.5
    Qwen's k=2 is 9.0 GB/token against plain decode's 8.90 — *worse*; MoE acceptance has never been
    measured (the 86–90 % on record is the DENSE P7 pairing and does not carry). And the drafter
    takes 3.09 GB of a 6.3 GB expert VRAM cache. A ~1.3× ceiling on routes, on a ~47 %-disk-stall
    decode, spent against a halved VRAM cache. **Re-open condition MET, 2026-09-05, and ACTED ON: a zero-VRAM proposer (PLD) accepts 0.907 on MiniMax and
    0.927 on Qwen-in-code with thinking suppressed — see S1e-bis. S1d (union-based sizing) was
    then BUILT the same day as `AdaptiveSpecGate`, and the tok/s A/B it called the open question
    is answered: MiniMax **1.307×**, Qwen **0.932×**. This "do not build it" verdict is HISTORY —
    read the `[x] S1d` row above, which supersedes it.**
  Suite **1491 → 1570 passed, 12 skipped**. `documentation/server.md` documents the two gates, the
  `speculative` field, and why the refusal says "could". The dense path (P7, 5.57×) must not weaken.

**Applied on the spot 2026-09-03, no row needed:** `DEFAULT_STREAM = True` in
`vaayu/server/config_local.py`. It was `False`, and with a client sending no `max_tokens` that meant
**864 s of silence** for an 809-token answer — most of the reported "slowness" was this, not tok/s.
The server already warns about the combination; nobody was reading the warning.

**Still unfixed on this box, both warned at every boot, neither a code row:** NVMe **APST** is
active (`scripts/vaayu_optimize_os.sh`, one-time, needs sudo) — Qwen is 47 % disk-stall, so this
is the one hardware tax landing on the dominant cost; and `rustdesk` holds 0.14 GiB of GPU, below
the 256 MiB refusal threshold but permanent, because budgets are sized once at load.


---

# ▶ STILL OPEN — carried forward from rows that closed

Three live items (the fourth, speculative decoding on dense, closed 2026-09-06; the fifth, S1d's
TRIAL exploration cost, closed the same day; the sixth, the content-blind latch it uncovered,
opened and REFUTED the same day — see SETTLED). Everything else in the old CARRIED FORWARD block
was `[x]` and is gone. **All three remaining are OPTIONAL by §0's test — none of them can make
Vaayu faster. The live perf work is §0's P5 and P8 (P7 was
refuted 2026-09-06 and P9 SHIPPED 2026-09-07 — see SETTLED and §0's vacancy notes).**

- **[x] Speculative decoding on streamed dense — CLOSED 2026-09-06. Both sub-rows below are done:
  the per-request bypass landed, and the dense default shipped as `USE_PROMPT_LOOKUP = "auto"`.
  Kept here for the measurements and for the decision's reasoning.** **5.57×** on Llama-3.3-70B (0.155 → 0.864 tok/s, 86 % acceptance, byte-identical)
  with the Llama-3.2-1B drafter, which shares its tokenizer. Re-measured 2026-09-02 on top of its
  three fixed bugs: **5.54× / 5.47× / 5.47×** over three back-to-back generations (0.149 → 0.828),
  18 rounds each, 90 % drafted-token acceptance, all three byte-identical to each other AND to the
  greedy baseline. Reserve-policy risk CHECKED and it does **not** fire (pinning_budget 10.75 GiB
  base vs 10.73 with the drafter resident; the −19 % clamp case was Llama-3.2-1B as the TARGET,
  a different premise). **All four P7 sub-bugs are fixed.**
  **Policy call DECIDED 2026-09-06 (with the user): a default is only safe once speculation stops
  changing requests that did not ask for it — bypass first, then default.**
  - **[x] Per-request bypass — LANDED 2026-09-06.** A loaded speculator used to change EVERY
    request on the process: `speculative_generate` is greedy at every decision point and has no
    stop-string machinery, so a sampled request came back greedy and a tool-enabled one generated
    straight past its own stop string. Dispatch (`vaayu/streaming.py`, the ONE call site) now asks
    `speculative_ignored_sampling` per request — the same helper the notice uses, so the two cannot
    disagree — and falls through to `model.generate` when this call asks for anything that loop
    would drop, printing one `[Vaayu-Spec] speculation SKIPPED` line naming it.
    `engine.warn_if_speculative_sampling_ignored` → `notice_if_speculation_skipped`: nothing is
    ignored any more, so the old "will be IGNORED / Load without drafter_model_path" text would have
    sent callers to unload a drafter costing them nothing. Wire field `warnings` unchanged.
    **Cost accepted:** the server appends stop strings to every tool-enabled request, so tool
    traffic no longer speculates — a correctness win, but it cuts speculation's reach.
    **Trap found:** `stream_generate`'s own defaults are SAMPLING (`temperature=0.7, top_p=0.9`) and
    `test_generate.py` passed neither, so the P7 harness would silently have stopped speculating; it
    now has an explicit `TEMPERATURE`/`TOP_P` knob, defaulting greedy. That also means **every P7/PLD
    number on record was taken through a harness whose NON-speculative arm ran sampled** — worth
    knowing before re-quoting a baseline, though the speculative arms were greedy regardless (that
    loop never read `temperature`).
    Verified: 7 new unit tests (watched fail first) + full suite **1654 passed, 12 skipped**, and
    two on-device gates on Llama-3.2-1B with prompt lookup, `delete_original=False`. Direct
    `stream_generate`: greedy arm speculates (`accept_rate` to 110 %) and is byte-identical across
    two runs; sampled arm prints the notice, runs NO draft rounds, and its two runs differ from each
    other and from greedy. **Through the SERVER** (`create_app` + `TestClient`, real routes — the
    interface that ships, and the one the first gate did not touch): streamed greedy = **10 draft
    rounds**, identical across two calls, no `: warning` on the wire; streamed `temperature=1.3` =
    **0 draft rounds**, the notice arrives as an SSE `: warning` comment, and the two calls differ
    from each other and from greedy. Before the fix both sampled calls came back greedy and
    identical. Scripts: `scratchpad/gate_bypass.py`, `scratchpad/gate_server.py` (not committed).
  - **[x] The dense default — LANDED 2026-09-06.** `USE_PROMPT_LOOKUP = "auto"` is now the shipped
    value in `vaayu/server/config.py` and `ServerConfig`, resolved once at load by
    `resolve_use_prompt_lookup` (`vaayu/vaayu_base.py`, beside `resolve_kv_cache_mode`) where
    `is_moe` is known: **dense → on** (1.94× streamed dense), **MoE → off**, explicit `True`/`False`
    always wins. Resolved BEFORE the speculator block reads it — `"auto"` is a truthy string, so an
    unresolved value would build a speculator on the very MoE models the default exists to spare.
    The resolved bool is kept as `self.use_prompt_lookup`; `speculative_status` is built from it, so
    nothing reports an unresolved `"auto"`.
    **MoE is excluded on EVIDENCE, not caution:** PLD is 1.285×/1.293× on MiniMax-M2.7 and **0.836×
    — 16 % slower — on Qwen3-235B** (S1f row above, ids byte-identical in every arm), and the gate's
    TRIAL phase costs a fixed 0.932× on short generations (row 5 below, still open). Defaulting it
    on would hand one actively-tested model a win and the other a loss — the asymmetry §5 forbids.
    `DRAFTER_MODEL_PATH` stays `None`: a drafter is a checkpoint path needing an exact vocab match,
    so it cannot be defaulted without baking in a model (§5).
    Verified: 9 new unit tests (watched fail first), full suite **1678 passed, 12 skipped**, and both
    on-device arms. **Dense** (Llama-3.2-1B through the server, nothing configured): resolved `True`,
    `PromptLookupDecoder` active, `speculative_status.requested == "prompt_lookup"`, 28 draft rounds,
    and the answer **byte-identical** to the same request with the speculator forced off. **MoE**
    (Qwen3-235B + the drafter from `config_local.py`, load only): `is_moe True` → resolved **False**,
    no speculator, and the pre-existing 53.4 GB drafter refusal unchanged — MoE behaves exactly as
    it shipped.
  **MoE speculation stays refuted** — see SETTLED.
- **[x] The non-streaming server route never speculated — FIXED 2026-09-06** (`56699d0`).
  Both routes now ask one helper, `choose_speculative_target` (`vaayu/vaayu_base.py`); the
  non-streaming arm went 0 → 5 draft rounds on the server gate and its text now matches the
  streamed greedy arm exactly. Original report follows, for the measurement.
  **[was] The non-streaming server route NEVER speculates.** `EngineHandle.generate` calls
  `self.model.generate(**kwargs)` directly (`vaayu/server/engine.py:442`); only the streaming route
  goes through `stream_generate`, which is where speculation is dispatched. **Measured through the
  server 2026-09-06** (Llama-3.2-1B, prompt lookup on, `temperature=0`): streamed = 10 draft rounds,
  non-streamed = **0**, same prompt, same process. Pre-existing, not caused by the per-request
  bypass — the bypass is what made it visible, because arm C was expected to speculate and did not.
  Two consequences: `POST /v1/chat/completions` with `stream:false` gets none of the 1.94×/5.57×,
  and the pending `USE_PROMPT_LOOKUP = "auto"` default would silently do nothing on that route.
  Fix is the same per-request choice of target the streaming path now makes; the two call sites
  should share one helper rather than answer the question twice. `DEFAULT_STREAM = True` is the
  shipped default, so the exposure is limited to clients that ask for `stream:false`.

- **[x] Speculative decoding corrupted output on non-EXL3 checkpoints — ROOT-CAUSED AND FIXED
  2026-09-06.** Symptom: on Llama-3.2-1B with prompt lookup, greedy speculative output diverged from
  greedy plain output at generated index 3 and 5 on two prompts, dropped tokens mid-sentence
  (`The list you delta echo foxtrot hotel`) and stopped early (17 vs 60 ids).
  **Root cause — NOT the proposer, and not PLD-specific.** `_trim_kv_cache`
  (`vaayu/vaayu_base.py:4576`) branched on `isinstance(cache, FP8DynamicCache)` then
  `hasattr(cache, 'key_cache')`. Under **transformers 5.5.4 a plain `DynamicCache` has neither** —
  its layers live in `.layers` — so the method matched no branch and **silently did nothing**. Its
  one caller is the speculative loop's Step 5, so a verify pass's rejected draft positions stayed in
  the cache and every later token was decoded against tokens that were never generated. Proof it is
  upstream of the accept logic: the divergent round was **k=0** (no draft, no accept path at all,
  just a 1-wide greedy forward) immediately after a 10-wide verify.
  **Blast radius is `kv_cache_mode` resolving to `fp16`, i.e. NON-EXL3 checkpoints only.**
  `FP8DynamicCache` (bf16/fp8) and `TurboQuantKVCache` both expose `crop` and took the working
  branch — verified by the new test, which passes on those two against the code that failed on
  `DynamicCache`. **The recorded 5.57× / 1.94× / 1.31× speculation numbers are therefore NOT
  affected**: all three were taken on EXL3 checkpoints (bf16 KV). An earlier note in this file said
  they were in doubt and blamed `PromptLookupDecoder` — that was wrong, and this row supersedes it.
  **Fix:** try `crop` first (every cache class this engine builds implements it), keep the
  `key_cache` slicing as the pre-`crop` fallback, and **raise** rather than fall through — a silent
  no-op here returns wrong text, which is the failure mode CLAUDE.md §5 forbids.
  Verified: 4 new tests, parametrized over all three cache classes plus an untrimmable one, watched
  fail first (`assert 70 == 60`); full suite **1669 passed, 12 skipped**; the losslessness probe now
  reports **IDENTICAL: True, 60/60 ids on both prompts**; and the server gate's streamed and
  non-streamed greedy arms return the same coherent text.

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
- **[x] S1d's TRIAL exploration cost — CLOSED 2026-09-06, "ACCEPT AND DOCUMENT" (the row's own
  second option), on the derivation below. No code changed on the generation path.** The
  Qwen3-235B anti-regression arm (`draft_tokens=1`, 128-token run) landed at **0.932× (6.8 %
  slower)** before `_decide_acceptance` latched `off_acceptance`. This row previously said the
  cost was "`CALIBRATE_STEPS(8) + TRIAL_ROUND_CAP(64)` = 72 non-payoff tokens" and offered
  "accept 8.6 %". **Both of those are wrong, and the corrections are the point of this closure:**

  - **CALIBRATE costs NOTHING.** With the gate not yet drafting, `draft_ids` is width 0, so
    `verify_input` is the single last token and the verify forward is the ordinary 1-position
    decode (`vaayu_base.py:4960-4972`). Those 8 steps are plain decode, not "non-payoff tokens".
  - **TRIAL's non-drafting rounds cost nothing either**, by the same width-0 path — PLD returns an
    empty draft whenever no n-gram matches. In the measured arm **27 of the 64 rounds drafted
    nothing**; only the **37 drafting rounds** were paid for.
    Both facts are now pinned by
    `tests/test_adaptive_spec_gate_wiring.py::test_a_suppressed_round_verifies_exactly_one_position`
    (verified to FAIL — `{2} == {1}` — when the verify is widened, and pass when it is not);
    the whole cost model below rests on them, so they are a test and not a comment.
  - **`TRIAL_ROUND_CAP` bounds TIME-TO-DECISION, not cost.** It is the cap that fired in the
    measured arm, and it cut the evidence to **37 of the designed 48 drafts while saving nothing**,
    because the rounds it cut off were the free ones.
  - **"8.6 %" is not reproducible from the shipped code.** `1 − 0.914 = 8.6 %` is the PRE-FIX arm,
    taken against the gate that rebuilt itself every request (Fix round 2, Critical 1). The final
    code measures **6.8 %**. Quote 6.8 %, at T=128, or the token-equivalent bound below.

  **The cost model, derived — every term is one the gate already computes and reports.** A
  drafting round drafts `d` tokens at acceptance `a`: it reads `union_k` and yields `1 + a·d`
  tokens, where plain decode would have read `(1 + a·d)·union_1`. Substituting the gate's own
  `breakeven = (union_k/union_1 − 1)/d` collapses the excess to `union_1·d·(breakeven − a)` per
  round, and `TRIAL_DRAFTED_TOKENS/d` rounds are run, so **in token-equivalents of decode the
  whole exploration costs `TRIAL_DRAFTED_TOKENS × (breakeven_acceptance − measured_acceptance)`,
  bounded above at `a = 0` by `TRIAL_DRAFTED_TOKENS × breakeven_acceptance`** — and it is zero at
  break-even and NEGATIVE (a gain) above it, which is the gate working. No model number appears in
  that expression (CLAUDE.md §5); both inputs are already in `gate.status()`.
  - Qwen3-235B: `48 × (0.5277 − 0.3514)` = **8.5 token-equivalents** designed; the arm stopped at
    37 drafts, so **6.5** — which is `6.5/128 = 5.1 %` of that run against the **6.8 %** measured,
    the ~1.7-point residual being per-round loop overhead, not bytes. Worst case `48 × 0.5277` =
    **25.3**. Cross-check that the derivation is not circular: it puts Qwen's `union_1` at
    `13.59/1.5277` = **8.90 GB/token**, which is S1e-bis's independently-measured 8.90.
  - MiniMax-M2.7: worst case `48 × 0.6094` = **29.3 token-equivalents**. (It latches `run`, so it
    pays a fraction of this and then earns 1.307×.)
  - **Exact at `draft_tokens=1`; an OVER-estimate for `d>1`**, because PLD may draft `k' < d`
    (`prompt_lookup.py`: `end_idx = min(start_idx + num_tokens, input_length)`, plus EOS
    truncation) and such a round reads `union_{k'+1} < union_k`. The bound errs high, which is the
    safe direction for a documented worst case.

  **Why it amortizes to nothing, and why that is a code property and not a hope.** The cost is paid
  **once per process**: `build_speculator` runs only at load (`vaayu_base.py:2165`), the gate is
  cached on `self._spec_gate` and its decision latches, and an `off` sets `self.speculator = None`
  so every later request takes `model.generate` — `choose_speculative_target` can only route AWAY
  from speculation, never resurrect it, and nothing reassigns `_expert_cache.budget_bytes` at
  runtime to trigger a rebuild. So the regression is `6.8 % × 128/T` in a process that generates
  `T` tokens: **< 1 % past ~870 tokens, < 0.1 % past ~8 700.** An agentic loop-engineering session
  (CLAUDE.md §0) is orders of magnitude past both. **6.8 % is a 128-token-benchmark artifact and
  must not be quoted as "the gate costs 6.8 %".**

  **Why bounding TRIAL by `max_new_tokens` was REFUSED — the other option this row offered.** The
  latch is permanent and per-process, so a decision error is permanent in BOTH directions, and both
  are far larger than the cost being saved: latching `off` on a good configuration forfeits the
  measured **1.307×** forever, and latching `run` on a bad one is the measured **0.836× (16 %
  slower)** forever. Shortening TRIAL buys the saving by making that error likelier. Qwen's own
  numbers: `p̂ = 0.3514` against a `0.5277` bar is a gap of 0.176, and the binomial SE at `n`
  drafted tokens is `sqrt(p̂(1−p̂)/n)` — **2.56 SE at n=48, 2.25 at n=37, 1.48 at n=16, 1.04 at
  n=8**. A TRIAL sized as a fraction of a 128-token request decides at n≈8–16, i.e. a **7–15 %
  chance of latching the wrong way permanently**, to save a one-time cost already worth ≤25 tokens.
  That is a bad trade at any session length, and it gets worse as sessions get longer. **Do not
  re-open this to shorten TRIAL.**


---

# ▶ SETTLED — DO NOT RE-OPEN

**Read this before proposing any perf work.** Each line is something that was BUILT or MEASURED and
then refuted. A refutation closes its row; it is not re-scoped. Full text in
`docs/2026-09-03-todo-archive-preprune.md` and `git log -p -- todo.md`.

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
  `speed-up-vaayu.md` §4 (file deleted 2026-09-03, `7d806e7` — see the bug row below; the argument
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

**This heading predates §0 and is now a LOCATION, not a priority: two rows here (6a = P5, 10 = P8)
are §0's top perf work.** §0 is the order. Read the `§0` column below before picking anything from this table.

| # | Item | §0 | Status | Cost | Why it sits here |
|---|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | OPTIONAL | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **13** | `[bug]` The adaptive spec gate's break-even is in EXPERT BYTES, so it admits configurations that LOSE wall-clock | OPTIONAL | `[ ]` | GPU round | Opened 2026-09-07 by the P9 end-to-end arm. MiniMax-M2.7, thinking ON: gate measured acceptance 0.6875 against a 0.6259 break-even and decided `run` — and speculation still cost **−12.9 % total / −16.0 % decode**. **Self-contained brief: [`docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md`](docs/2026-09-07-row13-spec-gate-wallclock-BRIEF.md).** See below. |
| **6a** | The intercept residual the VRAM calibration could not explain | **P5** | `[ ]` | GPU round | One intercept rule cannot fit both forwards: Llama-3.2-1B does NOT sawtooth in whole-prefill data, MiniMax-M2.7 does. Everything is over-predicted (safe), 15–63 % at small S and up to +231 % on MiniMax at a short trailing chunk. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | OPTIONAL | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. Now has a real consumer — the server's largest lever is off for every fp16 model. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | OPTIONAL (parked) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) — **TRIAGED + 2 probes run 2026-09-06** | **P2** (the outlier split) | `[ ]` **2 of 4 closed; 1 docs fix + 1 promoted remain** | CPU sweep | Two closed on new measurement (Plan B Phase 2 refuted directly; fold-`Pi` target removed by Plan I). Real-query gate RUN: **9 bits FAILS on Qwen (0.0649 vs 0.05 bar)** and 9 is the quantizer's ceiling. Outlier split promoted with a measured operating point. See below. |
| **10** | Does the FUSED forward leave MiniMax's P6 win on the table? | **P8** | `[ ]` | GPU round | Opened 2026-09-06 by P6. Qwen has a 56.9 %-hit VRAM cache buying it nothing. Brief: `docs/2026-09-06-P8-fused-predispatch-filter-BRIEF.md`. See below. |
| **11** | The VRAM prefix is sized on the zero-overlap BOUND, not the observed union | **P7** | `[x]` | GPU round | Opened 2026-09-06 by P6. Cheapest open perf row; stacks on P6's +11.9 %. Brief: `docs/2026-09-06-P7-observed-union-prefix-BRIEF.md`. See below. |
| **8** | `tensor_align=True` is production-dead code | OPTIONAL | `[~]` **docstrings FIXED 2026-09-03; branch deliberately kept** | done / rest not scheduled | Opened 2026-09-03 by P6, docs half closed the same day. Nothing calls it since `c5014cd` (2026-07-13), and the stale docstring is what produced row R4/P6 and cost that row its premise. **FOUR false claims corrected** in `safetensor_model_persister.py` — `load_expert_into_slab` (mode advertised as "the prefill stager, via stage_read_expert"), the `tensor_align=False` bullet (claimed prefill never routes through `views_from_packed_copy`; it always does now), `_group_layout`, and **`views_from_packed_copy`, the dangerous one — it promised its output was `_TENSOR_ALIGN`-aligned "by construction", which stopped being true at B1; production sources are raw, so its views are copy-SOURCE-safe only and `gpu_repack_aligned` is what makes them kernel-safe.** The BRANCH stays: `_group_layout`/`expert_span_bytes` size every slab `max(raw_aligned_length, packed_span)` for it, and 5 test sites exercise it — deleting it is a sizing change, not a cleanup, and nothing needs it. Docstrings only, no behaviour touched; suite 1322/3. |

# 10 · [x] Does the FUSED forward leave MiniMax's P6 win on the table?  (§0 **P8**)

**MEASURED 2026-09-07. The row's STATED MECHANISM is REFUTED; its PROPOSED CHANGE survives at a
much smaller, measured ceiling. Results: `docs/2026-09-07-P8-fused-predispatch-filter-RESULTS.md`;
probe `plans/probes/p8_trace.py`, data `plans/probes/p8-qwen-n3.jsonl` (all gitignored — numbers
restated here so the row survives without them).**

Qwen3-235B N=3, 5922 decode layer-visits, one attribution arm (NOT an A/B; its 1.777 tok/s carries
~1 % probe overhead and is not a result).

* **Dispatch-side work is not the prize.** The WHOLE `_dispatch_expert_misses` call costs
  **0.672 s = 0.632 %** of the 106.3 s decode wall; the part attributable to VRAM-RESIDENT experts
  is **0.0226 s = 0.021 %** (3.19 us x 7088 lookups, over-counted on purpose). A pre-dispatch filter
  cannot remove more than 0.021 % of decode. **MiniMax's +11.6 % did not come from dispatch-side
  work.** This also **corrects P6 §2**, which said the module-list filter removes "slab acquire,
  H2D, per-expert attach/detach" from dispatch: dispatch does NONE of those for a resident expert on
  either forward — the H2D + QMatrix skip happens in the COMPUTE loop (`_entry["bc"]`,
  `moe_patch.py:1938`) and **the fused forward already has it**.
* **What the probe found instead, and P1 could not see.** Routed experts that are VRAM-resident AND
  a RAM miss — a real 11.83 MB read for weights already on the GPU. **P1 measured exactly 0 against
  its 0.1 %-hit cache; against P6's working cache this arm measures 1246** = 14.74 GB =
  **2.45 % of the 602.3 GB read**. Ceiling **<= ~1.8 % tok/s**.
* **Root cause, derived — not a model literal.** On a VRAM-prefix layer the two caches have
  different per-layer capacities: VRAM `vram_budget/prefix_layers/expert_bytes` =
  3.925 GB/13/11.83 MB = **25.5 experts**, RAM `max_per_layer` (`vaayu_base.py:330`) =
  `expert_budget/num_moe_layers/expert_bytes` = 16.97 GB/94/11.83 MB = **15**. RAM's cap is charged
  over ALL layers, VRAM's budget is concentrated on the prefix, so **RAM evicts before VRAM on every
  prefix layer**. Two measured consequences: 17.6 % of resident lookups are RAM misses (the 1246),
  and an expert gone from VRAM is already gone from RAM — so **the RAM cache returns a 0.04 % hit
  rate to the non-resident experts that actually need it** (56.4 % on non-prefix layers). On prefix
  layers the RAM expert cache is strictly dominated by the VRAM cache.
* **This ANSWERS P1's blocking objection rather than ignoring it.** P1 warned that hoisting the
  filter stops refreshing the RAM LRU for the hottest experts, so they need a real read "the moment
  they fall out of VRAM". By the inequality above that read already happens. P1's mechanism is
  correct in general and **inert on this geometry** — and the change below **keeps the refresh
  anyway**, so a geometry where RAM's cap exceeds VRAM's stays safe by construction.

**[x] BUILT, MEASURED and SHIPPED 2026-09-07 at +3.21 % on Qwen3-235B.** NOT the brief's
pre-dispatch filter (that drops P1's RAM refresh for a 0.021 % prize). In
`_dispatch_expert_misses`, `get(pin=True)` still runs for every routed expert — **P1's LRU refresh
is preserved by construction, not by argument** — and only the **disk read** is suppressed, on a
miss, for an expert the caller already holds on the GPU (`VRAM_RESIDENT_NO_RAW`). The fused-EXL3
forward hoists its `_vram` gating above the dispatch, holds each routed expert's entry in
`_vram_hold` (eviction only pops the dict, so a held entry stays valid — the module-list forward
already relies on this), passes `set(_vram_hold) or None`, and serves the compute loop from the held
entry instead of a second `_vram.get()` (one LRU promotion per expert per visit, as before). On a
non-cacheable layer the set is empty -> `None` -> **dispatch is byte-identical to before**;
module-list and the non-EXL3 fused forward pass nothing and are untouched.

**Qwen3-235B N=3, interleaved 0,1,0,1, `NTOK=64 BALLAST_GB=25 LOOKAHEAD=0`, greedy, same box same
day: 1.785/1.797 -> 1.849/1.848 tok/s, mean 1.7909 -> 1.8485 = +3.21 %** (worst pairing +2.83 %,
best +3.60 %; p8=0 replicates 0.71 % apart, p8=1 0.04 %). **Bytes 602.2 -> 587.4 GB = -2.44 %,
reads 100 998 -> 98 513 = -2.46 %** — 2485 fewer io_uring reads at ~5.95 MB each is ~1242 experts
against the 1246 the attribution arm counted, so the predicted saving landed exactly. **Greedy ids
byte-identical in all four arms** (`b8e64e5bb17be1d8`). Prefix 13/94 and TTFT (24.66 -> 24.71 s)
unchanged — decode only.

**+3.21 % is LARGER than the <= ~1.8 % the byte saving alone predicts and the excess is NOT
attributed** (a suppressed read also removes a slab acquire, a `put_slab` and the RAM eviction it
forces, and NVMe is not a linear byte->time converter at this concurrency). **Never quote P8 as
"-2.44 % bytes => +3.21 % faster".** The one counter-moving number, stated not buried:
`ExpertRAMCache.misses` rose 38 312/38 337 -> 38 576/38 575 (+0.7 %, 6.47 -> 6.51 per decode visit),
expected and benign — a suppressed expert is never inserted into RAM, so it misses again next visit
and is suppressed again. The gate that matters is READS, and reads fell.

**MiniMax-M2.7 no-regression arms (module-list forward, passes no `vram_resident`, so the code it
runs is unchanged): bytes 378.4 -> 378.3 GB (-0.03 %), greedy ids identical, prefix 34/62
unchanged.** **MiniMax's tok/s reads -0.42 % across 8 arms, and that is ARM ORDERING, not the change** —
proved by reversing one pair: whichever arm runs SECOND is slower, whichever arm it is (0-first
pairs: -0.13/-0.84/-0.90 %; the one 1-first pair: **+0.18 %**). Replicate spreads on the box today
were 0.44-0.65 %, well outside the ledger's stated 0.06-0.18 % band — **treat that band as stale
until re-established.** Qwen's arms also ran 0-first in both pairs, so the same drift worked
AGAINST the treatment there: **+3.21 % is understated, not inflated.**

**SERVER REACHABILITY (CLAUDE.md §8): NOT REACHABLE TODAY — this win is BANKED, not delivered.**
The server runs one request at a time (`engine.py`'s `slot()` under `self._lock`) and
**`batched_generate` is deliberately not wired in** (its own module docstring: N>=2 at long context
peaks 14.17-15.57 GiB of 15.47). P8's prize exists only at N>1: at N=1 the prefix is the load-time
41/94 set, so VRAM per-layer capacity is 3.925/41/11.83 MB = **8.1 experts against RAM's 15** — the
inversion REVERSES and the suppression never fires. **P1 measured exactly that: 0 wasted reads at
N=1 on 5922 visits.** Unblocked by either (a) wiring batched decode into the server, or (b) opting
MoE speculation in (`USE_PROMPT_LOOKUP`), which re-sizes the prefix short enough to recreate the
inversion. **No config knob is needed or wanted** — the change is unconditional and inert where it
does not apply.

Tests `tests/test_moe_patch.py` +5; suite **1720 passed, 12 skipped**. Harness
`plans/probes/p8_trace.py` (attribution), `plans/probes/p8_ab.sh` + a `P8` arm gate in
`probe_fanout.py` (`P8=0` restores the old dispatch; the `_vram.get()` hoist runs in BOTH arms, so
the arm variable is the read suppression alone).

---

## Original row text (the premise above supersedes its table's causal reading)


**Self-contained brief: `docs/2026-09-06-P8-fused-predispatch-filter-BRIEF.md`** (gitignored, this
disk only — the numbers and the stop rule are restated here so the row survives without it).

Opened 2026-09-06 by P6's measurement. P6 gave both on-device MoE models a working VRAM expert
cache at N=3 and only ONE got faster:

| forward | model | hit rate after P6 | routed → dispatch / decode visit | aggregate tok/s |
|---|---|---|---|---|
| module-list | MiniMax-M2.7 | 2.35 % → **48.4 %** | 16.52 → **12.50 (−24 %)** | **+11.6–12.1 %** |
| fused EXL3 | Qwen3-235B | 0.04 % → **56.9 %** | 14.38 → **14.38** | +0.2–0.95 % (noise) |

The module-list forward filters VRAM-resident experts BEFORE dispatch (`_to_load`,
`moe_patch.py:1108`); the fused forward passes the whole routed set to `_dispatch_expert_misses`
(`:1774`) and consults `_vram.get()` only inside the compute loop (`:1822`). That is the entire
difference, and on the model that has it, it is worth 11.6 %.

**NOT a re-open of P1** (REFUTED 2026-09-06, SETTLED). P1 asked whether the redundant lookup cost
DISK READS — measured exactly zero on 5922 visits, and that stands. P1 also priced the lookup
itself at **0.665 µs/call = 0.012 % of decode**, so *do not build this to remove the lookup*. P1's
warning binds: the dispatch's RAM lookup is what keeps VRAM ⊆ RAM true, and hoisting the filter
would make the hottest experts the RAM LRU's first eviction candidates. What is new is only that P1
measured against a 0.1 %-hit fused cache and it now hits 56.9 %.

**First step is a MEASUREMENT, not a change:** with a working cache, what does
`_dispatch_expert_misses` actually do for an already-VRAM-resident expert — slab acquire, read
submit, H2D, or a cheap short-circuit? Copy `plans/probes/p1_trace.py`. **If it short-circuits
cheaply the row is REFUTED at zero cost.**

**Done means:** on-device A/B on Qwen3-235B at N=3, aggregate tok/s + bytes, greedy ids identical,
AND RAM misses/decode-visit not rising from the 6.47–6.48 measured under P6. Baseline to beat:
**1.791 / 1.805 tok/s**, 600.2–600.3 GB, prefix 14/94, hit rate 56.85–56.87 %.
**Refuted if** the dispatch does no real work for resident experts, or the filter cannot keep the
RAM refresh, or tok/s does not move.

---

# 11 · [x] **REFUTED 2026-09-06** — the VRAM prefix's bound IS the reuse headroom  (§0 **P7**)

**Self-contained brief: `docs/2026-09-06-P7-observed-union-prefix-BRIEF.md`** (gitignored, this disk
only — key numbers restated here).

Opened 2026-09-06 by P6, which shipped `batch_union_per_layer = min(num_experts, N*top_k)` — a
zero-overlap upper bound. The real union is smaller:

| model | bound at N=3 | measured union/visit | prefix now | prefix at measured |
|---|---|---|---|---|
| Qwen3-235B | 24 | **14.38** (1.80 × top_k) | 14 of 94 | ~24 of 94 |
| MiniMax-M2.7 | 24 | **16.51** (2.06 × top_k) | 34 of 62 | ~51 of 62 |

A short prefix is a real cost — layers outside it get no VRAM caching at all. **MiniMax-only
question:** on the fused forward a working cache buys nothing (see row 10), so a longer prefix
cannot help Qwen; run a Qwen arm as a no-regression check only.

**Do not rebuild the machinery:** `observed_union_per_layer` and `union_bytes`
(`vaayu/speculative/adaptive_gate.py`) already compute this from `_expert_history`, which all three
forwards write (`moe_patch.py:1281`, `:1419`, `:2028`) and which at batched decode ALREADY holds the
batch union. The missing piece is only a calibration window in `batched_generate` (P3's
`AdaptiveSpecGate` is the precedent; `CALIBRATE_STEPS = 8`).

**Trap that most threatens this row:** MiniMax's 16.51 is a POST-filter count, taken at a 2.4 % hit
rate where the filter removed almost nothing. **After P6 the hit rate is 48 %, so a fresh
measurement reads LOWER than the true union** — sizing on it would lengthen the prefix too far and
re-create the thrash P6 fixed. Measure pre-filter routing.

**Done means:** prefix derived from observed routing; on-device A/B on MiniMax-M2.7 at N=3 vs the
P6 build (beat **3.280 / 3.265 tok/s** at 48.4 % hits, 17 539 evictions, 376.5 GB), greedy ids
identical. **Refuted if** the longer prefix does not beat the bound-sized one.

**BUILT AND MEASURED 2026-09-06 — REFUTED on the target model, and the row's PREMISE is what
failed, not the implementation.** The calibrated prefix was built exactly as specified
(`calibrated_union_per_layer` + an 8-step window in `batched_generate`, 16 tests, suite 1716 passed
/ 12 skipped) and measured on MiniMax-M2.7 at N=3, interleaved 0,1,0,1, greedy ids byte-identical
across all four arms:

| arm | prefix 1st→last | hit rate | evictions | tok/s | vs P6 |
|---|---|---|---|---|---|
| p7=0 (shipped P6) | 34→34/62 | 48.45 / 48.44 % | 17 533 | 3.275 / 3.261 | — |
| p7=1 (calibrated) | 34→**ALL**/62 | **2.38 %** | **61 278** | 2.913 / 2.891 | **−10.85 / −11.54 %** |

P7 hands back P6's entire win: 2.913/2.891 IS the pre-P6 number (2.940/2.910).

**Root cause, and the number that must not be re-derived.** A `UNION_TRACE=1` arm recorded the
PRE-filter routing set of every MoE layer at every decode step
(`plans/probes/probe_fanout.py`, analysed by `plans/probes/p7_union_analysis.py`). Two findings:

1. **The union RISES across the decode: 10.47 → 20.55 per layer over 64 steps** (whole-decode mean
   16.91, which is the brief's 16.51 — the brief was RIGHT). An 8-step calibration window measures
   **10.47** and under-counts steady state by 2.0×. Any fixed early window has this problem.
2. **The bound's over-charge IS the reuse headroom, not slack.** `compute_vram_layer_range` sets
   capacity/layer = the charged load, so charging the TRUE union leaves capacity ≈ union and an
   entry cannot survive to its own reuse. Measured, at budget 7.341 GB / `expert_bytes` 8.875 MB:

   | charged | layers | capacity/layer | ÷ steady-state union 20.55 | measured |
   |---|---|---|---|---|
   | 24 (P6 bound) | 34 | 24.34 | **1.18×** | 48.4 % hits |
   | 11 (8-step calibration) | 62 | 13.34 | **0.65×** | 2.38 % hits |
   | 17 (whole-decode union) | 48 | 17.1 | 0.83× | below one step's routing |
   | 21 (steady-state union) | 39 | 21.2 | 1.03× | ~zero headroom |

   Independently: the union over **k=2** consecutive steps is **26.38**, already ABOVE the bound of
   24. P6's capacity of 24.34 sits between the k=1 union (20.55) and the k=2 union (26.38) — an
   expert recurring at distance 1 survives, longer distances miss. That IS the 48.4 % hit rate.

**The sweep proves the optimum is where P6 put it.** `UNION_FORCE=<n>` charges a fixed per-layer
load and lets `compute_vram_layer_range` turn it into a prefix (`plans/probes/p7_sweep.sh`,
`p7sweep_report.py`); two arms each, interleaved, greedy ids identical across all six:

| charge | prefix | hit rate | evictions | tok/s (arms) | mean | vs 24 |
|---|---|---|---|---|---|---|
| **24 (P6 bound)** | 34/62 | 48.45 % | 17 533 | 3.269 / 3.245 | 3.257 | +0.00 % |
| 21 | 39/62 | 45.34 % | 21 367 | 3.277 / 3.258 | 3.268 | **+0.34 %** |
| 18 | 46/62 | 26.21 % | 34 224 | 3.132 / 3.151 | 3.141 | **−3.54 %** |
| ~13.3 (ALL) | 62/62 | 2.38 % | 61 278 | 2.913 / 2.891 | 2.902 | **−10.9 %** |

+0.34 % at 21 is inside noise (arm spread 0.6 %; P6's own baseline pair spread 1.0 %). Flat from 24
to 21, monotone collapse below. **No prefix length beats P6's bound.**

**Do not re-open this row to "use a longer calibration window", "measure the union better", or
"charge the whole-decode mean instead."** A perfectly-measured union is the wrong quantity: sizing
capacity AT the working set removes the headroom the LRU needs.

**The code was REVERTED** — refuted work does not ship (CLAUDE.md §2/§3), suite back to the
recorded **1700 passed, 12 skipped**. The 20 lines are described above; do not rebuild them. Qwen
arms were deliberately not spent: the target regressed 11 %, and per P6 a Qwen no-regression check
protects nothing. Full detail: `docs/2026-09-06-P7-observed-union-prefix-RESULTS.md` (gitignored,
this disk only — everything load-bearing is restated here).

**One thing this row did NOT refute, and it is a different question.**
`compute_vram_layer_range` charges capacity = insert load with no reuse-distance term, and works
only because a zero-overlap bound happens to supply ~1.2× headroom. **Nothing derives that 1.2×.**
A geometry whose bound/union ratio differs has no reason to land in the right place. That is a gap
in the sizing contract, not a prefix-length question — open it on its own evidence, never by
re-running P7.

---

# 13 · [bug] The spec gate can say `run` on a configuration that loses wall-clock

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

# 14 · [ ] The server reaches 14 of the engine's 28 load parameters  (**A**)

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

---

# 18 · [bug] `output_hidden_states=True` crashes the forward

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
# 19 · [ ] Re-measure any drafter number taken under `sdpa` (fallout of row 16)

Row 16 left this consequence OPEN and it must not be lost inside a closed row.
`models/Llama-3.2-1B-Instruct` is the README's suggested speculative drafter and the dense row's
fast checkpoint. Before row 16's fix, ANY load whose `pinning_budget` reached 0 left the tied
`lm_head` on the meta device and drafted **uniform noise** — which would collapse acceptance
without any error. `sdpa` is what drove the budget to 0 on that checkpoint (measured: sdpa+auto 0,
flash+auto 10.97 GB).

**Task:** establish which recorded drafter/speculative numbers were taken under `sdpa`, and
re-measure those. **CLAUDE.md §0 cites 5.57x for dense speculative decoding with the 1B drafter** —
check that one first. If every such arm ran flash, the numbers stand: **say so explicitly and close
the row**, because "probably fine" is what this ledger exists to prevent.

Note row **17** interacts: a drafter prompt over `prefill_chunk_size` under sdpa is ALSO garbage,
independently of row 16. Fix 17 before re-measuring anything under sdpa.

---
# 17 · [bug] `sdpa` + CHUNKED PREFILL = garbage on MoE — BLOCKS row 15

Found 2026-09-07 while pricing row 15's real cost. **PRE-EXISTING: reproduced byte-identical on
`881aaaa`, before any of this branch's code changes** (verified in a clean `git worktree`).

**MiniMax-M2.7, `generate()`, `attn_implementation="sdpa"`, `prefill_chunk_size=512` held fixed:**

| prompt | chunks | output |
|---|---|---|
| 325 tok | 1 | coherent — *"The user is asking me to refactor the watcher code..."* |
| **709 tok** | **2** | **24 null bytes** |
| 4165 tok | 9 | 64 null bytes |

**Breaks as soon as there is more than one chunk.** `forward` (`vaayu_base.py:~4077`) sets
`attention_mask = None` for BOTH `flash_attention_2` and `sdpa`. With no mask, sdpa's implicit
causal masking is **top-left aligned**, which is wrong whenever `q_len < kv_len` — i.e. every
prefill chunk after the first, which carries past KV. Flash aligns bottom-right and survives.
Chunk 0 (`q_len == kv_len`) and decode (`q_len == 1`) are both correct, which is why this hid.

**NO recorded measurement is invalidated.** `batched_generate` builds an EXPLICIT 4D mask
(`build_batch_attention_mask`) and does its own single-forward prefill, so every P6/P7/P8/P10 sdpa
arm is unaffected; and the server ships `ATTN_IMPLEMENTATION = None` -> flash, so no user is
affected today. The exposure is exactly the combination row 15 needs.

**Why it BLOCKS row 15:** `batched_generate` hard-asserts `sdpa` (`vaayu_base.py:5194`), so wiring
batching means switching the server's kernel — and the server's single-stream path is
`generate()`/`stream_generate`, which chunks. **Switching the server to sdpa today would corrupt
every single-stream request with a prompt over `PREFILL_CHUNK_SIZE`.** Fix this first.

**Proposed fix (not built):** build an explicit causal mask for `sdpa` whenever `past_len > 0`,
instead of passing `None` — and LEAVE flash on the `None` path (it is correct there, and an explicit
mask would re-introduce the S-sized score buffer `SCORE_ELEM_BYTES` prices at 0.0 for it). Size the
mask off `past_len + q_len`, never `max_seq_len` — the sibling branch at `:4112` records a
268 MB / 12.9 GB regression from exactly that mistake. **Minimum reproducer is TWO chunks** (709
tokens at `prefill_chunk_size=512`); a single-chunk test passes on the broken code. Both MoE
geometries plus a dense check, since it touches every prefill.

**SELF-CONTAINED BRIEF, written to be picked up cold:**
[`docs/2026-09-07-row17-sdpa-chunked-prefill-BRIEF.md`](docs/2026-09-07-row17-sdpa-chunked-prefill-BRIEF.md)
(gitignored, this disk only — this row restates everything load-bearing).
Repro: `MODEL=minimax ATTN=sdpa PROMPT_TOK=700 NTOK=24 CHUNK=512 plans/probes/p10c_attn_cost.py`,
control `ATTN=flash_attention_2`.

**Also measured while pricing row 15 (MiniMax, 4165-token prompt), and it belongs to that row's
decision:** switching flash -> sdpa costs real VRAM before batching pays anything —
`reserve_bytes` **3.758 -> 5.146 GB**, `pinning_budget` **10.201 -> 8.814 GB**, expert VRAM cache
budget **8.929 -> 7.542 GB (-15.5 %)**. Cache size sets decode throughput (P6), so **row 15's
honest comparison is flash@N=1 (what users get today) vs sdpa@N=2 — NOT the sdpa-vs-sdpa +18.0 %
P10a measured.** That comparison does not exist yet and cannot be taken until this bug is fixed.

---
# 16 · [x] FIXED 2026-09-07 — tied lm_head never materialized on the PREFETCH path

Found 2026-09-07 by P10b's discriminator arm, which needed a dense model and could not get a sane
one. **Not a batching bug and not a dense bug -- an INTERSECTION.** Measured on
`models/Llama-3.2-1B-Instruct`, 12 greedy tokens, "Name three primary colors, then stop.":

| attn | `kv_cache_mode` | cache class | output |
|---|---|---|---|
| `flash_attention_2` | auto | `DynamicCache` | `'Red, blue, and yellow.'` |
| `flash_attention_2` | bf16 / fp8 | `FP8DynamicCache` | `'Red, blue, and yellow.'` |
| **`sdpa`** | **auto** | **`DynamicCache`** | **`' القرنuthor matching overallůjnavigate_ADDRESS...'`** |
| `sdpa` | bf16 / fp8 | `FP8DynamicCache` | `'Red, blue, and yellow.'` |

**`auto` resolves to `fp16` -> plain `DynamicCache` for every non-EXL3 checkpoint**, so that is the
pair a non-EXL3 model gets by DEFAULT. **Llama-3.3-70B-exl3 under `sdpa` is CORRECT**
(`'The three primary colors are:\n1. Red\n2.'`), so it is neither "sdpa is broken" nor "dense is
broken". Stock transformers on the same 1B checkpoint is correct, so it is not the checkpoint:
`'Here's a Python function that merges two sorted lists...'`.

**Why it matters even though no on-device target is non-EXL3:**
- **`batched_generate` hard-asserts `sdpa`** (`vaayu_base.py:5194`), so batched decode on ANY
  non-EXL3 model is in the broken pair by construction -- and P10's server wiring is batched decode.
- **`models/Llama-3.2-1B-Instruct` is the README's suggested speculative-decoding drafter** and the
  dense row's fast checkpoint. A drafter in the broken pair drafts garbage and acceptance collapses.
  **Check what the speculative path forces before trusting any drafter number taken under `sdpa`** --
  including the 5.57x dense speculative figure CLAUDE.md §0 cites.
- It is the same class as OPTIONAL row **3** (the prefix pool cannot use `DynamicCache` either), and
  it is exactly the hole CLAUDE.md §0 names: the dense gate `test_batched_dense_smoke.py` was
  deleted 2026-09-03 with **no committed replacement**, so nothing would have caught this.

**ROOT-CAUSED AND FIXED the same day, and the pair above was a SYMPTOM, not the cause.** Neither
`sdpa` nor `DynamicCache` is at fault. **Llama-3.2-1B has `tie_word_embeddings=True`, so the
checkpoint ships NO lm_head tensor** and Vaayu's splitter writes a valid-but-EMPTY shard for it
(bug **D1**). D1's fallback existed in **two** places — `_pin_shared_layers` (`:3254`) and
`load_layer_to_cpu` (`:3656`) — but **NOT in `_load_from_strategy` (`vaayu_base.py:3568`), the
PREFETCH route the forward uses for every non-pinned layer.** It came back `{}`, and
`lm_head.weight` was left **on the meta device**.

**Why `sdpa` looked like the trigger:** sdpa charges a score-buffer term flash does not
(`SCORE_ELEM_BYTES` 10.9 vs 0.0), which inflates `reserve_bytes` until **`pinning_budget` is
EXACTLY 0** — so nothing pins, and the head streams through the one unhandled path. Measured
`pinning_budget`: sdpa+auto **0** (garbage), sdpa+bf16 1.16 GB (correct), flash+auto 10.97 GB
(correct). **The head is normally pinned, which is why this hid for so long.**

**Evidence chain, all measured:** all 16 decoder layers bit-identical across the three configs
(final hidden norm 444.9 everywhere) → the body was never wrong; logits std **0.02** vs a correct
**2.21** → uniform noise; `lm_head.weight.is_meta == True` in the failing config against
`cuda:0, norm 334.577` in both working ones; `[VAAYU-IO] iouring 0.0MB lm_head`.

**Fix:** `_load_from_strategy` now applies the same `should_use_tied_lm_head` +
`rekey_embed_as_lm_head` fallback its two siblings already had. **Verified on device:** all 6
`attn × kv_cache_mode` cells on the 1B now emit `'Red, blue, and yellow.'`, and dense batched decode
is **token-identical at N=1 vs N=2 with both rows agreeing** — which independently confirms row 15's
verdict that the batched path is arithmetically sound and the MoE N=1/N=2 difference is routing
discontinuity alone. Tests: 4 in `tests/test_tied_lm_head.py` (2 were red before the fix). Suite
**1748 passed, 12 skipped**.

**Consequence still OPEN, not verified here:** `models/Llama-3.2-1B-Instruct` is the README's
suggested speculative drafter, and any drafter run that hit `pinning_budget == 0` before today was
drafting from a meta head. **Any drafter acceptance number taken under `sdpa` predates the fix and
should be re-measured** — including the 5.57x dense speculative figure CLAUDE.md §0 cites.
`docs/2026-09-07-P10b-ids-divergence-SETTLED.md` §4.

---
# 15 · [!] Wire batched decode into the SERVER  (§0 **P10**) — BLOCKED on row 17

**[!] BLOCKED 2026-09-07 on row 17** (`sdpa` + chunked prefill = garbage on MoE).
`batched_generate` hard-asserts `sdpa`; the server runs flash and its single-stream path chunks, so
switching the kernel today would corrupt every single-stream request with a prompt over
`PREFILL_CHUNK_SIZE`. **Unblocked by:** fixing row 17.

**Row 17 also carries the other half of this row's economics:** flash -> sdpa costs the expert VRAM
cache **-15.5 %** (8.929 -> 7.542 GB on MiniMax) before batching pays anything, so this row's real
comparison is **flash@N=1 vs sdpa@N=2**, not P10a's sdpa-vs-sdpa +18.0 %. Take that A/B before
building the scheduler; **it can still refute the row.**

**Design note already established (2026-09-07): NO arrival window is needed in v1.** `QUEUE_DEPTH`
is 8 and `slot()` already queues waiters, so a batch can be formed from requests ALREADY waiting —
zero added TTFT for anyone. An arrival window is a v2 question, not a v1 requirement.

**Self-contained brief: [`docs/2026-09-07-B-server-batched-decode-BRIEF.md`](docs/2026-09-07-B-server-batched-decode-BRIEF.md)**
(gitignored, this disk only — restated here). Opened 2026-09-07 by the CLAUDE.md §8 audit.

**P6 (+11.6-12.1 % MiniMax) and P8 (+3.21 % Qwen) are measured, shipped, green, and deliver ZERO to
users.** Both effects exist only at N>1; the server runs one request at a time (`slot()` holds
`self._lock` around the whole generation, `engine.py:185`) and its module docstring says
`batched_generate` "is deliberately not wired in". P8's N=1 inertness is measured, not assumed: at
N=1 the prefix is the load-time 41/94 set → VRAM per-layer capacity 8.1 experts against RAM's 15,
so the inversion P8 exploits REVERSES, and **P1 counted 0** of the reads P8 removes on 5922 N=1
visits.

**Read the refusal correctly before trying to defeat it.** `engine.py` cites a real number — N>=2 at
LONG context peaked **14.17-15.57 GiB of 15.47** (OPTIONAL row 4). It is a long-context refusal, not
an absolute one: **every P6/P7/P8/P1 probe ran N=3 successfully at ~690-token prompts.** And
admission for exactly this already exists — `plan_batch` (`batch.py:333`) returns
**`(admitted, deferred)`**, and `batched_generate` re-checks at the real width
(`_assert_context_fits(..., N=N)`, `vaayu_base.py:5195`). **The work is replacing an unconditional
refusal with the admission control that already exists, falling back to today's serialized path for
whatever it defers; a batch of 1 must be byte-identical to today.**

**[x] THE REFUTING MEASUREMENT RAN 2026-09-07. THE ROW PASSED — and it is worth an order of
magnitude more than this row claimed.** Full write-up:
[`docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md`](docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md);
probe `plans/probes/p10_admit.py`. Arms: ~4.15k-token prompts (system + synthetic file context +
task), `max_new_tokens=512` charged to KV, 16 tokens consumed then `gen.close()`, `BALLAST_GB=27`,
sdpa, greedy, one `systemd-run` scope each.

**Two gates, not one — and the brief named only the weaker.** `plan_batch` prices KV alone and
**admitted N=3 with zero deferrals on BOTH models**. `_assert_context_fits(n_tokens, N=N)`, which
`batched_generate` itself calls at the real width (`vaayu_base.py:5195`), prices the sdpa score
transient and **RAISED on every N>1 arm**. A probe checking `plan_batch` alone would have reported
"admits N=3" and then OOM'd in the prefill.

| model | N | `plan_batch` | gate 2 | forced run | peak VRAM | TTFT | aggregate tok/s |
|---|---|---|---|---|---|---|---|
| MiniMax-M2.7 | 1 | 1/0 | **PASS** | ok | 12.972 GB | 205.12 s | 1.634 |
| MiniMax-M2.7 | 2 | 2/0 | RAISE (max 3331) | **ok** | 15.157 GB | 228.73 s | **2.180** (+33.4 %) |
| MiniMax-M2.7 | 3 | 3/0 | RAISE (max 2048) | **OOM** `model.layers.22`, 14.93 of 15.47 GiB | — | — | — |
| Qwen3-235B | 1 | 1/0 | **PASS** | ok | 11.662 GB | 214.25 s | 0.927 |
| Qwen3-235B | 2 | 2/0 | RAISE (max 4096) | **ok** | 13.367 GB | 214.22 s | **1.582** (+70.7 %) |
| Qwen3-235B | 3 | 3/0 | RAISE (max 2048) | **ok** | 15.084 GB | 260.11 s | 1.278 (+37.9 %) |

**End-to-end (the number the row exists for): 1.77x / 1.98x / 2.34x at the measured 16 tokens;
1.48x / 1.78x / 1.57x projected at 512 tokens generated** (MiniMax N=2, Qwen N=2, Qwen N=3). **The
win is mostly PREFILL, not decode.** Qwen prefilled two 4.18k-token streams in **214.22 s against
214.25 s for one** — free within 0.02 %; MiniMax paid +11.5 %. Prefill at this context is dominated
by streaming weights off NVMe, which N streams share. That is CLAUDE.md §0's fan-out argument,
measured.

**[x] PREREQUISITE DONE 2026-09-07 — the giveback (P10a), BUILT, MEASURED, GREEN, BANKED.**
`admission_headroom` now takes `expert_giveback_bytes`, `batch.expert_giveback_bytes(model)` prices
the cache's ENTITLEMENT above one decode step's working set (budget, NOT usage -- usage is 0 on a
fresh load while the entitlements are 6.774 / 3.644 GiB), `_assert_context_fits` returns the
predicted peak so the caller charges itself what admission spent, and `batched_generate` at N>1
lowers `budget_bytes` by that amount for the batch and restores it in the existing `finally`
(GeneratorExit included). **N=1 is untouched.** **MiniMax-M2.7 at 4.15k, 64 tok/stream: N=1 1.750 ->
N=2 2.065 tok/s = +18.0 %**, peak VRAM 12.355 GB, budget 7.28 -> 4.48 GB, P6 prefix 62 -> 31/62.
Keeping the promise costs **3.3 %** against the cap-disabled 2.136 (+22.1 %) and returns **2.8 GB of
peak VRAM**; greedy ids byte-identical between capped and uncapped N=2. **N=3 still refused, and
correctly -- forced, it OOMs at `model.layers.22`.** **Qwen: giveback 0.00 GB, every decision
unchanged** -- its cache budget (3.644 GiB) is already below its working set (8.285), so it has
nothing to surrender; its false refusals are need-side over-prediction, untouched here. Suite **1744
passed, 12 skipped**. §8: **no knob needed** (derived from geometry); **NOT reachable** -- the server
still runs one request at a time, so this is worth 0 to users until the wiring below lands.
`docs/2026-09-07-P10a-expert-giveback-admission-RESULTS.md`; tests
`tests/test_batch_expert_giveback.py` + 11 in `tests/test_vram_plan.py`.

**[x] The greedy-id divergence is SETTLED 2026-09-07: BENIGN on MoE, and the CRITERION was wrong.**
`docs/2026-09-07-P10b-ids-divergence-SETTLED.md`. Nobody had decoded the tokens. MiniMax stream 0,
same prompt: N=1 *"in the provided **file** context"* vs N=2 *"in the provided context"* -- **one
word**, both coherent, a near-tie at position 42 resolved differently. Mechanism is the one
`tests/test_batch_prefix_equivalence.py` documented on 2026-08-15 for the cold/warm prefill pair:
batch width changes GEMM tiling, tiling perturbs router logits by ~1 ULP, and **top-k selection is
DISCONTINUOUS**. All properties match -- deterministic per width (3/3 N=1 arms and 4/4 N=2 arms
bitwise identical, across processes), late divergence, coherent both sides. Padding was excluded by
construction (the probe uses identical equal-length prompts, zero `max_R` padding).
**Row 15's criterion "greedy ids identical to the single-stream path" is UNACHIEVABLE on MoE and must
be replaced** (asserting it asserts a bug that cannot be fixed) with the three the prefix-equivalence
work already uses: deterministic per configuration; coherent as TEXT, not ids; bitwise identity only
within a route pair that does not change tiling (P10a's capped-vs-uncapped N=2 is one, and it held).
Dense may still assert token-identity -- once the bug below is fixed.

**What the gate-2 fix was, for anyone reading this cold:** Its own message says why — *"the expert
VRAM cache is entitled to everything above the reserve and fills during prefill, so only the reserve
is actually available here"* — so `admission_headroom` hands the request the reserve (4.79 GiB
MiniMax / 5.60 GiB Qwen) and refuses against it while 11.57 / 9.26 GiB is free. **But
`batched_generate` reclaims that cache to its working-set floor immediately AFTER the assert**
("Plan K", `vaayu_base.py:~5200`), so the assert prices a configuration the batch never runs in.
Forced past it, 3 of the 4 N>1 arms ran fine. **Gate 2 must price the POST-reclaim configuration**
(or the two gates must be reconciled into one) BEFORE the server work, or admission defers 100 % of
the time and the feature is decoration.

**Do NOT delete gate 2 and do NOT replace it with `plan_batch`:** gate 2 was RIGHT once — MiniMax
N=3 genuinely OOM'd — and `plan_batch` admitted exactly that arm. **And do not "fix" Qwen by
lowering the giveback floor:** its cache cannot hold one decode step's working set as it is.

**Trap for the next arm:** in `plans/probes/p10_admit.py`, `FORCE_ADMIT=1` replaces the guard
wholesale, so it returns no `need` and `batched_generate` takes NO giveback — a FORCE_ADMIT arm
measures the PRE-P10a configuration. `FORCE_N` alone caps the width on the shipped path. Do not mix
them and call the result shipped.

**Max width is model- and context-dependent, and it is not 3.** MiniMax takes 2 here; Qwen takes 3
but is FASTER at 2 (1.582 vs 1.278 aggregate — bigger KV starves the expert cache). Any
`MAX_BATCH_SIZE` is a ceiling, never a target.

**Caveats before quoting any of this:** every forced arm ran with a COLD expert VRAM cache
(`reclaimable_expert=0.000 GB`, fresh load) — the warm-cache margin is UNMEASURED and the tightest
arm had ~1.3 GiB of slack; 16 consumed tokens is a short decode window (ranks widths, not a
P6-grade A/B); TTFT arms were not first-touch-controlled.

**What the build actually is** (engine side exists; this is server work): batch formation with a
short arrival window, one worker driving `batched_generate` fanning `(stream_id, token_id)` to N
bridges; **the generator contract is the #1 risk** — its docstring says an abandoned generator
leaves `_batch_decode_active`/`_batch_n` set until GC, and **a client disconnect is exactly that**;
`batched_generate` asserts `sdpa`, so a `flash_attention_2` operator must get a fallback, never a
live assert; `BatchRequest.stop_ids` is token ids while `stopping.py` works in strings; **per-session
logging breaks under N>1 and `docs/issue-L1-triage-logging.md` predicted it and says what to do**;
`MAX_BATCH_SIZE = 1` knob in `config.py` in the same commit (§8). Per-stream `temperature`/`top_p`
already exist on `BatchRequest`.

**Done means** gate 2 prices the post-reclaim configuration (above) AND the server serves N
concurrent requests when admission allows and falls back otherwise; **a server-level on-device A/B** (N=2 and N=3 concurrent clients vs today's serialized
path at the same offered load — an engine probe number does NOT discharge this); greedy ids
identical to single-stream; no OOM at the real derived context; a disconnect mid-batch leaves the
engine usable, proven by a test that abandons the generator; suite green; L1 session attribution
still correct. **Refuted if** the server-level A/B shows no aggregate gain. (The other refutation
condition — "admission defers to N=1 at realistic context" — was TESTED 2026-09-07 and did NOT
fire: `plan_batch` admits N=3 on both models.)

**Trap, measured 2026-09-07:** interleave A/B arms BOTH ways. On MiniMax, whichever arm ran SECOND
was 0.4-0.9 % slower and replicate spreads were 0.44-0.65 % — not the 0.06-0.18 % the Environment
section still claims. A one-way 0,1,0,1 interleave manufactures a false ~0.5 % effect.

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

---

# 6a · [ ] The intercept residual the VRAM calibration could not explain  (§0 **P5**)

**Self-contained brief: `docs/2026-09-06-P5-intercept-residual-BRIEF.md`** (gitignored, this disk
only). Opened 2026-08-15 out of the kernel-aware-calibration row (archived; `plans/settled_and_method_notes.md`), with the data already in hand
(`plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md` §7).

**One intercept rule cannot be right for both forwards.** `transient − KV(capacity)`, flash arm:

| geometry | full trailing chunk | 1-token trailing chunk | charged now |
|---|---|---|---|
| llama-3.2-1b | 0.792 G | **0.792 G** (identical) | 0.788 ✅ |
| llama-3.3-70b | 0.928 G | 0.714 G | 1.051 (over) |
| minimax-m2.7 | 0.62–0.67 G | 0.082–0.089 G | 0.871 (over) |
| qwen3-235b | ≈ residency | ≈ residency | residency ✅ |

Llama-3.2-1B does **not** sawtooth in whole-prefill data; MiniMax-M2.7 does (0.3415 G at S=513
against 0.8690 G at S=512 — real, and consistent across all three of its `+1` points). Everything
is over-predicted (safe): 15–63 % at small S, up to +231 % on MiniMax at a short trailing chunk,
and on sdpa the score term and the intercept are summed although they do not fully coexist
(charging `max()` instead was tested and is worse in the DANGEROUS direction: −38 % on Qwen at
S=4096).

**The decisive experiment**, already named in that row's old brief: hold S fixed at one KV-capacity
plateau and vary only trailing-chunk length, plus per-chunk allocation instrumentation to find
which chunk sets the peak in the module-list forward. GPU round, own row.

**Stop rule:** if the per-forward difference turns out not to be derivable from geometry
(CLAUDE.md §5 forbids a per-model literal), say so and close the row — the current form is already
safe, just loose.

---

# 3 · [ ] The cross-turn prefix cache only works for EXL3 models

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

---

# 4 · [ ] 2× KV overshoot, ~0.83 GB at Qwen's derived operating context — PARKED

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

# 5 · [ ] DEFERRED write-ups — TRIAGED 2026-09-06, two probes run, two items left

Row 5 was never one row. Full evidence:
**[`docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md`](docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md)**
(**gitignored — this disk only**; the numbers below are the durable copy, per §7).

Method: one on-device capture per model at a realistic ~4.15k-token agentic context (re-sent
source + a codegen task), 64-token greedy decode, `MemoryMax=29G`. One run fed both probes —
real post-RoPE decode queries + stored-layout `K` for 7 quantile-sampled layers, and the ordered
decode routing trace. **Harness validated against Plan I's own published figure**: the reproduced
K-row proxy gives 0.0307 at S=694 against Plan I's 0.0372 for Qwen at 9 bits.

**[x] Plan B Tasks 8, 9 — Phase 2 live expert re-pin. REFUTED DIRECTLY 2026-09-06 → SETTLED.**
Four arms on one decode trace, all paying the same price for the pin, one cache shared across
window boundaries: `none` / `static` (Phase 1) / **`causal` = pin from the PREVIOUS window, which
IS Phase 2** / `oracle` = pin from the window being replayed (unrealizable). At every
non-degenerate budget on both models, **causal − none is between −0.082 and +0.011 hit-rate
points — negative more often than positive — while oracle − causal is +0.004 to +0.330.** Phase 2's
entire value sits in a predictor that recent heat does not approach. This no longer rests on Plan
F's static-sweep inheritance. **Do not re-open on "a better predictor"**: the oracle bound is
exactly what a better predictor is worth. The short horizon here (~4–8-token windows) *flatters*
recency; Phase 2's real between-turn horizon can only do worse.
**TRAP, recorded so this is not "revived" on an artifact:** at Qwen's 8.50/8.90 GB budgets
`derive_max_per_layer` returns exactly `top_k`, the LRU holds one token's routing per layer and
thrashes (`none` = 0.0000 at 8.50 GB), so **any** pin appears to win by up to +0.27. A sweep run
only at Qwen's ~8.5 GB operating budget would have concluded the opposite.

**[x] Plan H follow-up — fold `Π` into the query. CLOSED 2026-09-06, target gone → SETTLED.**
It existed to remove the 5–8 % MiniMax decode regression from compress-on-READ. **Plan I removed
that regression** by moving compression to the eviction boundary (Qwen 0.86→0.86, MiniMax
1.53→1.56). Its only remaining consumer is the dark `kv_cache_mode="turboquant"` mode that Plan H
§9 already says not to use.

**[x] Plan H follow-up — real-query quality gate. RUN 2026-09-06, docs corrected same day.**
Pre-stated bar unchanged (score rel-L2 < 0.05), Plan H/I's own statistic (whole `[H,S]` matrix per
layer). Worst sampled layer at ~4.16k tokens, real decode queries:

| model | forward | 9 bits | 8 bits | K-row proxy @9 | Gaussian @9 |
|---|---|---|---|---|---|
| Qwen3-235B | fused EXL3 | **0.0649 FAIL** | 0.0934 FAIL | 0.0413 pass | 0.0878 fail |
| MiniMax-M2.7 | module-list | **0.0426 PASS** | 0.0546 FAIL | 0.0413 pass | 0.0631 fail |

**Plan I's 8-vs-9 decision is CONFIRMED** (8 fails both models on real queries too). **Its 9-bit
certification is not.** The K-row proxy is **2.5–13.7× optimistic on Qwen** and only ~2× on
MiniMax — real queries fail Qwen at *every* S from 256 to 4096, so this is not a long-context
effect Plan I's short prompt missed, it is **a proxy validated on one forward that does not
transfer to the other** (§5, again). **`bits_total=9` is the quantizer's CEILING**, not a chosen
default — `turboquant.make_codebook` raises above `b_mse=8`, so there is no extra bit to spend.
**Nothing shipped is broken:** `PREFIX_COMPRESS = None` (`vaayu/server/config.py:109`), and no
size/TTFT/eviction number changes. **Docs corrected 2026-09-06** at all five sites that stated the
floor without its provenance: `vaayu/kv_compress.py`'s `_DEFAULT_BITS` comment,
`vaayu/server/README.md`, `documentation/server.md` (table + §6), and `documentation/TUNING.md` §2
(which carries the full table and the guidance). The 8-vs-9 advice is unchanged — 8 fails both
models on real queries too.
**[ ] STILL OPEN, deliberately unscheduled: does failing this bar degrade generated text?** The bar
is a pre-stated *proxy* for quality; Plan H §6 warns coherent output hides large perturbations (a
256-token coherence check passed even at the far worse 4-bit setting). Certifying that direction is
a separate experiment and **nothing here claims it**. Needs a real quality measure (perplexity on a
prefix-reused turn, or a task score), not another score-error number.

**[x] Mixed-precision outlier split — BUILT AND SHIPPED 2026-09-06 (§0 P2). → SETTLED.**
`vaayu/kv_compress.py` keeps each layer's **8 highest-variance channels** in the cache dtype and
packs the remaining 120 at 9 bits, on by default whenever `prefix_compress` is set. Measured
**through `compress_cache`/`decompress_cache`**, worst sampled layer, real decode queries at
~4.16k tokens: **Qwen3-235B 0.0683 FAIL → 0.0281 PASS**, **MiniMax-M2.7 0.0428 → 0.0263**, at
**1.211 B/elem, 0.605× bf16** against uniform 9-bit's 1.156 (0.578×) that does not pass — ~11 % of
the saving to buy the gate. Pool capacity therefore goes 5 → **9** sessions where uniform 9-bit
reached 10 (MiniMax, 7 751 tok, 11.49 GB pool), so the advisor's claim is now **~1.65×, not 2×**.
`PREFIX_COMPRESS` still defaults to `None`: this changes what `9` MEANS, not whether it is on.
Suite 1678 → **1684** (6 new tests). Detail: `docs/2026-09-06-P2-outlier-split-RESULTS.md`.
**Two things the promoted row got wrong, both now locked by test — do not "restore" either:**
- **Channels are picked ONCE PER LAYER, shared by every head.** Per-(layer, head) selection was
  measured and **FAILS MiniMax at m=8 (0.0725 — worse than not splitting at all, 0.0426)**, while
  looking better than per-layer on Qwen at m=16. A granularity validated on one forward does not
  transfer to the other (CLAUDE.md §5, again).
- **1.281 B/elem was an upper bound; the real figure is 1.211.** The promoting probe zeroed the
  outlier channels but still spent quantizer bits on all 128 dims. The shipped path quantizes a
  genuine 120-dim residual with its own rotation and codebook.
**`m` is not a knob, and m=4 is worse than useless:** `pack_codes` needs `(head_dim − m) % 8 == 0`,
so at head_dim 128 m=2 and m=4 cost **2.031 B/elem — above bf16's 2.000**. `_quantize_split` snaps
`m` up to the nearest bit-packable width. m=16 costs 1.266 and buys nothing (Qwen 0.0396, *worse*
than m=8's 0.0274).
**Mechanism, corrected as the row asked.** Plan H §7 and Plan I §8.5 both justify this split by
"both models' outliers are early layers". Measured: the layers that FAIL are the **diffuse-attention**
ones (Qwen L0 top-1 mass 0.058), and the two highest-kurtosis layers (Qwen L47/L93, 37.3/52.8) do
**not** fail. The split works anyway — a few channels dominate the inner product softmax sees — but
not for the stated reason. Nor is the failure a sink artifact: at Qwen L0, **6.5 % of attention
probability mass moves** (total variation), an independent metric agreeing with the gate.
**Still NOT established (see the row above):** that failing this bar degrades generated text, and
anything at all about V — the gate scores attention over K, and the split is applied to V by the
same code without a V number behind it.

---
