# TODO — Vaayu task ledger

Single source of truth for what is done and what is pending. **Both the user and Claude
add/update here, as work happens.** Protocol: `CLAUDE.md` §6.

Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

_Last updated: 2026-08-01._

---

## Execution order (priority)

Sequence is forced by dependencies, not preference:

1. **P1 — Plan A (VRAM decode-headroom fix).** The gate. Removes the multi-session
   prefill→decode OOM that blocks everything else. Start with Task 1 (tiny, in-code, TDD),
   then the on-device gate.
2. **P2 — Chain step 4 (measure round-2 HIT).** Only runnable once A stops the OOM. Cheap,
   and it's the original goal. Expect it to surface the prefix MISS-on-revisit bug (see
   Pending bugs) → likely spawns a small follow-up plan.
3. **P3 — Plan B (expert histogram).** Last. Blocked on A: its header mandates "after the
   VRAM OOM chain," it needs retargeting RAM→`ExpertVRAMCache`, and its VRAM-tier pin can't
   be **sized** until A fixes how much VRAM headroom exists. Do NOT start early — the infra
   is tier-agnostic but delivers zero value before the pin, so building it now is speculative
   (CLAUDE.md §2).

---

## Active plans

### Plan A — VRAM decode-headroom fix  **(P1)** — ✅ DONE 2026-08-01 (`045f387` `86a6ca5`)
`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md` (chain step 3)
RESULTS: `docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md`
- [x] **Task 1** — raise `_VRAM_RESERVE_BYTES` 2GB→3.5GB (in-code, TDD) — `045f387`
  - [x] Guard test in `tests/test_vram_budget.py` (fail → pass); full suite 391 green
  - [x] Constant + hardware-comment on `_VRAM_RESERVE_BYTES` in `vaayu_base.py`; docstring updated
  - [x] Mark on-hold KV-eviction spec superseded
- [x] **Task 2** — on-device gate — `86a6ca5`
  - [x] K=2 R=2 with `expandable_segments:True` → 4/4, no OOM, kernel OOM 0
  - [x] Attribution: reserve bump ALONE fixes it (no-exp run also 4/4); expandable_segments = optional insurance
  - [x] Decode unregressed: MiniMax 1.49 (floor 1.47); Qwen 0.84/0.83 vs pre-fix **measured** 0.82 → no back-off needed
  - [x] K=3 stretch — 6/6, no OOM
  - [x] RESULTS written + committed

### Plan C — Prefix-cache hit telemetry  **(P2 — chain step 4)**
`docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry.md`
Spec: `docs/superpowers/specs/2026-08-01-prefix-cache-hit-telemetry-design.md` (`140720c`)
- [ ] **Task 1** — counters + `stats()` + `reset_stats()` on `PrefixCacheStore` (TDD)
  - [ ] 9 unit tests in `tests/test_prefix_cache.py` (incl. degenerate hit, multi-evict, n/a rates)
  - [ ] 5 counters under the store's existing lock; no caching-behavior change
  - [ ] Full suite green (400)
- [ ] **Task 2** — `test_multiturn.py` becomes a gate
  - [ ] reset between rounds, print `stats()` per round
  - [ ] assert final-round `hits == K_SESSIONS`, exit non-zero on failure
  - [ ] loud failure if `model._prefix_cache` is None
- [ ] **Task 3** — on-device gate + RESULTS (closes chain step 4)
  - [ ] K=2 run, exit 0, 2 hits final round
  - [ ] K=3 run, exit 0, 3 hits final round; revert `K_SESSIONS` to 2
  - [ ] RESULTS doc + close chain step 4 here

### Plan B — Expert usage histogram (Subsystem B)  **(P3 — do last)**
`docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- [!] **BLOCKED / needs retarget** — premise warning (see plan header): RAM-tier pin is
  low-value now (RAM expert-cache hit 34%→5%; VRAM cache serves hot experts). Retarget the
  pin RAM→`ExpertVRAMCache`, and sequence AFTER Plan A / the VRAM OOM chain (VRAM contended).
- [ ] Tasks 1,2,4,5 — histogram infra (count/persist/rank/hook) — tier-agnostic, buildable now
- [!] Tasks 3,6 — pin: move from `ExpertRAMCache._hot` → `ExpertVRAMCache` before building
  (VRAM class has no pin mechanism yet)

---

## Chain — VRAM prefill OOM (memory: `vram_prefill_oom_chain`)
- [x] Step 1 — RAM budget rebalance (`6c9bc76..54d660b`)
- [x] Step 2 — VRAM diagnostic + RESULTS (2026-07-24, `855c09f` `fc01df0`); verdict = decode-boundary OOM
- [x] Step 3 — execute Plan A (above), 2026-08-01 (`045f387` `86a6ca5`); OOM closed at K=2 and K=3
- [~] Step 4 — round-2 HIT measurement **(P2)** — already OBSERVED in Plan A's gate runs: round-2
  TTFT collapsed ~90-101s → ~9-10s at both K=2 and K=3 (~10x cross-turn prefix-cache win), no CUDA
  OOM. Remaining: formalize as a step-4 measurement (explicit HIT-count assertion), not just an
  observation from the OOM gate logs.

---

## Pending bugs / open questions (found in review, not yet fixed)
- [x] ~~**Prefix cache MISS-on-revisit.**~~ Diagnostic run showed every `test_multiturn`
  session was a prefix MISS even revisiting its own session id. (found 2026-07-24)
  **NO LONGER REPRODUCES** after Plan A (2026-08-01): round-2 revisits HIT at both K=2 and
  K=3 (TTFT ~90-101s → ~9-10s). The MISSes were a downstream effect of the OOM aborting/
  poisoning round 2, not a separate keying bug.
- [x] ~~**Stale cross-reference comment.**~~ `vaayu/moe_patch.py` claimed
  `_PREFILL_VRAM_HEADROOM` was "the same class/value as `vaayu_base._VRAM_RESERVE_BYTES`" —
  false since Plan A (2 GB vs 3.5 GB). **FIXED** in `9aa2c61`: comment now states both are the
  same CLASS of hardware constant but deliberately DIFFERENT values, with the reason.
  (found + fixed 2026-08-01)
- [x] ~~**VRAM-cache-disabled message was too quiet.**~~ `vaayu/vaayu_base.py` announced
  "Expert VRAM cache: disabled (insufficient free VRAM)" as a plain info line, while the
  milder working-set-exceeds-budget case right below it uses `WARNING:`. CLAUDE.md §5 says a
  model that genuinely doesn't fit must WARN, not silently degrade — and this is the harder
  failure of the two (every expert streams per token). **FIXED 2026-08-01**: now a `WARNING:`
  carrying free VRAM, the post-reserve budget, the reserve, and the derived expert size.
  Found by Plan A's final review (parked there as out-of-scope). Not reachable on this box
  (11.7 GB free), so no on-device re-run needed. (found + fixed 2026-08-01)
- [x] ~~**Drifting line-number citations.**~~ Plan A's docs cited `vaayu_base.py:168`, then
  `:179`, while the constant is now at `:188` — each edit to the file re-broke them.
  **FIXED 2026-08-01** by citing the SYMBOL (`_VRAM_RESERVE_BYTES`) instead of a line number,
  which cannot drift. Historical line refs inside the 2026-07-23 diagnostic RESULTS are left
  as-is: they are a point-in-time record, accurate when written.

---

## Done (archive)
- [x] **VRAM prefill diagnostic** — `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md`
  (= chain step 2). All 3 tasks done 2026-07-24: Task 1 instrumentation `855c09f`, Task 2
  on-device run, Task 3 RESULTS `fc01df0`. Verdict: OOM is decode-boundary, not prefill →
  spawned Plan A.
- [x] KV prefix cache A1 + KV disk persistence (landed, ships dark)
- [x] Batched decode static v1 (landed; `plan_batch` collapses to N=1 on the 16GB box)
- [x] MiniMax-M2.7 bring-up (4 root causes fixed)
- [x] Expert-usage histogram plan — anchors re-verified + premise warning added (`417e0e0`)
