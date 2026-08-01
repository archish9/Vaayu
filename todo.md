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
3. **P3 — Plan D (expert histogram, MEASUREMENT ONLY).** Plan B's gate cleared, but its pin is
   premise-invalidated twice over (wrong tier, and a startup pre-warm is evicted by the first
   prefill). Plan D measures whether a pin has any target at all before one is designed. A
   negative result retires the pin on evidence — cheaper than building it on assumption
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

### Plan C — Prefix-cache hit telemetry  **(P2 — chain step 4)** — ✅ DONE 2026-08-01
`docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry.md`
Spec: `docs/superpowers/specs/2026-08-01-prefix-cache-hit-telemetry-design.md` (`140720c`)
RESULTS: `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md`
- [x] **Task 1** — counters + `stats()` + `reset_stats()` on `PrefixCacheStore` (TDD) — `626dcaf`
  - [x] 9 unit tests in `tests/test_prefix_cache.py` (incl. degenerate hit, multi-evict, n/a rates)
  - [x] 5 counters under the store's existing lock; no caching-behavior change
  - [x] Full suite green (400)
- [x] **Task 2** — `test_multiturn.py` becomes a gate — `cf7963e`
  - [x] reset between rounds, print `stats()` per round
  - [x] assert final-round `hits == K_SESSIONS`, exit non-zero on failure
  - [x] loud failure if `model._prefix_cache` is None
- [x] **Task 3** — on-device gate + RESULTS (closes chain step 4) — `e7f5939` `899ce91`
  - [x] K=2 run, exit 0, 2/2 hits final round, 0 evictions
  - [x] K=3 run, exit 0, 3/3 hits final round, 0 evictions; `K_SESSIONS` reverted to 2
  - [x] RESULTS doc + close chain step 4 here
- [x] **Final review fixes** — `aada3cd`: auto-match asymmetry documented + 2 tests,
  `.gitignore` for evidence logs, unlocked-read rationale, `R_ROUNDS<2` guard,
  kv_persist eviction note

### Plan B — Expert usage histogram (Subsystem B)  **(P3)** — RETARGETED 2026-08-01
`docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md` — Phase 1 **superseded**
by the measurement-only retarget below. Its gate (the VRAM OOM chain) closed 2026-08-01, but its
PREMISE WARNING still stands: do not build Phase 1 as written.
- [x] Blocker cleared — VRAM OOM chain closed (`045f387`..`aada3cd`)
- [!] Tasks 3,6 (the pin) — NOT built. Two reasons: `ExpertVRAMCache` still has no pin mechanism,
  and a startup pre-warm is a **no-op** — `_ensure_prefill_headroom` evicts the VRAM cache on
  every prefill and every generation starts with one (measured 9.04→2.64GB, 1362 evictions).
  Unblocks only if Plan D's measurement finds hot-but-evicted experts.
- [x] Tasks 2,5 (persist/decay, `compute_pin_set`) — **DROPPED as speculative** (§2): persistence
  matters only once a win is known; a pin set for a pin that doesn't exist is dead weight.
- [ ] Tasks 8,9 (Phase 2 live re-pin) — untouched, still deferred

### Plan D — Expert usage histogram, measurement only  **(P3 — active)**
`docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement.md`
Spec: `docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md` (`c38132e`)
Answers ONE question: are there experts decode routes to often but the VRAM cache keeps evicting?
A "no" retires Plan B's pin on evidence — that outcome is a success, not a failure.
- [ ] **Task 1** — `ExpertUsageTracker`: device-side route counter, no hot-path sync (TDD)
- [ ] **Task 2** — per-key attribution of **budget-pressure evictions only** in `ExpertVRAMCache`
  - [ ] Discriminating test: `evict_until_free()` evictions must NOT be attributed (needs CUDA —
        confirm it RAN, not skipped)
- [ ] **Task 3** — report joining routes with evictions; labels "no pin needed" as plainly as
  "pin candidate"
- [ ] **Task 4** — wire in behind `enable_expert_usage` (default False); hook at 3 routing sites,
  decode-only
- [ ] **Task 5** — on-device run on MiniMax + RESULTS + the pin VERDICT; check decode vs ~1.47 floor

---

## Chain — VRAM prefill OOM (memory: `vram_prefill_oom_chain`)
- [x] Step 1 — RAM budget rebalance (`6c9bc76..54d660b`)
- [x] Step 2 — VRAM diagnostic + RESULTS (2026-07-24, `855c09f` `fc01df0`); verdict = decode-boundary OOM
- [x] Step 3 — execute Plan A (above), 2026-08-01 (`045f387` `86a6ca5`); OOM closed at K=2 and K=3
- [x] Step 4 — round-2 HIT measurement **(P2)** — 2026-08-01, `626dcaf` `cf7963e` `e7f5939`.
  Harness now asserts HITs itself (exit status 0 is the verdict):
  K=2 → 2/2 HIT, 0evict, saved=3940/3980 tok (99%); K=3 → 3/3 HIT, 0evict, saved=5910/5970 tok
  (99%). TTFT collapsed ~90-101s → ~9-10s at both K. Kernel OOM count 0.

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

- [x] ~~**Dead code — `PrefixCacheStore.drop()`.**~~ No callers anywhere in the repo; dead since
  it was written (`cac8ccd`, the original A1 commit). **REMOVED 2026-08-01** on the user's call.
  `ExpertVRAMCache.drop()` is a different method and is genuinely used — untouched.
  (found + resolved 2026-08-01)
- [ ] **Prefix `hit_rate` is not comparable across lookup modes.** On a named session, an entry
  that shares ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the
  same case counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing semantics —
  documented in `stats()` and pinned by tests in `aada3cd`, NOT changed. Worth revisiting if a
  mixed-mode workload ever needs one comparable number. (found 2026-08-01, Plan C final review)

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
