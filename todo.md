# TODO — Vaayu task ledger

Single source of truth for what is done and what is pending. **Both the user and Claude
add/update here, as work happens.** Protocol: `CLAUDE.md` §6.

Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

_Last updated: 2026-07-24._

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

### Plan A — VRAM decode-headroom fix  **(P1)**
`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md` (chain step 3)
- [ ] **Task 1** — raise `_VRAM_RESERVE_BYTES` 2GB→3.5GB (in-code, TDD)
  - [ ] Guard test in `tests/test_vram_budget.py` (fail → pass)
  - [ ] Constant + hardware-comment in `vaayu_base.py:168`; docstring at `:171`
  - [ ] Mark on-hold KV-eviction spec superseded
  - [ ] Full suite green + commit
- [ ] **Task 2** — on-device gate (operator-run)
  - [ ] K=2 R=2 with `expandable_segments:True` → no OOM, kernel OOM 0
  - [ ] Isolate reserve vs expandable_segments contribution
  - [ ] Decode unregressed: MiniMax ≥1.47, Qwen ≥0.85 tok/s (back off to 3.0GB if not)
  - [ ] K=3 stretch
  - [ ] Write RESULTS + commit

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
- [ ] Step 3 — execute Plan A (above) **(= P1)**
- [ ] Step 4 — re-run `test_multiturn.py` K=2 then K=3: confirm round-2 HIT count > 1, no CUDA OOM **(P2)**

---

## Pending bugs / open questions (found in review, not yet fixed)
- [bug] **Prefix cache MISS-on-revisit.** Diagnostic run showed every `test_multiturn`
  session was a prefix MISS even revisiting its own session id — the round-2 HIT the chain
  wants is not landing even before the OOM. Separate cache eviction/keying question; chase
  after Plan A closes the OOM. (found 2026-07-24)

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
