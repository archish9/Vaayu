# Session Brief — The Misplaced-Feature Audit

> **Read this first, in a fresh session with no prior context. It is self-contained.**
> Written 2026-08-02 at the end of the Plan I session. Repo facts verified against the working
> tree at branch `plan-i-compress-on-evict`, commit `0459be2`.

## Your one job this session

Find every place in Vaayu where **a feature is correctly written but installed where it cannot
act.** Produce a verified register of them. Then — and only for the ones that clear the promotion
bar in §6 — write self-contained implementation plans.

**This session does NOT fix anything.** It finds, proves, ranks, and plans. Resist the pull to
patch something the moment you find it; a half-fixed defect with no measurement is how the last
three plans ended in refutation.

## 1. Why this session exists

Plan H (2026-08-02) implemented TurboQuant KV compression on the **live** cache, so every decode
step reconstructed the entire prefix: `O(S·d²)` per token, benchmarked at +0.28 s/token at 8k and
**+1.49 s/token at 32k** on Qwen. The algorithm was right, the tests passed, the size target was hit
exactly — and the feature was unusable, because it was installed in the hot path.

Plan I moved the same code to the eviction boundary (compress once per turn, on the way into the
RAM prefix pool). Same memory win, **zero** per-token cost: Qwen 0.86 → 0.86 tok/s, MiniMax
1.53 → 1.56. Nothing about the algorithm changed. Only its position did.

That is the pattern. A feature in the wrong position is invisible to tests (it is correct), to
profilers (it is doing what it was told), and to code review (it looks fine locally). It shows up
only when someone asks *"is this thing actually able to do its job where it sits?"*

The suspicion driving this session: **Vaayu has more of these.** Three are already identified (§5).
The audit is to find the rest.

## 2. What counts as a misplaced feature (taxonomy)

Use these five categories. If a candidate does not fit one, it is probably an ordinary bug — note
it and move on; it is out of scope.

| # | Category | Signature | Known example |
|---|---|---|---|
| A | **Wrong lifecycle position** | Built, filled, then destroyed before the phase that needed it | Expert VRAM cache flushed every prefill (§5.1) |
| B | **Stranded hook** | Lives in a branch/callback nothing reaches at runtime | `cache_aware_topk` (§5.2) |
| C | **Wrong tier** | Correct mechanism applied to the hot tier instead of the cold one (or vice versa) | Plan H TurboQuant on the live cache |
| D | **Inert by config/hardware** | Executes, but always collapses to a no-op case | Batched decode → N=1 (§5.3) |
| E | **Dead by ordering** | Two correct operations whose order makes one useless or harmful | `kv_persist` reading the cache after `pc.store()` (fixed in Plan I, `dc0da7a`) |

## 3. The method — execution-count auditing

Static reading is not enough. The `VAAYU_CACHE_BIAS` defect survived multiple readings of the code
because the branch it lives in *looks* reachable. It was caught by counting calls at runtime: **0
on both models.**

For each candidate, answer these four questions **in order**, and stop early when one fails:

1. **Does it execute at all?** Wrap the function/branch in a counter (a scratchpad wrapper — do not
   modify repo files for a probe) and run one real generation. Zero calls = category B, proven.
2. **Does it execute at the right time?** Log the phase (load / prefill chunk / decode step /
   turn boundary) alongside each call. Compare against the phase it exists to serve.
3. **Does its effect survive to that phase?** The hard one. A cache can be filled at the right time
   and emptied before use. Measure the *state at the point of use*, not the action.
4. **Would correcting its position change a measured number?** If you cannot name the number and
   its current value, the finding is not promotable (§6).

**Probe hygiene, learned the hard way:** never attach `py-spy` or any ptrace-based profiler to a
live Vaayu process — it corrupts `vaayu_rust`'s io_uring state (0 CQE overflows unprofiled vs 100+
profiled). Use in-process counters and log lines.

## 4. The audit surface

Every one of these is a shipped, gated, or configurable feature. Walk the list; for each, ask the
four questions. Tick them off in the register whether or not they turn up a defect — a clean bill
of health is a result, and stops the next person re-auditing it.

**Caching / memory**
- Expert VRAM cache (`ExpertVRAMCache`) and its interaction with `_ensure_prefill_headroom`
- Expert RAM cache (`ExpertRAMCache`), its budget, and its reset points
- Cross-turn RAM prefix cache (`PrefixCacheStore`, `enable_prefix_cache`, `auto_prefix_match`)
- KV disk persistence (`enable_kv_persist`, `kv_persist_sync`, `vaayu/kv_persist.py`)
- `prefix_compress` (Plan I, new — off by default)
- `kv_cache_mode` incl. `turboquant` (Plan H — known-bad at 4 bits, opt-in)

**MoE routing / experts**
- All three MoE forwards in `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3`,
  `_make_fused_sparse_forward`, `_make_modulelist_sparse_forward`) — **a feature wired into one
  forward silently no-ops on the others; this is how the bias defect happened**
- `VAAYU_CACHE_BIAS` / `cache_aware_topk` / `_build_resident_mask` (§5.2)
- Expert usage tracking (`enable_expert_usage`)
- Expert prefetch / lookahead (`VAAYU_LOOKAHEAD` — currently raises a RuntimeError on use; find out
  why it was disabled and whether the reason still holds)

**Execution / scheduling**
- Batched decode (`vaayu/batch.py`, `plan_batch`) (§5.3)
- Speculative decoding (`drafter_model_path`, `use_prompt_lookup`, `draft_tokens`)
- Layer-major chunked prefill and `prefill_chunk_size`
- Layer pinning (`max_pinned_layers`) and `_get_vram_pinning_budget`
- NVMe warmup (`_warmup_nvme`) — note it is skipped entirely for MoE models; check that is
  deliberate and still right
- `use_torch_compile`
- io_uring / `vaayu_rust` staging and the pinned-memory loader

**Reference reading before you start** (skim, do not read end-to-end):
- `CLAUDE.md` §0 — the dispatch table. Which forward each model takes. Non-negotiable context.
- `docs/superpowers/plans/2026-08-02-turboquant-kv-RESULTS.md` — the Plan H post-mortem
- `docs/superpowers/plans/2026-08-02-compress-on-evict-kv.md` — the fix, and the template for a
  good plan
- `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement-RESULTS.md` — the expert
  cache measurements you will need for §5.1
- `todo.md` — Pending bugs section, and the Plan B entry (it already records the §5.1 finding)

## 5. The three findings already in hand

These are **starting material, not the answer.** Two are proven, one is measured-but-unexplained.
Re-verify each with the §3 method before planning against it — and keep hunting; the point of the
session is the ones not yet on this list.

### 5.1 Expert VRAM cache is emptied on every prefill — **category A, highest expected value**

`_ensure_prefill_headroom` evicts the expert VRAM cache at each prefill. Measured (Plan D,
2026-08-01): **9.04 GB → 2.64 GB, 1362 evictions per prefill.** A startup pre-warm is therefore a
**no-op** — it is undone before the first decoded token.

Why it matters: expert streaming from NVMe is 0.1–0.3 s of the 0.3–0.5 s per-token decode budget —
the actual decode bottleneck. Every turn, decode begins with a cold expert cache and re-fetches
what it had.

**The honest counter-evidence, which is why this needs measuring not building:** Plans D and F
found the cache thrashes on *routing diversity*, not capacity — it holds ~6.2% of all experts, and
evictions ≈ misses per token during steady-state decode. A warm cache may buy far less than the
1362-eviction figure suggests. Plan F already refuted the related expert-pin idea on exactly this
kind of evidence (eviction share linear in N, 2–4× below the cost line, both models).

**The measurement that settles it:** instrument one turn to attribute decode's expert fetches into
(a) experts the cache held before prefill evicted them, vs (b) genuine first-touches. If (a) is
small, this finding dies like the pin did — and that is a good outcome, cheaply reached.

### 5.2 `VAAYU_CACHE_BIAS` is stranded upstream — **category B, proven**

`cache_aware_topk` / `_build_resident_mask` (`vaayu/moe_patch.py`) live in the plain-softmax `else`
branch of the router dispatch. Under transformers 5.5.4, **no MoE model reaches it**: Qwen takes the
`len(router_res) >= 3` tuple branch (5.x routers do their own top-k), and MiniMax is not even on
that forward — it runs the module-list one. **Probe at bias=2.0: 0 calls on both models.**

Status: a warning was added (`df9a52e`) so the user is told it is inert. The feature itself is still
unreachable. Re-targeting it to run *before* the model's own top-k is an unwritten spec, and is a
genuine candidate for this session's output.

Bias sweep evidence (Plan G): decode 1.52/1.50/1.51/1.50/1.54 and VRAM hit rate 36.8/36.1/36.7/
36.0/37.2% at bias 0/0.25/0.5/1.0/2.0 — flat and non-monotonic, consistent with zero effect.

### 5.3 Batched decode collapses to N=1 — **category D, needs classification**

`vaayu/batch.py`'s `plan_batch` collapses to N=1 on the 16 GB box. Multi-query batched decode is a
first-class lever for this workload (CLAUDE.md §0: the agent runs parallel fan-out, so aggregate
throughput matters). A batching feature that never batches delivers none of it.

**Open question this session must answer:** is this misplacement (the budget check is in the wrong
place, or charges the wrong thing), or is it an honest hardware limit? Note that Plan H found both
VRAM byte-accounting sites charge 2 B/elem for every non-fp8 mode — if `_resolve_kv_elem_bytes` or
`free_vram_budget` over-charges, `plan_batch` would refuse batches that would actually fit. Check
that arithmetic against measured occupancy before concluding.

## 6. The promotion bar — what earns an implementation plan

A finding graduates from register entry to plan **only if all four hold**:

1. **Proven inert or misplaced**, by execution count / phase log / state-at-use measurement — not by
   reading the code.
2. **Names the number it would move**, with that number's current measured value (decode tok/s,
   TTFT, prefix HIT rate, VRAM headroom, aggregate throughput).
3. **The fix is positional** — moving, re-hooking, or re-ordering existing code. If it requires a
   new algorithm, it is a different kind of work; register it and say so.
4. **No existing measurement already refutes it.** Check Plans D, F, G, H RESULTS first. Plan F
   killed the expert pin; do not resurrect it without new evidence.

Findings that fail the bar still go in the register, marked with which criterion they failed and
what evidence would change that. **Do not write a plan for them.** Three consecutive plans (F, G, H)
ended in refutation; the cost of planning a fix for a defect whose repair moves nothing is exactly
what this bar exists to avoid.

## 7. Deliverables

1. **`docs/superpowers/specs/2026-08-03-misplaced-feature-audit-RESULTS.md`** — the register. Every
   item from §4, each with: category (or "clean"), the evidence that settles it, the number it
   would move, and promoted/not-promoted with reason. This is the primary deliverable and is
   valuable even if nothing gets promoted.
2. **N self-contained implementation plans**, one per promoted finding, in
   `docs/superpowers/plans/YYYY-MM-DD-<name>.md`. Follow the structure of
   `2026-08-02-compress-on-evict-kv.md`: goal, architecture, global constraints, file structure,
   bite-sized TDD tasks with real code, an on-device gate task, success criteria, explicit
   out-of-scope. Each must stand alone for a fresh session.
3. **`todo.md` updated** — register linked, every promoted plan added with its tasks as checkboxes,
   every non-promoted finding recorded under Pending bugs so it cannot vanish (CLAUDE.md §6).

## 8. Out of scope

- **Fixing anything.** Probes live in the scratchpad; repo files stay untouched except `todo.md` and
  the two deliverable docs.
- Ordinary bug-hunting, style, refactors, dead code removal. If it is not one of the five categories
  in §2, note it in `todo.md` and move on.
- Re-litigating settled questions: the expert pin (Plan F), uniform-bit TurboQuant quality (Plan H).
  New evidence required to reopen.
- Performance work on the dense path (deferred per CLAUDE.md §0) unless a finding happens to sit in
  shared infrastructure.

## 9. State of the tree when this was written

- **Branch `plan-i-compress-on-evict`**, 4 commits ahead of `main` (`6d96049` `e84d101` `dc0da7a`
  `0459be2`). Suite **518 green**. Not merged.
- **Plan I Task 5 (on-device gate) was still running.** Step 1 passed (decode unchanged, §1).
  Steps 2–6 — pool capacity at K=4 with a capped 1050 MiB budget, long context, per-turn
  compression cost, restored-prefix quality, the live ordering trap — were mid-flight.
  Logs and scripts: `~/vaayu-evidence/2026-08-02-plan-i/` (`k4-off.log`, `k4-c8.log`,
  `step2-status.txt`, `restored_quality.py`). **Check `step2-status.txt` for `DONE` and finish that
  gate before starting the audit, or explicitly decide to park it.**
- `test_generate.py` and `test_multiturn.py` carry **uncommitted Plan I scaffolding** (env-var
  knobs, pool accounting prints, a budget cap). Revert with
  `git checkout test_generate.py test_multiturn.py` once the gate is done.
- `docs/superpowers/plans/2026-08-02-compress-on-evict-kv-RESULTS.md` exists with §1 filled in and
  later sections marked _(filled)_.
- Prior evidence archive: `~/vaayu-evidence/2026-08-02-plan-h/` — 9 gzipped run logs plus
  `quality_gate.py` / `sweep.py`, both re-runnable on CPU from persisted `.vaayu_kv` files.

## 10. Method notes worth carrying in

- **Two models, two forwards.** Qwen3-235B is fused-EXL3; MiniMax-M2.7 is module-list. A finding
  proven on one is **not** evidence for the other — Plan H's decode result differed between them for
  exactly this reason. Check both.
- **Derive, never hardcode.** Layer counts, expert counts, head dims come from config or
  `_moe_geometry` (CLAUDE.md §5).
- **A negative result is a real deliverable.** Plan F's "no viable N" saved the pin from being built.
  The best outcome here may be "four of these are fine, one is not."
