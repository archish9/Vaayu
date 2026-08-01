# Expert Pin + Routing Locality — Implementation Plan (Plan G)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything you need is
> in this document.

**Goal:** Raise MoE decode throughput by cutting VRAM expert-cache thrash. Two levers, cheapest
first: **(1) turn on the cache-aware routing bias that already exists and ships off**, measure it;
**(2) only if that is insufficient**, add a hot-expert pin to `ExpertVRAMCache`, sized by a prior
sweep.

**Architecture:** Task 1 is a zero-code on-device experiment — `VAAYU_CACHE_BIAS` is an existing
env var, default `0.0`. Tasks 2-4 add a pin mechanism (`_hot` set + eviction skip + a starvation
guard) to `ExpertVRAMCache`, warmed at the prefill→decode boundary rather than at startup. Task 5
gates everything on-device.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

---

## PRECONDITION — read this before writing any code

**This plan requires an N from the `top_n` sweep** (`docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep.md`).
Check whether its RESULTS exist:

```bash
ls docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep-RESULTS.md
```

- **If it does not exist:** Task 1 is still safe to run (it is zero-code and independent). Do NOT
  start Task 2 — you would be picking N by guess, and the one N ever measured (10) was decisively
  not viable. Stop after Task 1 and report that the sweep is required.
- **If it exists:** read its VERDICT. If it says no N is viable, **do not build the pin** — report
  that and stop after Task 1. If it names an N, use that N throughout.

---

## Why this exists — what is already known

Measured 2026-08-01 on MiniMax-M2.7 (62 layers × 256 experts, top_k=8, 256 output tokens, 16 GB
GPU):

- **126,480** expert routes; **78,726** budget-pressure evictions from the VRAM expert cache.
- Pinning each layer's top-10 experts would cover **19%** of those evictions while consuming
  620 × 8.9 MB = **5.5 GB of the 8.7 GB cache budget (63%)**. Share far below cost — self-defeating.
- The cache holds ~983 of 15,872 experts (**6.2%**) — about two tokens' worth. Its per-token
  working set (4.4 GB) already fits in the 8.7 GB budget, yet it thrashes anyway, because the
  *union* of experts routed across many tokens is far larger than any one token's set.
- Decode cost per token at 1.48 tok/s: `expert_disk` 0.1–0.3 s, `compute` 0.3–0.5 s.

**Two conclusions drive this plan's ordering.** First, **capacity is not the binding constraint —
distribution is.** Adding cache capacity does not change which experts the router asks for.
Second, **the ceiling for all expert-caching work is ~25–30% of token time**, because that is all
`expert_disk` is; compute dominates. Do not expect more than that from anything here.

That is why the routing lever comes first: it changes *what the router asks for* so requests land
on resident experts, attacking the actual constraint, and it requires no new code.

## The routing lever already exists (this is the key fact)

`vaayu/moe_patch.py` has, shipped and wired into the EXL3 fused forward:

- `cache_aware_topk(logits, resident_mask, top_k, base_bias)` — biases top-k **selection** toward
  resident experts while taking combination **weights** from the original logits, so weighting
  stays faithful. The bias is scaled by the per-row logit std, so a confident router is not
  overwhelmed. With `base_bias == 0.0` or `resident_mask is None` it reduces **exactly** to plain
  softmax-topk.
- `_build_resident_mask(vaayu_ref, expert_cache, layer_idx, num_experts, device)` — returns the
  RAM ∪ VRAM resident mask for the layer, or `None` when the feature is off, during prefill, or
  when nothing is resident.
- `vaayu/vaayu_base.py` sets `self._cache_bias = float(os.environ.get("VAAYU_CACHE_BIAS", "0.0"))`.

**So the feature is complete and defaults to off.** Task 1 is a one-env-var experiment.

Two caveats to carry into the measurement:
- Biasing selection **changes which experts run**, so outputs will differ from an unbiased run.
  That is a quality question, not just a speed one — judge both.
- It is wired into the EXL3 fused path. Confirm it is reached on the model you test before
  trusting a null result.

---

## Global Constraints

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No layer count, expert count, top_k, or
  expert-byte literal in code. The pin size derives from geometry (`model._moe_geometry`) and the
  cache budget, never a hardcoded number.
- **Ships dark.** Any pin must be behind a flag defaulting to off, and `VAAYU_CACHE_BIAS` keeps
  its `0.0` default. Nothing in this plan may change behavior for a user who sets nothing.
- **Do not regress decode.** Floors on the 16 GB box: MiniMax ≈ **1.47 tok/s**, Qwen ≈ **0.82
  tok/s**. A pin that misses these is a failed pin — the whole point is speed.
- **Never pin so much that the tail starves.** A pin consumes budget from the same cache it is
  helping. Enforce a hard ceiling (see Task 3) so the pinned set can never exceed a configured
  fraction of the cache.
- **Surgical (CLAUDE.md §3).** Do not refactor the MoE forwards, the eviction policy, or the
  router paths beyond what the pin requires.
- Run tests with `source venv/bin/activate`. Record the baseline suite count before starting.

---

## Task 1: Measure the routing bias — zero code (operator-run)

Needs a CUDA GPU and real weights. ~15 min per run. **Run this before writing any pin code — it
may make the pin unnecessary, or show that the distribution lever is the better investment.**

**Files:** Create `docs/superpowers/plans/2026-08-02-expert-pin-and-routing-locality-RESULTS.md`
(start it here; later tasks append).

- [ ] **Step 1: Baseline.** Run `test_generate.py` unchanged on MiniMax. Record decode tok/s, the
  per-token `vram=...hit_rate=` telemetry, and save the generated output text for comparison.

- [ ] **Step 2: Sweep the bias.** Re-run at several bias values:

```bash
source venv/bin/activate
for B in 0.25 0.5 1.0 2.0; do
  VAAYU_CACHE_BIAS=$B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u test_generate.py > run-bias-$B.log 2>&1
  echo "bias=$B exit=$?"
  grep -E "Decode throughput" run-bias-$B.log
done
```

- [ ] **Step 3: Confirm the feature actually engaged.** A null result is only meaningful if the
  code path ran. Verify the VRAM cache hit rate moved between bias=0 and bias>0 — grep the
  per-token telemetry (`vram=NNhit/NNmiss/NNevict hit_rate=NN%`) and compare. If hit rate is
  identical across bias values, the mask was `None` or the path was not reached; investigate
  before concluding anything.

- [ ] **Step 4: Judge quality, not just speed.** Biasing selection changes which experts run.
  Read the generated code from each run. Rising tok/s with degraded output is not a win. Record
  your qualitative read honestly.

- [ ] **Step 5: Record findings** in RESULTS: a table of `bias → decode tok/s → VRAM hit rate →
  output quality`, plus a VERDICT on whether routing bias alone is worth shipping (and at what
  default), and whether it makes the pin unnecessary.

- [ ] **Step 6: Commit RESULTS.** If the verdict is that bias alone suffices, **stop here** and
  say so — do not build the pin. Report that outcome; retiring unnecessary work on evidence is a
  success.

---

## Task 2: Pin mechanism on `ExpertVRAMCache` (TDD)

Only start this if the sweep named a viable N **and** Task 1 showed routing bias alone is
insufficient.

**Files:** Modify `vaayu/persist/expert_cache.py`; test in `tests/test_expert_vram_cache.py`.

**Interfaces produced:**
- `ExpertVRAMCache.pin(layer_idx, expert_idx) -> None`
- `ExpertVRAMCache.unpin_all() -> None`
- `.pinned` — a `set` of `(layer_idx, expert_idx)`
- Pinned entries are skipped by budget-pressure eviction in `put()`.

**Read the class first.** It is an `OrderedDict`-backed LRU with a single lock, keyed
`(layer_idx, expert_idx)`. `put()` evicts LRU under budget pressure and increments `evictions`
plus a per-key `evicted_keys` Counter. `evict_until_free()` is a separate wholesale prefill wipe
that increments `prefill_evictions` — **it is deliberately excluded from per-key attribution, and
your pin must not change that.**

- [ ] **Step 1: Write the failing tests.** The file already has a `_groups(nbytes)` helper —
  reuse it, do not redefine it. Cover:
  - A pinned entry survives budget pressure that would otherwise evict it (put enough to force
    eviction; assert the pinned key is still present and an unpinned one went instead).
  - **Starvation guard:** when *every* resident entry is pinned and a new `put()` cannot fit, the
    cache must refuse the insert (return `False`) rather than loop forever or evict a pinned
    entry. Write this test first — it is the one that turns a deadlock into a clean refusal.
  - `unpin_all()` clears the set and restores normal LRU behavior.
  - Pinning a key not resident in the cache is a no-op that does not raise.
  - `evict_until_free()` still evicts pinned entries — the prefill wipe is a correctness path for
    VRAM headroom and must not be blockable by a pin, or prefill can OOM. Assert this explicitly.

- [ ] **Step 2: Run them, confirm they fail.**

- [ ] **Step 3: Implement.** Add `self.pinned = set()` in `__init__`; skip pinned keys in `put()`'s
  eviction loop; if the loop can make no further progress (every remaining entry pinned), break
  and return `False` so the caller keeps ownership — mirroring the existing "single expert bigger
  than the whole budget" refusal. Leave `evict_until_free()` alone.

- [ ] **Step 4: Re-run, then the full suite. Step 5: Commit.**

---

## Task 3: Derive the pin set, with a hard budget ceiling (TDD)

**Files:** Modify `vaayu/expert_usage.py` (it already owns the histogram); test in
`tests/test_expert_usage.py`.

**Interface produced:** `compute_pin_set(counts, n_per_layer, expert_bytes, cache_budget_bytes, max_fraction, covered_layers=None) -> set[tuple[int, int]]`

- [ ] **Step 1: Write the failing tests.** Cover:
  - Returns each covered layer's top-`n_per_layer` experts by route count.
  - **The ceiling binds:** when `n_per_layer × layers × expert_bytes` would exceed
    `max_fraction × cache_budget_bytes`, the returned set is truncated to fit. Assert the byte
    total respects the ceiling. This is the guard that stops a pin from starving the tail it is
    supposed to help.
  - Layers with zero routes contribute nothing.
  - Layers outside `covered_layers` contribute nothing — pinning into a layer the VRAM cache never
    covers is meaningless.
  - Empty histogram returns an empty set.
  - Parametrized over Qwen-like (94 layers / 128 experts) and MiniMax-like (62 / 256) geometries
    per §5.

- [ ] **Step 2: Run, confirm failure. Step 3: Implement** as a pure function — no CUDA, no model.
  `max_fraction` is a caller-supplied float (no literal in library code).

- [ ] **Step 4: Re-run + full suite. Step 5: Commit.**

---

## Task 4: Apply the pin at the prefill→decode boundary

**Files:** Modify `vaayu/vaayu_base.py` and/or `vaayu/moe_patch.py`; test as feasible.

**Critical timing constraint — do not skip this reasoning.** A startup pre-warm is a **no-op**.
`_ensure_prefill_headroom` in `vaayu/moe_patch.py` evicts the VRAM expert cache on **every**
prefill pass whenever free VRAM drops under its headroom target, and **every generation begins
with a prefill** (measured: cache 9.04 → 2.64 GB, 1362 evictions in a single pass). Anything
warmed at startup is gone before the first decode step reads it.

So the pin must be applied **after prefill completes, at the prefill→decode boundary** — the same
place `ExpertRAMCache.reset_stats()` is already called. Find that call site and put the pin
application beside it.

- [ ] **Step 1: Add the flag.** `enable_expert_pin=False` on the model constructor, alongside the
  existing `enable_expert_usage`. Plus config for `n_per_layer` and `max_fraction`, both with
  documented defaults, both overridable.

- [ ] **Step 2: Apply at the boundary.** When the flag is on and a histogram exists, call
  `compute_pin_set(...)`, then `cache.pin(...)` for each key. Pass
  `covered_layers=self._vram_cacheable_layers`.

- [ ] **Step 3: Handle the cold-start case honestly.** On the first generation there is no
  histogram yet — the counts are empty, so the pin set is empty and nothing is pinned. Say so in a
  log line rather than silently doing nothing; a user who enabled the flag deserves to know why
  nothing was pinned. (Persisting the histogram across runs would fix this, and is deliberately
  out of scope — note it in `todo.md` rather than building it here.)

- [ ] **Step 4: Verify the feature is dark by default** — run the MoE test files
  (`tests/test_moe_patch.py`, `tests/test_moe_streaming_perf.py`) and the full suite.

- [ ] **Step 5: Commit.**

---

## Task 5: On-device gate (operator-run)

- [ ] **Step 1: Run the matrix** on MiniMax, appending to the Task 1 RESULTS doc: pin off/on ×
  bias off/on (4 runs). The interaction matters — a pin makes more experts resident, which gives
  the routing bias more to aim at, so the pair may beat either alone.

- [ ] **Step 2: Record for each cell** decode tok/s, VRAM cache hit rate, budget-pressure eviction
  count, and a qualitative read of output quality.

- [ ] **Step 3: Repeat on the second model** with a fresh `SESSION_ID` (a persisted KV under
  another model's id would poison the run).

- [ ] **Step 4: Decide.** The gate: **decode must beat the floors (MiniMax ≈1.47, Qwen ≈0.82) and
  beat the pin-off baseline.** If it does not, the pin does not ship — record that plainly. Given
  the ~25–30% ceiling on all expert-caching work and the 19%-share-for-63%-cost result at N=10, a
  null or negative outcome is a realistic result and reporting it honestly is the correct
  behavior, not a failure of execution.

- [ ] **Step 5: Revert scaffolding**, write the VERDICT, commit RESULTS, and update `todo.md`.

---

## Success Criteria

- Task 1's bias measurement is recorded with speed **and** quality, and the feature is confirmed
  to have actually engaged.
- If the pin is built: pinned entries survive budget pressure, `evict_until_free()` can still
  reclaim them, the starvation case refuses cleanly, and the pinned set can never exceed
  `max_fraction` of the cache.
- Everything defaults off; `pytest tests/ -q` green.
- A recorded VERDICT either way — including "neither lever is worth shipping," if that is what the
  numbers say.

## Deliberately NOT in scope

- **Startup pre-warm** — provably a no-op; the first prefill evicts it.
- **Persisting the histogram across runs** — would fix cold start; note it, do not build it.
- **Phase-2 live re-pinning mid-generation** — depends on this landing and proving out first.
