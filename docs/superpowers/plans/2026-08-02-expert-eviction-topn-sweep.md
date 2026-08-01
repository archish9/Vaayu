# Expert-Eviction `top_n` Sweep — Implementation Plan (Plan F)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything you need is
> in this document.

**Goal:** Find out *where* MoE expert-cache eviction pressure actually concentrates, so a future
hot-expert pin can be sized on evidence instead of a guess. Produces one number that a later plan
needs: **the smallest N for which pinning the top-N experts per layer captures a worthwhile share
of budget-pressure evictions.**

**Architecture:** Almost no new code. The measurement infrastructure already exists and ships dark
behind `enable_expert_usage`. This plan adds a small sweep helper that renders the *same*
histogram snapshot at several `top_n` values, then runs it on-device against both models.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

---

## Why this exists — what is already known

A prior measurement (2026-08-01, MiniMax-M2.7, 62 layers × 256 experts, top_k=8, 256 output
tokens) established:

- **126,480** expert routes during decode; **78,726** budget-pressure evictions from the VRAM
  expert cache.
- Pinning each layer's **top-10** experts by route count would cover only **15,255** of those
  evictions — **19%**. The other **81%** sit in the long tail.
- The VRAM cache holds ~983 of 15,872 total experts (**6.2%**) and evicts roughly as many experts
  as it misses per token. Its per-token working set (top_k × layers × expert_bytes = 4.4 GB) fits
  inside the 8.7 GB budget, yet it still thrashes — because the *union* of experts routed across
  many tokens vastly exceeds any single token's working set.
- Decode cost split, per token at 1.48 tok/s: `expert_disk` 0.1–0.3 s, `compute` 0.3–0.5 s. So
  **all** expert-caching work is chasing at most ~25–30% of token time.

Only N=10 has ever been measured. That single point cannot distinguish "evictions are spread
uniformly, pinning is hopeless" from "N=3 already captures most of it, and N=10 was simply the
wrong cut." **This plan produces the curve.**

## The decision this feeds

A later plan builds the pin. Pinning N experts per layer costs
`N × num_moe_layers × expert_bytes` of VRAM taken from the same cache it is trying to help — on
MiniMax, N=10 is 620 × 8.9 MB = **5.5 GB of an 8.7 GB budget (63%)**. Any N whose eviction share
is smaller than its budget share is self-defeating: it would evict more from the tail than it
saves at the head.

So the output of this plan is a table of `N → eviction share → VRAM cost`, and a recommendation of
either a specific N or "no N works — do not build the pin."

---

## Global Constraints

- **Measurement only.** This plan changes no engine behavior. The one code change is a reporting
  helper. `enable_expert_usage` stays default-False.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No layer count, expert count, or top_k
  literal in code. The sweep's N values are a caller-supplied list, not a hardcoded constant
  inside library code.
- **Do not regress decode.** The instrumentation measured 1.48 tok/s vs a ~1.47 floor on MiniMax
  when last run. If your run is materially below that, investigate before trusting the numbers.
- **Surgical (CLAUDE.md §3).** Every changed line traces to this plan.
- Run tests with `source venv/bin/activate`. Record the baseline suite count before starting;
  this document does not hardcode it.

## What already exists (read these before writing anything)

- `vaayu/expert_usage.py` — `ExpertUsageTracker` (device-side `[n_rows, n_experts]` route
  counter, `record()`, `snapshot()`, `report(vram_cache, top_n, covered_layers)`) and the pure
  `format_usage_report(counts, evicted_keys, top_n, covered_layers)`.
- `ExpertVRAMCache.evicted_keys` — a `Counter` keyed `(layer_idx, expert_idx)`, populated **only**
  by budget-pressure evictions in `put()`. `evict_until_free()` (the wholesale prefill wipe,
  measured at 1362 entries in one pass) is deliberately excluded; conflating them would make every
  expert look like a pin candidate.
- `enable_expert_usage` — constructor flag on the model, default False.
- `model._vram_cacheable_layers` — the set of layers the VRAM cache actually covers, or `None`
  meaning "all". **Pass this as `covered_layers`.** A layer that is never cached records zero
  evictions for a structural reason; without this the report labels it "resident, no pin needed",
  which is the exact label that would wrongly retire the pin. It is `None` on MiniMax (working set
  fits) but a restricted prefix on Qwen (~72 of 91 layers).

**Key efficiency note:** you do **not** need one GPU run per N. The tracker holds the counts and
the cache holds the eviction attribution; rendering at a different `top_n` is pure formatting over
the same snapshot. **One run per model, many reports.**

---

## Task 1: Sweep helper (TDD)

**Files:** Modify `vaayu/expert_usage.py`; test in `tests/test_expert_usage.py`.

**Interface produced:** `format_eviction_sweep(counts, evicted_keys, n_values, expert_bytes=None, cache_budget_bytes=None, covered_layers=None) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_usage.py`, matching the style already there.

```python
from vaayu.expert_usage import format_eviction_sweep


def test_sweep_share_is_monotonic_in_n():
    """A larger cut can only cover more evictions, never fewer."""
    counts = torch.zeros((1, 6), dtype=torch.int32)
    for e, r in enumerate([100, 90, 80, 70, 60, 50]):
        counts[0, e] = r
    ev = {(0, 0): 5, (0, 3): 20, (0, 5): 50}
    out = format_eviction_sweep(counts, ev, n_values=[1, 3, 6])
    shares = [int(tok.rstrip('%')) for tok in out.split() if tok.endswith('%')]
    assert shares == sorted(shares)


def test_sweep_reports_full_coverage_at_max_n():
    counts = torch.zeros((1, 3), dtype=torch.int32)
    counts[0, 0], counts[0, 1], counts[0, 2] = 10, 9, 8
    out = format_eviction_sweep(counts, {(0, 0): 4, (0, 2): 6}, n_values=[3])
    assert "10/10" in out          # every eviction accounted for at N = n_experts
    assert "100%" in out


def test_sweep_includes_vram_cost_when_geometry_supplied():
    """The cost side is the whole point: a share smaller than the budget share
    is self-defeating."""
    counts = torch.zeros((2, 4), dtype=torch.int32)
    counts[0, 0] = 10
    counts[1, 0] = 10
    out = format_eviction_sweep(counts, {(0, 0): 1}, n_values=[1],
                                expert_bytes=10 * 1024 ** 2,
                                cache_budget_bytes=100 * 1024 ** 2)
    # 1 expert x 2 layers x 10MB = 20MB of a 100MB budget
    assert "20" in out and "%" in out


def test_sweep_without_geometry_omits_cost_but_does_not_crash():
    counts = torch.zeros((1, 2), dtype=torch.int32)
    counts[0, 0] = 5
    out = format_eviction_sweep(counts, {}, n_values=[1])
    assert isinstance(out, str) and "N=1" in out


def test_sweep_on_empty_histogram():
    out = format_eviction_sweep(torch.zeros((2, 4), dtype=torch.int32), {}, n_values=[1, 2])
    assert isinstance(out, str)
```

- [ ] **Step 2: Run them, confirm they fail** on the missing import.

- [ ] **Step 3: Implement `format_eviction_sweep`** in `vaayu/expert_usage.py`.

Requirements:
- Pure function. No CUDA, no model, no global state — same contract as `format_usage_report`.
- For each `N` in `n_values`: sum `evicted_keys` over the union of each layer's top-N experts by
  route count (skipping layers with zero routes, and layers not in `covered_layers` when that set
  is supplied), and report `covered/total (share%)`.
- When `expert_bytes` and `cache_budget_bytes` are both given, also report the VRAM the pin would
  consume — `N × (number of layers with routes) × expert_bytes` — as an absolute figure and as a
  percentage of the budget. Both are caller-supplied, so §5 is satisfied.
- Emit one line per N plus a header. Keep it grep-friendly, e.g. prefix `[Vaayu-Sweep]`.
- Reuse the top-N selection logic from `format_usage_report` rather than duplicating it — extract
  a small shared helper if that keeps it clean.

- [ ] **Step 4: Confirm the tests pass, then the full suite.**

- [ ] **Step 5: Commit** (`feat(expert-usage): top_n eviction-share sweep`).

---

## Task 2: Print the sweep from the benchmark harness

**Files:** Modify `test_generate.py` (repo root).

`test_generate.py` already ends with a guarded usage-report print. Extend that block to also print
the sweep.

- [ ] **Step 1: Extend the existing guarded block.** Find the block near the end of the file that
  begins `if getattr(model, "_expert_usage", None) is not None:` and add a sweep print beside the
  existing report. Pass, from the live objects:
  - `counts` — `model._expert_usage.snapshot()`
  - `evicted_keys` — `model._expert_vram_cache.evicted_keys`
  - `n_values` — `[1, 2, 3, 5, 10, 20, 40]`
  - `expert_bytes` and `cache_budget_bytes` — read from `model._moe_geometry["expert_bytes"]` and
    `model._expert_vram_cache.budget_bytes`
  - `covered_layers` — `getattr(model, "_vram_cacheable_layers", None)`

- [ ] **Step 2: Do NOT set `enable_expert_usage=True` in the committed file.** The block is inert
  when the flag is off; Task 3 turns it on temporarily and reverts.

- [ ] **Step 3: Verify it compiles** (`python -m py_compile test_generate.py`) and that the suite
  is still green. Commit.

---

## Task 3: On-device sweep, both models (operator-run)

Needs a CUDA GPU and real model weights. ~15 minutes per run.

**Files:** Create `docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep-RESULTS.md`.

- [ ] **Step 1: Enable the flag temporarily.** In `test_generate.py`'s
  `AutoModel.from_pretrained(...)` call add `enable_expert_usage=True`. This is scaffolding —
  Step 5 reverts it.

- [ ] **Step 2: Run on MiniMax.** Point `MODEL_PATH` at the MiniMax path (both paths are in the
  file, one commented) and **set a fresh `SESSION_ID`** — a persisted KV under another model's id
  would poison the run; the file has a comment warning about exactly this.

```bash
source venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u test_generate.py > run-sweep-minimax.log 2>&1
echo "EXIT=$?"
grep -E "Vaayu-Sweep|Decode throughput|OutOfMemoryError" run-sweep-minimax.log
```

- [ ] **Step 3: Run on the second model** (Qwen), fresh `SESSION_ID` again. Qwen's VRAM cache is
  restricted to a layer prefix, so `covered_layers` will be a real set here, not `None` — confirm
  the report reflects that rather than labelling uncovered layers "resident".

- [ ] **Step 4: Sanity-check decode** in both logs against the floors (MiniMax ≈1.47, Qwen ≈0.82
  tok/s). The histogram hook adds one `index_add_` per MoE layer per token; last measured it cost
  nothing detectable. If decode is materially down, say so.

- [ ] **Step 5: Revert scaffolding.** `git checkout test_generate.py`; confirm it does not appear
  in `git status --short`.

- [ ] **Step 6: Write RESULTS** containing, per model:
  - The full sweep table: `N → evictions covered / total (share%) → pin VRAM cost (GB, % of budget)`.
  - Total routes and total budget-pressure evictions.
  - Decode tok/s vs floor.
  - A note that `run-sweep-*.log` files are operator-local and untracked.
  - **A one-line VERDICT** naming either a recommended N, or "no N is worth it."

  The decision rule, stated plainly so the next plan can act on it: **a candidate N is only viable
  if its eviction share materially exceeds its share of the cache budget.** N=10 on MiniMax was
  19% share for 63% of the budget — decisively not viable. If no N clears that bar on either
  model, the honest answer is that the pin should not be built, and that is a successful outcome
  of this plan, not a failure.

- [ ] **Step 7: Commit RESULTS and update `todo.md`** with the verdict.

---

## Success Criteria

- `format_eviction_sweep` is pure, tested, and reports both eviction share and VRAM cost per N.
- One on-device run per model produces the full curve — no re-running per N.
- RESULTS names a recommended N or states that none is viable, with the share-vs-cost comparison
  behind it.
- `pytest tests/ -q` green; `test_generate.py` reverted; `enable_expert_usage` still default False.

## Deliberately NOT in scope

- Building the pin. That is the next plan, and it depends on this one's N.
- Changing routing, eviction policy, or cache sizing.
- Persisting the histogram across runs — one run answers this question.
