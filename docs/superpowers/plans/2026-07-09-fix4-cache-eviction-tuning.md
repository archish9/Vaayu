# Fix #4: Resolve the Per-Layer-Cap vs Global-Budget Eviction Conflict

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first. Best applied AFTER Fix #1.**
>
> **Outcome (executed 2026-07-10, commits 1190434/f41b63b):** both invariants
> this fix targets were already satisfied by Fix #1 + earlier 2026-07-09 work —
> `ExpertRAMCache.get()` already propagated recency to the global LRU, and
> `compute_expert_budget`'s clamp already kept per-layer caps within the global
> budget for both target models. This fix landed as **regression tests only**,
> no production code changed. The measured `evictions ≈ 3× inserts` thrash is
> **Fix #9's** target (report §2.4, spec-decode working-set multiplier), not
> this fix's — do not read "no code change" as "skipped."

**Goal:** Stop the two eviction constraints (per-layer cap and global byte
budget) from fighting each other, which wastes cache capacity and lowers hit
rate. Give the LRU an explicit stickiness margin so recently-hot experts survive.

**Architecture:** `ExpertRAMCache` enforces two independent eviction rules: a
per-layer cap (`max_per_layer`) AND a global byte budget (`budget_bytes`). With
the current numbers these disagree: the per-layer caps allow `8 × 91 = 728`
experts, but the 7.5 GB global budget only affords `635` slots. So the global
budget evicts experts that the per-layer caps just admitted, and vice-versa —
cross-layer ping-pong. This fix makes the per-layer cap consistent with the
global budget and adds a "protect the most-recent N" rule so prefetched/just-used
experts are not the first evicted.

**Tech Stack:** Python 3.12. Files: `vaayu/persist/expert_cache.py`, `tests/`.

## Global Constraints

- **Model-agnostic:** all model numbers derived from `_moe_geometry` (Fix #1
  Task 0). The Qwen instance (11.28 MB / 91 layers / 128 experts / 8.02 GB) is
  shown only as illustration; the M2.5 instance (11.3 MB / 62 / 256 / 5.5 GB) must
  work through the same code. Never hardcode.
- The invariant this fix enforces is dimensionless:
  `max_per_layer × num_moe_layers × expert_bytes ≤ budget_bytes`. True for any
  model when `max_per_layer` is clamped by `compute_expert_budget` (Fix #1).
- Pure caching-policy change. Must not change what `get`/`put`/`has` return.
- Keep thread safety: all counter/state mutations under `self.lock`.
- The cache already has `hits`, `misses`, `evictions`, `reset_stats()`,
  `delta_stats()`, `stats()` (added 2026-07-09). Do not remove them.

---

## Background: exact current code

`vaayu/persist/expert_cache.py`, class `ExpertRAMCache`. Two constraints in
`put()` (per-layer mode, `max_per_layer > 0`):

```python
                # 1. Per-layer fairness: evict oldest within this layer
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self.current_usage -= sum(t.nbytes for t in evicted.values())
                    self._global_lru.pop((layer_idx, evicted_eidx), None)
                    self.evictions += 1

                # 2. Global budget enforcement: evict globally-oldest across ALL layers
                while self.current_usage + raw_size > self.budget_bytes:
                    if not self._evict_global_oldest():
                        return  # Can't free anything — refuse insert (OOM guard)
```

The measured problem (`run.log`): `evictions ≈ 3× inserts`, usage pegged at
`7.51/7.52 GB`. Even after Fix #1 raises the budget, the per-layer cap and global
budget can still disagree if `max_per_layer × moe_layers × expert_bytes ≠
budget`.

---

## The fix in one sentence

Set `max_per_layer` so that `max_per_layer × moe_layers × expert_bytes` is
**slightly below** `budget_bytes` (so the per-layer caps alone keep usage under
budget and the global-budget loop rarely fires), and skip the redundant
global-eviction pass when the per-layer cap already made room.

Worked per model (illustration only — code derives these):
- **Qwen** (budget 8.5 GB, 91 layers, 11.28 MB): `8.5/91/11.28MB = 8.4 → cap 8`,
  giving `8×91×11.28MB = 8.02 GB < 8.5 GB` ✓.
- **M2.5** (budget 8.5 GB, 62 layers, 11.3 MB): `8.5/62/11.3MB = 12.1 → cap 12`,
  giving `12×62×11.3MB = 8.21 GB < 8.5 GB` ✓ (and 12 > top_k 8 → real stickiness
  headroom, which is why M2.5 barely thrashes).

So the per-layer cap is already budget-consistent for both — the bug is that the
global-budget loop still runs and evicts *cross-layer* when a transient spike
(e.g. a step touching more experts in one layer via speculative draft positions,
see Fix #5) pushes usage up. Fix #4 makes the two rules cooperate for any model.

---

## Task 1: Add a "recently used" protection window to global eviction

**Files:**
- Modify: `vaayu/persist/expert_cache.py`
- Test: `tests/test_expert_cache_eviction.py` (create)

**Interfaces:**
- Consumes: existing `ExpertRAMCache(budget_bytes, max_per_layer)`.
- Produces: global eviction never evicts an expert accessed within the current
  `put()` batch; a new `stats()`-visible counter is not required.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_cache_eviction.py`:

```python
"""Fix #4: per-layer cap and global budget must cooperate, not thrash."""
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _expert(nbytes=11_280_000):
    # one tensor of the given byte size (approx real 11.28 MB expert)
    return {"w": torch.zeros(nbytes, dtype=torch.uint8)}


def test_per_layer_cap_consistent_with_budget_no_cross_layer_thrash():
    # Budget sized so 2 layers x 4 experts fit exactly with headroom.
    per_expert = 1_000_000
    budget = 9 * 4 * per_expert  # room for 9 experts across all layers
    c = ExpertRAMCache(budget_bytes=budget, max_per_layer=4)

    # Fill layer 0 and layer 1 to their caps (4 each = 8 total, under budget)
    for lid in (0, 1):
        for e in range(4):
            c.put(lid, e, _expert(per_expert))

    c.reset_stats()

    # Re-access all 8 — should ALL be hits, zero evictions (they fit).
    for lid in (0, 1):
        for e in range(4):
            assert c.get(lid, e) is not None
    assert c.evictions == 0, f"no eviction expected, got {c.evictions}"
    assert c.hits == 8


def test_recent_access_survives_global_eviction():
    # When global budget forces eviction, the just-accessed expert must NOT be
    # the one evicted.
    per_expert = 1_000_000
    budget = 4 * per_expert  # only 4 fit globally
    c = ExpertRAMCache(budget_bytes=budget, max_per_layer=8)  # per-layer allows more
    for e in range(4):
        c.put(0, e, _expert(per_expert))
    # Touch expert 0 so it is most-recent
    assert c.get(0, 0) is not None
    # Insert a 5th expert → forces one global eviction. Expert 0 (just touched)
    # must survive; expert 1 (now oldest) should go.
    c.put(0, 4, _expert(per_expert))
    assert c.get(0, 0) is not None, "recently-accessed expert must survive"
    assert c.get(0, 1) is None, "oldest expert should have been evicted"
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_cache_eviction.py -v`
Expected: `test_recent_access_survives_global_eviction` may already pass (LRU
moves accessed keys to end). `test_per_layer_cap_consistent...` should pass too
if the logic is sound. **If both pass, the current code is already correct on
these cases** — proceed to Task 2 (the real fix is the sizing consistency + the
`get()` touching global LRU, verified below). If either fails, fix as in Step 3.

- [ ] **Step 3: Ensure `get()` updates global LRU order in per-layer mode**

Confirm `get()` (per-layer branch) already calls `self._global_lru.move_to_end(key)`
on a hit. Locate:
```bash
grep -n "move_to_end" vaayu/persist/expert_cache.py
```
The per-layer `get()` branch must move the key to the end of BOTH `lc` and
`self._global_lru`. If the `_global_lru.move_to_end` call is missing, add it so
global eviction respects recent access:

```python
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                if lc and expert_idx in lc:
                    lc.move_to_end(expert_idx)
                    key = (layer_idx, expert_idx)
                    if key in self._global_lru:
                        self._global_lru.move_to_end(key)
                    self.hits += 1
                    return dict(lc[expert_idx])
                self.misses += 1
                return None
```
(As of 2026-07-09 this is already present. The test documents/locks it.)

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_cache_eviction.py -v`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_cache_eviction.py vaayu/persist/expert_cache.py
git commit -m "test: lock per-layer/global LRU cooperation (Fix #4)"
```

---

## Task 2: Size `max_per_layer` to the budget in `compute_expert_budget`

**Files:**
- Modify: `vaayu/vaayu_base.py` (`compute_expert_budget` if Fix #1 landed; else the
  inline `_max_per_layer` block)
- Test: extend `tests/test_expert_budget_sizing.py` (or create if Fix #1 not done)

**Interfaces:**
- Produces: `max_per_layer` such that
  `max_per_layer × num_moe_layers × expert_bytes <= budget_bytes` (per-layer caps
  never collectively exceed the global budget → global-eviction loop is a rare
  safety net, not the steady-state path).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_expert_budget_sizing.py`:

```python
import pytest
from vaayu.vaayu_base import compute_expert_budget
GB = 1024 ** 3
MB = 1024 ** 2

@pytest.mark.parametrize("layers,experts,expert_mb,ws_gb", [
    (91, 128, 11.28, 8.02),   # Qwen3-235B
    (62, 256, 11.30, 5.50),   # MiniMax-M2.5
])
def test_max_per_layer_never_exceeds_global_budget(layers, experts, expert_mb, ws_gb):
    expert_bytes = int(expert_mb * MB)
    budget, ne, mpl = compute_expert_budget(
        available_bytes=int(26.5 * GB),
        working_set_bytes=int(ws_gb * GB),
        num_moe_layers=layers, expert_bytes=expert_bytes,
        moe_top_k=8, moe_num_experts=experts,
    )
    # Collective per-layer capacity must fit within the global budget (any model).
    collective = mpl * layers * expert_bytes
    assert collective <= budget, (
        f"per-layer caps ({collective/GB:.2f}GB) exceed global budget "
        f"({budget/GB:.2f}GB) → cross-layer thrash"
    )
    assert mpl >= 8  # never below top_k
```

- [ ] **Step 2: Run to verify**

Run: `source venv/bin/activate && python -m pytest "tests/test_expert_budget_sizing.py::test_max_per_layer_never_exceeds_global_budget" -v`
Expected: PASS for both parametrized models if Fix #1's `compute_expert_budget`
clamps `max_per_layer` via `per_layer_bytes // expert_bytes`
(Qwen: `8.5GB/91/11.28MB = 8.4 → 8`, `8×91×11.28MB = 8.02GB ≤ 8.5GB` ✓;
M2.5: `8.5GB/62/11.3MB = 12.1 → 12`, `12×62×11.3MB = 8.21GB ≤ 8.5GB` ✓). If it
FAILS, tighten the clamp in Step 3.

- [ ] **Step 3: (Only if Step 2 failed) tighten the clamp**

In `compute_expert_budget`, ensure (note: `expert_bytes`, not `expert_mb`):
```python
    per_layer_bytes = expert_budget // max(1, num_moe_layers)
    max_per_layer = max(
        moe_top_k,
        min(moe_num_experts, int(per_layer_bytes // max(1, expert_bytes))),
    )
    # Guard: collective per-layer capacity must not exceed global budget.
    while max_per_layer > moe_top_k and (max_per_layer * num_moe_layers * expert_bytes) > expert_budget:
        max_per_layer -= 1
```

- [ ] **Step 4: Run full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_budget_sizing.py vaayu/vaayu_base.py
git commit -m "fix: keep per-layer caps within global budget (Fix #4)"
```

---

## Verification (real run)

- [ ] **Step 1: Run and inspect eviction ratio**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "this_token=" run.log | tail -10
```
Success criterion: `Kevict` per step is no longer ~3× the miss count. Ideally
`evict ≈ miss` (each real churn-in evicts one churn-out), and `hit_rate` is
higher than the Fix-#1-only number. Combined with Fix #6's `expert_disk` bucket,
`expert_disk` per step should drop.

## Gotchas

- Do not set `max_per_layer` below `top_k` (8) — a layer must be able to cache its
  entire active set or it re-misses every token by construction.
- This fix's standalone impact is modest (a few hit-rate points). Its real value
  is preventing Fix #1's larger budget from being squandered on cross-layer
  ping-pong. Run it *after* Fix #1.
- If Fix #1 was NOT applied, the inline `_max_per_layer` block in `vaayu_base.py`
  is the target instead of `compute_expert_budget` — same clamp logic.
