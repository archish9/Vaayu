# Fix #5: Deduplicate Expert Reads Across Speculative Draft Positions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first. Independent; smallest win — do last.**

**Goal:** Ensure that when a single forward pass processes multiple token
positions (speculative decoding verifies 5 draft tokens at once), each unique
expert is read from NVMe at most once, and confirm the mask/dispatch path already
executes a loaded expert for all positions that route to it.

**Architecture:** The run showed ~24 expert lookups per layer per step (≈ 3×
top_k), consistent with speculative decoding verifying ~5 draft positions in one
forward pass (`DRAFT_TOKENS = 5` in `test_generate.py`). `sparse_forward` already
computes `unique_experts` across all positions and loads each unique expert once —
so cross-position read dedup is *mostly* already correct. This fix (a) verifies
and locks that behavior with a test, and (b) removes any redundant per-position
work in the mask/gather path so a unique expert's weights are used for every
position that routed to it without re-reading.

**Tech Stack:** Python 3.12, PyTorch. Files: `vaayu/moe_patch.py`, `tests/`.

## Global Constraints

- Behavior-preserving on model outputs.
- top_k = 8, num_experts = 128, expert = 11.28 MB.
- The forward may process `B*S` positions where `S` > 1 during speculative
  verification. Dedup is across the flattened `B*S` positions.

---

## Background: exact current code

`vaayu/moe_patch.py`, `sparse_forward` (find via `grep -n "unique_experts" vaayu/moe_patch.py`):

```python
        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)
        router_res = router(flat_hidden)
        ...
        expert_indices_cpu = expert_indices.cpu()
        unique_experts: list[int] = expert_indices_cpu.flatten().unique().tolist()
        ...
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                misses.append(eidx)
        ...
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)
```

`unique_experts` is already deduped across all `B*S` positions, and each miss is
loaded once. So the **read** path is already dedup-correct. The remaining
opportunity is confirming the **compute** path runs each loaded expert once over
all its assigned positions (via `expert_mask`) rather than per-position.

---

## Task 1: Lock cross-position read dedup with a test

**Files:**
- Modify: `vaayu/moe_patch.py` (extract the dedup into a testable helper)
- Test: `tests/test_expert_dedup.py` (create)

**Interfaces:**
- Produces: `unique_expert_indices(expert_indices_cpu) -> list[int]` — a
  module-level helper returning the sorted unique expert ids across all
  positions. Used by `sparse_forward`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_dedup.py`:

```python
"""Fix #5: unique experts are computed once across all draft positions."""
import torch
from vaayu.moe_patch import unique_expert_indices


def test_dedup_across_positions():
    # 5 positions (rows), top_k=3 each; overlapping experts across positions.
    idx = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [1, 4, 5],
        [1, 2, 5],
        [3, 4, 5],
    ])
    uniq = unique_expert_indices(idx)
    assert uniq == [1, 2, 3, 4, 5]  # 5 unique, not 15 per-position


def test_dedup_single_position():
    idx = torch.tensor([[7, 7, 8]])  # duplicate within a position too
    assert unique_expert_indices(idx) == [7, 8]
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_dedup.py -v`
Expected: FAIL — `unique_expert_indices` undefined.

- [ ] **Step 3: Add the helper and use it in `sparse_forward`**

Add module-level in `vaayu/moe_patch.py`:
```python
def unique_expert_indices(expert_indices_cpu) -> list[int]:
    """Fix #5: unique expert ids across ALL routed positions (flattened B*S x top_k),
    so each expert is read/loaded at most once per layer per forward pass, even
    when speculative decoding verifies multiple draft positions at once."""
    return sorted(int(x) for x in expert_indices_cpu.flatten().unique().tolist())
```

In `sparse_forward`, replace:
```python
        unique_experts: list[int] = expert_indices_cpu.flatten().unique().tolist()
```
with:
```python
        unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)
```

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_dedup.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_dedup.py vaayu/moe_patch.py
git commit -m "refactor: extract+test cross-position expert dedup (Fix #5)"
```

---

## Task 2: Verify the compute path runs each expert once over all its positions

**Files:**
- Read/inspect: `vaayu/moe_patch.py` (the expert-execution loop after
  `expert_mask` is built)
- Test: `tests/test_expert_dedup.py` (add an assertion-style guard if a redundancy
  is found)

**Interfaces:**
- Consumes: `expert_mask` shape `(num_experts, top_k, B*S)`.

- [ ] **Step 1: Locate the expert execution loop**

```bash
grep -n "expert_mask\|for eidx in unique_experts\|_make_bc_linear\|BC_LinearEXL3\|idx, top_x" vaayu/moe_patch.py
```
Read the loop that, for each expert, gathers the positions routed to it (via
`expert_mask` / `torch.where`) and runs the expert's GEMM once over that batch of
positions.

- [ ] **Step 2: Confirm single-execution-per-expert**

Verify the loop iterates over `unique_experts` (NOT over positions), gathers all
positions for that expert with something like
`idx, top_x = torch.where(expert_mask[eidx])`, runs the expert once on
`flat_hidden[top_x]`, and scatters results back. This is the standard HF MoE
pattern and means each loaded expert is executed once for all its positions.

- [ ] **Step 3: If (and only if) a per-position re-execution or re-read is found**

If the loop re-reads `expert_cache.get(...)` inside a per-position sub-loop, or
re-instantiates the expert linear per position, hoist that out so it happens once
per `eidx`. Add a regression test capturing the specific redundancy you found
(e.g. assert `expert_cache.get` call count == number of unique experts, using a
counting wrapper). Do not invent a redundancy that is not there — if Step 2 shows
the path is already single-execution, record that finding and skip to
verification. **This task may legitimately be a no-op with a documented finding.**

- [ ] **Step 4: Run full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit (only if code changed)**

```bash
git add vaayu/moe_patch.py tests/test_expert_dedup.py
git commit -m "perf: run each expert once per forward over all routed positions (Fix #5)"
```

---

## Verification (real run)

- [ ] **Step 1: Compare per-step lookup count before/after**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "this_token=" run.log | tail -10
```
Success criterion: with Fix #6's buckets, `expert_disk` per step should not
increase; the number of unique experts loaded per layer stays ≈ the true union
(not `positions × top_k`). This fix mostly *confirms* correctness and prevents a
regression; the measurable win is small unless Task 2 found a real redundancy.

## Gotchas

- **This is the smallest-impact fix.** The read path is likely already correct;
  the value is a locked test + confirming the compute path. Do not over-engineer.
- Do not confuse "positions" (B*S rows, possibly 5 during speculative verify) with
  "experts". Dedup is over experts, executed over positions.
- If speculative decoding is disabled (`drafter_model_path=None`), S=1 and this
  fix is a no-op — still correct, just nothing to dedup.
