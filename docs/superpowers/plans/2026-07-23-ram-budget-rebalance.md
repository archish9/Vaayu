# RAM Budget Rebalance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Everything needed is in this file. No prior session context required.
>
> **PREREQUISITE:** run `docs/superpowers/plans/2026-07-23-ram-accounting-diagnostic.md` first. This plan's Task 2 consumes its "reclaimable RAM" figure. You *can* run this plan without it by measuring conservatively, but you will be sizing the budget from an estimate rather than a measurement — and this codebase has repeatedly burned people who did that.

**Goal:** Reclaim RAM from the expert RAM cache (currently ~9.13GB returning a 5% hit rate) so the cross-turn prefix cache gets a real budget, letting multiple concurrent sessions stay cached under parallel fan-out.

**Architecture:** One sizing function (`compute_expert_budget`) currently hands the expert RAM cache nearly all free RAM, and the prefix cache — constructed later, from whatever is left — gets zero. The expert cache's hit rate collapsed to ~5% when the VRAM expert cache started absorbing the hot experts, so it is now the largest low-value allocation on the box. This plan right-sizes it and verifies the freed RAM reaches the prefix cache, with decode throughput as the guard.

**Tech Stack:** Python, PyTorch, pytest.

## Background (read first — you have no prior context)

Vaayu is a low-resource LLM inference engine (`CLAUDE.md`). Relevant facts, all measured:

**The box:** 33.40GB RAM, 16GB GPU. Targets are MiniMax-M2.7 and Qwen3-235B, both EXL3-quantized MoE.

**Three RAM consumers, sized in construction order, first-come-first-served:**

1. **Expert RAM cache** — `compute_expert_budget` in `vaayu/vaayu_base.py` (search `def compute_expert_budget`) sizes it `available - overhead`, capped by `_EXPERT_CACHE_OOM_CEILING` (8.5GB). Backed by a pinned slab arena of the same budget.
2. **VRAM expert cache** — GPU-side, separate, not in scope.
3. **Cross-turn prefix cache** — `compute_prefix_cache_budget(free_ram)` returns `max(0, free_ram - _PREFIX_CACHE_RAM_RESERVE)` where the reserve is 12.88GB (`12 * 1024 ** 3`). Constructed AFTER the expert cache and arena have already taken their RAM.

**The measured outcome:** by the time the prefix cache measures, only ~6.5-8GB is free, which is below the 12.88GB reserve, so its budget is **0**.

**What a 0 budget actually costs — this is subtle and was misunderstood once already.** It does NOT disable the cache. `PrefixCacheStore.store()` in `vaayu/prefix_cache.py` evicts with `while self._current_bytes > self.budget_bytes and len(self._pool) > 1` — the `len(pool) > 1` guard means **one entry always survives regardless of budget**. So a single session's cross-turn reuse works fine at budget 0 (measured: TTFT 108s → 10s on a 2000-token shared prefix). What a 0 budget costs is **concurrency**: the pool holds exactly one session, so under parallel fan-out (CLAUDE.md §0 — concurrent sub-agents, each its own `session_id`) every stream but the most recent is evicted.

**Why the expert RAM cache is the right thing to shrink:** its hit rate fell from ~34% to **~5%** once the VRAM expert cache started serving the hot experts. It now holds ~9.13GB to serve a cold tail. It is the largest low-value allocation on the box.

**A rejected alternative, so you don't re-derive it:** shrinking the *slab arena* alone cannot fund the prefix cache. Free RAM before the arena is ~15.6GB against a 12.88GB reserve, leaving only ~2.76GB of slack — so any arena ≥2.76GB zeroes the prefix budget, and an arena small enough to preserve it (~1.76GB) gives up nearly all of the arena's measured +55% decode win. The lever must be the *reserve's* composition and the *expert cache budget*, not the arena size.

**A second key fact about the reserve.** `_PREFIX_CACHE_RAM_RESERVE`'s own comment says it reserves RAM "for the OS + **the expert RAM cache** + the io_uring pools." But by the time the prefix cache measures, the expert cache and arena are ALREADY ALLOCATED and therefore already excluded from `available`. So the reserve double-counts them. A prior experiment lowering the reserve 12GB→6GB was committed and then reverted with no stated reason; a later reproduction found 6GB completes cleanly with no OOM (kernel OOM count 0, peak RSS 26.63GB of 33.40GB), suggesting the revert was precautionary rather than a confirmed failure. **Treat that as evidence, not proof** — the reproduction was a single-turn benchmark, which under-stresses the prefix pool.

## Global Constraints

- **No model-specific hardcoded numbers** (CLAUDE.md §5). Budgets derive from measured free RAM, `_moe_geometry`, and hardware constants. `_EXPERT_CACHE_OOM_CEILING` (8.5GB), `_PREFIX_CACHE_RAM_RESERVE`, and the 33GB/16GB limits are hardware constants and may stay literal.
- **Do not regress decode throughput.** Baselines to beat, both measured: MiniMax **~1.47 tok/s**, Qwen **~0.85 tok/s** on `test_generate.py` with 256 new tokens. Shrinking the expert RAM cache could in principle raise disk load; the guard is that its hit rate is already only ~5%.
- **Do not break the multi-session prefill fix.** A separate fix (pinned-arena eviction + `expandable_segments`) makes two sequential large-context sessions work; `test_multiturn.py` must still complete without CUDA OOM.
- **Memory safety is the kill criterion.** This box has a history of OOM-kills. Any change that raises peak RSS materially, or produces a non-zero kernel OOM count, is a stop-and-revert.
- Run tests with `source venv/bin/activate` first; `pytest tests/ -q` (373 tests at time of writing) must stay green.

---

### Task 1: Make the RAM budget split observable and tested

Before changing any sizing, pin the current behavior in tests so a regression is visible.

**Files:**
- Test: `tests/test_ram_budget_split.py` (create)

**Interfaces:**
- Consumes: `compute_expert_budget(available_bytes, working_set_bytes, num_moe_layers, expert_bytes, moe_top_k, moe_num_experts) -> (expert_budget, ne_budget, max_per_layer)` and `compute_prefix_cache_budget(free_ram_bytes) -> int`, both in `vaayu/vaayu_base.py`.
- Produces: no new code — characterization tests only.

- [ ] **Step 1: Write characterization tests**

Create `tests/test_ram_budget_split.py`:

```python
"""Characterizes how free RAM is split between the expert cache and the prefix
cache. These tests document CURRENT behavior so the rebalance is a visible,
intentional change rather than an accident."""
import pytest
from vaayu.vaayu_base import (
    compute_expert_budget, compute_prefix_cache_budget,
    _PREFIX_CACHE_RAM_RESERVE, _EXPERT_CACHE_OOM_CEILING,
)

GB = 1024 ** 3

# Geometry stand-ins for the two on-device targets. Values are illustrative
# shapes, not asserted model facts — the point is that BOTH are exercised.
QWEN_LIKE = dict(working_set_bytes=int(8.9 * GB), num_moe_layers=94,
                 expert_bytes=int(11.28 * 1e6), moe_top_k=8, moe_num_experts=128)
MINIMAX_LIKE = dict(working_set_bytes=int(4.4 * GB), num_moe_layers=62,
                    expert_bytes=int(8.9 * 1e6), moe_top_k=8, moe_num_experts=256)


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_expert_budget_is_capped_by_the_oom_ceiling(geo):
    """With plenty of free RAM the expert cache stops at the hardware ceiling."""
    expert, _ne, _mpl = compute_expert_budget(available_bytes=30 * GB, **geo)
    assert expert <= _EXPERT_CACHE_OOM_CEILING


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_expert_budget_shrinks_when_ram_is_tight(geo):
    """The budget tracks available RAM below the ceiling."""
    big, _, _ = compute_expert_budget(available_bytes=30 * GB, **geo)
    small, _, _ = compute_expert_budget(available_bytes=6 * GB, **geo)
    assert small < big


def test_prefix_budget_is_zero_below_the_reserve():
    """CURRENT behavior: free RAM under the reserve yields a 0 budget."""
    assert compute_prefix_cache_budget(_PREFIX_CACHE_RAM_RESERVE - 1) == 0
    assert compute_prefix_cache_budget(0) == 0


def test_prefix_budget_is_free_minus_reserve_above_it():
    free = _PREFIX_CACHE_RAM_RESERVE + 3 * GB
    assert compute_prefix_cache_budget(free) == 3 * GB


@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_the_two_budgets_together_document_the_starvation(geo):
    """The whole problem in one assertion: after the expert cache takes its
    share of a realistic free-RAM figure, the prefix cache gets nothing."""
    free_at_start = 24 * GB          # measured: ~24.5GB free at cache-sizing
    expert, ne, _ = compute_expert_budget(available_bytes=free_at_start, **geo)
    # The arena mirrors the expert budget, so both come out of free RAM.
    free_for_prefix = free_at_start - expert - ne
    assert compute_prefix_cache_budget(free_for_prefix) == 0
```

- [ ] **Step 2: Run them**

```bash
source venv/bin/activate
pytest tests/test_ram_budget_split.py -v
```

Expected: all PASS — they characterize existing behavior. If any fails, the code has already changed from what this plan assumes; **stop and report** rather than editing the test to match.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ram_budget_split.py
git commit -m "test(ram): characterize the expert-vs-prefix RAM budget split

Documents current behavior before rebalancing: the expert RAM cache takes its
share first and the prefix cache, sized from what remains against a 12.88GB
reserve, gets 0.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Rebalance the budgets

**Files:**
- Modify: `vaayu/vaayu_base.py` — `compute_expert_budget` and/or `_PREFIX_CACHE_RAM_RESERVE`
- Test: `tests/test_ram_budget_split.py`

**Interfaces:**
- Produces: no new function names. Changed sizing behavior in the two existing budget functions.

**The design decision is yours to make from evidence, and there are two defensible levers.** Read both, pick one, and record the reasoning in the commit message. Do NOT apply both at once — you will not be able to attribute the result.

**Lever A — fix the reserve's double-count (recommended).** `_PREFIX_CACHE_RAM_RESERVE` reserves for "OS + expert RAM cache + io_uring pools", but the expert cache and arena are already allocated (and thus already excluded from `available`) by the time the prefix cache measures. Remove the expert-cache component, leaving OS + io_uring headroom. Redefine the constant with a comment naming exactly what it now covers, and derive the value rather than guessing: it should cover the OS working set plus the io_uring pools (a few hundred MB, see `get_expert_pool_size` in `vaayu/persist/safetensor_model_persister.py`), plus a safety margin for post-prefix-cache allocations — notably the KV cache, which grows during generation and is allocated after this point.

**Lever B — cap the expert RAM cache by measured value.** Its hit rate is ~5%, so most of its 9.13GB is dead weight. Cap it lower (the working set is a natural derived bound — see `_moe_geometry["working_set_bytes"]`, ~4.4GB for MiniMax) and let the freed RAM flow to the prefix cache. This is more model-adaptive but changes expert-cache behavior, so it carries more decode risk.

- [ ] **Step 1: Write the failing test for your chosen lever**

Add to `tests/test_ram_budget_split.py` — this test must FAIL before your change and PASS after. Adapt the numbers to your lever:

```python
@pytest.mark.parametrize("geo", [QWEN_LIKE, MINIMAX_LIKE], ids=["qwen", "minimax"])
def test_prefix_cache_gets_a_real_budget_after_rebalance(geo):
    """After the rebalance, a realistic post-expert-cache free-RAM figure must
    leave the prefix cache enough for MORE THAN ONE resident session — the whole
    point, since budget 0 already serves one (the pool keeps >=1 entry)."""
    free_at_start = 24 * GB
    expert, ne, _ = compute_expert_budget(available_bytes=free_at_start, **geo)
    free_for_prefix = free_at_start - expert - ne
    budget = compute_prefix_cache_budget(free_for_prefix)
    assert budget > 0, "prefix cache still starved"
    # A ~2000-token KV entry is on the order of a few hundred MB; require room
    # for at least two so concurrency is genuinely served, not just one entry.
    assert budget >= 1 * GB
```

- [ ] **Step 2: Run to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_ram_budget_split.py -k after_rebalance -v
```

Expected: FAIL on `assert budget > 0`.

- [ ] **Step 3: Apply your chosen lever**

Make the change in `vaayu/vaayu_base.py`. Requirements:
- Update the changed constant's or function's comment to state what it now covers and why, including that the expert cache is already excluded from `available` at that point (that reasoning is the whole justification and must not be lost).
- Keep it model-agnostic — no model numbers.
- Do not touch `_EXPERT_CACHE_OOM_CEILING` unless Lever B specifically requires it; it is a validated hardware OOM limit.

- [ ] **Step 4: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_ram_budget_split.py -v
pytest tests/ -q
```

Expected: the new test PASSES; **the characterization tests from Task 1 that assert the OLD starvation will now fail** — that is the intended, visible behavior change. Update `test_the_two_budgets_together_document_the_starvation` to assert the new behavior, and rename it accordingly (e.g. `test_the_two_budgets_leave_room_for_the_prefix_cache`). Do not delete it. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_ram_budget_split.py
git commit -m "feat(ram): rebalance so the prefix cache gets a real budget

<state which lever you chose and the evidence for it>

The expert RAM cache's hit rate fell to ~5% once the VRAM expert cache absorbed
the hot experts, making it the largest low-value allocation on the box, while
the prefix cache got 0 -- which collapses concurrent sessions to one resident
entry under parallel fan-out.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: On-device verification

Run by the operator. Needs the real models and the GPU. **Memory safety is the kill criterion — check it before reading any performance number.**

- [ ] **Step 1: Record the kernel OOM baseline**

```bash
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

Note the number. It must not increase across any run below.

- [ ] **Step 2: MiniMax single-stream — decode must not regress**

Fresh `SESSION_ID`, `MAX_NEW_TOKENS = 256`, `MODEL_PATH` = MiniMax.

```bash
source venv/bin/activate
python -u test_generate.py > run-rebalance-minimax.log 2>&1
grep -E "prefix cache ENABLED|Decode throughput|Time to first token" run-rebalance-minimax.log
grep -oE "rss=[0-9.]+GB" run-rebalance-minimax.log | tail -1
grep -oE "expert_cache=[^ ]+ hit_rate=[0-9]+%" run-rebalance-minimax.log | tail -1
```

Success: `prefix cache ENABLED: <non-zero> GB budget`; decode ≥ ~1.47 tok/s (within run-to-run variance, which is real on this stack — a <10% delta from one sample is not a regression); peak RSS not materially above ~22GB; kernel OOM count unchanged.

**If decode dropped materially,** the expert RAM cache was doing more than its 5% hit rate suggested — record the number and reconsider the lever rather than tuning blindly.

- [ ] **Step 3: Qwen single-stream — the actively-tested path**

Fresh `SESSION_ID`, `MODEL_PATH` = Qwen.

```bash
source venv/bin/activate
python -u test_generate.py > run-rebalance-qwen.log 2>&1
grep -E "prefix cache ENABLED|Decode throughput" run-rebalance-qwen.log
grep -oE "rss=[0-9.]+GB" run-rebalance-qwen.log | tail -1
```

Success: decode ≈0.85 tok/s, prefix budget non-zero, RSS not materially up.

- [ ] **Step 4: Multi-session — prove the concurrency win, the actual goal**

`test_multiturn.py` at the repo root runs K interleaved sessions across R rounds; each session shares a large system prefix and has its own `session_id`. Set `K_SESSIONS = 3`, `R_ROUNDS = 2`, `MODEL_PATH` = MiniMax.

Round 1 stores all K cold (MISS). Round 2 re-runs each — a session HITs only if the budget kept it resident.

```bash
source venv/bin/activate
python -u test_multiturn.py > run-rebalance-multiturn.log 2>&1
grep -E "prefix cache ENABLED|Vaayu-Prefix|R1 S|=== done|OutOfMemory" run-rebalance-multiturn.log
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

**This is the headline result.** Before the rebalance, budget 0 means only the LAST session survives to round 2 — expect 1 HIT and K-1 MISS. After, expect round 2 HITs for as many sessions as the budget holds. Count the round-2 HITs and report it as "N of K sessions stayed cached."

Also confirm: run completes (no CUDA OOM — the separate prefill-headroom fix must still hold), kernel OOM count unchanged.

- [ ] **Step 5: Record results**

Append a results section to `PERF-BRIEF-minimax-vram-cache.md`: prefix budget before (0) and after; round-2 HIT count out of K; decode tok/s both models vs the 1.47/0.85 baselines; peak RSS; kernel OOM count. State plainly if the concurrency win did not materialize.

- [ ] **Step 6: Commit**

```bash
git add PERF-BRIEF-minimax-vram-cache.md test_generate.py test_multiturn.py
git commit -m "docs(ram): record rebalance results

<prefix budget 0 -> N GB; round-2 HITs M of K; decode unregressed>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Success Criteria

- Prefix cache budget is non-zero on both models.
- **Multiple concurrent sessions stay cached** — round-2 HIT count > 1 of K in Task 3 Step 4. This is the actual goal; a non-zero budget that still holds one session has not fixed anything.
- Decode throughput unregressed on both models (MiniMax ~1.47, Qwen ~0.85).
- Peak RSS not materially above ~22GB; kernel OOM count unchanged.
- `test_multiturn.py` still completes without CUDA OOM.
- Full suite green.
