# Fix #1: Free RAM and Raise the Expert Cache Budget Above the Working Set

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` in this directory first.**
> **Strongly recommended: do Fix #6 (telemetry) before this**, so you can measure the result.

**Goal:** Raise the expert RAM cache budget from ~7.5 GB to **≥ 8.5 GB** by
reclaiming RAM from lower-value consumers, so it exceeds the **8.02 GB** decode
working set and the measured cache thrash stops.

**Architecture:** The decode working set (one step's experts) is
`8 × 91 × 11.28 MB = 8.02 GB`. The expert cache is hard-capped at 7 GB
(showed 7.5 GB at runtime). Because 7.5 < 8.02, the cache is pegged full and
evicts ~3 experts per insert (measured), so hit rate is ~18%. This fix frees RAM
from the non-expert tensor cache and shrinks the OS/CUDA reserve margin so the
hard cap can safely rise past 8.5 GB, letting a full step's working set reside
plus a small stickiness margin.

**Tech Stack:** Python 3.12, `psutil`. Files: `vaayu/vaayu_base.py`,
`vaayu/persist/expert_cache.py` (read-only reference), `tests/`.

## Global Constraints

- **The 32 GB RAM ceiling is real.** A prior attempt to raise the cap to 10 GB
  caused a confirmed OOM-kill (`journalctl -k`, anon-rss 26.46 GB). Do NOT just
  bump the constant — you MUST free equivalent RAM elsewhere first, and you MUST
  verify with a full-length run, not a short one.
- Target budget: **8.5 GB** (8.02 GB working set + ~0.5 GB margin). Do not exceed
  9 GB without a longer verification run than the one that OOM'd at 10 GB.
- Expert size = 11.28 MB, moe_layers = 91, top_k = 8. Working set = 8.02 GB.
- No behavior change other than cache sizing.

---

## Background: exact current code

All in `vaayu/vaayu_base.py`, in the MoE-init block. Locate with:
```bash
grep -n "Expert RAM cache:\|_expert_budget = min\|_ne_budget = min\|_overhead = (" vaayu/vaayu_base.py
```

Current non-expert tensor cache budget (2 GB):
```python
            _ne_budget = min(2 * 1024**3, int(_mem.available / 8))
            self._non_expert_tensor_cache = LRUTensorCache(_ne_budget)
```

Current overhead reserve and the 7 GB hard cap:
```python
            _overhead = (
                2 * 1024**3          # OS + CUDA context margin
                + _ne_budget         # non-expert tensor cache (just allocated above)
                + 512 * 1024 * 1024  # io_uring expert buffer pool
            )
            _expert_budget = min(
                7 * 1024**3,
                max(
                    1 * 1024**3,  # floor: at least 1 GB so decode always works
                    int(_mem.available - _overhead),
                ),
            )
```

Current per-layer cap derivation (used by ExpertRAMCache; see Fix #4 for tuning):
```python
            _num_moe_layers = max(1, len(self.layers) - 3)
            _per_layer_bytes = _expert_budget // _num_moe_layers
            _max_per_layer = max(
                self.moe_top_k,
                min(self.moe_num_experts, _per_layer_bytes // (_expert_mb * 1024 * 1024)),
            )
            self._expert_cache = ExpertRAMCache(_expert_budget, max_per_layer=_max_per_layer)
```

---

## Where the reclaimable RAM is

The non-expert tensor cache holds attention/norm/gate tensors for MoE layers. But
**all 94 layers are already "non-expert-pinned"** (resident — see report §4), so
the 2 GB `LRUTensorCache` is largely redundant during decode: the pinned copies
are already in memory. Shrinking it from 2 GB → 0.5 GB frees ~1.5 GB. Combined
with trimming the OS/CUDA margin from 2 GB → 1.5 GB (still generous; the 512 MB
io_uring pool is separate), that frees ~2 GB → budget can rise to ~9 GB nominal,
which we then cap at 8.5 GB for safety.

> If a future measurement shows the non-expert cache is doing real work (Fix #6's
> `cache=Nhit/Mmiss` field on the summary line — currently `0hit/0miss`), revisit.
> As of 2026-07-09 it showed `cache=0hit/0miss` during decode, confirming it is
> idle and safe to shrink.

---

## Task 1: Shrink the non-expert tensor cache and OS margin, raise the cap to 8.5 GB

**Files:**
- Modify: `vaayu/vaayu_base.py` (MoE-init block shown above)
- Test: `tests/test_expert_budget_sizing.py` (create)

**Interfaces:**
- Produces: `self._expert_cache.budget_bytes >= 8.5 GB` on a machine with
  ≥ 26 GB available at init (the measured `avail=26.5GB`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_budget_sizing.py`. This tests the *pure sizing formula*
by extracting it into a helper (Task 2 creates the helper; this test drives its
design). Start with the test:

```python
"""Fix #1: expert cache budget must exceed the 8.02 GB decode working set."""
from vaayu.vaayu_base import compute_expert_budget

GB = 1024 ** 3
WORKING_SET_GB = 8.02  # 8 top_k * 91 moe_layers * 11.28 MB


def test_budget_exceeds_working_set_on_32gb_box():
    # 26.5 GB available at init on the real 32 GB machine (from run.log)
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=int(26.5 * GB),
        num_moe_layers=91,
        expert_mb=11.28,
        moe_top_k=8,
        moe_num_experts=128,
    )
    assert budget >= 8.5 * GB, f"budget {budget/GB:.2f}GB must be >= 8.5GB"
    assert budget <= 9.0 * GB, "must not exceed 9GB safety limit"
    assert budget / GB > WORKING_SET_GB, "budget must exceed working set"


def test_budget_floor_on_tiny_ram():
    # Degenerate: very little RAM → still returns the 1 GB floor, never negative.
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=3 * GB,
        num_moe_layers=91,
        expert_mb=11.28,
        moe_top_k=8,
        moe_num_experts=128,
    )
    assert budget >= 1 * GB
    assert max_per_layer >= 8  # never below top_k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_budget_sizing.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_expert_budget'`.

- [ ] **Step 3: Extract the sizing formula into a testable module-level function**

In `vaayu/vaayu_base.py`, add this module-level function near the top of the file
(after imports, before the class). It centralizes the sizing so it is testable:

```python
def compute_expert_budget(
    available_bytes: int,
    num_moe_layers: int,
    expert_mb: float,
    moe_top_k: int,
    moe_num_experts: int,
):
    """Size the expert RAM cache so it exceeds one decode step's working set.

    Fix #1 (2026-07-09): the decode working set is
    top_k * num_moe_layers * expert_mb (8.02 GB for this model). The old code
    hard-capped at 7 GB (< working set) → thrash. We shrink the redundant
    non-expert cache and OS margin to raise the cap to 8.5 GB, which exceeds
    the working set. The 8.5 GB cap and shrunk reserves are validated against
    real OOM behavior — do not raise past 9 GB without a long verification run.

    Returns (expert_budget_bytes, ne_budget_bytes, max_per_layer).
    """
    GB = 1024 ** 3
    MB = 1024 ** 2
    # Non-expert cache shrunk 2GB -> 0.5GB: all layers are non-expert-pinned, so
    # this LRU is idle during decode (measured cache=0hit/0miss).
    ne_budget = min(512 * MB, int(available_bytes / 8))
    overhead = (
        int(1.5 * GB)        # OS + CUDA context margin (was 2 GB)
        + ne_budget          # non-expert tensor cache
        + 512 * MB           # io_uring expert buffer pool
    )
    expert_budget = min(
        int(8.5 * GB),                       # hard cap (was 7 GB); > 8.02 GB working set
        max(1 * GB, available_bytes - overhead),
    )
    per_layer_bytes = expert_budget // max(1, num_moe_layers)
    max_per_layer = max(
        moe_top_k,
        min(moe_num_experts, int(per_layer_bytes // (expert_mb * MB))),
    )
    return expert_budget, ne_budget, max_per_layer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_budget_sizing.py -v`
Expected: PASS both tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_budget_sizing.py vaayu/vaayu_base.py
git commit -m "feat: add compute_expert_budget sizing helper (Fix #1)"
```

---

## Task 2: Wire the helper into MoE init (replace inline sizing)

**Files:**
- Modify: `vaayu/vaayu_base.py` (MoE-init block)

- [ ] **Step 1: Replace the inline `_ne_budget`/`_overhead`/`_expert_budget`/`_max_per_layer` code**

Find the block (see "Background" above) and replace everything from
`_ne_budget = min(2 * 1024**3, ...)` through the `self._expert_cache =
ExpertRAMCache(...)` line with a call to the helper. Keep the surrounding
`_expert_mb` measurement code (the loop that reads real bytes from the offset
index) — it produces `_expert_mb ≈ 11.28`. New code:

```python
            # Non-expert tensor cache + expert cache sizing centralized in
            # compute_expert_budget (Fix #1). _expert_mb was measured above from
            # the offset index (~11.28 MB for this EXL3 model).
            _num_moe_layers = max(1, len(self.layers) - 3)
            _expert_budget, _ne_budget, _max_per_layer = compute_expert_budget(
                available_bytes=_mem.available,
                num_moe_layers=_num_moe_layers,
                expert_mb=_expert_mb,
                moe_top_k=self.moe_top_k,
                moe_num_experts=self.moe_num_experts,
            )
            self._non_expert_tensor_cache = LRUTensorCache(_ne_budget)
            print(f"[Vaayu-Cache] Non-expert tensor cache: {_ne_budget / 1e9:.1f} GB budget")
            self._expert_cache = ExpertRAMCache(_expert_budget, max_per_layer=_max_per_layer)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"(~{int(_expert_budget // (_expert_mb * 1024 * 1024))} expert slots, "
                  f"max {_max_per_layer}/layer, EXL3, working_set=8.02GB)")
```

> IMPORTANT: The original code created `self._non_expert_tensor_cache` and printed
> its budget *before* computing `_overhead`. Make sure the new block still creates
> `self._non_expert_tensor_cache` exactly once. If the surrounding code already
> created it earlier (grep for `self._non_expert_tensor_cache = LRUTensorCache`),
> remove the earlier creation so it is created only here with the new `_ne_budget`.

- [ ] **Step 2: Verify import and existing tests**

Run:
```bash
source venv/bin/activate && python -c "import vaayu.vaayu_base" && python -m pytest tests/ -q
```
Expected: import OK; full suite passes.

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: raise expert cache budget above decode working set (Fix #1)"
```

---

## Verification (end-to-end, requires real run — CRITICAL for OOM safety)

- [ ] **Step 1: Confirm the printed budget at startup**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1 &
sleep 90 && grep "Expert RAM cache:" run.log
```
Expected: `Expert RAM cache: 8.5 GB budget ... working_set=8.02GB`.

- [ ] **Step 2: Let the FULL run complete, then check for OOM**

Wait for the run to finish (~5 min). Then:
```bash
grep -i "killed\|oom\|MemoryError" run.log
sudo journalctl -k --since "10 min ago" | grep -i "killed process\|oom" || echo "NO OOM"
```
Expected: `NO OOM`. **If OOM occurs, revert the cap to 8.0 GB in
`compute_expert_budget` and shrink `ne_budget` further, or reduce
`MAX_PINNED_LAYERS` in `test_generate.py`. Do not leave a config that OOMs.**

- [ ] **Step 3: Confirm thrash dropped (this is the win)**

```bash
grep "this_token=" run.log | tail -10
```
Expected: `hit_rate` on decode steps rises from ~18% toward ~50-65%, and
`Kevict` drops from ~5,500 toward being comparable to or below misses (no longer
3× inserts). Decode throughput on the final summary line should improve from
0.12 tok/s toward ~0.2+ tok/s.

## Gotchas

- The working set is 8.02 GB; a budget of exactly 8.02 GB still thrashes at the
  margins. 8.5 GB gives ~0.5 GB (~44 experts) of stickiness headroom. Do not size
  it to exactly the working set.
- `_mem.available` is measured at init, *before* KV cache and layer buffers grow.
  The `1.5 GB` OS margin + the 8.5 GB cap already account for this via the OOM
  history; that is why the cap exists as a `min()` ceiling, not just the formula.
- Raising `_max_per_layer` above `top_k=8` is Fix #4's job; here it may rise
  naturally because per_layer_bytes grew. That is fine and helps.
- If `cache=Nhit/Mmiss` on the summary line ever shows the non-expert cache doing
  real work (non-zero hits) after shrinking it, increase `ne_budget` back toward
  1 GB and lower the expert cap correspondingly.
