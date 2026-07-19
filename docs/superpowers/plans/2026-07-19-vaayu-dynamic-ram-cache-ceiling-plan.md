# Vaayu Dynamic RAM Expert-Cache Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the literal `8.5 GB` RAM expert-cache OOM ceiling with a value derived from measured RAM + a peak-projection headroom, so the cache self-sizes on any box — behind a flag that defaults to today's behavior.

**Architecture:** One pure helper `compute_expert_cache_ceiling(available_bytes, overhead_bytes)` returns `available − overhead − peak_headroom` (peak_headroom = `max(floor, frac × available)`). `compute_expert_budget` uses it as the `min()` cap arm ONLY when `VAAYU_DYNAMIC_CACHE=1`; otherwise the literal `8.5 GB` stays. The `min` with the existing live-RAM arm guarantees the derived ceiling can only lower the budget, never over-commit.

**Tech Stack:** Python 3, pytest, `os.environ`, `psutil` (already used). No new deps, no new files.

**Design source:** `docs/superpowers/specs/2026-07-19-vaayu-dynamic-ram-cache-ceiling-design.md`.

## Global Constraints

Copied from the spec; every task implicitly includes these.

- **Ships behind a flag, default off = today.** `VAAYU_DYNAMIC_CACHE=1` → derived ceiling.
  Default (unset / not `"1"`) → literal `8.5 GB`, byte-identical to current behavior.
- **Model-agnostic (CLAUDE.md §5).** No model literals. Ceiling derived from measured RAM;
  peak-headroom floor + fraction are HARDWARE/safety constants (commented as such).
  `max_per_layer` still derives from geometry. Tests over Qwen-like AND MiniMax-like.
- **Safety.** Derived ceiling never exceeds `available − overhead` (kept as a `min` arm);
  clamps to a `1 GB` floor; calibrated so this 32 GB box lands ≈ 8.5 GB (below the known
  9–10 GB OOM boundary).
- **Shared-infra caution (CLAUDE.md §0).** Keep MoE tests green and the dense smoke green.
- **Repo layout.** Source in `vaayu/vaayu_base.py`; tests in `tests/`; run with
  `python -m pytest tests/<file> -v`.

## File Structure

- **Modify `vaayu/vaayu_base.py`** — add `compute_expert_cache_ceiling` (beside
  `compute_expert_budget:115`) + two hardware constants; change the cap arm in
  `compute_expert_budget:143`.
- **Modify `tests/test_expert_budget_sizing.py`** — add flag-off regression + flag-on
  derivation + helper tests.
- **Modify `CLAUDE.md`** — §5 hardware-constants bullet (Task 2).
- **Create `docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-RESULTS.md`**
  — on-device calibration record (Task 2).

---

### Task 1: derived ceiling helper + flag-gated integration

**Files:**
- Modify: `vaayu/vaayu_base.py` (add constants + `compute_expert_cache_ceiling` after
  `:112`; change the cap arm at `:143`)
- Test: `tests/test_expert_budget_sizing.py`

**Interfaces:**
- Produces: `compute_expert_cache_ceiling(available_bytes: int, overhead_bytes: int) -> int`;
  module constants `_PEAK_HEADROOM_FLOOR: int`, `_PEAK_HEADROOM_FRAC: float`.
- Consumes: existing `compute_expert_budget`, `_EXPERT_CACHE_OOM_CEILING`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_budget_sizing.py`:

```python
import os
from vaayu.vaayu_base import compute_expert_cache_ceiling


def test_ceiling_formula_and_floor():
    # available - overhead - max(floor, frac*available)
    from vaayu.vaayu_base import _PEAK_HEADROOM_FLOOR, _PEAK_HEADROOM_FRAC
    avail = int(26.5 * GB)
    overhead = int(2.5 * GB)
    peak = max(_PEAK_HEADROOM_FLOOR, int(_PEAK_HEADROOM_FRAC * avail))
    assert compute_expert_cache_ceiling(avail, overhead) == max(GB, avail - overhead - peak)


def test_ceiling_monotonic_and_bounded():
    o = int(2.5 * GB)
    c16 = compute_expert_cache_ceiling(int(16 * GB), o)
    c32 = compute_expert_cache_ceiling(int(32 * GB), o)
    c64 = compute_expert_cache_ceiling(int(64 * GB), o)
    assert c16 <= c32 <= c64                     # more RAM -> bigger ceiling
    for avail, c in [(16, c16), (32, c32), (64, c64)]:
        assert c <= int(avail * GB) - o          # never exceeds available - overhead
        assert c >= GB                           # 1 GB floor


def test_flag_off_uses_literal_ceiling(monkeypatch):
    monkeypatch.delenv("VAAYU_DYNAMIC_CACHE", raising=False)
    ws = int(8 * 91 * 11.28 * MB)
    budget, _, _ = compute_expert_budget(
        available_bytes=int(60 * GB), working_set_bytes=ws,     # huge RAM
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128)
    assert budget <= int(8.5 * GB) + 1           # capped at the literal, flag off


def test_flag_on_uses_derived_ceiling(monkeypatch):
    monkeypatch.setenv("VAAYU_DYNAMIC_CACHE", "1")
    ws = int(8 * 91 * 11.28 * MB)
    budget, _, _ = compute_expert_budget(
        available_bytes=int(60 * GB), working_set_bytes=ws,     # huge RAM
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128)
    assert budget > int(8.5 * GB)                # derived cap exceeds the old literal on a 60GB box
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_budget_sizing.py -k "ceiling or flag" -v`
Expected: FAIL with `ImportError: cannot import name 'compute_expert_cache_ceiling'`.

- [ ] **Step 3: Write minimal implementation**

In `vaayu/vaayu_base.py`, immediately after `_EXPERT_CACHE_OOM_CEILING` (`:112`), add:

```python
# Peak-projection headroom (Subsystem C). Hardware/safety constants: reserve for the
# transient decode-time spike (activations, io_uring buffers, KV growth) that
# MemAvailable does not see. Fraction-of-available so the reserve scales with the room
# a spike has to grow into. Calibrated so this 32 GB box lands ~8.5 GB (validate on
# device; see the RESULTS doc). NOT model numbers.
_PEAK_HEADROOM_FLOOR = int(4 * 1024 ** 3)
_PEAK_HEADROOM_FRAC = 0.58


def compute_expert_cache_ceiling(available_bytes: int, overhead_bytes: int) -> int:
    """Derive the expert-cache OOM ceiling from measured RAM instead of a literal.
    Portable across boxes: more free RAM -> higher ceiling; model weights (fixed) are
    already netted out of available_bytes. Behind VAAYU_DYNAMIC_CACHE=1."""
    GB = 1024 ** 3
    peak_headroom = max(_PEAK_HEADROOM_FLOOR, int(_PEAK_HEADROOM_FRAC * available_bytes))
    return max(1 * GB, available_bytes - overhead_bytes - peak_headroom)
```

Then change the cap arm inside `compute_expert_budget` (the `min(...)` at `:143`) from:

```python
    expert_budget = min(
        _EXPERT_CACHE_OOM_CEILING,               # hardware ceiling, not a model number
        max(1 * GB, available_bytes - overhead),
    )
```

to:

```python
    cap = (compute_expert_cache_ceiling(available_bytes, overhead)
           if os.environ.get("VAAYU_DYNAMIC_CACHE") == "1"
           else _EXPERT_CACHE_OOM_CEILING)      # literal ceiling unless flag on
    expert_budget = min(
        cap,
        max(1 * GB, available_bytes - overhead),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_budget_sizing.py -v`
Expected: PASS — new tests AND the pre-existing flag-off tests (regression lock).

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_expert_budget_sizing.py
git commit -m "feat(cache): derive RAM expert-cache ceiling from measured RAM (flag-gated)"
```

---

### Task 2: CLAUDE.md §5 update + on-device calibration record

**Files:**
- Modify: `CLAUDE.md` (§5 hardware-constants bullet)
- Create: `docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-RESULTS.md`

- [ ] **Step 1: Update CLAUDE.md §5**

In `CLAUDE.md` §5, the hardware-constants bullet currently reads (around the
"~8.5 GB expert-cache OOM ceiling" phrase). Change that clause to state the ceiling is
**derived** from measured RAM when `VAAYU_DYNAMIC_CACHE=1`:

Replace:

```
- **The only fixed constants are HARDWARE limits** — this box's VRAM (16 GB), RAM
  (32 GB), the ~8.5 GB expert-cache OOM ceiling, PCIe/NVMe bandwidth. Those are
  properties of the machine, not the model. They stay literal (and commented as
  hardware constants).
```

with:

```
- **The only fixed constants are HARDWARE limits** — this box's VRAM (16 GB), RAM
  (32 GB), PCIe/NVMe bandwidth. Those are properties of the machine, not the model.
  They stay literal (and commented as hardware constants). The expert-cache OOM
  ceiling is DERIVED from measured RAM (`compute_expert_cache_ceiling`, behind
  `VAAYU_DYNAMIC_CACHE=1`); its peak-headroom floor/fraction are the hardware
  constants, and ~8.5 GB is the expected value on this 32 GB box, not a literal the
  code depends on. (Default off still uses the 8.5 GB literal until the flag flip.)
```

- [ ] **Step 2: On-device calibration**

On this 32 GB box, with `VAAYU_DYNAMIC_CACHE=1`, run a real MoE generate and read the
printed expert-cache budget line. Confirm:
- The derived budget lands in a safe band ≈ 8.5 GB (target: ≥ the Qwen working set
  ~8.02 GB so it does not thrash, and < 9 GB so it does not OOM-kill).
- A full generate run completes with no OOM.
- Existing MoE-EXL3 generate tests green; dense load+generate smoke green.

If the derived value is out of band, tune `_PEAK_HEADROOM_FRAC` / `_PEAK_HEADROOM_FLOOR`
and re-run until it lands in band. Record the measured `available`, computed `overhead`,
derived ceiling, and final constant values.

- [ ] **Step 3: Write the RESULTS doc**

Create `docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-RESULTS.md`
capturing: measured `available_bytes`, `overhead`, `peak_headroom`, derived ceiling vs
the old 8.5 GB literal, the generate run's peak RSS (no OOM), and the final calibrated
constants. Note whether the default should be flipped to derived (the deferred follow-up).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-RESULTS.md
git commit -m "docs(cache): CLAUDE.md §5 derived-ceiling note + calibration results"
```

---

## Notes & known limitations

- **Default stays off** until Task 2 calibration proves parity; flipping the default to
  derived is a deferred follow-up (spec §8), not part of this plan.
- **Starting constants (`_PEAK_HEADROOM_FRAC=0.58`, `_PEAK_HEADROOM_FLOOR=4 GB`) are
  calibration seeds**, chosen so `available≈26.5 GB, overhead≈2.5 GB` on this box yields
  ceiling ≈ 8.6 GB. Task 2 tunes them to the real measured values.
- **Fraction-of-available, not geometry-derived** peak model (spec §8) — refine later only
  if calibration shows the fraction is too coarse across boxes.
- **VRAM cache and `LayerRAMCache` untouched** — both already auto-size.
