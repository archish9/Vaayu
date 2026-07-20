# Vaayu Dynamic RAM Expert-Cache Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the literal `8.5 GB` RAM expert-cache OOM ceiling with a value derived from measured RAM + a peak-projection headroom, so the cache self-sizes on any box — behind a flag that defaults to today's behavior.

**Architecture:** One pure helper `compute_expert_cache_ceiling(available_bytes, overhead_bytes)` returns `available − overhead − peak_headroom` (peak_headroom = `max(floor, frac × available)`). `compute_expert_budget` uses it as the `min()` cap arm ONLY when `VAAYU_DYNAMIC_CACHE=1`; otherwise the literal `8.5 GB` stays. The `min` with the existing live-RAM arm guarantees the derived ceiling can only lower the budget, never over-commit.

**Tech Stack:** Python 3, pytest, `os.environ`, `psutil` (already used). No new deps, no new files.

**Design source:** `docs/superpowers/specs/2026-07-19-vaayu-dynamic-ram-cache-ceiling-design.md`.

---

## On-device findings (2026-07-20) — READ BEFORE IMPLEMENTING

Measured on this box (32 GB RAM), Qwen3-235B-A22B-Thinking exl3-5.0bpw, from a real
on-device generate run (`run1.log`, produced while verifying the KV-persist plan):

| Quantity | Measured value |
|---|---|
| Expert RAM cache budget (today, literal ceiling) | **9.13 GB** (771 expert slots) |
| Per-token decode working set | **8.90 GB** (752 experts = top_k 8 × 94 layers) |
| Headroom (budget − working set) | 0.23 GB ≈ 19 slots |
| Steady-state expert-cache hit rate | 43–44 % |
| `expert_disk` share of token time | ~50 % (0.4–0.9 s of 0.9–1.7 s) |

Three findings that change this plan:

1. **`compute_expert_budget` ignores its own `working_set_bytes` parameter.** It is
   declared (`vaayu_base.py:126`), documented (`:135`), and passed by the caller
   (`:943`) — but never read in the function body (`:142-163`). The docstring's claim
   ("Size the expert RAM cache to exceed the DERIVED decode working set") is not
   implemented. Today's `budget > working_set` (9.13 > 8.90) is **coincidence**: the
   `8.5 GiB` literal happens to land above it. That literal is what is accidentally
   protecting decode from thrash — so replacing it with a derived value that *can* go
   lower, without adding a floor, is a live regression risk. Task 1 now fixes this.

2. **The plan's original seed constants regress this box.** `_PEAK_HEADROOM_FRAC = 0.58`
   gives `peak = max(4 GB, 0.58 × 26.5 GB) = 15.37 GB`, so
   `ceiling = 26.5 − 2.5 − 15.37 = 8.63 GB` — **below the 8.90 GB working set**, i.e.
   the exact pre-Fix1 thrash condition (hit rate 9–29 % before Fix1 vs 40–54 % after).
   Seed corrected to `0.56` below; derivation in Task 2.

3. **The original acceptance band was built on a stale number.** Task 2 said "≥ the Qwen
   working set ~8.02 GB". The real measured working set is **8.90 GB**, so `8.63` would
   have *passed* the plan's own stated check while failing in practice. Band corrected
   in Task 2 below.

**Scope note (not a change, but set expectations):** this plan delivers **no speedup on
this box** — by design it reproduces the current ~8.5 GB ceiling here. Its value is that
every user with more RAM than this box is currently hard-capped at *this box's* ceiling:
a 128 GB machine gets 771 expert slots when it could hold thousands. For a public release
that is the win, and it is a real one.

---

## Global Constraints

Copied from the spec; every task implicitly includes these.

- **Ships behind a flag, default off = today.** `VAAYU_DYNAMIC_CACHE=1` → derived ceiling.
  Default (unset / not `"1"`) → literal `8.5 GB`, byte-identical to current behavior.
- **Model-agnostic (CLAUDE.md §5).** No model literals. Ceiling derived from measured RAM;
  peak-headroom floor + fraction are HARDWARE/safety constants (commented as such).
  `max_per_layer` still derives from geometry. Tests over Qwen-like AND MiniMax-like.
- **Safety.** Derived ceiling never exceeds `available − overhead` (kept as a `min` arm);
  clamps to a `1 GB` floor; calibrated so this 32 GB box lands ≈ 9.13 GB (its known-good
  value today, above the 8.90 GB working set and below the confirmed 10 GB OOM boundary).
- **Never silently land below one decode working set (finding #1).** The derived ceiling
  is lifted to `working_set_bytes` when RAM demonstrably holds it and still leaves
  `_PEAK_HEADROOM_FLOOR` of spike room. When RAM genuinely cannot, the budget is left
  alone and the EXISTING caller warning fires (`vaayu_base.py:999-1003` — do **not** add
  a second warning). Raise only to the working set, never beyond: OOM kills the process,
  thrash only slows it.
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
- **Also activates `compute_expert_budget`'s existing-but-dead `working_set_bytes`
  parameter** (finding #1) as the thrash floor. No signature change — the parameter is
  already declared and already passed by the caller; it is simply never read today.

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


# --- thrash floor (on-device finding #1) ---

def test_thrash_floor_lifts_cap_to_working_set(monkeypatch):
    """A derived ceiling below one decode working set evicts, every token, exactly
    what the next token needs (pre-Fix1: 9-29% hit rate vs 40-54% after). When RAM
    demonstrably holds it, the cap must be lifted to the working set. Uses this
    box's REAL measured numbers: available 26.5GB, working set 8.90GB -- with
    FRAC=0.56 the raw derived ceiling is ~9.16GB, and with the older 0.58 it was
    8.63GB (below working set), which is exactly what this floor guards."""
    monkeypatch.setenv("VAAYU_DYNAMIC_CACHE", "1")
    ws = int(8.90 * GB)
    budget, _, _ = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=94, expert_bytes=int(11.8 * MB),
        moe_top_k=8, moe_num_experts=128)
    assert budget >= ws                          # never below one working set


def test_thrash_floor_not_applied_when_ram_cannot_hold_it(monkeypatch):
    """When RAM genuinely cannot hold working set + spike floor, leave the budget
    alone rather than eating the OOM safety margin -- OOM kills the process, thrash
    only slows it. The existing caller warning (vaayu_base.py:999) covers this."""
    monkeypatch.setenv("VAAYU_DYNAMIC_CACHE", "1")
    ws = int(20 * GB)                            # working set far exceeds this box
    budget, _, _ = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=94, expert_bytes=int(11.8 * MB),
        moe_top_k=8, moe_num_experts=128)
    assert budget < ws                           # not lifted; OOM safety wins


def test_thrash_floor_inert_when_flag_off(monkeypatch):
    """Ships-dark: with the flag off the literal ceiling governs, unchanged."""
    monkeypatch.delenv("VAAYU_DYNAMIC_CACHE", raising=False)
    ws = int(8.90 * GB)
    budget, _, _ = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=94, expert_bytes=int(11.8 * MB),
        moe_top_k=8, moe_num_experts=128)
    assert budget <= int(8.5 * GB) + 1           # literal ceiling, byte-identical to today
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
# a spike has to grow into. Calibrated so this 32 GB box lands ~9.13 GB, its known-good
# value today (above the 8.90 GB measured working set, below the confirmed 10 GB OOM
# boundary). Validate on device before trusting FRAC — see Task 2. NOT model numbers.
_PEAK_HEADROOM_FLOOR = int(4 * 1024 ** 3)
_PEAK_HEADROOM_FRAC = 0.56


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

    # Thrash floor (on-device finding #1). A cache below one decode working set
    # evicts, every token, exactly what the next token needs -- the pre-Fix1
    # condition (9-29% hit rate vs 40-54% after). The derived ceiling's peak
    # fraction is a GUESS; available_bytes is a MEASUREMENT. When RAM
    # demonstrably holds the working set AND still leaves the absolute spike
    # floor, don't let the guess push us under it. When RAM genuinely cannot,
    # leave the budget alone -- the caller already warns (vaayu_base.py:999).
    # Raise only TO the working set, never beyond: OOM kills the process,
    # thrash only slows it. This also activates working_set_bytes, which is
    # currently declared, documented, passed -- and never read.
    if (working_set_bytes > 0 and cap < working_set_bytes
            and available_bytes - overhead - working_set_bytes >= _PEAK_HEADROOM_FLOOR):
        cap = working_set_bytes

    expert_budget = min(
        cap,
        max(1 * GB, available_bytes - overhead),
    )
```

Note the floor is deliberately inside `compute_expert_budget` (not in
`compute_expert_cache_ceiling`): the ceiling helper stays a pure function of RAM, while
the working-set clamp is budget-sizing policy. Keeping the helper working-set-free also
keeps its unit tests (`test_ceiling_formula_and_floor`, `test_ceiling_monotonic_and_bounded`)
meaningful and geometry-independent.

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

**Step 2a — measure `available_bytes` FIRST (do not skip).** `_PEAK_HEADROOM_FRAC` is
derived from `available_bytes`, and the `26.5 GB` figure this plan assumes was **never
measured**. The on-device log only proves `available − overhead ≥ 9.13 GB` (because the
`min()` selected the 8.5 GiB literal), which does not pin down 26.5 at all. Add a
temporary print of the real `available_bytes` at the `compute_expert_budget` call site
(`vaayu_base.py:941-948`) and read it from a real run. If it is not ≈26.5 GB, recompute
FRAC with the derivation below before trusting anything.

**FRAC derivation** (redo with YOUR measured `available`):
```
target ceiling        = 9.13 GB   # known-good today: > 8.90 working set, < 10 GB OOM
available - overhead  = 26.5 - 2.5 = 24 GB
peak_headroom needed  = 24 - 9.13 = 14.87 GB
FRAC                  = 14.87 / 26.5 = 0.561    ->  0.56
check: 0.56 x 26.5 = 14.84 ; ceiling = 24 - 14.84 = 9.16 GB  ✓
```

On this 32 GB box, with `VAAYU_DYNAMIC_CACHE=1`, run a real MoE generate and read the
printed expert-cache budget line. Confirm:
- The derived budget lands in a safe band: **≥ 8.90 GB** — this box's REAL measured
  working set, NOT the stale `~8.02 GB` this plan originally stated (finding #3; 8.63 GB
  would have passed the old check while thrashing in practice) — and **< 10 GB**, the
  confirmed OOM-kill boundary. 9.13 GB is the known-good value the literal produces today.
- The thrash floor did NOT have to fire (i.e. the raw derived ceiling was already
  ≥ working set). If it fired, FRAC is still mis-tuned — the floor is a safety net, not
  the intended steady state.
- Steady-state expert-cache hit rate stays ≈43–44 % (the measured pre-change baseline).
  A drop toward 9–29 % means you landed below the working set.
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
- **Starting constants (`_PEAK_HEADROOM_FRAC=0.56`, `_PEAK_HEADROOM_FLOOR=4 GB`) are
  calibration seeds**, chosen so `available≈26.5 GB, overhead≈2.5 GB` on this box yields
  ceiling ≈ 9.16 GB — matching the 9.13 GB the literal produces today. Task 2 tunes them
  to the real measured values. (Was `0.58`/≈8.63 GB, which is BELOW this box's 8.90 GB
  working set — see finding #2.)
- **The fraction is doing heavy lifting.** Reserving 56 % of available RAM as spike
  headroom is very conservative on a large box (a 128 GB machine reserves ~67 GB). It is
  shaped that way to make a 32 GB box land on a specific number, which is
  tail-wagging-dog. It is acceptable for now because the thrash floor bounds the downside
  and the `min()` live-RAM arm bounds the upside — but if calibration on a second machine
  shows the fraction is a poor model of real spike size, replace it with a bounded
  absolute reserve rather than re-tuning the fraction again (spec §8's deferred refinement).
- **Do not add a working-set warning.** One already exists at `vaayu_base.py:999-1003`
  and fires on exactly this condition. The thrash floor complements it; it does not
  replace it.
- **Fraction-of-available, not geometry-derived** peak model (spec §8) — refine later only
  if calibration shows the fraction is too coarse across boxes.
- **VRAM cache and `LayerRAMCache` untouched** — both already auto-size.
