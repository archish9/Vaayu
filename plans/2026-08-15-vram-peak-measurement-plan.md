# VRAM Peak Measurement — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the whole-prefill VRAM peak observable from outside the engine again, then re-measure all four on-device geometries and publish the real curves.

**Architecture:** One line is deleted — the `torch.cuda.reset_peak_memory_stats()` call inside `_chunked_prefill`'s legacy per-chunk loop — because it wipes a process-global counter mid-forward and truncates every external measurement to the final chunk. A source-level regression test keeps it deleted. The probe script's sweep is then made to straddle the two boundaries where the prediction is piecewise, and the four geometries are re-probed into a RESULTS file. **No prediction, budget, or derived limit changes in this phase.**

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu130, transformers 5.5.4, pytest, `systemd-run --user --scope` for memory-capped arms.

**Spec:** `plans/2026-08-15-vram-peak-measurement-design.md`

## Global Constraints

- `python` is NOT on PATH → use `./venv/bin/python`. `vaayu` is not pip-installed → scripts need `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite baseline: `./venv/bin/python -m pytest tests/ -q` → **795 passed, 2 skipped, 1 xfailed**. This plan adds tests; the passed count rises accordingly and nothing may regress.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers anywhere (layer counts, expert counts, hidden sizes, per-token KV rates). Only HARDWARE limits may be literals — 16 GB VRAM / 15.47 GiB usable, 32 GB RAM, ~8.5 GB expert-cache ceiling, ~5.5 GB/s NVMe — each commented as such.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB> -p MemorySwapMax=0`; **28 G for Llama-3.3-70B** (22 G is no longer enough), 22 G for the others. Never attach py-spy — it corrupts `vaayu_rust`'s io_uring state. Discard each checkpoint's first touch. Redirect stdout to a file. Read logs with `grep -a`, never `cat` (300 MB–1.3 GB).
- **Never load two models in one pytest/python process** — freed pinned host blocks are cached, not returned; the second load gets OOM-killed (exit 137) under a cgroup cap.
- Same-box, same-day A/B or it does not count.
- **The ±10 % form gate (`scripts/fit_vram_calibration.py`) is EXPECTED to fail after this plan.** That is the correct outcome — it currently passes against a truncated measurement. Do NOT touch `predict_peak_vram`, `MATERIALIZES_SCORES`, `DEFAULT_ALLOC_FACTOR`, or any budget code to re-green it. That is Phase 2's decision, made with the data this plan produces.
- Checkpoint paths (`/home/a/code/ai-ml/models/`): Qwen3-235B `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`, MiniMax-M2.7 `NeuroSenko--MiniMax-M2.7-exl3`, Llama-3.3-70B `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw`, Llama-3.2-1B `Llama-3.2-1B-Instruct`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `vaayu/vaayu_base.py:3315-3334` | Modify | Legacy per-chunk prefill loop. Loses the reset; keeps the per-chunk print, whose meaning becomes a running high-water mark. |
| `tests/test_vram_peak_measurement.py` | Create | Source-level gate: the engine must never reset the process-global CUDA peak counter. Runs without a GPU. |
| `scripts/probe_vram_curve.py` | Modify | Peak-vs-S probe. Docstring corrected; sweep selection extracted into a testable function that straddles both piecewise boundaries. |
| `tests/test_probe_sweep.py` | Create | Unit tests for that sweep function — derived from chunk size and KV growth, no model literals. |
| `plans/2026-08-15-vram-peak-RESULTS.md` | Create | The deliverable: measured curves for all four geometries with signed error vs `predict_peak_vram`. |
| `todo.md` | Modify | Ledger: Phase 1 closed, Phase 2 rows opened with the data (CLAUDE.md §6). |

---

## Task 1: Stop the engine from resetting the global peak counter

**Files:**
- Modify: `vaayu/vaayu_base.py:3330-3334`
- Test: `tests/test_vram_peak_measurement.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: the invariant "no module under `vaayu/` calls `torch.cuda.reset_peak_memory_stats()`", which Task 2's probe and Task 3's arms depend on for their numbers to mean anything.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vram_peak_measurement.py`:

```python
"""The engine must never reset the process-global CUDA peak counter.

`torch.cuda.max_memory_allocated()` is ONE high-water mark per process, and
`reset_peak_memory_stats()` is how a caller says "start measuring here". Vaayu
used to call it once per prefill chunk, inside `_chunked_prefill`'s legacy
per-chunk loop, so any caller doing the standard reset -> forward -> read saw
only the FINAL chunk of a chunked prefill.

That truncated every point in scripts/fit_vram_calibration.py's MEASURED table
for the two DENSE geometries (the MoE layer-major branch returns before the
loop and was never affected). Measured cost of the blind spot on Llama-3.2-1B
at S=32768: 2.409 GB recorded, 8.13 GB actual.

Asserted at SOURCE level on purpose: tests/ has no GPU, and the defect is
"the engine mutates shared global state", not any particular byte count. A
value-asserting test would need a GPU and would not have caught this.
"""
import inspect
from pathlib import Path

import vaayu.vaayu_base as vaayu_base

_FORBIDDEN = "reset_peak_memory_stats"


def test_chunked_prefill_does_not_reset_the_peak_counter():
    src = inspect.getsource(vaayu_base.VaayuBaseModel._chunked_prefill)
    assert _FORBIDDEN not in src, (
        "_chunked_prefill resets the process-global CUDA peak counter again; "
        "this truncates every external measurement of a chunked prefill to its "
        "last chunk. See plans/2026-08-15-vram-peak-measurement-design.md."
    )


def test_no_engine_module_resets_the_peak_counter():
    """Same invariant, package-wide: the reset must not reappear elsewhere."""
    root = Path(vaayu_base.__file__).parent
    offenders = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if _FORBIDDEN in p.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        f"these engine modules reset the global CUDA peak counter: {offenders}. "
        f"Only measurement CALLERS (scripts/, tests/) may reset it."
    )
```

- [ ] **Step 2: Run it and verify BOTH tests fail**

```bash
./venv/bin/python -m pytest tests/test_vram_peak_measurement.py -v
```

Expected: 2 failed. The first names `_chunked_prefill`; the second lists `vaayu_base.py`. If either PASSES here, stop — the line you are about to delete is not the line the tests are looking at.

- [ ] **Step 3: Delete the reset**

In `vaayu/vaayu_base.py`, the per-chunk block at the end of the legacy loop currently reads:

```python
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
```

Replace it with:

```python
            if torch.cuda.is_available():
                # RUNNING high-water mark across the whole prefill, not this
                # chunk's own peak: max_memory_allocated() is one PROCESS-GLOBAL
                # counter, and it is deliberately NOT reset here. Resetting it
                # per chunk truncated every external measurement of a chunked
                # prefill to its final chunk -- which is how every dense point in
                # scripts/fit_vram_calibration.py's MEASURED table came to be a
                # last-chunk peak (llama-3.2-1b at S=32768: 2.409 GB recorded,
                # 8.13 GB actual). The counter belongs to whoever called
                # reset_peak_memory_stats() last; the engine must not touch it.
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
./venv/bin/python -m pytest tests/test_vram_peak_measurement.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Prove the gate can fail (a gate that cannot fail is not a gate)**

Temporarily restore the deleted line, re-run, confirm 2 failed, then remove it again and confirm 2 passed. Do not skip this — the repo has already shipped one gate that passed identically with and without the bug it existed to catch.

```bash
./venv/bin/python -m pytest tests/test_vram_peak_measurement.py -q   # after restore: 2 failed
./venv/bin/python -m pytest tests/test_vram_peak_measurement.py -q   # after re-delete: 2 passed
```

- [ ] **Step 6: Run the full suite**

```bash
./venv/bin/python -m pytest tests/ -q
```

Expected: 797 passed, 2 skipped, 1 xfailed (795 baseline + the 2 new tests). No failures.

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_vram_peak_measurement.py
git commit -m "fix(vram): stop resetting the global CUDA peak counter mid-prefill

The per-chunk reset in _chunked_prefill's legacy loop truncated every external
measurement of a chunked prefill to its final chunk, which is how the dense
points in fit_vram_calibration.py's MEASURED table became last-chunk peaks
(llama-3.2-1b at S=32768: 2.409 GB recorded, 8.13 GB actual).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Make the probe sweep straddle both piecewise boundaries

**Files:**
- Modify: `scripts/probe_vram_curve.py` (docstring at lines 1-13; add `sweep_points`; use it for the default sweep)
- Test: `tests/test_probe_sweep.py` (create)

**Interfaces:**
- Consumes: Task 1's invariant (the probe's `max_memory_allocated()` read is now whole-prefill).
- Produces: `sweep_points(*, chunk: int, growth: int, s_max: int) -> list[int]` in `scripts/probe_vram_curve.py`, used by Task 3's arms.

**Why this exists:** `predict_peak_vram` is piecewise in two independent ways — the KV term steps at each capacity doubling, and the logits intercept saws back down at each prefill-chunk boundary. A sweep that misses either boundary cannot distinguish a right form from a wrong one.

- [ ] **Step 1: Write the failing test**

Create `tests/test_probe_sweep.py`:

```python
"""scripts/probe_vram_curve.py's default S sweep must probe the two places
predict_peak_vram is piecewise: KV capacity doublings and prefill-chunk
boundaries. Every point is DERIVED from chunk/growth/s_max (CLAUDE.md S5) --
no model numbers appear here or in the function under test.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "probe_vram_curve",
    Path(__file__).resolve().parent.parent / "scripts" / "probe_vram_curve.py",
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192),      # a small derived ceiling
    (512, 2, 131072),    # a large one
    (256, 3, 20000),     # neither chunk nor growth is the common case
])
def test_sweep_is_sorted_unique_and_inside_the_ceiling(chunk, growth, s_max):
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    assert pts == sorted(set(pts))
    assert all(0 < s <= s_max for s in pts)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192), (512, 2, 131072), (256, 3, 20000),
])
def test_sweep_straddles_a_chunk_boundary(chunk, growth, s_max):
    """At least one exact multiple of chunk (full trailing chunk, where the
    logits intercept is largest) AND one point just past a multiple (trailing
    chunk of 1 token, where it is smallest)."""
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    assert any(s % chunk == 0 for s in pts)
    assert any(s % chunk == 1 for s in pts)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192), (512, 2, 131072), (256, 3, 20000),
])
def test_sweep_straddles_a_kv_capacity_doubling(chunk, growth, s_max):
    """Two points whose allocated KV capacity differs -- the step the KV term
    takes. Uses the engine's own capacity function so the test cannot drift
    from the implementation it is checking."""
    from vaayu.vram_plan import kv_capacity_tokens
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    caps = {kv_capacity_tokens(s, chunk=chunk, growth=growth, buckets=True)
            for s in pts}
    assert len(caps) >= 2


def test_sweep_degenerates_gracefully_when_the_ceiling_is_tiny():
    """A ceiling below one chunk still yields a usable, in-range sweep."""
    pts = probe.sweep_points(chunk=512, growth=2, s_max=300)
    assert pts and all(0 < s <= 300 for s in pts)
```

- [ ] **Step 2: Run it and verify it fails**

```bash
./venv/bin/python -m pytest tests/test_probe_sweep.py -v
```

Expected: FAIL — `AttributeError: module 'probe_vram_curve' has no attribute 'sweep_points'`.

- [ ] **Step 3: Implement `sweep_points` and correct the docstring**

In `scripts/probe_vram_curve.py`, replace the first docstring paragraph — the claim that peak-vs-S is not observable inside one call, which is the belief that produced the truncated table — with:

```python
"""P2/P3: peak VRAM vs context length, one model LOAD per model.

Peak-vs-S IS observable per call: the engine no longer resets the
process-global CUDA peak counter mid-prefill (see
plans/2026-08-15-vram-peak-measurement-design.md), so the standard
reset -> forward -> max_memory_allocated() sequence below reports the peak of
the WHOLE prefill. It previously reported only the final chunk on the dense
path, which is how every dense point in scripts/fit_vram_calibration.py's
MEASURED table became a last-chunk peak.

Usage:
  python scripts/probe_vram_curve.py <checkpoint> [S ...]

Run ONE model per process (METHOD NOTES: freed pinned blocks are cached, not
returned, so arms accumulate; one probe hit a 28.3 GB peak and systemd-oomd
killed the whole terminal cgroup).
"""
```

Then add, above `main()`:

```python
def sweep_points(*, chunk: int, growth: int, s_max: int) -> list[int]:
    """S values that probe both places predict_peak_vram is piecewise.

    The KV term steps at each capacity doubling; the logits intercept saws back
    down at each prefill-chunk boundary. A sweep that crosses neither cannot
    tell a right form from a wrong one.

    Everything is derived from the three arguments (CLAUDE.md S5): chunk and
    growth are properties of the prefill loop and the KV cache implementation,
    s_max of the box's free VRAM. No model number appears here.
    """
    from vaayu.vram_plan import kv_capacity_tokens

    pts: set[int] = set()

    # Chunk-boundary pairs: an exact multiple (full trailing chunk, intercept at
    # its largest) and the next token after one (trailing chunk of 1, smallest).
    k = 1
    while k * chunk <= s_max:
        pts.add(k * chunk)
        if k * chunk + 1 <= s_max:
            pts.add(k * chunk + 1)
        k *= 2

    # KV doubling pairs: the last S in a capacity bucket and the first in the
    # next one, found from the engine's own capacity function.
    s = chunk
    while s <= s_max:
        cap = kv_capacity_tokens(s, chunk=chunk, growth=growth, buckets=True)
        if cap < s_max:
            pts.add(min(cap, s_max))
            pts.add(min(cap + 1, s_max))
        s = max(s + 1, cap + 1)

    pts.add(s_max)
    return sorted(p for p in pts if 0 < p <= s_max)
```

Finally, replace the `DEFAULT_S` constant and its use in `main()` so the default sweep is derived rather than literal:

```python
    lengths = [int(x) for x in sys.argv[2:]] or sweep_points(
        chunk=model.prefill_chunk_size,
        growth=FP8DynamicCache._GROWTH,
        s_max=model.max_seq_len,
    )
```

`FP8DynamicCache` is imported from `vaayu.vaayu_base`; move the `lengths` assignment below the `from_pretrained` call, and pass `max_seq_len=None` there so the model derives its own ceiling. Delete `DEFAULT_S`.

- [ ] **Step 4: Run the tests and verify they pass**

```bash
./venv/bin/python -m pytest tests/test_probe_sweep.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Smoke the probe on the cheap checkpoint**

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python \
  scripts/probe_vram_curve.py /home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
  512 513 1024 2048 > /tmp/probe_smoke.log 2>&1
grep -a "^S,\|^[0-9]" /tmp/probe_smoke.log
```

Expected: four CSV rows, `status=ok`, `alloc_peak_bytes` **strictly increasing** with S. If it is flat or non-monotone across these small S, Task 1's deletion did not take effect — stop and investigate before probing anything expensive.

- [ ] **Step 6: Full suite, then commit**

```bash
./venv/bin/python -m pytest tests/ -q
git add scripts/probe_vram_curve.py tests/test_probe_sweep.py
git commit -m "test(vram): derive the probe sweep from chunk and KV growth

Straddles the two boundaries predict_peak_vram is piecewise across, and drops
the literal DEFAULT_S. Docstring no longer claims peak-vs-S is unobservable
within one call -- that belief produced the truncated MEASURED table.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: 807 passed, 2 skipped, 1 xfailed.

---

## Task 3: Re-probe all four geometries and publish the curves

**Files:**
- Create: `plans/2026-08-15-vram-peak-RESULTS.md`
- Modify: `todo.md`

**Interfaces:**
- Consumes: `sweep_points` from Task 2; the invariant from Task 1.
- Produces: the measured table Phase 2 needs. No code changes.

**This task is measurement, not implementation.** Nothing in `vaayu/` may change while running it.

- [ ] **Step 1: Run the four arms, one per subprocess, cheapest first**

Each arm runs alone. Discard the first touch of each checkpoint (page cache cold) by running the arm twice and keeping the second.

```bash
M=/home/a/code/ai-ml/models
run() {  # $1 = checkpoint dir, $2 = MemoryMax, $3 = output tag
  systemd-run --user --scope -p MemoryMax=$2 -p MemorySwapMax=0 \
    env PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python \
    scripts/probe_vram_curve.py "$1" > /tmp/vramcurve_$3.log 2>&1
  echo "$3 exit=$?"
}
run $M/Llama-3.2-1B-Instruct 22G llama-3.2-1b
run $M/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw 28G llama-3.3-70b
run $M/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw 22G qwen3-235b
run $M/NeuroSenko--MiniMax-M2.7-exl3 22G minimax-m2.7
```

**Trim the sweep on the expensive arms before running them.** `sweep_points` emits roughly
`2 * log2(s_max/chunk)` chunk-boundary points plus a KV pair per capacity bucket — around 30
points at the 1B's derived ceiling, where a single large-S prefill already costs minutes, and
around 8 at the 70B's, where each chunk streams weights off NVMe at ~40 s. Generate the full
list, then pass an explicit trimmed list as argv:

```bash
./venv/bin/python -c "import importlib.util,pathlib
spec=importlib.util.spec_from_file_location('p','scripts/probe_vram_curve.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.sweep_points(chunk=512, growth=2, s_max=8192))"
```

Keep at most ~8 points per arm, and the trim MUST retain: both sides of one chunk boundary
(an exact multiple and a multiple+1), both sides of one KV capacity doubling, and the largest
S the arm can complete. Dropping a boundary pair defeats the whole sweep.

Read every log with `grep -a`, never `cat`. Extract each arm's CSV rows:

```bash
for t in llama-3.2-1b llama-3.3-70b qwen3-235b minimax-m2.7; do
  echo "== $t"; grep -a "^[0-9]" /tmp/vramcurve_$t.log
done
```

An `OOM` in the `status` column is DATA, not a failure: record the S that produced it and keep the remaining arms. A killed arm (exit 137) is not data — rerun it alone with a higher `MemoryMax`.

- [ ] **Step 2: Compute the signed error against the current form**

For every point, `transient = alloc_peak_bytes - alloc_before_bytes` and `rel = (predicted - transient) / transient`, using `predict_peak_vram` with the same `VramShape` construction `scripts/fit_vram_calibration.py:shape_for` already uses. Reuse that function rather than rebuilding the shape, so the comparison is against exactly the form in the tree.

- [ ] **Step 3: Write `plans/2026-08-15-vram-peak-RESULTS.md`**

It must contain, and a reader with no GPU must be able to re-derive the verdict from it:

1. Provenance header: date, box, torch version, `PYTORCH_CUDA_ALLOC_CONF`, the cgroup cap per arm, and the commit SHA the arms ran at.
2. Per geometry: every `(S, alloc_before, alloc_peak, status)` row as measured.
3. Per geometry: fitted slope in MB/token, `predict_peak_vram`'s prediction at each point, and the signed relative error.
4. The before/after comparison against the old `MEASURED` table, called out per geometry — the two dense ones are expected to move, the two MoE ones are expected NOT to (the layer-major branch never hit the reset). **If a MoE geometry moves, that expectation was wrong and the reason must be found before Phase 2 starts.**
5. A one-paragraph verdict: what the corrected data says about the `KV(capacity) + last-chunk logits` form, and whether the surplus is the score/mask-shaped multiple of `n_heads · chunk · elem_bytes` that the 1B (5.2×) and 70B (5.9×) suggested.

- [ ] **Step 4: Record the expected gate failure**

```bash
./venv/bin/python scripts/fit_vram_calibration.py
```

Do NOT update `MEASURED` to the new numbers in this task, and do NOT change the form. Paste this run's output into the RESULTS file as the "before" state, and state plainly that re-pointing the table at the corrected data is Phase 2's first step, because it is what makes the gate meaningful again.

- [ ] **Step 5: Verify the on-device gates still pass**

The deletion is one line, but it is in the prefill path, so run all three (each under its own scope, per the gates table in `todo.md`):

```bash
systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python test_batched_dense_smoke.py
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu MOE_CKPT=$M/NeuroSenko--MiniMax-M2.7-exl3 \
  ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu PREFIX_CKPT=$M/NeuroSenko--MiniMax-M2.7-exl3 \
  ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q
```

Expected: dense smoke perplexity **8.78** on Llama-3.3-70B (unchanged); the other two pass.

- [ ] **Step 6: Update the ledger (CLAUDE.md §6)**

In `todo.md`: mark the measurement row `[x]` with a one-line verdict pointing at the RESULTS file, and open Phase 2 as rows carrying the four gated items from the spec's §4 — form, `MATERIALIZES_SCORES`, budget policy, `max_seq_len` re-derivation — each with its file, what is wrong, and what done means. State in the ledger that the ±10 % gate is knowingly red and why, so a later reader does not "fix" it by tuning the form.

- [ ] **Step 7: Commit**

```bash
git add plans/2026-08-15-vram-peak-RESULTS.md todo.md
git commit -m "docs(vram): measured peak curves for all four geometries

Re-probed with the per-chunk counter reset removed. Records the signed error
against the current KV(capacity)+last-chunk-logits form and leaves the +-10%
gate knowingly red: re-pointing MEASURED and re-deciding the form is Phase 2.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 1)

1. `grep -rn reset_peak_memory_stats vaayu/` returns nothing.
2. `tests/test_vram_peak_measurement.py` passes and was **observed to fail** with the line restored.
3. `tests/test_probe_sweep.py` passes; the probe's default sweep contains no model literals.
4. Full suite green at the new baseline (807 passed, 2 skipped, 1 xfailed).
5. The three on-device gates pass; dense smoke perplexity still 8.78.
6. `plans/2026-08-15-vram-peak-RESULTS.md` exists with all four geometries, every point labelled ok/OOM, and signed errors.
7. `todo.md` closes Phase 1 and opens Phase 2 with the data.
8. `predict_peak_vram`, `MATERIALIZES_SCORES`, `DEFAULT_ALLOC_FACTOR`, `MEASURED`, and every budget path are **unchanged**.
