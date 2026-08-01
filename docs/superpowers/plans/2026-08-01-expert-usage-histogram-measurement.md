# Expert Usage Histogram (Measurement Only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, per (layer, expert), how often decode routes to it and how often the VRAM
expert cache evicts it under budget pressure — so we can decide on evidence whether a
hot-expert pin is worth building at all.

**Architecture:** Two independent counters joined only at report time. A new
`vaayu/expert_usage.py` owns a device-side `counts[n_rows, n_experts]` tensor incremented by an
`index_add_` in each MoE forward (no host sync). Separately, `ExpertVRAMCache` gains opt-in
per-key attribution of **budget-pressure evictions only**. A pure formatting function joins them
into a report. Everything is off by default.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

## Design source

`docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md` (commit
`c38132e`). That spec supersedes Phase 1 of
`docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`, whose PREMISE WARNING
says "DO NOT BUILD PHASE 1 AS WRITTEN". Read the spec's "Relationship to the 2026-07-19 plan"
table before touching that older plan.

## Global Constraints

Every task implicitly includes these.

- **Ships dark.** `enable_expert_usage` defaults **False**. When off: no tensor allocated, no
  Counter, the hook returns on a single attribute check, and behavior is byte-identical to
  today. Dense models never construct a tracker.
- **No hot-path host sync (NON-NEGOTIABLE).** `record()` must not call `.item()`, `.cpu()`,
  `.tolist()`, `torch.cuda.synchronize()`, or anything else that forces a device→host
  transfer. The ONLY sync is `snapshot()` at report time, after generation.
- **Budget-pressure evictions only.** Per-key attribution counts evictions from `put()`. It must
  NEVER count `evict_until_free()`'s evictions — that is the deliberate wholesale prefill wipe
  (measured at 1362 evictions in one pass), and counting it would drown the signal so that every
  expert looks like a pin candidate. This is the single most important rule in this plan.
- **Decode only.** `record()` is skipped while `vaayu_ref._moe_prefill_pass` is set. The VRAM
  cache is not consulted during prefill, so there are no budget-pressure evictions then;
  counting prefill routes would compare decode-side evictions against prefill+decode routes.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `n_experts` comes from
  `compute_moe_geometry`; `n_rows` from `config.num_hidden_layers` (normalized by the loader's
  alias map at `vaayu/vaayu_base.py:962`). No layer count, expert count, or top_k literal in
  code. Tracker tests are parametrized over Qwen-like (94 layers / 128 experts) and
  MiniMax-like (62 / 256).
- **Surgical (CLAUDE.md §3).** Every changed line traces to this plan. Do not refactor the MoE
  forwards, the router paths, or `ExpertVRAMCache`'s existing eviction logic.
- Run tests with `source venv/bin/activate`; `python -m pytest tests/ -q` must stay green
  (402 passing before this plan).

## File Structure

- **Create `vaayu/expert_usage.py`** — `ExpertUsageTracker` (counts tensor, `record`,
  `snapshot`) and the pure `format_usage_report` function. One responsibility: counting and
  presenting expert routes. ~90 lines.
- **Create `tests/test_expert_usage.py`** — tracker + report tests.
- **Modify `vaayu/persist/expert_cache.py`** — `ExpertVRAMCache` opt-in per-key eviction
  attribution.
- **Modify `tests/test_expert_vram_cache.py`** — the discriminating attribution test.
- **Modify `vaayu/moe_patch.py`** — one `_maybe_record_usage` helper plus three call sites.
- **Modify `vaayu/vaayu_base.py`** — `enable_expert_usage` flag and tracker construction.
- **Create `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement-RESULTS.md`** —
  Task 5's on-device evidence and the pin decision.

---

## Task 1: `ExpertUsageTracker` — device-side route counting (TDD)

**Files:**
- Create: `vaayu/expert_usage.py`
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Produces, for Tasks 3 and 4:
  - `ExpertUsageTracker(n_rows: int, n_experts: int, device)` 
  - `.counts` — `torch.Tensor` of shape `[n_rows, n_experts]`, dtype `torch.int32`
  - `.record(layer_idx: int, expert_indices: torch.Tensor) -> None`
  - `.snapshot() -> torch.Tensor` — an int32 CPU copy of `.counts`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_expert_usage.py`:

```python
import pytest
import torch

from vaayu.expert_usage import ExpertUsageTracker


def test_record_accumulates_routes():
    t = ExpertUsageTracker(n_rows=8, n_experts=16, device="cpu")
    # two tokens, top_k=2 -> experts 3,5 and 3,9
    idx = torch.tensor([[3, 5], [3, 9]], dtype=torch.int64)
    t.record(layer_idx=4, expert_indices=idx)
    counts = t.snapshot()
    assert counts[4, 3].item() == 2
    assert counts[4, 5].item() == 1
    assert counts[4, 9].item() == 1
    assert counts[4].sum().item() == 4


def test_record_lands_in_the_model_layer_row_not_a_moe_ordinal():
    """Qwen's MoE layers start at model layer 3. Rows are indexed by the model's
    own layer_idx, so recording layer 3 must not land in row 0."""
    t = ExpertUsageTracker(n_rows=8, n_experts=16, device="cpu")
    t.record(layer_idx=3, expert_indices=torch.tensor([[1]], dtype=torch.int64))
    counts = t.snapshot()
    assert counts[3, 1].item() == 1
    assert counts[0].sum().item() == 0


def test_record_accumulates_across_calls():
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    idx = torch.tensor([[2]], dtype=torch.int64)
    for _ in range(5):
        t.record(layer_idx=1, expert_indices=idx)
    assert t.snapshot()[1, 2].item() == 5


@pytest.mark.parametrize("n_rows,n_experts,hot", [
    (94, 128, 127),   # Qwen-like:    94 layers, 128 experts
    (62, 256, 255),   # MiniMax-like: 62 layers, 256 experts
])
def test_shape_and_range_for_both_geometries(n_rows, n_experts, hot):
    """§5: sized from geometry, and the top expert index of each family is in
    range (a 256-expert index would overflow a 128-wide tensor)."""
    t = ExpertUsageTracker(n_rows=n_rows, n_experts=n_experts, device="cpu")
    assert tuple(t.counts.shape) == (n_rows, n_experts)
    t.record(layer_idx=n_rows - 1, expert_indices=torch.tensor([[hot]], dtype=torch.int64))
    assert t.snapshot()[n_rows - 1, hot].item() == 1


def test_snapshot_is_a_copy_not_a_view():
    """Report code must not be able to mutate live counters."""
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    t.record(layer_idx=0, expert_indices=torch.tensor([[1]], dtype=torch.int64))
    snap = t.snapshot()
    snap[0, 1] = 999
    assert t.snapshot()[0, 1].item() == 1


def test_record_accepts_flat_indices():
    """Some routing paths hand back a flat index tensor rather than [tokens, top_k]."""
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    t.record(layer_idx=2, expert_indices=torch.tensor([1, 1, 3], dtype=torch.int64))
    counts = t.snapshot()
    assert counts[2, 1].item() == 2
    assert counts[2, 3].item() == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.expert_usage'`.

- [ ] **Step 3: Create `vaayu/expert_usage.py`**

```python
"""Expert usage histogram (measurement only).

Counts which MoE experts DECODE routes to, so the hot-but-evicted population
can be measured before any hot-expert pin is designed. See
docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md.

Ships dark: nothing here is constructed unless enable_expert_usage is set.
"""
import torch


class ExpertUsageTracker:
    """Device-resident [n_rows, n_experts] route counter.

    Rows are indexed by the model's OWN layer_idx (Qwen's MoE layers start at 3),
    not by a zero-based MoE ordinal, so callers can pass layer_idx straight
    through. The few non-MoE rows cost ~48 KB at Qwen geometry and buy simpler,
    less error-prone indexing.
    """

    def __init__(self, n_rows: int, n_experts: int, device):
        self.n_rows = int(n_rows)
        self.n_experts = int(n_experts)
        self.counts = torch.zeros((self.n_rows, self.n_experts),
                                  dtype=torch.int32, device=device)

    def record(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        """Add one route per entry of expert_indices to row layer_idx.

        Hot path: this must stay a pure device op. No .item()/.cpu()/.tolist()
        and no synchronize() — the histogram is only read in snapshot().
        """
        flat = expert_indices.reshape(-1).to(torch.int64)
        ones = torch.ones_like(flat, dtype=self.counts.dtype)
        self.counts[layer_idx].index_add_(0, flat, ones)

    def snapshot(self) -> torch.Tensor:
        """int32 CPU copy of the counters. The ONLY device->host sync in this
        module; call it after generation, never inside a forward."""
        return self.counts.detach().to("cpu").clone()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py -v
```

Expected: PASS — all 7 tests (the parametrized one counts as 2).

- [ ] **Step 5: Run the full suite**

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Expected: green, 409 passed (402 + 7).

- [ ] **Step 6: Commit**

```bash
git add vaayu/expert_usage.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): device-side route counter

Counts which experts decode routes to, per (layer, expert), as a pure device
op — no .item()/.cpu()/synchronize() in the hot path; snapshot() is the only
device->host transfer and runs after generation.

Rows are indexed by the model's own layer_idx rather than a zero-based MoE
ordinal, since Qwen's MoE layers start at 3. Sized from geometry, tested over
Qwen-like (94/128) and MiniMax-like (62/256).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Per-key attribution of budget-pressure evictions (TDD)

The half that makes the histogram mean something. **Read the Global Constraints' third bullet
before starting.**

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (`ExpertVRAMCache`)
- Test: `tests/test_expert_vram_cache.py`

**Interfaces:**
- Consumes: nothing from Task 1 — these two halves are independent by design.
- Produces, for Tasks 3 and 4:
  - `ExpertVRAMCache(budget_bytes, track_evicted_keys: bool = False)`
  - `.evicted_keys` — a `collections.Counter` keyed `(layer_idx, expert_idx)` when tracking is
    on, otherwise `None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_vram_cache.py`. That file already defines a `_groups(nbytes)`
helper at line 7 and already has
`test_evict_until_free_counts_prefill_evictions_separately` (line 99) — **reuse `_groups`, do
not redefine it**, and put these tests in the same style alongside the existing ones:

```python
def test_evicted_keys_is_none_when_tracking_off():
    from vaayu.persist.expert_cache import ExpertVRAMCache
    c = ExpertVRAMCache(budget_bytes=10_000)
    assert c.evicted_keys is None


def test_put_pressure_evictions_are_attributed_per_key():
    from vaayu.persist.expert_cache import ExpertVRAMCache
    c = ExpertVRAMCache(budget_bytes=1000, track_evicted_keys=True)
    assert c.put(1, 7, _groups(600))
    assert c.put(1, 8, _groups(600))          # over budget -> evicts (1,7)
    assert c.evicted_keys[(1, 7)] == 1
    assert c.evictions == 1


def test_prefill_wipe_evictions_are_not_attributed():
    """THE discriminating test. evict_until_free() is the deliberate wholesale
    prefill wipe (1362 entries in one measured pass). Attributing it per-expert
    would drown the budget-pressure signal and make every expert look like a
    pin candidate. It must increment prefill_evictions only."""
    import torch
    from vaayu.persist.expert_cache import ExpertVRAMCache
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("evict_until_free reads CUDA free memory")
    c = ExpertVRAMCache(budget_bytes=10_000, track_evicted_keys=True)
    c.put(2, 5, _groups(1000))
    free, _ = torch.cuda.mem_get_info()
    c.evict_until_free(free + 10_000)          # force a wipe
    assert c.prefill_evictions >= 1
    assert c.evicted_keys[(2, 5)] == 0         # NOT attributed
    assert sum(c.evicted_keys.values()) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_vram_cache.py -v
```

Expected: FAIL — `AttributeError: 'ExpertVRAMCache' object has no attribute 'evicted_keys'`
(and a `TypeError` on the unexpected `track_evicted_keys` kwarg).

- [ ] **Step 3: Add the constructor flag**

In `vaayu/persist/expert_cache.py`, ensure `Counter` is imported at the top of the file
(`from collections import Counter, OrderedDict` — check what is already imported and extend it
rather than adding a second import line).

Change `ExpertVRAMCache.__init__`'s signature from:

```python
    def __init__(self, budget_bytes: int):
```

to:

```python
    def __init__(self, budget_bytes: int, track_evicted_keys: bool = False):
```

and add, after the existing `self.prefill_evictions = 0` line:

```python
        # Per-(layer, expert) attribution of BUDGET-PRESSURE evictions only, for
        # the expert usage histogram. None unless explicitly enabled (ships dark).
        #
        # Deliberately NOT incremented by evict_until_free(): that is the
        # wholesale prefill wipe (measured at 1362 entries in a single pass),
        # not the LRU failing to afford an expert. Counting it here would drown
        # the signal and make every expert look like a pin candidate.
        self.evicted_keys = Counter() if track_evicted_keys else None
```

- [ ] **Step 4: Attribute the eviction in `put()`**

In `put()`, the eviction loop currently discards the key. Replace:

```python
            while self.current_usage + size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= self._group_bytes(evicted)
                self.evictions += 1
                del evicted  # drop GPU references so the allocator can reuse
```

with:

```python
            while self.current_usage + size > self.budget_bytes and self.cache:
                _ekey, evicted = self.cache.popitem(last=False)
                self.current_usage -= self._group_bytes(evicted)
                self.evictions += 1
                if self.evicted_keys is not None:
                    self.evicted_keys[_ekey] += 1
                del evicted  # drop GPU references so the allocator can reuse
```

Do NOT touch `evict_until_free()`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_vram_cache.py -v
```

Expected: PASS. The prefill-wipe test SKIPs on a CPU-only box; that is expected and acceptable —
it runs on the dev machine, which has CUDA.

- [ ] **Step 6: Run the full suite**

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Expected: green, 412 passed (409 + 3).

- [ ] **Step 7: Commit**

```bash
git add vaayu/persist/expert_cache.py tests/test_expert_vram_cache.py
git commit -m "feat(expert-cache): attribute budget-pressure evictions per key

Opt-in Counter keyed (layer, expert), incremented only in put()'s eviction
loop. evict_until_free() is deliberately excluded: it is the wholesale prefill
wipe (1362 entries in one measured pass), not the LRU failing to afford an
expert, and attributing it would drown the signal so every expert looked like
a pin candidate. A test pins that distinction.

Off by default (evicted_keys is None) — no behavior change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: The report (TDD)

A pure function so it is testable without a GPU, a model, or a live cache.

**Files:**
- Modify: `vaayu/expert_usage.py`
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Consumes: Task 1's `snapshot()` tensor, Task 2's `evicted_keys` Counter.
- Produces, for Task 5: `format_usage_report(counts, evicted_keys, top_n=5) -> str`, and the
  thin method `ExpertUsageTracker.report(vram_cache, top_n=5) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_usage.py`:

```python
from vaayu.expert_usage import format_usage_report


def test_report_flags_hot_and_evicted_as_pin_candidate():
    counts = torch.zeros((2, 4), dtype=torch.int32)
    counts[1, 2] = 500      # hot AND evicted -> candidate
    counts[1, 3] = 400      # hot, never evicted -> not a candidate
    out = format_usage_report(counts, {(1, 2): 12}, top_n=5)
    assert "layer=1" in out
    assert "e2" in out and "500" in out and "12" in out
    assert "pin candidate" in out
    assert "no pin needed" in out          # e3 must be labelled the other way


def test_report_says_no_pin_needed_when_nothing_was_evicted():
    """The negative result must be as legible as the positive one — it is what
    retires the pin feature."""
    counts = torch.zeros((1, 3), dtype=torch.int32)
    counts[0, 1] = 900
    out = format_usage_report(counts, {}, top_n=5)
    assert "no pin needed" in out
    assert "pin candidate" not in out


def test_report_skips_layers_with_no_routes():
    counts = torch.zeros((3, 4), dtype=torch.int32)
    counts[2, 0] = 7
    out = format_usage_report(counts, {}, top_n=5)
    assert "layer=2" in out
    assert "layer=0" not in out
    assert "layer=1" not in out


def test_report_on_empty_histogram_does_not_crash():
    out = format_usage_report(torch.zeros((4, 8), dtype=torch.int32), {}, top_n=5)
    assert isinstance(out, str)
    assert "no expert routes recorded" in out


def test_report_respects_top_n():
    counts = torch.zeros((1, 10), dtype=torch.int32)
    for e in range(10):
        counts[0, e] = 100 - e
    out = format_usage_report(counts, {}, top_n=3)
    assert "e0" in out and "e1" in out and "e2" in out
    assert "e9" not in out
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py -v
```

Expected: FAIL — `ImportError: cannot import name 'format_usage_report'`.

- [ ] **Step 3: Implement the report**

Append to `vaayu/expert_usage.py`:

```python
def format_usage_report(counts, evicted_keys, top_n: int = 5) -> str:
    """Join route counts with per-key budget-pressure evictions.

    `counts` is a [n_rows, n_experts] CPU tensor (ExpertUsageTracker.snapshot()),
    `evicted_keys` a mapping {(layer, expert): count} (ExpertVRAMCache.evicted_keys)
    or {} when attribution was off.

    An expert with many routes and ZERO evictions needs no pin — the cache is
    already keeping it resident. That negative result is what retires the pin
    feature, so it is labelled just as plainly as a candidate.
    """
    lines = []
    total_routes = int(counts.sum().item())
    if total_routes == 0:
        return "[Vaayu-Usage] no expert routes recorded (feature off, or no decode ran)"

    hot_evictions = 0
    total_evictions = sum(evicted_keys.values()) if evicted_keys else 0

    for layer_idx in range(counts.shape[0]):
        row = counts[layer_idx]
        layer_total = int(row.sum().item())
        if layer_total == 0:
            continue
        order = sorted(range(counts.shape[1]), key=lambda e: -int(row[e].item()))
        lines.append(f"[Vaayu-Usage] layer={layer_idx} top experts by routes:")
        for e in order[:top_n]:
            routes = int(row[e].item())
            if routes == 0:
                continue
            ev = int(evicted_keys.get((layer_idx, e), 0)) if evicted_keys else 0
            if ev > 0:
                hot_evictions += ev
                tag = "<-- pin candidate"
            else:
                tag = "(resident, no pin needed)"
            lines.append(f"  e{e}  {routes} routes  {ev} evictions  {tag}")

    share = (hot_evictions / total_evictions) if total_evictions else 0.0
    lines.append(
        f"summary: {total_routes} routes, {total_evictions} budget-pressure evictions; "
        f"top-{top_n} experts account for {hot_evictions} of them ({share:.0%})")
    return "\n".join(lines)
```

And add this method to `ExpertUsageTracker`:

```python
    def report(self, vram_cache=None, top_n: int = 5) -> str:
        """Human-readable join of this histogram with the VRAM cache's per-key
        eviction attribution. Safe to call with no cache (reports routes only)."""
        evicted = getattr(vram_cache, "evicted_keys", None) or {}
        return format_usage_report(self.snapshot(), evicted, top_n=top_n)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py -v
```

Expected: PASS — all 12 tests.

- [ ] **Step 5: Run the full suite**

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Expected: green, 417 passed (412 + 5).

- [ ] **Step 6: Commit**

```bash
git add vaayu/expert_usage.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): report joining routes with evictions

Pure formatting function, testable without a GPU or a live cache. Labels a
high-routes/zero-evictions expert 'no pin needed' as plainly as it labels a
pin candidate — that negative result is what would retire the pin feature, so
it must be as legible as the positive one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Wire it in, dark by default

**Files:**
- Modify: `vaayu/vaayu_base.py` (flag + tracker construction)
- Modify: `vaayu/moe_patch.py` (one helper + three call sites)
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Consumes: Task 1's `ExpertUsageTracker`, Task 2's `track_evicted_keys` kwarg.
- Produces: `model._expert_usage` — a tracker or `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_usage.py`:

```python
def test_hook_is_a_noop_without_a_tracker_and_during_prefill():
    """Ships dark: no tracker -> the hook must return before touching anything.
    And with a tracker, a prefill pass must not be counted — the VRAM cache is
    not consulted during prefill, so counting prefill routes would compare
    decode-side evictions against prefill+decode routes."""
    from vaayu.moe_patch import _maybe_record_usage

    class _Ref:
        pass

    ref = _Ref()                      # no _expert_usage attribute at all
    _maybe_record_usage(ref, 0, torch.tensor([[1]], dtype=torch.int64))   # must not raise

    ref._expert_usage = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    ref._moe_prefill_pass = True
    _maybe_record_usage(ref, 1, torch.tensor([[2]], dtype=torch.int64))
    assert ref._expert_usage.snapshot().sum().item() == 0     # prefill not counted

    ref._moe_prefill_pass = False
    _maybe_record_usage(ref, 1, torch.tensor([[2]], dtype=torch.int64))
    assert ref._expert_usage.snapshot()[1, 2].item() == 1     # decode counted
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py::test_hook_is_a_noop_without_a_tracker_and_during_prefill -v
```

Expected: FAIL — `ImportError: cannot import name '_maybe_record_usage'`.

- [ ] **Step 3: Add the hook helper to `moe_patch.py`**

Insert immediately **after** the `_ensure_prefill_headroom` function ends (it is the
established precedent for a `vaayu_ref`/`layer_idx` helper, and the new hook follows the same
prefill guard):

```python
def _maybe_record_usage(vaayu_ref, layer_idx, expert_indices) -> None:
    """Count DECODE-time expert routes when the usage histogram is enabled.

    A no-op when the feature is off (the common case — one getattr). Skipped
    during prefill on purpose: the VRAM expert cache is not consulted in a
    prefill pass, so there are no budget-pressure evictions then, and counting
    prefill routes would make the report compare decode-side evictions against
    prefill+decode routes. See the measurement design spec.
    """
    tracker = getattr(vaayu_ref, "_expert_usage", None)
    if tracker is None:
        return
    if getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    tracker.record(layer_idx, expert_indices)
```

- [ ] **Step 4: Call it from the three routing sites**

Each site derives `unique_experts` right after routing; insert the hook on the line **before**
that derivation. All three already have `vaayu_ref` and `layer_idx` in scope.

**Site A — module-list sparse forward** (`moe_patch.py`, ~line 761). Find:

```python
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()
        if _prof:
```

and insert above it so it reads:

```python
        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()
        if _prof:
```

**Site B — fused sparse, non-EXL3** (`moe_patch.py`, ~line 1048). Find:

```python
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
```

and insert above it so it reads:

```python
        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
```

**Site C — fused sparse, EXL3** (`moe_patch.py`, ~line 1410). Find:

```python
        expert_indices_cpu = expert_indices.cpu()
        unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)
```

and insert above it so it reads:

```python
        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        expert_indices_cpu = expert_indices.cpu()
        unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)
```

Note Site A and Site B share an identical anchor line, so apply each edit within its own
function body — do not use a global find-and-replace.

- [ ] **Step 5: Add the flag and construct the tracker in `vaayu_base.py`**

Add `enable_expert_usage=False` to `from_pretrained`'s keyword arguments, alongside the existing
`enable_prefix_cache` / `enable_kv_persist` flags, and store it:

```python
        self.enable_expert_usage = enable_expert_usage
```

Then, where the VRAM expert cache is constructed (the `if torch.cuda.is_available() and
str(self.device).startswith("cuda"):` block that creates `ExpertVRAMCache`), pass the tracking
flag through:

```python
                    self._expert_vram_cache = ExpertVRAMCache(
                        _vram_budget,
                        track_evicted_keys=bool(getattr(self, "enable_expert_usage", False)),
                    )
```

And immediately after that block, construct the tracker (MoE only — dense has no experts):

```python
            # Expert usage histogram (measurement only, ships dark). Rows are
            # indexed by the model's own layer_idx; both dimensions derived (§5).
            self._expert_usage = None
            if getattr(self, "enable_expert_usage", False) and self.is_moe:
                from vaayu.expert_usage import ExpertUsageTracker
                self._expert_usage = ExpertUsageTracker(
                    n_rows=int(self.config.num_hidden_layers),
                    n_experts=int(self._moe_geometry["num_experts"]),
                    device=self.running_device,
                )
                print(f"[Vaayu-Usage] expert usage histogram ON "
                      f"({self.config.num_hidden_layers} rows x "
                      f"{self._moe_geometry['num_experts']} experts)")
```

Ensure `self._expert_usage = None` is also set on the path where no VRAM cache is built, so the
attribute always exists. If `self.running_device` is not the attribute name used nearby for the
compute device, use whichever one the surrounding code uses — match local style.

- [ ] **Step 6: Run the tests**

```bash
source venv/bin/activate
python -m pytest tests/test_expert_usage.py -v && python -m pytest tests/ -q
```

Expected: `tests/test_expert_usage.py` all PASS; full suite green at 418 passed (417 + 1).

- [ ] **Step 7: Verify the feature is genuinely dark**

```bash
source venv/bin/activate
python -m pytest tests/test_moe_patch.py tests/test_moe_streaming_perf.py -q
```

Expected: green — the MoE forwards behave identically with the flag off.

- [ ] **Step 8: Commit**

```bash
git add vaayu/moe_patch.py vaayu/vaayu_base.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): wire the histogram in, off by default

One _maybe_record_usage helper called from the three MoE routing sites, plus
an enable_expert_usage flag (default False) that constructs the tracker and
turns on the cache's per-key eviction attribution.

The hook skips prefill deliberately: the VRAM cache is not consulted during a
prefill pass, so counting prefill routes would compare decode-side evictions
against prefill+decode routes. With the flag off it returns on one getattr.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: On-device measurement + the pin decision

Run on the real GPU with MiniMax, read the report, and decide whether a pin is worth building.

**Files:**
- Create: `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement-RESULTS.md`

- [ ] **Step 1: Add a temporary reporting call to `test_generate.py`**

Set `enable_expert_usage=True` in the `AutoModel.from_pretrained(...)` call, and after
generation completes add:

```python
if getattr(model, "_expert_usage", None) is not None:
    print(model._expert_usage.report(getattr(model, "_expert_vram_cache", None), top_n=5))
```

This edit is temporary scaffolding for the measurement — revert it in Step 5.

- [ ] **Step 2: Run it on MiniMax**

`test_generate.py` currently points at the Qwen path. Switch `MODEL_PATH` to
`/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3` and set a fresh `SESSION_ID`
(a stale persisted KV under another model's id would poison the run — the file warns about
this). Then:

```bash
source venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u test_generate.py > run-expert-usage-minimax.log 2>&1
echo "EXIT=$?"
```

- [ ] **Step 3: Extract the evidence**

```bash
grep -A 40 "Vaayu-Usage" run-expert-usage-minimax.log | head -60
grep -E "Decode throughput|OutOfMemoryError" run-expert-usage-minimax.log
```

Record: the top experts per layer with their route and eviction counts, the summary line, the
total budget-pressure eviction count, and the decode tok/s.

- [ ] **Step 4: Check for a decode regression**

The hook adds one `index_add_` per MoE layer per token. Compare decode tok/s against the
MiniMax floor of **~1.47 tok/s** (measured 1.49 on 2026-08-01). If decode dropped materially,
that is a real finding — record it, and note that the feature being off by default means it
does not affect shipped performance.

- [ ] **Step 5: Revert the scaffolding**

Restore `test_generate.py` to its committed state (`MODEL_PATH` back to the Qwen path,
`SESSION_ID` back to `"ramdiag-qwen"`, `enable_expert_usage` and the report call removed):

```bash
git checkout test_generate.py
git status --short          # test_generate.py must NOT appear
```

- [ ] **Step 6: Write the RESULTS doc and make the decision**

Create `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement-RESULTS.md` with:
- The report output (top experts per layer, routes vs budget-pressure evictions, summary).
- Total budget-pressure evictions observed during decode.
- Decode tok/s with the feature ON vs the ~1.47 floor.
- A note that `run-expert-usage-minimax.log` is operator-local and untracked.
- **The decision, stated plainly as a one-line VERDICT:**
  - **Hot experts show meaningful budget-pressure evictions** → a VRAM-tier pin has a real
    target. Next step: a pin spec sized against the decode headroom Plan A established.
  - **Hot experts show ~zero budget-pressure evictions** → the cache already keeps them
    resident. **Do not build the pin.** Mark the 2026-07-19 plan's pin tasks (3 and 6) as
    premise-invalidated in `todo.md`, citing this measurement.

Record whichever outcome the data gives. The second is a success — it retires a planned feature
on evidence rather than building it on assumption.

- [ ] **Step 7: Update `todo.md` and commit**

Mark Plan B's status per the verdict, add this plan's tasks as complete, and update
`_Last updated:`. Commit the RESULTS doc and `todo.md` together.

---

## Success Criteria

- `ExpertUsageTracker` counts decode routes per (layer, expert) with no host sync in the hot
  path; `snapshot()` is the only device→host transfer.
- `ExpertVRAMCache.evicted_keys` attributes **only** `put()` budget-pressure evictions; a test
  proves `evict_until_free()` evictions are excluded.
- The report labels high-routes/zero-eviction experts as needing no pin, and renders on an empty
  histogram without crashing.
- With `enable_expert_usage=False` (the default): no tracker, no Counter, hook returns on one
  getattr, `pytest tests/ -q` green at 418.
- An on-device report exists, and the pin question is answered with a recorded VERDICT either
  way.

## Self-Review notes (author)

- **Spec coverage:** tracker (Task 1), per-key budget-pressure-only attribution (Task 2), the
  report including the negative-result labelling (Task 3), decode-only rule + three hook sites +
  ships-dark flag (Task 4), verification and the explicit two-branch decision (Task 5). Every
  spec section maps to a task.
- **Type consistency:** `ExpertUsageTracker(n_rows, n_experts, device)`, `.counts`, `.record()`,
  `.snapshot()`, `.report()`, `format_usage_report(counts, evicted_keys, top_n)`, and
  `ExpertVRAMCache(budget_bytes, track_evicted_keys)` / `.evicted_keys` are used identically in
  every task that references them.
- **Deliberately not built:** persistence, decay, `compute_pin_set`, and the pin itself — see
  the spec's disposition table. Recorded, not silently dropped.
- **Known risk:** the prefill-wipe test needs CUDA and SKIPs on a CPU-only box. It runs on the
  dev machine. It is the plan's most important test, so a reviewer should confirm it actually
  ran rather than skipped.
- **Known risk:** `index_add_` per MoE layer per token is small but not free. Task 5 Step 4
  measures it rather than assuming; the feature is off by default regardless.
