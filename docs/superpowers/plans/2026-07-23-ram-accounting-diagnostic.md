# RAM Accounting Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Everything needed is in this file. No prior session context required.

**Goal:** Produce a complete, evidence-backed accounting of where Vaayu's ~22GB steady-state RSS goes on a 33GB box, so the follow-on RAM-rebalance plan can be sized from measurement rather than estimate.

**Architecture:** This is an INVESTIGATION plan, not a feature. It adds temporary instrumentation, takes measurements, writes a findings document, and then REMOVES the instrumentation it doesn't want to keep. Its deliverable is a document plus (optionally) a small permanent diagnostic, not a performance change.

**Tech Stack:** Python, PyTorch, `/proc/self/smaps_rollup`, `psutil`.

## Background (read first — you have no prior context)

Vaayu is a low-resource LLM inference engine (see `CLAUDE.md`). On this box:
- 33.40GB total RAM, 16GB GPU.
- Loading MiniMax-M2.7 (an EXL3-quantized MoE) reaches **~22.2GB steady-state RSS**.
- A pinned expert slab arena takes **9.19GB** (unswappable). An expert RAM cache is budgeted **9.13GB**.
- Free RAM after load is only **~6.5-8GB**, which starves the cross-turn prefix cache (its budget formula is `max(0, free_ram - 12.88GB reserve)`).

A prior session established a partial ledger via startup prints (already committed, in `vaayu/vaayu_base.py`, search `[Vaayu-RAM] ledger`):

```
ledger @ cache-sizing: 24.52GB avail of 33.40GB (8.88GB already resident)
ledger @ post-arena:    7.05GB avail
```

So ~17.5GB of "available" disappears between those two points while the arena is only 9.19GB. About **5GB is unexplained**. Two untested hypotheses:
1. Pinned memory is unswappable and always counted resident, whereas the pre-arena baseline's pageable clones were OS-reclaimable — i.e. the old RSS figure understated the true working set.
2. Transient prefill staging buffers (the module-list prefill path loads misses through a pageable transient path) inflate the peak.

Already ruled out by inspection: the io_uring pinned buffer pools in `vaayu/persist/safetensor_model_persister.py` are small (`(32 if exl3 else 512) MB × ≤8` ≈ 256MB for EXL3), not the source.

**Why it matters:** CLAUDE.md §0 states the target workload is parallel fan-out (multiple concurrent inference streams). With only ~6.5GB free there is little headroom, and the follow-on rebalance plan needs to know how much RAM is genuinely reclaimable.

## Global Constraints

- **No model-specific hardcoded numbers** in any code you add (CLAUDE.md §5). Expert counts, layer counts, and byte sizes must be derived. Hardware limits (33GB RAM, 16GB VRAM, the 12.88GB prefix reserve) may be literals and are already defined.
- **Do not change engine behavior.** This plan measures; it must not alter cache sizing, eviction, or any inference path. If you find a bug, record it in the findings doc — do not fix it here.
- **Temporary instrumentation must be removed** before the final commit unless Task 4 explicitly decides to keep it.
- Run tests with the project venv: `source venv/bin/activate` first. The suite is `pytest tests/ -q` (373 tests at time of writing) and must stay green.
- On-device runs need the real models and the GPU; they are slow (~4-5 min each). Budget for that.

---

### Task 1: Add a categorized RAM ledger

**Files:**
- Modify: `vaayu/vaayu_base.py` — extend the existing `[Vaayu-RAM] ledger` prints (search for that string; there are two, at cache-sizing and post-arena).
- Test: `tests/test_ram_ledger.py` (create)

**Interfaces:**
- Produces: `vaayu.vaayu_base.ram_snapshot() -> dict` — returns a dict of RAM categories in bytes: `{"rss": int, "pss": int, "shared_clean": int, "shared_dirty": int, "private_clean": int, "private_dirty": int, "swap": int, "available": int, "total": int}`. Reads `/proc/self/smaps_rollup` (fast, one file, kernel-aggregated) with `psutil.virtual_memory()` for the system totals. Returns zeros for any field the kernel does not expose, never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ram_ledger.py`:

```python
"""RAM accounting snapshot — categorized /proc/self/smaps_rollup reader."""
import pytest
from vaayu.vaayu_base import ram_snapshot


def test_ram_snapshot_has_all_keys():
    snap = ram_snapshot()
    for key in ("rss", "pss", "shared_clean", "shared_dirty",
                "private_clean", "private_dirty", "swap", "available", "total"):
        assert key in snap, f"missing {key}"
        assert isinstance(snap[key], int)
        assert snap[key] >= 0


def test_ram_snapshot_rss_is_plausible():
    """RSS must be non-zero for a live process and not exceed total RAM."""
    snap = ram_snapshot()
    assert snap["rss"] > 0
    assert snap["total"] > 0
    assert snap["rss"] <= snap["total"]


def test_ram_snapshot_private_sums_under_rss():
    """private_clean + private_dirty cannot exceed RSS (they are subsets)."""
    snap = ram_snapshot()
    assert snap["private_clean"] + snap["private_dirty"] <= snap["rss"] * 1.05


def test_ram_snapshot_never_raises(monkeypatch):
    """A kernel without smaps_rollup must degrade to zeros, not crash."""
    import builtins
    real_open = builtins.open

    def _boom(path, *a, **kw):
        if "smaps_rollup" in str(path):
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _boom)
    snap = ram_snapshot()          # must not raise
    assert snap["pss"] == 0        # smaps-derived fields degrade to 0
    assert snap["total"] > 0       # psutil-derived fields still populated
```

- [ ] **Step 2: Run to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_ram_ledger.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ram_snapshot' from 'vaayu.vaayu_base'`.

- [ ] **Step 3: Implement `ram_snapshot`**

Add to `vaayu/vaayu_base.py`, near the other module-level helpers (put it just above `compute_prefix_cache_budget`, which you can find by searching for `def compute_prefix_cache_budget`):

```python
def ram_snapshot() -> dict:
    """Categorized process RAM accounting, for the memory-arbitration ledger.

    Reads /proc/self/smaps_rollup — the kernel's pre-aggregated rollup, one small
    file, far cheaper than parsing per-mapping /proc/self/smaps. Fields:
      rss            resident set size
      pss            proportional set size (shared pages divided by sharer count)
      shared_clean   file-backed pages shared and unmodified (e.g. mmapped weights)
      shared_dirty   shared and modified
      private_clean  private and unmodified
      private_dirty  private and modified — anonymous allocations live here, which
                     is where a pinned arena and PyTorch host buffers show up
      swap           pages pushed to swap (pinned memory can never appear here)
    plus `available`/`total` from psutil for the system view.

    Degrades to zeros for smaps-derived fields on a kernel without smaps_rollup;
    never raises, because this is diagnostics and must not break a load.
    """
    import psutil as _ps
    _FIELDS = {
        "Rss": "rss", "Pss": "pss",
        "Shared_Clean": "shared_clean", "Shared_Dirty": "shared_dirty",
        "Private_Clean": "private_clean", "Private_Dirty": "private_dirty",
        "Swap": "swap",
    }
    out = {v: 0 for v in _FIELDS.values()}
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in _FIELDS:
                    out[_FIELDS[key]] = int(rest.strip().split()[0]) * 1024
    except Exception:
        pass   # diagnostics only — a missing/!readable smaps_rollup must not break load
    _vm = _ps.virtual_memory()
    out["available"] = int(_vm.available)
    out["total"] = int(_vm.total)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_ram_ledger.py -v
pytest tests/ -q
```

Expected: 4 new tests PASS, full suite green.

- [ ] **Step 5: Wire the snapshot into the existing ledger prints**

In `vaayu/vaayu_base.py`, find the two existing ledger prints (search `[Vaayu-RAM] ledger`). Replace each with a categorized line. The first currently reads roughly:

```python
            print(f"[Vaayu-RAM] ledger @ cache-sizing: {_mem.available / 1e9:.2f}GB "
                  f"avail of {_mem.total / 1e9:.2f}GB "
                  f"({(_mem.total - _mem.available) / 1e9:.2f}GB already resident)")
```

Replace with:

```python
            _snap = ram_snapshot()
            print(f"[Vaayu-RAM] ledger @ cache-sizing: avail={_snap['available'] / 1e9:.2f}GB "
                  f"rss={_snap['rss'] / 1e9:.2f}GB priv_dirty={_snap['private_dirty'] / 1e9:.2f}GB "
                  f"shared_clean={_snap['shared_clean'] / 1e9:.2f}GB swap={_snap['swap'] / 1e9:.2f}GB")
```

And the post-arena one similarly, keeping its `@ post-arena` label. Add a THIRD identical print at the very end of model construction — put it immediately after the `cross-turn prefix cache ENABLED` print (search for that string), labelled `@ post-load`, so the ledger brackets the whole load.

- [ ] **Step 6: Commit**

```bash
source venv/bin/activate && pytest tests/ -q
git add vaayu/vaayu_base.py tests/test_ram_ledger.py
git commit -m "feat(diag): categorized RAM snapshot for the arbitration ledger

ram_snapshot() reads /proc/self/smaps_rollup for rss/pss/shared/private/swap
breakdown, so the load-time ledger distinguishes pinned anonymous allocations
(private_dirty) from file-backed mmapped weights (shared_clean) instead of
reporting a single opaque RSS number.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Measure — on-device, three configurations

**Files:** `test_generate.py` (config only). Run by the operator.

**Interfaces:** none. Deliverable is raw logs.

Each run: set a FRESH `SESSION_ID` (a reused id hits the prefix cache and changes the load path), `MAX_NEW_TOKENS = 32` (this task measures load-time memory, not throughput — keep runs short).

- [ ] **Step 1: MiniMax, current config (arena ON)**

```bash
source venv/bin/activate
# test_generate.py: MODEL_PATH = /home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3
#                   SESSION_ID = "ramdiag-minimax-arena"
python -u test_generate.py > run-ramdiag-arena.log 2>&1
grep -E "Vaayu-RAM|Expert slab arena|Expert RAM cache|prefix cache ENABLED" run-ramdiag-arena.log
```

- [ ] **Step 2: MiniMax, arena FORCED OFF**

This isolates the arena's contribution. Temporarily force the arena off by making the slab-capability predicate return False. In `vaayu/vaayu_base.py`, find `def moe_supports_slab_arena` and add `return False` as its first line (a temporary edit — revert after this run).

```bash
source venv/bin/activate
# SESSION_ID = "ramdiag-minimax-noarena"
python -u test_generate.py > run-ramdiag-noarena.log 2>&1
grep -E "Vaayu-RAM|Expert slab arena|prefix cache ENABLED" run-ramdiag-noarena.log
# REVERT the temporary `return False` now:
git checkout vaayu/vaayu_base.py   # if you have no other uncommitted changes to it
```

**Take care:** if you have other uncommitted edits to `vaayu_base.py`, revert the single line by hand instead of `git checkout`.

- [ ] **Step 3: Qwen, current config**

Qwen is the other on-device target and has always had the arena, so it is the control for "is this MiniMax-specific?"

```bash
source venv/bin/activate
# MODEL_PATH = /home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw
# SESSION_ID = "ramdiag-qwen"
python -u test_generate.py > run-ramdiag-qwen.log 2>&1
grep -E "Vaayu-RAM|Expert slab arena|prefix cache ENABLED" run-ramdiag-qwen.log
```

- [ ] **Step 4: Restore the harness**

Set `MODEL_PATH` back to MiniMax and `MAX_NEW_TOKENS` back to 256. Confirm `git diff vaayu/` is empty (no leftover temporary edits).

---

### Task 3: Analyze and write the findings document

**Files:**
- Create: `docs/superpowers/plans/2026-07-23-ram-accounting-RESULTS.md`

**Interfaces:** none.

- [ ] **Step 1: Build the comparison table**

From the three logs, tabulate at each of the three checkpoints (`@ cache-sizing`, `@ post-arena`, `@ post-load`): `avail`, `rss`, `priv_dirty`, `shared_clean`, `swap`.

- [ ] **Step 2: Answer these questions explicitly, each with the number that supports it**

1. **How much of RSS is `private_dirty`** (anonymous — pinned arena, PyTorch host buffers) vs **`shared_clean`** (file-backed mmapped model weights, reclaimable by the OS under pressure)? File-backed pages inflate RSS but are NOT a real memory cost, since the kernel can drop them.
2. **What is the arena's true marginal cost?** `arena-ON priv_dirty` minus `arena-OFF priv_dirty` at `@ post-load`. If ≈9.19GB, the arena is fully accounted and the "5GB mystery" is elsewhere. If materially less, the arena partly displaced allocations that existed anyway.
3. **Does the arena-OFF run show a correspondingly larger `shared_clean` or pageable footprint?** That would confirm hypothesis 1 (the old pageable-clone baseline understated the true working set because those pages were reclaimable).
4. **Is `swap` non-zero anywhere?** Pinned pages can never swap; if swap is non-zero the box is under genuine pressure and the "available" figure is optimistic.
5. **How much RAM is genuinely reclaimable** — i.e. what could a rebalance actually free? Give a number with its derivation. This is the single most important output; the follow-on rebalance plan consumes it.

- [ ] **Step 3: State the verdict on the 5GB**

Write one of: **explained** (with the breakdown), **partially explained** (with what is and isn't), or **still unexplained** (with what you ruled out and what instrument would settle it). Do not overclaim — "partially explained" with honest gaps is a valid and useful result.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-23-ram-accounting-RESULTS.md
git commit -m "docs(diag): RAM accounting findings

<one-line verdict on the ~5GB and the reclaimable-RAM number>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Decide what instrumentation to keep

**Files:** `vaayu/vaayu_base.py`, possibly `tests/test_ram_ledger.py`.

- [ ] **Step 1: Decide**

The categorized ledger prints three lines per load. Keep them if the findings show they are diagnostic-valuable on an ongoing basis (likely yes for a memory-constrained local-first engine); trim to one `@ post-load` line if three is noise. `ram_snapshot()` and its tests should be KEPT either way — it is a small, tested, reusable diagnostic the rebalance plan will want.

- [ ] **Step 2: Apply and commit**

```bash
source venv/bin/activate && pytest tests/ -q
git add -A vaayu/ tests/
git commit -m "chore(diag): finalize RAM ledger instrumentation

<what you kept and why>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Success Criteria

- `ram_snapshot()` exists, is tested (4 tests), and never raises.
- Three on-device logs captured with the categorized ledger.
- A results document that answers all five questions in Task 3 Step 2 with numbers.
- **A stated reclaimable-RAM figure** for the follow-on rebalance plan to consume.
- `git diff vaayu/` clean of temporary edits; full suite green.
