# Fix #1: Free RAM and Raise the Expert Cache Budget Above the Working Set

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` in this directory first.**
> **Strongly recommended: do Fix #6 (telemetry) before this**, so you can measure the result.

**Goal:** Size the expert RAM cache to exceed the model's **derived** decode
working set (not a hardcoded number) by reclaiming RAM from lower-value consumers,
up to the hardware OOM ceiling. For Qwen this raises the budget from ~7.5 GB to
~8.5 GB and stops the measured thrash; for MiniMax-M2.5 (working set ~5.5 GB) the
existing budget already suffices and this fix correctly becomes a no-op.

> **MODEL-AGNOSTIC.** Read the REPORT §1.5. This fix introduces `_moe_geometry`
> (the single source of truth) and sizes the budget from
> `_moe_geometry["working_set_bytes"]`. Never hardcode 11.28 MB / 91 / 8.02 GB —
> those are the *Qwen values* the derived formula happens to produce.

**Architecture:** The decode working set (one step's experts) is
`top_k × num_moe_layers × expert_bytes`, all measured at load time. The expert
cache is hard-capped at 7 GB (showed 7.5 GB at runtime). When working set >
budget (Qwen: 8.02 > 7.5), the cache is pegged full and evicts ~3 experts per
insert (measured), so hit rate is ~18%. This fix frees RAM from the non-expert
tensor cache and shrinks the OS/CUDA reserve margin so the hard cap can safely
rise toward a **hardware** ceiling of ~8.5 GB, and warns if the derived working
set still exceeds the achievable budget (model too big for cache on this box).

**Tech Stack:** Python 3.12, `psutil`. Files: `vaayu/vaayu_base.py`,
`vaayu/persist/expert_cache.py` (read-only reference), `tests/`.

## Global Constraints

- **The 32 GB RAM ceiling is real and model-independent.** A prior attempt to
  raise the cap to 10 GB caused a confirmed OOM-kill (`journalctl -k`, anon-rss
  26.46 GB). Do NOT just bump the constant — you MUST free equivalent RAM
  elsewhere first, and you MUST verify with a full-length run, not a short one.
- **Hardware OOM ceiling: 8.5 GB** — this is a *fixed* constant for this 32 GB
  box, NOT a model number. It is the max safe expert-cache size regardless of
  model. Do not exceed 9 GB without a longer verification run than the one that
  OOM'd at 10 GB.
- **All model numbers are derived**, never hardcoded: `expert_bytes`,
  `num_moe_layers`, `top_k`, `working_set_bytes` come from `_moe_geometry`
  (Task 0). Qwen produces 11.28 MB / 91 / 8 / 8.02 GB; M2.5 produces ~11.3 MB /
  62 / 8 / ~5.5 GB. The code must not care which.
- If `working_set > achievable_budget` (model too big for cache on this box),
  the fix prints a warning and proceeds — thrash is then a real hardware limit
  for that model, not a bug.
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

## Task 0: Add the model-agnostic `compute_moe_geometry` helper (single source of truth)

**Files:**
- Modify: `vaayu/vaayu_base.py` (add module-level helper; call it in MoE init)
- Test: `tests/test_moe_geometry.py` (create)

**Interfaces:**
- Produces: `compute_moe_geometry(expert_offset_index, num_layers, moe_top_k,
  moe_num_experts) -> dict` with keys `expert_bytes`, `num_moe_layers`, `top_k`,
  `num_experts`, `working_set_bytes`. Stored as `self._moe_geometry`. Consumed by
  Fix #1 (budget), Fix #3 (prefetch), and any other fix needing model numbers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_geometry.py`:

```python
"""Model-agnostic MoE geometry — derived from the offset index, never hardcoded."""
from vaayu.vaayu_base import compute_moe_geometry


class _FakeMeta:
    def __init__(self, byte_length): self.byte_length = byte_length


class _FakeIndex:
    """Mimics ExpertOffsetIndex.index: {layer: {expert: {key: meta}}}."""
    def __init__(self, num_layers, num_experts, expert_bytes):
        per_key = expert_bytes // 3
        self.index = {
            f"model.layers.{L}": {
                e: {
                    "gate.trellis": _FakeMeta(per_key),
                    "up.trellis": _FakeMeta(per_key),
                    "down.trellis": _FakeMeta(expert_bytes - 2 * per_key),
                }
                for e in range(num_experts)
            }
            for L in range(num_layers)
        }


def test_geometry_qwen_like():
    idx = _FakeIndex(num_layers=91, num_experts=128, expert_bytes=11_827_000)
    g = compute_moe_geometry(idx, num_layers=94, moe_top_k=8, moe_num_experts=128)
    assert g["num_moe_layers"] == 91
    assert abs(g["expert_bytes"] - 11_827_000) < 10
    # working set ~ 8 * 91 * 11.28MB ~ 8.02 GB
    assert 7.5e9 < g["working_set_bytes"] < 8.6e9


def test_geometry_minimax_like():
    idx = _FakeIndex(num_layers=62, num_experts=256, expert_bytes=11_850_000)
    g = compute_moe_geometry(idx, num_layers=62, moe_top_k=8, moe_num_experts=256)
    assert g["num_moe_layers"] == 62
    # working set ~ 8 * 62 * 11.3MB ~ 5.5 GB  → fits 7.5GB budget
    assert 5.0e9 < g["working_set_bytes"] < 6.2e9
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_moe_geometry.py -v`
Expected: FAIL — `compute_moe_geometry` undefined.

- [ ] **Step 3: Implement the helper**

Add module-level in `vaayu/vaayu_base.py` (near the top, after imports). Copy
exactly from REPORT §1.5:

```python
def compute_moe_geometry(expert_offset_index, num_layers, moe_top_k, moe_num_experts):
    """Model-agnostic MoE geometry, derived — never hardcoded (REPORT §1.5).

    Returns dict: expert_bytes, num_moe_layers, top_k, num_experts,
    working_set_bytes (= top_k * num_moe_layers * expert_bytes, one decode step).
    """
    expert_bytes = 0
    for layer_tensors in expert_offset_index.index.values():
        for expert_tensors in layer_tensors.values():
            expert_bytes = sum(t.byte_length for t in expert_tensors.values())
            break
        if expert_bytes:
            break
    num_moe_layers = len(expert_offset_index.index) or max(1, num_layers - 3)
    working_set_bytes = moe_top_k * num_moe_layers * expert_bytes
    return {
        "expert_bytes": expert_bytes,
        "num_moe_layers": num_moe_layers,
        "top_k": moe_top_k,
        "num_experts": moe_num_experts,
        "working_set_bytes": working_set_bytes,
    }
```

> Note on `num_moe_layers`: the offset index only contains MoE layers, so
> `len(index)` is the true MoE-layer count (91 for Qwen — some layers may be dense;
> 62 for M2.5). The `or max(1, num_layers-3)` is a fallback if the index is empty.

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_moe_geometry.py -v`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add tests/test_moe_geometry.py vaayu/vaayu_base.py
git commit -m "feat: add model-agnostic compute_moe_geometry (Fix #1 Task 0)"
```

---

## Task 1: Size the budget from the derived working set (up to the hardware ceiling)

**Files:**
- Modify: `vaayu/vaayu_base.py` (MoE-init block shown above)
- Test: `tests/test_expert_budget_sizing.py` (create)

**Interfaces:**
- Consumes: `working_set_bytes` from Task 0's geometry.
- Produces: `compute_expert_budget(available_bytes, working_set_bytes,
  num_moe_layers, expert_bytes, moe_top_k, moe_num_experts) -> (budget, ne_budget,
  max_per_layer)`, where `budget` exceeds `working_set_bytes` whenever RAM allows
  (capped at the 8.5 GB hardware ceiling).

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_budget_sizing.py`. This tests the *pure sizing formula*
by extracting it into a helper (Task 2 creates the helper; this test drives its
design). Start with the test:

```python
"""Fix #1: budget sizes above the DERIVED working set, up to the hardware ceiling.
Model-agnostic: tested with both Qwen-like and MiniMax-like geometry."""
from vaayu.vaayu_base import compute_expert_budget

GB = 1024 ** 3
MB = 1024 ** 2


def test_qwen_budget_exceeds_working_set_on_32gb_box():
    # Qwen: working set 8.02 GB. 26.5 GB available (from run.log).
    ws = int(8 * 91 * 11.28 * MB)  # ~8.02 GB
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128,
    )
    assert budget <= int(8.5 * GB) + 1, "must not exceed 8.5GB hardware ceiling"
    assert budget > ws, "budget must exceed Qwen working set (kills thrash)"


def test_minimax_budget_already_fits_no_change_needed():
    # M2.5: working set ~5.5 GB < existing budget. Budget still capped at 8.5GB,
    # and comfortably exceeds the smaller working set → no thrash.
    ws = int(8 * 62 * 11.3 * MB)  # ~5.5 GB
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=62, expert_bytes=int(11.3 * MB),
        moe_top_k=8, moe_num_experts=256,
    )
    assert budget > ws, "M2.5 working set must fit within budget"
    assert budget <= int(8.5 * GB) + 1


def test_budget_floor_on_tiny_ram():
    ws = int(8 * 91 * 11.28 * MB)
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=3 * GB, working_set_bytes=ws,
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128,
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
# Hardware constant for THIS 32 GB box: max safe expert-cache size before OOM.
# Model-independent. Raised from the old 7 GB only after shrinking other RAM
# consumers (below). Do NOT exceed 9 GB — 10 GB caused a confirmed OOM-kill.
_EXPERT_CACHE_OOM_CEILING = int(8.5 * 1024 ** 3)


def compute_expert_budget(
    available_bytes: int,
    working_set_bytes: int,
    num_moe_layers: int,
    expert_bytes: int,
    moe_top_k: int,
    moe_num_experts: int,
):
    """Size the expert RAM cache to exceed the DERIVED decode working set, up to
    the hardware OOM ceiling. Model-agnostic (Fix #1, REPORT §1.5).

    working_set_bytes comes from compute_moe_geometry — never hardcoded. When RAM
    allows, budget > working_set → no thrash (Qwen: raises to ~8.5 GB; M2.5: its
    5.5 GB working set already fits). When the model's working set exceeds the
    achievable budget, the caller warns and proceeds (real hardware limit).

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
        _EXPERT_CACHE_OOM_CEILING,               # hardware ceiling, not a model number
        max(1 * GB, available_bytes - overhead),
    )
    # max_per_layer derived from expert_bytes (any quant/model). Clamped so the
    # collective per-layer capacity never exceeds the global budget (Fix #4).
    per_layer_bytes = expert_budget // max(1, num_moe_layers)
    max_per_layer = max(
        moe_top_k,
        min(moe_num_experts, int(per_layer_bytes // max(1, expert_bytes))),
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
ExpertRAMCache(...)` line with the code below. The old inline `_expert_mb`
measurement loop is now **redundant** — `compute_moe_geometry` measures
`expert_bytes` itself. Remove that inline loop (grep `_expert_mb = _real_bytes`)
and let the geometry helper be the only place expert size is measured. New code:

```python
            # Model-agnostic geometry (Task 0) → single source of truth. No Qwen
            # numbers hardcoded; expert_bytes/num_moe_layers/working_set derived.
            self._moe_geometry = compute_moe_geometry(
                self._expert_offset_index,
                num_layers=len(self.layers),
                moe_top_k=self.moe_top_k,
                moe_num_experts=self.moe_num_experts,
            )
            _geo = self._moe_geometry
            _expert_budget, _ne_budget, _max_per_layer = compute_expert_budget(
                available_bytes=_mem.available,
                working_set_bytes=_geo["working_set_bytes"],
                num_moe_layers=_geo["num_moe_layers"],
                expert_bytes=_geo["expert_bytes"],
                moe_top_k=_geo["top_k"],
                moe_num_experts=_geo["num_experts"],
            )
            self._non_expert_tensor_cache = LRUTensorCache(_ne_budget)
            print(f"[Vaayu-Cache] Non-expert tensor cache: {_ne_budget / 1e9:.1f} GB budget")
            self._expert_cache = ExpertRAMCache(_expert_budget, max_per_layer=_max_per_layer)
            _ws_gb = _geo["working_set_bytes"] / 1e9
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"(~{int(_expert_budget // max(1, _geo['expert_bytes']))} expert slots, "
                  f"max {_max_per_layer}/layer, "
                  f"expert={_geo['expert_bytes']/1e6:.1f}MB, working_set={_ws_gb:.2f}GB)")
            if _geo["working_set_bytes"] > _expert_budget:
                print(f"[Vaayu-MoE] WARNING: working set {_ws_gb:.2f}GB exceeds cache "
                      f"budget {_expert_budget/1e9:.2f}GB — decode will thrash. This "
                      f"model is too large for the expert cache on this hardware; "
                      f"expect low hit rate (a real limit, not a bug).")
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

> **⚠️ DISABLE THE DRAFTER FOR THIS VERIFICATION** (REPORT §2.4): set
> `DRAFTER_MODEL_PATH = None` in `test_generate.py` for the verification run.
> With the drafter on, each verify pass routes ~(draft_tokens+1)×top_k unique
> experts per layer (~24 observed), a ~3× working set that no achievable budget
> fits — the hit-rate gain this fix promises is measurable only in single-token
> decode. Restore the drafter setting afterwards if desired.

- [ ] **Step 1: Confirm the printed budget at startup**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1 &
sleep 90 && grep "Expert RAM cache:" run.log
```
Expected (Qwen): `Expert RAM cache: 8.5 GB budget ... working_set=8.02GB`.
Expected (M2.5): `Expert RAM cache: 8.5 GB budget ... working_set=5.5GB` and NO
WARNING line (working set already fits — this fix is correctly a no-op there).

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
Expected (Qwen): `hit_rate` on decode steps rises from ~18% toward ~50-65%, and
`Kevict` drops from ~5,500 toward being comparable to or below misses (no longer
3× inserts). Decode throughput improves from 0.12 tok/s toward ~0.2+ tok/s.
Expected (M2.5): already high hit_rate before this fix (working set fits) — the
budget change does not regress it.

## Gotchas

- Budget must EXCEED the working set, not equal it — an exactly-working-set budget
  still thrashes at the margins. The 8.5 GB hardware ceiling gives Qwen (8.02 GB)
  ~0.5 GB headroom and M2.5 (5.5 GB) ~3 GB headroom. If a future model's *derived*
  working set exceeds 8.5 GB, the WARNING fires — that model genuinely does not fit
  this box's cache and will thrash (a real limit; consider harder quant / smaller
  model, not a code fix).
- `_mem.available` is measured at init, *before* KV cache and layer buffers grow.
  The `1.5 GB` OS margin + the 8.5 GB cap already account for this via the OOM
  history; that is why the cap exists as a `min()` ceiling, not just the formula.
- Raising `_max_per_layer` above `top_k=8` is Fix #4's job; here it may rise
  naturally because per_layer_bytes grew. That is fine and helps.
- If `cache=Nhit/Mmiss` on the summary line ever shows the non-expert cache doing
  real work (non-zero hits) after shrinking it, increase `ne_budget` back toward
  1 GB and lower the expert cap correspondingly.
