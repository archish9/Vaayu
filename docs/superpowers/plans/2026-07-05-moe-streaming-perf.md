# MoE Streaming Performance (Fix A + B + D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Mixtral-8x7B decode latency from ~9s/token to <0.5s/token by fixing the init-order bug, expanding the expert cache to cover the full working set, and adding pre-token batch prefetch.

**Architecture:** All three fixes touch only `vaayu/vaayu_base.py`. Fix A adds `non_expert_pinned_layer_names` so non-expert weight pinning skips inject/free without incorrectly bypassing expert loading. Fix B increases the expert RAM cache budget so all 64 decode-time experts fit. Fix D submits prefetch tasks for all 32 layers at the start of each forward pass using the previous token's routing history.

**Tech Stack:** Python 3.10+, PyTorch, psutil, concurrent.futures.ThreadPoolExecutor

## Global Constraints

- Run all tests from the repo root with: `source venv/bin/activate && python -m pytest tests/ -q`
- Baseline: 7 tests currently fail in `tests/test_moe_patch.py` — do not introduce new failures
- Integration verification after each task: `source venv/bin/activate && python -u test_generate.py > run.log 2>&1`
- Model: Mixtral-8x7B-Instruct-v0.1, 32 transformer layers, 8 experts per layer, top-2 routing
- Expert weight size: ~352 MB per expert (3 matrices: w_gate, w_up, w_down)
- Per-token working set: 2 experts × 32 layers = 64 experts = ~22.5 GB

---

## File Map

| File | Role | Changes |
|---|---|---|
| `vaayu/vaayu_base.py` | Engine core | Fix A (init order, pinning sets, forward loop), Fix B (budget), Fix D (prefetch) |
| `tests/test_moe_streaming_perf.py` | New test file | Unit tests for all three fixes |

---

## Task 1: Fix A — Non-Expert Pinning Infrastructure

**Files:**
- Modify: `vaayu/vaayu_base.py`
- Create: `tests/test_moe_streaming_perf.py`

**What this fixes:** `_expert_offset_index` is built at line 504, after `_init_pinned_layers()` at line 476. Phase 2 pinning checks `getattr(self, '_expert_offset_index', None) is not None` — always `None` at pin time — so loads full 2.9 GB layers instead of 80 MB non-expert only. Budget exhausts after 3 layers, leaving 29 layers to inject/free every token.

**Critical bug in previous approach:** Adding non-expert-pinned layers to `pinned_layer_names` would cause `sparse_forward` to call `original_forward` (the pre-patch MoE forward), which calls all 8 experts using weight tensors — but expert tensors are on **meta device** after non-expert-only pinning → crash. Solution: separate `non_expert_pinned_layer_names` set.

- [ ] **Step 1: Write failing tests**

Create `tests/test_moe_streaming_perf.py`:

```python
"""
Unit tests for MoE streaming performance fixes.
Tests are designed to run without a real GPU or model checkpoint.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ── Task 1: Fix A — non_expert_pinned_layer_names ─────────────────────────────

def test_non_expert_pinned_names_is_separate_set():
    """After init, non_expert_pinned_layer_names must exist and be distinct from pinned_layer_names."""
    from vaayu.vaayu_base import VaayuBase
    # Access the attribute declarations — the class must declare it
    # We test by checking it appears in a real instance's __init__ code path.
    # Use a mock subclass to avoid full model loading.
    obj = object.__new__(VaayuBase)
    obj.pinned_layer_names = set()
    obj.non_expert_pinned_layer_names = set()
    obj.pinned_layer_names.add("embed")
    obj.non_expert_pinned_layer_names.add("model.layers.5")
    assert "embed" not in obj.non_expert_pinned_layer_names
    assert "model.layers.5" not in obj.pinned_layer_names


def test_non_expert_pinned_excludes_transformer_layer_from_pinned():
    """Transformer layers pinned via non-expert strategy must NOT appear in pinned_layer_names."""
    # This documents the invariant: if a layer is in non_expert_pinned_layer_names,
    # original_forward bypass (which fires for pinned_layer_names) must NOT fire for it.
    pinned = {"model.embed_tokens", "model.norm", "lm_head"}
    non_expert_pinned = {f"model.layers.{i}" for i in range(32)}
    for layer in non_expert_pinned:
        assert layer not in pinned


def test_prefetch_all_predicted_experts_skips_cached():
    """_prefetch_all_predicted_experts submits tasks only for uncached experts."""
    from vaayu.vaayu_base import VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {0: [0, 1, 2], 1: [3, 4]}
    obj._expert_cache = MagicMock()
    # expert (layer=0, idx=0) is already cached; all others are not
    obj._expert_cache.has.side_effect = lambda layer, expert: (layer == 0 and expert == 0)
    obj._prefetch_experts = MagicMock()

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)

    # Layer 0: [0] cached → submit [1, 2]. Layer 1: [3, 4] uncached → submit [3, 4].
    assert executor.submit.call_count == 2
    submitted_args = [c.args[1:] for c in executor.submit.call_args_list]  # (layer_idx, uncached)
    assert (0, [1, 2]) in submitted_args
    assert (1, [3, 4]) in submitted_args


def test_prefetch_all_predicted_experts_empty_history():
    """_prefetch_all_predicted_experts does nothing when history is empty."""
    from vaayu.vaayu_base import VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {}
    obj._expert_cache = MagicMock()

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)
    executor.submit.assert_not_called()


def test_prefetch_all_predicted_experts_none_cache():
    """_prefetch_all_predicted_experts does nothing when cache is None."""
    from vaayu.vaayu_base import VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {0: [0, 1]}
    obj._expert_cache = None

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)
    executor.submit.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py -v
```

Expected: `test_prefetch_all_predicted_experts_*` fail with `AttributeError: type object 'VaayuBase' has no attribute '_prefetch_all_predicted_experts'`. The invariant tests (`test_non_expert_pinned_*`) pass immediately — they document constraints, not new code.

- [ ] **Step 3: Add `non_expert_pinned_layer_names` attribute**

In `vaayu/vaayu_base.py`, find line 464:
```python
        self.pinned_layer_names: set = set()
```
Change to:
```python
        self.pinned_layer_names: set = set()
        self.non_expert_pinned_layer_names: set = set()
```

- [ ] **Step 4: Move ExpertOffsetIndex build before `_init_pinned_layers()`**

In `vaayu/vaayu_base.py`, find lines 473–476:
```python
        self.draft_tokens = draft_tokens if draft_tokens is not None else 5

        if device.startswith("cuda"):
            self._init_pinned_layers()
```
Change to:
```python
        self.draft_tokens = draft_tokens if draft_tokens is not None else 5

        # Build offset index BEFORE _init_pinned_layers so Phase 2 can use
        # _load_non_expert_strategy (80 MB) instead of full layer load (2.9 GB).
        self._expert_offset_index = None
        if device.startswith("cuda") and self.is_moe:
            from .persist.expert_offset_index import ExpertOffsetIndex
            self._expert_offset_index = ExpertOffsetIndex(str(self.checkpoint_path))
            self._expert_offset_index.build_or_load()

        if device.startswith("cuda"):
            self._init_pinned_layers()
```

- [ ] **Step 5: Remove the now-redundant offset index init from the MoE block**

In `vaayu/vaayu_base.py`, find lines 490–505:
```python
        # MoE Expert-Aware Streaming: build offset index, expert cache, patch forwards
        self._expert_history: dict[int, list[int]] = {}
        self._expert_cache = None
        self._expert_offset_index = None
        self._moe_blocks: list = []
        # Non-expert tensor dict cache: avoids re-reading 80 MB per non-pinned MoE layer
        # per token. Populated lazily on first access, cleared on del. ~2.5 GB total.
        self._non_expert_tensor_cache: dict[str, dict] = {}
        if self.running_device.startswith("cuda") and self.is_moe:
            from .persist.expert_offset_index import ExpertOffsetIndex
            from .persist.expert_cache import ExpertRAMCache
            from .moe_patch import patch_moe_blocks
            import psutil as _psutil

            self._expert_offset_index = ExpertOffsetIndex(str(self.checkpoint_path))
            self._expert_offset_index.build_or_load()
```
Change to:
```python
        # MoE Expert-Aware Streaming: expert cache and patched forwards.
        # _expert_offset_index is initialized above (before _init_pinned_layers).
        self._expert_history: dict[int, list[int]] = {}
        self._expert_cache = None
        self._moe_blocks: list = []
        # Non-expert tensor dict cache: avoids re-reading 80 MB per non-pinned MoE layer
        # per token. Populated lazily on first access, cleared on del. ~2.5 GB total.
        self._non_expert_tensor_cache: dict[str, dict] = {}
        if self.running_device.startswith("cuda") and self.is_moe:
            from .persist.expert_cache import ExpertRAMCache
            from .moe_patch import patch_moe_blocks
            import psutil as _psutil
```

Note: `self._expert_offset_index = None` is removed (set above). The `from .persist.expert_offset_index import ExpertOffsetIndex` import is removed (moved above). The two `ExpertOffsetIndex` lines are removed (done above).

- [ ] **Step 6: Update Phase 2 to add to `non_expert_pinned_layer_names`**

In `vaayu/vaayu_base.py`, find lines 1083–1092 (the `inject_to_gpu` + `pinned_layer_names.add` block inside Phase 2):
```python
                    self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    self.pinned_layer_names.add(layer_name)
                    used += size
                    print(f"[Vaayu-Pin] pinned {layer_name}: "
                          f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
```
Change to:
```python
                    self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    _is_non_expert_pin = (
                        self.is_moe and getattr(self, "_expert_offset_index", None) is not None
                    )
                    if _is_non_expert_pin:
                        self.non_expert_pinned_layer_names.add(layer_name)
                    else:
                        self.pinned_layer_names.add(layer_name)
                    used += size
                    _pin_label = "non-expert-pinned" if _is_non_expert_pin else "pinned"
                    print(f"[Vaayu-Pin] {_pin_label} {layer_name}: "
                          f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
```

- [ ] **Step 7: Update the summary print at end of `_init_pinned_layers`**

Find lines 1095–1102:
```python
        pinned_xformer = sum(1 for n in self.pinned_layer_names
                             if n.startswith(layer_prefix + "."))
        pinned_shared = len(self.pinned_layer_names) - pinned_xformer
        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} layers pinned "
            f"({pinned_xformer} transformer + {pinned_shared} shared), "
            f"{used / 1e9:.2f} GB reserved"
        )
```
Change to:
```python
        pinned_xformer = sum(1 for n in self.pinned_layer_names
                             if n.startswith(layer_prefix + "."))
        pinned_shared = len(self.pinned_layer_names) - pinned_xformer
        non_expert_xformer = len(self.non_expert_pinned_layer_names)
        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} fully-pinned "
            f"({pinned_xformer} transformer + {pinned_shared} shared), "
            f"{non_expert_xformer} non-expert-pinned transformer, "
            f"{used / 1e9:.2f} GB reserved"
        )
```

- [ ] **Step 8: Update forward loop seed prefetch to skip both sets**

Find lines 1648–1654:
```python
            if self.prefetching:
                for _seed_name in self.layer_names:
                    if _seed_name not in self.pinned_layer_names:
                        if _seed_name in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                            future = executor.submit(self._load_non_expert_strategy, _seed_name)
                        else:
                            future = executor.submit(self._load_from_strategy, _seed_name)
                        break
```
Change to:
```python
            if self.prefetching:
                _all_pinned = self.pinned_layer_names | getattr(self, 'non_expert_pinned_layer_names', set())
                for _seed_name in self.layer_names:
                    if _seed_name not in _all_pinned:
                        if _seed_name in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                            future = executor.submit(self._load_non_expert_strategy, _seed_name)
                        else:
                            future = executor.submit(self._load_from_strategy, _seed_name)
                        break
```

- [ ] **Step 9: Update forward loop main pinned-layer fast-path check**

Find lines 1670–1683:
```python
                if layer_name in self.pinned_layer_names:
                    # ── Pinned: already in VRAM — zero disk/transfer cost ────
                    _lt_cache_hit = True
                    moved_layers = []
                    # If no future is outstanding, kick off the next non-pinned load
                    # so disk read overlaps with this layer's GPU compute.
                    if self.prefetching and future is None:
                        for _n in self.layer_names[i + 1:]:
                            if _n not in self.pinned_layer_names:
                                if _n in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                                    future = executor.submit(self._load_non_expert_strategy, _n)
                                else:
                                    future = executor.submit(self._load_from_strategy, _n)
                                break
```
Change to:
```python
                _non_expert_pinned = getattr(self, 'non_expert_pinned_layer_names', set())
                if layer_name in self.pinned_layer_names or layer_name in _non_expert_pinned:
                    # ── Pinned (full or non-expert): already in VRAM — zero disk/transfer cost ──
                    _lt_cache_hit = True
                    moved_layers = []
                    # If no future is outstanding, kick off the next non-pinned load
                    # so disk read overlaps with this layer's GPU compute.
                    if self.prefetching and future is None:
                        _all_pinned_fwd = self.pinned_layer_names | _non_expert_pinned
                        for _n in self.layer_names[i + 1:]:
                            if _n not in _all_pinned_fwd:
                                if _n in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                                    future = executor.submit(self._load_non_expert_strategy, _n)
                                else:
                                    future = executor.submit(self._load_from_strategy, _n)
                                break
```

- [ ] **Step 10: Run unit tests**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py -v
```

Expected: `test_non_expert_pinned_names_is_separate_set` and `test_non_expert_pinned_excludes_transformer_layer_from_pinned` PASS (pure Python logic, no new method needed). The 3 `test_prefetch_all_predicted_experts_*` tests still FAIL — method added in Task 3.

- [ ] **Step 11: Run full test suite to check no regressions**

```bash
source venv/bin/activate && python -m pytest tests/ -q
```

Expected: same 7 failures as baseline (pre-existing `test_moe_patch.py` failures), no new failures.

- [ ] **Step 12: Integration check — verify 32 non-expert-pinned layers in log**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "non-expert-pinned\|Vaayu-Pin" run.log | head -40
```

Expected output (32 lines like):
```
[Vaayu-Pin] non-expert-pinned model.layers.0: 0.08 GB  (cumulative N GB)
...
[Vaayu-Pin] non-expert-pinned model.layers.31: 0.08 GB  (cumulative ~3.1 GB)
[Vaayu-Pin] 3 fully-pinned (0 transformer + 3 shared), 32 non-expert-pinned transformer, ~3.1 GB reserved
```

And `gpu_load` in token summaries should be near 0.

- [ ] **Step 13: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_moe_streaming_perf.py
git commit -m "perf: fix init-order bug — pin all 32 MoE layers' non-expert weights"
```

---

## Task 2: Fix B — Expert Cache Budget

**Files:**
- Modify: `vaayu/vaayu_base.py`
- Modify: `tests/test_moe_streaming_perf.py`

**What this fixes:** Expert cache budget = 50% of total RAM = 46 expert slots. Per-token working set = 64 experts. Cache thrashes at 87% miss rate → ~8s NVMe reads per token. RAM layer cache gets ~20 GB budget but is never used for MoE models (pre-population skipped). Zeroing the RAM cache budget frees headroom; raising the expert budget to `max(50%, total - 6 GB)` gives 77 expert slots on a 33 GB machine.

- [ ] **Step 1: Write failing test**

Add to `tests/test_moe_streaming_perf.py`:

```python
# ── Task 2: Fix B — Expert cache budget ───────────────────────────────────────

def test_expert_budget_exceeds_working_set():
    """Expert cache budget must be large enough to hold 64-expert working set (~22.5 GB)."""
    import psutil
    mem = psutil.virtual_memory()
    _os_reserve = 6 * 1024**3
    expert_budget = max(int(mem.total * 0.50), int(mem.total - _os_reserve))
    expert_size_bytes = 352 * 1024 * 1024  # 352 MB per expert
    working_set_bytes = 64 * expert_size_bytes  # 64 experts
    assert expert_budget >= working_set_bytes, (
        f"Budget {expert_budget / 1e9:.1f} GB < working set {working_set_bytes / 1e9:.1f} GB"
    )


def test_expert_budget_greater_than_old_formula():
    """New budget formula must be larger than the old 50% formula on this machine."""
    import psutil
    mem = psutil.virtual_memory()
    old_budget = int(mem.total * 0.50)
    _os_reserve = 6 * 1024**3
    new_budget = max(int(mem.total * 0.50), int(mem.total - _os_reserve))
    assert new_budget >= old_budget
```

- [ ] **Step 2: Run tests to confirm they pass or show real failure**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py::test_expert_budget_exceeds_working_set tests/test_moe_streaming_perf.py::test_expert_budget_greater_than_old_formula -v
```

Expected: `test_expert_budget_exceeds_working_set` FAILS if old formula (46 slots < 64). `test_expert_budget_greater_than_old_formula` PASSES (trivially true when `total - 6 GB > total * 0.5`, i.e. total > 12 GB).

- [ ] **Step 3: Zero RAM layer cache budget for MoE**

In `vaayu/vaayu_base.py`, find lines 480–484:
```python
        if device.startswith("cuda"):
            from .persist.ram_cache import LayerRAMCache, auto_detect_cache_budget
            _cache_budget = auto_detect_cache_budget()
            self.ram_cache = LayerRAMCache(_cache_budget)
            print(f"[Vaayu-Cache] RAM layer cache: {_cache_budget / 1e9:.1f} GB budget")
```
Change to:
```python
        if device.startswith("cuda"):
            from .persist.ram_cache import LayerRAMCache, auto_detect_cache_budget
            # MoE models skip pre-population entirely; the RAM layer cache budget
            # is wasted allocation. Give that memory to the expert cache instead.
            _cache_budget = 0 if self.is_moe else auto_detect_cache_budget()
            self.ram_cache = LayerRAMCache(_cache_budget)
            print(f"[Vaayu-Cache] RAM layer cache: {_cache_budget / 1e9:.1f} GB budget")
```

- [ ] **Step 4: Increase expert cache budget formula**

In `vaayu/vaayu_base.py`, find lines 507–515:
```python
            _mem = _psutil.virtual_memory()
            # 50% of total RAM: frees budget from skipped MoE pre-population (Fix A).
            # 33 GB × 0.50 = 16.5 GB → 46 expert slots (vs 33 at 0.35).
            # Budget sanity: 16.5 GB expert + 0 GB pre-pop + 10.6 GB RAM cache + 6 GB OS/CUDA
            # = ~33 GB — fits on 33 GB machines. Adjust down if OOM observed.
            _expert_budget = int(_mem.total * 0.50)
            self._expert_cache = ExpertRAMCache(_expert_budget)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"(50% of {_mem.total / 1e9:.0f} GB total RAM)")
```
Change to:
```python
            _mem = _psutil.virtual_memory()
            # Budget = max(50% floor, total - 6 GB OS reserve).
            # 33 GB machine: max(16.5, 27) = 27 GB cap → ~77 expert slots.
            # Per-token working set = 64 experts × 352 MB = 22.5 GB.
            # Budget is a cap, not pre-allocation: cache grows to ~22.5 GB and stops.
            _os_reserve = 6 * 1024 ** 3
            _expert_budget = max(int(_mem.total * 0.50), int(_mem.total - _os_reserve))
            self._expert_cache = ExpertRAMCache(_expert_budget)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"({_expert_budget // (352 * 1024 * 1024)} expert slots on this machine)")
```

- [ ] **Step 5: Run tests**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py -v
```

Expected: all budget tests pass. Prefetch tests still fail (Task 3).

- [ ] **Step 6: Run full test suite**

```bash
source venv/bin/activate && python -m pytest tests/ -q
```

Expected: same 7 pre-existing failures, no new failures.

- [ ] **Step 7: Integration check — verify budget in log and hit rate improvement**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "Expert RAM cache\|hit_rate\|expert_cache" run.log | head -20
```

Expected:
- Log line like: `[Vaayu-MoE] Expert RAM cache: 27.0 GB budget (77 expert slots on this machine)`
- Token summary lines showing `hit_rate` jumping from 13% toward 85%+ by token 2–3

- [ ] **Step 8: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_moe_streaming_perf.py
git commit -m "perf: expand expert cache to cover full 64-expert working set"
```

---

## Task 3: Fix D — Pre-Token Batch Prefetch

**Files:**
- Modify: `vaayu/vaayu_base.py`
- Modify: `tests/test_moe_streaming_perf.py`

**What this fixes:** Current per-layer prefetch fires AFTER layer L completes, one layer at a time. Head-start ≈ 0. Adding a batch prefetch at the START of each forward pass submits tasks for all 32 layers using the previous token's routing history. Later layers (10–31) get several seconds of earlier layers' compute as head-start.

- [ ] **Step 1: Tests already written — run to confirm failure**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py::test_prefetch_all_predicted_experts_skips_cached tests/test_moe_streaming_perf.py::test_prefetch_all_predicted_experts_empty_history tests/test_moe_streaming_perf.py::test_prefetch_all_predicted_experts_none_cache -v
```

Expected: all three FAIL with `AttributeError: type object 'VaayuBase' has no attribute '_prefetch_all_predicted_experts'`.

- [ ] **Step 2: Add `_prefetch_all_predicted_experts` method**

In `vaayu/vaayu_base.py`, find the `_prefetch_experts` method ending at line ~1214. Add the new method immediately after it (after the last line of `_prefetch_experts`):

```python
    def _prefetch_all_predicted_experts(self, executor) -> None:
        """Submit prefetch tasks for all layers using previous token's routing history.

        Called once at the start of each forward pass (token N). Later layers get
        the compute time of all earlier layers as a prefetch window. Only submits
        tasks for experts not already in cache.
        """
        if not self._expert_history or self._expert_cache is None:
            return
        for layer_idx, expert_list in self._expert_history.items():
            uncached = [e for e in expert_list
                        if not self._expert_cache.has(layer_idx, e)]
            if uncached:
                executor.submit(self._prefetch_experts, layer_idx, uncached)
```

- [ ] **Step 3: Call batch prefetch at forward pass start**

In `vaayu/vaayu_base.py`, find lines 1643–1645:
```python
        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # ── Seed prefetch: skip pinned layers (already in VRAM) ──────────
```
Change to:
```python
        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # ── Pre-token batch prefetch: submit all layers using prev token's routing ──
            if self.is_moe and self._token_count > 0:
                self._prefetch_all_predicted_experts(executor)

            # ── Seed prefetch: skip pinned layers (already in VRAM) ──────────
```

- [ ] **Step 4: Run unit tests**

```bash
source venv/bin/activate && python -m pytest tests/test_moe_streaming_perf.py -v
```

Expected: all tests pass (including the three prefetch tests).

- [ ] **Step 5: Run full test suite**

```bash
source venv/bin/activate && python -m pytest tests/ -q
```

Expected: same 7 pre-existing failures, no new failures.

- [ ] **Step 6: Integration check — verify prefetch in log and latency improvement**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "tok/s\|s/tok\|compute\|gpu_load\|hit_rate" run.log | head -30
```

Expected after all three fixes:
- `gpu_load` ≈ 0s/token
- `compute` < 0.5s/token  
- Expert cache `hit_rate` ≥ 85% on decode tokens 2+
- Overall: <0.5s/token (vs ~9s before these fixes)

- [ ] **Step 7: Correctness check**

```bash
grep -i "asyncfilewatcher\|async.*file\|class.*watcher" run.log | head -5
python -c "
import re, sys
log = open('run.log').read()
# Extract generated text between first token output markers
tokens = re.findall(r'\b(import|def |class |async |await )\b', log)
print(f'Python keywords found: {len(tokens)} — model output looks valid')
"
```

Expected: generated output contains recognizable Python keywords. Not garbage tokens.

- [ ] **Step 8: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_moe_streaming_perf.py
git commit -m "perf: add pre-token batch prefetch for all 32 MoE layers"
```
