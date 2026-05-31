# Asymmetric Layer Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace shared-only VRAM pinning with topology-aware asymmetric pinning that also pins the first N transformer layers permanently, reducing NVMe reads per token.

**Architecture:** Add `max_pinned_layers` param to `VaayuBaseModel.__init__()`. Rewrite `_init_pinned_layers()` as a two-phase loop: Phase 1 pins shared layers with `continue` on budget miss; Phase 2 pins transformer layers (0…N) with `break` on budget miss. `forward()` is unchanged — it already handles any set of pinned layer names.

**Tech Stack:** Python, PyTorch, pytest, `unittest.mock`

---

## File Map

| File | Change |
|------|--------|
| `vaayu/vaayu_base.py` | Add `max_pinned_layers=0` to `__init__()` signature + store; rewrite `_init_pinned_layers()` lines 655–698 |
| `tests/test_vaayu_base_kwargs.py` | Add 2 tests: default and stored value for `max_pinned_layers` |
| `tests/test_pinned_layers.py` | New file — 6 unit tests for `_init_pinned_layers()` behavior |
| `test_quick.py` | Add `--max_pinned_layers` CLI arg |

---

## Task 1: Add `max_pinned_layers` parameter

**Files:**
- Modify: `vaayu/vaayu_base.py:161–165` (signature), `vaayu/vaayu_base.py:249` (store)
- Test: `tests/test_vaayu_base_kwargs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vaayu_base_kwargs.py`:

```python
def test_max_pinned_layers_default_is_zero():
    m = _make_base()
    assert m.max_pinned_layers == 0


def test_max_pinned_layers_stored():
    m = _make_base({"max_pinned_layers": -1})
    assert m.max_pinned_layers == -1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/a/code/ai-ml/Vaayu
python -m pytest tests/test_vaayu_base_kwargs.py::test_max_pinned_layers_default_is_zero tests/test_vaayu_base_kwargs.py::test_max_pinned_layers_stored -v
```

Expected: `AttributeError: 'VaayuBaseModel' object has no attribute 'max_pinned_layers'`

- [ ] **Step 3: Add parameter to `__init__()` signature**

In `vaayu/vaayu_base.py`, change lines 161–165 from:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto",
                 layer_names_dict=None, tokenizer_class=None):
```

to:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto", max_pinned_layers=0,
                 layer_names_dict=None, tokenizer_class=None):
```

- [ ] **Step 4: Store the parameter**

In `vaayu/vaayu_base.py`, after line 249 (`self.kv_cache_mode = kv_cache_mode`), add:

```python
        self.max_pinned_layers = max_pinned_layers
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_vaayu_base_kwargs.py::test_max_pinned_layers_default_is_zero tests/test_vaayu_base_kwargs.py::test_max_pinned_layers_stored -v
```

Expected: `2 passed`

- [ ] **Step 6: Run full existing test suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_vaayu_base_kwargs.py
git commit -m "feat: add max_pinned_layers param to VaayuBaseModel.__init__()"
```

---

## Task 2: Rewrite `_init_pinned_layers()`

**Files:**
- Modify: `vaayu/vaayu_base.py:655–698`
- Test: `tests/test_pinned_layers.py` (create new)

- [ ] **Step 1: Create test file with all 6 failing tests**

Create `tests/test_pinned_layers.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import patch

from tests.test_vaayu_base_kwargs import _make_base


def _setup_loader(m, layer_size_bytes=100):
    n = layer_size_bytes // 4  # float32 elements
    m.loader_strategy.load_from_disk.return_value = {"weight": torch.zeros(n, dtype=torch.float32)}
    m.loader_strategy.inject_to_gpu.return_value = None


def _run_pinning(m, budget_bytes):
    m.pinned_layer_names.clear()
    with patch.object(m, "_get_vram_pinning_budget", return_value=budget_bytes):
        m._init_pinned_layers()


def test_zero_budget_pins_nothing():
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m)
    _run_pinning(m, budget_bytes=0)
    assert len(m.pinned_layer_names) == 0


def test_shared_only_mode_pins_no_transformer_layers():
    m = _make_base({"max_pinned_layers": 0})
    _setup_loader(m)
    _run_pinning(m, budget_bytes=10_000)
    assert "model.layers.0" not in m.pinned_layer_names
    assert "model.embed_tokens" in m.pinned_layer_names
    assert "model.norm" in m.pinned_layer_names


def test_auto_mode_pins_transformer_layers_until_budget():
    # FakeModel has 4 transformer layers, each 100 bytes.
    # Budget 500: 3 shared (300) + 2 transformer (200) fit; 3rd transformer would hit 600.
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=500)
    xformer = [n for n in m.pinned_layer_names if n.startswith("model.layers.")]
    assert len(xformer) == 2
    assert "model.layers.0" in m.pinned_layer_names
    assert "model.layers.1" in m.pinned_layer_names
    assert "model.layers.2" not in m.pinned_layer_names


def test_explicit_n_caps_transformer_layers():
    # max_pinned_layers=2 caps at 2 even if budget allows all 4.
    m = _make_base({"max_pinned_layers": 2})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=10_000)
    xformer = [n for n in m.pinned_layer_names if n.startswith("model.layers.")]
    assert len(xformer) == 2
    assert "model.layers.2" not in m.pinned_layer_names


def test_shared_phase_continue_skips_oversized_layer():
    # embed=1000 bytes (too big for budget=300), norm+lm_head=100 bytes each.
    # With continue: norm and lm_head still get pinned after embed is skipped.
    m = _make_base({"max_pinned_layers": 0})

    def fake_load(layer_name, checkpoint_path):
        if layer_name == "model.embed_tokens":
            return {"weight": torch.zeros(250, dtype=torch.float32)}  # 1000 bytes
        return {"weight": torch.zeros(25, dtype=torch.float32)}       # 100 bytes

    m.loader_strategy.load_from_disk.side_effect = fake_load
    m.loader_strategy.inject_to_gpu.return_value = None
    _run_pinning(m, budget_bytes=300)
    assert "model.embed_tokens" not in m.pinned_layer_names
    assert "model.norm" in m.pinned_layer_names
    assert "lm_head" in m.pinned_layer_names  # layer_names_dict["lm_head"] == "lm_head"


def test_transformer_layers_pinned_in_order():
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=500)  # fits 3 shared + 2 transformer
    xformer = sorted(n for n in m.pinned_layer_names if n.startswith("model.layers."))
    assert xformer == ["model.layers.0", "model.layers.1"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_pinned_layers.py -v
```

Expected: most tests fail because current `_init_pinned_layers()` ignores `max_pinned_layers`.

- [ ] **Step 3: Replace `_init_pinned_layers()` (lines 655–698)**

In `vaayu/vaayu_base.py`, replace the entire `_init_pinned_layers` method (lines 655–698) with:

```python
    def _init_pinned_layers(self) -> None:
        budget = self._get_vram_pinning_budget()
        if budget <= 0:
            return

        layer_prefix = self.layer_names_dict["layer_prefix"]

        # Phase 1: shared layers — continue on budget miss (each has independent size)
        used = 0
        for layer_name in [self.layer_names_dict["embed"],
                           self.layer_names_dict["norm"],
                           self.layer_names_dict["lm_head"]]:
            try:
                tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                size = sum(t.nbytes for t in tensors.values())
                if used + size > budget:
                    print(f"[Vaayu-Pin] skip {layer_name}: budget exhausted "
                          f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                    continue
                self.loader_strategy.inject_to_gpu(
                    self.model, layer_name, tensors,
                    self.running_device, self.running_dtype,
                    self.hf_quantizer, getattr(self, "_path_remappings", {}),
                )
                self.pinned_layer_names.add(layer_name)
                used += size
                print(f"[Vaayu-Pin] pinned {layer_name}: "
                      f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
            except Exception as e:
                print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        # Phase 2: transformer layers — break on budget miss (homogeneous size)
        if self.max_pinned_layers != 0:
            model_attr = self.model
            for attr_name in layer_prefix.split("."):
                model_attr = getattr(model_attr, attr_name)
            total_layers = len(model_attr)

            for i in range(total_layers):
                if self.max_pinned_layers > 0 and i >= self.max_pinned_layers:
                    break
                layer_name = f"{layer_prefix}.{i}"
                try:
                    tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                    size = sum(t.nbytes for t in tensors.values())
                    if used + size > budget:
                        print(f"[Vaayu-Pin] budget exhausted at {layer_name} "
                              f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                        break
                    self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    self.pinned_layer_names.add(layer_name)
                    used += size
                    print(f"[Vaayu-Pin] pinned {layer_name}: "
                          f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        pinned_xformer = sum(1 for n in self.pinned_layer_names
                             if n.startswith(layer_prefix + "."))
        pinned_shared = len(self.pinned_layer_names) - pinned_xformer
        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} layers pinned "
            f"({pinned_xformer} transformer + {pinned_shared} shared), "
            f"{used / 1e9:.2f} GB reserved"
        )
```

- [ ] **Step 4: Run new tests to confirm they pass**

```bash
python -m pytest tests/test_pinned_layers.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_pinned_layers.py
git commit -m "feat: rewrite _init_pinned_layers() with two-phase asymmetric pinning"
```

---

## Task 3: Add `--max_pinned_layers` CLI arg to `test_quick.py`

**Files:**
- Modify: `test_quick.py`

- [ ] **Step 1: Add argparse argument**

In `test_quick.py`, after the existing `--kv_cache_mode` argument (line 17–19), add:

```python
parser.add_argument("--max_pinned_layers", type=int, default=0,
                    help="0=shared only (default), -1=auto fill VRAM budget, N=first N transformer layers")
```

- [ ] **Step 2: Pass arg to `from_pretrained()`**

In `test_quick.py`, change line 37 from:

```python
    kv_cache_mode=args.kv_cache_mode,
```

to:

```python
    kv_cache_mode=args.kv_cache_mode,
    max_pinned_layers=args.max_pinned_layers,
```

- [ ] **Step 3: Verify CLI arg is wired (dry run)**

```bash
cd /home/a/code/ai-ml/Vaayu
python test_quick.py --help
```

Expected output includes:
```
--max_pinned_layers MAX_PINNED_LAYERS
                    0=shared only (default), -1=auto fill VRAM budget, N=first N transformer layers
```

- [ ] **Step 4: Commit**

```bash
git add test_quick.py
git commit -m "feat: add --max_pinned_layers CLI arg to test_quick.py"
```

---

## Verification (manual, requires GPU)

After all tasks are committed, run on real hardware to confirm gains:

```bash
# Baseline: shared-only (current behavior)
python test_quick.py --kv_cache_mode fp8 --max_pinned_layers 0

# Full auto pinning: check [Vaayu-Pin] logs for ~28 transformer + 3 shared
python test_quick.py --kv_cache_mode fp8 --max_pinned_layers -1
```

Expected log (with 70B EXL3 4.5bpw on 16 GB GPU):
```
[Vaayu-Pin] pinned model.embed_tokens: 0.53 GB  (cumulative 0.53 GB)
[Vaayu-Pin] pinned model.norm: 0.00 GB  (cumulative 0.53 GB)
[Vaayu-Pin] pinned model.layers.0: 0.38 GB  (cumulative 0.91 GB)
...
[Vaayu-Pin] budget exhausted at model.layers.28 (11.64/11.64 GB)
[Vaayu-Pin] 31 layers pinned (28 transformer + 3 shared), 11.64 GB reserved
```

Check `nvidia-smi`: ~11–12 GB VRAM used, no OOM.
Check `[VAAYU-IO] token summary`: `io_wait` drops 30–50% vs baseline.
