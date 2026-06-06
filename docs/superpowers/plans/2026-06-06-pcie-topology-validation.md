# PCIe Topology Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a startup diagnostic that warns when the GPU is in a bandwidth-limited PCIe slot, preventing silent per-token latency penalties.

**Architecture:** A module-level function `_check_pcie_topology()` in `vaayu/vaayu_base.py` shells out to `nvidia-smi` to read the current PCIe link width and generation, then prints a warning (or confirmation) to stdout. It is called once in `VaayuBaseModel.__init__()` when a CUDA device is in use. All errors are silently swallowed so the function can never crash startup.

**Tech Stack:** Python stdlib `subprocess`, `pytest` + `unittest.mock` for tests. No new dependencies.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `test_pcie_topology.py` | Unit tests for `_check_pcie_topology` via mocked subprocess |
| Modify | `vaayu/vaayu_base.py:145-147` | Insert function before `VaayuBaseModel` class |
| Modify | `vaayu/vaayu_base.py:247` | Call function in `__init__` gated on CUDA device |

---

### Task 1: Write Failing Tests

**Files:**
- Create: `test_pcie_topology.py`

- [ ] **Step 1: Create the test file**

```python
"""Unit tests for _check_pcie_topology in vaayu/vaayu_base.py."""
import subprocess
from unittest.mock import patch, MagicMock


def _import_fn():
    from vaayu.vaayu_base import _check_pcie_topology
    return _check_pcie_topology


def _make_result(stdout, returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


def test_optimal_x16_gen4_prints_ok(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("16, 4\n")):
        fn()
    out = capsys.readouterr().out
    assert "✅" in out
    assert "Gen4 x16" in out
    assert "25.6 GB/s" in out


def test_suboptimal_x8_gen4_prints_warning(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("8, 4\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "Gen4 x8" in out
    assert "12.8 GB/s" in out
    assert "per token" in out


def test_suboptimal_x4_gen3_prints_warning(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("4, 3\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "Gen3 x4" in out
    assert "3.9 GB/s" in out


def test_nvidia_smi_nonzero_returncode_silent(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("", returncode=1)):
        fn()
    out = capsys.readouterr().out
    assert out == ""


def test_subprocess_exception_silent(capsys):
    fn = _import_fn()
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        fn()
    out = capsys.readouterr().out
    assert out == ""


def test_unknown_pcie_combo_prints_warning_no_crash(capsys):
    fn = _import_fn()
    # Gen6 doesn't exist in the table yet
    with patch("subprocess.run", return_value=_make_result("8, 6\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "unknown bandwidth" in out
```

- [ ] **Step 2: Run tests — confirm they all fail with ImportError**

```bash
source venv/bin/activate && python -m pytest test_pcie_topology.py -v
```

Expected: `ImportError: cannot import name '_check_pcie_topology'` (function doesn't exist yet)

---

### Task 2: Implement `_check_pcie_topology()`

**Files:**
- Modify: `vaayu/vaayu_base.py` — insert function at line 146 (before `class VaayuBaseModel`)

- [ ] **Step 1: Add `import subprocess` to the existing imports at the top of the file**

In `vaayu/vaayu_base.py`, the first line is `import os`. Add `subprocess` to it:

```python
import os
import subprocess
```

- [ ] **Step 2: Insert `_check_pcie_topology()` before `class VaayuBaseModel`**

Find the blank line at line 146 (between the closing `}` of `find_checkpoint_paths` and `class VaayuBaseModel`). Insert after it:

```python

def _check_pcie_topology():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return

        parts = result.stdout.strip().split(", ")
        width = int(parts[0])
        gen = int(parts[1])

        bandwidth_gbps = {
            (3, 4): 3.9, (3, 8): 7.9, (3, 16): 15.8,
            (4, 4): 6.4, (4, 8): 12.8, (4, 16): 25.6,
            (5, 4): 12.6, (5, 8): 25.2, (5, 16): 50.4,
        }.get((gen, width), 0)

        if width < 16:
            bw_str = f"{bandwidth_gbps:.1f} GB/s" if bandwidth_gbps > 0 else "unknown bandwidth"
            penalty = f"~{132/bandwidth_gbps:.0f}s per token" if bandwidth_gbps > 0 else "unknown"
            print(f"⚠️  [Vaayu-Warning] GPU running at PCIe Gen{gen} x{width} ({bw_str})")
            print(f"   gpu_load limited to {penalty} for 132GB models.")
            print(f"   Moving GPU to primary x16 slot doubles bandwidth.")
        else:
            print(f"✅ [Vaayu] GPU PCIe link: Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")

    except Exception:
        pass

```

- [ ] **Step 3: Run tests — confirm all 6 pass**

```bash
source venv/bin/activate && python -m pytest test_pcie_topology.py -v
```

Expected output:
```
test_pcie_topology.py::test_optimal_x16_gen4_prints_ok PASSED
test_pcie_topology.py::test_suboptimal_x8_gen4_prints_warning PASSED
test_pcie_topology.py::test_suboptimal_x4_gen3_prints_warning PASSED
test_pcie_topology.py::test_nvidia_smi_nonzero_returncode_silent PASSED
test_pcie_topology.py::test_subprocess_exception_silent PASSED
test_pcie_topology.py::test_unknown_pcie_combo_prints_warning_no_crash PASSED
6 passed
```

- [ ] **Step 4: Commit**

```bash
git add vaayu/vaayu_base.py test_pcie_topology.py
git commit -m "feat: add _check_pcie_topology startup diagnostic"
```

---

### Task 3: Wire Into `__init__`

**Files:**
- Modify: `vaayu/vaayu_base.py` — add call in `__init__` after dtype selection

- [ ] **Step 1: Locate the insertion point**

In `vaayu/vaayu_base.py`, find `self.dtype = self.running_dtype` (currently ~line 247). The next line after it begins `self.use_torch_compile`. Insert between them:

```python
        self.dtype = self.running_dtype

        if device.startswith("cuda"):
            _check_pcie_topology()

        self.use_torch_compile = use_torch_compile
```

- [ ] **Step 2: Smoke-test via `test_quick.py`**

```bash
source venv/bin/activate && python test_quick.py 2>&1 | head -30
```

Expected: within the first ~30 lines you see either:
- `✅ [Vaayu] GPU PCIe link: GenX xY (Z.Z GB/s)` — GPU in x16 slot
- `⚠️  [Vaayu-Warning] GPU running at PCIe ...` — GPU in sub-optimal slot

No exception, no crash.

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: invoke PCIe topology check on CUDA startup"
```
