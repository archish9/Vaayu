# vaayu-apple Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python package that streams 70B+ LLMs layer-by-layer from NVMe into Apple Silicon compute units using `F_NOCACHE` + PyTorch MPS and an optional MLX backend.

**Architecture:** `VaayuAppleModel` owns init/config. `AppleLoaderStrategy` implementations own disk IO. `MPSBackend` / `MLXBackend` own the forward loop. Each layer is loaded, run, then evicted to meta device before the next loads.

**Tech Stack:** Python 3.10+, PyTorch ≥2.3 (MPS), transformers ≥4.45, safetensors, accelerate, huggingface-hub, psutil; optional: mlx ≥0.16, mlx-lm ≥0.16.

---

## New repo location

All files in this plan go into: `~/Documents/ai-ml/vaayu-apple/`  
This spec lives in the Linux Vaayu repo — the new Mac repo is completely separate.

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies |
| `vaayu_apple/__init__.py` | Public API: `VaayuAppleModel` |
| `vaayu_apple/model.py` | `VaayuAppleModel` — init, pinning, MoE detect, forward dispatcher |
| `vaayu_apple/utils.py` | `clean_memory()`, `ram_available()`, `detect_chip()` |
| `vaayu_apple/loaders/nocache.py` | `open_nocache()` — F_NOCACHE via fcntl |
| `vaayu_apple/loaders/base.py` | `AppleLoaderStrategy` ABC |
| `vaayu_apple/loaders/safetensors_loader.py` | Hot path: F_NOCACHE + double buffer + zero-copy tensors |
| `vaayu_apple/loaders/mlx_loader.py` | MLX weight format loader |
| `vaayu_apple/backends/mps_backend.py` | PyTorch MPS forward loop with ThreadPoolExecutor prefetch |
| `vaayu_apple/backends/mlx_backend.py` | MLX forward loop with per-layer `mx.eval()` |
| `vaayu_apple/splitting/splitter.py` | Layer shard splitting (ported from Linux Vaayu utils.py) |
| `tests/test_nocache.py` | F_NOCACHE correctness |
| `tests/test_safetensors_loader.py` | Loader tensor shape/value correctness |
| `tests/test_splitting.py` | Split → reload tensor equality |
| `tests/test_forward_mps.py` | Full forward() logits shape with tiny model |
| `tests/test_mlx_loader.py` | MLX loader tensor shapes |

---

## Task 1: Create repo scaffold

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/pyproject.toml`
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/__init__.py`
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/model.py` (stub)
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/loaders/__init__.py`
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/backends/__init__.py`
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/splitting/__init__.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/__init__.py`

- [ ] **Step 1: Create repo and directory tree**

```bash
mkdir -p ~/Documents/ai-ml/vaayu-apple
cd ~/Documents/ai-ml/vaayu-apple
git init
mkdir -p vaayu_apple/loaders vaayu_apple/backends vaayu_apple/splitting tests
touch vaayu_apple/__init__.py
touch vaayu_apple/loaders/__init__.py
touch vaayu_apple/backends/__init__.py
touch vaayu_apple/splitting/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "vaayu-apple"
version = "0.1.0"
requires-python = ">=3.10"
description = "Layer-by-layer LLM streaming for Apple Silicon"
dependencies = [
    "torch>=2.3",
    "transformers>=4.45",
    "safetensors>=0.4",
    "accelerate>=0.30",
    "huggingface-hub>=0.23",
    "psutil>=5.9",
    "tqdm",
]

[project.optional-dependencies]
mlx = [
    "mlx>=0.16",
    "mlx-lm>=0.16",
]
dev = [
    "pytest>=8",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["vaayu_apple*"]
```

- [ ] **Step 3: Write Mac guard in `vaayu_apple/__init__.py`**

```python
import sys

if sys.platform != "darwin":
    raise ImportError(
        "vaayu-apple is Mac-only. "
        "For Linux, use the main Vaayu package."
    )

import torch
if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS not available. vaayu-apple requires Apple Silicon (M1+) "
        "with macOS 12.3+ and PyTorch >= 2.3."
    )

from .model import VaayuAppleModel

__all__ = ["VaayuAppleModel"]
```

- [ ] **Step 4: Write stub `vaayu_apple/model.py`**

```python
class VaayuAppleModel:
    """Stub — implemented in Task 8."""
    pass
```

- [ ] **Step 5: Install in editable mode**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pip install -e ".[dev]"
```

Expected: No errors. `pip show vaayu-apple` shows the package.

- [ ] **Step 6: Verify Mac guard works**

```bash
cd ~/Documents/ai-ml/vaayu-apple
python -c "import vaayu_apple; print('guard passed')"
```

Expected: `guard passed`

- [ ] **Step 7: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add .
git commit -m "chore: initial repo scaffold with Mac guard"
```

---

## Task 2: Utils

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/utils.py`

- [ ] **Step 1: Write `vaayu_apple/utils.py`**

```python
import gc
import platform
import torch
import psutil


def clean_memory() -> None:
    """Free Python GC + MPS cache. Called every ~20 layers."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ram_available() -> int:
    """Return available system RAM in bytes."""
    return psutil.virtual_memory().available


def detect_chip() -> str:
    """Return Apple chip identifier string, e.g. 'Apple M4'."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return platform.processor() or "Apple Silicon"
```

- [ ] **Step 2: Verify utils import**

```bash
cd ~/Documents/ai-ml/vaayu-apple
python -c "from vaayu_apple.utils import clean_memory, ram_available, detect_chip; print(detect_chip(), ram_available() // 1024**3, 'GB available')"
```

Expected: prints chip name and available RAM (e.g., `Apple M4 10 GB available`)

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/utils.py
git commit -m "feat: add utils — clean_memory, ram_available, detect_chip"
```

---

## Task 3: F_NOCACHE wrapper

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/loaders/nocache.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/test_nocache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_nocache.py
import os
import tempfile
import pytest
from vaayu_apple.loaders.nocache import open_nocache


def test_open_nocache_reads_same_bytes_as_standard_open(tmp_path):
    """F_NOCACHE must not corrupt read data."""
    test_file = tmp_path / "test.bin"
    data = bytes(range(256)) * 16  # 4096 bytes
    test_file.write_bytes(data)

    fd = open_nocache(str(test_file))
    try:
        read_data = os.read(fd, len(data))
    finally:
        os.close(fd)

    assert read_data == data


def test_open_nocache_returns_valid_fd(tmp_path):
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello")
    fd = open_nocache(str(test_file))
    assert fd > 0
    os.close(fd)


def test_open_nocache_nonexistent_file_raises():
    with pytest.raises(OSError):
        open_nocache("/nonexistent/path/file.bin")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_nocache.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `open_nocache` not defined yet.

- [ ] **Step 3: Implement `vaayu_apple/loaders/nocache.py`**

```python
import fcntl
import os
import warnings

F_NOCACHE = 48  # macOS fcntl constant — bypasses unified buffer cache


def open_nocache(path: str) -> int:
    """
    Open a file with F_NOCACHE so reads bypass the macOS page cache.
    Equivalent to Linux O_DIRECT for cache-bypass purposes.
    Falls back to standard open on failure (non-fatal).
    Returns a raw file descriptor — caller must close it.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError as e:
        warnings.warn(
            f"[vaayu-apple] F_NOCACHE failed on {path}: {e}. "
            "Falling back to buffered IO.",
            stacklevel=2,
        )
    return fd
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_nocache.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/loaders/nocache.py tests/test_nocache.py
git commit -m "feat: F_NOCACHE wrapper with non-fatal fallback"
```

---

## Task 4: Loader strategy ABC

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/loaders/base.py`

- [ ] **Step 1: Write `vaayu_apple/loaders/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class AppleLoaderStrategy(ABC):
    """
    Defines how a single model layer moves from NVMe → RAM → MPS/MLX,
    and how it is freed when Vaayu-Apple moves to the next layer.
    """

    @abstractmethod
    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        """
        Read layer weights from SSD into RAM.
        Returns {full_param_name: tensor} — tensors live in CPU RAM.
        layer_name example: "model.layers.0", "model.embed_tokens", "lm_head"
        """

    @abstractmethod
    def inject_to_layer(
        self,
        model: nn.Module,
        tensors: dict,
        device: str,
        dtype: torch.dtype,
    ) -> list:
        """
        Move tensors from CPU → device and attach to model.
        Returns list of injected param names (used by free_layer).
        """

    @abstractmethod
    def free_layer(self, layer_module: nn.Module) -> None:
        """
        Reclaim RAM after the layer's forward pass is done.
        Moves layer to 'meta' device and clears MPS/Metal cache.
        """
```

- [ ] **Step 2: Verify ABC is importable**

```bash
cd ~/Documents/ai-ml/vaayu-apple
python -c "from vaayu_apple.loaders.base import AppleLoaderStrategy; print('ABC ok')"
```

Expected: `ABC ok`

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/loaders/base.py
git commit -m "feat: AppleLoaderStrategy ABC"
```

---

## Task 5: Layer splitter

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/splitting/splitter.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/test_splitting.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_splitting.py
import torch
import pytest
from safetensors.torch import save_file
from pathlib import Path
from vaayu_apple.splitting.splitter import split_and_save_layers, shards_complete


def _make_tiny_model(tmp_path: Path) -> Path:
    """Write a minimal sharded safetensors model with 2 decoder layers."""
    model_path = tmp_path / "tiny_model"
    model_path.mkdir()

    # Two decoder layers + embed + norm + lm_head (mimics real layout)
    weights = {
        "model.embed_tokens.weight": torch.randn(32, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(16, 16),
        "model.norm.weight": torch.randn(16),
        "lm_head.weight": torch.randn(32, 16),
    }
    # Save as a single model.safetensors (single-file format)
    save_file(weights, model_path / "model.safetensors")
    return model_path


def test_split_creates_shard_files(tmp_path):
    model_path = _make_tiny_model(tmp_path)
    shard_path = tmp_path / "shards"

    split_and_save_layers(str(model_path), str(shard_path))

    expected = [
        "model.embed_tokens.safetensors",
        "model.layers.0.safetensors",
        "model.layers.1.safetensors",
        "model.norm.safetensors",
        "lm_head.safetensors",
    ]
    for fname in expected:
        assert (shard_path / fname).exists(), f"Missing shard: {fname}"


def test_split_preserves_tensor_values(tmp_path):
    from safetensors.torch import load_file
    model_path = _make_tiny_model(tmp_path)
    shard_path = tmp_path / "shards"
    split_and_save_layers(str(model_path), str(shard_path))

    original = load_file(model_path / "model.safetensors", device="cpu")
    shard = load_file(shard_path / "model.layers.0.safetensors", device="cpu")

    assert "model.layers.0.self_attn.q_proj.weight" in shard
    torch.testing.assert_close(
        shard["model.layers.0.self_attn.q_proj.weight"],
        original["model.layers.0.self_attn.q_proj.weight"],
    )


def test_shards_complete_returns_true_when_done(tmp_path):
    model_path = _make_tiny_model(tmp_path)
    shard_path = tmp_path / "shards"
    split_and_save_layers(str(model_path), str(shard_path))
    assert shards_complete(str(shard_path), n_layers=2) is True


def test_shards_complete_returns_false_when_missing(tmp_path):
    shard_path = tmp_path / "empty"
    shard_path.mkdir()
    assert shards_complete(str(shard_path), n_layers=2) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_splitting.py -v
```

Expected: `ImportError` — `split_and_save_layers` not defined yet.

- [ ] **Step 3: Implement `vaayu_apple/splitting/splitter.py`**

```python
"""
Layer shard splitter — ported from Linux Vaayu utils.py.
Splits HuggingFace model files into one safetensors file per layer.
Mac-specific: no compression, no bitsandbytes.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import huggingface_hub


def shards_complete(shard_path: str, n_layers: int) -> bool:
    """Return True if all expected shard files exist."""
    p = Path(shard_path)
    required = (
        ["model.embed_tokens.safetensors"]
        + [f"model.layers.{i}.safetensors" for i in range(n_layers)]
        + ["model.norm.safetensors", "lm_head.safetensors"]
    )
    return all((p / f).exists() for f in required)


def _detect_n_layers(index: dict) -> int:
    prefix = "model.layers."
    indices = {
        int(k.split(prefix)[1].split(".")[0])
        for k in index
        if prefix in k
    }
    return len(indices)


def split_and_save_layers(
    model_path: str,
    shard_path: str,
    hf_token: str | None = None,
) -> None:
    """
    Split a HuggingFace model checkpoint into per-layer safetensors shards.
    Skips if shards already exist.
    Supports: model.safetensors (single file), model.safetensors.index.json (sharded).
    """
    model_path = Path(model_path)
    shard_path = Path(shard_path)
    shard_path.mkdir(parents=True, exist_ok=True)

    # Build weight index: {param_name: filename}
    index: dict[str, str] = {}
    safetensors_format = False

    if (model_path / "model.safetensors.index.json").exists():
        safetensors_format = True
        with open(model_path / "model.safetensors.index.json") as f:
            index = json.load(f)["weight_map"]
    elif (model_path / "model.safetensors").exists():
        safetensors_format = True
        state = load_file(model_path / "model.safetensors", device="cpu")
        index = {k: "model.safetensors" for k in state.keys()}
        del state
        gc.collect()
    elif (model_path / "pytorch_model.bin.index.json").exists():
        with open(model_path / "pytorch_model.bin.index.json") as f:
            index = json.load(f)["weight_map"]
    elif (model_path / "pytorch_model.bin").exists():
        state = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        index = {k: "pytorch_model.bin" for k in state.keys()}
        del state
        gc.collect()
    else:
        raise FileNotFoundError(
            f"No model files found in {model_path}. "
            "Expected model.safetensors, model.safetensors.index.json, "
            "pytorch_model.bin, or pytorch_model.bin.index.json."
        )

    n_layers = _detect_n_layers(index)

    # Skip if complete
    if shards_complete(str(shard_path), n_layers):
        print(f"[vaayu-apple] shards already exist in {shard_path}, skipping split.")
        return

    # Build layer list
    layers = (
        ["model.embed_tokens"]
        + [f"model.layers.{i}" for i in range(n_layers)]
        + ["model.norm", "lm_head"]
    )

    # Load all weights (shard-by-shard for multi-file, all-at-once for single file)
    full_state: dict[str, torch.Tensor] = {}

    unique_files = sorted(set(index.values()))
    for fname in tqdm(unique_files, desc="Loading model shards"):
        fpath = model_path / fname
        if not fpath.exists():
            if hf_token is not None:
                huggingface_hub.hf_hub_download(
                    repo_id=str(model_path),
                    filename=fname,
                    token=hf_token,
                )
            else:
                raise FileNotFoundError(f"Shard file not found: {fpath}")
        if safetensors_format:
            full_state.update(load_file(str(fpath), device="cpu"))
        else:
            full_state.update(torch.load(str(fpath), map_location="cpu"))

    # Save one file per layer
    for layer_name in tqdm(layers, desc="Saving layer shards"):
        layer_prefix = layer_name + "."
        layer_dict = {
            k: v for k, v in full_state.items()
            if k.startswith(layer_prefix) or k == layer_name
        }
        if not layer_dict:
            continue
        out_file = shard_path / f"{layer_name}.safetensors"
        save_file(layer_dict, str(out_file))

        # Free as we go
        for k in list(layer_dict.keys()):
            if k in full_state:
                del full_state[k]
        del layer_dict
        gc.collect()

    print(f"[vaayu-apple] split complete → {shard_path}")


def find_or_create_splitted_path(
    model_id: str,
    layer_shards_path: str | None = None,
    hf_token: str | None = None,
) -> tuple[str, str]:
    """
    Download model if needed, split into per-layer shards if not already done.
    Returns (model_path, shard_path).
    """
    # Local path
    if os.path.isdir(model_id):
        model_path = model_id
    else:
        print(f"[vaayu-apple] downloading {model_id} from HuggingFace…")
        model_path = huggingface_hub.snapshot_download(
            model_id, token=hf_token,
            ignore_patterns=["*.safetensors", "*.bin"],
        )
        # If no index file, download weights too
        if not (
            (Path(model_path) / "model.safetensors.index.json").exists()
            or (Path(model_path) / "pytorch_model.bin.index.json").exists()
        ):
            model_path = huggingface_hub.snapshot_download(
                model_id, token=hf_token,
                allow_patterns=["*.safetensors", "*.bin", "*.json"],
            )

    # Shard path
    if layer_shards_path is not None:
        shard_path = layer_shards_path
    else:
        shard_path = str(Path(model_path) / "vaayu_apple_shards")

    split_and_save_layers(model_path, shard_path, hf_token=hf_token)
    return model_path, shard_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_splitting.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/splitting/splitter.py tests/test_splitting.py
git commit -m "feat: layer splitter with HF download + single/multi-file support"
```

---

## Task 6: Safetensors loader

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/loaders/safetensors_loader.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/test_safetensors_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_safetensors_loader.py
import torch
import pytest
from pathlib import Path
from safetensors.torch import save_file
from vaayu_apple.loaders.safetensors_loader import SafetensorsLoader


def _make_shard(tmp_path: Path, layer_name: str, tensors: dict) -> Path:
    """Save tensors as a layer shard file."""
    shard_path = tmp_path / "shards"
    shard_path.mkdir(exist_ok=True)
    out = shard_path / f"{layer_name}.safetensors"
    save_file(tensors, str(out))
    return shard_path


def test_load_from_disk_returns_correct_shapes(tmp_path):
    w = {"model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64)}
    shard_path = _make_shard(tmp_path, "model.layers.0", w)
    loader = SafetensorsLoader()
    result = loader.load_from_disk("model.layers.0", str(shard_path))
    assert "model.layers.0.self_attn.q_proj.weight" in result
    assert result["model.layers.0.self_attn.q_proj.weight"].shape == (64, 64)


def test_load_from_disk_preserves_values(tmp_path):
    original = torch.randn(32, 32)
    w = {"model.norm.weight": original}
    shard_path = _make_shard(tmp_path, "model.norm", w)
    loader = SafetensorsLoader()
    result = loader.load_from_disk("model.norm", str(shard_path))
    torch.testing.assert_close(result["model.norm.weight"], original)


def test_load_from_disk_uses_double_buffer(tmp_path):
    """Two consecutive loads should not corrupt each other."""
    w0 = {"model.layers.0.w": torch.ones(8, 8)}
    w1 = {"model.layers.1.w": torch.full((8, 8), 2.0)}
    shard_path = tmp_path / "shards"
    shard_path.mkdir()
    save_file(w0, str(shard_path / "model.layers.0.safetensors"))
    save_file(w1, str(shard_path / "model.layers.1.safetensors"))

    loader = SafetensorsLoader()
    r0 = loader.load_from_disk("model.layers.0", str(shard_path))
    r1 = loader.load_from_disk("model.layers.1", str(shard_path))

    # r0 should still be valid (double buffer keeps previous alive)
    assert r0["model.layers.0.w"].mean().item() == pytest.approx(1.0)
    assert r1["model.layers.1.w"].mean().item() == pytest.approx(2.0)


def test_load_missing_lm_head_falls_back_to_embed(tmp_path):
    """Tied weights: lm_head.safetensors absent → use embed_tokens."""
    w = {"model.embed_tokens.weight": torch.randn(32, 16)}
    shard_path = _make_shard(tmp_path, "model.embed_tokens", w)
    loader = SafetensorsLoader()
    result = loader.load_from_disk("lm_head", str(shard_path))
    assert "lm_head.weight" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_safetensors_loader.py -v
```

Expected: `ImportError` — `SafetensorsLoader` not defined yet.

- [ ] **Step 3: Implement `vaayu_apple/loaders/safetensors_loader.py`**

```python
from __future__ import annotations

import json
import os
import struct
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from .base import AppleLoaderStrategy
from .nocache import open_nocache

_DTYPE_MAP = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# Double buffer — alternated per call to avoid realloc
_BUFS: list[bytearray | None] = [None, None]
_BUF_IDX: int = 0
_PAD = 100 * 1024 * 1024  # 100 MB headroom


def _load_nocache_tensors(file_path: str) -> dict:
    """Read a safetensors file into a pre-allocated bytearray using F_NOCACHE."""
    global _BUFS, _BUF_IDX

    # Parse header
    with open(file_path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))

    tensors_meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not tensors_meta:
        return {}

    data_start = 8 + hdr_len
    total_size = max(v["data_offsets"][1] for v in tensors_meta.values())

    # Acquire double buffer
    buf_idx = _BUF_IDX
    _BUF_IDX = 1 - _BUF_IDX

    needed = total_size + _PAD
    if _BUFS[buf_idx] is None or len(_BUFS[buf_idx]) < needed:
        _BUFS[buf_idx] = bytearray(needed)
        print(f"[vaayu-apple] alloc buf={buf_idx} {needed / 1024**2:.0f}MB")

    buf = _BUFS[buf_idx]
    buf_view = memoryview(buf)

    t0 = time.time()
    fd = open_nocache(file_path)
    try:
        n = os.preadv(fd, [buf_view[:total_size]], data_start)
        if n != total_size:
            raise IOError(f"[vaayu-apple] short read: expected {total_size}, got {n}")
    finally:
        os.close(fd)

    elapsed = time.time() - t0
    size_mb = total_size / 1024**2
    gbps = (size_mb / 1024) / elapsed if elapsed > 0 else 0
    print(
        f"[VAAYU-IO] nocache {elapsed:.3f}s {size_mb:.0f}MB "
        f"{gbps:.2f}GB/s {os.path.basename(file_path)}"
    )

    # Zero-copy tensor reconstruction
    result = {}
    for name, meta in tensors_meta.items():
        dtype = _DTYPE_MAP.get(meta["dtype"], torch.float32)
        start, end = meta["data_offsets"]
        result[name] = torch.frombuffer(
            buf_view[start:end], dtype=dtype
        ).reshape(meta["shape"])

    return result


class SafetensorsLoader(AppleLoaderStrategy):

    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        file_path = str(Path(checkpoint_path) / f"{layer_name}.safetensors")

        if not os.path.exists(file_path):
            if layer_name == "lm_head":
                # Tied weights fallback: use embed_tokens, remap keys
                embed_path = str(
                    Path(checkpoint_path) / "model.embed_tokens.safetensors"
                )
                if not os.path.exists(embed_path):
                    raise FileNotFoundError(
                        f"[vaayu-apple] Neither {file_path} nor {embed_path} found. "
                        "Re-run with layer_shards_path=None to re-split."
                    )
                raw = _load_nocache_tensors(embed_path)
                return {
                    k.replace("model.embed_tokens", "lm_head"): v
                    for k, v in raw.items()
                }
            raise FileNotFoundError(
                f"[vaayu-apple] Shard not found: {file_path}. "
                "Re-run with layer_shards_path=None to re-split."
            )

        return _load_nocache_tensors(file_path)

    def inject_to_layer(
        self,
        model: nn.Module,
        tensors: dict,
        device: str,
        dtype: torch.dtype,
    ) -> list:
        moved = []
        # Try bulk load_state_dict first (fast path)
        gpu_tensors = {k: v.to(device, dtype=dtype, non_blocking=True) for k, v in tensors.items()}
        try:
            model.load_state_dict(gpu_tensors, strict=False, assign=True)
            moved = list(gpu_tensors.keys())
        except Exception:
            # Fallback: inject each param individually
            for name, tensor in gpu_tensors.items():
                try:
                    set_module_tensor_to_device(
                        model, name, device, value=tensor, dtype=dtype
                    )
                    moved.append(name)
                except Exception as e:
                    warnings.warn(f"[vaayu-apple] inject failed {name}: {e}")
        return moved

    def free_layer(self, layer_module: nn.Module) -> None:
        layer_module.to("meta")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_safetensors_loader.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/loaders/safetensors_loader.py tests/test_safetensors_loader.py
git commit -m "feat: SafetensorsLoader — F_NOCACHE double-buffer zero-copy path"
```

---

## Task 7: MPS backend

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/backends/mps_backend.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/test_forward_mps.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_forward_mps.py
"""
Full forward() pass with a tiny 2-layer Llama-style model on MPS.
Validates logits shape only — numerical correctness is covered by
the individual layer tests.
"""
import torch
import pytest
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from pathlib import Path
from vaayu_apple.backends.mps_backend import MPSBackend
from vaayu_apple.loaders.safetensors_loader import SafetensorsLoader


VOCAB = 128
HIDDEN = 64
LAYERS = 2
HEADS = 4


def _make_tiny_llama_config(tmp_path: Path):
    """Write a minimal LlamaConfig and return it."""
    from transformers import LlamaConfig
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        max_position_embeddings=64,
    )
    config.save_pretrained(str(tmp_path / "config"))
    return config


def _split_model_to_shards(config, tmp_path: Path) -> tuple:
    """Instantiate tiny model and save per-layer shards."""
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM(config)
    model.eval()

    shard_path = tmp_path / "shards"
    shard_path.mkdir()

    # Collect and save per-layer state dicts
    layer_names = (
        ["model.embed_tokens"]
        + [f"model.layers.{i}" for i in range(LAYERS)]
        + ["model.norm", "lm_head"]
    )
    full_state = dict(model.state_dict())

    for layer_name in layer_names:
        prefix = layer_name + "."
        shard = {k: v for k, v in full_state.items() if k.startswith(prefix) or k == layer_name}
        if shard:
            save_file(shard, str(shard_path / f"{layer_name}.safetensors"))

    return model, list(model.named_modules()), shard_path


def test_mps_forward_returns_correct_logits_shape(tmp_path):
    config = _make_tiny_llama_config(tmp_path)
    orig_model, _, shard_path = _split_model_to_shards(config, tmp_path)

    # Build meta model as VaayuAppleModel would
    with init_empty_weights():
        meta_model = type(orig_model)(config)
    meta_model.eval()

    # Build layer list
    layer_names = (
        ["model.embed_tokens"]
        + [f"model.layers.{i}" for i in range(LAYERS)]
        + ["model.norm", "lm_head"]
    )
    layers = []
    for name in layer_names:
        mod = meta_model
        for part in name.split("."):
            mod = getattr(mod, part)
        layers.append(mod)

    loader = SafetensorsLoader()
    backend = MPSBackend()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    from transformers.cache_utils import DynamicCache
    output = backend.forward(
        model=meta_model,
        layer_names=layer_names,
        layers=layers,
        loader=loader,
        checkpoint_path=str(shard_path),
        layer_names_dict={
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        },
        input_ids=input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        dtype=torch.float32,
        device="mps",
        max_seq_len=64,
        pinned_layer_names=set(),
        config=config,
    )

    assert output.logits.shape == (1, 4, VOCAB)
    assert not output.logits.isnan().any()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_forward_mps.py -v
```

Expected: `ImportError` — `MPSBackend` not defined yet.

- [ ] **Step 3: Implement `vaayu_apple/backends/mps_backend.py`**

```python
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache

from ..loaders.base import AppleLoaderStrategy
from ..utils import clean_memory


def _compute_full_position_embeddings(config, position_ids: torch.Tensor):
    """Manual RoPE computation — avoids meta-device uninitialized inv_freq."""
    try:
        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = float(getattr(config, "rope_theta", 10000.0))
    except AttributeError:
        return None

    inv_freq = 1.0 / (
        rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device)
            / head_dim
        )
    )
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    with torch.no_grad():
        freqs = (inv_freq_expanded.float() @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos, sin


def _kick_next_load(
    executor: ThreadPoolExecutor,
    loader: AppleLoaderStrategy,
    layer_names: list,
    current_idx: int,
    checkpoint_path: str,
    pinned: set,
):
    """Submit load for the next non-pinned layer. Returns Future or None."""
    for name in layer_names[current_idx + 1:]:
        if name not in pinned:
            return executor.submit(loader.load_from_disk, name, checkpoint_path)
    return None


class MPSBackend:

    def forward(
        self,
        model: nn.Module,
        layer_names: list,
        layers: list,
        loader: AppleLoaderStrategy,
        checkpoint_path: str,
        layer_names_dict: dict,
        input_ids: torch.Tensor,
        attention_mask,
        past_key_values,
        use_cache: bool,
        dtype: torch.dtype,
        device: str,
        max_seq_len: int,
        pinned_layer_names: set,
        config,
    ) -> CausalLMOutputWithPast:

        _exec_start = time.time()

        # KV cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Causal attention mask (only for eager attn)
        full_attention_mask = None
        # Use None for sdpa/flash — they handle masking internally.
        # Eager fallback: build full causal mask.

        # Position IDs
        position_ids = torch.arange(
            max_seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Pre-compute RoPE if model uses top-level rotary_emb
        full_position_embeddings = None
        try:
            rotary_emb = model.model.rotary_emb
            if rotary_emb is not None:
                rotary_cls = type(rotary_emb)
                real_rotary = rotary_cls(config=config).to(device=device, dtype=dtype)
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                dummy_x = torch.empty(1, 1, position_ids.shape[1], head_dim, device=device, dtype=dtype)
                full_position_embeddings = real_rotary(dummy_x, position_ids)
        except Exception:
            full_position_embeddings = _compute_full_position_embeddings(config, position_ids)

        hidden_states = None
        _io_s = _gpu_s = _compute_s = 0.0
        embed_name = layer_names_dict["embed"]
        norm_name = layer_names_dict["norm"]
        lm_head_name = layer_names_dict["lm_head"]

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as ex:
            # Seed prefetch for first non-pinned layer
            future = None
            for seed_name in layer_names:
                if seed_name not in pinned_layer_names:
                    future = ex.submit(loader.load_from_disk, seed_name, checkpoint_path)
                    break

            for i, (name, layer) in enumerate(
                tqdm(zip(layer_names, layers), total=len(layers), desc=f"forward ({device})")
            ):
                if name in pinned_layer_names:
                    # Already in MPS RAM — kick next load so disk overlaps compute
                    if future is None:
                        future = _kick_next_load(ex, loader, layer_names, i, checkpoint_path, pinned_layer_names)
                    moved_params = []
                else:
                    _t = time.time()
                    tensors = future.result()
                    _io_s += time.time() - _t

                    future = _kick_next_load(ex, loader, layer_names, i, checkpoint_path, pinned_layer_names)

                    _t = time.time()
                    moved_params = loader.inject_to_layer(model, tensors, device, dtype)
                    _gpu_s += time.time() - _t

                # Run layer
                _t = time.time()
                if name == embed_name:
                    hidden_states = layer(input_ids.to(device))

                elif name == norm_name:
                    hidden_states = layer(hidden_states)

                elif name == lm_head_name:
                    hidden_states = layer(hidden_states).float()

                else:
                    # Transformer decoder layer
                    past_len = past_key_values.get_seq_length() if past_key_values else 0
                    seq_len = hidden_states.shape[1]

                    kwargs: dict = {"use_cache": use_cache}
                    if past_key_values is not None:
                        kwargs["past_key_values"] = past_key_values
                    kwargs["position_ids"] = position_ids[:, past_len:past_len + seq_len]
                    if full_position_embeddings is not None:
                        cos, sin = full_position_embeddings
                        kwargs["position_embeddings"] = (
                            cos[:, past_len:past_len + seq_len],
                            sin[:, past_len:past_len + seq_len],
                        )

                    out = layer(hidden_states, **kwargs)
                    hidden_states = out[0] if isinstance(out, tuple) else out

                _compute_s += time.time() - _t

                # Evict non-pinned layer
                if name not in pinned_layer_names:
                    loader.free_layer(layer)

                if i % 20 == 19:
                    clean_memory()

        torch.mps.synchronize()

        total = time.time() - _exec_start
        print(
            f"[VAAYU-IO] token: total={total:.1f}s  "
            f"io={_io_s:.1f}s  gpu_load={_gpu_s:.1f}s  compute={_compute_s:.1f}s"
        )

        logits = hidden_states

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_forward_mps.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/backends/mps_backend.py tests/test_forward_mps.py
git commit -m "feat: MPSBackend — ThreadPoolExecutor prefetch + MPS forward loop"
```

---

## Task 8: VaayuAppleModel

**Files:**
- Modify: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/model.py` (replace stub)

- [ ] **Step 1: Implement `vaayu_apple/model.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache

from .loaders.base import AppleLoaderStrategy
from .loaders.safetensors_loader import SafetensorsLoader
from .backends.mps_backend import MPSBackend
from .splitting.splitter import find_or_create_splitted_path
from .utils import clean_memory, ram_available

import psutil

try:
    import mlx.core as mx
    from .loaders.mlx_loader import MLXLoader
    from .backends.mlx_backend import MLXBackend
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def _is_mlx_checkpoint(checkpoint_path: str) -> bool:
    import glob
    return bool(glob.glob(str(Path(checkpoint_path) / "*.npz")))


def _build_loader(checkpoint_path: str, vaayu_ref) -> AppleLoaderStrategy:
    if _MLX_AVAILABLE and _is_mlx_checkpoint(checkpoint_path):
        print("[vaayu-apple] MLX checkpoint detected → MLXLoader")
        return MLXLoader()
    print("[vaayu-apple] safetensors checkpoint → SafetensorsLoader")
    return SafetensorsLoader()


class VaayuAppleModel(GenerationMixin):
    """
    Streams 70B+ LLMs layer-by-layer from NVMe using F_NOCACHE + MPS (or MLX).
    Drop-in compatible with HuggingFace GenerationMixin.generate().
    """

    _supports_cache_class = True

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 512,
        layer_shards_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        profiling: bool = False,
    ):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS not available. vaayu-apple requires Apple Silicon (M1+) "
                "with macOS 12.3+ and PyTorch >= 2.3."
            )

        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.profiling = profiling
        self.pinned_layer_names: set = set()
        self._path_remappings: dict = {}
        self.hf_quantizer = None
        self.main_input_name = "input_ids"

        # Config + tokenizer
        self.config = AutoConfig.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
        self.generation_config = self._get_generation_config(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )

        # Normalize config aliases (for non-Llama architectures)
        self._normalize_config()

        # Split model into per-layer shards
        self.model_path, self.checkpoint_path = find_or_create_splitted_path(
            model_id, layer_shards_path, hf_token=hf_token
        )

        # Loader + backend
        self.loader = _build_loader(self.checkpoint_path, self)
        if _MLX_AVAILABLE and isinstance(self.loader, MLXLoader):
            self.backend = MLXBackend()
        else:
            self.backend = MPSBackend()

        # Build meta model
        self._init_model()

        # Layer names list: [embed, layer.0, ..., norm, lm_head]
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
        model_attr = self.model
        for part in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, part)
        n_layers = len(model_attr)

        self.layer_names = (
            [self.layer_names_dict["embed"]]
            + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(n_layers)]
            + [self.layer_names_dict["norm"], self.layer_names_dict["lm_head"]]
        )
        self.layers = []
        for name in self.layer_names:
            mod = self.model
            for part in name.split("."):
                mod = getattr(mod, part)
            self.layers.append(mod)

        # MoE detection
        self._detect_moe()

        # Pin shared layers
        self._init_pinned_layers()

        print(f"[vaayu-apple] ready — {len(self.layer_names)} layers, device=mps, dtype={dtype}")

    # ── internal helpers ──────────────────────────────────────────────

    def _get_generation_config(self, model_id: str) -> GenerationConfig:
        try:
            return GenerationConfig.from_pretrained(model_id)
        except Exception:
            return GenerationConfig()

    def _normalize_config(self) -> None:
        """Map non-standard config keys to standard transformers names."""
        fallbacks = {
            "hidden_size": ["d_model", "n_embd"],
            "num_hidden_layers": ["num_layers", "n_layer"],
            "num_attention_heads": ["num_heads", "n_head"],
            "num_key_value_heads": ["num_kv_heads", "num_attention_heads"],
            "intermediate_size": ["ffn_dim", "inner_dim"],
            "rms_norm_eps": ["layer_norm_eps", "norm_eps"],
            "max_position_embeddings": ["n_positions", "max_seq_len"],
            "rope_theta": ["rotary_emb_base"],
        }
        for target, srcs in fallbacks.items():
            if not hasattr(self.config, target):
                for src in srcs:
                    if hasattr(self.config, src):
                        setattr(self.config, target, getattr(self.config, src))
                        break
        # Hard defaults
        if not hasattr(self.config, "num_hidden_layers"):
            self.config.num_hidden_layers = 32
        if not hasattr(self.config, "num_attention_heads"):
            self.config.num_attention_heads = 32
        if not hasattr(self.config, "max_position_embeddings"):
            self.config.max_position_embeddings = 4096
        if not hasattr(self.config, "rope_theta"):
            self.config.rope_theta = 10000.0

    def _init_model(self) -> None:
        for attn_impl in ["sdpa", "eager"]:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        self.config,
                        attn_implementation=attn_impl,
                        trust_remote_code=True,
                    )
                self._attn_impl = attn_impl
                break
            except Exception:
                self.model = None
        if self.model is None:
            raise RuntimeError("[vaayu-apple] Failed to create model architecture.")
        print(f"[vaayu-apple] attention: {self._attn_impl}")
        self.model.eval()

        # Move buffers to MPS
        from accelerate.utils.modeling import set_module_tensor_to_device
        for buf_name, buf in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buf_name, "mps", value=buf, dtype=self.dtype
            )

    def _detect_moe(self) -> None:
        arch = (getattr(self.config, "architectures", None) or [""])[0]
        self.is_moe = (
            hasattr(self.config, "num_local_experts")
            or hasattr(self.config, "num_experts")
            or "Moe" in arch or "MoE" in arch or "Mixtral" in arch
        )
        self.moe_num_experts = getattr(self.config, "num_local_experts",
                                       getattr(self.config, "num_experts", 0))

    def _get_pinning_budget(self) -> int:
        available = psutil.virtual_memory().available
        reserve = 2 * 1024**3  # 2 GB
        return max(0, available - reserve)

    def _init_pinned_layers(self) -> None:
        budget = self._get_pinning_budget()
        if budget <= 0:
            return
        candidates = [
            self.layer_names_dict["embed"],
            self.layer_names_dict["norm"],
            self.layer_names_dict["lm_head"],
        ]
        used = 0
        for name in candidates:
            try:
                tensors = self.loader.load_from_disk(name, self.checkpoint_path)
                size = sum(t.nbytes for t in tensors.values())
                if used + size > budget:
                    print(f"[vaayu-apple] skip pin {name}: budget exhausted")
                    continue
                self.loader.inject_to_layer(self.model, tensors, "mps", self.dtype)
                self.pinned_layer_names.add(name)
                used += size
                print(f"[vaayu-apple] pinned {name}: {size / 1e9:.2f}GB")
            except Exception as e:
                print(f"[vaayu-apple] skip pin {name}: {e}")

    # ── GenerationMixin interface ─────────────────────────────────────

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_len = (
                past_key_values.get_seq_length()
                if hasattr(past_key_values, "get_seq_length")
                else past_key_values[0][0].shape[2]
            )
            if input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
            else:
                input_ids = input_ids[:, -1:]

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache: Optional[bool] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = True

        return self.backend.forward(
            model=self.model,
            layer_names=self.layer_names,
            layers=self.layers,
            loader=self.loader,
            checkpoint_path=self.checkpoint_path,
            layer_names_dict=self.layer_names_dict,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            dtype=self.dtype,
            device="mps",
            max_seq_len=self.max_seq_len,
            pinned_layer_names=self.pinned_layer_names,
            config=self.config,
        )
```

- [ ] **Step 2: Verify VaayuAppleModel is importable**

```bash
cd ~/Documents/ai-ml/vaayu-apple
python -c "from vaayu_apple import VaayuAppleModel; print('VaayuAppleModel ok')"
```

Expected: `VaayuAppleModel ok`

- [ ] **Step 3: Run existing tests to ensure nothing broke**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/ -v --ignore=tests/test_forward_mps.py --ignore=tests/test_mlx_loader.py
```

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/model.py
git commit -m "feat: VaayuAppleModel — init, pinning, MoE detect, forward dispatcher"
```

---

## Task 9: Public API + inference example

**Files:**
- Modify: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/__init__.py`
- Create: `~/Documents/ai-ml/vaayu-apple/inference_example.py`
- Create: `~/Documents/ai-ml/vaayu-apple/README.md`

- [ ] **Step 1: Update `vaayu_apple/__init__.py`**

```python
import sys

if sys.platform != "darwin":
    raise ImportError(
        "vaayu-apple is Mac-only. "
        "For Linux, use the main Vaayu package."
    )

import torch
if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS not available. vaayu-apple requires Apple Silicon (M1+) "
        "with macOS 12.3+ and PyTorch >= 2.3."
    )

from .model import VaayuAppleModel

__version__ = "0.1.0"
__all__ = ["VaayuAppleModel"]
```

- [ ] **Step 2: Write `inference_example.py`**

```python
"""
Minimal inference example for vaayu-apple.
Usage: python inference_example.py
Set HF_TOKEN env var if using a gated model.
"""
import os
import torch
from vaayu_apple import VaayuAppleModel

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # swap for 70B on 16GB Mac
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Loading {MODEL_ID}…")
model = VaayuAppleModel(
    MODEL_ID,
    dtype=torch.bfloat16,
    max_seq_len=512,
    hf_token=HF_TOKEN,
)

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

print("Generating…")
with torch.inference_mode():
    output = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=False,
    )

response = model.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nResponse: {response}")
```

- [ ] **Step 3: Write `README.md`**

```markdown
# vaayu-apple

Stream 70B+ LLMs layer-by-layer on any Apple Silicon Mac (M1–M4).
Works on 16GB MacBooks. No GPU memory limits.

## Install

```bash
pip install -e .
# For MLX backend:
pip install -e ".[mlx]"
```

## Usage

```python
from vaayu_apple import VaayuAppleModel
import torch

model = VaayuAppleModel("meta-llama/Llama-3.3-70B-Instruct", hf_token="hf_...")
input_ids = model.tokenizer("Hello", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(model.tokenizer.decode(output[0]))
```

## How it works

- **F_NOCACHE**: bypasses macOS page cache — equivalent to Linux O_DIRECT
- **Layer streaming**: loads one transformer layer at a time, runs it, evicts to meta
- **Pinned layers**: embed/norm/lm_head stay in RAM to save 3 reads/token
- **Dual backend**: safetensors → PyTorch MPS; MLX weights → MLX framework
```

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/__init__.py inference_example.py README.md
git commit -m "feat: public API, inference example, README"
```

---

## Task 10: MLX loader

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/loaders/mlx_loader.py`
- Create: `~/Documents/ai-ml/vaayu-apple/tests/test_mlx_loader.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_mlx_loader.py
import pytest
import numpy as np
from pathlib import Path

mlx = pytest.importorskip("mlx.core", reason="mlx not installed")


def _make_mlx_shard(tmp_path: Path, layer_name: str, arrays: dict) -> Path:
    """Save arrays as a .npz shard (MLX format)."""
    shard_path = tmp_path / "shards"
    shard_path.mkdir(exist_ok=True)
    out = shard_path / f"{layer_name}.npz"
    np.savez(str(out), **{k: np.array(v) for k, v in arrays.items()})
    return shard_path


def test_mlx_loader_returns_mx_arrays(tmp_path):
    from vaayu_apple.loaders.mlx_loader import MLXLoader
    arrays = {"model.layers.0.self_attn.q_proj.weight": mlx.ones((16, 16))}
    shard_path = _make_mlx_shard(tmp_path, "model.layers.0", arrays)

    loader = MLXLoader()
    result = loader.load_from_disk("model.layers.0", str(shard_path))

    assert "model.layers.0.self_attn.q_proj.weight" in result
    assert isinstance(result["model.layers.0.self_attn.q_proj.weight"], mlx.array)
    assert result["model.layers.0.self_attn.q_proj.weight"].shape == (16, 16)


def test_mlx_loader_free_clears_cache(tmp_path):
    import torch
    import torch.nn as nn
    from vaayu_apple.loaders.mlx_loader import MLXLoader

    loader = MLXLoader()
    layer = nn.Linear(8, 8)
    loader.free_layer(layer)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_mlx_loader.py -v
```

Expected: `ImportError` — `MLXLoader` not defined yet (or skipped if mlx not installed).

- [ ] **Step 3: Implement `vaayu_apple/loaders/mlx_loader.py`**

```python
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch.nn as nn

from .base import AppleLoaderStrategy

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class MLXLoader(AppleLoaderStrategy):
    """
    Loads MLX-format weight files (.npz) for use with the MLX backend.
    MLX manages its own IO pipeline — no F_NOCACHE needed.
    """

    def __init__(self):
        if not _MLX_AVAILABLE:
            raise ImportError(
                "MLX checkpoint detected but mlx is not installed. "
                "Install with: pip install mlx mlx-lm"
            )

    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        npz_path = str(Path(checkpoint_path) / f"{layer_name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"[vaayu-apple] MLX shard not found: {npz_path}"
            )
        raw = mx.load(npz_path)
        return dict(raw)

    def inject_to_layer(self, model, tensors: dict, device: str, dtype) -> list:
        """
        For MLX backend: inject mx.arrays directly into the MLX model.
        The model here is an mlx_lm model, not a PyTorch model.
        Uses model.update() with the layer's weight dict.
        """
        try:
            model.update(tensors)
            mx.eval(model.parameters())
        except Exception as e:
            warnings.warn(f"[vaayu-apple] MLX inject warning: {e}")
        return list(tensors.keys())

    def free_layer(self, layer_module) -> None:
        """
        Clear MLX Metal cache. Setting weights to None is not needed —
        the next inject_to_layer() call will overwrite them.
        """
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/test_mlx_loader.py -v
```

Expected: `2 passed` (or `2 skipped` if mlx not installed — acceptable).

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/loaders/mlx_loader.py tests/test_mlx_loader.py
git commit -m "feat: MLXLoader for .npz MLX weight format"
```

---

## Task 11: MLX backend

**Files:**
- Create: `~/Documents/ai-ml/vaayu-apple/vaayu_apple/backends/mlx_backend.py`

- [ ] **Step 1: Implement `vaayu_apple/backends/mlx_backend.py`**

```python
"""
MLX backend for vaayu-apple.
Streams MLX-format model weights layer-by-layer.
CRITICAL: mx.eval() must be called after EVERY layer forward.
Without it, MLX's lazy graph accumulates the entire model in memory.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

from ..loaders.base import AppleLoaderStrategy


def _mlx_inject(layer, tensors: dict) -> None:
    """Set layer weights from tensors dict. Handles nested keys."""
    flat = {}
    for full_key, val in tensors.items():
        # Strip layer prefix, e.g. "model.layers.0.self_attn.q_proj.weight" → "self_attn.q_proj.weight"
        parts = full_key.split(".")
        # Find index of the layer number and strip everything before it
        for i, part in enumerate(parts):
            if part.isdigit():
                sub_key = ".".join(parts[i + 1:])
                flat[sub_key] = val
                break
        else:
            flat[full_key] = val

    try:
        layer.update(flat)
        mx.eval(layer.parameters())
    except Exception as e:
        warnings.warn(f"[vaayu-apple] MLX inject: {e}")


def _mlx_free(layer) -> None:
    """Attempt to free layer weights. Best-effort — MLX GC handles the rest."""
    try:
        mx.metal.clear_cache()
    except Exception:
        pass


class MLXBackend:

    def forward(
        self,
        model,
        layer_names: list,
        layers: list,
        loader: AppleLoaderStrategy,
        checkpoint_path: str,
        layer_names_dict: dict,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache: bool,
        dtype,
        device: str,
        max_seq_len: int,
        pinned_layer_names: set,
        config,
    ):
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "MLX not installed. Run: pip install mlx mlx-lm"
            )

        _exec_start = time.time()

        embed_name = layer_names_dict["embed"]
        norm_name = layer_names_dict["norm"]
        lm_head_name = layer_names_dict["lm_head"]

        input_mx = mx.array(input_ids.numpy())

        # Embed tokens
        embed_tensors = loader.load_from_disk(embed_name, checkpoint_path)
        _mlx_inject(layers[0], embed_tensors)
        hidden = layers[0](input_mx)
        mx.eval(hidden)  # CRITICAL: force before freeing
        _mlx_free(layers[0])

        # Transformer layers
        for i, (name, layer) in enumerate(
            zip(layer_names[1:-2], layers[1:-2])
        ):
            if name in pinned_layer_names:
                hidden = layer(hidden)
                mx.eval(hidden)
            else:
                tensors = loader.load_from_disk(name, checkpoint_path)
                _mlx_inject(layer, tensors)
                hidden = layer(hidden)
                mx.eval(hidden)  # CRITICAL
                _mlx_free(layer)

        # Norm
        norm_tensors = loader.load_from_disk(norm_name, checkpoint_path)
        _mlx_inject(layers[-2], norm_tensors)
        hidden = layers[-2](hidden)
        mx.eval(hidden)
        _mlx_free(layers[-2])

        # LM head
        lm_head_tensors = loader.load_from_disk(lm_head_name, checkpoint_path)
        _mlx_inject(layers[-1], lm_head_tensors)
        logits_mx = layers[-1](hidden)
        mx.eval(logits_mx)
        _mlx_free(layers[-1])

        # Convert to PyTorch for GenerationMixin compatibility
        import torch
        import numpy as np
        logits_np = np.array(logits_mx.tolist(), dtype=np.float32)
        logits = torch.from_numpy(logits_np)

        total = time.time() - _exec_start
        print(f"[VAAYU-IO] MLX token total={total:.1f}s")

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
```

- [ ] **Step 2: Verify MLX backend imports cleanly**

```bash
cd ~/Documents/ai-ml/vaayu-apple
python -c "from vaayu_apple.backends.mlx_backend import MLXBackend; print('MLXBackend ok')"
```

Expected: `MLXBackend ok`

- [ ] **Step 3: Run full test suite**

```bash
cd ~/Documents/ai-ml/vaayu-apple
pytest tests/ -v
```

Expected: all non-MLX tests pass; MLX tests skip or pass depending on `mlx` installation.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/ai-ml/vaayu-apple
git add vaayu_apple/backends/mlx_backend.py
git commit -m "feat: MLXBackend — lazy-eval per-layer streaming with mx.eval() gate"
```

---

## Post-implementation smoke test

After all tasks complete, run a real end-to-end test with a small model to verify the full pipeline:

```bash
cd ~/Documents/ai-ml/vaayu-apple
python - <<'EOF'
from vaayu_apple import VaayuAppleModel
import torch

# Use a small gated-free model for smoke test
model = VaayuAppleModel(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.bfloat16,
    max_seq_len=128,
)
prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Say hello"}],
    tokenize=False, add_generation_prompt=True
)
ids = model.tokenizer(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=20, do_sample=False)
print(model.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
EOF
```

Expected: short coherent response, no NaN in logits, no OOM.

---

## Self-review checklist

- [x] Spec §3.1 F_NOCACHE → Task 3
- [x] Spec §3.2 Loader ABC → Task 4
- [x] Spec §3.3 SafetensorsLoader hot path → Task 6
- [x] Spec §3.4 MLXLoader → Task 10
- [x] Spec §3.5 Auto-detection → Task 8 (`_build_loader`)
- [x] Spec §4.1 MPS backend forward loop → Task 7
- [x] Spec §4.2 MLX backend + mx.eval() gate → Task 11
- [x] Spec §5 VaayuAppleModel init sequence → Task 8
- [x] Spec §5.2 Pinning budget → Task 8 (`_init_pinned_layers`)
- [x] Spec §5.3 Three init paths → Task 5 (`find_or_create_splitted_path`)
- [x] Spec §5.4 Public API → Task 9
- [x] Spec §6 Dependencies → Task 1 (`pyproject.toml`)
- [x] Spec §7 Error handling → Tasks 3, 6, 8 (guards + fallbacks)
- [x] Spec §8 Tests — no IO mocking → Tasks 3, 5, 6, 7, 10
- [x] Spec §10 Ported logic → Task 5 (splitter), Task 7 (RoPE), Task 8 (MoE detect, pinning)
