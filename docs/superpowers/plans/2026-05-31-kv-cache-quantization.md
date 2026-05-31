# KV Cache Quantization (Plan E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compress the KV cache from BF16 to FP8, freeing ~1.3 GB VRAM at 8k context with zero new dependencies.

**Architecture:** `FP8DynamicCache` subclasses `DynamicCache`, overriding `update()` to quantize only new tokens on each call (O(1) decode overhead). Per-head per-token float32 scales are stored alongside FP8 data; full cache is dequantized to BF16 before returning to attention. `VaayuBaseModel` gains a `kv_cache_mode` param (`"fp16"` | `"fp8"` | `"auto"`); `auto` picks `"fp8"` when `EXL3LoaderStrategy` is active.

**Tech Stack:** PyTorch 2.11 (`torch.float8_e4m3fn`), HuggingFace `transformers` `DynamicCache`, pytest

---

## File Map

| File | What changes |
|------|-------------|
| `vaayu/vaayu_base.py` | Add `FP8DynamicCache` class after line 47; add `kv_cache_mode` param to `__init__` signature (line 116) and store at line 199; add `_create_kv_cache()` method after `_get_vram_pinning_budget()`; update `_get_vram_pinning_budget()` (lines 552–566); change `forward()` line 912 |
| `tests/test_fp8_cache.py` | New: unit tests for `FP8DynamicCache` in isolation |
| `tests/test_vaayu_base_kwargs.py` | Append: tests for `kv_cache_mode` param acceptance and `_get_vram_pinning_budget()` output |
| `test_quick.py` | Add `argparse` + `--kv_cache_mode` arg |

---

## Task 1: `FP8DynamicCache` class

**Files:**
- Modify: `vaayu/vaayu_base.py` after line 47
- Create: `tests/test_fp8_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fp8_cache.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch


# Skip entire file if FP8 not available (PyTorch < 2.1)
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn not available"
)


@pytest.fixture
def cache():
    from vaayu.vaayu_base import FP8DynamicCache
    return FP8DynamicCache()


def _kv(batch=1, heads=4, tokens=3, head_dim=8, val=1.0):
    """Return (key, value) BF16 tensors filled with val."""
    shape = (batch, heads, tokens, head_dim)
    return torch.full(shape, val, dtype=torch.bfloat16), \
           torch.full(shape, val * 0.5, dtype=torch.bfloat16)


def test_update_stores_fp8(cache):
    k, v = _kv(tokens=1)
    cache.update(k, v, layer_idx=0)
    assert cache.key_cache[0].dtype == torch.float8_e4m3fn
    assert cache.value_cache[0].dtype == torch.float8_e4m3fn


def test_update_scales_shape(cache):
    batch, heads, tokens, head_dim = 1, 4, 1, 8
    k, v = _kv(batch, heads, tokens, head_dim)
    cache.update(k, v, layer_idx=0)
    # scale shape: [batch, heads, tokens, 1]
    assert cache._key_scales[0].shape == (batch, heads, tokens, 1)
    assert cache._value_scales[0].shape == (batch, heads, tokens, 1)


def test_update_returns_correct_shape(cache):
    batch, heads, tokens, head_dim = 1, 4, 1, 8
    k, v = _kv(batch, heads, tokens, head_dim)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (batch, heads, tokens, head_dim)
    assert v_out.shape == (batch, heads, tokens, head_dim)


def test_update_returns_bfloat16(cache):
    k, v = _kv(tokens=1)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.bfloat16


def test_dequant_approximate(cache):
    """Dequantized output should be close to original (FP8 has ~3 decimal digits)."""
    k, v = _kv(tokens=1, val=2.0)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert torch.allclose(k_out.float(), k.float(), atol=0.1)
    assert torch.allclose(v_out.float(), v.float(), atol=0.1)


def test_accumulation_across_decode_steps(cache):
    """Three sequential decode steps → seq_len grows to 3."""
    for step in range(3):
        k, v = _kv(tokens=1, val=float(step + 1))
        cache.update(k, v, layer_idx=0)
    assert cache.key_cache[0].shape[-2] == 3
    assert cache.value_cache[0].shape[-2] == 3
    assert cache._key_scales[0].shape[-2] == 3


def test_get_seq_length(cache):
    k, v = _kv(tokens=1)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length() == 2


def test_multi_layer(cache):
    """Different layer_idx entries are independent."""
    k0, v0 = _kv(tokens=1, val=1.0)
    k1, v1 = _kv(tokens=1, val=2.0)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)
    assert len(cache.key_cache) == 2
    assert cache.key_cache[0].shape[-2] == 1
    assert cache.key_cache[1].shape[-2] == 1


def test_prefill_then_decode(cache):
    """Prefill (4 tokens) + 2 decode steps → seq_len 6."""
    k_pre, v_pre = _kv(tokens=4, val=1.0)
    cache.update(k_pre, v_pre, layer_idx=0)
    for _ in range(2):
        k, v = _kv(tokens=1, val=2.0)
        cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length() == 6


def test_returned_shape_grows_with_seq(cache):
    """After decode steps, returned K/V shape includes ALL past tokens."""
    k, v = _kv(tokens=4, val=1.0)
    cache.update(k, v, layer_idx=0)  # prefill
    k2, v2 = _kv(tokens=1, val=2.0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)  # decode
    # Output must include all 5 tokens for the attention layer
    assert k_out.shape[-2] == 5
    assert v_out.shape[-2] == 5


def test_zero_activations_no_division_by_zero(cache):
    """All-zero K/V should not cause NaN (scale clamped to 1e-12)."""
    k = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
    v = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()
```

- [ ] **Step 2: Run tests — expect ImportError on `FP8DynamicCache`**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_fp8_cache.py -v 2>&1 | tail -20
```

Expected: all tests **ERROR** with `ImportError: cannot import name 'FP8DynamicCache'`

- [ ] **Step 3: Add `FP8DynamicCache` to `vaayu/vaayu_base.py`**

Insert after line 47 (after the `cache_utils` try/except block, before the blank lines leading to `def _layer_path_exists`). Add this block:

```python
class FP8DynamicCache(DynamicCache):
    """
    DynamicCache variant that stores K/V tensors in torch.float8_e4m3fn.

    Per-head per-token scales (shape [B, H, S, 1]) are stored alongside
    the FP8 data so decode steps are O(1): only new tokens are quantized.
    Full cache is dequantized to BF16/FP16 before returning to attention.
    """

    def __init__(self):
        super().__init__()
        self._key_scales: dict = {}
        self._value_scales: dict = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0

        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_fp8], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_fp8], dim=-2)
            self._key_scales[layer_idx] = torch.cat([self._key_scales[layer_idx], k_scale], dim=-2)
            self._value_scales[layer_idx] = torch.cat([self._value_scales[layer_idx], v_scale], dim=-2)
        else:
            self.key_cache.append(k_fp8)
            self.value_cache.append(v_fp8)
            self._key_scales[layer_idx] = k_scale
            self._value_scales[layer_idx] = v_scale

        k_out = self.key_cache[layer_idx].to(key_states.dtype) * self._key_scales[layer_idx]
        v_out = self.value_cache[layer_idx].to(value_states.dtype) * self._value_scales[layer_idx]
        return k_out, v_out
```

Note: `get_seq_length()` is deliberately NOT overridden — `DynamicCache.get_seq_length()` reads `self.key_cache[layer_idx].shape[-2]`, which is correct since we store FP8 tensors with the same shape.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_fp8_cache.py -v 2>&1 | tail -20
```

Expected: all 11 tests **PASSED**

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_fp8_cache.py
git commit -m "feat: add FP8DynamicCache with per-token per-head scales"
```

---

## Task 2: `kv_cache_mode` param on `VaayuBaseModel.__init__()`

**Files:**
- Modify: `vaayu/vaayu_base.py` lines 112–116 (signature) and ~199 (storage)
- Modify: `tests/test_vaayu_base_kwargs.py` (append new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vaayu_base_kwargs.py`:

```python
def test_kv_cache_mode_default_is_auto():
    m = _make_base()
    assert m.kv_cache_mode == "auto"


def test_kv_cache_mode_fp16_stored():
    m = _make_base({"kv_cache_mode": "fp16"})
    assert m.kv_cache_mode == "fp16"


def test_kv_cache_mode_fp8_stored():
    m = _make_base({"kv_cache_mode": "fp8"})
    assert m.kv_cache_mode == "fp8"
```

- [ ] **Step 2: Run tests — expect AttributeError**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_vaayu_base_kwargs.py::test_kv_cache_mode_default_is_auto -v 2>&1 | tail -10
```

Expected: **FAILED** with `AttributeError: 'VaayuBaseModel' object has no attribute 'kv_cache_mode'`

- [ ] **Step 3: Update `__init__` signature and store the param**

In `vaayu/vaayu_base.py`, change the `__init__` signature at line 112–116 from:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False,
                 layer_names_dict=None, tokenizer_class=None):
```

to:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto",
                 layer_names_dict=None, tokenizer_class=None):
```

Then in the `__init__` body, add storage right after `self.use_torch_compile = use_torch_compile` (line 199):

Change from:
```python
        self.use_torch_compile = use_torch_compile
        self._compiled_layer_fn = None
```

to:

```python
        self.use_torch_compile = use_torch_compile
        self.kv_cache_mode = kv_cache_mode
        self._compiled_layer_fn = None
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_vaayu_base_kwargs.py -v 2>&1 | tail -15
```

Expected: all tests **PASSED** (including the three new ones)

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_vaayu_base_kwargs.py
git commit -m "feat: add kv_cache_mode param to VaayuBaseModel"
```

---

## Task 3: `_create_kv_cache()` factory + updated `_get_vram_pinning_budget()`

**Files:**
- Modify: `vaayu/vaayu_base.py` lines 552–566 (`_get_vram_pinning_budget`) and ~567 (new `_create_kv_cache`)
- Modify: `tests/test_vaayu_base_kwargs.py` (append new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vaayu_base_kwargs.py`:

```python
def test_create_kv_cache_fp16_returns_dynamic_cache():
    from transformers.cache_utils import DynamicCache
    m = _make_base({"kv_cache_mode": "fp16"})
    cache = m._create_kv_cache()
    assert isinstance(cache, DynamicCache)


def test_create_kv_cache_fp8_returns_fp8_cache():
    import torch
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("FP8 not available")
    from vaayu.vaayu_base import FP8DynamicCache
    m = _make_base({"kv_cache_mode": "fp8"})
    cache = m._create_kv_cache()
    assert isinstance(cache, FP8DynamicCache)


def test_create_kv_cache_auto_fp16_for_standard_strategy():
    """auto mode with non-EXL3 strategy → FP16 DynamicCache."""
    from transformers.cache_utils import DynamicCache
    m = _make_base({"kv_cache_mode": "auto"})
    # _make_base patches build_loader_strategy to return MagicMock (not EXL3LoaderStrategy)
    cache = m._create_kv_cache()
    assert isinstance(cache, DynamicCache)


def test_vram_budget_fp8_reserves_less_than_fp16():
    """FP8 formula reserves less VRAM than BF16 formula (mocks GPU, no real hardware needed)."""
    from unittest.mock import patch, MagicMock
    import torch

    m_fp8 = _make_base({"kv_cache_mode": "fp8"})
    m_fp16 = _make_base({"kv_cache_mode": "fp16"})

    # Simulate a 24 GB GPU regardless of test machine
    m_fp8.device = torch.device("cuda:0")
    m_fp16.device = torch.device("cuda:0")
    fake_props = MagicMock()
    fake_props.total_memory = 24 * 1024 ** 3

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        budget_fp8 = m_fp8._get_vram_pinning_budget()
        budget_fp16 = m_fp16._get_vram_pinning_budget()

    assert budget_fp8 > budget_fp16
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_vaayu_base_kwargs.py::test_create_kv_cache_fp16_returns_dynamic_cache -v 2>&1 | tail -10
```

Expected: **FAILED** with `AttributeError: 'VaayuBaseModel' object has no attribute '_create_kv_cache'`

- [ ] **Step 3: Replace `_get_vram_pinning_budget()` and add `_create_kv_cache()`**

In `vaayu/vaayu_base.py`, replace the entire `_get_vram_pinning_budget` method (lines 552–566):

```python
    def _get_vram_pinning_budget(self) -> int:
        """
        VRAM budget available for permanently pinned layers.

        Reserves space for KV cache (sized by kv_cache_mode) + 1.5 GB overhead
        (CUDA context + streaming double-buffer).

        KV cache formulas:
          BF16: n_layers × 2 × n_kv_heads × head_dim × max_seq_len × 2 bytes
          FP8:  n_layers × 2 × n_kv_heads × (head_dim + 4) × max_seq_len × 1 byte
                (+ 4 bytes/head/token for float32 scale)
        """
        if not torch.cuda.is_available():
            return 0
        if not str(self.device).startswith("cuda"):
            return 0
        total = torch.cuda.get_device_properties(self.device).total_memory

        n_layers = getattr(self.config, 'num_hidden_layers', 80)
        n_kv_heads = getattr(self.config, 'num_key_value_heads', 8)
        head_dim = getattr(self.config, 'head_dim',
                           self.config.hidden_size // self.config.num_attention_heads)

        mode = getattr(self, 'kv_cache_mode', 'fp16')
        if mode == 'fp8':
            kv_bytes = n_layers * 2 * n_kv_heads * (head_dim + 4) * self.max_seq_len
        else:
            kv_bytes = n_layers * 2 * n_kv_heads * head_dim * self.max_seq_len * 2

        overhead = int(1.5 * 1024 ** 3)
        return max(0, total - kv_bytes - overhead)
```

Then add `_create_kv_cache()` immediately after (still inside the `# ── VRAM pinning ──` section, before `_init_pinned_layers`):

```python
    def _create_kv_cache(self):
        """Return a KV cache instance sized and typed for the configured kv_cache_mode."""
        mode = self.kv_cache_mode

        if mode == "auto":
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "fp8" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"

        if mode == "fp16":
            return DynamicCache()

        if not hasattr(torch, "float8_e4m3fn"):
            print("[Vaayu] FP8 KV cache unavailable (PyTorch < 2.1). Falling back to FP16.")
            return DynamicCache()

        print("[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)")
        return FP8DynamicCache()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/test_vaayu_base_kwargs.py -v 2>&1 | tail -20
```

Expected: all tests **PASSED**

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_vaayu_base_kwargs.py
git commit -m "feat: add _create_kv_cache() and update _get_vram_pinning_budget() for FP8"
```

---

## Task 4: Wire `forward()` to use `_create_kv_cache()`

**Files:**
- Modify: `vaayu/vaayu_base.py` line 912

This task has no separate test — the existing test suite + Task 1 tests cover the cache interface. The integration is validated in Task 5 via `test_quick.py`.

- [ ] **Step 1: Change `forward()` line 912**

In `vaayu/vaayu_base.py`, change:

```python
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
```

to:

```python
        if use_cache and past_key_values is None:
            past_key_values = self._create_kv_cache()
```

- [ ] **Step 2: Run full test suite — all still pass**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/ -v 2>&1 | tail -25
```

Expected: all tests **PASSED** with no regressions

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: wire forward() to use _create_kv_cache() instead of bare DynamicCache"
```

---

## Task 5: Add `--kv_cache_mode` to `test_quick.py`

**Files:**
- Modify: `test_quick.py`

- [ ] **Step 1: Add `argparse` and the new argument**

In `test_quick.py`, replace the existing script body (everything after the imports and before `print("Using vaayu from:")`) so the file reads:

```python
"""
Quick test without installation
This adds the local path to Python's import system
"""

import sys
import os
import time
import argparse
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="Vaayu quick test")
parser.add_argument("--kv_cache_mode", type=str, default="auto",
                    choices=["fp16", "fp8", "auto"],
                    help="KV cache precision: fp16, fp8 (~50%% VRAM savings), auto")
args = parser.parse_args()

print(f"Using vaayu from: {project_root}")

# Device selection: GPU by default, falls back to CPU if no GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Now import and test
from vaayu import AutoModel

MAX_LENGTH = 128

print("Loading model...")
model = AutoModel.from_pretrained(
    "/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw",
    device=DEVICE,
    kv_cache_mode=args.kv_cache_mode,
)

messages = [
    {"role": "user", "content": "What is the capital of United States?"}
]
formatted_prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

input_tokens = model.tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

print("Generating output...")
print(input_tokens['input_ids'].to(DEVICE))
start_time = time.time()
generation_output = model.generate(
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)
end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60.0

output = model.tokenizer.decode(generation_output.sequences[0])

print(f"\nOutput: {output}")
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
```

- [ ] **Step 2: Verify argparse works without a model**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/python test_quick.py --help
```

Expected output includes:
```
  --kv_cache_mode {fp16,fp8,auto}
                        KV cache precision: fp16, fp8 (~50% VRAM savings), auto
```

- [ ] **Step 3: Run full test suite — no regressions**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/pytest tests/ -v 2>&1 | tail -10
```

Expected: all tests **PASSED**

- [ ] **Step 4: Commit**

```bash
git add test_quick.py
git commit -m "feat: add --kv_cache_mode CLI arg to test_quick.py"
```

---

## Task 6: Integration verification (manual, with real model)

No code changes. Confirms VRAM savings and output quality on the actual 70B model.

- [ ] **Step 1: Run FP16 baseline, note VRAM peak**

In one terminal:
```bash
watch -n 0.5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

In another:
```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/python test_quick.py --kv_cache_mode fp16
```

Record peak VRAM from `nvidia-smi`. Confirm `[Vaayu] KV Cache:` line is NOT printed (FP16 is silent).

- [ ] **Step 2: Run FP8, compare VRAM and output**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/python test_quick.py --kv_cache_mode fp8
```

Expected:
- `[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)` printed during generation
- Peak VRAM measurably lower (exact savings depend on context length; at 128 tokens savings are small)
- Output text semantically identical to FP16 run

- [ ] **Step 3: Verify `auto` selects FP8 for EXL3 model**

```bash
cd /home/a/code/ai-ml/Vaayu
venv/bin/python test_quick.py --kv_cache_mode auto
```

Expected: `[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)` printed (model is EXL3, so `auto` → `fp8`).

---

## Self-Review Checklist

Spec requirements → tasks:

| Spec requirement | Covered by |
|-----------------|-----------|
| `FP8DynamicCache` subclass with FP8 storage | Task 1 |
| Per-head per-token scales | Task 1 (`_kv_scale` shape `[B,H,S,1]`) |
| O(1) decode (quantize new tokens only) | Task 1 (only `k_new`/`v_new` passed to quant) |
| `kv_cache_mode` param (`fp16`/`fp8`/`auto`) | Task 2 |
| `auto` → `fp8` for EXL3LoaderStrategy | Task 3 (`_create_kv_cache`) |
| PyTorch FP8 availability guard | Task 3 (`hasattr(torch, "float8_e4m3fn")`) |
| `_get_vram_pinning_budget()` uses actual KV size | Task 3 |
| `_get_vram_pinning_budget()` device guard (CPU model, CUDA available) | Task 3 (bugfix: `if not str(self.device).startswith("cuda"): return 0`) |
| `forward()` uses `_create_kv_cache()` | Task 4 |
| `--kv_cache_mode` CLI arg | Task 5 |
| Integration VRAM verification | Task 6 |
| `auto_model.py` no change needed | Confirmed — `**kwargs` already forwarded |
