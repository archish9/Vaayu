# Chunked Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process long prompts in 512-token chunks during prefill to prevent OOM crashes, and fix a latent position-ID overflow bug.

**Architecture:** `_chunked_prefill()` calls `forward()` iteratively with growing KV cache. A guard at the top of `forward()` routes long prompts (seq_len > prefill_chunk_size, no prior KV) transparently. Position IDs are computed dynamically from `max(max_seq_len, past_len + cur_len)` instead of the hardcoded `max_seq_len`.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers` (DynamicCache / CausalLMOutputWithPast), pytest

---

## File Map

| File | Change |
|------|--------|
| `vaayu/vaayu_base.py` | Add `prefill_chunk_size` param; add `_chunked_prefill()` method; add guard in `forward()`; fix position ID range |
| `tests/test_chunked_prefill.py` | New — all unit tests for this feature |
| `test_quick.py` | Add `PREFILL_CHUNK_SIZE` config var; pass to `from_pretrained()` |

`vaayu/auto_model.py` requires **no changes** — `**kwargs` already passes `prefill_chunk_size` through to the model constructor.

---

## Task 1: `prefill_chunk_size` parameter storage

**Files:**
- Modify: `vaayu/vaayu_base.py:161-165` (the `__init__` signature line)
- Create: `tests/test_chunked_prefill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunked_prefill.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import patch, MagicMock

from tests.test_vaayu_base_kwargs import _make_base


def test_prefill_chunk_size_default_is_512():
    m = _make_base()
    assert m.prefill_chunk_size == 512


def test_prefill_chunk_size_custom_stored():
    m = _make_base({"prefill_chunk_size": 256})
    assert m.prefill_chunk_size == 256


def test_prefill_chunk_size_zero_stored():
    m = _make_base({"prefill_chunk_size": 0})
    assert m.prefill_chunk_size == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
source venv/bin/activate
python -m pytest tests/test_chunked_prefill.py -v
```

Expected: `AttributeError: 'VaayuBaseModel' object has no attribute 'prefill_chunk_size'`

- [ ] **Step 3: Add `prefill_chunk_size` to `__init__` signature**

In `vaayu/vaayu_base.py`, change line 161-165 from:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto", max_pinned_layers=0,
                 layer_names_dict=None, tokenizer_class=None):
```

to:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto", max_pinned_layers=0,
                 layer_names_dict=None, tokenizer_class=None, prefill_chunk_size=512):
```

Then add the assignment inside `__init__`, after the existing `self.max_pinned_layers = max_pinned_layers` line (around line 250):

```python
        self.prefill_chunk_size = prefill_chunk_size
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/test_chunked_prefill.py::test_prefill_chunk_size_default_is_512 tests/test_chunked_prefill.py::test_prefill_chunk_size_custom_stored tests/test_chunked_prefill.py::test_prefill_chunk_size_zero_stored -v
```

Expected: 3 passed

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
python -m pytest tests/ -q
```

Expected: all prior tests + 3 new tests pass

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_chunked_prefill.py
git commit -m "feat: add prefill_chunk_size param to VaayuBaseModel (default=512)"
```

---

## Task 2: `_chunked_prefill()` method

**Files:**
- Modify: `vaayu/vaayu_base.py` — add new method before `forward()`
- Modify: `tests/test_chunked_prefill.py` — append new tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunked_prefill.py`:

```python
from transformers.modeling_outputs import CausalLMOutputWithPast


def _fake_kv(seq_len):
    """Return a MagicMock KV cache with get_seq_length() = seq_len."""
    kv = MagicMock()
    kv.get_seq_length.return_value = seq_len
    return kv


def _fake_out(seq_len, vocab=100, kv_seq_len=None):
    """Return a CausalLMOutputWithPast with the given logits shape."""
    return CausalLMOutputWithPast(
        logits=torch.zeros(1, seq_len, vocab),
        past_key_values=_fake_kv(kv_seq_len or seq_len),
    )


def test_chunked_prefill_calls_forward_twice_for_double_chunk():
    """1024-token prompt, chunk_size=512 → exactly 2 forward() calls."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512, kv_seq_len=512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 2


def test_chunked_prefill_calls_forward_three_times_for_triple_chunk():
    """1536-token prompt, chunk_size=512 → exactly 3 forward() calls."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1536))

    with patch.object(m, "forward", return_value=_fake_out(512, kv_seq_len=512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 3


def test_chunked_prefill_handles_remainder():
    """600 tokens, chunk_size=512 → 2 calls: chunk sizes [512, 88]."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 600))

    kv1 = _fake_kv(512)
    kv2 = _fake_kv(600)
    out1 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)
    out2 = CausalLMOutputWithPast(logits=torch.zeros(1, 88, 100), past_key_values=kv2)

    with patch.object(m, "forward", side_effect=[out1, out2]) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 2
    first_ids  = mock_fwd.call_args_list[0][1]["input_ids"]
    second_ids = mock_fwd.call_args_list[1][1]["input_ids"]
    assert first_ids.shape[1] == 512
    assert second_ids.shape[1] == 88


def test_chunked_prefill_threads_kv_cache_between_chunks():
    """KV cache from chunk N is passed as past_key_values to chunk N+1."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    kv1 = _fake_kv(512)
    kv2 = _fake_kv(1024)
    out1 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)
    out2 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv2)

    with patch.object(m, "forward", side_effect=[out1, out2]) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    # Second call must receive the KV cache returned by the first call
    second_kwargs = mock_fwd.call_args_list[1][1]
    assert second_kwargs["past_key_values"] is kv1


def test_chunked_prefill_first_call_has_no_past_kv():
    """First chunk is always prefilled from scratch (past_key_values=None)."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    first_kwargs = mock_fwd.call_args_list[0][1]
    assert first_kwargs["past_key_values"] is None


def test_chunked_prefill_returns_last_chunk_output():
    """Return value is the output of the final forward() call."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    out1 = _fake_out(512, kv_seq_len=512)
    out2 = _fake_out(512, kv_seq_len=1024)

    with patch.object(m, "forward", side_effect=[out1, out2]):
        result = m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                                    output_attentions=False, output_hidden_states=False, return_dict=True)

    assert result is out2


def test_chunked_prefill_slices_attention_mask_correctly():
    """Attention mask for chunk [0:512] should be mask[:, :512]; for [512:1024] mask[:, :1024]."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))
    attention_mask = torch.ones(1, 1024)

    kv1 = _fake_kv(512)
    out = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)

    with patch.object(m, "forward", return_value=out) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=attention_mask, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    first_mask  = mock_fwd.call_args_list[0][1]["attention_mask"]
    second_mask = mock_fwd.call_args_list[1][1]["attention_mask"]
    assert first_mask.shape[1] == 512
    assert second_mask.shape[1] == 1024


def test_chunked_prefill_none_attention_mask_stays_none():
    """When attention_mask=None (SDPA/flash path), forward() receives None each call."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    for call in mock_fwd.call_args_list:
        assert call[1]["attention_mask"] is None


def test_chunked_prefill_non_dict_return():
    """When return_dict=False, result is a tuple with logits first."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)):
        result = m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                                    output_attentions=False, output_hidden_states=False, return_dict=False)

    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)  # logits first
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_chunked_prefill.py -v -k "chunked_prefill"
```

Expected: `AttributeError: 'VaayuBaseModel' object has no attribute '_chunked_prefill'`

- [ ] **Step 3: Implement `_chunked_prefill()`**

In `vaayu/vaayu_base.py`, add this method immediately before `def forward(` (around line 968):

```python
    def _chunked_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        chunk_size = self.prefill_chunk_size
        total_len  = input_ids.shape[1]
        n_chunks   = (total_len + chunk_size - 1) // chunk_size
        print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

        past_key_values = None
        output = None

        for start in range(0, total_len, chunk_size):
            end        = min(start + chunk_size, total_len)
            chunk_ids  = input_ids[:, start:end]
            chunk_mask = attention_mask[:, :end] if attention_mask is not None else None

            output = self.forward(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            past_key_values = output.past_key_values

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

        if not return_dict:
            return tuple(v for v in [
                output.logits,
                output.past_key_values if use_cache else None,
                output.hidden_states,
                output.attentions,
            ] if v is not None)
        return output
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/test_chunked_prefill.py -v
```

Expected: all tests pass (3 from Task 1 + 9 from Task 2)

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -q
```

Expected: all prior tests + 12 new tests pass

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_chunked_prefill.py
git commit -m "feat: add _chunked_prefill() method to VaayuBaseModel"
```

---

## Task 3: Routing guard in `forward()`

**Files:**
- Modify: `vaayu/vaayu_base.py` — insert guard in `forward()` between line 990 and 992
- Modify: `tests/test_chunked_prefill.py` — append new tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunked_prefill.py`:

```python
from transformers.cache_utils import DynamicCache


def test_forward_routes_to_chunked_prefill_for_long_prompt():
    """Prompt longer than chunk_size, no prior KV → _chunked_prefill called."""
    m = _make_base({"prefill_chunk_size": 4})
    input_ids = torch.randint(0, 100, (1, 8))  # 8 > 4

    sentinel = CausalLMOutputWithPast(
        logits=torch.zeros(1, 4, 100),
        past_key_values=DynamicCache(),
    )
    with patch.object(m, "_chunked_prefill", return_value=sentinel) as mock_cp:
        result = m.forward(input_ids=input_ids, past_key_values=None,
                           use_cache=True, return_dict=True)

    mock_cp.assert_called_once()
    assert result is sentinel


def test_forward_skips_chunked_prefill_when_disabled():
    """chunk_size=0 → _chunked_prefill never called, even for long prompts."""
    m = _make_base({"prefill_chunk_size": 0})
    m._active_attn_impl = "eager"

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      use_cache=False, return_dict=True)
        except Exception:
            pass  # layer loop will fail without real weights; we only need the routing check

    mock_cp.assert_not_called()


def test_forward_skips_chunked_prefill_when_seq_not_exceeds_chunk():
    """Prompt exactly equal to chunk_size → no chunking."""
    m = _make_base({"prefill_chunk_size": 8})
    m._active_attn_impl = "eager"

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      use_cache=False, return_dict=True)
        except Exception:
            pass

    mock_cp.assert_not_called()


def test_forward_skips_chunked_prefill_when_past_kv_exists():
    """Decode step (past_kv_len > 0) → no chunking even for 'long' seq."""
    m = _make_base({"prefill_chunk_size": 4})
    m._active_attn_impl = "eager"

    past_kv = MagicMock()
    past_kv.get_seq_length.return_value = 4  # non-zero past

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      past_key_values=past_kv, use_cache=True, return_dict=True)
        except Exception:
            pass

    mock_cp.assert_not_called()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_chunked_prefill.py::test_forward_routes_to_chunked_prefill_for_long_prompt -v
```

Expected: FAIL — `_chunked_prefill` is not called because the guard doesn't exist yet.

- [ ] **Step 3: Add the guard in `forward()`**

In `vaayu/vaayu_base.py`, after the profiling block and before `batch = [...]` (currently line 992), insert the guard. The surrounding context looks like:

```python
        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()


        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
```

Change it to:

```python
        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        if (self.prefill_chunk_size > 0
                and input_ids is not None
                and input_ids.shape[1] > self.prefill_chunk_size):
            _past_kv_len = (self.get_past_key_values_cache_seq_len(past_key_values)
                            if past_key_values is not None else 0)
            if _past_kv_len == 0:
                return self._chunked_prefill(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
```

- [ ] **Step 4: Run routing tests**

```bash
python -m pytest tests/test_chunked_prefill.py -v -k "forward"
```

Expected: all 4 routing tests pass

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -q
```

Expected: all prior tests + all new tests pass

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_chunked_prefill.py
git commit -m "feat: add chunked prefill routing guard to forward()"
```

---

## Task 4: Fix position ID range

**Files:**
- Modify: `vaayu/vaayu_base.py:1002` — replace hardcoded `max_seq_len` with dynamic range
- Modify: `tests/test_chunked_prefill.py` — append regression test

The bug: `position_ids = torch.arange(self.max_seq_len, ...)`. For chunk 2 (past_len=512, seq_len=512), `get_position_ids_args(pos_ids, 512, 512)` tries to slice `pos_ids[:, 512:1024]` from a 512-element tensor. PyTorch silently returns an empty tensor, which corrupts hidden states.

- [ ] **Step 1: Write the failing regression tests**

Append to `tests/test_chunked_prefill.py`:

```python
def test_position_id_slice_is_empty_when_past_exceeds_max_seq_len():
    """Demonstrate the OLD bug: position_ids of max_seq_len size returns empty slice for past+seq."""
    m = _make_base()
    m.max_seq_len = 4
    # Old behavior: position_ids = arange(4) → only 4 elements
    old_pos_ids = torch.arange(4)[None, :]
    result = m.get_position_ids_args(old_pos_ids, len_p=4, len_s=4)
    # PyTorch slice [4:8] on a 4-element dim returns empty tensor — silent corruption
    assert result["position_ids"].shape[1] == 0  # BUG: should be 4


def test_position_id_slice_correct_with_dynamic_range():
    """NEW behavior: pos_range = max(max_seq_len, past+seq) → slice is [past : past+seq]."""
    m = _make_base()
    m.max_seq_len = 4
    past_len, seq_len = 4, 4
    pos_range = max(m.max_seq_len, past_len + seq_len)  # = 8
    new_pos_ids = torch.arange(pos_range)[None, :]
    result = m.get_position_ids_args(new_pos_ids, len_p=past_len, len_s=seq_len)
    assert result["position_ids"].shape == (1, 4)
    assert result["position_ids"][0, 0].item() == 4    # starts at past_len
    assert result["position_ids"][0, -1].item() == 7   # ends at past_len + seq_len - 1
```

- [ ] **Step 2: Run to confirm the bug-demonstration test passes and the fix-test passes**

```bash
python -m pytest tests/test_chunked_prefill.py::test_position_id_slice_is_empty_when_past_exceeds_max_seq_len tests/test_chunked_prefill.py::test_position_id_slice_correct_with_dynamic_range -v
```

Expected: both tests pass (they test the helper `get_position_ids_args` directly, which already works correctly — the fix is in how `forward()` computes the tensor it passes in).

- [ ] **Step 3: Apply the position ID fix in `forward()`**

In `vaayu/vaayu_base.py`, replace line 1002:

```python
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]
```

with:

```python
        _past_len_for_pos = (self.get_past_key_values_cache_seq_len(past_key_values)
                             if past_key_values is not None else 0)
        _cur_len_for_pos  = input_ids.shape[1] if input_ids is not None else 0
        _pos_range        = max(self.max_seq_len, _past_len_for_pos + _cur_len_for_pos)
        position_ids      = torch.arange(_pos_range, dtype=torch.long, device=self.running_device)[None, :]
```

- [ ] **Step 4: Run full suite to confirm no regressions**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_chunked_prefill.py
git commit -m "fix: compute position_ids range dynamically from past_len + seq_len"
```

---

## Task 5: Update `test_quick.py`

**Files:**
- Modify: `test_quick.py` — add `PREFILL_CHUNK_SIZE` to config block; pass to `from_pretrained()`

No automated test for this task — it's a config file change verified by running the script manually.

- [ ] **Step 1: Add `PREFILL_CHUNK_SIZE` to the config block**

In `test_quick.py`, the config block currently ends at line 31. Add one line after `MAX_LENGTH`:

```python
MAX_NEW_TOKENS    = 20
MAX_LENGTH        = 128
PREFILL_CHUNK_SIZE = 512    # 0 = disabled; tokens per prefill chunk
```

- [ ] **Step 2: Pass it to `from_pretrained()`**

Change the `AutoModel.from_pretrained(...)` call in `test_quick.py` from:

```python
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
)
```

to:

```python
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
)
```

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add test_quick.py
git commit -m "feat: add PREFILL_CHUNK_SIZE config var to test_quick.py"
```

---

## Self-Review

**Spec coverage:**
- ✅ `prefill_chunk_size=512` default → Task 1
- ✅ `_chunked_prefill()` method → Task 2
- ✅ Routing guard in `forward()` (auto-detect, no-past guard) → Task 3
- ✅ Position ID range fix → Task 4
- ✅ `test_quick.py` config var → Task 5
- ✅ `auto_model.py` requires no changes (noted in File Map)
- ✅ `return_dict` handling in `_chunked_prefill` (tuple vs CausalLMOutputWithPast) → Task 2 Step 3
- ✅ Attention mask slicing per chunk (mask[:, :end]) → Task 2 Step 1 test + Task 2 Step 3 impl

**Placeholder scan:** None found.

**Type consistency:**
- `_chunked_prefill` signature matches guard call in Task 3 ✓
- `_fake_out()` / `_fake_kv()` helpers defined before first use ✓
- `past_key_values` variable name consistent across tasks ✓
- Position ID fix uses `_past_len_for_pos` / `_cur_len_for_pos` / `_pos_range` — underscore prefix avoids shadowing `past_key_values` in outer scope ✓
