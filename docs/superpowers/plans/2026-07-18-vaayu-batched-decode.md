# Vaayu Feature B — Multi-Query Batched Decode (static v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run N parallel agent streams in one batched decode forward so each MoE expert in a layer's routed union is read from NVMe once and computed in one GEMM for all N streams — raising aggregate throughput for the coding agent's parallel fan-out.

**Architecture:** A new `@torch.no_grad()` method `VaayuBaseModel.batched_generate` (modeled on the existing `speculative_generate`) drives `self.forward` over a rectangular left-padded batch, managing N KV rows, per-stream samplers/stop conditions, and yielding `(stream_id, token)` live. VRAM sizing derives max N from live free VRAM and per-stream KV footprint. The MoE expert-dispatch/GEMM path is untouched; the **one** shared-path change is a gated padding-mask + position_ids branch in `forward` (the sdpa path currently nulls the mask and overrides positions, which corrupts a left-padded batch).

**Tech Stack:** Python, PyTorch (sdpa attention), pytest. Reference spec: `docs/superpowers/specs/2026-07-18-vaayu-batched-decode-design.md`.

## Known Limitations (v1)

- The on-device correctness gate (`test_batched_decode.py`, Task 7) has no automated pass/fail signal for a non-crashing wrong-output regression — only a crash fails it. Per-step logit-gap comparison is diagnostic/informational only, a deliberate tradeoff after discovering this stack's greedy decode has genuine pre-existing run-to-run nondeterminism unrelated to Feature B (see `docs/superpowers/plans/2026-07-18-decode-nondeterminism-investigation.md`).
- On the actively-tested box+model (Qwen3-235B EXL3, RTX 5060 Ti 16GB), `plan_batch` always collapses to N=1 once the expert cache is warm (free VRAM below the 1.5GB safety margin). Task 9's measured 1.31x aggregate throughput win was obtained by bypassing `plan_batch`, so it's a capability demonstration, not a production-realistic number on this specific hardware today.

## Global Constraints

Copied verbatim from the spec / CLAUDE.md §5 — every task implicitly includes these:

- **Model-agnostic, NON-NEGOTIABLE.** Never hardcode a model number (layer count, expert count, top_k, head_dim, KV bytes). Every geometry value is derived from `model.config` / `_moe_geometry`. The only literals allowed are **hardware** constants (16 GB VRAM, the ~1.5 GB CUDA/streaming safety margin) — comment them as hardware.
- **Test over both geometries.** Every pure sizing/mask/sampling helper is tested parametrized over a **Qwen-like** (94 layers, 8 kv-heads, head_dim 128) and a **MiniMax-like** (62 layers, geometry differs) shape, and over **fp8 (1 byte) and fp16 (2 byte)** KV.
- **Do not touch the MoE sparse-forward closures** (`_make_fused_sparse_forward*` in `vaayu/moe_patch.py`) except the single gated telemetry line in Task 8. The MoE-EXL3 suite must stay green.
- **The forward change is gated:** when the caller passes no `attention_mask`/`position_ids` (every existing caller — single-stream, MoE, dense), `forward` behavior is byte-identical. Only the new mask-supplied branch differs.
- **v1 requires sdpa attention** (`model._active_attn_impl == "sdpa"`). flash_attention_2 needs the varlen API — deferred to v2. `batched_generate` asserts this at entry.
- **Warn, don't silently degrade:** when N collapses to 1, emit a WARNING and run the single stream.
- **Frequent commits, TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit.

## File Structure

- `vaayu/batch.py` — **full rewrite** (replaces the 50-line HF-`generate` scaffold). Home of all pure, GPU-free helpers: `BatchRequest`, `kv_bytes_for`, `size_batch`, `sample_step`, `build_batch_attention_mask`, `batch_position_ids`, and the model-glue `geo_from_model` / `free_vram_budget` / `plan_batch`.
- `vaayu/vaayu_base.py` — **modify**: (a) gated padding-mask + position_ids + RoPE branch in `forward`; (b) new `VaayuBaseModel.batched_generate` method.
- `vaayu/moe_patch.py` — **modify** (Task 8 only): one gated telemetry line recording per-layer expert-union size when batching.
- `vaayu/telemetry.py` (or wherever `self.telemetry` is defined) — **modify** (Task 8): add `record_batch_overlap`.
- Tests (new): `tests/test_batch_sizing.py`, `tests/test_batch_sampling.py`, `tests/test_batch_masking.py`, `tests/test_batch_plan.py`, `tests/test_batch_overlap.py`.
- On-device scripts (new): `test_batched_decode.py` (correctness + throughput), `test_batched_dense_smoke.py` (dense smoke).

---

### Task 1: `BatchRequest` + `kv_bytes_for` (pure, GPU-free)

**Files:**
- Modify: `vaayu/batch.py` (replace entire file contents across Tasks 1-2, 3, 4, 8; start here)
- Test: `tests/test_batch_sizing.py`

**Interfaces:**
- Produces: `BatchRequest(prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, stop_ids: tuple[int, ...] = ())`; `kv_bytes_for(seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int, elem_bytes: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_sizing.py
import pytest
from vaayu.batch import BatchRequest, kv_bytes_for

# Geometry fixtures — derived shapes, NOT model literals baked into src.
QWEN = dict(n_layers=94, n_kv_heads=8, head_dim=128)
MINIMAX = dict(n_layers=62, n_kv_heads=8, head_dim=128)

@pytest.mark.parametrize("geo", [QWEN, MINIMAX])
@pytest.mark.parametrize("elem_bytes", [1, 2])
def test_kv_bytes_for_scales_linearly(geo, elem_bytes):
    b1 = kv_bytes_for(1000, elem_bytes=elem_bytes, **geo)
    b2 = kv_bytes_for(2000, elem_bytes=elem_bytes, **geo)
    # KV is linear in seq_len
    assert b2 == 2 * b1
    # exact formula: n_layers * 2 * n_kv_heads * head_dim * L * elem_bytes
    assert b1 == geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"] * 1000 * elem_bytes

def test_kv_bytes_fp8_half_of_fp16():
    assert kv_bytes_for(512, **QWEN, elem_bytes=1) * 2 == kv_bytes_for(512, **QWEN, elem_bytes=2)

def test_batch_request_defaults():
    r = BatchRequest(prompt_ids=[1, 2, 3], max_new_tokens=16)
    assert r.temperature == 0.0 and r.top_p == 1.0 and r.stop_ids == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_sizing.py -v`
Expected: FAIL with `ImportError: cannot import name 'BatchRequest'` (batch.py still has the scaffold).

- [ ] **Step 3: Write minimal implementation** (start the batch.py rewrite)

```python
# vaayu/batch.py
"""Multi-query batched decode (Feature B, static v1).

Pure, GPU-free helpers live here; the decode loop is VaayuBaseModel.batched_generate.
All sizing is DERIVED from model geometry — see CLAUDE.md §5.
"""
from dataclasses import dataclass, field


@dataclass
class BatchRequest:
    """One agent stream in a batch."""
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop_ids: tuple[int, ...] = ()


def kv_bytes_for(seq_len: int, n_layers: int, n_kv_heads: int,
                 head_dim: int, elem_bytes: int) -> int:
    """Per-stream KV footprint in bytes. Same formula as _get_vram_pinning_budget.
    elem_bytes: 1 (fp8 KV) or 2 (fp16). All geometry DERIVED, never literal."""
    return n_layers * 2 * n_kv_heads * head_dim * seq_len * elem_bytes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_sizing.py -v`
Expected: PASS (4 cases).

- [ ] **Step 5: Commit**

```bash
git add vaayu/batch.py tests/test_batch_sizing.py
git commit -m "feat(batch): BatchRequest + derived kv_bytes_for (Feature B Task 1)"
```

---

### Task 2: `size_batch` — rectangular greedy admission

**Files:**
- Modify: `vaayu/batch.py`
- Test: `tests/test_batch_sizing.py`

**Interfaces:**
- Consumes: `BatchRequest`, `kv_bytes_for`.
- Produces: `size_batch(requests: list[BatchRequest], budget_bytes: int, n_layers: int, n_kv_heads: int, head_dim: int, elem_bytes: int) -> tuple[list[BatchRequest], list[BatchRequest]]` returning `(admitted, deferred)`. Emits `warnings.warn(...)` when `len(admitted) == 1`.

**Rectangular cost note:** KV is one tensor `[N, heads, max_L, dim]`; every admitted stream allocates to `max_L = max(prompt_len_i + max_new_i)`. Admission cost is `count * kv_bytes_for(max_L)`, NOT the sum of per-stream needs. Sort ascending by `(prompt_len + max_new)` so a long stream doesn't inflate `max_L` for everyone.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_batch_sizing.py
import warnings
from vaayu.batch import size_batch

def _req(prompt_len, max_new):
    return BatchRequest(prompt_ids=[0] * prompt_len, max_new_tokens=max_new)

def test_size_batch_admits_under_rectangular_budget():
    reqs = [_req(100, 50) for _ in range(8)]           # each needs L=150
    per = kv_bytes_for(150, **QWEN, elem_bytes=1)
    budget = per * 4 + per // 2                          # room for exactly 4
    admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=1)
    assert len(admitted) == 4 and len(deferred) == 4

def test_size_batch_rectangular_max_L_inflation():
    # 3 short (L=110) + 1 long (L=2000). Rectangular cost with the long one is
    # 4 * kv(2000); without it 3 * kv(110). Budget fits the 3 short only.
    reqs = [_req(100, 10), _req(100, 10), _req(100, 10), _req(1990, 10)]
    budget = kv_bytes_for(110, **QWEN, elem_bytes=1) * 3 + 1
    admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=1)
    assert len(admitted) == 3
    assert all(len(r.prompt_ids) == 100 for r in admitted)   # long one deferred

def test_size_batch_warns_on_n1():
    reqs = [_req(5000, 100), _req(5000, 100)]
    budget = kv_bytes_for(5100, **QWEN, elem_bytes=2)        # room for exactly 1
    with pytest.warns(UserWarning, match="single-stream"):
        admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=2)
    assert len(admitted) == 1 and len(deferred) == 1

@pytest.mark.parametrize("geo", [QWEN, MINIMAX])
def test_size_batch_both_geometries(geo):
    reqs = [_req(200, 20) for _ in range(6)]
    per = kv_bytes_for(220, **geo, elem_bytes=1)
    admitted, _ = size_batch(reqs, per * 6, **geo, elem_bytes=1)
    assert len(admitted) == 6

def test_size_batch_force_admits_when_none_fit():
    # Even the smallest request alone exceeds the budget. Spec §7: run it alone,
    # never truncate — so admitted must be exactly the smallest, not empty.
    reqs = [_req(10000, 100), _req(5000, 100)]
    with pytest.warns(UserWarning):
        admitted, deferred = size_batch(reqs, budget_bytes=1, **QWEN, elem_bytes=2)
    assert len(admitted) == 1
    assert len(admitted[0].prompt_ids) == 5000     # smallest is force-admitted
    assert len(deferred) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_sizing.py -v -k size_batch`
Expected: FAIL with `ImportError: cannot import name 'size_batch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to vaayu/batch.py (import warnings at top)
import warnings


def size_batch(requests, budget_bytes, n_layers, n_kv_heads, head_dim, elem_bytes):
    """Greedy-admit under RECTANGULAR KV cost. Returns (admitted, deferred).
    Sorts ascending by (prompt_len + max_new) so wide batches of short streams win.
    Cost of admitting the first k sorted streams = k * kv_bytes_for(their max L)."""
    ordered = sorted(requests, key=lambda r: len(r.prompt_ids) + r.max_new_tokens)
    admitted = []
    running_max_L = 0
    cut = len(ordered)
    for i, r in enumerate(ordered):
        cand_max_L = max(running_max_L, len(r.prompt_ids) + r.max_new_tokens)
        cost = (i + 1) * kv_bytes_for(cand_max_L, n_layers, n_kv_heads,
                                      head_dim, elem_bytes)
        if cost <= budget_bytes:
            admitted.append(r)
            running_max_L = cand_max_L
        else:
            cut = i
            break
    deferred = ordered[cut:] if len(admitted) < len(ordered) else []
    if not admitted and ordered:
        # Even the smallest request exceeds budget. Must still serve it (never
        # truncate) — force-admit the smallest and defer the rest (spec §7).
        admitted = [ordered[0]]
        deferred = ordered[1:]
        warnings.warn(
            "batched decode: a single request exceeds the VRAM budget — "
            "running it alone", UserWarning, stacklevel=2)
        return admitted, deferred
    if len(admitted) == 1:
        warnings.warn(
            "batched decode: context too long to batch — running single-stream",
            UserWarning, stacklevel=2)
    return admitted, deferred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_sizing.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add vaayu/batch.py tests/test_batch_sizing.py
git commit -m "feat(batch): size_batch rectangular greedy admission + N=1 warn (Task 2)"
```

---

### Task 3: `sample_step` — per-stream sampling

**Files:**
- Modify: `vaayu/batch.py`
- Test: `tests/test_batch_sampling.py`

**Interfaces:**
- Produces: `sample_step(logits_row: torch.Tensor, temperature: float, top_p: float) -> int`. `logits_row` is 1-D `[vocab]`. temperature `<= 0` → greedy (argmax). Else temperature scale + optional top-p nucleus, then `multinomial`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_sampling.py
import torch
from vaayu.batch import sample_step

def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert sample_step(logits, temperature=0.0, top_p=1.0) == 1

def test_greedy_deterministic_across_calls():
    logits = torch.randn(1000)
    a = sample_step(logits, 0.0, 1.0)
    b = sample_step(logits, 0.0, 1.0)
    assert a == b == int(logits.argmax())

def test_top_p_restricts_support():
    # one dominant token; top_p=0.5 must always pick it under sampling
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    torch.manual_seed(0)
    picks = {sample_step(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}

def test_temperature_sampling_stays_in_vocab():
    logits = torch.randn(32)
    torch.manual_seed(0)
    for _ in range(20):
        tok = sample_step(logits, temperature=0.8, top_p=0.9)
        assert 0 <= tok < 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_sampling.py -v`
Expected: FAIL with `ImportError: cannot import name 'sample_step'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to vaayu/batch.py (import torch at top)
import torch


def sample_step(logits_row: "torch.Tensor", temperature: float, top_p: float) -> int:
    """Sample one token id from a 1-D logits row. Greedy when temperature <= 0."""
    if temperature <= 0.0:
        return int(logits_row.argmax().item())
    probs = torch.softmax(logits_row / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()   # keep the first token crossing top_p
        drop[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_sampling.py -v`
Expected: PASS (4 cases).

- [ ] **Step 5: Commit**

```bash
git add vaayu/batch.py tests/test_batch_sampling.py
git commit -m "feat(batch): per-stream sample_step (greedy/temp/top-p) (Task 3)"
```

---

### Task 4: `build_batch_attention_mask` + `batch_position_ids` — pure mask/position builders

**Files:**
- Modify: `vaayu/batch.py`
- Test: `tests/test_batch_masking.py`

**Interfaces:**
- Produces:
  - `build_batch_attention_mask(pad_mask: torch.Tensor, q_len: int, dtype: torch.dtype, device) -> torch.Tensor` → additive float mask `[N, 1, q_len, KV]` (0.0 attend, `finfo.min` masked) = causal AND key-not-pad. `pad_mask` is `[N, KV]` with 1=real, 0=pad; queries are the LAST `q_len` positions of KV.
  - `batch_position_ids(pad_mask: torch.Tensor) -> torch.Tensor` → `[N, KV]` long positions = `cumsum(pad_mask)-1` clamped `>= 0` (pads share position 0; real tokens count from 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_masking.py
import torch
from vaayu.batch import build_batch_attention_mask, batch_position_ids

NEG = torch.finfo(torch.float32).min

def test_position_ids_left_pad():
    # stream 0: 1 pad + 3 real ; stream 1: 0 pad + 4 real
    pad = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    pos = batch_position_ids(pad)
    assert pos.tolist() == [[0, 0, 1, 2], [0, 1, 2, 3]]

def test_mask_blocks_pad_keys_and_future():
    pad = torch.tensor([[0, 1, 1]])          # KV=3, first col is pad
    m = build_batch_attention_mask(pad, q_len=3, dtype=torch.float32, device="cpu")
    assert m.shape == (1, 1, 3, 3)
    # query row 1 (KV index 1) may attend key 1 only (key 0 is pad, key 2 is future)
    row1 = m[0, 0, 1]
    assert row1[0] == NEG and row1[1] == 0.0 and row1[2] == NEG

def test_decode_mask_single_query():
    pad = torch.tensor([[0, 1, 1, 1]])       # KV=4
    m = build_batch_attention_mask(pad, q_len=1, dtype=torch.float32, device="cpu")
    assert m.shape == (1, 1, 1, 4)
    # last query attends all real keys, not the pad
    assert m[0, 0, 0].tolist() == [NEG, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_masking.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to vaayu/batch.py
def batch_position_ids(pad_mask: "torch.Tensor") -> "torch.Tensor":
    """[N, KV] long positions from a left-pad mask (1=real, 0=pad)."""
    pos = pad_mask.long().cumsum(-1) - 1
    return pos.clamp_(min=0)


def build_batch_attention_mask(pad_mask, q_len, dtype, device):
    """Additive float mask [N, 1, q_len, KV] = causal AND key-not-pad.
    Queries are the LAST q_len positions of KV (right-aligned decode)."""
    N, KV = pad_mask.shape
    q_pos = torch.arange(KV - q_len, KV, device=device)[:, None]   # [q_len, 1]
    k_pos = torch.arange(KV, device=device)[None, :]               # [1, KV]
    causal = k_pos <= q_pos                                        # [q_len, KV]
    keep = causal[None, None] & pad_mask.bool()[:, None, None, :]  # [N, 1, q_len, KV]
    mask = torch.zeros(N, 1, q_len, KV, dtype=dtype, device=device)
    return mask.masked_fill(~keep, torch.finfo(dtype).min)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_masking.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
git add vaayu/batch.py tests/test_batch_masking.py
git commit -m "feat(batch): pure left-pad attention-mask + position-id builders (Task 4)"
```

---

### Task 5: Gated padding-mask + position_ids + RoPE path in `forward`

**Files:**
- Modify: `vaayu/vaayu_base.py` — `forward` (the sdpa/flash branch around lines 2209-2269).
- Test: `tests/test_batch_masking.py` (regression assertion) + the existing single-stream/MoE suite must stay green.

**Interfaces:**
- Consumes: `build_batch_attention_mask`, `batch_position_ids` (Task 4).
- Produces: `forward` now honors a caller-supplied 2D `attention_mask` (`[N, KV]` padding mask) and 2D `position_ids` (`[N, KV]`) on the sdpa path. When both are `None`, behavior is byte-identical to today (existing callers unaffected).

**Why:** `forward` today (line ~2211) sets `attention_mask = None` for sdpa and (line ~2224) overrides `position_ids` with a shared `arange`. A left-padded batch needs a per-stream padding mask and per-stream positions or it corrupts output (real tokens attend pads; short prompts get wrong RoPE). The RoPE precompute helper `_compute_full_position_embeddings` (line 2017) already handles a batch dim, so batched positions route cleanly through it.

- [ ] **Step 1: Write the failing regression test** (proves the gated branch is wired without a GPU model — uses a tiny stub exercising the mask builder path indirectly is impractical; instead assert the None-path is untouched by re-running an existing fast test, and add a focused source guard test)

```python
# append to tests/test_batch_masking.py
import inspect
import vaayu.vaayu_base as vb

def test_forward_has_gated_caller_mask_branch():
    """Guard: forward must branch on a captured caller mask/position, not
    unconditionally null them. Prevents a regression that silently reverts
    left-pad support."""
    src = inspect.getsource(vb.VaayuBaseModel.forward)
    assert "_caller_attention_mask" in src
    assert "_caller_position_ids" in src
    assert "build_batch_attention_mask" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_masking.py::test_forward_has_gated_caller_mask_branch -v`
Expected: FAIL (`_caller_attention_mask` not in forward source yet).

- [ ] **Step 3: Edit `forward` — capture caller args, gate mask + positions + RoPE**

In `vaayu/vaayu_base.py`, near the top of `forward` (just after the docstring/`_exec_start`), add capture of the caller's values before any reassignment:

```python
        # Feature B: capture caller-supplied batched mask/positions BEFORE the
        # sdpa branch nulls them. When present, forward builds a per-stream 4D
        # padding+causal mask and uses per-stream positions (left-padded batch).
        _caller_attention_mask = attention_mask
        _caller_position_ids = position_ids
```

Replace the sdpa mask block (currently):
```python
        if self._active_attn_impl in ("flash_attention_2", "sdpa"):
            attention_mask = None
        else:
```
with:
```python
        if self._active_attn_impl in ("flash_attention_2", "sdpa"):
            if _caller_attention_mask is not None and _caller_attention_mask.dim() == 2:
                from .batch import build_batch_attention_mask
                _q_len = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 0
                attention_mask = build_batch_attention_mask(
                    _caller_attention_mask, _q_len,
                    dtype=self.running_dtype, device=self.running_device)
            else:
                attention_mask = None
        else:
```

Wrap the `position_ids` arange construction so a caller value wins. Replace:
```python
        if getattr(self, '_position_ids_cache_key', None) != _pos_range:
            self._position_ids_cache = torch.arange(
                _pos_range, dtype=torch.long, device=self.running_device)[None, :]
            self._position_ids_cache_key = _pos_range
        position_ids = self._position_ids_cache
```
with:
```python
        if _caller_position_ids is not None:
            position_ids = _caller_position_ids.to(self.running_device)
        else:
            if getattr(self, '_position_ids_cache_key', None) != _pos_range:
                self._position_ids_cache = torch.arange(
                    _pos_range, dtype=torch.long, device=self.running_device)[None, :]
                self._position_ids_cache_key = _pos_range
            position_ids = self._position_ids_cache
```

Force the batch-capable RoPE path when positions are caller-supplied. Replace the RoPE cache-key guard head:
```python
        _rope_cache_key = (_pos_range, self.running_device, self.running_dtype)
        if getattr(self, '_rope_cache_key', None) != _rope_cache_key:
            if self.get_rotary_emb() is not None:
```
with:
```python
        _rope_cache_key = (_pos_range, self.running_device, self.running_dtype)
        if _caller_position_ids is not None:
            # Per-stream positions: never use the shared-arange cache; compute
            # RoPE directly from [N, S] positions (handles the batch dim natively).
            self._full_position_embeddings = self._compute_full_position_embeddings(position_ids)
            self._rope_cache_key = None
        elif getattr(self, '_rope_cache_key', None) != _rope_cache_key:
            if self.get_rotary_emb() is not None:
```

(The remaining body of that `elif` is unchanged from the original `if`.)

- [ ] **Step 4: Run the guard test + the existing regression suite**

Run: `pytest tests/test_batch_masking.py -v && pytest tests/test_moe_geometry.py tests/test_fp8_kv_cache.py tests/test_layer_major_prefill.py -v`
Expected: guard test PASS; existing tests PASS (None-path unchanged).

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_batch_masking.py
git commit -m "feat(forward): gated caller mask/position/RoPE for batched decode (Task 5)"
```

---

### Task 6: `VaayuBaseModel.batched_generate` decode loop

**Files:**
- Modify: `vaayu/vaayu_base.py` — add method near `speculative_generate` (line ~2734).
- Modify: `vaayu/batch.py` — add `geo_from_model`, `free_vram_budget`, `plan_batch` glue.
- Test: on-device (Task 7); here add the GPU-free glue test in `tests/test_batch_plan.py`.

**Interfaces:**
- Consumes: `BatchRequest`, `size_batch`, `sample_step`, `batch_position_ids`, `build_batch_attention_mask`.
- Produces:
  - `geo_from_model(model) -> tuple[int, int, int, int]` = `(n_layers, n_kv_heads, head_dim, elem_bytes)`, all derived from `model.config` + resolved kv mode.
  - `free_vram_budget(model) -> int` = `mem_get_info` free − 1.5 GB hardware safety margin.
  - `plan_batch(model, requests) -> tuple[list[BatchRequest], list[BatchRequest]]` = `size_batch` fed by the two above.
  - `VaayuBaseModel.batched_generate(self, requests: list[BatchRequest]) -> Iterator[tuple[int, int]]` yielding `(stream_id, token_id)`.

- [ ] **Step 1: Write the failing glue test** (GPU-free — a stub model with a `.config` and `kv_cache_mode`)

```python
# tests/test_batch_plan.py
import types
import pytest
from vaayu.batch import geo_from_model, plan_batch, BatchRequest, kv_bytes_for

def _stub_model(n_layers, n_kv_heads, head_dim, mode, free_bytes):
    cfg = types.SimpleNamespace(
        num_hidden_layers=n_layers, num_key_value_heads=n_kv_heads,
        head_dim=head_dim, hidden_size=head_dim * n_kv_heads,
        num_attention_heads=n_kv_heads)
    m = types.SimpleNamespace(config=cfg, kv_cache_mode=mode, loader_strategy=None,
                              _free_vram_override=free_bytes)
    return m

def test_geo_from_model_derives_shape_and_fp8():
    m = _stub_model(94, 8, 128, "fp8", free_bytes=0)
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(m)
    assert (n_layers, n_kv_heads, head_dim, elem) == (94, 8, 128, 1)

def test_geo_from_model_fp16_elem_bytes():
    m = _stub_model(62, 8, 128, "fp16", free_bytes=0)
    assert geo_from_model(m)[3] == 2

def test_plan_batch_uses_budget(monkeypatch):
    m = _stub_model(94, 8, 128, "fp8", free_bytes=0)
    monkeypatch.setattr("vaayu.batch.free_vram_budget",
                        lambda model: kv_bytes_for(150, 94, 8, 128, 1) * 3)
    reqs = [BatchRequest([0] * 100, 50) for _ in range(8)]
    admitted, deferred = plan_batch(m, reqs)
    assert len(admitted) == 3 and len(deferred) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_plan.py -v`
Expected: FAIL with `ImportError: cannot import name 'geo_from_model'`.

- [ ] **Step 3a: Add glue to `vaayu/batch.py`**

```python
# add to vaayu/batch.py
def _resolve_kv_elem_bytes(model) -> int:
    """1 byte (fp8) or 2 (fp16), mirroring VaayuBaseModel._create_kv_cache.
    Conservative: unknown/auto-non-EXL3 -> 2 (never under-reserves KV)."""
    mode = getattr(model, "kv_cache_mode", "fp16")
    if mode == "auto":
        try:
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "fp8" if isinstance(getattr(model, "loader_strategy", None),
                                       EXL3LoaderStrategy) else "fp16"
        except Exception:
            mode = "fp16"
    return 1 if mode == "fp8" else 2


def geo_from_model(model) -> tuple:
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    return n_layers, n_kv_heads, head_dim, _resolve_kv_elem_bytes(model)


def free_vram_budget(model) -> int:
    """Live free VRAM minus a hardware safety margin (CUDA context + streaming
    double-buffer). 1.5 GB is a HARDWARE constant for this box, not a model number."""
    import torch
    _SAFETY = int(1.5 * 1024 ** 3)   # hardware constant
    free, _ = torch.cuda.mem_get_info(model.device)
    return max(0, free - _SAFETY)


def plan_batch(model, requests):
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(model)
    return size_batch(requests, free_vram_budget(model),
                      n_layers, n_kv_heads, head_dim, elem)
```

- [ ] **Step 3b: Add `batched_generate` to `VaayuBaseModel` in `vaayu/vaayu_base.py`**

```python
    @torch.no_grad()
    def batched_generate(self, requests):
        """Static multi-query batched decode. Yields (stream_id, token_id) live.
        requests: list[vaayu.batch.BatchRequest] (already admitted, or pass raw and
        call plan_batch first). MoE forward is unchanged; batching is this loop
        plus the gated mask/position path in forward (Task 5)."""
        from .batch import (sample_step, batch_position_ids)
        assert self._active_attn_impl == "sdpa", (
            f"batched_generate requires sdpa attention, got {self._active_attn_impl}")
        N = len(requests)
        device = self.running_device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_L = max(len(r.prompt_ids) for r in requests)
        input_ids = torch.full((N, max_L), pad_id, dtype=torch.long, device=device)
        pad_mask = torch.zeros((N, max_L), dtype=torch.long, device=device)
        for i, r in enumerate(requests):
            L = len(r.prompt_ids)
            input_ids[i, max_L - L:] = torch.tensor(r.prompt_ids, device=device)
            pad_mask[i, max_L - L:] = 1
        position_ids = batch_position_ids(pad_mask)

        eos = self.config.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        stop_sets = [set(r.stop_ids) | eos_set for r in requests]
        active = [True] * N
        produced = [0] * N

        try:
            # Prefill — bypass chunked prefill (it assumes batch=1, indexes [0]).
            _saved_chunk = self.prefill_chunk_size
            self.prefill_chunk_size = 0
            try:
                out = self.forward(
                    input_ids=input_ids, attention_mask=pad_mask,
                    position_ids=position_ids, use_cache=True, return_dict=True)
            finally:
                self.prefill_chunk_size = _saved_chunk
            past = out.past_key_values
            logits = out.logits[:, -1, :]           # right edge aligned by left-pad

            next_tokens = torch.empty((N, 1), dtype=torch.long, device=device)
            for i in range(N):
                tok = sample_step(logits[i], requests[i].temperature, requests[i].top_p)
                next_tokens[i, 0] = tok
                yield (i, tok)
                produced[i] = 1
                if tok in stop_sets[i] or produced[i] >= requests[i].max_new_tokens:
                    active[i] = False

            while any(active):
                pad_mask = torch.cat(
                    [pad_mask, torch.ones((N, 1), dtype=torch.long, device=device)], dim=1)
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)
                out = self.forward(
                    input_ids=next_tokens, past_key_values=past,
                    attention_mask=pad_mask, position_ids=position_ids,
                    use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                for i in range(N):
                    if not active[i]:
                        next_tokens[i, 0] = pad_id      # dead slot, masked; static HoL
                        continue
                    tok = sample_step(logits[i], requests[i].temperature, requests[i].top_p)
                    next_tokens[i, 0] = tok
                    yield (i, tok)
                    produced[i] += 1
                    if tok in stop_sets[i] or produced[i] >= requests[i].max_new_tokens:
                        active[i] = False
        finally:
            self.telemetry.dump_json("vaayu_telemetry.json")
```

- [ ] **Step 4: Run the glue test to verify it passes**

Run: `pytest tests/test_batch_plan.py -v`
Expected: PASS (3 cases). (`batched_generate` itself is gated on-device — Task 7.)

- [ ] **Step 5: Commit**

```bash
git add vaayu/batch.py vaayu/vaayu_base.py tests/test_batch_plan.py
git commit -m "feat: batched_generate decode loop + VRAM plan glue (Task 6)"
```

---

### Task 7: On-device correctness — batched == single-stream at temp=0

**Files:**
- Create: `test_batched_decode.py` (repo root, mirrors `test_generate.py` load pattern).

**Interfaces:**
- Consumes: `AutoModel`, `BatchRequest`, `plan_batch`, `batched_generate`.

**This is THE correctness gate.** At temperature 0, batched decode must produce token-for-token identical output to running each prompt single-stream. Requires the loaded model (Qwen3-235B EXL3 per `test_generate.py`), sdpa attention. Not a CI unit test — run on the target box.

- [ ] **Step 1: Write the correctness script**

```python
# test_batched_decode.py
"""On-device: batched decode must equal single-stream at temp=0 (token-for-token)."""
import torch
from vaayu import AutoModel
from vaayu.batch import BatchRequest, plan_batch

MODEL_PATH = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"  # per test_generate.py
DEVICE = "cuda"

model = AutoModel.from_pretrained(MODEL_PATH, device=DEVICE, kv_cache_mode="auto",
                                  prefill_chunk_size=512)
# v1 requires sdpa (4D mask). If this fails, load with attn_implementation="sdpa".
assert model._active_attn_impl == "sdpa", f"need sdpa, got {model._active_attn_impl}"

tok = model.tokenizer
prompts = [
    "Write a Python function that reverses a string.",
    "Explain what a hash map is in two sentences.",
    "def fib(n):",
]
MAX_NEW = 24

def single_stream(prompt_ids):
    """Greedy single-stream reference via batched_generate with one request."""
    out = []
    for _sid, t in model.batched_generate([BatchRequest(prompt_ids, MAX_NEW)]):
        out.append(t)
    return out

req_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
reference = [single_stream(ids) for ids in req_ids]

admitted, deferred = plan_batch(model, [BatchRequest(ids, MAX_NEW) for ids in req_ids])
assert not deferred, "test prompts should all fit; shorten them if deferred"
batched = {i: [] for i in range(len(admitted))}
for sid, t in model.batched_generate(admitted):
    batched[sid].append(t)

# admitted is sorted by length; map back by matching prompt_ids
for i, r in enumerate(admitted):
    ref = reference[req_ids.index(r.prompt_ids)]
    got = batched[i]
    assert got == ref, f"stream {i} diverged:\n ref={ref}\n got={got}"
print("PASS: batched == single-stream token-for-token")
```

- [ ] **Step 2: Run it on the target box**

Run: `python test_batched_decode.py`
Expected: `PASS: batched == single-stream token-for-token`.

- [ ] **Step 3: If it diverges — debug (do NOT weaken the assert)**

Likely culprits, in order: (a) sdpa not active → fix `attn_implementation`; (b) mask sign/shape — print `build_batch_attention_mask` for a 2-stream toy and confirm pads are `finfo.min`; (c) position_ids off-by-one — print `batch_position_ids` vs expected cumsum; (d) chunked-prefill not bypassed — confirm `prefill_chunk_size` is 0 during the prefill forward. Use `superpowers:systematic-debugging`.

- [ ] **Step 4: Commit**

```bash
git add test_batched_decode.py
git commit -m "test: on-device batched==single-stream correctness gate (Task 7)"
```

---

### Task 8: Cross-stream overlap telemetry

**Files:**
- Modify: `vaayu/moe_patch.py` — one gated line in the sparse-forward closures.
- Modify: `vaayu/telemetry.py` — the `VaayuTelemetry` class (this is `self.telemetry`, constructed at `vaayu_base.py:565`) — add `record_batch_overlap`.
- Test: `tests/test_batch_overlap.py`.

**Interfaces:**
- Produces: `Telemetry.record_batch_overlap(layer_idx: int, union_size: int, max_possible: int) -> None`. `max_possible = N * top_k`. Ratio `union_size / max_possible` quantifies overlap (→1 divergent = small win; →`top_k/max` high overlap = near-N× disk amortization).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_overlap.py
from vaayu.telemetry import VaayuTelemetry

def test_record_batch_overlap_accumulates():
    t = VaayuTelemetry()
    t.record_batch_overlap(layer_idx=0, union_size=12, max_possible=16)
    t.record_batch_overlap(layer_idx=1, union_size=8, max_possible=16)
    stats = t.batch_overlap_summary()
    assert stats["layers"] == 2
    assert abs(stats["mean_ratio"] - ((12/16 + 8/16) / 2)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_overlap.py -v`
Expected: FAIL (`record_batch_overlap` missing).

- [ ] **Step 3: Add the telemetry methods + gated hook**

In the telemetry class:
```python
    def record_batch_overlap(self, layer_idx, union_size, max_possible):
        if not hasattr(self, "_batch_overlap"):
            self._batch_overlap = []
        self._batch_overlap.append((layer_idx, union_size, max_possible))

    def batch_overlap_summary(self):
        rows = getattr(self, "_batch_overlap", [])
        ratios = [u / m for _, u, m in rows if m]
        return {"layers": len(rows),
                "mean_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0}
```

In each `sparse_forward` in `vaayu/moe_patch.py`, right after `unique_experts` is computed, add the gated hook (default off → MoE behavior unchanged):
```python
        if getattr(vaayu_ref, "_batch_n", 1) > 1:
            vaayu_ref.telemetry.record_batch_overlap(
                layer_idx, len(unique_experts), vaayu_ref._batch_n * top_k)
```

In `batched_generate` (Task 6), set the flag around the loop: `self._batch_n = N` after computing `N`, and `self._batch_n = 1` in the `finally` block.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_overlap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py vaayu/telemetry.py vaayu/vaayu_base.py tests/test_batch_overlap.py
git commit -m "feat(telemetry): gated cross-stream expert-overlap metric (Task 8)"
```

---

### Task 9: Dense smoke + MoE regression + throughput verification

**Files:**
- Create: `test_batched_dense_smoke.py` (repo root).
- Doc: append a results section to `docs/superpowers/plans/2026-07-18-vaayu-batched-decode-RESULTS.md`.

**Interfaces:**
- Consumes: `AutoModel`, `BatchRequest`, `batched_generate`.

- [ ] **Step 1: Write the dense smoke script**

```python
# test_batched_dense_smoke.py
"""Dense (non-MoE) model must batch-decode without hitting any MoE path.
Usage: DENSE_CKPT=/path/to/small/dense/checkpoint python test_batched_dense_smoke.py
(same convention as the existing dense smoke test — a 1-8B dense model in Vaayu's
per-layer-safetensors layout.)"""
import os
import torch
from vaayu import AutoModel
from vaayu.batch import BatchRequest

DENSE_MODEL_PATH = os.environ["DENSE_CKPT"]
model = AutoModel.from_pretrained(DENSE_MODEL_PATH, device="cuda", kv_cache_mode="auto")
assert model._active_attn_impl == "sdpa"
assert not getattr(model, "is_moe", False), "expected a DENSE model for this smoke test"

tok = model.tokenizer
reqs = [BatchRequest(tok("Hello, ", add_special_tokens=True)["input_ids"], 8),
        BatchRequest(tok("The capital of France is ", add_special_tokens=True)["input_ids"], 8)]
seen = {0: 0, 1: 0}
for sid, t in model.batched_generate(reqs):
    seen[sid] += 1
assert seen[0] > 0 and seen[1] > 0
print("PASS: dense batched decode produced tokens for both streams")
```

- [ ] **Step 2: Run dense smoke + full MoE regression + single-stream sanity**

Run:
```bash
python test_batched_dense_smoke.py
pytest tests/ -v          # MoE geometry, KV, prefill, streaming — all must stay green
python test_generate.py   # single-stream still works (None-path unchanged)
```
Expected: dense smoke PASS; `pytest tests/` all green; `test_generate.py` generates as before.

- [ ] **Step 3: Measure aggregate throughput (the win)**

Run `test_batched_decode.py` (Task 7) with N divergent prompts and time it; compare aggregate tok/s to N sequential single-stream runs. Record `model.telemetry.batch_overlap_summary()` (mean overlap ratio). Success = aggregate tok/s **>** N × single-stream-share; magnitude tracks the overlap ratio.

- [ ] **Step 4: Write the results doc**

Record: N used, per-request lengths, aggregate vs sequential tok/s, mean overlap ratio, any VRAM headroom / N-collapse warnings observed. Note whether the win matched the overlap prediction.

- [ ] **Step 5: Commit**

```bash
git add test_batched_dense_smoke.py docs/superpowers/plans/2026-07-18-vaayu-batched-decode-RESULTS.md
git commit -m "test+docs: dense smoke, MoE regression, throughput results (Task 9)"
```

---

## Out of scope for v1 (do NOT build — documented in the spec §10)

Continuous batching / scheduler, paged KV, Feature-A shared-trunk composition, flash_attention_2 varlen support, eliding pad-token routing through MoE during prefill. These are v2.
