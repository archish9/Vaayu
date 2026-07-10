# Fix #8: Replace O(seq²) FP8 KV Cache Concatenation with a Pre-Allocated Ring Buffer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first.** Independent of all other fixes.

**Goal:** Eliminate the quadratic-time growth of `FP8DynamicCache`. Every decode
token currently `torch.cat`s the *entire* accumulated K/V cache (six tensors) —
O(seq) copy per layer per token = O(seq²) total. Replace with pre-allocated
buffers written in place (geometric doubling on overflow) so appends are
amortized O(1) and total copy work is O(seq).

**Architecture:** `FP8DynamicCache.update()` maintains six per-layer lists
(`key_cache`, `value_cache`, `_key_scales`, `_value_scales`, `_k_out_cache`,
`_v_out_cache`) and grows each by `torch.cat([existing, new], dim=-2)` every
token. Concatenation allocates a new tensor and copies all prior tokens — the
quadratic cost. This fix stores each as a pre-allocated buffer with a capacity ≥
current length, writes new tokens into the free slots in place, and reallocates
(doubling capacity, one copy) only when full. `get_seq_length` and the returned
`(k_out, v_out)` views are preserved exactly, so attention sees identical data.

**Tech Stack:** Python 3.12, PyTorch. Files: `vaayu/vaayu_base.py`
(`FP8DynamicCache`), `tests/`.

## Global Constraints

- **MODEL-AGNOSTIC (CLAUDE.md §5).** The cache infers all shapes (batch, heads,
  head_dim) from the first `update()`'s tensors. No layer count, head count, or
  head_dim is hardcoded. Works for Qwen (GQA) and MiniMax-M2.5 (GQA) identically.
- Behavior-preserving: `update()` must return the same `(k_out, v_out)` tensors
  (same values, same shape `[B, H, seq, D]`) as today. Attention output must be
  bit-identical for a given input.
- Memory profile unchanged (still fp8 + a dequantized out-buffer). This fix targets
  *time*, not memory. Do not change what is stored, only how it grows.
- Preserve `get_seq_length()` semantics and any `Cache` base-class contract used by
  `transformers` generation.
- Thread safety is not required here (KV update runs on the main forward thread).

---

## Background: exact current code

`vaayu/vaayu_base.py`, class `FP8DynamicCache`. Locate via
`grep -n "class FP8DynamicCache" vaayu/vaayu_base.py`. Current `update()`:

```python
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_new_out = k_fp8.to(key_states.dtype) * k_scale
        v_new_out = v_fp8.to(value_states.dtype) * v_scale
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_fp8], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_fp8], dim=-2)
            self._key_scales[layer_idx] = torch.cat([self._key_scales[layer_idx], k_scale], dim=-2)
            self._value_scales[layer_idx] = torch.cat([self._value_scales[layer_idx], v_scale], dim=-2)
            self._k_out_cache[layer_idx] = torch.cat([self._k_out_cache[layer_idx], k_new_out], dim=-2)
            self._v_out_cache[layer_idx] = torch.cat([self._v_out_cache[layer_idx], v_new_out], dim=-2)
        else:
            self.key_cache.append(k_fp8)
            self.value_cache.append(v_fp8)
            self._key_scales.append(k_scale)
            self._value_scales.append(v_scale)
            self._k_out_cache.append(k_new_out)
            self._v_out_cache.append(v_new_out)
        return self._k_out_cache[layer_idx], self._v_out_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]
```

The six `torch.cat` calls on the `dim=-2` (sequence) axis are the O(seq²) source.

---

## Task 1: Add a length-tracked, capacity-doubling ring buffer to `FP8DynamicCache`

**Files:**
- Modify: `vaayu/vaayu_base.py` (`FP8DynamicCache.__init__`, `update`, `get_seq_length`)
- Test: `tests/test_fp8_kv_ring.py` (create)

**Interfaces:**
- Preserved public surface: `update(key_states, value_states, layer_idx,
  cache_kwargs=None) -> (k_out, v_out)`; `get_seq_length(layer_idx=0) -> int`.
- New internal state: `self._seq_len: list[int]` (per-layer used length);
  buffers replace the six append-only lists but hold **capacity ≥ seq_len** and are
  sliced to `:seq_len` on return.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fp8_kv_ring.py`. It asserts the ring buffer returns identical
values to a reference cat-based accumulation, and that identity holds across a
capacity-doubling boundary:

```python
"""Fix #8: FP8 KV ring buffer must match cat-based accumulation, without O(n^2)."""
import torch
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="needs FP8 support"
)


def _ref_quant(k):
    scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    fp8 = (k / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.to(k.dtype) * scale


def test_ring_matches_reference_accumulation():
    from vaayu.vaayu_base import FP8DynamicCache
    cache = FP8DynamicCache()
    B, H, D = 1, 4, 8
    torch.manual_seed(0)
    ref_k = []
    # 20 single-token updates crossing the initial capacity (forces a doubling)
    for step in range(20):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        ref_k.append(_ref_quant(k))
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == step + 1
        assert k_out.shape == (B, H, step + 1, D)
    expected = torch.cat(ref_k, dim=-2)
    # Final returned k_out must equal the reference dequantized accumulation.
    torch.testing.assert_close(k_out, expected, rtol=0, atol=0)


def test_multi_token_prefill_then_decode():
    from vaayu.vaayu_base import FP8DynamicCache
    cache = FP8DynamicCache()
    B, H, D = 1, 4, 8
    k0 = torch.randn(B, H, 5, D); v0 = torch.randn(B, H, 5, D)  # prefill 5 tokens
    cache.update(k0, v0, layer_idx=0)
    assert cache.get_seq_length(0) == 5
    k1 = torch.randn(B, H, 1, D); v1 = torch.randn(B, H, 1, D)  # decode 1
    k_out, _ = cache.update(k1, v1, layer_idx=0)
    assert cache.get_seq_length(0) == 6
    assert k_out.shape == (B, H, 6, D)
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_fp8_kv_ring.py -v`
Expected: FAIL (current `update` works via cat but `test_ring_matches...` may pass
by luck; the point is to lock behavior before the rewrite). If both PASS on the
current code, that is fine — they are the regression guard for Step 3.

- [ ] **Step 3: Rewrite `__init__`, `update`, `get_seq_length` to use buffers**

Replace the class body's `__init__`/`update`/`get_seq_length` (keep the docstring).
New implementation:

```python
    def __init__(self):
        super().__init__()
        # Capacity buffers (allocated lazily on first update per layer). Each holds
        # >= _seq_len[layer] valid tokens on the dim=-2 (sequence) axis; the tail is
        # unused capacity. Fix #8: replaces per-token torch.cat (O(seq^2)) with
        # in-place writes + geometric doubling (amortized O(1) append).
        self.key_cache: list = []       # fp8 K, shape [B, H, capacity, D]
        self.value_cache: list = []     # fp8 V
        self._key_scales: list = []     # [B, H, capacity, 1]
        self._value_scales: list = []
        self._k_out_cache: list = []    # dequantized K, [B, H, capacity, D]
        self._v_out_cache: list = []
        self._seq_len: list = []        # per-layer used length

    _GROWTH = 2  # capacity doubling factor (amortized O(1) append)

    def _ensure_capacity(self, layer_idx, needed, template_kv, template_scale):
        """Grow layer_idx's buffers so capacity >= needed. Copies existing valid
        tokens once on growth. template_* provide dtype/device/leading dims."""
        cap = self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0
        if layer_idx < len(self.key_cache) and cap >= needed:
            return
        new_cap = max(needed, max(1, cap) * self._GROWTH)
        B, H, _, D = template_kv.shape
        dev, kv_dtype = template_kv.device, template_kv.dtype
        out_dtype = template_scale.dtype  # dequantized output dtype (fp16/bf16)

        def _grow(old_list, last_dim, dtype):
            buf = torch.zeros((B, H, new_cap, last_dim), dtype=dtype, device=dev)
            if layer_idx < len(old_list):
                used = self._seq_len[layer_idx]
                buf[..., :used, :] = old_list[layer_idx][..., :used, :]
                old_list[layer_idx] = buf
            else:
                old_list.append(buf)

        _grow(self.key_cache,     D, kv_dtype)
        _grow(self.value_cache,   D, kv_dtype)
        _grow(self._key_scales,   1, template_scale.dtype)
        _grow(self._value_scales, 1, template_scale.dtype)
        _grow(self._k_out_cache,  D, out_dtype)
        _grow(self._v_out_cache,  D, out_dtype)
        if layer_idx >= len(self._seq_len):
            self._seq_len.append(0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_new_out = k_fp8.to(key_states.dtype) * k_scale
        v_new_out = v_fp8.to(value_states.dtype) * v_scale

        n_new = key_states.shape[-2]
        prev = self._seq_len[layer_idx] if layer_idx < len(self._seq_len) else 0
        self._ensure_capacity(layer_idx, prev + n_new, k_fp8, k_scale)

        s = slice(prev, prev + n_new)
        self.key_cache[layer_idx][..., s, :] = k_fp8
        self.value_cache[layer_idx][..., s, :] = v_fp8
        self._key_scales[layer_idx][..., s, :] = k_scale
        self._value_scales[layer_idx][..., s, :] = v_scale
        self._k_out_cache[layer_idx][..., s, :] = k_new_out
        self._v_out_cache[layer_idx][..., s, :] = v_new_out
        self._seq_len[layer_idx] = prev + n_new

        used = self._seq_len[layer_idx]
        return (self._k_out_cache[layer_idx][..., :used, :],
                self._v_out_cache[layer_idx][..., :used, :])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._seq_len):
            return 0
        return self._seq_len[layer_idx]
```

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_fp8_kv_ring.py -v`
Expected: PASS both. If `assert_close` fails at atol=0, the in-place write path
diverged from the reference — check the `slice` indexing and that `k_new_out`
dtype matches `template_scale.dtype`.

- [ ] **Step 5: Run the existing FP8 KV test + full suite**

Run:
```bash
source venv/bin/activate && python -m pytest tests/test_fp8_kv_cache.py tests/ -q
```
Expected: PASS. The pre-existing `tests/test_fp8_kv_cache.py` must still pass —
it locks the public behavior.

- [ ] **Step 6: Commit**

```bash
git add tests/test_fp8_kv_ring.py vaayu/vaayu_base.py
git commit -m "perf: FP8 KV cache uses O(seq) ring buffer instead of O(seq^2) cat (Fix #8)"
```

---

## Task 2: Handle KV-cache truncation/crop if the generation loop uses it

**Files:**
- Inspect: `vaayu/vaayu_base.py` (grep for cache truncation)
- Modify: `FP8DynamicCache` only if a crop/reorder method is actually called

**Interfaces:**
- If generation crops the cache (sliding window / max length), the buffer's
  `_seq_len` must be updated, not the tensors re-cat.

- [ ] **Step 1: Find any crop/reorder/truncation on the cache**

```bash
grep -n "Cache truncation\|crop\|reorder_cache\|\.key_cache\[" vaayu/vaayu_base.py
```

- [ ] **Step 2: If a truncation path mutates `key_cache[layer]` by slicing**

If code does `cache.key_cache[i] = cache.key_cache[i][..., :n, :]` (or similar),
replace that pattern with setting the length only:
```python
        cache._seq_len[i] = min(cache._seq_len[i], n)
```
because the ring buffer keeps capacity; truncation is just a length change. Add a
helper on the class if the pattern appears in multiple places:
```python
    def crop(self, max_length: int) -> None:
        """Truncate every layer to max_length tokens (Fix #8: length-only, no copy)."""
        for i in range(len(self._seq_len)):
            self._seq_len[i] = min(self._seq_len[i], max_length)
```
and call `cache.crop(n)` at the truncation site.

> **CONFIRMED (2026-07-10 review): truncation EXISTS and this task is NOT a
> no-op.** `vaayu_base.py` has `_trim_kv_cache(cache, target_seq_len)` (tail-trim,
> called EVERY speculative-decode iteration, Step 5 of `speculative_generate`) and
> `_trim_kv_cache_front(cache, target_seq_len)` (front-trim on OOM pressure). Both
> have `isinstance(cache, FP8DynamicCache)` branches that slice all six tensor
> lists directly — under the ring buffer these slices would desync capacity from
> `_seq_len`. Rewrite the FP8 branch of `_trim_kv_cache` as a length-only update
> (`cache._seq_len[i] = min(cache._seq_len[i], target_seq_len)` via the `crop()`
> helper). `_trim_kv_cache_front` removes OLDEST tokens — with a ring buffer that
> requires shifting valid data left (one copy) before adjusting `_seq_len`; it is
> rare (OOM path only), so a simple copy-shift is fine.

- [ ] **Step 3: Run full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit (only if code changed)**

```bash
git add vaayu/vaayu_base.py
git commit -m "fix: FP8 KV crop is length-only under ring buffer (Fix #8)"
```

---

## Verification (real run — the win shows on LONG generations)

- [ ] **Step 1: Run with a longer output to expose the quadratic difference**

The O(seq²) cost is invisible at ~24 tokens. Temporarily raise `MAX_NEW_TOKENS`
in `test_generate.py` to 256 for this check (or use an existing long run), then:

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "token summary:" run.log | tail -20
```
Success criterion: per-token `total` time should NOT grow with token index due to
KV cat anymore (before this fix, later tokens paid more for the growing cat). The
`other`/`compute` bucket per token stays roughly flat across the run instead of
creeping up. (Expert streaming still dominates absolute time; this removes the
compounding KV overhead on top of it.)

- [ ] **Step 2: Confirm output unchanged**

```bash
grep -A5 "Output:" run.log | head
```
Expected: coherent output, same class of result as before. If garbage, the
in-place write diverged — revisit Task 1 Step 3.

## Gotchas

- **Bit-exactness matters.** The returned `(k_out, v_out)` must equal the old
  cat-based values exactly (atol=0), because attention consumes them. The test
  enforces this. The only behavior change is *how* the buffer grows.
- `_ensure_capacity` copies valid tokens on each doubling — that is O(n) total
  across the whole sequence (geometric series), not O(n²). Do not "optimize" it to
  copy every token.
- Leading dims (B, H) and head_dim D are read from the incoming tensors — model
  agnostic. Do not hardcode them. GQA models (Qwen, M2.5) have H = num_kv_heads
  here; the code does not care.
- Prefill passes `n_new > 1` (multiple tokens at once); decode passes `n_new == 1`
  (or > 1 under speculative verification). The `slice(prev, prev+n_new)` handles
  both. Test `test_multi_token_prefill_then_decode` covers this.
- Unrelated cleanup NOT in scope: `_check_pcie_topology(model_size_gb=132.0)` in
  this file hardcodes a model size for a startup warning. It is cosmetic; note it
  for a separate model-agnostic pass, do not fix it here.
