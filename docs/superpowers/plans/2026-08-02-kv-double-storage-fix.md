# KV Double-Storage Fix — Implementation Plan (Plan E)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything you need —
> the measurement, the anchors, the reasoning — is in this document. You do not need to read any
> other plan.

**Goal:** Stop Vaayu's KV cache from storing every key/value twice. Today it holds an fp8 copy
AND a full-size dequantized bf16 copy of the same data; the dequantized copies are **66% of the
KV footprint**. Add a bf16-only ring-buffer mode that keeps the amortized-O(1) append but drops
the redundant fp8 store and the scales.

**Architecture:** `FP8DynamicCache` (`vaayu/vaayu_base.py`) gains a `quantize: bool = True`
constructor flag. With `quantize=False` it allocates only `key_cache`/`value_cache` in bf16, skips
the fp8 store, the scale buffers, and the `_k_out_cache`/`_v_out_cache` buffers entirely, and
`update()` returns slices of `key_cache` directly. All ring-buffer, crop, `to_cpu`/`to_device`,
and budget-accounting logic is reused unchanged. A new `kv_cache_mode="bf16"` selects it.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

---

## Why this exists — the measurement (do not re-derive, but you may re-run it)

`FP8DynamicCache` allocates **six** per-layer buffers (`vaayu/vaayu_base.py`, `_BUFFER_LISTS`):

```
key_cache      fp8 K            value_cache    fp8 V
_key_scales    [B,H,cap,1]      _value_scales  [B,H,cap,1]
_k_out_cache   DEQUANTIZED K    _v_out_cache   DEQUANTIZED V
```

`update()` writes both the fp8 bytes and the dequantized result, then **returns slices of
`_k_out_cache`** — the fp8 copy is never read by attention. Measured, B=1 H=8 S=2048 D=128, one
layer:

```
key_cache      2.10 MB  ┐ fp8 store        4.20 MB  (33%)
value_cache    2.10 MB  ┘
_key_scales    0.03 MB
_value_scales  0.03 MB
_k_out_cache   4.19 MB  ┐ dequantized      8.38 MB  (66%)
_v_out_cache   4.19 MB  ┘
TOTAL         12.65 MB     = 3.016 bytes/element
```

Plain fp16 `DynamicCache` is 2.0 bytes/element. **So the current fp8 mode costs ~1.5× more VRAM
than fp16, not 2× less** — and `kv_cache_mode="auto"` selects fp8 for every EXL3 model, which is
both on-device targets.

What fp8 mode genuinely provides is **not** compression — it is the ring buffer (geometric
doubling, amortized O(1) append). Plain `DynamicCache` does per-step `torch.cat` (O(seq²)). So
the fix is *not* "switch to fp16"; it is "keep the ring buffer, store bf16 only."

| Configuration | bytes/elem | KV at the measured 1.09 GB | Frees |
|---|---|---|---|
| Today (fp8 + dequantized) | 3.016 | 1.09 GB | — |
| **bf16-only ring buffer (this plan)** | **2.00** | **0.72 GB** | **~366 MB (33.6%)** |
| fp8-only, dequantize on read | 1.02 | 0.37 GB | ~723 MB, but adds hot-path compute |

This plan builds the middle row: it frees ~33% of KV VRAM, **removes** quantization work rather
than adding any, and is **lossless** (fp8 is a lossy format; bf16 storage is not). The bottom row
is deliberately NOT built — dequantizing on every read costs O(seq_len) casts per layer per decode
step, and decode is already ~60-70% compute-bound on this hardware.

**Evidence this is safe:** `vaayu/kv_persist.py` already treats the dequantized buffers as
derived, not source-of-truth — it serializes only the fp8 store and the scales
(`kv_persist.py:118-121`), then **rebuilds** `_k_out_cache`/`_v_out_cache` by dequantizing on
restore (`kv_persist.py:153-155`).

---

## Global Constraints

Every task implicitly includes these.

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No layer count, head count, expert count, or
  hidden-size literal may enter the code. Shapes come from the tensors passed in.
- **Both paths must keep working.** `quantize=True` (today's fp8 behavior) must remain
  byte-identical. This plan ADDS a mode; it does not replace one. All existing fp8 tests must pass
  untouched.
- **Do not regress decode.** Floors on the 16 GB box: MiniMax ≈ **1.47 tok/s**, Qwen ≈ **0.82
  tok/s** (both measured 2026-08-01). Task 5 gates this on-device.
- **Surgical (CLAUDE.md §3).** Every changed line traces to this plan. Do not refactor the ring
  buffer, `crop`, `crop_front`, or the prefix-cache/persistence layers beyond what the new mode
  requires.
- Run tests with `source venv/bin/activate`; `python -m pytest tests/ -q` must stay green.
  **Record the baseline count before you start** (`python -m pytest tests/ -q | tail -1`) — this
  document does not hardcode it, because it drifts.

## Files you will touch

- **Modify `vaayu/vaayu_base.py`** — `FP8DynamicCache.__init__`, `_ensure_capacity`, `update`,
  `_BUFFER_LISTS`, and `_create_kv_cache` (mode selection).
- **Modify `vaayu/kv_cache_manager.py`** — `_extract_kv_as_bf16` branches on
  `isinstance(cache, FP8DynamicCache)` and unconditionally multiplies by `_key_scales`. That is
  wrong for an unquantized instance of the same class. **This is the sharpest trap in this plan.**
- **Modify `vaayu/kv_persist.py`** — the on-disk format records a `QUANT_SIG` derived from the
  quantization scheme string. An unquantized cache needs its own signature so old files are
  rejected rather than misread.
- **Modify/extend tests** — `tests/test_fp8_kv_cache.py`, `tests/test_fp8_kv_ring.py`,
  `tests/test_kv_cache_manager_fp8.py`, `tests/test_fp8_prefix_cache_moves.py`,
  `tests/test_kv_persist_*.py`.

---

## Task 1: `quantize=False` mode on the cache (TDD)

**Files:** Modify `vaayu/vaayu_base.py`; test in `tests/test_fp8_kv_cache.py`.

**Interfaces produced (later tasks depend on these exact names):**
- `FP8DynamicCache(quantize: bool = True)`
- `.quantize` — bool attribute
- `_BUFFER_LISTS` becomes an **instance property**, not a class constant: it must return only the
  buffers that actually exist for this instance.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fp8_kv_cache.py`. Read the file first and match its existing style.

```python
import torch
from vaayu.vaayu_base import FP8DynamicCache


def _kv(B=1, H=4, S=8, D=16, dt=torch.bfloat16):
    return torch.randn(B, H, S, D, dtype=dt), torch.randn(B, H, S, D, dtype=dt)


def test_unquantized_mode_allocates_no_fp8_or_scale_buffers():
    c = FP8DynamicCache(quantize=False)
    k, v = _kv()
    c.update(k, v, 0)
    assert c.quantize is False
    assert c.key_cache[0].dtype == k.dtype          # stored in the source dtype
    assert c._key_scales == []                      # no scales allocated
    assert c._value_scales == []
    assert c._k_out_cache == []                     # no dequantized mirror
    assert c._v_out_cache == []


def test_unquantized_mode_is_lossless():
    """fp8 is lossy; bf16 storage is not. Round-trip must be exact."""
    c = FP8DynamicCache(quantize=False)
    k, v = _kv()
    ko, vo = c.update(k, v, 0)
    assert torch.equal(ko, k)
    assert torch.equal(vo, v)


def test_unquantized_mode_is_two_bytes_per_element():
    c = FP8DynamicCache(quantize=False)
    k, v = _kv(S=8)
    c.update(k, v, 0)
    elems = k.numel()
    assert c.resident_bytes() == 2 * elems * 2      # K and V, 2 bytes each


def test_quantized_mode_is_unchanged():
    """Regression guard: the fp8 path must keep all six buffers."""
    c = FP8DynamicCache()                            # default quantize=True
    k, v = _kv()
    c.update(k, v, 0)
    assert c.quantize is True
    assert c.key_cache[0].dtype == torch.float8_e4m3fn
    assert len(c._key_scales) == 1 and len(c._k_out_cache) == 1


def test_unquantized_mode_appends_across_steps():
    """The ring buffer must still work — that is the whole reason this class exists."""
    c = FP8DynamicCache(quantize=False)
    k1, v1 = _kv(S=3)
    k2, v2 = _kv(S=1)
    c.update(k1, v1, 0)
    ko, _ = c.update(k2, v2, 0)
    assert c.get_seq_length(0) == 4
    assert ko.shape[-2] == 4
    assert torch.equal(ko[..., :3, :], k1)
    assert torch.equal(ko[..., 3:, :], k2)


def test_unquantized_mode_crop_and_to_cpu():
    c = FP8DynamicCache(quantize=False)
    k, v = _kv(S=6)
    c.update(k, v, 0)
    c.crop(4)
    assert c.get_seq_length(0) == 4
    c.to_cpu()
    assert c.key_cache[0].device.type == "cpu"
    assert c.key_cache[0].shape[-2] == 4              # compacted to used length
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
source venv/bin/activate
python -m pytest tests/test_fp8_kv_cache.py -v
```

Expected: the `quantize=False` tests fail on the unexpected constructor kwarg; the
`test_quantized_mode_is_unchanged` test passes already.

- [ ] **Step 3: Implement the mode**

In `vaayu/vaayu_base.py`, `FP8DynamicCache`:

1. Add the flag to `__init__` and record it. Keep every existing buffer list declared (they stay
   empty in unquantized mode, which keeps `crop`/`to_cpu` loops uniform):

```python
    def __init__(self, quantize: bool = True):
        super().__init__()
        # quantize=False: store K/V directly in their source dtype (bf16/fp16) and
        # skip both the fp8 copy and the dequantized mirror. Measured 2.0 vs 3.016
        # bytes/element — the fp8 store buys nothing once a full-precision copy has
        # to exist for attention anyway. What this class actually provides is the
        # ring buffer (amortized O(1) append), and that is mode-independent.
        self.quantize = bool(quantize)
```

2. `_BUFFER_LISTS` is currently a class constant. Make it an instance property so every consumer
   (`to_cpu`, `to_device`, `resident_bytes`, and anything else iterating it) sees only the buffers
   that exist:

```python
    @property
    def _BUFFER_LISTS(self):
        if self.quantize:
            return ("key_cache", "value_cache", "_key_scales",
                    "_value_scales", "_k_out_cache", "_v_out_cache")
        return ("key_cache", "value_cache")
```

Grep for every use of `_BUFFER_LISTS` before you change it and confirm each still behaves.

3. `_ensure_capacity`: allocate only what the mode needs. In unquantized mode `template_kv` is the
   raw K tensor, so its dtype is the storage dtype and there is no separate scale dtype.

4. `update()`: in unquantized mode write `key_states`/`value_states` straight into
   `key_cache`/`value_cache` and return slices of those, skipping quantization entirely.

Keep the quantized branch byte-identical to what is there today.

- [ ] **Step 4: Confirm the tests pass, then the whole suite**

```bash
source venv/bin/activate
python -m pytest tests/test_fp8_kv_cache.py tests/test_fp8_kv_ring.py -v
python -m pytest tests/ -q
```

The suite must be green at your recorded baseline + the new tests.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_fp8_kv_cache.py
git commit -m "feat(kv): bf16-only ring-buffer mode on FP8DynamicCache

The fp8 path stores every K/V twice: an fp8 copy AND a full-size dequantized
mirror that update() actually returns. Measured 3.016 bytes/element vs 2.0 for
plain fp16 — the dequantized buffers are 66% of the footprint, so 'fp8 KV cache'
costs ~1.5x more VRAM than no quantization at all.

quantize=False keeps the ring buffer (the real reason this class exists) and
stores bf16 directly: 2.0 bytes/element, lossless, and it removes quantization
work rather than adding any. The fp8 path is untouched and still the default.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Fix the `kv_cache_manager` scale assumption (TDD)

**This is the trap.** `vaayu/kv_cache_manager.py`'s `_extract_kv_as_bf16` does:

```python
if isinstance(cache, FP8DynamicCache):
    ks = cache._key_scales[layer_idx]      # IndexError when quantize=False
    k = k.to(torch.bfloat16) * ks
```

An unquantized instance **is** an `FP8DynamicCache`, so this branch fires and either raises
`IndexError` on the empty scale list or silently corrupts data. It must key off `cache.quantize`,
not the class.

**Files:** Modify `vaayu/kv_cache_manager.py`; test in `tests/test_kv_cache_manager_fp8.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_extract_handles_unquantized_cache():
    """An unquantized FP8DynamicCache has no scales — extraction must not
    assume they exist just because the class matches."""
    import torch
    from vaayu.vaayu_base import FP8DynamicCache
    from vaayu.kv_cache_manager import _extract_kv_as_bf16

    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    c.update(k, v, 0)
    ko, vo = _extract_kv_as_bf16(c, 0)
    assert ko.dtype == torch.bfloat16
    assert torch.equal(ko, k.cpu())      # lossless, no scale multiply
    assert torch.equal(vo, v.cpu())
```

- [ ] **Step 2: Run it, confirm it fails** (expect `IndexError` on the empty scale list).

- [ ] **Step 3: Fix the branch** — change the condition from `isinstance(cache, FP8DynamicCache)`
  to also require `getattr(cache, "quantize", True)`. Add a short comment saying why the class
  check alone is insufficient.

- [ ] **Step 4: Re-run** `tests/test_kv_cache_manager_fp8.py` and the full suite.

- [ ] **Step 5: Grep for other class-based assumptions.**

```bash
grep -rn "FP8DynamicCache" vaayu/ --include=*.py
```

Every `isinstance(..., FP8DynamicCache)` that then touches `_key_scales`, `_k_out_cache`, or
assumes fp8 dtype is the same bug. Fix each the same way, or state in your report that you checked
and no others exist.

- [ ] **Step 6: Commit** (message: `fix(kv): key dequantization off cache.quantize, not the class`).

---

## Task 3: Persistence signature for the unquantized format

`vaayu/kv_persist.py:19-20` derives `QUANT_SIG` from a scheme string:

```python
_QUANT_SIG_UNSIGNED = zlib.crc32(b"fp8_e4m3fn:448.0:per-head-per-token") & 0xFFFFFFFF
```

and `:410` rejects any file whose `quant_sig` differs. Serialization writes the fp8 store plus
scales (`:118-121`) and restore rebuilds the dequantized mirrors (`:153-155`).

An unquantized cache has no scales and a different storage dtype, so it needs its own signature —
otherwise a bf16 file could be read back as fp8 and silently produce garbage.

**Files:** Modify `vaayu/kv_persist.py`; test in `tests/test_kv_persist_format.py`.

- [ ] **Step 1: Decide and record the scope.** Two acceptable options — pick one, state which in
  your report, and implement only that one:
  - **(a) Signature-only (recommended, smaller).** Give the unquantized scheme its own
    `QUANT_SIG` (e.g. `crc32(b"bf16:none:ring")`), and have the writer refuse to persist an
    unquantized cache for now (log once, skip). Old fp8 files keep working. Disk persistence for
    bf16 caches lands later if wanted.
  - **(b) Full support.** Serialize the bf16 buffers, restore without a dequantization step, and
    round-trip test it.

  Option (a) is the honest minimum: the engine's KV disk persistence already ships dark
  (`enable_kv_persist` defaults False in production paths), so blocking it for a new mode is not a
  regression. Do NOT silently write bf16 data under the fp8 signature.

- [ ] **Step 2: Write the test** for whichever option you chose — either "an unquantized cache is
  refused with a clear message and no file is written", or a full save/restore round-trip
  asserting `torch.equal` on the restored tensors.

- [ ] **Step 3: Run it, confirm it fails. Step 4: Implement. Step 5: Re-run + full suite.**

- [ ] **Step 6: Commit.**

---

## Task 4: Wire the mode into `_create_kv_cache`

`vaayu/vaayu_base.py`'s `_create_kv_cache` currently maps `"auto"` → `"fp8"` for EXL3 models and
`"fp16"` → plain `DynamicCache`.

**Files:** Modify `vaayu/vaayu_base.py`; test in `tests/test_fp8_kv_cache.py`.

- [ ] **Step 1: Write the failing test** — `kv_cache_mode="bf16"` returns an `FP8DynamicCache`
  with `.quantize is False`; `"fp8"` still returns one with `.quantize is True`; `"fp16"` still
  returns a plain `DynamicCache`.

- [ ] **Step 2: Run, confirm failure. Step 3: Add the `"bf16"` branch.**

**Do NOT change what `"auto"` selects in this task.** Leave `auto` → fp8. Changing the default is
a behavioral change for every user and belongs behind the on-device gate in Task 5, not ahead of
it.

- [ ] **Step 4: Re-run + full suite. Step 5: Commit.**

---

## Task 5: On-device gate + RESULTS (operator-run, needs the real GPU)

This task needs a CUDA GPU and the real model weights. It takes ~15 minutes per run.

**Files:** Create `docs/superpowers/plans/2026-08-02-kv-double-storage-fix-RESULTS.md`.

- [ ] **Step 1: Baseline.** Run `test_generate.py` unchanged (fp8 mode) and record decode tok/s
  and peak VRAM.

- [ ] **Step 2: bf16 mode.** Set `KV_CACHE_MODE = "bf16"` in `test_generate.py`, re-run, record
  the same figures.

- [ ] **Step 3: Compare.** Report:
  - KV footprint before/after — get it from `model._prefix_cache` or by printing
    `past_key_values.resident_bytes()` after generation.
  - Decode tok/s vs the floors (MiniMax ≈1.47, Qwen ≈0.82). bf16 mode removes quantization work,
    so decode should be **flat or slightly better**. If it is worse, say so plainly and
    investigate rather than shipping it.
  - Output quality: bf16 storage is lossless where fp8 is not, so generated text may differ from
    the fp8 run. That is expected and is an improvement, not a regression — but eyeball the output
    for coherence.

- [ ] **Step 4: Repeat on the second model.** `test_generate.py`'s `MODEL_PATH` has both paths
  (one commented). Use a fresh `SESSION_ID` when switching models — a persisted KV under another
  model's id would poison the run; the file carries a comment warning about exactly this.

- [ ] **Step 5: Decide whether `auto` should switch to bf16.** State a recommendation with the
  measured numbers behind it. If bf16 is lossless, frees ~33% of KV VRAM, and does not regress
  decode on both models, flipping `auto` is justified — but make it an explicit, separately
  committed decision.

- [ ] **Step 6: Write RESULTS and commit**, including a note that any `run-*.log` evidence files
  are operator-local and untracked.

- [ ] **Step 7: Revert scaffolding.** `git checkout test_generate.py` and confirm it does not
  appear in `git status --short`.

---

## Success Criteria

- `FP8DynamicCache(quantize=False)` stores 2.0 bytes/element, is lossless, and keeps the ring
  buffer's amortized-O(1) append.
- The fp8 path is byte-identical to today; all pre-existing fp8 tests pass untouched.
- `kv_cache_manager` and `kv_persist` no longer assume "is an `FP8DynamicCache`" implies "has
  scales".
- On-device: KV footprint down ~33%, decode not regressed on either model.
- `pytest tests/ -q` green.

## Deliberately NOT in scope

- **Dequantize-on-read (the 1.02 bytes/element option).** Frees more but adds O(seq_len) casts per
  layer per decode step, and decode is already ~60-70% compute-bound here.
- **TurboQuant / any new quantization scheme.** Separate plan; its compression ratio applies to
  whatever remains after this fix.
- **Changing what `auto` selects** — Task 5 Step 5 recommends, a later commit decides.
