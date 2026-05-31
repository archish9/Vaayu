# Speculative Decoding Design — Vaayu Phase H

**Date:** 2026-05-31  
**Plans:** schedule_plan_h_speculative_decoding_part1.md + part2.md  
**Status:** Approved for implementation

---

## Problem

Vaayu streams 80 transformer layers from NVMe once per token — ~5 seconds/token. Speculative decoding turns that single expensive pass into a verifier that validates N drafted tokens at once, targeting 2–4× effective throughput.

---

## Approach Selected

**Approach A:** ExLlamaV3 EAGLE bridge first, manual weight loader as fallback. Standard drafter fully implemented. Both paths wired into CLI config.

Priority: EAGLE > Standard Drafter > None (no speculation).

---

## Architecture

### New Package: `vaayu/speculative/`

```
vaayu/speculative/
  __init__.py      — build_speculator() factory
  base.py          — SpeculativeDecoder ABC
  detection.py     — auto-detect checkpoint type
  eagle.py         — EAGLEDecoder (exllamav3 bridge + manual fallback)
  standard.py      — StandardDrafterDecoder (any HF small model)
```

### Abstract Interface (`base.py`)

All speculators implement four methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `draft` | `(input_ids, past_key_values, num_tokens) → (draft_ids, draft_state)` | Generate N candidate tokens |
| `verify_and_accept` | `(input_ids, draft_ids, verifier_logits, draft_state) → (accepted_ids, num_accepted)` | Accept longest correct prefix |
| `rollback_cache` | `(draft_state, num_accepted)` | Trim drafter KV cache to match accepted length |
| `vram_usage_bytes` | `() → int` | Report VRAM cost |

### Detection (`detection.py`)

`detect_speculative_type(model_dir) → 'eagle' | 'medusa' | 'none'`

Detection order:
1. `medusa_num_heads` / `medusa_num_layers` in config.json → `'medusa'`
2. Any file starting with `medusa_lm_head` → `'medusa'`
3. `speculators_config.method` contains `"eagle"` → `'eagle'`
4. `eagle_config` key present → `'eagle'`
5. Any architecture name contains `"eagle"` → `'eagle'`
6. Otherwise → `'none'`

### EAGLE Decoder (`eagle.py`)

**Construction:**
1. Load config.json from `eagle_head_path`
2. Load all `.safetensors` weights
3. Try `from exllamav3.speculative import EAGLEDraftModel` → delegate all ops
4. On ImportError: manual path — build single transformer layer from weights

**`draft()` implementation:**
1. Read `self.base_model._last_hidden_state` (set during most recent `forward()`)
2. Pass through EAGLE extrapolation layer → `next_hidden`
3. Apply base model's lm_head: `self.base_model.run_lm_head(self.base_model._lm_head_layer, next_hidden)`
4. Take argmax → draft token; feed `next_hidden` back for next step
5. Return `(draft_ids, {"hidden_states": last_hidden})`

**Bug fixes vs plan:**
- `run_lm_head(self.base_model.layers[-1], ...)` → `run_lm_head(self.base_model._lm_head_layer, ...)` — `layers[-1]` is a transformer block, not lm_head
- `_get_last_hidden_state()` → read `self.base_model._last_hidden_state` directly

**`rollback_cache()`:** EAGLE shares base model KV cache — rollback handled by `speculative_generate()` truncating the DynamicCache to accepted length. EAGLE's method is a no-op.

### Standard Drafter (`standard.py`)

Loads any HuggingFace causal LM entirely into VRAM:
```
self.drafter = AutoModelForCausalLM.from_pretrained(drafter_path, device_map=device, ...)
```

**`draft()`:** Autoregressive loop — start from last generated token, run drafter N times collecting greedy tokens. Maintains `self._drafter_past` across steps for KV cache reuse.

**`rollback_cache()`:** On partial acceptance, set `self._drafter_past = None` to discard stale drafter cache. On full acceptance, retain the cache. Simplified (non-crop) approach — optimize later.

**Recommended drafters:**

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen2.5-0.5B q4 | ~300 MB | Best fit, easy to test |
| Qwen2.5-Coder-1.5B q4 | ~1.0 GB | Better quality, still fits |
| Any 3B+ fp16 | >6 GB | Does NOT fit on 16 GB setup |

### Factory (`__init__.py`)

`build_speculator(eagle_head_path, drafter_model_path, base_model, device) → SpeculativeDecoder | None`

Priority: EAGLE (or Medusa via EAGLE loader) > StandardDrafter > None.

---

## VaayuBaseModel Changes (`vaayu_base.py`)

### 1. Constructor params

```python
def __init__(self, ...,
             eagle_head_path=None,
             drafter_model_path=None,
             draft_tokens=5,
             ...):
```

After model loading: `self.speculator = build_speculator(eagle_head_path, drafter_model_path, self, device)`

### 2. Hidden state capture

Inside the layer loop in `forward()`, capture the hidden state after the norm layer runs (post-norm, pre-lm_head). Use name matching via `layer_names_dict`:

```python
# Capture post-norm hidden state for EAGLE draft()
norm_name = self.layer_names_dict.get('norm')
if norm_name and layer_name == norm_name:
    self._last_hidden_state = seq.detach()
```

This is name-based (not index-based) so it works correctly regardless of model architecture or layer count.

Also store a reference to the lm_head module after model loading in `__init__`:
```python
# Resolve and store lm_head module for EAGLE draft()
lm_head_name = self.layer_names_dict.get('lm_head')
self._lm_head_layer = self._get_module_by_path(lm_head_name)
```

### 3. `speculative_generate()` method

Custom generation loop on `VaayuBaseModel`. Called from `test_quick.py` when `self.speculator` is set.

```
1. Prefill: output = self.forward(input_ids, use_cache=True)
   → past_key_values, _last_hidden_state set as side effect
2. Loop until max_new_tokens or EOS:
   a. draft_ids, draft_state = self.speculator.draft(generated, past_kv, num_tokens=self.draft_tokens)
   b. verify_input = concat(last generated token, draft_ids)
   c. verify_output = self.forward(verify_input, past_kv, use_cache=True)
   d. accepted_ids, n = self.speculator.verify_and_accept(generated, draft_ids, verify_output.logits, draft_state)
   e. generated = concat(generated, accepted_ids)
   f. self.speculator.rollback_cache(draft_state, n)
   g. Print: "[Vaayu-Spec] drafted X, accepted Y, total Z"
   h. EOS check
```

**Bug fix vs plan:** `self._chunked_forward(input_ids=input_ids, use_cache=True)` → `self.forward(input_ids=input_ids, use_cache=True)` — chunked prefill fires automatically inside `forward()` when `prefill_chunk_size > 0` and no past KV.

---

## test_quick.py Changes

Add to CONFIG block:

```python
EAGLE_HEAD_PATH    = None   # path to EAGLE-3/Medusa head checkpoint dir
DRAFTER_MODEL_PATH = None   # path to small HF drafter (e.g. Qwen2.5-0.5B)
DRAFT_TOKENS       = 5      # candidates per speculative iteration
```

Generation dispatch:

```python
if model.speculator:
    output_ids = model.speculative_generate(
        input_ids.to(DEVICE), max_new_tokens=MAX_NEW_TOKENS
    )
else:
    generation_output = model.generate(...)
    output_ids = generation_output.sequences[0]
```

---

## auto_model.py Changes

Pass new params through `from_pretrained()` → model constructor:
- `eagle_head_path`
- `drafter_model_path`  
- `draft_tokens`

---

## VRAM Budget (16 GB GPU, post Plans E+F)

```
Pinned layers (Plan F):     ~12.0 GB
KV Cache FP8 (Plan E):      ~0.6 GB
CUDA context:               ~0.8 GB
Streaming buffer:           ~0.5 GB
─────────────────────────────────────
Free for speculative:       ~2.1 GB

EAGLE-3 head:               ~0.1–0.2 GB  ✅ fits easily
Qwen2.5-0.5B q4 drafter:   ~0.3 GB      ✅ fits
Qwen2.5-1.5B q4 drafter:   ~1.0 GB      ✅ fits (tight)
3B fp16 drafter:            ~6 GB        ❌ does not fit
```

---

## Acceptance Criterion

**EAGLE path:**
```bash
python test_quick.py  # with EAGLE_HEAD_PATH set
# Expect: pause → multiple tokens appear at once, >1.5 t/s effective
```

**Standard drafter:**
```bash
python test_quick.py  # with DRAFTER_MODEL_PATH set to Qwen2.5-0.5B
# Expect: >1.0 t/s effective
```

**Detection test:**
```python
from vaayu.speculative.detection import detect_speculative_type
assert detect_speculative_type("/path/to/eagle3-head") == "eagle"
assert detect_speculative_type("/path/to/regular-model") == "none"
```

**VRAM check:** EAGLE head < 200 MB, 0.5B drafter < 500 MB.

---

## Files Changed

| File | Action |
|------|--------|
| `vaayu/speculative/__init__.py` | NEW — factory |
| `vaayu/speculative/base.py` | NEW — ABC |
| `vaayu/speculative/detection.py` | NEW — auto-detect |
| `vaayu/speculative/eagle.py` | NEW — EAGLE decoder |
| `vaayu/speculative/standard.py` | NEW — standard drafter |
| `vaayu/vaayu_base.py` | MODIFY — hidden state capture, speculative_generate(), constructor params |
| `vaayu/auto_model.py` | MODIFY — pass through new params |
| `test_quick.py` | MODIFY — config vars + dispatch |
| `README.md` | DOC — Phase H victory bullet + roadmap update |
| `documentation/speculative_decoding_and_moe.md` | DOC — as-built architecture |
| `documentation/architecture.md` | DOC — add speculative decoding section |
