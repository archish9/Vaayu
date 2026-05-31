# Plan E: KV Cache Quantization — Design Spec

**Date:** 2026-05-31
**Phase:** E (runs parallel with Phase F)
**Depends on:** Phase B (EXL3 Pivot) ✅
**Blocks:** Phase G (Chunked Prefill), Phase H (Speculative Decoding)

---

## Problem

After Phase B, weights are 4-bit EXL3. KV cache is still BF16. For Llama 3.3 70B:

```
BF16 KV @ 8192 tokens:
  80 layers × 2 × 8 kv_heads × 128 head_dim × 8192 tokens × 2 bytes ≈ 2.68 GB

FP8 KV @ 8192 tokens (with per-head-per-token scales):
  80 × 2 × 8 × (128 + 4) × 8192 × 1 byte ≈ 1.38 GB

Savings: ~1.3 GB freed at 8k context → available for layer pinning (Phase F) and drafter model (Phase H)
```

At 512-token default: savings are ~80 MB (small). At long contexts, savings compound.

---

## Scope

- Add `FP8DynamicCache`: a `DynamicCache` subclass storing K/V in `torch.float8_e4m3fn` with per-head per-token scales
- Add `kv_cache_mode` param to `VaayuBaseModel.__init__()` (`"fp16"` | `"fp8"` | `"auto"`)
- `auto` resolves to `"fp8"` when `EXL3LoaderStrategy` is active, else `"fp16"`
- Update `_get_vram_pinning_budget()` to use actual KV footprint instead of flat 4 GB reserve
- Add `--kv_cache_mode` CLI arg to `test_quick.py`

**Out of scope:** INT4 KV cache, ExLlamaV3 native cache integration, `quanto`/`hqq` backends.

---

## Architecture

### Why not ExLlamaV3 native cache?

`exllamav3.Cache.__init__(model, max_num_tokens, layer_type)` requires a persistent `exllamav3.Model` object and calls `attach_to_model()`. Vaayu streams layers one at a time with no persistent EXL3 model object. Incompatible.

### Why not HF `QuantizedCache`?

`QuantizedCache(backend, config, nbits=4, ...)` requires `"quanto"` or `"HQQ"` backend. Neither is installed in the Vaayu venv. Would add a new dependency.

### Chosen: Per-token FP8 scale (pure PyTorch, zero deps)

Subclass `DynamicCache`. On each `update()` call, quantize only the **new** tokens (O(1) per decode step). Append FP8 data and float32 scales separately. Dequantize full accumulated cache on readout.

Scale granularity: per-head per-token → shape `[batch, n_heads, new_tokens, 1]`. This eliminates cross-head scale corruption and allows O(1) append (no re-quantization of past tokens).

---

## `FP8DynamicCache` — Detailed Design

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
        self._key_scales: dict[int, torch.Tensor] = {}
        self._value_scales: dict[int, torch.Tensor] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # --- quantize new tokens only ---
        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0

        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # --- append to growing cache ---
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

        # --- dequantize full cache for attention ---
        k_out = self.key_cache[layer_idx].to(key_states.dtype) * self._key_scales[layer_idx]
        v_out = self.value_cache[layer_idx].to(value_states.dtype) * self._value_scales[layer_idx]
        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Inherited DynamicCache reads key_cache[0].shape[-2]; works unchanged.
        return super().get_seq_length(layer_idx)
```

**Key invariants:**
- `self.key_cache[i]` dtype is always `torch.float8_e4m3fn`
- `self._key_scales[i]` shape is always `[batch, heads, seq_len, 1]` in `float32`
- `get_seq_length()` still correct because `key_cache[0].shape[-2]` = accumulated seq_len

---

## `_create_kv_cache()` — Factory Method

```python
def _create_kv_cache(self):
    mode = self.kv_cache_mode

    if mode == "auto":
        from .persist.loader_strategy import EXL3LoaderStrategy
        mode = "fp8" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"

    if mode == "fp16":
        return DynamicCache()

    # FP8 path
    if not hasattr(torch, "float8_e4m3fn"):
        print("[Vaayu] FP8 KV cache unavailable (PyTorch < 2.1). Falling back to FP16.")
        return DynamicCache()

    print("[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)")
    return FP8DynamicCache()
```

---

## `_get_vram_pinning_budget()` — Updated Formula

Replaces the flat 4 GB reserve with a computed KV footprint + 1.5 GB overhead.

```
FP8 bytes  = n_layers × 2 × n_kv_heads × (head_dim + 4) × max_seq_len
             ↑ +4 bytes/head/token for float32 scale overhead

BF16 bytes = n_layers × 2 × n_kv_heads × head_dim × 2 × max_seq_len

budget = total_vram - kv_bytes - 1.5 GB (CUDA context + streaming double-buffer)
```

Falls back to `head_dim = hidden_size // num_attention_heads` if `config.head_dim` is absent (Qwen, etc.).

---

## Files to Modify

| File | Change |
|------|--------|
| `vaayu/vaayu_base.py` | Add `FP8DynamicCache` class; add `kv_cache_mode` param to `__init__`; add `_create_kv_cache()`; update `forward()` line 912; update `_get_vram_pinning_budget()` |
| `test_quick.py` | Add `--kv_cache_mode` arg (`choices=["fp16","fp8","auto"]`, default `"auto"`) |

`auto_model.py` needs no changes — `from_pretrained()` already forwards `**kwargs` to the model class.

---

## Verification

### VRAM savings
```bash
# terminal 1: watch VRAM
watch -n 0.5 nvidia-smi --query-gpu=memory.used --format=csv,noheader

# FP16 baseline (long context)
python test_quick.py --kv_cache_mode fp16

# FP8 (should be ~1–1.3 GB lower at 8k context)
python test_quick.py --kv_cache_mode fp8
```

### Output quality
- Same prompt, both modes → outputs should be semantically identical
- FP8 precision loss is below perceptible threshold for coding tasks (established in literature)

### Correctness
- `get_seq_length()` returns correct value after N decode steps
- Second-call generation (with non-None `past_key_values`) slices `input_ids` correctly
- No OOM at 8192 tokens with FP8 cache where FP16 cache would OOM

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FP8 overflow on extreme activations | Low | clamp to ±448 before cast; per-head scale bounds outliers |
| Quality regression on math/code | Very low | Per-head scale has sufficient granularity; test and compare |
| Decode speed regression from dequant | Negligible | Dequant is a single broadcast multiply, fused in CUDA |
| `get_seq_length()` mismatch | None | Inherits from DynamicCache unchanged |
