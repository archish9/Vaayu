# GPU Speed Optimization — Remove Per-Token Waste & Enable KV Cache

## Background

Vaayu running Llama 3.3 70B currently takes **~54 seconds per token** and generates 20 tokens in ~18 minutes. Two critical issues cause unnecessary work on every single token:

1. **Model reinit every token**: `forward()` calls `del self.model; clean_memory(); init_model()` on every forward pass (lines 697-701), destroying and recreating the model shell (~2-3s wasted per token).
2. **KV cache disabled**: `use_cache` is hardcoded to `False` (lines 687-689), forcing the model to reprocess the **entire input sequence** through all 83 layers on every token, instead of just the new token.

Both fixes are **mathematically lossless** — they produce bit-identical output.

> [!IMPORTANT]
> I/O bottleneck (disk→CPU→GPU weight loading) is explicitly **out of scope** — it will be addressed later with Rust.

---

## KV Cache Memory Scaling

KV cache size scales linearly with sequence length. The DynamicCache will automatically place tensors on whatever device the computation runs on — no hardcoded device assumptions.

**Formula per token per layer**: `2 (K+V) × num_kv_heads × head_dim × dtype_bytes`

**Example: Llama 3.3 70B (GQA, 8 KV heads, 80 layers, bf16):**

| Sequence Length | KV Cache Size |
|---|---|
| 128 tokens | 40 MB |
| 512 tokens | 160 MB |
| 2,048 tokens | 640 MB |
| 8,192 tokens | 2.5 GB |
| 131,072 (full context) | 40 GB |

For short sequences the cache is trivial. For very long sequences, CPU offloading can be added later — DynamicCache supports this natively.

---

## Subclass Audit

All changes must apply across every supported architecture. Here's what each subclass overrides for cache handling:

| Subclass | Overrides `get_past_key_value_args`? | KV arg name | Overrides `get_past_key_values_cache_seq_len`? | Notes |
|---|---|---|---|---|
| **VaayuLlama2** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuQWen2** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuMistral** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuMixtral** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuInternLM** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuBaichuan** | ❌ (uses base) | `past_key_value` | ❌ (uses base) | Modern arch, DynamicCache compatible |
| **VaayuQWen** (v1) | ✅ → `layer_past` | `layer_past` | ✅ → `shape[1]` | Legacy arch, uses custom tuple format |
| **VaayuChatGLM** | ✅ → `kv_cache` | `kv_cache` | ✅ → `shape[0]` | Legacy arch, uses custom tuple format + custom `get_sequence_len` |

**Strategy**: The 6 modern architectures (Llama, Qwen2, Mistral, Mixtral, InternLM, Baichuan) all use the base class defaults and are fully DynamicCache-compatible with transformers 5.5.4. The 2 legacy architectures (QWen v1, ChatGLM) use custom KV argument names — these will get a compatibility bridge in the base class that detects whether the subclass overrides `get_past_key_value_args` and falls back to the legacy tuple path when needed.

---

## Proposed Changes

### Phase 1: Remove Per-Token Model Reinit

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

**Problem**: Lines 697-701 destroy and rebuild the entire model on every forward call:
```python
# Reboot the model to make sure buffers are loaded and memory is clean
del self.model
clean_memory()
self.init_model()
self._compiled_layer_fn = None
```

The per-layer cleanup at lines 927-934 already handles memory correctly — each layer is moved to `"meta"` device after use, and `clean_memory()` is called. The model shell (module hierarchy, config, tied weights) doesn't need rebuilding.

**Changes**:
1. **Remove lines 697-701** entirely. The model shell from `__init__` persists between forward calls. Layer weights are loaded/unloaded per layer as before.
2. **Cache RoPE computation** (lines 725-741): Currently recomputed every forward(). Move to a lazy-init pattern — compute once on first forward, store in `self._full_position_embeddings`, reuse on subsequent calls. The RoPE values only depend on `max_seq_len` and config, which don't change.

**Expected impact**: Saves ~2-3 seconds per token (model reconstruction + RoPE recomputation overhead).

---

### Phase 2: Enable KV Cache with DynamicCache

This is the highest-impact change. With KV cache, token 2+ only processes **1 token** through all 83 layers instead of reprocessing the growing sequence.

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

**Change 1 — Enable cache class support** (line 99):
```diff
- self._supports_cache_class = False
+ self._supports_cache_class = True
```
This tells transformers 5.5.4's `generate()` to use `DynamicCache` natively. DynamicCache automatically places KV tensors on whatever device the computation runs on — no hardcoded device assumptions.

**Change 2 — Remove use_cache override** (lines 687-689):
```diff
- if cache_utils_installed:
-     # we don't support kv cache for new version yet
-     use_cache = False
```

**Change 3 — Fix `prepare_inputs_for_generation`** (lines 545-553):

Currently drops the DynamicCache object:
```python
if cache_utils_installed and isinstance(past_key_values, Cache):
    past_key_values = None  # ← drops it!
```

Change to pass DynamicCache through to forward:
```python
if past_key_values is not None:
    past_length = self.get_past_key_values_cache_seq_len(past_key_values)
    # Keep the cache object — pass it through to forward()
    if input_ids.shape[1] > past_length:
        remove_prefix_length = past_length
    else:
        remove_prefix_length = input_ids.shape[1] - 1
    input_ids = input_ids[:, remove_prefix_length:]
```

**Change 4 — Adapt decoder layer loop to use DynamicCache** (lines 826-920):

In transformers 5.5.4, decoder layers expect `past_key_value=<DynamicCache>` (the whole cache object, not a per-layer tuple). Each layer internally uses `self.layer_idx` to access its own KV slice via `cache.update(key, value, self.layer_idx)`.

The base class forward loop changes to:
```python
# At forward start:
if use_cache and past_key_values is None:
    past_key_values = DynamicCache()

# For each decoder layer:
has_cache = (past_key_values is not None 
             and hasattr(past_key_values, 'get_seq_length') 
             and past_key_values.get_seq_length() > 0)
len_p = past_key_values.get_seq_length() if has_cache else 0
len_s = self.get_sequence_len(seq)

kwargs = {
    'use_cache': use_cache,
    **self.get_past_key_value_args(past_key_values),  # passes full cache
    **self.get_pos_emb_args(len_p, len_s),
    **self.get_attention_mask_args(attention_mask, len_p, len_s),
    **self.get_position_ids_args(position_ids, len_p, len_s),
}
layer_out = layer(seq, **kwargs)
new_seq = layer_out[0] if isinstance(layer_out, tuple) else layer_out
```

DynamicCache is updated **in-place** by each layer — no manual cache list management needed.

**Change 5 — Update `get_past_key_value_args`** (line 651-652):

The base class method passes the full DynamicCache object:
```diff
- def get_past_key_value_args(self, k_cache, v_cache):
-     return {'past_key_value': (k_cache, v_cache)}
+ def get_past_key_value_args(self, past_key_values):
+     return {'past_key_value': past_key_values}
```

**Change 6 — Update `get_past_key_values_cache_seq_len`** (lines 591-598):

Simplify to always use DynamicCache's API:
```python
def get_past_key_values_cache_seq_len(self, past_key_values):
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values.get_seq_length()
    # Legacy tuple fallback for QWen v1 / ChatGLM
    return past_key_values[0][0].shape[2]
```

**Change 7 — Remove legacy cache list code** (lines 743-746, 937-941):

Remove the `kv_cache_list` initialization and post-processing. Return the DynamicCache directly:
```python
return CausalLMOutputWithPast(
    logits=logits,
    past_key_values=past_key_values,  # DynamicCache object
    ...
)
```

---

#### [MODIFY] [vaayu_qwen.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_qwen.py)

QWen v1 uses a legacy `layer_past` argument name and custom cache tensor shapes. Two options:

- **Option A (preferred)**: Update `get_past_key_value_args` to accept a DynamicCache and extract the per-layer tuple from it, converting to the `layer_past` kwarg name that the QWen v1 layer expects. This preserves backward compatibility with the older layer code.
- **Option B**: If QWen v1 in transformers 5.5.4 has been updated to use DynamicCache natively, simply remove the override.

The `get_past_key_values_cache_seq_len` override (shape[1]) becomes unnecessary since DynamicCache has `.get_seq_length()`.

#### [MODIFY] [vaayu_chatglm.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_chatglm.py)

Same approach as QWen v1 above. ChatGLM uses `kv_cache` as its kwarg name and shape[0] for seq length. The override adapts DynamicCache to the legacy format the ChatGLM layer expects.

#### No changes needed for: [vaayu_qwen2.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_qwen2.py), [vaayu_mistral.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_mistral.py), [vaayu_mixtral.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_mixtral.py), [vaayu_internlm.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_internlm.py), [vaayu_baichuan.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_baichuan.py)

These all use base class defaults and are DynamicCache-compatible out of the box.

---

### Phase 3: SDP Backend Verification Diagnostic

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

Add a one-time diagnostic in `__init__` to confirm which SDP backend is actually available:

```python
if torch.cuda.is_available():
    print(f"  Flash SDP available: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient SDP:   {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP (slow):     {torch.backends.cuda.math_sdp_enabled()}")
```

Logging-only change, no behavioral impact.

---

### Phase 4: Enable `torch.compile` (Optional / Lower Priority)

#### [MODIFY] [test_quick.py](file:///home/a/code/ai-ml/Vaayu/test_quick.py)

Document how to enable `torch.compile`:
```python
model = AutoModel.from_pretrained("...", device=DEVICE, use_torch_compile=True)
```

> [!WARNING]
> `torch.compile` has a significant **warmup cost** on the first forward call (30-60s compilation). It pays off only for longer generations (50+ tokens). For short tests, leave it disabled.

Phase 1 (removing model reinit) is a **prerequisite** for torch.compile to work, since the current code resets `self._compiled_layer_fn = None` every forward call, nullifying any compilation benefit.

---

## Verification Plan

### Automated Tests

1. **Correctness test**: Run `test_quick.py` with the same prompt before and after changes. Compare the output text — it must be **identical** (both fixes are lossless).
2. **Performance test**: Compare per-token times:
   - Before: ~54s/token (all tokens equal)
   - After: Token 1 should be ~54s (full prompt), tokens 2+ should be **significantly faster** (only 1 token through layers)
3. **Memory test**: Monitor GPU VRAM during generation to confirm KV cache doesn't cause OOM.

### Manual Verification

Run the test and confirm:
- [ ] Output text matches pre-change output
- [ ] Token 1 time ≈ previous per-token time
- [ ] Token 2+ times are measurably faster
- [ ] No `✅ Attention implementation: sdpa` repeated every token (model no longer reinited)
- [ ] SDP diagnostic shows Flash or Memory-efficient SDP, not just Math SDP
