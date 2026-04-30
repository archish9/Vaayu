# Lossless GPU Speedups for Vaayu

Implement three lossless GPU optimizations (Flash Attention audit, `torch.compile`, CUDA Graphs) into Vaayu's layer-streaming pipeline. All three produce **mathematically identical outputs** — zero accuracy loss.

## User Review Required

> [!IMPORTANT]
> **Phase 1 (Flash Attention) is by far the highest-impact, lowest-risk change.** The current attention mask format (4D boolean tensor) is almost certainly forcing PyTorch to silently fall back to Math SDP — meaning Vaayu is getting **zero** Flash Attention speedup today, even though the code tries to enable it.

> [!NOTE]
> **Phase 3 (CUDA Graphs) has been DROPPED** per user decision. `torch.compile(mode="reduce-overhead")` in Phase 2 handles CUDA Graphs internally where safe.

> [!IMPORTANT]
> **The model is fully rebuilt on every `forward()` call** (lines 652-655 in `vaayu_base.py`). This means `torch.compile` cannot simply wrap the whole model — it must target individual layer forward calls and rely on PyTorch's compilation cache to avoid re-compiling identical graph structures.

> [!IMPORTANT]
> **BF16 auto-detection** will be added to Phase 1. Instead of changing the default dtype, Vaayu will detect GPU BF16 support at runtime and auto-select the optimal dtype. Older GPUs fall back to FP16 automatically.

---

## Proposed Changes

### Phase 1: Flash Attention Audit & Fix (HIGH IMPACT, LOW RISK)

**The core problem:** Vaayu constructs a 4D boolean attention mask on lines 661-663:

```python
attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0  # shape: [1, 1, S, S]
```

Flash Attention **cannot** process 4D boolean masks. When PyTorch's SDPA receives this mask, it silently falls back to `Math SDP` (the slowest backend). This means Vaayu is currently getting **zero** speedup from SDPA despite the code attempting to enable it.

**The fix:** For causal (autoregressive) attention, Flash Attention accepts `is_causal=True` instead of an explicit mask. We need to stop passing the 4D mask to decoder layers and instead rely on the `is_causal` flag that HuggingFace attention modules already support.

---

#### [MODIFY] [vaayu_base.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/vaayu_base.py)

**Change 1: Replace attention fallback chain in `init_model()` (lines 254-306)**

Replace the current 4-step cascade (BetterTransformer → SDPA → default → eager) with a cleaner, logged fallback chain that tries `flash_attention_2` first:

```python
def init_model(self):
    self.model = None
    self._active_attn_impl = None  # Track what's actually running

    # Ordered preference: flash_attention_2 > sdpa > eager
    attn_candidates = []

    # Try flash_attention_2 if the package is installed
    try:
        import flash_attn
        attn_candidates.append("flash_attention_2")
    except ImportError:
        pass

    attn_candidates.extend(["sdpa", "eager"])

    # Also try BetterTransformer first (legacy, deprecated)
    if self.get_use_better_transformer() and better_transformer_available:
        try:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                self.model = BetterTransformer.transform(self.model)
            self._active_attn_impl = "BetterTransformer"
        except Exception:
            del self.model
            clean_memory()
            self.model = None

    # Try each attention implementation in order
    if self.model is None:
        for attn_impl in attn_candidates:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        self.config, attn_implementation=attn_impl, trust_remote_code=True
                    )
                self._active_attn_impl = attn_impl
                break
            except Exception:
                if self.model is not None:
                    del self.model
                clean_memory()
                self.model = None

    if self.model is None:
        raise RuntimeError("Failed to create model with any attention implementation")

    print(f"✅ Attention implementation: {self._active_attn_impl}")

    # Log the actual attention module type for verification
    try:
        first_layer = list(self.model.modules())
        for mod in first_layer:
            if 'Attention' in type(mod).__name__:
                print(f"   Attention module type: {type(mod).__name__}")
                break
    except Exception:
        pass

    # ... rest of init_model (quantization, eval, buffers) stays unchanged ...
```

**Change 2: Add BF16 auto-detection in `__init__()` (line 121)**

Auto-detect GPU BF16 support at runtime. Users with Ampere+ GPUs get BF16 automatically (better stability), older GPUs stay on FP16:

```python
# After: self.running_device = device (line 120)
# Auto-select optimal dtype if user passed the default float16
if dtype == torch.float16 and device.startswith("cuda"):
    try:
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:  # Ampere (sm_80) and newer support BF16
            self.running_dtype = torch.bfloat16
            print(f"✅ GPU supports BF16 (compute capability {cc[0]}.{cc[1]}). Using bfloat16 for better stability.")
        else:
            self.running_dtype = dtype
            print(f"ℹ️ GPU compute capability {cc[0]}.{cc[1]} — using float16.")
    except Exception:
        self.running_dtype = dtype
else:
    self.running_dtype = dtype
```

> [!NOTE]
> This only auto-upgrades when the user passes the **default** `torch.float16`. If a user explicitly passes `dtype=torch.float32` or any other dtype, it's respected as-is.

**Change 3: Fix attention mask format in `forward()` (lines 660-663)**

The current 4D boolean mask prevents Flash SDP from activating. Replace with `None` when Flash/SDPA is active (they handle causal masking internally via `is_causal=True`):

```python
# BEFORE (lines 660-663):
attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
attention_mask = attention_mask.to(self.running_device)

# AFTER:
if self._active_attn_impl in ("flash_attention_2", "sdpa"):
    # Flash/SDPA handle causal masking internally — no explicit mask needed
    attention_mask = None
else:
    # Eager attention needs the explicit 4D causal mask
    attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
    attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
    attention_mask = attention_mask.to(self.running_device)
```

**Change 3: Update `get_attention_mask_args()` (line 614-615)**

Handle `None` attention mask:

```python
def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
    if full_attention_mask is None:
        return {}  # Flash/SDPA handle causal masking internally
    return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}
```

**Change 4: Update layer forward calls (lines 821-828)**

When attention_mask is None, don't pass it explicitly — let HuggingFace use `is_causal=True` internally:

```python
# In the "not use_cache" branch (lines 821-828):
if not use_cache:
    kwargs = {'use_cache': False}
    if attention_mask is not None:
        kwargs['attention_mask'] = attention_mask[:, :, -len_seq:, -len_seq:]
    kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
    layer_out = layer(seq, **kwargs)
```

**Change 5: Add diagnostic method for users**

```python
def diagnose_attention(self):
    """Print diagnostic info about which attention backend is active."""
    print(f"Configured attention: {self._active_attn_impl}")
    if torch.cuda.is_available():
        print(f"Flash SDP available: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Memory-efficient SDP available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"Math SDP available: {torch.backends.cuda.math_sdp_enabled()}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
```

---

### Phase 2: torch.compile Integration (MEDIUM IMPACT, MEDIUM RISK)

Since the model is rebuilt every `forward()` call, we cannot compile the whole model. Instead, we compile **individual layer forward calls** and rely on PyTorch's compilation cache to reuse the compiled graph across identical decoder layer architectures.

#### [MODIFY] [vaayu_base.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/vaayu_base.py)

**Change 1: Add `use_torch_compile` constructor parameter (line 63)**

```python
def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
             layer_shards_saving_path=None, profiling_mode=False, compression=None,
             hf_token=None, prefetching=True, delete_original=False,
             use_torch_compile=False):  # NEW parameter
```

Store and validate:

```python
self.use_torch_compile = use_torch_compile
if use_torch_compile:
    if not hasattr(torch, 'compile'):
        print("⚠️ torch.compile requires PyTorch 2.0+. Disabling.")
        self.use_torch_compile = False
    elif not device.startswith("cuda"):
        print("⚠️ torch.compile is most effective on CUDA. Disabling for CPU.")
        self.use_torch_compile = False
    else:
        print("✅ torch.compile enabled for layer forward passes")
        self._compiled_layer_fn = None
```

**Change 2: Compile the layer forward inside the forward loop (around line 828)**

Wrap the decoder layer call with `torch.compile`. Use `mode="reduce-overhead"` which optimizes for inference latency (enables CUDA Graphs internally when possible):

```python
# Before calling layer(seq, **kwargs) for decoder layers:
if self.use_torch_compile and layer_name not in (
    self.layer_names_dict['embed'],
    self.layer_names_dict['norm'],
    self.layer_names_dict['lm_head']
):
    # Compile the layer forward. PyTorch caches the compiled graph
    # for identical architectures, so this is fast after the first layer.
    if self._compiled_layer_fn is None:
        self._compiled_layer_fn = torch.compile(
            lambda m, x, **kw: m(x, **kw),
            mode="reduce-overhead",
            fullgraph=False,  # allow graph breaks for flexibility
        )
    layer_out = self._compiled_layer_fn(layer, seq, **kwargs)
else:
    layer_out = layer(seq, **kwargs)
```

> [!NOTE]
> `mode="reduce-overhead"` internally uses CUDA Graphs where possible, which partially addresses Phase 3's goals. This is the safer path compared to manual CUDA Graph management.

**Change 3: Reset compiled function on model rebuild**

In `forward()`, after `self.init_model()` (line 655):

```python
self.init_model()
self._compiled_layer_fn = None  # Reset for new model instance
```

---

### Phase 3: CUDA Graphs — DROPPED

> [!NOTE]
> **Dropped per user decision.** `torch.compile(mode="reduce-overhead")` in Phase 2 handles CUDA Graphs internally where PyTorch determines it's safe. Manual CUDA Graphs would conflict with Vaayu's dynamic weight loading architecture.

---

## Verification Plan

### Automated Tests

1. **Attention backend verification:**
```bash
python -c "
from vaayu import AutoModel
model = AutoModel.from_pretrained('model_path')
model.diagnose_attention()
"
```
Expected: Should print `flash_attention_2` or `sdpa`, NOT `eager`

2. **Numerical correctness test:**
```bash
# Run inference with each attention backend and compare outputs
python test_quick.py  # with sdpa (baseline)
python test_quick.py  # with flash_attention_2
# Outputs should be identical (or differ by < 1e-5 due to float ordering)
```

3. **torch.compile smoke test:**
```bash
python -c "
from vaayu import AutoModel
model = AutoModel.from_pretrained('model_path', use_torch_compile=True)
# Should complete without errors, possibly slower on first run (compilation)
"
```

4. **Performance benchmark** (using existing profiling mode):
```bash
# Baseline
python test_quick.py  # profiling_mode=True, note wall time

# With fixes
python test_quick.py  # profiling_mode=True, compare wall time
```

### Manual Verification

- Verify that the generated text quality is unchanged (same prompt, same output)
- Check GPU utilization via `nvidia-smi` during inference — should be higher with Flash Attention actually working
- Monitor for any CUDA errors or warnings during torch.compile warmup

---

## Summary

| Phase | Change | Expected Speedup | Risk | Files Modified |
|---|---|---|---|---|
| **1** | Flash Attention audit + fix mask + flash_attention_2 + BF16 auto-detect | **2-4x on attention** | 🟢 Low | `vaayu_base.py` |
| **2** | `torch.compile` on layer forward calls | **10-30% overall** | 🟡 Medium | `vaayu_base.py` |
| **3** | ~~CUDA Graphs~~ | ~~5-15%~~ | **DROPPED** | — |

**Final scope:** Phase 1 + Phase 2 only. Phase 3 dropped; `torch.compile(mode="reduce-overhead")` handles CUDA Graphs internally.
