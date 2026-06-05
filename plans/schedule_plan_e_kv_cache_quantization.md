# Schedule Plan E: KV Cache Quantization (FP8 / Q4)

**Priority:** 3 (First thing after Plan B)
**Effort:** Low-Medium
**Risk:** Low
**VRAM Savings:** ~1.9 GB freed

## Objective
Quantize the KV cache from FP16/BF16 to Q4 (or Q8) to shrink the conversation context footprint by up to 75%. This frees ~1.9 GB of VRAM, which is critical for later enabling speculative decoding (Phase H) and supporting longer context windows without OOM.

## The Problem
After Plan B, model weights are compressed to 4-bit EXL3, but the **KV cache is still full precision**. For a 70B model:

```
KV Cache Size Formula:
  = num_layers × 2 × num_kv_heads × head_dim × seq_len × dtype_bytes

Llama 3.3 70B @ 8,192 tokens, BF16:
  = 80 × 2 × 8 × 128 × 8192 × 2 bytes
  = ~2.5 GB

With Q4 KV Cache:
  = ~0.625 GB  (75% savings → ~1.9 GB freed)
```

This 2.5 GB KV cache silently eats into the VRAM budget that should be reserved for layer pinning or speculative decoding.

## Current Code Architecture

### How the KV cache is created today
In `vaayu/vaayu_base.py` line 912-913:
```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache()
```

`DynamicCache` (from `transformers.cache_utils`) stores key/value tensors at the same dtype as the model's hidden states (BF16 on this GPU). There is no quantization applied.

### How ExLlamaV3 handles KV cache
ExLlamaV3 has its own cache system separate from HuggingFace's `DynamicCache`. Since Vaayu currently uses HuggingFace's cache with its layer-by-layer streaming, the integration path depends on which cache system is active.

## Implementation Steps

### Step 1: Check ExLlamaV3's KV Cache API

Before writing code, investigate the ExLlamaV3 cache API:

```python
# Run this in a Python shell with exllamav3 installed
import exllamav3
# Look for cache-related classes
print([x for x in dir(exllamav3) if 'cache' in x.lower() or 'Cache' in x])
```

Look for classes like:
- `ExLlamaV3Cache` — base cache class
- `ExLlamaV3Cache_Q4` — 4-bit quantized cache
- `ExLlamaV3Cache_Q8` — 8-bit quantized cache

Check the constructor signature:
```python
help(exllamav3.ExLlamaV3Cache_Q4.__init__)
# Typical signature: ExLlamaV3Cache_Q4(model, max_seq_len, batch_size=1)
```

### Step 2A: If Using ExLlamaV3's Native Cache (Preferred)

If ExLlamaV3 provides its own cache classes that work with the EXL3 forward pass, integrate them directly.

**File to modify:** `vaayu/vaayu_base.py`

**Change 1:** Add a `kv_cache_mode` parameter to `__init__()` (line 112):
```python
def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
             layer_shards_saving_path=None, profiling_mode=False, compression=None,
             hf_token=None, prefetching=True, delete_original=False,
             use_torch_compile=False,
             kv_cache_mode="auto",  # NEW: "fp16", "q8", "q4", or "auto"
             layer_names_dict=None, tokenizer_class=None):
```

Store it as `self.kv_cache_mode = kv_cache_mode`.

**Change 2:** Modify the cache creation in `forward()` (line 912-913):
```python
# BEFORE:
if use_cache and past_key_values is None:
    past_key_values = DynamicCache()

# AFTER:
if use_cache and past_key_values is None:
    past_key_values = self._create_kv_cache()
```

**Change 3:** Add a new method `_create_kv_cache()`:
```python
def _create_kv_cache(self):
    """Create a KV cache with optional quantization to save VRAM."""
    mode = self.kv_cache_mode
    
    # Auto-detect: use Q4 for EXL3 models, FP16 for standard
    if mode == "auto":
        if isinstance(self.loader_strategy, EXL3LoaderStrategy):
            mode = "q4"
        else:
            mode = "fp16"
    
    if mode == "fp16":
        return DynamicCache()
    
    # Try ExLlamaV3 quantized cache
    try:
        if mode == "q4":
            from exllamav3 import ExLlamaV3Cache_Q4
            cache = ExLlamaV3Cache_Q4(self.model, self.max_seq_len)
            print(f"[Vaayu] KV Cache: Q4 quantized (75% VRAM savings)")
            return cache
        elif mode == "q8":
            from exllamav3 import ExLlamaV3Cache_Q8
            cache = ExLlamaV3Cache_Q8(self.model, self.max_seq_len)
            print(f"[Vaayu] KV Cache: Q8 quantized (50% VRAM savings)")
            return cache
    except (ImportError, Exception) as e:
        print(f"[Vaayu] Quantized KV cache ({mode}) failed: {e}. Falling back to FP16.")
        return DynamicCache()
```

### Step 2B: If ExLlamaV3 Cache Doesn't Fit Vaayu's Streaming Model

ExLlamaV3's cache may expect all layers to be in VRAM simultaneously (standard ExLlama usage). If so, Vaayu's layer-by-layer streaming is incompatible with ExLlamaV3's native cache.

In that case, implement **manual KV cache quantization** using HuggingFace's `QuantizedCache`:

```python
# transformers >= 4.44 has QuantizedCacheConfig and QuantizedCache
from transformers.cache_utils import QuantizedCacheConfig, QuantizedCache

def _create_kv_cache(self):
    mode = self.kv_cache_mode
    if mode == "auto":
        mode = "q4" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"
    
    if mode == "fp16":
        return DynamicCache()
    
    try:
        nbits = 4 if mode == "q4" else 8
        cache_config = QuantizedCacheConfig(
            backend="quanto",  # or "HQQ" 
            nbits=nbits,
        )
        cache = QuantizedCache(cache_config)
        print(f"[Vaayu] KV Cache: {nbits}-bit quantized via transformers QuantizedCache")
        return cache
    except (ImportError, Exception) as e:
        print(f"[Vaayu] Quantized KV cache failed: {e}. Falling back to FP16.")
        return DynamicCache()
```

**Dependency check:** `QuantizedCache` requires either `quanto` or `HQQ` backend:
```bash
pip install quanto  # lightweight quantization backend from HuggingFace
```

### Step 3: Expose in CLI / Test Scripts

**File:** `test_quick.py` (line ~15, add argument):
```python
parser.add_argument("--kv_cache_mode", type=str, default="auto",
                    choices=["fp16", "q8", "q4", "auto"],
                    help="KV cache precision: fp16 (default), q8 (50%% savings), q4 (75%% savings)")
```

Pass it to VaayuBaseModel constructor:
```python
model = AutoModel.from_pretrained(args.model_path, kv_cache_mode=args.kv_cache_mode, ...)
```

### Step 4: Update VRAM Budget Calculation

**File:** `vaayu/vaayu_base.py`, method `_get_vram_pinning_budget()` (line 552-566)

The current method reserves a flat 4 GB. Update it to account for the actual KV cache size:

```python
def _get_vram_pinning_budget(self) -> int:
    if not torch.cuda.is_available():
        return 0
    total = torch.cuda.get_device_properties(self.device).total_memory
    
    # Calculate KV cache reservation based on quantization mode
    # Formula: n_layers × 2 × n_kv_heads × head_dim × max_seq_len × dtype_bytes
    n_layers = getattr(self.config, 'num_hidden_layers', 80)
    n_kv_heads = getattr(self.config, 'num_key_value_heads', 8)
    head_dim = getattr(self.config, 'head_dim', 
                       self.config.hidden_size // self.config.num_attention_heads)
    
    kv_bytes_fp16 = n_layers * 2 * n_kv_heads * head_dim * self.max_seq_len * 2
    
    mode = getattr(self, 'kv_cache_mode', 'fp16')
    if mode == "q4":
        kv_reserve = kv_bytes_fp16 // 4  # 75% savings
    elif mode == "q8":
        kv_reserve = kv_bytes_fp16 // 2  # 50% savings
    else:
        kv_reserve = kv_bytes_fp16
    
    # Add 1.5 GB for CUDA context + streaming buffer + workspace
    overhead = int(1.5 * 1024**3)
    reserve = kv_reserve + overhead
    
    return max(0, total - reserve)
```

## Key Files to Modify

| File | What Changes |
|------|-------------|
| `vaayu/vaayu_base.py` | Add `kv_cache_mode` param, `_create_kv_cache()` method, update `_get_vram_pinning_budget()`, update `forward()` cache init |
| `vaayu/auto_model.py` | Pass `kv_cache_mode` through to VaayuBaseModel |
| `test_quick.py` | Add `--kv_cache_mode` CLI argument |

## Verification

1. **VRAM measurement:**
   ```bash
   # Run with FP16 cache
   python test_quick.py --model_path /path/to/model --kv_cache_mode fp16
   # Note VRAM from nvidia-smi during generation
   
   # Run with Q4 cache
   python test_quick.py --model_path /path/to/model --kv_cache_mode q4
   # VRAM should drop by ~1.9 GB
   ```

2. **Quality check:**
   - Generate a 200-token coding response with both FP16 and Q4 cache
   - Compare outputs — they should be nearly identical for coding tasks
   - Q8 should be indistinguishable from FP16

3. **OOM test:**
   - Try a long context (8,192 tokens) with Q4 cache — should not OOM
   - Same context with FP16 cache may OOM depending on pinned layer count
