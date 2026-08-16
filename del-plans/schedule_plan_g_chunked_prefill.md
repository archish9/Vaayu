# Schedule Plan G: Chunked Prefill (Context Streaming)

**Priority:** 4 (After Plans E & F)
**Effort:** Medium
**Risk:** Low
**Benefit:** Eliminates OOM crashes on large prompts (4,000+ tokens)

## Objective
Process large input prompts in chunks of 512 tokens instead of all at once. This flattens the temporary VRAM spike during prefill, preventing Out-Of-Memory crashes when users paste large codebases.

## The Problem
When a user pastes a 2,000-line code file (4,000+ tokens), standard PyTorch processes all prompt tokens in one massive matrix multiplication. For a 70B model, this requires a huge temporary workspace in VRAM. On a 16 GB GPU with ~12 GB already pinned (Plan F), the remaining 4 GB cannot absorb this spike → **OOM crash before the first word is generated**.

## Current Code

### `forward()` in `vaayu/vaayu_base.py` (line 848)

The forward pass processes `input_ids` as a single batch. During prefill, the entire prompt is passed through each layer at once:

```python
# Line 872 — batch creation
batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]

# Line 1001 — layer execution processes the entire sequence at once
for j, seq in enumerate(batch):
    if layer_name == self.layer_names_dict['embed']:
        batch[j] = layer(seq)
    # ... processes full seq through each layer
```

The KV cache grows incrementally per layer (line 1018-1023), but the workspace memory for computing attention scales with `O(seq_len²)` during a single forward pass.

## Implementation Steps

### Step 1: Check if ExLlamaV3 Already Handles Chunking

Before writing custom code, check ExLlamaV3's generation API:

```python
import exllamav3
# Look for chunk-related parameters
help(exllamav3.ExLlamaV3)  # or ExLlamaV3Model
# Look for: max_chunk_size, chunk_size, max_input_len, etc.
```

**If ExLlamaV3 has built-in chunking** → just set the config parameter and skip the rest of this plan. 

**If not** → continue with Step 2.

### Step 2: Add `prefill_chunk_size` Parameter

**File:** `vaayu/vaayu_base.py`, `__init__()` (line 112)

Add to signature:
```python
prefill_chunk_size=512,  # NEW: tokens per prefill chunk (0 = no chunking)
```
Store as `self.prefill_chunk_size = prefill_chunk_size`.

### Step 3: Implement Chunked Prefill in forward()

**File:** `vaayu/vaayu_base.py`

Add a new method before `forward()`:

```python
def _chunked_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=True,
    **kwargs,
):
    """
    Process a long prompt in chunks to flatten VRAM spikes.
    
    Instead of processing all 4,000 tokens at once:
      1. Process tokens [0:512], build KV cache for those
      2. Process tokens [512:1024], extend KV cache
      3. ... repeat until all prompt tokens are cached
      4. Return the KV cache for autoregressive generation
    """
    chunk_size = self.prefill_chunk_size
    total_len = input_ids.shape[1]
    
    if chunk_size <= 0 or total_len <= chunk_size:
        # No chunking needed — fall through to standard forward
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
    
    print(f"[Vaayu] Chunked prefill: {total_len} tokens in "
          f"{(total_len + chunk_size - 1) // chunk_size} chunks of {chunk_size}")
    
    output = None
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        chunk_ids = input_ids[:, start:end]
        
        # Build attention mask for this chunk + all previous
        chunk_mask = None
        if attention_mask is not None:
            chunk_mask = attention_mask[:, :end]
        
        output = self.forward(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs,
        )
        
        # Extract updated KV cache for next chunk
        past_key_values = output.past_key_values
        
        print(f"[Vaayu] Prefill chunk [{start}:{end}] done, "
              f"KV cache seq_len={self.get_past_key_values_cache_seq_len(past_key_values)}")
    
    return output
```

### Step 4: Hook into Generation Pipeline

**File:** `vaayu/vaayu_base.py`

Modify `prepare_inputs_for_generation()` (line 724) or add a wrapper. The cleanest approach is to override in the generation entry point.

If using HuggingFace's `model.generate()`:
```python
# In the test script or wherever generate() is called:
if input_ids.shape[1] > model.prefill_chunk_size > 0:
    # Pre-fill the KV cache in chunks
    prefill_output = model._chunked_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    # Now generate autoregressively using the filled cache
    output = model.generate(
        input_ids=input_ids[:, -1:],  # only last token
        attention_mask=attention_mask,
        past_key_values=prefill_output.past_key_values,
        max_new_tokens=max_new_tokens,
    )
else:
    output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
```

### Step 5: Monitor Peak VRAM

Add VRAM logging to the chunked forward:
```python
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Vaayu] Peak VRAM after chunk: {peak:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
```

## Key Files to Modify

| File | What Changes |
|------|-------------|
| `vaayu/vaayu_base.py` | Add `prefill_chunk_size` param, add `_chunked_forward()` method |
| `vaayu/auto_model.py` | Pass `prefill_chunk_size` through |
| `test_quick.py` | Add `--prefill_chunk_size` CLI arg, use `_chunked_forward()` for long prompts |

## Verification

1. **OOM test:** Feed an 8,192-token prompt with chunking disabled (should OOM) vs enabled (should succeed)
2. **Peak VRAM:** Monitor `torch.cuda.max_memory_allocated()` — should stay flat across chunks instead of spiking
3. **Output quality:** Compare output of chunked vs non-chunked on a 1,000-token prompt — should be identical (chunking is mathematically lossless)
4. **Speed:** Chunked prefill is ~10-15% slower than non-chunked (due to multiple passes). This is the expected tradeoff for OOM safety.
