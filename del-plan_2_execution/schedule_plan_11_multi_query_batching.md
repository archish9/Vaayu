# Schedule Plan 11: Multi-Query Batching

**Phase:** 4 (Advanced Capabilities)
**Priority:** Low (Implement only when acting as a multi-user API)
**Effort:** High (2 days)
**Risk:** High
**Impact:** Drastically increases throughput by applying a single 132GB NVMe read cycle to multiple independent queries simultaneously.

## Objective
Currently, Vaayu processes exactly 1 user query at a time. Serving 5 queries sequentially requires loading the 132GB model 5 separate times (5 * 23s = 115s). By batching them, we can load each layer once and execute it against the hidden states of all 5 queries, completing all requests in ~25 seconds.

## Context for the Coding Agent
The `forward` pass in `vaayu_base.py` already supports a batch dimension (`seq.shape[0]`). The heavy lifting involves padding independent prompts to the same length (or using sophisticated continuous batching / chunked prefill coordination) and maintaining distinct KV caches for each sequence in the batch.

## Implementation Steps

### Step 1: Create `vaayu/batch.py`

Create a wrapper function designed to take multiple string prompts and coordinate the batch generation.

```python
import torch

def batch_generate(
    model,
    prompts: list[str],
    max_new_tokens: int = 128,
):
    """
    Process multiple prompts sharing the same NVMe reads.
    Each layer is loaded once and applied to all prompts' hidden states.
    """
    tokenizer = model.tokenizer
    
    # Left padding is required for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)
    
    # Note: Vaayu's layer weights are identical regardless of batch size.
    # The VRAM overhead is only the KV cache (num_seqs * context_len).
    
    # Delegate to the standard model.generate, but pass the batched inputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Step 2: Ensure Padding and Masking support in `vaayu_base.py`

**File to modify:** `vaayu/vaayu_base.py`

Check the `forward` function. Ensure `attention_mask` and `position_ids` correctly handle the batch dimension. Since Vaayu inherits from `LlamaForCausalLM` structure, HuggingFace's internal generation utilities will automatically generate a 2D `attention_mask` and `position_ids` for the batched inputs.

Ensure that the VRAM budget calculation (`_get_vram_pinning_budget`) accounts for the increased KV cache size:

```python
    # In _get_vram_pinning_budget
    # If batch size is known ahead of time, multiply the KV reserve by batch_size
    # kv_reserve = base_kv_reserve * expected_max_batch_size
```

### Step 3: Handle EOS in Batching

Since some sequences will finish generating before others, Hugging Face `generate` uses the `attention_mask` to stop updating sequences that have reached the EOS token. Ensure that `vaayu_base.py` properly passes the `attention_mask` to the `decoder_layer` block:

```python
    # This is usually already handled, but verify:
    layer_outputs = layer(
        seq,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
```

## Verification Plan
1. Create a script that defines an array of 4 different prompts.
2. Call `batch_generate(model, prompts)`.
3. Check the total execution time. It should be roughly equivalent to generating a single prompt (e.g., ~24 seconds), meaning you achieved a 4x throughput multiplier.
