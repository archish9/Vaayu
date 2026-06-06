# Schedule Plan 9: Graceful VRAM OOM Recovery

**Phase:** 3 (UX & Stability)
**Priority:** High
**Effort:** Low (3 hours)
**Risk:** Low
**Impact:** Prevents the engine from hard-crashing during long conversation loops or massive context window insertions by catching `OutOfMemoryError` and recovering safely.

## Objective
Currently, if the VRAM budget is exceeded during a forward pass (e.g., due to the KV cache ballooning from an 8,000-token prompt), PyTorch throws a `torch.cuda.OutOfMemoryError` and the entire script crashes. For an autonomous agent, this is a fatal failure. We must implement a try/catch mechanism around the GPU compute step that gracefully clears memory, truncates the context (if necessary), and retries the layer.

## Context for the Coding Agent
The layer forward pass happens in the hot loop inside `vaayu_base.py`. We need to wrap the `decoder_layer()` call in a retry loop. When an OOM occurs, we aggressively empty the PyTorch cache, trigger garbage collection, and optionally slice the `past_key_values` tensor down to a safe size before attempting the layer again.

## Implementation Steps

### Step 1: Implement the Retry Loop

**File to modify:** `vaayu/vaayu_base.py`

In `forward()`, locate the section where the decoder layer is called (around line 770+):

```python
        # Locate the layer execution block:
        # layer_outputs = layer(
        #     seq,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
```

Wrap it in a recovery mechanism:

```python
        MAX_OOM_RETRIES = 2
        layer_outputs = None

        for retry in range(MAX_OOM_RETRIES + 1):
            try:
                layer_outputs = layer(
                    seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                break  # success
            except torch.cuda.OutOfMemoryError:
                if retry == MAX_OOM_RETRIES:
                    print(f"[Vaayu] ❌ Fatal OOM on {layer_name}. Recovery failed.")
                    raise  # exhausted retries, propagate the crash
                
                print(f"[Vaayu] ⚠️ OOM on {layer_name} (attempt {retry+1}/{MAX_OOM_RETRIES}). "
                      f"Clearing cache and retrying...")
                
                # 1. Emergency Cleanup
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # 2. Context Truncation (Nuclear Option)
                # If we still OOM'd, the KV cache is likely too large. 
                # We slice the first 50% of tokens off the cache.
                if past_key_value is not None:
                    try:
                        seq_len = past_key_value.get_seq_length()
                        if seq_len > 512:
                            target_len = seq_len // 2
                            print(f"[Vaayu] 🔪 Truncating KV cache from {seq_len} to {target_len} tokens")
                            self._trim_kv_cache(past_key_value, target_len)
                            
                            # Also trim sequence and position_ids to match
                            if seq.size(1) > 1: # Only if we are in prefill
                                seq = seq[:, -target_len:, :]
                            if position_ids is not None:
                                position_ids = position_ids[:, -target_len:]
                            if attention_mask is not None:
                                attention_mask = attention_mask[:, :, -target_len:, -target_len:]
                    except Exception as e:
                        print(f"[Vaayu] Cache truncation failed: {e}")
```

### Step 2: Ensure `_trim_kv_cache` Exists

If `_trim_kv_cache` isn't fully robust for this use case (e.g. trimming from the *front* instead of the *back*), you may need to implement a front-trimming variant. `_trim_kv_cache` currently trims from the *end* (used for rolling back speculative decoding drafts). 

To trim from the front (forgetting oldest history):
```python
    def _trim_kv_cache_front(self, cache, target_seq_len: int) -> None:
        """Removes the OLDEST tokens from the cache to free VRAM."""
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i][:, :, -target_seq_len:, :]
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i][:, :, -target_seq_len:, :]
```

Update the try/catch block to use `_trim_kv_cache_front` instead.

## Verification Plan
1. In `test_quick.py`, artificially lower the VRAM budget (or pass a massive 15,000 token prompt).
2. Observe the console logs. It should hit the `⚠️ OOM on layer...` warning, execute the truncation, and successfully continue generating without crashing.
