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

### Step 1: Add `_trim_kv_cache_front` helper

**File to modify:** `vaayu/vaayu_base.py`

The existing `_trim_kv_cache(cache, target_seq_len)` (line ~1307) trims from the **end** — it's used for speculative decoding rollback. OOM recovery needs to trim from the **front** (forget oldest tokens). Add a separate method:

```python
def _trim_kv_cache_front(self, cache, target_seq_len: int) -> None:
    """Removes the OLDEST tokens from the KV cache to free VRAM."""
    for i in range(len(cache.key_cache)):
        if cache.key_cache[i] is not None:
            cache.key_cache[i] = cache.key_cache[i][:, :, -target_seq_len:, :]
        if cache.value_cache[i] is not None:
            cache.value_cache[i] = cache.value_cache[i][:, :, -target_seq_len:, :]
```

### Step 2: Implement the Retry Loop

**File to modify:** `vaayu/vaayu_base.py`

In `forward()`, locate the decoder layer call (around line 1220+). The variable is `past_key_values` (plural — that is the name used throughout `vaayu_base.py`).

> [!CAUTION]
> Do NOT modify `seq`, `position_ids`, or `attention_mask` inside the OOM handler.
> At the point of the decoder layer call, `seq` holds **hidden states** (shape `[batch, 1, hidden_dim]`), not input token IDs. Its shape is determined by the model architecture and cannot be safely changed mid-forward-pass without corrupting all subsequent layers.
> Only the **KV cache** can be safely trimmed on OOM — it is the only VRAM structure that scales with context length at this point.

```python
        MAX_OOM_RETRIES = 2
        layer_outputs = None

        for retry in range(MAX_OOM_RETRIES + 1):
            try:
                layer_outputs = layer(
                    seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                break  # success
            except torch.cuda.OutOfMemoryError:
                if retry == MAX_OOM_RETRIES:
                    print(f"[Vaayu] ❌ Fatal OOM on {layer_name}. Recovery failed.")
                    raise  # exhausted retries — propagate

                print(f"[Vaayu] ⚠️ OOM on {layer_name} (attempt {retry+1}/{MAX_OOM_RETRIES}). "
                      f"Clearing cache and retrying...")

                torch.cuda.empty_cache()
                import gc
                gc.collect()

                # Trim KV cache only — never touch hidden states (seq) mid-loop.
                if past_key_values is not None:
                    try:
                        seq_len = self.get_past_key_values_cache_seq_len(past_key_values)
                        if seq_len > 512:
                            target_len = seq_len // 2
                            print(f"[Vaayu] 🔪 Truncating KV cache: {seq_len} → {target_len} tokens")
                            self._trim_kv_cache_front(past_key_values, target_len)
                    except Exception as e:
                        print(f"[Vaayu] Cache truncation failed: {e}")
```

## Verification Plan
1. In `test_quick.py`, artificially lower the VRAM budget (or pass a massive 15,000 token prompt).
2. Observe the console logs. It should hit the `⚠️ OOM on layer...` warning, execute the truncation, and successfully continue generating without crashing.
