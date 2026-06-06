# Schedule Plan 10: Multi-Turn KV Cache Compression (INT8)

**Phase:** 4 (Advanced Capabilities)
**Priority:** Low (Implement only when building a chat server/wrapper)
**Effort:** Medium
**Risk:** Low
**Impact:** Halves the system RAM footprint required to store idle conversation histories between multi-turn user requests.

## Objective
In an autonomous agent loop or a chat UI, the user will send multiple back-and-forth messages. Re-evaluating the entire conversation history from scratch via NVMe streaming is prohibitively slow. Instead, a `KVCacheManager` wrapper can save the KV cache snapshot to system RAM when the model is idle. Since a large context can consume 6GB+ of RAM, we will quantize this *idle* snapshot to INT8 to compress it to 3GB, and dequantize it back to PyTorch floats when the user sends the next message.

## Context for the Coding Agent
This feature does **NOT** modify the active `FP8DynamicCache` used during `model.forward()`. The FP8 cache is hardware-accelerated for compute. The INT8 cache is strictly a storage format for idle state in system memory.

## Implementation Steps

### Step 1: Create `vaayu/kv_cache_manager.py`

Create a wrapper class to handle snapshotting.

```python
import torch
from transformers import DynamicCache

def _quantize_tensor(t):
    """Symmetric per-tensor INT8 quantization. Returns (int8_tensor, float_scale)."""
    scale = t.abs().max().item() / 127.0
    if scale == 0.0:
        return t.to(torch.int8), 1.0
    q = (t / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale

def _dequantize_tensor(q, scale, dtype):
    """Restore INT8 tensor to original dtype."""
    return q.to(dtype) * scale

class KVCacheManager:
    def __init__(self, model, compress_kv: bool = True):
        self.model = model
        self.compress_kv = compress_kv
        self._cached_kv_snapshot = None

    def save_snapshot(self, past_key_values: DynamicCache):
        """Clones the active cache into system RAM, optionally compressing to INT8."""
        # NOTE: When compress_kv=True, we directly append tuples to .key_cache / .value_cache
        # instead of calling .update(). This bypasses DynamicCache's internal validation but
        # is safe here because this snapshot is never passed back to a model forward pass in
        # compressed form — it is always decompressed via load_snapshot() first.
        new_cache = DynamicCache()
        for layer_idx in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]

            # Move to CPU
            k_cpu = k.cpu()
            v_cpu = v.cpu()

            if self.compress_kv:
                new_cache.key_cache.append(('int8', *_quantize_tensor(k_cpu), k_cpu.dtype))
                new_cache.value_cache.append(('int8', *_quantize_tensor(v_cpu), v_cpu.dtype))
            else:
                new_cache.update(k_cpu, v_cpu, layer_idx)
                
        self._cached_kv_snapshot = new_cache

    def load_snapshot(self, device="cuda") -> DynamicCache:
        """Restores the snapshot back to the GPU for the next generation turn."""
        if self._cached_kv_snapshot is None:
            return None
            
        out = DynamicCache()
        for layer_idx, (k_entry, v_entry) in enumerate(
            zip(self._cached_kv_snapshot.key_cache, self._cached_kv_snapshot.value_cache)
        ):
            if isinstance(k_entry, tuple) and k_entry[0] == 'int8':
                _, q_k, scale_k, dtype_k = k_entry
                _, q_v, scale_v, dtype_v = v_entry
                k = _dequantize_tensor(q_k, scale_k, dtype_k).to(device)
                v = _dequantize_tensor(q_v, scale_v, dtype_v).to(device)
            else:
                k = k_entry.to(device)
                v = v_entry.to(device)
                
            out.update(k, v, layer_idx)
        return out
```

### Step 2: Use Manager in the Agent Loop

When writing your autonomous agent `main.py` or FastAPI server:

```python
manager = KVCacheManager(model, compress_kv=True)

# Turn 1
outputs = model.generate(input_ids, use_cache=True, return_dict_in_generate=True)
manager.save_snapshot(outputs.past_key_values)

# Turn 2
past_kv = manager.load_snapshot(device="cuda")
# Provide only the NEW user tokens to the model, passing past_kv
outputs = model.generate(new_input_ids, past_key_values=past_kv, use_cache=True)
manager.save_snapshot(outputs.past_key_values)
```

## Verification Plan
1. Create a script that generates 100 tokens, calls `save_snapshot`, and then measures `sys.getsizeof()` or uses `psutil` to check system RAM consumption.
2. Verify that `compress_kv=True` results in exactly half the RAM usage of `compress_kv=False`.
3. Verify that generating the next turn using the restored snapshot produces identical logic and resumes generation correctly without delay.
