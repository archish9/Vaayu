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

def _quantize_tensor(t: torch.Tensor):
    """
    Symmetric per-tensor INT8 quantization.
    Input must be a float tensor (BF16, FP16, FP32).
    FP8DynamicCache stores FP8 tensors — dequantize those first before calling this.
    Returns (int8_tensor, float_scale).
    """
    # Upcast to float32 for safe abs/max — handles BF16, FP16, FP32
    t_f32 = t.to(torch.float32)
    scale = t_f32.abs().max().item() / 127.0
    if scale == 0.0:
        return t_f32.to(torch.int8), 1.0
    q = (t_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale

def _dequantize_tensor(q, scale, dtype):
    """Restore INT8 tensor to original dtype."""
    return q.to(torch.float32).mul_(scale).to(dtype)

def _extract_kv_as_bf16(cache, layer_idx: int):
    """
    Extract key/value tensors from cache as BF16 CPU tensors.
    Handles both FP8DynamicCache (needs dequantization) and standard DynamicCache.
    """
    from vaayu.vaayu_base import FP8DynamicCache
    k = cache.key_cache[layer_idx]
    v = cache.value_cache[layer_idx]
    if isinstance(cache, FP8DynamicCache):
        # FP8DynamicCache stores FP8 tensors + separate scales.
        # abs().max() fails on float8 — must dequantize to BF16 first.
        ks = cache._key_scales[layer_idx]
        vs = cache._value_scales[layer_idx]
        k = k.to(torch.bfloat16) * ks
        v = v.to(torch.bfloat16) * vs
    return k.cpu(), v.cpu()

class KVCacheManager:
    def __init__(self, model, compress_kv: bool = True):
        self.model = model
        self.compress_kv = compress_kv
        # Use a plain dict for the snapshot — more robust than mutating DynamicCache internals.
        # Schema: {layer_idx: {'k': tensor, 'v': tensor}}
        # or compressed: {layer_idx: {'k_q': int8, 'k_s': float, 'v_q': int8, 'v_s': float, 'dtype': dtype}}
        self._snapshot: dict | None = None

    def save_snapshot(self, past_key_values):
        """Clones the active cache into system RAM, optionally compressing to INT8."""
        snap = {}
        for layer_idx in range(len(past_key_values.key_cache)):
            k_cpu, v_cpu = _extract_kv_as_bf16(past_key_values, layer_idx)
            orig_dtype = k_cpu.dtype

            if self.compress_kv:
                q_k, s_k = _quantize_tensor(k_cpu)
                q_v, s_v = _quantize_tensor(v_cpu)
                snap[layer_idx] = {'k_q': q_k, 'k_s': s_k, 'v_q': q_v, 'v_s': s_v, 'dtype': orig_dtype}
            else:
                snap[layer_idx] = {'k': k_cpu, 'v': v_cpu}

        self._snapshot = snap

    def load_snapshot(self, device="cuda") -> DynamicCache:
        """Restores the snapshot back to the GPU as a standard DynamicCache."""
        if self._snapshot is None:
            return None

        out = DynamicCache()
        for layer_idx in sorted(self._snapshot.keys()):
            entry = self._snapshot[layer_idx]
            if 'k_q' in entry:
                k = _dequantize_tensor(entry['k_q'], entry['k_s'], entry['dtype']).to(device)
                v = _dequantize_tensor(entry['v_q'], entry['v_s'], entry['dtype']).to(device)
            else:
                k = entry['k'].to(device)
                v = entry['v'].to(device)
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
