# Plan 7: INT8 KV Cache Snapshot Compression

**Expected benefit: 50% snapshot RAM reduction; enables 2x longer conversation history**  
**Difficulty: Medium — ~60 lines**  
**Independent: Yes — only touches `kv_cache_manager.py`**  
**Spec:** `docs/superpowers/specs/2026-05-24-int8-kv-cache-design.md`

---

## Context

`KVCacheManager` snapshots the full KV state between turns. At BF16, a 48-layer 14B model with 2048-token context uses ~6 GB RAM for the snapshot alone. INT8 quantization of the snapshot halves this to ~3 GB, freeing RAM for more pinned decoder layers (Plan 5) or longer contexts.

Quantization applies **only to snapshot storage**. The live KV cache during forward passes stays in full precision — quantizing there would slow every attention computation.

---

## Changes

### Change 1: Add `compress_kv` parameter to `KVCacheManager.__init__`

```python
def __init__(self, model, compress_kv: bool = False):
    self.model = model
    self.compress_kv = compress_kv
    self._cached_input_ids = None
    self._cached_kv_snapshot = None
```

### Change 2: Add INT8 quantization helpers

Add to `kv_cache_manager.py` (before the class definition):

```python
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


# MLX variants
def _quantize_mx(arr):
    """Symmetric per-tensor INT8 quantization for mx.array."""
    import mlx.core as mx
    scale = float(mx.abs(arr).max().item()) / 127.0
    if scale == 0.0:
        return arr.astype(mx.int8), 1.0
    q = mx.round(arr / scale).clip(-128, 127).astype(mx.int8)
    mx.eval(q)
    return q, scale


def _dequantize_mx(q, scale, dtype):
    """Restore INT8 mx.array to original dtype."""
    import mlx.core as mx
    return q.astype(dtype) * scale
```

### Change 3: Compress MPS snapshot in `_clone_dynamic_cache()`

Replace current method:

```python
def _clone_dynamic_cache(self, cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        if self.compress_kv:
            # Store compressed; dequantize on restore
            new_cache.key_cache.append(('int8', *_quantize_tensor(k), k.dtype))
            new_cache.value_cache.append(('int8', *_quantize_tensor(v), v.dtype))
        else:
            new_cache.update(k.clone(), v.clone(), layer_idx)
    return new_cache
```

Add a `_decompress_dynamic_cache()` helper that restores INT8 entries:

```python
def _decompress_dynamic_cache(self, cache: DynamicCache) -> DynamicCache:
    """Dequantize a compressed snapshot back to a usable DynamicCache."""
    if not self.compress_kv:
        return cache
    out = DynamicCache()
    for layer_idx, (k_entry, v_entry) in enumerate(
        zip(cache.key_cache, cache.value_cache)
    ):
        if isinstance(k_entry, tuple) and k_entry[0] == 'int8':
            _, q_k, scale_k, dtype_k = k_entry
            _, q_v, scale_v, dtype_v = v_entry
            k = _dequantize_tensor(q_k, scale_k, dtype_k)
            v = _dequantize_tensor(q_v, scale_v, dtype_v)
        else:
            k, v = k_entry, v_entry
        out.update(k, v, layer_idx)
    return out
```

Call `_decompress_dynamic_cache()` in `generate()` before passing to model:

```python
if past_kv is not None and self.compress_kv:
    past_kv = self._decompress_dynamic_cache(past_kv)
```

### Change 4: Compress MLX snapshot in `_save_mlx_snapshot()`

```python
def _save_mlx_snapshot(self) -> dict:
    import mlx.core as mx
    from vaayu_apple.mlx_support.kv_cache import _MLXCacheSentinel

    backend = self.model.backend
    layers = []
    for cache in backend._kv_caches:
        if cache.keys is not None:
            if self.compress_kv:
                q_k, s_k = _quantize_mx(cache.keys)
                q_v, s_v = _quantize_mx(cache.values)
                orig_dtype = cache.keys.dtype
                layers.append(('int8', q_k, s_k, q_v, s_v, orig_dtype))
            else:
                k = mx.array(cache.keys); v = mx.array(cache.values)
                mx.eval(k, v)
                layers.append((k, v))
        else:
            layers.append((None, None))

    return {
        "layers": layers,
        "seq_len": backend._current_seq_len,
        "sentinel": _MLXCacheSentinel(backend._current_seq_len),
        "compressed": self.compress_kv,
    }
```

Update `_restore_mlx_snapshot()` to dequantize:

```python
def _restore_mlx_snapshot(self, snapshot: dict):
    backend = self.model.backend
    compressed = snapshot.get("compressed", False)
    for cache, entry in zip(backend._kv_caches, snapshot["layers"]):
        if entry[0] is None:
            cache.keys = None; cache.values = None
        elif compressed and entry[0] == 'int8':
            _, q_k, s_k, q_v, s_v, dtype = entry
            cache.keys = _dequantize_mx(q_k, s_k, dtype)
            cache.values = _dequantize_mx(q_v, s_v, dtype)
        else:
            cache.keys, cache.values = entry
    backend._current_seq_len = snapshot["seq_len"]
```

---

## Verification

```python
import psutil

agent_compressed = KVCacheManager(model, compress_kv=True)
agent_normal     = KVCacheManager(model, compress_kv=False)

# Turn 1
r1_c = agent_compressed.generate([{"role":"user","content":"What is Python?"}], max_new_tokens=50)
r1_n = agent_normal.generate([{"role":"user","content":"What is Python?"}], max_new_tokens=50)

# Check RAM after snapshot saved
print(f"Compressed snapshot size (check psutil or len(snapshot))")
print(f"Normal vs compressed output similar: {r1_c[:50]} | {r1_n[:50]}")

# Turn 2 — KV cache reuse
r2_c = agent_compressed.generate([
    {"role":"user","content":"What is Python?"},
    {"role":"assistant","content":r1_c},
    {"role":"user","content":"And Java?"},
], max_new_tokens=50)
# Should print: [KV-Cache] reusing N cached tokens
```

---

## Files Touched

| File | Lines Changed |
|---|---|
| `vaayu_apple/kv_cache_manager.py` | ~60 lines — helpers + updated save/restore/clone methods |
