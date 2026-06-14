import torch
from transformers import DynamicCache


def _quantize_tensor(t: torch.Tensor):
    """
    Symmetric per-tensor INT8 quantization.
    Input must be a float tensor (BF16, FP16, FP32).
    FP8DynamicCache stores FP8 tensors — dequantize those first before calling this.
    Returns (int8_tensor, float_scale).
    """
    t_f32 = t.to(torch.float32)
    scale = t_f32.abs().max().item() / 127.0
    if scale == 0.0:
        return t_f32.to(torch.int8), 1.0
    q = (t_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _dequantize_tensor(q: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
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
        ks = cache._key_scales[layer_idx]
        vs = cache._value_scales[layer_idx]
        k = k.to(torch.bfloat16) * ks
        v = v.to(torch.bfloat16) * vs
    return k.cpu(), v.cpu()


class KVCacheManager:
    """
    Saves and restores KV cache snapshots between multi-turn requests.

    Usage:
        manager = KVCacheManager(model, compress_kv=True)

        # Turn 1
        outputs = model.generate(input_ids, use_cache=True, return_dict_in_generate=True)
        manager.save_snapshot(outputs.past_key_values)

        # Turn 2
        past_kv = manager.load_snapshot(device="cuda")
        outputs = model.generate(new_input_ids, past_key_values=past_kv, use_cache=True)
        manager.save_snapshot(outputs.past_key_values)
    """

    def __init__(self, model, compress_kv: bool = True):
        self.model = model
        self.compress_kv = compress_kv
        # Schema (uncompressed):   {layer_idx: {'k': tensor, 'v': tensor}}
        # Schema (compressed INT8): {layer_idx: {'k_q': int8, 'k_s': float, 'v_q': int8, 'v_s': float, 'dtype': dtype}}
        self._snapshot: dict | None = None

    def save_snapshot(self, past_key_values) -> None:
        """Clone the active cache into system RAM, optionally compressing to INT8."""
        snap = {}
        for layer_idx in range(len(past_key_values.key_cache)):
            k_cpu, v_cpu = _extract_kv_as_bf16(past_key_values, layer_idx)
            orig_dtype = k_cpu.dtype

            if self.compress_kv:
                q_k, s_k = _quantize_tensor(k_cpu)
                q_v, s_v = _quantize_tensor(v_cpu)
                snap[layer_idx] = {
                    'k_q': q_k, 'k_s': s_k,
                    'v_q': q_v, 'v_s': s_v,
                    'dtype': orig_dtype,
                }
            else:
                snap[layer_idx] = {'k': k_cpu, 'v': v_cpu}

        self._snapshot = snap

    def load_snapshot(self, device: str = "cuda") -> DynamicCache | None:
        """Restore the snapshot back to the GPU as a standard DynamicCache."""
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

    def clear(self) -> None:
        """Free the in-RAM snapshot."""
        self._snapshot = None
