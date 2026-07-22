# TurboQuant × Vaayu: Feasibility Analysis

## 1. What TurboQuant Is

**Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Google Research, April 2025)

TurboQuant is a **data-oblivious, online vector quantization** algorithm that achieves near-optimal MSE distortion (within ≈2.7× of the Shannon lower bound). It has two variants:

### TurboQuant_MSE (for reconstruction quality)

```
Quant(x):
  1. Normalize: x̂ = x / ‖x‖,  store ‖x‖
  2. Random rotate: y = Π · x̂    (Π is a fixed random orthogonal matrix)
  3. For each coordinate y_j, find nearest centroid in a precomputed
     Lloyd-Max codebook (optimal scalar quantizer for Beta distribution)
  4. Store: b-bit index per coordinate + ‖x‖ scalar

DeQuant(indices, norm):
  1. Look up centroids from codebook
  2. Rotate back: x̃ = Π^T · centroids_vector
  3. Rescale: x̃ *= norm
```

### TurboQuant_prod (for unbiased inner products — what attention needs)

```
Quant(x):
  1. Apply TurboQuant_MSE at (b-1) bits → get reconstruction x̃_mse
  2. Compute residual r = x - x̃_mse,  store γ = ‖r‖
  3. Apply 1-bit QJL on normalized residual: sign(S · r/γ)
  4. Store: (b-1)-bit indices + 1-bit QJL signs + γ scalar

DeQuant(indices, qjl_signs, γ):
  1. x̃_mse = DeQuant_MSE(indices)
  2. x̃_qjl = γ · √(π/2) · S^T · qjl_signs / d
  3. Return x̃ = x̃_mse + x̃_qjl
```

### Key Results

| Bit-width | MSE Distortion | KV Cache Quality | Compression vs FP16 |
|---|---|---|---|
| 2.5 bits | ~0.08 | Marginal degradation | **6.4×** |
| 3.5 bits | ~0.02 | **Quality neutral** | **4.6×** |
| 4 bits | 0.009 | Lossless | 4× |
| 8 bits (FP8, Vaayu today) | ~0.0002 | Lossless | 2× |

### Why It Matters for Vaayu

Vaayu's current FP8 KV cache gives **2× compression** over FP16. TurboQuant at 3.5 bits gives **4.6× compression** with quality neutrality — that's **2.3× more VRAM savings** on top of what FP8 already provides.

For the 16 GB GPU target:
- **FP8 today**: ~1.38 GB KV cache at 8k context (Llama-70B geometry)
- **TurboQuant 3.5b**: ~0.60 GB — freeing **~780 MB** for more pinned layers or longer context
- **TurboQuant 2.5b**: ~0.43 GB — freeing **~950 MB**

---

## 2. Can You Implement This in Vaayu?

**Yes, absolutely.** TurboQuant is designed specifically for online KV cache quantization and maps cleanly onto Vaayu's architecture. Here's how:

### Where It Plugs In

The sole integration point is [FP8DynamicCache.update()](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py#L472-L495). Today it does:

```python
# Current FP8 scheme (per-head per-token scalar quantization)
k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
k_new_out = k_fp8.to(key_states.dtype) * k_scale  # dequantized for attention
```

TurboQuant replaces this with:

```python
# TurboQuant scheme (random rotation + optimal scalar quantization)
norm = key_states.norm(dim=-1, keepdim=True)          # store the norm
y = torch.matmul(key_states / norm, Pi.T)              # rotate (Π precomputed)
indices = codebook_lookup(y, centroids)                 # b-bit index per coord
# For inner-product variant: + QJL on residual
```

### Implementation Mapping

| Paper Concept | Vaayu Component | Implementation |
|---|---|---|
| Random rotation matrix Π | New: `self._rotation_matrix` on `FP8DynamicCache` or the model | Precomputed once at init, `[head_dim, head_dim]` orthogonal matrix via `torch.linalg.qr(randn(...))`. Shared across layers (data-oblivious). ~64KB for head_dim=128 |
| Lloyd-Max codebook centroids | New: `self._codebook` | Precomputed offline for target bit-width. Small lookup table: e.g., 2^3 = 8 centroids for 3-bit. Stored as a 1-D tensor |
| Codebook index storage | Replaces: `key_cache` (fp8) | Integer indices, b bits per coordinate. Packed into uint8/uint16 |
| Norm storage | Replaces: `_key_scales` | One scalar per token (vs current per-head-per-token scale) |
| QJL sign bits (inner-product variant) | New: `_key_qjl_signs` | 1 bit per coordinate, stored as packed bits |
| QJL random matrix S | New: `self._qjl_matrix` | `[head_dim, head_dim]` Gaussian random matrix. Shared. ~64KB |
| Dequantized output cache | Same role as: `_k_out_cache` | Reconstructed on the fly or cached as today |

### Computational Cost (Per Token, Per Layer)

| Operation | Current FP8 | TurboQuant_MSE | TurboQuant_prod |
|---|---|---|---|
| Quantize | abs + div + clamp + cast | matmul(Π) + nearest-centroid lookup | matmul(Π) + lookup + matmul(S) + sign |
| Dequantize | cast + multiply | centroid lookup + matmul(Π^T) | lookup + matmul(Π^T) + matmul(S^T) + add |
| Dominant cost | Elementwise (cheap) | Matrix-vector multiply (head_dim × head_dim) | Two matrix-vector multiplies |

> [!IMPORTANT]
> **The rotation matrix multiply is the main new cost.** For `head_dim=128`, this is a 128×128 matmul per head per token — on GPU this is ~16K FLOPs, negligible compared to the attention computation itself (~head_dim × seq_len FLOPs). The codebook lookup is a simple `torch.bucketize` or `torch.searchsorted` — O(head_dim × log(2^b)).

### Practical Considerations

1. **Randomized quantizer**: TurboQuant is stochastic (the rotation matrix Π is random). This means two quantizations of the same vector produce different indices. This is fine for KV cache (each token is quantized exactly once in `update()`), but means the **disk persistence format** needs updating.

2. **Rotation matrix sharing**: The paper uses one Π per quantizer instance. For Vaayu, you'd share one Π across all layers (same `head_dim`), stored once at model init. The QJL matrix S is separate.

3. **Bit-packing**: At 3 bits per coordinate, you'd pack indices into bytes. PyTorch doesn't have native 3-bit dtypes, so you'd use `uint8` with 2 coordinates packed per byte (wasting 2 bits) or use custom bit-packing via bitwise ops.

4. **Outlier channels**: The paper's best results (2.5/3.5 bits) split channels into outlier/non-outlier groups with different bit-widths. This maps to per-head configuration in Vaayu.

---

## 3. What Existing KV Cache Features Become Redundant?

### ✅ Features that STAY (all of them)

| Feature | Why it stays |
|---|---|
| **Ring buffer design** (geometric doubling, `_seq_len` tracking) | Buffer management is orthogonal to quantization scheme. Still needed for amortized O(1) append |
| **`crop()` / `crop_front()`** | Sequence truncation is about which tokens to keep, not how they're stored |
| **`to_cpu()` / `to_device()`** | Cross-turn mobility between GPU and RAM. Works the same regardless of quantization |
| **`resident_bytes()`** | Budget accounting. Would return *smaller* numbers with TurboQuant |
| **PrefixCacheStore** (RAM pool, LRU, checkout) | Session management layer. Quantization-agnostic |
| **KV disk persistence** (`kv_persist.py`) | Architecture stays. Format changes (new `QUANT_SIG`, different record layout) |
| **KVCacheManager** | Save/restore protocol stays. Internal representation changes |
| **OOM recovery** (`_trim_kv_cache_front`) | Still needed. OOM is about total VRAM pressure, not quantization scheme |
| **Chunked prefill** | VRAM spike flattening. Orthogonal |
| **VRAM pinning budget** (`_get_vram_pinning_budget`) | Still needed, but with smaller KV reservation → more pinned layers! |
| **Speculative decode trim** | Cache trimming after verification. Orthogonal |
| **Batched decode** | Shared cache for N streams. Orthogonal |

### ⚠️ Features that CHANGE (internal representation)

| Current component | What changes |
|---|---|
| **`key_cache` / `value_cache` (fp8 tensors)** | Replaced by **codebook index tensors** (uint8/uint16, b bits per coord) |
| **`_key_scales` / `_value_scales` (per-head-per-token)** | Replaced by **per-token norms** (one float per token, not per head). This is a simplification — 1 scalar vs H scalars per token |
| **`_k_out_cache` / `_v_out_cache` (dequantized bf16)** | Still needed (attention requires dequantized K/V). But could be computed on-the-fly instead of cached, saving 2× the dequantized buffer VRAM |
| **`_QUANT_SIG` in kv_persist.py** | New signature for TurboQuant's scheme (different record byte layout) |
| **`_create_kv_cache()` mode selection** | New mode: `"turboquant"` or `"tq3"` / `"tq3.5"` alongside existing `"fp8"` / `"fp16"` |

### ❌ Nothing Becomes Completely Unnecessary

No existing KV cache *feature* is eliminated by TurboQuant. TurboQuant replaces the **quantization scheme inside** `FP8DynamicCache`, not the cache management infrastructure around it. The scale factor storage shrinks from `[B, H, seq, 1]` to `[B, 1, seq, 1]` (per-token vs per-head-per-token), but the buffer still exists.

The **dequantized output caches** (`_k_out_cache`, `_v_out_cache`) are debatable: you *could* remove them and dequantize on-the-fly during attention, trading compute for VRAM. But this is an optimization choice, not a consequence of TurboQuant.

---

## 4. Summary & Recommendation

| Question | Answer |
|---|---|
| **Can you implement TurboQuant in Vaayu?** | **Yes.** It's a drop-in replacement for the quantization path inside `FP8DynamicCache.update()`. The surrounding infrastructure (ring buffer, prefix cache, disk persist, OOM recovery) is orthogonal and stays |
| **Is any KV feature not needed?** | **No feature is eliminated.** The per-head-per-token scale factors (`_key_scales`) *simplify* to per-token norms, and the raw storage format changes from fp8 tensors to codebook indices, but every architectural feature remains necessary |
| **What's the win?** | **2.3× more VRAM savings** over current FP8 (at 3.5-bit quality-neutral). Translates to ~780 MB freed on the 16 GB GPU → ~13 more pinned transformer layers or ~2× longer context |
| **What's the risk?** | The rotation matmul adds ~16K FLOPs per head per token per layer (negligible). The real engineering cost is bit-packing for non-byte-aligned widths (3 bits) and updating `kv_persist.py`'s binary format |

> [!TIP]
> **Recommended approach**: Implement TurboQuant as a new `kv_cache_mode="turboquant"` alongside the existing `"fp8"` and `"fp16"` modes. Create a `TurboQuantDynamicCache` class that extends the same interface (`update`, `crop`, `to_cpu`, etc.) but stores codebook indices + norms internally. This keeps the existing FP8 path as a proven fallback and lets you A/B test quality on real workloads.
