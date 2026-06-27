# Hybrid MoE Expert Compute: CPU FFN (BF16) + EXL3 GPU Sparse Loading

**Date:** 2026-06-27  
**Status:** Approved  
**Context:** `docs/superpowers/specs/2026-06-14-moe-expert-aware-streaming-design.md`

---

## Problem

Vaayu MoE expert-aware streaming is correct but still slow on PCIe Gen1 x8 (~2 GB/s H2D).
For full-precision BF16 Mixtral 8x7B: each expert = 336 MB → 168 ms PCIe transfer.
Decode token (2 unique experts × 32 layers): 2 × 32 × 168 ms = **10.7 s/token minimum**, even with 100% cache hit rate.

Root cause: PCIe is the bottleneck, not disk. Expert weights are in CPU RAM (cache hit) but must still cross PCIe to reach GPU for compute.

---

## Solution Overview

Two compute paths, selected at model init time based on model quantization format:

| Model format | Expert size | Strategy | Expected decode |
|---|---|---|---|
| BF16 / F16 full precision | 336 MB | CPU FFN (compute in RAM, zero PCIe for weights) | ~0.5–1.5 s/token |
| EXL3 quantized (4.5 bpw) | ~88 MB | GPU FFN with EXL3 sparse loading | ~3–4 s/token |

Both paths share: `ExpertRAMCache`, `ExpertOffsetIndex`, io_uring sparse loading infrastructure.

---

## Component 1: Detection

In `patch_moe_blocks()` (`vaayu/moe_patch.py`):

```python
from .persist.loader_strategy import EXL3LoaderStrategy
is_exl3 = isinstance(loader_strategy, EXL3LoaderStrategy)
compute_device = "cuda" if is_exl3 else "cpu"
```

`compute_device` is captured in the closure of `_make_fused_sparse_forward()`.

`patch_moe_blocks()` gains a new `loader_strategy` parameter (passed from `vaayu_base.py`).

---

## Component 2: CPU FFN Path (BF16/F16)

Activated when `compute_device == "cpu"`.

Modified `_make_fused_sparse_forward()` in `vaayu/moe_patch.py`:

```python
# hidden_states: [B, S, H] on GPU
flat_hidden = hidden_states.view(-1, H)

# Step 1: tiny GPU→CPU transfer (decode: 8 KB, prefill-512: 4 MB)
cpu_hidden = flat_hidden.cpu()
cpu_weights_tensor = routing_weights.cpu()

# Step 2: expert loop on CPU, weights already in RAM
final_cpu = torch.zeros(B * S, H, dtype=cpu_hidden.dtype)
for eidx in unique_experts:
    raw = get_expert(layer_idx, eidx)          # ExpertRAMCache → disk fallback
    w1 = raw[w1_key]                            # BF16, already on CPU
    w2 = raw[w2_key]
    w3 = raw[w3_key]
    
    top_k_pos, token_idx = torch.where(expert_mask[eidx])
    if token_idx.numel() == 0:
        continue
    x = cpu_hidden[token_idx]
    
    gate = act_fn_cpu(F.linear(x, w1))          # CPU GEMV/GEMM
    up   = F.linear(x, w3)
    out  = F.linear(gate * up, w2)
    
    out = out * cpu_weights_tensor[token_idx, top_k_pos, None]
    final_cpu.index_add_(0, token_idx, out)

# Step 3: tiny CPU→GPU transfer (same shape as input hidden_states)
return final_cpu.to(device, dtype=dtype).view(B, S, H)
```

**Performance model (decode, BF16, 32GB RAM @ 40 GB/s GEMV bandwidth):**
- GPU→CPU hidden: 1×4096×2 bytes = 8 KB → negligible
- CPU GEMM per expert (w1: 14336×4096 BF16 = 112 MB read, ~2.8 ms at 40 GB/s)
- 2 active experts × 32 layers = 64 expert calls × ~3 ms = **~0.2 s** compute
- CPU→GPU result: 8 KB → negligible
- **Estimated decode: ~0.3–0.5 s/token** (compute-bound on CPU RAM bandwidth)

**Prefill (512 tokens):** same structure; GEMM is bigger but still faster than 168 ms × 16 PCIe transfers.

**Key constraint:** `act_fn` must run on CPU. `SiLU` is available in `torch.nn.functional` on CPU. Store a CPU-side activation function reference at closure creation time.

---

## Component 3: EXL3 Expert Sparse Loading (GPU Path)

Activated when `compute_device == "cuda"`.

### 3a. ExpertOffsetIndex — no changes needed

Existing regex `\.experts\.(\d+)\.` already matches EXL3 checkpoint keys:
```
model.layers.0.block_sparse_moe.experts.0.w1.trellis
model.layers.0.block_sparse_moe.experts.0.w1.mcg
model.layers.0.block_sparse_moe.experts.0.w1.suh
...
```
EXL3 adds more tensors per expert (7 keys × 3 weight matrices = 21 tensors vs 3 for BF16), but indexing logic is unchanged.

### 3b. ExpertRAMCache — no changes needed

Stores raw tensor dicts. EXL3 tensors are just tensors; cache is format-agnostic.

### 3c. `load_expert_sparse()` — no changes needed

Reads byte ranges by offset; returns `{key: tensor}` dict. Works for EXL3 tensor dtypes (int8, uint8, float16 depending on EXL3 component).

### 3d. `_make_fused_sparse_forward_exl3()` — NEW function in `moe_patch.py`

For each active expert:
1. Get EXL3 tensor dict from cache (keys: `trellis`, `mcg`, `suh`, `svh`, `su`, `sv`, `mul1` per w1/w2/w3)
2. Move tensors to GPU (88 MB total per expert → 44 ms at 2 GB/s PCIe)
3. Instantiate temporary `ExLlamaV3Linear` and set q_matrix from loaded tensors
4. Run forward pass
5. Free q_matrix and GPU tensors immediately

```python
from exllamav3 import ExLlamaV3Linear  # or equivalent API

for eidx in unique_experts:
    raw = get_expert(layer_idx, eidx)          # EXL3 tensor dict on CPU
    
    # Move EXL3 quantized tensors to GPU (~88 MB per expert)
    gpu_tensors = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
    
    # Create temporary quantized linear for w1, w2, w3
    tmp_w1 = _make_exl3_linear(gpu_tensors, "w1", in_dim, mid_dim)
    tmp_w2 = _make_exl3_linear(gpu_tensors, "w2", mid_dim, in_dim)
    tmp_w3 = _make_exl3_linear(gpu_tensors, "w3", in_dim, mid_dim)
    
    gate = act_fn(tmp_w1(current_state))
    up   = tmp_w3(current_state)
    out  = tmp_w2(gate * up)
    
    # Apply weights, accumulate, free
    final_hidden.index_add_(0, token_idx, out * routing_weights[...])
    del tmp_w1, tmp_w2, tmp_w3, gpu_tensors
```

> **Open item:** `exllamav3` API for creating a temporary `ExLlamaV3Linear` from raw tensors (vs loading from a checkpoint) needs investigation. `EXL3LoaderStrategy.inject_to_gpu()` already does a version of this for full layers — the same pattern applies here per-expert. This must be confirmed against the installed `exllamav3` version before implementation.

**Performance model (EXL3, decode):**
- PCIe H2D per expert: 88 MB / 2 GB/s = **44 ms**
- 2 active experts × 32 layers = 64 transfers × 44 ms = **2.8 s/token**  
- GPU compute (fast, ExLlamaV3 CUDA kernels): ~50–100 ms total
- **Estimated decode: ~3–4 s/token** (vs 14 s BF16 GPU path)

---

## Component 4: RAM Cache Budget Fix

Current formula `max(2 GB, available_at_init - 8 GB)` produces 2.1 GB (6 expert slots) when RAM is near capacity at init time.

New formula in `vaayu_base.py`:
```python
import psutil
total_ram = psutil.virtual_memory().total
_expert_budget = int(total_ram * 0.35)  # 35% of total RAM
```

For 32 GB machine: 11.2 GB → **33 BF16 expert slots** (covers 1 full layer of all 8 experts × 4 decode tokens).

After warmup, decode hit rate: ~50–70%.  
Combined with CPU FFN path, cache misses only cost disk read (53 ms at 6.3 GB/s), not PCIe.

---

## Component 5: Gate Warning Fix (minor)

`detect_moe_blocks()` logs 32 warnings:
```
[Vaayu-MoE] warn: multiple expert lists at model.layers.X.mlp: ['gate', 'experts']
```

`MixtralTopKRouter` has `num_experts` attribute (value=8 > 4) → incorrectly classified as expert list.

Fix: in `detect_moe_blocks()`, check `child_name not in _ROUTER_NAMES` before classifying as expert list:
```python
elif (hasattr(child, 'num_experts') 
      and getattr(child, 'num_experts', 0) > _MIN_EXPERTS
      and not isinstance(child, nn.Linear)
      and child_name not in _ROUTER_NAMES):   # ← add this guard
```

---

## Architecture Diagram

```
model init
    │
    ├─ loader_strategy = SafetensorLoaderStrategy | EXL3LoaderStrategy
    │
    └─ patch_moe_blocks(loader_strategy=...)
            │
            ├─ is_exl3=False → compute_device="cpu"
            │       └─ _make_fused_sparse_forward(..., compute_device="cpu")
            │               ↓ at inference time
            │               [router on GPU] → [hidden to CPU] → [BF16 GEMM in RAM]
            │                               → [result to GPU]
            │
            └─ is_exl3=True → compute_device="cuda"
                    └─ _make_fused_sparse_forward_exl3(...)
                            ↓ at inference time
                            [router on GPU] → [EXL3 tensors to GPU] → [ExLlamaV3 GEMM]
```

---

## Files Modified

| File | Changes |
|---|---|
| `vaayu/moe_patch.py` | `patch_moe_blocks()` gains `loader_strategy` param; `_make_fused_sparse_forward()` gains `compute_device` param + CPU branch; new `_make_fused_sparse_forward_exl3()`; gate warning fix in `detect_moe_blocks()` |
| `vaayu/vaayu_base.py` | Pass `loader_strategy` to `patch_moe_blocks()`; replace RAM budget formula |

**No changes to:** `expert_cache.py`, `expert_offset_index.py`, `safetensor_model_persister.py`, `loader_strategy.py`.

---

## Out of Scope

- Non-MoE dense models: unaffected (CPU/GPU FFN path only activates inside `sparse_forward`)
- Other MoE architectures (DeepSeek, Kimi, Qwen): detection is generic, but CPU FFN branch assumes `w1/w2/w3` key patterns; EXL3 branch is Mixtral-specific in its tensor key mapping
- Prefill batching optimization: prefill already hits full GEMM throughput on CPU; further optimization (e.g., chunked expert batching) is future work
- Model parallelism / multi-GPU: out of scope

---

## Success Criteria

1. BF16 decode: < 2 s/token on Gen1 x8 PCIe hardware
2. EXL3 decode: < 5 s/token on Gen1 x8 PCIe hardware
3. No regression on non-MoE models
4. Expert cache hit rate > 50% after first decode token (verified via log)
5. Gate warning eliminated (0 "multiple expert lists" warnings)
