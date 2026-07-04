# Vaayu MoE Evaluation Report

An in-depth analysis of the current Mixture-of-Experts (MoE) streaming implementation in Vaayu, explaining why `test_quick.py` runs extremely slowly (13–15 seconds/token) and diagnosing the core architectural and implementation bugs.

---

## 🔍 Executive Summary

Your suspicion is correct: **Vaayu's MoE streaming is currently not working as intended**, leading to severe performance degradation. While the model is technically generating correct outputs, the latency is extremely high (~13–15s per token) instead of being "lightning fast."

We identified four core issues causing this slowdown:
1. **CPU FFN Bottleneck (Primary Slowdown)**: The hybrid MoE design forces all standard (non-EXL3) models to perform the expert Feed-Forward Network (FFN) computations on the CPU. Because PyTorch lacks optimized bfloat16/float16 GEMM/GEMV CPU kernels, a single token forward pass takes **7.9 seconds of CPU compute**, and the prefill chunk takes **over 100 seconds**.
2. **Pinned Layers Inefficiency**: Pinned transformer layers (like layers 0, 1, 2) have their expert weights fully loaded into GPU memory. However, because the MoE block forward is patched globally, even pinned layers run their FFN on the CPU and reload/prefetch weights unnecessarily.
3. **Speculative Prefetch Logic Error**: The speculative prefetcher predicts the next layer's active experts using `_pred = self._expert_history[_current_layer_idx]`. This is an index mismatch (using layer N's routing decisions to predict layer N+1). Additionally, the history table is cleared on every token, preventing cross-token prediction.
4. **Cache Thrashing in Prefill**: During the prefill phase (sequence length 512), the routing activates almost all experts in every layer, causing the total size of active experts (90 GB) to exceed the cache budget (16.7 GB). This leads to 100% cache thrashing and constant NVMe reads.

---

## 📈 Log Analysis (`run.log`)

The log shows that `test_quick.py` is **not hung** but is running extremely slowly:
- **Chunked Prefill**:
  - **Chunk [0:512]**: Took **104.04 seconds** to complete, with **101.2 seconds** spent on CPU compute (`compute=101.2s`).
  - **Chunk [512:673]**: Took **57.55 seconds** to complete, with **43.3 seconds** spent on CPU compute (`compute=43.3s`).
- **Autoregressive Generation (Decode)**:
  - Token 1 took **14.59 seconds** (`compute=7.9s`, `gpu_load=6.6s`).
  - Token 2 took **13.44 seconds**, and so on.
  - The model successfully generated **57 tokens** over 14 minutes (average **13–14s/token**), printing outputs like `import` and `float` into `run.log`.

Because the print statements are buffered/unbuffered and mixed up with intermediate logging lines (e.g. `import [Vaayu-Debug] forward...`), it appeared as if there was no output.

---

## 🛠️ Core Issues Diagnosed

### 1. The CPU FFN Bottleneck (Standard Models)
The MoE patching logic in `vaayu/moe_patch.py` determines the compute device as follows:
```python
is_exl3 = isinstance(loader_strategy, EXL3LoaderStrategy)
compute_device = "cuda" if is_exl3 else "cpu"
```
For standard safetensors checkpoints (like Mixtral BF16), `compute_device` is set to `"cpu"`. The design intended to avoid PCIe overhead (which was estimated at 168ms per expert on PCIe Gen1 x8). 

However, this design decision is highly problematic for two reasons:
* **CPU BF16 Performance**: PyTorch CPU lacks optimized BF16 vector-matrix math, so a single expert GEMV/GEMM is extremely slow.
* **PCIe Bandwidth**: The log shows your GPU is running at **PCIe Gen5 x8 (25.2 GB/s)**. Transferring 700 MB (2 active experts) from CPU to GPU takes **only 28 ms**. 

> [!IMPORTANT]
> Running the FFN on the GPU (including the 28ms transfer) takes **~29ms**, whereas running it on the CPU takes **246ms** (and over **3,000ms** during prefill).

### 2. Pinned Layers Running on CPU
In `_init_pinned_layers()`, layers 0 to 2 are pinned in GPU VRAM (9.23 GB). This means their gate and expert weights are already resident on the GPU.
However, because `moe_module.forward` is overwritten globally with the sparse forward, these pinned layers still compute on the CPU and attempt to load weights from the CPU cache.

### 3. Speculative Prefetch Logic & Lifecycle Errors
In `vaayu/vaayu_base.py` (lines 1856–1869):
```python
_next_layer_idx = _current_layer_idx + 1
_pred = self._expert_history[_current_layer_idx]
if _pred:
    executor.submit(self._prefetch_experts, _next_layer_idx, _pred)
```
* **Index Mismatch**: It fetches `self._expert_history[_current_layer_idx]` (the active experts of the *current* layer) and prefetches them for `_next_layer_idx` (the *next* layer). This assumes the next layer will activate the same expert indices as the current layer, which is incorrect.
* **Lifecycle Clearing**: `self._expert_history.clear()` is called at the start of every forward pass. This wipes out the history of the previous token, which makes it impossible to retrieve the routing indices of `_next_layer_idx` from the last token.

### 4. Cache Thrashing in Prefill
During prefill, the sequence length $S = 512$ causes almost all 8 experts in all 32 layers to be active. The total active expert footprint is $32 \times 8 \times 350 \text{ MB} \approx 90 \text{ GB}$. Because your expert cache budget is $16.7 \text{ GB}$ (~46 experts), the cache constantly evicts and reloads experts (thrashing), resulting in a $5\%$ cache hit rate and massive NVMe read overhead.

---

## 🚀 Recommended Fixes

To restore Vaayu's lightning-fast performance, we recommend three fixes:

### Fix 1: Enable GPU Compute Device for Standard Models
Update the `compute_device` selection in `vaayu/moe_patch.py` so that it uses `"cuda"` whenever a GPU is available:
```python
compute_device = "cuda" if (is_exl3 or device.startswith("cuda")) else "cpu"
```
This forces standard models to execute FFN computations on the GPU using fast PyTorch CUDA kernels.

### Fix 2: Bypass Patching for Pinned Layers
Save the original forward function during patching. If a layer is pinned, bypass the sparse forward entirely and call the original forward pass on the GPU:
```python
# In sparse_forward:
if layer_name in vaayu_ref.pinned_layer_names:
    return original_forward(hidden_states, **kwargs)
```

### Fix 3: Fix Speculative Prefetch Index and History Table
1. Stop clearing `self._expert_history` on every token. Keep it as a persistent dictionary mapping `layer_idx -> unique_experts`.
2. Look up the prediction using `_next_layer_idx` instead of the current index:
```python
_pred = self._expert_history.get(_next_layer_idx, [])
```
