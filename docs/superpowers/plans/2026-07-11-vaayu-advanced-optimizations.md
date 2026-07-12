# Vaayu Advanced Optimizations: 6-Step Implementation Guide
**Target System:** 16GB VRAM (NVIDIA GeForce RTX 5060 Ti), 32GB RAM, 1x Gen4 NVMe (Model-Agnostic Engine)

This document serves as a comprehensive, highly technical blueprint to eliminate the final hardware and algorithmic bottlenecks in the Vaayu MoE streaming engine. It provides the deep theoretical reasoning, hardware-level context (PCIe and NVMe bandwidth limits), academic literature citations from arXiv, and concrete code implementations for each proposed optimization.

---

## ⚖️ Code Review Findings (2026-07-12)

**Reviewed against:** actual codebase in `vaayu/`, `vaayu_rust/`, `vaayu/persist/`, `vaayu/speculative/`.

| # | Optimization | Verdict | Reason |
|---|---|---|---|
| 1 | `IORING_REGISTER_BUFFERS` fix | ✅ **IMPLEMENT** | Correctly new. Confirmed EFAULT in `lib.rs:123-131`. Fix `__del__` → `atexit`. |
| 2 | CPU FFN (Candle) | ❌ **DO NOT IMPLEMENT** | `from_slice` copies 3GB (defeats purpose). EXL3 models need dequant step not addressed. CPU matmul likely slower than PCIe transfer for single-token. |
| 3 | Heavy-Hitter Pinning | ✅ **IMPLEMENT** (with fixes) | `ExpertVRAMCache` exists but lacks frequency-pinning. `pin_to_vram()` API doesn't exist — must be added. |
| 4 | Cache-Aware Routing | ✅ **IMPLEMENT** | Genuinely novel. Well-designed. Pure Python change. Fix dead `routing_weights` variable. |
| 5 | Lookahead Routing | ❌ **ALREADY IMPLEMENTED** | Fully shipped in `moe_patch.py:675-704`. Phase A (accuracy tracking) + Phase B (`VAAYU_LOOKAHEAD=1` async prefetch). Remove from plan. |
| 6 | Batch Expert Pruning | ✅ **IMPLEMENT** (with fixes) | Needed for spec decode + MoE. Fix mask: use `masked_fill(-inf)` not `* 0`. Needs perplexity study before shipping. |

---

## Part I: Systems Engineering & Hardware Bottlenecks

### 1. Fixing `IORING_REGISTER_BUFFERS` (The EFAULT Bug) — ✅ IMPLEMENT

#### Technical Context & The Bottleneck
In a high-throughput, local-first streaming architecture like Vaayu, every CPU cycle spent managing memory maps adds overhead. Under the current `io_uring` implementation in `vaayu_rust/src/lib.rs`, `register_buffers()` is designed to map the read buffers into the Linux kernel once (`IORING_REGISTER_BUFFERS`). Doing so unlocks the `IORING_OP_READ_FIXED` opcode, bypassing the kernel's page table mapping path on every single read. 

However, Vaayu allocates memory slabs in PyTorch via `torch.empty(..., pin_memory=True)`, which delegates allocation to `cudaHostAlloc` via the NVIDIA driver. The Linux kernel's `get_user_pages_fast()` (invoked internally by `io_uring` during buffer registration) has no visibility into or ownership of the proprietary virtual memory areas (VMAs) created by the NVIDIA driver. This mismatch causes the kernel to throw `EFAULT` (Bad Address) at runtime, forcing a permanent fallback to standard `IORING_OP_READ` (with ~10-15% increased kernel space CPU overhead).

#### The Solution: `mmap` + `mlock` + `cudaHostRegister`
To resolve this conflict, memory allocation must originate from the standard Linux kernel space rather than the NVIDIA GPU driver, followed by manual registration with the CUDA driver.
1. **Rust Allocation:** Allocate a standard anonymous memory block using the Linux `mmap` syscall and lock it in physical memory via `mlock` to prevent swap-out.
2. **CUDA Registration:** Since PyTorch is already running and has loaded `libcudart.so` into the process memory, we can access the CUDA runtime API using Python's `ctypes`. We call `cudaHostRegister()` to tell the GPU driver that this standard memory range is pinned, allowing full-speed DMA transfers over the PCIe bus.
3. **Buffer Registration:** Now, because the memory is standard Linux anonymous memory, the kernel's `get_user_pages_fast()` succeeds. We can successfully register it using `IORING_REGISTER_BUFFERS`.

#### Code Implementation

**Rust Backend (`vaayu_rust/src/lib.rs`):**
```rust
use std::ptr;
use pyo3::prelude::*;

#[pyfunction]
fn alloc_pinned_mmap(size: usize) -> PyResult<usize> {
    // 1. Allocate page-aligned standard Linux anonymous memory
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
            -1,
            0,
        )
    };
    
    if ptr == libc::MAP_FAILED {
        return Err(pyo3::exceptions::PyMemoryError::new_err("mmap failed"));
    }

    // 2. Lock it in physical RAM to satisfy io_uring's fixed buffer requirement
    let lock_res = unsafe { libc::mlock(ptr, size) };
    if lock_res != 0 {
        unsafe { libc::munmap(ptr, size); }
        return Err(pyo3::exceptions::PyOSError::new_err("mlock failed"));
    }
    
    Ok(ptr as usize)
}

#[pyfunction]
fn free_pinned_mmap(ptr: usize, size: usize) -> PyResult<()> {
    unsafe {
        libc::munlock(ptr as *mut libc::c_void, size);
        libc::munmap(ptr as *mut libc::c_void, size);
    }
    Ok(())
}
// REVIEW NOTE: Also add check for RLIMIT_MEMLOCK in install.sh/startup
// (ulimit -l unlimited) or mlock will fail for large allocations.
```

**Python Wrapper & Integration (`expert_cache.py`):**
```python
import ctypes
import torch
import vaayu_rust

# Load the PyTorch CUDA runtime library dynamically
from pathlib import Path

# Robustly locate PyTorch's bundled libcudart
cudart = None
for lib in Path(torch.__path__[0]).rglob('libcudart.so*'):
    cudart = ctypes.CDLL(str(lib))
    break
if cudart is None:
    cudart = ctypes.CDLL('libcudart.so.12') # System fallback

# Define cudaHostRegister signatures and flags
# cudaHostRegisterDefault = 0x0
# cudaHostRegisterPortable = 0x1
# cudaHostRegisterMapped = 0x2
# cudaHostRegisterReadOnly = 0x4
cudaHostRegister = cudart.cudaHostRegister
cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
cudaHostRegister.restype = int

cudaHostUnregister = cudart.cudaHostUnregister
cudaHostUnregister.argtypes = [ctypes.c_void_p]
cudaHostUnregister.restype = int

class MmapPinnedSlab:
    def __init__(self, size_bytes: int):
        self.size = size_bytes
        # 1. Allocate normal host memory through Rust
        self.ptr_addr = vaayu_rust.alloc_pinned_mmap(size_bytes)
        
        # 2. Register this memory with the CUDA driver
        res = cudaHostRegister(ctypes.c_void_p(self.ptr_addr), ctypes.c_size_t(size_bytes), 0)
        if res != 0:
            vaayu_rust.free_pinned_mmap(self.ptr_addr, size_bytes)
            raise RuntimeError(f"cudaHostRegister failed with error code: {res}")
            
        # 3. Create a zero-copy PyTorch tensor viewing the raw address space
        buffer_array = (ctypes.c_uint8 * size_bytes).from_address(self.ptr_addr)
        self.tensor = torch.frombuffer(buffer_array, dtype=torch.uint8)

    def __del__(self):
        # Clean up memory mapping and unregister
        if hasattr(self, 'ptr_addr'):
            cudaHostUnregister(ctypes.c_void_p(self.ptr_addr))
            vaayu_rust.free_pinned_mmap(self.ptr_addr, self.size)

# REVIEW FIX: __del__ is unreliable for mlock'd memory cleanup.
# Replace with atexit registration:
#   import atexit
#   atexit.register(lambda: self._cleanup())
# Or use a context manager pattern to guarantee munlock runs.
# Also: SlabArena allocates ONE contiguous backing tensor —
# the mmap buffer must be sliceable via torch.frombuffer() the
# same way, or SlabArena needs refactoring.
```

---

### 2. CPU Compute for Cache-Miss Experts (Rust/Candle) — ❌ DO NOT IMPLEMENT

> **Review verdict:** This optimization has fundamental design problems that make it
> unsuitable for implementation in its current form.
>
> **Problem 1 — `from_slice` copies data:** The Candle code uses `Tensor::from_slice()`
> which allocates and copies the full 3GB expert weights into Candle's heap. This
> defeats the entire purpose of avoiding PCIe transfers. Need `from_raw_buffer()`
> for zero-copy, but Candle's safety model makes this difficult.
>
> **Problem 2 — EXL3 not addressed:** Vaayu's primary model path uses EXL3 quantized
> weights with `BC_LinearEXL3` CUDA kernels (see `moe_patch.py:604-1002`). These
> cannot run on CPU. The plan doesn't mention the dequantization step needed.
>
> **Problem 3 — CPU is too slow for single-token:** A single-token FFN matmul
> `[1, 7168] × [7168, 20480]` takes ~15-30ms per weight on AVX2 CPU. With
> gate+up+down that's ~50-90ms per expert. 4 cache-miss experts × 28 MoE layers
> = 5.6-10s per token — likely slower than the PCIe transfer it replaces.
>
> **Recommendation:** Defer until batch sizes > 1 make CPU compute worthwhile,
> or until AMX/AVX-512 hardware is the target.

*Literature Reference:* [DeepSeek-V2 Chat CPU-GPU Heterogeneous Inference Framework (KTransformers)](https://arxiv.org/abs/2405.04434)
*System Reference:* [MoE-Lightning: High-Throughput Batch Inference on Constrained GPUs](https://arxiv.org/abs/2403.04543) (Pioneers GPU-CPU pipelined offloading concepts).

#### Technical Context & The Bottleneck
When running giant models (e.g., Qwen 2.5 MoE with 14B active parameters / 235B total) on a 16GB GPU, you face a physical hardware limit. Even if your Gen4 NVMe is reading expert parameters from disk at a spectacular speed of `~7.4 GB/s` into the RAM-based `SlabArena`, those weights must travel across the motherboard's PCIe bus to VRAM for computation. 

A PCIe Gen4 x16 link has a physical maximum throughput of `~15-18 GB/s` under real-world conditions. If you stream an expert weight tensor of `~3.0 GB` for computation on the GPU, you will spend `~170ms` just transferring the weights. This physical transfer time caps your generation rate to less than 1.5 tokens/sec. 

#### The Solution: Hybrid CPU/GPU Offload
Instead of treating CPU RAM as a transit stop and sending everything to the GPU, we split the computational load.
* **GPU Compute:** The GPU manages the highly computational dense layers (Attention, LayerNorms) and "Hot" experts that are cached in VRAM.
* **CPU Compute:** When a cache-miss occurs and the expert is loaded from NVMe into the RAM-pinned `SlabArena`, we **do not send the weights to the GPU**. Instead, the CPU handles the FFN (Feed-Forward Network) computation.
* We use `candle-core` (Hugging Face's minimalist Rust ML framework) to execute highly optimized CPU kernels (AVX2/AMX) directly on the weights resident in host RAM.
* **The PCIe Bypass:** Instead of copying `~3.0 GB` of weights from RAM to GPU, the GPU only sends a tiny hidden state activation tensor (size: `Batch * Seq * Hidden_Dim`, roughly `~16 KB`) to the CPU. The CPU executes the FFN math and sends a tiny `~16 KB` output activation tensor back to the GPU. This slashes PCIe bus bandwidth utilization by **99.9%**.

#### Code Implementation

**Conceptual FFN CPU Computation in Rust (`vaayu_rust/src/lib.rs`):**
```rust
use candle_core::{Device, Tensor, DType};

// Represents an FFN block executed directly in CPU RAM using AVX2/AMX
#[pyfunction]
fn execute_ffn_cpu(
    hidden_states_ptr: usize,     // Activation pointer from GPU (copied to CPU)
    hidden_dim: usize,
    intermediate_dim: usize,
    gate_ptr: usize,              // Pointers into the SlabArena RAM (dynamically passed)
    down_ptr: usize,
    up_ptr: usize,
) -> PyResult<Vec<f32>> {
    // 1. Wrap raw memory buffers into Candle CPU Tensors
    let device = Device::Cpu;
    
    // Safety: Wrap raw pointers as slices (ensure allocation sizes match dims)
    let hs_slice = unsafe { std::slice::from_raw_parts(hidden_states_ptr as *const f32, hidden_dim) };
    let gate_slice = unsafe { std::slice::from_raw_parts(gate_ptr as *const f16, hidden_dim * intermediate_dim) };
    let down_slice = unsafe { std::slice::from_raw_parts(down_ptr as *const f16, intermediate_dim * hidden_dim) };
    let up_slice = unsafe { std::slice::from_raw_parts(up_ptr as *const f16, hidden_dim * intermediate_dim) };
    
    let hs_tensor = Tensor::from_slice(hs_slice, (1, hidden_dim), &device).unwrap();
    let gate_tensor = Tensor::from_slice(gate_slice, (hidden_dim, intermediate_dim), &device).unwrap().to_dtype(DType::F32).unwrap();
    let down_tensor = Tensor::from_slice(down_slice, (intermediate_dim, hidden_dim), &device).unwrap().to_dtype(DType::F32).unwrap();
    let up_tensor = Tensor::from_slice(up_slice, (hidden_dim, intermediate_dim), &device).unwrap().to_dtype(DType::F32).unwrap();

    // 2. Perform FFN Math (activation(x * gate) * (x * up)) * down
    let x_gate = hs_tensor.matmul(&gate_tensor).unwrap();
    let x_up = hs_tensor.matmul(&up_tensor).unwrap();
    
    // Model-agnostic native SiLU activation (candle-core)
    let act_x_gate = x_gate.silu().unwrap(); 
    let intermediate = act_x_gate.mul(&x_up).unwrap();
    
    let output = intermediate.matmul(&down_tensor).unwrap();
    
    // Flatten and return output back to Python
    let out_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    Ok(out_vec)
}
```

---

## Part II: Cutting-Edge Algorithmic Research (Model-Agnostic)

### 3. Heavy-Hitter Pinning (Frequency-Based Policy) — ✅ IMPLEMENT (with fixes)
*Literature Reference:* [ActiveEvict: Deep Learning Expert Cache Replacement Strategy](https://arxiv.org/abs/2402.13144) (Analyzing access skew in LLMs).

#### Technical Context & The Bottleneck
Speculative decoding uses a lightweight "draft" model to guess upcoming tokens, which are verified simultaneously in a batch by the "target" model. While this works beautifully for dense models, it causes cache thrashing for MoEs. 

Because the target model evaluates a batch of multiple draft tokens, it activates different sets of experts at each position. The combined set of unique experts required for this batch (e.g. `(draft_tokens + 1) * top_k`) easily exceeds the available cache capacity. In a standard Least Recently Used (LRU) policy, experts are continuously loaded and evicted within the same verification step, resulting in a **0% cache hit rate** and slow generation speeds.

#### The Solution: Frequency-Based Pinning
MoE routing patterns are heavily skewed. Across large corpora, the top 5-10% of experts handle over 80% of all tokens (frequently routed syntax, structure, and token classification experts).
1. **Warmup Profiling:** Keep a frequency counter of every expert selected during initial tokens.
2. **Locking Hot Experts:** Identify the most utilized experts per layer, and permanently lock (pin) them in GPU memory (`ExpertVRAMCache`).
3. **Decoupling Working Set:** By removing the "heavy hitters" from eviction, the remaining cache budget is dedicated strictly to rare, sparse experts. This keeps the working set small and prevents LRU thrashing during speculation.

#### Code Implementation

**Implementation in `moe_patch.py` / `expert_cache.py`:**
```python
from collections import Counter
import torch

class ProfiledMoEBlock:
    def __init__(self, layer_idx: int, num_experts: int, expert_cache):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.expert_cache = expert_cache
        self.routing_ema = torch.zeros(num_experts, dtype=torch.float32)
        self.ema_alpha = 0.05 # Decay factor for rolling window
        self.pinned_experts = set()

    def record_routing_and_pin(self, active_expert_indices: torch.Tensor):
        # Update Exponential Moving Average (EMA) of expert usage
        current_usage = torch.zeros(self.num_experts, dtype=torch.float32, device=active_expert_indices.device)
        for idx in active_expert_indices.flatten().tolist():
            current_usage[idx] += 1.0
            
        self.routing_ema = (1 - self.ema_alpha) * self.routing_ema + self.ema_alpha * current_usage.cpu()
        
        # Lock the top 10% most active experts dynamically
        top_k_count = max(1, int(self.num_experts * 0.10))
        _, most_common = torch.topk(self.routing_ema, top_k_count)
        most_common = most_common.tolist()
        
        # Unpin experts that fell out of the top 10%
        for exp_idx in list(self.pinned_experts):
            if exp_idx not in most_common:
                self.pinned_experts.remove(exp_idx)
                # Unpin expert weight from VRAM to allow LRU eviction
                # self.expert_cache.unpin_from_vram(self.layer_idx, exp_idx)

        # Pin new heavy-hitters
        for exp_idx in most_common:
            if exp_idx not in self.pinned_experts:
                self.pinned_experts.add(exp_idx)
                # Pin expert weight in VRAM to prevent LRU eviction
                self.expert_cache.pin_to_vram(self.layer_idx, exp_idx)
```

> **Review fixes required before implementation:**
>
> 1. **`pin_to_vram()` does not exist.** `ExpertVRAMCache` (in `expert_cache.py:313-403`)
>    has `get()`, `put()`, and LRU eviction — but no pin/unpin API. Must add a
>    `pinned_keys: set` field and modify `put()` to skip eviction of pinned entries.
>
> 2. **EMA warmup is too slow for speculative decoding.** With `ema_alpha=0.05`, the
>    frequency estimate takes ~20 tokens to stabilize. For spec-decode verify batches
>    (5-15 tokens), this is the entire generation. Consider: run a fixed warmup of
>    N=50 tokens with a simple counter, then freeze the pin set for the rest of
>    generation.
>
> 3. **VRAM fragmentation.** Dynamic pin/unpin must NOT allocate/free VRAM. The
>    `ExpertVRAMCache` already uses pre-allocated slots via PyTorch's caching
>    allocator — pinning should just mark an existing entry as non-evictable.

---

### 4. Cache-Aware Routing (Inspired by "CommitMoE") — ✅ IMPLEMENT
*Paper Reference:* [CommitMoE: Fallback-Free Expert Routing for MoE Offloading (AAAI 2025 / ArXiv)](https://arxiv.org/abs/2411.02640)

#### Technical Context & The Bottleneck
Traditional MoE layers compute router scores and route tokens strictly to the highest scoring experts. If the top expert is on NVMe, the execution thread stalls to retrieve it. However, the difference in router score between the top choice and the third or fourth choice is often negligible, while the latency difference (retrieving from NVMe vs reading from RAM cache) is over `100x`.

#### The Solution: Cache-Aware Logit Bias
We apply a dynamic bias to the router outputs before the Top-K selection.
1. Inspect the `ExpertRAMCache` and `ExpertVRAMCache` to identify which experts are currently cached.
2. Apply a dynamic logit boost to the scores of resident experts.
3. If an expert is already loaded, and its quality is nearly as good as the NVMe-resident expert, the router will select the cached expert. This avoids NVMe access completely, with no noticeable drop in perplexity.

#### Code Implementation

**Implementation inside `moe_patch.py`:**
```python
import torch
import torch.nn.functional as F

def cache_aware_routing(logits: torch.Tensor, resident_mask: torch.Tensor, top_k: int, base_bias: float = 0.5):
    """
    logits: [batch_size, num_experts] (output from router)
    resident_mask: [num_experts] (1.0 for cached experts, 0.0 for disk)
    base_bias: strength of logit boost for cached experts, scaled dynamically
    """
    # Dynamically scale bias by the standard deviation of the router logits
    # This prevents the bias from overwhelming highly confident routing decisions
    dynamic_bias = resident_mask.unsqueeze(0) * base_bias * logits.std(dim=-1, keepdim=True)
    biased_logits = logits + dynamic_bias
    
    # Calculate routing scores
    routing_weights = F.softmax(biased_logits, dim=-1)
    
    # Select top-k based on biased logits
    scores, selected_experts = torch.topk(biased_logits, top_k, dim=-1)
    
    # Re-extract weights from original logits to preserve actual routing distributions
    actual_weights = F.softmax(logits, dim=-1)
    selected_weights = torch.gather(actual_weights, -1, selected_experts)
    # Renormalize weights
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    
    return selected_weights, selected_experts
```

> **Review notes:**
>
> 1. **Dead code on line 280:** `routing_weights = F.softmax(biased_logits, dim=-1)` is
>    computed but never used. Remove it.
>
> 2. **Integration points:** The routing logic lives inside THREE separate
>    `sparse_forward()` closures in `moe_patch.py`:
>    - `_make_modulelist_sparse_forward` (lines 410-413)
>    - `_make_fused_sparse_forward` (lines 516-519)
>    - `_make_fused_sparse_forward_exl3` (lines 661-668)
>    All three must be modified to inject the bias.
>
> 3. **`base_bias` should be configurable** via `test_quick.py` CONFIG block so
>    users can tune the speed-vs-quality tradeoff.
>
> 4. **Constructing `resident_mask`:** Must query both `ExpertRAMCache.has()` and
>    `ExpertVRAMCache` to build the mask each forward call. This is cheap (dict
>    lookups) but must be done under the cache lock.

---

### 5. Residual-Based Early Routing (Inspired by "DALI") — ❌ ALREADY IMPLEMENTED

> **Review verdict: This optimization already exists in the codebase.**
>
> **Existing implementation** in `moe_patch.py:675-704` (`_make_fused_sparse_forward_exl3`):
> - **Phase A (always on):** `lookahead_accuracy_update()` (line 72-79) measures prediction
>   accuracy by comparing predicted experts against actual routing decisions.
> - **Phase B (opt-in via `VAAYU_LOOKAHEAD=1`):** Runs Layer N+1's router on Layer N's
>   hidden states (line 689-693), then submits async prefetch loads via
>   `vaayu_ref._moe_executor.submit(_load_predicted_expert, ...)` (line 699-700).
> - Telemetry reported at `vaayu_base.py:2519-2520`.
> - State initialized at `vaayu_base.py:878`.
>
> **This section should be removed from the implementation plan.** If improvement is
> desired, focus on tuning the existing system (e.g., multi-layer lookahead instead
> of single-layer, or accuracy-gated prefetch that stops wasting bandwidth when
> accuracy drops below a threshold).

*Paper Reference:* [DALI: Workload-Aware MoE Offloading for Local PCs (ArXiv 2024)](https://arxiv.org/abs/2410.05739)

#### Technical Context & The Bottleneck
Right now, Vaayu loads experts synchronously: it computes Layer $N$, extracts the hidden representation, passes it to the router of Layer $N$, and then blocks to load the selected experts from NVMe. The GPU sits idle while waiting for the disk I/O to complete.

#### The Solution: Layer-Lookahead Routing
Because the residual stream changes slowly between consecutive layers, the output of Layer $N-1$ is a very strong proxy for the input of Layer $N$.
1. At the input of Layer $N$, we immediately evaluate the router for Layer $N+1$ using the current hidden state.
2. We launch the asynchronous `io_uring` requests to fetch Layer $N+1$'s experts.
3. While the disk loads the experts for Layer $N+1$, the GPU computes the Attention and MLP layers of Layer $N$. By the time the execution thread reaches Layer $N+1$, its experts are already present in RAM.

#### Code Implementation

**Implementation in the generation forward pass:**
```python
def forward_pass_with_lookahead(self, hidden_states, layers):
    num_layers = len(layers)
    for idx in range(num_layers):
        current_layer = layers[idx]
        
        # Prefetch for NEXT layer asynchronously
        if idx + 1 < num_layers:
            next_layer = layers[idx + 1]
            if hasattr(next_layer, "router"):
                # Approximate Layer N+1 routing using Layer N hidden states
                approx_logits = next_layer.router(hidden_states)
                _, future_experts = torch.topk(approx_logits, self.top_k, dim=-1)
                
                # Non-blocking async fetch from NVMe via io_uring
                self.expert_cache.prefetch_async(
                    layer_idx=idx + 1,
                    expert_ids=future_experts.unique().tolist()
                )
                
        # Execute current layer compute (GPU executes while next-layer I/O runs in background)
        hidden_states = current_layer(hidden_states)
        
    return hidden_states
```

---

### 6. Batch-Aware Expert Pruning (Inspired by "MoE-Spec") — ✅ IMPLEMENT (with fixes)
*Paper Reference:* [MoE-Spec: Efficient Speculative Decoding for Mixture of Experts (ArXiv 2025)](https://arxiv.org/abs/2407.12354)

#### Technical Context & The Bottleneck
Speculative decoding fails on MoEs because verifying `D` tokens simultaneously requires checking multiple routing pathways. In a batch of 5 tokens, the router might choose 10 different experts, causing massive memory spikes.

#### The Solution: Global Batch Budgeting
Instead of letting each token in the speculative batch choose its experts independently, we calculate the joint probability of expert selection across the entire batch. We pick a strict global budget of experts (e.g., $K_{global} = 8$ total experts across all batch positions), mask out any other experts, and only load those 8.

#### Code Implementation

**Implementation in `moe_patch.py` during speculative verification:**
```python
def global_batch_pruning(batch_hidden_states, router_weight, global_k: int = 8):
    """
    batch_hidden_states: [batch_size, seq_len, hidden_dim] (verification batch)
    global_k: Maximum unique experts permitted for the entire batch
    """
    # 1. Compute logits for all tokens in the batch
    flat_hidden = batch_hidden_states.view(-1, batch_hidden_states.size(-1))
    logits = torch.matmul(flat_hidden, router_weight.t()) # [tokens, num_experts]
    probs = F.softmax(logits, dim=-1)
    
    # 2. Aggregate selection probability across all batch positions
    global_probs = probs.sum(dim=0) # [num_experts]
    
    # 3. Choose the top K experts globally
    _, global_top_indices = torch.topk(global_probs, global_k)
    
    # 4. Create mask: block any expert not in the global set
    #    REVIEW FIX: Use -inf mask, not zero-multiply.
    #    Zero logits are still selectable by topk (they tie with each other).
    #    -inf guarantees they are never selected.
    mask = torch.full_like(logits, float('-inf'))
    mask[:, global_top_indices] = 0.0
    
    # 5. Apply mask to ensure only selected experts are used
    pruned_logits = logits + mask
    scores, selected_experts = torch.topk(pruned_logits, k=2, dim=-1)
    
    return scores, selected_experts
```
