# Vaayu Engine: Core Architecture & System Design

Vaayu is engineered to break through the physical VRAM limits of consumer GPUs. By treating model storage (NVMe SSD) not as an offline asset, but as an active, online participant in the execution pipeline, Vaayu enables massive LLMs to run natively on affordable hardware.

This document details the low-level architectural mechanics of Vaayu's layer-streaming engine.

---

## 1. The Pipelined Execution Model

A naive layer-streaming implementation would suffer from complete serialization: read layer $N$ from disk $\rightarrow$ copy to GPU $\rightarrow$ run compute $\rightarrow$ evict layer $N$ $\rightarrow$ read layer $N+1$ from disk. This results in the GPU idling for over 80% of the execution cycle.

Vaayu resolves this by overlapping background file I/O with active GPU computation using a **producer-consumer prefetch architecture**:

```
Time ───────────────►
Main Thread (GPU):   [Transfer Layer N] [Compute Layer N]      [Transfer Layer N+1] [Compute Layer N+1]
Prefetch Thread:     [Read Layer N+1 from SSD ───────] [Read Layer N+2 from SSD ─────────]
```

### The Prefetch Execution Cycle

1. **Warm Startup**: During model initialization, Vaayu spins up a background `ThreadPoolExecutor` and immediately submits a task to pre-fetch the very first non-pinned layer from disk into CPU memory.
2. **The Execution Loop**: Inside the autoregressive layer loop in `vaayu_base.py`:
   * **Wait for Prefetch**: The main thread blocks on the prefetch `future.result()` to ensure Layer $N$ has finished loading into host RAM.
   * **Submit Next Prefetch**: The moment Layer $N$'s CPU read completes, and **before** the main thread begins copying Layer $N$ to the GPU, Vaayu immediately submits a prefetch task for Layer $N+1$. This ensures the SSD begins reading the next layer immediately, running concurrently with the PCIe transfer and GPU execution of the current layer.
   * **PCIe Transfer**: The main thread triggers a fast Host-to-Device (H2D) PCIe DMA copy of Layer $N$ to the active GPU using non-blocking CUDA streams.
   * **GPU Forward Pass**: The GPU executes the forward pass of Layer $N$ (Attention, Feed-Forward Networks).
   * **Explicit Eviction**: The moment computation completes, the layer is deleted from PyTorch's active cache, or its C++ memory is freed (under EXL3), and the model module is shifted back to the `meta` device.

---

## 2. Zero-Overhead Pinned Memory Double Buffering

Memory allocation in CUDA is a notoriously expensive system call. Specifically, registering host memory as "pinned" (`pin_memory=True`) requires the operating system kernel to block, lock contiguous physical RAM pages, and update page tables so they cannot be swapped to disk. 

Doing this dynamically inside the layer-streaming loop costs **~0.5 seconds per layer**, which adds **~41.5 seconds of overhead** to every single generated token.

### The Double-Buffer Solution

Vaayu completely eliminates this overhead by allocating exactly **two static global pinned buffers** (`_PINNED_BUFS[0]` and `_PINNED_BUFS[1]`) at startup:

```
                  ┌──────────────────────────────────────────────┐
                  │            Static Global Buffers             │
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
                                 ▼                ▼
                  ┌──────────────┐                ┌──────────────┐
                  │ Pinned Buf A │                │ Pinned Buf B │
                  │ (~1.8 GB)    │                │ (~1.8 GB)    │
                  └──────┬───────┘                └──────┬───────┘
                         │                               │
                         ▼                               ▼
                  Used for Loading Layer N        Used for Prefetching Layer N+1
                  (Currently copying to GPU)      (Currently reading from NVMe)
```

* **Size**: Pinned buffers are sized at ~1.8 GB to accommodate the largest single layer shard with comfortable headroom.
* **Synchronization**: A thread-safe locking mechanism rotates the active buffer index on every step using a bitwise XOR toggle:
  ```python
  with _BUF_LOCK:
      buf_idx = _BUF_IDX
      _BUF_IDX = 1 - _BUF_IDX
  ```
* **Views Instead of Copies**: After loading bytes into the active pinned buffer, Vaayu reconstructs PyTorch tensors as zero-copy views. They are sliced out of the pre-allocated buffer without a single byte copy:
  ```python
  raw = buf.narrow(0, start, nbytes)
  tensors[name] = raw.view(dtype).reshape(shape)
  ```
  The physical CPU memory remains locked and static. When the layer is copied to the GPU, it streams directly from this pre-registered host memory at maximum PCIe DMA hardware limits.

---

## 3. Pluggable Loader Strategy Pattern

To keep the core engine highly maintainable and clean, Vaayu decouples **disk-loading, parameter injection, and memory eviction** from the core Transformer framework. This is handled by a clean **Strategy Pattern** in `loader_strategy.py`:

```
                       ┌───────────────────────┐
                       │    LoaderStrategy     │
                       └───────────┬───────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│  StandardSafetensorsStrategy  │           │      EXL3LoaderStrategy       │
├───────────────────────────────┤           ├───────────────────────────────┤
│ Handles BF16/FP16 uncompressed│           │ Handles 4-bit ExLlamaV3       │
│ safetensors, loading directly │           │ quantized weights, performing │
│ through standard PyTorch      │           │ one-time model surgery, and   │
│ state dict injections.        │           │ calling custom C++ kernels.   │
└───────────────────────────────┘           └───────────────────────────────┘
```

This clean decoupling guarantees that future model backends (like Apple Silicon MPS structures, or new custom quantized formats) can be supported simply by writing a new subclass under `LoaderStrategy` without rewriting the core autoregressive forward loops in `vaayu_base.py`.
