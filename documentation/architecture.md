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

---

## 4. Asymmetric Layer Pinning

### The Problem

Before this feature, `_init_pinned_layers()` only pre-loaded 3 shared layers into VRAM:

| Layer | Size (70B EXL3 4.5bpw) |
|-------|------------------------|
| `model.embed_tokens` | ~530 MB |
| `model.norm` | ~0.5 MB |
| `lm_head` | ~530 MB |
| **Total** | **~1 GB** |

On a 16 GB GPU with FP8 KV cache active, the VRAM budget available for pinning is ~11–12 GB — leaving **~11 GB unused**. Every one of the remaining 80 transformer layers streamed from NVMe on every token.

### The Two-Phase Solution

`_init_pinned_layers()` now runs a two-phase loop controlled by `max_pinned_layers`:

```
Phase 1 — Shared layers (embed, norm, lm_head)
  Strategy: continue on budget miss
  Reason: each has an independent size; a large embed shouldn't block a tiny norm

Phase 2 — Transformer layers (model.layers.0, model.layers.1, …)
  Strategy: break on budget miss
  Reason: layers are homogeneous size; if one doesn't fit, none will
  Guard: only runs when max_pinned_layers != 0
```

```
VRAM (16 GB GPU, FP8 KV cache, 4096 ctx):
┌──────────────────────────────────────────────┐
│ KV Cache (FP8)         ~2.0 GB               │
│ CUDA context + buffer  ~1.5 GB               │
│ Shared layers          ~1.0 GB  (Phase 1)    │
│ Transformer layers 0–27 ~10.6 GB (Phase 2)   │
│ Free headroom          ~0.9 GB               │
└──────────────────────────────────────────────┘
→ 28 transformer + 3 shared = 31 layers pinned
→ 35% of the 70B model permanently in VRAM
```

### `max_pinned_layers` Parameter

| Value | Behavior |
|-------|----------|
| `0` | Shared layers only — default, preserves previous behavior |
| `-1` | Auto: fill VRAM budget with transformer layers after shared |
| `N > 0` | Shared + first N transformer layers (still budget-capped) |

### Impact on `forward()`

No changes to `forward()` were needed. It already keyed off the `pinned_layer_names` set at every layer boundary:

- **Pinned layer hit**: skip `load_from_disk` + `inject_to_gpu` + `free` — zero NVMe cost
- **Prefetch skip**: seed prefetch loop skips pinned layers so disk reads target only unpinned layers

With 28 layers pinned, each token saves 28 × ~380 MB = ~10.6 GB of NVMe reads, cutting prefill `io_wait` by 30–50%.

### Log Output

```
[Vaayu-Pin] pinned model.embed_tokens: 0.53 GB  (cumulative 0.53 GB)
[Vaayu-Pin] pinned model.norm: 0.00 GB  (cumulative 0.53 GB)
[Vaayu-Pin] pinned lm_head: 0.53 GB  (cumulative 1.06 GB)
[Vaayu-Pin] pinned model.layers.0: 0.38 GB  (cumulative 1.44 GB)
...
[Vaayu-Pin] budget exhausted at model.layers.28 (11.64/11.64 GB)
[Vaayu-Pin] 31 layers pinned (28 transformer + 3 shared), 11.64 GB reserved
```

## 5. Speculative Decoding

### Problem

Standard generation streams all 80 layers from NVMe once per token (~5 s/token). Even with EXL3 quantization and asymmetric pinning, each token requires a full model pass.

### Solution: Drafter-Verifier Loop

```
Standard:     [80-layer pass] → 1 token → [80-layer pass] → 1 token → …
Speculative:  draft 5 tokens (< 1 ms) → [80-layer pass, verifies all 5] → accept 3–5 → …
```

Vaayu acts as the **Verifier**. A resident drafter generates K candidates; the verifier processes them all in one pass and accepts the longest correct prefix.

### Package: `vaayu/speculative/`

| File | Responsibility |
|------|----------------|
| `base.py` | `SpeculativeDecoder` ABC: `draft()`, `verify_and_accept()`, `rollback_cache()`, `vram_usage_bytes()` |
| `detection.py` | `detect_speculative_type(dir)` → `'eagle'`/`'medusa'`/`'none'` |
| `eagle.py` | `EAGLEDecoder`: ExLlamaV3 bridge first, manual weight loader fallback |
| `standard.py` | `StandardDrafterDecoder`: any `AutoModelForCausalLM` resident in VRAM |
| `__init__.py` | `build_speculator()` factory: EAGLE > StandardDrafter > None |

### Key VaayuBaseModel Changes

**Hidden state capture:** After `run_norm()` in `forward()`, `self._last_hidden_state = batch[j][:, -1:, :].detach()` stores the post-norm hidden state of the last token. The EAGLE decoder reads this during drafting.

**`_lm_head_layer`:** Resolved after `set_layers_from_layer_names()` as `self.layers[-1]`. EAGLE calls `run_lm_head(self._lm_head_layer, next_hidden)` to produce token logits.

**`_trim_kv_cache(cache, target_len)`:** Trims both `DynamicCache` and `FP8DynamicCache` in-place after each verify pass. KV grows by `k+1` during verify; we trim back to `past_len + num_accepted`.

**`speculative_generate()` flow:**
1. Prefill: `forward(input_ids)` → `past_key_values`, first token
2. Loop until `max_new_tokens` or EOS:
   - Draft K tokens via `speculator.draft()`
   - Record `past_len_before`
   - Verify: `forward([last_token, d_0, …, d_{K-1}])` with past KV → logits shape `(1, K+1, vocab)`
   - Accept: `speculator.verify_and_accept()` → `accepted_ids`, `num_accepted`
   - Trim KV to `past_len_before + num_accepted`
   - Rollback drafter: `speculator.rollback_cache()`
   - Append `accepted_ids` to `generated`

### VRAM Budget (16 GB GPU, post Plans E+F)

| Component | VRAM |
|-----------|------|
| Pinned transformer layers | ~12.0 GB |
| FP8 KV cache | ~0.6 GB |
| CUDA context | ~0.8 GB |
| Streaming buffer | ~0.5 GB |
| **Free for drafter** | **~2.1 GB** |
| EAGLE-3 head | 0.1–0.2 GB ✅ |
| Qwen2.5-0.5B | ~0.3 GB ✅ |
| Qwen2.5-1.5B | ~1.0 GB ✅ |
| 3B fp16 | ~6 GB ❌ |

### Tuning

`DRAFT_TOKENS` (default 5) trades off NVMe waste on rejection vs throughput on acceptance:
- `accept_rate` > 70%: raise to 7–10
- `accept_rate` < 40%: drop to 3 or switch to better-matched drafter

---

## 6. MoE Expert-Aware Streaming

For Mixture-of-Experts (MoE) models, default layer-streaming loads all expert weights
per token (99.2% wasted for Kimi-K2.6). Expert-aware streaming reduces this to only
the top-k active experts per token via three layers of optimization:

1. **ExpertRAMCache** (`vaayu/persist/expert_cache.py`) — LRU RAM cache keyed by `(layer_idx, expert_idx)`. 70–90% of tokens reuse the same experts.
2. **Speculative prefetch** — After each MoE layer, current token's routing history predicts next layer's experts. Prefetch fires in background, overlapping with attention compute.
3. **Sparse io_uring** (`load_expert_sparse` in safetensor_model_persister.py) — On cache miss, precise byte-range read for only the active expert bytes.

Implementation uses monkey-patch (`vaayu/moe_patch.py`): replace each MoE block's `forward()` at init time. Generic detection via `detect_moe_blocks()` covers Mixtral, DeepSeek, Kimi, Qwen MoE. Dense models unaffected (`is_moe=False`).

See `documentation/moe_expert_aware_streaming.md` for full technical reference.

## 7. Decode CUDA Graphs

A one-token decode at batch 1 is host-bound on models whose weights stay resident: the
CPU spends longer issuing eager kernel launches than the GPU spends running them. Vaayu
therefore replays each resident decoder layer's static work as CUDA graphs
(`vaayu/decode_graphs.py`).

- **What is captured.** The stock transformers layer forward runs once under capture.
  The graph is broken at two places that must stay eager: the attention core
  (`cache.update` plus the attention kernel call — the KV cache grows every token, and
  flash-attention's wrapper syncs the host) and the MoE block after routing (expert ids
  must reach the CPU to dispatch). For the fused non-EXL3 MoE forward, routing and the
  shared expert are inside the graph (`decode_pre`).
- **When.** Only on eligible forwards: one token, batch 1, CUDA, no speculative decoding,
  no `profiling_mode`, no `output_hidden_states`. The KV cache must also be one of Vaayu's
  own seamed cache classes (`FP8DynamicCache` and its subclasses); a model that brings its
  own cache class — DeepSeek-V4's `DeepseekV4Cache` today — is never eligible, and no
  decode-graph line appears in its log at all. Prefill is always eager. The process's
  first eligible token warms up; each layer is captured on the next and replayed after
  that, and recaptured when a new request brings new cache tensors.
- **Correctness.** Graphs replay the same kernels in the same order, so outputs are
  bit-identical. A layer's first two tokens where its graph is used each run the graph,
  then run the stock forward last from the same starting state and compare the two
  bitwise; the stock result is always what is returned (and what the cache holds), so a
  layer only ever serves a graph result outright once both checks pass. A layer is refused
  for the process — stock forward from then on — the first time any of these happens: its
  capture raises (including attention that never reaches transformers' attention-function
  registry, e.g. `attn_implementation="eager"` as gpt-oss uses — `cache.update` runs but
  nothing closes the attention core, so every layer of such a model is refused); its
  checked tokens disagree bitwise with the stock forward; it needs a third capture within
  one request (an input is changing shape every token); or a forward kwarg hides a tensor
  inside a container the graphs cannot bind (a list, dict, or mixed tuple). Each distinct
  reason logs once: `[Vaayu] WARNING: decode graphs refused for <layer> (<reason>); it
  decodes eagerly.` — other layers refused for the same reason are not logged again.
- **Memory.** The graphs' memory pool is measured at the first capture and charged to the
  expert VRAM cache, which shrinks by that amount; the first forward after the first
  captures prints where it went (`decode graphs: pool X GB charged to the expert VRAM tier
  (a -> b GB)`). If nothing can pay for it, graphs are disabled for the process at once;
  the pool itself is released at the start of the next forward, with one WARNING
  (`decode graphs DISABLED for this process: <reason>. Decode runs eagerly.`). That WARNING
  also names the restored budget (`... The expert VRAM tier's budget is restored to <b>
  GB.`) only if the pool had already been charged to the expert VRAM tier; a model with no
  expert VRAM tier — dense, or disabled before any tier charge succeeded — has nothing to
  refund, and the WARNING ends without that sentence.
