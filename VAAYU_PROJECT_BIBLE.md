# VAAYU PROJECT BIBLE
### The Unfiltered Technical History of Running a 70B LLM on Consumer Hardware

**Last Updated:** 2026-05-04  
**Status:** Active Development  
**Model:** Llama 3.3 70B Instruct  
**Hardware:** i5-14600K · MSI Z790 Gaming Plus WiFi · RTX 5060 Ti 16GB · 32GB DDR5-6000 · Samsung 990 PRO NVMe  

---

> This document is the single source of truth for the Vaayu project. It records every problem encountered, every hypothesis tested, every benchmark run, and every fix applied. It is written to be brutally honest about what works, what failed, and why. If you are resuming work on this project, start here.

---

## TABLE OF CONTENTS

1. [What Vaayu Is](#1-what-vaayu-is)
2. [The Hardware Stack](#2-the-hardware-stack)
3. [The Fundamental Constraint: 132 GB in 16 GB](#3-the-fundamental-constraint-132-gb-in-16-gb)
4. [Resolved Issues (Before the I/O Focus)](#4-resolved-issues-before-the-io-focus)
   - 4.1 Flash Attention Confusion
   - 4.2 KV Cache Broken — The `past_key_values` Bug
   - 4.3 First Rust Attempt — The Triple Copy Disaster
5. [The I/O Bottleneck — Core Problem](#5-the-io-bottleneck--core-problem)  
   *(Continued in Part 2)*

---

## 1. What Vaayu Is

Vaayu is a custom inference engine built on top of the **AirLLM** pattern — a technique for running large language models that far exceed available GPU VRAM by streaming model weights layer-by-layer from NVMe storage. Instead of loading the full model into memory at once, Vaayu:

1. Loads a single transformer layer (~1.6 GB) from NVMe into system RAM
2. Transfers that layer over PCIe into GPU VRAM
3. Runs the GPU forward pass for that layer
4. Discards the layer from VRAM
5. Repeats for all 83 layers of Llama 3.3 70B

This allows a **132 GB model** to run on a **16 GB GPU** at the cost of extreme I/O pressure — the full 132 GB of weights must be read from disk for every single token generated.

The project is a custom Python + Rust codebase that has diverged significantly from the original AirLLM baseline, incorporating:
- Custom pinned memory management
- Multi-threaded parallel I/O
- A Rust `io_uring` module (`vaayu_rust`) for zero-copy NVMe streaming
- KV cache integration
- Per-token diagnostic timing (`[VAAYU-IO]` log lines)

---

## 2. The Hardware Stack

This section documents the exact hardware, because the optimization work is deeply hardware-specific. Every benchmark in this document was run on this exact machine.

| Component | Specification | Notes |
|---|---|---|
| **CPU** | Intel Core i5-14600K | 6 P-cores + 8 E-cores = 14 cores / 20 threads. Raptor Lake Refresh. Boost to 5.3 GHz. |
| **Motherboard** | MSI Z790 Gaming Plus WiFi (MS-7E06) | Intel Z790 chipset. LGA1700. |
| **RAM** | 32 GB DDR5-6000 (2×16 GB) | ~80–90 GB/s memory bandwidth. Critical for absorbing parallel NVMe streams. |
| **GPU** | NVIDIA RTX 5060 Ti 16 GB | Compute capability 12.0. Supports BF16 natively. PCIe 5.0 ×16 slot (PCI_E1 from CPU). |
| **NVMe SSD** | Samsung 990 PRO | Installed in M.2_1 slot — PCIe 4.0 ×4, directly from CPU. Box-rated: 7,450 MB/s sequential read. |
| **OS (dev machine)** | Linux (kernel 6.17) | All benchmarks ran on Linux. Rust `io_uring` requires kernel ≥ 6.14. |
| **Python** | 3.12 | |
| **PyTorch** | 2.x | |
| **transformers** | Latest (unified `LlamaAttention` class) | |

### Critical Hardware Facts Established by Benchmarking

**The NVMe slot is PCIe Gen 4 x4 — confirmed.**  
Running `cat /sys/class/nvme/nvme0/device/current_link_speed` reported `16.0 GT/s PCIe width 4`. The hardware is correctly configured. There are no BIOS settings to change.

**The Samsung 990 PRO's 7.4 GB/s rating requires Queue Depth ≥ 2 with multiple threads.**  
Samsung's rated speed is measured at QD2 multi-thread per their datasheet. Single-thread sequential reads (Queue Depth 1) are physically capped at **2.5–3.0 GB/s** regardless of PCIe generation. This is an NVMe controller architecture limit, not a software or configuration issue.

**The RTX 5060 Ti GPU compute time per layer is ~50ms.**  
GPU math is effectively free. The entire token generation time is dominated by NVMe-to-RAM-to-GPU data movement, not compute.

---

## 3. The Fundamental Constraint: 132 GB in 16 GB

This is the immovable physics of the project. It must be understood clearly before any optimization discussion.

### Why every token requires reading 132 GB from disk

The Llama 3.3 70B model weighs **132 GB** in bfloat16 precision. The RTX 5060 Ti has **16 GB** of VRAM. The model cannot fit in VRAM, so it cannot live on the GPU permanently.

Because tokens are generated **autoregressively** (one token at a time, each requiring a full forward pass through all 83 layers), and because the model weights must be present on the GPU during each layer's computation:

**Every single generated token = 132 GB read from NVMe.**

### The theoretical speed ceiling

```
Minimum time per token = 132 GB ÷ NVMe read speed

At Queue Depth 1 (Python default):  132 ÷ 2.8  GB/s = ~47 seconds
At Queue Depth 16 (our threading):  132 ÷ 4.4  GB/s = ~30 seconds
At theoretical max (io_uring):      132 ÷ 6.5  GB/s = ~20 seconds
At Samsung rated peak:              132 ÷ 7.45 GB/s = ~17.7 seconds
```

**KV Cache does NOT solve this problem.** KV Cache reduces the GPU compute work per token (avoiding redundant attention calculations for previous tokens), but it cannot reduce the number of bytes that must be read from disk. The weights still need to be loaded for every token. KV Cache saves milliseconds; the I/O bottleneck costs seconds.

### Layer-by-layer breakdown

The model is pre-split into 83 individual `.safetensors` files stored at:
```
/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/
  model.embed_tokens.safetensors   (~2.3 GB)
  model.layers.0.safetensors       (~1.6 GB each)
  model.layers.1.safetensors
  ...
  model.layers.79.safetensors
  model.norm.safetensors           (~0.001 GB)
  lm_head.safetensors              (~0.5 GB)
```

The `[VAAYU-IO]` log tag shows the exact load time for every component on every token pass.

---

## 4. Resolved Issues (Before the I/O Focus)

These are bugs and confusions that were fixed before the main I/O optimization work. They are recorded here so future contributors understand the current state of the codebase.

---

### 4.1 Flash Attention Confusion

**Symptom:** The logs showed `Attention module type: LlamaAttention` instead of a dedicated `LlamaFlashAttention2` class. This led to uncertainty about whether Flash Attention was actually active.

**Investigation:** Inspected `transformers.models.llama.modeling_llama.LlamaAttention` source code. Ran:
```bash
python3 -c "
from transformers import AutoConfig, AutoModelForCausalLM
import torch
config = AutoConfig.from_pretrained('/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct')
config._attn_implementation = 'sdpa'
with torch.device('meta'):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
print(f'Attention implementation: {config._attn_implementation}')
print(f'Attention module: {type(model.model.layers[0].self_attn).__name__}')
"
```

**Root Cause:** In modern `transformers` versions, Hugging Face unified all attention mechanisms into a single `LlamaAttention` class. The class internally dispatches to the correct kernel based on `config._attn_implementation`:
```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
attn_output, attn_weights = attention_interface(...)
```

With `_attn_implementation = 'sdpa'`, PyTorch's `torch.nn.functional.scaled_dot_product_attention` is used, which automatically dispatches to **FlashAttention v2** on supported hardware (RTX 5060 Ti with BF16).

**Resolution:** ✅ Flash Attention IS active. The log line `✅ Attention implementation: sdpa` confirms it. No code changes needed.

**Log confirmation:**
```
✅ GPU supports BF16 (compute capability 12.0). Using bfloat16 for better stability.
✅ Attention implementation: sdpa
   Attention module type: LlamaAttention
```

---

### 4.2 KV Cache Broken — The `past_key_values` Bug

**This was a critical correctness bug.** The KV cache was silently not working, causing the model to recompute attention over the full context on every token instead of only the new token.

**Symptom:** Each token pass processed the full sequence length instead of just the single new token. `past_length` was not incrementing in diagnostic logs.

**Root Cause — Two independent bugs:**

**Bug 1 (in `vaayu_base.py`):** The wrong kwarg name was being used when calling the decoder layer forward pass.

```python
# BROKEN — kwarg was 'past_key_value' (singular)
layer_outputs = decoder_layer(
    hidden_states,
    past_key_value=past_key_values[layer_idx],  # ← WRONG
    ...
)

# FIXED — kwarg must be 'past_key_values' (plural)
layer_outputs = decoder_layer(
    hidden_states,
    past_key_values=past_key_values[layer_idx],  # ← CORRECT
    ...
)
```

**Bug 2 (in `vaayu_base.py`):** The `use_cache=True` flag was being overridden to `False` somewhere in the call stack, preventing the cache from being populated even when the argument name was correct.

**Fix:** Both bugs were patched in `vaayu/vaayu_base.py`. The fix ensures `past_key_values` (plural) is passed correctly and `use_cache` is not overridden.

**Verification:** After the fix, `past_length` increments by 1 on each subsequent token pass, confirming the cache is being read and written correctly.

**Impact of this fix:** Without KV cache, generating the Nth token requires running N tokens through 83 layers. With KV cache working, every token after the first runs exactly 1 token through 83 layers. For a 10-token prompt, this eliminates 9 redundant token computations per pass.

**Files modified:**
- `vaayu/vaayu_base.py` — line ~659: kwarg rename from `past_key_value` to `past_key_values`

---

### 4.3 First Rust Attempt — The Triple Copy Disaster

**Background:** Before the pinned memory / threading approach, an early Rust implementation was attempted using PyO3 bindings to speed up file loading.

**What was built:** A `vaayu_rust` Python extension module with a `load_layer()` function that read a `.safetensors` file from disk and returned a dictionary of PyTorch tensors.

**Symptom:** After installing this module, per-token time **increased** from ~80 seconds to **~190 seconds** — a 2.4× regression.

**Root Cause — Accidental Triple Copy:**

The PyO3 binding architecture caused three full copies of each layer's 1.6 GB data:

```
Disk → Rust Vec<u8> (Copy 1)
     → Python bytes / NumPy array (Copy 2, at PyO3 boundary)
     → PyTorch .clone() (Copy 3, to create owned tensor)
```

Total data moved per layer: **4.8 GB** instead of 1.6 GB. Each copy took ~0.5s, adding ~1.5s of pure copy overhead per layer × 83 layers = **~125 extra seconds per token**.

**The Painful Irony:** The Python `safetensors` library already uses Rust internally. It memory-maps the file, providing true zero-copy tensor handles in `0.005s`. The naive Rust wrapper was dramatically slower than the Python library it was meant to replace.

**Resolution:** The triple-copy Rust layer loader was abandoned. The `vaayu_rust` module was redesigned from scratch with a completely different philosophy — instead of Rust **returning data to Python**, Rust would orchestrate `io_uring` to **write directly into a Python-allocated pinned memory buffer**, completely bypassing the PyO3 copy boundary.

The new Rust module (`vaayu_rust/src/lib.rs`) exposes:
- `io_uring_read_into(fd, buf_ptr, length, offset)` — uses `io_uring` to DMA data directly into a pre-allocated PyTorch pinned memory buffer
- `pread_into(fd, buf_ptr, length, offset)` — synchronous fallback using `pread`

Python allocates the pinned buffer. Rust only moves bytes into it. Zero copies at the PyO3 boundary.

---

*[Continued in Part 2: The I/O Bottleneck — Discovery, Diagnosis, and the Road to 4.4 GB/s]*
