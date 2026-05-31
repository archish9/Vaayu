<p align="center">
  <img src="vaayu_logo.svg" alt="Vaayu Logo" width="200"/>
</p>

# 🌪️ Vaayu: The Layer-Streaming LLM Inference Engine

Welcome to **Vaayu**—the high-performance, local-first inference engine designed to break through the physical boundaries of consumer hardware. 

Vaayu operates on a revolutionary principle: **You don't need a $10,000 enterprise GPU cluster to run world-class AI.** By streaming model weights layer-by-layer directly from an NVMe SSD into GPU VRAM on every single token, Vaayu enables massive Large Language Models (like Llama 3.3 70B) to execute natively on a single, affordable 16 GB GPU. 

We have achieved massive technical breakthroughs—shredding per-token latency from **116.8 seconds down to a highly responsive 29.6 seconds**, bypassing operating system caches, and engineering the ultimate zero-copy asynchronous disk pipeline.

This repository is no longer a formal script; it is a weaponized, local-first engine ready to power the next generation of AI coding assistants.

---

## 🚀 Our Engineering Victories & Breakthroughs

Through months of hardware-software co-design, we have systematically targeted and eliminated every major bottleneck. Here are our hard-won victories:

* **⚡ The NVMe Queue Depth Surge (+323% Speed)**: Standard sequential file reading is locked at a queue depth of 1 (QD1), capping NVMe drives at ~2.8 GB/s. We engineered a **16-thread parallel preadv reader** releasing the Python GIL to submit parallel requests, pushing speeds to **4.4 GB/s**. We then built a native **Rust `io_uring` kernel-bypass module** (`vaayu_rust`) to push asynchronous zero-copy reads to **6.0–6.7 GB/s**!
* **📦 The Double-Buffering Victory (Saves 41.5s/token)**: Allocating pinned memory host buffers dynamically inside the layer loop forced the OS kernel to lock contiguous pages on every call, costing ~0.5s per layer. We replaced this with **two static global pinned memory double-buffers** allocated once at startup, reducing dynamic allocation overhead to **0.0 seconds**.
* **🧠 The KV Cache Resurrection**: We tracked down and squashed a critical legacy cache bug where the engine was silently recomputing the entire prompt history on every step. By patching cache subscripts, subsequent tokens only evaluate the single new token, saving massive computation time.
* **🧹 CPU Cleanse (Saves 1.4s/token)**: We stripped defensive garbage collection sweeps (`gc.collect()`) and configured CUDA Host-to-Device memory copies to run asynchronously (`non_blocking=True`), completely flattening CPU latency spikes.
* **🔥 FP8 KV Cache Quantization (~50% VRAM savings)**: The KV cache accumulates at full BF16 precision during generation—2.68 GB for Llama-70B at 8k context, silently eating VRAM that should go to layer pinning and speculative decoding. We implemented `FP8DynamicCache`, a drop-in replacement storing K/V tensors in `torch.float8_e4m3fn` with per-head per-token scales. Pure PyTorch, zero new dependencies. KV footprint drops from **2.68 GB → 1.38 GB** at 8k context, freeing ~1.3 GB for the phases ahead.

---

## 🛠️ The Definitive Speed Guide: Setup in 4 Steps

To run Vaayu at its absolute hardware-ceiling sequential speeds, you must configure both your environment and your physical hardware correctly.

### Step 1: Install Core Dependencies
Set up your Python environment and compile `flash-attn` and `exllamav3` using our strict sandbox-bypass order:
```bash
# Create and activate environment
python3.12 -m venv venv
source venv/bin/activate

# Install core libraries & Triton JIT requirements
pip install "torch>=2.6.0" "transformers==5.5.4" "accelerate>=0.28.0" "safetensors>=0.4.0" "optimum>=1.17.0" "huggingface-hub>=0.20.0" "bitsandbytes>=0.43.0" ninja tqdm einops scipy sentencepiece peft evaluate scikit-learn wandb kernels maturin Pillow

# Compile CUDA FlashAttention-2 & ExLlamaV3 (compilation takes 5-15 mins)
pip install flash-attn --no-build-isolation
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
```

### Step 2: Compile the Native `vaayu_rust` Extension
Compile our zero-copy `io_uring` disk-streaming engine using Maturin:
```bash
cd vaayu_rust
maturin develop --release
cd ..
```

### Step 3: Relocate Your GPU to the CPU-Direct PCIe x16 Slot
*If your `gpu_load` times are hovering around ~20 seconds in your logs, your GPU is installed in the wrong slot.*
* **The Fix**: Open your PC chassis and move your GPU to the **topmost metal-armored PCIe slot (labeled PCI_E1 on your motherboard)**. This slot is wired directly to the CPU, increasing lane width to the maximum **x16** (PCIe 5.0/4.0), and dropping your PCIe transfer time from **20.4 seconds to ~2 seconds**!
* **Verify**: Run `nvidia-smi -q | grep -A 10 "Link Width"` and confirm `Link Width` reports **16x**.

### Step 4: Run the Linux NVMe Kernel Bypasses
Run our system optimization script to tell the Linux kernel to stop throttling your NVMe SSD:
```bash
sudo bash scripts/vaayu_optimize_os.sh /path/to/your/model/directory
```
* **I/O Scheduler**: Changes your NVMe block scheduler from `mq-deadline` (which wastes CPU reordering requests) to `none` (FIFO pass-through).
* **APST Power Management**: Prevents the SSD from entering micro-sleep states between layer queries, eliminating 50-500μs of wakeup latency jitter.

---

## 🗺️ Next Frontiers: Roadmaps to Interactive Speeds

The remaining phases follow a strict dependency chain. Each one unlocks the next.

### ✅ Phase E + F: VRAM Reclamation (In Progress)
**E — FP8 KV Cache** ✅ done: ~1.3 GB freed at 8k context (see Engineering Victories above).  
**F — Asymmetric Layer Pinning**: Pin the first 28 transformer layers permanently in VRAM (not just embed/norm/lm_head), delivering **15–25% faster prefill** with zero extra disk I/O for those layers.

### Phase G: Chunked Prefill
Large code file ingestion (4,000+ tokens) causes VRAM spikes that OOM. Processing prompts in 512-token chunks flattens the spike, making long-context use reliable. Depends on E+F being stable.

### Phase H: Speculative Decoding (The Speed Multiplier)
Standard generation streams the full 38 GB model once per token. Speculative Decoding turns Vaayu into a **Verifier**—a tiny drafter (EAGLE-3 self-speculation for near-zero VRAM, or a small resident model) guesses N tokens, Vaayu verifies all N in **one forward pass**. Target: **2–4× effective tokens/second**. Requires VRAM freed by Phase E.

### Phase D: Rust `io_uring` Polish (Final)
Update `vaayu_rust` with `IORING_REGISTER_BUFFERS` and fixed file descriptors for the final 10–15% NVMe throughput squeeze. Done last because access patterns finalize only after EXL3 + speculative decoding are locked in.

---

## 📚 The Complete Vaayu Documentation Suite

We have structured the entire architectural knowledge, installation guides, and performance histories into detailed, modular handbooks in the `/documentation` directory. Use these files as your technical guides:

* [**🌪️ Architectural Engine Design**](documentation/architecture.md): A deep mechanical analysis of our layer-streaming execution loops, producer-consumer prefetching models, and double-buffered host pinned memory strategies.
* [**🛠️ Setup & Hardware Tuning Guide**](documentation/setup_and_speed_guide.md): The step-by-step handbook to setting up virtual environments, Maturin Rust compilation, GPU PCIe lane width slots relocations, and OS-level NVMe block optimizations.
* [**📖 The Chronological Optimization Bible**](documentation/optimization_bible.md): The historical master ledger recording every performance epoch, tracking our benchmarks from the 116s baseline through the PyO3 regression, up to our current 29.6s steady state (and projected 8.0s future).
* [**📂 Quantization & Pluggable Strategies**](documentation/quantization_and_strategies.md): Explaining our decoupled `LoaderStrategy` pattern (Peeking safetensors headers to auto-detect strategies), compiling custom ExLlamaV3 4-bit CUDA kernels, managing heap memory safety, and executing `quantize.py`.
* [**🔮 Speculative Decoding & MoE Roadmap**](documentation/speculative_decoding_and_moe.md): Our technical blueprint detailing drafting-verification loops, Mixture-of-Experts routing gates, and expert-aware byte-offset disk streaming.
* [**🍎 Apple Silicon & The Unified Memory Paradigm**](documentation/mac_unified_memory_paradigm.md): Exploring UMA zero-copy superpowers (`gpu_load` drops to 0s), analyzing the 16 GB MacBook VM swap thrashing constraint, and outlining a Mac manual swap manager using Metal (MPS) and `F_NOCACHE` (via `fcntl`).
* [**🏆 The Bugfix & Maintenance Victory Log**](documentation/bugfix_log.md): A massive, structured developer ledger detailing every single code fix and API migration (including DynamicCache transitions, RoPE positional overrides, Gemma4 multi-modal remappings, 0-indexed non-sequential caching, and Triton compiler resolutions).

---

## 🏃 Quickstart Inference Test

Verify your installation by running our CLI text generation entrypoint:
```bash
# FP8 KV cache (default for EXL3 models, ~50% VRAM savings)
python test_quick.py --kv_cache_mode auto

# Force FP16 KV cache (baseline, full precision)
python test_quick.py --kv_cache_mode fp16
```

*Example API usage:*
```python
from vaayu import AutoModel

# Vaayu automatically peeks headers to detect the strategy (Standard vs EXL3)
# kv_cache_mode="auto" selects FP8 for EXL3 models, FP16 otherwise
model = AutoModel.from_pretrained(
    "/path/to/models/Llama-3.3-70B-Instruct-exl3-4.5bpw",
    kv_cache_mode="auto",  # or "fp8" / "fp16"
)

input_tokens = model.tokenizer(
    "What is the capital of India?",
    return_tensors="pt",
    padding=False
)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```
