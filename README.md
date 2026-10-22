<p align="center">
  <img src="vaayu_logo.svg" alt="Vaayu Logo" width="200"/>
</p>

# 🌪️ Vaayu: The Layer-Streaming LLM Inference Engine

Welcome to **Vaayu**—the high-performance, local-first inference engine designed to break through the physical boundaries of consumer hardware. 

Vaayu operates on a revolutionary principle: **You don't need a $10,000 enterprise GPU cluster to run world-class AI.** By streaming model weights layer-by-layer directly from an NVMe SSD into GPU VRAM on every single token, Vaayu enables massive Large Language Models (like Llama 3.3 70B) to execute natively on a single, affordable 16 GB GPU. 

We have achieved massive technical breakthroughs—shredding per-token latency from **116.8 seconds down to a highly responsive 29.6 seconds**, bypassing operating system caches, and engineering the ultimate zero-copy asynchronous disk pipeline.

This repository is no longer a formal script; it is a weaponized, local-first engine ready to power the next generation of AI coding assistants.

---

## ⛔ REQUIREMENT: Vaayu needs the machine to itself

**This is not a recommendation. Vaayu's speed is a direct function of how much VRAM and RAM
are free at the moment it loads — clearing the box first is the end user's responsibility.**

Vaayu streams model weights from NVMe on every token. The only way to avoid that cost is to
keep weights resident, so at load it fills whatever VRAM is free with pinned transformer
layers and whatever RAM is free with the layer/expert caches. **Every megabyte another
process holds is a layer Vaayu must re-read from disk on every single token.**

How much this costs is measured directly by the pinning A/B — the same mechanism a
crowded GPU triggers, since a smaller budget pins fewer layers:

| checkpoint | nothing pinned | budget filled | penalty |
|---|---|---|---|
| Llama-3.2-1B | 6.40 tok/s | **83.27 tok/s** | **13.0×** |
| MiniMax-M2.7 (MoE) | 0.760 tok/s | **1.537 tok/s** | **2.02×** |
| Llama-3.3-70B (dense) | 0.131 tok/s | **0.167 tok/s** | **1.27×** |

A shared GPU lands you somewhere on that scale — the less VRAM free at load, the closer to
the left column. (One observation, not yet an isolated A/B: MiniMax decoded at 0.631 tok/s
on a box where 9.3 GiB was held by unrelated containers, against 1.537 tok/s on record.)

**Before starting Vaayu:**

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # who holds VRAM
free -g                                                                     # who holds RAM
```

Stop everything that appears there — other model servers, notebooks, training jobs, GPU
containers (`docker stop <name>`), and any earlier Vaayu process. Then start Vaayu.

**Two things that surprise people:**

- **Budgets are derived once, at load.** Freeing VRAM *after* Vaayu has started does nothing
  for the running process — it keeps the smaller budget it computed at boot. Clear first,
  then start. If you clear afterwards, restart Vaayu.
- **Vaayu will never free another process's memory for you, and cannot.** GPU allocations
  belong to the owning process; only ending that process releases them. Vaayu reports what
  is held and by whom — the decision to end your own work stays yours.

**This is enforced, not just documented.** `REQUIRE_EXCLUSIVE_GPU = True` is the default in
`vaayu/server/config.py`: the server refuses to start while anything else holds GPU memory,
and prints the exact commands to free it:

```
[Vaayu-Server] refusing to load -- the GPU is not exclusive:
5 other process(es) hold 9.45 GiB of GPU memory. ...
  pid 3179  3.97 GiB  /usr/local/bin/python3.10 (container f599a5d0371c)
  ...
Free it with:
  docker stop f599a5d0371c
  kill 227646
```

Only **compute** processes count — your desktop (Xorg, gnome-shell, a browser) holds
graphics memory in the tens of MiB and is never something an inference engine should tell
you to kill. Two more things are excluded from the refusal, because neither costs the load
anything: holders on a **different GPU** than `DEVICE` (rows are matched on the GPU's UUID,
not on nvidia-smi's index, which does not track `cuda:N`), and holders under
`EXCLUSIVE_GPU_MIN_MIB` — a bare CUDA context is 128 MiB, so without a threshold any
process that merely imports torch refuses the boot. Both are still **reported**; only the
refusal is narrowed. Library callers (`AutoModel.from_pretrained`) get the same report as a
**warning** rather than a refusal, so existing scripts keep working. Set
`REQUIRE_EXCLUSIVE_GPU = False` for a box you deliberately share.

> **Containers come back by themselves.** A `docker compose` stack with
> `restart=unless-stopped` restarts seconds into every boot — powering the machine off does
> not count as stopping it. Use `docker compose down` (or `docker stop`) when you are done,
> or `docker update --restart=no <container>`; that is why Vaayu prints `docker stop <id>`
> for a containerised holder instead of `kill <pid>`, which would only respawn it.

---

## 🚀 Our Engineering Victories & Breakthroughs

Through months of hardware-software co-design, we have systematically targeted and eliminated every major bottleneck. Here are our hard-won victories:

* **⚡ The NVMe Queue Depth Surge (+323% Speed)**: Standard sequential file reading is locked at a queue depth of 1 (QD1), capping NVMe drives at ~2.8 GB/s. We engineered a **16-thread parallel preadv reader** releasing the Python GIL to submit parallel requests, pushing speeds to **4.4 GB/s**. We then built a native **Rust `io_uring` kernel-bypass module** (`vaayu_rust`) to push asynchronous zero-copy reads to **6.0–6.7 GB/s**!
* **📦 The Double-Buffering Victory (Saves 41.5s/token)**: Allocating pinned memory host buffers dynamically inside the layer loop forced the OS kernel to lock contiguous pages on every call, costing ~0.5s per layer. We replaced this with **two static global pinned memory double-buffers** allocated once at startup, reducing dynamic allocation overhead to **0.0 seconds**.
* **🧠 The KV Cache Resurrection**: We tracked down and squashed a critical legacy cache bug where the engine was silently recomputing the entire prompt history on every step. By patching cache subscripts, subsequent tokens only evaluate the single new token, saving massive computation time.
* **🧹 CPU Cleanse (Saves 1.4s/token)**: We stripped defensive garbage collection sweeps (`gc.collect()`) and configured CUDA Host-to-Device memory copies to run asynchronously (`non_blocking=True`), completely flattening CPU latency spikes.
* **🔥 FP8 KV Cache Quantization (~50% VRAM savings)**: The KV cache accumulates at full BF16 precision during generation—2.68 GB for Llama-70B at 8k context, silently eating VRAM that should go to layer pinning and speculative decoding. We implemented `FP8DynamicCache`, a drop-in replacement storing K/V tensors in `torch.float8_e4m3fn` with per-head per-token scales. Pure PyTorch, zero new dependencies. KV footprint drops from **2.68 GB → 1.38 GB** at 8k context, freeing ~1.3 GB for the phases ahead.
* **📌 Asymmetric Layer Pinning (15–25% Faster Prefill)**: Previously only 3 shared layers (embed, norm, lm_head, ~1 GB) were pre-loaded into VRAM, leaving ~11 GB of available budget unused. We rewrote `_init_pinned_layers()` with topology-aware two-phase pinning: Phase 1 pins shared layers; Phase 2 fills the remaining VRAM budget with the first N transformer layers in order, using `break` on budget exhaustion (homogeneous layer sizes mean if one doesn't fit, none will). On a 16 GB GPU with FP8 KV cache, this permanently eliminates **~35% of all NVMe reads per token**—delivering faster prefill on long prompts and more stable streaming throughput. Pass `max_pinned_layers=-1` to auto-fill, or `max_pinned_layers=N` to pin exactly N transformer layers.
* **🔮 Phase H: Speculative Decoding (2–4× Tokens/Second)**: Standard generation streams 80 layers once per token (~5 s/token). Speculative Decoding inserts a **resident drafter** (EAGLE-3 self-speculation ~0.1 GB VRAM, or any small HuggingFace model like Qwen2.5-0.5B) that guesses 5 tokens instantly, then Vaayu verifies all 5 in **one forward pass** and accepts the correct prefix. A 60–80% acceptance rate yields **2–4× effective throughput**. Configure via `EAGLE_HEAD_PATH` or `DRAFTER_MODEL_PATH` in `test_quick.py`.

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
pip install "torch>=2.6.0" "transformers==5.5.4" "accelerate>=0.28.0" "safetensors>=0.4.0" "optimum>=1.17.0" "huggingface-hub>=0.20.0" "bitsandbytes>=0.43.0" ninja tqdm einops scipy sentencepiece peft evaluate scikit-learn wandb kernels Pillow

# Compile CUDA FlashAttention-2 & ExLlamaV3 (compilation takes 5-15 mins)
pip install flash-attn --no-build-isolation
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
```

### Step 2: Install the Native `vaayu_rust` Extension
Vaayu's zero-copy `io_uring` disk-streaming engine ships as a prebuilt wheel on PyPI (Linux x86-64):
```bash
pip install vaayu-rust
```

It is a self-contained `manylinux` `abi3` wheel — no Rust toolchain or compilation needed, and one wheel covers Python 3.8+. If the installed extension is older than the engine expects, the load log prints a `[VAAYU-IO] WARNING` naming what is missing and the engine falls back to a slower read path.

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
* **APST Power Management**: Stops the SSD from entering low-power states when idle. Optional: measured with no decode or TTFT cost, because generation keeps the drive's idle gaps shorter than APST's sleep timeout.

---

## 🗺️ Next Frontiers: Roadmaps to Interactive Speeds

The remaining phases follow a strict dependency chain. Each one unlocks the next.

### ✅ Phase E + F: VRAM Reclamation
**E — FP8 KV Cache** ✅ done: ~1.3 GB freed at 8k context (see Engineering Victories above).  
**F — Asymmetric Layer Pinning** ✅ done: First 28 transformer layers permanently pinned in VRAM (not just embed/norm/lm_head), delivering **15–25% faster prefill**. See [Architecture: Asymmetric Layer Pinning](documentation/architecture.md#4-asymmetric-layer-pinning).

### ✅ Phase G: Chunked Prefill
**Done.** Large code file ingestion (4,000+ tokens) caused VRAM spikes that OOM. Processing prompts in 512-token chunks flattens the spike, making long-context use reliable. Controlled via `PREFILL_CHUNK_SIZE` in `test_quick.py` (default 512 tokens; set to 0 to disable).

### ✅ Phase H: Speculative Decoding
Implemented. Vaayu acts as the **Verifier** — a tiny drafter (EAGLE-3 self-speculation for near-zero VRAM, or any small HuggingFace model) guesses N tokens; Vaayu verifies all N in **one forward pass**. ExLlamaV3 native EAGLE bridge tried first; manual weight loader fallback. Configure in `test_quick.py`:

```python
EAGLE_HEAD_PATH    = "/path/to/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
DRAFTER_MODEL_PATH = "/path/to/Qwen2.5-0.5B"   # fallback or standalone
DRAFT_TOKENS       = 5
```

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
* [**🌐 Running Vaayu as a Server**](documentation/server.md): The complete OpenAI-compatible server reference — every CLI flag and request field, curl examples for streaming and tool calling, the coding-agent integration checklist, the measured speed levers (layer pinning is worth 13x on the 1B, 2.0x on MiniMax and 1.27x on the 70B), and the troubleshooting table.
* [**⚙️ Tuning Guide: what to set, and when**](documentation/TUNING.md): The decision guide for the caching knobs — when `prefix_compress` pays (and when it is pure cost), why the pool budget is not free RAM, and which knobs currently do nothing. Pairs with `tune_advisor.py`, which computes the crossover for your own model and box.
* [**🧩 Missing Buffers**](documentation/missing_buffers.md): What the load-time "persistent buffer(s) have no tensor in this checkpoint" WARNING means, why a dropped MoE routing bias (`e_score_correction_bias`, missing from the MiniMax-M2.7 EXL3 build) changes which experts run, and how to fetch just those tensors from the original repo with `scripts/fetch_missing_buffers.py`.
* [**🏆 The Bugfix & Maintenance Victory Log**](documentation/bugfix_log.md): A massive, structured developer ledger detailing every single code fix and API migration (including DynamicCache transitions, RoPE positional overrides, Gemma4 multi-modal remappings, 0-indexed non-sequential caching, and Triton compiler resolutions).

---

## ⚙️ Complete Configuration Parameter Reference

> **Just want to know what to turn on?** See [**documentation/TUNING.md**](documentation/TUNING.md) — a short
> decision guide for the caching knobs (`enable_prefix_cache`, `prefix_compress`,
> `enable_kv_persist`, `kv_cache_mode`), including which settings are proven no-ops today.
> Or run `python tune_advisor.py`, which derives the thresholds for **your** model and
> workload and prints the settings to copy. The reference below covers every parameter in
> depth.

All parameters are set in the `CONFIG` block at the top of `test_quick.py` and passed to `AutoModel.from_pretrained()`. This section explains **what each parameter does internally**, **how it affects generation speed and quality**, and **how to set the correct value** for your hardware and workload.

### Understanding Vaayu's Per-Token Cost

Before tuning parameters, understand the core cost model. Vaayu streams model weights from NVMe → GPU on **every token**. For Llama-3.3-70B at EXL3 4.5 bpw, this means reading ~37 GB from disk per token. The time to generate one token is:

```
token_time = (layers_streamed × layer_size / nvme_bandwidth) + gpu_compute_time
```

Every parameter below either **reduces layers_streamed** (pinning), **reduces layer_size** (quantization), **reduces the number of forward passes** (speculative decoding), or **prevents OOM crashes** (chunked prefill, KV cache mode).

---

### `MODEL_PATH`

```python
MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw"
```

**What it does:** Path to the local HuggingFace model checkpoint directory. Vaayu auto-detects the architecture (Llama, Qwen, Mistral, Mixtral, Gemma, ChatGLM, etc.) from `config.json` and the weight format (BF16/FP16 safetensors or EXL3 quantized) from safetensors headers.

**How it affects speed:**

| Format | Disk Payload (70B model) | Per-Token NVMe Read | Speed Impact |
|--------|-------------------------|---------------------|--------------|
| BF16/FP16 (unquantized) | ~132 GB | ~132 GB | Baseline — slowest |
| EXL3 4.5 bpw | ~37 GB | ~37 GB | **~3.5× faster** than BF16 |
| EXL3 3.0 bpw | ~25 GB | ~25 GB | ~5× faster (lower quality) |
| bitsandbytes 4-bit | ~35 GB | ~35 GB | Similar to EXL3 but slower CUDA kernels |

**Recommendation:** Always use **EXL3 4.5 bpw** quantized models. The disk payload shrinks by 3.5× and ExLlamaV3's native CUDA kernels are faster than dequantized BF16 matmuls. This is the single most impactful choice for generation speed.

**On first load**, Vaayu splits multi-shard HuggingFace checkpoints into per-layer `.safetensors` files (e.g., `model.layers.0.safetensors`, `model.layers.1.safetensors`, ...). This happens once and is cached in a `splitted_model/` subdirectory.

---

### `KV_CACHE_MODE`

```python
KV_CACHE_MODE = "auto"   # "auto" | "fp8" | "fp16"
```

**What it does:** Controls the precision of the KV (Key-Value) cache — the memory structure that stores attention state from all previously processed tokens. The KV cache lives permanently in VRAM and **grows with every token generated**.

**Internal mechanics:**
- `"fp16"` — Standard `DynamicCache`. Each K/V tensor stored in BF16 (2 bytes per element).
- `"fp8"` — Custom `FP8DynamicCache`. Stores K/V in `torch.float8_e4m3fn` (1 byte per element) with per-head per-token scale factors. Dequantizes to BF16 before returning to attention. Quality loss is negligible.
- `"auto"` — Selects `"fp8"` if the model uses EXL3 quantization, `"fp16"` otherwise.

**VRAM usage at 8k context (Llama-70B, 80 layers, 8 KV heads, 128 head dim):**

| Mode | Formula | VRAM |
|------|---------|------|
| `fp16` | 80 × 2 × 8 × 128 × 8192 × 2 bytes | **2.68 GB** |
| `fp8` | 80 × 2 × 8 × 128 × 8192 × 1 byte | **1.38 GB** |

**How it affects speed:** FP8 saves ~1.3 GB of VRAM. This freed VRAM is directly consumed by `MAX_PINNED_LAYERS` to pin more transformer layers permanently in GPU memory, which **eliminates NVMe reads** for those layers on every token. More pinned layers = fewer disk reads = faster tokens.

**Recommendation:** Use `"auto"` (or `"fp8"` explicitly). The quality difference is imperceptible and you gain ~1.3 GB for layer pinning.

---

### `MAX_PINNED_LAYERS`

```python
MAX_PINNED_LAYERS = -1   # 0=shared only, -1=auto fill VRAM budget, N=first N layers
```

**What it does:** Controls how many model layers are permanently loaded ("pinned") into VRAM at startup. Pinned layers **never touch the NVMe SSD** — they skip the disk read, CPU→GPU transfer, and VRAM free cycle entirely.

**Internal mechanics (two-phase asymmetric pinning):**

1. **Phase 1 (shared layers):** Always pins `embed_tokens`, `norm`, and `lm_head` (~1 GB total). On budget miss: `continue` to try next (different sizes).
2. **Phase 2 (transformer layers):** Pins `layers.0`, `layers.1`, ..., in order. On budget miss: `break` (all transformer layers are the same size — if one doesn't fit, none will).

Budget formula: `total_VRAM - KV_cache_reservation - 1.5 GB overhead - 1 layer streaming buffer`

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0` | Shared layers only (~1 GB pinned) | Minimum VRAM usage; all 80 transformer layers streamed |
| `-1` | **Auto-fill** remaining VRAM budget | **Recommended.** Pins as many layers as will fit |
| `N` (e.g., `10`) | Shared + first N transformer layers | Manual control; still budget-capped |

**How it affects speed (16 GB GPU, Llama-70B EXL3 4.5 bpw, FP8 KV):**

| Pinned Layers | NVMe Reads/Token | Approx. Speedup |
|---------------|-------------------|-----------------|
| 0 (shared only) | 83 layer reads | Baseline |
| 28 (auto-fill) | 55 layer reads | **~35% fewer disk reads** |
| 40 (hypothetical 24 GB GPU) | 43 layer reads | ~50% fewer disk reads |

**Recommendation:** Always use `-1`. Combined with `KV_CACHE_MODE="fp8"`, this yields ~28 pinned transformer layers on a 16 GB GPU, eliminating 35% of all NVMe I/O per token.

**Verify in startup logs:**
```
[Vaayu-Pin] pinned model.embed_tokens: 0.53 GB  (cumulative 0.53 GB)
[Vaayu-Pin] pinned model.layers.0: 0.38 GB  (cumulative 1.44 GB)
...
[Vaayu-Pin] 31 layers pinned (28 transformer + 3 shared), 11.64 GB reserved
```

---

### `MAX_NEW_TOKENS`

```python
MAX_NEW_TOKENS = 512
```

**What it does:** Maximum number of tokens Vaayu will generate in the response. Generation stops when either `MAX_NEW_TOKENS` is reached or an EOS (end-of-sequence) token is produced.

**How it affects speed:**
- Each new token requires a **full forward pass** through all non-pinned layers (streaming weights from disk). With ~55 streamed layers on a 16 GB GPU, each token takes ~3-5 seconds.
- Total generation time ≈ `MAX_NEW_TOKENS × per_token_time` (worst case, if no EOS hit).
- With speculative decoding, effective throughput is 2-4× higher (see `DRAFT_TOKENS`), so 512 tokens might take ~2-4 minutes instead of ~25-40 minutes.

**How it affects output quality:**
- Too low → response is truncated mid-sentence (especially for code generation).
- Too high → wasted time if the model hits EOS early (no penalty — generation stops at EOS).

| Workload | Recommended Value |
|----------|-------------------|
| Short Q&A, chat | `64–128` |
| Code generation (single function) | `256–512` |
| Full-file code generation | `512–1024` |
| Long-form writing | `1024–2048` |

**Recommendation:** Set to `512` for coding agent workloads. There is no VRAM cost to a high value — it only determines the maximum loop iterations.

> **⚠️ This parameter does not exist on the server, deliberately.** A fixed cap is exactly what
> truncates code mid-function. `POST /v1/chat/completions` treats `max_tokens` as optional and
> defaults it to the whole remaining context window (`max_seq_len − prompt_len`, derived at load),
> so generation stops when the model is done — and reports `finish_reason: "length"` when it
> genuinely was cut off. See [**Running Vaayu as a Server**](#-running-vaayu-as-a-server-openai-compatible).

---

### `MAX_LENGTH`

```python
MAX_LENGTH = 2048
```

**What it does:** Maximum prompt token length before truncation. Passed to the HuggingFace tokenizer's `max_length` parameter. If your prompt exceeds this, it is **silently truncated** from the left (oldest tokens dropped).

**How it affects speed:**
- Longer prompts → more tokens to process during **prefill** (the initial forward pass that builds the KV cache from the prompt). Prefill streams all non-pinned layers once per chunk.
- With `PREFILL_CHUNK_SIZE=512`, a 2048-token prompt requires 4 prefill passes (4 × full layer streaming ≈ 12-20 seconds).
- After prefill, each new generated token costs the same regardless of prompt length (KV cache handles the history).

**How it affects output quality:**
- Too low → system prompt or user context is truncated, degrading response quality.
- Must be **greater than** `PREFILL_CHUNK_SIZE` for chunked prefill to engage.

| Workload | Recommended Value |
|----------|-------------------|
| Simple Q&A | `512–1024` |
| Coding agent (system + user prompt) | `2048–4096` |
| Long document analysis | `4096–8192` |

**Recommendation:** `2048` is a good default for coding tasks. Increase to `4096` if your system prompt + code context is large. Keep in mind: longer prompts increase KV cache VRAM usage proportionally.

---

### `PREFILL_CHUNK_SIZE`

```python
PREFILL_CHUNK_SIZE = 512   # 0 = disabled; tokens per prefill chunk
```

**What it does:** Splits long prompts into chunks of this size for the initial prefill phase. Without chunking, processing a 4000-token prompt in one pass requires the attention mechanism to allocate a massive intermediate tensor (quadratic in sequence length), which can **OOM on a 16 GB GPU**.

**Internal mechanics:**
1. Prompt is split: `4000 tokens → 8 chunks of 512`
2. Each chunk runs a full forward pass (streaming all non-pinned layers from NVMe)
3. KV cache accumulates across chunks — chunk 2 sees KV from chunk 1, etc.
4. After all chunks complete, decode-phase generation begins (one token at a time)

**How it affects speed:**
- **Prefill time** = `num_chunks × single_forward_pass_time`. A 2048-token prompt with `chunk_size=512` requires 4 forward passes (~12-20s).
- Smaller chunks = more forward passes = slower prefill, but **lower peak VRAM**.
- Larger chunks = fewer forward passes = faster prefill, but **higher peak VRAM** (risk of OOM).
- `0` disables chunking entirely — the full prompt is processed in one pass.

| Value | Prefill Passes (2048 tokens) | VRAM Risk | Speed |
|-------|------------------------------|-----------|-------|
| `0` (disabled) | 1 pass | ⚠️ High OOM risk | Fastest prefill |
| `256` | 8 passes | Very safe | Slowest prefill |
| `512` | 4 passes | Safe | **Good balance** |
| `1024` | 2 passes | Moderate risk | Faster prefill |

**Recommendation:** `512` is the sweet spot for 16 GB GPUs. Only increase to `1024` if you have 24+ GB VRAM. Set to `0` only for very short prompts (<512 tokens).

---

### Speculative Decoding Parameters

Speculative decoding is Vaayu's **most impactful speed optimization** for generation throughput. Instead of streaming 80 layers from NVMe to produce 1 token, speculative decoding produces **2-4 tokens per streaming pass**.

#### How Speculative Decoding Works

```
1. DRAFTER generates K candidate tokens instantly (in VRAM, no NVMe reads)
2. VAAYU VERIFIER streams all 80 layers ONCE to verify all K candidates in parallel
3. Accept longest prefix where verifier agrees with drafter
4. If all K accepted → bonus token from verifier's extra logit = K+1 tokens
5. If mismatch at position i → accept i tokens + correction token = i+1 tokens
```

**Effective throughput:** If the drafter's acceptance rate is 60%, and you draft 5 tokens per round, you get ~3-4 tokens per verifier pass instead of 1. This is a **3-4× speedup** with the same NVMe bandwidth.

There are **two mutually exclusive modes**:

---

#### Mode A: Standard Drafter (`DRAFTER_MODEL_PATH`)

```python
DRAFTER_MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"
USE_PROMPT_LOOKUP  = False
```

**What it does:** Loads a small language model (0.5B-1.5B parameters) permanently into VRAM. This drafter model generates candidate tokens autoregressively at near-zero latency (no disk I/O — it's fully in VRAM).

**Requirements:**
- The drafter must share a **compatible tokenizer** with the verifier (pair within the same model family — Llama drafter for Llama verifier, Qwen drafter for Qwen verifier).
- The drafter consumes VRAM permanently, reducing budget for layer pinning.

| Drafter Model | VRAM Cost | Best For |
|---------------|-----------|----------|
| Llama-3.2-1B-Instruct | ~0.6 GB | General Llama-family tasks |
| Qwen2.5-0.5B | ~0.3 GB | Minimal VRAM impact (Qwen family) |
| Qwen2.5-Coder-1.5B | ~1.0 GB | Higher acceptance rate for code |

**Acceptance rate:** 40-80% depending on drafter/verifier alignment. Better for **instruction-following, chat, and novel creative responses** where the small model's priors partially align with the large model.

**Speed impact:** Each verifier pass produces `accepted + 1` tokens (where accepted ≈ 2-4 out of 5 drafts). With a 5s verifier pass, effective speed goes from ~5 s/token to ~1.2-2.5 s/token.

---

#### Mode B: Prompt Lookup Decoding (`USE_PROMPT_LOOKUP`)

```python
DRAFTER_MODEL_PATH = None
USE_PROMPT_LOOKUP  = True
```

**What it does:** Instead of using a neural network drafter, PLD searches the **existing generated context** for n-gram matches of the last few tokens. If it finds a match, it drafts the tokens that followed that n-gram in the context. This is a pure CPU operation with **zero VRAM cost**.

**Internal mechanics:**
1. Take the last 1-3 generated tokens as an n-gram query.
2. Search backwards through `input_ids` for matching n-grams.
3. If found: draft the tokens that followed the match in the original context.
4. If not found: fall back to single-token greedy decoding (no speedup, negligible overhead).

**When PLD works well (high match rate):**
- **Code generation**: Variable names, function signatures, import statements, and structural patterns (`async def`, `try:`, `except:`, `return`) repeat frequently.
- **Template/boilerplate text**: Repetitive documentation, configuration files.
- **Code completion**: Continuing partially written code where patterns from earlier in the file reappear.

**When PLD fails (0% match rate, no speedup):**
- **Instruction → novel response**: When the model generates creative text not present in the prompt.
- **Open-ended chat**: Free-form conversation where output tokens don't match input patterns.
- **Short prompts**: Fewer tokens in context = fewer n-grams to match against.

| Workload | PLD Match Rate | Effective Speedup |
|----------|---------------|-------------------|
| Code generation (large prompt) | 30-60% | 1.5-2.5× |
| Repetitive boilerplate | 50-80% | 2-4× |
| Novel instruction response | ~0% | No speedup (1×) |
| Short Q&A | ~0% | No speedup (1×) |

**Recommendation:** Use PLD (`Mode B`) for code generation workloads. Use Standard Drafter (`Mode A`) for general-purpose chat and instruction-following. PLD is free — if it finds no matches, it falls back to greedy decoding with near-zero overhead.

---

### Speeding up MoE with prompt lookup (1.31× on MiniMax-M2.7)

Speculative decoding used to be refused outright on MoE checkpoints. It is now
decided at **runtime**, per model, from that model's own behaviour — so on the
one configuration where it pays, you can have it.

**Measured 2026-09-06, this box, greedy, generated ids byte-identical to plain
decode in every arm:**

| checkpoint | result | what the gate decided |
|---|---|---|
| MiniMax-M2.7 | **1.307×** (1.7795 → 2.3253 tok/s) | `run` — acceptance 1.00 against a 0.61 break-even |
| Qwen3-235B | **0.932×** (1.080 → 1.0064 tok/s) | `off_acceptance` — acceptance 0.351 against a 0.528 bar |

#### Enabling it

```python
# vaayu/server/config_local.py
USE_PROMPT_LOOKUP = True     # zero VRAM; the gate decides whether it runs
DRAFT_TOKENS      = 3        # measured best on MiniMax-M2.7
```

That is the whole change. `ENABLE_THINKING` is already `False` on the server,
which matters more than it looks — see below.

#### The three conditions. All of them, or you get nothing

1. **The right model.** MiniMax-M2.7 wins. **Qwen3-235B loses 6.8 %** — the gate
   catches it and switches off, but the trial costs something. If you serve
   Qwen, leave `USE_PROMPT_LOOKUP` off.
2. **Thinking suppressed.** Both MoE templates end their assistant turn with a
   hard-coded `<think>`, and ignore the `enable_thinking` flag entirely. The
   server closes that block for them (`render_prompt`), which is why its
   `ENABLE_THINKING = False` default is load-bearing here. **A request that
   sends `"enable_thinking": true` gets prose, and prose kills the match rate.**
   Calling the library directly rather than through the server does *not* get
   this — the block-closing lives in the server's prompt path.
3. **Copy-heavy work.** PLD drafts by finding an n-gram from earlier in the
   context. Re-emitting a file with a small edit — the agentic loop this engine
   is built for — matches constantly. Writing novel prose matches nothing.

#### Checking that it actually ran

```bash
curl -s localhost:9999/v1/models | python -m json.tool
```

```json
"speculative": {
  "requested": "prompt_lookup", "active": true,
  "calibration": {
    "observed_union_gb": 12.45, "budget_gb": 14.37,
    "breakeven_acceptance": 0.6094, "measured_acceptance": 1.0,
    "drafting_rounds": 16, "decision": "run"
  }
}
```

`decision` is the answer: `run` (it is speculating), `off_acceptance` (it tried
and this model does not pay), `off_union` (a verify pass does not fit the expert
cache), or `off_no_routing` (no routing evidence — that one is a bug, report it).
The same object is written to `logs/latest/env.json` for a support bundle.

#### What it costs when it says no

The gate cannot know a model's acceptance rate without trying, so it spends its
calibration steps and up to 64 drafting rounds finding out. On Qwen over a
128-token run that is the 6.8 % above. It is paid **once per process**, not per
request — but on short generations against a model you already know does not
pay, switching `USE_PROMPT_LOOKUP` off by hand still beats letting the gate
discover it.

**A weighted drafter (`DRAFTER_MODEL_PATH`) is still refused on MoE** and that
has not changed: it costs 3.09 GB of VRAM that the expert cache needs, and the
same acceptance problem applies on top. This section is about PLD only.

---

#### `DRAFT_TOKENS`

```python
DRAFT_TOKENS = 15
```

**What it does:** Number of candidate tokens the drafter generates per speculative round. The verifier then checks all of them in **one forward pass**.

**How it affects speed:**

| Value | Verifier Input Size | Benefit | Risk |
|-------|---------------------|---------|------|
| `3` | 4 tokens verified at once | Low — max 4 tokens/pass | Very safe; almost all accepted |
| `5` | 6 tokens verified at once | Good balance — 3-6 tokens/pass | Standard choice |
| `10` | 11 tokens verified at once | Higher potential — up to 11 tokens/pass | Later drafts less likely to match |
| `15` | 16 tokens verified at once | Maximum throughput on matching workloads | Diminishing returns past ~10 |

**Trade-offs:**
- **Higher `DRAFT_TOKENS`** → More tokens verified per pass, but the acceptance rate drops for later positions (the drafter's predictions diverge from the verifier's). The verifier pass also takes slightly longer (processing 16 tokens vs 6), though this is marginal compared to NVMe I/O time.
- **Lower `DRAFT_TOKENS`** → Higher acceptance rate per draft, but fewer tokens per pass.
- With **PLD specifically**, higher values work well because when a match is found, the subsequent tokens tend to match too (they're literal copies from the context). Set to `10-15` for PLD.
- With **Standard Drafter**, `5-8` is the sweet spot. Beyond ~8, acceptance rates drop significantly for novel generation.

| Mode | Recommended DRAFT_TOKENS |
|------|--------------------------|
| Standard Drafter (chat/instruction) | `5` |
| Standard Drafter (code) | `5–8` |
| Prompt Lookup Decoding (code) | `10–15` |
| Prompt Lookup Decoding (boilerplate) | `15` |

---

### 🏁 Recommended Configurations

#### Coding Agent (16 GB GPU, EXL3 model) — Maximum Speed

```python
MODEL_PATH         = "/path/to/Llama-3.3-70B-Instruct-exl3-4.5bpw"
KV_CACHE_MODE      = "auto"          # bf16 ring buffer for EXL3 → 2.0 bytes/element
MAX_PINNED_LAYERS  = -1              # auto-fill → ~28 layers pinned
MAX_NEW_TOKENS     = 512             # room for full code output
MAX_LENGTH         = 2048            # system prompt + code context
PREFILL_CHUNK_SIZE = 512             # safe chunking for long prompts
DRAFTER_MODEL_PATH = None            # no drafter VRAM cost
USE_PROMPT_LOOKUP  = True            # zero-cost PLD for code patterns
DRAFT_TOKENS       = 15              # maximize PLD matches
```

#### General Chat (16 GB GPU, EXL3 model) — Balanced

```python
MODEL_PATH         = "/path/to/Llama-3.3-70B-Instruct-exl3-4.5bpw"
KV_CACHE_MODE      = "auto"
MAX_PINNED_LAYERS  = -1
MAX_NEW_TOKENS     = 256
MAX_LENGTH         = 1024
PREFILL_CHUNK_SIZE = 512
DRAFTER_MODEL_PATH = "/path/to/Llama-3.2-1B-Instruct"  # ~0.6 GB VRAM
USE_PROMPT_LOOKUP  = False
DRAFT_TOKENS       = 5
```

#### Minimum VRAM (16 GB GPU, BF16 model) — Conservative

```python
MODEL_PATH         = "/path/to/Llama-3.3-70B-Instruct"
KV_CACHE_MODE      = "fp16"
MAX_PINNED_LAYERS  = 0               # shared layers only
MAX_NEW_TOKENS     = 128
MAX_LENGTH         = 512
PREFILL_CHUNK_SIZE = 256             # small chunks to avoid OOM
DRAFTER_MODEL_PATH = None
USE_PROMPT_LOOKUP  = False
DRAFT_TOKENS       = 5
```

---

## 🌐 Running Vaayu as a Server (OpenAI-compatible)

> **Full reference:** [**documentation/server.md**](documentation/server.md) — every setting, every
> request field, curl examples for tool calling and streaming, and the coding-agent
> integration checklist.

Vaayu runs as a long-lived HTTP server speaking the OpenAI API. Any client that already
speaks it — a coding agent, Cline, Aider, LiteLLM, the `openai` SDK — works with **no adapter**.

The server takes **no command-line arguments** — set `MODEL_PATH` in
[`vaayu/server/config.py`](vaayu/server/config.py) (or in the gitignored
`vaayu/server/config_local.py`, which shadows it) and run:

```bash
./venv/bin/python -m vaayu.server
```

```bash
# streams by default (DEFAULT_STREAM = True) -- -N keeps curl from buffering it
# session_id is REQUIRED: give each independent agent stream its own
curl -sN localhost:9999/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "session_id": "agent-7",
  "messages": [{"role": "user", "content": "write a binary search in Python"}]
}'

# add "stream": false for one JSON body -- but nothing arrives until the LAST
# token, which on a slow checkpoint with no max_tokens can be an hour
```

**`session_id` is required** (`REQUIRE_SESSION_ID`, default on): without one the prefix
cache keys on a hash of the prompt's first 64 tokens, so two agents sharing a system
prompt collide on a single pool entry and reuse collapses to zero. OpenAI's `user` field
counts as a session, so an OpenAI-native client is not locked out. The session comes back
on the `x-vaayu-session-id` header, in the response body, and on every stream chunk.

Two more server defaults differ from OpenAI's on purpose: **`stream` defaults to true**, and
**`ENABLE_THINKING` defaults to false** — a reasoning trace costs the same tok/s as the
answer, so it is suppressed unless a request asks for it with `"enable_thinking": true`.

`POST /v1/chat/completions` (streaming and not) · `GET /v1/models` · `GET /health`.

### There is no `MAX_NEW_TOKENS` here — omit `max_tokens`

The single most important difference from the test scripts. **Omitting `max_tokens` is the
normal case**: the cap becomes `max_seq_len − prompt_len`, derived at load, so generation
stops when the model is **done** rather than at a fixed count. No more code truncated
mid-function.

When output genuinely was cut off you are **told**, and an agent must branch on it:

| `finish_reason` | meaning | what your agent does |
|---|---|---|
| `stop` | EOS or a stop string | use the output |
| `length` | cap reached — **output is truncated** | resend to continue; never write it to a file |
| `tool_calls` | the model called a tool | dispatch, append results, continue |

Streaming does **not** fix truncation — streaming is delivery, truncation is stopping policy.
It ships for time-to-first-token visibility, early abort, and mid-stream tool-call detection.

### The five must-knows for a coding agent

1. **Omit `max_tokens`, branch on `finish_reason`.** Most agent frameworks default it to
   1024 or 4096 — turn that off; a client-side number can only make the window smaller.
2. **Resend the full history every turn.** That *is* the fast path: prefix caching reuses the
   longest matching prefix and prefills only the tail (136–377× TTFT). OpenAI clients already
   do this, so the default is optimal. Don't trim from the front — it breaks the prefix.
3. **One request at a time.** FIFO queue (`QUEUE_DEPTH`, default 8), `503` + `Retry-After`
   past it. Parallel sub-agent calls will serialise; handle `503` with a retry.
4. **Context is the binding constraint, not speed.** Read the real number from `/v1/models`
   (8192 on all three big checkpoints). Keep MCP tool schemas terse — they eat it fast.
5. **MCP belongs in your agent.** Vaayu speaks `tools` / `tool_calls` and nothing more. Tool
   support is a property of the checkpoint's chat template, detected by **probing the
   template**, never by model name; a `tools` request against a template that can't render
   them is a `400` with a named reason rather than a silent drop.

### What the defaults buy

Measured end-to-end on Qwen3-235B — 2 concurrent sessions x 3 rounds, ~4.6k-token shared
prefix, agent-shaped. Same box, same day; only configuration differs:

| | first working config | shipped defaults | |
|---|---|---|---|
| total wall, 6 turns | 477.0 s | **198.1 s** | **2.41x** |
| warm turn | 80.3 s | **28.9 s** | 2.78x |
| prefix reuse | off | 83 % of prompt tokens skipped | |

Full breakdown and the per-lever numbers: [documentation/server.md §7](documentation/server.md#7-making-it-faster).

### Key flags

| flag | default | why you care |
|---|---|---|
| `MODEL_PATH` | `""` | required. One model per process — a load costs minutes, so swapping is a restart |
| `SERVED_MODEL_NAME` | `None` (checkpoint path) | short stable `id` for clients to hardcode |
| `QUEUE_DEPTH` | `8` | waiting requests before `503`. One runs at a time regardless |
| `MAX_PINNED_LAYERS` | **`-1`** | auto-fills VRAM with transformer layers. Setting `0` costs **13× on the 1B, 2.0× on MiniMax, 1.27× on the 70B** — the gain scales with how much of the model fits |
| `ENABLE_PREFIX_CACHE` | **`True`** | leave it on. Reuse is the largest lever here |
| `PREFIX_COMPRESS` | `None` (set `9`) | ~1.6× the pooled sessions (0.605× bf16: 9 bits plus 8 exact channels per layer, which is what clears the quality bar on both MoE forwards). Reuse collapses to **zero**, not gradually, once the pool is over capacity |
| `DRAFTER_MODEL_PATH` | `None` | **5.57×** on streamed dense. Still refused at load on MoE (a weighted drafter costs VRAM the expert cache needs) |
| `USE_PROMPT_LOOKUP` | `"auto"` | `"auto"` = ON for dense, OFF for MoE, resolved at load. **1.94×** on streamed dense, zero VRAM. On MoE a runtime gate decides per model — **1.31× on MiniMax-M2.7**, correctly refused on Qwen3-235B. See [Speeding up MoE with prompt lookup](#speeding-up-moe-with-prompt-lookup-131-on-minimax-m27) |
| `RAM_CACHE_BUDGET_GB` | `None` *(derived from total RAM)* | pageable RAM holding layers that don't fit VRAM. On the 70B: 33/80 layers, 264 hits/248 misses. Dense only — MoE experts use the **pinned** expert cache, a scarcer resource with its own 8.5 GiB ceiling |
| `--prefix-sessions` | `3` | concurrent sessions the prefix pool must hold before the MoE expert cache claims the rest. Lower = faster decode, fewer cached sessions |
| `ATTN_IMPLEMENTATION` | `None` (sdpa) | `"flash_attention_2"` derives more context at the same reserve |

Full list, request-body reference, tool-calling examples and the troubleshooting table:
[**documentation/server.md**](documentation/server.md).

### Limits, stated plainly

One model per process, no hot-swap. One request in flight, `n=1` only. No continuous
batching (N≥2 at long context is a hardware wall on 16 GB, not a missing feature). No
`/v1/completions`, no MCP client, **no auth or TLS** — bind to `127.0.0.1` or put a proxy in
front. Prefix caching is disabled under `KV_CACHE_MODE = "fp16"`; `/health` reports it.

---

## Reporting a problem

Zip the newest run directory and attach it:

```bash
zip -r vaayu-report.zip logs/run-2026-09-04T12-00-00Z/
```

`logs/latest` is a symlink to that directory, so `ls -l logs/latest` tells you
which one it is.

It contains:

| file | what it holds |
|---|---|
| `env.json` | the machine, the GPU, the NVMe, the effective config, the derived model geometry and budgets, and every startup warning |
| `server.log` | the whole run's output, timestamped, one line per line |
| `requests.jsonl` | one row per request — completed, refused before generation, raised, or abandoned mid-stream by the client. A request never ends without a row |
| `sessions/<id>.log` | the same stream, split per session |

**What is redacted.** Absolute paths are replaced by a basename plus a hash of
the full path, so `env.json` does not carry your username or your directory
layout. Nothing else in the bundle is filtered — skim `server.log` before
sending it if your prompts are sensitive. `requests.jsonl` is not filtered
either: an `error` row carries the exception's message and full traceback, and
those frames name absolute paths on your machine.

**If the server will not start**, there is no run directory to send. Run this
instead and paste the output:

```bash
./venv/bin/python -m vaayu.server.diagnose
```

It prints the same environment snapshot without loading any weights, and flags
the settings that commonly cause trouble.

---

## 🏃 Quickstart Inference Test

Edit the `CONFIG` block at the top of `test_quick.py`, then run:
```bash
source venv/bin/activate
python test_quick.py
```

*Example API usage:*
```python
from vaayu import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/models/Llama-3.3-70B-Instruct-exl3-4.5bpw",
    kv_cache_mode="auto",        # "auto" selects bf16 for EXL3, FP16 otherwise
                                 # ("fp8" still available; it is 1.5x LARGER — it
                                 #  keeps a dequantized mirror alongside the fp8 store)
    max_pinned_layers=-1,        # auto-fill remaining VRAM with transformer layers
    prefill_chunk_size=512,      # chunk long prompts to avoid VRAM spikes
    drafter_model_path=None,     # set to enable speculative decoding
    use_prompt_lookup=True,      # zero-cost PLD for code generation
    draft_tokens=15,             # candidates per speculative round
)

input_tokens = model.tokenizer(
    "What is the capital of India?",
    return_tensors="pt",
    padding=False
)

# Standard generation (no speculator set):
generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
)
print(model.tokenizer.decode(generation_output.sequences[0]))

# Speculative generation (when speculator is active):
# output_ids = model.speculative_generate(input_tokens['input_ids'].cuda(), max_new_tokens=512)
# print(model.tokenizer.decode(output_ids[0]))
```
