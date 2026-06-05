# Vaayu Engineering Bugfix & Maintenance Log

This document is the definitive chronicle of every engineering victory, bugfix, and systemic optimization applied during the development of the Vaayu engine. It serves as an active developer handbook and historic record, documenting error signatures, root cause analyses, and precise patching logics.

---

## 1. Hugging Face Core API Migrations

### Victory 1: The `DynamicCache` Subscript Resolution
* **Context**: Modern versions of `transformers` encapsulate key-value hidden states inside a dedicated `DynamicCache` class instead of legacy raw PyTorch tuple arrays.
* **Error Signature**: `TypeError: 'DynamicCache' object is not subscriptable`
* **Root Cause**: Because Vaayu's layer-streaming architecture handles and clears memory buffers independently, the complex caching object caused downstream slice unpacking failures during decoder calls.
* **The Resolution**: Wiped out dynamic cache references within `prepare_inputs_for_generation()` context logic. Modified `get_past_key_values_cache_seq_len()` to securely verify `hasattr(past_key_values, "get_seq_length")` rather than aggressively referencing list layers, seamlessly normalizing both cache types.

### Victory 2: RoPE Arguments Override
* **Context**: Modern attention modules (like `Qwen2Attention`) compute rotations dynamically at the top-level, deleting the `position_ids` kwarg and forcing layers to accept a massive Tuple Tensor of `position_embeddings` containing raw `(cos, sin)` data payloads.
* **Error Signature**: `TypeError: Qwen2Attention.forward() missing 1 required positional argument: 'position_embeddings'`
* **Root Cause**: Passing raw `position_ids` integers to modern decoder layers fails because they now expect pre-computed trigonometric embeddings.
* **The Resolution**: Implemented a top-level inference intercept: during sequence forward passes, Vaayu intercept-initializes the actual rotary embedding class (`real_rotary = rotary_emb_cls(config)`), computes the full sequence embeddings using configuration variables (bypassing uninitialized CUDA memory in `init_empty_weights()`), and slices the precise window bounds for rotary phases to pass back as `position_embeddings` in the layer loop.

### Victory 3: Layer Forward Dimension Collapses
* **Context**: Specific sequence projections and normalization return layers no longer rigidly adhere to standard tuple packing mechanisms. Returning raw 2D Tensors instead of Tuples functionally wiped batch dimensions when statically hardcoded scripts evaluated them via `layer_out[0]`.
* **Error Signature**: Dimensional tensor mismatches crashing iteration layers.
* **Root Cause**: Explicitly unpacking returns using `layer_out[0]` caused crashes when layers returned a raw `Tensor` instead of a `Tuple`.
* **The Resolution**: Inserted dynamic runtime validation constraints `isinstance(layer_out, tuple)` into state extraction processing scripts, securely capturing valid returns from normalization passes indiscriminately.

---

## 2. Universal Model Architecture Support

### Victory 4: Bypassing Tied Weights Missing Shards
* **Model Context**: `Qwen/Qwen2.5-3B`
* **Error Signature**: `FileNotFoundError: No such file or directory: splitted_model/lm_head.safetensors`
* **Root Cause**: To save disk storage and bandwidth, modern models merge string embeddings directly onto output sequence layers (`tie_word_embeddings=True`). Consequently, `lm_head.weight` is omitted from the safetensors index since it is identical to `embed_tokens.weight`.
* **The Resolution**: Encapsulated extraction paths targeting `lm_head` natively inside defensive `try/except FileNotFoundError` blocks within `load_layer_to_cpu()`. If triggered, the loader redirects to download the `embed_tokens` weight from disk, and modifies its dictionary payload keys inline (translating `"model.embed_tokens.weight"` seamlessly into `"lm_head.weight"`), synthesizing weight-tied parameters on the fly.

### Victory 5: Stripping Experimental Nested Configuration Paths
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Signature**: `ValueError: invalid literal for int() with base 10: 'layers'` & Splitting Execution Skips
* **Root Cause**: Experimental architectures bury components underneath complex parameter prefix structures (e.g. `transformer.model.layers.0` instead of `model.layers.0`), which broke legacy hardcoded character slice operations (e.g. `k[12:]`).
* **The Resolution**:
  * **Dynamic Layer Counting**: Replaced absolute character slices with dynamic delimiters: `k.split("model.layers.")[1].split('.')[0]` ensuring exact integer layer index retrieval globally.
  * **Keyword Homogenization**: Restructured mapping loops to search via the contextual wildcard `layer in k` instead of explicit prefix beginnings. Upon discovery, stripped the experimental repository prefixes permanently on extraction via `k[k.find(layer):]`, ensuring Vaayu receives consistently named payload layers.

### Victory 6: Shard Naming Format Normalization
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Signature**: `FileNotFoundError` seeking specific file padding formats (`00001-of-00001` vs `0001-of-00001`).
* **Root Cause**: Depending on how developers push shards to Hugging Face, file naming formats possess wildly differing padding.
* **The Resolution**: Dismantled manual string building predictions within shard caching logic. Transitioned execution mappings to strictly scrape filenames from `set(index.values())` dynamically, perfectly locking the chunk download loop to mirror exact developer formats.

### Victory 7: Core Configurations Sanitization Protocol
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Signature**: Successive downstream cascade crashes within HF: `AttributeError: Qwen3_5Config has no attribute...`
* **Root Cause**: Experimental or multi-modal configs failed to expose standard transformer properties, crashing PyTorch generation graph executions.
* **The Resolution**: Implemented a comprehensive **Configuration Sanitization** layer prior to instantiation. Unpacked deeply nested parameter aliases natively (e.g., mapping `hidden_size` targets via `config.text_config.hidden_size`, `config.d_model`, or `config.n_embd`), and blanketed omissions with standard generic fallbacks.

### Victory 8: Physical Tensor Embedding Scale Limits
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Signature**: `ValueError: Trying to set tensor [248320, 1024] to [248044, 1024]`
* **Root Cause**: Hugging Face built embedding matrices structured to tokenizer boundaries (e.g., `vocab_size = 248044`). However, physical files padded their internal sequence blocks higher for parallel batch compilation efficiencies (`248320`), causing shape mismatches.
* **The Resolution**: Implemented a **Physical Dimension Prober** algorithm. Upon processing downloaded arrays, Vaayu extracts index layouts using zero-cost safetensors header reads `get_slice("model.embed_tokens.weight").get_shape()`. Binding the native configuration structure (`vocab_size`, `hidden_size`) tightly to physical dimension offsets prevents shape mismatch crashes.

---

## 3. Single-File Caching & VLM Parameter Path Mapping

### Victory 9: Single-File SafeTensors Download Exclusion
* **Model Context**: Small/unsharded models (e.g., `Qwen3-0.6B`, under 2GB).
* **Error Signature**: `FileNotFoundError: No model files found in {...}. Expected one of: pytorch_model.bin.index.json, model.safetensors.index.json, model.safetensors, or pytorch_model.bin`
* **Root Cause**: To optimize bandwidth, Vaayu initially ignored large binary files (`*.safetensors`, `*.bin`) during initial caching, downloading only `index.json` manifests. Single-file configurations do not have an index file, leading to empty directories.
* **The Resolution**: Integrated dynamic index detection post-fetch. If `huggingface_hub` executes but no index files populate locally, the script gracefully infers it is a single-file model and triggers a forced secondary download targeting `['*.safetensors', '*.bin']`, allowing the monolithic loader to split the file.

### Victory 10: AutoModel Framework Fallbacks
* **Model Context**: Unmapped model classes (e.g., `Qwen3ForCausalLM`, `Gemma4ForConditionalGeneration`).
* **Error Signature**: `unknown architecture: Gemma4ForConditionalGeneration, trying to use Llama2...`
* **Root Cause**: `auto_model.py` mapped strings into specific handler modules. Missing classes fell back to Llama, causing configuration mismatches.
* **The Resolution**: Hard-mapped Qwen3 strings directly to `AirLLMQWen2` and Gemma strings to `AirLLMLlama2` to safely suppress fallback cascade failures.

### Victory 11: Dynamic Multi-Modal PyTorch Parameter Traversal & Injection
* **Model Context**: Complex/Wrapper Architectures and Vision-Language Models (e.g., `google/gemma-4-E2B`).
* **Error Signature**: `AttributeError: 'Gemma4Model' object has no attribute 'embed_tokens'`
* **Root Cause**: Multi-modal architectures wrap text logic inside deep nesting (e.g. `model.language_model.embed_tokens`). Slicing the file resulted in filenames matching shortcodes (`model.embed_tokens.safetensors`), but injecting them back into PyTorch crashed because the model's actual attribute was under `.language_model`.
* **The Resolution**:
  * Decoupled file naming from PyTorch attributes using a new dynamic path prober `_resolve_and_map_path()`.
  * The prober systematically crawls `model.named_modules()`, isolates the terminal module paths, and registers them in `self._path_remappings` (e.g. mapping `"model.embed_tokens"` $\rightarrow$ `"model.language_model.embed_tokens"`).
  * Immediately before parameter injection, the system intercepts `state_dict` keys, upgrading legacy keys back into PyTorch's Wrapper contexts natively.

---

## 4. Subclass Tensor Injectors & JIT Compilation Limits

### Victory 12: Non-Sequential SafeTensor Shard Caching
* **Model Context**: `gpt-oss-20b` (And generalized experimental models).
* **Error Signature**: `RuntimeError: Tensor on device cuda:0 is not on the expected device meta!` (Caused downstream by empty layer shards).
* **Root Cause**: Standard models publish sequentially indexed shards (`model-00001-of-00003`). `gpt-oss-20b` published 0-indexed shards (`model-00000-of-00002.safetensors`), and mapping was non-monotonic (e.g. `embed_tokens` was in chunk 2, but `layers.0` was in chunk 1). Legacy loader logic was strictly monotonic, skipping the 0-indexed chunk entirely.
* **The Resolution**: Restructured caching to initiate at an un-indexed state (`shard = -1`), recognizing `0` as an acceptable target. Mapped the download loops to dynamically query file paths directly via the parsed `index.values()` list, matching the developer's exact layout.

### Victory 13: PyTorch 2.x Tensor Subclasses & Advanced MXFP4 Quantizer Hooks
* **Model Context**: `gpt-oss-20b` / Experimental `mxfp4` implementations.
* **Error Signature**: `ValueError: Mxfp4GptOssExperts() does not have a parameter or a buffer named down_proj_blocks.`
* **Root Cause**: Advanced quantizers use custom PyTorch tensor subclasses to unflatten blocks (e.g., unflattening `.down_proj_blocks` into `.down_proj` in Python memory). Mapping parameters 1-to-1 using `accelerate` crashed because the attribute names on meta did not match the names on disk.
* **The Resolution**: Bypassed rigid `set_module_tensor_to_device` 1:1 loops. Bundled parameters neatly and invoked `self.model.load_state_dict(standard_params, strict=False, assign=True)` wrapped in graceful exception fallbacks. This delegates complex unflattening rules back to native PyTorch subclass implementations.

### Victory 14: Triton JIT C-Compiler Dependency
* **Environment Context**: Deploying dynamic JIT architectures (`mxfp4`, `flash-attn`, `deepspeed`) directly on OS environments.
* **Error Signature**: `subprocess.CalledProcessError... Python.h: No such file or directory`
* **Root Cause**: Triton compiles dynamic CUDA math kernels on the fly, which requires local Python developer headers (`Python.h`) to complete the compilation. Standard Python installs omit these files.
* **The Resolution**: Documented the two clean workarounds for end users:
  1. **OS-Level Headers**: Run `sudo apt-get install python3-dev` to install compilation headers.
  2. **Conda Isolation (Recommended)**: Run the project inside a Conda virtual environment. Conda packages the complete `Python.h` binaries internally within its environment, shielding compilation tasks from host machine mismatches.

---

## 5. Rust PyO3 Triple-Copy Resolution

### Victory 15: Overcoming the PyO3 Triple-Copy Bottleneck
* **Context**: The very first `vaayu_rust` implementation.
* **Error Signature**: Per-token execution latency regressed from ~80s to **190s**.
* **Root Cause**: PyO3 C-bindings made 3 copies of every layer:
  ```
  mmap File Region ────► Owned Rust Vec<u8> (Copy 1) 
                   ────► NumPy Array via PyO3 (Copy 2) 
                   ────► PyTorch .clone() Tensor (Copy 3)
  ```
  This wasted ~1.5s per layer in system memory copies.
* **The Resolution**: Completely redesigned the Rust module from the ground up:
  * Rust no longer owns or allocates data.
  * Python pre-allocates static pinned buffers using `torch.empty(..., pin_memory=True)`.
  * Python passes the raw buffer memory pointer (`buf.data_ptr()`) directly to Rust.
  * Rust orchestrates `io_uring` to read directly from NVMe into the PyTorch pinned memory buffer, bypassing the PyO3 boundary completely. Data never crosses copy boundaries.
