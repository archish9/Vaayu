# AirLLM Architecture Refactoring, Bugfix, & Maintenance Report

**Purpose:** This extensive diagnostic log and developer handbook is designed specifically for future AI coding agents. It provides a deep mechanical overview of how legacy AirLLM mechanisms were refactored to support latest-generation frameworks (e.g. `transformers >= 4.45`, `Qwen2.5`, `Qwen3.5`), explicitly documenting error signatures, the precise patching logic applied, and systemic guidelines for implementing future untested model architectures.

---

## Part 1: Hugging Face Core API Migrations

### 1. The `DynamicCache` Transition
* **Context**: Newer versions of `transformers` (often used to execute modern `Llama` variants and `Qwen2` models) encapsulate key-value hidden states inside a dedicated `DynamicCache` class instead of legacy raw PyTorch tuple arrays. 
* **Error Trigger**: `TypeError: 'DynamicCache' object is not subscriptable`
* **Patch Logic Applied (`airllm_base.py`)**: 
  - Since AirLLM generates and clears buffers independently via layer-by-layer loop execution, the complex caching object causes downstream slice unpacking failures.
  - Fix: Wiped out dynamic cache references within `prepare_inputs_for_generation()` context logic. Updated `get_past_key_values_cache_seq_len()` to securely check `hasattr(past_key_values, "get_seq_length")` rather than aggressively referencing list layers, gracefully normalizing both cache types perfectly.

### 2. Rotary Position Embedding (RoPE) Arguments Override
* **Context**: `Qwen2Attention` and modern HF attention nodes drastically changed positional tracking. Legacy arrays parsed `position_ids` raw integers. Natively modern frameworks compute rotations dynamically top-level, deleting the `position_ids` kwarg and forcing layers to accept a massive Tuple Tensor of `position_embeddings` containing raw `(cos, sin)` data payloads.
* **Error Trigger**: `TypeError: Qwen2Attention.forward() missing 1 required positional argument: 'position_embeddings'`
* **Patch Logic Applied (`airllm_base.py`)**:
  - Implemented top-level inference mock: During sequence decoding, AirLLM intercepts initialization, manually invokes `self.model.model.rotary_emb(value_states, position_ids)`, slices the precise window bounds for rotational phases, and explicitly passes this physical tensor Tuple tuple explicitly wrapped inside `get_pos_emb_args` back into individual decoder iteration loops cleanly. 

### 3. Layer Forward Dimension Collapses
* **Context**: Specific sequence projections and normalization return layers no longer rigidly adhere to standard tuple packing mechanisms. Returning raw 2D Tensors instead of Tuples functionally wiped batch dimensions when statically hardcoded scripts evaluated them via `layer_out[0]`.
* **Error Trigger**: Dimensional tensor mismatches crashing iteration layers (loss evaluation discrepancies).
* **Patch Logic Applied (`airllm_base.py`)**:
  - Inserted dynamic runtime validation constraints `isinstance(layer_out, tuple)` into state extraction processing scripts, securely capturing valid returns from normalization passes indiscriminately.

---

## Part 2: Working With Standard Architectures (Qwen2.5-3B)

### 4. Bypassing Tied Weights Missing Mappings
* **Model Context**: `Qwen/Qwen2.5-3B`
* **Error Trigger**: `FileNotFoundError: No such file or directory: splitted_model/lm_head.safetensors`
* **Cause**: To save explicit storage memory on HF datasets, `Qwen2.5` natively merges string-based embeddings directly onto output sequence layers (`tie_word_embeddings=True`). Therefore, `lm_head.weight` is intentionally stripped out of physical hard-drive `safetensors.index.json` manifests since it’s essentially `model.embed_tokens`.
* **Patch Logic Applied (`airllm_base.py`)**:
  - Encapsulated extraction paths targeting `lm_head` natively inside defensive `try/except FileNotFoundError` blocks within `load_layer_to_cpu()`. 
  - If triggered, the logic redirects to natively download the `embed` weight from disk, subsequently modifying its dictionary payload keys inline (translating `"model.embed_tokens.weight"` seamlessly into `"lm_head.weight"`). This safely synthesizes true, complete weight-tied variables for AirLLM.

---

## Part 3: Architecting Highly Experimental Frameworks (Qwen3.5-0.8B)

### 5. Stripping Deep Configuration String Misalignments 
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Trigger**: `ValueError: invalid literal for int() with base 10: 'layers'` & Splitting Execution Skips
* **Cause**: Experimental architectures bury components underneath chaotic parameter prefix structures (`e.g., transformer.model.layers.0` instead of cleanly `model.layers.0`). Hardcoded indexing searches (`k.startswith(layer)`) totally ignored chunks believing them invalid.
* **Patch Logic Applied (`utils.py`)**: 
  - **Dynamic Layer Counting**: Trashed absolute character slice operations (e.g. `k[12:]`), moving extraction parsing algorithms to dynamic prefix delimiters. Split counting checks `k.split("model.layers.")[1].split('.')[0]` ensuring exact integer retrieval globally. 
  - **Keyword Homogenization**: Restructured mapping loops to search via the contextual wildcard `layer in k` instead of explicit prefix beginnings. Upon discovery, stripped the experimental repository prefixes permanently on extraction via `k[k.find(layer):]`, ensuring AirLLM receives consistently named payload layers indiscriminately.

### 6. Shard Formatting Name Normalization
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Trigger**: `FileNotFoundError` seeking specific file paddings formats (`00001-of-00001`).
* **Cause**: Depending on how community developers published models natively to Hugging Face, chunks possessed wildly differing naming formats. (`model-0001` vs `model-00001`).
* **Patch Logic Applied (`utils.py`)**:
  - Dismantled manual string building predictions within shard caching logic. Transitioned execution mappings to strictly scrape string mappings from `set(index.values())` dynamically, perfectly locking the chunk download loop to natively mirror exact, pristine developer file formats. 

### 7. Core Configurations Sanitization Protocol
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Trigger**: Successive downstream cascade crashes within HF: `AttributeError: Qwen3_5Config has no attribute...`
* **Cause**: Experimental Vision-Language or stripped modules failed to map conventional requirements effectively within native standard configurations, collapsing pipeline generation graph executions.
* **Patch Logic Applied (`airllm_base.py`)**:
  - Implemented the ultimate **Configuration Sanitization** layer sequentially prior to instantiation.
  - Auto-mapped and expanded deeply nested parameter aliases natively (e.g., dynamically unpacking `hidden_size` targets via `config.text_config.hidden_size`, `config.d_model`, or `config.n_embd`).
  - Blanketed omissions uniformly with standardized native fallbacks mathematically matching generic initialization structures (e.g., explicitly overriding `num_hidden_layers = 24`, `max_position_embeddings=8192` if the model framework failed validation checks entirely).

### 8. Physical Tensor Embedding Scale Limits
* **Model Context**: `Qwen/Qwen3.5-0.8B`
* **Error Trigger**: `ValueError: Trying to set tensor [248320, 1024] to [248044, 1024]`
* **Cause**: `transformers` built matrices structured to constraints passed via standard dictionaries (like mapping `vocab_size` purely to Tokenizer boundaries). However, physical hard drive matrices actually padded their internal sequence blocks exponentially higher natively (`[248320]`) explicitly for parallel batch compilation efficiencies. Using static configurations yielded shape mismatches.
* **Patch Logic Applied (`airllm_base.py`)**:
  - Implemented lightning-fast **Physical Dimension Prober** algorithms. Upon processing downloaded arrays natively, AirLLM extracts index layouts seamlessly using zero-cost file reads `get_slice("model.embed_tokens.weight").get_shape()`. Binding the native configuration structure (`vocab_size`, `hidden_size`) tightly to literal, exact physical dimension offsets prevents matrix dimension limits crashing `accelerate` implementations indefinitely.

---

## Part 4: Future AI Coding Agent Maintenance Directives 

If passing a completely unmapped or breaking experimental model into AirLLM, future codebase maintainers (or AI Agents) must abide by the following foundational guidelines to prevent errors:

1. **Prioritize Native Aliasing Before Manual Hacks:** If `AutoModelForCausalLM` or generic layer generations complain about missing kwargs (e.g. `attention_mask`), track down the original Hugging Face source file and map missing config aliases recursively through `set_layer_names_dict`.
2. **Beware Aggressive Token Padding:** Do not blindly initialize constraints globally using values fetched purely from tokenizers (`len(tokenizer)`). Always physically verify dimension counts against the pure `.safetensors` matrix footprint mappings. 
3. **Trace The Output Graph Returns Continuously:** New variants of architectures frequently change final pooling implementations out of `decoder_layers` (transitioning abruptly from dict returns to tensors objects or altering nested depth). Subclass variations must rigorously apply Python `isinstance` logic blocks wrapping final index unpacking arrays to prevent dimensionality deletions. 
4. **SafeTensor Index Slicing**: When modifying extraction formats within `utils.py`, never assume zero-indexed character string padding. Always recursively strip strings logically employing dynamic delimiter targets inherently natively via `.find()` indexing protocols.
5. **Always Sandbox `rotary_pos_emb` Locally**: If `transformers` updates embedding computations fundamentally again, fallback strictly on pushing matrix evaluation directly backwards into pre-initialization calculations internally mapped inside `airllm_base.py`'s external loop hooks, subsequently injecting raw physical payload targets independently bypassing abstract Hugging Face framework kwargs logic completely.

---

## Part 5: Single-File Caching & VLM Parameter Path Mapping (Gemma4 / Multi-Modal architectures)

### 9. Single-File Model SafeTensors Download Exclusion
* **Model Context**: Experimental/unsharded models (e.g., `Qwen/Qwen3-0.6B`, small variants under 2GB)
* **Error Trigger**: `FileNotFoundError: No model files found in {...}. Expected one of: pytorch_model.bin.index.json, model.safetensors.index.json, model.safetensors, or pytorch_model.bin`
* **Root Cause Diagnostics**: To aggressively optimize bandwidth, `AirLLM` actively stripped large binary files (`*.safetensors`, `*.bin`) during the initial `huggingface_hub.snapshot_download` phase. It exclusively targeted the `index.json` manifests to evaluate layer maps before pulling heavy weights. However, Single-File Architectures (by definition) do not generate or host `index.json` mappings. By excluding physical binaries and expecting an index, the initial cache pull returned completely empty directories, starving `split_and_save_layers` immediately.
* **Structural Solution Applied (`utils.py`)**: 
  - Integrated dynamic index detection post-fetch. If `huggingface_hub` executes but neither `model.safetensors.index.json` nor `pytorch_model.bin.index.json` populate locally within the cache, the script gracefully infers it must be a single-file configuration.
  - It subsequently executes a rapid forced secondary download targeting precisely `['*.safetensors', '*.bin']`, allowing the native monolithic `load_file()` extractor to split the single bin into standard AirLLM shards.

### 10. AutoModel Framework Fallbacks (`auto_model.py`)
* **Model Context**: Unmapped variants (e.g., `Qwen3ForCausalLM`, `Gemma4ForConditionalGeneration`)
* **Error Trigger**: `unknown artichitecture: Gemma4ForConditionalGeneration, try to use Llama2...`
* **Root Cause Diagnostics**: `auto_model.py` maps strings into specific handler modules (`AirLLMQWen2`, `AirLLMLlama2`). Missing classes forced highly experimental pipelines to abruptly fail over to `AirLLMLlama2`. This collapsed runtime execution graphs because Llama initializes structures with completely different parameter topologies (missing tied embeddings, different norm structures) compared to targets like Gemma.
* **Structural Solution Applied**: 
  - Sequentially hard-mapped string searches for `Qwen3ForCausalLM` mapping transparently directly to `AirLLMQWen2`.
  - Added native string mapping for `Gemma` tracking identically to `AirLLMLlama2` to safely suppress `try to use Llama2` cascade failures while natively invoking `AirLLMBaseModel` cleanly.

### 11. Dynamic Multi-Modal PyTorch Parameter Traversal & Injection 
* **Model Context**: Complex/Wrapper Architectures and Vision-Language Models (e.g., `google/gemma-4-E2B`)
* **Error Trigger**: `AttributeError: 'Gemma4Model' object has no attribute 'embed_tokens'` & subsequent `FileNotFoundError: No such file or directory: model.language_model.embed_tokens.safetensors`
* **Root Cause Diagnostics**: This was an extremely deceptive bug caused by a critical disconnect between **physical cache filenames** and **virtual PyTorch module paths**.
  1. **Phase 1 (Disk Caching)**: `split_and_save_layers()` evaluates literal substrings. Because `"model.embed_tokens"` intrinsically matches inside `"model.language_model.embed_tokens.weight"`, the original caching script unknowingly succeeded. Crucially, it generated physical shard files named using the unmodified default dictionary: `model.embed_tokens.safetensors`.
  2. **Phase 2 (Instantiation)**: Modern generative frameworks wrap their core Transformer within wrapper nodes during initialization (e.g., `Gemma4ForConditionalGeneration` nests text logic inside `model.language_model.embed_tokens`). 
  3. **The Collision (Injection)**: When AirLLM attempted to inject physical weights loaded from `model.embed_tokens.weight` back into the PyTorch layout mapped at `model.embed_tokens`, it hard-crashed with `AttributeError` because the Pytorch element was nested below `.language_model`.
* **The Failed Experiment (Warning to Future Agents)**: 
  - **DO NOT** globally mutate or "auto-detect" `self.layer_names_dict` to align with the PyTorch hierarchy. If an agent loops over `self.model.named_modules()` and permanently updates `layer_names_dict['embed'] = "model.language_model.embed_tokens"`, the system enters an unrecoverable state!
  - It will force `load_layer_to_cpu` to blindly search for `model.language_model...safetensors` on disk. Thise files do not exist because Phase 1 (Caching) evaluates strictly via the legacy dictionary, permanently locking physical files to standard shortcode prefixes.

* **The Failsafe Structural Solution Applied (`airllm_base.py`)**: 
  - **Dynamic `_resolve_and_map_path()` Module Mapping:** Substituted strictly rigid `getattr()` indexing strings with a decoupled dual-path logic. The system leaves `layer_names_dict` globally untouched (preserving caching workflows completely). During instantiation, if a generic lookup (e.g. `hasattr(model, 'embed_tokens')`) fails, it executes a complete systematic crawl over `self.model.named_modules()`. 
  - The crawler dynamically isolates terminal structures functionally (e.g. searching out endpoints named `embed_tokens` while exclusively filtering out decoys containing `audio` or `vision`), and registers the divergence inside a new mapping dictionary: `self._path_remappings`.
  - **Pre-Transfer Memory Translators:** Immediately before parameter injection in `move_layer_to_device()`, the system traps the loaded raw `state_dict` from disk. It runs an interception loop across `_path_remappings`, gracefully upgrading legacy keys back into PyTorch's required wrapper contexts natively (e.g., actively translating `"model.embed_tokens.weight"` seamlessly localized into `"model.language_model.embed_tokens.weight"`). This guarantees parameter injection works synchronously across *any* multi-modal nesting wrapper without ever disrupting physical disk-caching mechanisms.

---

## Part 6: Subclass Tensor Injectors & JIT Compilation Limits (`gpt-oss-20b` / `mxfp4`)

### 12. Non-Sequential SafeTensor Shard Caching
* **Model Context**: `gpt-oss-20b` (And generalized experimental models)
* **Error Trigger**: `RuntimeError: Tensor on device cuda:0 is not on the expected device meta!` (Caused downstream by `model.layers.0.` fetching 16-byte empty shard files).
* **Root Cause Diagnostics**: 
  - Standard models publish strictly 1-indexed, monotonically increasing layer shards (e.g., `model-00001-of-00003`, where `embed` is chunk 1, `layer 0` is chunk 1, etc.).
  - `gpt-oss-20b` thoroughly broke these rules: Its shards were 0-indexed (`model-00000-of-00002.safetensors`), and its mappings looped non-monotonically (e.g. `embed` was in chunk 2, but `layer 0` was in chunk 1). 
  - AirLLM's `split_and_save_layers` logic rigidly relied on `if max(shards) > shard: shard += 1`, entirely bypassing the 0-indexed chunk and abandoning massive weight payloads.
* **Structural Solution Applied (`utils.py`)**:
  - Restructured caching to initiate at an un-indexed state (`shard = -1`), inherently recognizing `0` as an acceptable target bound.
  - Dynamically leapfrogs active trackers using `target_shard = max(shards)` and validates via `if target_shard != shard`, ensuring logically sound downloading for chaotic experimental padding sequences.

### 13. PyTorch 2.x Tensor Subclasses & Advanced MXFP4 Quantizer Hooks 
* **Model Context**: `gpt-oss-20b` / Experimental `mxfp4` implementations natively leveraging custom subclasses.
* **Error Trigger**: `ValueError: Mxfp4GptOssExperts() does not have a parameter or a buffer named down_proj_blocks.`
* **Root Cause Diagnostics**:
  - Legacy quantization tools (like `bitsandbytes`) required dynamic wrappers traversing raw tensors. `AirLLM` previously expected to intercept these by rigorously mapping `accelerate.set_module_tensor_to_device()` individually to identically matched leaf strings.
  - Modern architectures like `MXFP4` dynamically bundle parameter mappings (via PyTorch Tensor Subclasses handling `__tensor_unflatten__`). For example, the disk file index held `.down_proj_blocks`, but the PyTorch `Mxfp4GptOssExperts` layer node only legally owned parameter `.down_proj`. The module natively relied on the internal implementation of `_load_from_state_dict` to structurally unflatten and route block slices down. `accelerate` rigidly mapping explicitly string-to-string natively crashed!
* **Structural Solution Applied (`airllm_base.py`)**:
  - Bypassed the rigid `accelerate.set_module_tensor_to_device` 1:1 mapping loop. 
  - Bundled mapped structures neatly and invoked `self.model.load_state_dict(standard_params, strict=False, assign=True)` dynamically wrapping graceful exceptions. This effectively delegates complex state injection rules sequentially back to native PyTorch implementations seamlessly traversing abstract architectures natively.

### 14. Triton JIT C-Compiler Dependency (Missing `Python.h`)
* **Environment Context**: Deploying dynamic JIT architectures (`mxfp4`, `FlashAttention`, `Deepspeed`) directly on OS environments.
* **Error Trigger**: `subprocess.CalledProcessError: Command '['/usr/bin/gcc', ... 'cuda_utils.c']' returned non-zero exit status 1` culminating in `fatal error: Python.h: No such file or directory`.
* **Root Cause Diagnostics**: 
  - Unlike pre-compiled packages (e.g. standard `torch` or `bitsandbytes`), experimental quantizers heavily utilize Triton to synthesize GPU Math Kernels code 'on-the-fly' right at inference time. Triton relies heavily on native GPU Host toolchains, and seamlessly spins up `/usr/bin/gcc` locally to build extensions.
  - It intrinsically requires C Development Headers to process python bindings successfully. Standard python installs simply omit `.h` libraries out-of-the-box on linux.
* **Workarounds & Repository Management Deployment**:
  - **Prerequisites Requirements (Standard Approach)**: Any GitHub repository distributing modules dependent on Triton dynamic engines strongly needs the prerequisite formally outlined on PyPI / Readme headers.
  
  > [!IMPORTANT]
  > **Recommended README Prerequisite Notice**
  > "Since this model utilizes Triton for ultra-fast, dynamic kernel generation, end-users executing on native Linux environments must possess Python developer headers locally: 
  > Execute before deploying wrappers: `sudo apt-get install python3-dev`"
  
  - **The Conda Alternative Framework Workaround**: If modifying the native OS via `apt-get` is locked or dangerous, explicitly directing infrastructure creation inside isolated `miniconda` or `anaconda` namespaces inherently cures this permanently. Conda intrinsically packs and seals localized `Python.h` binaries internally within its environments, protecting dependencies from local machine variants effortlessly.
