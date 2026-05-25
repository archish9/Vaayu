import os
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler
from .persist.loader_strategy import build_loader_strategy, LoaderStrategy

# BetterTransformer is deprecated in newer optimum versions
try:
    from optimum.bettertransformer import BetterTransformer
    better_transformer_available = True
except (ImportError, ModuleNotFoundError):
    better_transformer_available = False
    print('>>>> BetterTransformer not available, will use sdpa attention implementation')

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False



try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed')
except ImportError:
    cache_utils_installed = False






def _layer_path_exists(model: nn.Module, path: str) -> bool:
    if not path:
        return False
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


def _auto_detect_layer_names(model: nn.Module) -> dict:
    EMBED = {"embed_tokens", "word_embeddings", "wte", "embed", "token_embeddings"}
    NORM = {"norm", "final_layernorm", "ln_f", "final_norm", "layer_norm", "ln_out"}
    HEAD = {"lm_head", "output_layer", "embed_out", "output", "head"}

    best_path, best_len = "", 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            best_len, best_path = len(mod), name

    if not best_path:
        return {}

    embed_path = norm_path = lm_head_path = None
    for name, mod in model.named_modules():
        if name.startswith(best_path + "."):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if embed_path is None and leaf in EMBED and isinstance(mod, nn.Embedding):
            embed_path = name
        if norm_path is None and leaf in NORM and not isinstance(mod, (nn.Embedding, nn.Linear, nn.ModuleList)):
            norm_path = name
        if lm_head_path is None and leaf in HEAD and isinstance(mod, nn.Linear):
            lm_head_path = name

    return {
        "embed": embed_path,
        "layer_prefix": best_path,
        "norm": norm_path,
        "lm_head": lm_head_path,
    }


class VaayuBaseModel(GenerationMixin):

    # customize layer names here; subclasses or layer_names_dict kwarg override
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
        self.layer_names_dict.update(self._layer_names_override)



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False,
                 layer_names_dict=None, tokenizer_class=None):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq lenght, by default 512
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """


        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = True
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')
            if not device.startswith("cuda"):
                raise ValueError(
                    "Compression (4bit/8bit) currently requires a CUDA GPU. "
                    "For CPU-only mode, use compression=None. "
                    "CPU compression support is planned for a future release."
                )


        self.compression = compression
        self.hf_token = hf_token

        # Save parameters
        self._layer_names_override = layer_names_dict or {}
        self._tokenizer_class = tokenizer_class
        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        self.running_device = device
        self.device = torch.device(self.running_device)
        if dtype == torch.float16 and device.startswith("cuda"):
            try:
                cc = torch.cuda.get_device_capability()
                if cc[0] >= 8:
                    self.running_dtype = torch.bfloat16
                    print(f"✅ GPU supports BF16 (compute capability {cc[0]}.{cc[1]}). Using bfloat16 for better stability.")
                else:
                    self.running_dtype = dtype
                    print(f"ℹ️ GPU compute capability {cc[0]}.{cc[1]} — using float16.")
            except Exception:
                self.running_dtype = dtype
        elif not device.startswith("cuda") and dtype == torch.float16:
            # CPU lacks native float16 ALUs — float32 is 10-50x faster on x86/ARM
            self.running_dtype = torch.float32
            print("ℹ️ CPU mode: using float32 (float16 is very slow on CPU).")
        else:
            self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.use_torch_compile = use_torch_compile
        self._compiled_layer_fn = None
        if use_torch_compile:
            if not hasattr(torch, 'compile'):
                print("⚠️ torch.compile requires PyTorch 2.0+. Disabling.")
                self.use_torch_compile = False
            elif not device.startswith("cuda"):
                print("⚠️ torch.compile is most effective on CUDA. Disabling for CPU.")
                self.use_torch_compile = False
            else:
                print("✅ torch.compile enabled for layer forward passes")

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        self.generation_config = self.get_generation_config()
        #print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # Probe exact physical vocab/hidden dimensions from extracted shard metadata to prevent PyTorch embedding mismatches
        embed_path = Path(self.checkpoint_path) / f"{self.layer_names_dict['embed']}.safetensors"
        if os.path.exists(embed_path):
            import safetensors.torch
            try:
                with safetensors.torch.safe_open(embed_path, framework="pt", device="cpu") as f:
                    real_embed_key = f"{self.layer_names_dict['embed']}.weight"
                    if real_embed_key in f.keys():
                        real_shape = f.get_slice(real_embed_key).get_shape()
                        self.config.vocab_size = real_shape[0]
                        self.config.hidden_size = real_shape[1]
            except Exception:
                pass

        # Native fallback if probe failed
        if not hasattr(self.config, "vocab_size"):
            if hasattr(self.tokenizer, "vocab_size"):
                self.config.vocab_size = self.tokenizer.vocab_size
            else:
                self.config.vocab_size = len(self.tokenizer)

        # Pull everything from nested configurations if this is a VLM wrapper
        if hasattr(self.config, "text_config"):
            for k in dir(self.config.text_config):
                if not k.startswith("_") and not hasattr(self.config, k):
                    setattr(self.config, k, getattr(self.config.text_config, k))

        # Dynamically map custom aliases
        fallback_mappings = {
            "hidden_size": ["d_model", "n_embd", "n_emb"],
            "num_hidden_layers": ["num_layers", "n_layer", "n_layers"],
            "num_attention_heads": ["num_heads", "n_head", "n_heads"],
            "num_key_value_heads": ["num_kv_heads", "n_kv_head", "num_attention_heads"],
            "intermediate_size": ["ffn_dim", "inner_dim"],
            "rms_norm_eps": ["layer_norm_eps", "norm_eps"],
            "max_position_embeddings": ["n_positions", "max_seq_len", "seq_length"],
            "rope_theta": ["rotary_emb_base"],
            "hidden_act": ["activation_function"],
        }

        for target_key, fallbacks in fallback_mappings.items():
            if not hasattr(self.config, target_key):
                for fb in fallbacks:
                    if hasattr(self.config, fb):
                        setattr(self.config, target_key, getattr(self.config, fb))
                        break

        # Token bounds protection
        if not hasattr(self.config, "pad_token_id"):
            self.config.pad_token_id = None
        if getattr(self.config, "pad_token_id", None) is not None and getattr(self.config, "vocab_size", 0) <= self.config.pad_token_id:
            self.config.pad_token_id = None

        if not hasattr(self.config, "eos_token_id"):
            self.config.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        if not hasattr(self.config, "bos_token_id"):
            self.config.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1

        # Hard generic defaults to blindly satisfy transformers instantiation for totally blank models
        if not hasattr(self.config, "num_hidden_layers"): self.config.num_hidden_layers = 24
        if not hasattr(self.config, "num_attention_heads"): self.config.num_attention_heads = 16
        if not hasattr(self.config, "num_key_value_heads"): self.config.num_key_value_heads = getattr(self.config, "num_attention_heads", 16)
        if not hasattr(self.config, "intermediate_size"): self.config.intermediate_size = getattr(self.config, "hidden_size", 4096) * 4
        if not hasattr(self.config, "rms_norm_eps"): self.config.rms_norm_eps = 1e-5
        if not hasattr(self.config, "max_position_embeddings"): self.config.max_position_embeddings = 8192
        if not hasattr(self.config, "hidden_act"): self.config.hidden_act = "silu"
        if not hasattr(self.config, "rope_theta"): self.config.rope_theta = 10000.0

        # Build loader strategy (auto-detects EXL3 vs standard safetensors)
        self.loader_strategy: LoaderStrategy = build_loader_strategy(
            str(self.checkpoint_path), self
        )

        self.init_model()

        # fill in any layer name keys that don't resolve on this model (unknown architectures)
        _detected = _auto_detect_layer_names(self.model)
        for key, val in _detected.items():
            if val and (key not in self.layer_names_dict or not _layer_path_exists(self.model, self.layer_names_dict[key])):
                self.layer_names_dict[key] = val

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.")

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        # MoE detection — must happen before pinning so setup_model has correct context
        self._detect_moe()

        # Pin shared layers (embed, norm, lm_head) permanently in VRAM.
        # Skips load/inject/free in forward() for these layers every token.
        self.pinned_layer_names: set = set()
        if device.startswith("cuda"):
            self._init_pinned_layers()

        # Let strategy do one-time model-level setup (e.g. EXL3: swap nn.Linear)
        self.loader_strategy.setup_model(
            self.model, self.config, self.running_device, self.running_dtype,
            pinned_layer_names=self.pinned_layer_names,
        )

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if self._tokenizer_class is not None:
            tok_kwargs = {"use_fast": False, "trust_remote_code": True}
            if hf_token is not None:
                tok_kwargs["token"] = hf_token
            return self._tokenizer_class.from_pretrained(self.model_local_path, **tok_kwargs)
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return False

    def init_model(self):
        self.model = None
        self._active_attn_impl = None

        attn_candidates = []
        try:
            import flash_attn
            attn_candidates.append("flash_attention_2")
        except ImportError:
            pass
        attn_candidates.extend(["sdpa", "eager"])

        if self.get_use_better_transformer() and better_transformer_available:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                    self.model = BetterTransformer.transform(self.model)
                self._active_attn_impl = "BetterTransformer"
            except Exception:
                del self.model
                clean_memory()
                self.model = None

        if self.model is None:
            for attn_impl in attn_candidates:
                try:
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(
                            self.config, attn_implementation=attn_impl, trust_remote_code=True
                        )
                    self._active_attn_impl = attn_impl
                    break
                except Exception:
                    if self.model is not None:
                        del self.model
                    clean_memory()
                    self.model = None

        if self.model is None:
            raise RuntimeError("Failed to create model with any attention implementation")

        print(f"✅ Attention implementation: {self._active_attn_impl}")
        try:
            for mod in self.model.modules():
                if 'Attention' in type(mod).__name__:
                    print(f"   Attention module type: {type(mod).__name__}")
                    break
        except Exception:
            pass

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)

        self.model.eval()
        self.model.tie_weights()

        self._path_remappings = {}
        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

        if 'rotary_pos_emb' in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

    def diagnose_attention(self):
        print(f"Configured attention: {self._active_attn_impl}")
        if torch.cuda.is_available():
            print(f"Flash SDP available: {torch.backends.cuda.flash_sdp_enabled()}")
            print(f"Memory-efficient SDP available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
            print(f"Math SDP available: {torch.backends.cuda.math_sdp_enabled()}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Compute capability: {torch.cuda.get_device_capability()}")

    def _resolve_and_map_path(self, target_path):
        import torch
        attr = self.model
        parts = target_path.split(".")
        current_path = ""
        resolved_full_path = ""

        # Attempt standard traversal
        success = True
        for part in parts:
            if hasattr(attr, part):
                attr = getattr(attr, part)
                current_path += ("." if current_path else "") + part
            else:
                success = False
                break
                
        if success:
            return attr

        # If it failed, search the model tree for the final node (e.g. embed_tokens, layers, norm, lm_head)
        target_tail = parts[-1]
        for name, mod in self.model.named_modules():
            if name.endswith(target_tail):
                # Ensure we avoid picking an audio/vision component arbitrarily if language is present
                if "audio" in name or "vision" in name:
                    continue
                # For layers, ensure it's a ModuleList
                if target_tail in ("layers", "blocks") and not isinstance(mod, torch.nn.ModuleList):
                    continue
                
                # Found a replacement! Map it so parameter injection succeeds later
                self._path_remappings[target_path] = name
                return mod
                
        # If absolute failure, return original root (will likely crash later, but preserves legacy fallback)
        return self.model

    def set_layers_from_layer_names(self):
        self.layers = []
        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["embed"]))
        
        layer_list = self._resolve_and_map_path(self.layer_names_dict["layer_prefix"])
        if isinstance(layer_list, (list, tuple, torch.nn.ModuleList)):
            self.layers.extend(list(layer_list))
        else:
            self.layers.append(layer_list)

        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["norm"]))
        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["lm_head"]))

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

    # ── MoE detection ────────────────────────────────────────────────────────

    def _detect_moe(self) -> None:
        """Detect Mixture-of-Experts architecture from model config."""
        arch = (getattr(self.config, "architectures", None) or [""])[0]
        self.is_moe = (
            hasattr(self.config, "num_local_experts")
            or hasattr(self.config, "num_experts")
            or "Moe" in arch
            or "MoE" in arch
            or "Mixtral" in arch
        )
        if self.is_moe:
            self.moe_num_experts = getattr(
                self.config, "num_local_experts",
                getattr(self.config, "num_experts", 8)
            )
            self.moe_top_k = getattr(
                self.config, "num_experts_per_tok",
                getattr(self.config, "top_k", 2)
            )
            print(
                f"[Vaayu] MoE detected: {self.moe_num_experts} experts, "
                f"top-{self.moe_top_k} active per token"
            )
            print(
                "[Vaayu] Expert-aware streaming (Phase 3): pending — "
                "currently streams full MoE layer (conservative baseline)"
            )
        else:
            self.moe_num_experts = 0
            self.moe_top_k = 0

    # ── VRAM pinning ─────────────────────────────────────────────────────────

    def _get_vram_pinning_budget(self) -> int:
        """
        VRAM budget available for permanently pinned layers.
        Reserves 4 GB for KV cache + io_uring double-buffers (~2 × largest layer).
        Formula is conservative; tune if KV cache overflows for long contexts.

        KV cache size = n_layers × 2 × n_kv_heads × head_dim × seq_len × dtype_bytes
        For Llama-70B @ 512 tok BF16: 80 × 2 × 8 × 128 × 512 × 2 ≈ 1.3 GB.
        For Qwen3-480B MoE the active KV is much smaller due to sparse routing.
        """
        if not torch.cuda.is_available():
            return 0
        total = torch.cuda.get_device_properties(self.device).total_memory
        reserve = 4 * 1024 ** 3  # 4 GB
        return max(0, total - reserve)

    def _init_pinned_layers(self) -> None:
        """
        Pre-load shared layers (embed, norm, lm_head) into VRAM once.
        forward() will skip load/inject/free for these layers every token,
        saving one full NVMe read per token for ~2–3 GB of shared weights.
        """
        budget = self._get_vram_pinning_budget()
        if budget <= 0:
            return

        candidates = [
            self.layer_names_dict["embed"],
            self.layer_names_dict["norm"],
            self.layer_names_dict["lm_head"],
        ]

        used = 0
        for layer_name in candidates:
            try:
                tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                size = sum(t.nbytes for t in tensors.values())
                if used + size > budget:
                    print(f"[Vaayu-Pin] skip {layer_name}: budget exhausted")
                    continue
                self.loader_strategy.inject_to_gpu(
                    self.model, layer_name, tensors,
                    self.running_device, self.running_dtype,
                    self.hf_quantizer, getattr(self, "_path_remappings", {}),
                )
                self.pinned_layer_names.add(layer_name)
                used += size
                print(
                    f"[Vaayu-Pin] pinned {layer_name}: "
                    f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)"
                )
            except Exception as e:
                # lm_head may fail if weights are tied; forward() handles that via
                # the existing tied-weights fallback in load_layer_to_cpu.
                print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} layers pinned, "
            f"{used / 1e9:.2f} GB reserved"
        )

    # ── Strategy helper ───────────────────────────────────────────────────────

    def _load_from_strategy(self, layer_name: str) -> dict:
        """Thin wrapper so ThreadPoolExecutor can call the strategy load."""
        return self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)

    def load_layer_to_cpu(self, layer_name):

        t = time.time()

        try:
            load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        except Exception as e:
            # For models with tied weights, lm_head.safetensors might not be generated
            # because it's implicitly tied to embed_tokens. Safely fallback to embed_tokens.
            if layer_name == self.layer_names_dict['lm_head'] or layer_name == self.layer_names_dict['lm_head'] + ".":
                print(f"Failed to load {layer_name}, attempting to use tied weights from {self.layer_names_dict['embed']}. (original error: {repr(e)})")
                embed_layer_name = self.layer_names_dict['embed']
                # Sometimes layer names are passed with dots depending on the execution context, so strip it for comparison but fetch the exact embed equivalent
                if layer_name.endswith("."):
                     embed_layer_name += "."
                load_layer_output = load_layer(self.checkpoint_path, embed_layer_name, self.profiling_mode)
                
                # Replace the 'model.embed_tokens' prefix with 'lm_head' in the returned state dict
                if self.profiling_mode:
                    state_dict, compression_time = load_layer_output
                else:
                    state_dict = load_layer_output
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace(self.layer_names_dict['embed'], self.layer_names_dict['lm_head'])
                    new_state_dict[new_k] = v
                
                if self.profiling_mode:
                    load_layer_output = (new_state_dict, compression_time)
                else:
                    load_layer_output = new_state_dict
            else:
                raise e

        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)

            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        return state_dict

    def move_layer_to_device(self, state_dict):
        layers = []
        
        # Apply path remappings to state dict keys so parameter injection succeeds 
        # on architectures heavily wrapped in VLMs (e.g., Gemma4)
        remapped_state_dict = {}
        for param_name, param_tensor in state_dict.items():
            new_param_name = param_name
            for old_path, new_path in getattr(self, "_path_remappings", {}).items():
                if param_name.startswith(old_path):
                    new_param_name = param_name.replace(old_path, new_path, 1)
                    break
            remapped_state_dict[new_param_name] = param_tensor

        for param_name, param in remapped_state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if '.weight' in param_name:
                    layer_name = param_name[:param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)
                else:
                    if param_name not in layers:
                        layers.append(param_name)

        standard_params = {}
        for param_name in layers:
            is_quantized = False
            if self.hf_quantizer is not None:
                if hasattr(self.hf_quantizer, "check_quantized_param"):
                    is_quantized = self.hf_quantizer.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

            if not is_quantized:
                standard_params[param_name] = remapped_state_dict[param_name].to(self.running_device, dtype=self.running_dtype, non_blocking=True)
            else:
                torch_dtype = self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(self.model, remapped_state_dict[param_name], param_name, self.running_device, remapped_state_dict)

        if len(standard_params) > 0:
            try:
                self.model.load_state_dict(standard_params, strict=False, assign=True)
            except Exception as e:
                for param_name, tensor in standard_params.items():
                    try:
                        set_module_tensor_to_device(self.model, param_name, self.running_device, value=tensor, dtype=self.running_dtype)
                    except Exception as fallback_e:
                        print(f"Warning: Could not inject {param_name} natively: {fallback_e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # Capture past length BEFORE normalizing the cache object to None,
            # because Vaayu manages its own KV cache layer-by-layer and does not
            # consume the HF DynamicCache directly.
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        # transformers >= 4.36 returns a DynamicCache object instead of a tuple-of-tuples.
        # Use duck-typing first so this works even if cache_utils failed to import.
        if hasattr(past_key_values, 'get_seq_length'):
            return past_key_values.get_seq_length()
        if cache_utils_installed and isinstance(past_key_values, Cache):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_rotary_emb(self):
        """Return the model-level rotary embedding module if the architecture exposes one.

        Transformers >= 4.45 computes RoPE once in the top-level model forward and passes
        position_embeddings=(cos, sin) to every decoder layer.  For models that follow this
        pattern the module lives at model.model.rotary_emb.
        """
        try:
            return self.model.model.rotary_emb
        except AttributeError:
            return None

    def _compute_full_position_embeddings(self, position_ids):
        """Compute RoPE (cos, sin) for all positions directly from model config.

        Avoids using the meta-initialized rotary_emb module whose inv_freq buffer
        contains uninitialized CUDA memory after init_empty_weights, which would
        produce garbage / overflowed values and corrupt the hidden states.
        """
        try:
            head_dim = getattr(self.config, 'head_dim',
                               self.config.hidden_size // self.config.num_attention_heads)
            rope_theta = float(getattr(self.config, 'rope_theta', 10000.0))
        except AttributeError:
            return None

        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim
        ))

        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.no_grad():
            freqs = (inv_freq_expanded.float() @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype=self.running_dtype)
            sin = emb.sin().to(dtype=self.running_dtype)
        return cos, sin

    def get_pos_emb_args(self, len_p, len_s):
        # If position_embeddings were pre-computed for the full sequence (new transformers API),
        # slice out just the tokens being processed in this call.
        pos_emb = getattr(self, '_full_position_embeddings', None)
        if pos_emb is not None:
            cos, sin = pos_emb
            return {'position_embeddings': (cos[:, len_p:len_p + len_s], sin[:, len_p:len_p + len_s])}
        return {}

    def get_past_key_value_args(self, past_key_values):
        return {'past_key_values': past_key_values}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        if full_attention_mask is None:
            return {}
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        _exec_start = time.time()
        print(f"[Vaayu] inference start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_exec_start))}")

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()


        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        if self._active_attn_impl in ("flash_attention_2", "sdpa"):
            attention_mask = None
        else:
            attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
            attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
            attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        # Pre-compute RoPE position embeddings for the full sequence length.
        # Transformers >= 4.45 moved this computation to the top-level model forward;
        # each decoder layer now receives position_embeddings=(cos, sin) rather than
        # recomputing them internally.
        #
        # We gate on whether the architecture exposes a top-level rotary_emb (the
        # signal that it uses this new API), but compute the values from the model
        # config rather than calling the meta-initialized module.  After
        # init_empty_weights(), rotary_emb.inv_freq is an uninitialized CUDA tensor
        # (garbage values) which would produce NaN/overflow and corrupt hidden states.
        if self.get_rotary_emb() is not None:
            try:
                # Natively bypass init_empty_weights uninitialized buffer by creating a local un-mocked instance.
                # Guarantees perfectly matched dimensionality outputs specific to the framework (e.g. concatenated Llama vs chunked GptOss)
                rotary_emb_cls = type(self.get_rotary_emb())
                real_rotary = rotary_emb_cls(config=self.config).to(device=self.running_device, dtype=self.running_dtype)

                # Provide a dummy sequence tensor; many modern RoPE forwards require x purely for querying .device and .dtype
                dummy_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
                dummy_x = torch.empty(1, 1, position_ids.shape[1], dummy_dim, device=self.running_device, dtype=self.running_dtype)

                self._full_position_embeddings = real_rotary(dummy_x, position_ids)
            except Exception as e:
                print(f"Native RoPE instantiation failed ({e}), falling back to manual fallback.")
                self._full_position_embeddings = self._compute_full_position_embeddings(position_ids)
        else:
            self._full_position_embeddings = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        _io_wait_s = 0.0    # time blocked on future.result() / load_layer_to_cpu
        _gpu_load_s = 0.0   # time spent on move_layer_to_device (CPU→GPU transfer)
        _compute_s = 0.0    # time spent on actual layer.forward() GPU compute

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # ── Seed prefetch: skip pinned layers (already in VRAM) ──────────
            future = None
            if self.prefetching:
                for _seed_name in self.layer_names:
                    if _seed_name not in self.pinned_layer_names:
                        future = executor.submit(self._load_from_strategy, _seed_name)
                        break

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers)):

                if layer_name in self.pinned_layer_names:
                    # ── Pinned: already in VRAM — zero disk/transfer cost ────
                    moved_layers = []
                    # If no future is outstanding, kick off the next non-pinned load
                    # so disk read overlaps with this layer's GPU compute.
                    if self.prefetching and future is None:
                        for _n in self.layer_names[i + 1:]:
                            if _n not in self.pinned_layer_names:
                                future = executor.submit(self._load_from_strategy, _n)
                                break

                elif self.prefetching:
                    # ── Non-pinned, prefetching ──────────────────────────────
                    if self.profiling_mode:
                        t = time.time()
                    _t = time.time()
                    state_dict = future.result()
                    _io_wait_s += time.time() - _t
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)

                    # Submit next non-pinned layer's load BEFORE GPU transfer
                    # so disk read and GPU H2D transfer overlap.
                    future = None
                    if self.profiling_mode:
                        t = time.time()
                    for _n in self.layer_names[i + 1:]:
                        if _n not in self.pinned_layer_names:
                            future = executor.submit(self._load_from_strategy, _n)
                            break
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)

                    if self.profiling_mode:
                        t = time.time()
                    _t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    _gpu_load_s += time.time() - _t
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                else:
                    # ── Non-pinned, no prefetch ──────────────────────────────
                    _t = time.time()
                    state_dict = self.load_layer_to_cpu(layer_name)
                    _io_wait_s += time.time() - _t
                    if self.profiling_mode:
                        t = time.time()
                    _t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    _gpu_load_s += time.time() - _t
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                # Run layer

                for j, seq in enumerate(batch):

                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                        batch[j] = self.run_norm(layer, seq)

                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:

                        if output_attentions:
                            all_hidden_states[i].append(new_seq)

                        len_p = past_key_values.get_seq_length() if past_key_values is not None else 0
                        len_s = self.get_sequence_len(seq)

                        kwargs = {
                            'use_cache': use_cache,
                            **self.get_past_key_value_args(past_key_values),
                            **self.get_pos_emb_args(len_p, len_s),
                            **self.get_attention_mask_args(attention_mask, len_p, len_s),
                            **self.get_position_ids_args(position_ids, len_p, len_s),
                        }

                        if self.use_torch_compile:
                            if self._compiled_layer_fn is None:
                                self._compiled_layer_fn = torch.compile(
                                    lambda m, x, **kw: m(x, **kw),
                                    mode="reduce-overhead",
                                    fullgraph=False,
                                )
                            _t = time.time(); layer_out = self._compiled_layer_fn(layer, seq, **kwargs); _compute_s += time.time() - _t
                        else:
                            _t = time.time(); layer_out = layer(seq, **kwargs); _compute_s += time.time() - _t
                        new_seq = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                        if output_attentions and isinstance(layer_out, tuple) and len(layer_out) > 1:
                            all_self_attns[i].append(layer_out[1])

                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Reclaim VRAM — skip for pinned layers (they stay resident)
                if layer_name not in self.pinned_layer_names:
                    self.loader_strategy.free_from_gpu(
                        self.model, layer, moved_layers, self.hf_quantizer
                    )

                if i % 20 == 19:
                    clean_memory()

        logits = torch.cat(batch, 0)

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        _exec_end = time.time()
        _total = _exec_end - _exec_start
        print(f"[Vaayu] inference end:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_exec_end))}")
        print(f"[Vaayu] total execution time: {_total:.3f}s")
        print(f"[VAAYU-IO] token summary: total={_total:.1f}s  io_wait={_io_wait_s:.1f}s  gpu_load={_gpu_load_s:.1f}s  compute={_compute_s:.1f}s  other={_total - _io_wait_s - _gpu_load_s - _compute_s:.1f}s")

        if not return_dict:
            return tuple(v for v in [logits,
                                     past_key_values if use_cache else None,
                                     tuple(all_hidden_states) if all_hidden_states is not None else None,
                                     tuple(all_self_attns) if all_self_attns is not None else None] if v is not None)
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()


            print(f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}")
            print(f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}")

            self.profiler.clear_profiling_time()


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_self_attns is not None else None,
        )