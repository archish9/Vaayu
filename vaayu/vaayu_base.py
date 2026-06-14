import gc
import os
import subprocess
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
from .telemetry import VaayuTelemetry, LayerTiming
from .persist.safetensor_model_persister import get_and_clear_pinned_timings as _get_layer_disk_timings

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


class FP8DynamicCache(DynamicCache):
    """
    DynamicCache variant that stores K/V tensors in torch.float8_e4m3fn.

    Per-head per-token scales (shape [B, H, S, 1]) are stored alongside
    the FP8 data so decode steps are O(1): only new tokens are quantized.
    Full cache is dequantized to BF16/FP16 before returning to attention.

    Note: this class manages its own key_cache / value_cache lists directly
    rather than delegating to the parent's layer-based storage, so that FP8
    tensors (and their companion scale tensors) are stored contiguously and
    can be concatenated cheaply across decode steps.
    """

    def __init__(self):
        super().__init__()
        # Override parent's layer-based storage with flat lists of FP8 tensors.
        self.key_cache: list = []
        self.value_cache: list = []
        self._key_scales: list = []
        self._value_scales: list = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # cache_kwargs intentionally unused
        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0

        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_fp8], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_fp8], dim=-2)
            self._key_scales[layer_idx] = torch.cat([self._key_scales[layer_idx], k_scale], dim=-2)
            self._value_scales[layer_idx] = torch.cat([self._value_scales[layer_idx], v_scale], dim=-2)
        else:
            self.key_cache.append(k_fp8)
            self.value_cache.append(v_fp8)
            self._key_scales.append(k_scale)
            self._value_scales.append(v_scale)

        k_out = self.key_cache[layer_idx].to(key_states.dtype) * self._key_scales[layer_idx]
        v_out = self.value_cache[layer_idx].to(value_states.dtype) * self._value_scales[layer_idx]
        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]




def _layer_path_exists(model: nn.Module, path: str) -> bool:
    if not path:
        return False
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
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


_pcie_topology_checked: set = set()

def _check_pcie_topology(gpu_index: int = 0, model_size_gb: float = 132.0):
    if gpu_index in _pcie_topology_checked:
        return
    _pcie_topology_checked.add(gpu_index)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"-i={gpu_index}",
             "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return

        parts = result.stdout.strip().splitlines()[0].split(", ")
        width = int(parts[0])
        gen = int(parts[1])

        bandwidth_gbps = {
            (3, 4): 3.9, (3, 8): 7.9, (3, 16): 15.8,
            (4, 4): 6.4, (4, 8): 12.8, (4, 16): 25.6,
            (5, 4): 12.6, (5, 8): 25.2, (5, 16): 50.4,
        }.get((gen, width), 0)

        if width < 16:
            bw_str = f"{bandwidth_gbps:.1f} GB/s" if bandwidth_gbps > 0 else "unknown bandwidth"
            penalty = f"~{model_size_gb/bandwidth_gbps:.1f}s per token" if bandwidth_gbps > 0 else "unknown"
            print(f"⚠️  [Vaayu-Warning] GPU running at PCIe Gen{gen} x{width} ({bw_str})")
            print(f"   Inference may be bandwidth-limited to {penalty} for {model_size_gb:.1f}GB models.")
            print(f"   Moving GPU to a primary x16 slot maximizes bandwidth.")
        else:
            print(f"✅ [Vaayu] GPU PCIe link: Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")

    except Exception:
        pass


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
                 use_torch_compile=False, kv_cache_mode="auto", max_pinned_layers=0,
                 layer_names_dict=None, tokenizer_class=None, prefill_chunk_size=512,
                 drafter_model_path=None, draft_tokens=5, use_prompt_lookup=False):
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
        self.telemetry = VaayuTelemetry()
        self._token_count = 0

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

        if device.startswith("cuda"):
            gpu_index = int(device.split(":")[-1]) if ":" in device else 0
            try:
                model_size_bytes = sum(
                    os.path.getsize(p) for p in Path(self.checkpoint_path).glob("*.safetensors")
                )
                model_size_gb = model_size_bytes / (1024 ** 3)
                if model_size_gb == 0:
                    model_size_gb = 132.0
            except Exception:
                model_size_gb = 132.0
            _check_pcie_topology(gpu_index, model_size_gb)

        self.use_torch_compile = use_torch_compile
        self.kv_cache_mode = kv_cache_mode
        self.max_pinned_layers = max_pinned_layers
        self.prefill_chunk_size = prefill_chunk_size
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

        if device.startswith("cuda"):
            from .persist.safetensor_model_persister import _should_use_direct_io, init_model_io
            self._use_direct_io = _should_use_direct_io(str(self.checkpoint_path))
            init_model_io(str(self.checkpoint_path), use_direct_io=self._use_direct_io)
        else:
            self._use_direct_io = False

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
        self.loader_strategy.setup_model(
            self.model, self.config, self.running_device, self.running_dtype,
            pinned_layer_names=self.pinned_layer_names,
        )
        
        # Refresh layers list to pick up newly replaced modules (e.g. ExLlamaV3Linear for lm_head)
        self.set_layers_from_layer_names()

        self.draft_tokens = draft_tokens if draft_tokens is not None else 5

        if device.startswith("cuda"):
            self._init_pinned_layers()

        # RAM layer cache — measured AFTER _init_pinned_layers so pinned-layer RAM
        # consumption is already reflected in available memory, preventing over-budget.
        if device.startswith("cuda"):
            from .persist.ram_cache import LayerRAMCache, auto_detect_cache_budget
            _cache_budget = auto_detect_cache_budget()
            self.ram_cache = LayerRAMCache(_cache_budget)
            print(f"[Vaayu-Cache] RAM layer cache: {_cache_budget / 1e9:.1f} GB budget")
            self._prepopulate_ram_cache()
            self.loader_strategy._cache = self.ram_cache
        else:
            self.ram_cache = None

        # MoE Expert-Aware Streaming: build offset index, expert cache, patch forwards
        self._expert_history: dict[int, list[int]] = {}
        self._expert_cache = None
        self._expert_offset_index = None
        self._moe_blocks: list = []
        if self.running_device.startswith("cuda") and self.is_moe:
            from .persist.expert_offset_index import ExpertOffsetIndex
            from .persist.expert_cache import ExpertRAMCache
            from .moe_patch import patch_moe_blocks
            import psutil as _psutil

            self._expert_offset_index = ExpertOffsetIndex(str(self.checkpoint_path))
            self._expert_offset_index.build_or_load()

            # Budget: available RAM minus 8 GB reserve
            _mem = _psutil.virtual_memory()
            _expert_budget = max(0, _mem.available - (8 * 1024 ** 3))
            self._expert_cache = ExpertRAMCache(_expert_budget)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget")

            self._moe_blocks = patch_moe_blocks(
                model=self.model,
                expert_cache=self._expert_cache,
                offset_index=self._expert_offset_index,
                checkpoint_path=str(self.checkpoint_path),
                top_k=self.moe_top_k,
                device=self.running_device,
                dtype=self.running_dtype,
                vaayu_ref=self,
            )
            print(f"[Vaayu-MoE] {len(self._moe_blocks)} MoE blocks patched for sparse streaming")
        elif self.is_moe:
            print("[Vaayu-MoE] MoE model on non-CUDA device — expert-aware streaming disabled, "
                  "falling back to full-layer loading")

        self.speculator = None
        if drafter_model_path or use_prompt_lookup:
            from .speculative import build_speculator
            self.speculator = build_speculator(
                drafter_model_path=drafter_model_path,
                base_model=self,
                device=device,
                use_prompt_lookup=use_prompt_lookup,
            )

        if device.startswith("cuda"):
            self._warmup_nvme()

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
            quant_method = None
            if isinstance(quantization_config, dict):
                quant_method = quantization_config.get("quant_method")
            else:
                quant_method = getattr(quantization_config, "quant_method", None)
                
            if quant_method == "exl3":
                print("[Vaayu] Skipping AutoHfQuantizer for exl3 (managed by LoaderStrategy)")
            else:
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

        Reserves space for KV cache (sized by kv_cache_mode) + 1.5 GB overhead
        (CUDA context + streaming double-buffer).

        KV cache formulas:
          BF16: n_layers × 2 × n_kv_heads × head_dim × max_seq_len × 2 bytes
          FP8:  n_layers × 2 × n_kv_heads × head_dim × max_seq_len × 1 byte
                (per-head scales are negligible vs 1.5 GB overhead and absorbed there)
        """
        if not torch.cuda.is_available():
            return 0
        if not str(self.device).startswith("cuda"):
            return 0
        total = torch.cuda.get_device_properties(self.device).total_memory

        n_layers = getattr(self.config, 'num_hidden_layers', 80)
        n_kv_heads = getattr(self.config, 'num_key_value_heads', 8)
        _head_dim_attr = getattr(self.config, 'head_dim', None)
        if isinstance(_head_dim_attr, int):
            head_dim = _head_dim_attr
        else:
            head_dim = self.config.hidden_size // self.config.num_attention_heads

        mode = getattr(self, 'kv_cache_mode', 'fp16')
        if mode == 'fp8':
            kv_bytes = n_layers * 2 * n_kv_heads * head_dim * self.max_seq_len
        else:
            kv_bytes = n_layers * 2 * n_kv_heads * head_dim * self.max_seq_len * 2

        # Reserve one transformer layer's worth of VRAM as a streaming buffer.
        # Without this slack, auto-fill pinning leaves <200 MB free, causing
        # H2D transfers to serialize instead of pipeline during short compute
        # windows (e.g., the final small prefill chunk with only a few tokens).
        layer0_path = Path(self.checkpoint_path) / f"{self.layer_names_dict['layer_prefix']}.0.safetensors"
        try:
            layer_slack = layer0_path.stat().st_size
        except Exception:
            layer_slack = int(0.6 * 1024 ** 3)  # 600 MB conservative fallback

        overhead = int(1.5 * 1024 ** 3) + layer_slack
        return max(0, total - kv_bytes - overhead)

    def _create_kv_cache(self):
        """Return a KV cache instance sized and typed for the configured kv_cache_mode."""
        mode = self.kv_cache_mode

        if mode == "auto":
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "fp8" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"

        if mode == "fp16":
            return DynamicCache()

        if not hasattr(torch, "float8_e4m3fn"):
            print("[Vaayu] FP8 KV cache unavailable (PyTorch < 2.1). Falling back to FP16.")
            return DynamicCache()

        print("[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)")
        return FP8DynamicCache()

    def _init_pinned_layers(self) -> None:
        """
        Pre-load layers into VRAM using two-phase asymmetric pinning.

        Phase 1 pins shared layers (embed, norm, lm_head) with continue on budget miss.
        Phase 2 pins transformer layers 0…N with break on budget miss; only runs when
        max_pinned_layers != 0 (0 = shared-only, -1 = fill budget, N = first N layers).
        """
        budget = self._get_vram_pinning_budget()
        if budget <= 0:
            return

        layer_prefix = self.layer_names_dict["layer_prefix"]

        # Phase 1: shared layers — continue on budget miss (each has independent size)
        used = 0
        for layer_name in [self.layer_names_dict["embed"],
                           self.layer_names_dict["norm"],
                           self.layer_names_dict["lm_head"]]:
            try:
                tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                size = sum(t.nbytes for t in tensors.values())
                if used + size > budget:
                    print(f"[Vaayu-Pin] skip {layer_name}: budget exhausted "
                          f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                    continue
                self.loader_strategy.inject_to_gpu(
                    self.model, layer_name, tensors,
                    self.running_device, self.running_dtype,
                    self.hf_quantizer, getattr(self, "_path_remappings", {}),
                )
                self.pinned_layer_names.add(layer_name)
                used += size
                print(f"[Vaayu-Pin] pinned {layer_name}: "
                      f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
            except Exception as e:
                print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        # Phase 1.5: Pin MoE router gates permanently in VRAM (MoE models only).
        # Gates are tiny (< 4 MB per layer) but must be in VRAM before the patched
        # MoE forward() can run the router. Detection is header-driven: any tensor
        # key matching r'\.(gate|router)\.weight$' but NOT containing 'experts.' is
        # a router weight. The tighter regex avoids matching gate_proj (MLP projection).
        if self.is_moe:
            import json as _json
            import re as _re
            import struct as _struct
            _gate_re = _re.compile(r'\.(gate|router)\.weight$')
            for shard_name in self.layer_names:
                if not shard_name.startswith(layer_prefix + "."):
                    continue
                shard_path = Path(self.checkpoint_path) / (shard_name + ".safetensors")
                if not shard_path.exists():
                    continue
                try:
                    with open(shard_path, "rb") as _f:
                        _hdr_len = _struct.unpack("<Q", _f.read(8))[0]
                        _hdr = _json.loads(_f.read(_hdr_len))
                    gate_keys = {
                        k for k in _hdr
                        if k != "__metadata__"
                        and _gate_re.search(k)
                        and "experts." not in k
                    }
                    if not gate_keys:
                        continue
                    from safetensors import safe_open
                    gate_tensors = {}
                    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                        for k in gate_keys:
                            if k in sf.keys():
                                gate_tensors[k] = sf.get_tensor(k)
                    if not gate_tensors:
                        continue
                    size = sum(t.nbytes for t in gate_tensors.values())
                    if used + size > budget:
                        print(f"[Vaayu-Pin] skip gate {shard_name}: budget exhausted")
                        continue
                    self.loader_strategy.inject_to_gpu(
                        self.model, shard_name, gate_tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    if not hasattr(self, '_moe_gate_pinned'):
                        self._moe_gate_pinned = set()
                    self._moe_gate_pinned.add(shard_name)
                    used += size
                    print(f"[Vaayu-Pin] pinned gate {shard_name}: "
                          f"{size / 1e6:.1f} MB  (cumulative {used / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"[Vaayu-Pin] gate pin failed for {shard_name}: {e}")

        # Phase 2: transformer layers — break on budget miss (homogeneous size)
        if self.max_pinned_layers != 0:
            model_attr = self.model
            for attr_name in layer_prefix.split("."):
                model_attr = getattr(model_attr, attr_name)
            total_layers = len(model_attr)

            for i in range(total_layers):
                if self.max_pinned_layers > 0 and i >= self.max_pinned_layers:
                    break
                layer_name = f"{layer_prefix}.{i}"
                try:
                    tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                    size = sum(t.nbytes for t in tensors.values())
                    if used + size > budget:
                        print(f"[Vaayu-Pin] budget exhausted at {layer_name} "
                              f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                        break
                    self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    self.pinned_layer_names.add(layer_name)
                    used += size
                    print(f"[Vaayu-Pin] pinned {layer_name}: "
                          f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        pinned_xformer = sum(1 for n in self.pinned_layer_names
                             if n.startswith(layer_prefix + "."))
        pinned_shared = len(self.pinned_layer_names) - pinned_xformer
        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} layers pinned "
            f"({pinned_xformer} transformer + {pinned_shared} shared), "
            f"{used / 1e9:.2f} GB reserved"
        )

    def _prepopulate_ram_cache(self) -> None:
        """Load cacheable layers from NVMe into RAM before the first inference token.

        Runs synchronously in __init__ (no inference, no DDR4 contention). Temporarily
        disconnects the async cache path on loader_strategy so load_from_disk returns
        plain tensors; put_sync() clones them inline and stores in the cache. Reconnects
        after all layers are loaded. Token 1 then gets pure cache hits with no clone
        overhead and no pinned-buffer race.
        """
        import time as _time
        t0 = _time.perf_counter()

        # Transformer layers not already pinned to VRAM
        prefix = self.layer_names_dict["layer_prefix"]
        cacheable = [
            name for name in self.layer_names
            if name not in self.pinned_layer_names and name.startswith(prefix)
        ]

        # Disable async cache path so load_from_disk returns raw tensors without
        # submitting background clones that would race with put_sync below.
        self.loader_strategy._cache = None
        n_cached = 0

        for layer_name in cacheable:
            if self.ram_cache.current_usage >= self.ram_cache.budget_bytes:
                break
            try:
                tensors = self.loader_strategy.load_from_disk(
                    layer_name, str(self.checkpoint_path))
            except Exception as e:
                print(f"[Vaayu-Cache] pre-pop skip {layer_name}: {e}")
                continue
            if self.ram_cache.put_sync(layer_name, tensors):
                n_cached += 1

        elapsed = _time.perf_counter() - t0
        cached_gb = self.ram_cache.current_usage / 1e9
        print(f"[Vaayu-Cache] pre-populated {n_cached} layers "
              f"({cached_gb:.1f} GB) in {elapsed:.1f}s")
        # loader_strategy._cache is reconnected by the caller immediately after return

    def _warmup_nvme(self) -> None:
        import threading

        def _do_warmup():
            warmup_count = min(10, len(self.layer_names))
            warmup_names = self.layer_names[:warmup_count]

            # Use plain file reads — NOT io_uring_read_into.
            # The persistent io_uring ring is not thread-safe for concurrent SQE
            # submission; using it here races with inference and corrupts CQE counts.
            # Plain reads still touch NVMe blocks and prime the controller's internal
            # DRAM cache, which is the only goal.
            buf = torch.empty(1700 * 1024 * 1024, dtype=torch.uint8)

            for name in warmup_names:
                path = str(Path(self.checkpoint_path) / (name + ".safetensors"))
                try:
                    with open(path, 'rb') as f:
                        f.readinto(buf.numpy())
                except Exception as e:
                    print(f"[Vaayu] Warm-up aborted on layer {name}: {e}")
                    break

            del buf
            print(f"[Vaayu] NVMe warm-up complete ({warmup_count} layers pre-read)")

        t = threading.Thread(target=_do_warmup, daemon=True)
        t.start()
        self._warmup_thread = t

    # ── MoE speculative prefetch ──────────────────────────────────────────────

    def _prefetch_experts(self, layer_idx: int, predicted_experts: list) -> None:
        """
        Background task: load predicted expert tensors into ExpertRAMCache.

        Called by the forward() loop after each MoE layer via ThreadPoolExecutor.
        Overlaps with next layer's shared attention compute to hide expert-load stall.
        """
        if self._expert_cache is None or self._expert_offset_index is None:
            return
        from .persist.safetensor_model_persister import load_expert_sparse
        layer_name = f"{self.layer_names_dict['layer_prefix']}.{layer_idx}"
        shard_file = str(Path(self.checkpoint_path) / f"{layer_name}.safetensors")
        if not Path(shard_file).exists():
            return
        for eidx in predicted_experts:
            if self._expert_cache.has(layer_idx, eidx):
                continue
            metas = self._expert_offset_index.get_expert_offsets(layer_name, eidx)
            if metas is None:
                continue
            try:
                raw = load_expert_sparse(shard_file, metas)
                self._expert_cache.put(layer_idx, eidx, raw)
            except Exception as e:
                print(f"[Vaayu-MoE] prefetch failed layer={layer_idx} expert={eidx}: {e}")

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

    def _chunked_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        chunk_size = self.prefill_chunk_size
        total_len  = input_ids.shape[1]
        n_chunks   = (total_len + chunk_size - 1) // chunk_size
        print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

        past_key_values = None
        output = None

        for start in range(0, total_len, chunk_size):
            end        = min(start + chunk_size, total_len)
            chunk_ids  = input_ids[:, start:end]
            chunk_mask = attention_mask[:, :end] if attention_mask is not None else None

            output = self.forward(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,  # must accumulate KV cache across chunks regardless of caller's use_cache
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            past_key_values = output.past_key_values

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

        if not return_dict:
            return tuple(v for v in [
                output.logits,
                output.past_key_values if use_cache else None,
                output.hidden_states,
                output.attentions,
            ] if v is not None)
        if not use_cache:
            output.past_key_values = None
        return output

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
        self.telemetry.begin_token(self._token_count)

        # Clear per-token routing history so stale entries from the previous token
        # don't interfere with speculative prefetch scheduling for the current token.
        if self.is_moe:
            self._expert_history.clear()

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        if (self.prefill_chunk_size > 0
                and input_ids is not None
                and input_ids.shape[1] > self.prefill_chunk_size):
            _past_kv_len = (self.get_past_key_values_cache_seq_len(past_key_values)
                            if past_key_values is not None else 0)
            if _past_kv_len == 0:
                return self._chunked_prefill(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        if self._active_attn_impl in ("flash_attention_2", "sdpa"):
            attention_mask = None
        else:
            attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
            attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
            attention_mask = attention_mask.to(self.running_device)
        _past_len_for_pos = (self.get_past_key_values_cache_seq_len(past_key_values)
                             if past_key_values is not None else 0)
        _cur_len_for_pos  = input_ids.shape[1] if input_ids is not None else 0
        _pos_range        = max(self.max_seq_len, _past_len_for_pos + _cur_len_for_pos)
        position_ids      = torch.arange(_pos_range, dtype=torch.long, device=self.running_device)[None, :]

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
            past_key_values = self._create_kv_cache()
        all_hidden_states = [[] for _ in range(len(self.layers))] if output_hidden_states else None
        all_self_attns = [[] for _ in range(len(self.layers))] if output_attentions else None

        # Capture KV length once before the layer loop. len_p stays constant for the
        # full forward pass — all layers process the same positions. Re-querying
        # get_seq_length() inside the loop would return a growing value as each layer
        # appends its K/V to the cache, causing cos/sin slices to walk off the end of
        # _full_position_embeddings and produce empty tensors (silent position corruption).
        _len_p = past_key_values.get_seq_length() if past_key_values is not None else 0

        _io_wait_s = 0.0    # time blocked on future.result() / load_layer_to_cpu
        _gpu_load_s = 0.0   # time spent on move_layer_to_device (CPU→GPU transfer)
        _compute_s = 0.0    # time spent on actual layer.forward() GPU compute

        # MoE: precompute set of shard names that contain MoE experts so we can
        # filter expert tensors out of inject_to_gpu. sparse_forward handles experts.
        _moe_shard_names: set[str] = (
            {b.layer_name for b in self._moe_blocks} if self._moe_blocks else set()
        )
        if _moe_shard_names:
            import re as _re_fwd
            _expert_tensor_re = _re_fwd.compile(r'\.experts\.\d+\.')
        else:
            _expert_tensor_re = None

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

                _lt_file_bytes = 0
                _lt_cache_hit = False
                _lt_header_parse_us = 0
                _lt_io_submit_us = 0
                _lt_io_wait_us = 0
                _lt_tensor_reconstruct_us = 0
                _lt_gpu_transfer_us = 0
                _lt_compute_us = 0

                if layer_name in self.pinned_layer_names:
                    # ── Pinned: already in VRAM — zero disk/transfer cost ────
                    _lt_cache_hit = True
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
                    _tl = time.perf_counter()
                    _t = time.time()
                    state_dict = future.result()
                    _io_wait_s += time.time() - _t
                    _lt_io_wait_us = int((time.perf_counter() - _tl) * 1e6)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)
                    _lt_fpath = str(Path(self.checkpoint_path) / (layer_name + ".safetensors"))
                    _lt_disk = _get_layer_disk_timings(_lt_fpath)
                    _lt_header_parse_us = _lt_disk.get("header_parse_us", 0)
                    _lt_io_submit_us = _lt_disk.get("io_submit_us", 0)
                    _lt_io_wait_us = _lt_disk.get("io_wait_us", _lt_io_wait_us)
                    _lt_tensor_reconstruct_us = _lt_disk.get("tensor_reconstruct_us", 0)
                    _lt_file_bytes = sum(v.nbytes for v in state_dict.values())

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
                    _tl = time.perf_counter()
                    _t = time.time()
                    _sd = state_dict
                    if _expert_tensor_re is not None and layer_name in _moe_shard_names:
                        _sd = {k: v for k, v in state_dict.items()
                               if not _expert_tensor_re.search(k)}
                    moved_layers = self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, _sd, self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {})
                    )
                    _gpu_load_s += time.time() - _t
                    _lt_gpu_transfer_us = int((time.perf_counter() - _tl) * 1e6)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                else:
                    # ── Non-pinned, no prefetch ──────────────────────────────
                    _tl = time.perf_counter()
                    _t = time.time()
                    state_dict = self.load_layer_to_cpu(layer_name)
                    _io_wait_s += time.time() - _t
                    _lt_io_wait_us = int((time.perf_counter() - _tl) * 1e6)
                    _lt_fpath = str(Path(self.checkpoint_path) / (layer_name + ".safetensors"))
                    _lt_disk = _get_layer_disk_timings(_lt_fpath)
                    _lt_header_parse_us = _lt_disk.get("header_parse_us", 0)
                    _lt_io_submit_us = _lt_disk.get("io_submit_us", 0)
                    _lt_io_wait_us = _lt_disk.get("io_wait_us", _lt_io_wait_us)
                    _lt_tensor_reconstruct_us = _lt_disk.get("tensor_reconstruct_us", 0)
                    _lt_file_bytes = sum(v.nbytes for v in state_dict.values())
                    if self.profiling_mode:
                        t = time.time()
                    _tl = time.perf_counter()
                    _t = time.time()
                    _sd = state_dict
                    if _expert_tensor_re is not None and layer_name in _moe_shard_names:
                        _sd = {k: v for k, v in state_dict.items()
                               if not _expert_tensor_re.search(k)}
                    moved_layers = self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, _sd, self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {})
                    )
                    _gpu_load_s += time.time() - _t
                    _lt_gpu_transfer_us = int((time.perf_counter() - _tl) * 1e6)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                # Run layer

                for j, seq in enumerate(batch):

                    if layer_name == self.layer_names_dict['embed']:
                        _tl = time.perf_counter()
                        batch[j] = layer(seq)
                        _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                    elif layer_name == self.layer_names_dict['norm']:
                        #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                        _tl = time.perf_counter()
                        batch[j] = self.run_norm(layer, seq)
                        _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)

                        if output_hidden_states:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict['lm_head']:
                        _tl = time.perf_counter()
                        batch[j] = self.run_lm_head(layer, seq)
                        _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                    else:

                        if output_hidden_states:
                            all_hidden_states[i].append(seq)

                        len_s = self.get_sequence_len(seq)

                        kwargs = {
                            'use_cache': use_cache,
                            **self.get_past_key_value_args(past_key_values),
                            **self.get_pos_emb_args(_len_p, len_s),
                            **self.get_attention_mask_args(attention_mask, _len_p, len_s),
                            **self.get_position_ids_args(position_ids, _len_p, len_s),
                        }

                        _MAX_OOM_RETRIES = 2
                        layer_out = None
                        for _oom_retry in range(_MAX_OOM_RETRIES + 1):
                            try:
                                if self.use_torch_compile:
                                    if self._compiled_layer_fn is None:
                                        self._compiled_layer_fn = torch.compile(
                                            lambda m, x, **kw: m(x, **kw),
                                            mode="reduce-overhead",
                                            fullgraph=False,
                                        )
                                    _tl = time.perf_counter(); _t = time.time(); layer_out = self._compiled_layer_fn(layer, seq, **kwargs); _compute_s += time.time() - _t; _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                                else:
                                    _tl = time.perf_counter(); _t = time.time(); layer_out = layer(seq, **kwargs); _compute_s += time.time() - _t; _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                                break
                            except torch.cuda.OutOfMemoryError:
                                if _oom_retry == _MAX_OOM_RETRIES:
                                    print(f"[Vaayu] ❌ Fatal OOM on {layer_name}. Recovery failed.")
                                    raise
                                print(f"[Vaayu] ⚠️ OOM on {layer_name} (attempt {_oom_retry + 1}/{_MAX_OOM_RETRIES}). "
                                      f"Clearing cache and retrying...")
                                torch.cuda.empty_cache()
                                gc.collect()
                                if past_key_values is not None:
                                    try:
                                        _oom_seq_len = self.get_past_key_values_cache_seq_len(past_key_values)
                                        if _oom_seq_len > 512:
                                            _oom_target = _oom_seq_len // 2
                                            print(f"[Vaayu] \U0001f52a Truncating KV cache: {_oom_seq_len} → {_oom_target} tokens")
                                            self._trim_kv_cache_front(past_key_values, _oom_target)
                                    except Exception as _oom_e:
                                        print(f"[Vaayu] Cache truncation failed: {_oom_e}")
                        new_seq = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                        if output_attentions and isinstance(layer_out, tuple) and len(layer_out) > 1:
                            all_self_attns[i].append(layer_out[1])

                        batch[j] = new_seq

                # MoE speculative prefetch: after MoE layer writes _expert_history,
                # prefetch predicted experts for the next MoE layer in background.
                if self.is_moe and self._expert_cache is not None:
                    _layer_prefix = self.layer_names_dict["layer_prefix"]
                    if layer_name.startswith(_layer_prefix + "."):
                        try:
                            _current_layer_idx = int(layer_name[len(_layer_prefix) + 1:])
                            if _current_layer_idx in self._expert_history:
                                _next_layer_idx = _current_layer_idx + 1
                                _pred = self._expert_history[_current_layer_idx]
                                if _pred:
                                    executor.submit(self._prefetch_experts, _next_layer_idx, _pred)
                        except (ValueError, IndexError):
                            pass

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Reclaim VRAM — skip for pinned layers (they stay resident)
                if layer_name not in self.pinned_layer_names:
                    self.loader_strategy.free_from_gpu(
                        self.model, layer, moved_layers, self.hf_quantizer
                    )

                self.telemetry.record_layer(LayerTiming(
                    layer_name=layer_name,
                    file_bytes=_lt_file_bytes,
                    header_parse_us=_lt_header_parse_us,
                    io_submit_us=_lt_io_submit_us,
                    io_wait_us=_lt_io_wait_us,
                    tensor_reconstruct_us=_lt_tensor_reconstruct_us,
                    gpu_transfer_us=_lt_gpu_transfer_us,
                    compute_us=_lt_compute_us,
                    cache_hit=_lt_cache_hit,
                ))

                if i % 40 == 39:
                    clean_memory()

        self.telemetry.end_token()
        self._token_count += 1
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
        _cache_stats = self.ram_cache.stats() if getattr(self, 'ram_cache', None) is not None else "disabled"
        _expert_cache_stats = self._expert_cache.stats() if self._expert_cache is not None else "n/a"
        print(f"[VAAYU-IO] token summary: total={_total:.1f}s  io_wait={_io_wait_s:.1f}s  gpu_load={_gpu_load_s:.1f}s  compute={_compute_s:.1f}s  other={_total - _io_wait_s - _gpu_load_s - _compute_s:.1f}s  cache={_cache_stats}  expert_cache={_expert_cache_stats}")

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

    def _trim_kv_cache(self, cache, target_seq_len: int) -> None:
        """Trim KV cache tensors in-place to target_seq_len tokens."""
        if cache is None:
            return
        if isinstance(cache, FP8DynamicCache):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i][:, :, :target_seq_len, :]
                cache.value_cache[i] = cache.value_cache[i][:, :, :target_seq_len, :]
                cache._key_scales[i] = cache._key_scales[i][:, :, :target_seq_len, :]
                cache._value_scales[i] = cache._value_scales[i][:, :, :target_seq_len, :]
        elif hasattr(cache, 'key_cache'):
            for i in range(len(cache.key_cache)):
                if len(cache.key_cache[i].shape) >= 3:
                    cache.key_cache[i] = cache.key_cache[i][:, :, :target_seq_len, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, :target_seq_len, :]

    def _trim_kv_cache_front(self, cache, target_seq_len: int) -> None:
        """Remove the oldest tokens from the KV cache to free VRAM."""
        if cache is None:
            return
        if isinstance(cache, FP8DynamicCache):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i]     = cache.key_cache[i][:, :, -target_seq_len:, :]
                cache.value_cache[i]   = cache.value_cache[i][:, :, -target_seq_len:, :]
                cache._key_scales[i]   = cache._key_scales[i][:, :, -target_seq_len:, :]
                cache._value_scales[i] = cache._value_scales[i][:, :, -target_seq_len:, :]
        elif hasattr(cache, 'key_cache'):
            for i in range(len(cache.key_cache)):
                if cache.key_cache[i] is not None and len(cache.key_cache[i].shape) >= 3:
                    cache.key_cache[i]   = cache.key_cache[i][:, :, -target_seq_len:, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, -target_seq_len:, :]

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        self.telemetry.dump_json("vaayu_telemetry.json")
        return result

    @torch.no_grad()
    def speculative_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        streamer=None,
        temperature: float = 0.0,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Custom generation loop using speculative decoding.

        Replaces model.generate() when self.speculator is set.
        The drafter guesses DRAFT_TOKENS candidates; Vaayu verifies all in one
        forward pass and accepts the longest correct prefix.

        Returns:
            generated: (1, prompt_len + new_tokens) tensor of all token IDs
        """
        assert self.speculator is not None, "speculative_generate called with no speculator"

        # Step 1: Prefill — forward() auto-chunks if prefill_chunk_size > 0
        prefill_out = self.forward(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill_out.past_key_values
        # Seed the first new token from prefill logits
        first_token = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([input_ids, first_token], dim=1)
        tokens_generated = 1

        if streamer is not None:
            streamer.put(first_token.cpu())

        eos_ids = self.config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_set = set(eos_ids) if eos_ids else set()

        if first_token.item() in eos_set or tokens_generated >= max_new_tokens:
            if streamer is not None:
                streamer.end()
            return generated

        while tokens_generated < max_new_tokens:
            # Step 2: Draft K tokens
            draft_ids, draft_state = self.speculator.draft(
                generated, past_key_values, num_tokens=self.draft_tokens
            )
            k = draft_ids.shape[1]

            # Step 3: Verify — one Vaayu pass over [last_token, d_0, ..., d_{k-1}]
            # past_key_values has prompt+accepted tokens; verify adds k+1 more
            past_len_before = past_key_values.get_seq_length()
            verify_input = torch.cat([generated[:, -1:], draft_ids], dim=1)
            verify_out = self.forward(
                input_ids=verify_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            verifier_logits = verify_out.logits   # shape: (1, k+1, vocab_size)
            past_key_values = verify_out.past_key_values

            # Step 4: Accept/reject longest correct prefix
            accepted_ids, num_accepted = self.speculator.verify_and_accept(
                generated, draft_ids, verifier_logits, draft_state
            )

            # Step 5: Trim verifier KV back to past_len + num_accepted
            # (verify pass grew cache by k+1; we only keep num_accepted new positions)
            self._trim_kv_cache(past_key_values, past_len_before + num_accepted)

            # Step 6: Sync drafter cache to match accepted length
            self.speculator.rollback_cache(draft_state, num_accepted)

            # Step 7: Extend generated sequence (cap at max_new_tokens)
            remaining = max_new_tokens - tokens_generated
            if num_accepted > remaining:
                accepted_ids = accepted_ids[:, :remaining]
                num_accepted = remaining
            generated = torch.cat([generated, accepted_ids], dim=1)
            tokens_generated += num_accepted

            if streamer is not None:
                streamer.put(accepted_ids.cpu())

            accept_rate_str = f"{num_accepted / k:.0%}" if k > 0 else "n/a (no match)"
            print(f"[Vaayu-Spec] drafted={k} accepted={num_accepted} "
                  f"total_new={tokens_generated}/{max_new_tokens} "
                  f"accept_rate={accept_rate_str}")

            if accepted_ids[0, -1].item() in eos_set:
                break

        if streamer is not None:
            streamer.end()
        self.telemetry.dump_json("vaayu_telemetry.json")
        return generated
