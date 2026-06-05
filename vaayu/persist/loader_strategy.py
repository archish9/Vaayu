"""
LoaderStrategy — pluggable disk-to-GPU pipeline for Vaayu.

Three implementations:
  StandardSafetensorsStrategy  — existing BF16/FP16 safetensors path (no change in behaviour)
  EXL3LoaderStrategy           — ExLlamaV3 4-bit quantized path (requires exllamav3 package)

Auto-detection via build_loader_strategy(checkpoint_path).
"""

from __future__ import annotations

import json
import struct
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

if TYPE_CHECKING:
    from transformers.quantizers import HfQuantizer


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LoaderStrategy(ABC):
    """
    Defines how a single model layer moves from NVMe → CPU pinned RAM → GPU,
    and how it is freed when Vaayu moves to the next layer.
    """

    @abstractmethod
    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        """
        Read raw bytes from SSD into CPU pinned RAM.
        Returns a {param_name: tensor} dict (tensors are CPU, may be pinned).
        """

    @abstractmethod
    def inject_to_gpu(
        self,
        model: nn.Module,
        layer_name: str,
        tensors: dict,
        device: str,
        dtype: torch.dtype,
        hf_quantizer=None,
        path_remappings: dict | None = None,
    ) -> list:
        """
        Move tensors from CPU → GPU and attach them to the model.
        Returns list of injected param names (used later by free_from_gpu).
        """

    @abstractmethod
    def free_from_gpu(
        self,
        model: nn.Module,
        layer_module: nn.Module,
        moved_layers: list,
        hf_quantizer=None,
    ) -> None:
        """
        Reclaim VRAM after the layer's forward pass is done.
        For C++ backends (EXL3) this MUST call explicit destructors — PyTorch GC
        will not free C++ QMatrix objects and the 16 GB GPU will OOM instantly.
        """

    def setup_model(
        self,
        model: nn.Module,
        config,
        device: str,
        dtype: torch.dtype,
        pinned_layer_names: set | None = None,
    ) -> None:
        """
        One-time model-level setup called once after init_empty_weights().
        Default: no-op. EXL3 overrides this to swap nn.Linear → ExLlamaV3Linear.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Standard safetensors strategy  (wraps existing Vaayu logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class StandardSafetensorsStrategy(LoaderStrategy):
    """
    Proxies load_from_disk → utils.load_layer (existing pinned/io_uring path).
    Proxies inject_to_gpu → vaayu_base.move_layer_to_device (existing logic).
    Zero behavioural change vs. the code before the strategy refactor.
    """

    def __init__(self, vaayu_base_ref):
        # Weak reference to avoid circular GC issues; we know base outlives strategy.
        self._base = vaayu_base_ref

    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        from ..utils import load_layer
        return load_layer(checkpoint_path, layer_name)

    def inject_to_gpu(
        self,
        model: nn.Module,
        layer_name: str,
        tensors: dict,
        device: str,
        dtype: torch.dtype,
        hf_quantizer=None,
        path_remappings: dict | None = None,
    ) -> list:
        # Delegate to vaayu_base.move_layer_to_device which contains all the
        # VLM path-remapping, HF quantizer, and load_state_dict fallback logic.
        return self._base.move_layer_to_device(tensors)

    def free_from_gpu(
        self,
        model: nn.Module,
        layer_module: nn.Module,
        moved_layers: list,
        hf_quantizer=None,
    ) -> None:
        if hf_quantizer is not None:
            for param_name in moved_layers:
                set_module_tensor_to_device(model, param_name, "meta")
        else:
            layer_module.to("meta")


# ─────────────────────────────────────────────────────────────────────────────
# EXL3 loader strategy
# ─────────────────────────────────────────────────────────────────────────────

# EXL3 quantized tensors use these specific key names inside safetensors files.
_EXL3_QTENSOR_KEYS = {"trellis", "mcg", "suh", "svh", "su", "sv", "mul1"}

# Required dtypes — ExLlamaV3 C++ kernels are strict about these.
_EXL3_DTYPE_MAP = {
    "trellis": torch.int16,
    "mcg":     torch.int32,
    "suh":     torch.float16,
    "svh":     torch.float16,
    "su":      torch.int16,
    "sv":      torch.int16,
    "mul1":    torch.int32,
}


class ExLlamaV3Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear that defers computation to ExLlamaV3's
    CUDA kernels operating directly on 4-bit compressed weights.

    q_matrix is a C++ QMatrix object set by EXL3LoaderStrategy.inject_to_gpu().
    It lives in C++ memory; PyTorch GC does NOT manage it.
    EXL3LoaderStrategy.free_from_gpu() must call its destructor explicitly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bc = None  # set by inject_to_gpu

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bc is None:
            raise RuntimeError(
                f"ExLlamaV3Linear (out={self.out_features}): bc not set. "
                "inject_to_gpu() must be called before forward()."
            )
        try:
            x_in = x if x.dtype == torch.float16 else x.to(torch.float16)
            x_in = x_in.contiguous()

            shape = x_in.shape
            x_flat = x_in.view(-1, self.in_features)

            # Try batched call first (single CUDA kernel for all rows).
            # Falls back to row-by-row only if the C++ backend rejects the batch.
            try:
                out = self.bc.run_alloc(x_flat, self.out_features, False)
                out = out.view(*shape[:-1], self.out_features)
            except Exception:
                # Fallback: row-by-row (slow but safe for all matrix configs)
                rows = x_flat.shape[0]
                out_flat = torch.empty((rows, self.out_features), dtype=torch.float16, device=x.device)
                for i in range(rows):
                    out_flat[i:i+1] = self.bc.run_alloc(x_flat[i:i+1].contiguous(), self.out_features, False)
                out = out_flat.view(*shape[:-1], self.out_features)

            if out.dtype != x.dtype:
                out = out.to(x.dtype)

            return out
        except ImportError:
            raise RuntimeError("exllamav3 not installed. Cannot run ExLlamaV3Linear.forward().")


class EXL3LoaderStrategy(LoaderStrategy):
    """
    Implements ExLlamaV3 (EXL3) 4-bit quantized inference.

    Key responsibilities:
      1. setup_model()    — swap nn.Linear → ExLlamaV3Linear once at init time.
      2. load_from_disk() — read EXL3 q_* tensors via existing io_uring path.
      3. inject_to_gpu()  — call make_q_matrix() C++ function per linear layer.
      4. free_from_gpu()  — explicitly destruct QMatrix C++ objects (no GC safety net).

    Memory safety: ExLlamaV3 QMatrix objects live in C++ heap.
    PyTorch GC is blind to them.  If free_from_gpu() is not called before
    the next inject_to_gpu(), the 16 GB GPU will OOM within a few layers.
    """

    def __init__(self):
        _require_exllamav3()

    # ── one-time init ────────────────────────────────────────────────────────

    def setup_model(
        self,
        model: nn.Module,
        config,
        device: str,
        dtype: torch.dtype,
        pinned_layer_names: set | None = None,
    ) -> None:
        """
        Traverse the model and replace nn.Linear with ExLlamaV3Linear.
        Excludes: embed_tokens (is Embedding, not Linear), lm_head (tied weights).
        Runs once after init_empty_weights() — all tensors are on meta at this point.
        """
        pinned = pinned_layer_names or set()
        swapped = 0

        for module_name, module in list(model.named_modules()):
            # Skip non-Linear modules
            if not isinstance(module, nn.Linear):
                continue
            # For ExLlamaV3, lm_head is typically quantized. If tied weights are used, 
            # it might not be, but in this checkpoint it is.
            # Pinned layers (like lm_head) should still be quantized if they are nn.Linear
            # They will simply not be freed from GPU during streaming.

            # Build replacement
            has_bias = module.bias is not None
            replacement = ExLlamaV3Linear(module.in_features, module.out_features, bias=has_bias)

            # Navigate to parent and swap the attribute
            *parent_path, attr = module_name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            setattr(parent, attr, replacement)
            swapped += 1

        print(f"[Vaayu-EXL3] swapped {swapped} nn.Linear → ExLlamaV3Linear")

    # ── per-layer I/O ────────────────────────────────────────────────────────

    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        """
        Reads EXL3 q_* tensors from disk using the existing io_uring/preadv path.
        Enforces the strict dtype requirements of ExLlamaV3's C++ kernels.
        Non-quantized tensors (bias, norm weights) pass through unchanged.
        """
        from ..utils import load_layer
        raw: dict = load_layer(checkpoint_path, layer_name)

        corrected = {}
        for key, tensor in raw.items():
            param_base = key.rsplit(".", 1)[-1]  # e.g. "q_weight" from "layer.q_proj.q_weight"
            if param_base in _EXL3_DTYPE_MAP:
                corrected[key] = tensor.to(_EXL3_DTYPE_MAP[param_base])
            else:
                corrected[key] = tensor
        return corrected

    def inject_to_gpu(
        self,
        model: nn.Module,
        layer_name: str,
        tensors: dict,
        device: str,
        dtype: torch.dtype,
        hf_quantizer=None,
        path_remappings: dict | None = None,
    ) -> list:
        """
        For each ExLlamaV3Linear in this layer:
          1. Move q_* tensors to GPU.
          2. Call make_q_matrix() to build the C++ QMatrix in VRAM.
          3. Attach QMatrix to the module.
        For non-quantized tensors (norms, biases): standard set_module_tensor_to_device.
        """
        try:
            from exllamav3 import ext as exl3_ext
        except ImportError:
            raise RuntimeError("exllamav3 not installed. Cannot inject EXL3 tensors.")

        remappings = path_remappings or {}
        moved = []
        qmatrices_this_layer = []

        # Group tensors by their parent module path
        module_tensors: dict[str, dict] = {}
        for param_name, tensor in tensors.items():
            # Apply any VLM path remappings (same logic as vaayu_base)
            mapped_name = param_name
            for old_path, new_path in remappings.items():
                if param_name.startswith(old_path):
                    mapped_name = param_name.replace(old_path, new_path, 1)
                    break
            # Split into module_path + param_base
            parts = mapped_name.rsplit(".", 1)
            if len(parts) == 2:
                mod_path, param_base = parts
            else:
                mod_path, param_base = "", parts[0]
            module_tensors.setdefault(mod_path, {})[param_base] = (mapped_name, tensor)

        for mod_path, params in module_tensors.items():
            # Resolve the module
            try:
                mod = model
                for part in mod_path.split("."):
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
            except (AttributeError, KeyError, IndexError, TypeError):
                mod = None

            if isinstance(mod, ExLlamaV3Linear) and any(
                base in _EXL3_QTENSOR_KEYS for base in params
            ):
                # ── EXL3 quantized linear: build QMatrix ──────────────────
                gpu_tensors = {}
                param_names_moved = []
                print(f"Resolving mod_path: {mod_path}, mod type: {type(mod)}")
                for base, (full_name, tensor) in params.items():
                    gpu_tensors[base] = tensor.to(device, non_blocking=True)
                    param_names_moved.append(full_name)
                    moved.append(full_name)

                trellis = gpu_tensors.get("trellis")
                suh = gpu_tensors.get("suh")
                svh = gpu_tensors.get("svh")
                mcg_tensor = gpu_tensors.get("mcg")
                mul1_tensor = gpu_tensors.get("mul1")

                # TODO: add fallback if su/sv packed weights are present
                if suh is None and "su" in gpu_tensors:
                    raise NotImplementedError("su packed weight detected, please unpack to suh")

                K = trellis.shape[-1] // 16 if trellis is not None else 0

                from exllamav3.util.tensor import g_tensor_cache
                bsz1_xh_args = (device, (1, mod.in_features), torch.float16)
                xh = g_tensor_cache.get(*bsz1_xh_args)

                mod.bc = exl3_ext.exllamav3_ext.BC_LinearEXL3(
                    trellis,
                    suh,
                    svh,
                    K,
                    mod.bias_param,
                    mcg_tensor is not None,
                    mul1_tensor is not None,
                    xh
                )
                qmatrices_this_layer.append((mod_path, mod.bc))

                # Free the intermediate GPU tensors
                del gpu_tensors

            else:
                # ── Non-quantized (norm, bias, embed) — standard injection ─
                for base, (full_name, tensor) in params.items():
                    try:
                        set_module_tensor_to_device(
                            model, full_name, device, value=tensor, dtype=dtype
                        )
                        moved.append(full_name)
                    except Exception as e:
                        print(f"[Vaayu-EXL3] warn: could not inject {full_name}: {e}")

        return moved

    def free_from_gpu(
        self,
        model: nn.Module,
        layer_module: nn.Module,
        moved_layers: list,
        hf_quantizer=None,
    ) -> None:
        """
        CRITICAL: explicitly destroy all QMatrix C++ objects for this layer.
        Failure to do so leaks VRAM and will OOM within a few streaming layers.
        Then move the layer to meta to release PyTorch-managed memory.
        """
        try:
            from exllamav3 import ext as exl3_ext
        except ImportError:
            pass
        else:
            # Walk ExLlamaV3Linear modules in this layer and clear bc
            for sub_module in layer_module.modules():
                if isinstance(sub_module, ExLlamaV3Linear) and sub_module.bc is not None:
                    # Rely on pybind11 refcount to free C++ memory
                    sub_module.bc = None

        # Move non-quantized parameters back to meta
        for param_name in moved_layers:
            param_base = param_name.rsplit(".", 1)[-1]
            if param_base not in _EXL3_QTENSOR_KEYS:
                try:
                    set_module_tensor_to_device(model, param_name, "meta")
                except Exception:
                    pass

        layer_module.to("meta")


# ─────────────────────────────────────────────────────────────────────────────
# Format detection + factory
# ─────────────────────────────────────────────────────────────────────────────

def _peek_safetensors_keys(file_path: str) -> set[str]:
    """Return the set of base parameter names (last path component) in a safetensors file."""
    try:
        with open(file_path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        return {k.rsplit(".", 1)[-1] for k in hdr if k != "__metadata__"}
    except Exception:
        return set()


def _is_exl3_checkpoint(checkpoint_path: str) -> bool:
    """
    Peek into the first available layer shard to detect EXL3 format.
    EXL3 uses standard safetensors containers but with q_* tensor names.
    """
    candidates = sorted(
        glob.glob(str(Path(checkpoint_path) / "model.layers.0.safetensors"))
    )
    if not candidates:
        candidates = sorted(glob.glob(str(Path(checkpoint_path) / "*.safetensors")))
    if not candidates:
        return False
    keys = _peek_safetensors_keys(candidates[0])
    return bool(keys & _EXL3_QTENSOR_KEYS)


def _require_exllamav3() -> None:
    try:
        import exllamav3  # noqa: F401
    except ImportError:
        raise ImportError(
            "EXL3 checkpoint detected but exllamav3 is not installed.\n"
            "Install with: MAX_JOBS=4 pip install exllamav3\n"
            "(requires CUDA Toolkit 12.x and gcc/g++)"
        )


def build_loader_strategy(checkpoint_path: str, vaayu_base_ref) -> LoaderStrategy:
    """
    Auto-detect checkpoint format and return the appropriate LoaderStrategy.
    Called once during VaayuBaseModel.__init__() after checkpoint_path is known.
    """
    if _is_exl3_checkpoint(checkpoint_path):
        print("[Vaayu] EXL3 checkpoint detected → EXL3LoaderStrategy")
        return EXL3LoaderStrategy()
    else:
        print("[Vaayu] Standard safetensors checkpoint → StandardSafetensorsStrategy")
        return StandardSafetensorsStrategy(vaayu_base_ref)
