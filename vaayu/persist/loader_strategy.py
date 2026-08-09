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
import time
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

def _assign_tensor_by_path(root: nn.Module, dotted: str, tensor) -> None:
    """Put `tensor` back at `dotted` under `root` (buffer or attribute)."""
    parts = dotted.split(".")
    mod = root
    for part in parts[:-1]:
        mod = getattr(mod, part)
    leaf = parts[-1]
    if leaf in mod._buffers:
        mod._buffers[leaf] = tensor
    else:
        setattr(mod, leaf, tensor)


def _to_meta_preserving_buffers(layer_module: nn.Module) -> None:
    """`layer_module.to("meta")`, but buffers survive.

    Sending PARAMETERS to meta is the point of layer streaming: they are the
    large tensors and the next per-layer load restores them from the
    checkpoint. BUFFERS are different. A module may register one the checkpoint
    has no tensor for -- MiniMax-M2's `MiniMaxM2SparseMoeBlock` does exactly
    this with `e_score_correction_bias` -- in which case the only thing that
    ever materialized it was the load-time `named_buffers()` pass in
    vaayu_base. Re-metaing it here destroyed it permanently: the layer's router
    worked on the FIRST forward pass and every later pass died with
    "Tensor on device meta is not on the expected device cuda:0!".

    Costs no extra memory: the saved entries are the SAME tensor objects the
    module already held, so holding a reference across the call keeps them
    alive rather than copying them.
    """
    saved = [(name, buf) for name, buf in layer_module.named_buffers(recurse=True)
             if buf is not None and buf.device.type != "meta"]
    layer_module.to("meta")
    for name, buf in saved:
        _assign_tensor_by_path(layer_module, name, buf)


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
        cache = getattr(self, '_cache', None)
        if cache is not None:
            cached = cache.get(layer_name)
            if cached is not None:
                return cached

        from ..utils import load_layer
        tensors = load_layer(checkpoint_path, layer_name)

        if cache is not None:
            cache.put(layer_name, tensors)

        return tensors

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
            _to_meta_preserving_buffers(layer_module)


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


# fp16 ceiling. A CUDA/kernel-contract constant (EXL3 kernels are fp16), not a
# model number — see CLAUDE.md §5.
_FP16_MAX = 65504.0
# Narrow to at most this before the cast, leaving headroom so the GEMM's own
# accumulation doesn't immediately overflow the fp16 output either.
_FP16_SAFE = _FP16_MAX / 2.0


def fp16_safe_narrow(x: torch.Tensor):
    """Cast `x` to fp16 without overflowing, returning (x_fp16, scale).

    bf16 carries fp32's exponent range (~3e38) while fp16 tops out at 65504, so
    a plain `.to(torch.float16)` silently turns large activations into `inf` —
    which the EXL3 kernel then propagates as `NaN`. Confirmed on-device: MoE
    `silu(w1(x)) * w3(x)` exceeded 65504 for a high-gain expert, and the `w2`
    call that consumed it produced NaN (`nan=1, inf=0`) with no crash.

    A linear operator is homogeneous, f(x/s) * s == f(x), so the caller can
    rescale the OUTPUT by `scale` and recover the true result exactly (to fp16
    rounding). That is why this rescales rather than clamping, which would
    silently corrupt magnitudes.

    The division happens BEFORE the narrowing — dividing after the cast would
    be too late, the `inf` already exists.

    `scale` is returned as a device TENSOR, never a Python float: this runs on a
    per-expert, per-token hot path (~350-400 calls/token), and `.item()` would
    force a host sync on every one. Returns scale == 1.0 whenever the input
    already fits, making the common path byte-identical to a plain cast.
    """
    if x.dtype == torch.float16:
        return x, torch.ones((), dtype=torch.float32, device=x.device)
    if x.numel() == 0:
        return x.to(torch.float16), torch.ones((), dtype=torch.float32, device=x.device)
    amax = x.detach().abs().amax().float()
    scale = torch.clamp(amax / _FP16_SAFE, min=1.0)
    if bool(scale.requires_grad):
        scale = scale.detach()
    return (x / scale.to(x.dtype)).to(torch.float16), scale


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
        self.bc = None  # set by inject_to_gpu (EXL3-quantized weight)
        # Some EXL3 checkpoints leave select layers unquantized (plain F16/BF16
        # weight) — e.g. small MoE router gates, kept unquantized for routing
        # stability. setup_model() swaps every nn.Linear to ExLlamaV3Linear
        # regardless, so this module must also support a plain-matmul fallback
        # for tensors inject_to_gpu() finds are NOT EXL3 quant tensors.
        self.weight_fallback = None  # set by inject_to_gpu (plain weight)

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bc is None and self.weight_fallback is None:
            raise RuntimeError(
                f"ExLlamaV3Linear (out={self.out_features}): bc not set. "
                "inject_to_gpu() must be called before forward()."
            )
        if self.bc is None:
            import torch.nn.functional as F
            bias = self.bias_param if self.bias_param is not None else None
            return F.linear(x, self.weight_fallback, bias)
        try:
            # Overflow-safe narrowing: a plain .to(float16) turns any bf16
            # activation above 65504 into inf, which this kernel propagates as
            # NaN. fp16_safe_narrow rescales instead; the output is scaled back
            # below (valid because a linear op is homogeneous). scale is 1.0 for
            # in-range input, making the common path byte-identical.
            x_in, _x_scale = fp16_safe_narrow(x)
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

            # Undo the input rescale (f(x/s) * s == f(x) for a linear f). Done
            # AFTER widening back to x's dtype on purpose: multiplying while
            # still fp16 could push the result straight back over 65504 and
            # reintroduce the very inf this guard exists to prevent. scale is
            # 1.0 whenever the input already fit, so this is a no-op then.
            return out * _x_scale.to(out.dtype)
        except ImportError:
            raise RuntimeError("exllamav3 not installed. Cannot run ExLlamaV3Linear.forward().")


def build_exl3_qmatrix(
    mod: "ExLlamaV3Linear", tensors_by_base: dict, device: str, non_blocking: bool = True,
    timing: dict | None = None,
) -> dict:
    """
    Build a C++ QMatrix (BC_LinearEXL3) from a group of EXL3 quant tensors
    (trellis/suh/svh/mcg/mul1, keyed by base name) and attach it to mod.bc.

    Returns the dict of GPU-resident quant tensors it built, so a caller that
    caches the QMatrix can also hold (and account for) the memory behind it.

    Shared by EXL3LoaderStrategy.inject_to_gpu (per-layer, pinned layers) and
    moe_patch.py's per-token expert injection (module-list MoE layout, e.g.
    MiniMax) — both need identical QMatrix construction, and the module-list
    per-token path additionally needs the caller to clear mod.bc afterward
    (see EXL3LoaderStrategy.free_from_gpu for why: C++-owned, not PyTorch-GC'd).

    non_blocking=True only actually overlaps when the source tensor is
    page-locked; for a pageable source, PyTorch silently routes it through an
    internal pinned staging-buffer pool (CachingHostAllocator) that
    torch.cuda.empty_cache() does NOT reclaim (that only clears the device
    allocator) — under thousands of small per-token copies from non-pinned
    disk-loaded tensors this pool balloons into a real host-RAM leak
    (confirmed: prefill alone drove RSS to ~22-27GB). The per-layer pinned
    path (inject_to_gpu, SlabArena-backed) keeps non_blocking=True since its
    sources are genuinely pinned; the per-token module-list path
    (moe_patch.py) passes non_blocking=False.

    timing: optional dict to accumulate {'h2d': seconds, 'construct': seconds}
    into (keys added with += so repeated calls, e.g. one per submodule role,
    sum correctly). Deliberately a plain, caller-owned dict rather than the
    MoE profiling dict itself — keeps this module unaware of _moe_profile's
    shape. None (the default, and what inject_to_gpu passes implicitly by
    omission) means "don't time" — zero overhead, unchanged behavior.
    """
    from exllamav3 import ext as exl3_ext
    from exllamav3.util.tensor import g_tensor_cache

    if timing is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
    gpu_tensors = {k: t.to(device, non_blocking=non_blocking) for k, t in tensors_by_base.items()}
    if timing is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing['h2d'] = timing.get('h2d', 0.0) + (time.perf_counter() - _t0)

    trellis = gpu_tensors.get("trellis")
    suh = gpu_tensors.get("suh")
    svh = gpu_tensors.get("svh")
    mcg_tensor = gpu_tensors.get("mcg")
    mul1_tensor = gpu_tensors.get("mul1")

    # TODO: add fallback if su/sv packed weights are present
    if suh is None and "su" in gpu_tensors:
        raise NotImplementedError("su packed weight detected, please unpack to suh")

    K = trellis.shape[-1] // 16 if trellis is not None else 0

    bsz1_xh_args = (device, (1, mod.in_features), torch.float16)
    xh = g_tensor_cache.get(*bsz1_xh_args)

    if timing is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
    mod.bc = exl3_ext.exllamav3_ext.BC_LinearEXL3(
        trellis,
        suh,
        svh,
        K,
        mod.bias_param,
        mcg_tensor is not None,
        mul1_tensor is not None,
        xh,
    )
    if timing is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing['construct'] = timing.get('construct', 0.0) + (time.perf_counter() - _t0)

    # Returned so callers can keep their own reference for byte accounting: the
    # QMatrix holds these tensors C++-side, where ExpertVRAMCache._group_bytes
    # cannot see them. inject_to_gpu ignores this; the module-list expert cache
    # (moe_patch.py) does not.
    return gpu_tensors


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
        Cache stores post-dtype-correction tensors so hits skip correction too.
        """
        cache = getattr(self, '_cache', None)
        if cache is not None:
            cached = cache.get(layer_name)
            if cached is not None:
                return cached

        from ..utils import load_layer
        raw: dict = load_layer(checkpoint_path, layer_name)

        corrected = {}
        for key, tensor in raw.items():
            param_base = key.rsplit(".", 1)[-1]  # e.g. "q_weight" from "layer.q_proj.q_weight"
            if param_base in _EXL3_DTYPE_MAP:
                corrected[key] = tensor.to(_EXL3_DTYPE_MAP[param_base])
            else:
                corrected[key] = tensor

        if cache is not None:
            cache.put(layer_name, corrected)

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
                for base, (full_name, tensor) in params.items():
                    moved.append(full_name)
                build_exl3_qmatrix(
                    mod, {base: tensor for base, (_, tensor) in params.items()}, device
                )
                qmatrices_this_layer.append((mod_path, mod.bc))

            else:
                # ── Non-quantized (norm, bias, embed, unquantized Linear) ──
                for base, (full_name, tensor) in params.items():
                    if isinstance(mod, ExLlamaV3Linear) and base == "weight":
                        # Checkpoint left this ExLlamaV3Linear's weight
                        # unquantized (e.g. a small MoE router gate) — no
                        # "weight" attribute exists on the module to target
                        # via set_module_tensor_to_device, use the plain
                        # fallback path instead.
                        mod.weight_fallback = tensor.to(device, dtype=dtype, non_blocking=True)
                        moved.append(full_name)
                        continue
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
            # Walk ExLlamaV3Linear modules in this layer and clear bc / plain fallback
            for sub_module in layer_module.modules():
                if isinstance(sub_module, ExLlamaV3Linear):
                    if sub_module.bc is not None:
                        # Rely on pybind11 refcount to free C++ memory
                        sub_module.bc = None
                    sub_module.weight_fallback = None

        # Move non-quantized parameters back to meta (no-ops harmlessly for
        # ExLlamaV3Linear plain-fallback weights — those aren't registered
        # torch Parameters, already cleared via weight_fallback above).
        for param_name in moved_layers:
            param_base = param_name.rsplit(".", 1)[-1]
            if param_base not in _EXL3_QTENSOR_KEYS:
                try:
                    set_module_tensor_to_device(model, param_name, "meta")
                except Exception:
                    pass

        _to_meta_preserving_buffers(layer_module)


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
