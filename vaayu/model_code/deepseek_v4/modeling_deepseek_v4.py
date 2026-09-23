"""DeepSeek-V4-Flash modeling code -- A5.3: dequant and `Linear`.

This checkpoint ships TWO weight formats and `config.json`'s
`quantization_config` names only one of them:

| what | weight | scale |
|---|---|---|
| routed experts (94.3 % of the bytes) | fp4 packed 2/byte, stored `I8` | `F8_E8M0`, per row per fp4 group |
| shared experts + attention | `F8_E4M3` | `F8_E8M0`, per `weight_block_size` tile |

The scale SHAPES differ, not just the dtypes, and the two GEMM kernels declare
them differently -- `[N, ceildiv(K, group)]` for fp4 against
`[ceildiv(N, block), ceildiv(K, block)]` for fp8.

Ported from the checkpoint's own `inference/model.py` with two deliberate
divergences, both forced by how Vaayu streams experts rather than by taste:

1. **`self.scale`, never `self.weight.scale`.** See `DeepseekV4Linear`.
2. **No module globals.** The reference takes `scale_fmt`, `scale_dtype` and
   `default_dtype` from globals that only `Transformer.__init__` sets
   (`model.py:881-886`), so a module built any other way loads quantized
   weights with the scale silently discarded. Everything here is read from the
   config or off the tensors themselves.

This file is copied verbatim into the checkpoint directory; the repo holds the
source of truth and `tests/test_deepseek_v4_linear.py` asserts the two are
byte-identical.
"""
from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# DeepSeek's TileLang kernels ship INSIDE the checkpoint (`inference/kernel.py`)
# and are not importable as a package. transformers copies remote code into its
# own module cache without the siblings, so the loader (A5.7) sets this to the
# checkpoint's `inference` directory; the fallback is the directory beside this
# file, which is where a copy next to the shards finds them.
KERNEL_DIR: Path | None = None

# Both pins are load-bearing and the tvm-ffi one is EXACT: tilelang 0.1.8 asks
# for `apache-tvm-ffi>=0.1.2,~=0.1.0`, so pip resolves a version that imports
# fine and then dies at kernel COMPILE. `import tilelang` is not a test of it.
_PINS = "tilelang==0.1.8 + apache-tvm-ffi==0.1.9"

_KERNELS = None

_QUANTIZED = (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)
_SCALE_DTYPE = torch.float8_e8m0fnu

# Tensors the module declares bf16 although the checkpoint stores them
# quantized: `Attention.forward` uses `wo_a`'s RAW weight in an einsum, so it
# never passes through a GEMM that could apply the scale. Declared here as a
# per-tensor rule rather than hidden in a forward.
DEQUANTIZE_AT_LOAD = ("wo_a",)


def kernels():
    """DeepSeek's `act_quant` / `fp4_gemm` / `fp8_gemm`, imported on first use.

    `kernel.py` imports tilelang, which imports TVM. Reading a config must not
    need either (CLAUDE.md Section 5), so nothing here is imported until a
    quantized weight is actually multiplied.
    """
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS

    directory = Path(KERNEL_DIR) if KERNEL_DIR is not None else Path(__file__).parent / "inference"
    source = directory / "kernel.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"DeepSeek-V4's TileLang kernels are not at {source}. They ship inside "
            f"the checkpoint, under `inference/`; point "
            f"`modeling_deepseek_v4.KERNEL_DIR` at that directory.")

    spec = importlib.util.spec_from_file_location("vaayu_deepseek_v4_kernel", source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as missing:
        raise ImportError(
            f"DeepSeek-V4's quantized weights go through its own TileLang kernels "
            f"({source}), which need {_PINS} -- both exact, and `import tilelang` "
            f"alone does not test the second.") from missing
    _KERNELS = module
    return module


def dequantize_block_scaled(weight: torch.Tensor, scale: torch.Tensor,
                            dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Multiply a block-scaled quantized weight out into `dtype`.

    `scale[i, j]` covers the tile `weight[i*bo:(i+1)*bo, j*bi:(j+1)*bi]`, and
    both block sizes are derived from the two shapes rather than passed -- the
    checkpoint is the only place they are stated together.
    """
    out_block = weight.shape[0] // scale.shape[0]
    in_block = weight.shape[-1] // scale.shape[-1]
    tiled = (weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
             * scale[:, None, :, None].float())
    return tiled.flatten(2, 3).flatten(0, 1).to(dtype)


def dequantize_at_load(weights: dict) -> dict:
    """Apply `DEQUANTIZE_AT_LOAD` to one module's state dict.

    Without this, `load_state_dict` copies raw F8_E4M3 values into a bf16
    Parameter WITHOUT the scale -- no exception, wrong numbers -- and leaves
    the scale over as an unexpected key.
    """
    out = {}
    for key, tensor in weights.items():
        prefix, _, leaf = key.rpartition(".")
        if prefix.rpartition(".")[2] not in DEQUANTIZE_AT_LOAD:
            out[key] = tensor
        elif leaf == "weight":
            out[key] = dequantize_block_scaled(tensor, weights[f"{prefix}.scale"])
        elif leaf != "scale":
            out[key] = tensor
    return out


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Both GEMM wrappers allocate their output with `torch.get_default_dtype()`
    (`kernel.py:268`, `:533`) while the kernel is compiled to write bf16. The
    reference gets away with it because `generate.py:77` sets the process
    default; an engine may not depend on a process-wide global.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class DeepseekV4Linear(nn.Module):
    """A `Linear` over this checkpoint's two quantized formats, plus bf16.

    **The mandatory divergence: `forward` reads `self.scale`, never
    `self.weight.scale`.** The reference aliases the scale onto the Parameter
    (`model.py:145`) and its `linear()` reads it back off there. Under Vaayu
    that alias does not survive: `_inject_expert_weights`
    (`moe_patch.py:1225`) rebinds every streamed key with
    `setattr(submod, base, nn.Parameter(t, requires_grad=False))`, so `weight`
    becomes a fresh object with no `.scale` -- an `AttributeError` on every
    streamed expert, on every token. Both keys DO bind, so streaming a
    dual-tensor quantized expert needs no engine changes at all.

    `fp4_group_size` has no config field: it is knowable only from the shipped
    scale shape (`in_features // scale.shape[-1]`), which is why the caller
    reads it off the checkpoint and passes it in.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 dtype: torch.dtype | None = None,
                 weight_block_size: list | tuple | None = None,
                 fp4_group_size: int | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16

        if dtype == torch.float4_e2m1fn_x2:
            if fp4_group_size is None:
                raise ValueError(
                    "fp4_group_size is in no config field -- read it off the shipped "
                    "scale shape (in_features // scale.shape[-1]) and pass it.")
            # fp4 is packed 2 per byte along K; the scale is per row, per group.
            self._parameter("weight", (out_features, in_features // 2), dtype)
            self._parameter("scale", (out_features, in_features // fp4_group_size),
                            _SCALE_DTYPE)
        elif dtype == torch.float8_e4m3fn:
            out_block, in_block = weight_block_size
            self._parameter("weight", (out_features, in_features), dtype)
            self._parameter("scale", (-(-out_features // out_block),
                                      -(-in_features // in_block)), _SCALE_DTYPE)
        else:
            self._parameter("weight", (out_features, in_features), dtype)
            self.register_parameter("scale", None)

        # Activations are quantized per `weight_block_size`'s K block for BOTH
        # formats -- only the WEIGHT scale is per fp4 group.
        self.act_block_size = weight_block_size[-1] if weight_block_size else None

    def _parameter(self, name: str, shape: tuple, dtype: torch.dtype) -> None:
        setattr(self, name, nn.Parameter(torch.empty(shape, dtype=dtype),
                                         requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dtype not in _QUANTIZED:
            return F.linear(x, weight)

        kernel = kernels()
        scale = self.scale                       # NOT weight.scale -- see the docstring
        # `act_quant` rounds its scales to powers of two only when `scale_fmt`
        # is set, and that has to match how the weights were quantized. The
        # scale tensor states it; the reference took it from a global.
        scale_fmt = "ue8m0" if scale.dtype == _SCALE_DTYPE else None
        gemm = kernel.fp4_gemm if weight.dtype == torch.float4_e2m1fn_x2 else kernel.fp8_gemm
        with _default_dtype(x.dtype):
            quantized, act_scale = kernel.act_quant(x, self.act_block_size,
                                                    scale_fmt, scale.dtype)
            return gemm(quantized, act_scale, weight, scale, scale.dtype)
