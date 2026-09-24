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
import contextvars
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

# Both are literals in the reference with NO config field, and both are
# load-bearing (CLAUDE.md §5 allows this only because there is no source to
# derive them from -- say so, do not hide them).
_ACT_QUANT_GROUP = 64          # model.py:512 -- 64, NOT weight_block_size's 128
_FP4_BLOCK_SIZE = 32           # model.py:18  -- the Indexer's fp4 simulation

# Tensors the module declares bf16 although the checkpoint stores them
# quantized: `Attention.forward` uses `wo_a`'s RAW weight in an einsum, so it
# never passes through a GEMM that could apply the scale. Declared here as a
# per-tensor rule rather than hidden in a forward.
DEQUANTIZE_AT_LOAD = ("wo_a",)

# `sparse_attn_kernel` is compiled per (heads, head_dim), and the tile has to be
# chosen before the source is exec'd -- before any config is in scope. These are
# this architecture's only shape, read back out of the config by the assertion in
# `DeepseekV4Attention.__init__` rather than trusted here.
_TILE_HEADS, _TILE_HEAD_DIM = 64, 512


def sparse_attn_smem(heads: int, head_dim: int, block: int) -> int:
    """Dynamic shared memory `sparse_attn_kernel` asks for, in bytes.

    Fitted to the three points A5's step-1 gate MEASURED on sm_120 (block 64 ->
    141,312 B, 32 -> 104,448, 16 -> fits): tilelang aliases `q_shared` and
    `o_shared` into one `heads x head_dim` bf16 buffer, adds ~2 KiB of index and
    alignment, and then pays per tile for `kv_shared` (`block x head_dim` bf16)
    and `acc_s_cast` (`heads x block` bf16). `num_stages` does not move it.
    """
    return heads * head_dim * 2 + 2048 + block * head_dim * 2 + heads * block * 2


def sparse_attn_tile(smem_optin: int, heads: int, head_dim: int) -> tuple:
    """The largest shipped-or-smaller `(block, num_stages)` this GPU can run.

    `smem_optin` is a HARDWARE property read off the device, not a constant:
    a datacenter card keeps the shipped `block = 64, num_stages = 2`, and only
    a consumer card (sm_120 offers 101,376 B) steps down. Stages drop to 1 with
    the block because a 1/4-size tile has nothing to pipeline against.
    """
    for block, stages in ((64, 2), (32, 1), (16, 1)):
        if sparse_attn_smem(heads, head_dim, block) <= smem_optin:
            return block, stages
    raise RuntimeError(
        f"no sparse_attn tile fits this GPU: the smallest asks "
        f"{sparse_attn_smem(heads, head_dim, 16)} B of dynamic shared memory and "
        f"the device offers {smem_optin} B per block.")


def _fit_sparse_attn(source: str, heads: int, head_dim: int, smem_optin: int) -> str:
    """`kernel.py` with `sparse_attn_kernel`'s tile constants rewritten.

    Both are LOCALS in the kernel factory, so there is no argument to turn --
    the source is the only handle. Returns the source unchanged when the
    shipped tile already fits, or when the source is not DeepSeek's kernel
    module at all (a test double standing in for the real `kernel.py`).

    "Is this DeepSeek's kernel module" is decided by `hc_split_sinkhorn_kernel`,
    a NEIGHBOR def -- never `sparse_attn_kernel`'s own name. The failure this
    guards against is exactly `sparse_attn_kernel` being renamed or moved;
    discriminating on its own name would make that failure indistinguishable
    from "not DeepSeek's module" and silently no-op instead of raising.
    """
    import re

    block, stages = sparse_attn_tile(smem_optin, heads, head_dim)
    if (block, stages) == (64, 2) or "def hc_split_sinkhorn_kernel" not in source:
        return source
    if "def sparse_attn_kernel" not in source:
        raise RuntimeError(
            "this is DeepSeek's kernel module (hc_split_sinkhorn_kernel is "
            "present) but sparse_attn_kernel is not -- it was renamed or moved, "
            "and the sm_120 fit must be re-derived.")
    start = source.index("def sparse_attn_kernel")
    end = source.index("def hc_split_sinkhorn_kernel")
    tile = source[start:end]
    tile, a = re.subn(r"\n    num_stages = \d+", f"\n    num_stages = {stages}",
                      tile, count=1)
    tile, b = re.subn(r"\n    block = \d+", f"\n    block = {block}", tile, count=1)
    if a != 1 or b != 1:
        raise RuntimeError(
            "sparse_attn_kernel's tile constants are not where this build expects "
            "them; the kernel source changed and the sm_120 fit must be re-derived.")
    return source[:start] + tile + source[end:]


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

    text = source.read_text()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        # `sparse_attn`'s tile is the one kernel whose shared-memory demand
        # exceeds a consumer card's budget. Fit it before tilelang's JIT calls
        # `inspect.getsource`, which is why this has to be a REAL file on disk.
        text = _fit_sparse_attn(text, _TILE_HEADS, _TILE_HEAD_DIM,
                                properties.shared_memory_per_block_optin)
    fitted = directory / "kernel.py"
    if text != source.read_text():
        import tempfile
        fitted = Path(tempfile.mkdtemp(prefix="vaayu_ds4_kernel_")) / "kernel.py"
        fitted.write_text(text)

    spec = importlib.util.spec_from_file_location("vaayu_deepseek_v4_kernel", fitted)
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


# ── A5.4: the MoE block ─────────────────────────────────────────────────────

_INPUT_IDS = contextvars.ContextVar("deepseek_v4_input_ids", default=None)


@contextlib.contextmanager
def input_ids_scope(input_ids: torch.Tensor):
    """Stash the batch's token ids for the hash-routed layers.

    Layers below `num_hash_layers` route by TOKEN ID, not by score, and Vaayu's
    `sparse_forward(hidden_states, **kwargs)` never receives `input_ids`.
    Threading them through would edit MoE infra that Qwen3-Next and MiniMax
    also run (CLAUDE.md §0 makes that a change to BOTH paths); excluding those
    layers from patching would strand three whole layers of experts resident.
    So the model stashes them here instead -- entirely inside this file, zero
    engine changes.

    A `ContextVar` rather than a plain global: the server runs one request at a
    time today, but two generations sharing a process must never read each
    other's ids.
    """
    token = _INPUT_IDS.set(input_ids)
    try:
        yield
    finally:
        _INPUT_IDS.reset(token)


def current_input_ids() -> torch.Tensor:
    """The stashed ids, or a message naming what forgot to stash them."""
    input_ids = _INPUT_IDS.get()
    if input_ids is None:
        raise RuntimeError(
            "a hash-routed MoE layer routes by token id and no input_ids are stashed "
            "-- the model's forward must run its layers inside `input_ids_scope`.")
    return input_ids


def _expert_dtypes(config) -> tuple:
    """(routed, shared) expert weight dtypes, both read off the config.

    `expert_dtype` names the routed experts' format; everything else takes the
    checkpoint's declared `fmt`. Neither is assumed -- a bf16 checkpoint of this
    architecture builds bf16 experts and never touches a kernel.
    """
    quant = config.checkpoint_quantization_config or {}
    shared = torch.float8_e4m3fn if quant.get("fmt") == "e4m3" else torch.bfloat16
    routed = torch.float4_e2m1fn_x2 if config.expert_dtype == "fp4" else shared
    return routed, shared


class DeepseekV4Gate(nn.Module):
    """Router scores, plus the selection Vaayu asks the model to own.

    Split from the reference's `Gate`, which returns `(weights, indices)` from
    its forward. Vaayu's forwards call `router(hidden_states) -> router_logits`
    and then `route_tokens_to_experts(logits)` (`moe_patch.py:155`) -- the
    contract transformers standardizes across 25+ MoE architectures. So
    `forward` returns the RAW scores and `route` does the scoring, the
    selection and the mixing.

    **The bias steers SELECTION ONLY.** Taking the weights from the biased
    scores is what made MiniMax emit incoherent text while loading, patching
    and streaming perfectly.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size,
                        dtype=torch.bfloat16), requires_grad=False)
        if self.hash:
            # `[vocab_size, top_k]`, and the shards hold it I64 -- declaring the
            # reference's int32 would cast every id table at load.
            self.tid2eid = nn.Parameter(
                torch.empty(config.vocab_size, config.num_experts_per_tok,
                            dtype=torch.int64), requires_grad=False)
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts,
                                                 dtype=torch.float32),
                                     requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight.float())

    def _score(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return router_logits.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return router_logits.sigmoid()
        return F.softplus(router_logits).sqrt()          # sqrtsoftplus

    def route(self, router_logits: torch.Tensor) -> tuple:
        """`(indices, weights)` -- `route_experts` tells them apart by dtype."""
        unbiased = self._score(router_logits)
        scores = unbiased if self.bias is None else unbiased + self.bias
        if self.hash:
            indices = self.tid2eid[current_input_ids().flatten()]
        else:
            indices = scores.topk(self.top_k, dim=-1)[1]
        weights = unbiased.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights * self.route_scale


class DeepseekV4Expert(nn.Module):
    """One SwiGLU FFN, with the reference's numerics kept exactly.

    `gate` and `up` are computed in float32, the `swiglu_limit` clamp is
    ASYMMETRIC on purpose (`up` two-sided, `gate` capped above only), and the
    product is cast back before `w2`. Each of those is a logits-level
    difference, not a style choice.

    `weights` is optional because Vaayu's module-list forward calls the expert
    with ONE argument and applies the routing weight itself, OUTSIDE
    (`moe_patch.py:1710`). The two are equal -- `w2` is a bias-free linear map
    and the weight is a per-token scalar, so it commutes through -- and this is
    written down because it reads like a bug.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, *,
                 dtype: torch.dtype | None = None, swiglu_limit: float = 0.0,
                 weight_block_size: list | tuple | None = None,
                 fp4_group_size: int | None = None):
        super().__init__()
        shape = dict(dtype=dtype, weight_block_size=weight_block_size,
                     fp4_group_size=fp4_group_size)
        self.w1 = DeepseekV4Linear(hidden_size, intermediate_size, **shape)
        self.w2 = DeepseekV4Linear(intermediate_size, hidden_size, **shape)
        self.w3 = DeepseekV4Linear(hidden_size, intermediate_size, **shape)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class DeepseekV4MoE(nn.Module):
    """`gate` + `experts` + an ungated `shared_experts`, on Vaayu's module-list
    forward with **zero engine changes**.

    The reference's names are load-bearing and are kept: `detect_moe_blocks`
    (`moe_patch.py:992`) wants an `nn.ModuleList` child with more than four
    entries and a sibling named one of `gate`/`router`/`w_gate`/
    `router_weights` carrying `.weight`, and `_shared_expert_output`
    (`moe_patch.py:536`) finds the shared expert by the `shared_expert` in its
    name, gating it only when a `*_gate` sibling exists. Renaming any of them to
    HF's `mlp`/`gate_proj` would cost a key remap AND a silent detector miss.
    """

    def __init__(self, config, layer_idx: int, *, fp4_group_size: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        routed_dtype, shared_dtype = _expert_dtypes(config)
        block = (config.checkpoint_quantization_config or {}).get("weight_block_size")

        def expert(dtype, intermediate_size):
            return DeepseekV4Expert(config.hidden_size, intermediate_size, dtype=dtype,
                                    swiglu_limit=config.swiglu_limit,
                                    weight_block_size=block,
                                    fp4_group_size=fp4_group_size)

        self.gate = DeepseekV4Gate(config, layer_idx)
        self.experts = nn.ModuleList([expert(routed_dtype, config.moe_intermediate_size)
                                      for _ in range(config.n_routed_experts)])
        # One fused module of `n_shared_experts` experts' width, always on.
        self.shared_experts = expert(
            shared_dtype, config.moe_intermediate_size * config.n_shared_experts)

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple:
        """Vaayu's routing contract (`route_experts`, `moe_patch.py:155`)."""
        return self.gate.route(router_logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        indices, weights = self.route_tokens_to_experts(self.gate(x))
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(),
                                minlength=self.n_routed_experts).tolist()
        for expert_idx, count in enumerate(counts):
            if count == 0:
                continue
            token, slot = torch.where(indices == expert_idx)
            # The routing weight OUTSIDE the expert, as the engine applies it.
            y[token] += weights[token, slot, None] * self.experts[expert_idx](x[token])
        y += self.shared_experts(x)
        return y.type_as(hidden_states).view(shape)


# ── A5.5: attention primitives ──────────────────────────────────────────────

import math
from functools import lru_cache


class DeepseekV4RMSNorm(nn.Module):
    """`model.py:189`. Normalises in fp32 and casts back.

    The weight is fp32 although the checkpoint stores it bf16 -- the reference
    declares it that way and relies on `load_state_dict` upcasting. Declaring
    bf16 would change the norm's precision on every layer of the model.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32),
                                   requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def _freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
               device_str):
    """`model.py:205`, with the device made explicit.

    `lru_cache(2)` is the reference's own size and it is exactly right: this
    model has TWO rope variants (YaRN-off at base theta for `compress_ratio ==
    0`, YaRN-on at `compress_rope_theta` otherwise). A third distinct call
    pattern evicts one and re-derives it every layer.

    The reference builds its `torch.arange` on the default device, which is why
    its harness must `set_default_device("cuda")` (A5.2 finding 2). Taking the
    device as an argument removes that trap at the source instead of working
    around it.
    """
    device = torch.device(device_str)

    def correction_dim(rotations):
        return (dim * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base)))

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                         device=device) / dim))
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
                / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor,
                         beta_fast, beta_slow, device) -> torch.Tensor:
    return _freqs_cis(dim, seqlen, original_seq_len, float(base), float(factor),
                      beta_fast, beta_slow, str(device))


def rope_settings(config, compress_ratio: int) -> dict:
    """The two variants, read off the config (never a literal theta).

    `compress_ratio == 0` layers are pure sliding-window attention: the
    reference disables YaRN for them by passing `original_seq_len = 0` and uses
    the BASE `rope_theta` (`model.py:484`). Everything else uses
    `compress_rope_theta` with YaRN.
    """
    rope = config.rope_parameters or {}
    if not compress_ratio:
        return {"original_seq_len": 0, "base": config.rope_theta,
                "factor": 1.0, "beta_fast": rope["beta_fast"],
                "beta_slow": rope["beta_slow"]}
    return {"original_seq_len": rope["original_max_position_embeddings"],
            "base": config.compress_rope_theta, "factor": rope["factor"],
            "beta_fast": rope["beta_fast"], "beta_slow": rope["beta_slow"]}


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor,
                     inverse: bool = False) -> torch.Tensor:
    """`model.py:238`. Rotates IN PLACE and returns the same tensor.

    In place is not a style choice: every call site in the reference mutates a
    SLICE and ignores the return (`apply_rotary_emb(q[..., -rd:], ...)`).
    `inverse=True` de-rotates, and the reference applies it a second time to
    the attention OUTPUT (`model.py:539`) -- unusual, and easy to drop.
    """
    out = x
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 3:
        freqs_cis = freqs_cis.view(1, z.size(1), z.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, z.size(1), 1, z.size(-1))
    out.copy_(torch.view_as_real(z * freqs_cis).flatten(-2))
    return out


def view_history(window_size: int, start_pos: int) -> int:
    """How many pre-chunk positions the `kv_view` must carry.

    A query at absolute `p` attends `[max(0, p - window_size + 1), p]`, so a
    chunk starting at `start_pos` needs `window_size - 1` earlier positions --
    capped by how many exist. THIS is why a chunk cannot attend the ring in
    place: 512 new positions overwrite a 128-slot ring before its own early
    queries have read it.
    """
    return min(start_pos, window_size - 1)


def window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                     device) -> torch.Tensor:
    """Sliding-window indices INTO THE `kv_view`, for any seqlen at any
    start_pos (`model.py:261` supports only seqlen==1 past position 0).

    Absolute position `a` sits at `a - start_pos + h`. Masked entries are -1,
    which the kernel reads as "skip" (`kernel.py:322-326`).
    """
    h = view_history(window_size, start_pos)
    positions = start_pos + torch.arange(seqlen, device=device).unsqueeze(1)
    oldest = (positions - window_size + 1).clamp(min=0)
    absolute = oldest + torch.arange(window_size, device=device)
    matrix = torch.where(absolute > positions, -1, absolute - start_pos + h)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int,
                       offset: int, device) -> torch.Tensor:
    """Compressed-block indices, for any seqlen at any start_pos
    (`model.py:275` masks causally only when start_pos == 0).

    Block `j` summarises positions `[j*ratio, (j+1)*ratio)`, so a query at
    absolute `p` may see blocks `j < (p + 1) // ratio`.
    """
    blocks = (start_pos + seqlen) // ratio
    positions = start_pos + torch.arange(seqlen, device=device).unsqueeze(1)
    index = torch.arange(blocks, device=device)
    visible = index < ((positions + 1) // ratio)
    matrix = torch.where(visible, index + offset, -1)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


# ── A5.5: the cache ─────────────────────────────────────────────────────────

from transformers.cache_utils import Cache, CacheLayerMixin

_ONE_LENGTH = (
    "a DeepSeek-V4 cache is servable at exactly one length: the circular "
    "sliding window holds the last `sliding_window` positions at slots "
    "`pos % sliding_window`, so the positions a shorter crop would need were "
    "overwritten. Drop the entry instead of cropping it."
)


class DeepseekV4CacheLayer(CacheLayerMixin):
    """One layer's whole per-sequence state.

    The reference keeps all of this in `register_buffer`s on the modules. It
    lives here instead so the modules hold weights only: two sequences cannot
    share state, and 42-B can snapshot a cache rather than reach into module
    internals.
    """

    is_compileable = False

    def __init__(self, *, compress_ratio, window_size, head_dim, max_seq_len,
                 index_head_dim, max_batch_size=1, dtype=torch.bfloat16,
                 device=None):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.cumulative_length = 0

        slots = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
        self.keys = torch.zeros(max_batch_size, slots, head_dim,
                                dtype=dtype, device=device)
        # Captured once, from the tensor torch just placed -- so `device=None`
        # resolves the same way torch resolves it. `prefetch` must move BACK to
        # this fixed target, never re-read it off `keys` (which `offload` has
        # by then moved to cpu, making that read report the wrong place).
        self.compute_device = self.keys.device
        # MLA keeps ONE latent per position and sparse_attn reads it as both
        # operands. The alias is the honest encoding -- see the overrides below.
        self.values = self.keys
        self.is_initialized = True

        coff = 1 + (compress_ratio == 4)
        if compress_ratio:
            shape = (max_batch_size, coff * compress_ratio, coff * head_dim)
            # fp32: the reference compresses in fp32 (`model.py:330`), and the
            # score state is seeded -inf so an unfilled slot cannot win a softmax.
            self.kv_state = torch.zeros(shape, dtype=torch.float32, device=device)
            self.score_state = torch.full(shape, float("-inf"),
                                          dtype=torch.float32, device=device)
        else:
            self.kv_state = self.score_state = None

        if compress_ratio == 4:
            self.index_keys = torch.zeros(max_batch_size,
                                          max_seq_len // compress_ratio,
                                          index_head_dim, dtype=dtype, device=device)
            ishape = (max_batch_size, coff * compress_ratio, coff * index_head_dim)
            self.index_kv_state = torch.zeros(ishape, dtype=torch.float32,
                                              device=device)
            self.index_score_state = torch.full(ishape, float("-inf"),
                                                dtype=torch.float32, device=device)
        else:
            self.index_keys = self.index_kv_state = self.index_score_state = None

    # -- CacheLayerMixin's five abstract methods ----------------------------
    def lazy_initialization(self, key_states, value_states):
        """Nothing to do: every buffer is preallocated from the config, because
        the ring's size IS `window + max_seq_len // ratio` and is not a function
        of what arrives."""

    def update(self, key_states, value_states, **kwargs):
        """`key_states` is the chunk's latents `[b, seqlen, head_dim]`.

        Returns the contiguous view attention reads -- history, then this
        chunk. The compressed tail is appended by the caller, which is the only
        thing that knows whether its Compressor flushed a block this forward.
        """
        start_pos = int(kwargs.get("start_pos", self.cumulative_length))
        view = torch.cat([self.history(start_pos), key_states], dim=1)
        self.write(key_states, start_pos)
        self.cumulative_length = start_pos + key_states.shape[1]
        return view, view

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Ring capacity and zero offset.

        **This model builds no HF attention mask.** `sparse_attn` masks by
        reading -1 out of `topk_idxs` (`kernel.py:322`), and causality lives in
        `window_topk_idxs` / `compress_topk_idxs`. The pair exists to satisfy
        `CacheLayerMixin`; nothing in the forward consults `attention_mask`.
        """
        return self.keys.shape[1], 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    # -- the ring -----------------------------------------------------------
    def history(self, start_pos: int) -> torch.Tensor:
        """The `window_size - 1` positions before `start_pos`, oldest first."""
        h = view_history(self.window_size, start_pos)
        if h == 0:
            return self.keys[:, :0]
        slots = (torch.arange(start_pos - h, start_pos, device=self.keys.device)
                 % self.window_size)
        return self.keys[:, slots]

    def write(self, kv: torch.Tensor, start_pos: int) -> None:
        """Carry the chunk's tail into the ring for the next forward."""
        seqlen = kv.shape[1]
        keep = min(seqlen, self.window_size)
        tail = kv[:, seqlen - keep:]
        slots = (torch.arange(start_pos + seqlen - keep, start_pos + seqlen,
                              device=self.keys.device) % self.window_size)
        self.keys[:, slots] = tail.to(self.keys.dtype)

    def nbytes(self) -> int:
        total = self.keys.numel() * self.keys.element_size()   # values IS keys
        for tensor in (self.kv_state, self.score_state, self.index_keys,
                       self.index_kv_state, self.index_score_state):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    # -- three inherited methods the alias breaks ---------------------------
    def offload(self):
        """The inherited version moves `keys` and `values` independently, which
        splits the alias into two tensors that then drift apart."""
        self._move("cpu")

    def prefetch(self):
        self._move(self.compute_device)

    def _move(self, device):
        self.keys = self.keys.to(device, non_blocking=True)
        self.values = self.keys
        for name in ("kv_state", "score_state", "index_keys",
                     "index_kv_state", "index_score_state"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device, non_blocking=True))

    def reorder_cache(self, beam_idx):
        raise RuntimeError(
            "DeepSeek-V4 does not support beam search: `values` aliases `keys`, "
            "so the inherited reorder would index_select the same ring twice "
            "and desynchronise the alias. Vaayu's server never beams.")

    def reset(self):
        self.keys.zero_()
        self.cumulative_length = 0
        if self.kv_state is not None:
            self.kv_state.zero_()
            self.score_state.fill_(float("-inf"))
        if self.index_keys is not None:
            self.index_keys.zero_()
            self.index_kv_state.zero_()
            self.index_score_state.fill_(float("-inf"))


class DeepseekV4Cache(Cache):
    """One `DeepseekV4CacheLayer` per body layer, sized from the config.

    Nothing below is a literal: `sliding_window`, `head_dim`, `index_head_dim`
    and `compress_ratios` all come off the config (CLAUDE.md §5).
    """

    def __init__(self, config, max_seq_len: int, max_batch_size: int = 1,
                 device=None, dtype=torch.bfloat16):
        layers = [
            DeepseekV4CacheLayer(
                compress_ratio=config.compress_ratios[layer_idx],
                window_size=config.sliding_window,
                head_dim=config.head_dim,
                max_seq_len=max_seq_len,
                index_head_dim=config.index_head_dim,
                max_batch_size=max_batch_size, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.max_seq_len = max_seq_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def advance(self, n: int) -> None:
        """Move every layer's counter. Tests and the model's own bookkeeping
        use it; a real forward advances through `update`."""
        for layer in self.layers:
            layer.cumulative_length = n

    def crop(self, max_length: int) -> None:
        """Refuses below the current length -- see `_ONE_LENGTH`.

        A no-op crop TO the current length is allowed, because that is what
        `prefix_cache`'s `min(...)` clamp produces on the common path and it
        asks nothing of the ring.
        """
        current = self.get_seq_length()
        if max_length >= current:
            return
        raise RuntimeError(_ONE_LENGTH)

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


# ── A5.5: the Compressor ─────────────────────────────────────────────────────


def _have_hadamard_extension() -> bool:
    """Is DeepSeek's own `fast_hadamard_transform` importable here?"""
    import importlib.util

    return importlib.util.find_spec("fast_hadamard_transform") is not None


def _torch_hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    """The Sylvester Hadamard transform in plain torch -- exact for a
    power-of-two last dimension, which both reference call sites have
    (`index_head_dim`). Ported from `tests/deepseek/parity.py`'s
    `torch_hadamard` so this path never hard-requires the compiled extension
    (CLAUDE.md §5).
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Hadamard needs a power-of-two dim, got {n}"
    out = x.float().reshape(-1, n)
    step = 1
    while step < n:
        out = out.view(-1, n // (2 * step), 2, step)
        top, bottom = out[:, :, 0], out[:, :, 1]
        out = torch.stack([top + bottom, top - bottom], dim=2)
        step *= 2
    return (out.reshape(x.shape) * scale).to(x.dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """`model.py:253`. DeepSeek's compiled extension when it is importable,
    the exact torch fallback otherwise -- a config-only import path must not
    need nvcc.
    """
    assert x.dtype == torch.bfloat16
    scale = x.size(-1) ** -0.5
    if _have_hadamard_extension():
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=scale)
    return _torch_hadamard(x, scale=scale)


def compressor_flush_positions(ratio: int, start_pos: int, seqlen: int) -> list:
    """The absolute positions at which a block completes inside this chunk.

    The reference's decode branch flushes iff `(start_pos + 1) % ratio == 0`
    (`model.py:352`) -- one position, one possible flush. A chunk may complete
    many blocks, and missing any of them leaves a hole in the compressed cache
    that nothing ever fills.
    """
    return [p for p in range(start_pos, start_pos + seqlen)
            if (p + 1) % ratio == 0]


class DeepseekV4Compressor(nn.Module):
    """`model.py:285`. Learned gated pooling over `compress_ratio` positions.

    `wkv`/`wgate` are fp32 Parameters although the checkpoint stores them bf16
    -- the reference declares them that way (`model.py:303`) and compresses in
    fp32. `overlap` (ratio == 4 only) doubles the projected width and pools
    over a window that straddles the block boundary.

    State lives in the cache layer, not here, so this module is reusable across
    sequences.
    """

    def __init__(self, config, compress_ratio: int, head_dim: int,
                 rotate: bool = False):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim,
                                            dtype=torch.float32),
                                requires_grad=False)
        self.wkv = DeepseekV4Linear(config.hidden_size, coff * head_dim,
                                    dtype=torch.float32)
        self.wgate = DeepseekV4Linear(config.hidden_size, coff * head_dim,
                                      dtype=torch.float32)
        self.norm = DeepseekV4RMSNorm(head_dim, config.rms_norm_eps)

    def _overlap_transform(self, tensor, value):
        """`model.py:314`. `[b, s, ratio, 2d] -> [b, s, 2*ratio, d]`, with the
        previous block's half shifted in so the pooling window overlaps."""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        out = tensor.new_full((b, s, 2 * ratio, d), value)
        out[:, :, ratio:] = tensor[:, :, :, d:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return out

    def forward(self, x, start_pos, state, freqs_cis):
        """Compress every block this chunk completes; buffer the remainder.

        Unlike the reference, `seqlen` may be anything at any `start_pos`. The
        chunk's leading partial block is completed out of `state`, the whole
        blocks run the reference's own pooling arithmetic, and the trailing
        partial block goes back into `state`.
        """
        ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
        bsz, seqlen, _ = x.size()
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x) + self.ape[
            (start_pos + torch.arange(seqlen, device=x.device)) % ratio]

        # Prepend whatever of the current block `state` is holding, so the
        # blocks below are always whole.
        held = start_pos % ratio
        offset = ratio if self.overlap else 0
        if held:
            kv = torch.cat([state.kv_state[:bsz, offset:offset + held], kv], dim=1)
            score = torch.cat([state.score_state[:bsz, offset:offset + held],
                               score], dim=1)
        first = start_pos - held
        total = kv.shape[1]
        cutoff = total - total % ratio

        if cutoff < ratio:                       # no block completes here
            self._hold(state, kv, score, offset, bsz)
            return None

        if self.overlap and first >= ratio:
            # The previous block's raw kv/score, carried forward the last
            # time a block flushed (see the write at the bottom of this
            # branch) -- the overlap window needs it whole.
            kv = torch.cat([state.kv_state[:bsz, :ratio], kv], dim=1)
            score = torch.cat([state.score_state[:bsz, :ratio], score], dim=1)
            cutoff += ratio

        whole_kv, tail_kv = kv[:, :cutoff], kv[:, cutoff:]
        whole_score = score[:, :cutoff]
        pooled = whole_kv.unflatten(1, (-1, ratio))
        pooled_score = whole_score.unflatten(1, (-1, ratio))
        if self.overlap:
            pooled = self._overlap_transform(pooled, 0)
            pooled_score = self._overlap_transform(pooled_score, float("-inf"))
        pooled = (pooled * pooled_score.softmax(dim=2)).sum(dim=2)
        if self.overlap:
            if first >= ratio:
                pooled = pooled[:, 1:]           # the carried block re-pooled
            # Carry the block THIS call just completed forward raw, for the
            # next call's overlap window. The reference does this by shifting
            # its "current window" into the "overlap window" once a block
            # flushes (model.py:359-360) -- that only works there because its
            # decode branch fills the current window one position at a time,
            # so by the time a block flushes the current window already IS
            # that block's raw data. This path computes a whole block from a
            # local tensor and never touches kv_state/score_state while doing
            # it, so a shift here would carry forward whatever stale (zero)
            # data happened to be sitting in the current-window slots, not the
            # block that was actually just completed. Write it explicitly.
            state.kv_state[:bsz, :ratio] = whole_kv[:, cutoff - ratio:cutoff]
            state.score_state[:bsz, :ratio] = whole_score[:, cutoff - ratio:cutoff]
        self._hold(state, tail_kv, score[:, cutoff:], offset, bsz)

        pooled = self.norm(pooled.to(dtype))
        blocks = compressor_flush_positions(ratio, start_pos, seqlen)
        rows = torch.tensor([p + 1 - ratio for p in blocks], device=x.device)
        apply_rotary_emb(pooled[..., -rd:], freqs_cis[rows])
        kernel = kernels()
        if self.rotate:
            pooled = rotate_activation(pooled)
            kernel.fp4_act_quant(pooled, _FP4_BLOCK_SIZE, True)
        else:
            kernel.act_quant(pooled[..., :-rd], _ACT_QUANT_GROUP, "ue8m0",
                             _SCALE_DTYPE, True)
        target = state.index_keys if self.rotate else state.keys
        base = 0 if self.rotate else state.window_size
        slots = torch.tensor([p // ratio for p in blocks], device=x.device)
        target[:bsz, base + slots] = pooled.to(target.dtype)
        return pooled

    def _hold(self, state, kv, score, offset, bsz):
        """Buffer the chunk's trailing partial block so a later call can
        complete it. Only `[offset:offset+held]` is ever read back (the
        `held` prepend above), so nothing else here needs clearing -- the
        overlap "previous block" carry lives in `[:offset]` and is written
        only when a block actually flushes, in `forward` above.
        """
        held = kv.shape[1]
        if held:
            state.kv_state[:bsz, offset:offset + held] = kv
            state.score_state[:bsz, offset:offset + held] = score
