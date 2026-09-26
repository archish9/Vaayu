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

Vaayu registers `DeepseekV4ForCausalLM` in-process
(`vaayu/vaayu_deepseek_v4.py`'s `_register_with_transformers`), so a plain
download -- no `auto_map`, no copied file -- loads as it comes, and nothing is
written into the model directory (S5, S5-W). A checkpoint that still carries a
hand-placed copy loads that copy instead; `tests/test_deepseek_v4_linear.py`
(and its `model`/`moe` twins) check such a copy has not drifted, and skip when
there is none.
"""
from __future__ import annotations

import contextlib
import contextvars
import importlib.util
import sys
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

# The reference's own discriminator for "this layer has an Indexer and its
# ring/compressor state overlaps `head_dim`" (`model.py`'s `Attention.__init__`
# and `Compressor.__init__`) -- a fixed property of the architecture, not a
# config field, so it is named here rather than typed as a bare `4` at each
# call site.
_INDEXER_COMPRESS_RATIO = 4

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


@contextlib.contextmanager
def no_bytecode():
    """Import a checkpoint's own source without caching bytecode beside it.

    `exec_module` writes `__pycache__/*.pyc` next to the file it runs, and that
    file is inside the user's download (S5-W). It stays a real import -- not
    S5's `exec(compile())` -- because tilelang's JIT reads the kernels back with
    `inspect.getsource`. The flag is process-global: it covers everything that
    file imports for the first time too (tilelang included), and it is not
    thread-safe, so it is held only for the one import. Every caller is on the
    single-request forward path.
    """
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


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
        import atexit
        import shutil
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="vaayu_ds4_kernel_")
        # tilelang's JIT is lazy (compiles -- and calls `inspect.getsource` --
        # on first kernel invocation, not here), so this directory must
        # outlive `kernels()`. Removed at process exit instead of never.
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
        fitted = Path(tmpdir) / "kernel.py"
        fitted.write_text(text)

    spec = importlib.util.spec_from_file_location("vaayu_deepseek_v4_kernel", fitted)
    module = importlib.util.module_from_spec(spec)
    try:
        with no_bytecode():
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
            if weight_block_size is None:
                raise ValueError(
                    "weight_block_size is in no config field -- an F8_E4M3 weight "
                    "carries a per-tile scale and the tile shape lives only in the "
                    "checkpoint's `checkpoint_quantization_config.weight_block_size`. "
                    "Read it off the config and pass it. (A bf16 checkpoint of this "
                    "architecture needs none and never reaches here.)")
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

from transformers import GenerationMixin
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_deepseek_v4 import DeepseekV4Config

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

        coff = 1 + (compress_ratio == _INDEXER_COMPRESS_RATIO)
        if compress_ratio:
            shape = (max_batch_size, coff * compress_ratio, coff * head_dim)
            # fp32: the reference compresses in fp32 (`model.py:330`), and the
            # score state is seeded -inf so an unfilled slot cannot win a softmax.
            self.kv_state = torch.zeros(shape, dtype=torch.float32, device=device)
            self.score_state = torch.full(shape, float("-inf"),
                                          dtype=torch.float32, device=device)
        else:
            self.kv_state = self.score_state = None

        if compress_ratio == _INDEXER_COMPRESS_RATIO:
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
    """Is DeepSeek's own `fast_hadamard_transform` importable here?

    This is an importability check, not a device check. If the package is
    installed, `rotate_activation` calls it unconditionally -- including on a
    CPU tensor, where it raises `RuntimeError: Expected x.is_cuda()` rather
    than falling back. Harmless today because the whole Indexer path
    (`rotate_activation`'s only caller) is CUDA-only, but the torch fallback
    below is reachable only when the extension is NOT installed, not whenever
    CUDA is unavailable.
    """
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
    need nvcc. The fallback is selected by importability alone, not by
    device: with the extension installed, a CPU `x` reaches the extension and
    raises rather than falling back (see `_have_hadamard_extension`).
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
        self.overlap = compress_ratio == _INDEXER_COMPRESS_RATIO
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

        # The Indexer builds a SECOND instance of this module over the same
        # cache layer (`rotate=True`, `head_dim=index_head_dim`), so its
        # scratch space cannot be the main compressor's `kv_state`/
        # `score_state` -- those are shaped off `head_dim`, not
        # `index_head_dim`, and the two compressors run in the same forward.
        kv_state = state.index_kv_state if self.rotate else state.kv_state
        score_state = state.index_score_state if self.rotate else state.score_state

        # Prepend whatever of the current block `state` is holding, so the
        # blocks below are always whole.
        held = start_pos % ratio
        offset = ratio if self.overlap else 0
        if held:
            kv = torch.cat([kv_state[:bsz, offset:offset + held], kv], dim=1)
            score = torch.cat([score_state[:bsz, offset:offset + held],
                               score], dim=1)
        first = start_pos - held
        total = kv.shape[1]
        cutoff = total - total % ratio

        if cutoff < ratio:                       # no block completes here
            self._hold(kv_state, score_state, kv, score, offset, bsz)
            return None

        if self.overlap and first >= ratio:
            # The previous block's raw kv/score, carried forward the last
            # time a block flushed (see the write at the bottom of this
            # branch) -- the overlap window needs it whole.
            kv = torch.cat([kv_state[:bsz, :ratio], kv], dim=1)
            score = torch.cat([score_state[:bsz, :ratio], score], dim=1)
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
            kv_state[:bsz, :ratio] = whole_kv[:, cutoff - ratio:cutoff]
            score_state[:bsz, :ratio] = whole_score[:, cutoff - ratio:cutoff]
        self._hold(kv_state, score_state, tail_kv, score[:, cutoff:], offset, bsz)

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

    def _hold(self, kv_state, score_state, kv, score, offset, bsz):
        """Buffer the chunk's trailing partial block so a later call can
        complete it. Only `[offset:offset+held]` is ever read back (the
        `held` prepend above), so nothing else here needs clearing -- the
        overlap "previous block" carry lives in `[:offset]` and is written
        only when a block actually flushes, in `forward` above.
        """
        held = kv.shape[1]
        if held:
            kv_state[:bsz, offset:offset + held] = kv
            score_state[:bsz, offset:offset + held] = score


# ── A5.5: the Indexer ────────────────────────────────────────────────────────


def indexer_causal_bound(ratio: int, start_pos: int, seqlen: int,
                         device) -> torch.Tensor:
    """`[seqlen, 1]` -- the exclusive block bound each query may select from.

    The reference computes this only for `start_pos == 0` (`model.py:431`); a
    chunk at any other position needs it just as much, and without it a query
    early in the chunk can select a compressed block summarising positions it
    has not reached. Same subexpression `compress_topk_idxs` already uses
    above, factored out because the Indexer needs the bound itself, not the
    finished index matrix.
    """
    positions = start_pos + torch.arange(seqlen, device=device).unsqueeze(1)
    return (positions + 1) // ratio


class DeepseekV4Indexer(nn.Module):
    """`model.py:386`. Scores compressed positions and keeps `index_topk`.

    Carries its own `Compressor` with the Hadamard rotation on
    (`rotate=True`), writing into the cache layer's `index_*` buffers rather
    than a buffer of its own.

    Generalises BOTH of the reference's `if start_pos == 0:` masks
    (`model.py:430-436`) to every `start_pos`: the score mask (so a query
    never scores its own future) and the post-topk mask (so a selection that
    only exists because there weren't `index_topk` visible blocks yet becomes
    `-1`, not a real index into the future). The reference's `else` branch
    (`topk_idxs += offset`, no masking at all) is exactly the bug a chunk
    boundary at `start_pos > 0` would hit.

    **`wq_b` is quantized, unlike the brief's plain-bf16 proposal.** The
    checkpoint ships `indexer.wq_b.{weight,scale}` -- F8_E4M3 with a block
    scale, the same "shared experts + attention" format `_expert_dtypes`
    already derives off the config, not the fp4 routed-expert format. A bf16
    `DeepseekV4Linear` here raises on load (`Unexpected key(s): "wq_b.scale"`)
    before a single token runs it.
    """

    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.index_head_dim ** -0.5
        _, attn_dtype = _expert_dtypes(config)
        block = (config.checkpoint_quantization_config or {}).get("weight_block_size")
        self.wq_b = DeepseekV4Linear(config.q_lora_rank,
                                     config.index_n_heads * config.index_head_dim,
                                     dtype=attn_dtype, weight_block_size=block)
        self.weights_proj = DeepseekV4Linear(config.hidden_size,
                                             config.index_n_heads,
                                             dtype=torch.bfloat16)
        self.compressor = DeepseekV4Compressor(config, compress_ratio,
                                               config.index_head_dim, rotate=True)

    def forward(self, x, qr, start_pos, offset, state, freqs_cis):
        bsz, seqlen, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        end_pos = start_pos + seqlen

        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis[start_pos:end_pos])
        q = rotate_activation(q)
        kernels().fp4_act_quant(q, _FP4_BLOCK_SIZE, True)

        self.compressor(x, start_pos, state, freqs_cis)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        blocks = end_pos // ratio
        score = torch.einsum("bshd,btd->bsht", q, state.index_keys[:bsz, :blocks])
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        bound = indexer_causal_bound(ratio, start_pos, seqlen, x.device)
        visible = torch.arange(blocks, device=x.device) < bound
        score = score + torch.where(visible, 0.0, float("-inf"))
        topk = score.topk(min(self.index_topk, blocks), dim=-1)[1]
        return torch.where(topk < bound, topk + offset, -1).int()


# ── A5.5: DeepseekV4Attention ────────────────────────────────────────────────


class DeepseekV4Attention(nn.Module):
    """`model.py:442`. MLA + sliding window + optional KV compression.

    Differs from the reference in exactly one structural way, and it is the
    point of A5.5: attention attends a contiguous `kv_view` it assembles per
    forward -- `[history | this chunk | compressed]` -- instead of the ring in
    place. A 512-token chunk overruns a 128-slot ring, so the reference's
    decode branch cannot be generalised, only replaced.

    At both of the reference's supported call shapes the view reduces to what
    it already does: `start_pos == 0` gives `history = 0` and index == absolute
    position, and `seqlen == 1` gives the same `window_size` positions in the
    same oldest-to-newest order the ring walk produces.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        if (config.num_attention_heads, config.head_dim) != (_TILE_HEADS, _TILE_HEAD_DIM):
            raise ValueError(
                f"the sparse_attn tile was fitted for {_TILE_HEADS} heads x "
                f"{_TILE_HEAD_DIM}, but this config asks for "
                f"{config.num_attention_heads} x {config.head_dim}; re-derive it "
                f"in `kernels()` before loading this checkpoint.")
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim ** -0.5
        eps = config.rms_norm_eps
        block = (config.checkpoint_quantization_config or {}).get("weight_block_size")
        fp8 = dict(dtype=torch.float8_e4m3fn, weight_block_size=block)

        # F32, per head, inside the softmax -- see `forward`.
        self.attn_sink = nn.Parameter(
            torch.empty(config.num_attention_heads, dtype=torch.float32),
            requires_grad=False)
        self.wq_a = DeepseekV4Linear(config.hidden_size, config.q_lora_rank, **fp8)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps)
        self.wq_b = DeepseekV4Linear(config.q_lora_rank,
                                     config.num_attention_heads * config.head_dim,
                                     **fp8)
        self.wkv = DeepseekV4Linear(config.hidden_size, config.head_dim, **fp8)
        self.kv_norm = DeepseekV4RMSNorm(config.head_dim, eps)
        # bf16 in the module, F8_E4M3 + per-128x128 scale on disk. A5.3's
        # `dequantize_at_load` multiplies the scale out once, at load, because
        # `forward` uses the raw weight in an einsum (`model.py:543`).
        self.wo_a = DeepseekV4Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = DeepseekV4Linear(config.o_groups * config.o_lora_rank,
                                     config.hidden_size, **fp8)

        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio,
                                                   config.head_dim)
            self.indexer = (DeepseekV4Indexer(config, self.compress_ratio)
                            if self.compress_ratio == _INDEXER_COMPRESS_RATIO
                            else None)
        else:
            self.compressor = self.indexer = None
        self._rope = rope_settings(config, self.compress_ratio)
        self._rope_dim = config.qk_rope_head_dim

    def _freqs(self, max_seq_len, device):
        return precompute_freqs_cis(self._rope_dim, max_seq_len,
                                    device=device, **self._rope)

    def forward(self, hidden_states, past_key_values, start_pos: int):
        state = past_key_values.layers[self.layer_idx]
        bsz, seqlen, _ = hidden_states.size()
        rd, win = self.rope_head_dim, self.window_size
        device = hidden_states.device
        freqs_cis = self._freqs(past_key_values.max_seq_len, device)
        positions = freqs_cis[start_pos:start_pos + seqlen]

        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.q_norm.eps)
        apply_rotary_emb(q[..., -rd:], positions)

        kv = self.kv_norm(self.wkv(hidden_states))
        apply_rotary_emb(kv[..., -rd:], positions)
        # FP8 QAT simulation on the NON-rope dims, group 64 (not 128), in place.
        kernels().act_quant(kv[..., :-rd], _ACT_QUANT_GROUP, "ue8m0",
                            _SCALE_DTYPE, True)

        history = state.history(start_pos)
        h = history.shape[1]
        topk_idxs = window_topk_idxs(win, bsz, seqlen, start_pos, device)
        view = torch.cat([history, kv], dim=1)

        if self.compress_ratio:
            offset = h + seqlen
            # The reference calls its OWN Attention-level `self.compressor`
            # unconditionally here (`model.py:530` / `:537`), regardless of
            # whether an Indexer exists -- it fills `state.keys`' compressed
            # half that `view` reads just below. The Indexer (present only on
            # compress_ratio == 4 layers) carries a SEPARATE `Compressor`
            # instance over `index_head_dim`, writing `state.index_keys` for
            # its own scoring; it is not a substitute for this call.
            self.compressor(hidden_states, start_pos, state, freqs_cis)
            if self.indexer is not None:
                compress_idxs = self.indexer(hidden_states, qr, start_pos,
                                             offset, state, freqs_cis)
            else:
                compress_idxs = compress_topk_idxs(
                    self.compress_ratio, bsz, seqlen, start_pos, offset, device)
            blocks = (start_pos + seqlen) // self.compress_ratio
            view = torch.cat([view, state.keys[:bsz, win:win + blocks]], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        state.write(kv, start_pos)
        state.cumulative_length = start_pos + seqlen

        o = kernels().sparse_attn(q, view, self.attn_sink, topk_idxs,
                                  self.softmax_scale)
        # Applied a SECOND time, INVERTED, to the output (`model.py:539`).
        apply_rotary_emb(o[..., -rd:], positions, True)

        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# ── A5.6: the block, hyper-connections and the model ────────────────────────

# `hc_*_scale` is `[3]` in the reference (`model.py:677-678`) because
# `hc_split_sinkhorn` splits its projection into exactly three parts --
# `pre`, `post` and `comb`. No config field carries it, so it is named here
# rather than typed as a bare 3 (CLAUDE.md §5 allows a literal only when there
# is no source to derive it from, and only if it says so).
_HC_SPLIT_PARTS = 3


class DeepseekV4DecoderLayer(nn.Module):
    """`model.py:652`. Attention and the MoE wrapped in hyper-connections.

    The residual stream is `hc_mult` copies wide -- `[b, s, hc, d]` -- and the
    block mixes it down to one copy before attention and again before the FFN,
    then expands it back. `hc_pre` and `hc_post` are the whole of it.

    **Both of them live inside this class, and that is load-bearing:** what
    reaches `self.ffn` is an ordinary `[b, s, d]`, which is why Vaayu's MoE
    patch needs no change (A5.4's "zero engine changes"). If the width ever
    escapes this block, that result no longer holds.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.attn = DeepseekV4Attention(config, layer_idx)
        # `fp4_group_size` reaches us on the config because `from_config` runs
        # under `init_empty_weights` with no checkpoint in reach and Vaayu's
        # call site passes nothing else (`vaayu_base.py:3361`). A5.7's loader
        # sets it; without it A5.4's named ValueError fires, which is the right
        # failure -- it says what is missing instead of inventing a group size.
        self.ffn = DeepseekV4MoE(config, layer_idx,
                                 fp4_group_size=config.fp4_group_size)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        # fp32 EXPLICITLY, never through the default dtype. These tensors are
        # F32 on disk; `from_config(dtype=bf16)` sets torch's default around
        # `__init__`, and accelerate then casts every injected tensor to the
        # dtype the skeleton already has -- so a bf16 skeleton would silently
        # round the checkpoint's fp32 hyper-connection weights down.
        fp32 = dict(dtype=torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, **fp32),
                                       requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, **fp32),
                                      requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, **fp32),
                                         requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, **fp32),
                                        requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(_HC_SPLIT_PARTS, **fp32),
                                          requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(_HC_SPLIT_PARTS, **fp32),
                                         requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        """`model.py:680`. `[b,s,hc,d]` -> `[b,s,d]`, plus the expand weights.

        The normalisation here is a BARE `rsqrt` over the flattened copies, not
        a `DeepseekV4RMSNorm`: there is no learned weight, and it runs in fp32
        whatever the stream's dtype.
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = kernels().hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters,
            self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        """`model.py:690`. `[b,s,d]` back out to `[b,s,hc,d]`."""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, hidden_states, past_key_values, start_pos: int,
                use_cache: bool | None = None):
        """`use_cache` is accepted and ignored.

        Vaayu's layer-major loop hardcodes it into every layer call
        (`vaayu_base.py:5209`) and no hook can remove it. This model's cache
        is preallocated and always live -- there is no uncached mode to
        select -- so the argument has nothing to do here.

        Accepted EXPLICITLY rather than through `**kwargs`: a catch-all would
        swallow a future engine kwarg without a word, which is the defect
        class this port has already paid for three times.
        """
        residual = hidden_states
        x, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn,
                                    self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, past_key_values, start_pos)
        hidden_states = self.hc_post(x, residual, post, comb)

        residual = hidden_states
        x, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn,
                                    self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        # The reference calls `self.ffn(x, input_ids)`. A5.4's MoE takes only
        # `hidden_states` -- Vaayu's forward contract -- and reads the batch's
        # ids from `current_input_ids()`, which the model's forward stashes.
        x = self.ffn(x)
        return self.hc_post(x, residual, post, comb)


class DeepseekV4PreTrainedModel(PreTrainedModel):
    """`config_class` is set explicitly rather than by annotation: transformers
    keeps the attribute path specifically for remote-code models
    (`modeling_utils.py:1220-1226`), which is what this is.
    """

    config_class = DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    # Honest: this model runs DeepSeek's own `sparse_attn` and participates in
    # no HF attention backend. The override below is what makes a request for
    # one harmless.
    _supports_flash_attn = False
    _supports_sdpa = False

    def _check_and_adjust_attn_implementation(self, attn_implementation,
                                              is_init_check: bool = False,
                                              allow_all_kernels: bool = False) -> str:
        """Accept any request and report `eager`.

        `DeepseekV4Attention` calls DeepSeek's `sparse_attn` unconditionally --
        it is the only attention kernel this architecture has, and A5 step 1
        fitted its tile to sm_120 at `block=16`. No HF backend is reachable, so
        there is nothing for this argument to select.

        Left to the base class, the honest `_supports_*` flags above would make
        `flash_attention_2` raise at `_flash_attn_can_dispatch` and `sdpa` at
        `_sdpa_can_dispatch`, and Vaayu's loop (`vaayu_base.py:3330-3378`)
        would build and throw away two 11,008-expert skeletons before reaching
        `eager`. The return value is what transformers stores as
        `config._attn_implementation_internal` (`modeling_utils.py:1255`).
        """
        return "eager"


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    """`model.py:877`'s `Transformer`, without the MTP half.

    embed -> expand to `hc_mult` copies -> the blocks -> `hc_head` -> norm.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV4DecoderLayer(config, layer_idx)
                                     for layer_idx in range(config.num_hidden_layers)])
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        # `hc_head_*` belong to the MODEL, not to a block -- see `hc_head`.
        fp32 = dict(dtype=torch.float32)
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, config.hc_mult * config.hidden_size, **fp32),
            requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, **fp32),
                                         requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, **fp32),
                                          requires_grad=False)

        self.post_init()

    def hc_head(self, x):
        """`model.py:709`. The final `hc_mult` -> 1 reduction.

        The reference defines this on `Block` and calls it as
        `layer.hc_head(...)` on whatever layer its loop left bound
        (`model.py:921`), passing the MODEL's parameters. The body reads `self`
        only for `norm_eps`, `hc_eps` and the `hc_mult` implied by the input's
        shape -- all three per-block copies of the same config fields -- so it
        lives here, where its parameters do. The `hc_head` golden proves the
        move is arithmetically free rather than asserting it.

        **This is NOT `hc_pre`.** Plain sigmoid, no Sinkhorn, no `post`, no
        `comb`, and `hc_head_fn` has `hc_mult` rows, not `(2+hc_mult)*hc_mult`.
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def _start_pos(self, past_key_values, cache_position, seqlen: int) -> int:
        """One absolute position for the whole forward.

        This model has no per-token position vector: the ring indexes by
        `pos % window_size`, the Compressor flushes on block boundaries counted
        from `start_pos`, and `window_topk_idxs` builds its indices from it. So
        `cache_position` must be exactly `arange(start_pos, start_pos+seqlen)`,
        and anything else is refused rather than silently mis-indexed.
        """
        if cache_position is None:
            return int(past_key_values.get_seq_length())
        if cache_position.numel() != seqlen:
            raise ValueError(
                f"cache_position has {cache_position.numel()} entries for a "
                f"{seqlen}-token forward")
        start_pos = int(cache_position[0])
        expected = torch.arange(start_pos, start_pos + seqlen,
                                device=cache_position.device,
                                dtype=cache_position.dtype)
        if not torch.equal(cache_position, expected):
            raise ValueError(
                "DeepSeek-V4 needs a contiguous cache_position -- the sliding "
                "window indexes by `pos % window_size` and the Compressor "
                "flushes on block boundaries counted from the first position, "
                f"so {cache_position.tolist()} would read the wrong slots "
                f"rather than fail. Expected {expected.tolist()}.")
        return start_pos

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                cache_position=None, **kwargs):
        if input_ids is None:
            raise ValueError(
                "DeepSeek-V4 needs input_ids: its first `num_hash_layers` MoE "
                "layers route by TOKEN ID, not by score, so inputs_embeds "
                "alone has nothing to route on. Refusing rather than routing "
                "every token to the same experts.")
        if past_key_values is None:
            raise ValueError(
                "DeepSeek-V4 needs a past_key_values built by the caller. Its "
                "cache is PREALLOCATED -- `sliding_window + max_seq_len // "
                "compress_ratio` slots per layer -- and the config's "
                f"max_position_embeddings ({self.config.max_position_embeddings}) "
                "is a context limit, not a cache size: defaulting to it would "
                "reserve hundreds of MB per layer. Pass "
                "DeepseekV4Cache(config, max_seq_len=<your budget>).")
        if not isinstance(past_key_values, DeepseekV4Cache):
            raise TypeError(
                f"DeepSeek-V4 needs a DeepseekV4Cache, got "
                f"{type(past_key_values).__name__}: a generic cache holds no "
                f"compressed half and no index state, so attention would read "
                f"an empty compressed view instead of failing.")

        # `attention_mask` and `position_ids` are accepted and unused: this
        # model builds no HF mask (`sparse_attn` masks by reading -1 out of
        # `topk_idxs`, `kernel.py:322`) and carries position in `start_pos`.
        hidden_states = self.embed_tokens(input_ids)
        start_pos = self._start_pos(past_key_values, cache_position,
                                    hidden_states.shape[1])
        # `repeat`, not `expand` (`model.py:916`): matches the reference, which
        # is this port's whole method. Checked on-device with `hc_mult=4`:
        # `expand` (stride `(15,5,0,1)`) is numerically indistinguishable here
        # -- its `flatten(2)` (hc_pre) is `torch.equal` to `repeat`'s, hc_post's
        # output is `torch.equal` either way (it never writes into `residual`),
        # and the residual stays unmutated -- so this is a fidelity choice, not
        # a correctness one.
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        with input_ids_scope(input_ids):
            for layer in self.layers:
                hidden_states = layer(hidden_states, past_key_values, start_pos)

        hidden_states = self.norm(self.hc_head(hidden_states))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    """The head. `tie_word_embeddings` is false in this checkpoint, so
    `lm_head` is a separate tensor and `_tied_weights_keys` stays unset.

    **The head takes whatever dtype the model is constructed with, where the
    reference's is unconditionally fp32.** `nn.Linear` here inherits the
    ambient default dtype rather than pinning one -- the dtype-agnostic
    behaviour this engine wants -- so under Vaayu's own construction path
    (`from_config(dtype=self.running_dtype)`) it comes out bf16; a plain
    `DeepseekV4ForCausalLM(config)` with no ambient dtype set yields an fp32
    head instead. `model.py:727-728` upcasts the checkpoint's bf16
    `head.weight` "for easier computation of logits later"; under Vaayu's
    bf16 construction, keeping the disk dtype returns 0.986 GiB (against the
    reference's unconditional fp32 upcast) to a pin budget that also carries
    the shared experts, the hc_* tensors and attention on a 15.47 GiB card --
    that saving does not apply if the head ends up fp32. The matmul still
    accumulates in fp32; only the output rounds through bf16 before the cast
    below, when the head is bf16. The `head` golden gates that choice on
    argmax, not just on rel-L2.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, **kwargs)

        slice_indices = (slice(-logits_to_keep, None)
                         if isinstance(logits_to_keep, int) else logits_to_keep)
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=outputs.past_key_values)
