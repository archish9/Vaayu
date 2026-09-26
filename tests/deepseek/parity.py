"""A5.2 -- the parity harness for the DeepSeek-V4 port.

DeepSeek's reference implementation cannot run whole here: 155.42 GiB of
weights against ~45.3 GiB of VRAM+RAM, and it shards with `torchrun`. But
`world_size = 1` / `rank = 0` are plain module globals (`model.py:15`) and
`generate.py:70` only calls `init_process_group` when `WORLD_SIZE > 1`, so ONE
module runs here with real weights -- 130.53 MiB for the heaviest attention
layer, 12.750 MiB per routed expert.

This module builds those modules and records what they produce, so A5.3-A5.7
have a golden reference that needs neither the checkpoint nor a GPU.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

GOLDENS = Path(__file__).parent / "goldens"

# numpy has no bf16 and no fp8, so a dtype numpy cannot name is stored as its
# raw BIT PATTERN in an integer array of the same width and rebuilt with
# `view`. Exact in both directions -- a golden that drifts by a ULP is not a
# golden -- and half the bytes of widening bf16 to fp32. The real dtype rides
# alongside under this prefix.
_DTYPE_KEY = "__dtype__"
_NOTE_KEY = "__note__"
_BITS = {2: torch.uint16, 1: torch.uint8, 4: torch.uint32, 8: torch.uint64}


def save_golden(name: str, tensors: dict, root: Path = GOLDENS,
                note: str = None) -> Path:
    """Write one golden: the seeded input AND the output, so the fixture is
    self-describing and a later session can prove it regenerates.

    `note` records what happened when there is no output to record -- a
    reference that RAISES is still a golden, and the most valuable one here.
    """
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, tensor in tensors.items():
        tensor = tensor.detach().cpu()
        storable = tensor if tensor.dtype in (torch.float32, torch.float64,
                                              torch.int32, torch.int64) else \
            tensor.view(_BITS[tensor.element_size()])
        arrays[key] = storable.numpy()
        arrays[_DTYPE_KEY + key] = np.array(str(tensor.dtype).removeprefix("torch."))
    if note is not None:
        arrays[_NOTE_KEY] = np.array(note)
    path = root / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_golden(name: str, root: Path = GOLDENS) -> dict:
    """Read a golden back with its dtypes restored. No checkpoint, no GPU."""
    with np.load(root / f"{name}.npz", allow_pickle=False) as store:
        return {key: torch.from_numpy(store[key]).view(
                    getattr(torch, str(store[_DTYPE_KEY + key])))
                for key in store.files if not key.startswith("__")}


def load_note(name: str, root: Path = GOLDENS) -> str | None:
    """The note recorded beside a golden, if it has one."""
    with np.load(root / f"{name}.npz", allow_pickle=False) as store:
        return str(store[_NOTE_KEY]) if _NOTE_KEY in store.files else None


CKPT = Path(os.environ.get(
    "A5_CKPT", "/home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731"))


def have_checkpoint() -> bool:
    return (CKPT / "model.safetensors.index.json").is_file()


@functools.lru_cache(maxsize=None)
def weight_map(ckpt: Path = None) -> dict:
    """The index's key -> original shard. It outlives the shards it names:
    `delete_original_shards` keeps it so the checkpoint stays discoverable."""
    ckpt = ckpt or CKPT
    return json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]


@functools.lru_cache(maxsize=None)
def _split_index(split_dir: Path) -> dict:
    """Split key -> the file holding it, read off the headers (~0.7 s for all
    46), so nothing here re-derives the split's file naming."""
    from safetensors import safe_open

    index = {}
    for path in sorted(split_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            index.update(dict.fromkeys(handle.keys(), path))
    return index


def locate(key: str, ckpt: Path = None) -> tuple:
    """(file, name inside it) for a key named the way the checkpoint names it.

    Two layouts are real. A fresh download holds only the ORIGINAL shards the
    index names. A checkpoint Vaayu has loaded with `delete_original=True` --
    the server's default -- holds only `splitted_model/`, every key renamed by
    the loader's own `remap_checkpoint_key` (this box since `4bc70b1`). The
    goldens were recorded off the originals; the split is their byte-for-byte
    copy. Going through the remap rather than a hardcoded `model.` prefix keeps
    the harness tracking the loader instead of guessing its naming.
    """
    from vaayu.model_code.deepseek_v4.checkpoint import remap_checkpoint_key

    ckpt = ckpt or CKPT
    original = ckpt / weight_map(ckpt)[key]
    if original.is_file():
        return original, key
    stored = remap_checkpoint_key(key)
    split = _split_index(ckpt / "splitted_model")
    if stored not in split:
        raise FileNotFoundError(
            f"{key}: its original shard {original.name} is gone and no shard under "
            f"{ckpt / 'splitted_model'} holds it as {stored!r}"
            + (" (remap_checkpoint_key drops it)" if stored is None else ""))
    return split[stored], stored


def read_checkpoint_tensors(keys, ckpt: Path = None) -> dict:
    """{key: tensor} exactly as stored, from whichever layout holds each key."""
    from safetensors import safe_open

    by_file = {}
    for key in keys:
        path, stored = locate(key, ckpt)
        by_file.setdefault(path, []).append((key, stored))
    out = {}
    for path, pairs in sorted(by_file.items()):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key, stored in pairs:
                out[key] = handle.get_tensor(stored)
    return out


def raw_module_weights(prefix: str, ckpt: Path = None) -> dict:
    """The tensors under `prefix`, keyed relative to it, with NO value
    transforms -- what the production loader consumes."""
    wanted = sorted(key for key in weight_map(ckpt or CKPT) if key.startswith(prefix))
    return {key[len(prefix):]: tensor
            for key, tensor in read_checkpoint_tensors(wanted, ckpt).items()}


def load_module_weights(prefix: str, ckpt: Path = None) -> dict:
    """Read exactly the tensors under `prefix`, keyed relative to it.

    Applies the two VALUE transforms `convert.py` does -- the key RENAMES it
    also does are no-ops on this checkpoint, whose shards already ship the
    reference's own names (`layers.7.attn.wq_a.weight`, not `self_attn`).
    """
    raw = raw_module_weights(prefix, ckpt)

    out = {}
    for key, tensor in raw.items():
        if key.endswith("wo_a.weight"):
            # Declared bf16 by the module and used raw in an einsum, so the
            # per-128x128-block scale has to be multiplied out once, here.
            scale = raw[key.replace("weight", "scale")]
            blk = tensor.shape[-1] // scale.shape[-1]
            tensor = (tensor.unflatten(0, (-1, blk)).unflatten(-1, (-1, blk)).float()
                      * scale[:, None, :, None].float())
            out[key] = tensor.flatten(2, 3).flatten(0, 1).bfloat16()
        elif key.endswith("wo_a.scale"):
            continue                          # consumed above
        elif ".experts." in prefix + key and tensor.dtype == torch.int8:
            out[key] = tensor.view(torch.float4_e2m1fn_x2)
        else:
            out[key] = tensor
    return out


# sm_120 offers 101,376 B of dynamic shared memory per block. `sparse_attn`'s
# shipped tile asks 141,312 B; `num_stages` does not move the demand and
# `block` is the only lever. A5 step 1 found block=16 is the first that fits.
# Both are locals inside `sparse_attn_kernel`, so the only way to turn them is
# to rewrite the file -- tilelang's JIT calls `inspect.getsource`, so it has to
# be a REAL file on disk, not an exec'd string.
SPARSE_ATTN_BLOCK = 16
SPARSE_ATTN_STAGES = 1

# Harness sizing, not model geometry: `Attention.__init__` allocates `kv_cache`
# and the RoPE table from `max_seq_len`, and the shipped 1,048,576 would size
# for a million positions to record a 512-token golden.
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 1

_reference = None


def have_cuda() -> bool:
    return torch.cuda.is_available()


def reference_modules(ckpt: Path = None):
    """Import the checkpoint's own `model.py`, with a sm_120-safe attention tile.

    `model.py` does `from kernel import ...`, so the patched kernel has to be
    importable under the name `kernel`. It is shadowed only for the duration of
    the import and then dropped from `sys.modules` -- a module called `kernel`
    left lying around is a collision waiting to happen.
    """
    global _reference
    if _reference is not None:
        return _reference

    import importlib.util
    import re
    import sys
    import tempfile

    inference = (ckpt or CKPT) / "inference"
    source = (inference / "kernel.py").read_text()
    start = source.index("def sparse_attn_kernel")
    end = source.index("def hc_split_sinkhorn_kernel")
    tile = source[start:end]
    tile, stages_hit = re.subn(r"\n    num_stages = \d+",
                               f"\n    num_stages = {SPARSE_ATTN_STAGES}", tile, count=1)
    tile, block_hit = re.subn(r"\n    block = \d+",
                              f"\n    block = {SPARSE_ATTN_BLOCK}", tile, count=1)
    assert stages_hit == block_hit == 1, "sparse_attn tile constants not found"

    variant = Path(tempfile.mkdtemp(prefix="a5_kernel_"))
    (variant / "kernel.py").write_text(source[:start] + tile + source[end:])

    sys.path.insert(0, str(variant))
    sys.path.insert(1, str(inference))
    shadowed = sys.modules.pop("kernel", None)
    try:
        spec = importlib.util.spec_from_file_location("a5_reference_model",
                                                      inference / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(variant))
        sys.path.remove(str(inference))
        # Keep the patched kernel alive on the model module, off `sys.modules`.
        module.a5_patched_kernel = sys.modules.pop("kernel", None)
        if shadowed is not None:
            sys.modules["kernel"] = shadowed

    # `rotate_activation` imports the extension directly, so the fallback only
    # reaches the reference if the reference's own binding is replaced. With
    # the extension present this is the same function by another name.
    module.rotate_activation = rotate_activation

    _reference = module
    return module


def apply_reference_globals(reference, args) -> None:
    """Replicate `Transformer.__init__`'s global assignment (model.py:881-886).

    Without this `default_dtype` stays bf16, so an FP8 weight loads into a bf16
    Parameter with its scale discarded -- silently wrong, no exception.
    """
    reference.world_size = 1
    reference.rank = 0
    reference.default_dtype = (torch.float8_e4m3fn if args.dtype == "fp8"
                               else torch.bfloat16)
    reference.scale_fmt = ("ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt)
    reference.scale_dtype = (torch.float8_e8m0fnu if args.scale_dtype == "fp8"
                             else torch.float32)


@contextlib.contextmanager
def reference_device(reference, device: str = "cuda"):
    """Torch's default device and dtype, the way `generate.py` leaves them.

    Not optional: `get_window_topk_idxs` builds its indices with a bare
    `torch.arange` (model.py:264), so without `set_default_device` every
    `sparse_attn` call is handed CPU indices and raises. Scoped and restored,
    because this is process-global state and the rest of the suite shares it.

    Those helpers are `lru_cache`d, so their entries are dropped on the way out
    -- a cached CUDA index tensor handed to a later CPU caller is the same bug
    with the devices swapped.
    """
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)          # generate.py:77
    try:
        # `Transformer.forward` is `@torch.inference_mode()`, and the reference
        # leans on it: `fp4_act_quant` writes in place over the view
        # `fast_hadamard_transform` returns, which autograd forbids. Run a bare
        # module the way the reference runs a whole one.
        with torch.inference_mode():
            yield
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
        reference.get_window_topk_idxs.cache_clear()
        reference.get_compress_topk_idxs.cache_clear()


def reference_args(reference, config, max_seq_len: int = MAX_SEQ_LEN,
                   max_batch_size: int = MAX_BATCH_SIZE):
    """Build the reference's `ModelArgs` and arm the globals it normally sets."""
    args = reference.ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size,
                               **model_args_kwargs(config))
    apply_reference_globals(reference, args)
    return args


def build_attention(reference, args, layer_id: int, ckpt: Path = None):
    with torch.device("cuda"):
        layer = reference.Attention(layer_id, args)
    layer.load_state_dict(load_module_weights(f"layers.{layer_id}.attn.", ckpt))
    return layer.eval()


def build_moe(reference, args, layer_id: int, ckpt: Path = None):
    """One whole MoE layer: `n_routed_experts` routed + 1 shared + the gate.

    The full expert set, not just the routed top-k: `MoE.forward` bincounts
    over every expert index and indexes `self.experts[i]`, so a partial layer
    would change the code path under test rather than the weights in it.
    """
    with torch.device("cuda"):
        ffn = reference.MoE(layer_id, args)
    ffn.load_state_dict(load_module_weights(f"layers.{layer_id}.ffn.", ckpt))
    return ffn.eval()


def seeded_token_ids(*shape: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
    """Token ids for the hash-routed layers, which index `tid2eid` with them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, vocab_size, shape, generator=generator,
                         device="cuda", dtype=torch.long)


def seeded_input(*shape: int, seed: int = 0, dtype=torch.bfloat16) -> torch.Tensor:
    """A fixed-seed activation, so a golden regenerates bit-for-bit."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)


def have_hadamard_extension() -> bool:
    """Is DeepSeek's own `fast_hadamard_transform` importable here?"""
    import importlib.util

    return importlib.util.find_spec("fast_hadamard_transform") is not None


def torch_hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    """The Sylvester Hadamard transform in plain torch, so the DeepSeek path
    never hard-requires a compiled per-model extension (CLAUDE.md §5).

    Both call sites rotate `index_head_dim`-wide vectors, a power of two, so
    the recursion is exact.
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


def hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    """DeepSeek's extension when it is installed, the torch transform when it
    is not. Imported lazily: a config-only import path must not need nvcc."""
    if have_hadamard_extension():
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=scale)
    return torch_hadamard(x, scale=scale)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """`model.py:253`, with the import routed through `hadamard`."""
    assert x.dtype == torch.bfloat16
    return hadamard(x, scale=x.size(-1) ** -0.5)


def model_args_kwargs(config) -> dict:
    """`DeepseekV4Config` -> kwargs for the reference's `ModelArgs`.

    One direction only (A5.1 forbids the reverse). `max_seq_len` and
    `max_batch_size` are deliberately NOT derived: `Attention.__init__` sizes
    `kv_cache` and the RoPE table off `max_seq_len`, and the shipped
    1,048,576 would allocate for a million positions to record a 512-token
    golden. The caller passes a small one.
    """
    rope = config.rope_parameters or {}
    quant = config.checkpoint_quantization_config or {}
    return {
        "vocab_size": config.vocab_size,
        "dim": config.hidden_size,
        "moe_inter_dim": config.moe_intermediate_size,
        "n_layers": config.num_hidden_layers,
        "n_hash_layers": config.num_hash_layers,
        "n_mtp_layers": config.num_nextn_predict_layers,
        "n_heads": config.num_attention_heads,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "n_activated_experts": config.num_experts_per_tok,
        "score_func": config.scoring_func,
        "route_scale": config.routed_scaling_factor,
        "swiglu_limit": config.swiglu_limit,
        "expert_dtype": config.expert_dtype,
        "q_lora_rank": config.q_lora_rank,
        "head_dim": config.head_dim,
        "rope_head_dim": config.qk_rope_head_dim,
        "o_groups": config.o_groups,
        "o_lora_rank": config.o_lora_rank,
        "window_size": config.sliding_window,
        "norm_eps": config.rms_norm_eps,
        "compress_ratios": config.compress_ratios,
        "compress_rope_theta": config.compress_rope_theta,
        # transformers 5.5.4 folds `rope_theta` INTO `rope_parameters` and
        # renames `original_max_position_embeddings` (modeling_rope_utils.py:706).
        "original_seq_len": rope["original_max_position_embeddings"],
        "rope_theta": rope.get("rope_theta", config.rope_theta),
        "rope_factor": rope["factor"],
        "beta_fast": rope["beta_fast"],
        "beta_slow": rope["beta_slow"],
        "index_n_heads": config.index_n_heads,
        "index_head_dim": config.index_head_dim,
        "index_topk": config.index_topk,
        "hc_mult": config.hc_mult,
        "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
        "hc_eps": config.hc_eps,
        # A5.1 renamed `quantization_config` so no FP8 quantizer attaches; the
        # block survives precisely because it is the only source for these two.
        "dtype": "fp8" if quant["fmt"] == "e4m3" else "bf16",
        "scale_fmt": quant["scale_fmt"],
        "dspark_block_size": config.dspark_block_size,
        "dspark_noise_token_id": config.dspark_noise_token_id,
        "dspark_target_layer_ids": config.dspark_target_layer_ids,
        "dspark_markov_rank": config.dspark_markov_rank,
    }


# ---------------------------------------------------------------- the recipes
#
# Each recipe builds ONE reference module with real weights, runs it on a
# fixed-seed input, and returns everything a later sub-plan needs to replay it:
# the inputs AND the outputs. Shapes are kept small on purpose -- these are
# committed fixtures, and correctness at a real sequence length is A5.5's
# step-0 gate, not something a repo should carry a 16 MB tensor for.

GOLDEN_SEQ_LEN = 8          # attention
GOLDEN_TOKENS = 4           # MoE / gate / block
GOLDEN_ROWS = 4             # bare Linear and Expert
CHUNK_LEN = 512             # Vaayu's prefill chunk -- the row-43-class trap

# Layers chosen by the axis that actually varies for attention: 21 layers carry
# an Indexer (compress_ratio == 4), 20 do not (== 128). Derived below, never
# hardcoded, so this still picks correctly if the ratios change.
def _layer_with_ratio(args, ratio: int) -> int:
    return next(i for i, r in enumerate(args.compress_ratios[:args.n_layers]) if r == ratio)


def build_expert(reference, args, prefix: str, ckpt: Path = None):
    """One `Expert`. `dtype` decides the whole weight format: routed experts
    are packed fp4, the shared expert takes `default_dtype` (fp8 here)."""
    routed = ".experts." in prefix
    dtype = torch.float4_e2m1fn_x2 if (routed and args.expert_dtype == "fp4") else None
    with torch.device("cuda"):
        expert = reference.Expert(args.dim, args.moe_inter_dim, dtype=dtype,
                                  swiglu_limit=args.swiglu_limit)
    expert.load_state_dict(load_module_weights(prefix, ckpt))
    return expert.eval()


def build_gate(reference, args, layer_id: int, ckpt: Path = None):
    with torch.device("cuda"):
        gate = reference.Gate(layer_id, args)
    gate.load_state_dict(load_module_weights(f"layers.{layer_id}.ffn.gate.", ckpt))
    return gate.eval()


def build_block(reference, args, layer_id: int, ckpt: Path = None):
    with torch.device("cuda"):
        block = reference.Block(layer_id, args)
    block.load_state_dict(load_module_weights(f"layers.{layer_id}.", ckpt))
    return block.eval()


def _recipe_linear_fp4(reference, args):
    layer_id = _layer_with_ratio(args, 128)
    expert = build_expert(reference, args, f"layers.{layer_id}.ffn.experts.0.")
    x = seeded_input(GOLDEN_ROWS, args.dim)
    return {"x": x, "out": expert.w1(x)}, None


def _recipe_linear_fp8(reference, args):
    layer_id = _layer_with_ratio(args, 128)
    shared = build_expert(reference, args, f"layers.{layer_id}.ffn.shared_experts.")
    x = seeded_input(GOLDEN_ROWS, args.dim)
    return {"x": x, "out": shared.w1(x)}, None


def _recipe_linear_bf16(reference, args):
    """`wo_a` is the load-time-dequant case: F8_E4M3 on disk, bf16 in the
    module, used raw in an einsum."""
    layer = build_attention(reference, args, _layer_with_ratio(args, 128))
    x = seeded_input(GOLDEN_ROWS, args.n_heads * args.head_dim // args.o_groups)
    return {"x": x, "out": layer.wo_a(x)}, None


def _recipe_expert(reference, args):
    layer_id = _layer_with_ratio(args, 128)
    expert = build_expert(reference, args, f"layers.{layer_id}.ffn.experts.0.")
    x = seeded_input(GOLDEN_ROWS, args.dim)
    weights = seeded_input(GOLDEN_ROWS, 1, seed=1)
    return {"x": x, "weights": weights, "out": expert(x, weights)}, None


def _gate_golden(reference, args, layer_id):
    gate = build_gate(reference, args, layer_id)
    x = seeded_input(GOLDEN_TOKENS, args.dim)
    input_ids = seeded_token_ids(GOLDEN_TOKENS, vocab_size=args.vocab_size)
    weights, indices = gate(x, input_ids)
    return {"x": x, "input_ids": input_ids, "weights": weights, "indices": indices}, None


def _recipe_gate_hash(reference, args):
    return _gate_golden(reference, args, layer_id=0)


def _recipe_gate_score(reference, args):
    return _gate_golden(reference, args, layer_id=args.n_hash_layers)


def _moe_golden(reference, args, layer_id):
    ffn = build_moe(reference, args, layer_id)
    x = seeded_input(1, GOLDEN_TOKENS, args.dim)
    input_ids = seeded_token_ids(1, GOLDEN_TOKENS, vocab_size=args.vocab_size)
    return {"x": x, "input_ids": input_ids, "out": ffn(x, input_ids)}, None


def _recipe_moe_hash(reference, args):
    return _moe_golden(reference, args, layer_id=0)


def _recipe_moe_score(reference, args):
    return _moe_golden(reference, args, layer_id=args.n_hash_layers)


def _attention_golden(reference, args, ratio: int, phase: str):
    """`prefill` is the `start_pos == 0` branch; `decode` and `chunked` are the
    ring-buffer branch, which `model.py:535` writes with `kv.squeeze(1)`."""
    layer = build_attention(reference, args, _layer_with_ratio(args, ratio))
    prefill = seeded_input(1, GOLDEN_SEQ_LEN, args.dim)
    out = layer(prefill, 0)
    if phase == "prefill":
        return {"x": prefill, "out": out}, None

    step = 1 if phase == "decode" else CHUNK_LEN
    x = seeded_input(1, step, args.dim, seed=1)
    try:
        return {"prefill_x": prefill, "x": x, "out": layer(x, GOLDEN_SEQ_LEN)}, None
    except Exception as failure:
        # THIS IS THE GOLDEN. The reference does not support a multi-token step
        # past position 0, and A5.5 needs that written down, not rediscovered.
        # The step input is not stored: it is `seeded_input(1, CHUNK_LEN, dim,
        # seed=1)`, reproducible from the recipe, and 3.2 MB of a fixture that
        # has no output to compare it against.
        return ({"prefill_x": prefill},
                f"seqlen={step} at start_pos={GOLDEN_SEQ_LEN} raised "
                f"{type(failure).__name__}: {failure}")


RECIPES = {
    "linear_fp4": _recipe_linear_fp4,
    "linear_fp8": _recipe_linear_fp8,
    "linear_bf16": _recipe_linear_bf16,
    "expert": _recipe_expert,
    "gate_hash": _recipe_gate_hash,
    "gate_score": _recipe_gate_score,
    "moe_hash": _recipe_moe_hash,
    "moe_score": _recipe_moe_score,
    **{f"attn_{phase}_r{ratio}": (lambda r, a, _r=ratio, _p=phase:
                                  _attention_golden(r, a, _r, _p))
       for ratio in (128, 4) for phase in ("prefill", "decode", "chunked")},
}


def _recipe_block(reference, args):
    """Hyper-connections end to end: `Block.forward` takes `[b, s, hc_mult, d]`,
    not `[b, s, d]` -- the residual stream is `hc_mult` copies wide."""
    layer_id = _layer_with_ratio(args, 128)
    block = build_block(reference, args, layer_id)
    x = seeded_input(1, GOLDEN_TOKENS, args.hc_mult, args.dim)
    input_ids = seeded_token_ids(1, GOLDEN_TOKENS, vocab_size=args.vocab_size)
    return {"x": x, "input_ids": input_ids, "out": block(x, 0, input_ids)}, None


RECIPES["block"] = _recipe_block


def _recipe_hc_head(reference, args):
    """`Block.hc_head` (`model.py:709`), the reduction `Transformer.forward`
    applies once, after the last block.

    The `Block` is built on META on purpose: `hc_head` reads no parameter of
    its own -- only `self.norm_eps`, `self.hc_eps` and, through
    `x.view(shape)`, `hc_mult` -- so the instance is there to supply three
    scalars. Verified 2026-09-24: `reference.Block(0, args)` constructs on meta
    and carries them. That makes this golden cost the 262 KB of `hc_head_*`
    instead of a 3.3 GiB layer.

    The three parameters are stored IN the fixture, so the test that reads it
    needs a GPU but not the 157 G checkpoint.
    """
    with torch.device("meta"):
        block = reference.Block(0, args)
    weights = load_module_weights("hc_head_")
    fn = weights["fn"].cuda()
    scale = weights["scale"].cuda()
    base = weights["base"].cuda()
    x = seeded_input(1, GOLDEN_TOKENS, args.hc_mult, args.dim)
    return {"x": x, "hc_head_fn": fn, "hc_head_scale": scale,
            "hc_head_base": base,
            "out": block.hc_head(x, fn, scale, base)}, None


RECIPES["hc_head"] = _recipe_hc_head


def _recipe_head(reference, args):
    """`RMSNorm` + `ParallelHead` (`model.py:719`) -- the fp32 logits.

    Only `x` and `out` are stored: `head.weight` is 0.986 GiB and does not
    belong in a committed fixture, so the test that reads this reads the two
    weights off the checkpoint itself.

    `full_logits=True` so the golden carries every position. The default
    (`model.py:733`) keeps only the last, which is the reference's own
    `logits_to_keep=1`.
    """
    with torch.device("cuda"):
        norm = reference.RMSNorm(args.dim, args.norm_eps)
        head = reference.ParallelHead(args.vocab_size, args.dim,
                                      args.norm_eps, args.hc_eps)
    norm.load_state_dict(load_module_weights("norm."))
    head.load_state_dict(load_module_weights("head."))
    x = seeded_input(1, GOLDEN_TOKENS, args.dim)
    return {"x": x, "out": head(norm(x), full_logits=True)}, None


RECIPES["head"] = _recipe_head


def _shipped_config():
    from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "deepseek_v4_config.json"
    return DeepseekV4Config(**json.loads(fixture.read_text()))


def record_golden(name: str, root: Path = GOLDENS) -> dict:
    """Run one recipe against the real checkpoint and write its fixture."""
    reference = reference_modules()
    args = reference_args(reference, _shipped_config())
    with reference_device(reference):
        tensors, note = RECIPES[name](reference, args)
        tensors = {key: value.detach().cpu() for key, value in tensors.items()}
    torch.cuda.empty_cache()
    save_golden(name, tensors, root=root, note=note)
    return tensors


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list(RECIPES)
    for golden in names:
        recorded = record_golden(golden)
        size = save_golden.__name__ and (GOLDENS / f"{golden}.npz").stat().st_size
        print(f"{golden:20s} {size/1024:8.1f} KiB  "
              + "  ".join(f"{k}{tuple(v.shape)}" for k, v in recorded.items())
              + (f"  NOTE: {load_note(golden)}" if load_note(golden) else ""), flush=True)
