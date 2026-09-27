"""Routed delivery for fused-3D expert checkpoints (S8, 2026-09-27).

A fused-3D checkpoint stores every expert of a layer in ONE tensor per projection
(`...mlp.experts.gate_up_proj_blocks`, expert axis leading), and its experts module
hands the whole tensor to a kernel that routes inside it. The whole-layer forward
streams that entire layer on every visit: on gpt-oss-20b that is 10.17 GB per token,
90 % of a 4.54 s token (docs/2026-09-27-S8-gptoss-attribution-RESULTS.md).

The kernel, though, only READS the rows of the experts it routed to -- measured:
random bytes in every other expert's rows leave the output bit-identical. So the
layer can be delivered sparsely with no new kernel: learn the routed set from the
routing the model itself computed, copy only those experts' rows (through the shared
per-expert RAM arena) into a persistent [num_experts, ...] device buffer, and run the
module's own forward over it.

Everything here is by capability, never by architecture name (CLAUDE.md §5): the
adapter recognises a routing object carrying a per-expert token histogram, and a
load-time self-check proves the kernel property on one real layer before a model is
switched to routed delivery.
"""
from __future__ import annotations

import torch


def routed_experts_from_call(args, kwargs, num_experts: int) -> list[int] | None:
    """The experts an experts-module call will actually process, or None.

    Recognises ONE argument shape: an object carrying `expt_hist`, a 1-D integer
    tensor of length `num_experts` -- the per-expert token histogram of the model's
    own routing (triton_kernels' RoutingData). The kernel processes exactly the
    experts with a non-zero count, so this is the kernel's own set, not a
    re-derivation: a separate top-k could break a tie differently and leave the
    kernel reading a row nobody uploaded.

    One device-to-host sync (`.tolist()`). Returns expert ids in ascending order.
    None means "not recognised"; the caller decides whether that is a fallback or
    an error.
    """
    for value in (*args, *kwargs.values()):
        hist = getattr(value, "expt_hist", None)
        if (isinstance(hist, torch.Tensor) and hist.dim() == 1
                and hist.numel() == int(num_experts)
                and not hist.dtype.is_floating_point and hist.dtype != torch.bool):
            return torch.nonzero(hist).flatten().tolist()
    return None


def fused3d_geometry_key(metas: dict) -> tuple:
    """Hashable identity of one fused-3D layer's tensor set: (leaf, shape, dtype)
    per tensor, sorted, layer name dropped. Layers with the same key share one
    device buffer and one self-check verdict."""
    return tuple(sorted((key.rsplit(".", 1)[-1], tuple(m.shape), str(m.dtype))
                        for key, m in metas.items()))


def allocate_expert_buffers(metas: dict, device) -> dict[str, torch.Tensor]:
    """One persistent [num_experts, ...] tensor per fused projection, keyed by leaf
    name, in the checkpoint's own dtype. Contents are undefined until a visit copies
    the routed experts' rows in; rows no visit wrote are never read (the load-time
    self-check proves that for the kernel in use)."""
    from .persist.safetensor_model_persister import _resolve_dtype
    return {key.rsplit(".", 1)[-1]:
                torch.empty(tuple(m.shape),
                            dtype=_resolve_dtype(m.dtype, key, "<fused-3D expert buffer>"),
                            device=device)
            for key, m in metas.items()}


def routed_delivery_enabled(block_info, offset_index, vaayu_ref) -> bool:
    """True iff this block takes routed delivery: it dispatches whole-layer AND the
    load-time self-check passed for its geometry.

    THE ONE PLACE this condition is written. The engine's arena guard and the
    forward factory both call it, so the RAM the arena pins and the forward that
    reads it cannot drift apart (the 47-m5 lesson).
    """
    from .moe_patch import dispatches_whole_layer
    if not dispatches_whole_layer(block_info, offset_index):
        return False
    verdicts = getattr(vaayu_ref, "_routed_fused3d_ok", None) or {}
    metas = offset_index.fused_layer_tensors[block_info.layer_name]
    return bool(verdicts.get(fused3d_geometry_key(metas)))
