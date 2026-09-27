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


def make_routed_experts_forward(*, experts_mod, prior_forward, experts_path: str,
                                layer_idx: int, layer_name: str, shard_file: str,
                                offset_index, expert_cache, vaayu_ref, buffers: dict,
                                device, snapshot: dict, active: list):
    """The experts SUBMODULE's forward under routed delivery.

    Inert -- `prior_forward` untouched -- unless `active[0]` is set. The block
    forward sets it for exactly the duration of a routed visit, so a whole-layer
    visit (the private A/B toggle) and any call outside a visit see the stock module.

    Per routed call: read the kernel's own routed set off the call; fetch those
    experts through the shared per-expert helpers (decode: RAM-arena hits, pool
    reads into pinned slabs, cached; prefill: staging slabs, not cached); copy each
    one's rows into the shared buffer; run the quantizer's conversions over the
    buffer AFTER the upload (so a layout conversion that copies rather than views
    still sees the fresh rows); call the stock forward; release.

    `snapshot` holds `meta_params` / `base_attrs`, taken by the factory AFTER this
    wrapper was installed, so a release restores the wrapper with everything else.
    """
    from .moe_patch import (_dispatch_expert_misses, _iter_expert_loads,
                            _populate_fused_experts, _release_fused_experts,
                            _release_prefill_staged, _settle_miss_upload)
    num_experts = int(experts_mod.num_experts)
    as_raw = {f"{experts_path}.{leaf}": buf for leaf, buf in buffers.items()}

    def forward(*args, **kwargs):
        if not active[0]:
            return prior_forward(*args, **kwargs)
        used = routed_experts_from_call(args, kwargs, num_experts)
        if used is None:
            raise RuntimeError(
                f"[Vaayu-MoE] {layer_name}: routed delivery is installed but this experts "
                f"call carries no routed set the adapter recognises (no `expt_hist` of "
                f"length {num_experts}). The load-time self-check passed on a different "
                f"call shape; running on would leave the kernel reading rows nobody "
                f"uploaded.")
        vaayu_ref._expert_history[layer_idx] = used
        expert_tensors, miss_futures, pinned_keys = _dispatch_expert_misses(
            used, layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache, vaayu_ref=vaayu_ref)
        uploaded: set = set()
        try:
            for eidx, raw in _iter_expert_loads(expert_tensors, miss_futures, vaayu_ref):
                for key, view in raw.items():
                    buffers[key.rsplit(".", 1)[-1]][eidx].copy_(view, non_blocking=True)
                # Every RAM-tier expert, hit or miss, owes this once its copies are
                # ISSUED: arm the slab's read event, unpin, return a staging slab.
                _settle_miss_upload(
                    eidx, raw, None, layer_idx=layer_idx, expert_cache=expert_cache,
                    vaayu_ref=vaayu_ref, pinned_keys=pinned_keys, vram=None, cm=None,
                    cm_last_call=False, cm_budget=0)
                uploaded.add(eidx)
            missing = sorted(set(used) - uploaded)
            if missing:
                raise RuntimeError(
                    f"[Vaayu-MoE] {layer_name}: routed experts {missing} were not uploaded "
                    f"(no offset, or the load returned nothing). Refusing to run: the "
                    f"kernel would read another layer's stale rows for them.")
            _populate_fused_experts(experts_mod, experts_path, as_raw, device, vaayu_ref)
            try:
                return prior_forward(*args, **kwargs)
            finally:
                _release_fused_experts(experts_mod, snapshot["meta_params"],
                                       snapshot["base_attrs"])
        finally:
            # Keys the loop never settled (an exception mid-visit). A leaked pin
            # blocks that key's eviction forever.
            for _pk in pinned_keys:
                expert_cache.unpin(*_pk)
            # Same close-then-sweep as the other forwards (S1-L1): a load still in
            # flight releases its own staging slab instead of registering one late.
            vaayu_ref._prefill_staging_closed = layer_idx
            _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == layer_idx)

    return forward
