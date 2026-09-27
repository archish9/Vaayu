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

from pathlib import Path

import torch
import torch.nn as nn


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


def self_check_routed_delivery(block_info, *, offset_index, checkpoint_path, vaayu_ref,
                               device, dtype, hidden_size: int, top_k: int,
                               seed: int = 0) -> tuple[bool, str]:
    """Prove, on ONE real layer, that this block's experts kernel reads only the
    routed experts' rows. Returns (passed, reason).

    Must run BEFORE patch_moe_blocks: it drives the block's own, still-unpatched
    forward. The router's parameters are swapped for fixed-seed random ones for the
    duration -- a streamed layer's router may still be on meta at load, and routing
    correctness is not what is checked -- and restored by identity afterwards.

    For M = 1 and M = max(2, E // (2 * top_k)) tokens: populate the real bytes, run,
    record the output and the routed set the adapter reads off the call; then
    overwrite every NON-routed expert's rows with random values, run again, and
    require the same routed set and a torch.equal output. Fails closed: an
    unrecognised call, a call that routes every expert (nothing left to corrupt),
    a difference, or any exception -> (False, reason).
    """
    from .moe_patch import _populate_fused_experts, _release_fused_experts
    from .persist.safetensor_model_persister import load_fused_layer_tensors

    moe = block_info.moe_module
    experts_mod = getattr(moe, block_info.experts_attr)
    router = getattr(moe, block_info.router_attr)
    experts_path = f"{block_info.module_path}.{block_info.experts_attr}"
    num_experts = int(experts_mod.num_experts)
    metas = offset_index.fused_layer_tensors[block_info.layer_name]
    shard_file = str(Path(checkpoint_path) / f"{block_info.layer_name}.safetensors")

    meta_params = dict(experts_mod._parameters)
    orig_attrs = dict(experts_mod.__dict__)
    router_params = dict(router._parameters)
    prior_forward = experts_mod.forward
    seen: dict = {}

    def capture(*args, **kwargs):
        seen["used"] = routed_experts_from_call(args, kwargs, num_experts)
        return prior_forward(*args, **kwargs)

    g = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=device).manual_seed(seed)
    try:
        for name, p in router_params.items():
            if p is not None:
                router._parameters[name] = nn.Parameter(
                    torch.randn(tuple(p.shape), generator=g).to(device=device, dtype=p.dtype),
                    requires_grad=False)
        experts_mod.forward = capture
        run_attrs = dict(experts_mod.__dict__)      # keeps `capture` across releases

        def run(layer, hidden):
            seen.pop("used", None)
            _populate_fused_experts(experts_mod, experts_path, layer, device, vaayu_ref)
            try:
                with torch.no_grad():
                    out = moe(hidden)
            finally:
                _release_fused_experts(experts_mod, meta_params, run_attrs)
            out = out[0] if isinstance(out, tuple) else out
            return out.detach().clone(), seen.get("used")

        m_multi = max(2, num_experts // (2 * max(1, int(top_k))))
        for m in (1, m_multi):
            layer = {k: t.to(device)
                     for k, t in load_fused_layer_tensors(shard_file, metas).items()}
            hidden = torch.randn(1, m, int(hidden_size), generator=g).to(device=device,
                                                                         dtype=dtype)
            out_real, used = run(layer, hidden)
            if used is None:
                return False, (f"the experts call carries no routed set the adapter "
                               f"recognises (no `expt_hist` of length {num_experts})")
            if len(used) >= num_experts:
                return False, (f"vacuous at M={m}: all {num_experts} experts were routed, "
                               f"so no row was left to corrupt")
            routed = set(used)
            for e in range(num_experts):
                if e in routed:
                    continue
                for t in layer.values():
                    row = t[e]
                    # Full-range random BYTES through a uint8 view, for every dtype:
                    # a finite-but-wrong value (the old float/int branches) only
                    # proves the kernel is insensitive to magnitude, not that it
                    # never reads the row. An unread row can hold NaN/Inf bf16
                    # patterns or an E8M0 scale of 0xFF in production, and only a
                    # byte-level corruption exercises that.
                    rb = row.view(torch.uint8)
                    rb.copy_(torch.randint(0, 256, rb.shape, dtype=torch.uint8,
                                           generator=g_dev, device=device))
            out_bad, used_bad = run(layer, hidden)
            if used_bad != used:
                return False, f"routing changed between the two arms at M={m}"
            if not torch.equal(out_real, out_bad):
                diff = (out_real.float() - out_bad.float()).abs().max().item()
                return False, (f"non-routed expert rows change the output at M={m} "
                               f"(max|diff| {diff:.3e}): the kernel reads rows it did "
                               f"not route to")
        return True, (f"non-routed expert rows do not change the output at M=1 "
                     f"and M={m_multi}")
    except Exception as e:  # noqa: BLE001 -- fails closed; the reason is printed
        return False, f"self-check raised {type(e).__name__}: {e}"
    finally:
        router._parameters.clear()
        router._parameters.update(router_params)
        _release_fused_experts(experts_mod, meta_params, orig_attrs)


def run_routed_self_checks(vaayu_ref, blocks, offset_index, *, checkpoint_path, device,
                           dtype, hidden_size: int, top_k: int) -> dict:
    """One self-check per distinct fused-3D geometry among the whole-layer blocks;
    returns {geometry_key: passed}. Prints one line per geometry either way: a FAIL
    is a WARNING (CLAUDE.md §5 -- the model still runs, on the slow path)."""
    from .moe_patch import dispatches_whole_layer
    verdicts: dict = {}
    for b in blocks:
        if not dispatches_whole_layer(b, offset_index):
            continue
        key = fused3d_geometry_key(offset_index.fused_layer_tensors[b.layer_name])
        if key in verdicts:
            continue
        ok, why = self_check_routed_delivery(
            b, offset_index=offset_index, checkpoint_path=checkpoint_path,
            vaayu_ref=vaayu_ref, device=device, dtype=dtype, hidden_size=hidden_size,
            top_k=top_k)
        verdicts[key] = ok
        if ok:
            print(f"[Vaayu-MoE] Routed fused-3D delivery ON for {b.layer_name}'s geometry "
                  f"({why}): only the routed experts' rows move per layer visit.")
        else:
            print(f"[Vaayu-MoE] WARNING: routed fused-3D delivery OFF for {b.layer_name}'s "
                  f"geometry -- {why}. This model keeps whole-layer delivery: every token "
                  f"moves every expert (slow, but correct).")
    return verdicts
