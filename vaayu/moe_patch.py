"""
MoE Expert-Aware Streaming: monkey-patch utilities for Vaayu.

detect_moe_blocks() — generic detection of MoE blocks in any architecture.
patch_moe_blocks()  — replaces each MoE block's forward() with a sparse version.

Design rationale: see docs/superpowers/specs/2026-06-14-moe-expert-aware-streaming-design.md
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, wait as _futures_wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .persist.expert_cache import ExpertRAMCache
    from .persist.expert_offset_index import ExpertOffsetIndex

# Maps checkpoint key substrings to logical roles used in CPU FFN path.
# To support a new MoE architecture: add 3 lines here, nothing else changes.
_EXPERT_WEIGHT_ROLES: list[tuple[str, str]] = [
    # Mixtral format
    (".w1.", "gate"),
    (".w2.", "down"),
    (".w3.", "up"),
    # DeepSeek, Qwen, Kimi, LLaMA-MoE
    (".gate_proj.", "gate"),
    (".down_proj.", "down"),
    (".up_proj.",   "up"),
]


def unique_expert_indices(expert_indices_cpu) -> list[int]:
    """Fix #5: unique expert ids across ALL routed positions (flattened B*S x top_k),
    so each expert is read/loaded at most once per layer per forward pass, even
    when speculative decoding verifies multiple draft positions at once."""
    return sorted(int(x) for x in expert_indices_cpu.flatten().unique().tolist())


_FP16_MAX = 65504.0  # torch.finfo(torch.float16).max — kernel-contract constant


def _trace_expert(vaayu_ref, layer_idx, eidx, expert_mod, x_in, expert_out) -> None:
    """VAAYU_TRACE_EXPERT=<layer_idx> diagnostic: per-expert magnitudes for ONE
    layer, prefill + first decode step only. Scoped to a single layer because
    the checks force a device sync per expert (~200/layer) — far too expensive
    to run model-wide, fine for one.

    Recomputes w1's PRE-ACTIVATION output on purpose. EXL3 kernels run in fp16
    (`ExLlamaV3Linear.forward` casts bf16 -> fp16), whose max is 65504 while
    bf16 reaches ~3e38. An overflow there gives -inf, and silu(-inf) =
    -inf * sigmoid(-inf) = -inf * 0 = NaN — which would present as nan=1/inf=0
    downstream, exactly the signature seen at the last MoE layer. Reporting the
    pre-activation absmax against _FP16_MAX confirms or refutes that directly
    instead of inferring it.
    """
    target = getattr(vaayu_ref, "_trace_expert_layer", None)
    if target is None or target != layer_idx:
        return
    if getattr(vaayu_ref, "_token_count", 0) > 1:
        return
    bad_out = bool(torch.isnan(expert_out).any().item() or
                   torch.isinf(expert_out).any().item())
    pre, pre_err = None, ""
    try:
        # No manual cast: ExLlamaV3Linear.forward does its own bf16->fp16
        # conversion, so replicate the real call exactly rather than a variant
        # of it. Surface any failure instead of silently reporting nan, which
        # would be indistinguishable from a genuine overflow.
        w1 = getattr(expert_mod, "w1", None) or getattr(expert_mod, "gate_proj", None)
        if w1 is not None:
            pre = w1(x_in)
    except Exception as e:
        pre_err = f" pre_err={type(e).__name__}:{e}"
    pre_absmax = float(pre.abs().max().item()) if pre is not None else float("nan")
    pre_bad = bool(pre is not None and (torch.isnan(pre).any().item()
                                        or torch.isinf(pre).any().item()))
    # Only print offenders plus a periodic sample, so the log stays readable.
    if not (bad_out or pre_bad or pre_absmax > 0.1 * _FP16_MAX or eidx % 64 == 0):
        return
    print(f"[VAAYU-EXPERT] tok={getattr(vaayu_ref, '_token_count', 0)} "
          f"layer={layer_idx} e={eidx:>3} rows={x_in.shape[0]} "
          f"x_absmax={x_in.abs().max().item():.4e} "
          f"w1pre_absmax={pre_absmax:.4e} "
          f"(fp16_max={_FP16_MAX:.0f}, frac={pre_absmax / _FP16_MAX:.3f}) "
          f"pre_bad={int(pre_bad)} out_bad={int(bad_out)} "
          f"out_absmax={expert_out.abs().max().item():.4e}{pre_err}", flush=True)


def _trace_moe_block(vaayu_ref, layer_idx, style, block_in, block_out,
                     routing_weights, unique_experts) -> None:
    """VAAYU_TRACE_ACT=1 diagnostic: one line per MoE block, prefill + first
    decode step only. No-op (one getattr) when the flag is off.

    Reports the expert output's magnitude RELATIVE to the block input, because
    the transformer adds this as a residual: a ratio near 0 means the experts
    contributed nothing (weights/QMatrix wrong), while a very large ratio means
    they are swamping the residual. Routing-weight min/max/sum localizes a
    routing defect separately from an expert-compute defect — sum should be
    ~top_k per token when each token's weights are a convex mix.
    """
    if not getattr(vaayu_ref, "_trace_act", False):
        return
    if getattr(vaayu_ref, "_token_count", 0) > 1:
        return
    from .vaayu_base import act_stats
    w = routing_weights.detach().float()
    n_tok = max(1, w.numel() // max(1, w.shape[-1]))
    print(f"[VAAYU-MOE-ACT] tok={getattr(vaayu_ref, '_token_count', 0)} "
          f"layer={layer_idx:>3} style={style} "
          f"in[{act_stats(block_in)}] out[{act_stats(block_out)}] "
          f"w_min={w.min().item():.4f} w_max={w.max().item():.4f} "
          f"w_sum_per_tok={w.sum().item() / n_tok:.4f} "
          f"n_experts={len(unique_experts)}", flush=True)


def route_experts(moe_module, router_logits, top_k: int):
    """Combination weights + expert indices for one MoE layer, using the
    MODEL's own scoring function (CLAUDE.md §5: detection over assumption).

    Delegates to `moe_module.route_tokens_to_experts(router_logits)` when the
    model defines it — the contract transformers standardizes across 25+ MoE
    architectures (DeepSeek v2/v3, GLM-MoE, dots1, dbrx, jamba, MiniMax-M2...).
    Those models do NOT score with softmax. MiniMax-M2 scores with sigmoid and
    adds an `e_score_correction_bias` that steers SELECTION
    (`scoring_func: "sigmoid"`, `use_routing_bias: true` in its config), then
    takes weights from the UNBIASED sigmoid. Normalizing softmax over the top-k
    instead yields a far peakier mix — for logits [3,2,1] the top expert gets
    ~0.67 rather than ~0.37 — wrong on every layer, which is what made MiniMax
    emit incoherent text while loading, patching and streaming perfectly.

    Falls back to plain softmax-topk when no contract method exists, which is
    what Qwen3-MoE and Mixtral genuinely do — byte-identical to the previous
    hardcoded expression for them, so the actively-tested path is unchanged.

    Return order is NOT stable across implementors (DeepSeek/GLM/dots1/MiniMax
    return `(indices, weights)`; dbrx returns `(weights, indices)`), so the two
    are told apart by dtype — indices are integral, weights floating point —
    never by position.

    `router_logits` may be (N, E) or (B, S, E). The contract methods only accept
    2-D (they `gather(1, ...)`), so leading dims are flattened and restored here.
    Returns (weights, indices), each shaped like router_logits with E -> top_k.
    """
    fn = getattr(moe_module, "route_tokens_to_experts", None)
    if not callable(fn):
        w, idx = torch.topk(router_logits.float().softmax(dim=-1), top_k, dim=-1)
        return w / w.sum(dim=-1, keepdim=True), idx

    lead_shape = router_logits.shape[:-1]
    flat = router_logits.reshape(-1, router_logits.shape[-1])
    first, second = fn(flat)
    idx, w = (first, second) if not first.is_floating_point() else (second, first)
    return w.reshape(*lead_shape, -1), idx.reshape(*lead_shape, -1)


def cache_aware_topk(logits, resident_mask, top_k: int, base_bias: float):
    """Bias top-k SELECTION toward resident experts; take combination WEIGHTS
    from the ORIGINAL logits (selection changes, weighting stays faithful —
    advanced-opt #4). base_bias == 0.0 or resident_mask is None reduces exactly
    to plain softmax-topk (feature OFF). The bias is scaled by the per-row logit
    std so it never overwhelms a confident router (large gaps => bias is a small
    fraction of the gap). resident_mask: [num_experts], 1.0 for cached else 0.0."""
    logits = logits.float()
    if base_bias and resident_mask is not None:
        std = logits.std(dim=-1, keepdim=True)
        biased = logits + resident_mask.unsqueeze(0) * base_bias * std
    else:
        biased = logits
    _, sel = torch.topk(biased, top_k, dim=-1)
    orig = logits.softmax(dim=-1)
    w = torch.gather(orig, -1, sel)
    w = w / w.sum(dim=-1, keepdim=True)
    return w, sel


def _warn_cache_bias_inert(vaayu_ref, reason: str) -> None:
    """Warn once per model when VAAYU_CACHE_BIAS is set but cannot be applied.

    cache_aware_topk only runs in the plain-softmax branch of the EXL3 fused
    forward. Measured 2026-08-02 (Plan G): no current MoE model reaches it —
    transformers 5.x routers do their own top-k, and MiniMax-family models own
    their scoring function. Silently ignoring the setting is the degradation
    CLAUDE.md §5 forbids, so say so. Call sites are per layer per token, hence
    the once-per-instance latch."""
    if not getattr(vaayu_ref, "_cache_bias", 0.0):
        return
    if getattr(vaayu_ref, "_cache_bias_warned", False):
        return
    vaayu_ref._cache_bias_warned = True
    _log.warning(
        f"[Vaayu-MoE] VAAYU_CACHE_BIAS={vaayu_ref._cache_bias} has NO EFFECT on this "
        f"model: {reason}, so cache-aware expert routing never runs. Routing is "
        f"unchanged; unset the variable or expect no bias.")


def _build_resident_mask(vaayu_ref, expert_cache, layer_idx: int,
                         num_experts: int, device):
    """[num_experts] mask of experts resident in RAM ∪ VRAM for `layer_idx`, or
    None when cache-aware routing is off (bias 0 or prefill) or nothing is
    resident — so the caller pays nothing in the default (feature-off) case.
    One lock acquisition per cache (resident_set), non-mutating."""
    bias = getattr(vaayu_ref, "_cache_bias", 0.0)
    if not bias or getattr(vaayu_ref, "_moe_prefill_pass", False):
        return None
    res = expert_cache.resident_set(layer_idx)
    _vr = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vr is not None:
        res = res | _vr.resident_set(layer_idx)
    if not res:
        return None
    mask = torch.zeros(num_experts, device=device)
    mask[torch.tensor(sorted(res), device=device)] = 1.0
    return mask


def compute_expert_dispatch(expert_indices_cpu: "torch.Tensor", num_experts: int):
    """Permutation-based MoE dispatch (CPU side, one pass per layer).

    expert_indices_cpu: (N, top_k) long tensor of routed expert ids.
    Returns (pair, starts, counts):
      pair   — (2, N*top_k) long tensor; row 0 = flat token index, row 1 = top-k
               slot; entries sorted by expert id (stable), so expert e's routed
               positions are the contiguous slice [starts[e] : starts[e]+counts[e]).
      starts — list[int] per expert; counts — list[int] per expert.
    Replaces the per-layer int64 one_hot mask + per-expert torch.where/H2D pair:
    the caller uploads `pair` to the GPU ONCE and slices per expert (views, free).
    """
    n, top_k = expert_indices_cpu.shape
    eids = expert_indices_cpu.reshape(-1)
    tok = torch.arange(n).repeat_interleave(top_k)
    kpos = torch.arange(top_k).repeat(n)
    order = torch.argsort(eids, stable=True)
    counts_t = torch.bincount(eids[order], minlength=num_experts)
    starts_t = counts_t.cumsum(0) - counts_t
    pair = torch.stack([tok[order], kpos[order]])
    return pair, starts_t.tolist(), counts_t.tolist()


def lookahead_accuracy_update(state: dict, predicted, layer_idx: int, actual: set) -> None:
    """Phase-A accounting for the lookahead routing predictor: when a stored
    prediction targets THIS layer, count how many actually-routed experts it
    covered. state = {"hits": int, "total": int}; accuracy = hits/total."""
    if predicted is None or predicted[0] != layer_idx:
        return
    state["hits"] += len(predicted[1] & actual)
    state["total"] += len(actual)


def predict_next_experts(next_block: "MoEBlockInfo", hidden_states,
                         top_k: int) -> "tuple[set, set]":
    """(predicted, to_stage) for the NEXT MoE layer, scored the way that layer
    itself scores (CLAUDE.md §5: detection over assumption).

    `predicted` is the whole predicted set — what the accuracy telemetry scores.
    `to_stage` is the prefetch ADMISSION set: it drops each routed row's
    LOWEST-weight pick. Measured 2026-09-03 on both models, the marginal misses
    removed per prefetch read is flat down the weight order and then falls off a
    cliff at the last rank (Qwen 0.607 -> 0.418, MiniMax 0.654 -> 0.312): the
    prediction routes the next layer on the CURRENT layer's hidden state, and
    the weakest pick is the first to fall out of the true top-k. Dropping it
    removes ~11 % of decode reads. Per ROW, not globally — batched decode routes
    N rows, and one global drop would leave N-1 of them staging a full top-k.
    See tests/test_lookahead_tail_drop.py for the measurement this encodes.

    Three cases, in the same order the on-path forwards resolve them:
      * a router that returns its own (logits, weights, indices) — believe it;
      * a model that owns its scoring function — `route_experts` delegates to
        `route_tokens_to_experts` (MiniMax: sigmoid + e_score_correction_bias,
        group routing). A hardcoded topk-on-logits is simply WRONG there;
      * everything else — plain softmax-topk, which is what Qwen3-MoE and
        Mixtral genuinely do. softmax is monotonic, so the predicted SET is
        identical to the topk-on-logits this used to compute for them.

    `hidden_states` may be (N, H) or (B, S, H): the fused forward passes its
    flat view, the module-list forward the 3-D states, and both the routers and
    `route_experts` accept either. This is the CURRENT layer's hidden states —
    the next layer's input is not known yet, and that approximation is the
    predictor's whole premise, not a defect introduced here."""
    router = getattr(next_block.moe_module, next_block.router_attr)
    res = router(hidden_states)
    if isinstance(res, tuple) and len(res) >= 3:
        # (logits, weights, indices) in some order — told apart by dtype, never
        # by position, for the reason route_experts documents.
        a, b = res[1], res[2]
        idx, w = (a, b) if not a.is_floating_point() else (b, a)
    else:
        logits = res[0] if isinstance(res, tuple) else res
        w, idx = route_experts(next_block.moe_module, logits, top_k)
    idx = idx.reshape(-1, idx.shape[-1])
    # One D2H for both sets: the narrowing is a gather on device, so admission
    # costs no extra sync on the per-layer decode path.
    keep = idx
    if idx.shape[-1] > 1:
        _, order = torch.topk(w.reshape(-1, w.shape[-1]).float(),
                              idx.shape[-1] - 1, dim=-1)
        keep = torch.gather(idx, -1, order)
    both = torch.cat((idx.reshape(-1), keep.reshape(-1))).cpu()
    n_all = idx.numel()
    predicted = set(int(x) for x in both[:n_all].unique().tolist())
    to_stage = set(int(x) for x in both[n_all:].unique().tolist())
    return predicted, to_stage


def _stage_predicted_expert(expert_cache, offset_index, checkpoint_path: Path,
                            next_layer_name: str, next_layer_idx: int, eidx: int) -> None:
    """Phase-B prefetch: load a predicted next-layer expert into the RAM cache
    ahead of its layer via the SAME zero-clone slab path decode's on-path misses
    use (CR-2 fix: no double-clone). prefetch=True so the prefetch_used telemetry
    measures usefulness. Never raises.

    Layout-agnostic: it reads bytes by (layer_name, expert id) through the
    offset index, which both expert layouts share, so the fused-EXL3 and
    module-list forwards stage identically."""
    from .persist.safetensor_model_persister import (
        load_expert_sparse, load_expert_into_slab)
    try:
        if expert_cache.has(next_layer_idx, eidx):
            return
        metas = offset_index.get_expert_offsets(next_layer_name, eidx)
        if metas is None:
            return
        _shard = str(checkpoint_path / f"{next_layer_name}.safetensors")
        if expert_cache.arena is None:
            # No arena (unit-test / CUDA-less) — legacy clone path.
            raw = load_expert_sparse(_shard, metas)
            expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
            return
        # Reservation: if anyone is already loading this expert, skip
        # entirely. Prefetch is opportunistic — re-reading bytes another
        # thread is already fetching costs real NVMe bandwidth and buys
        # nothing. Unlike decode, the stager never waits: it has nothing to
        # deliver and no caller blocked on it, and waiting here would put a
        # prefetch worker to sleep on the on-path reader.
        _claim = expert_cache.claim_load(next_layer_idx, eidx, "stager")
        if _claim is None:
            return
        try:
            expert_cache.mark_running(_claim)
            slab_idx = expert_cache.acquire_slab()
            if slab_idx is None:
                raw = load_expert_sparse(_shard, metas)
                expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
                return
            slab = expert_cache.arena.tensor(slab_idx)
            try:
                views = load_expert_into_slab(
                    _shard, metas, slab,
                    allow_direct=expert_cache.arena.pin, tensor_align=False)
            except Exception:
                expert_cache.arena.release(slab_idx)
                raise
            if not expert_cache.put_slab(next_layer_idx, eidx, views, slab_idx,
                                         prefetch=True):
                # Declined (no room). This thread takes no further read of
                # `views`, and holds no pin to hang a deferred release on,
                # so give the slab back now.
                expert_cache.arena.release(slab_idx)
        finally:
            expert_cache.finish_load(_claim)
    except Exception as e:
        _log.warning(
            f"Lookahead prefetch failed for layer {next_layer_idx} expert {eidx}: {e}"
        )


def _lookahead_step(vaayu_ref, expert_cache, offset_index, checkpoint_path: Path,
                    layer_idx: int, hidden_states, top_k: int, actual_experts) -> None:
    """Lookahead routing predictor (Task 12, decode only) — ONE implementation,
    called by BOTH MoE forwards.

    It lived only in `_make_fused_sparse_forward_exl3` until 2026-09-03, which
    made `VAAYU_LOOKAHEAD=1` a silent no-op on every module-list model (MiniMax)
    — CLAUDE.md §0's single-forward trap, the same one that left VAAYU_CACHE_BIAS
    doing nothing. Anything added here reaches both layouts by construction.

    Phase A (accounting) and Phase B (prefetch) are BOTH gated behind
    VAAYU_LOOKAHEAD=1 (CR-1): the extra next-layer router GEMM + the prediction's
    D2H sync are per-layer/per-token decode overhead that only Phase B consumes.
    Default off → steady-state decode pays nothing. Reads
    vaayu_ref._lookahead_enabled (captured ONCE at init, not a live os.environ
    read) so mutating the env var after engine construction can't bypass the
    hard-block guard in vaayu_base.py — see check_lookahead_flag_safe()."""
    if (getattr(vaayu_ref, "_moe_prefill_pass", False)
            or not getattr(vaayu_ref, "_lookahead_enabled", False)):
        return
    state = getattr(vaayu_ref, "_lookahead_state", None)
    if state is not None:
        lookahead_accuracy_update(
            state, getattr(vaayu_ref, "_lookahead_pred", None),
            layer_idx, set(actual_experts))
    blocks = getattr(vaayu_ref, "_moe_block_by_idx", None)
    next_block = blocks.get(layer_idx + 1) if isinstance(blocks, dict) else None
    if next_block is None:
        return
    try:
        pred_set, stage_set = predict_next_experts(next_block, hidden_states, top_k)
        vaayu_ref._lookahead_pred = (layer_idx + 1, pred_set)
        stager = getattr(vaayu_ref, "_stager_executor", None)
        if stager is None:
            return
        next_name = next_block.layer_name
        for _pe in stage_set:
            if not expert_cache.has(layer_idx + 1, _pe):
                stager.submit(_stage_predicted_expert, expert_cache, offset_index,
                              checkpoint_path, next_name, layer_idx + 1, _pe)
    except Exception as e:
        _log.warning(
            f"Lookahead router probe failed for layer {layer_idx + 1}: {e}"
        )
        vaayu_ref._lookahead_pred = None


def _classify_expert_tensors(
    raw: dict,
) -> "tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]":
    """Returns (w_gate, w_down, w_up) tensors from a raw per-expert tensor dict.

    Works with any MoE checkpoint key format listed in _EXPERT_WEIGHT_ROLES.
    Returns None for any role not found in raw.
    """
    roles: dict[str, "torch.Tensor | None"] = {"gate": None, "down": None, "up": None}
    for key, tensor in raw.items():
        for pattern, role in _EXPERT_WEIGHT_ROLES:
            if pattern in key and roles[role] is None:
                roles[role] = tensor
                break
    return roles["gate"], roles["down"], roles["up"]


_EXL3_COMPONENT_KEYS = {"trellis", "mcg", "suh", "svh", "su", "sv", "mul1"}

# compressed-tensors `pack-quantized` (GPTQ/AWQ int4 checkpoints): one expert
# linear is stored as bit-packed int32 codes + per-group scales, not as a dense
# weight. Row 30 / S2 — the HF quantizer binds none of these for a fused-experts
# MoE, so Vaayu dequantizes them itself on the compute device, which is also
# what keeps the DISK read at 4 bits per weight (the whole point of the model).
_PACKED_COMPONENT_KEYS = {
    "weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx",
}


def _group_packed_tensors_by_weight(raw: dict) -> dict[str, dict[str, "torch.Tensor"]]:
    """Groups compressed-tensors pack-quantized tensors by weight role.

    Returns {"gate": {"weight_packed": t, "weight_scale": t, ...}, "down": ..., "up": ...}.
    Empty per-role dicts mean the expert is NOT packed (dense or EXL3) — that is
    how the forward picks its path.
    """
    result: dict[str, dict[str, "torch.Tensor"]] = {"gate": {}, "down": {}, "up": {}}
    for key, tensor in raw.items():
        role = None
        for pattern, r in _EXPERT_WEIGHT_ROLES:
            if pattern in key:
                role = r
                break
        if role is None:
            continue
        component = key.rsplit(".", 1)[-1]
        if component in _PACKED_COMPONENT_KEYS:
            result[role][component] = tensor
    return result


def _shared_expert_output(moe_mod: nn.Module, flat_hidden: "torch.Tensor"):
    """The block's always-on shared-expert contribution, flat (tokens, hidden).

    None when the block has no shared expert — which is every model Vaayu has
    run on device so far (Qwen3-235B and MiniMax-M2.7 route only), and why this
    was missed: `patch_moe_blocks` replaces the whole MoE BLOCK forward, so a
    shared expert sitting beside the router is silently dropped.

    Two shapes exist in the wild and both are handled by name, the same way the
    router is found (`_ROUTER_NAMES`): Qwen2 / Qwen3-Next gate it with a sibling
    `shared_expert_gate` (sigmoid over one logit), DeepSeek adds `shared_experts`
    ungated. The gate is identified by its `_gate` suffix, not by its shape.
    """
    shared = gate = None
    for name, child in moe_mod.named_children():
        if "shared_expert" not in name:
            continue
        if name.endswith("_gate"):
            gate = child
        elif shared is None:
            shared = child
    if shared is None:
        return None
    out = shared(flat_hidden)
    if gate is not None:
        out = torch.sigmoid(gate(flat_hidden)) * out
    return out


def _expert_dense_weights(raw: dict, device: str, dtype: "torch.dtype", timing: dict = None):
    """(w_gate, w_down, w_up) for one expert, on `device`, ready for `F.linear`.

    Pack-quantized experts (compressed-tensors int4/int8 GPTQ-AWQ checkpoints)
    are dequantized HERE, on the compute device, so the disk read and the H2D
    copy stay at the checkpoint's bit width — dequantizing at persist time would
    multiply both by 4x and delete the reason for running a 4-bit model.
    Dense/EXL3-adjacent checkpoints keep the plain cast path unchanged.

    `timing`, when given, receives the split between the host-to-device copy
    ('h2d') and the dequant ('construct'). Those two happen back to back here,
    and the phase-4 gate is a ratio of exactly them — with no split it reads
    zero. Passing it also switches on the `synchronize()` calls that make the
    numbers truthful, so it is for the profiled arm only.

    Returns (None, None, None) if any of the three roles is missing.
    """
    def _mark():
        if timing is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    packed = _group_packed_tensors_by_weight(raw)
    if packed["gate"]:
        out = []
        for role in ("gate", "down", "up"):
            comps = packed[role]
            if "weight_packed" not in comps or "weight_scale" not in comps:
                return None, None, None
            _t0 = _mark()
            on_device = {
                k: (v if k == "weight_shape" else v.to(device, non_blocking=True))
                for k, v in comps.items()
            }
            if timing is not None:
                _t1 = _mark()
                timing["h2d"] += _t1 - _t0
            out.append(dequantize_packed_weight(on_device, dtype))
            if timing is not None:
                timing["construct"] += _mark() - _t1
        return out[0], out[1], out[2]

    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    if w_gate is None or w_down is None or w_up is None:
        return None, None, None
    _t0 = _mark()
    dense = (w_gate.to(device, dtype=dtype, non_blocking=True),
             w_down.to(device, dtype=dtype, non_blocking=True),
             w_up.to(device, dtype=dtype, non_blocking=True))
    if timing is not None:
        timing["h2d"] += _mark() - _t0
    return dense


def unpack_int32_to_int8(packed: "torch.Tensor", num_bits: int, cols: int) -> "torch.Tensor":
    """Inverse of compressed-tensors' `pack_to_int32` along the last dim.

    32 consecutive elements occupy exactly `num_bits` int32 words, packed
    densely from bit 0 with an offset of 2**(num_bits-1) so the stored codes are
    unsigned; this returns the signed int8 codes. Held to the reference
    implementation bit-for-bit by tests/test_packed_expert_dequant.py, which is
    why nothing on the decode path has to import compressed_tensors.
    """
    rows, num_words = packed.shape
    if 32 % num_bits == 0:
        # No element straddles a word (num_bits in 1/2/4/8 — every int4 and int8
        # checkpoint), so the unpack is a broadcast shift with no gather. The
        # general path below costs ~3x more on GPU, which at 480 expert-linears
        # per token is the difference between 50 ms and 150 ms.
        per_word = 32 // num_bits
        shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * num_bits
        out = (packed.unsqueeze(-1) >> shifts) & ((1 << num_bits) - 1)
        out = out.view(rows, num_words * per_word)[:, :cols]
        return (out - (1 << (num_bits - 1))).to(torch.int8)

    if num_words % num_bits:
        packed = torch.nn.functional.pad(packed, (0, num_bits - num_words % num_bits))
        num_words = packed.shape[1]
    num_groups = num_words // num_bits
    value_g = packed.reshape(rows * num_groups, num_bits)

    elem = torch.arange(32, device=packed.device, dtype=torch.int32)
    bit_start = elem * num_bits
    word_idx = (bit_start // 32).long()
    bit_off = bit_start % 32
    lo_bits = torch.clamp(32 - bit_off, max=num_bits)

    out = (value_g[:, word_idx] >> bit_off.unsqueeze(0)) & ((1 << lo_bits) - 1).unsqueeze(0)
    straddles = lo_bits < num_bits
    if bool(straddles.any()):
        hi_bits = num_bits - lo_bits[straddles]
        out[:, straddles] |= (
            value_g[:, word_idx[straddles] + 1] & ((1 << hi_bits) - 1).unsqueeze(0)
        ) << lo_bits[straddles].unsqueeze(0)

    out = out.view(rows, num_groups * 32)[:, :cols]
    return (out - (1 << (num_bits - 1))).to(torch.int8)


def dequantize_packed_weight(components: dict, dtype: "torch.dtype") -> "torch.Tensor":
    """One expert linear's dense weight from its pack-quantized components.

    Bit width and group size are DERIVED from the tensors (packed vs original
    element count; scale's last dim) — never read from a config and never a
    literal, so int4-g32, int4-g128 and int8 checkpoints all take the same path
    (CLAUDE.md §5). Runs wherever `components` already live, so calling it with
    GPU tensors keeps the dequant off the CPU.
    """
    packed = components["weight_packed"]
    scale = components["weight_scale"]
    if "weight_zero_point" in components or "weight_g_idx" in components:
        raise NotImplementedError(
            "asymmetric / act-order pack-quantized experts carry a zero point or "
            "g_idx that Vaayu does not apply; this checkpoint would decode to "
            "shifted weights. Supported today: symmetric group quantization."
        )
    rows, cols = (int(x) for x in components["weight_shape"].tolist())

    num_bits, remainder = divmod(packed.numel() * 32, rows * cols)
    if remainder or not 1 <= num_bits <= 8:
        raise ValueError(
            f"cannot derive bit width: {packed.numel()} int32 words for a "
            f"{rows}x{cols} weight"
        )

    codes = unpack_int32_to_int8(packed, num_bits, cols)
    groups = scale.shape[-1]
    return (
        codes.view(rows, groups, cols // groups).to(dtype)
        * scale.view(rows, groups, 1).to(dtype)
    ).view(rows, cols)


def _group_exl3_tensors_by_weight(raw: dict) -> dict[str, dict[str, "torch.Tensor"]]:
    """Groups EXL3 checkpoint tensors by weight role and component.

    Returns {"gate": {"trellis": t, ...}, "down": {...}, "up": {...}}.
    """
    result: dict[str, dict[str, "torch.Tensor"]] = {"gate": {}, "down": {}, "up": {}}
    for key, tensor in raw.items():
        role = None
        for pattern, r in _EXPERT_WEIGHT_ROLES:
            if pattern in key:
                role = r
                break
        if role is None:
            continue
        component = key.rsplit(".", 1)[-1]
        if component in _EXL3_COMPONENT_KEYS:
            result[role][component] = tensor
    return result


def accumulate_expert_outputs(final_hidden, contributions) -> None:
    """Add every expert's contribution to `final_hidden` in EXPERT-ID order.

    `contributions` is an iterable of (expert_idx, token_idx, hidden) built as
    experts finish. Experts are CONSUMED in arrival order on purpose -- that is
    what overlaps disk reads with compute -- but arrival order varies run to
    run: VRAM-resident experts run first, and the rest arrive in disk COMPLETION
    order (_iter_expert_loads waits FIRST_COMPLETED). Since each token routes to
    top_k experts, several contributions land on the SAME rows, and floating
    point addition is not associative at any width, so summing in arrival order
    makes the engine a non-deterministic function of its input.

    Measured on Qwen3-235B (2026-08-15): three identical greedy generate runs
    produced TWO distinct token sequences, diverging at the 4th new token, with
    per-run BC construction counts of 25380 / 24570 / 24696 -- cache warmth
    changing the order. An earlier fix widened this accumulator from bf16 to
    fp32 and its comment claimed fp32 made "the cast result identical whatever
    the order": that is false. fp32's error is normally far below the bf16
    output's own precision, but any value sitting on a bf16 rounding boundary
    tips regardless, and across 94 layers something always does. Width narrows
    the window; only a fixed summation order closes it.

    Sorting here rather than sorting execution keeps the overlap: the disk reads
    still complete in whatever order the NVMe gives them.
    """
    for _eidx, token_idx, hidden in sorted(contributions, key=lambda c: c[0]):
        final_hidden.index_add_(0, token_idx, hidden.to(final_hidden.dtype))


def _make_bc_linear_exl3(
    gpu_components: "dict[str, torch.Tensor]",
    in_features: int,
    out_features: int,
    device: str,
):
    """Build a temporary BC_LinearEXL3 C++ object from pre-moved GPU tensors.

    Caller must set result to None when done — PyTorch GC does not manage C++ heap.
    """
    from exllamav3 import ext as exl3_ext
    from exllamav3.util.tensor import g_tensor_cache

    trellis = gpu_components.get("trellis")
    suh = gpu_components["suh"] if "suh" in gpu_components else gpu_components.get("su")
    svh = gpu_components["svh"] if "svh" in gpu_components else gpu_components.get("sv")
    mcg = gpu_components.get("mcg")
    mul1 = gpu_components.get("mul1")

    if trellis is None or suh is None or svh is None:
        raise ValueError(
            f"EXL3 expert missing required tensors (trellis/suh/svh); "
            f"got: {list(gpu_components.keys())}"
        )

    K = trellis.shape[-1] // 16
    xh = g_tensor_cache.get(device, (1, in_features), torch.float16)

    return exl3_ext.exllamav3_ext.BC_LinearEXL3(
        trellis, suh, svh, K,
        None,
        mcg is not None,
        mul1 is not None,
        xh,
    )


_ROUTER_NAMES = {"gate", "router", "w_gate", "router_weights"}
_MIN_EXPERTS = 4  # ModuleList must have > this many children to be an expert list
_EXPERT_PREFIX_RE = re.compile(r'^.*?\.experts\.\d+\.')

_log = logging.getLogger("vaayu.moe_patch")


@dataclass
class MoEBlockInfo:
    layer_idx: int         # index in model.layers (0-indexed transformer layer number)
    layer_name: str        # shard name e.g. "model.layers.3" (matches checkpoint filename stem)
    module_path: str       # full dotted path e.g. "model.layers.3.block_sparse_moe"
    moe_module: nn.Module
    router_attr: str       # attribute on moe_module holding the gate Linear
    experts_attr: str      # attribute on moe_module holding the experts ModuleList
    expert_style: str      # "module_list" or "fused"
    # Which forward factory actually produced this block's forward, recorded by
    # _make_sparse_moe_forward. None until the block is patched. env.json reports
    # it; nothing at runtime branches on it.
    forward_name: str | None = None


def detect_moe_blocks(model: nn.Module) -> list[MoEBlockInfo]:
    """
    Generic MoE block detector. Finds any module that directly contains:
      1. An nn.ModuleList with > _MIN_EXPERTS children  (the expert FFN list)
         OR a fused expert module (e.g. MixtralExperts) with num_experts > _MIN_EXPERTS
      2. A sibling nn.Linear whose attribute name is in _ROUTER_NAMES  (the gate)
         OR a router module (e.g. MixtralTopKRouter) with a .weight parameter
         OR an ExLlamaV3Linear (a plain nn.Linear router that EXL3LoaderStrategy's
         setup_model() already swapped, before detection runs — some EXL3
         checkpoints leave small modules like routers unquantized)
    """
    from .persist.loader_strategy import ExLlamaV3Linear

    results: list[MoEBlockInfo] = []
    seen_paths: set[str] = set()

    for mod_path, mod in model.named_modules():
        experts_attr: str | None = None
        router_attr: str | None = None
        expert_style: str | None = None

        _found_expert_lists: list[str] = []
        for child_name, child in mod.named_children():
            # Old style: nn.ModuleList with > 4 children
            if isinstance(child, nn.ModuleList) and len(child) > _MIN_EXPERTS:
                _found_expert_lists.append(child_name)
                expert_style = "module_list"
            # New style: any module with a num_experts attribute > 4 (and not nn.Linear)
            # Skip router-named modules — MixtralTopKRouter has num_experts but is NOT an expert list
            elif (hasattr(child, 'num_experts')
                  and getattr(child, 'num_experts', 0) > _MIN_EXPERTS
                  and not isinstance(child, nn.Linear)
                  and child_name not in _ROUTER_NAMES):
                _found_expert_lists.append(child_name)
                expert_style = "fused"

            # Router check
            if isinstance(child, nn.Linear) and child_name in _ROUTER_NAMES:
                router_attr = child_name
            elif isinstance(child, ExLlamaV3Linear) and child_name in _ROUTER_NAMES:
                router_attr = child_name
            elif (child_name in _ROUTER_NAMES
                  and hasattr(child, 'weight')
                  and isinstance(getattr(child, 'weight', None), (nn.Parameter, torch.Tensor))):
                router_attr = child_name

        if len(_found_expert_lists) == 1:
            experts_attr = _found_expert_lists[0]
        elif len(_found_expert_lists) > 1:
            # Prefer "experts" over other names; log warning for ambiguous cases
            preferred = next((n for n in _found_expert_lists if n == "experts"), None)
            experts_attr = preferred or _found_expert_lists[0]
            print(f"[Vaayu-MoE] warn: multiple expert lists at {mod_path}: "
                  f"{_found_expert_lists}, using '{experts_attr}'")

        if experts_attr is None or router_attr is None:
            continue
        if mod_path in seen_paths:
            continue

        # Extract layer index from path (e.g. "model.layers.3.block_sparse_moe" -> 3)
        m = re.search(r'\.?(\d+)\.', mod_path + ".")
        if m is None:
            continue
        layer_idx = int(m.group(1))

        # layer_name = path prefix up to and including the numeric index
        m2 = re.match(r'(.*\.\d+)', mod_path)
        layer_name = m2.group(1) if m2 else mod_path

        seen_paths.add(mod_path)
        results.append(MoEBlockInfo(
            layer_idx=layer_idx,
            layer_name=layer_name,
            module_path=mod_path,
            moe_module=mod,
            router_attr=router_attr,
            experts_attr=experts_attr,
            expert_style=expert_style,
        ))

    return results


def _inject_expert_weights(
    expert_mod: nn.Module,
    raw: dict,
    expert_idx: int,
    layer_name: str,
    device: str,
    dtype,
    timing: dict | None = None,
) -> dict | None:
    """
    Inject expert tensor weights into expert_mod for inference.
    Returns the per-submodule state it injected on success, None if any submodule
    failed (expert skipped). The returned dict is
    {submod_path: {tensor_name: gpu_tensor}, ..., "bc": {submod_path: qmatrix}} —
    the exact shape ExpertVRAMCache stores, whose _group_bytes skips the "bc" key.

    Uses regex to strip the full key prefix generically:
        "layers.0.block_sparse_moe.experts.0.w1.weight" → "w1.weight"
    Works for Mixtral, DeepSeek, Kimi, Qwen MoE without hardcoding architecture names.

    Groups tensors by their target submodule first (not injected one-by-one)
    because an EXL3-quantized submodule (setup_model() may have swapped a
    module-list expert's w1/w2/w3 to ExLlamaV3Linear) needs all its quant
    tensors (trellis/suh/svh/...) together to build one QMatrix — see
    persist.loader_strategy.build_exl3_qmatrix.

    timing: optional dict accumulating {'h2d': seconds, 'construct': seconds}
    across every submodule this call injects (EXL3 QMatrix build and plain
    tensor copies alike land in 'h2d'; only the QMatrix construction lands in
    'construct'). None (the default) means "don't time" — callers that don't
    profile pay nothing extra.
    """
    from .persist.loader_strategy import ExLlamaV3Linear, build_exl3_qmatrix, _EXL3_QTENSOR_KEYS

    state: dict = {}
    bc_objs: dict = {}
    groups: dict[str, dict] = {}
    for param_key, tensor in raw.items():
        m = _EXPERT_PREFIX_RE.match(param_key)
        if m is None:
            continue  # not an expert tensor (shouldn't happen but be safe)
        local_key = param_key[m.end():]  # e.g. "w1.weight" or "w1.trellis"
        submod_path, _, base = local_key.rpartition(".")
        groups.setdefault(submod_path, {})[base] = tensor

    for submod_path, bases in groups.items():
        try:
            submod = expert_mod
            for part in submod_path.split("."):
                submod = getattr(submod, part)

            if isinstance(submod, ExLlamaV3Linear) and any(
                b in _EXL3_QTENSOR_KEYS for b in bases
            ):
                # non_blocking=False: these tensors come from disk loads /
                # the RAM cache, not guaranteed page-locked — see
                # build_exl3_qmatrix's docstring for why async here leaks host RAM.
                state[submod_path] = build_exl3_qmatrix(
                    submod, bases, device, non_blocking=False, timing=timing)
                bc_objs[submod_path] = submod.bc
                continue

            for base, tensor in bases.items():
                if timing is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                t = tensor.to(device, dtype=dtype, non_blocking=False)
                if timing is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timing['h2d'] = timing.get('h2d', 0.0) + (time.perf_counter() - _t0)
                state.setdefault(submod_path, {})[base] = t
                if isinstance(submod, ExLlamaV3Linear):
                    # Unquantized weight/bias landing on a module setup_model()
                    # swapped anyway — ExLlamaV3Linear has no "weight"/"bias"
                    # attribute of its own, target the real fallback fields.
                    if base == "weight":
                        submod.weight_fallback = t
                    elif base == "bias":
                        submod.bias_param = nn.Parameter(t, requires_grad=False)
                    else:
                        setattr(submod, base, nn.Parameter(t, requires_grad=False))
                else:
                    setattr(submod, base, nn.Parameter(t, requires_grad=False))
        except Exception as e:
            _log.warning(
                f"Expert weight inject failed: {submod_path} ({list(bases)}): {e}"
            )
            return None
    if bc_objs:
        state["bc"] = bc_objs
    return state


def _attach_cached_expert(expert_mod: nn.Module, entry: dict) -> bool:
    """
    Re-bind a VRAM-cached expert (an entry built by _inject_expert_weights) onto a
    live expert module. The inverse of injection, minus the disk read and the H2D
    copy — that saving is the entire point of the VRAM expert cache.

    Returns True if every role in the entry was attached, False on any structural
    mismatch (caller falls through to a normal load rather than computing with a
    half-attached module).

    Layout-agnostic on purpose: it walks whatever submodule paths the entry
    carries, so it serves EXL3 and non-EXL3 module-list experts alike without
    knowing anything about w1/w2/w3 or a gated-MLP shape.
    """
    from .persist.loader_strategy import ExLlamaV3Linear

    bc_objs = entry.get("bc") or {}
    for submod_path, tensors in entry.items():
        if submod_path == "bc":
            continue
        try:
            submod = expert_mod
            for part in submod_path.split("."):
                submod = getattr(submod, part)

            bc = bc_objs.get(submod_path)
            if bc is not None:
                # EXL3: the QMatrix already wraps the cached GPU tensors, so
                # re-binding it is the whole attach — no Parameter to restore.
                submod.bc = bc
                continue

            for base, t in tensors.items():
                if isinstance(submod, ExLlamaV3Linear):
                    if base == "weight":
                        submod.weight_fallback = t
                    elif base == "bias":
                        submod.bias_param = nn.Parameter(t, requires_grad=False)
                    else:
                        setattr(submod, base, nn.Parameter(t, requires_grad=False))
                else:
                    setattr(submod, base, nn.Parameter(t, requires_grad=False))
        except Exception as e:
            _log.warning(f"Cached expert attach failed: {submod_path}: {e}")
            return False
    return True


def patch_moe_blocks(
    model: nn.Module,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: str,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
    loader_strategy=None,
    blocks=None,
) -> list[MoEBlockInfo]:
    """
    Detect all MoE blocks in model and replace their forward() with a sparse version.

    The sparse forward:
      1. Runs the pinned router gate to determine active expert indices
      2. Loads only those expert tensors (ExpertRAMCache → io_uring fallback)
      3. Injects each expert's weights to GPU, runs it, unloads back to meta
      4. Writes routing result to vaayu_ref._expert_history[layer_idx]

    vaayu_ref: reference to VaayuBaseModel instance.

    _expert_history has NO production reader (audit F7). The docstring used to
    claim "the forward() loop reads _expert_history after each layer and submits
    speculative prefetch" -- that mechanism does not exist and never did, and
    the claim sent readers looking for it. The write stays because the tests
    read it: it is the observation hook they use to assert WHICH experts a
    forward actually routed to (tests/test_moe_patch.py, ~8 assertions across
    all three forwards). Removing it would cost those tests their only view of
    routing and buy a runtime saving of approximately zero -- one dict
    assignment per layer.

    Returns list of patched MoEBlockInfo for logging.
    """
    try:
        from .persist.loader_strategy import EXL3LoaderStrategy
        is_exl3 = isinstance(loader_strategy, EXL3LoaderStrategy)
    except ImportError:
        is_exl3 = False
    compute_device = "cuda" if (is_exl3 or device.startswith("cuda")) else "cpu"

    # `blocks` lets a caller that already ran detect_moe_blocks (the engine does,
    # to size the slab arena before this point) pass the result in rather than
    # walking the module tree a second time. None = detect here, as before.
    if blocks is None:
        blocks = detect_moe_blocks(model)
    for block_info in blocks:
        original_forward = block_info.moe_module.forward
        sparse_fwd = _make_sparse_moe_forward(
            block_info=block_info,
            expert_cache=expert_cache,
            offset_index=offset_index,
            checkpoint_path=Path(checkpoint_path),
            top_k=top_k,
            device=device,
            dtype=dtype,
            vaayu_ref=vaayu_ref,
            compute_device=compute_device,
            original_forward=original_forward,
            is_exl3=is_exl3,
        )
        # Fix 1: direct assignment — no __get__ binding needed
        block_info.moe_module.forward = sparse_fwd
        block_info.forward_name = getattr(sparse_fwd, "__vaayu_forward__", None)
        experts_mod = getattr(block_info.moe_module, block_info.experts_attr)
        if hasattr(experts_mod, 'num_experts'):
            n_experts = experts_mod.num_experts
        else:
            n_experts = len(experts_mod)
            
        print(
            f"[Vaayu-MoE] patched {block_info.module_path} "
            f"(router={block_info.router_attr}, experts={block_info.experts_attr}, "
            f"n={n_experts}, top_k={top_k})"
        )
    return blocks


def _make_sparse_moe_forward(
    block_info: MoEBlockInfo,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
    compute_device: str = "cpu",
    original_forward = None,
    is_exl3: bool = False,
):
    """
    Returns a replacement forward() for the MoE block, dispatching based on expert_style.

    The returned function is tagged `__vaayu_forward__` with the name of the
    factory that produced it. env.json reports that tag rather than re-deriving
    the choice from expert_style + is_exl3: a second copy of this dispatch table
    drifts, and when it drifts it misreports which of the four forwards ran --
    which is triage step one (CLAUDE.md 0).
    """
    if block_info.expert_style == "fused":
        if is_exl3:
            fwd = _make_fused_sparse_forward_exl3(
                block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
            )
            fwd.__vaayu_forward__ = "_make_fused_sparse_forward_exl3"
            return fwd
        fwd = _make_fused_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            compute_device=compute_device,
            original_forward=original_forward,
        )
        fwd.__vaayu_forward__ = "_make_fused_sparse_forward"
        return fwd
    fwd = _make_modulelist_sparse_forward(
        block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
        original_forward=original_forward,
    )
    fwd.__vaayu_forward__ = "_make_modulelist_sparse_forward"
    return fwd


# Hardware constant: free VRAM to guarantee before each MoE layer DURING PREFILL,
# by evicting the (prefill-idle) VRAM expert cache. A property of the card, not
# the model (CLAUDE.md §5) — same CLASS of constant as vaayu_base._VRAM_RESERVE_FLOOR_BYTES
# (both are hardware headroom reserves), but a deliberately DIFFERENT value (2 GB
# here vs 3.5 GB there): this one only needs to exceed one layer's prefill-chunk
# activation, since the poll runs per layer, while the other must additionally
# cover the decode-boundary cache re-warm + sampling transient (see its comment
# in vaayu_base.py). Do not conflate the two or assume they move together.
_PREFILL_VRAM_HEADROOM = 2 * 1024 ** 3


def _format_vram_diag(layer_idx, free, alloc, reserved,
                      cache_usage, cache_budget, prefill_evict):
    """One grep-able diagnostic line for the VRAM prefill trajectory. All byte
    inputs; emits GB. Pure — no torch/CUDA calls, so it is unit-testable."""
    g = 1024 ** 3
    return (f"[VRAM-DIAG] layer={layer_idx} "
            f"free={free / g:.2f}GB alloc={alloc / g:.2f}GB "
            f"reserved={reserved / g:.2f}GB "
            f"cache={cache_usage / g:.2f}/{cache_budget / g:.2f}GB "
            f"prefill_evict={prefill_evict}")


def _ensure_prefill_headroom(vaayu_ref, layer_idx=-1) -> None:
    """During a prefill pass, evict the VRAM expert cache until free VRAM meets
    _PREFILL_VRAM_HEADROOM. The cache is not read during prefill — the two
    forwards that consult it (module-list and fused-EXL3) set their local
    _vram = None when _moe_prefill_pass is set, and the non-EXL3 fused forward
    never touches it at all — so its residency gives prefill no benefit while
    starving activations. Evicting here is therefore free for prefill; the
    evicted experts re-warm during the decode that follows.
    A no-op during decode, without a VRAM cache, or without CUDA.

    When VAAYU_VRAM_DIAG is set, emits one grep-able [VRAM-DIAG] line per MoE
    layer capturing the prefill VRAM trajectory (read-only; behavior unchanged)."""
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    _vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vram is None or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
    if os.environ.get("VAAYU_VRAM_DIAG"):
        free_now, _ = torch.cuda.mem_get_info()
        print(_format_vram_diag(
            layer_idx, free_now,
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
            _vram.current_usage, _vram.budget_bytes, _vram.prefill_evictions),
            flush=True)


def _maybe_record_usage(vaayu_ref, layer_idx, expert_indices) -> None:
    """Count DECODE-time expert routes when the usage histogram is enabled.

    A no-op when the feature is off (the common case — one getattr). Skipped
    during prefill on purpose: the VRAM expert cache is not consulted in a
    prefill pass, so there are no budget-pressure evictions then, and counting
    prefill routes would make the report compare decode-side evictions against
    prefill+decode routes. See the measurement design spec.
    """
    tracker = getattr(vaayu_ref, "_expert_usage", None)
    if tracker is None:
        return
    if getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    tracker.record(layer_idx, expert_indices)


def _make_modulelist_sparse_forward(
    block_info: MoEBlockInfo,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
    original_forward = None,
):
    from .persist.loader_strategy import ExLlamaV3Linear

    layer_name = block_info.layer_name
    layer_idx  = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
        _ensure_prefill_headroom(vaayu_ref, layer_idx)
        if layer_name in vaayu_ref.pinned_layer_names:
            return original_forward(hidden_states, **kwargs)

        _prof = getattr(vaayu_ref, 'profiling_mode', False) and getattr(vaayu_ref, '_moe_profile', None) is not None
        _mp = vaayu_ref._moe_profile if _prof else None

        # 1. Run router gate (permanently pinned in VRAM by _init_pinned_layers)
        # 'route' brackets the router forward + route_experts + the
        # unique().tolist() sync below. Sync BEFORE starting the clock so any
        # GPU work still in flight from the previous layer isn't misattributed
        # to routing; no sync is needed at the end — .tolist() below already
        # forces a blocking D2H copy, so the window it closes is truthful
        # without an extra synchronize() call.
        if _prof:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
        router = getattr(block_info.moe_module, block_info.router_attr)
        router_logits = router(hidden_states)
        # Use the model's OWN scoring function when it defines one — never a
        # hardcoded softmax. MiniMax-M2 (this layout's on-device target) scores
        # with sigmoid + e_score_correction_bias; softmax produced a far peakier
        # mix on every layer and made the model emit incoherent text.
        routing_weights, expert_indices = route_experts(
            block_info.moe_module, router_logits, top_k
        )
        _warn_cache_bias_inert(
            vaayu_ref, "this model uses the module-list expert layout, whose forward "
                       "has no cache-aware routing")

        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()
        if _prof:
            _mp['route'] += time.perf_counter() - _t0

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
            vaayu_ref.telemetry.record_batch_overlap(
                layer_idx, len(unique_experts), vaayu_ref._batch_n * top_k)

        # 2/3. Load -> inject -> compute -> free, ONE expert at a time. Bounds
        # peak concurrent QMatrix/weight residency to a single expert
        # regardless of how many experts this token's routing touches in this
        # layer. An earlier version loaded+injected ALL unique_experts before
        # computing/freeing any of them — peak residency scaled with
        # unique_experts count, which spiked hard on wide-routing tokens
        # (worst case: prefill's whole-prompt chunk) and caused an OOM-kill
        # that gradual periodic gc.collect()/empty_cache() didn't prevent,
        # since it was a single-token spike, not a slow leak.
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        experts_list = getattr(block_info.moe_module, block_info.experts_attr)
        # fp32 accumulator, cast back at the return -- see the extended note in
        # _make_fused_sparse_forward_exl3. Experts reach index_add_ in an order
        # that varies run to run (VRAM residency, then disk completion order),
        # and a bf16 accumulator turns that into ~1 ULP per layer that compounds
        # into different tokens over 62 layers. fp32 shrinks that error but does
        # not remove the order dependence -- accumulate_expert_outputs below is
        # what makes the sum order-independent.
        final_hidden = torch.zeros_like(hidden_states, dtype=torch.float32)

        # Permutation dispatch (ported from _make_fused_sparse_forward_exl3): one
        # CPU argsort per layer instead of one torch.where scan per expert
        # (~496 scans/token on the target model). hidden_states is (B, S, H), not
        # flat like the fused path's flat_hidden, so expert_indices/routing_weights
        # are flattened to (B*S, top_k) here — flat index b*S+s agrees between
        # .reshape(-1, top_k) and .view(-1, H) since both simply drop the B/S
        # split of the same row-major layout.
        if _prof:
            _t0 = time.perf_counter()
        # Lookahead routing predictor (Task 12, decode only) — the SAME
        # _lookahead_step the fused-EXL3 forward calls, so VAAYU_LOOKAHEAD is
        # not a no-op on this layout (P2). Placed before the miss dispatch
        # below so predicted reads are queued ahead of this layer's own, and
        # inside 'mask_sync' so profiling still accounts for every window.
        # Layers in pinned_layer_names return above and predict nothing: their
        # experts are resident, so there is no read for a prefetch to hide.
        _lookahead_step(vaayu_ref, expert_cache, offset_index, checkpoint_path,
                        layer_idx, hidden_states, top_k, unique_experts)
        B, S, H = hidden_states.shape
        _pair, _starts, _counts_l = compute_expert_dispatch(
            expert_indices.cpu().reshape(-1, top_k), len(experts_list))
        _pair_gpu = _pair.to(hidden_states.device, non_blocking=True)
        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0
        # Gather source only (read-only): a copy-if-non-contiguous is harmless.
        _flat_hidden = hidden_states.reshape(-1, H)
        # Accumulation target: MUST share storage with final_hidden, so .view()
        # (not .reshape()) — final_hidden is fresh from torch.zeros_like() and
        # therefore always contiguous, so .view() is safe and never copies. If
        # this ever silently copied, index_add_ below would write into a
        # discarded temporary and the whole layer would output zeros.
        _flat_final = final_hidden.view(-1, H)
        # Buffered, then summed in expert-id order after every expert has run
        # (accumulate_expert_outputs). Experts are still CONSUMED in completion
        # order -- that overlap is the point -- but the SUM must not depend on
        # it. Holds tokens x top_k x H of fp16, which is 64 KB at decode and
        # ~32 MB for a 512-token prefill chunk.
        _contribs: list = []
        _flat_routing = routing_weights.reshape(-1, top_k)

        # VRAM expert cache (Fix #7, ported from the fused path). Same two gates:
        # prefill routes ~every expert once, so caching there only churns the LRU;
        # and _vram_cacheable_layers restricts to the layer prefix that fits when
        # the budget can't hold a whole token's working set (None = no restriction).
        _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
        if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
            _vram = None
        _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
        if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
            _vram = None

        # Experts already VRAM-resident need no RAM lookup and no disk read at all.
        # ONE get() per expert, entry held for the compute loop below — not
        # has()-then-get(), which would take the lock twice and leave a window
        # where an expert could vanish between the two calls.
        _vram_resident: dict = {}
        if _vram is not None:
            for _e in unique_experts:
                _entry = _vram.get(layer_idx, _e)
                if _entry is not None:
                    _vram_resident[_e] = _entry
        _to_load = [e for e in unique_experts if e not in _vram_resident]

        # Dispatch disk misses up front and consume hits first, so compute overlaps
        # the in-flight reads. These futures are CPU-side (RAM cache / slab arena,
        # bounded by the RAM expert budget) — GPU injection below stays strictly
        # one expert at a time, so the residency bound is untouched.
        _expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            _to_load,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref,
        )

        def _run_expert(eidx: int, expert_mod) -> None:
            # 'scatter' covers the gather/scatter bracketing expert_mod(_x_in):
            # the contiguous permutation-slice + gather before gemm, and the
            # weighted index_add_ accumulate after it. It is timed in two
            # halves so 'gemm' keeps measuring ONLY the expert call itself.
            # Both halves reuse gemm's own pre-existing synchronize() calls as
            # their boundary instead of adding a second one back-to-back —
            # gemm already drains the queue right before and right after the
            # expert call, so that sync is both gemm's boundary and scatter's.
            if _prof:
                _t_scatter0 = time.perf_counter()
            _s, _c = _starts[eidx], _counts_l[eidx]
            if _c == 0:
                return
            token_idx_gpu = _pair_gpu[0, _s:_s + _c]
            kpos_gpu = _pair_gpu[1, _s:_s + _c]
            _x_in = _flat_hidden[token_idx_gpu]
            if _prof:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _mp['scatter'] += time.perf_counter() - _t_scatter0
                _t0 = time.perf_counter()
            expert_out = expert_mod(_x_in)                               # (M, H)
            if _prof:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _mp['gemm'] += time.perf_counter() - _t0
            _trace_expert(vaayu_ref, layer_idx, eidx, expert_mod, _x_in, expert_out)
            if _prof:
                _t_scatter1 = time.perf_counter()
            w = _flat_routing[token_idx_gpu, kpos_gpu].unsqueeze(-1)
            _contribs.append((eidx, token_idx_gpu, w * expert_out))
            if _prof:
                # This is the one new synchronize() this phase actually adds
                # (the pre-gather boundary above is free, borrowed from gemm's
                # existing sync). It runs once per routed expert (~496x/token
                # on the target model) — real added serialization, but the
                # index_add_ accumulate is a real GPU kernel and an unsynced
                # timer here would measure launch time, not completion.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _mp['scatter'] += time.perf_counter() - _t_scatter1

        def _detach_expert(eidx: int, expert_mod) -> None:
            if _prof:
                _t0 = time.perf_counter()
            try:
                # Detach from the live module — ALWAYS, hit or miss. See the
                # extended rationale on the miss arm below.
                for submod in expert_mod.modules():
                    if isinstance(submod, ExLlamaV3Linear):
                        submod.bc = None
                        submod.weight_fallback = None
                expert_mod.to("meta")
            except Exception as e:
                _log.warning(f"Failed to unload expert {eidx} to meta: {e}")
            if _prof:
                # Pure CPU-side: .modules() iteration + reference clears, and
                # .to("meta") swaps storage metadata without launching a GPU
                # kernel — no synchronize here, it would only add cost and
                # distortion without making this figure any more truthful.
                _mp['detach'] += time.perf_counter() - _t0

        def _load_inject_run(eidx: int, raw) -> None:
            # Shared by the normal miss-load loop and the attach-failure
            # fallback below: inject -> (optionally cache) -> compute -> detach,
            # with the continue-on-failed-injection check INSIDE the try so
            # _detach_expert still runs when injection binds some submodules
            # (e.g. w1) before a later group raises (e.g. w2) and returns None —
            # skipping it would leave those bound tensors off-meta forever.
            expert_mod = experts_list[eidx]
            try:
                # _inject_expert_weights does two different things for an EXL3
                # submodule (H2D tensor copy, then QMatrix construction) — a
                # single bracket around the whole call would conflate them, so
                # thread a scratch dict through and let it attribute each
                # phase to its own _mp key (see build_exl3_qmatrix's timing
                # param). Non-EXL3 submodules only ever populate 'h2d'.
                _timing = {} if _prof else None
                entry = _inject_expert_weights(
                    expert_mod, raw, eidx, layer_name, device, dtype, timing=_timing)
                if _prof:
                    _mp['h2d'] += _timing.get('h2d', 0.0)
                    _mp['construct'] += _timing.get('construct', 0.0)
                if not entry:
                    return
                if _vram is not None:
                    # Return value deliberately ignored: unlike the fused path there
                    # is no _owned_by_cache branch. _detach_expert must run whether
                    # or not the cache accepted the entry — on a hit it frees
                    # nothing (the entry holds the refs); without it an evicted
                    # entry would still be referenced by the live ModuleList, so
                    # eviction would reclaim no VRAM. It also keeps peak GPU
                    # residency at one uncached expert regardless of routing width.
                    _vram.put(layer_idx, eidx, entry)
                _run_expert(eidx, expert_mod)
            finally:
                _detach_expert(eidx, expert_mod)

        try:
            _attach_failed: list[int] = []
            for eidx, entry in _vram_resident.items():
                expert_mod = experts_list[eidx]
                try:
                    if _prof:
                        _t0 = time.perf_counter()
                    _attached = _attach_cached_expert(expert_mod, entry)
                    if _prof:
                        # Pure CPU-side: rebinds references (nn.Parameter
                        # wrapping an already-resident GPU tensor, or an EXL3
                        # QMatrix pointer swap) — no kernel is launched, so no
                        # synchronize is needed for this figure to be truthful.
                        _mp['attach'] += time.perf_counter() - _t0
                    if not _attached:
                        # Structural mismatch — a real bug, not a race. Drop the
                        # now-unusable entry so it isn't reported as a HIT (and
                        # fail to attach) on every subsequent token forever.
                        # _attach_cached_expert may have partially bound real
                        # submodules before returning False, so the detach below
                        # (finally) must still run. The project owner decided a
                        # failed attach must not silently drop this expert's
                        # contribution to final_hidden — fall through to a real
                        # disk load below instead of skipping compute.
                        _vram.drop(layer_idx, eidx)
                        _attach_failed.append(eidx)
                        continue
                    _run_expert(eidx, expert_mod)
                finally:
                    _detach_expert(eidx, expert_mod)

            for eidx, raw in _iter_expert_loads(_expert_tensors, _miss_futures, vaayu_ref, _mp):
                _load_inject_run(eidx, raw)

            if _attach_failed:
                # These experts were excluded from _to_load (they looked like
                # VRAM hits when dispatch ran), so _dispatch_expert_misses has
                # never seen them — dispatch their disk loads now. Run this
                # AFTER the main loop above: it blocks on brand-new disk reads
                # for these rare attach-failure experts, and running it first
                # would stall the main loop's RAM-cache hits and already-
                # in-flight futures behind that stall instead of overlapping.
                _fb_tensors, _fb_futures, _fb_pinned = _dispatch_expert_misses(
                    _attach_failed,
                    layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
                    offset_index=offset_index, expert_cache=expert_cache,
                    vaayu_ref=vaayu_ref,
                )
                _pinned_keys.update(_fb_pinned)
                for eidx, raw in _iter_expert_loads(_fb_tensors, _fb_futures, vaayu_ref, _mp):
                    _load_inject_run(eidx, raw)
        finally:
            # Release every key THIS call pinned, even if compute raised partway
            # through — a leaked pin permanently blocks that key's eviction.
            for _pk in _pinned_keys:
                expert_cache.unpin(*_pk)

        accumulate_expert_outputs(_flat_final, _contribs)

        # 4. Write routing result to vaayu_ref for speculative prefetch scheduling
        _shared = _shared_expert_output(block_info.moe_module, hidden_states.view(-1, H))
        if _shared is not None:
            _flat_final += _shared.to(_flat_final.dtype)

        vaayu_ref._expert_history[layer_idx] = unique_experts
        _trace_moe_block(vaayu_ref, layer_idx, "module_list", hidden_states,
                         final_hidden, routing_weights, unique_experts)
        return final_hidden.to(hidden_states.dtype), router_logits

    return sparse_forward


def _make_fused_sparse_forward(
    block_info: MoEBlockInfo,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
    compute_device: str = "cpu",
    original_forward = None,
):
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
        _ensure_prefill_headroom(vaayu_ref, layer_idx)
        if layer_name in vaayu_ref.pinned_layer_names:
            return original_forward(hidden_states, **kwargs)

        # Phase profile, same key set and same bracketing rules as the other two
        # forwards. This one recorded nothing until row 30 phase 3, whose whole
        # deliverable is the (h2d + gemm + tail) / T ratio these keys carry.
        _prof = (getattr(vaayu_ref, 'profiling_mode', False)
                 and getattr(vaayu_ref, '_moe_profile', None) is not None)
        _mp = vaayu_ref._moe_profile if _prof else None

        def _mark():
            if _prof and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

        # 'route' brackets the router forward, the top-k and the
        # unique().tolist() sync below -- the same window the other two forwards
        # call route.
        _t_route = _mark()
        # Router runs on GPU (pinned in VRAM)
        router_res = router(flat_hidden)
        if isinstance(router_res, tuple) and len(router_res) >= 3:
            router_logits, routing_weights, expert_indices = router_res[:3]
        else:
            router_logits = router_res
            routing_weights, expert_indices = torch.topk(
                router_logits.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        _warn_cache_bias_inert(
            vaayu_ref, "this model takes the non-EXL3 fused forward, which has no "
                       "cache-aware routing")

        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        unique_experts: list[int] = expert_indices.flatten().unique().tolist()
        if _prof:
            _mp['route'] += time.perf_counter() - _t_route

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
            vaayu_ref.telemetry.record_batch_overlap(
                layer_idx, len(unique_experts), vaayu_ref._batch_n * top_k)

        # Load active expert weights: cache → parallel sparse io_uring for misses
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        misses: list[int] = []
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                misses.append(eidx)

        _t_disk = _mark() if _prof else 0.0
        if misses:
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Reuse the persistent MoE executor (created once at model init) —
            # a fresh ThreadPoolExecutor per layer call paid thread create/join
            # on every MoE layer. Fallback covers unit tests whose mock
            # vaayu_ref has no executor.
            _ex = getattr(vaayu_ref, "_moe_executor", None)
            if _ex is not None:
                _results = _ex.map(_load_one, misses)
            else:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with _TPE(max_workers=min(len(misses), 8)) as _tmp_ex:
                    _results = list(_tmp_ex.map(_load_one, misses))
            for eidx, raw in _results:
                if raw is not None:
                    expert_tensors[eidx] = raw
        if _prof:
            _mp['disk_load'] += time.perf_counter() - _t_disk

        # fp32 accumulator, cast back at the return -- see the identical note in
        # _make_fused_sparse_forward_exl3. Experts reach index_add_ in disk
        # completion order, and a bf16 accumulator makes that order observable.
        # Expert mask on CPU (avoids PCIe for index tensors)
        _t_mask = _mark() if _prof else 0.0
        expert_indices_cpu = expert_indices.cpu()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)
        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t_mask

        # ── Compute FFN on the target device ──────────────────────────────────
        final_hidden = torch.zeros_like(flat_hidden, dtype=torch.float32)

        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue

            _t_scatter = _mark() if _prof else 0.0
            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]
            if _prof:
                _mp['scatter'] += time.perf_counter() - _t_scatter

            # h2d / construct are split inside _expert_dense_weights: the copy
            # and the dequant run back to back there, and the phase-4 gate is a
            # ratio of exactly those two.
            w_gate_gpu, w_down_gpu, w_up_gpu = _expert_dense_weights(
                raw, device, dtype, timing=_mp)
            if w_gate_gpu is None:
                continue

            _t_gemm = _mark() if _prof else 0.0
            gate_out = act_fn(F.linear(current_state, w_gate_gpu))
            up_out = F.linear(current_state, w_up_gpu)
            out = F.linear(gate_out * up_out, w_down_gpu)
            if _prof:
                _mp['gemm'] += _mark() - _t_gemm

            _t_scatter = _mark() if _prof else 0.0
            out = out * routing_weights[token_idx_gpu, top_k_pos.to(device), None]
            final_hidden.index_add_(0, token_idx_gpu, out.to(final_hidden.dtype))
            if _prof:
                _mp['scatter'] += _mark() - _t_scatter

            del w_gate_gpu, w_down_gpu, w_up_gpu

        result = final_hidden.to(flat_hidden.dtype)
        _shared = _shared_expert_output(moe_mod, flat_hidden)
        if _shared is not None:
            result = result + _shared.to(result.dtype)

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return result.view(B, S, H)

    return sparse_forward


# Backstop for a decode worker waiting on an in-flight load it lost the claim
# to. NOT a performance knob: it exists so a worker that dies without running
# its finally cannot hang the layer. A healthy wait is one expert read off NVMe
# (single-digit ms); anything near this value means something is wrong.
_LOAD_WAIT_BACKSTOP_S = 5.0


# Placeholder for a routed expert that is ALREADY resident in the VRAM expert
# cache and missed the RAM cache. It travels through expert_tensors so the
# compute loop still runs the expert, but carries no CPU tensors: its weights
# are on the GPU already, so reading them off NVMe again is pure waste. See
# _dispatch_expert_misses' vram_resident parameter.
VRAM_RESIDENT_NO_RAW = object()


def _dispatch_expert_misses(
    unique_experts,
    *,
    layer_idx: int,
    layer_name: str,
    shard_file: str,
    offset_index,
    expert_cache,
    vaayu_ref,
    staged_layer=None,
    vram_resident=None,
) -> tuple[dict, list, set]:
    """
    Split this layer's routed experts into RAM/staged hits and disk misses, submit
    the misses to the persistent pool, and return without waiting — so the caller
    computes hits while the reads are in flight.

    Returns (expert_tensors, miss_futures, pinned_keys). The caller MUST unpin
    every key in pinned_keys in a finally block; a leaked pin permanently blocks
    that key's eviction (a slow cache-capacity leak).

    `vram_resident` (fused-EXL3 forward only) is the set of routed experts the
    caller already holds a VRAM cache entry for. Their RAM lookup still runs —
    that LRU refresh is the residency invariant P1 measured, and dropping it is
    the change P1 refuted — but on a MISS no disk read is issued: the caller
    serves those experts from their GPU copies and gets VRAM_RESIDENT_NO_RAW in
    expert_tensors instead. The module-list forward filters these experts out
    before it calls this helper and passes nothing, so its behaviour is
    unchanged.

    Shared by the fused-EXL3 and module-list forwards: the slab/pin/arena
    discipline below is subtle enough that a second copy would rot.
    """
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab

    # Pin mechanism (race fix, round 2): a slab-backed cache entry checked out
    # here (get() hit or put_slab() miss) but not yet H2D-copied must not be
    # evicted-and-reused by a concurrent evictor — chiefly the VAAYU_LOOKAHEAD
    # stager, which drives heavy eviction churn — in that window. We refcount-
    # pin every checked-out key and the caller unpins once its copy is issued
    # (after which SlabArena.record_read()/acquire() guards the in-flight-copy
    # window). Only arena-backed (zero-clone) entries carry the race; the
    # legacy clone path (arena is None) copies independent memory, so pinning
    # is skipped there. pinned_keys tracks exactly which keys THIS call
    # pinned, so the caller's finally unpins each exactly once even if
    # compute raises partway through.
    _do_pin = expert_cache.arena is not None and not getattr(
        vaayu_ref, "_moe_prefill_pass", False)
    pinned_keys: set = set()
    expert_tensors: dict = {}
    misses: list = []

    for eidx in unique_experts:
        if staged_layer is not None and eidx in staged_layer:
            expert_tensors[eidx] = staged_layer[eidx]
            continue
        cached = expert_cache.get(layer_idx, eidx, pin=_do_pin)
        if cached is not None:
            expert_tensors[eidx] = cached
            if _do_pin:
                pinned_keys.add((layer_idx, eidx))
        elif vram_resident is not None and eidx in vram_resident:
            # On the GPU already — reading it off NVMe would fill a RAM slot
            # with a copy of something the compute loop will not look at.
            # Measured on Qwen3-235B at N=3: 1246 such reads, 2.45 % of decode
            # bytes, because a VRAM-prefix layer's RAM cap (budget over ALL MoE
            # layers) sits below its VRAM cap (budget over the prefix only), so
            # RAM evicts first. The get() above already refreshed the RAM LRU.
            expert_tensors[eidx] = VRAM_RESIDENT_NO_RAW
        else:
            misses.append(eidx)

    # Pin misses BEFORE dispatching their loads: the key isn't cached yet, so
    # it can't be evicted until put_slab() inserts it — at which point the pin
    # is already set and the evictor will skip it.
    if _do_pin:
        for _e in misses:
            expert_cache.pin(layer_idx, _e)
            pinned_keys.add((layer_idx, _e))

    miss_futures = []
    if misses:
        def _load_one(eidx: int):
            metas = offset_index.get_expert_offsets(layer_name, eidx)
            if metas is None:
                _log.warning(
                    f"[Vaayu-MoE] warn: no offset for {layer_name} expert {eidx}")
                return eidx, None
            # Prefill: transient load, never cached (routes ~all experts
            # once; caching would clone content decode replaces).
            if getattr(vaayu_ref, "_moe_prefill_pass", False):
                return eidx, load_expert_sparse(shard_file, metas)
            if expert_cache.arena is None:
                # No arena (e.g. compute_slab_bytes()==0 or CUDA-less unit
                # tests constructing ExpertRAMCache directly) — legacy
                # clone path, still cached.
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
            # Reservation: the lookahead stager may already be reading this
            # exact expert. Losing that race used to mean reading it twice off
            # NVMe and discarding one copy (~190 decode + ~200 stager duplicate
            # reads per token, measured). Wait for the in-flight read instead —
            # but ONLY if it has actually started. A claim that is merely queued
            # belongs to a stager worker that may sit behind other prefetches,
            # and blocking on that would let prefetch preempt an on-path decode
            # read, which is the inversion the dedicated _stager_executor exists
            # to prevent. The key is already pinned (dispatch pins every miss
            # before submitting), so a waited-for entry cannot be evicted out
            # from under us between the wait and the H2D copy.
            _claim = expert_cache.claim_load(layer_idx, eidx, "decode")
            if _claim is None:
                _other = expert_cache.peek_load(layer_idx, eidx)
                if _other is not None and _other.running:
                    # Backstop only, not a tuning knob: bounds the wait if the
                    # winning worker dies without running its finally. A healthy
                    # read is ~one expert off NVMe.
                    _other.done.wait(timeout=_LOAD_WAIT_BACKSTOP_S)
                    _cached = expert_cache.get(layer_idx, eidx)
                    if _cached is not None:
                        return eidx, _cached
                # Queued-only holder, or the winner failed and cached nothing:
                # fall through and read it ourselves, exactly as before.
            try:
                expert_cache.mark_running(_claim)
                return _load_one_uncontended(eidx, metas)
            finally:
                expert_cache.finish_load(_claim)

        def _load_one_uncontended(eidx: int, metas):
            # Zero-clone decode path: io_uring reads straight into a pinned
            # slab; the cache adopts views (no clone anywhere disk→RAM→GPU).
            slab_idx = expert_cache.acquire_slab()
            if slab_idx is None:
                # arena exhausted and nothing evictable — legacy fallback
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
            slab = expert_cache.arena.tensor(slab_idx)
            try:
                # tensor_align=False: decode's cached experts are only ever
                # consumed via per-component .to(device) copies (always
                # naturally aligned on a fresh cross-device copy), never via
                # views_from_packed_copy — skip the scratch-pool repack, which
                # regressed decode throughput when applied unconditionally.
                views = load_expert_into_slab(
                    shard_file, metas, slab,
                    allow_direct=expert_cache.arena.pin,
                    tensor_align=False,
                )
            except Exception:
                expert_cache.arena.release(slab_idx)
                raise
            if not expert_cache.put_slab(layer_idx, eidx, views, slab_idx):
                # Declined (duplicate key — the lookahead stager loaded this same
                # expert first — or no evictable room). We still own the slab AND
                # we are about to hand `views` into it to the forward, which DMAs
                # from them. Hang its release on the pin this key already holds:
                # the caller's finally unpins after the copy is issued.
                expert_cache.orphan_slab(layer_idx, eidx, slab_idx)
            return eidx, views

        # Submit and DON'T wait: hit experts compute while these reads run.
        # _moe_executor only exists for CUDA MoE (see vaayu_base.py's
        # is_moe/running_device gate) — this helper is also the module-list
        # forward's loader, which is the generic CPU-compute fallback, so it
        # must not hard-depend on that attribute. Mirrors the same fallback
        # in _make_fused_sparse_forward's own miss loader.
        _ex = getattr(vaayu_ref, "_moe_executor", None)
        if _ex is not None:
            miss_futures = [_ex.submit(_load_one, e) for e in misses]
        else:
            # No persistent executor (CPU MoE / mock vaayu_ref in unit tests) —
            # build the fallback pool once and cache it on vaayu_ref instead of
            # per-call, mirroring why _moe_executor itself is created once at
            # model init rather than per-layer-per-token (see the comment at
            # its construction site in vaayu_base.py).
            _tmp_ex = getattr(vaayu_ref, "_moe_fallback_executor", None)
            if _tmp_ex is None:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                # Fixed width, NOT min(len(misses), 8): this pool is cached on
                # vaayu_ref for the model's lifetime (see above), so sizing it
                # from the first call's miss count would permanently freeze it
                # at that width — a later call with more misses would then run
                # its disk loads serially, defeating _iter_expert_loads'
                # FIRST_COMPLETED overlap. get_expert_pool_size() mirrors how
                # the persistent _moe_executor is sized in vaayu_base.py.
                try:
                    from .persist.safetensor_model_persister import get_expert_pool_size as _gps
                    _pool_n = _gps()
                except ImportError:
                    _pool_n = 8
                _tmp_ex = _TPE(max_workers=_pool_n)
                vaayu_ref._moe_fallback_executor = _tmp_ex
                atexit.register(_tmp_ex.shutdown, wait=False)
            miss_futures = [_tmp_ex.submit(_load_one, e) for e in misses]

    return expert_tensors, miss_futures, pinned_keys


def _iter_expert_loads(expert_tensors: dict, miss_futures: list, vaayu_ref, mp=None):
    """Yield (eidx, raw): cache hits immediately (compute overlaps the in-flight
    miss reads), then misses in completion order. Blocking time in the wait() is
    the true disk stall and is accumulated into vaayu_ref._expert_disk_s
    (telemetry: stall, not total read time)."""
    # Both branches filter None so a producer that puts an optional value in
    # expert_tensors (or whose future resolves to a None raw) can't reach the
    # compute loop and raise deep inside it — it's just skipped instead.
    for _e, _raw in expert_tensors.items():
        if _raw is not None:
            yield _e, _raw
    _pending = set(miss_futures)
    while _pending:
        _w0 = time.perf_counter()
        _done, _pending = _futures_wait(_pending, return_when=FIRST_COMPLETED)
        _stall = time.perf_counter() - _w0
        vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
        if mp is not None:
            mp['disk_load'] += _stall
        for _f in _done:
            _e, _raw = _f.result()
            if _raw is not None:
                yield _e, _raw


def _make_fused_sparse_forward_exl3(
    block_info: MoEBlockInfo,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
):
    """Sparse forward for EXL3 quantized MoE.

    Loads ~88 MB quantized tensors per expert, builds a temporary BC_LinearEXL3,
    runs ExLlamaV3 CUDA kernel, frees immediately.
    PCIe cost: 88 MB / 2 GB/s = 44 ms per expert (vs 168 ms BF16).
    """
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import (
        views_from_packed_copy, gpu_repack_aligned)

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def shard_file_for(name: str) -> str:
        return str(checkpoint_path / f"{name}.safetensors")

    def sparse_forward(hidden_states, **kwargs):
        _ensure_prefill_headroom(vaayu_ref, layer_idx)
        _prof = getattr(vaayu_ref, 'profiling_mode', False) and getattr(vaayu_ref, '_moe_profile', None) is not None
        _mp = vaayu_ref._moe_profile if _prof else None

        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

        # Router on GPU (pinned)
        router_res = router(flat_hidden)
        if isinstance(router_res, tuple) and len(router_res) >= 3:
            router_logits, routing_weights, expert_indices = router_res[:3]
            _warn_cache_bias_inert(vaayu_ref, "the router returned its own top-k")
        else:
            router_logits = router_res
            if callable(getattr(moe_mod, "route_tokens_to_experts", None)):
                # Model defines its own scoring function (sigmoid / correction
                # bias / group routing — DeepSeek, GLM-MoE, MiniMax family).
                # It owns routing; cache-aware biasing cannot apply to a scoring
                # function we don't own, and softmax here would be simply wrong.
                routing_weights, expert_indices = route_experts(
                    moe_mod, router_logits, top_k)
                _warn_cache_bias_inert(
                    vaayu_ref, "the model owns its scoring function "
                               "(route_tokens_to_experts)")
            else:
                # Cache-aware routing (advanced-opt #4): bias top-k toward resident
                # experts. ONLY in this branch — the tuple branch above means the
                # model returned its OWN top-k (possibly grouped / correction-biased,
                # e.g. DeepSeek/MiniMax family) and must never be re-routed.
                # _build_resident_mask returns None when the feature is off, so
                # cache_aware_topk reduces to plain softmax-topk (no behavior change).
                _res_mask = _build_resident_mask(
                    vaayu_ref, expert_cache, layer_idx,
                    router_logits.shape[-1], router_logits.device)
                routing_weights, expert_indices = cache_aware_topk(
                    router_logits, _res_mask, top_k,
                    getattr(vaayu_ref, "_cache_bias", 0.0))

        if _prof:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        expert_indices_cpu = expert_indices.cpu()
        unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
            vaayu_ref.telemetry.record_batch_overlap(
                layer_idx, len(unique_experts), vaayu_ref._batch_n * top_k)

        # Lookahead routing predictor (Task 12, decode only) — shared with the
        # module-list forward, see _lookahead_step. Timed inside 'mask_sync'.
        _lookahead_step(vaayu_ref, expert_cache, offset_index, checkpoint_path,
                        layer_idx, flat_hidden, top_k, unique_experts)

        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0

        shard_file = shard_file_for(layer_name)

        # Prefill lookahead: if the stager pre-read this layer, WAIT for it
        # (that wait IS the disk time — no duplicate reads) and consume it.
        _staged_layer = None
        _staged_slabs = {}
        _is_last_call = True
        _cm_gpu = None
        if (getattr(vaayu_ref, "_moe_prefill_pass", False)
                and getattr(vaayu_ref, "_prefill_stager_active", False)):
            _ev = vaayu_ref._prefill_stage_events.get(layer_idx)
            if _ev is not None:
                if getattr(vaayu_ref, "_overlap_trace", False):
                    print(f"[VAAYU-OVERLAP] wait_start layer={layer_idx} "
                          f"t={time.perf_counter():.6f}")
                _w0 = time.perf_counter()
                _ev.wait()
                _w1 = time.perf_counter()
                _stall = _w1 - _w0
                vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
                if getattr(vaayu_ref, "_overlap_trace", False):
                    print(f"[VAAYU-OVERLAP] wait_end layer={layer_idx} "
                          f"t={_w1:.6f} stall={_stall:.6f}")
                # Chunk-major prefill runs this layer once per chunk: PEEK the
                # staged dict; pop + release only on the final chunk's call.
                _n_calls = getattr(vaayu_ref, "_chunk_major_n", None) or 1
                _cnt = getattr(vaayu_ref, "_staged_consume_count", {})
                _seen = _cnt.get(layer_idx, 0) + 1
                _cnt[layer_idx] = _seen
                _is_last_call = _seen >= _n_calls
                # Chunk-major H2D retention: keep this layer's GPU expert tensors
                # alive across its chunk calls (all chunks run back-to-back within
                # the layer), so each expert uploads once per layer instead of
                # once per chunk. Freed at the last chunk call (below).
                if _n_calls > 1:
                    if _seen == 1:
                        vaayu_ref._chunk_major_gpu = {}
                        vaayu_ref._chunk_major_gpu_bytes = 0
                if _is_last_call:
                    _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
                else:
                    _staged_layer = vaayu_ref._prefill_staged.get(layer_idx)
                _staged_slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).get(layer_idx, {})
                if _staged_layer is not None and _is_last_call:
                    vaayu_ref._prefill_stage_sem.release()  # free a stage slot

        _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
        # Prefill: don't churn the GPU LRU with a whole-model sweep.
        if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
            _vram = None
        # Layer-range restriction (Fix #7 follow-up): only layers in the
        # cacheable prefix use the VRAM cache; None means no restriction.
        _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
        if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
            _vram = None

        # Check GPU residency BEFORE dispatching, so a resident expert that has
        # fallen out of the RAM cache is not read off NVMe for nothing (P8).
        # The entry is HELD for the whole compute loop, exactly as the
        # module-list forward holds its _vram_resident entries: eviction only
        # pops the dict and drops its reference, so a held entry stays valid
        # even if a later put() in this same layer-visit evicts it.
        _vram_hold: dict = {}
        if _vram is not None:
            for _e in unique_experts:
                _held = _vram.get(layer_idx, _e)
                if _held is not None:
                    _vram_hold[_e] = _held

        expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            unique_experts,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref, staged_layer=_staged_layer,
            vram_resident=set(_vram_hold) or None,
        )

        if _prof:
            _t0 = time.perf_counter()
        # Permutation dispatch: one CPU argsort + ONE H2D of the index pairs per
        # layer, replacing the int64 one_hot mask and the two tiny per-expert
        # .to(device) copies (~1,500 launches/token).
        _pair, _starts, _counts_l = compute_expert_dispatch(
            expert_indices_cpu.view(-1, expert_indices_cpu.shape[-1]),
            experts_mod.num_experts)
        _pair_gpu = _pair.to(device, non_blocking=True)
        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0

        # fp32 accumulator, cast back at the return. Experts land in
        # index_add_ order, and that order is NOT deterministic: cache-resident
        # experts run first (residency varies with warmth) and the rest arrive
        # in disk COMPLETION order (_iter_expert_loads waits FIRST_COMPLETED).
        # Accumulating in bf16 makes that order loudly observable -- bf16
        # addition rounds at ~2^-8, so a reordered sum differs by ~1 ULP per
        # layer and compounds into different tokens over a deep stack. fp32
        # rounds at ~2^-24 and shrinks that error ~2^13x.
        #
        # It does NOT remove it. This comment used to claim "the cast result is
        # identical whatever the order"; that was disproven on Qwen3-235B
        # 2026-08-15 -- three identical greedy generate runs gave TWO distinct
        # token sequences. fp32's error is normally far below the bf16 output's
        # own precision, but a value sitting on a bf16 rounding boundary tips
        # regardless, and across 94 layers something always does. Width narrows
        # the window; only a fixed summation order closes it, which is what
        # accumulate_expert_outputs now does.
        final_hidden = torch.zeros_like(flat_hidden, dtype=torch.float32)
        # See the module-list forward and accumulate_expert_outputs: summed in
        # expert-id order once the loop is done, so a run-to-run change in
        # arrival order cannot change the result.
        _contribs: list = []
        intermediate_size = None  # inferred lazily from the first expert we see

        if getattr(vaayu_ref, "_overlap_trace", False):
            print(f"[VAAYU-OVERLAP] compute_start layer={layer_idx} "
                  f"t={time.perf_counter():.6f} n_experts={len(unique_experts)}")
        try:
            for eidx, raw in _iter_expert_loads(expert_tensors, _miss_futures, vaayu_ref, _mp if _prof else None):
                # No CPU tensors: dispatch suppressed this expert's read because
                # it is already in _vram_hold (P8). Its shapes come off the GPU
                # copy instead — the same tensors, uploaded on an earlier visit.
                _no_raw = raw is VRAM_RESIDENT_NO_RAW
                if _no_raw:
                    groups = None
                else:
                    groups = _group_exl3_tensors_by_weight(raw)
                    if not groups["gate"] or not groups["down"] or not groups["up"]:
                        _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                        continue
                if intermediate_size is None:
                    _t_gate = (_vram_hold[eidx]["gate"] if _no_raw else groups["gate"]).get("trellis")
                    if _t_gate is None:
                        continue
                    intermediate_size = _t_gate.shape[1] * 16

                _s, _c = _starts[eidx], _counts_l[eidx]
                if _c == 0:
                    continue
                token_idx_gpu = _pair_gpu[0, _s:_s + _c]
                top_k_pos_gpu = _pair_gpu[1, _s:_s + _c]
                current_state = flat_hidden[token_idx_gpu]
                if current_state.dtype != torch.float16:
                    current_state = current_state.to(torch.float16)

                # Held since before dispatch (one LRU promotion per expert per
                # visit, as before — the lookup simply moved earlier).
                _cached_gpu = _vram_hold.get(eidx)
                _cm_gpu = getattr(vaayu_ref, "_chunk_major_gpu", None)
                if _cached_gpu is None and _cm_gpu is not None:
                    _cached_gpu = _cm_gpu.get(eidx)   # retained from an earlier chunk
                _entry = None
                if _cached_gpu is not None:
                    gpu_gate, gpu_up, gpu_down = _cached_gpu["gate"], _cached_gpu["up"], _cached_gpu["down"]
                    _owned_by_cache = True
                    _entry = _cached_gpu
                else:
                    if _prof:
                        torch.cuda.synchronize(); _t0 = time.perf_counter()
                    _sidx = _staged_slabs.get(eidx)
                    if _sidx is not None:
                        # Slab-backed staged expert (prefill): ONE pinned H2D of the
                        # packed slab prefix, then free GPU re-slicing — replaces 9
                        # per-component copies (Task 9).
                        _slab_cpu = expert_cache.arena.tensor(_sidx)
                        _base = _slab_cpu.data_ptr()
                        _used = max((v.data_ptr() - _base) + v.nbytes for v in raw.values())
                        # ONE H2D of the raw slab prefix (volume unchanged), then
                        # views_from_packed_copy reslices the RAW (tensor_align=False)
                        # layout into GPU views — valid as copy SOURCES (a copy_'s
                        # source needs only itemsize alignment, which even offsets
                        # satisfy), NOT as kernel inputs. gpu_repack_aligned then
                        # packs each role onto 16-aligned CUDA offsets for EXL3 (B1).
                        _gpu_blob = _slab_cpu[:_used].to(device, non_blocking=True)
                        # Guard slab reuse against this in-flight async H2D read
                        # (arena.acquire() will sync on it before rewriting _sidx).
                        expert_cache.arena.record_read(_sidx)
                        gpu_gate = gpu_repack_aligned(
                            views_from_packed_copy(groups["gate"], _slab_cpu, _gpu_blob), device)
                        gpu_down = gpu_repack_aligned(
                            views_from_packed_copy(groups["down"], _slab_cpu, _gpu_blob), device)
                        gpu_up   = gpu_repack_aligned(
                            views_from_packed_copy(groups["up"],   _slab_cpu, _gpu_blob), device)
                    else:
                        gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                        gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                        gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
                        # These copies read FROM slab-backed views (decode cache
                        # hit/miss). Record a read event against the source slab so a
                        # concurrent evict→reuse (esp. the lookahead stager) can't
                        # rewrite it CPU-side before the DMA drains — silent torn-read
                        # corruption otherwise. Legacy cloned experts aren't in the
                        # arena → slab_index_of returns None → skipped.
                        _arena = expert_cache.arena
                        if _arena is not None:
                            _rd_slab = _arena.slab_index_of(next(iter(groups["gate"].values())))
                            if _rd_slab is not None:
                                _arena.record_read(_rd_slab)
                        # The async H2D copies for this expert are now issued;
                        # record_read()/acquire() guards the slab from here on, so the
                        # pin is no longer needed — release it promptly to keep
                        # eviction pressure low. The finally below covers any expert
                        # that never reaches this point (VRAM hit, continue, raise).
                        _uk = (layer_idx, eidx)
                        if _uk in _pinned_keys:
                            expert_cache.unpin(layer_idx, eidx)
                            _pinned_keys.discard(_uk)
                    if _prof:
                        torch.cuda.synchronize(); _mp['h2d'] += time.perf_counter() - _t0
                    if _vram is not None:
                        # put() returns True only if the cache actually accepted ownership
                        # (it can refuse if this exact key raced in, or if a single expert
                        # is larger than the whole budget) — only then do we skip the free.
                        _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                        _owned_by_cache = _vram.put(layer_idx, eidx, _entry)
                        if not _owned_by_cache:
                            _entry = None
                    elif _cm_gpu is not None:
                        # Chunk-major retention (prefill: _vram is gated off). Budget
                        # bound is the same hardware-derived VRAM cache budget; overflow
                        # experts simply re-upload on later chunks.
                        _sz = sum(t.nbytes for d in (gpu_gate, gpu_up, gpu_down)
                                  for t in d.values())
                        _budget = getattr(getattr(vaayu_ref, "_expert_vram_cache", None),
                                          "budget_bytes", 0)
                        if vaayu_ref._chunk_major_gpu_bytes + _sz <= _budget:
                            _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                            _cm_gpu[eidx] = _entry
                            vaayu_ref._chunk_major_gpu_bytes += _sz
                            _owned_by_cache = True
                        else:
                            _owned_by_cache = False
                    else:
                        _owned_by_cache = False

                bc_gate = bc_down = bc_up = None
                try:
                    _bc_cached = _entry.get("bc") if _entry is not None else None
                    if _bc_cached is not None:
                        bc_gate, bc_up, bc_down = _bc_cached
                    else:
                        if _prof:
                            _t0 = time.perf_counter()
                        bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                        bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                        bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)
                        if _prof:
                            torch.cuda.synchronize(); _mp['construct'] += time.perf_counter() - _t0
                        if _entry is not None:
                            # Cache the BC objects with the entry: they wrap the SAME
                            # resident GPU tensors, so their lifetime matches; eviction
                            # drops the tuple and the pybind destructors free the C++
                            # heap. Mutation outside the cache lock is safe — the decode
                            # compute loop is the sole consumer of VRAM-cache entries.
                            _entry["bc"] = (bc_gate, bc_up, bc_down)

                    if _prof:
                        _t0 = time.perf_counter()
                    x_flat = current_state.view(-1, H)
                    gate_out = act_fn(bc_gate.run_alloc(x_flat, intermediate_size, False))
                    up_out   = bc_up.run_alloc(x_flat, intermediate_size, False)
                    hidden   = bc_down.run_alloc((gate_out * up_out).contiguous(), H, False)
                    if _prof:
                        torch.cuda.synchronize(); _mp['gemm'] += time.perf_counter() - _t0

                    weights_slice = routing_weights[token_idx_gpu, top_k_pos_gpu, None]
                    hidden = hidden * weights_slice
                    _contribs.append((eidx, token_idx_gpu, hidden))
                finally:
                    # Free C++ QMatrix objects ONLY when the cache does not hold them:
                    # cached BC objects (entry["bc"]) live until entry eviction. For
                    # uncached experts the local refs are the only refs — null them so
                    # the pybind destructors run now.
                    if not _owned_by_cache:
                        bc_gate = bc_down = bc_up = None
                        del gpu_gate, gpu_down, gpu_up
        finally:
            # Guarantee every key THIS call pinned is released, even if compute
            # raised partway through — a leaked pin would permanently block that
            # key's eviction (a slow cache-capacity leak). Early-unpin above
            # already released experts whose H2D copy issued; this drains the
            # rest (VRAM hits, skipped/continued experts, exceptions).
            for _pk in _pinned_keys:
                expert_cache.unpin(*_pk)

        if getattr(vaayu_ref, "_overlap_trace", False):
            print(f"[VAAYU-OVERLAP] compute_end layer={layer_idx} "
                  f"t={time.perf_counter():.6f}")

        if intermediate_size is None:
            _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx}: cannot determine intermediate_size, returning zeros")

        # Give staged slabs back to the arena once this layer's compute (and
        # therefore all its H2D copies, which are ordered on the same stream)
        # has completed. synchronize() is required: the next user of the slab
        # writes it CPU-side via io_uring, which the CUDA stream can't see.
        if _staged_layer is not None and _is_last_call:
            _slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).pop(layer_idx, None)
            if _slabs:
                _arena = expert_cache.arena
                if _arena is not None:
                    if torch.cuda.is_available():
                        # Stream-scoped wait, not a full-device drain (CR-3):
                        # the staged-consume H2Ds that read these slabs are
                        # already enqueued on the current stream; an event
                        # recorded here and synchronized blocks the host until
                        # exactly those complete, so the CPU-side io_uring write
                        # that reuses a released slab can't race the H2D — WITHOUT
                        # draining unrelated device work (e.g. the background
                        # stager's own in-flight copies).
                        _rel_ev = torch.cuda.Event()
                        _rel_ev.record()
                        _rel_ev.synchronize()
                    for _sidx in _slabs.values():
                        if _sidx is not None:
                            _arena.release(_sidx)

        if _is_last_call and getattr(vaayu_ref, "_chunk_major_gpu", None) is not None:
            # Drop the retained GPU tensors + BC objects; the CUDA caching
            # allocator reuses the memory for the next layer's retention dict.
            vaayu_ref._chunk_major_gpu = None
            vaayu_ref._chunk_major_gpu_bytes = 0

        accumulate_expert_outputs(final_hidden, _contribs)

        _shared = _shared_expert_output(moe_mod, flat_hidden)
        if _shared is not None:
            final_hidden += _shared.to(final_hidden.dtype)

        vaayu_ref._expert_history[layer_idx] = unique_experts
        final_hidden = final_hidden.to(flat_hidden.dtype)
        _trace_moe_block(vaayu_ref, layer_idx, "fused_exl3", flat_hidden,
                         final_hidden, routing_weights, unique_experts)
        return final_hidden.view(B, S, H)

    return sparse_forward

