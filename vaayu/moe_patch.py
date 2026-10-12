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

from .packed_expert_kernels import (packed_experts_decode, packed_experts_decode_graphed,
                                    packed_experts_prefill)

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


def truncate_top_k(weights, indices, k, p=None):
    """Decode-time expert budget on an already-routed (weights, indices) pair.
    Row 33 / T2, and DECODE_TOP_K="auto".

    `k`: keep the `k` largest-weight slots. `p`: the routing-mass cutoff --
    keep, per row, the smallest set of slots whose normalised weight sums to
    >= p (always >= 1), and RE-POINT each dropped slot at that row's top-1
    expert with weight 0. Shapes are unchanged, so no forward's dispatch
    changes, and a dropped slot loads nothing (it duplicates an expert already
    routed). The top-1 is found by WEIGHT, not position: MiniMax routes with
    `sorted=False`. If both are given, `k` applies first.

    Kept weights are rescaled to each row's ORIGINAL sum, not to 1 (D1).
    Routers that multiply by `routed_scaling_factor` (DeepSeek-V2/V3,
    GLM-4-MoE, dots1, ...) or skip renormalisation do not sum to 1, and forcing
    1 would silently rescale every MoE block's output by 1 / original_sum.
    A sum that misses 1 only by rounding is snapped to exactly 1.0, so
    unit-sum routers stay bit-identical to plain renormalisation: on device a
    <= 1.8e-7 weight change moved Qwen3-Next's decode perplexity by 0.6 %.

    `k is None or k >= K` with `p is None` returns the SAME tensors, not equal
    copies: expert accumulation is order-dependent, so re-running topk on a
    path the user did not turn on would be a bit-level change."""
    num_slots = weights.shape[-1]
    if (k is None or k >= num_slots) and p is None:
        return weights, indices
    row_sum = weights.float().sum(dim=-1, keepdim=True)
    # Numerical tolerance, not a model number: bf16 rounding leaves a unit-sum
    # row up to 2.44e-3 off 1 (measured, D1); 1e-2 keeps a 4x margin.
    row_sum = torch.where((row_sum - 1.0).abs() < 1e-2,
                          torch.ones_like(row_sum), row_sum)
    if k is not None and k < num_slots:
        w, pos = torch.topk(weights.float(), k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True) * row_sum
        weights, indices = w.to(weights.dtype), indices.gather(-1, pos)
    if p is None:
        return weights, indices
    w = weights.float()
    w = w / w.sum(dim=-1, keepdim=True)
    sorted_w, order = torch.sort(w, dim=-1, descending=True)
    keep_sorted = (sorted_w.cumsum(dim=-1) - sorted_w) < p
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    top1 = indices.gather(-1, order[..., :1]).expand_as(indices)
    w = torch.where(keep, w, torch.zeros_like(w))
    w = w / w.sum(dim=-1, keepdim=True) * row_sum
    return w.to(weights.dtype), torch.where(keep, indices, top1)


def _decode_limits(vaayu_ref):
    """The decode-time `(k, p)` budget, or `(None, None)` when it must not
    apply.

    Decode-only by design (row 33 / T2): prefill routes ~every expert anyway
    (224.23 per layer per chunk, measured), so truncating it buys nearly
    nothing and costs prompt quality."""
    if getattr(vaayu_ref, "_moe_prefill_pass", False):
        return None, None
    return (getattr(vaayu_ref, "_decode_top_k", None),
            getattr(vaayu_ref, "_decode_top_p", None))



def _routing_to_host(expert_indices):
    """S28 C11: a MoE visit's routing on the host through ONE device-to-host sync.

    Returns (the ids' CPU copy, the routed experts ascending) -- the same list
    `expert_indices.flatten().unique().tolist()` gives, which costs a device sort (3 cub
    kernels) and two syncs (its output's size, then the copy) before the dispatch's own
    `.cpu()` adds a third. One token's ids are a handful, where a Python set beats a CPU
    `torch.unique` call; a prefill chunk's are thousands, where the reverse holds."""
    cpu = expert_indices.cpu()
    if cpu.numel() <= cpu.shape[-1]:
        return cpu, sorted(set(cpu.reshape(-1).tolist()))
    return cpu, cpu.unique().tolist()


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


def _timing_mark(timing) -> float:
    """perf_counter, synchronised first when a profiling `timing` dict is live -- the
    split between an async copy and the work after it is only truthful after a sync."""
    if timing is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _packed_roles_complete(packed: dict) -> bool:
    """Every role of a `_group_packed_tensors_by_weight` result has its codes and scales."""
    return all("weight_packed" in packed[r] and "weight_scale" in packed[r]
               for r in ("gate", "down", "up"))


def _upload_packed_expert(packed: dict, device: str, timing: dict = None) -> dict:
    """On-device copies of one pack-quantized expert's components (weight_shape stays on the
    host: it is read, never computed on). The H2D half of `_expert_dense_weights`, shared
    with the S7 kernel branch, which skips the dequant half. Books `timing["h2d"]`."""
    _t0 = _timing_mark(timing)
    on_device = {
        r: {k: (v if k == "weight_shape" else v.to(device, non_blocking=True))
            for k, v in packed[r].items()}
        for r in ("gate", "down", "up")
    }
    if timing is not None:
        timing["h2d"] += _timing_mark(timing) - _t0
    return on_device


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

    Returns (w_gate, w_down, w_up, entry), where `entry` is what the VRAM expert
    cache should store for this expert — the PACKED BLOB for a pack-quantized
    checkpoint (the checkpoint's own bytes, so a hit skips this whole function),
    the dense weights for anything else, None when there is nothing cacheable.

    Returns (None, None, None, None) if any of the three roles is missing.
    """
    packed = _group_packed_tensors_by_weight(raw)
    if packed["gate"]:
        roles = ("gate", "down", "up")
        if not _packed_roles_complete(packed):
            return None, None, None, None
        on_device = _upload_packed_expert(packed, device, timing)
        _t1 = _timing_mark(timing)
        # ONE dequant pass for the whole expert, not three. The cost is kernel
        # launches, not arithmetic -- 114.3 us/expert for three per-linear calls
        # against 17.6 for one compiled pass, on work a trivial kernel does in
        # 2.7 (measured on a real expert, 300 reps). At 480 routed experts per
        # token that is ~46 ms of a ~244 ms token.
        blob = build_expert_blob(on_device)
        if blob is None:
            dense = {r: dequantize_packed_weight(on_device[r], dtype) for r in roles}
        else:
            dense = dequantize_expert_blob(blob, dtype)
        if timing is not None:
            timing["construct"] += _timing_mark(timing) - _t1
        return dense["gate"], dense["down"], dense["up"], blob

    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    if w_gate is None or w_down is None or w_up is None:
        return None, None, None, None
    _t0 = _timing_mark(timing)
    dense = (w_gate.to(device, dtype=dtype, non_blocking=True),
             w_down.to(device, dtype=dtype, non_blocking=True),
             w_up.to(device, dtype=dtype, non_blocking=True))
    if timing is not None:
        timing["h2d"] += _timing_mark(timing) - _t0
    # Nothing to dequantize on this path, so the cache entry IS the weights.
    return dense + ({"dense": dict(zip(("gate", "down", "up"), dense))},)


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


_FUSED_DEQUANT_CACHE: dict = {}


def _fused_dequant_kernel(flat_packed, flat_scale, num_bits, group_size, dtype):
    """Every code in one expert, unpacked and scaled in a single pass.

    Correct only under the preconditions `dequantize_expert_weights` checks:
    `32 % num_bits == 0`, every role packed at the same bit width and the same
    group size, and each role's `cols` a whole number of codes per int32 word.
    Given those, the three projections concatenate into one flat code stream
    whose consecutive runs of `group_size` line up with the flat scales.

    The offset is folded into the float subtract instead of taking the int8 hop
    the per-linear path uses -- measured 114.3 -> 75.7 us/expert, and asserted
    BIT-IDENTICAL by tests/test_packed_expert_dequant.py rather than assumed
    (bf16 represents every code in 0..255 exactly, so the two orders agree).
    """
    per_word = 32 // num_bits
    shifts = torch.arange(per_word, device=flat_packed.device,
                          dtype=torch.int32) * num_bits
    codes = (flat_packed.unsqueeze(-1) >> shifts) & ((1 << num_bits) - 1)
    codes = codes.view(-1, group_size).to(dtype) - (1 << (num_bits - 1))
    return (codes * flat_scale.reshape(-1, 1).to(dtype)).view(-1)


def _fused_dequant_fn(num_bits, group_size, dtype, device_type):
    """`_fused_dequant_kernel`, compiled once per shape on CUDA.

    The remaining cost is kernel LAUNCHES, not arithmetic: 75.7 us/expert for
    work a trivial kernel does in 2.7, falling to 17.6 compiled (measured on one
    real expert, 300 reps). Compiling is therefore worth it exactly where launch
    overhead exists, so CPU keeps the eager function -- and a dynamo failure
    falls back to it rather than taking the model down.
    """
    if device_type != "cuda":
        return _fused_dequant_kernel
    key = (num_bits, group_size, dtype)
    if key not in _FUSED_DEQUANT_CACHE:
        try:
            _FUSED_DEQUANT_CACHE[key] = torch.compile(_fused_dequant_kernel, dynamic=False)
        except Exception as exc:                      # pragma: no cover - env-dependent
            _log.warning("torch.compile unavailable for the expert dequant (%s); "
                         "running it eagerly", exc)
            _FUSED_DEQUANT_CACHE[key] = _fused_dequant_kernel
    return _FUSED_DEQUANT_CACHE[key]


def build_expert_blob(by_role: dict) -> "dict | None":
    """One expert's pack-quantized tensors as a single blob, or None.

    `flat_packed` / `flat_scale` concatenated ONCE, plus the geometry needed to
    slice the decoded stream back into roles. This is what the VRAM expert cache
    stores (S2b (iii)): the entry is the checkpoint's own bytes, so a hit skips
    the host-to-device copy AND the two `torch.cat` calls, leaving only the
    compiled fused kernel -- measured 55.5 -> 31.8 us/expert on 64 real experts
    interleaved with the forward's own gemms (2026-09-13).

    Returns None when this expert cannot be decoded in one fused pass -- a dense
    (non-packed) expert, or roles packed at different bit widths / group sizes.
    Decoding those through one flat group-size view would decode them WRONGLY,
    so the compatibility test is a precondition, not an optimisation hint, and
    the caller falls back to the per-role path.
    """
    geom = []
    for role, comps in by_role.items():
        if "weight_zero_point" in comps or "weight_g_idx" in comps:
            # RC1's refusal, restated here so neither the fused path nor the
            # cache can lose it.
            raise NotImplementedError(
                "asymmetric / act-order pack-quantized experts carry a zero point or "
                "g_idx that Vaayu does not apply; this checkpoint would decode to "
                "shifted weights. Supported today: symmetric group quantization.")
        if not {"weight_packed", "weight_scale", "weight_shape"} <= set(comps):
            return None
        rows, cols = (int(x) for x in comps["weight_shape"].tolist())
        num_bits, remainder = divmod(comps["weight_packed"].numel() * 32, rows * cols)
        geom.append((role, rows, cols, num_bits,
                     cols // comps["weight_scale"].shape[-1], remainder))

    if not geom:
        return None
    num_bits, group_size = geom[0][3], geom[0][4]
    fusable = (
        num_bits and 32 % num_bits == 0
        and all(r == 0 and nb == num_bits and g == group_size
                and c % (32 // num_bits) == 0
                for _role, _rows, c, nb, g, r in geom)
    )
    if not fusable:
        return None

    roles = [g[0] for g in geom]
    return {
        "flat": {
            "packed": torch.cat([by_role[r]["weight_packed"].reshape(-1) for r in roles]),
            "scale": torch.cat([by_role[r]["weight_scale"].reshape(-1) for r in roles]),
        },
        # Derived from the tensors, never a config read (CLAUDE.md 5).
        "geom": (num_bits, group_size, tuple((g[0], g[1], g[2]) for g in geom)),
    }


def dequantize_expert_blob(blob: dict, dtype: "torch.dtype") -> dict:
    """{role: dense weight} from a blob built by `build_expert_blob`.

    All that is left of the dequant once the concatenation is done -- one
    compiled kernel over the flat code stream, then metadata-only role views.
    """
    num_bits, group_size, roles = blob["geom"]
    flat_packed, flat_scale = blob["flat"]["packed"], blob["flat"]["scale"]
    fn = _fused_dequant_fn(num_bits, group_size, dtype, flat_packed.device.type)
    flat = fn(flat_packed, flat_scale, num_bits, group_size, dtype)

    out, at = {}, 0
    for role, rows, cols in roles:
        out[role] = flat[at:at + rows * cols].view(rows, cols)
        at += rows * cols
    return out


def expert_weights_from_entry(entry: dict, dtype: "torch.dtype") -> tuple:
    """(w_gate, w_down, w_up) for one VRAM-cached expert.

    The mirror of the `entry` `_expert_dense_weights` returns: a packed blob
    still owes the compiled fused kernel (measured 31.8 us/expert on 64 real
    experts interleaved with the forward's own gemms, against 55.5 for the
    upload path's cat-and-decode), dense weights are already the weights.
    Either way the host-to-device copy is skipped, which is the 121.9 us.
    """
    dense = entry["dense"] if "dense" in entry else dequantize_expert_blob(entry, dtype)
    return dense["gate"], dense["down"], dense["up"]


def dequantize_expert_weights(by_role: dict, dtype: "torch.dtype") -> dict:
    """{role: dense weight} for one expert, in ONE dequant pass where possible.

    `dequantize_packed_weight` per role costs three launches of everything; the
    blob concatenates the roles and pays them once. Falls back to the per-role
    path, role by role, for any checkpoint `build_expert_blob` refuses.
    """
    blob = build_expert_blob(by_role)
    if blob is None:
        return {role: dequantize_packed_weight(comps, dtype)
                for role, comps in by_role.items()}
    return dequantize_expert_blob(blob, dtype)


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
    # Which delivery the whole-layer forward installed ("routed" | "whole"), recorded
    # by patch_moe_blocks from the forward's __vaayu_delivery__ tag (S8). None for the
    # other three forwards. env.json reports it beside forward_name.
    delivery: str | None = None


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


def fused_param_experts(experts_module) -> bool:
    """True iff this experts module holds FUSED parameters, one per projection
    with the expert count leading, rather than per-expert weights.

    Detected from the parameters' own shapes (CLAUDE.md §5), never from a
    parameter name or an architecture string: a direct parameter whose leading
    dimension equals `num_experts` and which has at least 2 dimensions.

    `recurse=False` is deliberate: a `ModuleList` of per-expert submodules
    exposes its children's parameters under `recurse=True` and would
    false-positive.

    NOT SUFFICIENT TO CHOOSE A FORWARD ON ITS OWN, and measured to be so before
    it was written: `Qwen3NextExperts` reports True here too -- its live module
    carries `gate_up_proj [512, 1024, 2048]` and `down_proj [512, 2048, 512]`,
    with and without its quantizer applied -- while its CHECKPOINT stores one
    tensor per expert, which is what the tuned fused forward streams. What
    separates the two is the checkpoint layout, so dispatch pairs this with the
    offset index's `fused_layer_tensors` for the layer; this predicate only
    answers "could this module take a whole-layer tensor at all".
    """
    num_experts = getattr(experts_module, "num_experts", None)
    if not isinstance(num_experts, int) or num_experts <= 0:
        return False
    for _name, param in experts_module.named_parameters(recurse=False):
        if param.dim() >= 2 and param.shape[0] == num_experts:
            return True
    return False


def dispatches_whole_layer(block_info: "MoEBlockInfo", offset_index) -> bool:
    """True iff this block's forward will be `_make_whole_layer_sparse_forward`.

    THE ONE PLACE the whole-layer condition is written. `_make_sparse_moe_forward`
    calls it to choose the forward, and the engine calls it before sizing the
    pinned slab arena -- because under WHOLE delivery that forward reads NEITHER
    the expert RAM cache nor the VRAM one, so an arena allocated for it is pinned,
    unswappable RAM nothing ever touches (47-m5). ROUTED delivery (S8,
    vaayu/routed_fused3d.py) does read the RAM arena, which is why the engine's
    arena guard also consults `routed_delivery_enabled`. Measured on
    gpt-oss-20b: skipping it took load RSS 19.20 -> 3.81 GB with the generated
    logits bit-identical. The arena is sized from FREE RAM AT LOAD, so the figure
    freed moves by arm (15.97 GB there, ~18.5 GB under the server gate) -- quote
    the arm, never a single number.

    Both conditions are load-bearing, and the module predicate alone does NOT
    discriminate: `Qwen3NextExperts` reports fused parameters too (measured
    2026-09-21) while its checkpoint stores one tensor per expert, which the tuned
    per-expert fused forward streams -- and that forward DOES read the arena. So
    the index's whole-tensor metas decide and the module predicate only confirms
    the module can take them. Derived from the checkpoint layout and the live
    module tree, never from an architecture string (CLAUDE.md §5).
    """
    if block_info.expert_style != "fused" or offset_index is None:
        return False
    metas = offset_index.fused_layer_tensors.get(block_info.layer_name)
    if not metas:
        return False
    return fused_param_experts(
        getattr(block_info.moe_module, block_info.experts_attr))


class PackedInjectionError(RuntimeError):
    """A streamed expert tensor could not be reinterpreted into its target's
    packed dtype.

    Distinct from every other injection failure because skipping the expert
    cannot recover it: the bytes on disk are right and the engine's handling of
    them is wrong. _inject_expert_weights re-raises this past the warn-and-skip
    every other failure takes — print-and-continue is how 769 parameters were
    left on meta while a load reported success (A5.7b).
    """


# Target dtypes an injected tensor must be REINTERPRETED into (.view), never
# numerically cast into. Two different silent corruptions live here:
#   * float4_e2m1fn_x2 — packed, two values per byte. A cast reads the packed
#     bytes as literal values, and the unpacked K axis is half the real one.
#   * float8_e8m0fnu — a scale format a kernel reads back off scale.dtype
#     (DeepseekV4Linear.forward: scale_fmt = "ue8m0" if scale.dtype is this).
#     A cast to bf16 SUCCEEDS SILENTLY and turns scale_fmt into None.
# This names a class of torch dtype, not a model number (CLAUDE.md §5). A model
# that declares none of them injects exactly as it did before this list existed.
# hasattr-guarded because moe_patch is shared infrastructure every model
# imports: on a torch without these dtypes the tuple shrinks and the injector
# keeps its old behaviour, rather than failing at import for every model.
NO_CAST_INJECTION_DTYPES: tuple = tuple(
    getattr(torch, _name)
    for _name in ("float4_e2m1fn_x2", "float8_e8m0fnu")
    if hasattr(torch, _name)
)


def _inject_no_cast(tensor, target, target_dtype, device, submod_path, base,
                    layer_name, expert_idx, pinned_source):
    """Move ONE streamed tensor to `device` and reinterpret its bytes as
    `target_dtype`, which the target parameter declared and which must never be
    cast into (see NO_CAST_INJECTION_DTYPES).

    Both guards exist because `.view(dtype)` is not defensive: with a differing
    element size it silently RESCALES the last dimension (a float32 (4, 2)
    viewed as int8 is (4, 8)), so a wrong source dtype would reach the kernel as
    a wrongly-shaped weight rather than as an error. No cast is requested, so
    _copy_async_ok can still grant a non-blocking copy from the slab arena
    (row 49) — which a cast would have forfeited.

    A third guard runs first: the element-size check alone would pass a source
    dtype that merely happens to share the target's width (e.g. float8_e4m3fn
    against a float8_e8m0fnu target — both 1 B/elem, verified: `.view` between
    them succeeds). This checkpoint ships both formats (routed-expert scales as
    e8m0, shared-expert/attention weights as e4m3), so a repack or sibling
    checkpoint that ships the wrong one against this skeleton would pass both
    the width and shape guards and have its bits silently read as the wrong
    format. Refusing anything but the target dtype itself or a raw byte
    container (uint8/int8) closes that on the source side, the same way the
    allowlist already closes it on the target side.
    """
    if tensor.dtype is not target_dtype and tensor.dtype not in (torch.uint8, torch.int8):
        raise PackedInjectionError(
            f"{layer_name} expert {expert_idx} {submod_path}.{base}: refusing to "
            f"reinterpret a {tensor.dtype} tensor as {target_dtype} -- only the "
            f"target dtype itself or a raw byte container (uint8/int8) may be "
            f"reinterpreted, and every 1-byte dtype would pass the width check")
    if tensor.element_size() != target_dtype.itemsize:
        raise PackedInjectionError(
            f"{layer_name} expert {expert_idx} {submod_path}.{base}: cannot "
            f"reinterpret {tensor.dtype} ({tensor.element_size()} B/elem) as "
            f"{target_dtype} ({target_dtype.itemsize} B/elem) — .view would "
            f"rescale the last dimension instead of refusing")
    if tuple(tensor.shape) != tuple(target.shape):
        raise PackedInjectionError(
            f"{layer_name} expert {expert_idx} {submod_path}.{base}: the "
            f"checkpoint ships {tuple(tensor.shape)} but the skeleton declares "
            f"{tuple(target.shape)} for {target_dtype}")
    try:
        return tensor.to(
            device, non_blocking=_copy_async_ok(pinned_source, tensor, None)
        ).view(target_dtype)
    except torch.cuda.OutOfMemoryError:
        # A transient VRAM shortage, not a packed-dtype defect: it must keep its
        # own type so vaayu_base's isinstance-based OOM handler can still see it,
        # and so it is not reported as "the bytes are right, the engine is wrong".
        raise
    except Exception as e:
        raise PackedInjectionError(
            f"{layer_name} expert {expert_idx} {submod_path}.{base}: "
            f"{tensor.dtype} → {target_dtype} reinterpret failed: {e}") from e


def _copy_async_ok(pinned_source: bool, tensor, dtype) -> bool:
    """Whether ONE host-to-device copy of `tensor` may run non_blocking.

    Two conditions, both necessary: the source must be page-locked (the caller
    knows — see _load_inject_run), and the copy must not carry a dtype change.
    A `.to(device, dtype=other, non_blocking=True)` casts on the HOST first, so
    the copy's real source is a fresh PAGEABLE temp, which PyTorch stages
    through its CachingHostAllocator pool — the host-RAM leak build_exl3_qmatrix's
    docstring describes, from a call site that looks page-locked. dtype None
    means "no cast requested" (build_exl3_qmatrix's copies).
    """
    return bool(pinned_source) and (dtype is None or tensor.dtype == dtype)


def _inject_expert_weights(
    expert_mod: nn.Module,
    raw: dict,
    expert_idx: int,
    layer_name: str,
    device: str,
    dtype,
    timing: dict | None = None,
    pinned_source: bool = False,
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

    pinned_source: the caller's statement that `raw`'s tensors live in the
    pinned slab arena, which is what makes an async copy both faster and safe
    (row 49). It is a claim about the MEMORY, not a request: each copy still
    goes through _copy_async_ok, which refuses async for a dtype change. False
    (the default) keeps every copy synchronous, which is what the pageable
    clone paths — prefill and the arena-exhausted fallback — need. A caller
    that passes True MUST also arm the arena's read event for the source slab
    (SlabArena.record_read) before the slab can be recycled.
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
                # Async only from the pinned slab arena (row 49): a pageable
                # source would be staged through PyTorch's CachingHostAllocator
                # pool, which leaks host RAM — see build_exl3_qmatrix's
                # docstring. No dtype is passed here, so no host-side cast can
                # turn a page-locked source back into a pageable one.
                state[submod_path] = build_exl3_qmatrix(
                    submod, bases, device,
                    non_blocking=_copy_async_ok(
                        pinned_source, next(iter(bases.values())), None),
                    timing=timing)
                bc_objs[submod_path] = submod.bc
                continue

            for base, tensor in bases.items():
                if timing is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                # The target parameter's OWN declared dtype decides this, read
                # off the live module: one attribute read per streamed tensor
                # (~6 per expert) instead of self.model.get_parameter()'s string
                # walk through the whole tree, in the decode hot path. Safe
                # because the dtype is a fixpoint of this forward's cycle —
                # .to("meta") in _detach_expert preserves it, and
                # _attach_cached_expert re-binds an already-reinterpreted
                # tensor. None (no such attribute, as on an EXL3-swapped
                # ExLlamaV3Linear, or a parameter registered None) is not in the
                # tuple, so it takes the cast path unchanged.
                target = getattr(submod, base, None)
                target_dtype = target.dtype if target is not None else None
                if target_dtype in NO_CAST_INJECTION_DTYPES:
                    t = _inject_no_cast(tensor, target, target_dtype, device,
                                        submod_path, base, layer_name,
                                        expert_idx, pinned_source)
                else:
                    t = tensor.to(device, dtype=dtype,
                                  non_blocking=_copy_async_ok(pinned_source, tensor, dtype))
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
        except PackedInjectionError:
            # Not skippable: the disk bytes are right and this engine's handling
            # of them is wrong, so every expert in every layer would hit it.
            # _load_inject_run's `finally` still runs _detach_expert, so nothing
            # is left bound off-meta on the way out.
            raise
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
        block_info.delivery = getattr(sparse_fwd, "__vaayu_delivery__", None)
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
        # Whole-layer delivery, ahead of the quant split: the question is where the
        # CHECKPOINT keeps this layer's experts, not how they are quantised. The
        # condition lives in dispatches_whole_layer() because the engine's arena
        # guard has to ask the same question before it pins RAM this forward never
        # reads (47-m5); written twice, the two drift.
        if dispatches_whole_layer(block_info, offset_index):
            fwd = _make_whole_layer_sparse_forward(
                block_info, offset_index, checkpoint_path, device, vaayu_ref,
                original_forward=original_forward, expert_cache=expert_cache,
            )
            fwd.__vaayu_forward__ = "_make_whole_layer_sparse_forward"
            return fwd
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
# by evicting the VRAM expert cache (prefill reads it, never fills it). A property of the card, not
# the model (CLAUDE.md §5) — same CLASS of constant as vaayu_base._VRAM_RESERVE_FLOOR_BYTES
# (both are hardware headroom reserves), but a deliberately DIFFERENT value (2 GB
# here vs 3.5 GB there): this one only needs to exceed one layer's prefill-chunk
# activation, since the poll runs per layer, while the other must additionally
# cover the decode-boundary cache re-warm + sampling transient (see its comment
# in vaayu_base.py). Do not conflate the two or assume they move together.
_PREFILL_VRAM_HEADROOM = 2 * 1024 ** 3


def _format_vram_diag(layer_idx, free, alloc, reserved,
                      cache_usage, cache_budget, prefill_evict, prefill_hit):
    """One grep-able diagnostic line for the VRAM prefill trajectory. All byte
    inputs; emits GB. Pure — no torch/CUDA calls, so it is unit-testable."""
    g = 1024 ** 3
    return (f"[VRAM-DIAG] layer={layer_idx} "
            f"free={free / g:.2f}GB alloc={alloc / g:.2f}GB "
            f"reserved={reserved / g:.2f}GB "
            f"cache={cache_usage / g:.2f}/{cache_budget / g:.2f}GB "
            f"prefill_evict={prefill_evict} prefill_hit={prefill_hit}")


def _ensure_prefill_headroom(vaayu_ref, layer_idx=-1) -> None:
    """During a prefill pass, evict the VRAM expert cache until free VRAM meets
    _PREFILL_VRAM_HEADROOM. Activations come first. A prefill pass reads the
    tier but never fills it (S17, _vram_tier_visit), so an entry evicted here
    costs that prefill one upload it would otherwise have skipped, and re-warms
    during the decode that follows. This runs at forward entry, before the visit
    takes its entries, so it never evicts one the visit is computing with.
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
        # S35-E: on a request the elastic tier lends to, the whole predicted peak was
        # in driver-free VRAM at its grant, so a low reading here is usually the
        # allocator's own cache (each prefill chunk's score buffer is larger than the
        # last, and the old ones stay cached: 4.2 GiB behind an 8k prompt). Return
        # that and look again before evicting -- measured without this: 233 of 582
        # tier entries evicted during one cold 8k prefill. A request that is lent
        # nothing (near max_seq_len) runs the static plan, and with it today's rule
        # below, unchanged. `is True`: a stand-in model's attribute is not a grant.
        _elastic = getattr(vaayu_ref, "_elastic_tier", None)
        if _elastic is not None and _elastic.lent is True:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
    if os.environ.get("VAAYU_VRAM_DIAG"):
        free_now, _ = torch.cuda.mem_get_info()
        print(_format_vram_diag(
            layer_idx, free_now,
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
            _vram.current_usage, _vram.budget_bytes, _vram.prefill_evictions,
            _vram.prefill_hits),
            flush=True)


def _vram_tier_visit(vaayu_ref, layer_idx, unique_experts):
    """One layer visit's view of the VRAM expert tier: (held, vram).

    `held` maps each routed expert that is resident to its entry. The visit keeps
    these references for its whole compute loop: eviction only pops the dict and
    drops the cache's reference, so a held entry stays valid even if a later
    put() in the same visit evicts it. ONE lookup per expert, not has()-then-get().

    `vram` is the cache this visit may put() its misses into, or None.

    Two gates, shared by all three per-expert forwards and, since S18, the routed
    fused-3D wrapper (vaayu/routed_fused3d.py), so they cannot drift:
    _vram_cacheable_layers restricts the tier to the layer prefix that fits when
    the budget cannot hold a whole token's working set (None = no restriction);
    and a PREFILL pass is read-only (S17) -- it routes ~every expert once, so
    filling the tier from it would only churn the LRU, but an expert already
    resident needs no second upload. Prefill reads through peek(), which leaves
    the LRU order and the decode hit/miss counters alone.
    """
    vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    layers = getattr(vaayu_ref, "_vram_cacheable_layers", None)
    if vram is None or (layers is not None and layer_idx not in layers):
        return {}, None
    prefill = getattr(vaayu_ref, "_moe_prefill_pass", False)
    lookup = vram.peek if prefill else vram.get
    held = {}
    for eidx in unique_experts:
        entry = lookup(layer_idx, eidx)
        if entry is not None:
            held[eidx] = entry
    return held, (None if prefill else vram)


def _maybe_record_usage(vaayu_ref, layer_idx, expert_indices) -> None:
    """Count DECODE-time expert routes when the usage histogram is enabled.

    A no-op when the feature is off (the common case — one getattr). Skipped
    during prefill on purpose: a prefill pass never fills the VRAM expert cache
    (it only reads it), so there are no budget-pressure evictions then, and counting
    prefill routes would make the report compare decode-side evictions against
    prefill+decode routes. See the measurement design spec.
    """
    tracker = getattr(vaayu_ref, "_expert_usage", None)
    if tracker is None:
        return
    if getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    # The routed fused-3D forward knows its routed set as a plain list of expert
    # ids (one visit = one token at decode, so the set IS that token's top-k);
    # the other forwards pass a [tokens, top_k] device tensor. Convert here, once
    # we already know the tracker is on, so the off path stays a single getattr.
    if not torch.is_tensor(expert_indices):
        expert_indices = torch.as_tensor(
            expert_indices, dtype=torch.int64, device=tracker.counts.device)
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
    from .persist.expert_cache import ExpertVRAMCache

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
        # routing sync below (_routing_to_host). Sync BEFORE starting the clock
        # so any GPU work still in flight from the previous layer isn't
        # misattributed to routing; no sync is needed at the end — its .cpu()
        # already forces a blocking D2H copy, so the window it closes is
        # truthful without an extra synchronize() call.
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
        routing_weights, expert_indices = truncate_top_k(
            routing_weights, expert_indices, *_decode_limits(vaayu_ref))

        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        expert_indices_cpu, unique_experts = _routing_to_host(expert_indices)
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
            expert_indices_cpu.reshape(-1, expert_indices.shape[-1]),
            len(experts_list))
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
        _flat_routing = routing_weights.reshape(-1, routing_weights.shape[-1])

        # VRAM expert cache (Fix #7, ported from the fused path). Experts already
        # VRAM-resident need no RAM lookup and no disk read at all. Prefill reads
        # the tier and does not fill it (_vram is None then) -- see _vram_tier_visit.
        _vram_resident, _vram = _vram_tier_visit(vaayu_ref, layer_idx, unique_experts)

        # Chunk-major retention (row 45, the module-list half of row 44's F2). The
        # layer-major chunked prefill calls this layer once per chunk, back to
        # back, for any MoE model -- but until this, only the two fused forwards
        # kept anything across those calls. Measured on MiniMax-M2.7: a 2048-token
        # prefill issued 55,871 expert loads against a per-layer union of 15,268
        # (3.659x, 495.86 GB read), and the repeated per-load H2D + dequant +
        # detach was 63 % of a 113.9 s wall. An expert uploaded by an earlier chunk
        # call is kept as the same entry the VRAM cache would store and served
        # through the hit branch below; the dict is dropped at the layer's last
        # chunk call and bounded by the VRAM cache budget; prefill
        # puts nothing in that cache. _chunked_prefill's finally clears
        # _chunk_major_gpu, which is why a missing dict always starts a fresh count.
        _cm = None
        _cm_last_call = False
        _cm_budget = 0
        _n_calls = getattr(vaayu_ref, "_chunk_major_n", None) or 1
        if getattr(vaayu_ref, "_moe_prefill_pass", False) and _n_calls > 1:
            _cm = getattr(vaayu_ref, "_chunk_major_gpu", None)
            _owner, _seen = getattr(vaayu_ref, "_chunk_major_owner", (None, 0))
            if _cm is None or _owner != layer_idx:
                _cm = vaayu_ref._chunk_major_gpu = {}
                vaayu_ref._chunk_major_gpu_bytes = 0
                _seen = 0
            _seen += 1
            vaayu_ref._chunk_major_owner = (layer_idx, _seen)
            _cm_last_call = _seen >= _n_calls
            _cm_budget = getattr(getattr(vaayu_ref, "_expert_vram_cache", None),
                                 "budget_bytes", 0)

        if _cm is not None:
            # An expert the tier lacked on an earlier chunk call was uploaded then
            # and retained here; tier residents never enter this dict.
            for _e in unique_experts:
                _entry = _cm.get(_e)
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
                # Row 49: upload async when — and only when — these bytes sit in
                # the pinned slab arena. slab_index_of is pointer arithmetic on
                # the arena's chunks (no CUDA call), and it answers both
                # questions at once: whether the source is page-locked, and
                # which slab the in-flight DMA reads from. A prefill clone or an
                # arena-exhausted fallback is in no slab -> None -> synchronous,
                # which is what those pageable buffers need.
                _arena = expert_cache.arena
                _rd = (_arena.slab_index_of(next(iter(raw.values())))
                       if _arena is not None and raw else None)
                entry = _inject_expert_weights(
                    expert_mod, raw, eidx, layer_name, device, dtype,
                    timing=_timing, pinned_source=_rd is not None)
                if _rd is not None:
                    # The copies are issued but not drained. Arm the arena's read
                    # event so acquire() cannot hand this slab to a fresh io_uring
                    # write under the in-flight DMA (silent torn weights) — the
                    # same guarantee both fused forwards take. This forward's pin
                    # on the key is held until the layer's finally, so the window
                    # between the copy and the event is covered by the pin.
                    _arena.record_read(_rd)
                # A prefill staging slab's job ends here: the copies are issued and
                # the read event now guards them, so hand it back for the next
                # expert's read (S1-L1). A decode slab is the cache's, not ours.
                _release_prefill_staged(expert_cache, vaayu_ref,
                                        lambda k: k == (layer_idx, eidx))
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
                elif _cm is not None and not _cm_last_call:
                    # Retain for this layer's remaining chunk calls. Overflow past
                    # the budget simply re-reads on a later chunk, as F2 does.
                    # _detach_expert still runs below: the entry owns the GPU
                    # tensors and the BC objects, the module only borrows them,
                    # which is exactly why the VRAM-cache path above can detach too.
                    _sz = ExpertVRAMCache._group_bytes(entry)
                    if vaayu_ref._chunk_major_gpu_bytes + _sz <= _cm_budget:
                        _cm[eidx] = entry
                        vaayu_ref._chunk_major_gpu_bytes += _sz
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
                        # The entry came from the VRAM cache (decode, or a
                        # prefill read) or from the chunk-major dict. _vram is
                        # None in prefill, so the drop goes to the cache itself;
                        # drop() on a key it does not hold is a no-op.
                        _tier = getattr(vaayu_ref, "_expert_vram_cache", None)
                        if _tier is not None:
                            _tier.drop(layer_idx, eidx)
                        if _cm is not None and eidx in _cm:
                            vaayu_ref._chunk_major_gpu_bytes -= (
                                ExpertVRAMCache._group_bytes(_cm.pop(eidx)))
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
            # Same for this layer's prefill staging slabs: the loop returns each one
            # as its upload issues, this drains experts it never reached (S1-L1).
            # The flag closes the layer first, so a load still in flight releases its
            # own slab instead of registering one after this sweep has passed.
            vaayu_ref._prefill_staging_closed = layer_idx
            _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == layer_idx)

        if _cm_last_call:
            vaayu_ref._chunk_major_gpu = None
            vaayu_ref._chunk_major_gpu_bytes = 0

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


def _weight_converter_target(converter) -> str:
    """The attribute name a weight converter writes, from its target pattern.

    The patterns are regexes anchored for matching parameter names (`gate_up_proj$`);
    what the conversion op wants is the plain attribute, because it resolves the
    module by name. Either a string or a one-element list, both shapes ship in
    transformers 5.5.4.
    """
    targets = converter.target_patterns
    if isinstance(targets, str):
        targets = [targets]
    return re.sub(r"[\^$]", "", targets[0])


def _populate_fused_experts(experts_mod, experts_path: str, raw: dict,
                            device: str, vaayu_ref) -> None:
    """Put ONE layer's fused expert tensors on the experts module.

    Quantised projections go through the QUANTIZER'S OWN weight conversions
    (`hf_quantizer.get_weight_conversions()`), never through a format-specific
    branch here: that hook returns the two MXFP4 deserialisers for gpt-oss and an
    EMPTY list for Qwen3-Next's compressed-tensors quantizer (measured
    2026-09-21), so this stays quant-agnostic and inert for a checkpoint whose
    fused tensors need no conversion. Whatever the converters do not consume --
    the expert biases here -- is assigned directly, in the dtype the module's own
    (meta) parameter declares, which is how the stock loader fixes the dtype too.
    """
    leaves = {key.rsplit(".", 1)[-1]: tensor for key, tensor in raw.items()}
    on_device = {name: t.to(device, non_blocking=True) for name, t in leaves.items()}

    hf_quantizer = getattr(vaayu_ref, "hf_quantizer", None)
    converters = []
    if hf_quantizer is not None and hasattr(hf_quantizer, "get_weight_conversions"):
        converters = hf_quantizer.get_weight_conversions() or []

    consumed: set = set()
    for converter in converters:
        sources = list(getattr(converter, "source_patterns", []) or [])
        if not sources or not all(name in on_device for name in sources):
            continue
        target = _weight_converter_target(converter)
        payload = {name: on_device[name] for name in sources}
        for op in converter.operations:
            op.convert(payload, model=vaayu_ref.model,
                       full_layer_name=f"{experts_path}.{target}",
                       missing_keys=set())
        consumed.update(sources)

    for name, tensor in on_device.items():
        if name in consumed:
            continue
        declared = experts_mod._parameters.get(name)
        if declared is not None and declared.dtype != tensor.dtype:
            tensor = tensor.to(declared.dtype)
        experts_mod._parameters[name] = nn.Parameter(tensor, requires_grad=False)


def _release_fused_experts(experts_mod, meta_params: dict, base_attrs: dict) -> None:
    """Undo one _populate_fused_experts, exactly.

    The swizzled weight a conversion op leaves behind is neither a parameter nor
    a buffer -- it is a plain attribute holding a kernel-side tensor object, plus
    a `{proj}_precision_config` beside it -- so nothing in Vaayu's usual teardown
    (`set_module_tensor_to_device(..., "meta")`, `_to_meta_preserving_buffers`)
    frees it. Left alone it would hold this layer's expert bytes on the GPU for
    the model's lifetime: 423.6 MB per layer on gpt-oss-20b, 10.17 GB per sweep.
    Restoring the patch-time snapshot puts the module back on meta whatever the
    converters did, including when populate raised part-way through.
    """
    experts_mod._parameters.clear()
    experts_mod._parameters.update(meta_params)
    for name in [n for n in experts_mod.__dict__ if n not in base_attrs]:
        del experts_mod.__dict__[name]
    experts_mod.__dict__.update(base_attrs)


def _make_whole_layer_sparse_forward(
    block_info: MoEBlockInfo,
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    device: str,
    vaayu_ref,
    original_forward=None,
    expert_cache=None,
):
    """Stream a fused-3D layer WHOLE and let the module's own forward route it.

    For a checkpoint that stores every expert of a layer in one tensor per
    projection, the routing happens inside the module's kernel -- a grouped GEMM
    is handed the entire fused tensor -- so there is no per-expert delivery to
    make and no sparsity for the expert cache to exploit. The layer's bytes are
    read by the whole-tensor metas the offset index records, handed to the module,
    computed by `original_forward`, and released before the call returns.

    The block's `original_forward` is the module's own forward as it stood when
    Vaayu patched it -- for a quantised model that is the quantizer's forward,
    installed by `preprocess_model` -- so its return value is passed back
    untouched: gpt-oss's MoE block returns `(hidden_states, router_scores)`, not a
    bare tensor. Per-expert streaming of a fused-3D checkpoint is row 47's A6 and
    needs a kernel that takes one expert at a time.

    S8: when the load-time self-check proved this geometry's kernel reads only the
    routed experts' rows (`routed_delivery_enabled`), the layer is delivered ROUTED
    instead: the experts submodule's forward is wrapped (vaayu/routed_fused3d.py) and
    only the routed experts' rows move, through the shared per-expert RAM arena, into
    one persistent device buffer per geometry. The private model attribute
    `_routed_fused3d = False` switches a routed install back to whole-layer delivery
    for A/B only.
    """
    from .persist.safetensor_model_persister import load_fused_layer_tensors
    from .routed_fused3d import (allocate_expert_buffers, fused3d_geometry_key,
                                 make_routed_experts_forward, routed_delivery_enabled)

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx
    experts_mod = getattr(block_info.moe_module, block_info.experts_attr)
    experts_path = f"{block_info.module_path}.{block_info.experts_attr}"
    shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
    # S8 routed delivery: installed BEFORE the snapshot below, so a release (either
    # path) restores the wrapper rather than deleting it.
    routed = (expert_cache is not None
              and routed_delivery_enabled(block_info, offset_index, vaayu_ref))
    active = [False]
    snapshot: dict = {}
    if routed:
        _layer_metas = offset_index.fused_layer_tensors[layer_name]
        _all_buffers = getattr(vaayu_ref, "_fused3d_buffers", None)
        if _all_buffers is None:
            _all_buffers = {}
            vaayu_ref._fused3d_buffers = _all_buffers
        _gkey = fused3d_geometry_key(_layer_metas)
        if _gkey not in _all_buffers:
            # One per geometry, shared by every layer of it: copies run on the
            # compute stream, so layer i+1 cannot overwrite rows layer i's kernels
            # are still reading. Same peak as the whole-layer transient it replaces.
            _all_buffers[_gkey] = allocate_expert_buffers(_layer_metas, device)
            if len(_all_buffers) > 1:
                # The planner's layer_slack budgets ONE fused layer, so only the
                # first geometry's buffer is inside the plan (CLAUDE.md §5: warn,
                # don't silently degrade). No model on this box has two.
                def _fmt(bufs):
                    n = sum(t.nbytes for t in bufs.values())
                    return f"{n / 1e6:.1f} MB ({n} B)"
                _held = ", ".join(_fmt(b) for k, b in _all_buffers.items() if k != _gkey)
                print(f"[Vaayu-MoE] WARNING: {layer_name} is another routed fused-3D "
                      f"geometry; its expert buffer {_fmt(_all_buffers[_gkey])} is allocated "
                      f"beside the {_held} already held. The VRAM planner budgets one "
                      f"fused layer (layer_slack), so this extra buffer is not charged to "
                      f"it: expect less VRAM headroom than the plan printed.")
        experts_mod.forward = make_routed_experts_forward(
            experts_mod=experts_mod, prior_forward=experts_mod.forward,
            experts_path=experts_path, layer_idx=layer_idx, layer_name=layer_name,
            shard_file=shard_file, offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref, buffers=_all_buffers[_gkey], device=device,
            snapshot=snapshot, active=active)
    # Snapshot taken at PATCH time, while the module is still meta-initialised:
    # this is what a visit restores, so a release can never leave the module
    # holding a live expert tensor.
    meta_params = dict(experts_mod._parameters)
    base_attrs = dict(experts_mod.__dict__)
    snapshot.update(meta_params=meta_params, base_attrs=base_attrs)
    # Every expert, because the kernel is handed every expert. The spec-decode
    # gate sizes a verify pass off this (vaayu_base.py, gate.observe) and an empty
    # dict reads to it as "this forward does not report routing".
    all_experts = list(range(int(getattr(experts_mod, "num_experts", 0))))

    def sparse_forward(hidden_states, **kwargs):
        _ensure_prefill_headroom(vaayu_ref, layer_idx)
        if layer_name in vaayu_ref.pinned_layer_names:
            return original_forward(hidden_states, **kwargs)
        if routed and getattr(vaayu_ref, "_routed_fused3d", True):
            active[0] = True
            try:
                return original_forward(hidden_states, **kwargs)
            finally:
                active[0] = False

        metas = offset_index.fused_layer_tensors.get(layer_name)
        if not metas:
            raise RuntimeError(
                f"[Vaayu-MoE] {layer_name}: the whole-layer forward was installed but the "
                f"expert offset index holds no fused tensors for this layer. Vaayu would "
                f"run the layer with meta-initialised experts and emit garbage."
            )
        raw = load_fused_layer_tensors(shard_file, metas)
        try:
            _populate_fused_experts(experts_mod, experts_path, raw, device, vaayu_ref)
            vaayu_ref._expert_history[layer_idx] = all_experts
            return original_forward(hidden_states, **kwargs)
        finally:
            _release_fused_experts(experts_mod, meta_params, base_attrs)

    sparse_forward.__vaayu_delivery__ = "routed" if routed else "whole"
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

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def _route(flat_hidden):
        """Router -> top-k -> the decode budget: the routed slots, on the GPU."""
        router = getattr(block_info.moe_module, block_info.router_attr)
        router_res = router(flat_hidden)
        if isinstance(router_res, tuple) and len(router_res) >= 3:
            _logits, routing_weights, expert_indices = router_res[:3]
        else:
            routing_weights, expert_indices = torch.topk(
                router_res.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return truncate_top_k(routing_weights, expert_indices, *_decode_limits(vaayu_ref))

    def decode_pre(hidden_states):
        """S12: the part of a visit BEFORE the routing sync -- routing and the shared
        expert -- so a decode graph can capture it. None where sparse_forward would not
        take this path (a fully pinned layer). No side effect beyond its tensors:
        usage recording and the expert history stay in sparse_forward."""
        if layer_name in vaayu_ref.pinned_layer_names:
            return None
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        routing_weights, expert_indices = _route(flat_hidden)
        return (routing_weights, expert_indices,
                _shared_expert_output(block_info.moe_module, flat_hidden))

    _shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
    _moe_mod = block_info.moe_module

    def _lean_decode(hidden_states, _pre):
        """S28 C4: a one-token decode visit on the packed kernels, with nothing a decode
        never needs -- no prefill headroom, chunk-major retention, prefill-kernel lookup,
        profile marks or per-call path join, and no zeroed accumulator (the kernel
        overwrites it). The same helpers, ops and order as the `_use_pk` branch below,
        so the same bits."""
        _act_fn = getattr(_moe_mod, block_info.experts_attr).act_fn
        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)
        if _pre is None:
            routing_weights, expert_indices = _route(flat_hidden)
            _shared = None
        else:
            routing_weights, expert_indices, _shared = _pre
        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        expert_indices_cpu, unique_experts = _routing_to_host(expert_indices)
        _eids = expert_indices_cpu.view(-1).tolist()
        _vram_hold, _vram = _vram_tier_visit(vaayu_ref, layer_idx, unique_experts)
        expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            unique_experts, layer_idx=layer_idx, layer_name=layer_name,
            shard_file=_shard_file, offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref, vram_resident=set(_vram_hold))
        try:
            _blobs: dict = {}
            for eidx, raw in _iter_expert_loads(expert_tensors, _miss_futures, vaayu_ref):
                _held = _vram_hold.get(eidx)
                if _held is not None:
                    _blobs[eidx] = _held
                    continue
                _math = _packed_miss_entry(
                    eidx, raw, device=device, dtype=dtype, timing=None, layer_idx=layer_idx,
                    expert_cache=expert_cache, vaayu_ref=vaayu_ref, pinned_keys=_pinned_keys,
                    vram=_vram, cm=None, cm_last_call=False, cm_budget=0)
                if _math is not None:
                    _blobs[eidx] = _math
            _pairs = [(_blobs[_eids[s]], s)
                      for s in sorted(range(len(_eids)), key=_eids.__getitem__)
                      if _eids[s] in _blobs]
            if _pre is None:
                _shared = _shared_expert_output(_moe_mod, flat_hidden)
            result = None
            if _pairs and _pre is not None:
                result = packed_experts_decode_graphed(
                    layer_idx, flat_hidden, _pairs, routing_weights, _shared, _act_fn,
                    tail=True)
            if result is None:
                out = torch.empty(flat_hidden.shape, dtype=torch.float32,
                                  device=flat_hidden.device)
                if _pairs:
                    packed_experts_decode(flat_hidden, _pairs, routing_weights, out, _act_fn)
                else:
                    out.zero_()
                result = out.to(flat_hidden.dtype)
                if _shared is not None:
                    result = result + _shared.to(result.dtype)
        finally:
            for _pk in _pinned_keys:
                expert_cache.unpin(*_pk)
            if getattr(vaayu_ref, "_prefill_staged", None):
                vaayu_ref._prefill_staging_closed = layer_idx
                _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == layer_idx)
        vaayu_ref._expert_history[layer_idx] = unique_experts
        return result.view(B, S, H)

    def sparse_forward(hidden_states, _pre=None, **kwargs):
        if (hidden_states.shape[0] * hidden_states.shape[1] == 1
                and not getattr(vaayu_ref, "_moe_prefill_pass", False)
                and getattr(vaayu_ref, "_packed_moe_kernel", None) is not None
                and not getattr(vaayu_ref, "profiling_mode", False)
                and getattr(vaayu_ref, "_batch_n", 1) == 1
                and layer_name not in vaayu_ref.pinned_layer_names):
            return _lean_decode(hidden_states, _pre)
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
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

        # 'route' brackets the router forward, the top-k and the routing
        # sync below (_routing_to_host) -- the same window the other two
        # forwards call route.
        _t_route = _mark()
        if _pre is None:
            routing_weights, expert_indices = _route(flat_hidden)
            _shared_pre = None
        else:
            # S12: a decode graph already ran routing + the shared expert (decode_pre).
            routing_weights, expert_indices, _shared_pre = _pre
        _warn_cache_bias_inert(
            vaayu_ref, "this model takes the non-EXL3 fused forward, which has no "
                       "cache-aware routing")

        _maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
        expert_indices_cpu, unique_experts = _routing_to_host(expert_indices)
        if _prof:
            _mp['route'] += time.perf_counter() - _t_route

        if getattr(vaayu_ref, "_batch_n", 1) > 1:
            vaayu_ref.telemetry.record_batch_overlap(
                layer_idx, len(unique_experts), vaayu_ref._batch_n * top_k)

        # VRAM expert cache (S2b (iii)). Until 2026-09-13 this forward reached it
        # through ZERO code paths while its budget sat reserved and idle — the
        # VAAYU_CACHE_BIAS failure mode (CLAUDE.md §0) in a fourth place. Same
        # two gates and the same hold as the other two forwards. Prefill reads
        # the tier and does not fill it (_vram is None then) -- see _vram_tier_visit.
        _vram_hold, _vram = _vram_tier_visit(vaayu_ref, layer_idx, unique_experts)

        # Chunk-major retention (row 44 F2). The layer-major chunked prefill calls
        # this layer once per chunk, back to back; without this, each call re-read
        # its experts -- 53,938 loads for a 2k Qwen3-Next prefill against a
        # per-layer union of 17,445 (10.9x at 8k). An expert uploaded by an
        # earlier chunk call is kept as the same entry the VRAM cache would store
        # and served through the hit branch below; the dict is dropped at the
        # layer's last chunk call. Same mechanism as the fused-EXL3 forward's
        # _chunk_major_gpu, with its own call counter since that one rides the
        # EXL3-only prefill stager. Bounded by the VRAM cache budget, which prefill
        # does not fill. _chunked_prefill's finally clears the dict on an exception,
        # which is why a missing dict always starts a fresh count.
        _cm = None
        _cm_last_call = False
        _cm_budget = 0
        _n_calls = getattr(vaayu_ref, "_chunk_major_n", None) or 1
        if getattr(vaayu_ref, "_moe_prefill_pass", False) and _n_calls > 1:
            _cm = getattr(vaayu_ref, "_chunk_major_gpu", None)
            _owner, _seen = getattr(vaayu_ref, "_chunk_major_owner", (None, 0))
            if _cm is None or _owner != layer_idx:
                _cm = vaayu_ref._chunk_major_gpu = {}
                vaayu_ref._chunk_major_gpu_bytes = 0
                _seen = 0
            _seen += 1
            vaayu_ref._chunk_major_owner = (layer_idx, _seen)
            _cm_last_call = _seen >= _n_calls
            _cm_budget = getattr(getattr(vaayu_ref, "_expert_vram_cache", None),
                                 "budget_bytes", 0)
            for _e in unique_experts:
                _held = _cm.get(_e)
                if _held is not None:
                    _vram_hold[_e] = _held

        # Load active expert weights through the SHARED miss loader (row 30-P).
        # With a slab arena -- the engine allocates one for this forward on CUDA
        # since 30-P -- RAM-tier entries are views into PINNED slabs, so the miss
        # path's .to(device, non_blocking=True) is a real async copy instead of a
        # pageable one that holds the host (157 us vs 32-40 us per expert,
        # measured; +10.62 % decode tok/s on Qwen3-Next-80B int4). Without an
        # arena (CPU compute, unit tests) the helper takes the legacy clone path,
        # which is what this forward used to do inline.
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            unique_experts,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref,
            # On the GPU already: a RAM miss on these reads nothing off NVMe (P8).
            # Their RAM lookup still runs inside the helper (LRU refresh).
            vram_resident=set(_vram_hold),
        )
        # S7: a one-token decode call runs the fused packed-expert kernels (installed at
        # load by install_packed_expert_kernels). S20: every prefill call of a layer the
        # installer found eligible runs the grouped ones -- the flag is the set of those
        # layers' indices; absent or None means no layer. A multi-token call outside
        # prefill, a layer outside the set and every checkpoint the installer did not
        # accept keep the per-slot loop below, which holds one expert's dense weights
        # at a time.
        _pf_layers = getattr(vaayu_ref, "_packed_moe_prefill_kernel", None) or ()
        _use_pf = (layer_idx in _pf_layers
                   and getattr(vaayu_ref, "_moe_prefill_pass", False))
        _use_pk = _use_pf or (getattr(vaayu_ref, "_packed_moe_kernel", None) is not None
                              and flat_hidden.shape[0] == 1
                              and not getattr(vaayu_ref, "_moe_prefill_pass", False))
        try:
            # fp32 accumulator, cast back at the return. Experts are CONSUMED in
            # arrival order (that is what overlaps the reads with compute, row 44
            # F6) and summed in expert-id order afterwards by
            # accumulate_expert_outputs, exactly as the other two forwards do --
            # so the sum never depends on which read finished first. Until F6
            # this forward waited for EVERY read before computing anything,
            # which put 9.10 s of a 25 s Qwen3-Next 2k prefill in a stall with
            # ~19 s of compute queued behind it.
            # Permutation dispatch (row 44 F5, ported from the other two
            # forwards): ONE CPU argsort and ONE H2D of the index pairs per
            # layer, replacing an int64 one-hot mask of num_experts x top_k x
            # tokens (21 MB per 512-token Qwen3-Next chunk) and a torch.where
            # plus two tiny per-expert copies. Each expert's positions are a
            # contiguous slice of `_pair_gpu`, so slicing them is free.
            _t_mask = _mark() if _prof else 0.0
            if _use_pk:
                # The per-slot loop below never runs on this branch: the kernel branch
                # consumes every load first. Bound anyway, so no name in that loop is
                # ever unbound (tests/test_no_unbound_locals.py).
                _pair_gpu = _starts = _counts_l = None
            else:
                _pair, _starts, _counts_l = compute_expert_dispatch(
                    expert_indices_cpu.view(-1, expert_indices_cpu.shape[-1]),
                    experts_mod.num_experts)
                _pair_gpu = _pair.to(device, non_blocking=True)
            if _prof:
                _mp['mask_sync'] += time.perf_counter() - _t_mask

            # ── Compute FFN on the target device ──────────────────────────────
            final_hidden = torch.zeros_like(flat_hidden, dtype=torch.float32)
            _contribs: list = []

            _loads = _iter_expert_loads(
                expert_tensors, _miss_futures, vaayu_ref, _mp if _prof else None)
            if _use_pk:
                # Every routed expert, VRAM hit or miss alike, goes through fused
                # kernels over the layer's packed blobs instead of ~14 launches each in
                # the loop below (~135 us of host per expert for ~33 us of GPU work at
                # one token, measured -- S7; a 512-token prefill chunk: 48 ms of loop
                # against 4 ms of kernels -- S20). Hits and misses MUST share them:
                # the accumulator gate compares logits across different residency,
                # so residency may not change an expert's arithmetic.
                _blobs: dict = {}
                for eidx, raw in _loads:
                    _held = _vram_hold.get(eidx)
                    if _held is not None:
                        _blobs[eidx] = _held
                        if _prof:
                            _mp['slots_hit'] += 1
                        continue
                    # The installer read ONE expert, so a later layer may be unpacked or
                    # mixed; packed_experts_decode runs any such call the stock way.
                    _math = _packed_miss_entry(
                        eidx, raw, device=device, dtype=dtype, timing=_mp,
                        layer_idx=layer_idx, expert_cache=expert_cache, vaayu_ref=vaayu_ref,
                        pinned_keys=_pinned_keys, vram=_vram, cm=_cm,
                        cm_last_call=_cm_last_call, cm_budget=_cm_budget)
                    if _math is not None:
                        _blobs[eidx] = _math
                        if _prof:
                            _mp['slots_miss'] += 1
                if _use_pf:
                    # One launch set for the whole call, after its last load arrived:
                    # the kernels group the rows by expert themselves and sum each
                    # token's experts in expert-id order (row 44 F6).
                    if _blobs:
                        _t_gemm = _mark() if _prof else 0.0
                        packed_experts_prefill(
                            flat_hidden,
                            expert_indices_cpu.view(-1, expert_indices_cpu.shape[-1]),
                            routing_weights, _blobs, final_hidden, act_fn)
                        if _prof:
                            _mp['gemm'] += _mark() - _t_gemm
                else:
                    # The per-slot loop's order -- expert id, then slot -- which is the
                    # order the fp32 sum runs in (row 44 F6).
                    _eids = expert_indices_cpu.view(-1).tolist()
                    _pairs = [(_blobs[_eids[s]], s)
                              for s in sorted(range(len(_eids)), key=_eids.__getitem__)
                              if _eids[s] in _blobs]
                    if _pairs:
                        _t_gemm = _mark() if _prof else 0.0
                        packed_experts_decode(flat_hidden, _pairs, routing_weights,
                                              final_hidden, act_fn)
                        if _prof:
                            _mp['gemm'] += _mark() - _t_gemm
                _loads = ()      # every expert was consumed above
            for eidx, raw in _loads:
                _s, _c = _starts[eidx], _counts_l[eidx]
                if _c == 0:
                    continue

                _t_scatter = _mark() if _prof else 0.0
                token_idx_gpu = _pair_gpu[0, _s:_s + _c]
                top_k_pos_gpu = _pair_gpu[1, _s:_s + _c]
                current_state = flat_hidden[token_idx_gpu]
                if _prof:
                    _mp['scatter'] += time.perf_counter() - _t_scatter

                _held = _vram_hold.get(eidx)
                if _held is not None:
                    # A hit skips the host-to-device copy entirely; a packed entry
                    # still owes the fused dequant, which is 'construct' either way.
                    # It is booked to 'construct_hit' as well, because S3 cannot move
                    # a GPU-resident expert to the CPU without copying it back.
                    _t_con = _mark() if _prof else 0.0
                    w_gate_gpu, w_down_gpu, w_up_gpu = expert_weights_from_entry(
                        _held, dtype)
                    if _prof:
                        _d_con = _mark() - _t_con
                        _mp['construct'] += _d_con
                        _mp['construct_hit'] += _d_con
                        _mp['slots_hit'] += 1
                else:
                    # h2d / construct are split inside _expert_dense_weights: the copy
                    # and the dequant run back to back there, and the phase-4 gate is a
                    # ratio of exactly those two.
                    w_gate_gpu, w_down_gpu, w_up_gpu, _entry = _expert_dense_weights(
                        raw, device, dtype, timing=_mp)
                    _settle_miss_upload(
                        eidx, raw, _entry, layer_idx=layer_idx, expert_cache=expert_cache,
                        vaayu_ref=vaayu_ref, pinned_keys=_pinned_keys, vram=_vram, cm=_cm,
                        cm_last_call=_cm_last_call, cm_budget=_cm_budget)
                    if w_gate_gpu is None:
                        continue
                    if _prof:
                        _mp['slots_miss'] += 1

                _t_gemm = _mark() if _prof else 0.0
                gate_out = act_fn(F.linear(current_state, w_gate_gpu))
                up_out = F.linear(current_state, w_up_gpu)
                out = F.linear(gate_out * up_out, w_down_gpu)
                if _prof:
                    _d_gemm = _mark() - _t_gemm
                    _mp['gemm'] += _d_gemm
                    if _held is not None:
                        _mp['gemm_hit'] += _d_gemm

                _t_scatter = _mark() if _prof else 0.0
                out = out * routing_weights[token_idx_gpu, top_k_pos_gpu, None]
                # Buffered, not added here: accumulate_expert_outputs sums in
                # expert-id order once every expert has run, so arrival order
                # cannot reach the arithmetic (row 44 F6).
                _contribs.append((eidx, token_idx_gpu, out))
                if _prof:
                    _mp['scatter'] += _mark() - _t_scatter

                del w_gate_gpu, w_down_gpu, w_up_gpu

            accumulate_expert_outputs(final_hidden, _contribs)
        finally:
            # Every key the helper pinned that the loop did not release (VRAM
            # hits, skipped experts, an exception mid-loop). A leaked pin blocks
            # that key's eviction forever.
            for _pk in _pinned_keys:
                expert_cache.unpin(*_pk)
            # Same for this layer's prefill staging slabs: the loop returns each one
            # as its upload issues, this drains experts it never reached (S1-L1).
            # The flag closes the layer first, so a load still in flight releases its
            # own slab instead of registering one after this sweep has passed.
            vaayu_ref._prefill_staging_closed = layer_idx
            _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == layer_idx)

        if _cm_last_call:
            vaayu_ref._chunk_major_gpu = None
            vaayu_ref._chunk_major_gpu_bytes = 0

        result = final_hidden.to(flat_hidden.dtype)
        _shared = (_shared_pre if _pre is not None
                   else _shared_expert_output(moe_mod, flat_hidden))
        if _shared is not None:
            result = result + _shared.to(result.dtype)

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return result.view(B, S, H)

    sparse_forward.decode_pre = decode_pre
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


# Not a tuning knob, and the same idea as _LOAD_WAIT_BACKSTOP_S: a staging slab that
# leaked (a worker that registered one after its layer already tore down) must make
# prefill fall back to the clone path, never hang a layer.
_STAGING_WAIT_BACKSTOP_S = 5.0


def _prefill_staged(vaayu_ref) -> dict:
    """{(layer_idx, expert_idx): slab_idx} for prefill experts whose bytes are still
    sitting in a staging slab the forward has not finished uploading."""
    staged = getattr(vaayu_ref, "_prefill_staged", None)
    if staged is None:
        staged = {}
        vaayu_ref._prefill_staged = staged
    return staged


def _stage_prefill_expert(expert_cache, vaayu_ref, layer_idx, eidx, shard_file, metas):
    """Read ONE prefill expert into a reserved staging slab, or return None.

    Prefill does not cache, so before S1-L1 it read through load_expert_sparse, which
    clones out of its pinned pool buffer into PAGEABLE RAM — and a pageable copy
    cannot honour non_blocking, so every expert's upload drained the stream (measured:
    202.8 us host vs 202.9 us device, i.e. fully serialising, against 113.6/159.8 from
    pinned; 67-89 us of host stall per expert = 6.6-8.8 % of a prefill). A staging
    slab is page-locked, so the upload the forward already issues becomes a real async
    copy, and the read drops a 1.77 MB host clone per expert on the way.

    None means "no staging slab available" — the caller takes the legacy clone path,
    which is correct, just slower.
    """
    from .persist.safetensor_model_persister import load_expert_into_slab
    arena = expert_cache.arena
    if arena is None or not arena.staging_slabs:
        return None
    slab_idx = arena.acquire_staging(timeout=_STAGING_WAIT_BACKSTOP_S)
    if slab_idx is None:
        return None
    try:
        views = load_expert_into_slab(
            shard_file, metas, arena.tensor(slab_idx),
            # Mirrors the decode miss path exactly: O_DIRECT only from pinned
            # memory, and no per-tensor repack (no production caller wants it).
            allow_direct=arena.pin, tensor_align=False)
    except Exception:
        arena.release_staging(slab_idx)
        raise
    # Registered AFTER the read so a failed read cannot leave a phantom key, and
    # keyed by (layer, expert) so the consuming forward can return exactly this slab.
    _prefill_staged(vaayu_ref)[(layer_idx, eidx)] = slab_idx
    # Register BEFORE this check, never after: the layer's finally may be sweeping
    # right now (an exception mid-compute does not wait for in-flight loads). Either
    # the sweep sees the key and releases it, or this does -- both are idempotent, so
    # the slab cannot be stranded either way. A closed layer's expert falls back to
    # the clone path rather than hold a slab nobody will consume.
    if getattr(vaayu_ref, "_prefill_staging_closed", None) == layer_idx:
        _release_prefill_staged(expert_cache, vaayu_ref,
                                lambda k: k == (layer_idx, eidx))
        return None
    return views


def _release_prefill_staged(expert_cache, vaayu_ref, pred) -> None:
    """Return every staging slab whose (layer_idx, expert_idx) key matches `pred`.

    Called three times per layer, and each one covers a case the others do not:
    after an expert's uploads are issued (the fast path — the arena's read event,
    already armed by record_read, covers the in-flight DMA); in the layer's finally
    (experts the compute loop skipped, and an exception mid-layer); and at the next
    layer's dispatch for any OTHER layer's leftovers, which self-heals a slab a
    worker registered after its own layer had already swept.
    """
    arena = expert_cache.arena
    staged = getattr(vaayu_ref, "_prefill_staged", None)
    if arena is None or not staged:
        return
    for key in [k for k in list(staged) if pred(k)]:
        slab_idx = staged.pop(key, None)
        if slab_idx is not None:
            arena.release_staging(slab_idx)


def _packed_miss_entry(eidx, raw, *, device, dtype, timing, layer_idx, expert_cache, vaayu_ref,
                       pinned_keys, vram, cm, cm_last_call, cm_budget):
    """What a missed expert computes with on the fused forward's packed-kernel decode paths
    (its lean branch and its `_use_pk` branch -- one copy, S28, so they cannot drift): the
    entry is uploaded and settled, and the tensors this call computes with are returned --
    None when stock skips the expert (a missing role).

    The entry is what the VRAM tier stores; the return value is what this call computes with.
    They differ only for packed roles at different widths (no fused pass): stock dequantizes
    those role by role and caches nothing, and so does this."""
    packed = _group_packed_tensors_by_weight(raw)
    if not packed["gate"]:
        wg, _wd, _wu, entry = _expert_dense_weights(raw, device, dtype, timing=timing)
        math = entry if wg is not None else None
    elif _packed_roles_complete(packed):
        on_dev = _upload_packed_expert(packed, device, timing)
        entry = build_expert_blob(on_dev)
        math = (entry if entry is not None
                else {"dense": dequantize_expert_weights(on_dev, dtype)})
    else:
        entry = math = None      # a missing role: stock skips it too
    _settle_miss_upload(eidx, raw, entry, layer_idx=layer_idx, expert_cache=expert_cache,
                        vaayu_ref=vaayu_ref, pinned_keys=pinned_keys, vram=vram, cm=cm,
                        cm_last_call=cm_last_call, cm_budget=cm_budget)
    return math


def _settle_miss_upload(eidx, raw, entry, *, layer_idx, expert_cache, vaayu_ref,
                        pinned_keys, vram, cm, cm_last_call, cm_budget) -> None:
    """Everything a missed expert owes once its host-to-device copies are ISSUED.

    Shared by the non-EXL3 fused forward's per-slot loop, its S7 kernel branch, and
    the routed fused-3D wrapper (S8, vaayu/routed_fused3d.py) so the order below
    cannot drift between them. The routed wrapper settles every expert it UPLOADS, RAM
    hit or miss: its rows go into a shared device buffer, and `entry` is the private
    copy of them the VRAM tier takes (S18) or None when nothing will keep one (prefill,
    no tier, a layer outside the cacheable range). If the copies read from a pinned slab,
    arm the arena's read event so acquire() cannot recycle the slab under the in-flight
    DMA, then release the forward's pin on the key -- the same order as
    _make_fused_sparse_forward_exl3. A legacy clone is in no slab (slab_index_of ->
    None) and needs neither. A prefill staging slab goes back to the staging pool
    (S1-L1). Then the cache entry, if there is one: the VRAM tier in decode, the
    chunk-major dict in a chunked prefill.
    """
    arena = expert_cache.arena
    if arena is not None:
        _rd = arena.slab_index_of(next(iter(raw.values())))
        if _rd is not None:
            arena.record_read(_rd)
    _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k == (layer_idx, eidx))
    if (layer_idx, eidx) in pinned_keys:
        expert_cache.unpin(layer_idx, eidx)
        pinned_keys.discard((layer_idx, eidx))
    if entry is None:
        return
    if vram is not None:
        # put() False means the cache refused ownership (duplicate key, or one expert
        # larger than the whole budget). These are plain torch tensors -- unlike the
        # EXL3 forward's BC objects there is no C++ heap to free, so dropping the
        # reference is the free.
        vram.put(layer_idx, eidx, entry)
    elif cm is not None and not cm_last_call:
        # Overflow past the budget simply re-reads on a later chunk.
        from .persist.expert_cache import ExpertVRAMCache
        _sz = ExpertVRAMCache._group_bytes(entry)
        if vaayu_ref._chunk_major_gpu_bytes + _sz <= cm_budget:
            cm[eidx] = entry
            vaayu_ref._chunk_major_gpu_bytes += _sz


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

    `vram_resident` (both fused forwards pass it -- fused-EXL3, and since row
    30-P the non-EXL3 fused forward -- and since S18 the routed fused-3D wrapper)
    is the set of routed experts the caller
    already holds a VRAM cache entry for. Their RAM lookup still runs —
    that LRU refresh is the residency invariant P1 measured, and dropping it is
    the change P1 refuted — but on a MISS no disk read is issued: the caller
    serves those experts from their GPU copies and gets VRAM_RESIDENT_NO_RAW in
    expert_tensors instead. The module-list forward filters these experts out
    before it calls this helper and passes nothing, so its behaviour is
    unchanged.

    Shared by all three per-expert MoE forwards (fused-EXL3, module-list, and
    since row 30-P the non-EXL3 fused forward) and, since S8, by the routed
    fused-3D wrapper (vaayu/routed_fused3d.py): the slab/pin/arena discipline
    below is subtle enough that a second copy would rot.
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
    if not _do_pin and expert_cache.arena is not None:
        # Prefill. This layer is open for staging again, and any slab still
        # registered for ANOTHER layer is a leftover nothing else will return.
        vaayu_ref._prefill_staging_closed = None
        _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] != layer_idx)
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
            # once; caching would clone content decode replaces) -- but read into a
            # reserved PINNED staging slab when one is armed, so the forward's
            # upload is async instead of a stream-draining pageable copy (S1-L1).
            if getattr(vaayu_ref, "_moe_prefill_pass", False):
                staged = _stage_prefill_expert(
                    expert_cache, vaayu_ref, layer_idx, eidx, shard_file, metas)
                if staged is not None:
                    return eidx, staged
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
        # must not hard-depend on that attribute.
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

        routing_weights, expert_indices = truncate_top_k(
            routing_weights, expert_indices, *_decode_limits(vaayu_ref))

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

        # Check GPU residency BEFORE dispatching, so a resident expert that has
        # fallen out of the RAM cache is not read off NVMe for nothing (P8).
        # Prefill reads the tier and does not fill it (_vram is None then), and
        # the layer-range restriction covers both -- see _vram_tier_visit.
        _vram_hold, _vram = _vram_tier_visit(vaayu_ref, layer_idx, unique_experts)

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
                # visit at decode; a prefill visit looks up through peek() and
                # promotes nothing).
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
                    # Both branches above have issued this expert's copies and armed
                    # the read event, so a prefill staging slab can go back to the
                    # staging pool for the next expert's read (S1-L1).
                    _release_prefill_staged(expert_cache, vaayu_ref,
                                            lambda k: k == (layer_idx, eidx))
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
                        # Chunk-major retention (prefill: nothing is put in the tier). Budget
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
                            # heap. Mutation outside the cache lock is safe — every consumer
                            # of VRAM-cache entries (decode or prefill) runs on the forward
                            # thread, one layer visit at a time.
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
            # Same for this layer's prefill staging slabs (S1-L1); the flag closes
            # the layer so an in-flight load releases its own slab rather than
            # registering one after this sweep.
            vaayu_ref._prefill_staging_closed = layer_idx
            _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == layer_idx)

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

