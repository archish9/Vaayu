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

    vaayu_ref: reference to VaayuBaseModel instance. The forward() loop reads
    _expert_history after each layer and submits speculative prefetch.

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
    """
    if block_info.expert_style == "fused":
        if is_exl3:
            return _make_fused_sparse_forward_exl3(
                block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
            )
        return _make_fused_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            compute_device=compute_device,
            original_forward=original_forward,
        )
    else:
        return _make_modulelist_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            original_forward=original_forward,
        )


# Hardware constant: free VRAM to guarantee before each MoE layer DURING PREFILL,
# by evicting the (prefill-idle) VRAM expert cache. A property of the card, not
# the model (CLAUDE.md §5) — same CLASS of constant as vaayu_base._VRAM_RESERVE_BYTES
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
        final_hidden = torch.zeros_like(hidden_states)

        # Permutation dispatch (ported from _make_fused_sparse_forward_exl3): one
        # CPU argsort per layer instead of one torch.where scan per expert
        # (~496 scans/token on the target model). hidden_states is (B, S, H), not
        # flat like the fused path's flat_hidden, so expert_indices/routing_weights
        # are flattened to (B*S, top_k) here — flat index b*S+s agrees between
        # .reshape(-1, top_k) and .view(-1, H) since both simply drop the B/S
        # split of the same row-major layout.
        if _prof:
            _t0 = time.perf_counter()
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
            _flat_final.index_add_(0, token_idx_gpu, (w * expert_out).to(final_hidden.dtype))
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

        # 4. Write routing result to vaayu_ref for speculative prefetch scheduling
        vaayu_ref._expert_history[layer_idx] = unique_experts
        _trace_moe_block(vaayu_ref, layer_idx, "module_list", hidden_states,
                         final_hidden, routing_weights, unique_experts)
        return final_hidden, router_logits

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

        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

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

        # Expert mask on CPU (avoids PCIe for index tensors)
        expert_indices_cpu = expert_indices.cpu()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)

        # ── Compute FFN on the target device ──────────────────────────────────
        final_hidden = torch.zeros_like(flat_hidden)

        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue

            w_gate, w_down, w_up = _classify_expert_tensors(raw)
            if w_gate is None or w_down is None or w_up is None:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue

            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]

            w_gate_gpu = w_gate.to(device, dtype=dtype, non_blocking=True)
            w_down_gpu = w_down.to(device, dtype=dtype, non_blocking=True)
            w_up_gpu   = w_up.to(device, dtype=dtype, non_blocking=True)

            gate_out = act_fn(F.linear(current_state, w_gate_gpu))
            up_out = F.linear(current_state, w_up_gpu)
            out = F.linear(gate_out * up_out, w_down_gpu)
            out = out * routing_weights[token_idx_gpu, top_k_pos.to(device), None]
            final_hidden.index_add_(0, token_idx_gpu, out.to(final_hidden.dtype))

            del w_gate_gpu, w_down_gpu, w_up_gpu

        result = final_hidden

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return result.view(B, S, H)

    return sparse_forward


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
) -> tuple[dict, list, set]:
    """
    Split this layer's routed experts into RAM/staged hits and disk misses, submit
    the misses to the persistent pool, and return without waiting — so the caller
    computes hits while the reads are in flight.

    Returns (expert_tensors, miss_futures, pinned_keys). The caller MUST unpin
    every key in pinned_keys in a finally block; a leaked pin permanently blocks
    that key's eviction (a slow cache-capacity leak).

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
            expert_cache.put_slab(layer_idx, eidx, views, slab_idx)
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
        load_expert_sparse, load_expert_into_slab, views_from_packed_copy,
        gpu_repack_aligned)

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def shard_file_for(name: str) -> str:
        return str(checkpoint_path / f"{name}.safetensors")

    def _load_predicted_expert(next_layer_name: str, next_layer_idx: int, eidx: int):
        """Phase-B prefetch: load a predicted next-layer expert into the RAM
        cache ahead of its layer via the SAME zero-clone slab path decode's
        on-path misses use (CR-2 fix: no double-clone). prefetch=True so the
        prefetch_used telemetry measures usefulness. Never raises."""
        try:
            if expert_cache.has(next_layer_idx, eidx):
                return
            metas = offset_index.get_expert_offsets(next_layer_name, eidx)
            if metas is None:
                return
            _shard = shard_file_for(next_layer_name)
            if expert_cache.arena is None:
                # No arena (unit-test / CUDA-less) — legacy clone path.
                raw = load_expert_sparse(_shard, metas)
                expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
                return
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
            expert_cache.put_slab(next_layer_idx, eidx, views, slab_idx, prefetch=True)
        except Exception as e:
            _log.warning(
                f"Lookahead prefetch failed for layer {next_layer_idx} expert {eidx}: {e}"
            )

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

        # ── Lookahead routing predictor (Task 12, decode only) ──────────────
        # Phase A (accounting) and Phase B (prefetch) are BOTH gated behind
        # VAAYU_LOOKAHEAD=1 (CR-1): the extra next-layer router GEMM +
        # _pred_idx.cpu() D2H sync are per-layer/per-token decode overhead
        # that only Phase B consumes. Default off → steady-state decode
        # pays nothing. Reads vaayu_ref._lookahead_enabled (captured ONCE at
        # init, not a live os.environ read) so mutating the env var after
        # engine construction can't bypass the hard-block guard in
        # vaayu_base.py — see check_lookahead_flag_safe()'s docstring.
        if (not getattr(vaayu_ref, "_moe_prefill_pass", False)
                and getattr(vaayu_ref, "_lookahead_enabled", False)):
            _la_state = getattr(vaayu_ref, "_lookahead_state", None)
            if _la_state is not None:
                lookahead_accuracy_update(
                    _la_state, getattr(vaayu_ref, "_lookahead_pred", None),
                    layer_idx, set(unique_experts))
            _next_block = getattr(vaayu_ref, "_moe_block_by_idx", {}).get(layer_idx + 1) \
                if isinstance(getattr(vaayu_ref, "_moe_block_by_idx", None), dict) else None
            if _next_block is not None:
                try:
                    _next_router = getattr(_next_block.moe_module, _next_block.router_attr)
                    _nr = _next_router(flat_hidden)
                    _nl = _nr[0] if isinstance(_nr, tuple) else _nr
                    _, _pred_idx = torch.topk(_nl.float(), top_k, dim=-1)
                    _pred_set = set(int(x) for x in _pred_idx.flatten().cpu().unique().tolist())
                    vaayu_ref._lookahead_pred = (layer_idx + 1, _pred_set)
                    if getattr(vaayu_ref, "_lookahead_enabled", False):
                        _next_name = _next_block.layer_name
                        for _pe in _pred_set:
                            if not expert_cache.has(layer_idx + 1, _pe):
                                vaayu_ref._stager_executor.submit(
                                    _load_predicted_expert, _next_name, layer_idx + 1, _pe)
                except Exception as e:
                    _log.warning(
                        f"Lookahead router probe failed for layer {layer_idx + 1}: {e}"
                    )
                    vaayu_ref._lookahead_pred = None

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

        expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            unique_experts,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref, staged_layer=_staged_layer,
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

        final_hidden = torch.zeros_like(flat_hidden)
        intermediate_size = None  # inferred lazily from the first expert we see

        _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
        # Prefill: don't churn the GPU LRU with a whole-model sweep.
        if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
            _vram = None
        # Layer-range restriction (Fix #7 follow-up): only layers in the
        # cacheable prefix use the VRAM cache; None means no restriction.
        _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
        if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
            _vram = None

        if getattr(vaayu_ref, "_overlap_trace", False):
            print(f"[VAAYU-OVERLAP] compute_start layer={layer_idx} "
                  f"t={time.perf_counter():.6f} n_experts={len(unique_experts)}")
        try:
            for eidx, raw in _iter_expert_loads(expert_tensors, _miss_futures, vaayu_ref, _mp if _prof else None):
                groups = _group_exl3_tensors_by_weight(raw)
                if not groups["gate"] or not groups["down"] or not groups["up"]:
                    _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                    continue
                if intermediate_size is None:
                    _t_gate = groups.get("gate", {}).get("trellis")
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

                _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
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
                    final_hidden.index_add_(0, token_idx_gpu, hidden.to(final_hidden.dtype))
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

        vaayu_ref._expert_history[layer_idx] = unique_experts
        _trace_moe_block(vaayu_ref, layer_idx, "fused_exl3", flat_hidden,
                         final_hidden, routing_weights, unique_experts)
        return final_hidden.view(B, S, H)

    return sparse_forward

