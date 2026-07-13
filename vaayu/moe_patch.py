"""
MoE Expert-Aware Streaming: monkey-patch utilities for Vaayu.

detect_moe_blocks() — generic detection of MoE blocks in any architecture.
patch_moe_blocks()  — replaces each MoE block's forward() with a sparse version.

Design rationale: see docs/superpowers/specs/2026-06-14-moe-expert-aware-streaming-design.md
"""

from __future__ import annotations

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
    """
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
) -> bool:
    """
    Inject expert tensor weights into expert_mod for inference.
    Returns True if all weights injected, False if any failed (expert skipped).

    Uses regex to strip the full key prefix generically:
        "layers.0.block_sparse_moe.experts.0.w1.weight" → "w1.weight"
    Works for Mixtral, DeepSeek, Kimi, Qwen MoE without hardcoding architecture names.
    """
    for param_key, tensor in raw.items():
        m = _EXPERT_PREFIX_RE.match(param_key)
        if m is None:
            continue  # not an expert tensor (shouldn't happen but be safe)
        local_key = param_key[m.end():]  # e.g. "w1.weight"
        local_parts = local_key.split(".")
        try:
            submod = expert_mod
            for part in local_parts[:-1]:
                submod = getattr(submod, part)
            t = tensor.to(device, dtype=dtype, non_blocking=True)
            setattr(submod, local_parts[-1], nn.Parameter(t, requires_grad=False))
        except Exception as e:
            _log.warning(
                f"Expert weight inject failed: {param_key} → {local_key}: {e}"
            )
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
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx  = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
        if layer_name in vaayu_ref.pinned_layer_names:
            return original_forward(hidden_states, **kwargs)

        # 1. Run router gate (permanently pinned in VRAM by _init_pinned_layers)
        router = getattr(block_info.moe_module, block_info.router_attr)
        router_logits = router(hidden_states)
        routing_weights, expert_indices = torch.topk(
            router_logits.float().softmax(dim=-1), top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        # 2. Load missing experts: ExpertRAMCache → sparse io_uring
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    print(f"[Vaayu-MoE] warn: no offset for {layer_name} expert {eidx}")
                    continue
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        # 3. Temporarily inject active expert weights → device, run FFN, unload
        experts_list = getattr(block_info.moe_module, block_info.experts_attr)
        B, S, H = hidden_states.shape
        final_hidden = torch.zeros_like(hidden_states)

        injected_experts: set[int] = set()
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue
            expert_mod = experts_list[eidx]
            ok = _inject_expert_weights(expert_mod, raw, eidx, layer_name, device, dtype)
            if ok:
                injected_experts.add(eidx)

        # Vectorized: gather all positions routed to each expert and run the
        # expert FFN once per expert on the batch of its tokens, instead of a
        # triple Python loop with a .item() GPU sync per position.
        _idx_cpu = expert_indices.cpu()
        for eidx in injected_experts:
            b_idx, s_idx, k_idx = torch.where(_idx_cpu == eidx)
            if b_idx.numel() == 0:
                continue
            b_gpu = b_idx.to(hidden_states.device)
            s_gpu = s_idx.to(hidden_states.device)
            k_gpu = k_idx.to(hidden_states.device)
            expert_mod = experts_list[eidx]
            expert_out = expert_mod(hidden_states[b_gpu, s_gpu])          # (M, H)
            w = routing_weights[b_gpu, s_gpu, k_gpu].unsqueeze(-1)
            final_hidden.index_put_((b_gpu, s_gpu),
                                    (w * expert_out).to(final_hidden.dtype),
                                    accumulate=True)

        for eidx in injected_experts:
            try:
                experts_list[eidx].to("meta")
            except Exception as e:
                _log.warning(f"Failed to unload expert {eidx} to meta: {e}")

        # 4. Write routing result to vaayu_ref for speculative prefetch scheduling
        vaayu_ref._expert_history[layer_idx] = unique_experts
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

        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

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
        load_expert_sparse, load_expert_into_slab, views_from_packed_copy)

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def shard_file_for(name: str) -> str:
        return str(checkpoint_path / f"{name}.safetensors")

    def _load_predicted_expert(next_layer_name: str, next_layer_idx: int, eidx: int):
        """Phase-B prefetch: load a predicted next-layer expert into the RAM
        cache ahead of its layer. Uses the legacy clone path + prefetch=True so
        the existing prefetch_used telemetry measures usefulness. Never raises."""
        try:
            if expert_cache.has(next_layer_idx, eidx):
                return
            metas = offset_index.get_expert_offsets(next_layer_name, eidx)
            if metas is None:
                return
            raw = load_expert_sparse(shard_file_for(next_layer_name), metas)
            expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
        except Exception as e:
            _log.warning(
                f"Lookahead prefetch failed for layer {next_layer_idx} expert {eidx}: {e}"
            )

    def sparse_forward(hidden_states, **kwargs):
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
        else:
            router_logits = router_res
            routing_weights, expert_indices = torch.topk(
                router_logits.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        if _prof:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        expert_indices_cpu = expert_indices.cpu()
        unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)

        # ── Lookahead routing predictor (Task 12, decode only) ──────────────
        # Phase A (always on): score the PREVIOUS layer's prediction for this
        # layer against the actual routing — pure accounting, no behavior change.
        # Phase B (env VAAYU_LOOKAHEAD=1): prefetch predicted next-layer misses.
        # Lookahead subsystem (Phase A accounting + Phase B prefetch) is now
        # fully gated behind VAAYU_LOOKAHEAD=1 (CR-1): the extra next-layer
        # router GEMM + _pred_idx.cpu() D2H sync are per-layer/per-token decode
        # overhead that only Phase B consumes. Default off → steady-state decode
        # pays nothing. The inner os.environ check on the prefetch submit below
        # is now redundant but harmless; leave it.
        if (not getattr(vaayu_ref, "_moe_prefill_pass", False)
                and os.environ.get("VAAYU_LOOKAHEAD") == "1"):
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
                    if os.environ.get("VAAYU_LOOKAHEAD") == "1":
                        _next_name = _next_block.layer_name
                        for _pe in _pred_set:
                            if not expert_cache.has(layer_idx + 1, _pe):
                                vaayu_ref._moe_executor.submit(
                                    _load_predicted_expert, _next_name, layer_idx + 1, _pe)
                except Exception as e:
                    _log.warning(
                        f"Lookahead router probe failed for layer {layer_idx + 1}: {e}"
                    )
                    vaayu_ref._lookahead_pred = None

        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0

        shard_file = shard_file_for(layer_name)
        expert_tensors: dict[int, dict] = {}
        misses: list[int] = []

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
                _w0 = time.perf_counter()
                _ev.wait()
                _stall = time.perf_counter() - _w0
                vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
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

        for eidx in unique_experts:
            if _staged_layer is not None and eidx in _staged_layer:
                expert_tensors[eidx] = _staged_layer[eidx]
                continue
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                misses.append(eidx)

        _miss_futures = []
        if misses:
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
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
                    # consumed via the per-component .to(device) copies below
                    # (always naturally aligned on a fresh cross-device copy),
                    # never via views_from_packed_copy — skip the scratch-pool
                    # repack, which regressed decode throughput when applied
                    # unconditionally (see load_expert_into_slab's docstring).
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

            # Submit misses to the persistent pool and DON'T wait: hit experts
            # compute below while these reads run. Only the remainder that hasn't
            # landed when hits are done shows up as disk stall (_expert_disk_s).
            _miss_futures = [vaayu_ref._moe_executor.submit(_load_one, e) for e in misses]

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

        def _iter_experts_hits_then_misses():
            """Yield (eidx, raw): cache hits immediately (compute overlaps the
            in-flight miss reads), then misses in completion order. Blocking
            time in the wait() is the true disk stall and is accumulated into
            vaayu_ref._expert_disk_s (telemetry: stall, not total read time)."""
            for _e in unique_experts:
                _raw = expert_tensors.get(_e)
                if _raw is not None:
                    yield _e, _raw
            _pending = set(_miss_futures)
            while _pending:
                _w0 = time.perf_counter()
                _done, _pending = _futures_wait(_pending, return_when=FIRST_COMPLETED)
                _stall = time.perf_counter() - _w0
                vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
                if _prof:
                    _mp['disk_load'] += _stall
                for _f in _done:
                    _e, _raw = _f.result()
                    if _raw is not None:
                        yield _e, _raw

        _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
        # Prefill: don't churn the GPU LRU with a whole-model sweep.
        if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
            _vram = None
        # Layer-range restriction (Fix #7 follow-up): only layers in the
        # cacheable prefix use the VRAM cache; None means no restriction.
        _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
        if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
            _vram = None

        for eidx, raw in _iter_experts_hits_then_misses():
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
                    _gpu_blob = _slab_cpu[:_used].to(device, non_blocking=True)
                    gpu_gate = views_from_packed_copy(groups["gate"], _slab_cpu, _gpu_blob)
                    gpu_down = views_from_packed_copy(groups["down"], _slab_cpu, _gpu_blob)
                    gpu_up   = views_from_packed_copy(groups["up"],   _slab_cpu, _gpu_blob)
                else:
                    gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                    gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                    gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
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
                        torch.cuda.synchronize()
                    for _sidx in _slabs.values():
                        if _sidx is not None:
                            _arena.release(_sidx)

        if _is_last_call and getattr(vaayu_ref, "_chunk_major_gpu", None) is not None:
            # Drop the retained GPU tensors + BC objects; the CUDA caching
            # allocator reuses the memory for the next layer's retention dict.
            vaayu_ref._chunk_major_gpu = None
            vaayu_ref._chunk_major_gpu_bytes = 0

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return final_hidden.view(B, S, H)

    return sparse_forward

