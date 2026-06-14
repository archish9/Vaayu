"""
MoE Expert-Aware Streaming: monkey-patch utilities for Vaayu.

detect_moe_blocks() — generic detection of MoE blocks in any architecture.
patch_moe_blocks()  — replaces each MoE block's forward() with a sparse version.

Design rationale: see docs/superpowers/specs/2026-06-14-moe-expert-aware-streaming-design.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .persist.expert_cache import ExpertRAMCache
    from .persist.expert_offset_index import ExpertOffsetIndex

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


def detect_moe_blocks(model: nn.Module) -> list[MoEBlockInfo]:
    """
    Generic MoE block detector. Finds any module that directly contains:
      1. An nn.ModuleList with > _MIN_EXPERTS children  (the expert FFN list)
      2. A sibling nn.Linear whose attribute name is in _ROUTER_NAMES  (the gate)

    Works across Mixtral (block_sparse_moe), DeepSeek (mlp), Qwen MoE (mlp),
    and any future architecture following the same structural pattern.
    """
    results: list[MoEBlockInfo] = []
    seen_paths: set[str] = set()

    for mod_path, mod in model.named_modules():
        experts_attr: str | None = None
        router_attr: str | None = None

        _found_expert_lists: list[str] = []
        for child_name, child in mod.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > _MIN_EXPERTS:
                _found_expert_lists.append(child_name)
            elif isinstance(child, nn.Linear) and child_name in _ROUTER_NAMES:
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

        # Extract layer index from path (e.g. "model.layers.3.block_sparse_moe" → 3)
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
    blocks = detect_moe_blocks(model)
    for block_info in blocks:
        sparse_fwd = _make_sparse_moe_forward(
            block_info=block_info,
            expert_cache=expert_cache,
            offset_index=offset_index,
            checkpoint_path=Path(checkpoint_path),
            top_k=top_k,
            device=device,
            dtype=dtype,
            vaayu_ref=vaayu_ref,
        )
        # Fix 1: direct assignment — no __get__ binding needed
        block_info.moe_module.forward = sparse_fwd
        n_experts = len(getattr(block_info.moe_module, block_info.experts_attr))
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
):
    """
    Returns a replacement forward() for the MoE block.

    Closure captures: block_info, expert_cache, offset_index, checkpoint_path,
    top_k, device, dtype, vaayu_ref.

    Called as: moe_block.forward(hidden_states, **kwargs)
    Returns:   (hidden_states, router_logits) matching standard MoE forward signature
    """
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx  = block_info.layer_idx

    # Fix 2: simple signature — no bound-method dual-mode needed
    def sparse_forward(hidden_states, **kwargs):
        # 1. Run router gate (permanently pinned in VRAM by _init_pinned_layers)
        router = getattr(block_info.moe_module, block_info.router_attr)
        router_logits = router(hidden_states)
        routing_weights, expert_indices = torch.topk(
            router_logits.float().softmax(dim=-1), top_k, dim=-1
        )
        # Fix 3: normalize routing weights so they sum to 1 (matches actual MoE behavior)
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
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        # 3. Temporarily inject active expert weights → device, run FFN, unload
        experts_list = getattr(block_info.moe_module, block_info.experts_attr)
        B, S, H = hidden_states.shape
        final_hidden = torch.zeros_like(hidden_states)

        # Fix 4 + Fix 5: Phase A — inject all unique experts before token loop
        injected_experts: set[int] = set()
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue
            expert_mod = experts_list[eidx]
            # Fix 5: use _inject_expert_weights helper; skip expert if any weight fails
            ok = _inject_expert_weights(expert_mod, raw, eidx, layer_name, device, dtype)
            if ok:
                injected_experts.add(eidx)

        # Fix 4: Phase B — token × top_k accumulation with correct batch handling
        for s in range(S):
            for k in range(top_k):
                # expert_indices shape: (B, S, top_k)
                for b in range(B):
                    eidx = int(expert_indices[b, s, k].item())
                    if eidx not in injected_experts:
                        continue
                    weight = routing_weights[b, s, k]
                    expert_mod = experts_list[eidx]
                    expert_out = expert_mod(hidden_states[b:b+1, s, :])  # (1, H)
                    final_hidden[b, s, :] += weight * expert_out.squeeze(0)

        # Fix 4: Phase C — unload all injected experts back to meta
        for eidx in injected_experts:
            try:
                experts_list[eidx].to("meta")
            except Exception as e:
                _log.warning(f"Failed to unload expert {eidx} to meta: {e}")

        # 4. Write routing result to vaayu_ref for speculative prefetch scheduling
        vaayu_ref._expert_history[layer_idx] = unique_experts
        # Return tuple matching original MoE forward signature: (hidden_states, router_logits)
        return final_hidden, router_logits

    return sparse_forward
