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
    compute_device = "cuda" if is_exl3 else "cpu"

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
            compute_device=compute_device,
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
):
    """
    Returns a replacement forward() for the MoE block, dispatching based on expert_style.
    """
    if block_info.expert_style == "fused":
        if compute_device == "cuda":
            return _make_fused_sparse_forward_exl3(
                block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
            )
        return _make_fused_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            compute_device=compute_device,
        )
    else:
        return _make_modulelist_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
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
):
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx  = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
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

        for s in range(S):
            for k in range(top_k):
                for b in range(B):
                    eidx = int(expert_indices[b, s, k].item())
                    if eidx not in injected_experts:
                        continue
                    weight = routing_weights[b, s, k]
                    expert_mod = experts_list[eidx]
                    expert_out = expert_mod(hidden_states[b:b+1, s, :])  # (1, H)
                    final_hidden[b, s, :] += weight * expert_out.squeeze(0)

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
):
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
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
            from concurrent.futures import ThreadPoolExecutor as _TPE

            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                # use_pinned_buf=False: fresh buffer per call, no lock → concurrent reads.
                # GIL released inside vaayu_rust.io_uring_read_into (py.allow_threads),
                # so N workers genuinely overlap NVMe reads in parallel.
                raw = load_expert_sparse(shard_file, metas, use_pinned_buf=False)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Up to 8 concurrent reads: amortizes NVMe APST wake-up latency (~100ms)
            # over all misses instead of paying it once per expert serially.
            with _TPE(max_workers=min(len(misses), 8)) as _ex:
                for eidx, raw in _ex.map(_load_one, misses):
                    if raw is not None:
                        expert_tensors[eidx] = raw

        # Expert mask on CPU (avoids PCIe for index tensors)
        expert_indices_cpu = expert_indices.cpu()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)

        if compute_device == "cpu":
            # ── CPU FFN: zero PCIe cost for expert weights ─────────────────
            # Only hidden state crosses PCIe (8 KB decode, 4 MB prefill-512)
            cpu_hidden = flat_hidden.cpu()
            cpu_routing_weights = routing_weights.cpu()
            final_cpu = torch.zeros(flat_hidden.shape[0], H, dtype=cpu_hidden.dtype)

            for eidx in unique_experts:
                raw = expert_tensors.get(eidx)
                if raw is None:
                    continue

                w_gate, w_down, w_up = _classify_expert_tensors(raw)
                if w_gate is None or w_down is None or w_up is None:
                    _log.warning(f"[Vaayu-MoE] layer {layer_idx} expert {eidx}: missing weights, skipping")
                    continue

                top_k_pos, token_idx = torch.where(expert_mask[eidx])
                if token_idx.numel() == 0:
                    continue

                x = cpu_hidden[token_idx]
                gate_out = act_fn(F.linear(x, w_gate))
                up_out = F.linear(x, w_up)
                out = F.linear(gate_out * up_out, w_down)
                out = out * cpu_routing_weights[token_idx, top_k_pos, None]
                final_cpu.index_add_(0, token_idx, out.to(final_cpu.dtype))

            result = final_cpu.to(device, dtype=dtype)

        else:
            # ── GPU path (used for EXL3 or explicit cuda override) ──────────
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
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
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

        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    continue
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        expert_indices_cpu = expert_indices.cpu()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)

        # Infer intermediate_size from first available expert's gate trellis
        intermediate_size = None
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is not None:
                groups = _group_exl3_tensors_by_weight(raw)
                t = groups.get("gate", {}).get("trellis")
                if t is not None:
                    intermediate_size = t.shape[0]
                    break
        if intermediate_size is None:
            _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx}: cannot determine intermediate_size, returning zeros")
            vaayu_ref._expert_history[layer_idx] = unique_experts
            return flat_hidden.view(B, S, H)

        final_hidden = torch.zeros_like(flat_hidden)

        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue

            groups = _group_exl3_tensors_by_weight(raw)
            if not groups["gate"] or not groups["down"] or not groups["up"]:
                _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                continue

            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue
            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]
            if current_state.dtype != torch.float16:
                current_state = current_state.to(torch.float16)

            gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
            gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
            gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}

            bc_gate = bc_down = bc_up = None
            try:
                bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)

                x_flat = current_state.view(-1, H)
                gate_out = act_fn(bc_gate.run_alloc(x_flat, intermediate_size, False))
                up_out   = bc_up.run_alloc(x_flat, intermediate_size, False)
                hidden   = bc_down.run_alloc((gate_out * up_out).contiguous(), H, False)

                weights_slice = routing_weights[token_idx_gpu, top_k_pos.to(device), None]
                hidden = hidden * weights_slice
                final_hidden.index_add_(0, token_idx_gpu, hidden.to(final_hidden.dtype))
            finally:
                # CRITICAL: free C++ QMatrix objects — Python GC cannot manage them
                bc_gate = bc_down = bc_up = None
                del gpu_gate, gpu_down, gpu_up

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return final_hidden.view(B, S, H)

    return sparse_forward

