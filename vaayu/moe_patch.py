"""
MoE Expert-Aware Streaming: monkey-patch utilities for Vaayu.

detect_moe_blocks() — generic detection of MoE blocks in any architecture.
patch_moe_blocks()  — replaces each MoE block's forward() with a sparse version.

Design rationale: see docs/superpowers/specs/2026-06-14-moe-expert-aware-streaming-design.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

if TYPE_CHECKING:
    from .persist.expert_cache import ExpertRAMCache
    from .persist.expert_offset_index import ExpertOffsetIndex

_ROUTER_NAMES = {"gate", "router", "w_gate", "router_weights", "gate_up_proj"}
_MIN_EXPERTS = 4  # ModuleList must have > this many children to be an expert list


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

        for child_name, child in mod.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > _MIN_EXPERTS:
                experts_attr = child_name
            elif isinstance(child, nn.Linear) and child_name in _ROUTER_NAMES:
                router_attr = child_name

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
