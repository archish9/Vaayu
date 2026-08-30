"""P1(2): expert-tensor role classification, and the table it is derived from.

`_classify_expert_tensors` and `_group_exl3_tensors_by_weight` run per expert,
per layer, PER TOKEN over a mapping that is fixed when the checkpoint is opened.
The key shapes below are the REAL ones, read off the two on-device checkpoints'
`model.safetensors.index.json` -- Qwen's fused-EXL3 layout and MiniMax's
module-list layout, which are different rows of CLAUDE.md §0's dispatch table
and must both keep working.

These were written to guard a speed-up of the role lookup that was then
REFUTED: the whole function costs 0.13-0.16 % of a token on the two on-device
models, so the best available rewrite banks 0.02-0.05 %. The tests are kept
because nothing else covered these two functions across both expert layouts.
"""
import torch
import pytest

from vaayu.moe_patch import (
    _classify_expert_tensors,
    _group_exl3_tensors_by_weight,
)

# role -> checkpoint segment, per layout. Real prefixes, real component names.
QWEN_PREFIX = "model.layers.7.mlp.experts.42."
QWEN_ROLE_SEG = {"gate": "gate_proj", "down": "down_proj", "up": "up_proj"}
QWEN_COMPONENTS = ["trellis", "suh", "svh"]

MINIMAX_PREFIX = "model.layers.7.block_sparse_moe.experts.42."
MINIMAX_ROLE_SEG = {"gate": "w1", "down": "w2", "up": "w3"}
MINIMAX_COMPONENTS = ["trellis", "suh", "svh", "mcg"]

LAYOUTS = [
    pytest.param(QWEN_PREFIX, QWEN_ROLE_SEG, QWEN_COMPONENTS, id="qwen-fused-exl3"),
    pytest.param(MINIMAX_PREFIX, MINIMAX_ROLE_SEG, MINIMAX_COMPONENTS, id="minimax-module-list"),
]


def _raw(prefix, role_seg, components):
    """A per-expert raw dict as the loader hands it over, tensors distinguishable."""
    raw = {}
    for i, (role, seg) in enumerate(sorted(role_seg.items())):
        for j, comp in enumerate(components):
            raw[f"{prefix}{seg}.{comp}"] = torch.tensor([float(i), float(j)])
    return raw


@pytest.mark.parametrize("prefix,role_seg,components", LAYOUTS)
def test_exl3_grouping_maps_every_component_to_its_role(prefix, role_seg, components):
    groups = _group_exl3_tensors_by_weight(_raw(prefix, role_seg, components))
    assert set(groups) == {"gate", "down", "up"}
    for role, seg in role_seg.items():
        assert set(groups[role]) == set(components), f"{role} lost components"
        for comp in components:
            expected = _raw(prefix, role_seg, components)[f"{prefix}{seg}.{comp}"]
            assert torch.equal(groups[role][comp], expected), f"{role}.{comp} mismatched"


@pytest.mark.parametrize("prefix,role_seg,components", LAYOUTS)
def test_exl3_grouping_drops_unknown_components(prefix, role_seg, components):
    raw = _raw(prefix, role_seg, components)
    seg = role_seg["gate"]
    raw[f"{prefix}{seg}.not_a_component"] = torch.tensor([9.0])
    groups = _group_exl3_tensors_by_weight(raw)
    assert "not_a_component" not in groups["gate"]
    assert set(groups["gate"]) == set(components)


@pytest.mark.parametrize("prefix,role_seg,components", LAYOUTS)
def test_exl3_grouping_ignores_non_expert_keys(prefix, role_seg, components):
    raw = _raw(prefix, role_seg, components)
    raw["model.layers.7.input_layernorm.weight"] = torch.tensor([1.0])
    raw["model.layers.7.self_attn.q_proj.trellis"] = torch.tensor([2.0])
    groups = _group_exl3_tensors_by_weight(raw)
    for role in ("gate", "down", "up"):
        assert set(groups[role]) == set(components)


@pytest.mark.parametrize("prefix,role_seg,components", LAYOUTS)
def test_classify_returns_gate_down_up_in_that_order(prefix, role_seg, components):
    # The non-EXL3 path: one tensor per role, plain `.weight` components.
    raw = {f"{prefix}{seg}.weight": torch.tensor([float(i)])
           for i, (role, seg) in enumerate(sorted(role_seg.items()))}
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    order = sorted(role_seg)          # down, gate, up
    for role, tensor in (("gate", w_gate), ("down", w_down), ("up", w_up)):
        assert tensor is not None, f"{role} not classified"
        assert torch.equal(tensor, torch.tensor([float(order.index(role))]))


def test_classify_returns_none_for_a_role_that_is_absent():
    raw = {f"{QWEN_PREFIX}gate_proj.weight": torch.tensor([0.0])}
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    assert w_gate is not None
    assert w_down is None and w_up is None
