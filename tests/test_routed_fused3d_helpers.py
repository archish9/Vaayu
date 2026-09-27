"""S8 Task 1: the pure pieces of routed fused-3D delivery.

The adapter must return the KERNEL'S OWN routed set (the experts its routing
histogram counts), never a re-derivation -- a separate top-k can break a tie
differently and leave the kernel reading a row nobody uploaded.
"""
import pytest
import torch
import torch.nn as nn

from vaayu.moe_patch import MoEBlockInfo
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.routed_fused3d import (allocate_expert_buffers, fused3d_geometry_key,
                                  routed_delivery_enabled, routed_experts_from_call)


class _Routing:
    def __init__(self, expt_hist):
        self.expt_hist = expt_hist


def test_adapter_reads_the_histogram_the_routing_computed():
    hist = torch.tensor([0, 3, 0, 1, 0, 0, 2, 0], dtype=torch.int32)
    assert routed_experts_from_call((torch.zeros(2, 4), _Routing(hist), None), {}, 8) == [1, 3, 6]


def test_adapter_finds_the_routing_in_kwargs_too():
    hist = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int64)
    assert routed_experts_from_call((), {"routing_data": _Routing(hist)}, 8) == [0, 7]


@pytest.mark.parametrize("hist", [
    torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),     # float: not a count
    torch.tensor([True, False] * 4),                               # bool
    torch.tensor([0, 1, 0, 0], dtype=torch.int32),                 # wrong length
    torch.zeros(2, 8, dtype=torch.int32),                          # 2-D
    None,                                                          # absent
])
def test_adapter_refuses_anything_that_is_not_a_per_expert_count(hist):
    assert routed_experts_from_call((_Routing(hist),), {}, 8) is None


def test_adapter_returns_none_for_a_call_with_no_routing_object():
    assert routed_experts_from_call((torch.zeros(3, 4), torch.tensor([[1, 2]])), {}, 8) is None


def _metas(prefix, gate_shape, dtype="U8"):
    return {f"{prefix}.mlp.experts.gate_up_proj_blocks":
                ExpertTensorMeta(0, 0, list(gate_shape), dtype),
            f"{prefix}.mlp.experts.gate_up_proj_bias":
                ExpertTensorMeta(0, 0, [gate_shape[0], gate_shape[1]], "BF16")}


def test_geometry_key_ignores_the_layer_name():
    assert (fused3d_geometry_key(_metas("model.layers.0", (32, 5760, 90, 16)))
            == fused3d_geometry_key(_metas("model.layers.23", (32, 5760, 90, 16))))


def test_geometry_key_separates_a_different_shape_or_dtype():
    base = fused3d_geometry_key(_metas("model.layers.0", (32, 5760, 90, 16)))
    assert base != fused3d_geometry_key(_metas("model.layers.0", (32, 2880, 90, 16)))
    assert base != fused3d_geometry_key(_metas("model.layers.0", (32, 5760, 90, 16), "I8"))


def test_buffers_take_the_checkpoint_shape_and_dtype_per_leaf():
    bufs = allocate_expert_buffers(_metas("model.layers.0", (4, 6, 3, 2)), "cpu")
    assert set(bufs) == {"gate_up_proj_blocks", "gate_up_proj_bias"}
    assert bufs["gate_up_proj_blocks"].shape == (4, 6, 3, 2)
    assert bufs["gate_up_proj_blocks"].dtype is torch.uint8
    assert bufs["gate_up_proj_bias"].shape == (4, 6)
    assert bufs["gate_up_proj_bias"].dtype is torch.bfloat16


class _FakeIndex:
    def __init__(self, fused_layer_tensors):
        self.fused_layer_tensors = fused_layer_tensors


class _Ref:
    def __init__(self, verdicts):
        self._routed_fused3d_ok = verdicts


def _block(num_experts, fused_param_shape, expert_style="fused"):
    experts = nn.Module()
    experts.num_experts = num_experts
    if fused_param_shape is not None:
        # meta: only the SHAPE is read, and Qwen3-Next's is 1.07 G elements.
        experts.register_parameter(
            "gate_up_proj", nn.Parameter(torch.empty(fused_param_shape, device="meta"),
                                         requires_grad=False))
    moe = nn.Module()
    moe.experts = experts
    moe.gate = nn.Linear(8, num_experts, bias=False)
    return MoEBlockInfo(layer_idx=0, layer_name="model.layers.0",
                        module_path="model.layers.0.mlp", moe_module=moe,
                        router_attr="gate", experts_attr="experts", expert_style=expert_style)


# CLAUDE.md §5: two contrasting geometries at least -- here the three that decide it.
def test_gptoss_like_block_is_enabled_only_by_a_passing_verdict():
    b = _block(32, (32, 5760, 90))
    metas = _metas("model.layers.0", (32, 5760, 90, 16))
    idx = _FakeIndex({"model.layers.0": metas})
    assert routed_delivery_enabled(b, idx, _Ref({fused3d_geometry_key(metas): True})) is True
    assert routed_delivery_enabled(b, idx, _Ref({fused3d_geometry_key(metas): False})) is False
    assert routed_delivery_enabled(b, idx, _Ref({})) is False
    assert routed_delivery_enabled(b, idx, object()) is False     # no verdicts at all


def test_qwen3next_like_block_is_never_routed_fused3d():
    """Fused module params, PER-EXPERT checkpoint: no whole-layer metas, so it
    stays on the tuned per-expert fused forward whatever the verdict map says."""
    b = _block(512, (512, 1024, 2048))
    assert routed_delivery_enabled(b, _FakeIndex({}), _Ref({("anything",): True})) is False


def test_minimax_like_module_list_block_is_never_routed_fused3d():
    b = _block(256, None, expert_style="module_list")
    idx = _FakeIndex({"model.layers.0": _metas("model.layers.0", (256, 8, 4, 2))})
    assert routed_delivery_enabled(b, idx, _Ref({})) is False
