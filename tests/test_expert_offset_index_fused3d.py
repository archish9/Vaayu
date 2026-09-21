"""A2: checkpoints that fuse a layer's experts into one tensor.

gpt-oss-20b stores `model.layers.N.mlp.experts.gate_up_proj_blocks` of shape
[32, 5760, 90, 16] -- all 32 experts on a leading axis -- so the `.experts.N.`
regex matches nothing. Detection is by capability: an `experts.` prefix with no
numeric component, and a leading dimension equal to the config's expert count.
Both conditions, so a coincidentally-3D tensor is not captured.
"""
import json
import struct

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex


@pytest.mark.parametrize("key,shape,expected,want", [
    # gpt-oss shape: fused, leading axis is the expert count
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [32, 5760, 90, 16], 32, True),
    ("model.layers.0.mlp.experts.down_proj_scales", [32, 2880, 90], 32, True),
    ("model.layers.0.mlp.experts.gate_up_proj_bias", [32, 5760], 32, True),
    # a DIFFERENT fused geometry -- nothing here may be gpt-oss-specific
    ("model.layers.3.block_sparse_moe.experts.w1", [8, 1024, 512], 8, True),
    # per-expert layout: NOT fused, the existing path owns it
    ("model.layers.0.mlp.experts.0.w1.weight", [4096, 4096], 32, False),
    # leading dim does not match the expert count -> not the expert axis
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [16, 5760, 90], 32, False),
    # not under an experts prefix at all
    ("model.layers.0.self_attn.q_proj.weight", [32, 4096], 32, False),
    # no expectation available -> cannot claim the leading axis means experts
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [32, 5760, 90], None, False),
    # 1-D tensor under an experts prefix: no axis to split
    ("model.layers.0.mlp.experts.gate_up_proj_bias", [32], 32, False),
])
def test_fused_expert_axis_detection(key, shape, expected, want):
    assert ExpertOffsetIndex._fused_expert_axis(key, shape, expected) is want
