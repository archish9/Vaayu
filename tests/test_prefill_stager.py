"""Prefill lookahead stager: gate + stage-depth sizing (both target geometries)."""
import pytest

from vaayu.vaayu_base import compute_stage_gate, compute_stage_depth


GB = 1024 ** 3
MB = 1024 ** 2

QWEN_GEO = {"expert_bytes": 11 * MB, "top_k": 8, "num_experts": 128,
            "num_moe_layers": 91, "working_set_bytes": 8 * 91 * 11 * MB}
MINIMAX_GEO = {"expert_bytes": 11 * MB, "top_k": 8, "num_experts": 256,
               "num_moe_layers": 62, "working_set_bytes": 8 * 62 * 11 * MB}


@pytest.mark.parametrize("geo,long_needed", [(QWEN_GEO, 32), (MINIMAX_GEO, 64)])
def test_stage_gate_long_vs_short_prompt(geo, long_needed):
    # threshold: seq_len * top_k >= 2 * num_experts
    assert compute_stage_gate(seq_len=long_needed, geometry=geo) is True
    assert compute_stage_gate(seq_len=long_needed - 1, geometry=geo) is False
    assert compute_stage_gate(seq_len=439, geometry=geo) is True
    assert compute_stage_gate(seq_len=1, geometry=geo) is False


@pytest.mark.parametrize("geo", [QWEN_GEO, MINIMAX_GEO])
def test_stage_depth_bounded_by_budget_and_cap(geo):
    layer_bytes = geo["num_experts"] * geo["expert_bytes"]
    # generous budget -> capped at 2
    assert compute_stage_depth(8 * GB, geo) == min(2, 8 * GB // layer_bytes - 1)
    # tight budget (one layer barely fits) -> still at least 1
    assert compute_stage_depth(layer_bytes, geo) == 1
    assert compute_stage_depth(0, geo) == 1
