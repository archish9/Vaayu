"""Fix #9: speculative verify working set derived from geometry — model-agnostic."""
import pytest
from vaayu.vaayu_base import compute_spec_working_set

MB = 1024 ** 2
GB = 1024 ** 3


@pytest.mark.parametrize("layers,experts,expert_mb,expect_lo_gb,expect_hi_gb", [
    (91, 128, 11.28, 40.0, 55.0),   # Qwen-like: min(128, 6*8)=48 unique/layer -> ~48 GB
    (62, 256, 11.30, 28.0, 40.0),   # MiniMax-like: 48 unique/layer -> ~33 GB
])
def test_spec_working_set_amplifies(layers, experts, expert_mb, expect_lo_gb, expect_hi_gb):
    geo = {
        "expert_bytes": int(expert_mb * MB),
        "num_moe_layers": layers,
        "top_k": 8,
        "num_experts": experts,
        "working_set_bytes": 8 * layers * int(expert_mb * MB),
    }
    # draft_tokens=5 -> 6 verify positions -> up to min(num_experts, 48) unique/layer
    spec_ws = compute_spec_working_set(geo, verify_positions=6)
    single = geo["working_set_bytes"]
    assert spec_ws > 2 * single, "verify pass must be recognized as >2x single-token"
    assert expect_lo_gb * GB < spec_ws < expect_hi_gb * GB


def test_single_position_equals_normal_working_set():
    geo = {"expert_bytes": 10 * MB, "num_moe_layers": 50, "top_k": 8,
           "num_experts": 128, "working_set_bytes": 8 * 50 * 10 * MB}
    assert compute_spec_working_set(geo, verify_positions=1) == geo["working_set_bytes"]


def test_capped_at_num_experts():
    # Tiny expert count: positions can't route to more experts than exist.
    geo = {"expert_bytes": 10 * MB, "num_moe_layers": 50, "top_k": 8,
           "num_experts": 16, "working_set_bytes": 8 * 50 * 10 * MB}
    spec_ws = compute_spec_working_set(geo, verify_positions=100)
    assert spec_ws == 16 * 50 * 10 * MB


def test_spec_decode_fits_decision():
    from vaayu.vaayu_base import spec_decode_fits
    GB2 = 1024 ** 3
    geo_qwen = {"expert_bytes": int(11.28 * MB), "num_moe_layers": 91,
                "top_k": 8, "num_experts": 128,
                "working_set_bytes": 8 * 91 * int(11.28 * MB)}
    geo_mini = {"expert_bytes": int(11.30 * MB), "num_moe_layers": 62,
                "top_k": 8, "num_experts": 256,
                "working_set_bytes": 8 * 62 * int(11.30 * MB)}
    budget = int(8.5 * GB2)
    # draft_tokens=5: neither model's verify pass fits the hardware ceiling.
    assert spec_decode_fits(geo_qwen, draft_tokens=5, budget_bytes=budget) is False
    assert spec_decode_fits(geo_mini, draft_tokens=5, budget_bytes=budget) is False
    # draft_tokens=0 (no speculation): reduces to single-token working set.
    assert spec_decode_fits(geo_mini, draft_tokens=0, budget_bytes=budget) is True
