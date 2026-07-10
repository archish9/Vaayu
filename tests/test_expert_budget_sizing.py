"""Fix #1: budget sizes above the DERIVED working set, up to the hardware ceiling.
Model-agnostic: tested with both Qwen-like and MiniMax-like geometry."""
from vaayu.vaayu_base import compute_expert_budget

GB = 1024 ** 3
MB = 1024 ** 2


def test_qwen_budget_exceeds_working_set_on_32gb_box():
    # Qwen: working set 8.02 GB. 26.5 GB available (from run.log).
    ws = int(8 * 91 * 11.28 * MB)  # ~8.02 GB
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128,
    )
    assert budget <= int(8.5 * GB) + 1, "must not exceed 8.5GB hardware ceiling"
    assert budget > ws, "budget must exceed Qwen working set (kills thrash)"


def test_minimax_budget_already_fits_no_change_needed():
    # M2.5: working set ~5.5 GB < existing budget. Budget still capped at 8.5GB,
    # and comfortably exceeds the smaller working set → no thrash.
    ws = int(8 * 62 * 11.3 * MB)  # ~5.5 GB
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=int(26.5 * GB), working_set_bytes=ws,
        num_moe_layers=62, expert_bytes=int(11.3 * MB),
        moe_top_k=8, moe_num_experts=256,
    )
    assert budget > ws, "M2.5 working set must fit within budget"
    assert budget <= int(8.5 * GB) + 1


def test_budget_floor_on_tiny_ram():
    ws = int(8 * 91 * 11.28 * MB)
    budget, ne_budget, max_per_layer = compute_expert_budget(
        available_bytes=3 * GB, working_set_bytes=ws,
        num_moe_layers=91, expert_bytes=int(11.28 * MB),
        moe_top_k=8, moe_num_experts=128,
    )
    assert budget >= 1 * GB
    assert max_per_layer >= 8  # never below top_k
