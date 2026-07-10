"""Fix #5: unique experts are computed once across all draft positions."""
import torch
from vaayu.moe_patch import unique_expert_indices


def test_dedup_across_positions():
    # 5 positions (rows), top_k=3 each; overlapping experts across positions.
    idx = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [1, 4, 5],
        [1, 2, 5],
        [3, 4, 5],
    ])
    uniq = unique_expert_indices(idx)
    assert uniq == [1, 2, 3, 4, 5]  # 5 unique, not 15 per-position


def test_dedup_single_position():
    idx = torch.tensor([[7, 7, 8]])  # duplicate within a position too
    assert unique_expert_indices(idx) == [7, 8]
