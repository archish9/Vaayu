import pytest
import torch

from vaayu.expert_usage import ExpertUsageTracker


def test_record_accumulates_routes():
    t = ExpertUsageTracker(n_rows=8, n_experts=16, device="cpu")
    # two tokens, top_k=2 -> experts 3,5 and 3,9
    idx = torch.tensor([[3, 5], [3, 9]], dtype=torch.int64)
    t.record(layer_idx=4, expert_indices=idx)
    counts = t.snapshot()
    assert counts[4, 3].item() == 2
    assert counts[4, 5].item() == 1
    assert counts[4, 9].item() == 1
    assert counts[4].sum().item() == 4


def test_record_lands_in_the_model_layer_row_not_a_moe_ordinal():
    """Qwen's MoE layers start at model layer 3. Rows are indexed by the model's
    own layer_idx, so recording layer 3 must not land in row 0."""
    t = ExpertUsageTracker(n_rows=8, n_experts=16, device="cpu")
    t.record(layer_idx=3, expert_indices=torch.tensor([[1]], dtype=torch.int64))
    counts = t.snapshot()
    assert counts[3, 1].item() == 1
    assert counts[0].sum().item() == 0


def test_record_accumulates_across_calls():
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    idx = torch.tensor([[2]], dtype=torch.int64)
    for _ in range(5):
        t.record(layer_idx=1, expert_indices=idx)
    assert t.snapshot()[1, 2].item() == 5


@pytest.mark.parametrize("n_rows,n_experts,hot", [
    (94, 128, 127),   # Qwen-like:    94 layers, 128 experts
    (62, 256, 255),   # MiniMax-like: 62 layers, 256 experts
])
def test_shape_and_range_for_both_geometries(n_rows, n_experts, hot):
    """§5: sized from geometry, and the top expert index of each family is in
    range (a 256-expert index would overflow a 128-wide tensor)."""
    t = ExpertUsageTracker(n_rows=n_rows, n_experts=n_experts, device="cpu")
    assert tuple(t.counts.shape) == (n_rows, n_experts)
    t.record(layer_idx=n_rows - 1, expert_indices=torch.tensor([[hot]], dtype=torch.int64))
    assert t.snapshot()[n_rows - 1, hot].item() == 1


def test_snapshot_is_a_copy_not_a_view():
    """Report code must not be able to mutate live counters."""
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    t.record(layer_idx=0, expert_indices=torch.tensor([[1]], dtype=torch.int64))
    snap = t.snapshot()
    snap[0, 1] = 999
    assert t.snapshot()[0, 1].item() == 1


def test_record_accepts_flat_indices():
    """Some routing paths hand back a flat index tensor rather than [tokens, top_k]."""
    t = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    t.record(layer_idx=2, expert_indices=torch.tensor([1, 1, 3], dtype=torch.int64))
    counts = t.snapshot()
    assert counts[2, 1].item() == 2
    assert counts[2, 3].item() == 1
