import pytest
import torch

from vaayu.expert_usage import ExpertUsageTracker, format_usage_report


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


def test_report_flags_hot_and_evicted_as_pin_candidate():
    counts = torch.zeros((2, 4), dtype=torch.int32)
    counts[1, 2] = 500      # hot AND evicted -> candidate
    counts[1, 3] = 400      # hot, never evicted -> not a candidate
    out = format_usage_report(counts, {(1, 2): 12}, top_n=5)
    assert "layer=1" in out
    assert "e2" in out and "500" in out and "12" in out
    assert "pin candidate" in out
    assert "no pin needed" in out          # e3 must be labelled the other way


def test_report_says_no_pin_needed_when_nothing_was_evicted():
    """The negative result must be as legible as the positive one — it is what
    retires the pin feature."""
    counts = torch.zeros((1, 3), dtype=torch.int32)
    counts[0, 1] = 900
    out = format_usage_report(counts, {}, top_n=5)
    assert "no pin needed" in out
    assert "pin candidate" not in out


def test_report_skips_layers_with_no_routes():
    counts = torch.zeros((3, 4), dtype=torch.int32)
    counts[2, 0] = 7
    out = format_usage_report(counts, {}, top_n=5)
    assert "layer=2" in out
    assert "layer=0" not in out
    assert "layer=1" not in out


def test_report_on_empty_histogram_does_not_crash():
    out = format_usage_report(torch.zeros((4, 8), dtype=torch.int32), {}, top_n=5)
    assert isinstance(out, str)
    assert "no expert routes recorded" in out


def test_report_warns_when_evictions_land_outside_top_n():
    """An expert just outside the top-N cutoff can absorb all the evictions —
    the summary must not silently report 0% in that case."""
    counts = torch.zeros((1, 6), dtype=torch.int32)
    counts[0, 0] = 100
    counts[0, 1] = 90
    counts[0, 2] = 80
    counts[0, 3] = 70
    counts[0, 4] = 60
    counts[0, 5] = 55       # just outside top_n=5, holds all the evictions
    out = format_usage_report(counts, {(0, 5): 200}, top_n=5)
    assert "WARNING" in out
    assert "200" in out
    assert "top-5" in out
    assert "top_n" in out


def test_report_no_warning_when_evictions_are_all_inside_top_n():
    """When every eviction belongs to a printed top-N expert, no WARNING line
    should appear — the fix must not add noise to the common case."""
    counts = torch.zeros((2, 4), dtype=torch.int32)
    counts[1, 2] = 500
    counts[1, 3] = 400
    out = format_usage_report(counts, {(1, 2): 12}, top_n=5)
    assert "WARNING" not in out


def test_report_respects_top_n():
    counts = torch.zeros((1, 10), dtype=torch.int32)
    for e in range(10):
        counts[0, e] = 100 - e
    out = format_usage_report(counts, {}, top_n=3)
    assert "e0" in out and "e1" in out and "e2" in out
    assert "e9" not in out


def test_hook_is_a_noop_without_a_tracker_and_during_prefill():
    """Ships dark: no tracker -> the hook must return before touching anything.
    And with a tracker, a prefill pass must not be counted — the VRAM cache is
    not consulted during prefill, so counting prefill routes would compare
    decode-side evictions against prefill+decode routes."""
    from vaayu.moe_patch import _maybe_record_usage

    class _Ref:
        pass

    ref = _Ref()                      # no _expert_usage attribute at all
    _maybe_record_usage(ref, 0, torch.tensor([[1]], dtype=torch.int64))   # must not raise

    ref._expert_usage = ExpertUsageTracker(n_rows=4, n_experts=8, device="cpu")
    ref._moe_prefill_pass = True
    _maybe_record_usage(ref, 1, torch.tensor([[2]], dtype=torch.int64))
    assert ref._expert_usage.snapshot().sum().item() == 0     # prefill not counted

    ref._moe_prefill_pass = False
    _maybe_record_usage(ref, 1, torch.tensor([[2]], dtype=torch.int64))
    assert ref._expert_usage.snapshot()[1, 2].item() == 1     # decode counted
