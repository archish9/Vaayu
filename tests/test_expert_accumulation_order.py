"""The MoE forwards must be a deterministic function of their input.

Root cause found 2026-08-15: two of the three forwards accumulated expert
outputs in ARRIVAL order -- VRAM-resident experts first, the rest in disk
COMPLETION order (`_iter_expert_loads` waits FIRST_COMPLETED) -- and floating
point addition is not associative at any width. Measured on Qwen3-235B: three
identical greedy `generate` runs produced TWO distinct token sequences.

The fp32 accumulator that preceded this shrank the per-step error ~2^13x versus
bf16 and its comment claimed fp32 made "the cast result identical whatever the
order". That is false: fp32 error is normally below the bf16 output's precision,
but any value sitting on a bf16 rounding boundary tips anyway, and across 94
layers something always does.

Execution stays in completion order -- the disk-overlap win is the point of it.
Only the SUMMATION order is pinned.
"""
import torch

from vaayu.moe_patch import accumulate_expert_outputs


def _contributions():
    """Values chosen so fp32 addition is visibly non-associative.

    fp32 spacing at 1e8 is ~8, so (1e8 + 1.0) rounds straight back to 1e8 and
    the small term is lost -- but (1e8 + -1e8) + 1.0 keeps it. Same three
    numbers, two different sums, which is the whole defect in miniature.
    Expert ids are deliberately out of arrival order."""
    idx = torch.tensor([0])
    return [
        (7, idx, torch.tensor([[1e8]], dtype=torch.float32)),
        (2, idx, torch.tensor([[-1e8]], dtype=torch.float32)),
        (5, idx, torch.tensor([[1.0]], dtype=torch.float32)),
    ]


def _accumulate(order):
    out = torch.zeros(1, 1, dtype=torch.float32)
    accumulate_expert_outputs(out, order)
    return out


def test_the_fixture_really_is_order_sensitive():
    """Assert the premise: a naive in-order sum of these values DOES depend on
    order, or this file proves nothing (todo.md METHOD NOTES)."""
    c = _contributions()
    naive_a = torch.zeros(1, 1, dtype=torch.float32)
    for _e, i, h in c:
        naive_a.index_add_(0, i, h)
    naive_b = torch.zeros(1, 1, dtype=torch.float32)
    for _e, i, h in reversed(c):
        naive_b.index_add_(0, i, h)
    assert not torch.equal(naive_a, naive_b), "fixture is not order-sensitive"


def test_accumulation_is_bitwise_identical_whatever_the_arrival_order():
    c = _contributions()
    import itertools
    results = [_accumulate(list(p)) for p in itertools.permutations(c)]
    assert all(torch.equal(results[0], r) for r in results[1:]), \
        "same contributions, different arrival order -> different sum"


def test_it_sums_in_EXPERT_ID_order_specifically():
    """Not merely 'some fixed order' -- expert id, so the result is stable
    across runs AND reproducible by hand."""
    c = _contributions()
    expected = torch.zeros(1, 1, dtype=torch.float32)
    for _e, i, h in sorted(c, key=lambda r: r[0]):
        expected.index_add_(0, i, h)
    assert torch.equal(_accumulate(list(reversed(c))), expected)


def test_contributions_are_cast_to_the_accumulator_dtype():
    """Experts hand back fp16; the accumulator is fp32 and must stay fp32."""
    out = torch.zeros(1, 4, dtype=torch.float32)
    idx = torch.tensor([0])
    accumulate_expert_outputs(out, [(0, idx, torch.full((1, 4), 0.5, dtype=torch.float16))])
    assert out.dtype is torch.float32
    assert torch.equal(out, torch.full((1, 4), 0.5, dtype=torch.float32))


def test_overlapping_tokens_from_several_experts_all_land():
    """Each token routes to top_k experts, so rows genuinely overlap -- that
    overlap is why order mattered in the first place."""
    out = torch.zeros(3, 2, dtype=torch.float32)
    accumulate_expert_outputs(out, [
        (1, torch.tensor([0, 2]), torch.ones(2, 2, dtype=torch.float32)),
        (0, torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.float32)),
    ])
    assert torch.equal(out, torch.tensor([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]))
