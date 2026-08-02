"""Pinned layer-staging buffer sizing (Plan J Task 2).

The caching host allocator charges the next power of two for a pinned request,
so ask for that size explicitly and use all of it. CPU-only — pure arithmetic,
no allocation.
"""
import pytest

from vaayu.persist.safetensor_model_persister import (
    _ALIGN,
    next_pow2_at_least,
    pinned_alloc_size,
)


def test_next_pow2_is_never_smaller_than_the_request():
    for req in (1, 1023, 1024, 1025, 1_303_000_000):
        n = next_pow2_at_least(req)
        assert n >= req
        assert n & (n - 1) == 0


def test_next_pow2_is_identity_on_exact_powers():
    for p in range(0, 35):
        assert next_pow2_at_least(1 << p) == (1 << p)


def test_next_pow2_of_nonpositive_is_zero():
    assert next_pow2_at_least(0) == 0
    assert next_pow2_at_least(-1) == 0


@pytest.mark.parametrize("total_size", [
    1,                    # degenerate tiny header-only shard
    100 * 1024 ** 2,      # small non-expert shard
    1_229_000_000,        # MiniMax-like layer shard
    1_500_000_000,        # Qwen-like layer shard
])
def test_alloc_size_covers_the_o_direct_worst_case(total_size):
    # io_uring reads _align_up(total_size + prefix_pad) bytes with
    # prefix_pad < _ALIGN, so the buffer must cover total_size + 2*_ALIGN.
    assert pinned_alloc_size(total_size) >= total_size + 2 * _ALIGN


@pytest.mark.parametrize("total_size", [1, 100 * 1024 ** 2, 1_229_000_000])
def test_alloc_size_is_a_power_of_two(total_size):
    n = pinned_alloc_size(total_size)
    assert n & (n - 1) == 0


def test_alloc_size_no_longer_pays_the_100mb_pad():
    # A ~0.1 MB shard used to request 100 MB of pad and be charged a 128 MiB
    # block. It must now land in a far smaller power-of-two block.
    assert pinned_alloc_size(100_000) < 8 * 1024 ** 2


def test_alloc_size_is_monotonic():
    prev = 0
    for total_size in (1, 4096, 10 ** 6, 10 ** 8, 10 ** 9, 2 * 10 ** 9):
        cur = pinned_alloc_size(total_size)
        assert cur >= prev
        prev = cur


def test_reuse_boundary_is_pinned_to_pinned_alloc_size():
    """The call site's reuse test is
    `_PINNED_SIZES[buf_idx] < pinned_alloc_size(total_size)`. Pin the
    invariant it depends on with pure arithmetic over `pinned_alloc_size` --
    no allocation, no `vaayu_rust`. A recorded buffer size may only be
    treated as sufficient for a later `total_size` when
    `total_size + 2 * _ALIGN <= recorded size`; testing the old bound
    (`recorded_size < total_size`) is not enough, since it ignores the
    io_uring O_DIRECT slack and the power-of-two step.
    """
    # First shard: a 100 MiB request is charged a 128 MiB block.
    t0 = 100 * 1024 ** 2
    recorded_size = pinned_alloc_size(t0)
    assert recorded_size == 128 * 1024 ** 2

    # A later shard of exactly 128 MiB (== the recorded block size) must NOT
    # be treated as reusable: its own pinned_alloc_size need steps up to the
    # next power of two, since total_size + 2*_ALIGN pushes past 128 MiB.
    later_total_size = 134_217_728  # 128 MiB
    assert later_total_size + 2 * _ALIGN > recorded_size
    must_reallocate = recorded_size < pinned_alloc_size(later_total_size)
    assert must_reallocate

    # The old, buggy bound (`recorded_size < later_total_size`) would have
    # said "no reallocation needed" here -- silently reusing an undersized
    # buffer and writing past its end.
    assert not (recorded_size < later_total_size)


def test_alloc_size_exact_values():
    # Exact figure from the on-device dense smoke run (Task 2 report): a
    # ~116.02 MiB shard is charged a 128 MiB block. Every other assertion in
    # this file is a property (>= request, power of two, monotonic, < 8 MiB)
    # that a regression like `4 * next_pow2_at_least(...)` would still pass.
    assert pinned_alloc_size(121_643_008) == 128 * 1024 ** 2

    # Boundary: total_size itself already an exact power of two must still
    # step up one block, since the +2*_ALIGN slack pushes past the boundary.
    assert pinned_alloc_size(128 * 1024 ** 2) == 256 * 1024 ** 2
