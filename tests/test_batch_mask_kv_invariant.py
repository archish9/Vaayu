"""The batched mask must agree with the KV cache it is built against.

build_batch_attention_mask sizes its KV axis from the caller's 2D padding
mask alone -- it never looks at the cache. When the two disagree, the failure
surfaces ~30 frames away inside attention as a bare torch expand error naming
head counts and no cause:

    RuntimeError: The expanded size of the tensor (1690) must match the
    existing size (1127) at non-singleton dimension 3.
    Target sizes: [3, 64, 1127, 1690]. Tensor sizes: [3, 1, 1127, 1127]

That is bug D3's shape in a new subsystem, and it cost one full debugging
cycle on 2026-08-09 without ever being explained. These tests pin the two
invariants that make such a state impossible to reach silently.
"""
import pytest
import torch

from vaayu.vaayu_base import FP8DynamicCache, check_batch_kv_invariant


def _cache(lengths, n=1, heads=2, dim=4):
    """FP8DynamicCache whose layer i holds lengths[i] tokens."""
    c = FP8DynamicCache(quantize=False)
    for i, L in enumerate(lengths):
        if L:
            k = torch.randn(n, heads, L, dim)
            c.update(k, k.clone(), i)
        else:
            c.update(torch.randn(n, heads, 1, dim),
                     torch.randn(n, heads, 1, dim), i)
            c._seq_len[i] = 0
    return c


def test_agreeing_mask_and_cache_pass():
    c = _cache([5, 5], n=3)
    check_batch_kv_invariant(torch.ones(3, 12, dtype=torch.long), 7, c)


def test_no_cache_requires_mask_width_to_be_the_query_length():
    check_batch_kv_invariant(torch.ones(3, 7, dtype=torch.long), 7, None)
    with pytest.raises(RuntimeError, match="past=0"):
        check_batch_kv_invariant(torch.ones(3, 12, dtype=torch.long), 7, None)


def test_the_reported_failure_is_caught_at_its_source():
    """The exact 2026-08-09 state: cache holds 563, mask claims 0 past."""
    c = _cache([563, 563], n=3)
    with pytest.raises(RuntimeError) as e:
        check_batch_kv_invariant(torch.ones(3, 1127, dtype=torch.long), 1127, c)
    msg = str(e.value)
    assert "1127" in msg and "563" in msg and "1690" in msg, msg


def test_ragged_per_layer_lengths_are_caught():
    """get_seq_length() reads layer 0 only, so a cache whose layers disagree
    passes every length check in forward() and then mismatches inside
    attention at whichever layer differs."""
    c = _cache([5, 9])
    with pytest.raises(RuntimeError, match="per-layer"):
        check_batch_kv_invariant(torch.ones(1, 12, dtype=torch.long), 7, c)


def test_caches_without_per_layer_lengths_are_not_rejected():
    """Only Vaayu's own cache classes expose _seq_len; a plain DynamicCache
    must still pass the length check it can support."""
    class Bare:
        def get_seq_length(self, layer_idx=0):
            return 5
    check_batch_kv_invariant(torch.ones(1, 12, dtype=torch.long), 7, Bare())
