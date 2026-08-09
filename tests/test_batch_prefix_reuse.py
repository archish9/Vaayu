"""F3 Phase 1: the batch's shared prefix, the cache broadcast, and the row slice.

All three are pure/duck-typed so they run without a model load -- CLAUDE.md's
method note about load-time decisions that can only be verified by hand-run
harnesses.
"""
import pytest
import torch

from vaayu.batch import BatchRequest, batch_common_prefix


def _req(ids):
    return BatchRequest(prompt_ids=list(ids), max_new_tokens=4)


def test_identical_prompts_share_everything():
    reqs = [_req([1, 2, 3, 4]), _req([1, 2, 3, 4])]
    assert batch_common_prefix(reqs) == [1, 2, 3, 4]


def test_shared_head_divergent_tail():
    reqs = [_req([1, 2, 3, 9]), _req([1, 2, 3, 8, 8]), _req([1, 2, 3])]
    assert batch_common_prefix(reqs) == [1, 2, 3]


def test_no_overlap_returns_empty():
    assert batch_common_prefix([_req([1, 2]), _req([9, 9])]) == []


def test_single_stream_is_its_whole_prompt():
    assert batch_common_prefix([_req([5, 6, 7])]) == [5, 6, 7]


def test_empty_inputs():
    assert batch_common_prefix([]) == []
    assert batch_common_prefix([_req([]), _req([1])]) == []


def test_a_later_request_value_mismatch_truncates():
    """A request beyond index 1 must be able to shorten the prefix by VALUE,
    not just by length -- otherwise a two-request-only implementation that
    truncates by min length passes the whole suite."""
    reqs = [_req([1, 2, 3, 9]), _req([1, 2, 3, 8]), _req([1, 9, 9])]
    assert batch_common_prefix(reqs) == [1]


from vaayu.batch import broadcast_cache_rows, slice_cache_row


class _FakeCache:
    """Duck-types the parts of FP8DynamicCache these helpers touch."""

    def __init__(self, buffers=("key_cache", "value_cache"), n_layers=2,
                 b=1, h=2, s=3, d=4):
        self._buffers = buffers
        for name in buffers:
            setattr(self, name, [torch.randn(b, h, s, d) for _ in range(n_layers)])
        self._seq_len = [s] * n_layers

    @property
    def _BUFFER_LISTS(self):
        return self._buffers


def test_broadcast_expands_every_buffer_to_n_rows():
    c = _FakeCache()
    broadcast_cache_rows(c, 3)
    for name in c._BUFFER_LISTS:
        for t in getattr(c, name):
            assert t.shape[0] == 3


def test_broadcast_rows_do_not_alias():
    """The .expand() trap: shape checks pass, then every stream's decode
    writes land in one shared storage. Write row 0, row 1 must not move."""
    c = _FakeCache()
    broadcast_cache_rows(c, 2)
    before = c.key_cache[0][1].clone()
    c.key_cache[0][0].fill_(99.0)
    assert torch.equal(c.key_cache[0][1], before)


def test_broadcast_preserves_values():
    c = _FakeCache()
    original = c.key_cache[0][0].clone()
    broadcast_cache_rows(c, 4)
    for row in range(4):
        assert torch.equal(c.key_cache[0][row], original)


def test_broadcast_covers_quantized_buffer_names():
    """Cache-mode-agnostic: TurboQuant exposes six differently-named lists."""
    c = _FakeCache(buffers=("key_cache", "value_cache", "_k_signs",
                            "_v_signs", "_k_norms", "_v_norms"))
    broadcast_cache_rows(c, 2)
    for name in c._BUFFER_LISTS:
        assert getattr(c, name)[0].shape[0] == 2


def test_slice_row_reduces_the_batch_dimension():
    """store()'s to_cpu trims only the sequence axis, so a cache handed to it
    still holding N rows would be charged N-fold against the pool budget."""
    c = _FakeCache()
    broadcast_cache_rows(c, 5)
    one = slice_cache_row(c, 0)
    for name in one._BUFFER_LISTS:
        assert getattr(one, name)[0].shape[0] == 1


def test_slice_row_is_independent_of_the_source():
    c = _FakeCache()
    broadcast_cache_rows(c, 2)
    one = slice_cache_row(c, 0)
    one.key_cache[0].fill_(7.0)
    assert not torch.equal(c.key_cache[0][0], one.key_cache[0][0])


def test_slice_row_copies_seq_len_list():
    c = _FakeCache()
    one = slice_cache_row(c, 0)
    one._seq_len[0] = 999
    assert c._seq_len[0] != 999
