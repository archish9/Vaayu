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

    def crop(self, max_length: int) -> None:
        for i in range(len(self._seq_len)):
            self._seq_len[i] = min(self._seq_len[i], max_length)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len[layer_idx]


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


from vaayu.batch import batch_position_ids, build_batch_attention_mask


def test_prefix_mask_and_positions_are_absolute():
    """With m cached tokens and left-padded tails, tail tokens must land on
    absolute positions m..m+r_i-1, and tail queries must see the cache."""
    m, max_R = 3, 4
    tail_mask = torch.tensor([[1, 1, 1, 1],      # r=4, no pad
                              [0, 0, 1, 1]])     # r=2, two pads
    pad_mask = torch.cat([torch.ones(2, m, dtype=torch.long), tail_mask], dim=1)

    pos = batch_position_ids(pad_mask)
    assert pos.shape == (2, m + max_R)
    assert pos[0, m:].tolist() == [3, 4, 5, 6]     # r=4 -> m..m+3
    assert pos[1, m:].tolist() == [2, 2, 3, 4]     # pads clamp, reals m..m+1

    mask = build_batch_attention_mask(pad_mask, max_R, torch.float32, "cpu")
    assert mask.shape == (2, 1, max_R, m + max_R)
    neg = torch.finfo(torch.float32).min
    assert (mask[0, 0, 0, :m] != neg).all()        # first tail query sees cache
    assert (mask[1, 0, :, m:m + 2] == neg).all()   # row 1's pads are masked


from vaayu.batch import prefix_store_row, resolve_matched_len


def test_prefix_store_row_picks_the_longest_prompt_not_row_zero():
    """size_batch sorts ascending, so row 0 is typically the SHORTEST prompt
    -- the one with the MOST left-pad. A `_row = 0` implementation would
    pass a test built any other way; put the shortest prompt at index 0 on
    purpose so that bug fails loudly here."""
    reqs = [_req([1, 2]), _req([1, 2, 3, 4, 5, 6, 7, 8]), _req([1, 2, 3, 4])]
    assert prefix_store_row(reqs) == 1


def test_prefix_store_row_ties_pick_a_longest_row():
    reqs = [_req([1, 2, 3]), _req([9, 9, 9])]
    assert prefix_store_row(reqs) in (0, 1)
    assert len(reqs[prefix_store_row(reqs)].prompt_ids) == 3


def test_resolve_matched_len_reads_back_the_real_cache_length():
    """crop() is min(existing_len, requested) -- a silent no-op when the
    cache holds FEWER tokens than requested (e.g. a generate()-stored entry
    of cache_len + 1, per _persist_and_store). resolve_matched_len must
    return what the cache actually has, not the requested value, so a
    caller deriving tails from it never skips a token the model never saw."""
    c = _FakeCache(s=5)                 # cache only holds 5 tokens
    real_m = resolve_matched_len(c, 8)  # lookup() claimed 8
    assert real_m == 5
    assert c.get_seq_length() == 5

    reqs = [_req(list(range(10))), _req(list(range(12)))]
    tails = [r.prompt_ids[real_m:] for r in reqs]
    assert tails[0] == list(range(5, 10))
    assert tails[1] == list(range(5, 12))


def test_resolve_matched_len_is_a_true_no_op_when_cache_is_longer():
    c = _FakeCache(s=10)
    assert resolve_matched_len(c, 6) == 6
    assert c.get_seq_length() == 6
