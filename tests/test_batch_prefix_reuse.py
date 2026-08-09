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

    def to_device(self, device) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for name in self._BUFFER_LISTS for t in getattr(self, name))


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


# --- IMPORTANT 3 coverage: store-back on generator abandonment -------------
#
# batched_generate's own docstring treats early abandonment as real:
# .close() (or GC dropping the generator) raises GeneratorExit at the
# currently-suspended yield. lookup() already POPPED the pooled entry by
# that point; only the finally block's store-back puts one back. Reasoning
# about that in prose isn't verification -- this drives the real method
# end-to-end against the smallest fake VaayuBaseModel-shaped object that
# lets it reach its first yield, so abandonment is exercised for real.

from types import SimpleNamespace

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import VaayuBaseModel, _auto_session_id


class _FakeBatchedModel:
    """Just enough of VaayuBaseModel's surface for batched_generate to run:
    sdpa attn, a device string, a tokenizer/config/telemetry stub, and a
    forward() that manufactures a fresh _FakeCache sized off whatever
    attention_mask/input_ids it's handed -- generic enough to serve both
    the prefill call and (if ever reached) the decode loop."""
    _active_attn_impl = "sdpa"
    running_device = "cpu"
    prefill_chunk_size = 0

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    class _Cfg:
        eos_token_id = -999   # unreachable: the fake forward's logits are

    class _Telemetry:
        def reset_batch_overlap(self):
            pass

        def dump_json(self, path):
            pass

    def __init__(self, pc):
        self._prefix_cache = pc
        self.tokenizer = self._Tok()
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()

    def forward(self, input_ids, attention_mask, position_ids, use_cache=True,
                return_dict=True, past_key_values=None):
        cache = _FakeCache(s=attention_mask.shape[1], b=input_ids.shape[0])
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        return SimpleNamespace(past_key_values=cache, logits=logits)


def test_abandoning_the_generator_still_stores_the_prefix_back():
    common_ids = [1, 2, 3, 4, 5]
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    pc.store("preexisting", common_ids, _FakeCache(s=len(common_ids), b=1))
    assert len(pc._pool) == 1

    requests = [_req([1, 2, 3, 4, 5, 6, 7]), _req([1, 2, 3, 4, 5, 8, 9, 10])]
    model = _FakeBatchedModel(pc)

    gen = VaayuBaseModel.batched_generate(model, requests)
    next(gen)     # advance to the first yield -- lookup() has already popped
                  # the entry and `past` is now bound from the prefill call
    gen.close()   # GeneratorExit at the suspended yield

    assert model._batch_decode_active is False
    assert len(pc._pool) == 1              # entry restored, not lost
    restored = next(iter(pc._pool.values()))
    assert restored.token_ids == common_ids


def test_abandoning_before_the_first_yield_does_not_crash():
    """A generator-iterator runs none of its body until first resumed, so
    closing one before any next() is a true no-op in CPython -- nothing in
    batched_generate has executed yet, including lookup(). Pin that this
    stays a clean no-op (no crash, pool untouched) rather than a change
    here accidentally forcing eager work."""
    common_ids = [1, 2, 3]
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    pc.store("preexisting", common_ids, _FakeCache(s=len(common_ids), b=1))

    requests = [_req([1, 2, 3, 6]), _req([1, 2, 3, 7])]
    model = _FakeBatchedModel(pc)

    gen = VaayuBaseModel.batched_generate(model, requests)
    gen.close()   # closed before even calling next() once
    assert not hasattr(model, "_batch_decode_active")   # body never ran
    assert len(pc._pool) == 1                           # lookup() never ran


class _RaisingStorePC:
    """Stub prefix-cache store whose store() always fails, to exercise the
    store-back's own try/except without needing a real budget-eviction pool."""

    def lookup(self, session_id, input_ids, device=None):
        return _FakeCache(s=len(input_ids), b=1), len(input_ids)

    def store(self, session_id, token_ids, cache):
        raise RuntimeError("disk full (simulated)")


def test_store_back_failure_does_not_skip_cleanup():
    """A raise inside slice_cache_row/crop/pc.store must not skip clearing
    _batch_decode_active/_batch_n or dumping telemetry -- those three lines
    sit after the store-back in the same finally block, so if the store-back
    isn't guarded, a failure there propagates straight out and skips them."""
    common_ids = [1, 2, 3]
    requests = [_req([1, 2, 3, 6]), _req([1, 2, 3, 7])]
    model = _FakeBatchedModel(_RaisingStorePC())

    gen = VaayuBaseModel.batched_generate(model, requests)
    next(gen)
    gen.close()   # must not raise RuntimeError out of close()

    assert model._batch_decode_active is False
    assert model._batch_n == 1


# ── Prefix cache requires a Vaayu cache class ────────────────────────────────

def test_prefix_cache_support_detects_a_vaayu_cache():
    """FP8DynamicCache/TurboQuantKVCache carry the to_cpu/to_device/
    resident_bytes surface PrefixCacheStore drives -- and so does _FakeCache,
    which is modelled on them."""
    from vaayu.batch import cache_supports_prefix_store
    assert cache_supports_prefix_store(_FakeCache()) is True


def test_prefix_cache_support_needs_every_method():
    """Missing any one of the three is enough to break store()."""
    from vaayu.batch import cache_supports_prefix_store

    class _Partial:
        def to_cpu(self): pass
        def to_device(self, d): pass
        # no resident_bytes -- store() reads it immediately after to_cpu()

    assert cache_supports_prefix_store(_Partial()) is False
    assert cache_supports_prefix_store(object()) is False


def test_prefix_cache_support_rejects_plain_dynamic_cache():
    """The real reason this exists: `_create_kv_cache` returns a bare
    transformers DynamicCache for kv_cache_mode "fp16" (what "auto" resolves
    to for every non-EXL3 model), and it has no to_cpu/resident_bytes at all.
    PrefixCacheStore.store() would AttributeError on it."""
    import torch as _t
    from transformers.cache_utils import DynamicCache
    from vaayu.batch import cache_supports_prefix_store

    c = DynamicCache()
    c.update(_t.randn(1, 2, 3, 4), _t.randn(1, 2, 3, 4), 0)
    # Pin the premise rather than assuming it -- if a future transformers
    # grows these methods, this test should tell us, not silently pass.
    assert not hasattr(c, "to_cpu")
    assert not hasattr(c, "resident_bytes")
    assert cache_supports_prefix_store(c) is False


def test_slice_row_lands_on_the_cpu_not_the_source_device():
    """The store-back runs when VRAM is at its peak -- right after a long
    prefill. Cloning the row on the source device needs a second full copy of
    the KV cache in VRAM exactly then. Measured on MiniMax at 7 700 tokens,
    that allocation failed with 19.56 MiB free, the pool was never populated,
    and round 2 re-prefilled from scratch. Slicing straight to host keeps the
    device allocation at zero.
    """
    c = _FakeCache()
    broadcast_cache_rows(c, 3)
    one = slice_cache_row(c, 1)
    for name in one._BUFFER_LISTS:
        for t in getattr(one, name):
            assert t.device.type == "cpu", f"{name} landed on {t.device}"
