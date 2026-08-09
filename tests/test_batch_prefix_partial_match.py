"""batched_generate's prefix reuse under PARTIAL matches.

todo.md's suspect for the unexplained 2026-08-09 mask/KV failure was "round 2
with a partial match" -- a batch whose common prefix is a STRICT prefix of a
longer stored entry, the one shape every successful measurement run avoided.
These tests build that shape (and its inverse, and a ragged-length version)
and drive batched_generate through it, asserting at the forward boundary that
the padding mask always agrees with the cache it will be used against.

Driven through a duck-typed `self` rather than a model load: the contract
under test is between batched_generate and the mask, and both are pure tensor
bookkeeping. A real checkpoint would cost minutes per case and prove the same
thing (tests/test_batch_prefix_equivalence.py already carries the on-device
gate).
"""
import torch

from vaayu.batch import BatchRequest
from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import (FP8DynamicCache, VaayuBaseModel,
                              check_batch_kv_invariant)

N_LAYERS, N_HEADS, HEAD_DIM, VOCAB = 2, 2, 4, 16


class _Recorder:
    """Minimal `self` for VaayuBaseModel.batched_generate, whose forward
    enforces the same mask/KV invariant the real forward does."""

    _active_attn_impl = "sdpa"
    running_device = "cpu"
    running_dtype = torch.float32
    prefill_chunk_size = 0
    _batch_decode_active = False
    _batch_n = 1

    def __init__(self, pc):
        self.tokenizer = type("T", (), {"pad_token_id": 0})()
        self.config = type("C", (), {"eos_token_id": -1})()
        self.telemetry = type("Tel", (), {
            "reset_batch_overlap": lambda s: None,
            "dump_json": lambda s, *a, **k: None})()
        self._prefix_cache = pc
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, return_dict=True):
        n, q = input_ids.shape
        check_batch_kv_invariant(attention_mask, q, past_key_values)
        # `is not None`, never a truth test: FP8DynamicCache manages its own
        # buffer lists and leaves the parent DynamicCache's `layers` empty, so
        # a populated cache is FALSY and `past_key_values or ...` silently
        # discards it and prefills from scratch every step.
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.calls.append((int(attention_mask.shape[1]), past, q))
        cache = past_key_values if past_key_values is not None else FP8DynamicCache(quantize=False)
        for layer in range(N_LAYERS):
            kv = torch.randn(n, N_HEADS, q, HEAD_DIM)
            cache.update(kv, kv.clone(), layer)
        out = type("O", (), {})()
        out.past_key_values = cache
        out.logits = torch.randn(n, q, VOCAB)
        return out


def _drive(pc, prompts, max_new_tokens=2):
    rec = _Recorder(pc)
    reqs = [BatchRequest(list(p), max_new_tokens) for p in prompts]
    list(VaayuBaseModel.batched_generate(rec, reqs))
    return rec


def _rounds(round1, round2):
    pc = PrefixCacheStore(budget_bytes=1 << 30, auto_prefix_match=True)
    return _drive(pc, round1), _drive(pc, round2)


HEAD = list(range(100, 160))          # 60-token shared head


def test_round2_common_prefix_is_a_strict_prefix_of_the_stored_entry():
    """The suspect shape. Stored entry covers 60 tokens; the next batch shares
    only its first 40, so lookup() matches less than the cache holds."""
    r1, r2 = _rounds([HEAD + [1, 2], HEAD + [3, 4], HEAD + [5, 6]],
                     [HEAD[:40] + [7, 7], HEAD[:40] + [8, 8], HEAD[:40] + [9, 9]])
    assert r1.calls[0][1] == 0, "round 1 must be cold"
    assert r2.calls[0][1] == 40, f"round 2 must reuse exactly 40: {r2.calls[0]}"


def test_round2_common_prefix_extends_beyond_the_stored_entry():
    """The inverse: the cache is shorter than what the batch now shares, so
    the reused length is capped by the cache, not by the token match."""
    r1, r2 = _rounds([HEAD[:40] + [1, 2], HEAD[:40] + [3, 4], HEAD[:40] + [5, 6]],
                     [HEAD + [7, 7], HEAD + [8, 8], HEAD + [9, 9]])
    assert r1.calls[0][1] == 0
    assert r2.calls[0][1] == 40, f"cache holds 40; cannot reuse more: {r2.calls[0]}"


def test_partial_match_with_ragged_prompt_lengths():
    """Partial match plus unequal prompts, so the left-pad block and the
    reused prefix are both non-trivial."""
    _, r2 = _rounds([HEAD + [1, 2], HEAD + [3, 4, 4, 4], HEAD + [5]],
                    [HEAD[:31] + [7], HEAD[:31] + [8, 8, 8, 8, 8], HEAD[:31] + [9, 9]])
    width, past, q = r2.calls[0]
    assert past == 31 and width == past + q


def test_match_reaches_the_shortest_prompt_minus_one():
    """The clamp case: every stream must keep at least one token to run."""
    _, r2 = _rounds([HEAD + [1, 2], HEAD + [3, 4], HEAD + [5, 6]],
                    [HEAD[:50] + [7], HEAD[:50] + [8], HEAD[:50] + [9]])
    width, past, q = r2.calls[0]
    assert q >= 1 and width == past + q


def test_decode_steps_keep_the_mask_and_cache_in_step():
    """Every decode step grows the mask by one and the cache by one; the
    invariant is checked on each, so a drift of even one token fails here."""
    _, r2 = _rounds([HEAD + [1, 2], HEAD + [3, 4], HEAD + [5, 6]],
                    [HEAD + [1, 2], HEAD + [3, 4], HEAD + [5, 6]])
    assert len(r2.calls) > 1, "decode loop must have run"
    for width, past, q in r2.calls:
        assert width == past + q
