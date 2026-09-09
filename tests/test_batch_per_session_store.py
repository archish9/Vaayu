"""A batched turn must leave the prefix pool as a serial turn would.

Found 2026-09-09 by P10 T10's second on-device arm. MiniMax-M2.7, 2 sessions x
2 turns at ~4.17k tokens, `ENABLE_PREFIX_CACHE=True` (the server's default):

    MAX_BATCH_SIZE=1   aggregate  99.52 s   turn-2 hits 4164/4223, 4167/4226
    MAX_BATCH_SIZE=2   aggregate 443.78 s   turn-2 hits   51/4223,   51/4226

4.46x SLOWER end to end -- while batching itself WON its turn: the cold turn
ran 288.8 s batched against 461.2 s serial, 1.60x, reproducing the banked
1.606x. The whole loss is on turn 2, because `batched_generate` stored exactly
ONE pool entry per batch, keyed by a hash of the batch's COMMON prefix (the
51-token shared system prompt), and nothing per session. Every session
therefore re-prefilled its entire context on its next turn.

The old comment justified one-entry-per-batch with a K=7-sessions-against-a-
5-slot-pool measurement where writing N gave ZERO hits. That is a pool SIZING
cliff (PREFIX_SESSIONS), not a property of who writes the entries -- seven
serial sessions hit the same cliff. Writing one entry per row is what makes a
batched turn and a serial turn leave the same pool state, which is the only
behaviour a user can reason about when MAX_BATCH_SIZE is a ceiling that a batch
may or may not reach.

The row slice is not a plain crop: tails are LEFT-padded, so row i's cache is
[0:m] shared prefix, then (max_R - len(tail_i)) pads, then its real tokens. The
pad block sits in the MIDDLE and has to be spliced out.
"""
from types import SimpleNamespace

import torch

from vaayu.batch import BatchRequest, slice_cache_row_unpadded
from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import VaayuBaseModel, _auto_session_id

from tests.test_batch_prefix_reuse import _FakeBatchedModel, _FakeCache


# ── the splice ──────────────────────────────────────────────────────────────

def _numbered(b, s, layers=2):
    """Cache whose every element equals its sequence index, so a wrong splice
    is visible as a wrong number rather than as a wrong shape."""
    c = _FakeCache(b=b, s=s, n_layers=layers)
    for name in c._BUFFER_LISTS:
        for t in getattr(c, name):
            t[:] = torch.arange(s, dtype=t.dtype).reshape(1, 1, s, 1)
    return c


def test_the_pad_block_is_spliced_out_not_cropped():
    # m=3 prefix, 4 pads, 5 real tail tokens, 2 decode steps.
    cache = _numbered(b=2, s=3 + 4 + 5 + 2)
    out = slice_cache_row_unpadded(cache, row=0, prefix_len=3, pad_len=4,
                                   valid_len=3 + 5 + 2)

    kept = out.key_cache[0][0, 0, :, 0].tolist()
    assert kept == [0, 1, 2, 7, 8, 9, 10, 11, 12, 13], (
        "the shared prefix and the row's real tokens, with the interior pad "
        f"block [3:7) dropped; got {kept}")
    assert out.get_seq_length() == 10


def test_the_splice_is_batch_one_and_shares_no_storage():
    cache = _numbered(b=3, s=12)
    out = slice_cache_row_unpadded(cache, row=1, prefix_len=2, pad_len=3,
                                   valid_len=9)
    assert out.key_cache[0].shape[0] == 1, "the batch dimension must shrink"
    out.key_cache[0][:] = -1
    assert cache.key_cache[0][1, 0, 0, 0] != -1, "must not alias the source"


def test_a_row_with_no_padding_keeps_everything_after_the_prefix():
    """The longest row -- the only one the old code could store."""
    cache = _numbered(b=2, s=10)
    out = slice_cache_row_unpadded(cache, row=0, prefix_len=4, pad_len=0,
                                   valid_len=10)
    assert out.key_cache[0][0, 0, :, 0].tolist() == list(range(10))


# ── the store-back ──────────────────────────────────────────────────────────

def _run(model, requests, steps=1):
    gen = VaayuBaseModel.batched_generate(model, requests)
    for _ in range(steps * len(requests)):
        next(gen)
    gen.close()


def test_every_session_in_the_batch_gets_its_own_entry():
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    reqs = [BatchRequest(prompt_ids=[1, 2, 3, 4, 5, 6, 7], max_new_tokens=4,
                         session_id="agent-a"),
            BatchRequest(prompt_ids=[1, 2, 3, 4, 5, 8, 9, 10, 11],
                         max_new_tokens=4, session_id="agent-b")]
    _run(_FakeBatchedModel(pc), reqs)

    assert set(pc._pool) == {"agent-a", "agent-b"}, (
        "one entry per session, not one per batch keyed by the shared prefix")


def test_each_entry_holds_that_session_s_own_prompt():
    """The point of the entry: its next turn must match its WHOLE prompt, not
    just the prefix it happened to share with a sibling."""
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [1, 2, 3, 4, 5, 8, 9, 10, 11]
    reqs = [BatchRequest(prompt_ids=a, max_new_tokens=4, session_id="agent-a"),
            BatchRequest(prompt_ids=b, max_new_tokens=4, session_id="agent-b")]
    _run(_FakeBatchedModel(pc), reqs)

    for sid, prompt in (("agent-a", a), ("agent-b", b)):
        stored = pc._pool[sid].token_ids
        assert stored[:len(prompt)] == prompt, (
            f"{sid} stored {stored[:len(prompt)]}, wanted its own prompt {prompt}")
        # Its next turn resends that prompt plus what came back.
        match = pc.peek_match(sid, prompt + [77])
        assert match >= len(prompt), (
            f"{sid} would re-prefill {len(prompt) - match} of its own tokens")


def test_the_generated_tokens_are_stored_too_except_the_last():
    """A next turn resends the answer, so an entry that stopped at the prompt
    would make the model re-read its own output.

    All but the LAST token: a yielded token's KV is written by the NEXT forward,
    which a finished or abandoned generator never makes. The entry is clamped to
    what the cache actually backs rather than claiming a position it cannot
    serve -- one token, and the next turn matches up to it and prefills the
    rest."""
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    reqs = [BatchRequest(prompt_ids=[1, 2, 3, 4], max_new_tokens=4,
                         session_id="a"),
            BatchRequest(prompt_ids=[1, 2, 5, 6], max_new_tokens=4,
                         session_id="b")]
    _run(_FakeBatchedModel(pc), reqs, steps=2)

    entry = pc._pool["a"]
    assert len(entry.token_ids) == 5, (
        f"4 prompt + 2 generated - 1 unbacked, got {len(entry.token_ids)}")
    assert entry.token_ids[:4] == [1, 2, 3, 4]
    assert entry.cache.get_seq_length() == 5, "ids and KV must agree in length"


def test_a_row_without_a_session_id_falls_back_to_the_hashed_id():
    """Library callers do not name sessions; they must still get per-row
    entries, keyed exactly as the serial path keys an unnamed turn."""
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    a, b = [1, 2, 3, 7], [1, 2, 3, 9]
    _run(_FakeBatchedModel(pc), [BatchRequest(prompt_ids=a, max_new_tokens=2),
                                 BatchRequest(prompt_ids=b, max_new_tokens=2)])

    assert len(pc._pool) == 2
    for key, entry in pc._pool.items():
        assert key == _auto_session_id(entry.token_ids)
