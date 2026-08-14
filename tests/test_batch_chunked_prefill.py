"""Batched prefill must chunk, or long contexts are unreachable at any N.

`batched_generate` disabled chunked prefill outright ("it assumes batch=1,
indexes [0]"), so a whole batch prompt ran as ONE forward. Measured on
MiniMax, same box same day: N=1 fits at 4 000 tokens and OOMs at 7 768
(tried 5.79 GiB with 5.13 free); N=3 fits at 2 000 and OOMs at 4 000. The
constraint is prefill ACTIVATIONS and it scales with N x context, which is
why Plan K's admission question could not even be asked at the length it was
posed -- no admission policy helps if N=1 does not run.

forward()'s own chunked path cannot be reused: its MoE fast path turns chunks
INTO batch elements, so it needs the batch dimension this caller is already
using. The chunking therefore lives in batched_generate, where the padding
mask and per-stream positions are built.

Each chunk must satisfy mask_width == past + q_len; check_batch_kv_invariant
enforces that inside the real forward, and the stub here mirrors it.
"""
import torch

from vaayu.batch import BatchRequest
from vaayu.vaayu_base import (FP8DynamicCache, VaayuBaseModel,
                              check_batch_kv_invariant)

N_LAYERS, N_HEADS, HEAD_DIM, VOCAB = 2, 2, 4, 16


class _ChunkRecorder:
    _active_attn_impl = "sdpa"
    running_device = "cpu"
    running_dtype = torch.float32
    _batch_decode_active = False
    _batch_n = 1
    _prefix_cache = None
    _expert_vram_cache = None

    def __init__(self, chunk_size):
        self.prefill_chunk_size = chunk_size
        self.tokenizer = type("T", (), {"pad_token_id": 0})()
        self.config = type("C", (), {"eos_token_id": -1})()
        self.telemetry = type("Tel", (), {
            "reset_batch_overlap": lambda s: None,
            "dump_json": lambda s, *a, **k: None})()
        self.calls = []          # (past, q, mask_width) per forward
        self.chunk_sizes_seen = []

    def _assert_context_fits(self, n_tokens, N=1):
        # Duck-typed stub: no _vram_plan on this fake, real method no-ops too.
        pass

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, return_dict=True):
        n, q = input_ids.shape
        check_batch_kv_invariant(attention_mask, q, past_key_values)
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.calls.append((past, q, int(attention_mask.shape[1])))
        assert position_ids.shape[1] == past + q, (
            f"positions must cover the whole sequence so get_pos_emb_args can "
            f"slice [len_p:len_p+len_s]; got {position_ids.shape[1]} for "
            f"past={past} q={q}")
        cache = past_key_values if past_key_values is not None else FP8DynamicCache(quantize=False)
        for layer in range(N_LAYERS):
            kv = torch.randn(n, N_HEADS, q, HEAD_DIM)
            cache.update(kv, kv.clone(), layer)
        out = type("O", (), {})()
        out.past_key_values = cache
        out.logits = torch.randn(n, q, VOCAB)
        return out


def _drive(engine, reqs, max_new=2):
    list(VaayuBaseModel.batched_generate(engine, reqs))


def _reqs(n, prompt_len, max_new=2):
    """Exactly `prompt_len` tokens per stream, last one distinct per stream."""
    return [BatchRequest([1] * (prompt_len - 1) + [i], max_new) for i in range(n)]


def _split(calls, max_R):
    """Prefill calls are those up to the one whose past+q covers the prompt.
    A ragged LAST chunk can have q == 1, which is indistinguishable from a
    decode step by shape alone -- so split by coverage, not by width."""
    for i, (past, q, _w) in enumerate(calls):
        if past + q >= max_R:
            return calls[:i + 1], calls[i + 1:]
    return calls, []


def test_long_prompt_is_split_into_chunks():
    eng = _ChunkRecorder(chunk_size=8)
    _drive(eng, _reqs(3, 40))                       # 40 tokens -> 5 chunks of 8
    prefill, _ = _split(eng.calls, 40)
    assert len(prefill) == 5, f"expected 5 chunks, got {len(prefill)}: {prefill}"
    assert all(q <= 8 for _p, q, _w in prefill), prefill


def test_a_ragged_final_chunk_is_still_prefill():
    """41 tokens at chunk 8 is 5 full chunks plus a 1-token remainder. That
    remainder looks exactly like a decode step from its shape, so nothing may
    depend on width to tell them apart."""
    eng = _ChunkRecorder(chunk_size=8)
    _drive(eng, _reqs(2, 41))
    prefill, decode = _split(eng.calls, 41)
    assert len(prefill) == 6, prefill
    assert prefill[-1][1] == 1, prefill[-1]
    assert sum(q for _p, q, _w in prefill) == 41


def test_each_chunk_sees_the_right_past_and_mask_width():
    eng = _ChunkRecorder(chunk_size=8)
    _drive(eng, _reqs(2, 20))
    prefill, _ = _split(eng.calls, 20)
    expected_past = 0
    for past, q, width in prefill:
        assert past == expected_past, f"chunk past {past} != {expected_past}"
        assert width == past + q
        expected_past += q
    assert expected_past == 20, expected_past


def test_chunk_size_zero_is_one_forward():
    """The pre-existing behaviour, unchanged for callers that disable chunking."""
    eng = _ChunkRecorder(chunk_size=0)
    _drive(eng, _reqs(3, 40))
    prefill, _ = _split(eng.calls, 40)
    assert len(prefill) == 1 and prefill[0] == (0, 40, 40)


def test_prompt_shorter_than_a_chunk_is_one_forward():
    eng = _ChunkRecorder(chunk_size=64)
    _drive(eng, _reqs(3, 10))
    prefill, _ = _split(eng.calls, 10)
    assert len(prefill) == 1


def test_final_cache_length_is_the_whole_prompt():
    """Chunking must not lose or double-count positions."""
    eng = _ChunkRecorder(chunk_size=8)
    _drive(eng, _reqs(3, 40))
    prefill, _ = _split(eng.calls, 40)
    assert sum(q for _p, q, _w in prefill) == 40


def test_decode_still_runs_after_a_chunked_prefill():
    eng = _ChunkRecorder(chunk_size=8)
    _drive(eng, _reqs(2, 30, max_new=3))
    _pre, decode = _split(eng.calls, 30)
    assert len(decode) >= 2, eng.calls
    for past, q, width in decode:
        assert width == past + q
