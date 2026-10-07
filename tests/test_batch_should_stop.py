"""A retired row must stop yielding AND stop holding the batch open.

`while any(active)` is the only thing that ends batched_generate's decode
loop, and nothing outside can reach `active`. Without this hook a client that
disconnects at cap=8000 holds the whole batch for 8000 steps -- the other
streams pay for a connection that is gone. Cooperative and once-per-step,
mirroring the server's CancelToken rather than inventing a second mechanism.

The recorder below is the same CPU stub tests/test_batch_chunked_prefill.py
uses: no checkpoint, no GPU.
"""
import torch

from vaayu.batch import BatchRequest
from vaayu.vaayu_base import (FP8DynamicCache, VaayuBaseModel,
                              check_batch_kv_invariant)

N_LAYERS, N_HEADS, HEAD_DIM, VOCAB = 2, 2, 4, 16


class _Recorder:
    _active_attn_impl = "sdpa"
    running_device = "cpu"
    running_dtype = torch.float32
    _batch_decode_active = False
    _batch_n = 1
    _prefix_cache = None
    _expert_vram_cache = None

    def __init__(self):
        self.prefill_chunk_size = 0
        self.tokenizer = type("T", (), {"pad_token_id": 0})()
        self.config = type("C", (), {"eos_token_id": -1})()
        self.telemetry = type("Tel", (), {
            "reset_batch_overlap": lambda s: None})()
        self.forwards = 0

    def _assert_context_fits(self, n_tokens, N=1, expert_giveback_bytes=0,
                             batched_prefill=False):
        return None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, return_dict=True):
        self.forwards += 1
        n, q = input_ids.shape
        check_batch_kv_invariant(attention_mask, q, past_key_values)
        cache = (past_key_values if past_key_values is not None
                 else FP8DynamicCache(quantize=False))
        for layer in range(N_LAYERS):
            kv = torch.randn(n, N_HEADS, q, HEAD_DIM)
            cache.update(kv, kv.clone(), layer)
        out = type("O", (), {})()
        out.past_key_values = cache
        out.logits = torch.randn(n, q, VOCAB)
        return out


def _reqs(n, prompt_len=4, max_new=20):
    return [BatchRequest([1] * (prompt_len - 1) + [i], max_new) for i in range(n)]


def test_no_hook_is_todays_behaviour():
    eng = _Recorder()
    yielded = list(VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=5)))
    assert sorted(i for i, _tok in yielded) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_a_retired_row_stops_yielding():
    eng = _Recorder()
    gen = VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=5),
                                          should_stop=lambda i: i == 0)
    got = list(gen)
    assert {i for i, _tok in got} == {1}, "row 0 was retired before its first token"
    assert len(got) == 5


def test_retiring_every_row_ends_the_generator_promptly():
    """The whole point: a batch whose clients have all gone must not run to
    cap. Both rows retire before their first token, so the loop never runs."""
    eng = _Recorder()
    got = list(VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=5000),
                                               should_stop=lambda i: True))
    assert got == []
    assert eng.forwards == 1, "one prefill forward, then nothing to decode for"


def test_a_row_retired_mid_decode_stops_and_the_others_finish():
    seen = {"steps": 0}

    def _stop(i):
        # Retire row 0 only after it has produced two tokens.
        return i == 0 and seen["steps"] >= 2

    eng = _Recorder()
    out = []
    for i, tok in VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=6),
                                                  should_stop=_stop):
        out.append(i)
        if i == 0:
            seen["steps"] += 1
    assert out.count(0) == 2
    assert out.count(1) == 6


def test_flags_are_cleared_after_a_hooked_run():
    eng = _Recorder()
    list(VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=3),
                                         should_stop=lambda i: i == 1))
    assert eng._batch_decode_active is False
    assert eng._batch_n == 1


def test_flags_are_cleared_when_the_generator_is_abandoned():
    """The contract in batched_generate's own docstring: an abandoned
    generator must not leave the model in batched-decode mode. A client
    disconnect is exactly this abandonment."""
    eng = _Recorder()
    gen = VaayuBaseModel.batched_generate(eng, _reqs(2, max_new=50))
    next(gen)
    assert eng._batch_decode_active is True
    gen.close()
    assert eng._batch_decode_active is False
    assert eng._batch_n == 1
