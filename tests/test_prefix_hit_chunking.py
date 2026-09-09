"""A prefix-cache HIT must not turn the rest of the prompt into one forward.

Found 2026-09-09 by P10 T10's first on-device arm. MiniMax-M2.7, ~4.17k-token
agentic prompts, `ENABLE_PREFIX_CACHE = True`, `sdpa`, `MAX_BATCH_SIZE = 1` --
so not a batching path at all. The FIRST request (cold, no hit) chunked and
succeeded at a 12.84 GB peak; every request after it hit the pool and died:

    CUDA out of memory. Tried to allocate 3.07 GiB.
    GPU 0 has a total capacity of 15.47 GiB of which 2.08 GiB is free.

3 of 4 requests in that arm, one after another, each allocation scaling with
its own prompt (4171 -> 3.07, 4191 -> 3.09, 4223 -> 3.11 GiB).

`forward()` gated chunked prefill on `_past_kv_len == 0`. That guard was added
2026-05-31 (`6cfd252`) to keep DECODE out of the chunked path -- its own test
was called "Decode step (past_kv_len > 0) -> no chunking" -- but decode is
q_len == 1 and the outer `input_ids.shape[1] > prefill_chunk_size` test already
excludes it. Cross-turn prefix caching landed later (2026-07-19) and created the
case the guard actually caught: a SHORT hit leaving a LONG remainder, which is
the ordinary shape when two agent sessions share a system prompt and nothing
else. That remainder then ran as one q_len ~= 4000 forward, and sdpa with an
explicit mask falls off its fused backends onto the math one, which
materializes num_heads x q x kv scores -- 48 x 4171 x 4171 x 4 B = 3.1 GiB,
matching the OOM.

`_chunked_prefill` never needed the guard: it takes `past_key_values`, threads
it through both paths, and `compute_chunk_past_lens(chunk_lens, past0)` exists
precisely to offset each chunk element by a non-zero starting past.
"""
from unittest.mock import MagicMock, patch

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from vaayu.vaayu_base import VaayuBaseModel


def _bare_model(chunk_size):
    """A VaayuBaseModel with only what forward()'s routing block reads."""
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.prefill_chunk_size = chunk_size
    m._active_attn_impl = "sdpa"
    m.profiling_mode = False
    m._token_count = 0
    m.telemetry = type("Tel", (), {"begin_token": lambda s, n: None})()
    return m


def _past(length):
    kv = MagicMock()
    kv.get_seq_length.return_value = length
    return kv


def test_a_long_remainder_after_a_prefix_hit_still_chunks():
    """The bug. past > 0 must not disable chunking -- only q_len does."""
    m = _bare_model(chunk_size=512)
    sentinel = CausalLMOutputWithPast(logits=torch.zeros(1, 1, 8),
                                      past_key_values=DynamicCache())

    with patch.object(m, "_chunked_prefill", return_value=sentinel) as chunked:
        out = m.forward(input_ids=torch.zeros(1, 4000, dtype=torch.long),
                        past_key_values=_past(150), use_cache=True,
                        return_dict=True)

    chunked.assert_called_once()
    # The past must reach it, or every chunk would be positioned from 0 and
    # `compute_chunk_past_lens`'s `past0` would be a lie.
    assert chunked.call_args.kwargs["past_key_values"] is not None
    assert out is sentinel


def test_a_cold_long_prompt_still_chunks():
    """The case that always worked -- pinned so the fix does not trade one for
    the other."""
    m = _bare_model(chunk_size=512)
    sentinel = CausalLMOutputWithPast(logits=torch.zeros(1, 1, 8),
                                      past_key_values=DynamicCache())

    with patch.object(m, "_chunked_prefill", return_value=sentinel) as chunked:
        m.forward(input_ids=torch.zeros(1, 4000, dtype=torch.long),
                  past_key_values=None, use_cache=True, return_dict=True)

    chunked.assert_called_once()


def test_a_decode_step_never_chunks():
    """What the 2026-05-31 guard was actually written for, and it holds without
    that guard: a decode step is q_len == 1, which the length test excludes on
    its own however long the past is."""
    m = _bare_model(chunk_size=512)

    with patch.object(m, "_chunked_prefill") as chunked:
        try:
            m.forward(input_ids=torch.zeros(1, 1, dtype=torch.long),
                      past_key_values=_past(4000), use_cache=True,
                      return_dict=True)
        except Exception:
            pass          # no weights on this fake; only the routing matters

    chunked.assert_not_called()


def test_a_speculative_verify_pass_never_chunks():
    """A verify pass is `draft_tokens + 1` wide (5-16 here) against a long past.
    It must keep taking the single forward it was measured on -- chunking a
    16-token window would be pure overhead."""
    m = _bare_model(chunk_size=512)

    with patch.object(m, "_chunked_prefill") as chunked:
        try:
            m.forward(input_ids=torch.zeros(1, 4, dtype=torch.long),
                      past_key_values=_past(4000), use_cache=True,
                      return_dict=True)
        except Exception:
            pass

    chunked.assert_not_called()


def test_chunking_stays_off_when_it_is_disabled():
    """prefill_chunk_size = 0 is the documented off switch, and
    `batched_generate` sets it to 0 around its own prefill (vaayu_base.py's
    `_saved_chunk`) -- so this is what keeps the batch path out of here."""
    m = _bare_model(chunk_size=0)

    with patch.object(m, "_chunked_prefill") as chunked:
        try:
            m.forward(input_ids=torch.zeros(1, 4000, dtype=torch.long),
                      past_key_values=_past(150), use_cache=True,
                      return_dict=True)
        except Exception:
            pass

    chunked.assert_not_called()
