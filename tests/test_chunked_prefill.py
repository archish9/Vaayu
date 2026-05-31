import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import patch, MagicMock

from tests.test_vaayu_base_kwargs import _make_base


def test_prefill_chunk_size_default_is_512():
    m = _make_base()
    assert m.prefill_chunk_size == 512


def test_prefill_chunk_size_custom_stored():
    m = _make_base({"prefill_chunk_size": 256})
    assert m.prefill_chunk_size == 256


def test_prefill_chunk_size_zero_stored():
    m = _make_base({"prefill_chunk_size": 0})
    assert m.prefill_chunk_size == 0


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


def _fake_kv(seq_len):
    """Return a MagicMock KV cache with get_seq_length() = seq_len."""
    kv = MagicMock()
    kv.get_seq_length.return_value = seq_len
    return kv


def _fake_out(seq_len, vocab=100, kv_seq_len=None):
    """Return a CausalLMOutputWithPast with the given logits shape."""
    return CausalLMOutputWithPast(
        logits=torch.zeros(1, seq_len, vocab),
        past_key_values=_fake_kv(kv_seq_len or seq_len),
    )


def test_chunked_prefill_calls_forward_twice_for_double_chunk():
    """1024-token prompt, chunk_size=512 → exactly 2 forward() calls."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512, kv_seq_len=512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 2


def test_chunked_prefill_calls_forward_three_times_for_triple_chunk():
    """1536-token prompt, chunk_size=512 → exactly 3 forward() calls."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1536))

    with patch.object(m, "forward", return_value=_fake_out(512, kv_seq_len=512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 3


def test_chunked_prefill_handles_remainder():
    """600 tokens, chunk_size=512 → 2 calls: chunk sizes [512, 88]."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 600))

    kv1 = _fake_kv(512)
    kv2 = _fake_kv(600)
    out1 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)
    out2 = CausalLMOutputWithPast(logits=torch.zeros(1, 88, 100), past_key_values=kv2)

    with patch.object(m, "forward", side_effect=[out1, out2]) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    assert mock_fwd.call_count == 2
    first_ids  = mock_fwd.call_args_list[0][1]["input_ids"]
    second_ids = mock_fwd.call_args_list[1][1]["input_ids"]
    assert first_ids.shape[1] == 512
    assert second_ids.shape[1] == 88


def test_chunked_prefill_threads_kv_cache_between_chunks():
    """KV cache from chunk N is passed as past_key_values to chunk N+1."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    kv1 = _fake_kv(512)
    kv2 = _fake_kv(1024)
    out1 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)
    out2 = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv2)

    with patch.object(m, "forward", side_effect=[out1, out2]) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    # Second call must receive the KV cache returned by the first call
    second_kwargs = mock_fwd.call_args_list[1][1]
    assert second_kwargs["past_key_values"] is kv1


def test_chunked_prefill_first_call_has_no_past_kv():
    """First chunk is always prefilled from scratch (past_key_values=None)."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    first_kwargs = mock_fwd.call_args_list[0][1]
    assert first_kwargs["past_key_values"] is None


def test_chunked_prefill_returns_last_chunk_output():
    """Return value is the output of the final forward() call."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    out1 = _fake_out(512, kv_seq_len=512)
    out2 = _fake_out(512, kv_seq_len=1024)

    with patch.object(m, "forward", side_effect=[out1, out2]):
        result = m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                                    output_attentions=False, output_hidden_states=False, return_dict=True)

    assert result is out2


def test_chunked_prefill_slices_attention_mask_correctly():
    """Attention mask for chunk [0:512] should be mask[:, :512]; for [512:1024] mask[:, :1024]."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))
    attention_mask = torch.ones(1, 1024)

    kv1 = _fake_kv(512)
    out = CausalLMOutputWithPast(logits=torch.zeros(1, 512, 100), past_key_values=kv1)

    with patch.object(m, "forward", return_value=out) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=attention_mask, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    first_mask  = mock_fwd.call_args_list[0][1]["attention_mask"]
    second_mask = mock_fwd.call_args_list[1][1]["attention_mask"]
    assert first_mask.shape[1] == 512
    assert second_mask.shape[1] == 1024


def test_chunked_prefill_none_attention_mask_stays_none():
    """When attention_mask=None (SDPA/flash path), forward() receives None each call."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)) as mock_fwd:
        m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                           output_attentions=False, output_hidden_states=False, return_dict=True)

    for call in mock_fwd.call_args_list:
        assert call[1]["attention_mask"] is None


def test_chunked_prefill_non_dict_return():
    """When return_dict=False, result is a tuple with logits first."""
    m = _make_base({"prefill_chunk_size": 512})
    input_ids = torch.randint(0, 100, (1, 1024))

    with patch.object(m, "forward", return_value=_fake_out(512)):
        result = m._chunked_prefill(input_ids, attention_mask=None, use_cache=True,
                                    output_attentions=False, output_hidden_states=False, return_dict=False)

    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)  # logits first


def test_forward_routes_to_chunked_prefill_for_long_prompt():
    """Prompt longer than chunk_size, no prior KV → _chunked_prefill called."""
    m = _make_base({"prefill_chunk_size": 4})
    input_ids = torch.randint(0, 100, (1, 8))  # 8 > 4

    sentinel = CausalLMOutputWithPast(
        logits=torch.zeros(1, 4, 100),
        past_key_values=DynamicCache(),
    )
    with patch.object(m, "_chunked_prefill", return_value=sentinel) as mock_cp:
        result = m.forward(input_ids=input_ids, past_key_values=None,
                           use_cache=True, return_dict=True)

    mock_cp.assert_called_once()
    assert result is sentinel


def test_forward_skips_chunked_prefill_when_disabled():
    """chunk_size=0 → _chunked_prefill never called, even for long prompts."""
    m = _make_base({"prefill_chunk_size": 0})
    m._active_attn_impl = "eager"

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      use_cache=False, return_dict=True)
        except Exception:
            pass  # layer loop will fail without real weights; we only need the routing check

    mock_cp.assert_not_called()


def test_forward_skips_chunked_prefill_when_seq_not_exceeds_chunk():
    """Prompt exactly equal to chunk_size → no chunking."""
    m = _make_base({"prefill_chunk_size": 8})
    m._active_attn_impl = "eager"

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      use_cache=False, return_dict=True)
        except Exception:
            pass

    mock_cp.assert_not_called()


def test_forward_skips_chunked_prefill_when_past_kv_exists():
    """Decode step (past_kv_len > 0) → no chunking even for 'long' seq."""
    m = _make_base({"prefill_chunk_size": 4})
    m._active_attn_impl = "eager"

    past_kv = MagicMock()
    past_kv.get_seq_length.return_value = 4  # non-zero past

    with patch.object(m, "_chunked_prefill") as mock_cp:
        try:
            m.forward(input_ids=torch.randint(0, 100, (1, 8)),
                      past_key_values=past_kv, use_cache=True, return_dict=True)
        except Exception:
            pass

    mock_cp.assert_not_called()


def test_position_id_slice_is_empty_when_past_exceeds_max_seq_len():
    """Demonstrate the OLD bug: position_ids of max_seq_len size returns empty slice for past+seq."""
    m = _make_base()
    m.max_seq_len = 4
    # Old behavior: position_ids = arange(4) → only 4 elements
    old_pos_ids = torch.arange(4)[None, :]
    result = m.get_position_ids_args(old_pos_ids, len_p=4, len_s=4)
    # PyTorch slice [4:8] on a 4-element dim returns empty tensor — silent corruption
    assert result["position_ids"].shape[1] == 0  # BUG: should be 4


def test_position_id_slice_correct_with_dynamic_range():
    """NEW behavior: pos_range = max(max_seq_len, past+seq) → slice is [past : past+seq]."""
    m = _make_base()
    m.max_seq_len = 4
    past_len, seq_len = 4, 4
    pos_range = max(m.max_seq_len, past_len + seq_len)  # = 8
    new_pos_ids = torch.arange(pos_range)[None, :]
    result = m.get_position_ids_args(new_pos_ids, len_p=past_len, len_s=seq_len)
    assert result["position_ids"].shape == (1, 4)
    assert result["position_ids"][0, 0].item() == 4    # starts at past_len
    assert result["position_ids"][0, -1].item() == 7   # ends at past_len + seq_len - 1
