# tests/test_speculative_standard.py
import torch
import pytest
from unittest.mock import MagicMock, patch


class _FakeDrafter:
    """Callable fake drafter for testing."""
    def __init__(self, predicted_tokens):
        self.predicted_tokens = predicted_tokens
        self.call_count = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        tok = self.predicted_tokens[min(self.call_count, len(self.predicted_tokens) - 1)]
        self.call_count += 1
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, tok] = 10.0
        out = MagicMock()
        out.logits = logits
        out.past_key_values = MagicMock()
        out.past_key_values.get_seq_length = MagicMock(return_value=self.call_count)
        return out

    def parameters(self):
        return [torch.zeros(10)]

    def eval(self):
        pass


def _make_drafter_mock(predicted_tokens):
    drafter = _FakeDrafter(predicted_tokens)
    return drafter


def _new_decoder(tokens):
    from vaayu.speculative.standard import StandardDrafterDecoder
    dec = StandardDrafterDecoder.__new__(StandardDrafterDecoder)
    dec.device = "cpu"
    dec.drafter = _make_drafter_mock(tokens)
    dec._drafter_past = None
    dec._vram_bytes = 0
    return dec


def test_cold_start_draft_prefills_then_generates():
    dec = _new_decoder([10, 20, 30])
    input_ids = torch.tensor([[1, 2, 3]])
    draft_ids, state = dec.draft(input_ids, num_tokens=3)
    assert draft_ids.shape == (1, 3)
    assert state["num_drafted"] == 3


def test_warm_start_draft_uses_cached_past():
    dec = _new_decoder([10, 20, 30])
    input_ids = torch.tensor([[1, 2, 3]])
    dec.draft(input_ids, num_tokens=2)
    assert dec._drafter_past is not None
    dec.draft(torch.tensor([[1, 2, 3, 10]]), num_tokens=2)
    assert dec._drafter_past is not None


def test_verify_and_accept_all_match():
    dec = _new_decoder([])
    draft_ids = torch.tensor([[10, 20]])
    logits = torch.zeros(1, 3, 100)
    logits[0, 0, 10] = 10.0
    logits[0, 1, 20] = 10.0
    logits[0, 2, 30] = 10.0
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 3
    assert accepted.tolist() == [[10, 20, 30]]


def test_verify_and_accept_first_mismatch():
    dec = _new_decoder([])
    draft_ids = torch.tensor([[10, 20]])
    logits = torch.zeros(1, 3, 100)
    logits[0, 0, 99] = 10.0
    logits[0, 1, 20] = 10.0
    logits[0, 2, 30] = 10.0
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 1
    assert accepted.tolist() == [[99]]


def test_rollback_clears_cache_on_partial_accept():
    dec = _new_decoder([])
    dec._drafter_past = MagicMock()
    dec.rollback_cache({"num_drafted": 5}, num_accepted=3)
    assert dec._drafter_past is None


def test_rollback_keeps_cache_on_full_accept():
    dec = _new_decoder([])
    sentinel = MagicMock()
    dec._drafter_past = sentinel
    dec.rollback_cache({"num_drafted": 5}, num_accepted=6)
    assert dec._drafter_past is sentinel
