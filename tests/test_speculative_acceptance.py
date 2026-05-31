import torch
import pytest
from vaayu.speculative.base import SpeculativeDecoder


class ConcreteDecoder(SpeculativeDecoder):
    """Minimal subclass to test the acceptance algorithm in isolation."""

    def draft(self, input_ids, past_key_values=None, num_tokens=5):
        raise NotImplementedError

    def vram_usage_bytes(self):
        return 0

    def rollback_cache(self, draft_state, num_accepted):
        pass

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1


def _logits(predicted_tokens, vocab=100):
    """Logits where argmax produces the given token at each position."""
    logits = torch.zeros(1, len(predicted_tokens), vocab)
    for i, tok in enumerate(predicted_tokens):
        logits[0, i, tok] = 10.0
    return logits


@pytest.fixture
def dec():
    return ConcreteDecoder()


def test_all_drafts_accepted_returns_k_plus_bonus(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # verifier predicts 10, 20, 30 at positions 0-2; bonus=40 at position 3
    logits = _logits([10, 20, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 4
    assert accepted.tolist() == [[10, 20, 30, 40]]


def test_first_draft_rejected_returns_correction_only(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # verifier disagrees at pos 0: predicts 99, not 10
    logits = _logits([99, 20, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 1
    assert accepted.tolist() == [[99]]


def test_partial_acceptance_returns_correct_prefix(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # pos 0 ok (10), pos 1 mismatch (verifier=99, draft=20)
    logits = _logits([10, 99, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 2
    assert accepted.tolist() == [[10, 99]]
