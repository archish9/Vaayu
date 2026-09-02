"""The drafter's KV cache must not survive a speculative_generate call.

Measured on-device 2026-09-02 (Llama-3.2-1B as its own drafter, greedy, same
prompt, one process): three back-to-back calls returned three DIFFERENT token
sequences, and only the first was coherent -- later ones degraded into repeated
fragments. Clearing the drafter cache between calls made 3/3 byte-identical AND
equal to the first-ever run, so the cache carry-over is the whole cause.

StandardDrafterDecoder keeps `_drafter_past`/`_drafter_seen` across calls by
design (Fix #9 -- within one generation loop that avoids an O(seq) re-prefill
per rejection). Nothing reset them at the START of a generation, so call 2 fed
a NEW prompt's tail on top of call 1's cache: `input_ids[:, self._drafter_seen:]`
is sliced by a length that belongs to a different sequence.

Latent only because the drafter is off by default; the moment P7 ships it makes
every request after the first in a server process wrong.
"""
import torch

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.speculative.standard import StandardDrafterDecoder
from vaayu.vaayu_base import VaayuBaseModel

from .test_speculative_prefix_reuse import _FakeSpecModel, _FakeSpeculator


def _bare_drafter():
    """A StandardDrafterDecoder without loading weights: reset() must touch only
    the two cache attributes, so it is testable off-device."""
    d = object.__new__(StandardDrafterDecoder)
    d._drafter_past = object()
    d._drafter_seen = 41
    return d


def test_standard_drafter_reset_clears_its_cache():
    d = _bare_drafter()
    d.reset()
    assert d._drafter_past is None
    assert d._drafter_seen == 0


def test_the_base_contract_has_a_reset_every_speculator_answers():
    # PromptLookupDecoder holds no cross-call state; it must still answer the
    # call, or the engine has to type-check its own speculator.
    from vaayu.speculative.prompt_lookup import PromptLookupDecoder
    PromptLookupDecoder(eos_token_id=1).reset()


class _ResetRecordingSpeculator(_FakeSpeculator):
    def __init__(self):
        self.events = []

    def reset(self):
        self.events.append("reset")

    def draft(self, generated, past_key_values, num_tokens=5):
        self.events.append("draft")
        return super().draft(generated, past_key_values, num_tokens)


def _run(model, sid):
    return VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=3, session_id=sid)


def test_speculative_generate_resets_the_drafter_before_it_drafts():
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _ResetRecordingSpeculator()
    _run(model, "s1")
    assert model.speculator.events[0] == "reset", \
        "the drafter cache from a previous call was still live when drafting began"


def test_every_call_gets_its_own_reset():
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    model.speculator = _ResetRecordingSpeculator()
    _run(model, "s1")
    _run(model, "s2")
    assert model.speculator.events.count("reset") == 2
