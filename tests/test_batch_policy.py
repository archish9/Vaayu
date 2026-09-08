"""Batch formation: prefill chunk passes, batched against serial.

The rule exists because batching and the cross-turn prefix cache are in direct
conflict. batched_generate does ONE lookup on the batch's COMMON prefix and
never looks up each stream's own session entry, so batching two warm agent
sessions discards each one's warm prefix -- the largest measured lever in this
engine -- to buy the batch win. Prefill here is expert-weight streaming per
chunk, so both plans are priced in chunk passes, derived from the prompts and
prefill_chunk_size. No model constants (CLAUDE.md section 5).

Shapes below are the two on-device geometries' real agentic contexts: MiniMax
4165 tokens, Qwen 4177, at the shipped PREFILL_CHUNK_SIZE of 512.
"""
import pytest

from vaayu.server.batch_policy import (Candidate, batched_passes, chunk_passes,
                                       select_width, serial_passes)

CHUNK = 512                       # server config default, not a model number


def _prompt(n, tail_seed=0):
    """`n` tokens: a shared 200-token system prompt, then distinct content."""
    return [7] * 200 + [tail_seed + i for i in range(n - 200)]


@pytest.mark.parametrize("total", [4165, 4177])     # MiniMax-like, Qwen-like
def test_two_cold_divergent_prompts_batch(total):
    cands = [Candidate(_prompt(total, 0)), Candidate(_prompt(total, 10 ** 6))]
    assert select_width(cands, max_batch_size=2, chunk=CHUNK) == 2


@pytest.mark.parametrize("total", [4165, 4177])
def test_two_warm_prompts_serialize(total):
    """Each stream's own entry covers all but its last 40 tokens -- one chunk
    pass each serially, against eight for the batch. Batching would throw the
    warm prefixes away."""
    cands = [Candidate(_prompt(total, 0), own_match=total - 40),
             Candidate(_prompt(total, 10 ** 6), own_match=total - 40)]
    assert select_width(cands, max_batch_size=2, chunk=CHUNK) == 1


def test_width_is_capped_by_max_batch_size():
    cands = [Candidate(_prompt(4165, s)) for s in (0, 10 ** 6, 2 * 10 ** 6)]
    assert select_width(cands, max_batch_size=2, chunk=CHUNK) == 2


def test_max_batch_size_of_one_never_forms_a_batch():
    cands = [Candidate(_prompt(4165, s)) for s in (0, 10 ** 6)]
    assert select_width(cands, max_batch_size=1, chunk=CHUNK) == 1


def test_first_candidate_with_warm_prefix_still_batches_on_cost():
    """Pure cost comparison: even a candidate with warm prefix (own_match=4155)
    batches if the cost metric favors it. Candidates 0,1,2: serial 9+9+1=19,
    batched 8 (common prefix 200). Batching wins, so width=3 not 1."""
    cands = [Candidate(_prompt(4165, 0)),
             Candidate(_prompt(4165, 10 ** 6)),
             Candidate(_prompt(4165, 2 * 10 ** 6), own_match=4165 - 10)]
    assert select_width(cands, max_batch_size=3, chunk=CHUNK) == 3


def test_a_single_candidate_is_always_width_one():
    assert select_width([Candidate(_prompt(4165))], max_batch_size=3,
                        chunk=CHUNK) == 1


def test_small_nonzero_own_match_batches_when_cost_favors():
    """Small own_match does not veto batching if the cost comparison says batch.
    5000-token prompts, common prefix 200: serial costs 20 (10+10), batched costs
    10. Must batch (width=2) not serialize on the strength of one token match."""
    cands = [Candidate(_prompt(5000, 0), own_match=0),
             Candidate(_prompt(5000, 10 ** 6), own_match=1)]
    assert select_width(cands, max_batch_size=2, chunk=CHUNK) == 2


def test_chunk_passes_rounds_up_and_floors_at_zero():
    assert chunk_passes(0, 512) == 0
    assert chunk_passes(1, 512) == 1
    assert chunk_passes(512, 512) == 1
    assert chunk_passes(513, 512) == 2
    assert chunk_passes(-5, 512) == 0


def test_chunk_of_zero_is_one_unchunked_pass():
    """PREFILL_CHUNK_SIZE = 0 disables chunking; it must not divide by zero."""
    assert chunk_passes(4165, 0) == 1


def test_serial_and_batched_price_the_same_pair_the_two_ways():
    cands = [Candidate(_prompt(4165, 0), own_match=4000),
             Candidate(_prompt(4165, 10 ** 6), own_match=4000)]
    assert serial_passes(cands, CHUNK) == 2            # ceil(165/512) each
    assert batched_passes(cands, CHUNK) == 8           # ceil((4165-200)/512)
