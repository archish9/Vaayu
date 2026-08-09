"""F3 Phase 1: the batch's shared prefix, the cache broadcast, and the row slice.

All three are pure/duck-typed so they run without a model load -- CLAUDE.md's
method note about load-time decisions that can only be verified by hand-run
harnesses.
"""
import pytest
import torch

from vaayu.batch import BatchRequest, batch_common_prefix


def _req(ids):
    return BatchRequest(prompt_ids=list(ids), max_new_tokens=4)


def test_identical_prompts_share_everything():
    reqs = [_req([1, 2, 3, 4]), _req([1, 2, 3, 4])]
    assert batch_common_prefix(reqs) == [1, 2, 3, 4]


def test_shared_head_divergent_tail():
    reqs = [_req([1, 2, 3, 9]), _req([1, 2, 3, 8, 8]), _req([1, 2, 3])]
    assert batch_common_prefix(reqs) == [1, 2, 3]


def test_no_overlap_returns_empty():
    assert batch_common_prefix([_req([1, 2]), _req([9, 9])]) == []


def test_single_stream_is_its_whole_prompt():
    assert batch_common_prefix([_req([5, 6, 7])]) == [5, 6, 7]


def test_empty_inputs():
    assert batch_common_prefix([]) == []
    assert batch_common_prefix([_req([]), _req([1])]) == []


def test_a_later_request_value_mismatch_truncates():
    """A request beyond index 1 must be able to shorten the prefix by VALUE,
    not just by length -- otherwise a two-request-only implementation that
    truncates by min length passes the whole suite."""
    reqs = [_req([1, 2, 3, 9]), _req([1, 2, 3, 8]), _req([1, 9, 9])]
    assert batch_common_prefix(reqs) == [1]
