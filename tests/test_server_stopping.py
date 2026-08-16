"""Stopping policy for the server. Pure -- no torch, no model, no checkpoint.

The max_seq_len values below are the DERIVED per-load context windows of the four
checkpoints on this box (todo.md's "Checkpoints on disk" table, verified on-device
4/4 by row 9). They are test DATA, not constants the implementation may contain:
CLAUDE.md §5 forbids a literal context length in vaayu/.
"""
import pytest

from vaayu.server.stopping import ContextTooLong, finish_reason, resolve_max_tokens

DERIVED_WINDOWS = [
    ("qwen3-235b", 8192),
    ("minimax-m2.7", 8192),
    ("llama-3.3-70b", 8192),
    ("llama-3.2-1b", 71981),
]


@pytest.mark.parametrize("name,max_seq_len", DERIVED_WINDOWS)
def test_omitted_max_tokens_defaults_to_the_whole_remaining_window(name, max_seq_len):
    prompt_len = 1000
    assert resolve_max_tokens(None, prompt_len, max_seq_len) == max_seq_len - prompt_len


@pytest.mark.parametrize("name,max_seq_len", DERIVED_WINDOWS)
def test_requested_cap_is_clamped_to_the_remaining_window(name, max_seq_len):
    prompt_len = max_seq_len - 10
    assert resolve_max_tokens(1_000_000, prompt_len, max_seq_len) == 10


@pytest.mark.parametrize("name,max_seq_len", DERIVED_WINDOWS)
def test_a_smaller_request_is_honoured_verbatim(name, max_seq_len):
    assert resolve_max_tokens(64, 100, max_seq_len) == 64


@pytest.mark.parametrize("name,max_seq_len", DERIVED_WINDOWS)
def test_prompt_filling_the_window_is_refused_not_truncated(name, max_seq_len):
    with pytest.raises(ContextTooLong) as exc:
        resolve_max_tokens(None, max_seq_len, max_seq_len)
    # The message must name both numbers -- a refusal the caller cannot act on
    # is the diagnostic failure this repo keeps paying for.
    assert str(max_seq_len) in str(exc.value)


def test_a_prompt_longer_than_the_window_is_refused():
    with pytest.raises(ContextTooLong):
        resolve_max_tokens(None, 9000, 8192)


def test_non_positive_max_tokens_is_rejected():
    with pytest.raises(ValueError):
        resolve_max_tokens(0, 100, 8192)
    with pytest.raises(ValueError):
        resolve_max_tokens(-5, 100, 8192)


def test_eos_reports_stop():
    assert finish_reason(hit_eos=True, hit_stop_string=False, hit_tool_call=False,
                         produced=17, cap=100) == "stop"


def test_stop_string_reports_stop():
    assert finish_reason(hit_eos=False, hit_stop_string=True, hit_tool_call=False,
                         produced=17, cap=100) == "stop"


def test_hitting_the_cap_reports_length_so_the_client_knows_it_is_truncated():
    assert finish_reason(hit_eos=False, hit_stop_string=False, hit_tool_call=False,
                         produced=100, cap=100) == "length"


def test_a_tool_call_outranks_eos():
    # Templates emit the tool-call close marker and then EOS in the same breath.
    # Reporting "stop" there would make the client treat a tool call as prose.
    assert finish_reason(hit_eos=True, hit_stop_string=True, hit_tool_call=True,
                         produced=17, cap=100) == "tool_calls"


def test_cap_reached_and_eos_together_reports_stop():
    # The model finished ON the last admissible token: not a truncation.
    assert finish_reason(hit_eos=True, hit_stop_string=False, hit_tool_call=False,
                         produced=100, cap=100) == "stop"
