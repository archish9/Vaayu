"""Model-agnostic tool-format DISCOVERY by self-probe (CLAUDE.md §5, Tier B).

Instead of a marker table keyed on model names, derive the checkpoint's native
tool-call syntax by rendering a sentinel tool call through its OWN chat template
and reading the delimiters back out. A tool-call render is diffed against a
PLAIN assistant render so the role header (which also ends in `>` on Llama) is
not mistaken for a tool-call wrapper -- the same two-render-and-diff trick
`chat.generation_tail` uses for the thinking block.

The pure derivation is tested against the EXACT strings four real on-disk
templates render. A separate on-device gate renders through the live tokenizers
and derives, skipped when a model dir is absent.
"""
import json
import os

import pytest

from vaayu.server.tools import (derive_from_render, derive_tool_format,
                                parse_tool_calls)

N, K, V = "ZZNAMEZZ", "KKKEYKK", "VVVALVV"

# Exact (call, plain) assistant renders captured from each checkpoint's template.
FIXTURES = {
    "minimax": {
        "call": ']~b]ai\n\n<minimax:tool_call>\n<invoke name="ZZNAMEZZ">\n'
                '<parameter name="KKKEYKK">VVVALVV</parameter>\n</invoke>\n'
                '</minimax:tool_call>[e~[\n',
        "plain": ']~b]ai\nPLAINTEXTPAYLOAD[e~[\n',
    },
    "qwen36": {
        "call": '<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n'
                '<function=ZZNAMEZZ>\n<parameter=KKKEYKK>\nVVVALVV\n</parameter>\n'
                '</function>\n</tool_call><|im_end|>\n',
        "plain": '<|im_start|>assistant\n<think>\n\n</think>\n\n'
                 'PLAINTEXTPAYLOAD<|im_end|>\n',
    },
    "qwen25": {
        "call": '<|im_start|>assistant\n<tool_call>\n'
                '{"name": "ZZNAMEZZ", "arguments": {"KKKEYKK": "VVVALVV"}}\n'
                '</tool_call><|im_end|>\n',
        "plain": '<|im_start|>assistant\nPLAINTEXTPAYLOAD<|im_end|>\n',
    },
    "llama": {
        "call": '<|start_header_id|>assistant<|end_header_id|>\n\n'
                '{"name": "ZZNAMEZZ", "parameters": {"KKKEYKK": "VVVALVV"}}<|eot_id|>',
        "plain": '<|start_header_id|>assistant<|end_header_id|>\n\n'
                 'PLAINTEXTPAYLOAD<|eot_id|>',
    },
}


def _derive(key):
    f = FIXTURES[key]
    return derive_from_render(f["call"], f["plain"], N, K, V)


def _payload(key):
    """What the model actually emits -- the call render minus the shared header
    with a plain render, i.e. what a parser sees at runtime."""
    call, plain = FIXTURES[key]["call"], FIXTURES[key]["plain"]
    i = 0
    while i < len(call) and i < len(plain) and call[i] == plain[i]:
        i += 1
    return call[i:]


def test_derive_minimax_xml_attribute_form():
    fmt = _derive("minimax")
    assert fmt.grammar == "xml"
    assert fmt.open_marker == "<minimax:tool_call>"
    assert fmt.close_marker == "</minimax:tool_call>"
    assert fmt.call_open_prefix == '<invoke name="'
    assert fmt.call_open_suffix == '">'
    assert fmt.call_close == "</invoke>"
    assert fmt.param_open_prefix == '<parameter name="'
    assert fmt.param_open_suffix == '">'
    assert fmt.param_close == "</parameter>"


def test_derive_qwen36_xml_equals_form():
    fmt = _derive("qwen36")
    assert fmt.grammar == "xml"
    assert fmt.open_marker == "<tool_call>"
    assert fmt.close_marker == "</tool_call>"
    assert fmt.call_open_prefix == "<function="
    assert fmt.call_open_suffix == ">"
    assert fmt.call_close == "</function>"
    assert fmt.param_open_prefix == "<parameter="
    assert fmt.param_close == "</parameter>"


def test_derive_qwen25_json_in_a_marker():
    fmt = _derive("qwen25")
    assert fmt.grammar == "json"
    assert fmt.open_marker == "<tool_call>"
    assert fmt.close_marker == "</tool_call>"
    assert fmt.name_key == "name"
    assert fmt.args_key == "arguments"
    assert fmt.require_declared is False


def test_derive_llama_bare_json_with_the_parameters_key():
    fmt = _derive("llama")
    assert fmt.grammar == "json"
    assert fmt.open_marker == ""            # role header is NOT mistaken for one
    assert fmt.name_key == "name"
    assert fmt.args_key == "parameters"
    assert fmt.require_declared is True     # a bare object needs a declared name


@pytest.mark.parametrize("key", list(FIXTURES))
def test_a_derived_format_round_trips_the_emitted_payload(key):
    fmt = _derive(key)
    out = _payload(key)
    _, calls = parse_tool_calls(out, fmt, allowed_names={N},
                                tool_schemas={N: {K: "string"}})
    assert len(calls) == 1, f"{key}: {out!r}"
    assert calls[0].function.name == N
    assert json.loads(calls[0].function.arguments) == {K: V}


# ── on-device: derive through the live tokenizer for each model present ───────
MODELS = os.environ.get("MODELS_DIR", "/home/a/code/ai-ml/models")
CASES = {
    "NeuroSenko--MiniMax-M2.7-exl3": "xml",
    "darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6": "xml",
    "Qwen2.5-Coder-1.5B": "json",
    "Llama-3.2-1B-Instruct": "json",
}


@pytest.mark.parametrize("subdir,grammar", list(CASES.items()))
def test_derive_tool_format_through_the_real_tokenizer(subdir, grammar):
    path = os.path.join(MODELS, subdir)
    if not os.path.isdir(path):
        pytest.skip(f"model dir absent: {path}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    fmt = derive_tool_format(tok)
    assert fmt is not None, f"probe failed to derive a format for {subdir}"
    assert fmt.grammar == grammar
