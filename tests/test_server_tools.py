"""Tool-call detection and parsing. Pure -- strings in, ToolCall objects out.

Detection reads the chat TEMPLATE, never a model name (CLAUDE.md §5: detection
over assumption). The template snippets below are trimmed from the shapes real
checkpoints ship.
"""
from vaayu.server.tools import (GENERIC_JSON, HERMES, detect_tool_format,
                                parse_tool_calls)

HERMES_TEMPLATE = (
    "{% for message in messages %}{% if tools %}"
    "You may call tools. Return a call as <tool_call>{\"name\": ..., "
    "\"arguments\": ...}</tool_call>{% endif %}{% endfor %}"
)
TOOLS_BUT_NO_MARKER_TEMPLATE = (
    "{% if tools %}Available tools: {{ tools | tojson }}{% endif %}"
    "{% for message in messages %}{{ message.content }}{% endfor %}"
)
PLAIN_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"


def test_hermes_marker_in_the_template_is_detected():
    assert detect_tool_format(HERMES_TEMPLATE) is HERMES


def test_a_template_that_takes_tools_but_uses_no_known_marker_falls_back_to_json():
    assert detect_tool_format(TOOLS_BUT_NO_MARKER_TEMPLATE) is GENERIC_JSON


def test_a_template_with_no_tool_support_is_reported_as_none():
    assert detect_tool_format(PLAIN_TEMPLATE) is None


def test_a_missing_template_is_reported_as_none():
    assert detect_tool_format(None) is None


def test_detection_never_looks_at_a_model_name():
    # A template naming Qwen but supporting no tools must NOT be detected.
    assert detect_tool_format("Qwen3 assistant. {{ messages }}") is None


def test_a_single_hermes_call_is_parsed_and_stripped_from_the_text():
    out = ('Reading it now.\n'
           '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>')
    text, calls = parse_tool_calls(out, HERMES)
    assert text == "Reading it now."
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert calls[0].function.arguments == '{"path": "a.py"}'
    assert calls[0].id.startswith("call_")
    assert calls[0].type == "function"


def test_two_calls_are_parsed_in_order_with_distinct_ids():
    out = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
    text, calls = parse_tool_calls(out, HERMES)
    assert text == ""
    assert [c.function.name for c in calls] == ["a", "b"]
    assert calls[0].id != calls[1].id


def test_an_unterminated_call_is_left_alone_rather_than_guessed():
    out = 'thinking <tool_call>{"name": "a", "arg'
    text, calls = parse_tool_calls(out, HERMES)
    assert calls == []
    assert out == text


def test_malformed_json_inside_the_markers_yields_no_call_and_loses_no_text():
    # Never invent a call from JSON we could not read. The raw text goes back to
    # the client so the failure is visible rather than swallowed.
    out = '<tool_call>{"name": "a", "arguments": <<broken>>}</tool_call>'
    text, calls = parse_tool_calls(out, HERMES)
    assert calls == []
    assert text == out


def test_a_call_missing_its_name_is_rejected():
    out = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    _, calls = parse_tool_calls(out, HERMES)
    assert calls == []


def test_arguments_are_always_a_json_string_even_when_the_model_emits_an_object():
    out = '<tool_call>{"name": "a", "arguments": {"n": 1}}</tool_call>'
    _, calls = parse_tool_calls(out, HERMES)
    assert isinstance(calls[0].function.arguments, str)


def test_missing_arguments_become_an_empty_json_object():
    out = '<tool_call>{"name": "ping"}</tool_call>'
    _, calls = parse_tool_calls(out, HERMES)
    assert calls[0].function.arguments == "{}"


def test_generic_json_parses_a_fenced_block():
    out = 'Sure.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    text, calls = parse_tool_calls(out, GENERIC_JSON, {"read_file"})
    assert calls[0].function.name == "read_file"
    assert text == "Sure."


def test_generic_json_leaves_a_block_naming_an_UNDECLARED_tool_as_prose():
    # B7: a coding agent asks for a package.json. The fence carries a perfectly
    # good `name`, so the old fallback deleted it from the prose and returned a
    # fabricated tool call `my-app`. The request declared `read_file` and
    # nothing else, so this is a code answer, not a call.
    out = 'Here is package.json:\n```json\n{"name": "my-app", "version": "1.0.0"}\n```\ndone'
    text, calls = parse_tool_calls(out, GENERIC_JSON, {"read_file"})
    assert calls == []
    assert text == out


def test_generic_json_with_no_declared_names_can_never_produce_a_call():
    # Fail-safe default: with nothing declared, no fenced block can be a call.
    # The worst case of getting this wrong is destroyed prose, so the default
    # errs towards keeping the text.
    out = 'x\n```json\n{"name": "read_file", "arguments": {}}\n```'
    text, calls = parse_tool_calls(out, GENERIC_JSON)
    assert calls == []
    assert text == out


def test_generic_json_strips_only_the_declared_call_and_keeps_the_other_block():
    out = ('call:\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```\n'
           'and config:\n```json\n{"name": "my-app"}\n```')
    text, calls = parse_tool_calls(out, GENERIC_JSON, {"read_file"})
    assert [c.function.name for c in calls] == ["read_file"]
    assert '"my-app"' in text
    assert "read_file" not in text


def test_generic_json_ignores_a_fenced_block_that_is_not_a_tool_call():
    # Ordinary JSON in a code answer must not be hijacked into a tool call.
    out = 'Here is the config:\n```json\n{"port": 8000}\n```'
    text, calls = parse_tool_calls(out, GENERIC_JSON)
    assert calls == []
    assert text == out


def test_no_format_means_no_parsing_and_the_text_is_untouched():
    out = '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
    text, calls = parse_tool_calls(out, None)
    assert calls == []
    assert text == out
