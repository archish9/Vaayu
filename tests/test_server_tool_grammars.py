"""Model-agnostic tool-call GRAMMARS: parsing a format's native tokens into
OpenAI ToolCall objects.

Vaayu must emit the OpenAI `tool_calls` shape for ANY checkpoint, not one model
(CLAUDE.md §5). Real checkpoints ship at least three native shapes, found by
probing the template (never a model name):

  * JSON in a marker pair   -- Qwen2.5-Coder: <tool_call>{"name":..}</tool_call>
  * bare JSON, no marker     -- Llama-3.x: {"name":.., "parameters":{..}}
  * nested XML               -- MiniMax:   <minimax:tool_call><invoke name=..>..
                             -- Qwen3.6:   <tool_call><function=..><parameter=..>

The strings below are the EXACT tokens these templates render for a tool call
(captured by rendering a sentinel call through each checkpoint's own template).
"""
import pytest

from vaayu.server.tools import (ToolCallFormat, parse_tool_calls)


# ── XML grammar, attribute form (MiniMax) ─────────────────────────────────────
MINIMAX = ToolCallFormat(
    name="minimax", grammar="xml",
    open_marker="<minimax:tool_call>", close_marker="</minimax:tool_call>",
    call_open_prefix='<invoke name="', call_open_suffix='">', call_close="</invoke>",
    param_open_prefix='<parameter name="', param_open_suffix='">',
    param_close="</parameter>")

# ── XML grammar, equals form (Qwen3.6) ────────────────────────────────────────
QWEN36 = ToolCallFormat(
    name="qwen36_xml", grammar="xml",
    open_marker="<tool_call>", close_marker="</tool_call>",
    call_open_prefix="<function=", call_open_suffix=">", call_close="</function>",
    param_open_prefix="<parameter=", param_open_suffix=">",
    param_close="</parameter>")

# ── JSON grammar, bare (Llama-3.x) -- args live under `parameters`, no marker ──
LLAMA = ToolCallFormat(name="llama_json", grammar="json",
                       open_marker="", close_marker="",
                       name_key="name", args_key="parameters",
                       require_declared=True)


def test_minimax_xml_call_parses_to_a_toolcall():
    out = ('12 times 7 equals 84.\n'
           '<minimax:tool_call>\n<invoke name="multiply">\n'
           '<parameter name="a">12</parameter>\n'
           '<parameter name="b">7</parameter>\n</invoke>\n</minimax:tool_call>')
    schemas = {"multiply": {"a": "integer", "b": "integer"}}
    text, calls = parse_tool_calls(out, MINIMAX, tool_schemas=schemas)
    assert text == "12 times 7 equals 84."
    assert len(calls) == 1
    assert calls[0].function.name == "multiply"
    # arguments is a JSON string, and the schema coerces "12" -> 12 (not "12").
    import json
    assert json.loads(calls[0].function.arguments) == {"a": 12, "b": 7}
    assert calls[0].type == "function"
    assert calls[0].id.startswith("call_")


def test_minimax_without_a_schema_keeps_values_as_strings():
    # No schema to coerce against -> never guess a type; leave the raw string.
    out = ('<minimax:tool_call>\n<invoke name="f">\n'
           '<parameter name="x">hello</parameter>\n</invoke>\n</minimax:tool_call>')
    _, calls = parse_tool_calls(out, MINIMAX)
    import json
    assert json.loads(calls[0].function.arguments) == {"x": "hello"}


def test_minimax_two_invokes_in_one_block_make_two_calls():
    out = ('<minimax:tool_call>\n'
           '<invoke name="a">\n<parameter name="x">1</parameter>\n</invoke>\n'
           '<invoke name="b">\n<parameter name="y">2</parameter>\n</invoke>\n'
           '</minimax:tool_call>')
    schemas = {"a": {"x": "integer"}, "b": {"y": "integer"}}
    _, calls = parse_tool_calls(out, MINIMAX, tool_schemas=schemas)
    assert [c.function.name for c in calls] == ["a", "b"]
    assert calls[0].id != calls[1].id


def test_qwen36_equals_form_xml_parses_with_value_on_its_own_line():
    out = ('<tool_call>\n<function=search>\n'
           '<parameter=query>\nrust borrow checker\n</parameter>\n'
           '</function>\n</tool_call>')
    text, calls = parse_tool_calls(out, QWEN36)
    assert text == ""
    assert calls[0].function.name == "search"
    import json
    # value is trimmed of the surrounding newlines the template adds.
    assert json.loads(calls[0].function.arguments) == {"query": "rust borrow checker"}


def test_bare_json_llama_uses_the_parameters_key():
    out = '{"name": "read_file", "parameters": {"path": "a.py"}}'
    text, calls = parse_tool_calls(out, LLAMA, allowed_names={"read_file"})
    assert calls[0].function.name == "read_file"
    import json
    assert json.loads(calls[0].function.arguments) == {"path": "a.py"}
    assert text == ""


def test_bare_json_is_ignored_when_it_names_an_undeclared_tool():
    # A bare object is what an ordinary answer can look like too; only a declared
    # name may become a call (B7 safety, carried over from GENERIC_JSON).
    out = '{"name": "my-app", "version": "1.0.0"}'
    text, calls = parse_tool_calls(out, LLAMA, allowed_names={"read_file"})
    assert calls == []
    assert text == out


def test_bare_json_list_makes_several_calls():
    out = ('[{"name": "a", "parameters": {}}, '
           '{"name": "b", "parameters": {"x": 1}}]')
    _, calls = parse_tool_calls(out, LLAMA, allowed_names={"a", "b"})
    assert [c.function.name for c in calls] == ["a", "b"]


def test_an_unterminated_xml_block_is_left_as_prose():
    out = 'working <minimax:tool_call>\n<invoke name="a">\n<parameter name="x">1'
    text, calls = parse_tool_calls(out, MINIMAX)
    assert calls == []
    assert text == out
