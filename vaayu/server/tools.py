"""Tool-call detection and parsing.

Pure: strings in, schemas.ToolCall out.

The format is found by PROBING the tokenizer's chat template text -- never by
matching a model name or architecture string (CLAUDE.md §5: detection over
assumption). A model-name table would silently no-op on the next checkpoint, and
this repo has already paid for that class of bug once (VAAYU_CACHE_BIAS).
"""
import hashlib
import json
import re
from dataclasses import dataclass

from .schemas import ToolCall, ToolCallFunction


@dataclass(frozen=True)
class ToolCallFormat:
    name: str
    open_marker: str
    close_marker: str

    def stop_strings(self) -> list[str]:
        """Registered as generation stop strings, so decode halts AT the call
        instead of running on to hallucinate the tool's result."""
        return [self.close_marker]


# The Hermes/Qwen family: an explicit marker pair around a JSON object.
HERMES = ToolCallFormat("hermes", "<tool_call>", "</tool_call>")

# A template that accepts `tools` but ships no distinctive marker. Falls back to
# a fenced JSON block, and only accepts one that actually looks like a call.
GENERIC_JSON = ToolCallFormat("generic_json", "```json", "```")

_KNOWN = (HERMES,)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def detect_tool_format(chat_template):
    """Which tool-call syntax this checkpoint's template speaks, or None.

    None means the template has no tool support at all -- the caller must then
    REFUSE a request carrying `tools` rather than silently dropping them
    (CLAUDE.md §5: warn, never silently degrade).
    """
    if not chat_template:
        return None
    for fmt in _KNOWN:
        if fmt.open_marker in chat_template:
            return fmt
    # No known marker. A template that references a `tools` variable still
    # supports tools -- it just renders them without a distinctive wrapper.
    if "tools" in chat_template:
        return GENERIC_JSON
    return None


def _call_id(index: int, name: str, arguments: str) -> str:
    """Deterministic, so tests can assert on it and a retry produces the same id."""
    digest = hashlib.sha1(f"{index}:{name}:{arguments}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _build(index: int, obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", {})
    # The OpenAI wire format carries arguments as a JSON *string*. Models emit
    # either an object or an already-encoded string; normalise both.
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args)
    return ToolCall(id=_call_id(index, name, arguments),
                    function=ToolCallFunction(name=name, arguments=arguments))


def _parse_marked(text: str, fmt: ToolCallFormat):
    calls, cleaned, cursor, index = [], [], 0, 0
    while True:
        start = text.find(fmt.open_marker, cursor)
        if start == -1:
            break
        end = text.find(fmt.close_marker, start + len(fmt.open_marker))
        if end == -1:
            break                      # unterminated -- leave the tail alone
        body = text[start + len(fmt.open_marker):end]
        try:
            obj = json.loads(body)
        except ValueError:
            # Never invent a call from JSON we could not read. Skip past this
            # marker and keep the raw text so the failure is visible.
            cursor = end + len(fmt.close_marker)
            continue
        call = _build(index, obj)
        if call is None:
            cursor = end + len(fmt.close_marker)
            continue
        calls.append(call)
        index += 1
        cleaned.append(text[cursor:start])
        cursor = end + len(fmt.close_marker)
    if not calls:
        return text, []
    cleaned.append(text[cursor:])
    return "".join(cleaned).strip(), calls


def _parse_fenced(text: str):
    calls, spans, index = [], [], 0
    for m in _FENCED_JSON.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            continue
        call = _build(index, obj)
        if call is None:
            continue        # ordinary JSON in a code answer -- not a tool call
        calls.append(call)
        index += 1
        spans.append(m.span())
    if not calls:
        return text, []
    out, cursor = [], 0
    for start, end in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out).strip(), calls


def parse_tool_calls(text: str, fmt):
    """Split generated text into (prose, tool_calls).

    Returns the text UNCHANGED and an empty list whenever nothing parseable was
    found, so a parsing failure degrades to "the model wrote some text" rather
    than to a swallowed response.
    """
    if fmt is None:
        return text, []
    if fmt is GENERIC_JSON:
        return _parse_fenced(text)
    return _parse_marked(text, fmt)
