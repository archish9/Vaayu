"""Tool-call detection and parsing.

Pure: strings in, schemas.ToolCall out.

The format is found by PROBING -- never by matching a model name or architecture
string (CLAUDE.md §5: detection over assumption). A model-name table would
silently no-op on the next checkpoint, and this repo has already paid for that
class of bug once (VAAYU_CACHE_BIAS).

Two ways to get a format, both probe-based:

  * `derive_tool_format(tokenizer)` renders a sentinel tool call through the
    checkpoint's OWN chat template and reads the delimiters back out. This is
    the primary path: it handles ANY template that can render a tool call,
    including the three native shapes real targets ship -- JSON in a marker
    (Qwen2.5), bare JSON (Llama), and nested XML in two dialects (MiniMax's
    `name="x"` attribute form and Qwen3.6's `<function=x>` equals form).

  * `detect_tool_format(chat_template)` is the fallback marker table, used only
    when the self-probe cannot run or derive a format (a model with no chat
    template, or one whose template will not render an assistant tool call).
"""
import hashlib
import json
import re
from dataclasses import dataclass

from .schemas import ToolCall, ToolCallFunction


@dataclass(frozen=True)
class ToolCallFormat:
    """How one checkpoint spells a tool call. `grammar` picks the parser; the
    rest are the literal delimiters that parser needs. A JSON format uses
    `open_marker`/`close_marker` + `name_key`/`args_key`; an XML format uses the
    wrapper markers plus the per-call and per-parameter tag delimiters."""
    name: str
    grammar: str                       # "json" | "xml"
    open_marker: str = ""              # wrapper around ONE call; "" = none
    close_marker: str = ""
    # json grammar
    name_key: str = "name"
    args_key: str = "arguments"
    require_declared: bool = False     # an unmarked object must name a declared tool (B7)
    # xml grammar
    call_open_prefix: str = ""         # '<invoke name="'  or '<function='
    call_open_suffix: str = ""         # '">'              or '>'
    call_close: str = ""               # '</invoke>'       or '</function>'
    param_open_prefix: str = ""        # '<parameter name="' or '<parameter='
    param_open_suffix: str = ""        # '">'                or '>'
    param_close: str = ""              # '</parameter>'

    def stop_strings(self) -> list[str]:
        """Registered as generation stop strings, so decode halts AT the call
        instead of running on to hallucinate the tool's result. A format with no
        close marker (bare JSON) relies on the model's own EOS."""
        return [self.close_marker] if self.close_marker else []


# The Hermes/Qwen family: an explicit marker pair around a JSON object.
HERMES = ToolCallFormat("hermes", "json", "<tool_call>", "</tool_call>")

# A template that accepts `tools` but ships no distinctive marker. Falls back to
# a fenced JSON block, and only accepts one NAMING A TOOL THE REQUEST DECLARED
# -- the fence itself is no evidence at all, since an ordinary code answer
# opens exactly the same one.
GENERIC_JSON = ToolCallFormat("generic_json", "json", "```json", "```",
                              require_declared=True)

_KNOWN = (HERMES,)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def detect_tool_format(chat_template):
    """Fallback: which tool-call syntax this template speaks, or None.

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


# ── self-probe: derive a format from the template's own rendering ─────────────

_PROBE_NAME, _PROBE_KEY, _PROBE_VAL = "ZZNAMEZZ", "KKKEYKK", "VVVALVV"


def derive_tool_format(tokenizer):
    """Render a sentinel tool call through `tokenizer`'s chat template and derive
    the format from what it produces. Returns None on ANY failure, so the caller
    falls back to `detect_tool_format`. Never raises."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    user = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {
        "name": _PROBE_NAME, "description": "probe",
        "parameters": {"type": "object",
                       "properties": {_PROBE_KEY: {"type": "string"}},
                       "required": [_PROBE_KEY]}}}]
    call = user + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": _PROBE_NAME, "arguments": {_PROBE_KEY: _PROBE_VAL}}}]}]
    plain = user + [{"role": "assistant", "content": "PLAINTEXTPAYLOAD"}]
    try:
        kw = {"tools": tools, "tokenize": False, "add_generation_prompt": False}
        call_render = tokenizer.apply_chat_template(call, **kw)
        plain_render = tokenizer.apply_chat_template(plain, **kw)
    except Exception:                               # noqa: BLE001 -- any template error
        return None
    if not isinstance(call_render, str) or not isinstance(plain_render, str):
        return None
    try:
        return derive_from_render(call_render, plain_render,
                                  _PROBE_NAME, _PROBE_KEY, _PROBE_VAL)
    except Exception:                               # noqa: BLE001 -- derivation is best-effort
        return None


def _tagname(tag: str) -> str:
    """'<invoke name="' -> 'invoke'; '<function=' -> 'function';
    '<minimax:tool_call>' -> 'minimax:tool_call'."""
    t = tag.lstrip("<").lstrip("/")
    for i, c in enumerate(t):
        if c in ' =>/':
            return t[:i]
    return t


def _tag_before(text: str, pos: int) -> str:
    """The last complete `<...>` tag ending (ignoring trailing whitespace) at
    `pos`, or "" if the text there does not end in a tag."""
    seg = text[:pos].rstrip()
    if not seg.endswith(">"):
        return ""
    start = seg.rfind("<")
    return seg[start:] if start != -1 else ""


def _payload(call_render: str, plain_render: str) -> str:
    """The call render minus its shared prefix with a plain assistant render --
    i.e. the assistant's own output, with the role header (which can itself end
    in `>`) stripped off so it is never read as a tool-call wrapper."""
    i = 0
    while (i < len(call_render) and i < len(plain_render)
           and call_render[i] == plain_render[i]):
        i += 1
    return call_render[i:]


def derive_from_render(call_render, plain_render, name, key, val):
    """Derive a ToolCallFormat from one sentinel render. `call_render` is the
    template's output for an assistant tool call carrying (`name`, {`key`:`val`});
    `plain_render` is its output for a plain assistant message, used only to
    strip the shared role header."""
    r = _payload(call_render, plain_render)
    i_name = r.find(name)
    if i_name == -1:
        return None
    before = r[:i_name]
    if before.endswith('="'):
        return _derive_xml(r, name, key, attr=True)
    if before.endswith("="):
        return _derive_xml(r, name, key, attr=False)
    return _derive_json(r, name, key)


def _derive_xml(r, name, key, attr):
    i_name = r.find(name)
    before = r[:i_name]
    call_open_prefix = r[before.rfind("<"):i_name]
    after = r[i_name + len(name):]
    gt = after.find(">")
    if gt == -1:
        return None
    call_open_suffix = after[:gt + 1]
    call_close = "</%s>" % _tagname(call_open_prefix)

    i_key = r.find(key)
    if i_key == -1:
        return None
    beforek = r[:i_key]
    param_open_prefix = r[beforek.rfind("<"):i_key]
    afterk = r[i_key + len(key):]
    pgt = afterk.find(">")
    if pgt == -1:
        return None
    param_open_suffix = afterk[:pgt + 1]
    param_close = "</%s>" % _tagname(param_open_prefix)

    # The wrapper is the tag immediately before the call tag.
    open_marker = _tag_before(r, before.rfind("<"))
    close_marker = ("</%s>" % _tagname(open_marker)) if open_marker else ""
    return ToolCallFormat(
        name="probed_xml", grammar="xml",
        open_marker=open_marker, close_marker=close_marker,
        call_open_prefix=call_open_prefix, call_open_suffix=call_open_suffix,
        call_close=call_close, param_open_prefix=param_open_prefix,
        param_open_suffix=param_open_suffix, param_close=param_close)


def _balanced(s, start):
    """The `{...}` object beginning at `s[start]`, string-aware. (substring, end)."""
    depth, i, in_str, esc = 0, start, False, False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1], i + 1
        i += 1
    return None, len(s)


def _derive_json(r, name, key):
    i_name = r.find(name)
    bo = r.rfind("{", 0, i_name)
    if bo == -1:
        return None
    obj_str, _ = _balanced(r, bo)
    if obj_str is None:
        return None
    try:
        obj = json.loads(obj_str)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name_key = next((k for k, v in obj.items() if v == name), None)
    args_key = next((k for k, v in obj.items() if isinstance(v, dict)), None)
    if name_key is None or args_key is None:
        return None
    open_marker = _tag_before(r, bo)
    close_marker = ("</%s>" % _tagname(open_marker)) if open_marker else ""
    return ToolCallFormat(
        name="probed_json", grammar="json",
        open_marker=open_marker, close_marker=close_marker,
        name_key=name_key, args_key=args_key,
        require_declared=(open_marker == ""))


# ── building ToolCall objects ─────────────────────────────────────────────────

def _call_id(index: int, name: str, arguments: str) -> str:
    """Deterministic, so tests can assert on it and a retry produces the same id."""
    digest = hashlib.sha1(f"{index}:{name}:{arguments}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _make_call(index: int, name: str, arguments: str):
    return ToolCall(id=_call_id(index, name, arguments),
                    function=ToolCallFunction(name=name, arguments=arguments))


def _build(index: int, obj, name_key="name", args_key="arguments"):
    if not isinstance(obj, dict):
        return None
    name = obj.get(name_key)
    if not isinstance(name, str) or not name:
        return None
    args = obj.get(args_key, {})
    # The OpenAI wire format carries arguments as a JSON *string*. Models emit
    # either an object or an already-encoded string; normalise both.
    arguments = args if isinstance(args, str) else json.dumps(args)
    return _make_call(index, name, arguments)


_SCALAR = {None, "string"}


def _coerce(value: str, jtype):
    """XML parameter values arrive as raw text; coerce to the schema's JSON type
    so `arguments` parses to what the client's `tools` declared. With no type
    (unknown tool or param) the raw string is kept -- never a guessed type."""
    if jtype == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if jtype == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if jtype == "boolean":
        low = value.strip().lower()
        return True if low == "true" else False if low == "false" else value
    if jtype in ("object", "array"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


# ── parsing generated text ────────────────────────────────────────────────────

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
        call = _build(index, obj, fmt.name_key, fmt.args_key)
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


def _parse_fenced(text: str, allowed_names):
    calls, spans, index = [], [], 0
    for m in _FENCED_JSON.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            continue
        call = _build(index, obj)
        if call is None or call.function.name not in allowed_names:
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


def _scan_json(s, start):
    """The balanced JSON value (object or array) beginning at `s[start]`,
    string-aware. (substring, ok)."""
    openers = {"{": "}", "[": "]"}
    stack, i, in_str, esc = [], start, False, False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in openers:
            stack.append(openers[c])
        elif c in "}]":
            if not stack or stack[-1] != c:
                return None, False
            stack.pop()
            if not stack:
                return s[start:i + 1], True
        i += 1
    return None, False


def _first_json(text):
    """The first balanced `{...}` / `[...]` in `text`, or None. Lets a bare call
    be read past trailing scaffolding (e.g. a decoded end-of-turn token) without
    hardcoding any such token."""
    for i, ch in enumerate(text):
        if ch in "{[":
            frag, ok = _scan_json(text, i)
            return frag if ok else None
    return None


def _parse_bare_json(text: str, fmt: ToolCallFormat, allowed_names):
    """An unmarked object (or list of objects) that IS the whole response. A
    bare object looks like an ordinary answer, so every call must name a declared
    tool, and a single unparseable item sends the whole text back as prose."""
    frag = _first_json(text)
    if frag is None:
        return text, []
    try:
        obj = json.loads(frag)
    except ValueError:
        return text, []
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for index, it in enumerate(items):
        call = _build(index, it, fmt.name_key, fmt.args_key)
        if call is None:
            return text, []
        if fmt.require_declared and call.function.name not in allowed_names:
            return text, []
        calls.append(call)
    if not calls:
        return text, []
    return "", calls


def _parse_params(body: str, fmt: ToolCallFormat, schema):
    args, cursor = {}, 0
    while True:
        po = body.find(fmt.param_open_prefix, cursor)
        if po == -1:
            break
        kstart = po + len(fmt.param_open_prefix)
        kend = body.find(fmt.param_open_suffix, kstart)
        if kend == -1:
            break
        pkey = body[kstart:kend]
        vstart = kend + len(fmt.param_open_suffix)
        vend = body.find(fmt.param_close, vstart)
        if vend == -1:
            break
        args[pkey] = _coerce(body[vstart:vend].strip(), schema.get(pkey))
        cursor = vend + len(fmt.param_close)
    return args


def _parse_xml(text: str, fmt: ToolCallFormat, tool_schemas):
    schemas = tool_schemas or {}
    calls, cleaned, cursor, index = [], [], 0, 0
    while True:
        wopen = text.find(fmt.open_marker, cursor)
        if wopen == -1:
            break
        wclose = text.find(fmt.close_marker, wopen + len(fmt.open_marker))
        if wclose == -1:
            break                      # unterminated wrapper -- leave the tail
        block = text[wopen + len(fmt.open_marker):wclose]
        block_calls = _parse_invokes(block, fmt, schemas, index)
        if block_calls:
            cleaned.append(text[cursor:wopen])
            calls.extend(block_calls)
            index += len(block_calls)
        cursor = wclose + len(fmt.close_marker)
    if not calls:
        return text, []
    cleaned.append(text[cursor:])
    return "".join(cleaned).strip(), calls


def _parse_invokes(block: str, fmt: ToolCallFormat, schemas, start_index):
    calls, cursor, index = [], 0, start_index
    while True:
        co = block.find(fmt.call_open_prefix, cursor)
        if co == -1:
            break
        nstart = co + len(fmt.call_open_prefix)
        nend = block.find(fmt.call_open_suffix, nstart)
        if nend == -1:
            break
        name = block[nstart:nend]
        cclose = block.find(fmt.call_close, nend)
        if cclose == -1:
            break                      # unterminated invoke -- stop here
        body = block[nend + len(fmt.call_open_suffix):cclose]
        args = _parse_params(body, fmt, schemas.get(name, {}))
        if name:
            calls.append(_make_call(index, name, json.dumps(args)))
            index += 1
        cursor = cclose + len(fmt.call_close)
    return calls


def declared_tool_names(tools):
    """The tool names carried by THIS request -- the only names a bare object
    may claim. `tools` is the request's `list[ToolDef]`, or None."""
    return frozenset(t.function.name for t in (tools or []))


def declared_tool_schemas(tools):
    """{tool name: {param name: json type string}} from THIS request, for XML
    value coercion. Params with no declared type simply map to None."""
    out = {}
    for t in (tools or []):
        params = (t.function.parameters or {}).get("properties", {}) \
            if isinstance(t.function.parameters, dict) else {}
        out[t.function.name] = {
            k: (v.get("type") if isinstance(v, dict) else None)
            for k, v in params.items()}
    return out


def parse_tool_calls(text: str, fmt, allowed_names=frozenset(), tool_schemas=None):
    """Split generated text into (prose, tool_calls).

    Returns the text UNCHANGED and an empty list whenever nothing parseable was
    found, so a parsing failure degrades to "the model wrote some text" rather
    than to a swallowed response.

    `allowed_names` gates the formats where a block could be an ordinary answer
    (the ```json fence and an unmarked bare object): a call must name a declared
    tool. It DEFAULTS TO EMPTY, so a caller that forgets it loses tool calls
    rather than prose -- the failure that costs nothing, versus the one that
    deletes the answer. `tool_schemas` (from `declared_tool_schemas`) types XML
    parameter values; without it XML values stay strings.
    """
    if fmt is None:
        return text, []
    if fmt is GENERIC_JSON:
        return _parse_fenced(text, allowed_names)
    if fmt.grammar == "xml":
        return _parse_xml(text, fmt, tool_schemas)
    if fmt.open_marker:
        return _parse_marked(text, fmt)
    return _parse_bare_json(text, fmt, allowed_names)
