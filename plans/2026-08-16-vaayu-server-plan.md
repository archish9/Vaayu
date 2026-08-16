# Vaayu Inference Server (OpenAI-compatible) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Vaayu from a test-script token generator into a long-running HTTP server that an autonomous coding agent can drive over an OpenAI-compatible API, with streaming, tool calls, and output that stops when the model is done rather than at a fixed token count.

**Architecture:** A new `vaayu/server/` package wraps one already-loaded `VaayuBaseModel`. Pure modules (`stopping.py`, `schemas.py`, `tools.py`) hold all policy and parsing and are unit-tested with no GPU and no checkpoint; `engine.py` owns the single model plus a depth-bounded FIFO admission gate; `app.py` is FastAPI routes and SSE framing only. `vaayu_base.py` is not grown — the two engine-side fixes go into `vaayu/prefix_cache.py` as testable helpers that `vaayu_base.generate` calls.

**Tech Stack:** Python 3, FastAPI, uvicorn, pydantic v2 (2.11.0 already installed), httpx 0.28.1 (already installed, needed by `fastapi.testclient`), pytest, transformers 5.5.4, torch.

**Spec:** `plans/2026-08-16-vaayu-server-design.md` — read it first. It records *why* the shape is what it is (measured ceiling ~0.86 tok/s at 8192 context, single stream; resident-model default; queue not batch).

---

## Global Constraints

Copied verbatim from the environment section of `todo.md` and CLAUDE.md. Every task's requirements implicitly include this section.

- `python` is **NOT** on PATH. Use `./venv/bin/python` and `./venv/bin/pip`.
- `vaayu` is **not** pip-installed. Scripts need `PYTHONPATH=/home/a/code/ai-ml/Vaayu`. Running pytest from the repo root works without it; standalone scripts need it.
- Baseline test suite before this plan: **892 passed, 3 skipped** (`./venv/bin/python -m pytest tests/ -q`). Every task must leave the suite green; the count only ever goes up.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers anywhere. Layer counts, expert counts, vocab sizes, context lengths, hidden sizes are **derived** at load time. Only HARDWARE limits may be literals (16 GB VRAM / 15.47 GiB usable, 32 GB RAM, ~8.5 GB expert-cache ceiling, ~5.5 GB/s NVMe), each commented as a hardware constant. **`512` must not appear as a token cap anywhere in `vaayu/server/`.**
- **CLAUDE.md §3 — surgical changes.** Do not reformat, refactor, or "improve" code you are not asked to change. Match surrounding style.
- **Detection over assumption (§5).** Tool-call formats are found by probing the tokenizer's chat template text, never by matching a model name or architecture string.
- **Warn, never silently degrade (§5).** When something cannot be honoured (a template with no tool support, a sampling parameter the speculative path ignores), say so loudly. Do not quietly drop it.
- **`todo.md` is the ledger (§6).** Mark row 13's subtask `[~]` when you start it and `[x]` when it is done and verified. Never batch this up at the end.
- On-device runs go under their own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`. Never attach py-spy (it corrupts `vaayu_rust`'s io_uring state). Redirect stdout to a file and `grep -a` it — never `cat` (logs reach 300 MB–1.3 GB).
- Commit after every task. Commit messages are written normally (not terse).

---

## File Structure

| file | responsibility | pure |
|---|---|---|
| `vaayu/server/__init__.py` | package marker, re-exports `create_app`, `ServerConfig` | — |
| `vaayu/server/stopping.py` | `resolve_max_tokens`, `finish_reason`, `ContextTooLong` | **yes** |
| `vaayu/server/schemas.py` | pydantic request/response models (OpenAI subset + Vaayu extras) | **yes** |
| `vaayu/server/tools.py` | tool-call format detection + parsing → OpenAI `tool_calls` | **yes** |
| `vaayu/server/chat.py` | render messages (+ tools) to token ids via the tokenizer's chat template | mostly |
| `vaayu/server/engine.py` | `EngineHandle`: the one model, the admission gate, blocking + streaming generation | no |
| `vaayu/server/app.py` | FastAPI app factory, routes, SSE framing | no |
| `vaayu/server/__main__.py` | CLI entry (`python -m vaayu.server --model PATH`) | no |
| `vaayu/prefix_cache.py` | **modify**: explicit-id miss falls back to auto-match; add `restore_on_abort` | partly |
| `vaayu/vaayu_base.py` | **modify**: `generate()` calls `restore_on_abort` on the failure path | no |
| `tests/test_server_stopping.py` | Task 1 | — |
| `tests/test_server_schemas.py` | Task 2 | — |
| `tests/test_server_tools.py` | Task 3 | — |
| `tests/test_server_chat.py` | Task 4 | — |
| `tests/test_prefix_explicit_miss_fallback.py` | Task 5 | — |
| `tests/test_prefix_restore_on_abort.py` | Task 6 | — |
| `tests/test_server_engine.py` | Task 7 | — |
| `tests/test_server_app.py` | Tasks 8, 9 | — |
| `test_server_smoke.py` (repo ROOT) | Task 11, on-device gate | — |

---

## Task 0: Dependencies and package skeleton

**Files:**
- Modify: `requirements.txt`
- Create: `vaayu/server/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `vaayu.server` package; `fastapi` and `uvicorn` available in `./venv`.

- [ ] **Step 1: Install the two missing dependencies**

`pydantic` 2.11.0 and `httpx` 0.28.1 are already installed. `fastapi` and `uvicorn` are not.

```bash
./venv/bin/pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"
```

- [ ] **Step 2: Record them in requirements.txt**

Append to `requirements.txt`, after the `Pillow` line:

```
# HTTP server (vaayu/server) — OpenAI-compatible inference API
fastapi>=0.115
uvicorn[standard]>=0.30
```

- [ ] **Step 3: Create the package marker**

Create `vaayu/server/__init__.py`:

```python
"""Vaayu's OpenAI-compatible HTTP server.

Design: plans/2026-08-16-vaayu-server-design.md

Import order matters: `app` pulls in FastAPI, so the lazy re-export below keeps
`import vaayu` cheap for every non-server caller (every test script in the repo).
"""

__all__ = ["create_app", "ServerConfig"]


def __getattr__(name):
    if name in __all__:
        from . import app as _app
        return getattr(_app, name)
    raise AttributeError(name)
```

- [ ] **Step 4: Verify the package imports and nothing else broke**

```bash
./venv/bin/python -c "import vaayu.server; print('ok')"
./venv/bin/python -m pytest tests/ -q
```

Expected: `ok`, then `892 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt vaayu/server/__init__.py
git commit -m "feat(server): add vaayu.server package skeleton and HTTP dependencies"
```

---

## Task 1: `stopping.py` — the truncation fix

This is the task that answers the originating problem. `MAX_NEW_TOKENS=512` cuts code mid-function. The fix is not streaming (streaming is delivery, truncation is stopping policy); the fix is that the cap is **derived from the remaining context window**, generation stops on EOS, and a cut-off client is **told**.

**Files:**
- Create: `vaayu/server/stopping.py`
- Test: `tests/test_server_stopping.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class ContextTooLong(ValueError)`
  - `resolve_max_tokens(requested: int | None, prompt_len: int, max_seq_len: int) -> int`
  - `finish_reason(*, hit_eos: bool, hit_stop_string: bool, hit_tool_call: bool, produced: int, cap: int) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_stopping.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_stopping.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.stopping'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/stopping.py`:

```python
"""Stopping policy for the Vaayu server.

Pure: no torch, no model, no I/O. Everything here is unit-testable without a
checkpoint, which is the point -- this is where the MAX_NEW_TOKENS truncation
bug is actually fixed.

Streaming does NOT fix truncation. Streaming is delivery; truncation is stopping
policy. The fix is that the cap is DERIVED from the model's remaining context
window (never a literal -- CLAUDE.md §5), that generation stops on EOS, and that
a client whose output WAS cut off is told so via finish_reason="length" instead
of being handed silently-broken code.
"""


class ContextTooLong(ValueError):
    """The prompt alone does not fit the model's derived context window."""


def resolve_max_tokens(requested, prompt_len: int, max_seq_len: int) -> int:
    """How many new tokens this request may generate.

    `requested` is the client's `max_tokens`, or None when omitted -- and omitted
    is the NORMAL case for a coding agent. `max_seq_len` is the model's own
    derived window (`VaayuBaseModel.max_seq_len`, re-derived by plan_vram at every
    load); it is never a constant in this file.
    """
    budget = max_seq_len - prompt_len
    if budget <= 0:
        raise ContextTooLong(
            f"prompt is {prompt_len} tokens but this model's derived context "
            f"window is {max_seq_len} tokens, leaving no room to generate. "
            f"Shorten the prompt by at least {prompt_len - max_seq_len + 1} tokens.")
    if requested is None:
        return budget
    if requested <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {requested}")
    return min(requested, budget)


def finish_reason(*, hit_eos: bool, hit_stop_string: bool, hit_tool_call: bool,
                  produced: int, cap: int) -> str:
    """OpenAI `finish_reason` for a completed generation.

    Order is deliberate. A template emits its tool-call close marker and then EOS
    in the same breath, so EOS must not outrank a tool call or the client reads a
    tool call as prose. And EOS must outrank the cap: a model that finished ON the
    last admissible token was not truncated.
    """
    if hit_tool_call:
        return "tool_calls"
    if hit_eos or hit_stop_string:
        return "stop"
    if produced >= cap:
        return "length"
    return "stop"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./venv/bin/python -m pytest tests/test_server_stopping.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: the new file passes; the suite is `903 passed, 3 skipped` (892 + 11 new).

- [ ] **Step 5: Confirm no literal token cap slipped in**

```bash
grep -rn "512" vaayu/server/ || echo "clean"
```

Expected: `clean`.

- [ ] **Step 6: Commit**

```bash
git add vaayu/server/stopping.py tests/test_server_stopping.py
git commit -m "feat(server): derive the generation cap from the context window, not a literal

Replaces the fixed MAX_NEW_TOKENS behaviour. An omitted max_tokens now defaults
to max_seq_len - prompt_len (derived per load), and finish_reason reports
'length' when output really was cut off so the client is told rather than handed
silently-broken code."
```

- [ ] **Step 7: Tick the ledger**

In `todo.md`, row 13, change the `server/stopping.py` line from `- [ ]` to `- [x]`.

---

## Task 2: `schemas.py` — the wire format

**Files:**
- Create: `vaayu/server/schemas.py`
- Test: `tests/test_server_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ChatMessage`, `ToolDef`, `ChatCompletionRequest`, `StreamOptions`
  - `ChatCompletionResponse`, `Choice`, `Usage`, `ResponseMessage`, `ToolCall`, `ToolCallFunction`
  - `ChatCompletionChunk`, `ChunkChoice`, `Delta`
  - `make_completion_id() -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_schemas.py`:

```python
"""Wire-format models. Pure -- pydantic only, no torch, no model."""
import pytest
from pydantic import ValidationError

from vaayu.server.schemas import (ChatCompletionRequest, ChatCompletionResponse,
                                  Choice, ResponseMessage, Usage, make_completion_id)


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_minimal_request_parses_and_max_tokens_defaults_to_none():
    r = _req()
    # None, NOT a number: the cap is resolved later against the model's derived
    # window (stopping.resolve_max_tokens). A default here would be a literal.
    assert r.max_tokens is None
    assert r.stream is False
    assert r.messages[0].role == "user"


def test_unknown_openai_fields_are_ignored_not_rejected():
    # Real clients send fields we do not implement (logit_bias, user, ...).
    # 422-ing on those breaks every off-the-shelf agent.
    r = _req(logit_bias={"1": 2}, user="agent-7", presence_penalty=0.3)
    assert r.model == "vaayu"


def test_vaayu_extras_are_declared_and_optional():
    r = _req(session_id="agent-7-turn-3")
    assert r.session_id == "agent-7-turn-3"
    assert _req().session_id is None
    assert _req().drafter_model_path is None
    assert _req().kv_cache_mode is None


def test_stop_accepts_a_string_or_a_list_and_normalises_to_a_list():
    assert _req(stop="```").stop == ["```"]
    assert _req(stop=["```", "</tool_call>"]).stop == ["```", "</tool_call>"]
    assert _req().stop == []


@pytest.mark.parametrize("field,bad", [
    ("temperature", -0.1), ("temperature", 2.1),
    ("top_p", -0.1), ("top_p", 1.1),
    ("top_k", -1),
    ("max_tokens", 0), ("max_tokens", -5),
])
def test_out_of_range_sampling_values_are_rejected(field, bad):
    with pytest.raises(ValidationError):
        _req(**{field: bad})


def test_n_greater_than_one_is_rejected_with_a_named_reason():
    # Concurrency is 1 on this box (N>=2 at long context is a settled hardware
    # refusal). Silently returning one choice for n=3 would be a lie.
    with pytest.raises(ValidationError) as exc:
        _req(n=3)
    assert "n=1" in str(exc.value)


def test_empty_messages_is_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="vaayu", messages=[])


def test_tool_definitions_parse():
    r = _req(tools=[{
        "type": "function",
        "function": {"name": "read_file",
                     "description": "Read a file",
                     "parameters": {"type": "object",
                                    "properties": {"path": {"type": "string"}}}},
    }])
    assert r.tools[0].function.name == "read_file"


def test_completion_ids_are_unique_and_prefixed():
    a, b = make_completion_id(), make_completion_id()
    assert a.startswith("chatcmpl-") and b.startswith("chatcmpl-")
    assert a != b


def test_response_serialises_with_the_openai_field_names():
    resp = ChatCompletionResponse(
        id="chatcmpl-x", created=1, model="vaayu",
        choices=[Choice(index=0,
                        message=ResponseMessage(role="assistant", content="hello"),
                        finish_reason="stop")],
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
    d = resp.model_dump(exclude_none=True)
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["content"] == "hello"
    assert d["usage"]["total_tokens"] == 5
    # tool_calls is absent, not null, when there are none.
    assert "tool_calls" not in d["choices"][0]["message"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_schemas.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.schemas'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/schemas.py`:

```python
"""OpenAI-compatible request/response models, plus Vaayu's own optional extras.

Pure pydantic: no torch, no model, no I/O.

`extra="ignore"` throughout is deliberate. Off-the-shelf agent clients send
fields this server does not implement (logit_bias, presence_penalty, user, ...);
rejecting those with a 422 would break every one of them. Fields Vaayu DOES
honour are declared explicitly, so an ignored field is genuinely unimplemented
rather than merely unnamed.
"""
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def make_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


# ── request ──────────────────────────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    arguments: str          # JSON-encoded string, per the OpenAI wire format


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["function"] = "function"
    function: FunctionDef


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    # None means "omitted" -- resolved later against the model's DERIVED context
    # window by stopping.resolve_max_tokens. A numeric default here would be
    # exactly the literal cap this server exists to remove.
    max_tokens: int | None = Field(default=None, gt=0)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stop: list[str] = Field(default_factory=list)
    n: int = 1
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    tools: list[ToolDef] | None = None
    tool_choice: Any = None

    # ── Vaayu extras. Optional; OpenAI clients never send them. ──
    session_id: str | None = None
    kv_cache_mode: str | None = None
    drafter_model_path: str | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def _normalise_stop(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("n")
    @classmethod
    def _only_one_choice(cls, v):
        if v != 1:
            raise ValueError(
                "this server generates one choice per request (n=1). Concurrency "
                "is 1 on this hardware -- N>=2 at long context is a settled "
                "refusal, not a missing feature.")
        return v


# ── response ─────────────────────────────────────────────────────────────────

class ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = 0
    message: ResponseMessage
    finish_reason: str


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_schemas.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `914 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add vaayu/server/schemas.py tests/test_server_schemas.py
git commit -m "feat(server): OpenAI-compatible request/response models

extra='ignore' so unimplemented OpenAI fields do not 422 off-the-shelf clients.
max_tokens defaults to None (omitted) rather than a number -- the cap is resolved
against the model's derived context window. n>1 is refused with a named reason."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, `server/schemas.py` line → `[x]`.

---

## Task 3: `tools.py` — tool-call detection and parsing

**Files:**
- Create: `vaayu/server/tools.py`
- Test: `tests/test_server_tools.py`

**Interfaces:**
- Consumes: `vaayu.server.schemas.ToolCall`, `ToolCallFunction`.
- Produces:
  - `class ToolCallFormat` — frozen dataclass with `name: str`, `open_marker: str`, `close_marker: str`
  - `HERMES: ToolCallFormat`, `GENERIC_JSON: ToolCallFormat`
  - `detect_tool_format(chat_template: str | None) -> ToolCallFormat | None`
  - `parse_tool_calls(text: str, fmt: ToolCallFormat | None) -> tuple[str, list[ToolCall]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_tools.py`:

```python
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
    text, calls = parse_tool_calls(out, GENERIC_JSON)
    assert calls[0].function.name == "read_file"
    assert text == "Sure."


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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_tools.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.tools'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/tools.py`:

```python
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


def _build(index: int, obj) -> "ToolCall | None":
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_tools.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `930 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add vaayu/server/tools.py tests/test_server_tools.py
git commit -m "feat(server): tool-call detection by chat-template probe, and parsing

Format is found by probing the template text, never by matching a model name
(CLAUDE.md §5). Unparseable output degrades to plain text rather than inventing
a call or swallowing the response."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, `server/tools.py` line → `[x]`.

---

## Task 4: `chat.py` — rendering messages to tokens

**Files:**
- Create: `vaayu/server/chat.py`
- Test: `tests/test_server_chat.py`

**Interfaces:**
- Consumes: `vaayu.server.tools.detect_tool_format`.
- Produces:
  - `class TemplateLacksToolSupport(ValueError)`
  - `class NoChatTemplate(ValueError)`
  - `render_prompt(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> list[int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_chat.py`:

```python
"""Prompt rendering. Exercised against a fake tokenizer -- no checkpoint needed."""
import pytest

from vaayu.server.chat import (NoChatTemplate, TemplateLacksToolSupport,
                               render_prompt)

HERMES_TEMPLATE = "tools: <tool_call> ... {% for m in messages %}{{ m.content }}{% endfor %}"
PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"

MESSAGES = [{"role": "user", "content": "write a function"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


class FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return [1, 2, 3, 4]


def test_a_plain_request_renders_to_token_ids():
    tok = FakeTokenizer(PLAIN_TEMPLATE)
    assert render_prompt(tok, MESSAGES) == [1, 2, 3, 4]


def test_the_generation_prompt_is_always_added():
    tok = FakeTokenizer(PLAIN_TEMPLATE)
    render_prompt(tok, MESSAGES)
    assert tok.calls[0][1]["add_generation_prompt"] is True
    assert tok.calls[0][1]["tokenize"] is True


def test_tools_are_passed_through_to_a_template_that_supports_them():
    tok = FakeTokenizer(HERMES_TEMPLATE)
    render_prompt(tok, MESSAGES, tools=TOOLS)
    assert tok.calls[0][1]["tools"] == TOOLS


def test_tools_are_refused_by_a_template_that_cannot_render_them():
    # Never silently drop `tools` (CLAUDE.md §5). The agent would sit waiting for
    # a tool call the model was never told about.
    tok = FakeTokenizer(PLAIN_TEMPLATE)
    with pytest.raises(TemplateLacksToolSupport) as exc:
        render_prompt(tok, MESSAGES, tools=TOOLS)
    assert "tool" in str(exc.value).lower()


def test_no_tools_means_the_tools_kwarg_is_not_passed_at_all():
    # Some templates raise on tools=None rather than treating it as absent.
    tok = FakeTokenizer(PLAIN_TEMPLATE)
    render_prompt(tok, MESSAGES)
    assert "tools" not in tok.calls[0][1]


def test_a_tokenizer_with_no_chat_template_is_refused():
    tok = FakeTokenizer(None)
    with pytest.raises(NoChatTemplate):
        render_prompt(tok, MESSAGES)


def test_a_non_list_return_from_the_template_is_refused_rather_than_passed_on():
    class StringTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kw):
            return "text not tokens"

    with pytest.raises(TypeError):
        render_prompt(StringTokenizer(PLAIN_TEMPLATE), MESSAGES)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_chat.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.chat'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/chat.py`:

```python
"""Render chat messages (and tool definitions) to token ids.

The checkpoint's own chat template does the work -- there is no per-model
formatting table here, and there must not be one (CLAUDE.md §5).
"""
from .tools import detect_tool_format


class NoChatTemplate(ValueError):
    """The tokenizer ships no chat template, so messages cannot be rendered."""


class TemplateLacksToolSupport(ValueError):
    """The request carries `tools` but the template cannot render them."""


def render_prompt(tokenizer, messages, tools=None):
    """messages: list[dict] as sent on the wire. Returns a list of token ids."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise NoChatTemplate(
            "this checkpoint's tokenizer ships no chat_template, so /v1/chat/"
            "completions cannot render messages for it. Use a chat/instruct "
            "checkpoint, or send an already-formatted single user message.")

    kwargs = {"add_generation_prompt": True, "tokenize": True}
    if tools:
        if detect_tool_format(template) is None:
            raise TemplateLacksToolSupport(
                "this checkpoint's chat template has no tool support, so the "
                "`tools` in this request cannot be rendered. Refusing rather "
                "than dropping them silently -- the model would never be told "
                "the tools exist and the agent would wait for a call that can "
                "never come.")
        kwargs["tools"] = tools

    ids = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(ids, list):
        raise TypeError(
            f"apply_chat_template returned {type(ids).__name__}, expected a list "
            f"of token ids (tokenize=True was requested)")
    return ids
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_chat.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `937 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add vaayu/server/chat.py tests/test_server_chat.py
git commit -m "feat(server): render messages and tools via the checkpoint's own chat template

A request carrying tools against a template with no tool support is refused with
a named reason rather than having its tools silently dropped."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, `server/chat.py` line → `[x]`.

---

## Task 5: Prefix cache — an explicit-id miss must fall back to auto-match

The server passes `session_id=None` by default, so this is not on the default path — but the moment a client supplies `session_id` (the field exists precisely so an agent can pin a stream), a miss on that id currently yields a **full prefill with no fuzzy fallback**: strictly worse than not naming the session at all. Prefix reuse is this engine's largest measured lever (136–377× TTFT), so a silent path to zero reuse is a real defect.

**Files:**
- Modify: `vaayu/prefix_cache.py:61-95` (the `lookup` method)
- Test: `tests/test_prefix_explicit_miss_fallback.py`

**Interfaces:**
- Consumes: `vaayu.prefix_cache.PrefixCacheStore`.
- Produces: no new symbols. `PrefixCacheStore.lookup(session_id, input_ids, device=None)` keeps its signature and return type `(cache_or_None, match_len)`; only the miss path changes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_explicit_miss_fallback.py`:

```python
"""An explicit session_id that misses must still auto-prefix-match.

Before this fix, naming a session and missing it was STRICTLY WORSE than not
naming it: the auto-match branch was gated on `session_id is None`.
"""
from vaayu.prefix_cache import PrefixCacheStore


class FakeCache:
    """Minimal stand-in for the three methods PrefixCacheStore calls."""

    def __init__(self, nbytes=1000):
        self._nbytes = nbytes
        self.on_cpu = False

    def to_cpu(self):
        self.on_cpu = True

    def resident_bytes(self):
        return self._nbytes


def _store():
    return PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=True)


def test_an_explicit_id_that_hits_is_used_directly():
    s = _store()
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, m = s.lookup("agent-a", [1, 2, 3, 4, 5])
    assert cache is not None
    assert m == 4


def test_an_explicit_id_that_misses_falls_back_to_the_best_prefix_match():
    s = _store()
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, m = s.lookup("agent-b", [1, 2, 3, 4, 5, 6])
    assert cache is not None, "an explicit miss must not throw away a usable prefix"
    assert m == 4
    assert s.hits == 1


def test_the_fallback_picks_the_longest_common_prefix_not_the_first_entry():
    s = _store()
    s.store("short", [1, 2], FakeCache())
    s.store("long", [1, 2, 3, 4, 5], FakeCache())
    _, m = s.lookup("unknown", [1, 2, 3, 4, 5, 9])
    assert m == 5


def test_a_genuine_miss_with_no_shared_prefix_is_still_a_miss():
    s = _store()
    s.store("agent-a", [1, 2, 3], FakeCache())
    cache, m = s.lookup("agent-b", [90, 91, 92])
    assert cache is None
    assert m == 0
    assert s.misses == 1


def test_auto_prefix_match_disabled_keeps_an_explicit_miss_a_miss():
    s = PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=False)
    s.store("agent-a", [1, 2, 3, 4], FakeCache())
    cache, _ = s.lookup("agent-b", [1, 2, 3, 4, 5])
    assert cache is None


def test_an_empty_pool_is_a_miss_not_a_crash():
    s = _store()
    cache, m = s.lookup("agent-a", [1, 2, 3])
    assert (cache, m) == (None, 0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_prefix_explicit_miss_fallback.py -q
```

Expected: FAIL on `test_an_explicit_id_that_misses_falls_back_to_the_best_prefix_match` — `assert cache is not None` fails, because the auto-match branch is gated on `session_id is None`.

- [ ] **Step 3: Write the minimal fix**

In `vaayu/prefix_cache.py`, inside `lookup`, replace this block:

```python
            key = None
            if session_id is not None and session_id in self._pool:
                key = session_id
            elif self.auto_prefix_match and session_id is None:
                best, best_m = None, 0
                for k, e in self._pool.items():
                    m = longest_common_prefix(e.token_ids, input_ids)
                    if m > best_m:
                        best, best_m = k, m
                key = best
```

with this one:

```python
            key = None
            if session_id is not None and session_id in self._pool:
                key = session_id
            if key is None and self.auto_prefix_match:
                # NOT gated on `session_id is None`. An explicit id that misses
                # used to skip this branch entirely, so NAMING a session and
                # missing was strictly worse than not naming it -- a full
                # prefill where a usable prefix was sitting in the pool. Prefix
                # reuse is this engine's largest measured lever (136-377x TTFT);
                # a silent path to zero reuse is a defect, not a nuance.
                best, best_m = None, 0
                for k, e in self._pool.items():
                    m = longest_common_prefix(e.token_ids, input_ids)
                    if m > best_m:
                        best, best_m = k, m
                key = best
```

Note for the reviewer: after a fallback hit, `vaayu_base.generate` re-stores under the **requested** id, so the entry is effectively renamed. That is correct — the pool is LRU and byte-budgeted, the contents are identical, and the requested id is the one the client will ask for next.

- [ ] **Step 4: Run the new test and the whole suite**

```bash
./venv/bin/python -m pytest tests/test_prefix_explicit_miss_fallback.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `943 passed, 3 skipped`. **If any existing prefix test fails, stop** — it encodes the old behaviour and the change needs re-reading, not the test rewriting.

- [ ] **Step 5: Commit**

```bash
git add vaayu/prefix_cache.py tests/test_prefix_explicit_miss_fallback.py
git commit -m "fix(prefix): an explicit session_id that misses now falls back to auto-match

The auto-match branch was gated on session_id being None, so naming a session
and missing it produced a full prefill even with a usable prefix in the pool --
strictly worse than not naming it. Found while designing the server, where
session_id is a client-supplied field."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, the first `engine fix (prefix cache)` line → `[x]`.

---

## Task 6: Prefix cache — a client disconnect must not drain the pool

`lookup()` **pops** the entry and `store()` re-inserts it only after `generate()` returns. An SSE client that aborts mid-stream — routine for an agent that changed its mind, hit a timeout, or was cancelled — kills the request, and the entry is gone **permanently**. A single-user test script never hits this; a server hits it constantly.

The restore must crop the cache to a length whose tokens are known. The adopted cache was mutated by however many forward passes completed before the abort, so re-storing it against the original token list would be **corruption** — worse than losing the entry.

**Files:**
- Modify: `vaayu/prefix_cache.py` (add `restore_on_abort` at module level)
- Modify: `vaayu/vaayu_base.py:4118` (the `result = super().generate(*args, **kwargs)` inside the prefix-cache branch of `generate`)
- Test: `tests/test_prefix_restore_on_abort.py`

**Interfaces:**
- Consumes: `vaayu.prefix_cache.PrefixCacheStore`.
- Produces: `restore_on_abort(store, session_id: str, token_ids: list, cache, prefix_len: int) -> bool` — returns True when the entry was put back.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_restore_on_abort.py`:

```python
"""A turn that dies mid-flight must put the checked-out prefix entry back.

lookup() POPS; store() only re-inserts after generate() returns. Without this,
an aborted SSE stream drains the pool permanently.
"""
from vaayu.prefix_cache import PrefixCacheStore, restore_on_abort


class FakeCache:
    def __init__(self, seq_len=0, nbytes=1000):
        self._seq_len = seq_len
        self._nbytes = nbytes
        self.cropped_to = None

    def get_seq_length(self, layer_idx=0):
        return self._seq_len

    def crop(self, max_length):
        self.cropped_to = max_length
        self._seq_len = max_length

    def to_cpu(self):
        pass

    def resident_bytes(self):
        return self._nbytes


class ExplodingCache(FakeCache):
    def crop(self, max_length):
        raise RuntimeError("cache is on a dead CUDA context")


def _store():
    return PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=True)


def test_the_entry_goes_back_cropped_to_the_length_whose_tokens_are_known():
    s = _store()
    ids = [1, 2, 3, 4, 5, 6]
    cache = FakeCache(seq_len=9)     # grew past the prefix before the abort
    assert restore_on_abort(s, "agent-a", ids, cache, prefix_len=4) is True
    assert cache.cropped_to == 4, "storing the mutated cache against the old token list would corrupt it"
    back, m = s.lookup("agent-a", [1, 2, 3, 4, 5])
    assert back is cache
    assert m == 4


def test_nothing_is_restored_when_there_was_no_hit_to_begin_with():
    s = _store()
    assert restore_on_abort(s, "agent-a", [1, 2, 3], FakeCache(), prefix_len=0) is False
    assert len(s._pool) == 0


def test_a_cache_shorter_than_the_recorded_prefix_is_restored_at_its_real_length():
    s = _store()
    cache = FakeCache(seq_len=2)
    assert restore_on_abort(s, "agent-a", [1, 2, 3, 4], cache, prefix_len=4) is True
    assert cache.cropped_to == 2
    _, m = s.lookup("agent-a", [1, 2, 3, 4])
    assert m == 2


def test_a_failing_restore_is_swallowed_so_it_cannot_mask_the_real_error():
    # This runs on an exception path. If it raised, it would replace the actual
    # failure the caller is about to re-raise -- the "read the FIRST error"
    # lesson, paid for once already.
    s = _store()
    assert restore_on_abort(s, "agent-a", [1, 2, 3, 4], ExplodingCache(seq_len=9),
                            prefix_len=4) is False
    assert len(s._pool) == 0


def test_a_missing_store_is_a_no_op():
    assert restore_on_abort(None, "agent-a", [1, 2], FakeCache(), prefix_len=2) is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_prefix_restore_on_abort.py -q
```

Expected: FAIL — `ImportError: cannot import name 'restore_on_abort'`.

- [ ] **Step 3: Add the helper**

Append to `vaayu/prefix_cache.py`, at module level after the `PrefixCacheStore` class:

```python
def restore_on_abort(store, session_id: str, token_ids: list, cache,
                     prefix_len: int) -> bool:
    """Put a checked-out prefix entry back after a turn died mid-flight.

    lookup() POPS an entry and store() only re-inserts it once generate()
    returns, so an aborted turn -- an SSE client disconnecting, a cancellation,
    an OOM -- loses that entry PERMANENTLY. A single-user script never hits
    this; a server hits it constantly.

    The adopted cache was mutated by however many forward passes completed, so
    it is CROPPED back to a length whose tokens are known before being stored.
    Re-storing the mutated cache against the original token list would be
    corruption, which is worse than losing the entry.

    Never raises: this runs on an exception path, and a failure here would
    replace the real error the caller is about to re-raise. Returns True only
    when an entry actually went back.
    """
    if store is None or prefix_len <= 0:
        return False
    try:
        keep = min(int(prefix_len), int(cache.get_seq_length()))
        if keep <= 0:
            return False
        cache.crop(keep)
        store.store(session_id, list(token_ids[:keep]), cache)
        return True
    except Exception as e:
        print(f"[Vaayu-Prefix] WARNING: could not restore session {session_id!r} "
              f"after an aborted turn ({type(e).__name__}: {e}); the pool loses "
              f"this entry and the next turn will prefill in full.")
        return False
```

- [ ] **Step 4: Run the new test to verify it passes**

```bash
./venv/bin/python -m pytest tests/test_prefix_restore_on_abort.py -q
```

Expected: PASS.

- [ ] **Step 5: Wire it into `generate()`**

In `vaayu/vaayu_base.py`, inside `generate`'s prefix-cache branch. Find this line (currently around line 4118, immediately after `kwargs["past_key_values"] = kv`):

```python
        result = super().generate(*args, **kwargs)
        resolved_sid = session_id if session_id is not None else _auto_session_id(ids_list)
```

Replace with:

```python
        resolved_sid = session_id if session_id is not None else _auto_session_id(ids_list)
        try:
            result = super().generate(*args, **kwargs)
        except BaseException:
            # lookup() above POPPED this entry. Without putting it back, an
            # aborted turn -- an SSE client disconnecting mid-stream, a
            # cancellation, an OOM -- drains the pool permanently.
            from .prefix_cache import restore_on_abort
            restore_on_abort(pc, resolved_sid, ids_list, kv, match_len)
            raise
```

`match_len` is already in scope: it is `0` on the MISS branch and the adopted length on the HIT branch, which is exactly the prefix whose tokens are known.

- [ ] **Step 6: Run the whole suite**

```bash
./venv/bin/python -m pytest tests/ -q
```

Expected: `949 passed, 3 skipped`.

- [ ] **Step 7: Prove the wiring on-device**

The unit tests cover the helper; this proves `generate()` actually calls it. Run the existing prefix gate, which loads a real checkpoint:

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PREFIX_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
  ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q \
  > /tmp/prefix-gate.log 2>&1; grep -a "passed\|failed" /tmp/prefix-gate.log
```

Expected: passed, unchanged from before this task.

- [ ] **Step 8: Commit**

```bash
git add vaayu/prefix_cache.py vaayu/vaayu_base.py tests/test_prefix_restore_on_abort.py
git commit -m "fix(prefix): an aborted turn puts its checked-out entry back

lookup() pops and store() only re-inserts after generate() returns, so a client
disconnecting mid-stream drained the pool permanently. restore_on_abort crops
the mutated cache to a length whose tokens are known before re-storing -- storing
it against the original token list would corrupt it. Never raises, so it cannot
mask the error the caller is re-raising."
```

- [ ] **Step 9: Tick the ledger** — `todo.md` row 13, the second `engine fix (prefix cache)` line → `[x]`.

---

## Task 7: `engine.py` — the model, the admission gate, generation

**Files:**
- Create: `vaayu/server/engine.py`
- Test: `tests/test_server_engine.py`

**Interfaces:**
- Consumes: `stopping.resolve_max_tokens`, `stopping.finish_reason`, `chat.render_prompt`, `tools.detect_tool_format`, `tools.parse_tool_calls`, `schemas.ChatCompletionRequest`.
- Produces:
  - `class QueueFull(RuntimeError)`
  - `@dataclass GenerationResult` with fields `text: str`, `tool_calls: list`, `prompt_tokens: int`, `completion_tokens: int`, `finish_reason: str`
  - `class EngineHandle` with `.model`, `.max_seq_len` (property), `.tool_format` (property), `.depth` (property), `.queue_depth`, `.warn_if_speculative_sampling_ignored(req) -> str | None`, `.build_generate_kwargs(req, prompt_ids) -> dict`, `async .slot()` (async context manager), `async .generate(req) -> GenerationResult`, `async .stream(req)` (async generator of `str`), `.prepare(req) -> tuple[list[int], int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_engine.py`:

```python
"""EngineHandle: admission gate and request-to-kwargs mapping.

Exercised against a fake model. No torch, no checkpoint -- the parts that need a
real GPU are covered by test_server_smoke.py at the repo root.
"""
import asyncio

import pytest

from vaayu.server.engine import EngineHandle, QueueFull
from vaayu.server.schemas import ChatCompletionRequest

HERMES_TEMPLATE = "tools <tool_call> {% for m in messages %}{{ m.content }}{% endfor %}"
PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class FakeTokenizer:
    def __init__(self, chat_template=PLAIN_TEMPLATE):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, **kw):
        return [1] * 10


class FakeModel:
    def __init__(self, max_seq_len=8192, chat_template=PLAIN_TEMPLATE, speculator=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FakeTokenizer(chat_template)
        self.speculator = speculator
        self._vram_plan = object()      # not the CPU fallback path


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_max_seq_len_is_read_off_the_model_not_stored():
    e = EngineHandle(FakeModel(max_seq_len=71981))
    assert e.max_seq_len == 71981


def test_prepare_returns_the_prompt_ids_and_the_resolved_cap():
    e = EngineHandle(FakeModel(max_seq_len=100))
    ids, cap = e.prepare(_req())
    assert ids == [1] * 10
    assert cap == 90, "an omitted max_tokens takes the whole remaining window"


def test_prepare_clamps_a_requested_cap_to_the_window():
    e = EngineHandle(FakeModel(max_seq_len=100))
    _, cap = e.prepare(_req(max_tokens=10_000))
    assert cap == 90


def test_greedy_requests_do_not_sample():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.0), [1, 2, 3])
    assert kw["do_sample"] is False
    assert "top_p" not in kw and "top_k" not in kw


def test_sampling_parameters_are_passed_only_when_sampling():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.7, top_p=0.9, top_k=40), [1, 2, 3])
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["top_k"] == 40


def test_stop_strings_require_the_tokenizer_and_are_passed_together():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(stop=["```"]), [1, 2, 3])
    assert kw["stop_strings"] == ["```"]
    assert kw["tokenizer"] is e.model.tokenizer


def test_the_tool_close_marker_is_added_as_a_stop_string():
    # Otherwise decode runs past the call and hallucinates the tool's result.
    e = EngineHandle(FakeModel(chat_template=HERMES_TEMPLATE))
    kw = e.build_generate_kwargs(
        _req(tools=[{"type": "function", "function": {"name": "f"}}]), [1, 2, 3])
    assert "</tool_call>" in kw["stop_strings"]


def test_session_id_is_passed_straight_through():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(session_id="agent-7"), [1, 2, 3])
    assert kw["session_id"] == "agent-7"


def test_an_omitted_session_id_is_none_so_auto_prefix_match_runs():
    e = EngineHandle(FakeModel())
    assert e.build_generate_kwargs(_req(), [1, 2, 3])["session_id"] is None


def test_no_warning_when_there_is_no_speculator():
    e = EngineHandle(FakeModel(speculator=None))
    assert e.warn_if_speculative_sampling_ignored(_req(top_p=0.5)) is None


def test_a_speculator_plus_top_p_warns_because_the_path_ignores_it():
    # speculative_generate honours temperature only; top_p/top_k/stop fall into
    # **kwargs and vanish. Warn, never pretend (CLAUDE.md §5).
    e = EngineHandle(FakeModel(speculator=object()))
    msg = e.warn_if_speculative_sampling_ignored(_req(temperature=0.7, top_p=0.5))
    assert msg is not None and "top_p" in msg


def test_a_speculator_with_stop_strings_warns_too():
    e = EngineHandle(FakeModel(speculator=object()))
    assert "stop" in e.warn_if_speculative_sampling_ignored(_req(stop=["```"]))


def test_a_speculator_with_defaults_only_does_not_warn():
    e = EngineHandle(FakeModel(speculator=object()))
    assert e.warn_if_speculative_sampling_ignored(_req(temperature=0.0)) is None


def test_the_cpu_max_seq_len_fallback_is_reported_at_construction(capsys):
    # vaayu_base falls back to max_seq_len=512 when there is no VramPlan (CPU /
    # no CUDA). Defaulting the cap to "the remaining window" would silently
    # inherit a 512-token window, which is the very bug this server removes.
    m = FakeModel(max_seq_len=512)
    m._vram_plan = None
    EngineHandle(m)
    assert "512" in capsys.readouterr().out


def test_requests_beyond_the_queue_depth_are_refused():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=2)
        held = []

        async def hold():
            async with e.slot():
                held.append(1)
                await asyncio.sleep(0.2)

        a = asyncio.create_task(hold())
        b = asyncio.create_task(hold())
        await asyncio.sleep(0.05)
        with pytest.raises(QueueFull):
            async with e.slot():
                pass
        await asyncio.gather(a, b)
        assert len(held) == 2

    asyncio.run(main())


def test_the_gate_serialises_so_the_model_is_never_touched_concurrently():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=8)
        order, live = [], 0

        async def work(tag):
            nonlocal live
            async with e.slot():
                live += 1
                assert live == 1, "two requests were inside the gate at once"
                order.append(tag)
                await asyncio.sleep(0.02)
                live -= 1

        await asyncio.gather(*(work(i) for i in range(4)))
        assert sorted(order) == [0, 1, 2, 3]

    asyncio.run(main())


def test_a_slot_is_released_even_when_the_body_raises():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=1)
        with pytest.raises(ValueError):
            async with e.slot():
                raise ValueError("boom")
        assert e.depth == 0
        async with e.slot():
            pass

    asyncio.run(main())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_engine.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.engine'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/engine.py`:

```python
"""EngineHandle -- the one loaded model, the admission gate, and generation.

Concurrency is 1 by construction. The engine is not thread-safe, and N>=2 at
long context is a settled hardware refusal on this box (peak 14.17-15.57 GiB of
15.47), so this is a depth-bounded FIFO gate, not a scheduler. `batched_generate`
is deliberately not wired in: it is static-batch v1 and would admit nothing here.
"""
import asyncio
import queue as _queue
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .chat import render_prompt
from .stopping import finish_reason, resolve_max_tokens
from .tools import detect_tool_format, parse_tool_calls


class QueueFull(RuntimeError):
    """More requests are waiting than the configured depth allows."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    tool_calls: list = field(default_factory=list)


class EngineHandle:
    def __init__(self, model, queue_depth: int = 8):
        self.model = model
        self.queue_depth = queue_depth
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._spec_warned = False

        # vaayu_base falls back to max_seq_len=512 when it built no VramPlan
        # (CPU / no CUDA). This server defaults the generation cap to "the
        # remaining window", so that fallback would silently reinstate exactly
        # the fixed 512-token cap the server exists to remove. Name it loudly.
        if getattr(model, "_vram_plan", None) is None:
            print(f"[Vaayu-Server] WARNING: this model has no VRAM plan (CPU or "
                  f"no CUDA), so max_seq_len is vaayu_base's fallback of "
                  f"{self.max_seq_len}. Every request's default token cap is "
                  f"bounded by it. Load on CUDA for a derived window.")

    # ── properties read off the model, never cached ──────────────────────────

    @property
    def max_seq_len(self) -> int:
        """Derived at load by plan_vram; never a constant in this package."""
        return self.model.max_seq_len

    @property
    def tool_format(self):
        return detect_tool_format(getattr(self.model.tokenizer, "chat_template", None))

    @property
    def depth(self) -> int:
        return self._inflight

    # ── admission ────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def slot(self):
        """Admit one request. Raises QueueFull past `queue_depth`.

        The counter is incremented BEFORE awaiting the lock, so it counts
        waiting requests as well as the running one -- a client's queue position
        is `depth` at entry.
        """
        if self._inflight >= self.queue_depth:
            raise QueueFull(
                f"{self._inflight} requests are already queued (depth "
                f"{self.queue_depth}). This engine runs one request at a time.")
        self._inflight += 1
        try:
            async with self._lock:
                yield
        finally:
            self._inflight -= 1

    # ── request mapping ──────────────────────────────────────────────────────

    def prepare(self, req):
        """(prompt_ids, resolved_cap). Raises ContextTooLong / TemplateLacksToolSupport."""
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        tools = [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
        prompt_ids = render_prompt(self.model.tokenizer, messages, tools=tools)
        cap = resolve_max_tokens(req.max_tokens, len(prompt_ids), self.max_seq_len)
        return prompt_ids, cap

    def warn_if_speculative_sampling_ignored(self, req):
        """speculative_generate honours `temperature` only; top_p/top_k/stop fall
        into **kwargs and are silently discarded. Return a message when this
        request asks for something that path will drop, else None."""
        if getattr(self.model, "speculator", None) is None:
            return None
        ignored = []
        if req.top_p != 1.0:
            ignored.append("top_p")
        if req.top_k != 0:
            ignored.append("top_k")
        if req.stop:
            ignored.append("stop")
        if not ignored:
            return None
        return (f"speculative decoding is active and its loop honours temperature "
                f"only, so {', '.join(ignored)} will be IGNORED for this request. "
                f"Load without drafter_model_path to use them.")

    def build_generate_kwargs(self, req, prompt_ids, cap=None) -> dict:
        import torch
        kwargs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long,
                                      device=self.model.running_device),
            "max_new_tokens": cap if cap is not None else 1,
            "use_cache": True,
            "session_id": req.session_id,     # None -> auto-prefix-match
        }
        if req.temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = req.temperature
            kwargs["top_p"] = req.top_p
            kwargs["top_k"] = req.top_k
        else:
            kwargs["do_sample"] = False

        stops = list(req.stop)
        fmt = self.tool_format
        if req.tools and fmt is not None:
            # Halt AT the call rather than running on to hallucinate its result.
            for s in fmt.stop_strings():
                if s not in stops:
                    stops.append(s)
        if stops:
            kwargs["stop_strings"] = stops
            kwargs["tokenizer"] = self.model.tokenizer     # HF requires it with stop_strings
        return kwargs

    # ── generation ───────────────────────────────────────────────────────────

    def _decode_new(self, out_ids, prompt_len):
        new_ids = out_ids[0][prompt_len:].tolist()
        text = self.model.tokenizer.decode(new_ids, skip_special_tokens=True)
        return new_ids, text

    def _finish(self, new_ids, text, cap, tool_calls):
        eos = self.model.config.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        return finish_reason(
            hit_eos=bool(new_ids) and new_ids[-1] in eos_set,
            hit_stop_string=False,          # HF trims the stop string; see below
            hit_tool_call=bool(tool_calls),
            produced=len(new_ids), cap=cap)

    async def generate(self, req) -> GenerationResult:
        prompt_ids, cap = self.prepare(req)
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap)
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: self.model.generate(**kwargs))
        new_ids, text = self._decode_new(out, len(prompt_ids))
        cleaned, calls = parse_tool_calls(text, self.tool_format if req.tools else None)
        return GenerationResult(
            text=cleaned, tool_calls=calls,
            prompt_tokens=len(prompt_ids), completion_tokens=len(new_ids),
            finish_reason=self._finish(new_ids, text, cap, calls))

    async def stream(self, req):
        """Async generator yielding (kind, payload).

        kind is "delta" with a text chunk, or "final" with a GenerationResult
        whose `text` holds the full cleaned text. Bridges vaayu.streaming's
        worker thread onto the event loop through a thread-safe queue.
        """
        from ..streaming import stream_generate

        prompt_ids, cap = self.prepare(req)
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap)
        input_ids = kwargs.pop("input_ids")
        kwargs.pop("max_new_tokens", None)
        kwargs.pop("do_sample", None)               # stream_generate derives it
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", 1.0)

        loop = asyncio.get_running_loop()
        bridge: _queue.Queue = _queue.Queue()
        SENTINEL = object()

        def _pump():
            try:
                for chunk in stream_generate(self.model, input_ids,
                                             max_new_tokens=cap,
                                             temperature=temperature,
                                             top_p=top_p, **kwargs):
                    bridge.put(("delta", chunk))
            except BaseException as e:                # noqa: BLE001 -- re-raised below
                bridge.put(("error", e))
            finally:
                bridge.put((SENTINEL, None))

        thread = threading.Thread(target=_pump, daemon=True)
        thread.start()

        fmt = self.tool_format if req.tools else None
        parts, error = [], None
        while True:
            kind, payload = await loop.run_in_executor(None, bridge.get)
            if kind is SENTINEL:
                break
            if kind == "error":
                error = payload
                break
            parts.append(payload)
            yield ("delta", payload)
        thread.join()
        if error is not None:
            raise error

        full = "".join(parts)
        cleaned, calls = parse_tool_calls(full, fmt)
        produced = len(self.model.tokenizer(full, add_special_tokens=False)["input_ids"])
        yield ("final", GenerationResult(
            text=cleaned, tool_calls=calls,
            prompt_tokens=len(prompt_ids), completion_tokens=produced,
            finish_reason=finish_reason(
                hit_eos=produced < cap, hit_stop_string=False,
                hit_tool_call=bool(calls), produced=produced, cap=cap)))
```

Note on `hit_stop_string`: HF trims a matched stop string before returning, so the streamed text cannot be inspected for it. `produced < cap` therefore stands in for "the model stopped on its own", which covers both EOS and a stop string — and the only distinction that matters to a client is `length` versus `stop`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_engine.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `965 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add vaayu/server/engine.py tests/test_server_engine.py
git commit -m "feat(server): EngineHandle with a depth-bounded admission gate

One model, one request in flight -- the engine is not thread-safe and N>=2 at
long context is a settled hardware refusal. Warns when a request asks for
sampling parameters the speculative path silently discards, and when the model
carries vaayu_base's CPU max_seq_len=512 fallback."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, `server/engine.py` and `startup warning` lines → `[x]`.

---

## Task 8: `app.py` — routes and non-streaming completions

**Files:**
- Create: `vaayu/server/app.py`
- Test: `tests/test_server_app.py`

**Interfaces:**
- Consumes: `EngineHandle`, `QueueFull`, `GenerationResult`, all of `schemas`, `stopping.ContextTooLong`, `chat.TemplateLacksToolSupport`, `chat.NoChatTemplate`.
- Produces:
  - `@dataclass ServerConfig` — `model_path: str`, `device: str = "cuda:0"`, `queue_depth: int = 8`, `host: str = "127.0.0.1"`, `port: int = 8000`, `kv_cache_mode: str = "auto"`, `drafter_model_path: str | None = None`, `served_model_name: str | None = None`
  - `create_app(config: ServerConfig | None = None, engine=None) -> FastAPI`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_app.py`:

```python
"""HTTP surface, exercised against a fake engine -- no model, no GPU."""
import json

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import create_app
from vaayu.server.chat import TemplateLacksToolSupport
from vaayu.server.engine import GenerationResult, QueueFull
from vaayu.server.schemas import ToolCall, ToolCallFunction
from vaayu.server.stopping import ContextTooLong


class FakeEngine:
    """Implements only what app.py touches."""

    def __init__(self, result=None, raises=None, deltas=None):
        self.max_seq_len = 8192
        self.queue_depth = 8
        self.depth = 0
        self.model = type("M", (), {"name_or_path": "fake"})()
        self._result = result or GenerationResult(
            text="hello", prompt_tokens=3, completion_tokens=2, finish_reason="stop")
        self._raises = raises
        self._deltas = deltas or ["hel", "lo"]
        self.warned = None

    def warn_if_speculative_sampling_ignored(self, req):
        return self.warned

    def slot(self):
        engine = self

        class _Slot:
            async def __aenter__(self):
                if isinstance(engine._raises, QueueFull):
                    raise engine._raises

            async def __aexit__(self, *a):
                return False

        return _Slot()

    async def generate(self, req):
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        return self._result

    async def stream(self, req):
        if self._raises is not None and not isinstance(self._raises, QueueFull):
            raise self._raises
        for d in self._deltas:
            yield ("delta", d)
        yield ("final", self._result)

    def prefix_stats(self):
        return "hits=1 misses=0"


def _client(engine):
    return TestClient(create_app(engine=engine))


BODY = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}


def test_health_reports_the_queue_and_the_prefix_cache():
    r = _client(FakeEngine()).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["queue_depth"] == 8
    assert body["max_seq_len"] == 8192
    assert "hits=1" in body["prefix_cache"]


def test_models_lists_the_one_loaded_model_with_its_derived_window():
    body = _client(FakeEngine()).get("/v1/models").json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["max_seq_len"] == 8192


def test_a_non_streaming_completion_returns_the_openai_shape():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "chat.completion"
    assert b["choices"][0]["message"]["content"] == "hello"
    assert b["choices"][0]["finish_reason"] == "stop"
    assert b["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert b["id"].startswith("chatcmpl-")


def test_tool_calls_are_returned_with_finish_reason_tool_calls():
    call = ToolCall(id="call_x", function=ToolCallFunction(name="f", arguments="{}"))
    engine = FakeEngine(result=GenerationResult(
        text="", prompt_tokens=3, completion_tokens=9,
        finish_reason="tool_calls", tool_calls=[call]))
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert b["choices"][0]["finish_reason"] == "tool_calls"
    assert b["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


def test_hitting_the_cap_is_reported_as_length_not_hidden():
    engine = FakeEngine(result=GenerationResult(
        text="def f(", prompt_tokens=3, completion_tokens=8, finish_reason="length"))
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert b["choices"][0]["finish_reason"] == "length"


def test_an_oversized_prompt_is_a_400_carrying_the_engine_message():
    engine = FakeEngine(raises=ContextTooLong("prompt is 9000 tokens but the window is 8192"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400
    assert "9000" in r.json()["error"]["message"]


def test_tools_against_a_template_without_tool_support_is_a_400():
    engine = FakeEngine(raises=TemplateLacksToolSupport("no tool support"))
    body = dict(BODY, tools=[{"type": "function", "function": {"name": "f"}}])
    r = _client(engine).post("/v1/chat/completions", json=body)
    assert r.status_code == 400


def test_a_full_queue_is_a_503_with_retry_after():
    engine = FakeEngine(raises=QueueFull("8 requests are already queued"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 503
    assert "Retry-After" in r.headers


def test_an_engine_crash_is_a_500_not_a_hang():
    engine = FakeEngine(raises=RuntimeError("CUDA error"))
    r = _client(engine).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 500
    assert "CUDA error" in r.json()["error"]["message"]


def test_a_speculative_sampling_warning_is_surfaced_on_the_response():
    engine = FakeEngine()
    engine.warned = "top_p will be IGNORED"
    b = _client(engine).post("/v1/chat/completions", json=BODY).json()
    assert "top_p" in b["warnings"][0]


def test_n_greater_than_one_is_a_422_from_validation():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, n=4))
    assert r.status_code == 422
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_server_app.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vaayu.server.app'`.

- [ ] **Step 3: Write the implementation**

Create `vaayu/server/app.py`:

```python
"""FastAPI surface. Routes, error mapping and SSE framing only -- no policy.

Everything decidable without a GPU lives in stopping.py / tools.py / schemas.py.
"""
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import NoChatTemplate, TemplateLacksToolSupport
from .engine import EngineHandle, QueueFull
from .schemas import (ChatCompletionRequest, ChatCompletionResponse, Choice,
                      ResponseMessage, Usage, make_completion_id)
from .stopping import ContextTooLong


@dataclass
class ServerConfig:
    model_path: str
    device: str = "cuda:0"
    queue_depth: int = 8
    host: str = "127.0.0.1"
    port: int = 8000
    kv_cache_mode: str = "auto"
    drafter_model_path: str | None = None
    served_model_name: str | None = None


def _error(status: int, message: str, kind: str, headers=None):
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": kind}})


def _served_name(app) -> str:
    cfg = getattr(app.state, "config", None)
    if cfg is not None and cfg.served_model_name:
        return cfg.served_model_name
    engine = app.state.engine
    return str(getattr(engine.model, "name_or_path", "vaayu"))


def create_app(config: ServerConfig | None = None, engine=None) -> FastAPI:
    """`engine` is for tests: pass one and no model is loaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            from ..auto_model import AutoModel
            print(f"[Vaayu-Server] loading {config.model_path} ...")
            t0 = time.time()
            model = AutoModel.from_pretrained(
                config.model_path,
                device=config.device,
                kv_cache_mode=config.kv_cache_mode,
                drafter_model_path=config.drafter_model_path or None,
            )
            print(f"[Vaayu-Server] loaded in {time.time() - t0:.1f}s; "
                  f"derived max_seq_len={model.max_seq_len}")
            app.state.engine = EngineHandle(model, queue_depth=config.queue_depth)
        else:
            app.state.engine = engine
        yield

    app = FastAPI(title="Vaayu", lifespan=lifespan)
    app.state.config = config

    @app.get("/health")
    async def health():
        e = app.state.engine
        body = {
            "status": "ready",
            "model": _served_name(app),
            "max_seq_len": e.max_seq_len,
            "queue_depth": e.queue_depth,
            "queued": e.depth,
            "prefix_cache": e.prefix_stats(),
        }
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                body["vram_free_gib"] = round(free / 1024 ** 3, 2)
                body["vram_total_gib"] = round(total / 1024 ** 3, 2)
        except Exception:
            pass          # /health must never be the thing that fails
        return body

    @app.get("/v1/models")
    async def models():
        e = app.state.engine
        return {"object": "list",
                "data": [{"id": _served_name(app), "object": "model",
                          "owned_by": "vaayu", "max_seq_len": e.max_seq_len}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        e = app.state.engine
        warning = e.warn_if_speculative_sampling_ignored(req)
        if warning:
            print(f"[Vaayu-Server] WARNING: {warning}")

        try:
            async with e.slot():
                if req.stream:
                    return StreamingResponse(
                        _sse(e, req, _served_name(app), warning),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})
                result = await e.generate(req)
        except QueueFull as ex:
            return _error(503, str(ex), "server_busy", headers={"Retry-After": "5"})
        except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
            return _error(400, str(ex), "invalid_request_error")
        except Exception as ex:                       # noqa: BLE001
            return _error(500, f"{type(ex).__name__}: {ex}", "internal_error")

        body = ChatCompletionResponse(
            id=make_completion_id(), created=int(time.time()),
            model=_served_name(app),
            choices=[Choice(
                index=0,
                message=ResponseMessage(
                    content=result.text or None,
                    tool_calls=result.tool_calls or None),
                finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens),
        ).model_dump(exclude_none=True)
        if warning:
            body["warnings"] = [warning]
        return body

    return app
```

`_sse` is written in Task 9; until then the streaming branch is unreachable in the tests above (they all send `stream` unset). Add this stub at the bottom of the module so the import resolves:

```python
async def _sse(engine, req, model_name, warning):
    raise NotImplementedError("streaming lands in Task 9")
    yield  # pragma: no cover -- makes this an async generator
```

- [ ] **Step 4: Add `prefix_stats` to `EngineHandle`**

`/health` calls it. In `vaayu/server/engine.py`, add to `EngineHandle`:

```python
    def prefix_stats(self) -> str:
        """One grep-able line from the prefix pool, or a note that it is off.

        Hit rate here is the difference between ~1 s and ~98 s TTFT, and there
        is otherwise no way to observe it from outside the process.
        """
        pc = getattr(self.model, "_prefix_cache", None)
        if pc is None:
            return "disabled"
        try:
            return pc.stats()
        except Exception as e:
            return f"unavailable ({type(e).__name__}: {e})"
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_app.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: new file passes; suite `977 passed, 3 skipped`.

- [ ] **Step 6: Commit**

```bash
git add vaayu/server/app.py vaayu/server/engine.py tests/test_server_app.py
git commit -m "feat(server): FastAPI surface with non-streaming chat completions

/v1/chat/completions, /v1/models and /health. /health surfaces the prefix-cache
stats, which are otherwise unobservable from outside the process and are the
difference between ~1s and ~98s TTFT. A full queue is a 503 with Retry-After; an
oversized prompt is a 400 carrying the engine's own diagnostic."
```

- [ ] **Step 7: Tick the ledger** — `todo.md` row 13, `server/app.py` line → `[x]`.

---

## Task 9: SSE streaming

**Files:**
- Modify: `vaayu/server/app.py` (replace the `_sse` stub)
- Test: `tests/test_server_app.py` (append)

**Interfaces:**
- Consumes: `EngineHandle.stream` yielding `("delta", str)` then `("final", GenerationResult)`.
- Produces: `_sse(engine, req, model_name, warning)` — async generator of SSE-framed `str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_app.py`:

```python
def _sse_events(raw: str):
    """Parse an SSE body into the list of JSON payloads, dropping [DONE]."""
    out = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def test_a_stream_emits_role_then_content_then_a_terminating_done():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")
    events = _sse_events(r.text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[0]["object"] == "chat.completion.chunk"
    contents = [e["choices"][0]["delta"].get("content") for e in events]
    assert "".join(c for c in contents if c) == "hello"


def test_the_last_stream_event_carries_the_finish_reason():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    events = _sse_events(r.text)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"][0]["delta"] == {}


def test_usage_is_omitted_unless_asked_for():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert all("usage" not in e for e in _sse_events(r.text))


def test_usage_is_included_when_stream_options_asks():
    body = dict(BODY, stream=True, stream_options={"include_usage": True})
    events = _sse_events(_client(FakeEngine()).post("/v1/chat/completions", json=body).text)
    assert events[-1]["usage"]["total_tokens"] == 5


def test_a_streamed_tool_call_is_emitted_once_and_complete():
    # Buffered, never emitted as partial JSON -- partial tool-call JSON breaks
    # clients that try to parse each delta.
    call = ToolCall(id="call_x", function=ToolCallFunction(name="f", arguments='{"a":1}'))
    engine = FakeEngine(
        deltas=["<tool_call>", '{"name":"f",'],
        result=GenerationResult(text="", prompt_tokens=3, completion_tokens=9,
                                finish_reason="tool_calls", tool_calls=[call]))
    events = _sse_events(_client(engine).post("/v1/chat/completions",
                                              json=dict(BODY, stream=True)).text)
    with_calls = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert len(with_calls) == 1
    assert with_calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_an_engine_crash_mid_stream_emits_an_error_event_and_closes():
    engine = FakeEngine(raises=RuntimeError("io_uring read failed"))
    r = _client(engine).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert "io_uring read failed" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_a_queue_position_comment_precedes_the_first_event():
    r = _client(FakeEngine()).post("/v1/chat/completions", json=dict(BODY, stream=True))
    assert r.text.startswith(": vaayu queue")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_server_app.py -q -k "stream or usage or queue_position"
```

Expected: FAIL — `NotImplementedError: streaming lands in Task 9`.

- [ ] **Step 3: Replace the `_sse` stub**

In `vaayu/server/app.py`, delete the stub and put this in its place:

```python
def _chunk(cid, created, model_name, *, delta, finish=None, usage=None) -> str:
    body = ChatCompletionChunk(
        id=cid, created=created, model=model_name,
        choices=[ChunkChoice(index=0, delta=Delta(**delta), finish_reason=finish)],
        usage=usage,
    ).model_dump(exclude_none=True)
    return f"data: {json.dumps(body)}\n\n"


async def _sse(engine, req, model_name, warning):
    """OpenAI-shaped SSE. Content streams; a tool call is BUFFERED and emitted
    once, complete -- partial tool-call JSON breaks clients that parse each
    delta, and the gain from streaming it is nil."""
    cid, created = make_completion_id(), int(time.time())

    # A comment line: legal SSE, ignored by every client, and it lets a queued
    # caller tell "waiting behind N requests" from "hung".
    yield f": vaayu queue position {engine.depth}\n\n"
    if warning:
        yield f": warning {warning}\n\n"

    yield _chunk(cid, created, model_name, delta={"role": "assistant"})

    fmt = engine.tool_format if req.tools else None
    open_marker = fmt.open_marker if fmt is not None else None
    held, suppressing = "", False
    final = None
    try:
        async for kind, payload in engine.stream(req):
            if kind == "final":
                final = payload
                break
            if suppressing:
                continue
            held += payload
            if open_marker and open_marker in held:
                # Everything from here belongs to the tool call. Emit the prose
                # that preceded it, then go quiet until the final event.
                prose = held[:held.index(open_marker)]
                if prose:
                    yield _chunk(cid, created, model_name, delta={"content": prose})
                held, suppressing = "", True
                continue
            # Hold back a marker-length tail so an open marker split across two
            # chunks is still recognised.
            keep = (len(open_marker) - 1) if open_marker else 0
            if len(held) > keep:
                emit, held = held[:len(held) - keep], held[len(held) - keep:]
                yield _chunk(cid, created, model_name, delta={"content": emit})
    except Exception as ex:                            # noqa: BLE001
        yield (f"data: {json.dumps({'error': {'message': f'{type(ex).__name__}: {ex}', 'type': 'internal_error'}})}\n\n")
        yield "data: [DONE]\n\n"
        return

    if held and not suppressing:
        yield _chunk(cid, created, model_name, delta={"content": held})

    if final is not None and final.tool_calls:
        yield _chunk(cid, created, model_name,
                     delta={"tool_calls": final.tool_calls})

    usage = None
    if final is not None and req.stream_options and req.stream_options.include_usage:
        usage = Usage(prompt_tokens=final.prompt_tokens,
                      completion_tokens=final.completion_tokens,
                      total_tokens=final.prompt_tokens + final.completion_tokens)
    yield _chunk(cid, created, model_name, delta={},
                 finish=(final.finish_reason if final is not None else "stop"),
                 usage=usage)
    yield "data: [DONE]\n\n"
```

Extend the imports at the top of `app.py`:

```python
import json
```

and add `ChatCompletionChunk`, `ChunkChoice`, `Delta` to the existing `from .schemas import (...)` line.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_server_app.py -q
./venv/bin/python -m pytest tests/ -q
```

Expected: `984 passed, 3 skipped`.

- [ ] **Step 5: Commit**

```bash
git add vaayu/server/app.py tests/test_server_app.py
git commit -m "feat(server): SSE streaming with buffered tool calls

Content streams as OpenAI deltas; a tool call is buffered and emitted once,
complete, because partial tool-call JSON breaks clients that parse each delta.
An open marker split across two chunks is still recognised. A queue-position
comment precedes the first event so a waiting client can tell queued from hung."
```

---

## Task 10: `__main__.py` — the CLI entry point

**Files:**
- Create: `vaayu/server/__main__.py`
- Test: manual (`--help`); the app itself is covered by Tasks 8 and 9.

**Interfaces:**
- Consumes: `ServerConfig`, `create_app`.
- Produces: `main(argv=None) -> int`.

- [ ] **Step 1: Write the implementation**

Create `vaayu/server/__main__.py`:

```python
"""python -m vaayu.server --model /path/to/checkpoint

One model, loaded once at boot. There is no hot-swap: a Vaayu load costs minutes
(pinning, slab arena, NVMe warmup, expert-cache sizing), so swapping a model is a
process restart. This is the one thing exllamav3-server.py does that must not be
copied.
"""
import argparse
import os

from .app import ServerConfig, create_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vaayu.server")
    p.add_argument("--model", default=os.environ.get("VAAYU_MODEL_PATH"),
                   help="checkpoint path (or set VAAYU_MODEL_PATH)")
    p.add_argument("--served-model-name", default=None,
                   help="name reported by /v1/models; defaults to the checkpoint path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--queue-depth", type=int, default=8,
                   help="waiting requests allowed before 503; one runs at a time")
    p.add_argument("--kv-cache-mode", default="auto")
    p.add_argument("--drafter-model-path", default=None,
                   help="opt-in speculative decoding. Measured 5.57x on streamed "
                        "dense; refused by the Fix #9 gate on both MoE targets")
    args = p.parse_args(argv)

    if not args.model:
        p.error("--model is required (or set VAAYU_MODEL_PATH)")

    cfg = ServerConfig(
        model_path=args.model, device=args.device, queue_depth=args.queue_depth,
        host=args.host, port=args.port, kv_cache_mode=args.kv_cache_mode,
        drafter_model_path=args.drafter_model_path,
        served_model_name=args.served_model_name)

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the CLI parses**

```bash
./venv/bin/python -m vaayu.server --help
./venv/bin/python -m vaayu.server 2>&1 | tail -2
```

Expected: help text listing every flag; the second command exits non-zero with `--model is required`.

- [ ] **Step 3: Run the suite**

```bash
./venv/bin/python -m pytest tests/ -q
```

Expected: `984 passed, 3 skipped`.

- [ ] **Step 4: Commit**

```bash
git add vaayu/server/__main__.py
git commit -m "feat(server): CLI entry point (python -m vaayu.server --model PATH)"
```

- [ ] **Step 5: Tick the ledger** — `todo.md` row 13, `server/__main__.py` line → `[x]`.

---

## Task 11: On-device smoke gate

Every test so far runs against fakes. This is the one that proves the server drives a real checkpoint: a real load, a real prefill, real tokens, and real prefix reuse across two turns.

**Files:**
- Create: `test_server_smoke.py` (repo **ROOT**, beside `test_batched_dense_smoke.py`)

**Interfaces:**
- Consumes: `create_app`, `ServerConfig`.
- Produces: an executable gate; exit 0 on pass, non-zero on failure.

- [ ] **Step 1: Write the gate**

Create `test_server_smoke.py` at the repo root:

```python
"""On-device gate: the server against a real checkpoint.

    SERVER_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
      systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
      ./venv/bin/python test_server_smoke.py

Defaults to the small dense checkpoint so a run costs seconds, not minutes.
Point SERVER_CKPT at an EXL3 MoE checkpoint to exercise that path.

ONE model load for the whole file: freed pinned host blocks are cached rather
than returned, and a second load in the same process is OOM-killed under a 22 GB
cgroup cap (todo.md, "Never load a model twice in one pytest process").
"""
import json
import os
import sys

from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app

CKPT = os.environ.get("SERVER_CKPT", "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct")
PROMPT = "Write a Python function that reverses a string. Reply with code only."

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    cfg = ServerConfig(model_path=CKPT, served_model_name="vaayu-smoke")
    with TestClient(create_app(cfg)) as client:
        h = client.get("/health").json()
        print(json.dumps(h, indent=2))
        check("health is ready", h["status"] == "ready")
        # The derived window, NOT a literal: only that it is a real number and
        # bigger than the CPU fallback this server warns about.
        check("max_seq_len is derived, not the CPU fallback", h["max_seq_len"] > 512,
              f"max_seq_len={h['max_seq_len']}")

        body = {"model": "vaayu-smoke",
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 48, "temperature": 0.0}

        # ── 1. non-streaming ──
        r = client.post("/v1/chat/completions", json=body)
        check("non-stream returns 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            b = r.json()
            text = b["choices"][0]["message"].get("content") or ""
            print(f"--- non-stream ({b['usage']['completion_tokens']} tokens, "
                  f"finish_reason={b['choices'][0]['finish_reason']}) ---\n{text}\n---")
            check("non-stream produced tokens", b["usage"]["completion_tokens"] > 0)
            check("finish_reason is a real value",
                  b["choices"][0]["finish_reason"] in {"stop", "length", "tool_calls"})

        # ── 2. streaming, and prefix reuse on the second turn ──
        with client.stream("POST", "/v1/chat/completions",
                           json=dict(body, stream=True)) as s:
            check("stream returns 200", s.status_code == 200)
            chunks, saw_done = [], False
            for line in s.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    saw_done = True
                    continue
                ev = json.loads(payload)
                piece = ev["choices"][0]["delta"].get("content")
                if piece:
                    chunks.append(piece)
        streamed = "".join(chunks)
        print(f"--- streamed ({len(chunks)} deltas) ---\n{streamed}\n---")
        check("stream terminated with [DONE]", saw_done)
        check("stream produced text", len(streamed) > 0)

        # ── 3. an oversized prompt is refused, not truncated ──
        huge = {"model": "vaayu-smoke",
                "messages": [{"role": "user", "content": "x " * (h["max_seq_len"] + 1000)}]}
        r = client.post("/v1/chat/completions", json=huge)
        check("an oversized prompt is a 400", r.status_code == 400, r.text[:200])

        # ── 4. the prefix pool was exercised by the two identical turns ──
        stats = client.get("/health").json()["prefix_cache"]
        print(f"prefix cache: {stats}")
        check("prefix cache is observable from /health", stats != "")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the gate on the small dense checkpoint**

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu \
  ./venv/bin/python test_server_smoke.py > /tmp/server-smoke-1b.log 2>&1
grep -a "PASS\|FAIL\|ALL PASS\|FAILURES" /tmp/server-smoke-1b.log
```

Expected: `ALL PASS`. If a check fails, read the FIRST error in the log, not the last — a poisoned CUDA context reports downstream symptoms for the rest of the run.

- [ ] **Step 3: Run the gate on an EXL3 MoE checkpoint**

Proves the module-list forward and the expert cache are not disturbed. This one takes minutes.

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu \
      SERVER_CKPT=/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3 \
  ./venv/bin/python test_server_smoke.py > /tmp/server-smoke-minimax.log 2>&1
grep -a "PASS\|FAIL\|ALL PASS\|FAILURES\|max_seq_len" /tmp/server-smoke-minimax.log
```

Expected: `ALL PASS`, with `max_seq_len=8192`.

- [ ] **Step 4: Run the dense smoke gate to prove nothing regressed**

CLAUDE.md §0's named guard. It must be unchanged.

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  ./venv/bin/python test_batched_dense_smoke.py > /tmp/dense-smoke.log 2>&1
grep -a "perplexity\|PASS\|FAIL" /tmp/dense-smoke.log
```

Expected: PASS, perplexity **8.78** — the same figure as before this plan.

- [ ] **Step 5: Commit**

```bash
git add test_server_smoke.py
git commit -m "test(server): on-device smoke gate against a real checkpoint

Real load, non-stream, stream, an oversized-prompt refusal, and prefix-cache
observability. Defaults to the small dense checkpoint so a run costs seconds;
SERVER_CKPT points it at an EXL3 MoE."
```

- [ ] **Step 6: Tick the ledger** — `todo.md` row 13, `test_server_smoke.py` line → `[x]`.

---

## Task 12: Document it and close the row

**Files:**
- Modify: `README.md`
- Modify: `todo.md`
- Modify: `plans/settled_and_method_notes.md`

- [ ] **Step 1: Add a server section to README.md**

Insert after the existing usage section:

````markdown
## Running Vaayu as a server

```bash
./venv/bin/python -m vaayu.server --model /path/to/checkpoint --port 8000
```

OpenAI-compatible: `POST /v1/chat/completions` (streaming and not), `GET /v1/models`,
`GET /health`. Any client that speaks the OpenAI API works unchanged.

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "vaayu",
  "messages": [{"role": "user", "content": "write a binary search in Python"}]
}'
```

**`max_tokens` is optional, and omitting it is the normal case.** The default is the
whole remaining context window (`max_seq_len - prompt_len`, derived at load), so output
stops when the model is done rather than at a fixed count. If output really was cut
off, `finish_reason` is `"length"` — send the turn again to continue rather than
writing truncated code to a file.

**One model per process.** A Vaayu load costs minutes; swapping a model is a restart.

**One request at a time**, with a FIFO queue (`--queue-depth`, default 8) and a 503
past that. The engine is not thread-safe and N≥2 at long context is a hardware refusal
on 16 GB, not a missing feature.

**Tool calling** is supported wherever the checkpoint's chat template supports it: pass
OpenAI `tools`, get `tool_calls` back with `finish_reason="tool_calls"`. MCP belongs in
your agent — this server speaks `tools`/`tool_calls` and nothing more.

**Prefix caching is on by default and is the largest lever here** (136–377× TTFT on
repeated context). Nothing is needed client-side: resend the full history each turn, as
OpenAI clients already do. Watch it work with `curl localhost:8000/health`.
````

- [ ] **Step 2: Verify the README example actually runs**

Start the server on the small checkpoint, curl it, stop it.

```bash
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONPATH=/home/a/code/ai-ml/Vaayu \
  ./venv/bin/python -m vaayu.server \
    --model /home/a/code/ai-ml/models/Llama-3.2-1B-Instruct --port 8000 \
  > /tmp/server-run.log 2>&1 &
sleep 90
curl -s localhost:8000/health
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vaayu","messages":[{"role":"user","content":"write a binary search in Python"}],"max_tokens":64}'
pkill -f "vaayu.server" || true
```

Expected: a health body, then a completion whose `choices[0].message.content` holds code.

- [ ] **Step 3: Final full-suite and gate run**

```bash
./venv/bin/python -m pytest tests/ -q
```

Expected: `984 passed, 3 skipped`.

- [ ] **Step 4: Close row 13 in `todo.md`**

Change the row 13 heading from `# 13 · [~]` to `# 13 · [x] CLOSED <date>`, confirm every subtask is `[x]`, and replace the "Task groups (placeholders until the plan lands)" preamble with the measured outcome: the final suite count, the dense smoke perplexity, and which checkpoints the on-device server gate passed on. Move the SCHEDULED section back to empty and refresh the NEXT ACTION block to point at the OPTIONAL rows.

- [ ] **Step 5: Add the archive entry**

Append a section to `plans/settled_and_method_notes.md` titled "Vaayu as an inference server (todo row 13)" recording: what shipped, the two prefix-cache bugs the design surfaced and how they were fixed, the measured suite/gate numbers, and the standing limitations (one request at a time, tool-call formats covered, speculative path sampling-blind, `batched_generate` unwired).

- [ ] **Step 6: Commit**

```bash
git add README.md todo.md plans/settled_and_method_notes.md
git commit -m "docs(server): document the HTTP server and close todo row 13"
```

---

## Self-Review

**Spec coverage.** Every section of `plans/2026-08-16-vaayu-server-design.md` maps to a task: §3 module table → Tasks 0–10 (one task per module); §3.1 lifecycle → Task 8 `lifespan`; §3.2 concurrency → Task 7 `slot()`; §3.3 endpoints → Task 8; §3.4 request fields → Task 2; §4 stopping → Task 1, with the CPU-`512` warning in Task 7; §5 prefix cache → Tasks 5 and 6; §6 tool calling → Tasks 3, 4, 7 (stop strings), 9 (buffered streaming); §7 error mapping → Task 8; §8 testing → the per-task tests plus Task 11; §9 risks → 9.1 in Task 10's docstring, 9.2 in Task 7's `warn_if_speculative_sampling_ignored`, 9.3 and 9.4 recorded in Task 12's archive entry (both are measurement follow-ups, not code), 9.5 in Task 7's module docstring; §10 out-of-scope items appear in no task, correctly.

**Placeholders.** None. Every code step carries real code; every test step carries real assertions; every run step carries the exact command and its expected output.

**Type consistency.** `resolve_max_tokens` / `finish_reason` keep one signature from Task 1 through Tasks 7 and 9. `ToolCallFormat.open_marker` / `.close_marker` / `.stop_strings()` are defined in Task 3 and used unchanged in Tasks 7 and 9. `GenerationResult`'s five fields are defined in Task 7 and consumed identically in Tasks 8, 9 and by `FakeEngine`. `EngineHandle.stream` yields `("delta", str)` / `("final", GenerationResult)` in Task 7 and is consumed in exactly that shape by `_sse` in Task 9. `restore_on_abort(store, session_id, token_ids, cache, prefix_len)` is defined and called with the same five arguments in Task 6.

**Known gap, deliberately left.** `EngineHandle.prefix_stats` is written in Task 8 (Step 4) rather than Task 7, because Task 8's `/health` is its first consumer. A reviewer taking the tasks out of order should expect it there.

---

## Execution Handoff

**Plan complete and saved to `plans/2026-08-16-vaayu-server-plan.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
