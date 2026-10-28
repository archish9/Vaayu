# Vaayu tool calling — how it works now (for Rudra)

**Audience:** the Rudra team.
**Status:** the tool-calling blocker from `vaayu-tool-calling.md` is **FIXED on the Vaayu side.**
**Date:** 2026-10-28. Verified on-device against MiniMax-M2.7 through the live server.

This document is self-contained: what changed, the exact request/response shapes Rudra now gets,
and whether Rudra needs to change anything (short answer: **no** for MiniMax).

---

## 1. What was wrong, in one line

Vaayu returned the model's tool-call intent as raw `<minimax:tool_call>` XML inside
`message.content`, with `finish_reason: "stop"` and no `tool_calls`. Vaayu now translates the
model's native tokens into the OpenAI `tool_calls` shape before responding.

## 2. What Vaayu does now (server-side, no Rudra change)

Vaayu discovers each model's native tool-call syntax **at load**, by rendering a sentinel call
through that checkpoint's own chat template and reading the delimiters back — so it is not specific
to MiniMax. It then parses the model's generated tokens into OpenAI `tool_calls`, strips them out of
`content`, and sets `finish_reason: "tool_calls"`. This covers every native shape Vaayu's models
use today:

| Model (Rudra-relevant) | Native shape | Parsed to `tool_calls`? |
|---|---|---|
| **MiniMax-M2.7** (Rudra's coder) | `<minimax:tool_call><invoke name=..><parameter name=..>` XML | **yes** |
| Qwen3.6-35B | `<tool_call><function=..><parameter=..>` XML | yes |
| Qwen2.5 | `<tool_call>{json}</tool_call>` | yes |
| Llama-3.x | bare `{"name":.., "parameters":{..}}` | yes (non-stream; see §6) |

## 3. Non-streaming response (`stream: false`)

Request is unchanged from §2 of the original doc (`tools`, `tool_choice`, `session_id` in body).
Vaayu returns:

```json
{
  "id": "chatcmpl-103eff5076cc4abaa7a83671",
  "object": "chat.completion",
  "model": "vaayu",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I need to calculate 12 times 7 using the multiply tool.",
      "tool_calls": [{
        "id": "call_508fdcc304f9dc25fe74d6d2",
        "type": "function",
        "function": { "name": "multiply", "arguments": "{\"a\": 12, \"b\": 7}" }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": { "prompt_tokens": 210, "completion_tokens": 47, "total_tokens": 257 },
  "session_id": "repro-tools"
}
```

- `tool_calls` is an **array** on `message`.
- `function.arguments` is a **JSON-encoded string** (OpenAI spec), and it parses to the schema
  Rudra sent in `tools[].function.parameters` — Vaayu coerces XML values to the declared types
  (e.g. `"12"` → `12` for an `integer` parameter).
- `finish_reason` is `"tool_calls"`.
- `content` is `null` (or any prose that preceded the call) — the XML is gone.
- Multiple calls in one turn → multiple entries, each with a distinct `id`.

## 4. Streaming response (`stream: true`)

Rudra runs with `--stream`. MiniMax's wrapper is a marker Vaayu buffers from, so the raw XML is
**never** emitted as a content delta. The call arrives as a single structured `delta.tool_calls`
chunk, and the final chunk carries `finish_reason: "tool_calls"`:

```
: vaayu queue position 0

data: {"choices":[{"index":0,"delta":{"role":"assistant"}}], ...}
data: {"choices":[{"index":0,"delta":{"content":"I "}}], ...}
data: {"choices":[{"index":0,"delta":{"content":"nee"}}], ...}
... (prose deltas only — NO <minimax:tool_call> text ever appears here) ...
data: {"choices":[{"index":0,"delta":{"content":"he multiply tool.\n"}}], ...}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_508fdcc304f9dc25fe74d6d2","type":"function","function":{"name":"multiply","arguments":"{\"a\": 12, \"b\": 7}"}}]}}], ...}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}], ...}
data: [DONE]
```

(`langchain-openai` concatenates `delta.tool_calls[].function.arguments` across chunks per the
streaming spec; here the whole call lands in one chunk.)

## 5. Does Rudra need to change anything?

**For MiniMax (and Qwen): no.** The response is now standard OpenAI; `langchain-openai` maps
`message.tool_calls` as usual, and `rudra models test` should report `tools: ok`. Keep Rudra
provider-agnostic — do **not** add any `<minimax:tool_call>` scraping.

## 6. One caveat worth knowing (not MiniMax)

A model whose native tool call has **no wrapper marker** (Llama-3.x emits a bare JSON object) cannot
be buffered mid-stream, so when such a model is *streamed* its tool call also appears in `content`
(the first balanced JSON object). Non-streamed, it is clean. **This does not affect MiniMax or Qwen**
(both have markers and stream clean). If Rudra ever points a role at a Llama checkpoint and relies on
streamed tool calls, call that role non-streamed until Vaayu closes this (tracked on the Vaayu side).

## 7. Reproduce

The `curl` from §2 of `vaayu-tool-calling.md` now returns the §3 shape; add `"stream": true` for §4.
Both were run on-device against MiniMax-M2.7 on 2026-10-28 — the outputs above are the real captures.
