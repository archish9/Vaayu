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


def test_openai_user_is_the_session_id_fallback():
    # OpenAI-native clients tag streams with `user`, never with Vaayu's
    # `session_id`. Dropping it silently cost them exact prefix matching with
    # no way to tell from the response.
    assert _req(user="agent-7").session_key == "agent-7"


def test_an_explicit_session_id_wins_over_user():
    # `user` is a stable END-USER id; session_id names one conversation. When
    # both arrive the narrower one is the right pool key.
    assert _req(user="alice", session_id="agent-7-turn-3").session_key == "agent-7-turn-3"


def test_session_key_is_none_when_neither_is_sent_so_auto_match_runs():
    assert _req().session_key is None
    assert _req(user="").session_key is None, "an empty user is not a session"
