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


def test_return_dict_false_is_requested_so_transformers_5_returns_bare_ids():
    # transformers 5.x defaults return_dict=True and hands back a BatchEncoding.
    # Found on-device by test_server_smoke.py, not by these fakes.
    tok = FakeTokenizer(PLAIN_TEMPLATE)
    render_prompt(tok, MESSAGES)
    assert tok.calls[0][1]["return_dict"] is False


def test_a_batchencoding_style_mapping_is_unwrapped():
    class MappingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kw):
            return {"input_ids": [[5, 6, 7]], "attention_mask": [[1, 1, 1]]}

    assert render_prompt(MappingTokenizer(PLAIN_TEMPLATE), MESSAGES) == [5, 6, 7]


def test_a_nested_batch_of_one_is_flattened():
    class NestedTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kw):
            return [[9, 8, 7]]

    assert render_prompt(NestedTokenizer(PLAIN_TEMPLATE), MESSAGES) == [9, 8, 7]


def test_a_tensor_like_return_is_converted():
    class TensorLike:
        def tolist(self):
            return [[1, 2]]

    class TensorTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kw):
            return TensorLike()

    assert render_prompt(TensorTokenizer(PLAIN_TEMPLATE), MESSAGES) == [1, 2]
