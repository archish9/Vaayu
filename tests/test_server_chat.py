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


# ── a model-supplied renderer (S5) ──────────────────────────────────────────


class FakeRenderer:
    def __init__(self, text="ab", error=None):
        self.text, self.error, self.calls = text, error, []

    def render(self, messages, enable_thinking):
        self.calls.append((messages, enable_thinking))
        if self.error is not None:
            raise self.error
        return self.text


class TextTokenizer(FakeTokenizer):
    """Encodes by character code, so the ids prove WHICH text was tokenized."""

    def __call__(self, text, add_special_tokens=True):
        self.encoded = (text, add_special_tokens)
        return {"input_ids": [ord(c) for c in text]}


def test_a_model_renderer_is_used_instead_of_the_template():
    tok = TextTokenizer(PLAIN_TEMPLATE)
    assert render_prompt(tok, MESSAGES, renderer=FakeRenderer("ab")) == [97, 98]
    assert tok.calls == [], "the template must not be consulted when the model renders"


def test_a_model_renderer_works_where_there_is_no_template():
    assert render_prompt(TextTokenizer(None), MESSAGES,
                         renderer=FakeRenderer("ab")) == [97, 98]


def test_the_renderers_text_is_tokenized_without_adding_special_tokens():
    # The renderer writes BOS itself; a second one shifts every position.
    tok = TextTokenizer(None)
    render_prompt(tok, MESSAGES, renderer=FakeRenderer("ab"))
    assert tok.encoded == ("ab", False)


@pytest.mark.parametrize("thinking", [True, False])
def test_enable_thinking_reaches_the_renderer(thinking):
    renderer = FakeRenderer()
    render_prompt(TextTokenizer(None), MESSAGES, enable_thinking=thinking, renderer=renderer)
    assert renderer.calls == [(MESSAGES, thinking)]


def test_tools_are_refused_by_a_model_renderer():
    renderer = FakeRenderer()
    with pytest.raises(TemplateLacksToolSupport):
        render_prompt(TextTokenizer(None), MESSAGES, tools=TOOLS, renderer=renderer)
    assert renderer.calls == [], "refused before rendering, never rendered without them"


def test_an_empty_tools_list_is_no_tools_for_a_model_renderer():
    # Review Focus 1: the Jinja path treats `tools=[]` as absent; so must this one.
    assert render_prompt(TextTokenizer(None), MESSAGES, tools=[],
                         renderer=FakeRenderer("ab")) == [97, 98]


def test_a_renderer_that_cannot_load_its_encoder_is_refused_naming_the_file():
    renderer = FakeRenderer(error=FileNotFoundError("/ckpt/encoding/encoding_dsv4.py is missing"))
    with pytest.raises(NoChatTemplate, match="encoding_dsv4.py"):
        render_prompt(TextTokenizer(None), MESSAGES, renderer=renderer)


def test_a_renderer_rejecting_a_message_propagates_unchanged():
    # Review Focus 2: same as a Jinja template that raises today -- no relabel.
    renderer = FakeRenderer(error=ValueError("unknown role: narrator"))
    with pytest.raises(ValueError, match="narrator") as raised:
        render_prompt(TextTokenizer(None), MESSAGES, renderer=renderer)
    assert not isinstance(raised.value, NoChatTemplate)
