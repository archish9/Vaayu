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
