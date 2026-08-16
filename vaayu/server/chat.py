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

    # return_dict=False asks for bare ids. transformers 5.x defaults it to True
    # and hands back a BatchEncoding, which _looks_ list-like enough to travel
    # several frames before failing -- caught on-device, not by the fakes.
    kwargs = {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
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
    return _as_token_ids(ids)


def _as_token_ids(ids):
    """Normalise whatever apply_chat_template returned into a flat list of ids.

    Templates and transformers versions disagree here: a bare list, a nested
    batch of one, or a BatchEncoding/dict keyed on "input_ids" are all seen in
    the wild. Normalise the shapes we recognise and refuse the rest loudly --
    a wrong-shaped prompt otherwise surfaces many frames away as a tensor error.
    """
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):                       # torch tensor / numpy array
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]                                 # batch of one
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise TypeError(
            f"apply_chat_template returned {type(ids).__name__}, expected a flat "
            f"list of token ids (tokenize=True, return_dict=False were requested)")
    return ids
