"""Render chat messages (and tool definitions) to token ids.

The checkpoint's own chat template does the work -- there is no per-model
formatting table here, and there must not be one (CLAUDE.md §5).
"""
from .tools import detect_tool_format


# Opening/closing pairs seen in the wild. This is a table of CONVENTIONS, not of
# models: which one applies is decided by probing the rendered generation prompt,
# exactly as tools.detect_tool_format probes for tool markers. A checkpoint whose
# template spells thinking some other way simply matches nothing and is left
# alone -- never mis-prompted.
THINKING_MARKERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|thinking|>", "<|/thinking|>"),
)


def unclosed_thinking_marker(text: str) -> str | None:
    """The closing marker `text` still owes, or None if it owes nothing."""
    for opening, closing in THINKING_MARKERS:
        opened = text.rfind(opening)
        if opened != -1 and text.rfind(closing) < opened:
            return closing
    return None


def generation_tail(tokenizer, messages, tools=None, **template_kwargs) -> str:
    """Just the generation prompt -- what `add_generation_prompt=True` appends.

    Rendered twice and diffed rather than scanned whole: a user message that
    happens to contain the literal `<think>` must not decide how the model is
    prompted.

    `template_kwargs` must match what the real render will send. Probing without
    them reports the block a flag-honouring template was about to drop, and the
    caller then injects a close tag for a block that no longer opens.
    """
    kwargs = {"tokenize": False, **template_kwargs}
    if tools:
        kwargs["tools"] = tools
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    body = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    return full[len(body):] if full.startswith(body) else full


class NoChatTemplate(ValueError):
    """The tokenizer ships no chat template, so messages cannot be rendered."""


class TemplateLacksToolSupport(ValueError):
    """The request carries `tools` but the template cannot render them."""


def render_prompt(tokenizer, messages, tools=None, enable_thinking=True, renderer=None):
    """messages: list[dict] as sent on the wire. Returns a list of token ids.

    `enable_thinking=False` suppresses a reasoning block two ways, because
    templates disagree: the template's own flag when it has one (Qwen3), and
    closing a block the template hard-codes when it does not (MiniMax-M2.7,
    whose generation prompt ends with a bare `<think>` and offers no flag).

    `renderer` is the model's own (`VaayuBaseModel.get_prompt_renderer`), for a
    checkpoint whose prompt format is not a Jinja template. Given, it is used
    instead of the template; None leaves the template path exactly as it was.
    """
    if renderer is not None:
        return _render_with(renderer, tokenizer, messages, tools, enable_thinking)

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

    if not enable_thinking:
        # Templates that know the flag drop the block themselves; the rest
        # ignore an unknown Jinja variable, which costs nothing.
        kwargs["enable_thinking"] = False
        closing = unclosed_thinking_marker(
            generation_tail(tokenizer, messages, tools=tools if tools else None,
                            enable_thinking=False))
        if closing is not None:
            # Nothing else can close it: the block is already open when
            # generation starts. Render as text, close it, re-tokenize --
            # verified byte-identical to the tokenize=True path on-device.
            text_kwargs = {k: v for k, v in kwargs.items() if k != "tokenize"}
            text = tokenizer.apply_chat_template(messages, tokenize=False, **text_kwargs)
            return _as_token_ids(
                tokenizer(text + closing + "\n\n", add_special_tokens=False))

    ids = tokenizer.apply_chat_template(messages, **kwargs)
    return _as_token_ids(ids)


def _render_with(renderer, tokenizer, messages, tools, enable_thinking):
    """A model-supplied renderer: its text, tokenized with NO special tokens
    added -- the renderer writes BOS itself, and a second one shifts every
    position."""
    if tools:
        raise TemplateLacksToolSupport(
            "this model's prompt renderer takes no tools, so the `tools` in this "
            "request cannot be rendered. Refusing rather than dropping them "
            "silently -- the model would never be told the tools exist and the "
            "agent would wait for a call that can never come.")
    try:
        text = renderer.render(messages, enable_thinking)
    except FileNotFoundError as missing:
        raise NoChatTemplate(
            f"this model renders prompts with its own encoder, and it cannot be "
            f"loaded: {missing}") from missing
    return _as_token_ids(tokenizer(text, add_special_tokens=False))


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
