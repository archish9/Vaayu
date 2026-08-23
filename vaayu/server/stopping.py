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
                  produced: int, cap: int, cancelled: bool = False) -> str:
    """OpenAI `finish_reason` for a completed generation.

    Order is deliberate. A template emits its tool-call close marker and then EOS
    in the same breath, so EOS must not outrank a tool call or the client reads a
    tool call as prose. And EOS must outrank the cap: a model that finished ON the
    last admissible token was not truncated.

    `cancelled` sits between them. A run halted by cancel_criteria -- client
    disconnect, SIGTERM, _teardown -- produced fewer tokens than the cap, which
    is exactly what the hit_stop_string inference below reads as "the model chose
    to stop". It did not, so cancellation must be consulted first. It is reported
    as "length" rather than an invented value: the vocabulary here is OpenAI's,
    and "length" is what already means "this answer is incomplete, do not trust
    it as final" -- the only distinction a client acts on. Which of the two
    truncations it was is in the server log. But a real EOS still outranks it:
    if the model had already finished, cancelling afterwards changes nothing.
    """
    if hit_tool_call:
        return "tool_calls"
    if hit_eos:
        return "stop"
    if cancelled:
        return "length"
    if hit_stop_string:
        return "stop"
    if produced >= cap:
        return "length"
    return "stop"
