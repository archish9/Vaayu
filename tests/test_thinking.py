"""Thinking-mode control, without a per-model table (CLAUDE.md §5).

Templates disagree on how thinking is expressed, so two mechanisms:

1. `chat_template_kwargs` -- templates with their own flag (Qwen3's
   `enable_thinking`) honour it; the rest ignore an unknown Jinja variable.
2. Close-tag injection -- MiniMax-M2.7 hard-codes `<think>` into the generation
   prompt with no flag at all, so the only way out is to close the block.

Which one applies is DETECTED from the rendered generation prompt, the same way
tools.detect_tool_format probes the template for tool markers.
"""
import pytest

from vaayu.server.chat import (generation_tail, render_prompt,
                               unclosed_thinking_marker)


class FakeTokenizer:
    """Renders a MiniMax-shaped prompt: the generation prompt opens <think>."""

    chat_template = "x"

    def __init__(self, tail="]~b]ai\n<think>\n", honours_flag=False):
        self._tail = tail
        self._honours_flag = honours_flag
        self.seen_kwargs = None

    def apply_chat_template(self, messages, **kw):
        self.seen_kwargs = kw
        body = "".join(m["content"] for m in messages)
        tail = self._tail
        if self._honours_flag and kw.get("enable_thinking") is False:
            tail = "]~b]ai\n"
        text = body + (tail if kw.get("add_generation_prompt") else "")
        if kw.get("tokenize"):
            return [len(text)]
        return text

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


# ── detection ────────────────────────────────────────────────────────────────

def test_an_open_think_tag_in_the_generation_prompt_is_detected():
    assert unclosed_thinking_marker("]~b]ai\n<think>\n") == "</think>"


def test_a_closed_block_needs_no_injection():
    assert unclosed_thinking_marker("<think>\nreasoning\n</think>\n\n") is None


def test_a_prompt_with_no_thinking_at_all_is_left_alone():
    assert unclosed_thinking_marker("<|im_start|>assistant\n") is None


def test_alternative_marker_spellings_are_recognised():
    assert unclosed_thinking_marker("<thinking>\n") == "</thinking>"


def test_the_tail_is_the_diff_between_with_and_without_generation_prompt():
    """Detection runs on the generation prompt ONLY. Scanning the whole rendered
    text would let a user message containing the literal '<think>' decide how
    the model is prompted."""
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "does <think> in my text matter?"}]
    assert generation_tail(tok, msgs) == "]~b]ai\n<think>\n"


def test_user_text_containing_a_think_tag_does_not_trigger_injection():
    tok = FakeTokenizer(tail="]~b]ai\n")          # template opens no block
    msgs = [{"role": "user", "content": "explain <think> tags"}]
    ids = render_prompt(tok, msgs, enable_thinking=False)
    assert "</think>" not in "".join(chr(i) for i in ids)


# ── the two mechanisms ───────────────────────────────────────────────────────

def test_thinking_off_closes_a_hardcoded_block():
    tok = FakeTokenizer()
    ids = render_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert "".join(chr(i) for i in ids).endswith("<think>\n</think>\n\n")


def test_thinking_on_leaves_the_prompt_exactly_as_the_template_wrote_it():
    tok = FakeTokenizer()
    ids = render_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert "</think>" not in "".join(chr(i) for i in ids)


def test_a_template_with_its_own_flag_is_told_directly():
    tok = FakeTokenizer(honours_flag=True)
    ids = render_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert tok.seen_kwargs["enable_thinking"] is False
    text = "".join(chr(i) for i in ids)
    assert "<think>" not in text and "</think>" not in text, (
        "the template already dropped the block; nothing to inject")


def test_the_flag_is_not_sent_when_thinking_is_on():
    # Sending enable_thinking=True to a template that has never heard of it is
    # harmless, but sending nothing keeps the default path byte-identical.
    tok = FakeTokenizer(honours_flag=True)
    render_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert "enable_thinking" not in tok.seen_kwargs


def test_the_default_is_unchanged_rendering_for_library_callers():
    tok = FakeTokenizer()
    assert render_prompt(tok, [{"role": "user", "content": "hi"}]) == \
           render_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=True)


# ── wiring: config default, per-request override ─────────────────────────────

def _req(**over):
    from vaayu.server.schemas import ChatCompletionRequest
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_thinking_is_off_by_default_on_the_server():
    from vaayu.server.app import ServerConfig
    assert ServerConfig(model_path="/x").enable_thinking is False


def test_a_request_can_ask_for_thinking_back():
    assert _req(enable_thinking=True).enable_thinking is True
    assert _req().enable_thinking is None, "unset means 'use the server default'"


def test_the_engine_resolves_request_over_server_default():
    from vaayu.server.engine import EngineHandle
    from .test_server_engine import FakeModel

    e = EngineHandle(FakeModel(), enable_thinking=False)
    assert e.resolve_thinking(_req()) is False
    assert e.resolve_thinking(_req(enable_thinking=True)) is True

    e_on = EngineHandle(FakeModel(), enable_thinking=True)
    assert e_on.resolve_thinking(_req()) is True
    assert e_on.resolve_thinking(_req(enable_thinking=False)) is False


# ── wiring: streaming default ────────────────────────────────────────────────

def test_stream_is_unset_by_default_so_the_server_decides():
    assert _req().stream is None
    assert _req(stream=False).stream is False


def test_the_server_default_is_streaming():
    from vaayu.server.app import ServerConfig
    assert ServerConfig(model_path="/x").default_stream is True


def test_an_explicit_stream_false_still_wins():
    from vaayu.server.app import ServerConfig, resolve_stream
    cfg = ServerConfig(model_path="/x")            # default_stream=True
    assert resolve_stream(_req(stream=False), cfg) is False
    assert resolve_stream(_req(), cfg) is True
    assert resolve_stream(_req(stream=True), cfg) is True


def test_with_no_server_config_the_openai_spec_default_applies():
    # create_app(engine=...) with no config: embedded/test use, where surprising
    # a caller with SSE would be worse than following the spec.
    from vaayu.server.app import resolve_stream
    assert resolve_stream(_req(), None) is False
