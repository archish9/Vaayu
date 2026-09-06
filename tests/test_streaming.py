"""
Unit tests for vaayu/streaming.py's stream_generate().
Tests are designed to run without a real GPU or model checkpoint.
"""
import threading

from vaayu.streaming import stream_generate


class _FakeModel:
    """Minimal stand-in: only what stream_generate touches (.tokenizer,
    .speculator, .generate)."""
    tokenizer = object()
    speculator = None

    def __init__(self, behavior):
        self._behavior = behavior

    def generate(self, **kwargs):
        self._behavior(kwargs["streamer"])


def _happy_path(streamer):
    streamer.on_finalized_text("hello ", stream_end=False)
    streamer.on_finalized_text("world", stream_end=True)


def _crash_path(streamer):
    raise RuntimeError("boom")


def test_stream_generate_yields_chunks_normally():
    model = _FakeModel(_happy_path)
    chunks = list(stream_generate(model, input_ids=None, max_new_tokens=1))
    assert chunks == ["hello ", "world"]


def test_stream_generate_propagates_exception_instead_of_hanging():
    """A background-thread exception (e.g. the io_uring CQE overflow crash)
    must propagate to the caller, not deadlock stream_generate's consumer
    forever on the streamer's unbounded queue."""
    model = _FakeModel(_crash_path)
    result = {}

    def _drain():
        try:
            list(stream_generate(model, input_ids=None, max_new_tokens=1))
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "stream_generate hung instead of propagating the exception"
    assert isinstance(result.get("exc"), RuntimeError)
    assert str(result["exc"]) == "boom"


# ── real generated-id capture (B2) ───────────────────────────────────────────
# The server cannot count generated tokens by re-tokenizing the decoded text:
# the text is decoded with skip_special_tokens=True, so EOS is gone, and merges
# differ across a decode boundary. generate() hands every real id to the
# streamer's put(), so capturing them there is exact and costs nothing.

import torch

from vaayu.streaming import _IdCountingStreamer


class _FakeIdTokenizer:
    """Enough of a tokenizer for TextStreamer.put(): decode() only."""

    def decode(self, ids, **kw):
        return "".join(f"<{i}>" for i in ids)


def _sink_streamer(**kw):
    sink = []
    return sink, _IdCountingStreamer(_FakeIdTokenizer(), sink, **kw)


def test_id_sink_records_the_real_generated_ids():
    sink, s = _sink_streamer(skip_prompt=False)
    s.put(torch.tensor([11]))
    s.put(torch.tensor([12]))
    s.end()
    assert sink == [11, 12]


def test_id_sink_excludes_the_prompt_when_skip_prompt_is_set():
    # generate() puts the prompt first; TextStreamer.put swallows that one call
    # and clears the flag. The sink must skip exactly the same call, or every
    # count is long by a whole prompt.
    sink, s = _sink_streamer(skip_prompt=True)
    s.put(torch.tensor([[1, 2, 3, 4]]))     # the prompt
    s.put(torch.tensor([99]))
    s.end()
    assert sink == [99]


def test_id_sink_records_every_id_of_a_multi_token_put():
    # speculative_generate puts a whole accepted run at once, shaped (1, N).
    sink, s = _sink_streamer(skip_prompt=False)
    s.put(torch.tensor([[7, 8, 9]]))
    s.end()
    assert sink == [7, 8, 9]


def _puts_ids(streamer):
    streamer.put(torch.tensor([[1, 2]]))    # prompt, skipped
    streamer.put(torch.tensor([41]))
    streamer.put(torch.tensor([42]))
    streamer.end()


def test_stream_generate_fills_token_ids_out_when_asked():
    model = _FakeModel(_puts_ids)
    model.tokenizer = _FakeIdTokenizer()
    ids = []
    list(stream_generate(model, input_ids=None, max_new_tokens=2, token_ids_out=ids))
    assert ids == [41, 42]


def test_stream_generate_is_unchanged_when_no_sink_is_passed():
    model = _FakeModel(_happy_path)
    assert list(stream_generate(model, input_ids=None, max_new_tokens=1)) == ["hello ", "world"]


# ── the speculative path's first token (P7) ──────────────────────────────────
# TextStreamer.put() swallows its FIRST call when skip_prompt=True, and
# stream_generate always sets skip_prompt=True. generate() spends that call on
# the prompt; speculative_generate has no prompt put, so the call that gets
# eaten is its first GENERATED token -- the one seeded from prefill logits.
# Latent only because the drafter is off by default.

from vaayu.vaayu_base import VaayuBaseModel

from .test_speculative_prefix_reuse import _FakeSpecModel
from vaayu.prefix_cache import PrefixCacheStore


def _spec_run(streamer, max_new_tokens=3):
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    return VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]),
        max_new_tokens=max_new_tokens, streamer=streamer, session_id="s1")


def test_speculative_streams_every_generated_id_under_skip_prompt():
    sink, s = _sink_streamer(skip_prompt=True)
    out = _spec_run(s)
    generated = out[0, 5:].tolist()
    assert len(generated) == 3, "premise: 3 tokens were generated"
    assert sink == generated, "the prefill-seeded first token never reached the streamer"


def test_speculative_streams_every_generated_token_as_text():
    from transformers import TextIteratorStreamer
    s = TextIteratorStreamer(_FakeIdTokenizer(), skip_prompt=True)
    out = _spec_run(s)
    text = "".join(list(s))
    assert text == "".join(f"<{i}>" for i in out[0, 5:].tolist())


def test_speculative_still_skips_the_prompt_under_skip_prompt():
    # The fix must not swing the other way: the prompt is not generated text.
    sink, s = _sink_streamer(skip_prompt=True)
    out = _spec_run(s)
    assert 1 not in sink and 5 not in sink, "prompt ids leaked into the sink"
    assert len(sink) == out.shape[1] - 5


def test_speculative_with_skip_prompt_off_streams_prompt_then_generated():
    # skip_prompt=False is generate()'s contract: prompt first, then tokens.
    sink, s = _sink_streamer(skip_prompt=False)
    out = _spec_run(s)
    assert sink == out[0].tolist()


# ── per-request speculation bypass ───────────────────────────────────────────
# `speculative_generate` is greedy and has no stop-string machinery, so loading
# a speculator used to change EVERY request on the process: a sampled request
# got greedy text and a tool request ran past its own stop string. Dispatch is
# per request instead -- speculate only when this call asks for nothing that
# loop would drop, which is the same question `speculative_ignored_sampling`
# already answers for the warning, asked once so the two cannot disagree.

class _DispatchRecordingModel:
    """Records which of the two generate paths stream_generate chose."""
    tokenizer = object()

    def __init__(self):
        self.speculator = object()      # loaded, i.e. speculation is available
        self.called = None

    def generate(self, **kwargs):
        self.called = "generate"
        _happy_path(kwargs["streamer"])

    def speculative_generate(self, **kwargs):
        self.called = "speculative_generate"
        _happy_path(kwargs["streamer"])


def _dispatch(**kwargs):
    model = _DispatchRecordingModel()
    list(stream_generate(model, input_ids=None, max_new_tokens=1, **kwargs))
    return model.called


def test_a_greedy_request_speculates():
    assert _dispatch(temperature=0.0, top_p=1.0) == "speculative_generate"


def test_a_sampled_request_takes_the_ordinary_path():
    # The whole point: temperature was asked for, so it must be HONOURED by
    # HF's loop rather than swallowed by the greedy one.
    assert _dispatch(temperature=0.7) == "generate"


def test_a_request_with_stop_strings_takes_the_ordinary_path():
    # The server appends stop strings to every tool-enabled request; the
    # speculative loop drops them and generates straight past the tool call.
    assert _dispatch(temperature=0.0, top_p=1.0,
                     stop_strings=["```"]) == "generate"


def test_stream_generates_own_sampling_defaults_bypass_too():
    # temperature=0.7/top_p=0.9 are this function's defaults, so a caller that
    # passes neither is asking for sampling and must not be given greedy text.
    assert _dispatch() == "generate"


def test_a_model_with_no_speculator_is_unaffected():
    model = _FakeModel(_happy_path)
    assert list(stream_generate(model, input_ids=None, max_new_tokens=1,
                                temperature=0.7)) == ["hello ", "world"]


def test_the_bypass_says_why_it_skipped_speculation(capsys):
    _dispatch(temperature=0.7, stop_strings=["```"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "Vaayu-Spec" in ln]
    assert len(lines) == 1, f"expected exactly one notice, got: {lines}"
    assert "temperature" in lines[0] and "stop_strings" in lines[0]


def test_a_speculating_request_prints_no_notice(capsys):
    _dispatch(temperature=0.0, top_p=1.0)
    assert "Vaayu-Spec" not in capsys.readouterr().out
