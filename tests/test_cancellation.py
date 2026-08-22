"""A client that goes away must stop the generation it started.

Measured failure this exists to prevent: a Postman request was abandoned and
`/health` still reported `"queued": 1` with `ss` showing NO connection on the
port -- an 8148-token generation (44-token prompt, 8192-token window) kept the
GPU and the single slot for what would have been ~88 min at MiniMax-M2.7's
measured 1.537 tok/s, and every later request queued behind a ghost.

The thread running generate() cannot be killed from the event loop, so
cancellation is cooperative: a token set from the loop, read once per token by
a StoppingCriteria (transformers path) and by the while-loop condition
(speculative path, which has no criteria support of its own).
"""
import asyncio

import pytest
import torch

from vaayu.server.app import _watch_disconnect
from vaayu.server.engine import CancelToken, EngineHandle
from vaayu.server.schemas import ChatCompletionRequest
from vaayu.vaayu_base import VaayuBaseModel

from .test_server_engine import FakeModel
from .test_speculative_prefix_reuse import _FakeSpecModel

from vaayu.prefix_cache import PrefixCacheStore


class _GeneratingModel(FakeModel):
    """FakeModel plus the three things a real generate() round touches."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tokenizer.decode = lambda ids, **k: "x" * len(ids)
        self.tokenizer.__class__.__call__ = lambda _s, t, **k: {"input_ids": [1] * len(t)}
        self.config = type("C", (), {"eos_token_id": 2})()

    def generate(self, **kw):
        new = torch.tensor([[7, 7]])
        streamer = kw.get("streamer")
        if streamer is not None:
            streamer.put(new)
            streamer.end()
        return torch.cat([kw["input_ids"], new], dim=1)


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


# ── the token ────────────────────────────────────────────────────────────────

def test_a_fresh_token_is_not_cancelled():
    assert CancelToken().cancelled is False


def test_cancel_is_visible_and_sticky():
    t = CancelToken()
    t.cancel()
    assert t.cancelled is True
    t.cancel()
    assert t.cancelled is True


# ── the transformers path ────────────────────────────────────────────────────

def test_no_token_means_no_stopping_criteria():
    # Library callers and the tests' fakes must see the kwargs they always saw.
    kw = EngineHandle(FakeModel()).build_generate_kwargs(_req(), [1, 2, 3])
    assert "stopping_criteria" not in kw


def test_a_token_installs_a_criteria_that_flips_when_cancelled():
    t = CancelToken()
    kw = EngineHandle(FakeModel()).build_generate_kwargs(_req(), [1, 2, 3], cancel=t)
    crit = kw["stopping_criteria"]
    ids = torch.tensor([[1, 2, 3]])
    assert bool(crit(ids, None).any()) is False, "not cancelled -> generation runs"
    t.cancel()
    assert bool(crit(ids, None).any()) is True, "cancelled -> generation stops"


def test_the_criteria_answers_per_batch_row_as_transformers_requires():
    # StoppingCriteriaList ORs criteria into a (batch,) bool tensor; returning a
    # bare Python bool works until something batches and then fails obscurely.
    t = CancelToken()
    kw = EngineHandle(FakeModel()).build_generate_kwargs(_req(), [1, 2, 3], cancel=t)
    out = kw["stopping_criteria"](torch.tensor([[1, 2], [3, 4]]), None)
    assert out.shape == (2,) and out.dtype == torch.bool


def test_generate_stops_early_once_the_token_is_cancelled():
    """The real end-to-end shape: a model that keeps generating until the
    criteria says stop, cancelled from the event loop while it runs."""
    class SpinningModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.steps = 0
            self.tokenizer.decode = lambda ids, **kw: "x" * len(ids)
            self.config = type("C", (), {"eos_token_id": 2})()

        def generate(self, **kw):
            crit = kw["stopping_criteria"]
            ids = kw["input_ids"]
            while not bool(crit(ids, None).any()):
                self.steps += 1
                if self.steps > 10_000:              # test-only runaway guard
                    raise AssertionError("criteria never fired")
                ids = torch.cat([ids, torch.tensor([[7]])], dim=1)
            return ids

    model = SpinningModel()
    e = EngineHandle(model)
    token = CancelToken()

    async def main():
        task = asyncio.create_task(e.generate(_req(max_tokens=8000), cancel=token))
        await asyncio.sleep(0.05)
        token.cancel()
        return await asyncio.wait_for(task, timeout=5)

    result = asyncio.run(main())
    assert model.steps > 0, "premise: it was generating"
    assert result.completion_tokens < 8000, "it stopped short of the cap"


# ── the speculative path (its own loop, no criteria support) ─────────────────

def test_speculative_generate_honours_a_cancelled_criteria():
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _FakeSpecModel(pc)
    token = CancelToken()
    token.cancel()

    from vaayu.server.engine import cancel_criteria
    out = VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=50,
        session_id="s1", stopping_criteria=cancel_criteria(token))
    # 5 prompt + the ONE token the prefill step seeds before the draft/verify
    # loop is ever entered (vaayu_base.py, "Seed the first new token from
    # prefill logits"). Cancellation cannot undo work already done; what it must
    # prevent is the other 49.
    assert out.shape[1] == 6, "cancelled at the loop head -> no draft rounds ran"


def test_speculative_generate_runs_normally_without_a_criteria():
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _FakeSpecModel(pc)
    out = VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=3, session_id="s1")
    assert out.shape[1] == 8, "5 prompt + 3 generated"


# ── the disconnect watchdog ──────────────────────────────────────────────────

class _FakeRequest:
    """is_disconnected() flips to True after `after` calls."""

    def __init__(self, after=1):
        self.after = after
        self.calls = 0

    async def is_disconnected(self):
        self.calls += 1
        return self.calls > self.after


def test_the_watchdog_cancels_when_the_client_goes_away():
    t = CancelToken()
    asyncio.run(asyncio.wait_for(
        _watch_disconnect(_FakeRequest(after=1), t, poll_s=0.01), timeout=2))
    assert t.cancelled is True


def test_the_watchdog_returns_when_generation_finishes_first():
    # Generation done -> the token is cancelled by the caller's finally, and the
    # watchdog must exit rather than poll a completed request forever.
    t = CancelToken()

    class _Connected:
        async def is_disconnected(self):
            return False

    async def main():
        task = asyncio.create_task(_watch_disconnect(_Connected(), t, poll_s=0.01))
        await asyncio.sleep(0.05)
        t.cancel()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(main())


# ── the silent-request warning ───────────────────────────────────────────────

def test_a_non_streaming_request_with_no_cap_says_how_long_it_may_be_silent(capsys):
    """The request that started all this looked hung for 15 minutes. It was not
    hung -- it was an 8148-token generation that sends nothing until the last
    token. Say so at admission, where a log reader can act on it."""
    e = EngineHandle(_GeneratingModel(max_seq_len=8192))
    asyncio.run(e.generate(_req()))
    out = capsys.readouterr().out
    assert "no max_tokens" in out and "8182" in out, out
    assert "stream" in out, "the fix for the caller is streaming; name it"


def test_a_capped_request_is_not_warned_about(capsys):
    e = EngineHandle(_GeneratingModel(max_seq_len=8192))
    asyncio.run(e.generate(_req(max_tokens=64)))
    assert "no max_tokens" not in capsys.readouterr().out


def test_a_streaming_request_is_not_warned_about(capsys):
    e = EngineHandle(_GeneratingModel(max_seq_len=8192))

    async def drain():
        async for _ in e.stream(_req(stream=True)):
            pass

    asyncio.run(drain())
    assert "no max_tokens" not in capsys.readouterr().out
