"""D2 (c): a request that dies must not leave its KV cache behind for the next one.

The row's failure: request 1 dies mid-forward; request 2, ~50 ms later, is refused
at max_seq_len=8192 of a planned 32768. Identified on device (2026-11-27): the dead
request's FP8DynamicCache (8.32 GB) was unreachable cyclic garbage -- exception ->
traceback -> dead frames -> cache -- and the server freezes gc after load, so
nothing ever collected it; `gc.collect()` returned exactly those bytes.

Two pieces, tested separately:
  A. `release_exception_frames` at every place an error crosses from the worker
     thread, so the cache dies by refcount; the error and its traceback text stay.
  B. `_assert_context_fits` re-plans ONCE after gc.collect() + empty_cache(), only
     when it would refuse.
"""
import asyncio
import gc
import json
import weakref

import pytest
import torch

from vaayu.server.engine import EngineHandle, release_exception_frames
from vaayu.vram_plan import VramPlanError

from .test_reqlog_engine import M, _req, rows
from .test_server_run_batch import BatchModel, _drain, _Job


class _Cache:
    """Stands in for the KV cache: weakref-able, held by a frame local."""


@pytest.fixture
def gc_frozen_like_the_server():
    """The server's gc is frozen after load: cyclic garbage is never collected.
    Disabling it is the strict form of that -- only refcounts free anything."""
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    yield
    if was:
        gc.enable()


def _dying_forward(holder):
    cache = _Cache()                      # the dead request's KV, a frame local
    holder.append(weakref.ref(cache))
    try:
        raise torch_oom()
    except RuntimeError as oom:
        raise RuntimeError("CUDA OOM on model.layers.1. Not retried: (test)") from oom


def torch_oom():
    return RuntimeError("CUDA out of memory. (test cause)")


# ── A: the helper ───────────────────────────────────────────────────────────

def test_release_drops_the_locals_but_keeps_the_error_and_its_traceback_text():
    import traceback
    holder = []
    try:
        _dying_forward(holder)
    except RuntimeError as ex:
        assert holder[0]() is not None, "premise: the traceback holds the cache"
        release_exception_frames(ex)
        assert holder[0]() is None, "the cache survived release_exception_frames"
        text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        assert "_dying_forward" in text and "CUDA OOM on model.layers.1" in text
        assert "CUDA out of memory. (test cause)" in text, "the __cause__ chain must stay printable"
        assert str(ex).startswith("CUDA OOM on model.layers.1. Not retried")


def test_release_is_safe_on_an_exception_with_no_traceback_and_on_a_cycle():
    release_exception_frames(RuntimeError("never raised"))
    release_exception_frames(None)
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a                # a cycle must terminate
    release_exception_frames(a)


# ── A: every site an error crosses the worker-thread boundary ───────────────

def _dying_model(holder):
    class Dying(M):
        def generate(self, **kw):
            cache = _Cache()
            holder.append(weakref.ref(cache))
            raise RuntimeError("CUDA OOM on model.layers.1. Not retried: (test)")
    return Dying()


def _assert_diagnostics(tmp_path, exc_info):
    assert exc_info.type is RuntimeError
    assert str(exc_info.value).startswith("CUDA OOM on model.layers.1. Not retried")
    r = rows(tmp_path)
    assert len(r) == 1 and r[0]["finish"] == "error"
    assert r[0]["error"]["type"] == "RuntimeError"
    assert r[0]["error"]["message"].startswith("CUDA OOM on model.layers.1. Not retried")
    assert "generate" in r[0]["error"]["traceback"], "error.traceback lost its frames"
    assert "test_failed_request_release.py" in r[0]["error"]["traceback"]


def test_a_failed_non_streaming_request_does_not_keep_its_cache(tmp_path, gc_frozen_like_the_server):
    holder = []
    e = EngineHandle(_dying_model(holder), log_dir=tmp_path)
    with pytest.raises(RuntimeError) as ei:
        asyncio.run(e.generate(_req()))
    assert holder and holder[0]() is None, "the dead request's cache is still alive"
    _assert_diagnostics(tmp_path, ei)


def test_a_failed_streaming_request_does_not_keep_its_cache(tmp_path, gc_frozen_like_the_server):
    holder = []
    e = EngineHandle(_dying_model(holder), log_dir=tmp_path)

    async def go():
        async for _ in e.stream(_req()):
            pass

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(go())
    assert holder and holder[0]() is None, "the dead request's cache is still alive"
    _assert_diagnostics(tmp_path, ei)


def test_a_failed_batch_does_not_keep_its_cache(gc_frozen_like_the_server):
    holder = []

    class Boom(BatchModel):
        def batched_generate(self, requests, reclaim=False, should_stop=None):
            cache = _Cache()
            holder.append(weakref.ref(cache))
            yield (0, ord("a"))
            raise RuntimeError("CUDA OOM")

    e = EngineHandle(Boom([]))
    jobs = [_Job("s0", [1] * 10, 8), _Job("s1", [1] * 10, 8)]

    async def go():
        await e.run_batch(jobs)
        return [await _drain(j) for j in jobs]

    a, b = asyncio.run(go())
    kinds = [kind for kind, _p in a]
    assert "error" in kinds and any(k == "error" for k, _p in b)
    err = next(p for k, p in a if k == "error")
    assert holder and holder[0]() is None, "the dead batch's cache is still alive"
    assert str(err) == "CUDA OOM"


# ── B: the safety net ───────────────────────────────────────────────────────

class _Fits:
    """Stands in for `_assert_context_fits_once`: refuses until `reclaimed`."""

    def __init__(self, refuse_always=False):
        self.calls, self.reclaimed, self.refuse_always = 0, False, refuse_always
        self.nothing_to_check = False

    def __call__(self, model, n_tokens, **kw):
        self.calls += 1
        if self.nothing_to_check:                    # CPU / no plan: returns None
            return None
        if self.refuse_always or not self.reclaimed:
            raise VramPlanError("Context does not fit. The largest context that fits RIGHT NOW is max_seq_len=8192.\n")
        return 123


@pytest.fixture
def guard(monkeypatch):
    from vaayu.vaayu_base import VaayuBaseModel
    fits = _Fits()
    monkeypatch.setattr(VaayuBaseModel, "_assert_context_fits_once", fits, raising=False)
    collects = []
    real_collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a, **k: (collects.append(1), setattr(fits, "reclaimed", True),
                                                        real_collect(*a, **k))[-1])
    return VaayuBaseModel._assert_context_fits, fits, collects


def test_a_refusal_is_replanned_once_after_a_reclaim(guard):
    fit, fits, collects = guard
    assert fit(object(), 21566) == 123
    assert fits.calls == 2 and len(collects) == 1


def test_a_refusal_that_stands_is_not_looped_and_says_a_retry_ran(guard):
    fit, fits, collects = guard
    fits.refuse_always = True
    with pytest.raises(VramPlanError) as ei:
        fit(object(), 21566)
    assert fits.calls == 2 and len(collects) == 1, "exactly one re-plan, no loop"
    assert "max_seq_len=8192" in str(ei.value)
    assert "re-planned once" in str(ei.value)


def test_an_admitted_request_never_pays_for_a_collection(guard):
    fit, fits, collects = guard
    fits.reclaimed = True
    assert fit(object(), 21566) == 123
    assert fits.calls == 1 and collects == []


def test_a_guard_with_nothing_to_check_never_collects(guard):
    fit, fits, collects = guard
    fits.nothing_to_check = True
    assert fit(object(), 5) is None
    assert collects == []


# ── the cancellation split of _release_failed_forward ───────────────────────

class _Loader:
    def __init__(self):
        self.freed = []

    def free_from_gpu(self, model, layer, moved, quantizer):
        self.freed.append(layer)


def _release(exc, monkeypatch):
    import types
    import vaayu.vaayu_base as vb
    calls = []
    monkeypatch.setattr(vb, "clean_memory", lambda *a, **k: calls.append(1))
    m = types.SimpleNamespace(loader_strategy=_Loader(), model=None, hf_quantizer=None)
    vb.VaayuBaseModel._release_failed_forward(m, ("layer", ["w"]), [1, 2], exc)
    return m.loader_strategy.freed, calls


def test_a_cancellation_frees_the_layer_but_never_runs_a_collection(monkeypatch):
    for exc in (GeneratorExit(), asyncio.CancelledError(), KeyboardInterrupt()):
        freed, calls = _release(exc, monkeypatch)
        assert freed == ["layer"], f"{type(exc).__name__}: the layer's weights must be freed"
        assert calls == [], f"{type(exc).__name__}: gc + empty_cache on a cancellation"


def test_a_memory_class_failure_collects_even_when_wrapped(monkeypatch):
    wrapped = RuntimeError("CUDA OOM on model.layers.1. Not retried: (test)")
    wrapped.__cause__ = torch.OutOfMemoryError("CUDA out of memory. (test)")
    freed, calls = _release(wrapped, monkeypatch)
    assert freed == ["layer"] and calls == [1]
    assert _release(RuntimeError("something else"), monkeypatch)[1] == []
