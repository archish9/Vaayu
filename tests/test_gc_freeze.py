"""P1(1): freeze the permanent object graph once the model is loaded.

A loaded model is a large, permanent object graph -- module tree, expert-cache
views, the pinned-slab bookkeeping -- and every gen-2 collection traverses all
of it for objects that will never be freed. `gc.freeze()` moves what already
exists into a permanent generation the collector skips.

Load-time garbage must be collected FIRST: whatever is frozen is retained for
the life of the process, so freezing over uncollected garbage leaks it forever.
"""
import gc

import pytest

from vaayu.utils import freeze_after_load


def test_freeze_after_load_moves_live_objects_out_of_the_collector():
    gc.unfreeze()
    assert gc.get_freeze_count() == 0
    try:
        freeze_after_load()
        assert gc.get_freeze_count() > 0, "nothing was frozen"
    finally:
        # Permanent by design, so the test process must undo it or every later
        # test runs against a collector that skips this module's objects.
        gc.unfreeze()


def test_freeze_after_load_collects_garbage_before_freezing_it():
    """A cycle dropped during load must not be frozen into the process."""
    class Node:
        pass

    gc.unfreeze()
    try:
        a, b = Node(), Node()
        a.peer, b.peer = b, a           # unreachable cycle once the names go
        del a, b
        before = gc.get_freeze_count()
        freeze_after_load()
        frozen = gc.get_freeze_count() - before
        # Collect again: anything the freeze should have caught is now
        # unreachable AND frozen, so a second pass finds nothing to free.
        assert gc.collect() == 0, "load-time garbage survived into the frozen set"
        assert frozen > 0
    finally:
        gc.unfreeze()


# ── the server is the only caller: it loads one model and exits ─────────────

@pytest.fixture
def fake_load(monkeypatch):
    """The real load path, with AutoModel and the GPU audit stubbed out."""
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ga, "audit", lambda **kw: [])


@pytest.fixture
def freeze_calls(monkeypatch):
    calls = []
    import vaayu.server.app as app_mod
    monkeypatch.setattr(app_mod, "freeze_after_load", lambda: calls.append(1))
    return calls


def test_startup_freezes_once_after_the_model_is_loaded(fake_load, freeze_calls):
    from fastapi.testclient import TestClient
    from vaayu.server.app import ServerConfig, create_app

    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert freeze_calls == [1], "the loaded model's object graph was not frozen"


def test_an_injected_engine_is_not_frozen(freeze_calls):
    """Tests and library callers inject an engine and keep using the process;
    freezing their objects would retain them for its whole life."""
    from fastapi.testclient import TestClient
    from vaayu.server.app import ServerConfig, create_app

    engine = type("E", (), {"cancel_all": lambda self: None, "model": None,
                            "max_seq_len": 8192, "queue_depth": 8, "depth": 0})()
    with TestClient(create_app(ServerConfig(model_path="/ckpt"), engine=engine)):
        pass
    assert freeze_calls == []
