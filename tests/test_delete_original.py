"""`DELETE_ORIGINAL` reclaims the source shards a completed split replaced.

Irreversible by nature -- recovery is a re-download of 39-66 GB -- so two things
are gates, not niceties: the flag must actually reach the loader (a flag that
silently does nothing is how disk quietly fills), and the server must SAY it is
on before the load that acts on it.
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(kwargs)
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ga, "audit", lambda: [])
    return seen


def test_delete_original_is_on_by_default():
    assert ServerConfig(model_path="/x").delete_original is True


def test_it_reaches_the_loader(captured):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert captured["delete_original"] is True


def test_it_can_be_turned_off(captured):
    cfg = ServerConfig(model_path="/ckpt", delete_original=False)
    with TestClient(create_app(cfg)):
        pass
    assert captured["delete_original"] is False


def test_the_server_announces_an_irreversible_default_before_it_loads(captured, capsys):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    out = capsys.readouterr().out
    assert "DELETE_ORIGINAL" in out
    assert "irreversible" in out.lower(), "name what cannot be undone"


def test_no_announcement_when_it_is_off(captured, capsys):
    with TestClient(create_app(ServerConfig(model_path="/ckpt", delete_original=False))):
        pass
    assert "DELETE_ORIGINAL" not in capsys.readouterr().out


def test_the_config_module_exposes_it():
    from vaayu.server import config
    assert hasattr(config, "DELETE_ORIGINAL")
    names = {f.name for f in dataclasses.fields(ServerConfig)}
    assert "delete_original" in names
