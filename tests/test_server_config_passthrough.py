"""Every ServerConfig knob must actually reach AutoModel.from_pretrained.

This is the bug class that shipped once already: `max_pinned_layers` defaults to
0 in vaayu_base and create_app did not pass it, so the server pinned ZERO
transformer layers while 11.46 GiB of pinning budget sat unused and every layer
streamed from NVMe on every token. A field that exists but is never forwarded is
indistinguishable from a field that works.
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app

# Fields that are the SERVER's own concern and are deliberately not model kwargs.
NOT_MODEL_KWARGS = {"model_path", "queue_depth", "host", "port", "served_model_name",
                    "require_exclusive_gpu", "exclusive_gpu_min_mib",
                    "enable_thinking", "default_stream",
                    "log_dir", "capture_stdio", "log_rotate_mb",
                    "require_session_id"}


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["path"] = path
            seen.update(kwargs)
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    # These tests are about kwarg passthrough, not the exclusive-GPU gate. Left
    # unpatched they pass or fail on whatever else happens to hold the GPU.
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(ga, "audit", lambda **kw: [])
    return seen


def test_every_non_server_config_field_is_forwarded_to_the_loader(captured):
    cfg = ServerConfig(model_path="/ckpt")
    with TestClient(create_app(cfg)):
        pass
    for f in dataclasses.fields(ServerConfig):
        if f.name in NOT_MODEL_KWARGS:
            continue
        assert f.name in captured, (
            f"ServerConfig.{f.name} is declared but never reaches "
            f"AutoModel.from_pretrained -- it silently does nothing")


def test_pinning_auto_fills_by_default_rather_than_pinning_nothing(captured):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert captured["max_pinned_layers"] == -1, (
        "at 0 the engine pins no transformer layers and the whole pinning budget "
        "is wasted; a server holds one model for its whole life")


def test_prefix_cache_is_on_by_default(captured):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert captured["enable_prefix_cache"] is True


def test_explicit_values_win(captured):
    cfg = ServerConfig(model_path="/ckpt", max_pinned_layers=4,
                       use_prompt_lookup=True, draft_tokens=15,
                       prefix_compress=9, attn_implementation="flash_attention_2")
    with TestClient(create_app(cfg)):
        pass
    assert captured["max_pinned_layers"] == 4
    assert captured["use_prompt_lookup"] is True
    assert captured["draft_tokens"] == 15
    assert captured["prefix_compress"] == 9
    assert captured["attn_implementation"] == "flash_attention_2"


# The old test here asserted the argparse defaults matched this dataclass. There
# are no CLI flags any more -- `vaayu/server/config.py` is the only input, and
# tests/test_server_config_module.py holds that same defaults-must-not-drift gate.
