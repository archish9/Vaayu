"""`vaayu/server/config.py` is the server's ONLY input -- there are no CLI flags.

Same bug class as test_server_config_passthrough.py, one layer earlier: a
ServerConfig field with no constant in config.py is unreachable for the user,
and a constant whose value never lands in ServerConfig silently does nothing.
Both are indistinguishable from a knob that works until someone measures.
"""
import dataclasses
import inspect

import pytest

from vaayu.server import config
from vaayu.server.app import ServerConfig

# config.py speaks GB because humans do; ServerConfig speaks bytes.
CONST_NAME = {"ram_cache_budget": "RAM_CACHE_BUDGET_GB"}


def const_for(field_name: str) -> str:
    return CONST_NAME.get(field_name, field_name.upper())


@pytest.fixture
def launched(monkeypatch):
    """Run main() with the loader and the HTTP server stubbed; return the cfg."""
    seen = {}

    import uvicorn

    from vaayu.server import __main__ as entry

    monkeypatch.setattr(entry, "create_app", lambda cfg: seen.setdefault("cfg", cfg))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    return seen


def test_every_serverconfig_field_is_configurable(monkeypatch):
    for f in dataclasses.fields(ServerConfig):
        assert hasattr(config, const_for(f.name)), (
            f"ServerConfig.{f.name} has no {const_for(f.name)} in config.py -- "
            f"there are no CLI flags, so the user cannot set it at all")


def test_config_defaults_match_the_serverconfig_defaults():
    # A config.py default that drifts from the dataclass default means the
    # server and every library caller behave differently for no stated reason.
    for f in dataclasses.fields(ServerConfig):
        if f.name in ("model_path", "ram_cache_budget"):
            continue  # required / unit-converted, both covered below
        assert getattr(config, const_for(f.name)) == f.default, (
            f"config.{const_for(f.name)} != ServerConfig.{f.name} default")


def test_the_fast_server_defaults_survive():
    assert config.MAX_PINNED_LAYERS == -1, "0 streams every layer from NVMe per token"
    assert config.ENABLE_PREFIX_CACHE is True, "reuse is the largest lever here"
    assert config.PREFILL_CHUNK_SIZE == 512


def test_main_builds_serverconfig_from_the_config_module(monkeypatch, launched):
    monkeypatch.setattr(config, "MODEL_PATH", "/ckpt")
    monkeypatch.setattr(config, "PORT", 9123)
    monkeypatch.setattr(config, "MAX_PINNED_LAYERS", 4)
    monkeypatch.setattr(config, "PREFIX_COMPRESS", 9)
    monkeypatch.setattr(config, "USE_PROMPT_LOOKUP", True)

    from vaayu.server.__main__ import main
    assert main() == 0

    cfg = launched["cfg"]
    assert (cfg.model_path, cfg.port) == ("/ckpt", 9123)
    assert (cfg.max_pinned_layers, cfg.prefix_compress, cfg.use_prompt_lookup) == (4, 9, True)
    assert launched["port"] == 9123, "uvicorn must bind the configured port"


def test_ram_cache_budget_gb_is_converted_to_bytes(monkeypatch, launched):
    monkeypatch.setattr(config, "MODEL_PATH", "/ckpt")
    monkeypatch.setattr(config, "RAM_CACHE_BUDGET_GB", 2.5)

    from vaayu.server.__main__ import main
    main()
    assert launched["cfg"].ram_cache_budget == int(2.5 * 1024 ** 3)


def test_none_ram_cache_budget_stays_none(monkeypatch, launched):
    monkeypatch.setattr(config, "MODEL_PATH", "/ckpt")
    monkeypatch.setattr(config, "RAM_CACHE_BUDGET_GB", None)

    from vaayu.server.__main__ import main
    main()
    assert launched["cfg"].ram_cache_budget is None, "None derives it from TOTAL RAM"


def test_empty_model_path_names_the_file_to_edit(monkeypatch, launched):
    monkeypatch.setattr(config, "MODEL_PATH", "")
    from vaayu.server.__main__ import main
    with pytest.raises(SystemExit) as e:
        main()
    assert "config.py" in str(e.value)


def test_there_are_no_cli_flags(monkeypatch):
    from vaayu.server import __main__ as entry
    src = inspect.getsource(entry)
    assert "argparse" not in src, "config.py is the only input; flags were removed"
    assert "config" in src


def test_a_gitignored_local_override_can_shadow_any_value():
    src = inspect.getsource(config)
    assert "config_local" in src, (
        "config.py must import config_local last so machine-specific values "
        "(MODEL_PATH) stay out of git")
