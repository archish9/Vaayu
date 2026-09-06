"""`use_prompt_lookup="auto"` — a REQUEST, resolved once the model is known.

Prompt lookup is worth 1.94x on streamed dense and costs ~nothing when it finds
no n-gram match (k=0 rounds are plain greedy decode), so the server defaults it
on. It must NOT default on for MoE: the adaptive gate's TRIAL phase has a fixed
exploration cost that does not amortize on short generations, measured at 0.932x
on a 128-token Qwen3-235B run, and that is still open.

`is_moe` is not known until the checkpoint is loaded, so config.py cannot answer
this -- the resolution happens where the answer exists, the same shape of
problem `resolve_kv_cache_mode` solves for `"auto"` KV dtype.
"""
import pytest

from vaayu.vaayu_base import resolve_use_prompt_lookup


def test_auto_turns_prompt_lookup_on_for_dense():
    assert resolve_use_prompt_lookup("auto", is_moe=False) is True


def test_auto_leaves_moe_exactly_as_it_ships_today():
    # Not a performance call: MoE's TRIAL cost is an open row. Defaulting this
    # on would ship a measured 6.8% regression to one of the two MoE targets.
    assert resolve_use_prompt_lookup("auto", is_moe=True) is False


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("is_moe", [True, False])
def test_an_explicit_choice_always_wins(explicit, is_moe):
    assert resolve_use_prompt_lookup(explicit, is_moe=is_moe) is explicit


def test_an_unrecognised_request_is_refused_by_name():
    # Silently treating a typo as False would turn speculation off and look
    # like a perf regression with no message anywhere.
    with pytest.raises(ValueError, match="use_prompt_lookup"):
        resolve_use_prompt_lookup("on", is_moe=False)


def test_the_server_ships_auto_as_its_default():
    from vaayu.server import config
    assert config.USE_PROMPT_LOOKUP == "auto"



def test_the_config_default_reaches_serverconfig_unchanged(monkeypatch):
    # "auto" has to survive the config -> ServerConfig -> from_pretrained hop as
    # itself. Resolving it early -- anywhere before `is_moe` is known -- would
    # answer the question with the information the answer depends on missing.
    import dataclasses
    from vaayu.server.app import ServerConfig
    field = {f.name: f for f in dataclasses.fields(ServerConfig)}["use_prompt_lookup"]
    assert field.default == "auto"
