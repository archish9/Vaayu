"""Every engine load parameter is either reachable from the server or declined.

`vaayu/server` is the ONLY interface users have (CLAUDE.md §8), so an engine
parameter the server cannot reach is a feature that does not exist. The gap
opened silently once already -- 14 of 28 unreachable, found by an audit rather
than by the suite -- because nothing connected `vaayu_base.__init__` to
`vaayu/server/config.py`.

This test is that connection. A NEW engine parameter breaks the suite until
someone puts it in one of the two sets below, which is exactly the §8 decision
forced at the moment it is cheapest to make. "No knob" is a valid answer; it is
not a valid silence.
"""
import dataclasses
import inspect

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app
from vaayu.vaayu_base import VaayuBaseModel

# The checkpoint itself: positional, and reachable as `MODEL_PATH`. It is the
# one parameter that is neither a knob nor a decline.
CHECKPOINT_PATH = "model_local_path_or_repo_id"

# Reachable from `vaayu/server/config.py` and passed to from_pretrained.
SERVER_EXPOSED = {
    "device",
    "delete_original",
    "kv_cache_mode",
    "max_pinned_layers",
    "prefill_chunk_size",
    "attn_implementation",
    "ram_cache_budget",
    "enable_prefix_cache",
    "prefix_sessions",
    "prefix_compress",
    "auto_prefix_match",
    "max_seq_len",
    "enable_kv_persist",
    "kv_persist_dir",
    "kv_persist_sync",
    "drafter_model_path",
    "draft_tokens",
    "use_prompt_lookup",
    "enable_expert_usage",
}

# Declined on purpose. The reason is the deliverable -- §8 requires the decision
# be stated, and this dict is where it is stated.
SERVER_DELIBERATELY_NOT_EXPOSED = {
    "dtype": "loader plumbing; the quantized paths pick their own compute dtype",
    "hf_token": "HF auth belongs in the environment, not in a served config file",
    "layer_names_dict": "dev override for an unrecognised architecture; detection is the "
                        "supported path (CLAUDE.md §5)",
    "tokenizer_class": "same -- a dev override for a checkpoint the loader misreads",
    "layer_shards_saving_path": "library-facing: where a one-off split is written",
    "profiling_mode": "dev instrumentation, not an operator control",
    "compression": "`__init__` overrides it in places; not a live path",
    "use_torch_compile": "same -- overridden in `__init__`, not a live path",
    "prefetching": "loader plumbing with no measured operator reason to turn off",
}


def _engine_params() -> set:
    return {name for name in inspect.signature(VaayuBaseModel.__init__).parameters
            if name != "self"} - {CHECKPOINT_PATH}


def test_every_engine_parameter_is_classified():
    declared = _engine_params()
    classified = SERVER_EXPOSED | set(SERVER_DELIBERATELY_NOT_EXPOSED)
    unclassified = declared - classified
    stale = classified - declared
    assert not unclassified, (
        f"engine parameters with no server decision: {sorted(unclassified)}. "
        f"CLAUDE.md §8: expose it in vaayu/server/config.py, or add it to "
        f"SERVER_DELIBERATELY_NOT_EXPOSED with the reason.")
    assert not stale, (
        f"classified names that are no longer engine parameters: {sorted(stale)}")


def test_the_two_classifications_are_disjoint():
    assert not (SERVER_EXPOSED & set(SERVER_DELIBERATELY_NOT_EXPOSED))


def test_every_exposed_parameter_is_a_serverconfig_field():
    fields = {f.name for f in dataclasses.fields(ServerConfig)}
    missing = SERVER_EXPOSED - fields
    assert not missing, f"exposed but no ServerConfig field: {sorted(missing)}"


def test_every_exposed_parameter_actually_reaches_the_loader(monkeypatch):
    """A knob from_pretrained never receives is decoration (CLAUDE.md §8 #2)."""
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
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(ga, "audit", lambda **kw: [])

    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass

    missing = SERVER_EXPOSED - set(seen)
    assert not missing, f"never passed to from_pretrained: {sorted(missing)}"


@pytest.mark.parametrize("declined", sorted(SERVER_DELIBERATELY_NOT_EXPOSED))
def test_each_decline_carries_a_reason(declined):
    assert SERVER_DELIBERATELY_NOT_EXPOSED[declined].strip()
