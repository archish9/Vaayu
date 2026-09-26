"""A5.1: `deepseek_v4` must resolve through `AutoConfig`.

transformers 5.5.4 ships `deepseek_v2/v3/vl` and no `deepseek_v4`, and the
checkpoint ships no `auto_map`, so Vaayu dies at `vaayu_base.py:2131` before it
reaches a single weight. The fix was MiniMax's (a `PretrainedConfig` subclass in
the checkpoint dir plus an `auto_map` entry); since S5 Vaayu registers the
subclass in-process instead, so a plain download needs neither.

Every test here runs off `tests/fixtures/deepseek_v4_config.json`, a byte copy
of the shipped config -- so A5.2-A5.7 can test this path without the 157 G
checkpoint or a GPU.
"""
import dataclasses
import json
import os
from pathlib import Path

import pytest

from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v4_config.json"


@pytest.fixture
def shipped() -> dict:
    return json.loads(FIXTURE.read_text())


def test_config_declares_the_model_type_transformers_lacks(shipped):
    assert DeepseekV4Config(**shipped).model_type == "deepseek_v4"


def test_quantization_config_is_renamed_so_no_fp8_quantizer_attaches(shipped):
    """`vaayu_base.py:3394` reads `config.quantization_config` and, for any
    `quant_method` but exl3, builds an `AutoHfQuantizer` -- here a
    `FineGrainedFP8HfQuantizer` that would swap every Linear for an FP8 one.
    A5.3 dequantises through the TileLang kernels instead, so nothing named
    `quantization_config` may survive on this config.
    """
    config = DeepseekV4Config(**shipped)

    assert getattr(config, "quantization_config", None) is None


def test_the_renamed_quantization_block_keeps_every_field(shipped):
    """Renamed, not dropped: A5.2 builds the reference's `ModelArgs` from it
    (`fmt` -> `dtype="fp8"`, `scale_fmt` -> `scale_fmt="ue8m0"`).
    """
    config = DeepseekV4Config(**shipped)

    assert config.checkpoint_quantization_config == shipped["quantization_config"]


# The three keys the base class owns, and the three it renames. Everything else
# in the shipped config is this class's own field.
BASE_OWNED = {"architectures", "model_type", "transformers_version"}
RENAMED = {
    "torch_dtype": "dtype",                                  # PretrainedConfig BC, configuration_utils.py:239
    "rope_scaling": "rope_parameters",                       # RoPE standardization, modeling_rope_utils.py:706
    "quantization_config": "checkpoint_quantization_config",  # ours, so no FP8 quantizer attaches
}


def _attr_for(key: str) -> str:
    return RENAMED.get(key, key)


@pytest.mark.parametrize("key", sorted(set(json.loads(FIXTURE.read_text())) - BASE_OWNED - set(RENAMED)))
def test_every_shipped_field_is_declared_and_invents_no_default(key):
    """A field the checkpoint sets must be declared here, and must default to
    `None` -- a plausible-but-wrong number loads silently and is worse than a
    missing one. 43 layers is not 61 (deepseek_v3's default), and nothing here
    may quietly supply the difference.
    """
    declared = {f.name: f for f in dataclasses.fields(DeepseekV4Config)}

    assert key in declared, f"{key} is in the checkpoint's config.json but not declared"
    assert declared[key].default is None, f"{key} invents a default: {declared[key].default!r}"


@pytest.mark.parametrize("key", sorted(json.loads(FIXTURE.read_text())))
def test_every_shipped_field_round_trips(shipped, key):
    """Field by field over all 52 keys, no spot check: a silently-defaulted or
    mangled field is exactly the failure this test exists to catch.
    """
    config = DeepseekV4Config(**shipped)
    value = shipped[key]

    if key == "torch_dtype":                       # base BC: str -> torch.dtype
        import torch
        assert config.dtype == getattr(torch, value)
    elif key == "rope_scaling":                    # standardized into rope_parameters
        assert config.rope_parameters.items() >= value.items()
        assert config.rope_parameters["rope_type"] == value["type"]
        assert config.rope_parameters["rope_theta"] == shipped["rope_theta"]
    else:
        assert getattr(config, _attr_for(key)) == value


def test_the_expert_guard_resolves_this_checkpoints_expert_count(tmp_path):
    """A1's guard reads the ordered `("num_experts", "num_local_experts",
    "n_routed_experts")`; DeepSeek supplies only the third. A1 Task 4 exists
    because that lookup silently returned `None` on every production load, so
    confirm it against the real shipped config, not a hand-written dict.
    """
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex
    (tmp_path / "config.json").write_text(FIXTURE.read_text())

    assert ExpertOffsetIndex(str(tmp_path)).expected_expert_count() == 256


# ── a plain download: no `auto_map`, no copied files (S5) ──────────────────


def test_a_plain_download_resolves_through_vaayus_registration(tmp_path):
    """The upstream `config.json` has no `auto_map`, and transformers 5.5.4 has
    no `deepseek_v4`, so without Vaayu's registration `AutoConfig` raises and a
    user's fresh download never loads. The fixture IS the shipped config minus
    `auto_map` (`test_the_fixture_still_matches_the_shipped_config`)."""
    import vaayu.vaayu_deepseek_v4  # noqa: F401 -- registers deepseek_v4
    from transformers import AutoConfig

    (tmp_path / "config.json").write_text(FIXTURE.read_text())

    config = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=True)

    assert type(config) is DeepseekV4Config
    assert getattr(config, "auto_map", None) is None


def test_registering_twice_is_harmless():
    """It runs at every import of the module; a second call must not raise."""
    from vaayu import vaayu_deepseek_v4

    vaayu_deepseek_v4._register_with_transformers()
    vaayu_deepseek_v4._register_with_transformers()


# ── on the checkpoint itself ────────────────────────────────────────────────

CKPT = Path(os.environ.get(
    "VAAYU_DEEPSEEK_V4_CKPT",
    "/home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731"))

_needs_ckpt = pytest.mark.skipif(not CKPT.is_dir(), reason=f"checkpoint not on this box: {CKPT}")


@_needs_ckpt
def test_the_checkpoints_copy_of_this_file_has_not_drifted():
    """No copy is NEEDED: Vaayu registers `DeepseekV4Config` in-process (S5).
    But a checkpoint that still carries one (an older hand-placed `auto_map`)
    loads THAT copy under `trust_remote_code`, so a stale one runs stale code
    silently. Guard it while it exists."""
    copy = CKPT / "configuration_deepseek_v4.py"
    if not copy.exists():
        pytest.skip("no checkpoint-side copy -- none is needed since S5 registers "
                    "DeepseekV4Config in-process")
    repo = Path(__file__).parents[1] / "vaayu/model_code/deepseek_v4/configuration_deepseek_v4.py"

    assert copy.read_bytes() == repo.read_bytes()


@_needs_ckpt
def test_the_fixture_still_matches_the_shipped_config():
    shipped = json.loads((CKPT / "config.json").read_text())
    shipped.pop("auto_map", None)

    assert shipped == json.loads(FIXTURE.read_text())


@_needs_ckpt
def test_auto_config_resolves_deepseek_v4_from_the_checkpoint():
    """The deliverable: `vaayu_base.py:2131` gets a config instead of a raise."""
    import vaayu.vaayu_deepseek_v4  # noqa: F401 -- holds with or without an auto_map
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(CKPT), trust_remote_code=True)

    assert type(config).__name__ == "DeepseekV4Config"
    assert config.model_type == "deepseek_v4"
    assert config.num_hidden_layers == 43
