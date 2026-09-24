"""A5.7b -- reading DeepSeek-V4-Flash's checkpoint: key names and fp4 group size.

The mapping half needs neither the 157 G checkpoint nor a GPU; it is a pure
function over strings. The one test that walks the real index is gated.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import checkpoint
from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")


@pytest.mark.parametrize("disk,module", [
    ("embed.weight", "model.embed_tokens.weight"),
    ("layers.0.attn.wq_a.weight", "model.layers.0.attn.wq_a.weight"),
    ("layers.42.ffn.experts.255.w1.scale", "model.layers.42.ffn.experts.255.w1.scale"),
    ("norm.weight", "model.norm.weight"),
    ("hc_head_fn", "model.hc_head_fn"),
    ("hc_head_base", "model.hc_head_base"),
    ("hc_head_scale", "model.hc_head_scale"),
    ("head.weight", "lm_head.weight"),
])
def test_every_kept_root_maps_to_its_module_path(disk, module):
    assert checkpoint.remap_checkpoint_key(disk) == module


@pytest.mark.parametrize("key", [
    "mtp.0.ffn.experts.0.w1.weight",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.1.attn.wkv.scale",
])
def test_the_mtp_draft_module_is_dropped(key):
    """MTP is excluded (user, 2026-09-23). Not merely 10.117 GiB of waste:
    `detect_moe_blocks` reads the layer index from the first `.N.` in the path,
    so `mtp.0.ffn` collides with the real layer 0."""
    assert checkpoint.remap_checkpoint_key(key) is None


def test_an_unrecognised_root_raises_rather_than_vanishing():
    """A dropped key is a silently missing weight. Only `mtp` is dropped, and
    only because someone decided to."""
    with pytest.raises(ValueError, match="somethingelse"):
        checkpoint.remap_checkpoint_key("somethingelse.weight")


def test_lm_head_lands_outside_model_by_hf_convention():
    """`vaayu/utils.py:359` builds its layer list with a bare `lm_head.`, and
    `drop_absent_lm_head` looks for that name."""
    assert not checkpoint.remap_checkpoint_key("head.weight").startswith("model.")


def test_the_subclass_returns_the_mapping():
    assert (VaayuDeepseekV4.get_checkpoint_key_remap(object())
            is checkpoint.remap_checkpoint_key)


@needs_checkpoint
def test_the_mapping_covers_every_key_in_the_real_index():
    """The whole checkpoint, headers only -- 72,317 tensors, of which 4,705 are
    `mtp`. Zero unmapped: a root nobody thought about would raise here rather
    than during a multi-hour split."""
    weight_map = json.loads(
        (parity.CKPT / "model.safetensors.index.json").read_text())["weight_map"]

    mapped, dropped = {}, 0
    for key in weight_map:
        new = checkpoint.remap_checkpoint_key(key)
        if new is None:
            dropped += 1
        else:
            assert new not in mapped, f"{key} and {mapped[new]} both map to {new}"
            mapped[new] = key

    assert len(weight_map) == 72317
    assert len(mapped) == 67612
    assert dropped == 4705


def _fake_split_layer(tmp_path, in_features=64, scale_last=32):
    """One split layer shard carrying a routed expert's weight/scale pair.

    fp4 packs two values per byte, so `weight`'s last dim is `in_features // 2`
    and the group size is `in_features // scale_last`.
    """
    save_file(
        {"model.layers.0.ffn.experts.0.w1.weight":
            torch.zeros(8, in_features // 2, dtype=torch.uint8),
         "model.layers.0.ffn.experts.0.w1.scale":
            torch.zeros(8, scale_last, dtype=torch.uint8)},
        str(tmp_path / "model.layers.0.safetensors"))
    return tmp_path


def test_the_group_size_is_derived_from_a_weight_scale_pair(tmp_path):
    _fake_split_layer(tmp_path, in_features=64, scale_last=32)
    assert checkpoint.fp4_group_size_from_split(tmp_path) == 2


def test_the_group_size_is_not_a_literal(tmp_path):
    """Second geometry, same code path (CLAUDE.md §5: derive, never hardcode)."""
    _fake_split_layer(tmp_path, in_features=2048, scale_last=64)
    assert checkpoint.fp4_group_size_from_split(tmp_path) == 32


def test_a_split_with_no_expert_pair_names_what_it_could_not_find(tmp_path):
    save_file({"model.layers.0.attn.wq_a.weight": torch.zeros(4, 4)},
              str(tmp_path / "model.layers.0.safetensors"))
    with pytest.raises(ValueError, match="fp4_group_size"):
        checkpoint.fp4_group_size_from_split(tmp_path)


def test_a_missing_layer_shard_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.layers.0"):
        checkpoint.fp4_group_size_from_split(tmp_path)


@needs_checkpoint
def test_the_real_checkpoint_measures_32():
    """The value A5.4's skeleton refuses to invent. Read off the ORIGINAL
    shards here, because the split may not exist yet when this runs; the
    loader reads the same quantity off the split ones."""
    weight_map = json.loads(
        (parity.CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    key = next(k for k in weight_map
               if k.endswith("w1.weight") and ".ffn.experts." in k
               and not k.startswith("mtp."))
    from safetensors import safe_open
    shapes = {}
    for name in (key, key.replace(".weight", ".scale")):
        with safe_open(parity.CKPT / weight_map[name], framework="pt") as handle:
            shapes[name] = handle.get_slice(name).get_shape()
    assert shapes[key][-1] * 2 // shapes[key.replace(".weight", ".scale")][-1] == 32


def test_init_model_sets_the_group_size_before_calling_super(monkeypatch):
    """A5.4's skeleton raises a named ValueError if the config has no
    `fp4_group_size`, so it has to be on the config BEFORE `from_config`."""
    seen = {}

    class Config:
        fp4_group_size = None

    def fake_init_model(self):
        seen["at_super"] = self.config.fp4_group_size

    monkeypatch.setattr("vaayu.vaayu_base.VaayuBaseModel.init_model", fake_init_model)
    monkeypatch.setattr(checkpoint, "fp4_group_size_from_split", lambda *a, **k: 32)

    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    model.config = Config()
    model.checkpoint_path = "/some/split"
    model.layer_names_dict = {"layer_prefix": "model.layers"}
    model.init_model()

    assert seen["at_super"] == 32
