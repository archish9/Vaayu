"""`split_and_save_layers(key_remap=...)` -- renaming checkpoint keys BEFORE the split.

DeepSeek-V4-Flash's keys (`layers.3.ffn.experts.7.w1.weight`) do not name the
module tree, and the splitter matches keys by SUBSTRING (`if layer in k`), so a
disk-named `layer_names` would pull 248 layer-internal `*norm` keys into the
`norm` shard. Renaming first is what keeps every consumer downstream -- the
shard names, `ExpertOffsetIndex`, the embed-shape probe at `vaayu_base.py:2141`
-- on the one convention they already speak.

None of this needs a real checkpoint: the fixtures are 2x2 tensors.
"""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from vaayu.utils import split_and_save_layers


def _write_checkpoint(tmp_path, layout):
    """layout: {shard_filename: {tensor_key: tensor}} -> writes shards + index."""
    weight_map = {}
    for fname, tensors in layout.items():
        save_file(tensors, str(tmp_path / fname))
        for k in tensors:
            weight_map[k] = fname
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return weight_map


def _t(v):
    return torch.full((2, 2), float(v))


def _deepseek_like(tmp_path):
    """Two layers named the way DeepSeek-V4 names them, plus an MTP module."""
    return _write_checkpoint(tmp_path, {
        "model-00001-of-00002.safetensors": {
            "embed.weight": _t(1),
            "layers.0.attn_norm.weight": _t(2),
            "layers.0.ffn.experts.0.w1.weight": _t(3),
        },
        "model-00002-of-00002.safetensors": {
            "layers.1.attn_norm.weight": _t(4),
            "norm.weight": _t(5),
            "head.weight": _t(6),
            "mtp.0.ffn.experts.0.w1.weight": _t(7),
        },
    })


def _remap(key):
    if key.startswith("mtp."):
        return None
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    return "model." + key


def test_a_remapped_split_is_written_under_the_new_names(tmp_path):
    _deepseek_like(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False, key_remap=_remap)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    assert set(written) == {
        "model.embed_tokens.weight",
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn.experts.0.w1.weight",
        "model.layers.1.attn_norm.weight",
        "model.norm.weight",
        "lm_head.weight",
    }


def test_the_shard_files_are_named_for_the_modules_not_the_disk(tmp_path):
    _deepseek_like(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False, key_remap=_remap)

    names = {p.name for p in (tmp_path / "splitted_model").glob("*.safetensors")}
    assert names == {
        "model.embed_tokens.safetensors",
        "model.layers.0.safetensors",
        "model.layers.1.safetensors",
        "model.norm.safetensors",
        "lm_head.safetensors",
    }


def test_a_key_mapped_to_none_is_dropped_everywhere(tmp_path):
    """MTP is excluded. It must not reach a shard, and -- the sharper reason --
    it must not reach `ExpertOffsetIndex`, whose `\\.experts\\.(\\d+)\\.` would
    index its 768 experts, or `detect_moe_blocks`, which reads the layer index
    from the FIRST `.N.` and would call `mtp.0.ffn` layer 0."""
    _deepseek_like(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False, key_remap=_remap)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    assert not [k for k in written if "mtp" in k]
    assert not list((tmp_path / "splitted_model").glob("mtp*"))


def test_the_values_survive_the_rename(tmp_path):
    """A rename that loses which tensor it renamed would still pass the name
    assertions above."""
    _deepseek_like(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False, key_remap=_remap)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    assert torch.equal(written["model.embed_tokens.weight"], _t(1))
    assert torch.equal(written["model.layers.1.attn_norm.weight"], _t(4))
    assert torch.equal(written["lm_head.weight"], _t(6))


def test_two_keys_mapping_to_one_name_raise(tmp_path):
    """Silently writing one and losing the other is the failure mode this
    whole pass exists to avoid."""
    _write_checkpoint(tmp_path, {
        "model-00001-of-00001.safetensors": {
            "embed.weight": _t(1),
            "layers.0.a.weight": _t(2),
            "norm.weight": _t(3),
            "head.weight": _t(4),
        },
    })

    with pytest.raises(ValueError, match="model.norm.weight"):
        split_and_save_layers(str(tmp_path), delete_original=False,
                              key_remap=lambda k: "model.norm.weight")


def test_no_remap_leaves_the_split_exactly_as_it_was(tmp_path):
    """The default for every other model on this box."""
    weight_map = _write_checkpoint(tmp_path, {
        "model-00001-of-00001.safetensors": {
            "model.embed_tokens.weight": _t(1),
            "model.layers.0.a.weight": _t(2),
            "model.norm.weight": _t(3),
            "lm_head.weight": _t(4),
        },
    })

    split_and_save_layers(str(tmp_path), delete_original=False)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))
    assert set(written) == set(weight_map)


def test_the_index_left_behind_by_a_delete_carries_the_ORIGINAL_keys(tmp_path):
    """Single-file checkpoints get a synthesised index before deletion
    (`_leave_an_index_behind`). If that index held REMAPPED keys, the next load
    would re-split from already-renamed keys and the remap would raise on roots
    it has never seen. The delete path must see the original index."""
    save_file({"embed.weight": _t(1),
               "layers.0.a.weight": _t(2),
               "norm.weight": _t(3),
               "head.weight": _t(4)},
              str(tmp_path / "model.safetensors"))

    split_and_save_layers(str(tmp_path), delete_original=True, key_remap=_remap)

    left = json.loads(
        (tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(left) == {"embed.weight", "layers.0.a.weight",
                         "norm.weight", "head.weight"}


def test_the_base_hook_defaults_to_no_remap():
    from vaayu.vaayu_base import VaayuBaseModel

    assert VaayuBaseModel.get_checkpoint_key_remap(object()) is None


# Task 4b (G2): a key matching no shard prefix is silently DROPPED by the
# substring assignment loop -- the split still writes every expected shard and
# stamps every .done marker, so completeness looks fine. Found on DeepSeek-V4's
# real, remapped index: 3 of 67,612 keys (`model.hc_head_base/_fn/_scale`)
# matched no prefix. Measured 2026-09-24 against every on-device checkpoint
# with the DEFAULT layer names: Qwen3-Next-80B-A3B leaves 4,625 `mtp.*` keys
# homeless TODAY, unremapped -- so a hard raise unconditionally would break a
# model that works. The guard therefore raises only when a key_remap ran, and
# otherwise warns (CLAUDE.md 5).

def _well_formed_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path, {
        "model-00001-of-00001.safetensors": {
            "model.embed_tokens.weight": _t(1),
            "model.layers.0.a.weight": _t(2),
            "model.norm.weight": _t(3),
            "lm_head.weight": _t(4),
        },
    })


def _checkpoint_with_a_homeless_key(tmp_path):
    return _write_checkpoint(tmp_path, {
        "model-00001-of-00001.safetensors": {
            "model.embed_tokens.weight": _t(1),
            "model.layers.0.a.weight": _t(2),
            "model.norm.weight": _t(3),
            "lm_head.weight": _t(4),
            "homeless.weight": _t(5),
        },
    })


def test_a_homeless_key_raises_when_a_key_remap_ran(tmp_path):
    """A caller that took the trouble to rename every root (DeepSeek-V4's
    `remap_checkpoint_key`) means every renamed key to land somewhere -- a
    homeless one is a real bug, not a pre-existing gap."""
    _checkpoint_with_a_homeless_key(tmp_path)

    with pytest.raises(RuntimeError, match="homeless.weight"):
        split_and_save_layers(str(tmp_path), delete_original=False,
                              key_remap=lambda k: k)


def test_a_homeless_key_only_warns_with_no_key_remap(tmp_path, capsys):
    """Absent a key_remap, this is a pre-existing gap this guard did not
    create (Qwen3-Next's own unremapped `mtp.*` keys) -- warn, don't raise."""
    _checkpoint_with_a_homeless_key(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "homeless.weight" in out

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))
    assert "homeless.weight" not in written


def test_a_well_formed_checkpoint_splits_cleanly_with_no_warning(tmp_path, capsys):
    weight_map = _well_formed_checkpoint(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False)

    assert "match no shard prefix" not in capsys.readouterr().out

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))
    assert set(written) == set(weight_map)


def test_a_homeless_key_raises_before_the_already_split_early_return_can_delete(tmp_path):
    """Review finding (2026-09-24): the guard must run before EITHER
    `delete_original_shards` call site, not just before the write loop. A
    checkpoint split by an earlier run that silently dropped a homeless key
    presents every expected shard AND every `.done` marker -- exactly what
    `all(found_layers.values())` in the "already split" early return calls
    complete. Fabricate exactly that state, then load again with
    `delete_original=True` and a `key_remap` in play: the guard must raise
    before that branch's `delete_original_shards(...)` call, so the untouched
    originals are still on disk afterwards."""
    weight_map = _checkpoint_with_a_homeless_key(tmp_path)
    original_files = {tmp_path / fname for fname in weight_map.values()}
    original_index = tmp_path / "model.safetensors.index.json"
    assert all(f.exists() for f in original_files)

    # The state an earlier run that silently dropped "homeless.weight" would
    # leave: every OTHER expected shard, each with its .done marker. Content
    # does not matter -- G2 must fire before anything downstream reads these.
    split_dir = tmp_path / "splitted_model"
    split_dir.mkdir()
    for layer in ("model.embed_tokens", "model.layers.0", "model.norm", "lm_head"):
        (split_dir / f"{layer}.safetensors").touch()
        (split_dir / f"{layer}.safetensors.done").touch()

    with pytest.raises(RuntimeError, match="homeless.weight"):
        split_and_save_layers(str(tmp_path), delete_original=True,
                              key_remap=lambda k: k)

    assert all(f.exists() for f in original_files), (
        "the homeless-key guard ran too late: the original shard(s) were "
        "deleted before it fired")
    assert original_index.exists(), (
        "the homeless-key guard ran too late: the original index was "
        "deleted before it fired")
