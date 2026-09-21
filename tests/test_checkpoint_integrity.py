"""G0: a checkpoint that is still downloading must fail loudly.

Row 30's RC5 shipped a `.done` marker over dropped tensors once. The
DeepSeek-V4-Flash checkpoint on this box was 46-of-48 shards with no index for
part of 2026-09-20 -- a download that stopped, not an exotic case.
"""
import json

import pytest

from vaayu.persist.checkpoint_integrity import (
    assert_checkpoint_complete,
    assert_expert_layout_indexable,
)
from vaayu.utils import find_or_create_local_splitted_path, split_and_save_layers


def _shards(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"\0" * 16)


def test_missing_shards_raise(tmp_path):
    _shards(tmp_path, ["model-00001-of-00003.safetensors",
                       "model-00002-of-00003.safetensors"])
    with pytest.raises(RuntimeError) as excinfo:
        assert_checkpoint_complete(str(tmp_path))
    message = str(excinfo.value)
    assert "3" in message and "2" in message
    assert "00003" in message          # names the missing shard


def test_complete_shard_set_passes(tmp_path):
    _shards(tmp_path, [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)])
    assert_checkpoint_complete(str(tmp_path))


def test_index_referencing_a_missing_file_raises(tmp_path):
    _shards(tmp_path, ["model-00001-of-00002.safetensors",
                       "model-00002-of-00002.safetensors"])
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00009-of-00002.safetensors"}}))
    with pytest.raises(RuntimeError, match="model-00009-of-00002.safetensors"):
        assert_checkpoint_complete(str(tmp_path))


def test_unsharded_checkpoint_passes(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 16)
    assert_checkpoint_complete(str(tmp_path))


def test_directory_with_no_safetensors_passes(tmp_path):
    """Not this check's job to decide whether a directory is a checkpoint."""
    assert_checkpoint_complete(str(tmp_path))


# G1: the same layout question, asked BEFORE a split writes or deletes anything.

def test_assert_expert_layout_indexable_passes_for_fused_3d(tmp_path):
    """A2 taught the index this layout, so refusing it here would block a
    checkpoint Vaayu can now read -- at the one seam that deletes originals.

    This test asserted the opposite until A2: the guard and the index must
    answer the same question, and the index's answer changed.
    """
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    keys = ["model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.self_attn.q_proj.weight"]
    assert_expert_layout_indexable(str(tmp_path), keys)   # must not raise


def test_assert_expert_layout_indexable_raises_for_a_layout_neither_path_covers(tmp_path):
    """Experts under no `experts.` prefix at all: still unindexable, still a
    raise, and still BEFORE anything is written or deleted."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    keys = ["model.layers.0.mlp.expert_weights",
            "model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(RuntimeError) as excinfo:
        assert_expert_layout_indexable(str(tmp_path), keys)
    assert "32" in str(excinfo.value)
    assert "expert_weights" in str(excinfo.value)


def test_assert_expert_layout_indexable_passes_for_per_expert_keys(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    keys = ["model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.1.w1.weight"]
    assert_expert_layout_indexable(str(tmp_path), keys)   # must not raise


def test_assert_expert_layout_indexable_passes_for_dense_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 80}))
    keys = ["model.layers.0.self_attn.q_proj.weight"]
    assert_expert_layout_indexable(str(tmp_path), keys)   # must not raise


def test_split_and_save_layers_refuses_an_unindexable_original_before_writing(tmp_path):
    """The finding's actual guarantee: nothing is written before the raise."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.mlp.expert_weights": "model.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        }
    }))
    with pytest.raises(RuntimeError, match="32"):
        split_and_save_layers(str(tmp_path))
    assert not (tmp_path / "splitted_model").exists()


def test_split_and_save_layers_does_not_refuse_a_fused_3d_original(tmp_path):
    """The gpt-oss shape must get PAST the layout guard now that the index
    understands it. It still fails afterwards -- the shard file this fixture
    names does not exist -- but not with the layout message, which is the
    whole point: A2's work is unreachable if this seam keeps refusing."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": "model.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        }
    }))
    with pytest.raises(Exception) as excinfo:
        split_and_save_layers(str(tmp_path))
    assert "per-expert layout" not in str(excinfo.value)


def test_find_or_create_local_splitted_path_raises_g0_for_shards_with_no_index(tmp_path):
    """The finding's motivating case: shards present, no index -- an
    interrupted download -- must raise G0's message, not HFValidationError."""
    (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"\0" * 16)
    (tmp_path / "model-00002-of-00003.safetensors").write_bytes(b"\0" * 16)
    with pytest.raises(RuntimeError) as excinfo:
        find_or_create_local_splitted_path(str(tmp_path))
    message = str(excinfo.value)
    assert "3" in message and "2" in message
