"""G0: a checkpoint that is still downloading must fail loudly.

Row 30's RC5 shipped a `.done` marker over dropped tensors once. The
DeepSeek-V4-Flash checkpoint on this box was 46-of-48 shards with no index for
part of 2026-09-20 -- a download that stopped, not an exotic case.
"""
import json

import pytest

from vaayu.persist.checkpoint_integrity import assert_checkpoint_complete


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
