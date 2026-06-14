import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vaayu.persist.expert_offset_index import ExpertOffsetIndex, ExpertTensorMeta


def make_synthetic_moe_shard(tmp_path: Path, layer_name: str, num_experts: int = 4) -> Path:
    """Write a minimal safetensors shard containing expert tensors."""
    tensors = {}
    tensors[f"{layer_name}.self_attn.q_proj.weight"] = torch.randn(8, 8)
    tensors[f"{layer_name}.block_sparse_moe.gate.weight"] = torch.randn(4, 8)
    for eidx in range(num_experts):
        tensors[f"{layer_name}.block_sparse_moe.experts.{eidx}.w1.weight"] = torch.randn(8, 8)
        tensors[f"{layer_name}.block_sparse_moe.experts.{eidx}.w2.weight"] = torch.randn(8, 8)
    shard_path = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard_path))
    return shard_path


def test_build_finds_expert_entries(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=4)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    offsets = idx.get_expert_offsets("model.layers.0", 0)
    assert offsets is not None
    assert len(offsets) == 2  # w1 and w2


def test_expert_meta_fields(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    offsets = idx.get_expert_offsets("model.layers.0", 0)
    for key, meta in offsets.items():
        assert isinstance(meta, ExpertTensorMeta)
        assert meta.file_offset > 0
        assert meta.byte_length > 0
        assert meta.shape == [8, 8]
        assert meta.dtype == "F32"


def test_non_expert_tensors_excluded(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    # Gate weights should not appear in expert offsets
    for eidx_tensors in idx.index.get("model.layers.0", {}).values():
        for key in eidx_tensors:
            assert ".gate." not in key or ".experts." in key


def test_moe_layer_names(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    make_synthetic_moe_shard(tmp_path, "model.layers.1", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    names = idx.moe_layer_names()
    assert "model.layers.0" in names
    assert "model.layers.1" in names


def test_json_cache_written(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache_file = tmp_path / "vaayu_expert_offsets.json"
    assert cache_file.exists()
    with open(cache_file) as f:
        data = json.load(f)
    assert "model.layers.0" in data


def test_load_cached(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    idx2 = ExpertOffsetIndex(str(tmp_path))
    loaded = idx2.load_cached()
    assert loaded is True
    assert idx2.get_expert_offsets("model.layers.0", 0) is not None


def test_stale_cache_triggers_rebuild(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    # Make cache appear older than shard by writing stale JSON
    cache_file = tmp_path / "vaayu_expert_offsets.json"
    import time; time.sleep(0.05)
    # Touch shard to make it newer than cache
    shard = tmp_path / "model.layers.0.safetensors"
    shard.touch()

    idx2 = ExpertOffsetIndex(str(tmp_path))
    loaded = idx2.load_cached()
    assert loaded is False  # cache stale — must rebuild


def test_missing_expert_returns_none(tmp_path):
    make_synthetic_moe_shard(tmp_path, "model.layers.0", num_experts=2)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    # Expert index 99 doesn't exist
    assert idx.get_expert_offsets("model.layers.0", 99) is None
    # Layer that doesn't exist
    assert idx.get_expert_offsets("model.layers.99", 0) is None
