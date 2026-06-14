import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.persist.safetensor_model_persister import load_expert_sparse


def make_shard(tmp_path: Path, layer_name: str = "model.layers.0") -> Path:
    tensors = {
        f"{layer_name}.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{layer_name}.block_sparse_moe.gate.weight": torch.randn(4, 8),
        f"{layer_name}.block_sparse_moe.experts.0.w1.weight": torch.ones(8, 8) * 1.0,
        f"{layer_name}.block_sparse_moe.experts.0.w2.weight": torch.ones(8, 8) * 2.0,
        f"{layer_name}.block_sparse_moe.experts.1.w1.weight": torch.ones(8, 8) * 3.0,
        f"{layer_name}.block_sparse_moe.experts.1.w2.weight": torch.ones(8, 8) * 4.0,
    }
    shard = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard))
    return shard


def test_load_expert_sparse_correct_values(tmp_path):
    layer_name = "model.layers.0"
    shard = make_shard(tmp_path, layer_name)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    metas = idx.get_expert_offsets(layer_name, 0)
    assert metas is not None

    result = load_expert_sparse(str(shard), metas, use_direct_io=False)
    assert f"{layer_name}.block_sparse_moe.experts.0.w1.weight" in result
    t = result[f"{layer_name}.block_sparse_moe.experts.0.w1.weight"]
    assert t.shape == (8, 8)
    assert abs(t[0, 0].item() - 1.0) < 1e-3


def test_load_expert_sparse_two_experts(tmp_path):
    layer_name = "model.layers.0"
    shard = make_shard(tmp_path, layer_name)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    metas0 = idx.get_expert_offsets(layer_name, 0)
    metas1 = idx.get_expert_offsets(layer_name, 1)

    # Merge both experts' metas into one call
    merged = {**metas0, **metas1}
    result = load_expert_sparse(str(shard), merged, use_direct_io=False)

    key0 = f"{layer_name}.block_sparse_moe.experts.0.w1.weight"
    key1 = f"{layer_name}.block_sparse_moe.experts.1.w1.weight"
    assert key0 in result and key1 in result
    assert abs(result[key0][0, 0].item() - 1.0) < 1e-3
    assert abs(result[key1][0, 0].item() - 3.0) < 1e-3


def test_load_expert_sparse_empty_metas(tmp_path):
    shard = make_shard(tmp_path)
    result = load_expert_sparse(str(shard), {}, use_direct_io=False)
    assert result == {}
