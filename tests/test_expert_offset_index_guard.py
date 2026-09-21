"""A1: the expert-offset index must know what it SHOULD have found.

Without an expected count there is nothing to compare an empty index against,
which is why a fused-3D checkpoint could index zero experts and say nothing.
The field name differs per model family and substring matching is a trap --
`num_experts_per_tok` is the top-k, not the total (10 on Qwen3-Next), and
`shared_expert_intermediate_size` is a width. Both would be captured by a
naive scan and both are wrong.
"""
import json

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex


def _ckpt(tmp_path, config: dict | None):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    return ExpertOffsetIndex(str(tmp_path))


@pytest.mark.parametrize("config,expected", [
    ({"num_experts": 512, "num_experts_per_tok": 10,
      "shared_expert_intermediate_size": 512}, 512),   # Qwen3-Next shape
    ({"num_local_experts": 256, "num_experts_per_tok": 8}, 256),   # MiniMax shape
    ({"num_local_experts": 32, "num_experts_per_tok": 4,
      "experts_per_token": 4}, 32),                    # gpt-oss shape
    ({"n_routed_experts": 256, "n_shared_experts": 1,
      "num_experts_per_tok": 6}, 256),                 # DeepSeek shape
])
def test_expected_expert_count_reads_the_total_not_the_topk(tmp_path, config, expected):
    assert _ckpt(tmp_path, config).expected_expert_count() == expected


def test_dense_config_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, {"num_hidden_layers": 80}).expected_expert_count() is None


def test_missing_config_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, None).expected_expert_count() is None


def test_zero_or_negative_count_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, {"num_experts": 0}).expected_expert_count() is None


import struct


def _write_shard(path, tensors: dict):
    """Minimal safetensors file: {key: (dtype, shape, nbytes)}."""
    header, offset = {}, 0
    for key, (dtype, shape, nbytes) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_build_raises_when_config_implies_experts_and_none_were_found(tmp_path):
    """A layout NEITHER detector covers: no `.experts.<N>.` key, and no tensor
    under an `experts.` prefix either, so both paths index nothing.

    This was written against the gpt-oss fused shape, which A2 now indexes --
    so the case moved to one still outside both layouts. The guard's question
    is unchanged and deliberately layout-agnostic: config says E, index holds
    zero.
    """
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.expert_weights": ("U8", [32, 64, 9, 16], 294912),
        "model.layers.0.mlp.expert_scales": ("U8", [32, 64, 9], 18432),
    })
    index = ExpertOffsetIndex(str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        index.build(write_cache=False)
    message = str(excinfo.value)
    assert "32" in message                                   # what was expected
    assert "expert_weights" in message                        # what was actually seen
    assert str(tmp_path) in message                           # which checkpoint


def test_build_does_not_raise_for_a_dense_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8], 128),
    })
    ExpertOffsetIndex(str(tmp_path)).build(write_cache=False)   # must not raise


def test_build_still_indexes_a_per_expert_checkpoint(tmp_path):
    """Regression guard for every model on disk that works today."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 2}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.1.w1.weight": ("BF16", [4, 4], 32),
    })
    index = ExpertOffsetIndex(str(tmp_path))
    index.build(write_cache=False)
    shard = index.index["model-00001-of-00001"]
    assert sorted(shard) == [0, 1]
