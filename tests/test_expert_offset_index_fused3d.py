"""A2: checkpoints that fuse a layer's experts into one tensor.

gpt-oss-20b stores `model.layers.N.mlp.experts.gate_up_proj_blocks` of shape
[32, 5760, 90, 16] -- all 32 experts on a leading axis -- so the `.experts.N.`
regex matches nothing. Detection is by capability: an `experts.` prefix with no
numeric component, and a leading dimension equal to the config's expert count.
Both conditions, so a coincidentally-3D tensor is not captured.
"""
import json
import struct

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex


@pytest.mark.parametrize("key,shape,expected,want", [
    # gpt-oss shape: fused, leading axis is the expert count
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [32, 5760, 90, 16], 32, True),
    ("model.layers.0.mlp.experts.down_proj_scales", [32, 2880, 90], 32, True),
    ("model.layers.0.mlp.experts.gate_up_proj_bias", [32, 5760], 32, True),
    # a DIFFERENT fused geometry -- nothing here may be gpt-oss-specific
    ("model.layers.3.block_sparse_moe.experts.w1", [8, 1024, 512], 8, True),
    # per-expert layout: NOT fused, the existing path owns it
    ("model.layers.0.mlp.experts.0.w1.weight", [4096, 4096], 32, False),
    # leading dim does not match the expert count -> not the expert axis
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [16, 5760, 90], 32, False),
    # not under an experts prefix at all
    ("model.layers.0.self_attn.q_proj.weight", [32, 4096], 32, False),
    # no expectation available -> cannot claim the leading axis means experts
    ("model.layers.0.mlp.experts.gate_up_proj_blocks", [32, 5760, 90], None, False),
    # 1-D tensor under an experts prefix: no axis to split
    ("model.layers.0.mlp.experts.gate_up_proj_bias", [32], 32, False),
])
def test_fused_expert_axis_detection(key, shape, expected, want):
    assert ExpertOffsetIndex._fused_expert_axis(key, shape, expected) is want


def _write_shard(path, tensors):
    """Minimal safetensors file: {key: (dtype, shape, nbytes)}."""
    header, offset = {}, 0
    for key, (dtype, shape, nbytes) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return 8 + len(blob)          # where tensor data starts


@pytest.mark.parametrize("experts,rows,cols", [(32, 64, 16), (8, 128, 4)])
def test_fused_tensor_yields_one_contiguous_range_per_expert(tmp_path, experts, rows, cols):
    per_expert = rows * cols
    total = experts * per_expert
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": experts}))
    data_start = _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks":
            ("U8", [experts, rows, cols], total),
    })

    index = ExpertOffsetIndex(str(tmp_path))
    index.build(write_cache=False)
    shard = index.index["model-00001-of-00001"]

    assert sorted(shard) == list(range(experts))
    key = "model.layers.0.mlp.experts.gate_up_proj_blocks"
    for i in range(experts):
        meta = shard[i][key]
        assert meta.byte_length == per_expert
        assert meta.file_offset == data_start + i * per_expert
        assert meta.shape == [rows, cols]          # expert axis consumed
        assert meta.dtype == "U8"

    # Contiguous and gapless: the slices tile the whole tensor exactly.
    spans = sorted((shard[i][key].file_offset, shard[i][key].byte_length)
                   for i in range(experts))
    assert spans[0][0] == data_start
    for (off_a, len_a), (off_b, _) in zip(spans, spans[1:]):
        assert off_a + len_a == off_b
    assert spans[-1][0] + spans[-1][1] == data_start + total


def test_whole_tensor_meta_is_kept_for_layer_granular_consumers(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 4}))
    data_start = _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.down_proj_blocks": ("U8", [4, 8, 2], 64),
    })
    index = ExpertOffsetIndex(str(tmp_path))
    index.build(write_cache=False)
    whole = index.fused_layer_tensors["model-00001-of-00001"][
        "model.layers.0.mlp.experts.down_proj_blocks"]
    assert whole.file_offset == data_start
    assert whole.byte_length == 64
    assert whole.shape == [4, 8, 2]


def test_a_fused_checkpoint_no_longer_raises(tmp_path):
    """A1's guard fired on exactly this shape; now the layout is understood."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 4}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [4, 8, 2], 64),
    })
    ExpertOffsetIndex(str(tmp_path)).build(write_cache=False)   # must not raise


def test_per_expert_checkpoints_are_untouched(tmp_path):
    """Regression guard: Qwen3-Next, MiniMax and DeepSeek all use this layout."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 2}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.1.w1.weight": ("BF16", [4, 4], 32),
    })
    index = ExpertOffsetIndex(str(tmp_path))
    index.build(write_cache=False)
    shard = index.index["model-00001-of-00001"]
    assert sorted(shard) == [0, 1]
    assert shard[0]["model.layers.0.mlp.experts.0.w1.weight"].shape == [4, 4]
    assert index.fused_layer_tensors == {}


def test_a_fused_tensor_that_does_not_divide_by_the_expert_count_raises(tmp_path):
    """The leading axis was not the expert axis, or the config is wrong.

    Truncating the last expert silently is the same class of bug row 47 exists
    to remove, so this is an error, not a floor division.
    """
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 3}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [3, 5, 1], 16),
    })
    with pytest.raises(RuntimeError, match="does not divide evenly"):
        ExpertOffsetIndex(str(tmp_path)).build(write_cache=False)


def test_cached_index_round_trips_a_fused_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 4}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [4, 8, 2], 64),
    })
    first = ExpertOffsetIndex(str(tmp_path))
    first.build(write_cache=True)

    second = ExpertOffsetIndex(str(tmp_path))
    assert second.load_cached() is True
    assert sorted(second.index["model-00001-of-00001"]) == [0, 1, 2, 3]
    assert second.fused_layer_tensors["model-00001-of-00001"], (
        "a reloaded fused index lost its whole-tensor metas")


def test_a_cache_written_before_this_change_still_loads(tmp_path):
    """Back-compat: the caches on disk today carry no fused section.

    Qwen3-Next's cache holds 24576 experts and MiniMax's 15872; invalidating
    them would cost a full header re-scan on the next load of every model that
    works today, for a layout they do not have.
    """
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 2}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.1.w1.weight": ("BF16", [4, 4], 32),
    })
    (tmp_path / ExpertOffsetIndex.CACHE_FILENAME).write_text(json.dumps({
        "model-00001-of-00001": {"0": {"model.layers.0.mlp.experts.0.w1.weight": {
            "file_offset": 200, "byte_length": 32, "shape": [4, 4], "dtype": "BF16"}}},
    }))

    index = ExpertOffsetIndex(str(tmp_path))
    assert index.load_cached() is True
    assert index.index["model-00001-of-00001"][0][
        "model.layers.0.mlp.experts.0.w1.weight"].file_offset == 200
    assert index.fused_layer_tensors == {}


def test_fused_layout_indexes_in_the_two_directory_production_shape(tmp_path):
    """Shards in splitted_model/, config in the original dir -- the shape
    A1 task 4 showed is the only one production actually takes. Detection
    needs the expert count, so it needs that config lookup to work.
    """
    shards_dir = tmp_path / "splitted_model"
    shards_dir.mkdir()
    config_dir = tmp_path / "original"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"num_local_experts": 4}))
    _write_shard(shards_dir / "model.layers.0.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [4, 8, 2], 64),
    })

    index = ExpertOffsetIndex(str(shards_dir), config_path=str(config_dir))
    index.build(write_cache=False)   # must not raise
    assert sorted(index.index["model.layers.0"]) == [0, 1, 2, 3]
    assert index.index["model.layers.0"][2][
        "model.layers.0.mlp.experts.gate_up_proj_blocks"].byte_length == 16
