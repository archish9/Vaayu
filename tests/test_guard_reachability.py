"""A1 task 4: both load guards are reachable on the production shape.

Measured defect: Vaayu indexes its own `splitted_model/` directory, which
carries no `config.json` -- the config lives in the ORIGINAL checkpoint dir.
`expected_expert_count()` therefore read `config_path = self.checkpoint_path
/ "config.json"`, found nothing, returned None, and `_assert_experts_found`
returned early on every production load. G0 (`assert_checkpoint_complete`)
was inert for a different reason: it only ever ran against the splitted
output, which has neither `-of-NNNNN` shard names nor an index, so it saw
nothing to check and never inspected the original shard set an interrupted
download truncates.

These tests reproduce the two-directory production shape no prior test does.
"""
import json
import struct

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.utils import split_and_save_layers


def _write_shard(path, tensors: dict):
    """Minimal safetensors file: {key: (dtype, shape, nbytes)}."""
    header, offset = {}, 0
    for key, (dtype, shape, nbytes) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_config_in_a_separate_directory_is_found(tmp_path):
    """The production shape: shards in the splitted dir, config in the
    original checkpoint dir. The expert keys match NEITHER layout, so the
    index comes back empty -- and now there is a config to compare it
    against, so the guard must fire.

    The unindexable shape here used to be the fused-3D one; A2 indexes that,
    so this test moved to a layout still outside both detectors. What it
    covers is the two-directory config lookup, not any particular layout."""
    shards_dir = tmp_path / "splitted_model"
    shards_dir.mkdir()
    config_dir = tmp_path / "original"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    _write_shard(shards_dir / "model.layers.0.safetensors", {
        "model.layers.0.mlp.expert_weights": ("U8", [32, 64, 9, 16], 294912),
    })

    index = ExpertOffsetIndex(str(shards_dir), config_path=str(config_dir))
    with pytest.raises(RuntimeError) as excinfo:
        index.build(write_cache=False)
    assert "32" in str(excinfo.value)


def test_separate_config_directory_does_not_raise_when_experts_are_indexed(tmp_path):
    """Same two-directory split, but the shards hold real per-expert keys --
    the shape both working MoE models are actually in. Must NOT raise."""
    shards_dir = tmp_path / "splitted_model"
    shards_dir.mkdir()
    config_dir = tmp_path / "original"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"num_local_experts": 2}))
    _write_shard(shards_dir / "model.layers.0.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.1.w1.weight": ("BF16", [4, 4], 32),
    })

    index = ExpertOffsetIndex(str(shards_dir), config_path=str(config_dir))
    index.build(write_cache=False)   # must not raise
    assert sorted(index.index["model.layers.0"]) == [0, 1]


def test_config_path_defaults_to_the_checkpoint_directory(tmp_path):
    """Omitting config_path must behave exactly as it does today: config
    and shards in one directory."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 2}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": ("BF16", [4, 4], 32),
        "model.layers.0.mlp.experts.1.w1.weight": ("BF16", [4, 4], 32),
    })

    index = ExpertOffsetIndex(str(tmp_path))
    index.build(write_cache=False)
    assert sorted(index.index["model-00001-of-00001"]) == [0, 1]


def test_build_or_load_raises_on_a_cached_empty_index(tmp_path):
    """A cache written before this guard existed can hold an empty index.
    build() never runs on the cached branch, so build_or_load() must assert
    there too."""
    (tmp_path / "config.json").write_text(json.dumps({"num_local_experts": 32}))
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [32, 64, 9, 16], 294912),
    })
    cache_path = tmp_path / ExpertOffsetIndex.CACHE_FILENAME
    cache_path.write_text(json.dumps({}))
    # Make the cache newer than the shard so load_cached() accepts it.
    shard_mtime = (tmp_path / "model-00001-of-00001.safetensors").stat().st_mtime
    import os
    os.utime(cache_path, (shard_mtime + 10, shard_mtime + 10))

    index = ExpertOffsetIndex(str(tmp_path))
    with pytest.raises(RuntimeError):
        index.build_or_load()


def test_split_and_save_layers_refuses_an_incomplete_checkpoint(tmp_path):
    """G0 must run on the ORIGINAL checkpoint, before splitting -- the
    splitted output has neither -of-NNNNN names nor an index, so nothing
    downstream can ever see this truncation."""
    (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"\0" * 16)
    (tmp_path / "model-00002-of-00003.safetensors").write_bytes(b"\0" * 16)

    with pytest.raises(RuntimeError) as excinfo:
        split_and_save_layers(str(tmp_path))
    assert "00003" in str(excinfo.value)
