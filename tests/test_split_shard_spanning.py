"""A layer's tensors may live in ANY set of shards, in any order.

`split_and_save_layers` used to load, per layer, only `max(shard number)` and
only when that number differed from the shard it already held. That assumes the
checkpoint index is layer-major and monotonic: every shard below the maximum was
supposedly pulled in by an earlier layer. Modern quantized MoE checkpoints break
the assumption -- one layer's tensors are spread over several NON-ADJACENT shards
and the shard numbers do not rise with the layer index.

When the assumption fails the loop writes the layer anyway, with the tensors from
the un-loaded shards simply missing, and stamps the `.done` marker beside it. The
completeness check then reports the split as good. Measured on
`cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`: 24356 of 24576 experts written,
two layers short, 41.57 GB instead of ~48.2 -- with no error anywhere.
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


def test_layer_spanning_non_adjacent_shards_loses_no_tensor(tmp_path):
    """Layer 1 lives in shards 1 and 3; nothing before it ever needed shard 1.

    `max(1, 3) == 3` and shard 3 is already resident from layer 0, so the old
    loop loaded nothing and dropped every layer-1 tensor that lived in shard 1.
    """
    layout = {
        "model-00001-of-00003.safetensors": {"model.layers.1.a.weight": _t(1)},
        "model-00002-of-00003.safetensors": {"model.norm.weight": _t(2)},
        "model-00003-of-00003.safetensors": {
            "model.embed_tokens.weight": _t(3),
            "model.layers.0.a.weight": _t(4),
            "model.layers.1.b.weight": _t(5),
            "lm_head.weight": _t(6),
        },
    }
    weight_map = _write_checkpoint(tmp_path, layout)

    saved = split_and_save_layers(str(tmp_path), delete_original=False)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    missing = sorted(set(weight_map) - set(written))
    assert not missing, f"tensors dropped by the split: {missing}"


def test_every_tensor_survives_a_scrambled_shard_order(tmp_path):
    """Shard numbers that do not rise with the layer index, layers spanning 3.

    This is the shape the 80B checkpoint actually has (its load order was
    1, 4, 7, 3, 9, 6, 2, 8, 5), and it exercises the reverse case too: a layer
    whose only shard is numerically BELOW one already held.
    """
    layout = {
        "model-00001-of-00004.safetensors": {
            "model.embed_tokens.weight": _t(0),
            "model.layers.2.a.weight": _t(1),
        },
        "model-00002-of-00004.safetensors": {
            "model.layers.0.a.weight": _t(2),
            "model.layers.3.a.weight": _t(3),
        },
        "model-00003-of-00004.safetensors": {
            "model.layers.1.a.weight": _t(4),
            "model.layers.2.b.weight": _t(5),
            "model.norm.weight": _t(6),
        },
        "model-00004-of-00004.safetensors": {
            "model.layers.0.b.weight": _t(7),
            "model.layers.2.c.weight": _t(8),
            "lm_head.weight": _t(9),
        },
    }
    weight_map = _write_checkpoint(tmp_path, layout)

    saved = split_and_save_layers(str(tmp_path), delete_original=False)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    assert sorted(written) == sorted(weight_map)
    for key, fname in weight_map.items():
        expected = load_file(str(tmp_path / fname))[key]
        assert torch.equal(written[key], expected), f"{key} has the wrong values"


def test_a_layer_key_is_not_confused_with_a_same_suffixed_group(tmp_path):
    """`mtp.layers.0.*` must not be swept into `model.layers.0.*`.

    The 80B checkpoint ships an MTP draft module the causal-LM never builds; its
    keys sit under their own top-level prefix and must be left out of the split.
    """
    layout = {
        "model-00001-of-00002.safetensors": {
            "model.embed_tokens.weight": _t(0),
            "model.layers.0.a.weight": _t(1),
            "mtp.layers.0.a.weight": _t(2),
        },
        "model-00002-of-00002.safetensors": {
            "model.norm.weight": _t(3),
            "lm_head.weight": _t(4),
        },
    }
    _write_checkpoint(tmp_path, layout)

    split_and_save_layers(str(tmp_path), delete_original=False)

    written = {}
    for shard in sorted((tmp_path / "splitted_model").glob("*.safetensors")):
        written.update(load_file(str(shard)))

    assert "mtp.layers.0.a.weight" not in written
    assert "model.layers.0.a.weight" in written
