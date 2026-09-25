"""Regression tests for the F8_E8M0 dtype gap in `_DTYPE_MAP`.

DeepSeek-V4-Flash's layer-0 shard carries 776 `F8_E8M0` tensors (MXFP8-style
block scales, e.g. `model.layers.0.attn.wkv.scale`, shape [4, 32], 128 bytes).
`_DTYPE_MAP` mapped every other on-disk dtype but not this one, so every
reader's `_DTYPE_MAP.get(meta['dtype'], torch.float32)` silently reinterpreted
those 128 bytes as float32 (32 elements instead of 128), which only surfaced
200 lines away as `RuntimeError: shape '[4, 32]' is invalid for input of size
32` -- a confusing reshape error, not a "this dtype is unknown" error.

Fix has two parts:
  1. Map F8_E8M0 -> torch.float8_e8m0fnu (same guarded-hasattr pattern as the
     existing F8_E4M3 / F8_E5M2 entries).
  2. Replace the silent `.get(..., torch.float32)` fallback at all five call
     sites with a shared resolver that RAISES, naming the unmapped dtype
     string, the tensor name, and the file -- so the NEXT unmapped dtype
     reports itself instead of presenting as a shape error.

No checkpoint and no GPU needed -- these build tiny safetensors files with
safetensors.torch.save_file and read them back through the pinned loader
(the path load_layer uses in production, see SafetensorModelPersister.load_model).
"""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from vaayu.persist import safetensor_model_persister as smp
from vaayu.persist.model_persister import ModelPersister


def _e8m0(shape) -> torch.Tensor:
    """A float8_e8m0fnu tensor with non-trivial (non-all-zero) byte content,
    built by viewing random uint8 bytes -- e8m0 has no arithmetic constructor
    worth relying on, but every byte pattern is a valid encoding for it."""
    n = 1
    for d in shape:
        n *= d
    raw = torch.randint(0, 256, (n,), dtype=torch.uint8)
    return raw.view(torch.float8_e8m0fnu).reshape(shape)


def test_e8m0_scale_tensor_round_trips_shape_and_dtype(tmp_path):
    """The regression, at the real shape: model.layers.0.attn.wkv.scale is
    [4, 32] F8_E8M0, 128 bytes on disk. Read through the pinned loader (the
    path load_layer uses), it must come back as [4, 32] float8_e8m0fnu --
    not silently reinterpreted as a 32-element float32 tensor."""
    layer_name = "model.layers.0"
    key = f"{layer_name}.attn.wkv.scale"
    tensors = {key: _e8m0((4, 32))}
    save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))

    result = ModelPersister.get_model_persister().load_model(layer_name, str(tmp_path))

    assert key in result
    assert result[key].shape == torch.Size([4, 32])
    assert result[key].dtype == torch.float8_e8m0fnu


def test_e8m0_values_survive_not_just_shape(tmp_path):
    """A reader that guessed the wrong dtype could still produce a
    right-shaped tensor by coincidence for some other geometry; compare the
    actual bytes against what safetensors.torch.load_file returns for the
    same file (bitwise, via uint8 view -- float8 comparison ops are awkward)."""
    layer_name = "model.layers.0"
    key = f"{layer_name}.attn.wkv.scale"
    tensors = {key: _e8m0((4, 32))}
    shard = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard))

    expected = load_file(str(shard))[key]
    result = ModelPersister.get_model_persister().load_model(layer_name, str(tmp_path))
    actual = result[key]

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_mixed_dtype_shard_round_trips(tmp_path):
    """The real layer-0 shard shape: F8_E8M0, F8_E4M3, I8, BF16 and F32
    tensors together in one file. Every dtype must come back correctly,
    not just the one under test in isolation."""
    layer_name = "model.layers.0"
    tensors = {
        f"{layer_name}.attn.wkv.scale": _e8m0((4, 32)),
        f"{layer_name}.attn.wkv.weight": torch.randint(
            -128, 127, (8, 8), dtype=torch.int8
        ),
        f"{layer_name}.mlp.gate.weight_scale": torch.randn(4, 8).to(torch.float8_e4m3fn),
        f"{layer_name}.input_layernorm.weight": torch.randn(8).to(torch.bfloat16),
        f"{layer_name}.self_attn.q_proj.bias": torch.randn(8, dtype=torch.float32),
    }
    shard = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard))

    expected = load_file(str(shard))
    result = ModelPersister.get_model_persister().load_model(layer_name, str(tmp_path))

    assert set(result.keys()) == set(expected.keys())
    for key, exp_t in expected.items():
        act_t = result[key]
        assert act_t.dtype == exp_t.dtype, key
        assert act_t.shape == exp_t.shape, key
        assert torch.equal(act_t.view(torch.uint8), exp_t.view(torch.uint8)), key


def test_unmapped_dtype_raises_and_names_itself(tmp_path, monkeypatch):
    """Part 2: an unknown dtype must raise, not silently become float32.
    Simulate "unmapped" by deleting a real entry from _DTYPE_MAP and
    confirm the raise, through a real call site (_load_pinned), names both
    the offending dtype string and the tensor whose read hit it."""
    layer_name = "model.layers.0"
    key = f"{layer_name}.attn.wkv.scale"
    tensors = {key: _e8m0((4, 32))}
    shard = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard))

    monkeypatch.delitem(smp._DTYPE_MAP, "F8_E8M0", raising=False)

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        smp._load_pinned(str(shard))

    msg = str(exc_info.value)
    assert "F8_E8M0" in msg
    assert key in msg
