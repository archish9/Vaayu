"""`compression` is REFUSED at load -- warn and disable -- where it cannot work (S10b).

`compression="4bit"|"8bit"` re-splits the checkpoint through bitsandbytes'
`quantize_nf4` / `quantize_blockwise` over EVERY tensor (`compress_layer_state_dict`).
Measured 2026-09-28:
- an already-quantized checkpoint crashes the split at its first packed tensor
  (bnb 0.49.2 raises on int32 / uint8 / int8 / fp8, both modes; gpt-oss-20b at
  layer 0), leaving a partial split dir behind;
- a checkpoint whose original shards are gone (split earlier, then deleted) dies
  on a bare AssertionError in the splitter's `_fetch` (Qwen3-Next);
- a MoE checkpoint streams its experts through the offset index, which reads raw
  split bytes and never decompresses (static read, not measured).
On an unquantized dense checkpoint both modes run (Llama-3.2-1B).

The decision is a pure function of the ORIGINAL shards' safetensors headers and
config.json, so it is testable without loading a model.
"""
import inspect
import json

import pytest
import torch
from safetensors.torch import save_file

from vaayu.vaayu_base import VaayuBaseModel, resolve_compression


def _checkpoint(tmp_path, tensors, config=None):
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config or {"hidden_size": 8}))
    return str(tmp_path)


def _float_tensors():
    return {"model.embed_tokens.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
            "model.layers.0.mlp.up_proj.weight": torch.zeros(8, 8, dtype=torch.float16),
            "model.norm.weight": torch.zeros(8, dtype=torch.float32)}


@pytest.mark.parametrize("mode", ["4bit", "8bit"])
def test_an_unquantized_dense_checkpoint_keeps_its_request(tmp_path, mode, capsys):
    assert resolve_compression(mode, _checkpoint(tmp_path, _float_tensors())) == mode
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("dtype, name", [(torch.int32, "I32"),          # AWQ qweight, EXL3
                                         (torch.uint8, "U8"),           # MXFP4 blocks
                                         (torch.int8, "I8"),            # DeepSeek-V4 experts
                                         (torch.float8_e4m3fn, "F8_E4M3")])  # DeepSeek-V4 attention
@pytest.mark.parametrize("mode", ["4bit", "8bit"])
def test_a_quantized_checkpoint_is_refused_naming_the_dtype(tmp_path, mode, dtype, name, capsys):
    tensors = _float_tensors()
    tensors["model.layers.0.self_attn.q_proj.qweight"] = torch.zeros(8, 8).to(dtype)
    assert resolve_compression(mode, _checkpoint(tmp_path, tensors)) is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "DISABLED" in out
    assert name in out


def test_the_dtype_scan_reads_every_shard(tmp_path, capsys):
    # A quantized tensor in the SECOND shard must still refuse.
    save_file(_float_tensors(), str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"model.layers.1.mlp.qweight": torch.zeros(8, 8, dtype=torch.int32)},
              str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "config.json").write_text("{}")
    assert resolve_compression("8bit", str(tmp_path)) is None
    assert "I32" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["num_experts", "num_local_experts", "n_routed_experts"])
def test_a_moe_checkpoint_is_refused_even_in_bf16(tmp_path, field, capsys):
    path = _checkpoint(tmp_path, _float_tensors(), config={field: 4})
    assert resolve_compression("4bit", path) is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "DISABLED" in out
    assert "expert" in out


def test_no_original_shards_is_refused(tmp_path, capsys):
    # Originals deleted after an earlier split: only the index and config remain.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    assert resolve_compression("4bit", str(tmp_path)) is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "DISABLED" in out


def test_a_repo_id_is_refused(capsys):
    # No local headers exist before the download, so nothing can be decided.
    assert resolve_compression("8bit", "some-org/some-model") is None
    assert "DISABLED" in capsys.readouterr().out


def test_not_requesting_it_is_off_and_silent(tmp_path, capsys):
    assert resolve_compression(None, str(tmp_path)) is None
    assert capsys.readouterr().out == ""


def test_an_unknown_mode_raises_before_anything_is_split(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        resolve_compression("16bit", _checkpoint(tmp_path, _float_tensors()))


def test_the_engine_resolves_compression_before_the_split():
    # §8 "is it plumbed?": the split must receive the RESOLVED value, so a refused
    # request writes nothing -- not a copy of the request taken before the refusal.
    src = inspect.getsource(VaayuBaseModel.__init__)
    resolve = "compression = resolve_compression(compression, model_local_path_or_repo_id)"
    assert resolve in src
    assert src.index(resolve) < src.index("find_or_create_local_splitted_path(")
    assert src.count("self.compression =") == 1


# --- 4bit on an unquantized checkpoint (S10b) ---------------------------------
# `__init__` re-reads the vocab/hidden sizes off the split's embedding shard. A
# 4bit split stores that weight nf4-PACKED -- Llama-3.2-1B's [128256, 2048] as
# [131334144, 1] -- so the config got hidden_size=1, every weight inject was
# refused, and the first forward died on meta tensors (measured 2026-09-28).

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes quantizes on CUDA")


def _split_embed_shard(tmp_path, compression):
    from vaayu.utils import compress_layer_state_dict
    layer = {"model.embed_tokens.weight": torch.randn(96, 64, dtype=torch.bfloat16)}
    layer = {k: v.cpu() for k, v in compress_layer_state_dict(layer, compression).items()}
    path = tmp_path / "model.embed_tokens.safetensors"
    save_file(layer, str(path))
    return path


@cuda
@pytest.mark.parametrize("compression", [None, "8bit", "4bit"])
def test_the_embed_probe_reads_the_logical_shape_under_every_mode(tmp_path, compression):
    from vaayu.vaayu_base import embed_shard_shape
    path = _split_embed_shard(tmp_path, compression)
    assert list(embed_shard_shape(path, "model.embed_tokens.weight")) == [96, 64]


@pytest.mark.parametrize("compression, fits", [(None, False), ("8bit", False), ("4bit", True)])
def test_check_space_asks_for_the_compressed_size_not_more(tmp_path, monkeypatch, compression, fits):
    # A 4bit split is ~0.28x the checkpoint; check_space used to DIVIDE by 0.2813
    # and demand 3.55x (Llama-3.2-1B: 8.2 GiB demanded, 0.66 GiB written).
    import shutil
    from vaayu import utils
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"\0" * 1000)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(utils.shutil, "disk_usage",
                        lambda _p: shutil._ntuple_diskusage(10_000, 9_600, 400))
    if fits:
        utils.check_space(ckpt, out, compression)
    else:
        with pytest.raises(utils.NotEnoughSpaceException):
            utils.check_space(ckpt, out, compression)
