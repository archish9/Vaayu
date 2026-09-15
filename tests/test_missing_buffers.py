"""Persistent buffers the checkpoint has no tensor for.

A module can register a PERSISTENT buffer (part of its state_dict) that a
checkpoint does not ship. Nothing restores it, so it silently keeps its
init value. Measured 2026-09-15: `NeuroSenko--MiniMax-M2.7-exl3` has no
`e_score_correction_bias` on any of its 62 layers (upstream
`MiniMaxAI/MiniMax-M2.7` ships them), so MiniMax routed with a zero bias --
prose ppl +3.3 % vs the upstream bias. Qwen3-235B, Qwen3-Next, Llama-3.3-70B
and Llama-3.2-1B register no persistent buffers, so the check is free there.

Vaayu warns about every such buffer (CLAUDE.md §5: warn, don't silently
degrade) and applies a sidecar file of the missing tensors when the user has
fetched one with scripts/fetch_missing_buffers.py.
"""
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vaayu.missing_buffers import (
    SIDECAR_FILENAME, apply_sidecar, missing_persistent_buffers, read_remote_tensors,
    warn_missing,
)

E = 3


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, E, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(E))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)


class _Model(nn.Module):
    """Module names mirror a HF causal LM: model.layers.N.<block>."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(n_layers)])
        for layer in self.model.layers:
            layer.block = _Block()


def _layer_names(n):
    return ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(n)] + ["model.norm", "lm_head"]


def _split(path, model, carries_bias=()):
    """One file per layer, as the split dir holds them."""
    for i, layer in enumerate(model.model.layers):
        tensors = {f"model.layers.{i}.block.gate.weight": layer.block.gate.weight.detach().clone()}
        if i in carries_bias:
            tensors[f"model.layers.{i}.block.e_score_correction_bias"] = torch.ones(E)
        save_file(tensors, str(Path(path) / f"model.layers.{i}.safetensors"))


def _bias(i):
    return f"model.layers.{i}.block.e_score_correction_bias"


# ── detection ───────────────────────────────────────────────────────────────

def test_a_persistent_buffer_with_no_checkpoint_tensor_is_reported(tmp_path):
    model = _Model()
    _split(tmp_path, model)

    missing = missing_persistent_buffers(model, tmp_path, _layer_names(2))

    assert missing == [_bias(0), _bias(1)]  # the non-persistent `scratch` is never reported


def test_a_buffer_the_checkpoint_carries_is_not_reported(tmp_path):
    model = _Model()
    _split(tmp_path, model, carries_bias=(0,))

    assert missing_persistent_buffers(model, tmp_path, _layer_names(2)) == [_bias(1)]


def test_the_owning_layer_file_is_matched_on_a_dot_boundary(tmp_path):
    """`model.layers.1` is a string prefix of `model.layers.10`; reading layer 1's
    file for layer 10's buffer would report a buffer the checkpoint does carry."""
    model = _Model(n_layers=11)
    _split(tmp_path, model, carries_bias=(10,))

    missing = missing_persistent_buffers(model, tmp_path, _layer_names(11))

    assert _bias(10) not in missing
    assert _bias(1) in missing


def test_a_model_with_no_persistent_buffers_reads_no_file(tmp_path):
    model = nn.Module()
    model.register_buffer("inv_freq", torch.zeros(2), persistent=False)

    assert missing_persistent_buffers(model, tmp_path / "does-not-exist", ["lm_head"]) == []


# ── sidecar ────────────────────────────────────────────────────────────────

def test_the_sidecar_fills_missing_buffers_and_keeps_its_dtype(tmp_path):
    """Routing adds the bias to float32 scores; a bf16 cast would round away a
    third of the selection-relevant spread (bf16 step ~0.016 at ~2.15)."""
    model = _Model().to(torch.bfloat16)
    values = {_bias(0): torch.tensor([2.1, 2.15, 2.2]), _bias(1): torch.tensor([1.0, 3.0, 5.0])}
    save_file(values, str(tmp_path / SIDECAR_FILENAME))

    applied = apply_sidecar(model, tmp_path, [_bias(0), _bias(1)])

    assert applied == [_bias(0), _bias(1)]
    got = model.model.layers[0].block.e_score_correction_bias
    assert got.dtype == torch.float32
    assert torch.equal(got, values[_bias(0)])


def test_no_sidecar_applies_nothing(tmp_path):
    assert apply_sidecar(_Model(), tmp_path, [_bias(0)]) == []


def test_a_sidecar_tensor_of_the_wrong_shape_raises(tmp_path):
    """A sidecar fetched for a different model must not load half-silently."""
    save_file({_bias(0): torch.zeros(E + 1)}, str(tmp_path / SIDECAR_FILENAME))

    with pytest.raises(ValueError, match="e_score_correction_bias"):
        apply_sidecar(_Model(), tmp_path, [_bias(0)])


def test_sidecar_keys_that_are_not_missing_buffers_are_ignored_loudly(tmp_path, capsys):
    save_file({_bias(0): torch.ones(E), "model.layers.0.block.not_a_buffer": torch.ones(E)},
              str(tmp_path / SIDECAR_FILENAME))
    model = _Model()

    applied = apply_sidecar(model, tmp_path, [_bias(1)])

    assert applied == []
    assert torch.equal(model.model.layers[0].block.e_score_correction_bias, torch.zeros(E))
    out = capsys.readouterr().out
    assert "ignored" in out and "not_a_buffer" in out


def test_the_sidecar_changes_the_auto_fingerprint(tmp_path):
    """Guard, not new behaviour: a cached DECODE_TOP_K=auto pick measured without
    the bias must re-measure once the bias is added. That holds only while the
    sidecar lives in the split dir the fingerprint globs."""
    from vaayu.decode_top_k_auto import fingerprint
    _split(tmp_path, _Model())
    before = fingerprint("{}", tmp_path)

    save_file({_bias(0): torch.ones(E)}, str(tmp_path / SIDECAR_FILENAME))

    assert fingerprint("{}", tmp_path) != before


# ── warning ────────────────────────────────────────────────────────────────

def test_the_warning_names_the_buffers_and_how_to_fetch_them(capsys):
    warn_missing([_bias(i) for i in range(62)])

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "62" in out and "e_score_correction_bias" in out
    assert "scripts/fetch_missing_buffers.py" in out


def test_nothing_missing_prints_nothing(capsys):
    warn_missing([])

    assert capsys.readouterr().out == ""


# ── ranged fetch core (scripts/fetch_missing_buffers.py) ────────────────────

def test_read_remote_tensors_reads_only_the_named_tensors(tmp_path):
    big = torch.arange(4096, dtype=torch.float32)
    save_file({"a.weight": big, _bias(0): torch.tensor([1.5, 2.5, 3.5]), _bias(1): torch.ones(E)},
              str(tmp_path / "shard-1.safetensors"))
    raw = (tmp_path / "shard-1.safetensors").read_bytes()
    fetched = []

    def get(filename, start, end):
        fetched.append((filename, start, end))
        return raw[start:end + 1]

    weight_map = {"a.weight": "shard-1.safetensors", _bias(0): "shard-1.safetensors",
                  _bias(1): "shard-1.safetensors"}
    out = read_remote_tensors(get, weight_map, [_bias(0)])

    assert list(out) == [_bias(0)]
    assert torch.equal(out[_bias(0)], torch.tensor([1.5, 2.5, 3.5]))
    assert sum(e - s + 1 for _, s, e in fetched) < len(raw) // 4  # never the 16 KB weight


# ── engine wiring ───────────────────────────────────────────────────────────

def test_the_engine_applies_the_sidecar_then_warns_about_the_rest(tmp_path, capsys):
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.model = _Model()
    m.checkpoint_path = tmp_path
    m.layer_names = _layer_names(2)
    _split(tmp_path, m.model)
    save_file({_bias(0): torch.full((E,), 2.0)}, str(tmp_path / SIDECAR_FILENAME))

    m._apply_missing_buffers()

    assert torch.equal(m.model.model.layers[0].block.e_score_correction_bias, torch.full((E,), 2.0))
    out = capsys.readouterr().out
    assert _bias(1) in out or "1 persistent buffer" in out
    assert "WARNING" in out


def test_init_applies_missing_buffers_once_the_modules_are_final():
    """setup_model may replace modules (e.g. ExLlamaV3Linear); buffers must be
    checked on the final tree, i.e. after set_layers_from_layer_names()."""
    import inspect
    from vaayu.vaayu_base import VaayuBaseModel
    src = inspect.getsource(VaayuBaseModel.__init__)

    assert "self._apply_missing_buffers()" in src
    assert src.index("self._apply_missing_buffers()") > src.index("self.set_layers_from_layer_names()")
