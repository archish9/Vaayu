"""S7 T4: install_packed_expert_kernels -- when the fused packed-expert decode kernel is
switched on at load, and when it is (loudly or quietly) left off."""
import inspect
import types

import pytest
import torch
import torch.nn as nn

import vaayu.packed_expert_kernels as pk
from tests.test_fused_packed_kernel_branch import _cuda_model, cuda


def _ref(**kw):
    base = dict(running_device="cuda", running_dtype=torch.bfloat16, _moe_blocks=[],
                _expert_offset_index=None, checkpoint_path="", moe_top_k=2)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_nothing_on_cpu():
    ref = _ref(running_device="cpu")
    assert pk.install_packed_expert_kernels(ref) is False
    assert not hasattr(ref, "_packed_moe_kernel")


def test_nothing_without_triton(monkeypatch):
    monkeypatch.setattr(pk, "triton", None)
    assert pk.install_packed_expert_kernels(_ref()) is False


def test_nothing_without_a_fused_nonexl3_block():
    blocks = [types.SimpleNamespace(forward_name="_make_modulelist_sparse_forward")]
    assert pk.install_packed_expert_kernels(_ref(_moe_blocks=blocks,
                                                 _expert_offset_index=object())) is False


def test_another_cuda_device_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    blocks = [types.SimpleNamespace(forward_name="_make_fused_sparse_forward")]
    ref = _ref(running_device="cuda:1", _moe_blocks=blocks, _expert_offset_index=object())
    assert pk.install_packed_expert_kernels(ref) is False
    assert "WARNING" in capsys.readouterr().out


@cuda
def test_installs_on_a_packed_fused_block(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    _model, ref, _ = _cuda_model(tmp_path, kernel=False)
    assert pk.install_packed_expert_kernels(ref) is True
    bits, gs, _roles = ref._packed_moe_kernel
    assert (bits, gs) == (4, 32)
    assert "fused packed-expert decode kernel on 1 MoE layers (4-bit, group 32)" \
        in capsys.readouterr().out
    assert len(pk._VERDICTS) == 2, "fp32 and activation-dtype routing weights both checked"


@cuda
def test_a_non_silu_activation_is_left_alone(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    model, ref, _ = _cuda_model(tmp_path, kernel=False)
    model.layers[0].mlp.experts.act_fn = nn.GELU()
    assert pk.install_packed_expert_kernels(ref) is False
    out = capsys.readouterr().out
    assert "not SiLU" in out and "WARNING" not in out
    assert not hasattr(ref, "_packed_moe_kernel")


@cuda
def test_a_dense_fused_checkpoint_is_left_alone(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    _model, ref, _ = _cuda_model(tmp_path, packed=False, kernel=False)
    assert pk.install_packed_expert_kernels(ref) is False
    assert "WARNING" not in capsys.readouterr().out


@cuda
def test_a_disagreeing_kernel_is_refused_at_load(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    monkeypatch.setattr(pk, "_launch", lambda x, pairs, w, out: out.zero_())
    _model, ref, _ = _cuda_model(tmp_path, kernel=False)
    assert pk.install_packed_expert_kernels(ref) is False
    assert "WARNING" in capsys.readouterr().out
    assert not hasattr(ref, "_packed_moe_kernel")


def test_the_engine_installs_after_patching():
    """Review focus 5 (§8 'is it plumbed?'): the call exists, after patch_moe_blocks --
    the installer reads each block's forward_name, which patching sets."""
    from vaayu.vaayu_base import VaayuBaseModel
    src = inspect.getsource(VaayuBaseModel.__init__)
    assert "install_packed_expert_kernels(self)" in src
    assert src.index("patch_moe_blocks(") < src.index("install_packed_expert_kernels(self)")
