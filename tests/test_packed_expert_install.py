"""S7 T4: install_packed_expert_kernels -- when the fused packed-expert decode kernel is
switched on at load, and when it is (loudly or quietly) left off."""
import inspect
import types

import pytest
import torch
import torch.nn as nn

import vaayu.moe_patch as mp
import vaayu.packed_expert_kernels as pk
from tests.test_fused_packed_kernel_branch import NUM_EXPERTS, _cuda_model, cuda


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


# ---- S20: the grouped prefill kernels ride the same installer, on their own verdict ----

def _prefill_model(tmp_path, **kw):
    """A geometry the prefill kernels can tile (the 64-wide default cannot)."""
    return _cuda_model(tmp_path, kernel=False, hidden=256, intermediate=128, **kw)


def _fresh(monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})


@cuda
def test_installs_the_prefill_kernels_too(tmp_path, capsys, monkeypatch):
    _fresh(monkeypatch)
    _model, ref, _ = _prefill_model(tmp_path)
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_prefill_kernel == frozenset({ref._moe_blocks[0].layer_idx})
    out = capsys.readouterr().out
    assert "fused packed-expert prefill kernel on 1 MoE layers (4-bit, group 32)" in out
    assert "WARNING" not in out
    assert len(pk._PF_VERDICTS) == 2, "fp32 and activation-dtype routing weights both checked"


@cuda
def test_a_disagreeing_prefill_kernel_keeps_the_decode_kernel(tmp_path, capsys, monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(pk, "_prefill_launch", lambda x, e, w, b, out, plan=None: out.zero_())
    _model, ref, _ = _prefill_model(tmp_path)
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_kernel is not None
    assert not hasattr(ref, "_packed_moe_prefill_kernel")
    out = capsys.readouterr().out
    assert "prefill kernel refused" in out and "WARNING" in out


@cuda
def test_a_geometry_the_prefill_kernels_cannot_tile_keeps_the_decode_kernel(tmp_path, capsys,
                                                                            monkeypatch):
    """The 64-wide mock: S7's kernels take it, a tensor-core tile does not exist."""
    _fresh(monkeypatch)
    _model, ref, _ = _cuda_model(tmp_path, kernel=False)
    assert pk.install_packed_expert_kernels(ref) is True
    assert not hasattr(ref, "_packed_moe_prefill_kernel")
    out = capsys.readouterr().out
    assert "prefill kernel refused" in out and "K tile" in out


@cuda
def test_a_refused_decode_kernel_installs_no_prefill_kernel(tmp_path, monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(pk, "_launch", lambda x, pairs, w, out: out.zero_())
    _model, ref, _ = _prefill_model(tmp_path)
    assert pk.install_packed_expert_kernels(ref) is False
    assert not hasattr(ref, "_packed_moe_prefill_kernel")


@cuda
def test_layers_too_large_to_hold_at_once_keep_the_per_expert_prefill(tmp_path, capsys,
                                                                      monkeypatch):
    """The grouped kernels hold every blob a call uploaded until they launch -- at worst a
    layer's whole expert set. Over half the prefill headroom that is refused, loudly, with
    the derived size; the decode kernel (ten experts a token) is unaffected. Here every
    layer is too large, so no prefill kernel is installed at all."""
    _fresh(monkeypatch)
    _model, ref, _ = _prefill_model(tmp_path, more_layers=({},))
    monkeypatch.setattr(mp, "_PREFILL_VRAM_HEADROOM", 64 * 1024)
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_kernel is not None
    assert not hasattr(ref, "_packed_moe_prefill_kernel")
    warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert len(warnings) == 1
    assert "prefill kernel not installed: one call can hold a layer's 8 experts" in warnings[0]
    assert warnings[0].endswith("prefill headroom; prefill keeps the per-expert path")
    assert pk._PF_VERDICTS == {}, "refused before the kernels are compiled"


def _no_fused_pass_after_the_first(monkeypatch):
    """`build_expert_blob` refuses every expert after the first it is handed: the second
    layer's roles read as packed at different widths."""
    real, calls = mp.build_expert_blob, []

    def build(by_role):
        calls.append(1)
        return real(by_role) if len(calls) == 1 else None
    monkeypatch.setattr(mp, "build_expert_blob", build)


@cuda
@pytest.mark.parametrize("second, why", [
    (None, "not pack-quantized"),
    ("no-fused-pass", "no fused pass"),
    ({"group_size": 64}, "a different geometry or scale dtype"),
], ids=["unquantized", "no-fused-pass", "another-group-size"])
def test_a_layer_the_prefill_kernels_cannot_take_is_left_out_and_named(tmp_path, capsys,
                                                                       monkeypatch, second, why):
    """Eligibility is per MoE layer, from one real expert of each: the layer that is not
    packed like the checked one keeps the per-expert prefill, named in ONE WARNING."""
    _fresh(monkeypatch)
    if second == "no-fused-pass":
        second = {}
        _no_fused_pass_after_the_first(monkeypatch)
    _model, ref, _ = _prefill_model(tmp_path, more_layers=(second,))
    first, other = (b.layer_idx for b in ref._moe_blocks)
    assert first != other, "premise: two MoE layers"
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_prefill_kernel == frozenset({first})
    assert ref._packed_moe_kernel is not None
    out = capsys.readouterr().out
    assert "fused packed-expert prefill kernel on 1 MoE layers (4-bit, group 32)" in out
    warnings = [line for line in out.splitlines() if "WARNING" in line]
    assert len(warnings) == 1
    assert (f"prefill kernel not used on 1 of 2 MoE layers (layer indices {other}): "
            in warnings[0]) and why in warnings[0]
    assert warnings[0].endswith("prefill keeps the per-expert path there")
    assert "decode kernel not installed" not in out, "the probe of the other layer is quiet"


@cuda
def test_layers_left_out_for_different_reasons_share_one_warning(tmp_path, capsys, monkeypatch):
    _fresh(monkeypatch)
    _model, ref, _ = _prefill_model(tmp_path, more_layers=(None, {"group_size": 64}))
    first, dense, other = (b.layer_idx for b in ref._moe_blocks)
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_prefill_kernel == frozenset({first})
    warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert len(warnings) == 1
    assert warnings[0].endswith(
        f"not used on 2 of 3 MoE layers (layer indices {dense}, {other}): experts not "
        f"pack-quantized, or packed with no fused pass (layers {dense}); a different geometry "
        f"or scale dtype than the checked layer (layers {other}); prefill keeps the "
        f"per-expert path there")


@cuda
def test_a_layer_with_more_experts_than_fit_is_left_out_with_its_size(tmp_path, capsys,
                                                                      monkeypatch):
    """The size guard is each layer's own: twice the experts, twice the bytes."""
    _fresh(monkeypatch)
    _model, ref, _ = _prefill_model(tmp_path, more_layers=({},))
    small, large = ref._moe_blocks
    blob = pk._one_real_blob(ref, small, "cuda")
    held = NUM_EXPERTS * sum(t.numel() * t.element_size() for t in blob["flat"].values())
    large.moe_module.experts.num_experts = 2 * NUM_EXPERTS
    monkeypatch.setattr(mp, "_PREFILL_VRAM_HEADROOM", 3 * held)     # half: 1.5x one layer
    assert pk.install_packed_expert_kernels(ref) is True
    assert ref._packed_moe_prefill_kernel == frozenset({small.layer_idx})
    warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert len(warnings) == 1
    assert (f"not used on 1 of 2 MoE layers (layer indices {large.layer_idx}): one call can "
            f"hold a layer's {2 * NUM_EXPERTS} experts on the GPU at once "
            f"({2 * held / 2**30:.2f} GiB)") in warnings[0]
