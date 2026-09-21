"""A3: whole-layer delivery for a fused-3D checkpoint.

A checkpoint that stores every expert of a layer in ONE tensor per projection
cannot be streamed per expert: the module's kernel is handed the whole tensor and
routes inside it. These tests cover the dispatch that picks that path, the
populate/release round trip that keeps the module on meta between visits, and the
one case that must NOT change -- a per-expert checkpoint whose live module also
carries fused parameters, which is Qwen3-Next, the tuned path.
"""
import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vaayu.moe_patch import (_populate_fused_experts, _release_fused_experts,
                             patch_moe_blocks)
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex

NUM_EXPERTS = 8
WIDTH = 4


class _FusedExperts(nn.Module):
    """One fused parameter per projection, expert count leading (gpt-oss shape)."""

    def __init__(self, num_experts=NUM_EXPERTS, width=WIDTH):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, width, width))
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, width))
        # Present and None at init, exactly as Mxfp4GptOssExperts declares it --
        # a converter fills it in and release has to put the None back.
        self.gate_up_proj_precision_config = None


class _Block(nn.Module):
    """The MoE block. Its forward returns a TUPLE, as gpt-oss's does."""

    def __init__(self, num_experts=NUM_EXPERTS, width=WIDTH):
        super().__init__()
        self.router = nn.Linear(width, num_experts)
        self.experts = _FusedExperts(num_experts, width)

    def forward(self, hidden_states):
        w = self.experts.gate_up_proj
        b = self.experts.gate_up_proj_bias
        return hidden_states @ w.sum(0) + b.sum(0), None


class _PerExpertBlock(nn.Module):
    """Qwen3-Next's shape: fused parameters on the module, per-expert checkpoint."""

    def __init__(self, num_experts=NUM_EXPERTS, width=WIDTH):
        super().__init__()
        self.router = nn.Linear(width, num_experts)
        self.experts = _FusedExperts(num_experts, width)
        self.experts.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return hidden_states, None


class _Layer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


class _Model(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(block)])


class _MockVaayuRef:
    def __init__(self, model, hf_quantizer=None):
        self.model = model
        self.hf_quantizer = hf_quantizer
        self._expert_history: dict[int, list[int]] = {}
        self.pinned_layer_names: set = set()


def _write_config(path, num_experts=NUM_EXPERTS):
    (path / "config.json").write_text(json.dumps({"num_experts": num_experts}))


def _write_fused_checkpoint(path, weights=None, biases=None):
    """One layer, one tensor per projection, expert count leading."""
    _write_config(path)
    w = weights if weights is not None else torch.randn(NUM_EXPERTS, WIDTH, WIDTH)
    b = biases if biases is not None else torch.randn(NUM_EXPERTS, WIDTH)
    save_file({"layers.0.mlp.experts.gate_up_proj": w.contiguous(),
               "layers.0.mlp.experts.gate_up_proj_bias": b.contiguous()},
              str(path / "layers.0.safetensors"))
    return w, b


def _write_per_expert_checkpoint(path):
    _write_config(path)
    tensors = {}
    for eidx in range(NUM_EXPERTS):
        tensors[f"layers.0.mlp.experts.{eidx}.w1.weight"] = torch.randn(WIDTH, WIDTH)
    save_file(tensors, str(path / "layers.0.safetensors"))


def _meta_model(block_cls=_Block):
    with torch.device("meta"):
        return _Model(block_cls())


def _patch(tmp_path, model, vaayu_ref):
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    blocks = patch_moe_blocks(
        model=model,
        expert_cache=ExpertRAMCache(budget_bytes=4 * 1024 * 1024),
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=2,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    return idx, blocks


def test_fused_checkpoint_gets_the_whole_layer_forward(tmp_path):
    _write_fused_checkpoint(tmp_path)
    model = _meta_model()
    _idx, blocks = _patch(tmp_path, model, _MockVaayuRef(model))
    assert len(blocks) == 1
    assert blocks[0].forward_name == "_make_whole_layer_sparse_forward"


def test_per_expert_checkpoint_keeps_the_tuned_fused_forward(tmp_path):
    """The regression this dispatch exists to avoid.

    The live module carries fused parameters -- Qwen3NextExperts does, measured --
    so a module-only predicate would move the fastest model on this box onto an
    unmeasured forward. The checkpoint layout decides instead.
    """
    _write_per_expert_checkpoint(tmp_path)
    model = _meta_model(_PerExpertBlock)
    _idx, blocks = _patch(tmp_path, model, _MockVaayuRef(model))
    assert blocks[0].forward_name == "_make_fused_sparse_forward"


def test_forward_streams_the_layer_and_leaves_the_module_on_meta(tmp_path):
    w, b = _write_fused_checkpoint(tmp_path)
    model = _meta_model()
    ref = _MockVaayuRef(model)
    _idx, blocks = _patch(tmp_path, model, ref)
    experts = blocks[0].moe_module.experts

    hidden = torch.randn(3, WIDTH)
    out, extra = blocks[0].moe_module.forward(hidden)

    assert extra is None, "the block's own return shape must survive untouched"
    torch.testing.assert_close(out, hidden @ w.sum(0) + b.sum(0))
    assert experts.gate_up_proj.is_meta, "the layer's bytes must not outlive the call"
    assert experts.gate_up_proj_bias.is_meta
    assert ref._expert_history[0] == list(range(NUM_EXPERTS))


def test_two_visits_do_not_accumulate(tmp_path):
    w, b = _write_fused_checkpoint(tmp_path)
    model = _meta_model()
    ref = _MockVaayuRef(model)
    _idx, blocks = _patch(tmp_path, model, ref)
    hidden = torch.randn(2, WIDTH)

    first, _ = blocks[0].moe_module.forward(hidden)
    second, _ = blocks[0].moe_module.forward(hidden)

    torch.testing.assert_close(first, second)
    assert blocks[0].moe_module.experts.gate_up_proj.is_meta


def test_a_raising_forward_still_releases(tmp_path):
    _write_fused_checkpoint(tmp_path)
    model = _meta_model()
    ref = _MockVaayuRef(model)
    _idx, blocks = _patch(tmp_path, model, ref)
    experts = blocks[0].moe_module.experts

    with pytest.raises(TypeError):
        blocks[0].moe_module.forward("not a tensor")

    assert experts.gate_up_proj.is_meta


class _StubOp:
    """Stands in for a quantizer conversion op: writes an attribute, not a param."""

    def convert(self, input_dict, model=None, full_layer_name=None,
                missing_keys=None, **kwargs):
        module = model.layers[0].mlp.experts
        packed = input_dict["gate_up_proj_packed"]
        scale = input_dict["gate_up_proj_scale"]
        object.__setattr__(module, "gate_up_proj", packed.float() * scale)
        object.__setattr__(module, "gate_up_proj_precision_config", "live")
        return {}


class _StubConverter:
    source_patterns = ["gate_up_proj_packed", "gate_up_proj_scale"]
    target_patterns = [r"gate_up_proj$"]
    operations = [_StubOp()]


class _StubQuantizer:
    def get_weight_conversions(self):
        return [_StubConverter()]


def test_quantizer_conversions_are_applied_and_then_released():
    """The weights are populated through the quantizer's own hook, and whatever
    it leaves behind -- an attribute, not a parameter -- is released with it."""
    model = _meta_model()
    experts = model.layers[0].mlp.experts
    ref = _MockVaayuRef(model, hf_quantizer=_StubQuantizer())
    meta_params = dict(experts._parameters)
    base_attrs = dict(experts.__dict__)

    raw = {"layers.0.mlp.experts.gate_up_proj_packed":
               torch.ones(NUM_EXPERTS, WIDTH, WIDTH, dtype=torch.uint8),
           "layers.0.mlp.experts.gate_up_proj_scale": torch.tensor(2.0),
           "layers.0.mlp.experts.gate_up_proj_bias": torch.zeros(NUM_EXPERTS, WIDTH)}
    _populate_fused_experts(experts, "layers.0.mlp.experts", raw, "cpu", ref)

    assert experts.gate_up_proj_precision_config == "live"
    torch.testing.assert_close(experts.gate_up_proj,
                               torch.full((NUM_EXPERTS, WIDTH, WIDTH), 2.0))
    assert not experts.gate_up_proj_bias.is_meta

    _release_fused_experts(experts, meta_params, base_attrs)

    assert experts.gate_up_proj.is_meta
    assert experts.gate_up_proj_precision_config is None
    assert "gate_up_proj_packed" not in experts.__dict__


def test_unconverted_tensors_take_the_dtype_the_module_declares():
    """The stock loader fixes a tensor's dtype from the module's own parameter;
    a bf16 bias for an fp32 parameter must not arrive as bf16."""
    with torch.device("meta"):
        experts = _FusedExperts()
    ref = _MockVaayuRef(_meta_model())
    raw = {"layers.0.mlp.experts.gate_up_proj_bias":
               torch.zeros(NUM_EXPERTS, WIDTH, dtype=torch.bfloat16)}
    _populate_fused_experts(experts, "layers.0.mlp.experts", raw, "cpu", ref)
    assert experts.gate_up_proj_bias.dtype is torch.float32
