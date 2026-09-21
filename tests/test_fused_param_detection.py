"""A3: telling a fused-PARAMETER experts module from a per-expert one.

Both are `expert_style == "fused"` to detect_moe_blocks, because both expose
num_experts on the module. They differ in where the weights live: gpt-oss keeps
one parameter per projection with the expert count leading; Qwen3-Next keeps
per-expert tensors. Only the first can be handed to a grouped-GEMM kernel.
"""
import torch
import torch.nn as nn

from vaayu.moe_patch import fused_param_experts


class _FusedParams(nn.Module):
    """gpt-oss shape, miniaturised."""
    def __init__(self, num_experts=4, rows=8, cols=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, rows, cols))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, cols, rows))


class _PerExpert(nn.Module):
    """Qwen3-Next shape, miniaturised."""
    def __init__(self, num_experts=4, rows=8):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(nn.Linear(rows, rows) for _ in range(num_experts))


def test_fused_parameter_module_is_detected():
    assert fused_param_experts(_FusedParams()) is True


def test_per_expert_module_is_not():
    assert fused_param_experts(_PerExpert()) is False


def test_module_without_num_experts_is_not():
    assert fused_param_experts(nn.Linear(4, 4)) is False


def test_detection_does_not_depend_on_parameter_names():
    """Capability, not naming (CLAUDE.md §5): a different family's fused module
    must be detected too."""
    mod = _FusedParams()
    mod.w13 = mod.gate_up_proj
    del mod.gate_up_proj
    assert fused_param_experts(mod) is True


class _Qwen3NextLike(nn.Module):
    """The live Qwen3-Next experts module, miniaturised.

    Measured on device 2026-09-21 (`plans/probes/a3_qwen_experts_shape.py`):
    `Qwen3NextExperts` carries DIRECT fused parameters -- `gate_up_proj
    [512, 1024, 2048]`, `down_proj [512, 2048, 512]`, num_experts=512 -- both
    before and after CompressedTensorsHfQuantizer.preprocess_model. It also
    carries an `act_fn` child, because that forward computes the FFN itself.
    """
    def __init__(self, num_experts=4, rows=8, cols=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, rows, cols))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, cols, rows))
        self.act_fn = nn.SiLU()


def test_a_per_expert_checkpoints_module_also_reports_fused():
    """The predicate is about the MODULE, and on its own it cannot pick a forward.

    Qwen3-Next -- the fastest, most perf-tuned model on this box -- reports True
    here while its checkpoint is per-expert. Dispatch must therefore also require
    the offset index to hold whole-layer tensors for the layer. This test exists
    so that the day someone dispatches on this predicate alone, it fails here
    rather than on the tuned model.
    """
    assert fused_param_experts(_Qwen3NextLike()) is True


def test_one_dimensional_parameter_is_not_enough():
    """A 1-D tensor under an experts module has no axis left once the expert
    axis is consumed -- the same rule the offset index applies."""
    mod = nn.Module()
    mod.num_experts = 4
    mod.expert_bias = nn.Parameter(torch.zeros(4))
    assert fused_param_experts(mod) is False
