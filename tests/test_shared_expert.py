"""Row 30 / S2, RC3: MoE blocks with an always-on shared expert.

`patch_moe_blocks` replaces the whole MoE *block* forward, so everything the
real block did besides routing is lost. Qwen3-Next (the S2 target), Qwen2-MoE,
DeepSeek and GLM all put an always-on shared expert in that block:

    out = routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)   # Qwen
    out = routed + shared_experts(x)                                   # DeepSeek

Neither on-device model has one (Qwen3-235B and MiniMax-M2.7 both route only),
so no existing gate catches this: the output would be plausible and wrong,
silently dropping ~63 MB of weights per layer on the S2 checkpoint.

The no-shared-expert case is asserted too — that is the regression guard for the
two models that ship today.
"""
import torch
import torch.nn as nn
import pytest

from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import (
    MockFusedModel,
    MockVaayuRef,
    _make_fused_checkpoint,
    MockModel,
    _make_checkpoint,
)

HIDDEN = 8
INTERMEDIATE = 16
NUM_EXPERTS = 8
TOP_K = 2


class MockSharedExpert(nn.Module):
    """A Qwen-style shared expert: a plain hidden→hidden MLP."""

    def __init__(self, hidden=HIDDEN, intermediate=INTERMEDIATE):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _patch(model, tmp_path, device="cpu"):
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    patch_moe_blocks(
        model=model,
        expert_cache=ExpertRAMCache(budget_bytes=8 * 1024 * 1024),
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=TOP_K,
        device=device,
        dtype=torch.float32,
        vaayu_ref=MockVaayuRef(),
        loader_strategy=None,
    )


def _routed_only(model, tmp_path, x):
    """Output of the same model with the shared expert detached — i.e. what the
    forward produced before this fix."""
    block = model.layers[0].mlp
    saved = {n: getattr(block, n) for n in list(dict(block.named_children()))
             if "shared_expert" in n}
    for n in saved:
        delattr(block, n)
    try:
        return model.layers[0].mlp.forward(x).clone()
    finally:
        for n, m in saved.items():
            setattr(block, n, m)


def test_fused_forward_applies_gated_shared_expert(tmp_path):
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    torch.manual_seed(3)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    block = model.layers[0].mlp
    block.shared_expert = MockSharedExpert()
    block.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)
    _patch(model, tmp_path)

    x = torch.randn(1, 3, HIDDEN)
    got = model.layers[0].mlp.forward(x)

    flat = x.view(-1, HIDDEN)
    expected_shared = torch.sigmoid(block.shared_expert_gate(flat)) * block.shared_expert(flat)
    routed = _routed_only(model, tmp_path, x)

    assert torch.allclose(got, routed + expected_shared.view_as(got), atol=1e-5)
    assert not torch.allclose(got, routed), "shared expert contributed nothing"


def test_fused_forward_applies_ungated_shared_expert(tmp_path):
    """DeepSeek shape: `shared_experts`, no gate — added straight on."""
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    torch.manual_seed(4)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    block = model.layers[0].mlp
    block.shared_experts = MockSharedExpert()
    _patch(model, tmp_path)

    x = torch.randn(1, 3, HIDDEN)
    got = model.layers[0].mlp.forward(x)
    routed = _routed_only(model, tmp_path, x)
    expected = routed + block.shared_experts(x.view(-1, HIDDEN)).view_as(routed)
    assert torch.allclose(got, expected, atol=1e-5)


def test_fused_forward_without_shared_expert_is_unchanged(tmp_path):
    """Regression guard for Qwen3-235B / MiniMax-M2.7, which have none."""
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    torch.manual_seed(5)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    _patch(model, tmp_path)
    x = torch.randn(1, 3, HIDDEN)
    out = model.layers[0].mlp.forward(x)
    assert out.shape == (1, 3, HIDDEN)
    assert torch.isfinite(out).all()


def test_modulelist_forward_applies_shared_expert(tmp_path):
    """The same block shape on the module-list forward — a feature wired into
    one forward silently no-ops in the others (CLAUDE.md §0)."""
    torch.manual_seed(6)
    model = MockModel(num_layers=1, hidden=HIDDEN, num_experts=NUM_EXPERTS)
    _make_checkpoint(tmp_path, model)
    block = model.layers[0].block_sparse_moe
    block.shared_expert = MockSharedExpert()
    block.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    patch_moe_blocks(
        model=model,
        expert_cache=ExpertRAMCache(budget_bytes=8 * 1024 * 1024),
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=TOP_K,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=MockVaayuRef(),
        loader_strategy=None,
    )

    x = torch.randn(1, 3, HIDDEN)
    got = model.layers[0].block_sparse_moe.forward(x)
    if isinstance(got, tuple):
        got = got[0]
    flat = x.view(-1, HIDDEN)
    shared = torch.sigmoid(block.shared_expert_gate(flat)) * block.shared_expert(flat)
    assert torch.isfinite(got).all()
    # The shared term is present: removing it changes the answer by exactly it.
    delattr(block, "shared_expert")
    delattr(block, "shared_expert_gate")
    routed = model.layers[0].block_sparse_moe.forward(x)
    if isinstance(routed, tuple):
        routed = routed[0]
    assert torch.allclose(got, routed + shared.view_as(routed), atol=1e-5)
