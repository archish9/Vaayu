"""Lookahead routing predictor on the MODULE-LIST forward (P2).

The predictor (Task 12) was wired ONLY into `_make_fused_sparse_forward_exl3`,
so `VAAYU_LOOKAHEAD=1` was a silent no-op on every module-list model — the
MiniMax layout, and exactly the single-forward trap CLAUDE.md §0 names. These
tests pin the wiring on the module-list forward, and pin that the prediction is
taken from the MODEL's own scoring function (MiniMax scores with sigmoid +
`e_score_correction_bias`, not softmax-topk), never from a hardcoded topk.
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vaayu.moe_patch import patch_moe_blocks, predict_next_expert_set
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex

from tests.test_moe_patch import MockModel, MockVaayuRef, _make_checkpoint


class RecordingStager:
    """Stand-in for vaayu_base's _stager_executor: records and runs inline."""

    def __init__(self):
        self.calls: list = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append(args)
        fn(*args, **kwargs)


def _patch_for_lookahead(tmp_path, num_layers=2, num_experts=8, top_k=2,
                         enabled=True):
    """Patched module-list model with the lookahead wiring a real engine sets up."""
    model = MockModel(num_layers=num_layers, hidden=8, num_experts=num_experts)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vaayu_ref._lookahead_enabled = enabled
    vaayu_ref._lookahead_state = {"hits": 0, "total": 0}
    vaayu_ref._stager_executor = RecordingStager()

    blocks = patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    # What vaayu_base.py builds right after patch_moe_blocks().
    vaayu_ref._moe_block_by_idx = {b.layer_idx: b for b in blocks}
    return model, vaayu_ref, cache


def test_modulelist_forward_predicts_next_layer(tmp_path):
    """Layer 0's forward must leave a prediction for layer 1."""
    model, vaayu_ref, _ = _patch_for_lookahead(tmp_path)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    pred = getattr(vaayu_ref, "_lookahead_pred", None)
    assert pred is not None, "module-list forward made no lookahead prediction"
    layer, experts = pred
    assert layer == 1
    assert experts and all(isinstance(e, int) for e in experts)


def test_modulelist_forward_prefetches_predicted_experts(tmp_path):
    """Phase B: predicted next-layer experts are staged into the RAM cache."""
    model, vaayu_ref, cache = _patch_for_lookahead(tmp_path)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    _, predicted = vaayu_ref._lookahead_pred
    assert vaayu_ref._stager_executor.calls, "nothing submitted to the stager"
    for eidx in predicted:
        assert cache.has(1, eidx), f"layer-1 expert {eidx} was not prefetched"


def test_modulelist_lookahead_off_makes_no_prediction(tmp_path):
    """Flag off => no next-layer router probe, no prefetch, no overhead."""
    model, vaayu_ref, _ = _patch_for_lookahead(tmp_path, enabled=False)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert getattr(vaayu_ref, "_lookahead_pred", None) is None
    assert vaayu_ref._stager_executor.calls == []


def test_modulelist_lookahead_scores_accuracy(tmp_path):
    """Phase A: a prediction for THIS layer is scored against its real routing."""
    model, vaayu_ref, _ = _patch_for_lookahead(tmp_path, num_layers=1)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)
    # What layer 0 actually routes to, per the model's own scoring function.
    router_logits = moe.gate(hidden)
    from vaayu.moe_patch import route_experts
    _, actual_idx = route_experts(moe, router_logits, 2)
    actual = set(int(x) for x in actual_idx.flatten().unique().tolist())

    vaayu_ref._lookahead_pred = (0, actual)
    moe.forward(hidden)
    assert vaayu_ref._lookahead_state == {"hits": len(actual), "total": len(actual)}


def test_modulelist_last_layer_predicts_nothing(tmp_path):
    """No layer_idx+1 block => no prediction, no crash."""
    model, vaayu_ref, _ = _patch_for_lookahead(tmp_path, num_layers=1)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))
    assert getattr(vaayu_ref, "_lookahead_pred", None) is None
    assert vaayu_ref._stager_executor.calls == []


# ── The prediction must use the model's OWN scoring function ─────────────────

class _InvertedScoringMoE(nn.Module):
    """A MoE block whose contract method selects the LOWEST-scoring experts.

    Stands in for any model that owns its scoring (MiniMax: sigmoid +
    e_score_correction_bias, group routing). Deliberately disagrees with plain
    topk-on-logits so a hardcoded topk predictor is caught.
    """

    def __init__(self, hidden=8, num_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)])
        self.top_k = top_k

    def route_tokens_to_experts(self, router_logits):
        weights, idx = torch.topk(-router_logits.float(), self.top_k, dim=-1)
        return idx, weights


def test_prediction_uses_model_scoring_function():
    """The predictor must ask the model, not assume softmax-topk."""
    from vaayu.moe_patch import MoEBlockInfo, route_experts

    torch.manual_seed(0)
    moe = _InvertedScoringMoE()
    block = MoEBlockInfo(
        layer_idx=1, layer_name="layers.1",
        module_path="layers.1.block_sparse_moe", moe_module=moe,
        router_attr="gate", experts_attr="experts", expert_style="module_list")

    hidden = torch.randn(1, 1, 8)
    got = predict_next_expert_set(block, hidden, top_k=2)

    logits = moe.gate(hidden)
    _, own_idx = route_experts(moe, logits, 2)
    expected = set(int(x) for x in own_idx.flatten().unique().tolist())
    _, naive_idx = torch.topk(logits.float(), 2, dim=-1)
    naive = set(int(x) for x in naive_idx.flatten().unique().tolist())

    assert got == expected
    assert got != naive, "test model does not distinguish the two — fix the fixture"


def test_prediction_uses_router_returned_topk():
    """A router that returns its own (logits, weights, indices) is believed."""
    from vaayu.moe_patch import MoEBlockInfo

    class _TupleRouter(nn.Module):
        def forward(self, x):
            n = x.reshape(-1, x.shape[-1]).shape[0]
            return (torch.zeros(n, 8), torch.ones(n, 2),
                    torch.tensor([[5, 6]] * n))

    class _MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = _TupleRouter()
            self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])

    block = MoEBlockInfo(
        layer_idx=1, layer_name="layers.1",
        module_path="layers.1.block_sparse_moe", moe_module=_MoE(),
        router_attr="gate", experts_attr="experts", expert_style="module_list")
    assert predict_next_expert_set(block, torch.randn(1, 1, 8), top_k=2) == {5, 6}
