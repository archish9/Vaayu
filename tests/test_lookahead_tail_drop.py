"""Prefetch ADMISSION: the lowest-weight prediction per row is not staged (P4).

Measured 2026-09-03 on both actively-tested MoE models (64-token greedy decode,
`plans/probes/p4_trace.py` + `p4_sim.py`, simulator gated against the run's own
counters to within 1.2 %): the marginal misses a prefetch read removes, taken
rank by rank down the router's weight order, is roughly flat until the LAST
rank, where it falls off a cliff on BOTH models —

    Qwen3-235B    0.884 0.853 0.867 0.822 0.786 0.726 0.607 **0.418**
    MiniMax-M2.7  0.686 0.654 0.666 0.647 0.675 0.687 0.654 **0.312**

so the top_k-th prediction buys about half of what every other one buys. It is
the least robust member of the set: the predictor routes the NEXT layer on the
CURRENT layer's hidden state, and the lowest-weight pick is the one that falls
out of the true top-k first. Dropping it removes ~11 % of decode reads.

This is an ADMISSION rule, not a prediction rule: `_lookahead_pred` still
carries the FULL predicted set, so the accuracy telemetry keeps measuring the
predictor rather than the staging policy.
"""
import pytest
import torch
import torch.nn as nn

from vaayu.moe_patch import MoEBlockInfo, predict_next_experts, route_experts

from tests.test_lookahead_modulelist import _patch_for_lookahead


class _SoftmaxMoE(nn.Module):
    """No `route_tokens_to_experts` — the softmax-topk fallback Qwen3-MoE and
    Mixtral genuinely use."""

    def __init__(self, hidden=8, num_experts=16, top_k=4):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)])
        self.top_k = top_k


class _OwnScoringMoE(_SoftmaxMoE):
    """Owns its scoring function, and returns (indices, weights) — the
    MiniMax/DeepSeek/GLM contract order, which is the opposite of dbrx's."""

    def route_tokens_to_experts(self, router_logits):
        weights, idx = torch.topk((-router_logits.float()).sigmoid(),
                                  self.top_k, dim=-1)
        return idx, weights


def _block(moe):
    return MoEBlockInfo(
        layer_idx=1, layer_name="layers.1",
        module_path="layers.1.block_sparse_moe", moe_module=moe,
        router_attr="gate", experts_attr="experts", expert_style="module_list")


@pytest.mark.parametrize("cls", [_SoftmaxMoE, _OwnScoringMoE],
                         ids=["softmax-topk", "own-scoring"])
def test_staging_drops_the_lowest_weight_prediction(cls):
    """Both routing contracts: staged == predicted minus the weakest pick."""
    torch.manual_seed(0)
    moe = cls(top_k=4)
    hidden = torch.randn(1, 1, 8)

    predicted, staged = predict_next_experts(_block(moe), hidden, top_k=4)

    w, idx = route_experts(moe, moe.gate(hidden), 4)
    order = w.reshape(-1)[torch.argsort(w.reshape(-1), descending=True)]
    weakest = int(idx.reshape(-1)[int(torch.argmin(w.reshape(-1)))])
    assert len(order) == 4
    assert predicted == set(int(x) for x in idx.reshape(-1).tolist())
    assert staged == predicted - {weakest}
    assert len(staged) == 3


def test_prediction_set_is_not_narrowed():
    """The accuracy telemetry must keep scoring the PREDICTOR, so the predicted
    set stays whole even though the staging set is narrowed."""
    torch.manual_seed(1)
    moe = _SoftmaxMoE(top_k=4)
    predicted, staged = predict_next_experts(_block(moe), torch.randn(1, 1, 8),
                                             top_k=4)
    assert len(predicted) == 4
    assert staged < predicted


def test_drop_is_per_row_not_global():
    """Batched decode routes N rows. Each row's own weakest pick is dropped —
    dropping one id globally would leave N-1 rows staging their whole top-k."""
    torch.manual_seed(2)
    moe = _SoftmaxMoE(top_k=4)
    hidden = torch.randn(1, 3, 8)          # three routed positions

    predicted, staged = predict_next_experts(_block(moe), hidden, top_k=4)

    w, idx = route_experts(moe, moe.gate(hidden), 4)
    w2, idx2 = w.reshape(-1, 4), idx.reshape(-1, 4)
    weakest = {int(idx2[r, int(torch.argmin(w2[r]))]) for r in range(idx2.shape[0])}
    kept = set()
    for r in range(idx2.shape[0]):
        wk = int(idx2[r, int(torch.argmin(w2[r]))])
        kept |= {int(e) for e in idx2[r].tolist() if int(e) != wk}
    assert predicted == set(int(x) for x in idx2.reshape(-1).tolist())
    assert staged == kept
    assert weakest - kept, "fixture is degenerate: every weakest pick is kept by another row"


def test_top_k_1_stages_its_only_prediction():
    """Nothing to drop when the router picks one expert — the rule must not
    empty the staging set."""
    torch.manual_seed(3)
    moe = _SoftmaxMoE(top_k=1)
    predicted, staged = predict_next_experts(_block(moe), torch.randn(1, 1, 8),
                                             top_k=1)
    assert len(predicted) == 1
    assert staged == predicted


def test_router_returned_topk_is_believed_and_narrowed():
    """A router that returns its own (logits, weights, indices) is believed, and
    its weakest pick is still the one dropped."""

    class _TupleRouter(nn.Module):
        def forward(self, x):
            n = x.reshape(-1, x.shape[-1]).shape[0]
            return (torch.zeros(n, 8),
                    torch.tensor([[0.6, 0.3, 0.1]] * n),
                    torch.tensor([[5, 6, 7]] * n))

    class _MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = _TupleRouter()
            self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])

    predicted, staged = predict_next_experts(_block(_MoE()),
                                             torch.randn(1, 1, 8), top_k=3)
    assert predicted == {5, 6, 7}
    assert staged == {5, 6}


def test_forward_stages_only_the_admitted_set(tmp_path):
    """End-to-end on the module-list forward: the stager is asked for the
    narrowed set, while _lookahead_pred still records the full prediction."""
    model, vaayu_ref, _cache = _patch_for_lookahead(tmp_path, num_experts=8,
                                                    top_k=4)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    layer, predicted = vaayu_ref._lookahead_pred
    assert layer == 1 and len(predicted) == 4
    submitted = {args[-1] for args in vaayu_ref._stager_executor.calls}
    assert len(submitted) == 3, f"stager was asked for {submitted}, want 3 of 4"
    assert submitted < predicted
