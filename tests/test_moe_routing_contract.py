"""MoE routing must follow the MODEL's own scoring function, not a hardcoded
softmax (CLAUDE.md §5: detection over assumption).

Root cause this guards: `_make_modulelist_sparse_forward` hardcoded
softmax-topk. MiniMax-M2 scores with sigmoid + an e_score_correction_bias
(`scoring_func: "sigmoid"`, `use_routing_bias: true`), so Vaayu produced
badly wrong combination weights on every MoE layer -> incoherent output.
Selection happened to survive (topk is monotone in the logits either way);
the WEIGHTS did not.

transformers standardizes `route_tokens_to_experts(router_logits)` across 25+
MoE architectures (DeepSeek v2/v3, GLM-MoE, dots1, dbrx, jamba, MiniMax...).
Qwen3-MoE and Mixtral do NOT define it and genuinely use softmax-topk, so the
fallback must stay byte-identical to today's expression for them.
"""
import pytest
import torch

from vaayu.moe_patch import route_experts


# ── geometry parametrization (CLAUDE.md §5: never one model's numbers) ──
GEOMETRIES = [
    pytest.param(128, 8, id="qwen-like-128e-top8"),
    pytest.param(256, 8, id="minimax-like-256e-top8"),
]


class _SigmoidBiasRouter(torch.nn.Module):
    """MiniMax/DeepSeek-family contract: returns (indices, weights), scores with
    sigmoid, selects on sigmoid+bias, weights from the UNBIASED sigmoid."""

    def __init__(self, num_experts, top_k, bias=None):
        super().__init__()
        self.top_k = top_k
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts) if bias is None else bias,
        )

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)


class _WeightsFirstRouter(torch.nn.Module):
    """dbrx-family contract: same method, but returns (weights, indices)."""

    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def route_tokens_to_experts(self, router_logits):
        probs = torch.softmax(router_logits.float(), dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        return w, idx


class _NoContractRouter(torch.nn.Module):
    """Qwen3-MoE / Mixtral: no route_tokens_to_experts at all."""


def _softmax_reference(router_logits, top_k):
    """Verbatim the expression route_experts must fall back to."""
    w, idx = torch.topk(router_logits.float().softmax(dim=-1), top_k, dim=-1)
    return w / w.sum(dim=-1, keepdim=True), idx


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_delegates_to_model_contract_indices_first(num_experts, top_k):
    """Sigmoid-scored model: weights must match the MODEL's sigmoid routing,
    NOT softmax. This is the regression that produced gibberish."""
    torch.manual_seed(0)
    mod = _SigmoidBiasRouter(num_experts, top_k)
    logits = torch.randn(4, num_experts)

    w, idx = route_experts(mod, logits, top_k)

    exp_idx, exp_w = mod.route_tokens_to_experts(logits)
    assert torch.equal(idx, exp_idx)
    assert torch.allclose(w.float(), exp_w.float(), atol=1e-6)

    # ...and must be materially DIFFERENT from the old softmax behaviour,
    # otherwise this test would pass even against the buggy implementation.
    soft_w, _ = _softmax_reference(logits, top_k)
    assert not torch.allclose(w.float(), soft_w.float(), atol=1e-3)


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_correction_bias_changes_selection(num_experts, top_k):
    """A non-zero e_score_correction_bias must steer SELECTION (it is added to
    the choice score). Ignoring it picks the wrong experts outright."""
    torch.manual_seed(1)
    logits = torch.randn(4, num_experts)
    bias = torch.zeros(num_experts)
    bias[-1] = 50.0  # force the last expert into every row's top-k

    mod = _SigmoidBiasRouter(num_experts, top_k, bias=bias)
    _, idx = route_experts(mod, logits, top_k)

    assert (idx == num_experts - 1).any(dim=-1).all(), \
        "correction bias ignored: biased expert not selected"


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_handles_weights_first_return_order(num_experts, top_k):
    """dbrx returns (weights, indices) while DeepSeek/MiniMax return
    (indices, weights). Order must be resolved by dtype, not position."""
    torch.manual_seed(2)
    mod = _WeightsFirstRouter(top_k)
    logits = torch.randn(4, num_experts)

    w, idx = route_experts(mod, logits, top_k)

    assert w.is_floating_point()
    assert idx.dtype in (torch.int32, torch.int64)
    assert idx.shape == (4, top_k) and w.shape == (4, top_k)
    assert (idx >= 0).all() and (idx < num_experts).all()


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_fallback_is_byte_identical_to_softmax(num_experts, top_k):
    """Qwen3-MoE / Mixtral have no contract method and genuinely use
    softmax-topk. The fallback must not change their behaviour at all."""
    torch.manual_seed(3)
    mod = _NoContractRouter()
    logits = torch.randn(6, num_experts)

    w, idx = route_experts(mod, logits, top_k)
    exp_w, exp_idx = _softmax_reference(logits, top_k)

    assert torch.equal(idx, exp_idx)
    assert torch.equal(w, exp_w)


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_preserves_3d_batch_seq_shape(num_experts, top_k):
    """The module-list forward feeds (B, S, E) logits and indexes the results
    as (b, s, k). The model contract only accepts 2-D, so route_experts must
    flatten and restore around it."""
    torch.manual_seed(4)
    B, S = 2, 5
    mod = _SigmoidBiasRouter(num_experts, top_k)
    logits = torch.randn(B, S, num_experts)

    w, idx = route_experts(mod, logits, top_k)

    assert idx.shape == (B, S, top_k)
    assert w.shape == (B, S, top_k)

    flat_idx, flat_w = mod.route_tokens_to_experts(logits.reshape(-1, num_experts))
    assert torch.equal(idx.reshape(-1, top_k), flat_idx)
    assert torch.allclose(w.reshape(-1, top_k).float(), flat_w.float(), atol=1e-6)


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_model_weights_passed_through_verbatim(num_experts, top_k):
    """route_experts must NOT renormalize what the model returned. Normalization
    is the model's business and is not uniform: MiniMax divides by the top-k
    sum, dbrx applies an OPTIONAL p-norm (`moe_normalize_expert_weights`, may be
    None -> raw softmax values that do not sum to 1). Re-normalizing here would
    silently corrupt every model that deliberately does not."""
    torch.manual_seed(5)
    logits = torch.randn(4, num_experts)

    mod = _WeightsFirstRouter(top_k)          # returns un-normalized top-k probs
    w, _ = route_experts(mod, logits, top_k)

    exp_w, _ = mod.route_tokens_to_experts(logits)
    assert torch.equal(w, exp_w)
    # Guard the guard: this fixture's weights genuinely do not sum to 1, so a
    # renormalizing implementation would fail the equality above.
    assert not torch.allclose(w.float().sum(-1), torch.ones(4), atol=1e-3)


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_fallback_and_sum_normalizing_models_are_convex(num_experts, top_k):
    """Where the model (or the fallback) does normalize by the sum, the result
    is a convex mix — the property the expert-combination step relies on."""
    torch.manual_seed(6)
    logits = torch.randn(4, num_experts)
    for mod in (_SigmoidBiasRouter(num_experts, top_k), _NoContractRouter()):
        w, _ = route_experts(mod, logits, top_k)
        assert torch.allclose(w.float().sum(-1), torch.ones(4), atol=1e-5), \
            f"{type(mod).__name__} weights not normalized"
