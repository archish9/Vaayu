"""Decode-time `top_k` truncation (row 33 / sequence step T2).

`bytes_read` and every per-slot phase scale with the number of routed experts,
so decoding at a smaller k than the checkpoint's `num_experts_per_tok` is a
direct speed lever. It is LOSSY by construction — greedy ids WILL diverge, so
id-identity is the wrong correctness check here (it is the right one for every
lossless arm in this repo).

Row 33 proposed clamping each MoE module's own `top_k` attribute by detection.
That is fragile: there are THREE routing owners in this engine, not the two the
row named --

  1. `route_experts`'s softmax fallback      -> reads the engine's `k` argument
  2. `moe_module.route_tokens_to_experts`    -> reads `moe_module.top_k`
  3. a router returning (logits, w, indices) -> reads `router.top_k`
     (Qwen3NextTopKRouter — the flagship int4 checkpoint)

and a missed owner is either a silent no-op or a shape crash. Truncating the
ROUTED TENSORS instead cannot silently no-op: it operates on what routing
actually returned, whoever produced it.
"""
import pytest
import torch

from vaayu.moe_patch import truncate_top_k


# CLAUDE.md §5: never one model's numbers.
GEOMETRIES = [
    pytest.param(128, 8, id="qwen3-235b-like-128e-top8"),
    pytest.param(256, 8, id="minimax-like-256e-top8"),
    pytest.param(512, 10, id="qwen3-next-like-512e-top10"),
]


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_truncation_keeps_the_k_largest_weights(num_experts, top_k):
    """The kept experts are exactly those carrying the largest combination
    weights — dropping the smallest contributions, not an arbitrary subset."""
    torch.manual_seed(0)
    logits = torch.randn(7, num_experts)
    w, idx = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)

    k = top_k - 2
    w_t, idx_t = truncate_top_k(w, idx, k)

    assert w_t.shape == (7, k)
    assert idx_t.shape == (7, k)
    for row in range(7):
        kept = set(idx_t[row].tolist())
        order = torch.argsort(w[row], descending=True)
        expected = set(idx[row][order[:k]].tolist())
        assert kept == expected


@pytest.mark.parametrize("num_experts,top_k", GEOMETRIES)
def test_truncated_weights_are_renormalised(num_experts, top_k):
    """Dropping slots removes their share of the convex mix. Without
    renormalisation the block's output is silently scaled DOWN by the dropped
    mass — a quiet quality loss on every layer, on top of the intended one."""
    torch.manual_seed(1)
    logits = torch.randn(5, num_experts)
    w, idx = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)

    w_t, _ = truncate_top_k(w, idx, top_k - 4)

    assert torch.allclose(w_t.sum(dim=-1), torch.ones(5), atol=1e-5)


@pytest.mark.parametrize("k", [None, 8, 9])
def test_off_path_returns_the_very_same_tensors(k):
    """k=None (feature off) and k >= K must be a true no-op — the SAME objects,
    not equal copies. MiniMax routes with `sorted=False`, so re-running topk
    would reorder the slots; expert accumulation is order-dependent
    (test_expert_accumulation_order), so a reorder is a bit-level change on a
    path the user did not turn on."""
    torch.manual_seed(2)
    w = torch.rand(4, 8)
    w = w / w.sum(dim=-1, keepdim=True)
    idx = torch.randint(0, 256, (4, 8))

    w_t, idx_t = truncate_top_k(w, idx, k)

    assert w_t is w
    assert idx_t is idx


# ── the three routing owners, driven through the real forwards ──────────────
#
# `_expert_history[layer]` is the list of experts the layer actually computed,
# written by all three forwards. With a single token it is exactly the routed
# slot set, so its length IS the number of slots the truncation was supposed to
# remove. A forward that never calls the truncation leaves it at K — which is
# the silent no-op this file exists to catch.
import torch.nn as nn

from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import (
    MockFusedModel, MockVaayuRef, _make_fused_checkpoint,
)

HIDDEN, NUM_EXPERTS, K, DECODE_K = 8, 8, 4, 2
BUDGET = 64 * 1024 * 1024


class _TupleRouter(nn.Module):
    """Owner 3 — Qwen3NextTopKRouter's contract: the ROUTER does the top-k and
    returns (logits, weights, indices). The engine's `top_k` argument never
    reaches selection, so clamping it alone is a no-op on this model."""

    def __init__(self, hidden, num_experts, top_k):
        super().__init__()
        self._gate = nn.Linear(hidden, num_experts, bias=False)
        self.num_experts = num_experts
        self.weight = self._gate.weight
        self.top_k = top_k

    def forward(self, x):
        logits = self._gate(x)
        w, idx = logits.float().softmax(-1).topk(self.top_k, dim=-1)
        return logits, w / w.sum(-1, keepdim=True), idx


def _fused(tmp_path, *, decode_k, prefill=False):
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=HIDDEN * 2, num_experts=NUM_EXPERTS)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    model.layers[0].mlp.gate = _TupleRouter(HIDDEN, NUM_EXPERTS, K)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._moe_prefill_pass = prefill
    ref._decode_top_k = decode_k
    patch_moe_blocks(
        model=model, expert_cache=ExpertRAMCache(budget_bytes=BUDGET),
        offset_index=idx, checkpoint_path=str(tmp_path), top_k=K,
        device="cpu", dtype=torch.float32, vaayu_ref=ref, loader_strategy=None,
    )
    return model, ref


def test_fused_forward_computes_only_decode_top_k_experts(tmp_path):
    """Owner 3. The router still selects K; the forward must compute k."""
    model, ref = _fused(tmp_path, decode_k=DECODE_K)

    model.layers[0].mlp.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == DECODE_K


def test_fused_forward_leaves_prefill_at_the_full_top_k(tmp_path):
    """Decode-only is the chosen scope: prefill routes ~every expert anyway
    (224.23/layer measured, SETTLED), so truncating it buys nearly nothing and
    costs prompt quality."""
    model, ref = _fused(tmp_path, decode_k=DECODE_K, prefill=True)

    model.layers[0].mlp.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == K


from tests.test_moe_patch import MockModel, _make_checkpoint


class _ContractBlock(nn.Module):
    """Owner 2 — MiniMax-M2's contract: selects on sigmoid + a correction bias
    with `sorted=False`, weights from the UNBIASED sigmoid, and takes NO `k`
    argument (it reads `self.top_k`). Truncating the returned tensors keeps the
    k largest COMBINATION WEIGHTS of its K, which is not identical to what the
    model would have selected at k — see the RESULTS doc."""

    def __init__(self, block, num_experts, top_k):
        super().__init__()
        self.gate = block.gate
        self.experts = block.experts
        self.top_k = top_k
        self.register_buffer("e_score_correction_bias", torch.randn(num_experts))

    def route_tokens_to_experts(self, router_logits):
        w = torch.sigmoid(router_logits.float())
        _, idx = torch.topk(w + self.e_score_correction_bias, self.top_k,
                            dim=-1, sorted=False)
        tw = w.gather(1, idx)
        return idx, (tw / tw.sum(dim=-1, keepdim=True)).to(router_logits.dtype)

    def forward(self, hidden_states, **kwargs):
        raise AssertionError("patched forward must replace this")


def _modulelist(tmp_path, *, decode_k, contract=False):
    model = MockModel(num_layers=1, hidden=HIDDEN)
    _make_checkpoint(tmp_path, model)
    if contract:
        layer = model.layers[0]
        layer.block_sparse_moe = _ContractBlock(
            layer.block_sparse_moe, NUM_EXPERTS, K)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._moe_prefill_pass = False
    ref._decode_top_k = decode_k
    patch_moe_blocks(
        model=model, expert_cache=ExpertRAMCache(budget_bytes=BUDGET),
        offset_index=idx, checkpoint_path=str(tmp_path), top_k=K,
        device="cpu", dtype=torch.float32, vaayu_ref=ref, loader_strategy=None,
    )
    return model, ref


def test_modulelist_forward_truncates_the_softmax_fallback_owner(tmp_path):
    """Owner 1 — `route_experts`'s softmax fallback, which DOES read the
    engine's `k`. Truncation must reach it too, or the engine clamps budgets
    the forward does not respect."""
    model, ref = _modulelist(tmp_path, decode_k=DECODE_K)

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == DECODE_K


def test_modulelist_forward_truncates_the_contract_owner(tmp_path):
    """Owner 2 — the model owns its scoring function and ignores the engine's
    `k` entirely."""
    model, ref = _modulelist(tmp_path, decode_k=DECODE_K, contract=True)

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == DECODE_K


from unittest.mock import MagicMock, patch as mock_patch
from pathlib import Path

from vaayu.moe_patch import (
    detect_moe_blocks, _make_fused_sparse_forward_exl3,
)
from tests.test_moe_patch import _make_mock_exl3_raw


class _LogitsRouter(nn.Module):
    """Qwen3-235B's own shape: the router returns plain logits, so the EXL3
    fused forward takes its cache-aware-softmax branch and the engine's `k`
    does the selecting."""

    def __init__(self, hidden, num_experts):
        super().__init__()
        self._gate = nn.Linear(hidden, num_experts, bias=False)
        self.num_experts = num_experts
        self.weight = self._gate.weight

    def forward(self, x):
        return self._gate(x)


def test_exl3_forward_computes_only_decode_top_k_experts(tmp_path):
    """The third forward. It shares the routing dispatch but not the call site,
    so it needs its own wiring — and its own proof."""
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    model.layers[0].mlp.gate = _LogitsRouter(HIDDEN, NUM_EXPERTS)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=HIDDEN * 2, num_experts=NUM_EXPERTS)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=BUDGET)
    for eidx in range(NUM_EXPERTS):
        cache.put(0, eidx, _make_mock_exl3_raw("layers.0", eidx,
                                               in_dim=HIDDEN, out_dim=HIDDEN * 2))
    ref = MockVaayuRef()
    ref._moe_prefill_pass = False
    ref._decode_top_k = DECODE_K

    block_info = detect_moe_blocks(model)[0]
    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = (
        lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16))
    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", return_value=mock_bc):
        block_info.moe_module.forward = _make_fused_sparse_forward_exl3(
            block_info=block_info, expert_cache=cache, offset_index=idx,
            checkpoint_path=Path(str(tmp_path)), top_k=K, device="cpu",
            dtype=torch.float32, vaayu_ref=ref,
        )
        model.layers[0].mlp.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == DECODE_K


# ── engine: resolving the requested k against the checkpoint's own ──────────
from types import SimpleNamespace

from vaayu.vaayu_base import VaayuBaseModel, resolve_decode_top_k


@pytest.mark.parametrize("config_k,requested,expected", [
    (10, None, 10),   # feature off -> the checkpoint's own value
    (10, 6, 6),
    (8, 6, 6),
    (10, 10, 10),
    (10, 99, 10),     # never ABOVE config: the router selected only 10
    (10, 0, 1),       # degenerate request still routes one expert
])
def test_resolve_decode_top_k(config_k, requested, expected):
    assert resolve_decode_top_k(config_k, requested) == expected


def test_a_request_above_the_checkpoints_top_k_warns(capsys):
    """CLAUDE.md §5: when the hardware/model genuinely cannot do what was
    asked, WARN — never silently degrade."""
    resolve_decode_top_k(8, 12)

    assert "WARNING" in capsys.readouterr().out


def test_detect_moe_publishes_the_clamped_decode_k():
    """`_detect_moe` is where `moe_top_k` is read off the config; the clamped
    decode value has to be published there too, because the geometry built
    right after it sizes the RAM/VRAM budgets from one decode step."""
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.config = SimpleNamespace(num_experts=512, num_experts_per_tok=10,
                               architectures=["Qwen3NextForCausalLM"])
    m._requested_decode_top_k = 6

    m._detect_moe()

    assert m.moe_top_k == 10, "prefill must keep the checkpoint's own top_k"
    assert m._decode_top_k == 6


def test_geometry_is_sized_from_the_decode_k(tmp_path):
    """`working_set_bytes` is ONE DECODE STEP by its own definition, and every
    RAM/VRAM expert budget is derived from it. Sizing it at the checkpoint's
    top_k while decode routes fewer slots reserves cache for experts that are
    never asked for — the waste row 31 closed, reintroduced."""
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                           intermediate=HIDDEN * 2, num_experts=NUM_EXPERTS)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.is_moe = True
    m.moe_top_k = K
    m._decode_top_k = DECODE_K
    m.moe_num_experts = NUM_EXPERTS
    m._expert_offset_index = idx
    m.layers = [None] * 4

    m._build_moe_geometry()

    assert m._moe_geometry["top_k"] == DECODE_K
