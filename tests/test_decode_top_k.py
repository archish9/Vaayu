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


# ── the lossy knob has to announce itself ───────────────────────────────────
#
# DECODE_TOP_K is the only knob in vaayu/server/config.py whose wrong value
# fails SILENTLY: no error, no crash, just quietly worse output. Every other
# default there preserves the model's output exactly. CLAUDE.md §5 asks for a
# warning rather than a silent degradation, and a degradation the operator
# opted into still needs saying -- they cannot know THIS checkpoint's cost
# unless they measured it, and the two models measured here differ by 4x.


def test_an_active_clamp_warns_that_output_is_degraded(capsys):
    resolve_decode_top_k(10, 6)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "DECODE_TOP_K" in out


@pytest.mark.parametrize("requested", [None, 10])
def test_no_warning_when_the_clamp_is_inert(capsys, requested):
    """None is the feature off, and k == the checkpoint's own value routes
    exactly what it always routed. Neither degrades anything, so neither may
    cry wolf."""
    resolve_decode_top_k(10, requested)

    assert "WARNING" not in capsys.readouterr().out


# ── routing-mass cutoff: DECODE_TOP_K="auto" (spec §7) ─────────────────────
#
# The cutoff keeps, per token, the smallest set of routed slots whose weight
# sums to >= p (always >= 1) and RE-POINTS each dropped slot at that token's
# top-1 expert with weight 0. Shapes never change, so no forward's dispatch
# code changes, and a dropped slot loads nothing: it duplicates an expert the
# token already routes to. That is the exact mechanism whose speed was measured
# (+12.9 % Qwen3-Next, +10.75 % MiniMax at p=0.90).


def test_cutoff_repoints_dropped_slots_at_the_top1_expert():
    w = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    i = torch.tensor([[7, 3, 9, 1]])

    w2, i2 = truncate_top_k(w, i, None, p=0.9)

    assert i2.tolist() == [[7, 3, 9, 7]]
    assert w2[0, 3].item() == 0.0
    assert abs(w2.sum().item() - 1.0) < 1e-6


def test_cutoff_finds_the_top1_by_weight_on_unsorted_routing():
    """MiniMax routes with sorted=False: slot 0 is not the top expert."""
    w = torch.tensor([[0.05, 0.5, 0.15, 0.3]])
    i = torch.tensor([[1, 7, 9, 3]])

    w2, i2 = truncate_top_k(w, i, None, p=0.9)

    assert i2.tolist() == [[7, 7, 9, 3]]
    assert w2[0, 0].item() == 0.0


def test_cutoff_always_keeps_the_top1():
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    i = torch.tensor([[5, 6, 7, 8]])

    w2, i2 = truncate_top_k(w, i, None, p=0.01)

    assert set(i2.flatten().tolist()) == {5}
    assert abs(w2[0, 0].item() - 1.0) < 1e-6


def test_no_cutoff_and_no_clamp_returns_the_very_same_tensors():
    w = torch.rand(3, 8)
    w = w / w.sum(dim=-1, keepdim=True)
    i = torch.randint(0, 64, (3, 8))

    a, b = truncate_top_k(w, i, 8, p=None)

    assert a is w and b is i


def test_fused_forward_cutoff_computes_only_the_kept_experts(tmp_path):
    """p=0.01 keeps exactly the top-1 per token, whatever the weights."""
    model, ref = _fused(tmp_path, decode_k=K)
    ref._decode_top_p = 0.01

    model.layers[0].mlp.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == 1


def test_fused_forward_cutoff_is_off_during_prefill(tmp_path):
    model, ref = _fused(tmp_path, decode_k=K, prefill=True)
    ref._decode_top_p = 0.01

    model.layers[0].mlp.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == K


def test_modulelist_forward_cutoff_reaches_the_softmax_owner(tmp_path):
    model, ref = _modulelist(tmp_path, decode_k=K)
    ref._decode_top_p = 0.01

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == 1


def test_modulelist_forward_cutoff_reaches_the_contract_owner(tmp_path):
    model, ref = _modulelist(tmp_path, decode_k=K, contract=True)
    ref._decode_top_p = 0.01

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, HIDDEN))

    assert len(ref._expert_history[0]) == 1


def test_exl3_forward_cutoff_computes_only_the_kept_experts(tmp_path):
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
    ref._decode_top_k = K
    ref._decode_top_p = 0.01

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

    assert len(ref._expert_history[0]) == 1


# ── engine wiring for DECODE_TOP_K="auto" (spec §7) ─────────────────────
from vaayu.vaayu_base import DEFAULT_DECODE_TOP_K_BUDGET, validate_decode_top_k_budget


def test_auto_resolves_to_the_checkpoints_own_k_at_load(capsys):
    assert resolve_decode_top_k(10, "auto") == 10
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["fast", "8", True, 7.5])
def test_an_unknown_decode_top_k_value_raises(bad):
    with pytest.raises(ValueError, match="auto"):
        resolve_decode_top_k(10, bad)


@pytest.mark.parametrize("budget", [0, 1, -0.1, 1.5])
def test_a_budget_outside_zero_one_raises(budget):
    with pytest.raises(ValueError):
        validate_decode_top_k_budget(budget)


def test_a_valid_budget_is_returned():
    assert validate_decode_top_k_budget(0.035) == 0.035


def _bare_moe(requested, budget=DEFAULT_DECODE_TOP_K_BUDGET):
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.config = SimpleNamespace(num_experts=512, num_experts_per_tok=10,
                               architectures=["Qwen3NextForCausalLM"])
    m._requested_decode_top_k = requested
    m._decode_top_k_budget = budget
    return m


def test_detect_moe_marks_auto_pending_without_clamping():
    m = _bare_moe("auto")

    m._detect_moe()

    assert (m._decode_top_k, m._decode_top_k_auto) == (10, True)
    assert m.decode_top_k_status()["mode"] == "off"


def test_detect_moe_reports_an_explicit_clamp_as_config():
    m = _bare_moe(8)

    m._detect_moe()

    assert m.decode_top_k_status() == {"mode": "k", "value": 8, "cost": None, "source": "config"}


def test_a_budget_with_an_explicit_int_is_noted(capsys):
    m = _bare_moe(8, budget=0.02)

    m._detect_moe()

    assert "only applies" in capsys.readouterr().out


def test_auto_on_a_dense_checkpoint_is_a_noted_no_op(capsys):
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    m._requested_decode_top_k = "auto"
    m._decode_top_k_budget = DEFAULT_DECODE_TOP_K_BUDGET

    m._detect_moe()

    assert m._decode_top_k_auto is False
    assert "not an MoE checkpoint" in capsys.readouterr().out


def test_the_hook_runs_resolve_only_when_auto(monkeypatch):
    import vaayu.decode_top_k_auto as dtka
    seen = []

    def fake_resolve(model, budget):
        seen.append(budget)
        return {"mode": "k", "value": 8, "cost": 0.03, "source": "measured"}

    monkeypatch.setattr(dtka, "resolve", fake_resolve)
    m = _bare_moe("auto", budget=0.02)
    m._detect_moe()
    m._resolve_decode_top_k_auto()
    assert seen == [0.02]
    assert m.decode_top_k_status()["source"] == "measured"

    off = _bare_moe(None)
    off._detect_moe()
    off._resolve_decode_top_k_auto()
    assert seen == [0.02]


def test_the_hook_is_the_last_statement_of_init():
    """Measurement needs a fully constructed model -- prefix cache, KV persist
    and the post-load RAM ledger included. AutoModel.from_pretrained does no
    setup after construction, so the end of __init__ IS load complete."""
    import inspect
    lines = inspect.getsource(VaayuBaseModel.__init__).rstrip().splitlines()
    assert lines[-1].strip() == "self._resolve_decode_top_k_auto()"


@pytest.mark.parametrize("bad", ["fast", "8", True, 7.5])
def test_an_unknown_decode_top_k_value_raises_on_a_dense_checkpoint_too(bad):
    """The load-time ValueError must not depend on the architecture: a dense
    checkpoint never reaches resolve_decode_top_k, so it validates on its own."""
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    m._requested_decode_top_k = bad
    m._decode_top_k_budget = DEFAULT_DECODE_TOP_K_BUDGET
    with pytest.raises(ValueError, match="auto"):
        m._detect_moe()
