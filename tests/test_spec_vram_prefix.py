"""P9: speculative_generate must size the VRAM cacheable prefix for its verify width.

A verify pass is (draft_tokens + 1) positions pushed through every MoE layer in
ONE forward, so its per-layer insert load is the routing union over those
positions -- structurally identical to P6's batch width, and bounded by the same
`compute_spec_working_set` bound Fix #9 already uses for admission. The load-time
call (vaayu_base.py:2610) charges one stream's top_k, and speculative_generate
never re-sizes, so at the shipped draft_tokens=3 the prefix is charged 4x under
what the forward inserts: MiniMax caches ALL 62 layers, which is exactly the
configuration P6 measured at a 2.35% hit rate against 48.4% when sized right.

These tests drive the REAL speculative_generate against the smallest
model-shaped fake that reaches its verify loop.
"""
from types import SimpleNamespace

import pytest
import torch

from vaayu.vaayu_base import (VaayuBaseModel, batch_union_per_layer,
                              compute_vram_layer_range)

QWEN_GEO = {"expert_bytes": int(11.83e6), "top_k": 8, "num_moe_layers": 94,
            "num_experts": 128, "working_set_bytes": 8 * 94 * int(11.83e6)}
QWEN_BUDGET = int(4.09e9)
# MiniMax is the case with NO load-time restriction at all (its single-stream
# working set fits), so it is the one that must gain one during speculation --
# and it is the model P3 measured speculation's 1.307x on.
MINIMAX_GEO = {"expert_bytes": int(8.88e6), "top_k": 8, "num_moe_layers": 62,
               "num_experts": 256, "working_set_bytes": 8 * 62 * int(8.88e6)}
MINIMAX_BUDGET = int(7.45e9)

GEOMETRIES = [("qwen", QWEN_GEO, QWEN_BUDGET, list(range(3, 97))),
              ("minimax", MINIMAX_GEO, MINIMAX_BUDGET, list(range(62)))]

DRAFT_TOKENS = 3          # P3's shipped default; 4 verify positions


class _FakeCache:
    def __init__(self, s):
        self._s = s

    def get_seq_length(self):
        return self._s


class _FakeSpeculator:
    """Drafts `num_tokens` ids and has exactly one of them accepted."""

    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def draft(self, generated, past_key_values, num_tokens):
        return (torch.full((1, num_tokens), 7, dtype=generated.dtype,
                           device=generated.device),
                SimpleNamespace())

    def verify_and_accept(self, generated, draft_ids, logits, state):
        return torch.full((1, 1), 42, dtype=generated.dtype,
                          device=generated.device), 1

    def rollback_cache(self, state, num_accepted):
        pass


class _FakeMoEModel:
    """Enough of VaayuBaseModel for speculative_generate to reach its verify
    loop, plus the three MoE attributes the prefix sizing reads."""
    running_device = "cpu"
    _prefix_cache = None
    speculative_status = None

    class _Cfg:
        eos_token_id = -999          # unreachable: the fake never emits it

    class _Telemetry:
        def dump_json(self, path):
            pass

    def __init__(self, geo, budget, layer_ids, load_time_range,
                 draft_tokens=DRAFT_TOKENS):
        self._moe_geometry = geo
        self._moe_blocks = [SimpleNamespace(layer_idx=i) for i in layer_ids]
        self._expert_vram_cache = SimpleNamespace(budget_bytes=budget)
        self._vram_cacheable_layers = load_time_range
        self.draft_tokens = draft_tokens
        self.speculator = _FakeSpeculator()
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()
        self.seen_during_forward = []
        self.verify_widths = []

    def _build_spec_gate(self):
        # None is the steady state this sizing is for: the gate latches RUN for
        # the process, so every later round drafts at full width.
        return None

    def _assert_context_fits(self, n_tokens):
        pass

    def _create_kv_cache(self):
        return _FakeCache(s=0)

    def _trim_kv_cache(self, past_key_values, length):
        past_key_values._s = length

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                return_dict=True):
        # What the MoE forwards read per layer-visit -- record what they WOULD see.
        self.seen_during_forward.append(self._vram_cacheable_layers)
        self.verify_widths.append(input_ids.shape[1])
        seq = (past_key_values.get_seq_length() if past_key_values is not None
               else 0) + input_ids.shape[1]
        return SimpleNamespace(
            past_key_values=_FakeCache(s=seq),
            logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 8))


def _model(geo, budget, layer_ids, draft_tokens=DRAFT_TOKENS):
    load_time = compute_vram_layer_range(budget, geo, layer_ids)
    return (_FakeMoEModel(geo, budget, layer_ids, load_time, draft_tokens),
            load_time)


def _run(model, max_new_tokens=3):
    return VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4]]), max_new_tokens=max_new_tokens)


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_verify_prefix_is_sized_for_the_verify_width(name, geo, budget, layer_ids):
    model, load_time = _model(geo, budget, layer_ids)
    expected = compute_vram_layer_range(
        budget, geo, layer_ids,
        # Same min(num_experts, positions * top_k) bound, as its docstring says.
        experts_per_layer=batch_union_per_layer(geo, DRAFT_TOKENS + 1))

    _run(model)

    assert DRAFT_TOKENS + 1 in model.verify_widths, "verify pass never ran"
    assert model.seen_during_forward, "forward never ran"
    assert all(s == expected for s in model.seen_during_forward)
    assert expected is not None and len(expected) < len(layer_ids)
    # and it is genuinely tighter than what the load-time sizing left in place
    assert load_time is None or len(expected) < len(load_time)


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_load_time_prefix_is_restored_after_speculation(name, geo, budget,
                                                        layer_ids):
    model, load_time = _model(geo, budget, layer_ids)
    _run(model)
    assert model._vram_cacheable_layers == load_time


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_prefix_is_restored_when_the_loop_raises(name, geo, budget, layer_ids):
    """The loop has several exit points and no pre-existing finally; a leaked
    verify-sized prefix would silently shrink caching for every later call."""
    model, load_time = _model(geo, budget, layer_ids)
    boom = RuntimeError("verify blew up")

    def _raise(*a, **kw):
        raise boom
    model.forward = _raise

    with pytest.raises(RuntimeError):
        _run(model)
    assert model._vram_cacheable_layers == load_time


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_prefix_is_restored_on_the_early_eos_return(name, geo, budget, layer_ids):
    """max_new_tokens=1 returns straight after the prefill seed, before the
    verify loop -- an exit path of its own."""
    model, load_time = _model(geo, budget, layer_ids)
    _run(model, max_new_tokens=1)
    assert model._vram_cacheable_layers == load_time


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_draft_tokens_0_leaves_the_load_time_prefix_untouched(name, geo, budget,
                                                              layer_ids):
    """A 1-position verify is an ordinary decode: the union IS top_k, so the
    recomputed set equals the load-time one and nothing may be touched."""
    model, load_time = _model(geo, budget, layer_ids, draft_tokens=0)
    _run(model)
    assert all(s == load_time for s in model.seen_during_forward)
    assert model._vram_cacheable_layers == load_time


def test_model_without_a_vram_cache_is_untouched():
    """No expert VRAM cache (dense, or MoE loaded without one): the sizing must
    not run at all (CLAUDE.md §0: MoE work must not touch the dense path)."""
    model, _ = _model(QWEN_GEO, QWEN_BUDGET, list(range(94)))
    model._vram_cacheable_layers = None
    del model._expert_vram_cache
    _run(model)
    assert model._vram_cacheable_layers is None
    assert all(s is None for s in model.seen_during_forward)


class _FakeGate:
    """The runtime gate's only question this sizing may ask."""

    def __init__(self, drafts):
        self._drafts = drafts
        self.phase = "run" if drafts else "calibrate"
        self.decision = "run" if drafts else None

    def should_draft(self):
        return self._drafts

    def observe(self, history, elapsed_s=None):
        pass

    def record_round(self, drafted, accepted, elapsed_s=None,
                     draft_overhead_s=None):
        pass

    def decided_off(self):
        return False


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_a_gate_that_will_not_draft_leaves_the_load_time_prefix_alone(
        name, geo, budget, layer_ids):
    """A round the gate refuses to draft is an ordinary 1-wide decode, so the
    verify-width charge is simply wrong for it. Measured on Qwen3-235B
    2026-09-07: its gate latches off_union after calibration and NEVER drafts,
    so the whole call ran 1-wide under a 4-wide prefix -- 65/94 layers -> 16/94
    and 0.9740 -> 0.9423 tok/s, -3.25%."""
    model, load_time = _model(geo, budget, layer_ids)
    model._build_spec_gate = lambda: _FakeGate(drafts=False)

    _run(model)

    assert all(s == load_time for s in model.seen_during_forward)
    assert model._vram_cacheable_layers == load_time


def _gate_at_a_suppressed_trial_leg(geo):
    """A REAL gate, mid-TRIAL, whose next round falls in a suppressed leg --
    the state any call after the first can now start in (row 13, `even-4`)."""
    from vaayu.speculative import adaptive_gate as ag
    gate = ag.AdaptiveSpecGate(geo, DRAFT_TOKENS, budget_bytes=10 ** 12)
    for _ in range(max(ag.CALIBRATE_STEPS, DRAFT_TOKENS + 1)):
        gate.observe({l: list(range(8)) for l in range(geo["num_moe_layers"])})
    for _ in range(ag.TRIAL_LEG_ROUNDS):
        gate.record_round(DRAFT_TOKENS, DRAFT_TOKENS)
    assert gate.phase == "trial" and gate.should_draft() is False
    return gate


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_a_call_starting_in_a_suppressed_trial_leg_still_gets_the_verify_width_prefix(
        name, geo, budget, layer_ids):
    """The per-call read asks whether this CALL will speculate, not whether
    its first round drafts. TRIAL alternates legs, so a call that opens on a
    suppressed leg still drafts a few rounds later -- and those verify passes
    need the verify-width prefix. Charging the suppressed rounds for it
    over-charges, the safe direction."""
    model, _ = _model(geo, budget, layer_ids)
    gate = _gate_at_a_suppressed_trial_leg(geo)
    model._build_spec_gate = lambda: gate
    expected = compute_vram_layer_range(
        budget, geo, layer_ids,
        experts_per_layer=batch_union_per_layer(geo, DRAFT_TOKENS + 1))

    _run(model)

    assert model.seen_during_forward, "forward never ran"
    assert all(s == expected for s in model.seen_during_forward)


@pytest.mark.parametrize("name,geo,budget,layer_ids", GEOMETRIES)
def test_a_gate_that_drafts_still_gets_the_verify_width_prefix(
        name, geo, budget, layer_ids):
    """The steady state: once the gate has latched run it drafts every round,
    and that is the case the whole row exists for."""
    model, _ = _model(geo, budget, layer_ids)
    model._build_spec_gate = lambda: _FakeGate(drafts=True)
    expected = compute_vram_layer_range(
        budget, geo, layer_ids,
        experts_per_layer=batch_union_per_layer(geo, DRAFT_TOKENS + 1))

    _run(model)

    assert DRAFT_TOKENS + 1 in model.verify_widths, "verify pass never ran"
    assert all(s == expected for s in model.seen_during_forward)
