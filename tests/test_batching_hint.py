"""S56 — the server's client-fan-out (N>1 batching) hint.

The server alone knows the model, so it tells the client whether fanning out
concurrent requests into an N>1 batch is worth it. The rule (established by the
2026-10 batching sweep): batching HELPS only when the batch's lanes stream the
SAME bytes per decode step (dense — shared weights) and HURTS when they DIVERGE
(streamed MoE — the batch streams the per-chunk routing UNION off disk).

Two layers are tested here:
  * `derive_batching_hint` — the pure SHARE-vs-DIVERGE + admission rule.
  * `compute_and_store_batching_hint` — reads a loaded model and stores the
    static property on it (what EngineHandle calls once at load).
  * `batching_hint_payload` — folds the static property with the server's live
    batch config into the wire payload (`recommended_client_concurrency`).
  * the `/v1/models` and `/health` endpoints expose the payload.

Geometries parametrized per CLAUDE.md §5: Qwen3-Next-like (48L, 512 experts,
~1.69 MiB each) and MiniMax-like (62L, 256 experts, ~8 MiB each).
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vaayu.batch import derive_batching_hint, compute_and_store_batching_hint
from vaayu.server.app import batching_hint_payload, create_app, ServerConfig
from vaayu.vram_plan import VramPlanError

MiB = 1024 ** 2
GiB = 1024 ** 3

# (num_moe_layers, num_experts, expert_bytes) for two contrasting streamed MoE
# geometries. Both have a total far above any 16 GiB-card VRAM budget, so both
# stream and both DIVERGE.
STREAMED_MOE_GEOMETRIES = {
    "qwen3-next-like": (48, 512, int(1.69 * MiB)),
    "minimax-like": (62, 256, int(8 * MiB)),
}


# ── derive_batching_hint: the pure rule ──────────────────────────────────────

def test_dense_model_shares_and_helps_when_a_width2_batch_admits():
    helps, reason = derive_batching_hint(
        is_moe=False, total_expert_bytes=0, expert_vram_budget_bytes=0,
        prefix_lost_under_batch=False, width2_admits=True)
    assert helps is True
    assert "dense" in reason.lower() or "share" in reason.lower()


def test_dense_does_not_help_when_a_width2_batch_is_refused():
    helps, reason = derive_batching_hint(
        is_moe=False, total_expert_bytes=0, expert_vram_budget_bytes=0,
        prefix_lost_under_batch=False, width2_admits=False)
    assert helps is False
    assert "width-2" in reason.lower() or "admit" in reason.lower()


def test_dense_does_not_help_when_batching_drops_the_prefix_cache():
    helps, reason = derive_batching_hint(
        is_moe=False, total_expert_bytes=0, expert_vram_budget_bytes=0,
        prefix_lost_under_batch=True, width2_admits=True)
    assert helps is False
    assert "prefix" in reason.lower()


@pytest.mark.parametrize("geo", STREAMED_MOE_GEOMETRIES.values(),
                         ids=list(STREAMED_MOE_GEOMETRIES))
def test_streamed_moe_diverges_and_does_not_help(geo):
    layers, experts, ebytes = geo
    total = layers * experts * ebytes
    helps, reason = derive_batching_hint(
        is_moe=True, total_expert_bytes=total,
        expert_vram_budget_bytes=2 * GiB,   # far below `total` -> streams
        prefix_lost_under_batch=False, width2_admits=True)
    assert helps is False
    assert "stream" in reason.lower() or "union" in reason.lower()


def test_vram_resident_moe_is_false_but_flagged_unmeasured():
    # All experts fit the VRAM budget -> no per-chunk routing union -> it WOULD
    # share, but that is unmeasured (S55 on hold), so we do not advertise it.
    helps, reason = derive_batching_hint(
        is_moe=True, total_expert_bytes=1 * GiB,
        expert_vram_budget_bytes=10 * GiB,   # >= total -> fully resident
        prefix_lost_under_batch=False, width2_admits=True)
    assert helps is False
    assert "unmeasured" in reason.lower()


# ── compute_and_store_batching_hint: reads a loaded model ─────────────────────

def _fake_dense_model(*, admits=True, raises=False, max_seq_len=8192):
    def _assert(n_tokens, N=1, expert_giveback_bytes=0, batched_prefill=False,
                prompt_tokens=None):
        if raises:
            raise VramPlanError("context does not fit at width 2")
        return 123456 if admits else None

    return SimpleNamespace(
        is_moe=False, _moe_geometry=None, _expert_vram_cache=None,
        _cache_holds_recurrent_state=False, _cache_is_one_length=False,
        _cache_is_windowed=False, max_seq_len=max_seq_len,
        _assert_context_fits=_assert)


def test_store_sets_helps_true_on_an_admitting_dense_model():
    model = _fake_dense_model(admits=True)
    out = compute_and_store_batching_hint(model)
    assert model._batching_helps is True
    assert out["helps"] is True
    assert isinstance(model._batching_reason, str) and model._batching_reason


def test_store_sets_helps_false_when_width2_is_refused():
    model = _fake_dense_model(raises=True)
    compute_and_store_batching_hint(model)
    assert model._batching_helps is False
    assert "width-2" in model._batching_reason.lower() or "admit" in model._batching_reason.lower()


def test_store_sets_helps_false_when_there_is_no_vram_plan():
    # `_assert_context_fits` returns None on CPU / no plan -> cannot confirm a
    # width-2 batch admits -> conservatively False.
    model = _fake_dense_model(admits=False)
    compute_and_store_batching_hint(model)
    assert model._batching_helps is False


def test_store_never_raises_on_a_model_missing_attrs():
    model = SimpleNamespace()   # no is_moe, no _assert_context_fits, nothing
    out = compute_and_store_batching_hint(model)
    assert model._batching_helps is False
    assert out["helps"] is False


# ── batching_hint_payload: fold in the server's live batch config ─────────────

def _sched(*, enabled, mbs):
    return SimpleNamespace(batching_enabled=enabled, max_batch_size=mbs)


def _model(*, helps, reason="because"):
    return SimpleNamespace(_batching_helps=helps, _batching_reason=reason)


def test_payload_recommends_serial_when_the_model_does_not_benefit():
    p = batching_hint_payload(_model(helps=False), _sched(enabled=True, mbs=3))
    assert p["helps"] is False
    assert p["recommended_client_concurrency"] == 1


def test_payload_recommends_serial_when_the_server_batching_is_off():
    p = batching_hint_payload(_model(helps=True), _sched(enabled=False, mbs=3))
    assert p["recommended_client_concurrency"] == 1


def test_payload_recommends_max_batch_size_when_it_helps_and_is_enabled():
    p = batching_hint_payload(_model(helps=True), _sched(enabled=True, mbs=3))
    assert p["helps"] is True
    assert p["recommended_client_concurrency"] == 3


# ── endpoints expose the payload ──────────────────────────────────────────────

def _hint_engine(*, helps, attn="sdpa"):
    """The project's FakeEngine (complete enough for the app lifespan) with the
    load-time batching attrs set on its model."""
    from .test_server_app import FakeEngine
    eng = FakeEngine()
    eng.model._active_attn_impl = attn
    eng.model._batching_helps = helps
    eng.model._batching_reason = "dense: all lanes share identical streamed weights"
    return eng


def _client(engine, **cfg_overrides):
    cfg = ServerConfig(model_path="unused", **cfg_overrides)
    return TestClient(create_app(config=cfg, engine=engine))


def test_v1_models_exposes_the_batching_hint_with_recommended_concurrency():
    with _client(_hint_engine(helps=True),
                 max_batch_size=2, attn_implementation="sdpa") as c:
        data = c.get("/v1/models").json()["data"][0]
        assert data["batching"]["helps"] is True
        assert data["batching"]["recommended_client_concurrency"] == 2
        assert isinstance(data["batching"]["reason"], str)


def test_health_exposes_the_batching_hint():
    with _client(_hint_engine(helps=True),
                 max_batch_size=2, attn_implementation="sdpa") as c:
        body = c.get("/health").json()
        assert body["batching_hint"]["helps"] is True
        assert body["batching_hint"]["recommended_client_concurrency"] == 2


def test_v1_models_recommends_serial_when_server_batching_is_off():
    with _client(_hint_engine(helps=True)) as c:   # default MBS=1 -> disabled
        data = c.get("/v1/models").json()["data"][0]
        assert data["batching"]["recommended_client_concurrency"] == 1
