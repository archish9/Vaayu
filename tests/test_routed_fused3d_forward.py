"""S8 Task 2: routed delivery through the whole-layer forward.

Routed and whole-layer delivery run the SAME module forward over the same bytes for
every routed expert, so their outputs must be torch.equal -- not close.
"""
import pytest
import torch

from tests.routed_fused3d_fakes import NUM_EXPERTS, WIDTH, Ref, build, patch, routed_ok
from vaayu.expert_usage import ExpertUsageTracker


def _run(block, hidden):
    out, _logits = block.moe_module.forward(hidden)
    return out


def _whole(ref, block, hidden):
    ref._routed_fused3d = False
    try:
        return _run(block, hidden)
    finally:
        ref._routed_fused3d = True


def test_delivery_tag_says_which_path_is_installed(tmp_path):
    model, idx, _ = build(tmp_path)
    blocks, _ = patch(tmp_path, model, idx, Ref(model, routed_ok(idx)))
    assert blocks[0].forward_name == "_make_whole_layer_sparse_forward"
    assert blocks[0].delivery == "routed"


def test_no_verdict_keeps_whole_layer_delivery(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model)
    blocks, _ = patch(tmp_path, model, idx, ref)
    assert blocks[0].delivery == "whole"
    _run(blocks[0], torch.randn(3, WIDTH))
    assert ref._expert_history[0] == list(range(NUM_EXPERTS))


def test_routed_output_equals_whole_layer_and_the_reference(tmp_path):
    model, idx, weights = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(3, WIDTH)

    routed = _run(blocks[0], hidden)
    whole = _whole(ref, blocks[0], hidden)

    assert torch.equal(routed, whole)
    w, b = weights[0]
    router = blocks[0].moe_module.router
    scal, eidx = torch.topk(router(hidden).softmax(-1), 2, dim=-1)
    expected = torch.stack([sum(scal[t, j] * (hidden[t] @ w[eidx[t, j]] + b[eidx[t, j]])
                                for j in range(2)) for t in range(3)])
    torch.testing.assert_close(routed, expected)


def test_history_is_the_routed_set_and_the_module_ends_on_meta(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(1, WIDTH)
    _run(blocks[0], hidden)
    router = blocks[0].moe_module.router
    want = sorted(set(torch.topk(router(hidden), 2, dim=-1).indices.flatten().tolist()))
    assert ref._expert_history[0] == want
    experts = blocks[0].moe_module.experts
    assert experts.gate_up_proj.is_meta and experts.gate_up_proj_bias.is_meta
    assert "forward" in experts.__dict__, "the wrapper must survive the release"


def test_stale_rows_in_the_buffer_are_never_read(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(2, WIDTH)
    whole = _whole(ref, blocks[0], hidden)
    for buf in next(iter(ref._fused3d_buffers.values())).values():
        buf.fill_(float("nan"))
    routed = _run(blocks[0], hidden)
    assert torch.isfinite(routed).all()
    assert torch.equal(routed, whole)


def test_two_layers_share_the_buffer_without_leaking_rows(tmp_path):
    model, idx, _ = build(tmp_path, n_layers=2)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    assert len(ref._fused3d_buffers) == 1, "one geometry -> one shared buffer"
    hidden = torch.randn(2, WIDTH)
    r0, r1 = _run(blocks[0], hidden), _run(blocks[1], hidden)
    w0, w1 = _whole(ref, blocks[0], hidden), _whole(ref, blocks[1], hidden)
    assert torch.equal(r0, w0) and torch.equal(r1, w1)
    assert not torch.equal(r0, r1)


def test_a_decode_visit_caches_the_routed_experts(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, cache = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(1, WIDTH)
    _run(blocks[0], hidden)
    assert cache.resident_set(0) == set(ref._expert_history[0])
    hits_before = cache.hits
    _run(blocks[0], hidden)
    assert cache.hits > hits_before


def test_a_decode_visit_records_routed_experts_when_usage_is_on(tmp_path):
    """S36 item 3: the routed fused-3D forward must feed the expert-usage histogram
    like the other MoE forwards do; before the fix `_maybe_record_usage` was never
    called on this path, so ENABLE_EXPERT_USAGE recorded nothing on gpt-oss."""
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    ref._expert_usage = ExpertUsageTracker(n_rows=4, n_experts=NUM_EXPERTS, device="cpu")
    blocks, _ = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(1, WIDTH)
    router = blocks[0].moe_module.router
    want = sorted(set(torch.topk(router(hidden), 2, dim=-1).indices.flatten().tolist()))
    _run(blocks[0], hidden)
    counts = ref._expert_usage.snapshot()
    assert counts[0].sum().item() == len(want)
    assert [e for e in range(NUM_EXPERTS) if counts[0, e].item() > 0] == want
    assert all(counts[0, e].item() == 1 for e in want)


def test_a_prefill_visit_records_no_usage(tmp_path):
    """Prefill routes are skipped on purpose (see _maybe_record_usage): a prefill
    pass fills no VRAM cache, so counting its routes would skew the report."""
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    ref._expert_usage = ExpertUsageTracker(n_rows=4, n_experts=NUM_EXPERTS, device="cpu")
    blocks, _ = patch(tmp_path, model, idx, ref)
    ref._moe_prefill_pass = True
    _run(blocks[0], torch.randn(6, WIDTH))
    assert ref._expert_usage.snapshot().sum().item() == 0


def test_a_prefill_visit_caches_nothing(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, cache = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(6, WIDTH)
    ref._moe_prefill_pass = True
    router = blocks[0].moe_module.router
    want = sorted(set(torch.topk(router(hidden), 2, dim=-1).indices.flatten().tolist()))
    assert len(want) < NUM_EXPERTS, "a visit routing every expert cannot tell the paths apart"
    routed = _run(blocks[0], hidden)
    assert ref._expert_history[0] == want, "the routed branch did not run"
    whole = _whole(ref, blocks[0], hidden)
    assert cache.resident_set(0) == set()
    assert torch.equal(routed, whole)


def test_a_failed_expert_load_releases_the_layer_and_raises(tmp_path, monkeypatch):
    import vaayu.persist.safetensor_model_persister as smp
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)

    def boom(*a, **k):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(smp, "load_expert_sparse", boom)
    with pytest.raises(RuntimeError, match="disk gone"):
        _run(blocks[0], torch.randn(1, WIDTH))
    experts = blocks[0].moe_module.experts
    assert experts.gate_up_proj.is_meta
    assert "forward" in experts.__dict__


def test_a_routed_expert_with_no_offset_raises_instead_of_being_skipped(tmp_path, monkeypatch):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    monkeypatch.setattr(idx, "get_expert_offsets", lambda layer, e: None)
    with pytest.raises(RuntimeError, match="were not uploaded"):
        _run(blocks[0], torch.randn(1, WIDTH))
    assert blocks[0].moe_module.experts.gate_up_proj.is_meta


def test_an_unrecognised_call_raises_instead_of_falling_back(tmp_path):
    model, idx, _ = build(tmp_path, pass_hist=False)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref)
    with pytest.raises(RuntimeError, match="no routed set"):
        _run(blocks[0], torch.randn(1, WIDTH))


def test_the_toggle_restores_whole_layer_delivery(tmp_path):
    model, idx, _ = build(tmp_path)
    ref = Ref(model, routed_ok(idx))
    blocks, cache = patch(tmp_path, model, idx, ref)
    hidden = torch.randn(2, WIDTH)
    routed = _run(blocks[0], hidden)
    ref._routed_fused3d = False
    whole = _run(blocks[0], hidden)
    assert ref._expert_history[0] == list(range(NUM_EXPERTS))
    assert torch.equal(routed, whole)
    assert blocks[0].moe_module.experts.gate_up_proj.is_meta


def test_a_second_geometry_buffer_warns_that_the_planner_does_not_charge_it(tmp_path, capsys):
    """The VRAM planner's layer_slack budgets ONE fused layer, so one routed
    geometry's buffer fits the plan and a second one does not (CLAUDE.md §5: warn,
    don't silently degrade). Two geometries here: layer 1 stored in another dtype."""
    from safetensors.torch import save_file
    model, idx, weights = build(tmp_path, n_layers=2)
    w, b = weights[1]
    save_file({"layers.1.mlp.experts.gate_up_proj": w.half().contiguous(),
               "layers.1.mlp.experts.gate_up_proj_bias": b.half().contiguous()},
              str(tmp_path / "layers.1.safetensors"))
    idx.build()
    ref = Ref(model, routed_ok(idx))
    capsys.readouterr()
    patch(tmp_path, model, idx, ref)
    out = capsys.readouterr().out
    assert len(ref._fused3d_buffers) == 2
    warnings = [line for line in out.splitlines() if "WARNING" in line and "geometry" in line]
    assert len(warnings) == 1, out
    for bufs in ref._fused3d_buffers.values():
        assert f"({sum(t.nbytes for t in bufs.values())} B)" in warnings[0]
    assert "not charged" in warnings[0]


def test_one_geometry_allocates_one_buffer_and_warns_nothing(tmp_path, capsys):
    model, idx, _ = build(tmp_path, n_layers=2)
    ref = Ref(model, routed_ok(idx))
    capsys.readouterr()
    patch(tmp_path, model, idx, ref)
    assert len(ref._fused3d_buffers) == 1
    assert "WARNING" not in capsys.readouterr().out
