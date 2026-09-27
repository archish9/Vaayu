"""S8 Task 3: the load-time proof that a fused-3D kernel reads only routed rows.

It FAILS CLOSED: an unrecognised call, a vacuous check (every expert routed), a
difference, or an exception all keep whole-layer delivery. And it must leave no
residue on the model it ran against, pass or fail.
"""
import torch

import vaayu.routed_fused3d as rf
from tests.routed_fused3d_fakes import (NUM_EXPERTS, TOP_K, WIDTH, Ref, RoutedExperts,
                                        build, patch)


def _check(tmp_path, model, idx, block_idx=0, top_k=TOP_K):
    from vaayu.moe_patch import detect_moe_blocks
    block = detect_moe_blocks(model)[block_idx]
    return rf.self_check_routed_delivery(
        block, offset_index=idx, checkpoint_path=str(tmp_path), vaayu_ref=Ref(model),
        device="cpu", dtype=torch.float32, hidden_size=WIDTH, top_k=top_k), block


def test_an_honest_kernel_passes(tmp_path):
    model, idx, _ = build(tmp_path)
    (ok, why), _ = _check(tmp_path, model, idx)
    assert ok, why
    assert "M=1" in why


def test_a_kernel_that_reads_every_row_fails(tmp_path):
    model, idx, _ = build(tmp_path, read_all_rows=True)
    (ok, why), _ = _check(tmp_path, model, idx)
    assert not ok
    assert "change the output" in why


def test_a_check_that_routes_every_expert_is_vacuous_and_fails(tmp_path):
    model, idx, _ = build(tmp_path, top_k=NUM_EXPERTS)
    (ok, why), _ = _check(tmp_path, model, idx, top_k=NUM_EXPERTS)
    assert not ok
    assert "vacuous" in why


def test_a_call_with_no_histogram_fails(tmp_path):
    model, idx, _ = build(tmp_path, pass_hist=False)
    (ok, why), _ = _check(tmp_path, model, idx)
    assert not ok
    assert "no routed set" in why


def _assert_no_residue(block, router_params_before):
    experts = block.moe_module.experts
    assert "forward" not in experts.__dict__, "the capture wrapper was left installed"
    assert experts.gate_up_proj.is_meta and experts.gate_up_proj_bias.is_meta
    router = block.moe_module.router
    for name, p in router_params_before.items():
        assert router._parameters[name] is p, f"router.{name} not restored by identity"


def test_a_passing_check_leaves_no_residue(tmp_path):
    model, idx, _ = build(tmp_path)
    before = dict(model.layers[0].mlp.router._parameters)
    (ok, _), block = _check(tmp_path, model, idx)
    assert ok
    _assert_no_residue(block, before)


def test_a_raising_check_leaves_no_residue(tmp_path, monkeypatch):
    model, idx, _ = build(tmp_path)
    before = dict(model.layers[0].mlp.router._parameters)

    def boom(self, *a, **k):
        raise ValueError("kernel exploded")
    monkeypatch.setattr(RoutedExperts, "forward", boom)
    (ok, why), block = _check(tmp_path, model, idx)
    assert not ok
    assert "raised ValueError" in why
    _assert_no_residue(block, before)


def test_one_check_per_geometry_and_the_verdict_enables_routing(tmp_path, monkeypatch):
    from vaayu.moe_patch import detect_moe_blocks
    model, idx, weights = build(tmp_path, n_layers=2)
    calls = []
    real = rf.self_check_routed_delivery

    def counting(*a, **k):
        calls.append(a[0].layer_name)
        return real(*a, **k)
    monkeypatch.setattr(rf, "self_check_routed_delivery", counting)
    ref = Ref(model)
    blocks = detect_moe_blocks(model)
    verdicts = rf.run_routed_self_checks(
        ref, blocks, idx, checkpoint_path=str(tmp_path), device="cpu",
        dtype=torch.float32, hidden_size=WIDTH, top_k=TOP_K)
    assert calls == ["layers.0"], "both layers share one geometry -> one check"
    assert list(verdicts.values()) == [True]

    ref._routed_fused3d_ok = verdicts
    patched, _ = patch(tmp_path, model, idx, ref, cache=None)
    assert [b.delivery for b in patched] == ["routed", "routed"]
    hidden = torch.randn(2, WIDTH)
    out, _ = patched[1].moe_module.forward(hidden)
    w, b = weights[1]
    router = patched[1].moe_module.router
    scal, eidx = torch.topk(router(hidden).softmax(-1), TOP_K, dim=-1)
    expected = torch.stack([sum(scal[t, j] * (hidden[t] @ w[eidx[t, j]] + b[eidx[t, j]])
                                for j in range(TOP_K)) for t in range(2)])
    torch.testing.assert_close(out, expected)


def test_a_failing_geometry_prints_a_warning(tmp_path, capsys):
    from vaayu.moe_patch import detect_moe_blocks
    model, idx, _ = build(tmp_path, read_all_rows=True)
    verdicts = rf.run_routed_self_checks(
        Ref(model), detect_moe_blocks(model), idx, checkpoint_path=str(tmp_path),
        device="cpu", dtype=torch.float32, hidden_size=WIDTH, top_k=TOP_K)
    assert list(verdicts.values()) == [False]
    assert "WARNING: routed fused-3D delivery OFF" in capsys.readouterr().out
