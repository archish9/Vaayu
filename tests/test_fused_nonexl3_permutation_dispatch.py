"""Row 44 F5: the non-EXL3 fused forward dispatches by permutation, not by mask.

It was the last forward still building an int64 one-hot mask per layer call
(`num_experts x top_k x tokens` — 21 MB for one 512-token Qwen3-Next chunk) and
running `torch.where` plus two tiny H2D copies per routed expert. The other two
forwards replaced that years ago with `compute_expert_dispatch`: one CPU argsort
per layer, ONE H2D of the index pairs, and free GPU slices per expert.

Order inside an expert changes (token-major instead of slot-major) but cannot
change the result: an expert never sees the same token twice in one layer call
(top-k ids are distinct), so its `index_add_` has no duplicate indices.
"""
import torch
import torch.nn.functional as F

import vaayu.moe_patch as mp
from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import MockFusedModel, MockVaayuRef
from tests.test_packed_expert_dequant import _write_packed_fused_checkpoint

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 32, 64, 8, 2


def _model(tmp_path):
    dense = _write_packed_fused_checkpoint(
        tmp_path, hidden=HIDDEN, intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build(write_cache=False)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    ref = MockVaayuRef()
    patch_moe_blocks(model=model, expert_cache=ExpertRAMCache(budget_bytes=8 << 20),
                     offset_index=idx, checkpoint_path=str(tmp_path), top_k=TOP_K,
                     device="cpu", dtype=torch.float32, vaayu_ref=ref,
                     loader_strategy=None)
    torch.manual_seed(11)
    return model, ref, dense, torch.randn(1, 4, HIDDEN)


def test_no_one_hot_mask_is_built(tmp_path, monkeypatch):
    model, _ref, _dense, x = _model(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the forward still builds a one-hot dispatch mask")
    monkeypatch.setattr(F, "one_hot", boom)

    assert torch.isfinite(model.layers[0].mlp.forward(x)).all()


def test_every_routed_pair_is_dispatched_exactly_once(tmp_path, monkeypatch):
    model, _ref, _dense, x = _model(tmp_path)
    seen = {}
    real = mp.compute_expert_dispatch

    def spy(indices_cpu, num_experts):
        pair, starts, counts = real(indices_cpu, num_experts)
        seen.update(n=indices_cpu.shape[0], top_k=indices_cpu.shape[1],
                    pair=pair, counts=counts)
        return pair, starts, counts
    monkeypatch.setattr(mp, "compute_expert_dispatch", spy)

    model.layers[0].mlp.forward(x)

    assert seen, "the forward does not use compute_expert_dispatch"
    assert sum(seen["counts"]) == seen["n"] * seen["top_k"]
    assert seen["pair"].shape == (2, seen["n"] * seen["top_k"])


def test_output_matches_an_independent_reference(tmp_path):
    """The same routing run against the checkpoint's own dequantized weights."""
    model, _ref, dense, x = _model(tmp_path)
    got = model.layers[0].mlp.forward(x)

    flat = x.view(-1, HIDDEN)
    _, weights, indices = model.layers[0].mlp.gate(flat)
    expected = torch.zeros_like(flat, dtype=torch.float32)
    for tok in range(flat.shape[0]):
        for pos in range(TOP_K):
            e = int(indices[tok, pos])
            h = flat[tok:tok + 1]
            gate = F.silu(F.linear(h, dense[(e, "gate_proj")]))
            up = F.linear(h, dense[(e, "up_proj")])
            expected[tok] += (F.linear(gate * up, dense[(e, "down_proj")])
                              * weights[tok, pos]).squeeze(0)

    assert torch.allclose(got.view(-1, HIDDEN), expected, atol=1e-4), (
        f"max|d| = {(got.view(-1, HIDDEN) - expected).abs().max().item():.3e}")
