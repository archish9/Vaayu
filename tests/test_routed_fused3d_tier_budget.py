"""S18: the routed fused-3D kernel buffers are charged to the VRAM expert tier.

The tier is sized as free VRAM minus the reserve BEFORE patch_moe_blocks allocates the
routed buffers, so its budget includes their bytes. Left uncharged, a full tier plus the
reserve plus the resident weights exceed the card by one fused layer.
"""
import types

import pytest
import torch

from tests.routed_fused3d_fakes import Ref, build, patch, routed_ok
from vaayu.persist.expert_cache import ExpertVRAMCache
from vaayu.routed_fused3d import charge_routed_buffers_to_vram_tier
from vaayu.vaayu_base import VaayuBaseModel

# Fused-3D geometries (these allocate a routed buffer): layers, experts, width, top-k.
FUSED3D = {
    "small": dict(n_layers=2, num_experts=8, width=4, top_k=2),
    "wide": dict(n_layers=3, num_experts=12, width=6, top_k=3),
}
# Per-expert-forward geometries (no routed buffer): expert bytes, tier budget in experts.
PER_EXPERT = {"qwen3next_like": (1_769_520, 4251), "minimax_like": (8 * 1024 ** 2, 700)}


def _expert_bytes(g):
    return (g["width"] * g["width"] + g["width"]) * 4


def _routed_ref(tmp_path, name):
    g = FUSED3D[name]
    model, idx, _ = build(tmp_path, g["n_layers"], top_k=g["top_k"],
                          num_experts=g["num_experts"], width=g["width"])
    ref = Ref(model, routed_ok(idx))
    patch(tmp_path, model, idx, ref)
    buffer_bytes = sum(t.nbytes for bufs in ref._fused3d_buffers.values() for t in bufs.values())
    assert buffer_bytes == g["num_experts"] * _expert_bytes(g), "one geometry, one buffer"
    return g, ref, buffer_bytes


@pytest.mark.parametrize("name", list(FUSED3D))
def test_the_buffers_bytes_come_out_of_the_tier_budget(tmp_path, name, capsys):
    g, ref, buffer_bytes = _routed_ref(tmp_path, name)
    budget = buffer_bytes + 5 * _expert_bytes(g)
    ref._expert_vram_cache = tier = ExpertVRAMCache(budget)
    capsys.readouterr()
    assert charge_routed_buffers_to_vram_tier(ref, _expert_bytes(g)) == buffer_bytes
    assert ref._expert_vram_cache is tier
    assert tier.budget_bytes == 5 * _expert_bytes(g)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert f"{buffer_bytes} B" in out, out


@pytest.mark.parametrize("name", list(FUSED3D))
def test_a_charged_budget_under_one_expert_drops_the_tier_and_warns(tmp_path, name, capsys):
    g, ref, buffer_bytes = _routed_ref(tmp_path, name)
    ref._expert_vram_cache = ExpertVRAMCache(buffer_bytes + _expert_bytes(g) - 1)
    capsys.readouterr()
    assert charge_routed_buffers_to_vram_tier(ref, _expert_bytes(g)) == buffer_bytes
    assert ref._expert_vram_cache is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "Expert VRAM cache disabled" in out, out


@pytest.mark.parametrize("name", list(FUSED3D))
def test_a_budget_smaller_than_the_buffers_drops_the_tier_too(tmp_path, name):
    g, ref, buffer_bytes = _routed_ref(tmp_path, name)
    ref._expert_vram_cache = ExpertVRAMCache(buffer_bytes // 2)
    assert charge_routed_buffers_to_vram_tier(ref, _expert_bytes(g)) == buffer_bytes
    assert ref._expert_vram_cache is None


@pytest.mark.parametrize("name", list(FUSED3D))
def test_no_tier_means_nothing_to_charge(tmp_path, name, capsys):
    g, ref, _ = _routed_ref(tmp_path, name)
    capsys.readouterr()
    assert charge_routed_buffers_to_vram_tier(ref, _expert_bytes(g)) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", list(PER_EXPERT))
@pytest.mark.parametrize("buffers", ["absent", "empty"])
def test_a_model_with_no_routed_buffer_keeps_its_budget(name, buffers, capsys):
    expert_bytes, n = PER_EXPERT[name]
    tier = ExpertVRAMCache(n * expert_bytes)
    ref = types.SimpleNamespace(_expert_vram_cache=tier)
    if buffers == "empty":
        ref._fused3d_buffers = {}
    assert charge_routed_buffers_to_vram_tier(ref, expert_bytes) == 0
    assert ref._expert_vram_cache is tier and tier.budget_bytes == n * expert_bytes
    assert capsys.readouterr().out == ""


def test_entries_over_the_charged_budget_are_evicted():
    """A tier that already holds entries (not the load-time case, but set_budget's
    contract) must come out within the charged budget."""
    eb = 64
    tier = ExpertVRAMCache(10 * eb)
    for e in range(10):
        assert tier.put(0, e, {"fused3d": {"w": torch.zeros(eb // 4)}})
    ref = types.SimpleNamespace(
        _expert_vram_cache=tier,
        _fused3d_buffers={"geom": {"w": torch.zeros(6 * eb // 4)}})
    assert charge_routed_buffers_to_vram_tier(ref, eb) == 6 * eb
    assert tier.budget_bytes == 4 * eb and tier.current_usage <= 4 * eb
    assert len(tier.cache) == 4


def test_the_load_charges_the_routed_buffers_before_it_derives_the_layer_range():
    """Wiring (CLAUDE.md §8: a helper nothing calls is decoration): __init__ charges
    the routed buffers after the blocks are patched and before the tier's layer range
    is computed from its budget."""
    import inspect
    src = inspect.getsource(VaayuBaseModel.__init__)
    patched = src.index("MoE blocks patched for sparse streaming")
    charge = src.index("charge_routed_buffers_to_vram_tier(self")
    layer_range = src.index("compute_vram_layer_range(")
    assert patched < charge < layer_range
