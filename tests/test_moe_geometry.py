"""Model-agnostic MoE geometry — derived from the offset index, never hardcoded."""
import pytest
from vaayu.vaayu_base import compute_moe_geometry, largest_injection_group_bytes


class _FakeMeta:
    def __init__(self, byte_length): self.byte_length = byte_length


class _FakeIndex:
    """Mimics ExpertOffsetIndex.index: {layer: {expert: {key: meta}}}."""
    def __init__(self, num_layers, num_experts, expert_bytes):
        per_key = expert_bytes // 3
        self.index = {
            f"model.layers.{L}": {
                e: {
                    "gate.trellis": _FakeMeta(per_key),
                    "up.trellis": _FakeMeta(per_key),
                    "down.trellis": _FakeMeta(expert_bytes - 2 * per_key),
                }
                for e in range(num_experts)
            }
            for L in range(num_layers)
        }


def test_geometry_qwen_like():
    idx = _FakeIndex(num_layers=91, num_experts=128, expert_bytes=11_827_000)
    g = compute_moe_geometry(idx, num_layers=94, moe_top_k=8, moe_num_experts=128)
    assert g["num_moe_layers"] == 91
    assert abs(g["expert_bytes"] - 11_827_000) < 10
    # working set ~ 8 * 91 * 11.28MB ~ 8.02 GB
    assert 7.5e9 < g["working_set_bytes"] < 8.62e9


def test_geometry_minimax_like():
    idx = _FakeIndex(num_layers=62, num_experts=256, expert_bytes=11_850_000)
    g = compute_moe_geometry(idx, num_layers=62, moe_top_k=8, moe_num_experts=256)
    assert g["num_moe_layers"] == 62
    # working set ~ 8 * 62 * 11.3MB ~ 5.5 GB  → fits 7.5GB budget
    assert 5.0e9 < g["working_set_bytes"] < 6.2e9


class _NonUniformIndex:
    """Experts with DIFFERENT byte sizes across a layer (uniformity violated)."""
    def __init__(self, num_layers, num_experts, base_expert_bytes=1000):
        # Each expert e in each layer has size (e+1) * base_expert_bytes
        # ensuring non-uniformity and making max = num_experts * base_expert_bytes
        self.index = {
            f"model.layers.{L}": {
                e: {
                    "gate.trellis": _FakeMeta((e + 1) * base_expert_bytes // 3),
                    "up.trellis": _FakeMeta((e + 1) * base_expert_bytes // 3),
                    "down.trellis": _FakeMeta(
                        (e + 1) * base_expert_bytes - 2 * ((e + 1) * base_expert_bytes // 3)
                    ),
                }
                for e in range(num_experts)
            }
            for L in range(num_layers)
        }


@pytest.mark.parametrize(
    "num_layers,num_experts,base_bytes",
    [
        (91, 128, 1000),     # Qwen-like
        (62, 256, 1000),     # MiniMax-like
    ],
)
def test_geometry_nonuniform_takes_max_and_warns(
    capsys, num_layers, num_experts, base_bytes
):
    idx = _NonUniformIndex(
        num_layers=num_layers, num_experts=num_experts, base_expert_bytes=base_bytes
    )
    g = compute_moe_geometry(
        idx, num_layers=num_layers, moe_top_k=8, moe_num_experts=num_experts
    )
    max_expert_bytes = num_experts * base_bytes
    assert g["expert_bytes"] == max_expert_bytes  # max, not first
    assert "non-uniform expert sizes" in capsys.readouterr().out


class _GroupIndex:
    """Mimics ExpertOffsetIndex.index with the FULL checkpoint keys it stores
    (`...experts.<n>.<submodule>.<tensor>`), taking an explicit layer map."""
    def __init__(self, index):
        self.index = index


def _expert(prefix, group_bytes, extra_per_group=0):
    """One expert: per projection, a `.trellis` plus a small `.suh` (same group)."""
    out = {}
    for name, nbytes in zip(("w1", "w3", "w2"), group_bytes):
        out[f"{prefix}.{name}.trellis"] = _FakeMeta(nbytes - extra_per_group)
        out[f"{prefix}.{name}.suh"] = _FakeMeta(extra_per_group)
    return out


def _index_with(num_layers, num_experts, group_bytes):
    return _GroupIndex({
        f"model.layers.{L}": {
            e: _expert(f"model.layers.{L}.block_sparse_moe.experts.{e}",
                       group_bytes, extra_per_group=1_000)
            for e in range(num_experts)
        }
        for L in range(num_layers)
    })


def _index_with_one_outlier(normal, outlier, num_layers=48, outlier_layer=40):
    index = {}
    for L in range(num_layers):
        size = outlier if L == outlier_layer else normal
        index[f"model.layers.{L}"] = {
            e: _expert(f"model.layers.{L}.mlp.experts.{e}", [normal, normal, size if e == 3 else normal])
            for e in range(4)
        }
    return _GroupIndex(index)


def test_injection_group_none_and_empty_index_are_zero():
    assert largest_injection_group_bytes(None) == 0
    assert largest_injection_group_bytes(_GroupIndex({})) == 0
    assert largest_injection_group_bytes(_GroupIndex({"model.layers.0": {0: {}}})) == 0


def test_injection_group_unprefixed_keys_are_not_expert_tensors():
    """Fused-3D keys carry no `.experts.<n>.` and are a supported zero."""
    idx = _GroupIndex({"model.layers.0": {0: {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": _FakeMeta(5_000)}}})
    assert largest_injection_group_bytes(idx) == 0


@pytest.mark.parametrize("num_layers,num_experts,group_bytes", [
    (62, 256, [3_000_000, 3_000_000, 2_400_000]),   # MiniMax-like, 3 projections
    (48, 512,  [600_000,   600_000,   490_000]),    # Qwen3-Next-like
])
def test_injection_group_largest_submodule_not_the_expert_total(
        num_layers, num_experts, group_bytes):
    idx = _index_with(num_layers, num_experts, group_bytes)
    assert largest_injection_group_bytes(idx) == max(group_bytes)
    assert largest_injection_group_bytes(idx) < sum(group_bytes)


def test_injection_group_single_outsized_expert_sets_the_size():
    """One fat expert in layer 40 must not be averaged away, nor missed by
    sampling only the first layer the way compute_moe_geometry does."""
    idx = _index_with_one_outlier(normal=3_000_000, outlier=9_000_000)
    assert largest_injection_group_bytes(idx) == 9_000_000
