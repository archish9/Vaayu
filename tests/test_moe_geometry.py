"""Model-agnostic MoE geometry — derived from the offset index, never hardcoded."""
import pytest
from vaayu.vaayu_base import compute_moe_geometry


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
