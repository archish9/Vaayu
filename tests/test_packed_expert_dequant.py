"""Row 30 / S2: dequantizing compressed-tensors `pack-quantized` experts.

Vaayu streams expert tensors itself and hands them straight to `F.linear`, so a
checkpoint whose experts are stored as `weight_packed` (int32 bit-packed) +
`weight_scale` needs a dequantiser of Vaayu's own — the HF quantizer binds
nothing for these models (row 30, RC2: the routed experts are fused 3D
`nn.Parameter`s, not `nn.Linear`, so `apply_quantization_config` skips them).

The spec these tests hold Vaayu to is `compressed_tensors`' OWN reference
implementation: `pack_to_int32` / `unpack_from_int32` / `dequantize`. Vaayu
reimplements the unpack in torch so nothing on the decode path imports the
package; these tests are what makes that reimplementation honest.

Everything is derived from the tensors themselves — bit width from the packed
vs original element counts, group size from the scale's last dim — so the same
code covers int4-g32 (Qwen3-Next-80B-AWQ, the S2 target) and int4-g128
(RedHatAI/Intel builds) with no per-model branch (CLAUDE.md §5).
"""
import pytest
import torch

from vaayu.moe_patch import (
    _group_packed_tensors_by_weight,
    dequantize_packed_weight,
    unpack_int32_to_int8,
)

ct_pack = pytest.importorskip(
    "compressed_tensors.compressors.pack_quantized.helpers",
    reason="reference implementation; only needed to validate Vaayu's copy",
)

# (rows, cols, num_bits, group_size) — g32 is the on-disk S2 target
# (cyankiwi Qwen3-Next-80B-AWQ, gate/up [512, 2048], down [2048, 512]);
# g128 is the RedHatAI / Intel / AutoRound layout of the same family.
GEOMETRIES = [
    pytest.param(512, 2048, 4, 32, id="qwen3next-int4-g32-gate"),
    pytest.param(2048, 512, 4, 32, id="qwen3next-int4-g32-down"),
    pytest.param(768, 1536, 4, 128, id="int4-g128"),
    pytest.param(256, 512, 8, 128, id="int8-g128"),
]


def _reference_pack(weight: torch.Tensor, num_bits: int, group_size: int):
    """Quantize + pack `weight` exactly the way compressed-tensors would.

    Returns the three tensors a `pack-quantized` checkpoint stores.
    """
    rows, cols = weight.shape
    groups = cols // group_size
    grouped = weight.reshape(rows, groups, group_size).float()
    qmax = 2 ** (num_bits - 1) - 1
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-6) / qmax
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-qmax - 1, qmax)
    q = q.reshape(rows, cols).to(torch.int8)
    packed = ct_pack.pack_to_int32(q, num_bits)
    return packed, scale.to(torch.bfloat16), torch.tensor([rows, cols]), q


@pytest.mark.parametrize("rows,cols,num_bits,group_size", GEOMETRIES)
def test_unpack_matches_compressed_tensors_reference(rows, cols, num_bits, group_size):
    torch.manual_seed(0)
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)
    packed, scale, shape, q = _reference_pack(weight, num_bits, group_size)

    reference = ct_pack.unpack_from_int32(packed, num_bits, torch.Size([rows, cols]))
    ours = unpack_int32_to_int8(packed, num_bits, cols)

    assert ours.dtype == torch.int8
    assert torch.equal(ours, reference), "unpack diverges from compressed-tensors"
    assert torch.equal(ours, q), "unpack does not recover the quantized values"


@pytest.mark.parametrize("rows,cols,num_bits,group_size", GEOMETRIES)
def test_dequantize_recovers_the_weight(rows, cols, num_bits, group_size):
    torch.manual_seed(1)
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)
    packed, scale, shape, q = _reference_pack(weight, num_bits, group_size)

    got = dequantize_packed_weight(
        {"weight_packed": packed, "weight_scale": scale, "weight_shape": shape},
        dtype=torch.bfloat16,
    )

    assert got.shape == (rows, cols)
    assert got.dtype == torch.bfloat16
    # Same arithmetic the reference dequantizer does: q * per-group scale.
    expected = (
        q.reshape(rows, cols // group_size, group_size).to(torch.bfloat16)
        * scale.reshape(rows, cols // group_size, 1)
    ).reshape(rows, cols)
    assert torch.equal(got, expected)
    # And it is a real reconstruction of the original, not noise.
    rel = (got.float() - weight.float()).norm() / weight.float().norm()
    assert rel < 0.15, f"reconstruction rel-L2 {rel:.4f} too large for {num_bits} bits"


def test_bit_width_and_group_size_are_derived_not_passed():
    """No caller tells the dequantiser the bit width or the group size — a
    hardcoded 4/32 would pass the g32 cases above and fail here (CLAUDE.md §5)."""
    torch.manual_seed(2)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    packed, scale, shape, q = _reference_pack(weight, 8, 64)
    got = dequantize_packed_weight(
        {"weight_packed": packed, "weight_scale": scale, "weight_shape": shape},
        dtype=torch.bfloat16,
    )
    expected = (
        q.reshape(64, 4, 64).to(torch.bfloat16) * scale.reshape(64, 4, 1)
    ).reshape(64, 256)
    assert torch.equal(got, expected)


def test_asymmetric_packed_experts_are_refused_by_name():
    """Vaayu implements the symmetric case only. An asymmetric checkpoint must
    say so, not silently drop the zero point and emit shifted weights."""
    torch.manual_seed(3)
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    packed, scale, shape, _ = _reference_pack(weight, 4, 32)
    with pytest.raises(NotImplementedError, match="zero"):
        dequantize_packed_weight(
            {
                "weight_packed": packed,
                "weight_scale": scale,
                "weight_shape": shape,
                "weight_zero_point": torch.zeros_like(packed),
            },
            dtype=torch.bfloat16,
        )


# ── role grouping ────────────────────────────────────────────────────────────

# Real key shape, read off cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit's
# model.safetensors.index.json.
PACKED_PREFIX = "model.layers.3.mlp.experts.7."
PACKED_ROLE_SEG = {"gate": "gate_proj", "down": "down_proj", "up": "up_proj"}
PACKED_COMPONENTS = ["weight_packed", "weight_scale", "weight_shape"]


def _packed_raw():
    raw = {}
    for i, (role, seg) in enumerate(sorted(PACKED_ROLE_SEG.items())):
        for j, comp in enumerate(PACKED_COMPONENTS):
            raw[f"{PACKED_PREFIX}{seg}.{comp}"] = torch.tensor([float(i), float(j)])
    return raw


def test_packed_grouping_maps_every_component_to_its_role():
    groups = _group_packed_tensors_by_weight(_packed_raw())
    assert set(groups) == {"gate", "down", "up"}
    raw = _packed_raw()
    for role, seg in PACKED_ROLE_SEG.items():
        assert set(groups[role]) == set(PACKED_COMPONENTS)
        for comp in PACKED_COMPONENTS:
            assert torch.equal(groups[role][comp], raw[f"{PACKED_PREFIX}{seg}.{comp}"])


def test_packed_grouping_is_empty_for_a_dense_checkpoint():
    """A bf16/EXL3 expert dict must not be mistaken for a packed one — that is
    what tells the forward which path to take."""
    dense = {f"{PACKED_PREFIX}{seg}.weight": torch.zeros(2) for seg in PACKED_ROLE_SEG.values()}
    assert all(not v for v in _group_packed_tensors_by_weight(dense).values())


# ── the fused forward, end to end ────────────────────────────────────────────

def _write_packed_fused_checkpoint(tmp_path, *, hidden, intermediate, num_experts,
                                   num_bits=4, group_size=32, layer_name="layers.0"):
    """A one-layer fused MoE checkpoint whose experts are pack-quantized, in the
    key shape cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit uses.

    Returns the dense weights the packed tensors decode to, so a test can
    compute the expected FFN output without trusting the code under test.
    """
    from safetensors.torch import save_file

    tensors, dense = {}, {}
    torch.manual_seed(7)
    for eidx in range(num_experts):
        for seg, shape in (("gate_proj", (intermediate, hidden)),
                           ("up_proj", (intermediate, hidden)),
                           ("down_proj", (hidden, intermediate))):
            w = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
            packed, scale, wshape, q = _reference_pack(w, num_bits, group_size)
            base = f"{layer_name}.mlp.experts.{eidx}.{seg}"
            tensors[f"{base}.weight_packed"] = packed
            tensors[f"{base}.weight_scale"] = scale
            tensors[f"{base}.weight_shape"] = wshape
            dense[(eidx, seg)] = dequantize_packed_weight(
                {"weight_packed": packed, "weight_scale": scale, "weight_shape": wshape},
                dtype=torch.float32,
            )
    tensors[f"{layer_name}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
    save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))
    return dense


def test_fused_forward_runs_packed_experts(tmp_path):
    """The non-EXL3 fused forward must dequantize pack-quantized experts rather
    than handing `F.linear` an int32 pack (row 30, RC1)."""
    import torch.nn.functional as F
    from vaayu.moe_patch import patch_moe_blocks
    from vaayu.persist.expert_cache import ExpertRAMCache
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex
    from tests.test_moe_patch import MockFusedModel, MockVaayuRef

    hidden, intermediate, num_experts, top_k = 32, 64, 8, 2
    dense = _write_packed_fused_checkpoint(
        tmp_path, hidden=hidden, intermediate=intermediate, num_experts=num_experts)

    model = MockFusedModel(num_layers=1, hidden=hidden)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    assert len(idx.index["layers.0"]) == num_experts, "index did not see packed experts"

    patch_moe_blocks(
        model=model,
        expert_cache=ExpertRAMCache(budget_bytes=8 * 1024 * 1024),
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=MockVaayuRef(),
        loader_strategy=None,
    )

    torch.manual_seed(11)
    x = torch.randn(1, 4, hidden)
    got = model.layers[0].mlp.forward(x)
    assert got.shape == (1, 4, hidden)
    assert torch.isfinite(got).all(), "packed experts produced NaN/Inf"

    # Expected: the same routing, run against the dequantized dense weights.
    flat = x.view(-1, hidden)
    router = model.layers[0].mlp.gate
    _, weights, indices = router(flat)
    expected = torch.zeros_like(flat, dtype=torch.float32)
    for tok in range(flat.shape[0]):
        for pos in range(top_k):
            e = int(indices[tok, pos])
            h = flat[tok:tok + 1]
            gate = F.silu(F.linear(h, dense[(e, "gate_proj")]))
            up = F.linear(h, dense[(e, "up_proj")])
            expected[tok] += (F.linear(gate * up, dense[(e, "down_proj")])
                              * weights[tok, pos]).squeeze(0)
    assert torch.allclose(got.view(-1, hidden), expected, atol=1e-4), (
        f"max|d| = {(got.view(-1, hidden) - expected).abs().max().item():.3e}")
