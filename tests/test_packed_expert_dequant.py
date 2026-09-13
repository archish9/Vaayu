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


# ── the fused forward's phase profile (row 30, phase 3) ──────────────────────
#
# The non-EXL3 fused forward is the only one of the three that recorded NOTHING
# into `_moe_profile`. Phase 3's whole deliverable is the gate
# `(h2d + gemm + tail) / T`, which reads that dict — unmeasured, the gate reads
# zero and would close the hybrid idea on an artefact.

# Must carry every key `VaayuBase.forward` puts in `_moe_profile` — a missing
# one is a KeyError inside the forward, not a soft miss.
_BLANK_PROFILE = {"mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0,
                  "gemm": 0.0, "route": 0.0, "attach": 0.0, "detach": 0.0,
                  "scatter": 0.0, "construct_hit": 0.0, "gemm_hit": 0.0,
                  "slots_hit": 0.0, "slots_miss": 0.0}


def _patch_packed_fused(tmp_path, *, hidden=32, intermediate=64, num_experts=8, top_k=2):
    from vaayu.moe_patch import patch_moe_blocks
    from vaayu.persist.expert_cache import ExpertRAMCache
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex
    from tests.test_moe_patch import MockFusedModel, MockVaayuRef

    _write_packed_fused_checkpoint(
        tmp_path, hidden=hidden, intermediate=intermediate, num_experts=num_experts)
    model = MockFusedModel(num_layers=1, hidden=hidden)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    vaayu_ref = MockVaayuRef()
    patch_moe_blocks(
        model=model,
        expert_cache=ExpertRAMCache(budget_bytes=8 * 1024 * 1024),
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
        loader_strategy=None,
    )
    return model, vaayu_ref, hidden


def test_fused_forward_records_its_phase_profile(tmp_path):
    """Every phase the decode split is built from must be populated."""
    model, vaayu_ref, hidden = _patch_packed_fused(tmp_path)
    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = dict(_BLANK_PROFILE)

    model.layers[0].mlp.forward(torch.randn(1, 4, hidden))

    prof = vaayu_ref._moe_profile
    for key in ("route", "disk_load", "mask_sync", "h2d", "construct", "gemm", "scatter"):
        assert prof[key] > 0.0, f"fused forward recorded zero {key}"
    # This forward has no VRAM cache and no per-expert module to detach.
    assert prof["attach"] == 0.0 and prof["detach"] == 0.0


def test_fused_forward_profile_is_fully_gated(tmp_path):
    """Off means untouched — the instrumentation may not cost a decode anything."""
    model, vaayu_ref, hidden = _patch_packed_fused(tmp_path)
    vaayu_ref.profiling_mode = False
    vaayu_ref._moe_profile = dict(_BLANK_PROFILE)

    model.layers[0].mlp.forward(torch.randn(1, 4, hidden))

    assert vaayu_ref._moe_profile == _BLANK_PROFILE


def test_fused_forward_dense_checkpoint_still_profiles_h2d(tmp_path):
    """A non-packed (dense) fused checkpoint has no dequant, so `construct`
    stays 0 — but the H2D cast must still be timed, or the gate under-reads on
    every model that is not pack-quantized."""
    from vaayu.moe_patch import _expert_dense_weights

    raw = {f"{PACKED_PREFIX}{seg}.weight": torch.randn(4, 4)
           for seg in PACKED_ROLE_SEG.values()}
    timing = {"h2d": 0.0, "construct": 0.0}
    w_gate, w_down, w_up, entry = _expert_dense_weights(
        raw, "cpu", torch.float32, timing=timing)

    assert w_gate is not None and w_down is not None and w_up is not None
    assert timing["h2d"] > 0.0
    assert timing["construct"] == 0.0
    # Cacheable as-is: a dense expert's entry IS its weights, so a VRAM hit on
    # one skips the upload and owes no dequant (row 30, S2b (iii)).
    assert entry == {"dense": {"gate": w_gate, "down": w_down, "up": w_up}}


# ── S2b (ii): one fused dequant per expert instead of three ──────────────────
#
# Phase 3 measured `construct` -- this dequant -- at 25.1 % of a decode token,
# which is the condition RC1 banked the faster variants behind. Re-measured on
# one real expert, 300 reps: three per-linear calls 114.3 us, one fused call
# 75.7 us, the fused call under torch.compile 17.6 us (a trivial kernel is 2.7),
# so it is kernel-launch overhead, not arithmetic. At 480 routed experts per
# token that is ~46 ms of a 244 ms token.
#
# The whole win is only collectable if the output stays BIT-IDENTICAL: these
# weights feed F.linear directly, and the suite above holds the unpack to
# compressed_tensors' own reference bit-for-bit. `allclose` would not do.

def _components(rows, cols, num_bits, group_size, seed):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols, dtype=torch.bfloat16) * 0.1
    packed, scale, shape, _ = _reference_pack(w, num_bits, group_size)
    return {"weight_packed": packed, "weight_scale": scale, "weight_shape": shape}


def _expert_components(num_bits=4, group_size=32, hidden=256, intermediate=128):
    """gate/up [intermediate, hidden] and down [hidden, intermediate], the real shape."""
    return {
        "gate": _components(intermediate, hidden, num_bits, group_size, 1),
        "up": _components(intermediate, hidden, num_bits, group_size, 2),
        "down": _components(hidden, intermediate, num_bits, group_size, 3),
    }


@pytest.mark.parametrize("num_bits,group_size", [(4, 32), (4, 128), (8, 128)])
def test_fused_expert_dequant_is_bit_identical_to_three_calls(num_bits, group_size):
    from vaayu.moe_patch import dequantize_expert_weights

    by_role = _expert_components(num_bits, group_size)
    expected = {r: dequantize_packed_weight(c, torch.bfloat16) for r, c in by_role.items()}

    got = dequantize_expert_weights(by_role, torch.bfloat16)

    assert set(got) == set(expected)
    for role in expected:
        assert got[role].shape == expected[role].shape
        assert got[role].dtype == expected[role].dtype
        assert torch.equal(got[role], expected[role]), f"{role} is not bit-identical"


def test_fused_dequant_falls_back_when_the_roles_disagree():
    """The fused pass flattens all three roles into one group-size-32 view. A
    checkpoint whose roles were packed differently must still decode, one role
    at a time, rather than decode wrongly."""
    from vaayu.moe_patch import dequantize_expert_weights

    by_role = {
        "gate": _components(128, 256, 4, 32, 1),
        "up": _components(128, 256, 4, 32, 2),
        "down": _components(256, 128, 4, 64, 3),      # different group size
    }
    expected = {r: dequantize_packed_weight(c, torch.bfloat16) for r, c in by_role.items()}

    got = dequantize_expert_weights(by_role, torch.bfloat16)

    for role in expected:
        assert torch.equal(got[role], expected[role]), f"{role} decoded wrongly"


def test_fused_dequant_refuses_asymmetric_by_name():
    """The NotImplementedError RC1 added must not be lost to the fused path."""
    from vaayu.moe_patch import dequantize_expert_weights

    by_role = _expert_components()
    by_role["gate"]["weight_zero_point"] = torch.zeros(1)
    with pytest.raises(NotImplementedError, match="zero point"):
        dequantize_expert_weights(by_role, torch.bfloat16)


def test_fused_dequant_actually_fuses():
    """Guard against the optimization silently becoming a no-op.

    A feature that quietly stops firing is this repo's recurring failure mode
    (CLAUDE.md §0). One fused pass must not call the per-role dequantiser at all
    on the geometry the target checkpoint actually has.
    """
    import vaayu.moe_patch as mp

    calls = []
    real = mp.dequantize_packed_weight
    mp.dequantize_packed_weight = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        mp.dequantize_expert_weights(_expert_components(), torch.bfloat16)
    finally:
        mp.dequantize_packed_weight = real
    assert calls == [], f"fused path fell back: {len(calls)} per-role calls"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled path is CUDA-only")
def test_compiled_fused_dequant_matches_eager_bit_for_bit():
    """torch.compile is where the 6.5x is. It may not change a single bit."""
    from vaayu.moe_patch import dequantize_expert_weights

    by_role = {r: {k: (v if k == "weight_shape" else v.cuda()) for k, v in c.items()}
               for r, c in _expert_components().items()}
    expected = {r: dequantize_packed_weight(c, torch.bfloat16) for r, c in by_role.items()}

    got = dequantize_expert_weights(by_role, torch.bfloat16)

    for role in expected:
        assert torch.equal(got[role], expected[role]), f"{role} differs under compile"


def test_dequant_falls_back_to_eager_when_compile_is_unavailable(monkeypatch):
    """The compile fallback is a branch that never runs in a healthy env, which
    is exactly how a NameError ships in one. Force it and check the output is
    still right -- and that the warning it logs can actually be emitted."""
    import vaayu.moe_patch as mp

    mp._FUSED_DEQUANT_CACHE.clear()
    monkeypatch.setattr(mp.torch, "compile",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no dynamo")))
    monkeypatch.setattr(mp.torch.cuda, "is_available", lambda: True)

    by_role = _expert_components()
    expected = {r: dequantize_packed_weight(c, torch.bfloat16) for r, c in by_role.items()}

    fn = mp._fused_dequant_fn(4, 32, torch.bfloat16, "cuda")
    assert fn is mp._fused_dequant_kernel

    got = mp.dequantize_expert_weights(by_role, torch.bfloat16)
    for role in expected:
        assert torch.equal(got[role], expected[role])
    mp._FUSED_DEQUANT_CACHE.clear()


# ── S2b (iii): the packed-blob VRAM cache entry ──────────────────────────────
#
# The VRAM expert cache stores what the two on-device `torch.cat`s in
# `dequantize_expert_weights` build -- `flat_packed` / `flat_scale` already
# concatenated -- so a hit skips the H2D AND the cats and pays only the compiled
# fused kernel. Measured 2026-09-13 on 64 real experts interleaved with the
# forward's own gemms: today's construct 55.5 us/expert, the blob's 31.8.
#
# The entry feeds `F.linear` exactly as before, so the roundtrip has to be
# BIT-identical to the path it replaces -- `allclose` would not do.

@pytest.mark.parametrize("num_bits,group_size", [(4, 32), (4, 128), (8, 128)])
def test_blob_roundtrip_is_bit_identical_to_the_fused_path(num_bits, group_size):
    from vaayu.moe_patch import (build_expert_blob, dequantize_expert_blob,
                                 dequantize_expert_weights)

    by_role = _expert_components(num_bits, group_size)
    expected = dequantize_expert_weights(by_role, torch.bfloat16)

    blob = build_expert_blob(by_role)
    got = dequantize_expert_blob(blob, torch.bfloat16)

    assert set(got) == set(expected)
    for role in expected:
        assert got[role].shape == expected[role].shape
        assert torch.equal(got[role], expected[role]), f"{role} is not bit-identical"


def test_blob_entry_holds_only_the_two_flat_tensors():
    """The entry is the cache's byte cost. It must carry the packed bytes and
    the scales and nothing else -- `weight_shape` never crosses PCIe, and a
    dense copy would be 3.56x the bytes and a different layout decision."""
    from vaayu.moe_patch import build_expert_blob

    by_role = _expert_components()
    blob = build_expert_blob(by_role)

    assert set(blob["flat"]) == {"packed", "scale"}
    expected_bytes = sum(c["weight_packed"].nbytes + c["weight_scale"].nbytes
                         for c in by_role.values())
    assert sum(t.nbytes for t in blob["flat"].values()) == expected_bytes


def test_blob_is_refused_when_the_roles_disagree():
    """Not fusable -> no blob, so the caller keeps the per-role path rather than
    decoding three differently-packed roles through one flat view."""
    from vaayu.moe_patch import build_expert_blob

    by_role = {
        "gate": _components(128, 256, 4, 32, 1),
        "up": _components(128, 256, 4, 32, 2),
        "down": _components(256, 128, 4, 64, 3),      # different group size
    }
    assert build_expert_blob(by_role) is None


def test_blob_refuses_asymmetric_by_name():
    """RC1's refusal may not be lost on the way into the cache."""
    from vaayu.moe_patch import build_expert_blob

    by_role = _expert_components()
    by_role["gate"]["weight_zero_point"] = torch.zeros_like(
        by_role["gate"]["weight_packed"])
    with pytest.raises(NotImplementedError, match="zero"):
        build_expert_blob(by_role)


def test_blob_is_none_for_a_dense_expert():
    """A bf16 fused checkpoint has nothing to pack -- the caller caches its
    dense weights instead, and must not be handed a half-built blob."""
    from vaayu.moe_patch import build_expert_blob

    assert build_expert_blob({"gate": {"weight": torch.zeros(4, 4)}}) is None
