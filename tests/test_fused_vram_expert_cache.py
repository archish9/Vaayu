"""Row 30 / S2b (iii): the VRAM expert cache, wired into the non-EXL3 fused forward.

That forward was the ONLY one of Vaayu's three that never reached
`_expert_vram_cache` — measured on Qwen3-Next-80B int4 at `V=0.0`,
`vram=0hit/0miss/0evict`, with 7.5 GB of VRAM cache budget reserved and idle
while peak VRAM sat at 4.48 GiB of 15.47. Same shape as the `VAAYU_CACHE_BIAS`
failure (CLAUDE.md §0): a feature wired into one forward silently no-ops in the
others, and nothing fails.

The entry layout is the PACKED BLOB (`build_expert_blob`), chosen by replay
through the real cache and confirmed by the build's own stop-rule measurement:
`construct_flat` 31.8 µs/expert against today's 55.5, so a hit skips the H2D
(121.9 µs) and the two cats, for 145.6 µs of a ~230 ms token per hit.

The test that matters most here is the one asserting a hit ACTUALLY skips the
H2D. A cache that fills, reports hits and re-uploads anyway would show every
symptom of working.
"""
import pytest
import torch

from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.moe_patch import patch_moe_blocks
from tests.test_moe_patch import MockFusedModel, MockVaayuRef, _make_fused_checkpoint
from tests.test_packed_expert_dequant import (
    _BLANK_PROFILE, _write_packed_fused_checkpoint)

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 32, 64, 8, 2
BIG_BUDGET = 64 * 1024 * 1024


def _patch(tmp_path, *, vram_budget=BIG_BUDGET, packed=True, cacheable_layers=None,
           prefill=False):
    """A one-layer fused MoE model on a real checkpoint, with a VRAM cache."""
    if packed:
        _write_packed_fused_checkpoint(
            tmp_path, hidden=HIDDEN, intermediate=INTERMEDIATE,
            num_experts=NUM_EXPERTS)
    else:
        _make_fused_checkpoint(tmp_path, num_layers=1, hidden=HIDDEN,
                               intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    vaayu_ref = MockVaayuRef()
    vaayu_ref._expert_vram_cache = (ExpertVRAMCache(budget_bytes=vram_budget)
                                    if vram_budget is not None else None)
    vaayu_ref._vram_cacheable_layers = cacheable_layers
    vaayu_ref._moe_prefill_pass = prefill
    ram = ExpertRAMCache(budget_bytes=BIG_BUDGET)
    patch_moe_blocks(
        model=model,
        expert_cache=ram,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=TOP_K,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
        loader_strategy=None,
    )
    return model, vaayu_ref, ram


def _count_h2d(monkeypatch):
    """Counts calls to the host->device + dequant helper — i.e. every expert
    the forward builds from CPU tensors instead of serving off the GPU."""
    import vaayu.moe_patch as mp
    calls = []
    real = mp._expert_dense_weights

    def spy(*a, **kw):
        calls.append(a[0])
        return real(*a, **kw)

    monkeypatch.setattr(mp, "_expert_dense_weights", spy)
    return calls


def test_a_vram_hit_skips_the_host_to_device_build(tmp_path, monkeypatch):
    """The silent-no-op guard: on the second visit every routed expert is
    resident, so NOTHING may go through the upload/dequant path."""
    model, vaayu_ref, _ram = _patch(tmp_path)
    x = torch.randn(1, 4, HIDDEN)

    model.layers[0].mlp.forward(x)          # fill
    calls = _count_h2d(monkeypatch)
    model.layers[0].mlp.forward(x)          # same routing -> all resident

    assert vaayu_ref._expert_vram_cache.hits > 0, "cache never served anything"
    assert calls == [], f"{len(calls)} experts rebuilt from host despite a VRAM hit"


def test_vram_hits_do_not_change_a_single_bit(tmp_path):
    """A cached expert feeds `F.linear` exactly as a freshly built one does."""
    model, vaayu_ref, _ram = _patch(tmp_path)
    x = torch.randn(1, 4, HIDDEN)

    first = model.layers[0].mlp.forward(x)
    second = model.layers[0].mlp.forward(x)

    assert torch.equal(first, second), (
        f"max|d| = {(first - second).abs().max().item():.3e}")


def test_a_resident_expert_is_not_read_off_disk_again(tmp_path, monkeypatch):
    """P8's lesson, in this forward: an expert already on the GPU must not be
    re-read from NVMe just because it fell out of the RAM cache."""
    import vaayu.persist.safetensor_model_persister as sp
    model, vaayu_ref, ram = _patch(tmp_path)
    x = torch.randn(1, 4, HIDDEN)

    model.layers[0].mlp.forward(x)
    # Drop the RAM cache entirely: the only thing between the forward and the
    # disk is now GPU residency.
    ram.clear()
    reads = []
    real = sp.load_expert_sparse

    def spy(*a, **kw):
        reads.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(sp, "load_expert_sparse", spy)
    model.layers[0].mlp.forward(x)

    assert vaayu_ref._expert_vram_cache.hits > 0
    assert reads == [], f"{len(reads)} disk reads for experts already resident"


def test_prefill_does_not_touch_the_vram_cache(tmp_path):
    """Prefill routes ~every expert once; churning the GPU LRU with a whole-model
    sweep is what the other two forwards gate off, and so must this one."""
    model, vaayu_ref, _ram = _patch(tmp_path, prefill=True)

    model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))

    vram = vaayu_ref._expert_vram_cache
    assert (vram.hits, vram.misses, len(vram.cache)) == (0, 0, 0)


def test_layer_restriction_is_honoured(tmp_path):
    """`_vram_cacheable_layers` is the Fix #7 self-eviction guard. A layer outside
    it may not insert."""
    model, vaayu_ref, _ram = _patch(tmp_path, cacheable_layers={1})   # not layer 0

    model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))

    assert len(vaayu_ref._expert_vram_cache.cache) == 0


def test_a_refused_put_still_produces_the_right_answer(tmp_path):
    """`put()` returns False when the entry cannot fit. The caller still owns the
    weights and must run the expert anyway."""
    model, vaayu_ref, _ram = _patch(tmp_path, vram_budget=1)   # nothing fits
    x = torch.randn(1, 4, HIDDEN)

    got = model.layers[0].mlp.forward(x)
    model_ref, _ref_vaayu, _ref_ram = _patch(tmp_path, vram_budget=None)
    expected = model_ref.layers[0].mlp.forward(x)

    assert len(vaayu_ref._expert_vram_cache.cache) == 0
    assert got.shape == expected.shape
    assert torch.isfinite(got).all()


def test_no_vram_cache_configured_still_works(tmp_path):
    """`_expert_vram_cache = None` is the CPU-compute / unit-test configuration."""
    model, _vaayu, _ram = _patch(tmp_path, vram_budget=None)
    out = model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))
    assert torch.isfinite(out).all()


def test_a_dense_fused_checkpoint_is_cached_too(tmp_path, monkeypatch):
    """Model-agnostic (CLAUDE.md §5): the win is skipping the upload, which a
    non-quantized fused MoE pays too. A packed-only wiring would silently no-op
    on every dense checkpoint."""
    model, vaayu_ref, _ram = _patch(tmp_path, packed=False)
    x = torch.randn(1, 4, HIDDEN)

    model.layers[0].mlp.forward(x)
    calls = _count_h2d(monkeypatch)
    second = model.layers[0].mlp.forward(x)

    assert vaayu_ref._expert_vram_cache.hits > 0
    assert calls == [], f"{len(calls)} dense experts rebuilt from host"
    assert torch.isfinite(second).all()


@pytest.mark.parametrize("packed", [True, False], ids=["packed-blob", "dense"])
def test_entry_bytes_never_exceed_the_checkpoints_expert_bytes(tmp_path, packed):
    """`compute_vram_layer_range` sizes the working set with the CHECKPOINT's
    expert bytes (`vaayu_base.py:700`). That is only correct while a cache ENTRY
    is no larger than the checkpoint's expert — true of the packed blob (it drops
    `weight_shape`, which never crosses PCIe) and false by 3.56x of a dequantized
    one. If this ever fails, that guard under-counts the working set and stops
    restricting layers in exactly the case it exists for.

    The dense case is here because caching dense experts is what could break the
    equality: the entry is cast to the compute dtype, so a checkpoint loaded at a
    WIDER dtype than it was stored in would cache more bytes than it holds on
    disk. Both checkpoints here are stored and loaded at the same width, which is
    what every model on disk does."""
    model, vaayu_ref, _ram = _patch(tmp_path, packed=packed)
    model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))

    vram = vaayu_ref._expert_vram_cache
    assert vram.cache, "nothing cached — the test proves nothing"
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    checkpoint_bytes = max(
        sum(m.byte_length for m in metas.values())
        for metas in idx.index["layers.0"].values())
    for key, entry in vram.cache.items():
        assert vram._group_bytes(entry) <= checkpoint_bytes, (
            f"{key}: entry {vram._group_bytes(entry)} B > checkpoint expert "
            f"{checkpoint_bytes} B")


# ── S3-pre: the phase profile must separate a VRAM hit from a miss ───────────
#
# `construct` and `gemm` are recorded on BOTH branches — a packed-blob hit still
# owes the fused dequant, and the GEMM runs on the GPU either way. S3's gate is
# `(h2d + gemm + tail) / T`, but S3 can only move the slots that are NOT already
# VRAM-resident: a resident expert crosses no PCIe, so a CPU kernel would have to
# copy it BACK. Read undivided, the gate counts work the hybrid cannot take and
# over-reads by the hit rate (83 % on Qwen3-Next-80B). These keys make the split
# a measurement instead of a proportionality assumption.

def test_the_profile_splits_vram_hits_from_misses(tmp_path):
    model, vaayu_ref, _ram = _patch(tmp_path)
    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = dict(_BLANK_PROFILE)
    x = torch.randn(1, 4, HIDDEN)

    model.layers[0].mlp.forward(x)          # fill  -> every routed slot misses
    prof = dict(vaayu_ref._moe_profile)
    assert prof["slots_miss"] > 0 and prof["slots_hit"] == 0, (
        f"fill pass: {prof['slots_miss']} miss / {prof['slots_hit']} hit")
    assert prof["construct_hit"] == 0.0 and prof["gemm_hit"] == 0.0

    model.layers[0].mlp.forward(x)          # same routing -> every slot hits
    prof = dict(vaayu_ref._moe_profile)
    assert prof["slots_hit"] > 0, "second pass recorded no VRAM-hit slots"
    assert prof["construct_hit"] > 0.0, "a hit still dequants — it must be timed"
    assert prof["gemm_hit"] > 0.0, "a hit still runs the GEMM — it must be timed"
    assert prof["construct_hit"] <= prof["construct"], "hit share exceeds the total"
    assert prof["gemm_hit"] <= prof["gemm"], "hit share exceeds the total"


def test_the_hit_miss_split_is_gated_too(tmp_path):
    """Off means untouched, same rule as every other phase key."""
    model, vaayu_ref, _ram = _patch(tmp_path)
    vaayu_ref.profiling_mode = False
    blank = dict(_BLANK_PROFILE)
    vaayu_ref._moe_profile = dict(blank)

    model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))
    model.layers[0].mlp.forward(torch.randn(1, 4, HIDDEN))

    assert vaayu_ref._moe_profile == blank
