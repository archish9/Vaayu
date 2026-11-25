"""FP8DynamicCache round-trips GPU<->CPU for prefix-cache storage, preserving
data and sequence length. CPU-only: uses device='cpu' as the 'GPU' stand-in
(the move logic is device-agnostic)."""
import pytest
import torch
from vaayu.vaayu_base import FP8DynamicCache


def _fill(cache, n_layers, seq, heads=2, dim=8, device="cpu"):
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        cache.update(k, v, L)


def test_to_cpu_compacts_and_preserves():
    c = FP8DynamicCache()
    _fill(c, n_layers=3, seq=5)
    before = [c._k_out_cache[i][..., :c._seq_len[i], :].clone() for i in range(3)]
    c.to_cpu()
    assert all(t.device.type == "cpu" for t in c.key_cache)
    # compacted: capacity now equals used length (no growth tail)
    assert all(c.key_cache[i].shape[-2] == c._seq_len[i] == 5 for i in range(3))
    for i in range(3):
        assert torch.equal(c._k_out_cache[i][..., :5, :], before[i])


def test_to_device_roundtrip_then_append():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=4)
    c.to_cpu()
    c.to_device("cpu")           # 'cpu' stands in for the GPU move
    assert c.get_seq_length() == 4
    # a subsequent update appends after the restored length
    _fill(c, n_layers=2, seq=1)  # NOTE: appends to BOTH layers' seq -> 5 each
    assert c.get_seq_length() == 5


def test_resident_bytes_positive_and_shrinks_after_compact():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=3)   # capacity doubled to >=... tail present
    big = c.resident_bytes()
    c.to_cpu()                    # compaction drops the capacity tail
    assert 0 < c.resident_bytes() <= big


# ── D1: the park must allocate nothing on the device ────────────────────────
#
# `to_cpu` compacted with `[..., :used, :].contiguous().to("cpu")`, and the
# `.contiguous()` materialises the compacted layer ON THE DEVICE before the
# D2H. Those staging bytes are in nobody's VRAM budget -- the planner hands
# every spare byte to the expert cache -- so the park raised a CUDA OOM
# *after* a 130 s prefill and an already-streamed reply (D1,
# logs/run-2026-11-25T07-45-35Z: 44.00 MiB wanted, 42.12 MiB free). Same class
# of defect as row 12a, where the drafter's VRAM was in nobody's budget.

# (kv heads, head_dim, layers) of the two EXL3 checkpoints on this box, per
# CLAUDE.md §5's two-geometry rule. They contrast on every axis a park sees:
# 8 x 128 over 62 layers against 2 x 256 over 40.
GEOMETRIES = {
    "minimax-like": dict(heads=8, dim=128, layers=62),
    "qwen36-35b-like": dict(heads=2, dim=256, layers=40),
}
GEO = pytest.mark.parametrize("geo", sorted(GEOMETRIES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_park_allocates_no_device_memory():
    c = FP8DynamicCache(quantize=False)
    _fill(c, n_layers=4, seq=96, heads=4, dim=128, device="cuda")
    _fill(c, n_layers=4, seq=32, heads=4, dim=128, device="cuda")
    # There must be a capacity tail to compact, or the park has nothing to
    # stage and the measurement proves nothing.
    assert c.key_cache[0].shape[-2] > c._seq_len[0] > 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    c.to_cpu()
    # The PEAK, not the post-call reading: the staged temp is freed before
    # to_cpu returns, so the post-call delta is ~0 either way (it is negative,
    # the device buffers are gone).
    assert torch.cuda.max_memory_allocated() <= base
    del c
    torch.cuda.empty_cache()   # test_decode_graphs measures reserved() deltas


@GEO
def test_park_bytes_identical_to_contiguous_reference(geo):
    """Bit-identical to `[..., :used, :].contiguous()`, which is what the park
    produced before the block-by-block copy replaced it."""
    g = GEOMETRIES[geo]
    gen = torch.Generator().manual_seed(7)
    cap = 11
    src = torch.randn(1, g["heads"], cap, g["dim"], generator=gen).bfloat16()
    for used in (cap, cap - 4, 1, 0):          # incl. used == S_cap and used == 1
        got = FP8DynamicCache._compact_to_cpu(src, used)
        assert got.shape == (1, g["heads"], used, g["dim"])
        assert got.device.type == "cpu"
        assert torch.equal(got, src[..., :used, :].contiguous())
    # no leading dims at all
    flat = src[0, 0]
    assert torch.equal(FP8DynamicCache._compact_to_cpu(flat, 3), flat[:3])
    # more than two leading dims
    deep = src.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    assert torch.equal(FP8DynamicCache._compact_to_cpu(deep, 3),
                       deep[..., :3, :].contiguous())
    # a strided source has no contiguous [used, D] prefixes: the fallback path
    strided = src.transpose(1, 2)
    assert not strided.is_contiguous()
    assert torch.equal(FP8DynamicCache._compact_to_cpu(strided, 1),
                       strided[..., :1, :].contiguous())
    # and through to_cpu over the whole geometry's layer count
    c = FP8DynamicCache(quantize=False)
    for L in range(g["layers"]):
        for n in (3, 2):      # two appends, so a growth tail exists
            k = torch.randn(1, g["heads"], n, g["dim"], dtype=torch.float16)
            c.update(k, k.clone(), L)
    assert c.key_cache[0].shape[-2] > c._seq_len[0]      # a tail to compact
    want = {name: [t[..., :c._seq_len[i], :].contiguous()
                   for i, t in enumerate(getattr(c, name))]
            for name in c._BUFFER_LISTS}
    c.to_cpu()
    for name in c._BUFFER_LISTS:
        for i, t in enumerate(getattr(c, name)):
            assert torch.equal(t, want[name][i]), (name, i)
