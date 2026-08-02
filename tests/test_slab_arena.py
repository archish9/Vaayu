"""SlabArena + zero-clone expert cache path (pinned slab arena task).

CLAUDE.md §5: sizing helpers are tested over BOTH target geometries
(Qwen-like: 91 layers x 128 experts, MiniMax-like: 62 x 256). All tests are
CPU-only (pin=False) so the suite runs without CUDA.
"""
import pytest
import torch

from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import (
    _ALIGN,
    _TENSOR_ALIGN,
    _align_down,
    _align_up,
    compute_slab_bytes,
    expert_span_bytes,
    load_expert_into_slab,
)


def _meta(offset, length, shape, dtype="F16"):
    return ExpertTensorMeta(file_offset=offset, byte_length=length, shape=shape, dtype=dtype)


class _FakeIndex:
    """Duck-types ExpertOffsetIndex.index for compute_slab_bytes."""
    def __init__(self, index):
        self.index = index


# ── sizing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_layers,n_experts", [(91, 4), (62, 4)])  # Qwen-like, MiniMax-like layer counts
def test_compute_slab_bytes_is_max_aligned_span(n_layers, n_experts):
    # expert 0: contiguous 1000B span; expert 1: 3000B span with a gap
    index = {}
    for li in range(n_layers):
        base = li * 100_000
        index[f"model.layers.{li}"] = {
            0: {"a": _meta(base + 100, 600, [300]), "b": _meta(base + 700, 400, [200])},
            1: {"a": _meta(base + 5000, 1000, [500]), "b": _meta(base + 7000, 1000, [500])},
        }
    slab = compute_slab_bytes(_FakeIndex(index))
    # worst span: expert 1 -> tensors "a" (1000B) + "b" (1000B) = 2000B of real
    # tensor data. The persistent slab only needs each tensor individually
    # _TENSOR_ALIGN-packed (no O_DIRECT/_ALIGN rounding, and no padding for the
    # 1000B GAP between "a" and "b" within their group -- that gap only exists
    # in the transient scratch read, see _group_layout) -> must be >= 2000 and
    # a _TENSOR_ALIGN-multiple.
    assert slab % _TENSOR_ALIGN == 0
    assert slab >= 2000
    # NOTE: alignment padding depends on the ABSOLUTE file offset (_align_down),
    # which varies per layer since the 100_000 stride isn't _ALIGN-aligned, so
    # the worst-case span isn't guaranteed to live on layer 0. Recompute the
    # ground truth independently rather than assuming layer 0 is worst.
    expected = max(
        expert_span_bytes(expert_metas)
        for layer_experts in index.values()
        for expert_metas in layer_experts.values()
    )
    assert slab == expected


def test_expert_span_bytes_alignment():
    metas = {"x": _meta(4097, 100, [50]), "y": _meta(4300, 100, [50])}
    span = expert_span_bytes(metas)
    assert span % _TENSOR_ALIGN == 0
    # expert_span_bytes must cover BOTH of load_expert_into_slab's write
    # modes (tensor_align=True's packed_span, tensor_align=False's
    # raw_aligned_length) since the same arena slab is shared between
    # decode (tensor_align=False) and prefill staging (tensor_align=True) —
    # see _group_layout's slab_span = max(...). For this small a tensor pair
    # near a 4096-byte O_DIRECT boundary, the raw (O_DIRECT-rounded) mode
    # dominates: group spans [4097, 4400), aligned down to 4096, giving a
    # 304-byte real range rounded up to one 4096-byte sector.
    packed_span = 2 * ((100 + _TENSOR_ALIGN - 1) // _TENSOR_ALIGN * _TENSOR_ALIGN)
    raw_aligned_length = _align_up((4400 - 4097) + (4097 - _align_down(4097)))
    assert span == max(packed_span, raw_aligned_length)
    # Independent ground truth (hand-derived, not the production formula):
    # prefix_pad = 4097 - 4096 = 1; raw real range = 304B; align_up(304+1)
    # rounds up to a single 4096B sector; packed = 2*align_up_to(100,16) =
    # 2*112 = 224. max(4096, 224) = 4096.
    assert span == 4096


def test_expert_span_bytes_multiple_far_apart_groups():
    """Reproduces the real checkpoint bug: an expert's tensors are NOT
    file-contiguous (small metadata block, then a large data block far
    away). expert_span_bytes must sum each group's own aligned size, not
    the raw span from first-tensor-start to last-tensor-end (which would
    hugely overcount — this is exactly what caused Task 7's real-hardware
    regression: a 128x oversized slab, OOM on pinned alloc, and 0% cache
    hit rate)."""
    small = _meta(1_000_000, 100, [50])                    # tiny tensor near 1MB mark
    big = _meta(500_000_000, 4_000_000, [1000])             # 4MB tensor ~500MB away
    metas = {"small": small, "big": big}
    span = expert_span_bytes(metas)
    # Two groups (gap >> _ALIGN): ~4096 (small, aligned) + ~4005888ish (big, aligned).
    # Must be close to the SUM of the two tensors' own sizes, nowhere near the
    # ~499MB raw span between them.
    assert span < 10 * 1024 * 1024          # nowhere near the ~499MB naive span
    assert span >= small.byte_length + big.byte_length
    assert span % _TENSOR_ALIGN == 0


def test_load_expert_into_slab_handles_far_apart_groups(tmp_path):
    """Same scenario through the real read path: two groups, packed
    back-to-back into the slab (slab layout != file layout), each
    reconstructed correctly."""
    payload_small = bytes(range(50)) * 2                    # 100B
    payload_big = bytes(reversed(range(256))) * 16           # 4096B
    f = tmp_path / "shard.bin"
    blob = bytearray(600_000)
    off_small, off_big = 1000, 500_000
    blob[off_small:off_small + 100] = payload_small
    blob[off_big:off_big + 4096] = payload_big
    f.write_bytes(bytes(blob))

    metas = {
        "small.trellis": _meta(off_small, 100, [100], dtype="U8"),
        "big.trellis": _meta(off_big, 4096, [4096], dtype="U8"),
    }
    slab_bytes = expert_span_bytes(metas)
    slab = torch.empty(slab_bytes, dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=True)
    assert bytes(views["small.trellis"].numpy().tobytes()) == payload_small
    assert bytes(views["big.trellis"].numpy().tobytes()) == payload_big
    # both views live inside the (much smaller than the file span) slab
    for v in views.values():
        assert v.data_ptr() >= slab.data_ptr()
        assert v.data_ptr() < slab.data_ptr() + slab.nbytes
        assert v.data_ptr() % _TENSOR_ALIGN == 0


# ── arena mechanics ───────────────────────────────────────────────────────

def test_arena_acquire_release_recycle():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    a = arena.acquire()
    b = arena.acquire()
    assert a is not None and b is not None and a != b
    assert arena.acquire() is None  # exhausted
    arena.release(a)
    assert arena.acquire() == a     # recycled


# ── cache adoption / eviction / duplicate ─────────────────────────────────

def _views_for(arena, idx, nbytes=64):
    t = arena.tensor(idx)
    return {"w": t[:nbytes].view(torch.uint8)}


def test_put_slab_adopts_without_clone_and_evict_releases():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    cache = ExpertRAMCache(budget_bytes=2 * _ALIGN, arena=arena)
    i0 = cache.acquire_slab(); cache.put_slab(0, 0, _views_for(arena, i0), i0)
    i1 = cache.acquire_slab(); cache.put_slab(0, 1, _views_for(arena, i1), i1)
    assert arena.acquire() is None                      # both slabs owned by cache
    # zero-clone: cached tensor shares storage with the slab
    got = cache.get(0, 0)
    assert got["w"].data_ptr() == arena.tensor(i0).data_ptr()
    # acquiring a third slab must evict the LRU entry (0,1 is MRU via the get above)
    i2 = cache.acquire_slab()
    assert i2 is not None
    assert cache.get(0, 1) is None                      # (0,1) was evicted (LRU after (0,0) was touched)
    arena.release(i2)


def test_slab_index_of_maps_views_and_rejects_foreign():
    arena = SlabArena(n_slabs=3, slab_bytes=_ALIGN, pin=False)
    for idx in range(3):
        v = arena.tensor(idx)[:64].view(torch.uint8)
        assert arena.slab_index_of(v) == idx
    # a tensor that isn't a view into the arena backing → None (legacy clone)
    assert arena.slab_index_of(torch.zeros(16, dtype=torch.uint8)) is None


def test_acquire_consumes_pending_read_event():
    """The reuse-race guard bookkeeping: record_read() registers a pending
    event for a slab; acquire() must consume (pop + synchronize) it before
    handing the slab back out, so a CPU rewrite can't race an in-flight H2D
    read. CUDA-only (needs a real event); skipped on CPU-only boxes."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for a real torch.cuda.Event")
    arena = SlabArena(n_slabs=1, slab_bytes=_ALIGN, pin=True)
    idx = arena.acquire()
    assert idx is not None
    arena.record_read(idx)
    assert idx in arena._pending_read       # event registered
    arena.release(idx)
    assert idx in arena._pending_read       # release does NOT clear it
    got = arena.acquire()
    assert got == idx
    assert idx not in arena._pending_read   # acquire popped + synced the event


def test_acquire_syncs_before_slab_reuse():
    """End-to-end: a large async H2D read from a slab, then release+acquire.
    acquire() must have synchronized the recorded read, so overwriting the
    slab CPU-side AFTER acquire cannot corrupt the already-drained GPU copy —
    i.e. the GPU tensor still reflects the pre-overwrite bytes."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    slab_bytes = 16 * 1024 * 1024  # large enough that the H2D is non-trivial
    arena = SlabArena(n_slabs=1, slab_bytes=slab_bytes, pin=True)
    idx = arena.acquire()
    slab = arena.tensor(idx)
    slab.fill_(0xAB)                                  # pattern A
    gpu = slab.to("cuda", non_blocking=True)          # async read of pattern A
    arena.record_read(idx)
    arena.release(idx)
    reused = arena.acquire()                          # must sync the read event
    assert reused == idx
    slab.fill_(0xCD)                                  # pattern B (would corrupt if unsynced)
    assert torch.all(gpu == 0xAB).item()             # GPU copy is pure pattern A


def test_record_read_is_noop_without_cuda():
    from unittest.mock import patch as mock_patch
    arena = SlabArena(n_slabs=1, slab_bytes=_ALIGN, pin=False)
    idx = arena.acquire()
    with mock_patch("torch.cuda.is_available", return_value=False):
        arena.record_read(idx)
    assert idx not in arena._pending_read            # no event recorded
    arena.release(idx)
    assert arena.acquire() == idx                    # acquire still works (no event to sync)


def test_put_slab_duplicate_key_releases_new_slab():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    cache = ExpertRAMCache(budget_bytes=4 * _ALIGN, arena=arena)
    i0 = cache.acquire_slab(); cache.put_slab(3, 7, _views_for(arena, i0), i0)
    i1 = cache.acquire_slab(); cache.put_slab(3, 7, _views_for(arena, i1), i1)  # duplicate
    # duplicate insert must give i1 back to the arena, not leak it
    assert arena.acquire() == i1


def test_legacy_clone_put_still_works_without_arena():
    cache = ExpertRAMCache(budget_bytes=10**6)
    cache.put(0, 0, {"w": torch.zeros(16, dtype=torch.uint8)})
    assert cache.get(0, 0) is not None


# ── real file IO through a slab ───────────────────────────────────────────

def test_load_expert_into_slab_reconstructs_bytes(tmp_path):
    # Craft a file where the expert's two tensors sit at known offsets.
    payload_a = bytes(range(256)) * 4          # 1024 B
    payload_b = bytes(reversed(range(256))) * 2  # 512 B
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    off_a, off_b = 8192, 8192 + 1024 + 512     # contiguous-ish (gap 512 < _ALIGN)
    blob[off_a:off_a + 1024] = payload_a
    blob[off_b:off_b + 512] = payload_b
    f.write_bytes(bytes(blob))

    metas = {
        "gate.trellis": _meta(off_a, 1024, [1024], dtype="U8"),
        "up.trellis": _meta(off_b, 512, [512], dtype="U8"),
    }
    slab = torch.empty(compute_slab_bytes(_FakeIndex({"L": {0: metas}})), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=False)
    assert bytes(views["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload_b
    # zero-clone: views live inside the slab buffer
    assert views["gate.trellis"].data_ptr() >= slab.data_ptr()
    assert views["gate.trellis"].data_ptr() < slab.data_ptr() + slab.nbytes


def test_load_expert_into_slab_views_are_tensor_aligned(tmp_path):
    """Real-hardware regression (2026-07-12): the CUDA-side one-copy GPU
    rebuild (views_from_packed_copy) reslices these exact CPU views straight
    into a GPU tensor handed to EXL3's kernel, which dereferences the weight
    buffer as uint4* (16-byte vector loads). A tensor's natural file-relative
    position is only itemsize-aligned (safe for .view(dtype), which is all
    the pre-fix code checked) — not _TENSOR_ALIGN-safe, since that position
    is derived from the file's own byte layout and is essentially arbitrary
    mod 16. This crashed with `cudaErrorMisalignedAddress` on real hardware
    on the very first prefill MoE layer.

    Deliberately adversarial: awkward, non-16-aligned file offsets AND an
    odd byte-length (13B) tensor whose length itself isn't a multiple of
    anything convenient — proving the fix doesn't rely on tensor shapes
    happening to be nicely sized (not guaranteed across MiniMax-like
    geometries per CLAUDE.md §5), only on the explicit re-packing."""
    payload_a = bytes(range(13))                    # 13 B — deliberately awkward length
    payload_b = bytes(reversed(range(256))) * 3      # 768 B
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    off_a, off_b = 8191, 8191 + 13 + 7               # neither offset is 16-aligned
    blob[off_a:off_a + 13] = payload_a
    blob[off_b:off_b + 768] = payload_b
    f.write_bytes(bytes(blob))

    metas = {
        "gate.trellis": _meta(off_a, 13, [13], dtype="U8"),
        "up.trellis": _meta(off_b, 768, [768], dtype="U8"),
    }
    slab = torch.empty(compute_slab_bytes(_FakeIndex({"L": {0: metas}})), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=True)

    assert bytes(views["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload_b
    for v in views.values():
        assert v.data_ptr() % 16 == 0, f"{v.data_ptr()} not 16-byte aligned"


def test_load_expert_into_slab_tensor_align_false_skips_repack(tmp_path):
    """tensor_align=False (decode's cache-miss path) must NOT pay the
    _EXPERT_BUF_POOL/repack cost tensor_align=True incurs — applying it
    unconditionally regressed real-hardware decode throughput (avg token
    time ~1.0-1.2s -> ~1.56s, from added pool contention + per-tensor copy_
    overhead on a path hit 350-400x/token). This does not assert alignment
    (none is guaranteed or needed in this mode) -- only that the read still
    reconstructs correctly and lands directly in `slab` with no scratch
    buffer involved."""
    payload_a = bytes(range(13))
    payload_b = bytes(reversed(range(256))) * 3
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    off_a, off_b = 8191, 8191 + 13 + 7
    blob[off_a:off_a + 13] = payload_a
    blob[off_b:off_b + 768] = payload_b
    f.write_bytes(bytes(blob))

    metas = {
        "gate.trellis": _meta(off_a, 13, [13], dtype="U8"),
        "up.trellis": _meta(off_b, 768, [768], dtype="U8"),
    }
    slab = torch.empty(compute_slab_bytes(_FakeIndex({"L": {0: metas}})), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=False)

    assert bytes(views["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload_b
    for v in views.values():
        assert v.data_ptr() >= slab.data_ptr()
        assert v.data_ptr() < slab.data_ptr() + slab.nbytes


def test_load_expert_into_slab_tensor_align_gates_pool_usage(tmp_path):
    """Regression guard for the actual claim in the test above's name: with
    the pinned scratch pool populated (simulating a CUDA machine),
    tensor_align=False must never draw from _EXPERT_BUF_POOL, while
    tensor_align=True draws exactly once per contiguous group. Without this,
    a future change could silently reintroduce the unconditional-repack
    regression and no test would catch it."""
    import vaayu.persist.safetensor_model_persister as persister
    from unittest.mock import patch as mock_patch

    payload = bytes(range(100))
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    off = 8191  # deliberately not 16- or 4096-aligned
    blob[off:off + 100] = payload
    f.write_bytes(bytes(blob))
    metas = {"w": _meta(off, 100, [100], dtype="U8")}

    scratch = torch.empty(1024 * 1024, dtype=torch.uint8)
    slab = torch.empty(expert_span_bytes(metas), dtype=torch.uint8)

    with mock_patch.object(persister, "_USE_PINNED", True), \
         mock_patch.object(persister, "_EXPERT_PINNED_SIZE", scratch.nbytes), \
         mock_patch.object(persister._EXPERT_BUF_POOL, "get",
                            wraps=persister._EXPERT_BUF_POOL.get) as mock_get:
        persister._EXPERT_BUF_POOL.put(scratch)  # seed one buffer, drained below

        views_false = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=False)
        assert mock_get.call_count == 0, "tensor_align=False drew from the scratch pool"
        assert bytes(views_false["w"].numpy().tobytes()) == payload

        views_true = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=True)
        assert mock_get.call_count == 1, "tensor_align=True should draw exactly one buffer (one group)"
        assert bytes(views_true["w"].numpy().tobytes()) == payload
        # buffer returned to the pool afterward, so it doesn't leak into other tests
        assert persister._EXPERT_BUF_POOL.qsize() == 1


# ── power-of-two pinned arena clamp ─────────────────────────────────────────

from vaayu.persist.safetensor_model_persister import largest_pow2_at_most
from vaayu.vaayu_base import clamp_slab_count, unclamped_slab_count

_GiB = 1024 ** 3

# Both on-device geometries, from the shipped load logs (CLAUDE.md §5: these are
# TEST FIXTURES describing real models, not constants in production code).
_GEO_MINIMAX = dict(budget_bytes=int(8.5 * _GiB), slab_bytes=8_900_000, pool_size=6)
_GEO_QWEN = dict(budget_bytes=int(8.5 * _GiB), slab_bytes=11_800_000, pool_size=6)


def test_largest_pow2_never_exceeds_request():
    for req in (1, 2, 1023, 1024, 1025, 9_203_000_000):
        n = largest_pow2_at_most(req)
        assert n <= req
        assert n & (n - 1) == 0


def test_largest_pow2_is_identity_on_exact_powers():
    # Measured: the allocator charges an exact power of two exactly, so the
    # clamp must not shrink a request that is already one.
    for p in range(0, 35):
        assert largest_pow2_at_most(1 << p) == (1 << p)


def test_largest_pow2_of_nonpositive_is_zero():
    assert largest_pow2_at_most(0) == 0
    assert largest_pow2_at_most(-5) == 0


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_clamp_wastes_less_than_one_slab_of_the_block_it_pays_for(geo):
    # Tightness: the clamped backing must fill the power-of-two block to within
    # one slab. A looser clamp would give away cache for no extra RAM — the
    # backing size is what is charged, so every slab below the boundary is pure
    # loss. (The backing itself is NOT a power of two: it is the largest whole
    # number of slabs that fits inside one.)
    target = largest_pow2_at_most(unclamped_slab_count(**geo) * geo["slab_bytes"])
    backing = clamp_slab_count(**geo) * geo["slab_bytes"]
    assert target - backing < geo["slab_bytes"]


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_clamped_backing_fits_under_the_pow2_the_allocator_charges(geo):
    # The whole point: the backing store must fit inside the power of two we are
    # already paying for, so the allocator does not double the charge.
    n = clamp_slab_count(**geo)
    assert n * geo["slab_bytes"] <= largest_pow2_at_most(
        unclamped_slab_count(**geo) * geo["slab_bytes"])


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_clamp_never_grows_the_arena(geo):
    assert clamp_slab_count(**geo) <= unclamped_slab_count(**geo)


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_clamp_keeps_the_staging_addend(geo):
    # The +get_expert_pool_size() in-flight slabs must survive the clamp, or the
    # loader starves while the cache is full.
    assert clamp_slab_count(**geo) >= geo["pool_size"]


def test_tiny_budget_still_yields_the_staging_slabs():
    # Degenerate box: budget smaller than one slab. Never return 0 slabs.
    assert clamp_slab_count(budget_bytes=1, slab_bytes=8_900_000, pool_size=6) >= 6


def test_clamp_is_a_noop_when_nothing_was_wasted():
    # The clamp shrinks a box IFF the allocator was over-charging it, i.e. iff the
    # nominal backing size is not already an exact power of two. Construct a
    # budget whose unclamped backing lands exactly on 2 GiB with slab_bytes a
    # power of two, and assert the count is untouched.
    slab_bytes = 1 << 20                      # 1 MiB slabs
    pool_size = 6
    n_target = 2048                           # 2 GiB backing, exactly
    budget = (n_target - pool_size) * slab_bytes
    assert unclamped_slab_count(budget, slab_bytes, pool_size) == n_target
    assert clamp_slab_count(budget, slab_bytes, pool_size) == n_target


# ── Finding 1: the budget trim must exclude the pool_size addend ───────────
#
# ExpertRAMCache.put_slab (vaayu/persist/expert_cache.py) charges arena.slab_bytes
# per entry against budget_bytes, evicting while current_usage + size >
# budget_bytes -- so the cache's steady-state entry cap is exactly
# budget_bytes // slab_bytes. The arena is deliberately n_slabs = cache slabs +
# pool_size in-flight staging slabs (unclamped_slab_count's docstring). The trim
# at the arena-mirroring call site must size the budget off the CACHE's share of
# the arena (n_slabs - pool_size), not the whole arena, or the loader's landing
# zone gets folded into the cache's own cap and the freelist reserve collapses
# to 0. This test pins that arithmetic identity directly -- it is the fixed
# expression from vaayu_base.py's trim block, exercised as pure arithmetic
# (CLAUDE.md: no VaayuBaseModel construction).

def _trimmed_budget(n_slabs: int, slab_bytes: int, pool_size: int) -> int:
    """Mirrors the vaayu_base.py trim block's corrected _arena_bytes expression."""
    return max(0, n_slabs - pool_size) * slab_bytes


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
@pytest.mark.parametrize("pool_size", [6, 9])
def test_trimmed_budget_leaves_the_pool_size_reserve_free(geo, pool_size):
    geo = dict(geo, pool_size=pool_size)
    n_slabs = clamp_slab_count(**geo)
    trimmed_budget = _trimmed_budget(n_slabs, geo["slab_bytes"], pool_size)
    cache_cap = trimmed_budget // geo["slab_bytes"]
    # The cache can never fill the arena: its steady-state entry cap plus the
    # staging addend must equal (not exceed) the arena's slab count.
    assert cache_cap + pool_size == n_slabs


@pytest.mark.parametrize("pool_size", [6, 9])
def test_trimmed_budget_is_strictly_less_than_whole_arena_when_pool_size_positive(pool_size):
    # Regression guard for the bug as found: using the WHOLE arena (n_slabs *
    # slab_bytes) as the trimmed budget folds the staging addend into the
    # cache's own cap, giving a freelist reserve of 0.
    n_slabs, slab_bytes = 965, 8_900_000
    whole_arena = n_slabs * slab_bytes
    trimmed = _trimmed_budget(n_slabs, slab_bytes, pool_size)
    assert trimmed < whole_arena
    assert (whole_arena - trimmed) == pool_size * slab_bytes


# ── Finding 2: the clamp must not fire on a pageable (non-pinned) arena ─────
#
# Power-of-two charging is a caching HOST allocator property -- pinned memory
# only. A pageable torch.empty allocation (CPU-only box, or the pinned-alloc
# RuntimeError retry path) pays no round-up, so clamping it trades cache for
# zero RAM freed. The call site guards clamp_slab_count behind `if _pin`; this
# helper mirrors that gate as pure arithmetic.

def _n_slabs_for(pin: bool, budget_bytes: int, slab_bytes: int, pool_size: int) -> int:
    """Mirrors the vaayu_base.py call-site gate: clamp only when pinning."""
    unclamped = unclamped_slab_count(budget_bytes, slab_bytes, pool_size)
    return clamp_slab_count(budget_bytes, slab_bytes, pool_size) if pin else unclamped


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_unpinned_arena_gets_the_unclamped_count(geo):
    unclamped = unclamped_slab_count(**geo)
    assert _n_slabs_for(False, **geo) == unclamped
    # Sanity: on these real geometries the clamp DOES bite when pinned, so this
    # isn't vacuously true because clamping happened to be a no-op.
    assert clamp_slab_count(**geo) < unclamped


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_pinned_arena_gets_the_clamped_count(geo):
    assert _n_slabs_for(True, **geo) == clamp_slab_count(**geo)


# ── Finding 3: the printed "frees" figure must be exact, not an upper bound ─
#
# The RAM actually returned is _charged_bytes - next_pow2_at_least(actual
# backing bytes), not _charged_bytes - (n_slabs * slab_bytes) directly -- those
# only coincide when the clamped backing itself lands exactly on the power-of-
# two floor. In the degenerate max(pool_size, ...) branch of clamp_slab_count,
# the backing can exceed that floor, and the naive subtraction overstates the
# freed bytes.

from vaayu.persist.safetensor_model_persister import next_pow2_at_least


def test_freed_bytes_is_zero_when_degenerate_branch_exceeds_the_floor():
    # Tiny budget forces clamp_slab_count's max(pool_size, ...) branch: the
    # backing (pool_size slabs) exceeds the power-of-two floor computed from
    # the (much smaller) unclamped request actually asked for.
    budget_bytes, slab_bytes, pool_size = 1, 8_900_000, 6
    unclamped = unclamped_slab_count(budget_bytes, slab_bytes, pool_size)
    n_slabs = clamp_slab_count(budget_bytes, slab_bytes, pool_size)
    charged_bytes = next_pow2_at_least(unclamped * slab_bytes)
    naive_freed = charged_bytes - n_slabs * slab_bytes
    exact_freed = charged_bytes - next_pow2_at_least(n_slabs * slab_bytes)
    # The degenerate branch actually exceeds the floor here, so the naive and
    # exact figures must differ -- and the exact one must be the smaller,
    # honest bound (never negative-implying nonsense in real use, since the
    # clamp guarantees n_slabs * slab_bytes <= the floor in the non-degenerate
    # case; here it's the reverse, so exact_freed <= naive_freed).
    assert exact_freed <= naive_freed
    assert exact_freed == 0


@pytest.mark.parametrize("geo", [_GEO_MINIMAX, _GEO_QWEN], ids=["minimax", "qwen"])
def test_freed_bytes_close_to_naive_figure_at_real_geometries(geo):
    # At the real on-device geometries (non-degenerate branch) the two figures
    # differ by less than one slab -- per the finding's own observation ("both
    # round to the same printed value" at typical precision). The exact figure
    # is still the correct one to print; this just confirms the fix is not a
    # behavior change at the geometries that matter on-device.
    unclamped = unclamped_slab_count(**geo)
    n_slabs = clamp_slab_count(**geo)
    charged_bytes = next_pow2_at_least(unclamped * geo["slab_bytes"])
    naive_freed = charged_bytes - n_slabs * geo["slab_bytes"]
    exact_freed = charged_bytes - next_pow2_at_least(n_slabs * geo["slab_bytes"])
    assert exact_freed <= naive_freed
    assert naive_freed - exact_freed < geo["slab_bytes"]


# ── Task 5: floor the clamp at the model's working set ─────────────────────
#
# Task 4's on-device gate FAILED on Qwen: 0.43 tok/s against a 0.82 floor (-48%),
# with the engine's own WARNING that the clamped 8.49 GB cache budget no longer
# covers Qwen's 8.90 GB working set (the unclamped 9.13 GB budget did, by 0.23 GB).
# MiniMax's 4.40 GB working set is nowhere near either boundary, which is why it
# passed. The clamp must not fire when doing so would convert a fitting model
# into a thrashing one -- only when the clamped cache still covers the working
# set, or the working set doesn't fit either way (thrashes regardless, so the
# RAM is worth more).

_WS_QWEN = 8.90 * 1e9        # measured working_set_bytes, Qwen3-235B (log line)
_WS_MINIMAX = 4.40 * 1e9     # measured working_set_bytes, MiniMax-M2.7 (log line)


def test_clamp_is_skipped_when_it_would_break_a_fitting_working_set():
    # Qwen: unclamped cache 9.13 GB fits the 8.90 GB working set; clamped 8.49 GB does not.
    # Clamping here costs 48% of decode, so it must not happen.
    n = clamp_slab_count(**_GEO_QWEN, working_set_bytes=_WS_QWEN)
    assert n == unclamped_slab_count(**_GEO_QWEN)


def test_clamp_still_applies_when_the_working_set_fits_after_clamping():
    # MiniMax: 4.40 GB working set fits inside the clamped cache, so the win is free.
    n = clamp_slab_count(**_GEO_MINIMAX, working_set_bytes=_WS_MINIMAX)
    assert n == clamp_slab_count(**_GEO_MINIMAX)
    assert n < unclamped_slab_count(**_GEO_MINIMAX)


def test_clamp_still_applies_when_the_working_set_fits_neither_way():
    # A working set larger than even the unclamped cache thrashes either way; take the RAM.
    huge = 50 * 1e9
    n = clamp_slab_count(**_GEO_QWEN, working_set_bytes=huge)
    assert n == clamp_slab_count(**_GEO_QWEN)


def test_working_set_defaults_to_zero_and_changes_nothing():
    for geo in (_GEO_QWEN, _GEO_MINIMAX):
        assert clamp_slab_count(**geo) == clamp_slab_count(**geo, working_set_bytes=0)
