"""Chunked slab-arena backing (Plan L).

The caching host allocator charges the next power of two for every pinned
allocation, so ONE 9.13 GB backing tensor costs 16 GiB resident. Several
exact-power-of-two chunks are charged exactly, recovering ~7.9 GB while
keeping every slab (measured 2026-08-07, E1/E4).

CLAUDE.md §5: sizing is tested over BOTH target geometries. All tests are
CPU-only (pin=False) so the suite runs without CUDA.
"""
import pytest
import torch

from vaayu.persist.expert_cache import SlabArena
from vaayu.persist.safetensor_model_persister import (
    _ALIGN,
    chunk_decomposition,
    largest_pow2_at_most,
    next_pow2_at_least,
)

# --- Task 2: SlabArena chunked backing ---


@pytest.mark.parametrize("n_slabs,slab_bytes", [(20, 4096), (7, 8192)])
def test_arena_allocates_chunks_not_one_backing_tensor(n_slabs, slab_bytes):
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    assert not hasattr(arena, "_backing"), \
        "the single backing tensor must be gone, not shadowed"
    assert len(arena._chunks) >= 1
    assert sum(arena._chunk_nslabs) == arena.n_slabs


@pytest.mark.parametrize("n_slabs,slab_bytes", [(20, 4096), (7, 8192)])
def test_every_slab_is_distinct_and_the_right_size(n_slabs, slab_bytes):
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    seen = set()
    for i in range(arena.n_slabs):
        t = arena.tensor(i)
        assert t.numel() == slab_bytes
        assert t.data_ptr() not in seen, f"slab {i} aliases an earlier slab"
        seen.add(t.data_ptr())


@pytest.mark.parametrize("n_slabs,slab_bytes", [(20, 4096), (7, 8192)])
def test_writing_one_slab_does_not_disturb_its_neighbours(n_slabs, slab_bytes):
    # The real corruption symptom, expressed directly: a slab must be exactly
    # its own bytes, including the first and last slab of every chunk.
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    for i in range(arena.n_slabs):
        arena.tensor(i).fill_(0)
    target = arena.n_slabs // 2
    arena.tensor(target).fill_(0xAB)
    for i in range(arena.n_slabs):
        expected = 0xAB if i == target else 0
        assert int(arena.tensor(i).min()) == expected
        assert int(arena.tensor(i).max()) == expected


@pytest.mark.parametrize("n_slabs,slab_bytes", [(20, 4096), (7, 8192)])
def test_slab_lives_entirely_inside_one_chunk(n_slabs, slab_bytes):
    # If a slab straddled two allocations, tensor() would hand io_uring a span
    # crossing a boundary -- the failure this design exists to prevent.
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    spans = [(c.data_ptr(), c.data_ptr() + c.numel()) for c in arena._chunks]
    for i in range(arena.n_slabs):
        t = arena.tensor(i)
        start, end = t.data_ptr(), t.data_ptr() + t.numel()
        assert any(lo <= start and end <= hi for lo, hi in spans), \
            f"slab {i} is not contained in a single chunk"


def test_chunk_bases_are_o_direct_aligned():
    # slab_bytes is _ALIGN-aligned so slab starts stay sector-aligned -- but
    # only RELATIVE TO THE CHUNK BASE, so each base must be aligned too or
    # O_DIRECT reads break. Found by the E2 audit, 2026-08-07.
    arena = SlabArena(20, _ALIGN * 2, pin=False)
    for c in arena._chunks:
        assert c.data_ptr() % _ALIGN == 0


def test_the_arena_requests_only_powers_of_two(monkeypatch):
    # THE property this whole plan exists to establish, and the one a green
    # suite otherwise cannot see: chunk_decomposition returning pow2 sizes is
    # worthless if SlabArena does not REQUEST them. A non-pow2 request is
    # charged the next power of two -- an 8 GiB chunk becomes a 16 GiB charge.
    real_empty = torch.empty
    requested = []

    def spy(*args, **kwargs):
        if kwargs.get("pin_memory") is not None and args:
            requested.append(int(args[0]))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", spy)
    SlabArena(20, 4096, pin=False)
    assert requested, "no allocations observed -- the spy did not attach"
    for n in requested:
        assert n & (n - 1) == 0, f"non-power-of-two allocation request: {n}"


def test_acquire_release_still_cover_every_slab():
    # acquire/release are index-based and must be untouched by chunking.
    arena = SlabArena(9, 4096, pin=False)
    got = []
    while (idx := arena.acquire()) is not None:
        got.append(idx)
    assert sorted(got) == list(range(arena.n_slabs))
    assert arena.acquire() is None
    for idx in got:
        arena.release(idx)
    assert arena.acquire() is not None


# Real derived geometries, used as test INPUTS only (never in vaayu/).
# MiniMax-M2.7 and Qwen3-235B as measured 2026-08-07 by compute_slab_bytes().
GEOMETRIES = [
    pytest.param(1027, 8_888_320, id="minimax-like"),
    pytest.param(771, 11_841_536, id="qwen-like"),
]


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


@pytest.mark.parametrize("n_slabs,slab_bytes", GEOMETRIES)
def test_every_chunk_is_an_exact_power_of_two(n_slabs, slab_bytes):
    # The whole point: the allocator charges an exact pow2 exactly, so any
    # non-pow2 chunk silently reintroduces the round-up this plan removes.
    chunks = chunk_decomposition(n_slabs, slab_bytes)
    assert chunks, "decomposition must not be empty for a positive slab count"
    for c in chunks:
        assert _is_pow2(c), f"chunk {c} is not a power of two"


@pytest.mark.parametrize("n_slabs,slab_bytes", GEOMETRIES)
def test_capacity_covers_the_requested_slab_count(n_slabs, slab_bytes):
    # A slab must never straddle a chunk boundary, so each chunk contributes
    # floor(chunk / slab_bytes) slabs and the tail remainder is unusable.
    chunks = chunk_decomposition(n_slabs, slab_bytes)
    capacity = sum(c // slab_bytes for c in chunks)
    assert capacity >= n_slabs


@pytest.mark.parametrize("n_slabs,slab_bytes", GEOMETRIES)
def test_charge_beats_the_single_tensor_round_up(n_slabs, slab_bytes):
    # This is the entire deliverable, stated as an assertion.
    chunks = chunk_decomposition(n_slabs, slab_bytes)
    single_tensor_charge = next_pow2_at_least(n_slabs * slab_bytes)
    chunked_charge = sum(chunks)
    assert chunked_charge < single_tensor_charge
    # Measured 2026-08-07: 9.14 GB vs 17.18 GB on both models. Assert the
    # order of magnitude of the win, not the exact bytes.
    assert chunked_charge < 0.6 * single_tensor_charge


@pytest.mark.parametrize("n_slabs,slab_bytes", GEOMETRIES)
def test_k_stays_small(n_slabs, slab_bytes):
    # E4 measured k=36 at full bandwidth, so a large k is not a correctness
    # problem -- but the decomposition should not be pathological either.
    assert len(chunk_decomposition(n_slabs, slab_bytes)) <= 8


def test_empty_for_non_positive_slab_count():
    assert chunk_decomposition(0, 8_888_320) == []
    assert chunk_decomposition(-1, 8_888_320) == []


def test_single_slab_gets_one_chunk_that_holds_it():
    chunks = chunk_decomposition(1, 11_841_536)
    assert len(chunks) == 1
    assert _is_pow2(chunks[0])
    assert chunks[0] // 11_841_536 >= 1


def test_slab_larger_than_any_natural_chunk_still_terminates():
    # A hypothetical model with a very large expert: the first candidate chunk
    # (largest_pow2_at_most of the need) can be SMALLER than one slab, which
    # would yield zero capacity and loop forever without the fallback.
    chunks = chunk_decomposition(3, 700 * 1024 * 1024)
    assert sum(c // (700 * 1024 * 1024) for c in chunks) >= 3


@pytest.mark.parametrize("n_slabs,slab_bytes", [(20, 4096), (7, 8192)])
def test_slab_index_of_round_trips_every_slab(n_slabs, slab_bytes):
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    for i in range(arena.n_slabs):
        assert arena.slab_index_of(arena.tensor(i)) == i


@pytest.mark.parametrize("n_slabs,slab_bytes", [(5, 3 * 4096)])
def test_slab_index_of_handles_first_and_last_slab_of_every_chunk(n_slabs, slab_bytes):
    # The boundaries a single-_backing arena could not have. (20,4096)/(7,8192)
    # were dropped: Task 2's alignment-escalation path collapses them to a
    # single chunk under pytest (non-deterministically -- observed 1 chunk in
    # 1/40 standalone runs too), which silently degenerates this test into a
    # subset of the round-trip test. (5, 3*4096) is deterministically
    # multi-chunk (_chunk_first == [0, 2, 4] on 40/40 constructions).
    arena = SlabArena(n_slabs, slab_bytes, pin=False)
    assert len(arena._chunk_first) > 1, (
        f"fixture must produce a multi-chunk arena, got {len(arena._chunk_first)} "
        f"chunk(s) -- this test is vacuous otherwise")
    for c, first in enumerate(arena._chunk_first):
        last = first + arena._chunk_nslabs[c] - 1
        assert arena.slab_index_of(arena.tensor(first)) == first
        assert arena.slab_index_of(arena.tensor(last)) == last


def test_slab_index_of_is_global_not_chunk_local():
    # THE failure mode this function's one production caller cannot survive:
    # a chunk-LOCAL index is silently correct for chunk 0 and wrong for every
    # chunk after it, which attaches the H2D read-guard to the wrong slab and
    # lets the right one be overwritten mid-DMA. A chunk-local implementation
    # returns [0,1,0,1,0] here where the correct one returns [0,1,2,3,4].
    slab_bytes = 3 * 4096          # forces >1 chunk deterministically
    arena = SlabArena(5, slab_bytes, pin=False)
    assert len(arena._chunk_first) > 1, (
        f"fixture must produce a multi-chunk arena, got {len(arena._chunk_first)} "
        f"chunk(s) -- this test is vacuous otherwise")
    assert [arena.slab_index_of(arena.tensor(i)) for i in range(arena.n_slabs)] == \
        list(range(arena.n_slabs))


def test_pointer_in_a_chunk_tail_remainder_returns_none():
    # THE new failure mode. Chunks are powers of two and slab_bytes is not, so
    # each chunk ends with chunk_bytes % slab_bytes bytes belonging to no slab.
    # A pointer there must be rejected, not mapped to a bogus index -- a wrong
    # index frees/guards a slab another expert is still reading.
    slab_bytes = 3 * 4096  # deliberately not a power of two
    arena = SlabArena(5, slab_bytes, pin=False)
    tails = 0
    for c, chunk in enumerate(arena._chunks):
        used = arena._chunk_nslabs[c] * slab_bytes
        if used >= chunk.numel():
            continue                      # this chunk happens to have no tail
        tails += 1
        tail = chunk[used:used + 1]
        assert arena.slab_index_of(tail) is None
    assert tails > 0, "fixture must produce at least one chunk with a tail"


def test_pointer_outside_every_chunk_returns_none():
    arena = SlabArena(6, 4096, pin=False)
    foreign = torch.zeros(16, dtype=torch.uint8)
    assert arena.slab_index_of(foreign) is None
    # And a view into an unrelated arena, which is the realistic case
    # (a legacy cloned expert, or a second model's arena).
    other = SlabArena(6, 4096, pin=False)
    assert arena.slab_index_of(other.tensor(0)) is None


def test_interior_pointer_maps_to_its_own_slab():
    # moe_patch passes a VIEW of a tensor inside the slab, not the slab start.
    slab_bytes = 4096
    arena = SlabArena(6, slab_bytes, pin=False)
    for i in range(arena.n_slabs):
        interior = arena.tensor(i)[100:200]
        assert arena.slab_index_of(interior) == i
        # Last byte of the slab: the offset adjacent to the next slab, and the
        # one an off-by-one in the `//` bound would break.
        assert arena.slab_index_of(arena.tensor(i)[slab_bytes - 1:]) == i
