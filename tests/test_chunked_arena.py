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
