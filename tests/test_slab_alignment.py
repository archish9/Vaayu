"""H2: SlabArena's O_DIRECT premise was asserted in prose and enforced nowhere.

Its docstring states "slab_bytes is _ALIGN-aligned and every chunk base is
_ALIGN-aligned, so every slab start stays sector-aligned for O_DIRECT". Slab
starts are multiples of slab_bytes from the chunk base, so a slab_bytes that is
not _ALIGN-aligned breaks that for every slab after the first.

`compute_slab_bytes` could produce exactly that. `_group_layout` returns
`slab_span = max(raw_aligned_length, packed_span)`: raw_aligned_length goes
through `_align_up` (4096), but packed_span only rounds each tensor to
_TENSOR_ALIGN (16). A group of many small tensors makes packed_span the larger
of the two, and 16-aligned is not 4096-aligned.

Narrow -- never observed on either on-device model, and equally unenforced
before and after Plan L's chunking. The unchecked ASSERTION is the defect.
"""
import pytest

from vaayu.persist.expert_cache import SlabArena
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import (
    _ALIGN,
    _TENSOR_ALIGN,
    compute_slab_bytes,
    expert_span_bytes,
)


class _FakeIndex:
    def __init__(self, index):
        self.index = index


def _many_small_tensors(n=300, size=1):
    """Contiguous 1-byte tensors: the raw span rounds up to one _ALIGN page,
    but per-tensor _TENSOR_ALIGN padding makes packed_span far larger."""
    return {f"t{i}": ExpertTensorMeta(file_offset=i * size, byte_length=size,
                                      shape=[1], dtype="BF16")
            for i in range(n)}


def test_the_fixture_really_does_make_packed_span_dominate():
    """Assert the premise, or this file can silently stop testing anything --
    Plan L's slab_index_of mutant passed 84/84 on fixtures that had collapsed."""
    span = expert_span_bytes(_many_small_tensors())
    assert span > _ALIGN, "packed_span must exceed one page for this to bite"
    assert span % _TENSOR_ALIGN == 0, "premise: packed_span is 16-aligned"


def test_compute_slab_bytes_is_align_aligned_even_when_packed_span_wins():
    metas = _many_small_tensors()
    slab_bytes = compute_slab_bytes(_FakeIndex({"layer0": {0: metas}}))
    assert slab_bytes % _ALIGN == 0, (
        f"slab_bytes={slab_bytes} is not _ALIGN-aligned, so every slab after "
        f"the first starts off a sector boundary")
    assert slab_bytes >= expert_span_bytes(metas), "must still hold the expert"


def test_slab_arena_refuses_a_misaligned_slab_size():
    """Enforce the docstring at the boundary, so a caller computing slab_bytes
    some other way cannot silently break O_DIRECT."""
    with pytest.raises(ValueError) as e:
        SlabArena(2, _ALIGN + _TENSOR_ALIGN, pin=False)
    assert "_ALIGN" in str(e.value) or "4096" in str(e.value)


def test_slab_arena_still_accepts_an_aligned_slab_size():
    arena = SlabArena(2, _ALIGN * 2, pin=False)
    assert arena.slab_bytes == _ALIGN * 2
