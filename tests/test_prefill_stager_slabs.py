"""Stager slab-staging helpers: read-into-slab with fallback + release accounting.

CPU-only (pin=False); no CUDA needed.
"""
import torch

from vaayu.persist.expert_cache import SlabArena
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import _ALIGN, expert_span_bytes
from vaayu.vaayu_base import stage_read_expert


def _meta(offset, length, shape, dtype="U8"):
    return ExpertTensorMeta(file_offset=offset, byte_length=length, shape=shape, dtype=dtype)


def _make_shard(tmp_path):
    payload = bytes(range(256)) * 16  # 4096 B
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    blob[8192:8192 + 4096] = payload
    f.write_bytes(bytes(blob))
    metas = {"up.trellis": _meta(8192, 4096, [4096])}
    return str(f), metas, payload


def test_stage_read_uses_slab_when_available(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    arena = SlabArena(n_slabs=2, slab_bytes=expert_span_bytes(metas), pin=False)
    views, sidx = stage_read_expert(shard, metas, arena)
    assert sidx is not None
    slab = arena.tensor(sidx)
    v = views["up.trellis"]
    assert bytes(v.numpy().tobytes()) == payload
    assert slab.data_ptr() <= v.data_ptr() < slab.data_ptr() + slab.nbytes
    arena.release(sidx)


def test_stage_read_falls_back_when_arena_exhausted(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    arena = SlabArena(n_slabs=1, slab_bytes=expert_span_bytes(metas), pin=False)
    held = arena.acquire()          # exhaust the arena
    views, sidx = stage_read_expert(shard, metas, arena)
    assert sidx is None             # pageable fallback
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload
    arena.release(held)


def test_stage_read_no_arena(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    views, sidx = stage_read_expert(shard, metas, None)
    assert sidx is None
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload


def test_views_from_packed_copy_rebuilds_offsets(tmp_path):
    """Rebuilding views inside a copied buffer must reproduce every tensor
    byte-exactly at the same relative offsets (CPU stand-in for the GPU copy).

    Includes an F16 (multi-byte-itemsize) group at a 2-byte-aligned file
    offset: an all-U8 test can't exercise PyTorch's `.view(dtype)` storage-
    offset alignment constraint (itemsize 1 offsets are trivially aligned),
    but production expert tensors are F16/BF16, where that constraint is the
    real risk this helper must not silently violate."""
    from vaayu.persist.safetensor_model_persister import (
        expert_span_bytes, load_expert_into_slab, views_from_packed_copy)

    payload_a = bytes(range(256)) * 4              # 1024 B
    payload_b = bytes(reversed(range(256))) * 2    # 512 B
    f16_tensor = torch.tensor([1.5, -2.25, 3.0, 0.125], dtype=torch.float16)
    payload_c = f16_tensor.numpy().tobytes()       # 8 B, F16
    f = tmp_path / "shard.bin"
    blob = bytearray(600_000)
    off_a, off_b, off_c = 8192, 500_000, 550_000   # three far-apart groups
    blob[off_a:off_a + 1024] = payload_a
    blob[off_b:off_b + 512] = payload_b
    blob[off_c:off_c + 8] = payload_c
    f.write_bytes(bytes(blob))
    metas = {
        "gate.trellis": _meta(off_a, 1024, [1024]),
        "up.trellis": _meta(off_b, 512, [512]),
        "down.proj": _meta(off_c, 8, [4], dtype="F16"),
    }
    slab = torch.empty(expert_span_bytes(metas), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False, tensor_align=True)

    used = max((v.data_ptr() - slab.data_ptr()) + v.nbytes for v in views.values())
    copied = slab[:used].clone()                    # stand-in for the H2D copy
    rebuilt = views_from_packed_copy(views, slab, copied)
    assert bytes(rebuilt["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(rebuilt["up.trellis"].numpy().tobytes()) == payload_b
    assert rebuilt["down.proj"].dtype == torch.float16
    assert torch.equal(rebuilt["down.proj"], f16_tensor)
    for v in rebuilt.values():
        assert v.data_ptr() >= copied.data_ptr()


def test_gpu_repack_aligned_offsets_bytes_dtype():
    """Repacked buffer: every view 16-byte aligned relative to the buffer base,
    byte-exact, dtype/shape preserved. CPU stands in for CUDA (device='cpu');
    the >=256-byte CUDA base alignment is verified on-device separately."""
    from vaayu.persist.safetensor_model_persister import (
        gpu_repack_aligned, _TENSOR_ALIGN)

    trellis = torch.arange(10, dtype=torch.float16)      # 20 B -> padded to 32
    suh = torch.arange(4, dtype=torch.float16)           # 8 B
    src = {"trellis": trellis, "suh": suh}

    out = gpu_repack_aligned(src, "cpu")
    base = out["trellis"].data_ptr()                     # first key sits at off 0
    for v in out.values():
        assert (v.data_ptr() - base) % _TENSOR_ALIGN == 0
    assert torch.equal(out["trellis"], trellis)
    assert torch.equal(out["suh"], suh)
    assert out["suh"].dtype == torch.float16 and out["suh"].shape == suh.shape
    # suh must land AFTER trellis's 16-aligned span (20 -> 32), not overlapping
    assert out["suh"].data_ptr() - base == _TENSOR_ALIGN * 2
