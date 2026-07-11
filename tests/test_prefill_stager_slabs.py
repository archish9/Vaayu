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
    byte-exactly at the same relative offsets (CPU stand-in for the GPU copy)."""
    from vaayu.persist.safetensor_model_persister import (
        expert_span_bytes, load_expert_into_slab, views_from_packed_copy)

    payload_a = bytes(range(256)) * 4              # 1024 B
    payload_b = bytes(reversed(range(256))) * 2    # 512 B
    f = tmp_path / "shard.bin"
    blob = bytearray(600_000)
    off_a, off_b = 8192, 500_000                   # two far-apart groups
    blob[off_a:off_a + 1024] = payload_a
    blob[off_b:off_b + 512] = payload_b
    f.write_bytes(bytes(blob))
    metas = {
        "gate.trellis": _meta(off_a, 1024, [1024]),
        "up.trellis": _meta(off_b, 512, [512]),
    }
    slab = torch.empty(expert_span_bytes(metas), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False)

    used = max((v.data_ptr() - slab.data_ptr()) + v.nbytes for v in views.values())
    copied = slab[:used].clone()                    # stand-in for the H2D copy
    rebuilt = views_from_packed_copy(views, slab, copied)
    assert bytes(rebuilt["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(rebuilt["up.trellis"].numpy().tobytes()) == payload_b
    for v in rebuilt.values():
        assert v.data_ptr() >= copied.data_ptr()
