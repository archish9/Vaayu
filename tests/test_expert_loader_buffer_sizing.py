"""The expert loader's pinned buffers are sized from the checkpoint, not guessed.

Row 44 (2026-09-19): `init_model_io` sized each buffer at `(32 if exl3 else 512)
MB` against a `max(0.5, avail - 27 GB)` budget, which is 0.5 GB on this 31 GB
box. 0.5 GB / 512 MB = ONE buffer for every non-EXL3 checkpoint, and the buffer
count is `get_expert_pool_size()` -- the width of `_moe_executor`, the ring pool
and the stager executor. Qwen3-Next's largest expert is ~1.8 MB, so every one of
its expert reads ran serially: 66 % of a 2k prefill (86 s). Pre-filling the pool
with 8 small buffers, engine otherwise untouched, measured 86.3 -> 57.1 s.

The size is now the checkpoint's own largest aligned expert span, read off its
expert offset index (CLAUDE.md §5: expert byte-size is derived, never a
literal). A checkpoint with no per-expert tensors keeps the old sizing.
"""
from pathlib import Path
from types import SimpleNamespace
import queue

import pytest
import torch
from safetensors.torch import save_file

from vaayu.persist import safetensor_model_persister as P
from vaayu.persist.expert_offset_index import ExpertOffsetIndex


def _packed_int4_expert(prefix: str, rows: int, cols: int) -> dict:
    """compressed-tensors pack-quantized expert (Qwen3-Next AWQ layout)."""
    t = {}
    for role, (r, c) in (("gate_proj", (rows, cols)), ("up_proj", (rows, cols)),
                         ("down_proj", (cols, rows))):
        t[f"{prefix}.{role}.weight_packed"] = torch.zeros(r, c // 8, dtype=torch.int32)
        t[f"{prefix}.{role}.weight_scale"] = torch.zeros(r, c // 32, dtype=torch.bfloat16)
        t[f"{prefix}.{role}.weight_shape"] = torch.tensor([r, c], dtype=torch.int64)
    return t


def _exl3_like_expert(prefix: str, rows: int, cols: int) -> dict:
    """EXL3 trellis layout (Qwen3-235B / MiniMax-M2.7)."""
    t = {}
    for role in ("w1", "w2", "w3"):
        t[f"{prefix}.{role}.trellis"] = torch.zeros(rows // 16, cols // 16, 64,
                                                    dtype=torch.int16)
        t[f"{prefix}.{role}.suh"] = torch.zeros(rows, dtype=torch.float16)
        t[f"{prefix}.{role}.svh"] = torch.zeros(cols, dtype=torch.float16)
    return t


# Two geometries with different expert byte sizes, so a test that passes on one
# by coincidence cannot pass on the other.
LAYOUTS = {
    "packed-int4 (Qwen3-Next-like)": (_packed_int4_expert, 64, 256, 8),
    "exl3 (MiniMax-like)": (_exl3_like_expert, 128, 512, 16),
}


def _write_moe_checkpoint(root: Path, make_expert, rows, cols, n_experts,
                          n_layers: int = 2) -> None:
    for li in range(n_layers):
        name = f"model.layers.{li}"
        tensors = {f"{name}.self_attn.q_proj.weight": torch.zeros(8, 8),
                   f"{name}.mlp.gate.weight": torch.zeros(n_experts, 8)}
        for e in range(n_experts):
            tensors.update(make_expert(f"{name}.mlp.experts.{e}", rows, cols))
        save_file(tensors, str(root / f"{name}.safetensors"))


def _write_dense_checkpoint(root: Path) -> None:
    save_file({"model.layers.0.mlp.up_proj.weight": torch.zeros(64, 32)},
              str(root / "model.layers.0.safetensors"))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_buffer_is_the_checkpoints_largest_expert_span(tmp_path, layout):
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)

    got = P.expert_read_buffer_bytes(str(tmp_path))

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build(write_cache=False)
    assert got == P.compute_slab_bytes(idx)
    # Every single read load_expert_sparse can issue must fit one buffer.
    for experts in idx.index.values():
        for metas in experts.values():
            for group in P._group_contiguous_ranges(metas):
                _, _, raw_aligned, _, _ = P._group_layout(group)
                assert raw_aligned <= got


def test_the_two_layouts_derive_different_sizes(tmp_path):
    """Guards the derivation itself: a constant would pass the test above."""
    sizes = []
    for i, (make, rows, cols, n) in enumerate(LAYOUTS.values()):
        d = tmp_path / str(i)
        d.mkdir()
        _write_moe_checkpoint(d, make, rows, cols, n)
        sizes.append(P.expert_read_buffer_bytes(str(d)))
    assert sizes[0] != sizes[1]


def test_a_checkpoint_without_experts_derives_zero_and_writes_nothing(tmp_path):
    _write_dense_checkpoint(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert P.expert_read_buffer_bytes(str(tmp_path)) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _run_pool_sizing(monkeypatch, root: Path, avail_gib: float):
    """init_model_io's buffer-pool branch only, with allocations recorded
    instead of pinned (no CUDA needed) and no ring init."""
    allocs = []

    def fake_empty(n, dtype=None, pin_memory=False):
        allocs.append((n, pin_memory))
        return SimpleNamespace(nbytes=n)

    monkeypatch.setattr(P, "_EXPERT_BUF_POOL", queue.Queue())
    monkeypatch.setattr(P, "_EXPERT_PINNED_SIZE", 0)
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 1)
    monkeypatch.setattr(P, "_USE_PINNED", True)
    monkeypatch.setattr(P, "_USE_IOURING", False)
    monkeypatch.setattr(P.torch, "empty", fake_empty)
    monkeypatch.setattr(P.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=int(avail_gib * 1024 ** 3)))
    P.init_model_io(str(root), use_direct_io=True)
    return allocs


# 22.3 GiB is what this box reported at init_model_io in row 44's runs (BALLAST_GB=25):
# a 0.5 GB pool budget, the floor, which is where every run here lands.
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_moe_checkpoint_gets_a_full_width_pool_on_this_box(tmp_path, monkeypatch, layout):
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    want = P.expert_read_buffer_bytes(str(tmp_path))

    allocs = _run_pool_sizing(monkeypatch, tmp_path, avail_gib=22.3)

    assert P._EXPERT_PINNED_SIZE == want
    assert P.get_expert_pool_size() == P._EXPERT_POOL_N, (
        f"pool={P.get_expert_pool_size()} -- one buffer means one loader worker "
        f"and one ring: every expert read serialized (row 44)")
    assert allocs == [(want, True)] * P._EXPERT_POOL_N


def test_a_checkpoint_without_experts_keeps_the_previous_sizing(tmp_path, monkeypatch):
    """Dense checkpoints never call the expert loader; their pool (and so their
    ring count) is left exactly as it was."""
    _write_dense_checkpoint(tmp_path)
    _run_pool_sizing(monkeypatch, tmp_path, avail_gib=22.3)
    assert P._EXPERT_PINNED_SIZE == 512 * 1024 * 1024
    assert P.get_expert_pool_size() == 1
