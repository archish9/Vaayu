"""Buffered or O_DIRECT: the page cache can only use RAM the engine does not take.

`_should_use_direct_io` goes buffered when the page cache could hold 40 % of the
checkpoint. Its rule dates from before the pinned expert RAM cache existed and
counted ALL the free RAM as page-cache room -- but on a checkpoint with
per-expert tensors the engine sizes its own expert cache from that same free RAM
a moment later and pins it. Measured 2026-10-04 on Qwen3-Next (48.24 GB, 43.9 GB
of it experts, 27 GiB free): the rule said "43 % cacheable" and went buffered;
the page cache then served 1-3 % of the bytes read while the kernel reclaimed
6-9 million pages per turn, and the agent's prefix-HIT turn took 23.55 s against
19.44 s with O_DIRECT (-17.4 %, greedy ids identical).

Free RAM is the function's input, so it is the one thing faked here; the
checkpoints are real files and the expert index is the real one.
"""
from types import SimpleNamespace

import pytest

from vaayu.persist import safetensor_model_persister as P
from vaayu.persist.expert_offset_index import ExpertOffsetIndex

from tests.test_expert_loader_buffer_sizing import (
    LAYOUTS, _write_dense_checkpoint, _write_moe_checkpoint)

GIB = 1024 ** 3
RESERVE = 8 * GIB       # what the rule sets aside for the OS, PyTorch and pinned buffers


def _model_bytes(root) -> int:
    return sum(p.stat().st_size for p in root.glob("*.safetensors"))


def _all_experts_bytes(root) -> int:
    """Pinned RAM that holds every expert of the checkpoint: experts x slab size."""
    idx = ExpertOffsetIndex(str(root))
    idx.build(write_cache=False)
    return P.compute_slab_bytes(idx) * sum(len(layer) for layer in idx.index.values())


def _choose(monkeypatch, root, room_bytes) -> bool:
    """The engine's choice with `room_bytes` of free RAM beyond the reserve."""
    monkeypatch.setattr(P.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=int(RESERVE + room_bytes)))
    return P._should_use_direct_io(str(root))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_o_direct_when_the_expert_cache_will_take_the_free_ram(tmp_path, monkeypatch, layout):
    """The bug. Free RAM is 43 % of the checkpoint, but the experts alone exceed
    it: the expert cache pins all of it and the page cache gets nothing."""
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    room = int(0.43 * _model_bytes(tmp_path))
    assert _all_experts_bytes(tmp_path) > room          # the premise

    assert _choose(monkeypatch, tmp_path, room) is True


@pytest.mark.parametrize("layout", LAYOUTS)
def test_o_direct_when_little_is_left_after_every_expert_is_cached(tmp_path, monkeypatch, layout):
    """Every expert fits, but what is left could hold only 20 % of the checkpoint."""
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    room = _all_experts_bytes(tmp_path) + int(0.20 * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room) is True


@pytest.mark.parametrize("layout", LAYOUTS)
def test_buffered_when_ram_remains_after_every_expert_is_cached(tmp_path, monkeypatch, layout):
    """A small MoE on a big box (gpt-oss here): the expert cache holds every
    expert and half the checkpoint still fits beside it -- the page cache is real."""
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    room = _all_experts_bytes(tmp_path) + int(0.50 * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room) is False


@pytest.mark.parametrize("share,direct", [(0.43, False), (0.39, True)])
def test_a_checkpoint_without_experts_keeps_the_forty_percent_rule(tmp_path, monkeypatch,
                                                                    share, direct):
    """Dense: nothing of the engine's competes for the free RAM, so the rule is
    what it was -- buffered from 40 % of the checkpoint up."""
    _write_dense_checkpoint(tmp_path)
    room = int(share * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room) is direct


def test_the_choice_writes_nothing_into_the_checkpoint(tmp_path, monkeypatch):
    """Some checkpoints must never be written to (DeepSeek-V4's model dir)."""
    make, rows, cols, n = LAYOUTS["packed-int4 (Qwen3-Next-like)"]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    before = sorted(p.name for p in tmp_path.iterdir())

    _choose(monkeypatch, tmp_path, int(0.43 * _model_bytes(tmp_path)))

    assert sorted(p.name for p in tmp_path.iterdir()) == before
