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

A dense checkpoint has the same problem with a different cache (S29,
2026-10-07): its RAM layer cache is sized from total RAM right after the choice
and holds its layers as clones. Llama-3.3-70B (41.4 GB, 27.2 GiB free) went
buffered at "50 % cacheable"; ~16.7 GB streamed per token through what was left,
the page cache served at most 0.6 % of it while the kernel reclaimed ~34 million
pages per request, and decode took 2.48 s/token against 1.96 with O_DIRECT
(-21.0 %, 8 mirrored arms, greedy ids identical).

Free RAM (and the total RAM the dense cache is derived from) is the function's
input, so it is the one thing faked here; the checkpoints are real files and the
expert index is the real one.
"""
from types import SimpleNamespace

import pytest

from vaayu.persist import safetensor_model_persister as P
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.persist.ram_cache import RAM_CACHE_RESERVE

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


def _choose(monkeypatch, root, room_bytes, layer_cache_bytes=0, **kw) -> bool:
    """The engine's choice with `room_bytes` of free RAM beyond the reserve, on a box
    whose TOTAL RAM derives a dense layer cache of `layer_cache_bytes`."""
    available = int(RESERVE + room_bytes)
    total = RAM_CACHE_RESERVE + int(layer_cache_bytes)
    monkeypatch.setattr(P.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=available, total=total))
    return P._should_use_direct_io(str(root), **kw)


def _no_ram_failure(capsys) -> bool:
    """The rule falls back to O_DIRECT on ANY exception -- a fake that breaks it would
    pass every O_DIRECT assertion for the wrong reason."""
    return "Failed to detect RAM" not in capsys.readouterr().out


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


def test_o_direct_when_the_dense_layer_cache_will_take_the_free_ram(tmp_path, monkeypatch, capsys):
    """The S29 bug. Free RAM is half the checkpoint, but the dense RAM layer cache
    is derived from total RAM and takes all of it: the page cache gets nothing."""
    _write_dense_checkpoint(tmp_path)
    room = int(0.50 * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room, layer_cache_bytes=room) is True
    assert _no_ram_failure(capsys)


def test_buffered_when_ram_remains_after_the_dense_layer_cache(tmp_path, monkeypatch, capsys):
    """A small dense model on a big box: half the checkpoint still fits beside the
    layer cache, so the page cache is real."""
    _write_dense_checkpoint(tmp_path)
    layer_cache = _model_bytes(tmp_path)
    room = layer_cache + int(0.50 * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room, layer_cache_bytes=layer_cache) is False
    assert _no_ram_failure(capsys)


@pytest.mark.parametrize("share,direct", [(0.43, False), (0.39, True)])
def test_a_dense_load_with_the_layer_cache_off_keeps_the_forty_percent_rule(
        tmp_path, monkeypatch, capsys, share, direct):
    """RAM_CACHE_BUDGET_GB = 0 (the server's knob, passed through) turns the layer
    cache off; nothing of the engine's then competes for the free RAM, so the rule
    is what it was -- buffered from 40 % of the checkpoint up. Total RAM would
    derive a cache that swallows everything if the override were ignored."""
    _write_dense_checkpoint(tmp_path)
    room = int(share * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room, layer_cache_bytes=100 * room,
                   ram_cache_budget=0) is direct
    assert _no_ram_failure(capsys)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_expert_checkpoint_is_not_charged_the_dense_layer_cache(tmp_path, monkeypatch, capsys,
                                                                   layout):
    """A MoE load builds no RAM layer cache (vaayu_base: budget 0 when is_moe), so
    only its expert cache is charged -- gpt-oss stays buffered, where the page cache
    serves its prefill re-reads (S29: 0.98 GB requested, 0 GB from the drive on a
    HIT turn)."""
    make, rows, cols, n = LAYOUTS[layout]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    room = _all_experts_bytes(tmp_path) + int(0.50 * _model_bytes(tmp_path))

    assert _choose(monkeypatch, tmp_path, room, layer_cache_bytes=100 * room) is False
    assert _no_ram_failure(capsys)


def test_the_choice_writes_nothing_into_the_checkpoint(tmp_path, monkeypatch):
    """Some checkpoints must never be written to (DeepSeek-V4's model dir)."""
    make, rows, cols, n = LAYOUTS["packed-int4 (Qwen3-Next-like)"]
    _write_moe_checkpoint(tmp_path, make, rows, cols, n)
    before = sorted(p.name for p in tmp_path.iterdir())

    _choose(monkeypatch, tmp_path, int(0.43 * _model_bytes(tmp_path)))

    assert sorted(p.name for p in tmp_path.iterdir()) == before
