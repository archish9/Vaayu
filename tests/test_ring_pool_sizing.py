"""The io_uring ring pool must be sized to the DRIVE, not to the caller pool.

P8 (2026-09-03) measured this path directly (`plans/probes/p8_samepath.py`):
with 8 rings, aggregate read bandwidth collapses from ~7.09 GB/s to ~5.15 GB/s
as soon as 8 callers are in flight — which is exactly what decode does, since
`get_expert_pool_size()` is 8 on this box. With 2-4 rings the same callers hold
7.09 GB/s flat at every k from 2 to 16. Both models, byte-identical curves.

The ring pool is what actually throttles device concurrency (a caller queues on
a ring's Mutex), so sizing it to the caller pool over-drives the NVMe past its
read-concurrency knee. These tests pin the sizing rule.
"""
import pytest

from vaayu.persist import safetensor_model_persister as P


def test_ring_pool_is_sized_to_the_drive_not_the_caller_pool(monkeypatch):
    """8 pinned buffers must NOT mean 8 rings — that is the P8 ceiling."""
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 8)
    monkeypatch.delenv("VAAYU_RING_POOL_N", raising=False)
    assert P.get_expert_pool_size() == 8
    assert P._ring_pool_size() == P._NVME_READ_CONCURRENCY
    assert P._ring_pool_size() < 8


def test_ring_pool_never_exceeds_the_caller_pool(monkeypatch):
    """A ring no caller can reach is dead weight; on a RAM-tight box the buffer
    pool falls to 1-2 and the ring pool must follow it down."""
    monkeypatch.delenv("VAAYU_RING_POOL_N", raising=False)
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 1)
    assert P._ring_pool_size() == 1
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 2)
    assert P._ring_pool_size() == 2


def test_env_override_selects_the_ring_count(monkeypatch):
    """`VAAYU_RING_POOL_N` is the A/B knob this row was measured with; it must
    keep working, including for the pre-fix value it is compared against."""
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 8)
    monkeypatch.setenv("VAAYU_RING_POOL_N", "8")
    assert P._ring_pool_size() == 8
    monkeypatch.setenv("VAAYU_RING_POOL_N", "2")
    assert P._ring_pool_size() == 2


def test_bad_env_override_falls_back_to_the_measured_default(monkeypatch):
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 8)
    for bad in ("", "banana", "0", "-3"):
        monkeypatch.setenv("VAAYU_RING_POOL_N", bad)
        assert P._ring_pool_size() == P._NVME_READ_CONCURRENCY


def test_init_model_io_builds_the_ring_pool_at_the_drive_size(monkeypatch, tmp_path):
    """The regression itself: `init_model_io` used to hand the ring builder
    `get_expert_pool_size()`. This asserts on the value it actually passes."""
    seen = {}
    monkeypatch.setattr(P, "_USE_IOURING", True)
    monkeypatch.setattr(P, "_RING_INITIALIZED", False)
    monkeypatch.setattr(P, "_FILES_REGISTERED", True)   # skip file registration
    monkeypatch.setattr(P, "_EXPERT_POOL_ACTUAL_N", 8)
    monkeypatch.delenv("VAAYU_RING_POOL_N", raising=False)
    monkeypatch.setattr(P, "_init_iouring_ring",
                        lambda n, qd=32: seen.update(n=n, qd=qd) or True)
    # Pool already populated → init_model_io skips buffer allocation entirely.
    monkeypatch.setattr(P, "_USE_PINNED", False)

    P.init_model_io(str(tmp_path), use_direct_io=True)

    assert seen["n"] == P._NVME_READ_CONCURRENCY, (
        f"ring pool built with n={seen['n']}, the caller-pool size, not the "
        f"drive's concurrency knee"
    )
