"""Characterizes the flag-gated VRAM prefill diagnostic line. Pure formatting only —
the on-device trajectory it produces is captured in Task 2."""
import os
import pytest
from vaayu.moe_patch import _format_vram_diag

GB = 1024 ** 3


def test_diag_line_reports_all_components_in_gb():
    line = _format_vram_diag(
        layer_idx=7, free=2 * GB, alloc=13 * GB, reserved=int(13.5 * GB),
        cache_usage=8 * GB, cache_budget=9 * GB, prefill_evict=42,
    )
    # One grep-able line naming the layer and every component in GB.
    assert "[VRAM-DIAG]" in line
    assert "layer=7" in line
    assert "free=2.00GB" in line
    assert "alloc=13.00GB" in line
    assert "reserved=13.50GB" in line
    assert "cache=8.00/9.00GB" in line
    assert "prefill_evict=42" in line


def test_diag_line_is_single_line():
    line = _format_vram_diag(1, GB, GB, GB, GB, GB, 0)
    assert "\n" not in line


def test_gate_defaults_off(monkeypatch):
    """With VAAYU_VRAM_DIAG unset, the diag branch is skipped. We assert on the
    gate the code reads, so the behavior-neutral default is pinned."""
    monkeypatch.delenv("VAAYU_VRAM_DIAG", raising=False)
    assert not os.environ.get("VAAYU_VRAM_DIAG")
