"""RAM accounting snapshot — categorized /proc/self/smaps_rollup reader."""
import pytest
from vaayu.vaayu_base import ram_snapshot


def test_ram_snapshot_has_all_keys():
    snap = ram_snapshot()
    for key in ("rss", "pss", "shared_clean", "shared_dirty",
                "private_clean", "private_dirty", "swap", "available", "total"):
        assert key in snap, f"missing {key}"
        assert isinstance(snap[key], int)
        assert snap[key] >= 0


def test_ram_snapshot_rss_is_plausible():
    """RSS must be non-zero for a live process and not exceed total RAM."""
    snap = ram_snapshot()
    assert snap["rss"] > 0
    assert snap["total"] > 0
    assert snap["rss"] <= snap["total"]


def test_ram_snapshot_private_sums_under_rss():
    """private_clean + private_dirty cannot exceed RSS (they are subsets)."""
    snap = ram_snapshot()
    assert snap["private_clean"] + snap["private_dirty"] <= snap["rss"] * 1.05


def test_ram_snapshot_never_raises(monkeypatch):
    """A kernel without smaps_rollup must degrade to zeros, not crash."""
    import builtins
    real_open = builtins.open

    def _boom(path, *a, **kw):
        if "smaps_rollup" in str(path):
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _boom)
    snap = ram_snapshot()          # must not raise
    assert snap["pss"] == 0        # smaps-derived fields degrade to 0
    assert snap["total"] > 0       # psutil-derived fields still populated
