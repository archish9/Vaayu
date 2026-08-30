"""Hard-block guard for VAAYU_LOOKAHEAD=1.

**The reason has changed TWICE on 2026-08-30; this docstring tracks the third
and current one.** The guard originally existed because the prefetch path
corrupted memory. That was root-caused and fixed (ExpertRAMCache.put_slab
released a DECLINED slab the caller was still reading — see
tests/test_slab_arena.py::test_declined_slab_is_not_handed_to_the_next_acquirer).
It then stood on performance, because the flag cost 15% of decode. An in-flight
load reservation removed the duplicate stager/decode reads behind that cost and
the flag became a measured **+11.4%** on Qwen3-235B.

It still refuses, now purely on COVERAGE: the predictor is wired only into the
fused-EXL3 forward, so it is a silent no-op on the module-list forward
(MiniMax), and it has never been measured under parallel fan-out. The
assertions below are unchanged across all three revisions — refuse when set,
no-op when unset — because what the guard DOES has never been the thing in
question. What changes is why, and a reader who comes here for the reason must
get the current one.
"""
import os
import pytest
from vaayu.vaayu_base import check_lookahead_flag_safe


def test_raises_when_flag_is_set(monkeypatch):
    monkeypatch.setenv("VAAYU_LOOKAHEAD", "1")
    with pytest.raises(RuntimeError, match="VAAYU_LOOKAHEAD"):
        check_lookahead_flag_safe()


def test_noop_when_flag_unset(monkeypatch):
    monkeypatch.delenv("VAAYU_LOOKAHEAD", raising=False)
    check_lookahead_flag_safe()  # must not raise


def test_noop_when_flag_set_to_other_value(monkeypatch):
    monkeypatch.setenv("VAAYU_LOOKAHEAD", "0")
    check_lookahead_flag_safe()  # must not raise — only "1" is blocked
