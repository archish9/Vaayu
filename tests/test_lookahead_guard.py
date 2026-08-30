"""Hard-block guard for VAAYU_LOOKAHEAD=1.

**The reason changed on 2026-08-30 and this docstring changed with it.** The
guard used to exist because the prefetch path corrupted memory (device-side
assert / NaN-Inf routing probabilities). That was root-caused and fixed:
ExpertRAMCache.put_slab released a DECLINED slab while the caller still held
views into it and was about to DMA from them. See
check_lookahead_flag_safe.__doc__ and tests/test_slab_arena.py's
test_declined_slab_is_not_handed_to_the_next_acquirer.

The guard still refuses, on PERFORMANCE: with the corruption fixed the flag
runs clean and cuts decode's dead time 37.6% -> 15.1%, but reads 60% more bytes
(prefetched experts evict the on-path working set) and costs 15% of decode
throughput on Qwen3-235B. So the assertions below are unchanged -- it must
still refuse, and still be a no-op when the flag is unset -- but a reader
coming here for the reason must get the true one.
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
