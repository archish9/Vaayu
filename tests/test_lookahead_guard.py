"""Hard-block guard for VAAYU_LOOKAHEAD=1 (Task 5 follow-up).

VAAYU_LOOKAHEAD=1 causes confirmed on-device memory corruption (device-side
assert / NaN-Inf routing probabilities) in the lookahead prefetch path after
two independent race-fix rounds. check_lookahead_flag_safe() must refuse to
proceed rather than silently corrupt memory, and must be a no-op when the
flag is unset (the default path).
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
