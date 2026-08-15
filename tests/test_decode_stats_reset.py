"""F13: the decode-stats reset latched for the PROCESS, not the turn.

`_decode_stats_reset` was set True on the first decode pass and never cleared,
so the reset that exists to exclude prefill's ~100%-miss pollution ran once per
model lifetime. From turn 2 on, the cumulative expert-cache hit rate was diluted
by every new prefill -- telemetry only, but telemetry is how every cache
decision on this branch got made.
"""
from vaayu.vaayu_base import decode_stats_reset


def test_the_first_decode_pass_of_a_turn_resets_and_latches():
    reset_now, latched = decode_stats_reset(
        is_moe=True, has_expert_cache=True, is_prefill_pass=False,
        already_reset=False)
    assert reset_now is True
    assert latched is True


def test_later_decode_passes_in_the_same_turn_do_not_reset_again():
    reset_now, latched = decode_stats_reset(
        is_moe=True, has_expert_cache=True, is_prefill_pass=False,
        already_reset=True)
    assert reset_now is False
    assert latched is True


def test_a_new_prefill_re_arms_the_reset():
    """THE bug. Turn 2's prefill pollutes the counters exactly as turn 1's did,
    so the latch must drop when a prefill runs -- otherwise the next decode
    never clears it."""
    reset_now, latched = decode_stats_reset(
        is_moe=True, has_expert_cache=True, is_prefill_pass=True,
        already_reset=True)
    assert reset_now is False, "a prefill pass never resets -- it is the polluter"
    assert latched is False, "but it must re-arm the next decode's reset"


def test_turn_two_resets_again_after_that_re_arm():
    _, latched = decode_stats_reset(is_moe=True, has_expert_cache=True,
                                    is_prefill_pass=True, already_reset=True)
    reset_now, latched = decode_stats_reset(
        is_moe=True, has_expert_cache=True, is_prefill_pass=False,
        already_reset=latched)
    assert reset_now is True and latched is True


def test_nothing_resets_without_a_moe_expert_cache():
    for kwargs in ({"is_moe": False, "has_expert_cache": True},
                   {"is_moe": True, "has_expert_cache": False}):
        reset_now, latched = decode_stats_reset(
            is_prefill_pass=False, already_reset=False, **kwargs)
        assert reset_now is False
        assert latched is False, "no reset happened, so nothing to latch"
