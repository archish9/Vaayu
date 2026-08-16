"""The RAM layer cache budget must be reproducible, and overridable.

Before this, the budget was `psutil.virtual_memory().available - 8 GB` sampled
once at load, so throughput depended on whatever else happened to be running at
boot. Measured on the reference box, two runs of an identical command against
Llama-3.3-70B: 9.7 GB / 18 layers cached / 0.167 tok/s versus 16.4 GB / 33 layers
/ 0.230 tok/s -- a 38 % decode swing decided by luck.

GB here is binary (GiB), matching the implementation's hardware constants.
"""
import pytest

from vaayu.persist.ram_cache import (RAM_CACHE_RESERVE, auto_detect_cache_budget,
                                     describe_cache_budget)

GB = 1024 ** 3


def test_the_budget_is_derived_from_total_ram_not_from_what_is_free_right_now():
    # Same box, two different moments. Same answer -- that is the whole point.
    busy = auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=30 * GB)
    idle = auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=28 * GB)
    assert busy == idle == 32 * GB - RAM_CACHE_RESERVE


def test_the_reserve_is_held_back_from_total():
    assert auto_detect_cache_budget(total_bytes=32 * GB,
                                    available_bytes=32 * GB) == 32 * GB - RAM_CACHE_RESERVE


def test_it_never_claims_more_ram_than_currently_exists():
    # Deterministic is not the same as reckless: RAM that is genuinely occupied
    # cannot be cached into, so the live figure still clamps.
    b = auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=6 * GB)
    assert b < 32 * GB - RAM_CACHE_RESERVE
    assert b <= 6 * GB


def test_a_nearly_full_machine_yields_zero_rather_than_a_negative_budget():
    assert auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=0) == 0


def test_a_small_machine_yields_zero_rather_than_a_negative_budget():
    assert auto_detect_cache_budget(total_bytes=8 * GB, available_bytes=8 * GB) == 0


def test_an_explicit_override_wins_over_both_bounds():
    assert auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=30 * GB,
                                    override_bytes=4 * GB) == 4 * GB


def test_an_explicit_override_of_zero_disables_the_cache():
    # Distinct from None, which means "derive it".
    assert auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=30 * GB,
                                    override_bytes=0) == 0


def test_an_explicit_override_is_still_clamped_to_what_exists():
    # Asking for more RAM than the machine has must not be honoured silently.
    assert auto_detect_cache_budget(total_bytes=32 * GB, available_bytes=5 * GB,
                                    override_bytes=100 * GB) <= 5 * GB


@pytest.mark.parametrize("total,available,expect", [
    (32 * GB, 30 * GB, "total"),        # the deterministic bound applies
    (32 * GB, 6 * GB, "available"),     # the live bound applies
    (32 * GB, 30 * GB, "total"),
])
def test_the_binding_constraint_is_named_so_degradation_is_visible(total, available, expect):
    # CLAUDE.md §5: warn, never silently degrade. A budget that quietly collapsed
    # because the box was busy is exactly the failure this row exists to remove,
    # so which bound applied has to be printable.
    msg = describe_cache_budget(total_bytes=total, available_bytes=available)
    assert expect in msg


def test_the_description_names_an_override_when_one_is_given():
    msg = describe_cache_budget(total_bytes=32 * GB, available_bytes=30 * GB,
                                override_bytes=4 * GB)
    assert "override" in msg


def test_the_reserve_is_documented_as_a_hardware_constant_not_a_model_number():
    # CLAUDE.md §5 permits hardware literals only, and requires them commented.
    import inspect

    import vaayu.persist.ram_cache as rc
    src = inspect.getsource(rc)
    i = src.index("RAM_CACHE_RESERVE")
    window = src[max(0, i - 600):i + 200]
    assert "hardware" in window.lower()


def test_calling_with_no_arguments_still_works_against_the_live_machine():
    # The zero-argument form is what vaayu_base uses; it must not require psutil
    # values to be threaded through.
    b = auto_detect_cache_budget()
    assert isinstance(b, int) and b >= 0
