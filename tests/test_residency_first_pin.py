"""Residency-first pin decision (S42, builds S40's lever).

Pure, no CUDA / no model load (same discipline as test_vram_plan.py). The
helper decides whether phase 2 of `_init_pinned_layers` should pin ONE MORE
non-expert transformer layer into VRAM, trading expert-VRAM-tier capacity for
guaranteed-every-token non-expert residency.

The rule (S40 measured +15.18 % on DeepSeek by pinning its 3 streamed
non-expert layers at the tier's expense):
  * ALWAYS pin while the original plan budget covers it (fits_b0) -- today's
    behaviour, byte-for-byte.
  * Keep pinning PAST the budget only while the expert tier is already BELOW
    one routed working set (thrashing -- its marginal experts re-stream anyway)
    AND the pin would not breach the reserve + a small VRAM safety margin.

Model-agnostic (CLAUDE.md §5): the caller derives free_vram, reserve,
working_set_bytes and the per-layer size; this helper hardcodes no model
number. Contrasting regimes are parametrized below (thrashing tier vs healthy
tier), per the §5 two-geometry mandate.
"""
from vaayu.vram_plan import residency_first_allows_pin

GB = 1024 ** 3
MARGIN = 0.20 * GB  # the oracle's margin (ran bit-identical, no OOM)


def _allows(*, used, size, budget, free_vram, reserve, working_set):
    return residency_first_allows_pin(
        used=int(used), size=int(size), budget=int(budget),
        free_vram=int(free_vram), reserve=int(reserve),
        working_set_bytes=int(working_set), margin=int(MARGIN))


def test_pins_while_the_original_budget_covers_it():
    # fits_b0 path: the layer fits under today's pinning_budget -> always pin,
    # regardless of tier health. (Qwen3-Next: non-expert layers all fit -> no-op.)
    assert _allows(used=5 * GB, size=0.167 * GB, budget=8 * GB,
                   free_vram=13 * GB, reserve=3.5 * GB, working_set=100 * GB)


def test_extends_past_budget_when_the_tier_is_thrashing():
    # DeepSeek-like: budget exhausted (used==budget), but the expert tier
    # (free-used-reserve = 1.5 GB) is below one 3.4 GB working set -> keep pinning.
    assert _allows(used=8 * GB, size=0.167 * GB, budget=8 * GB,
                   free_vram=13 * GB, reserve=3.5 * GB, working_set=3.4 * GB)


def test_stops_at_budget_when_the_tier_is_healthy():
    # A tier at/above one working set (1.0 GB here, tier=1.5 GB) is a real cache;
    # shrinking it below the working set would fall off the LRU cliff -> stop.
    assert not _allows(used=8 * GB, size=0.167 * GB, budget=8 * GB,
                       free_vram=13 * GB, reserve=3.5 * GB, working_set=1.0 * GB)


def test_stops_before_breaching_the_reserve_margin():
    # Thrashing tier, but pinning this layer would leave < margin above reserve.
    assert not _allows(used=9.3 * GB, size=0.167 * GB, budget=9.3 * GB,
                       free_vram=13 * GB, reserve=3.5 * GB, working_set=3.4 * GB)


def test_tier_exactly_at_the_working_set_is_not_thrashing():
    # Boundary: tier_now == working_set is NOT below it -> do not extend.
    assert not _allows(used=8 * GB, size=0.167 * GB, budget=8 * GB,
                       free_vram=13 * GB, reserve=3.5 * GB, working_set=1.5 * GB)


def test_last_thrashing_pin_exactly_at_the_margin_is_allowed():
    # free-(used+size)-reserve == margin -> fits_vram is inclusive -> pin.
    free, reserve, size = 13 * GB, 3.5 * GB, 0.167 * GB
    used = free - reserve - MARGIN - size  # so the post-pin headroom == margin
    assert _allows(used=used, size=size, budget=used,  # budget exhausted
                   free_vram=free, reserve=reserve, working_set=3.4 * GB)
