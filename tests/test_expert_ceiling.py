"""The expert-cache ceiling must be DERIVED, and must not starve the prefix pool.

`_EXPERT_CACHE_OOM_CEILING = 8.5 GiB` was set 2026-07-10, when the pinned
allocator charged the next power of two for every request (a 9.13 GB arena cost
16 GiB). The chunked-arena fix landed 2026-08-07 and charges exactly; the ceiling
was never revisited. Measured on Qwen3-235B, 2 sessions x 3 rounds, real context:

  ceiling 8.5 GiB -> expert  9.12 GB, prefix 11.73 GB, warm wall 36.6 s
  ceiling 13  GiB -> expert 13.95 GB, prefix  7.45 GB, warm wall 32.0 s
  ceiling 18  GiB -> expert 19.33 GB, prefix  2.00 GB, warm wall 27.3 s

Prefix reuse held at 83 % in every arm — but only because those sessions were
~4.6k tokens (~0.90 GB each). At a full 8192-token context they are 1.58 GB each,
so two would need 3.16 GB against that 2.00 GB pool and the pool would thrash.
Reuse collapses to ZERO, not proportionally, one session past capacity (Plan I).
Hence: reserve for the sessions we intend to guarantee, then give the expert
cache what is left.
"""
import pytest

from vaayu.vaayu_base import derive_expert_ceiling

GB = 1024 ** 3
QWEN_KV_PER_TOK = 193_000        # decimal bytes/token, bf16 (todo checkpoints table)
MINIMAX_KV_PER_TOK = 254_000


def _ceiling(**over):
    kw = dict(available_bytes=24 * GB, prefix_sessions=3,
              kv_bytes_per_token=QWEN_KV_PER_TOK, max_seq_len=8192,
              prefix_enabled=True)
    kw.update(over)
    return derive_expert_ceiling(**kw)


def test_the_ceiling_grows_with_available_ram():
    small = _ceiling(available_bytes=16 * GB)
    big = _ceiling(available_bytes=28 * GB)
    assert big > small


def test_room_is_reserved_for_the_guaranteed_sessions():
    # Each guaranteed session costs max_seq_len * kv_bytes_per_token.
    one = _ceiling(prefix_sessions=1)
    three = _ceiling(prefix_sessions=3)
    per_session = 8192 * QWEN_KV_PER_TOK
    assert one - three == pytest.approx(2 * per_session, rel=0.01)


def test_a_disabled_prefix_cache_reserves_nothing_for_it():
    off = _ceiling(prefix_enabled=False)
    on = _ceiling(prefix_enabled=True)
    assert off > on
    assert off - on == pytest.approx(3 * 8192 * QWEN_KV_PER_TOK, rel=0.01)


def test_a_longer_context_reserves_more_and_leaves_less():
    short = _ceiling(max_seq_len=4096)
    long = _ceiling(max_seq_len=8192)
    assert short > long


@pytest.mark.parametrize("kv", [QWEN_KV_PER_TOK, MINIMAX_KV_PER_TOK])
def test_both_on_device_geometries_derive_a_usable_ceiling(kv):
    # CLAUDE.md §5: never prove a sizing helper on one model.
    c = _ceiling(kv_bytes_per_token=kv)
    assert 4 * GB <= c <= 24 * GB, c


def test_the_result_never_goes_below_the_historical_floor():
    # A tiny box must still get a working cache rather than a nonsensical one:
    # the shipped 8.5 GiB was known to run, so it is the floor, not the cap.
    tiny = _ceiling(available_bytes=6 * GB, prefix_sessions=8)
    assert tiny == int(8.5 * GB)


def test_it_never_exceeds_available_ram():
    c = _ceiling(available_bytes=10 * GB, prefix_sessions=0, prefix_enabled=False)
    assert c <= 10 * GB


def test_zero_sessions_is_allowed_and_reserves_nothing():
    assert _ceiling(prefix_sessions=0) == _ceiling(prefix_enabled=False)


def test_the_shipped_default_would_now_exceed_the_stale_literal_on_this_box():
    # The whole point: on a 32 GB box with 3 guaranteed sessions, the derived
    # ceiling must be materially larger than 8.5 GiB, or nothing changed.
    c = _ceiling(available_bytes=24 * GB, prefix_sessions=3)
    assert c > int(10 * GB), f"derived {c / GB:.2f} GiB is not better than the stale literal"


def test_compute_expert_budget_accepts_a_derived_ceiling():
    from vaayu.vaayu_base import compute_expert_budget

    budget_default, _, _ = compute_expert_budget(
        available_bytes=24 * GB, working_set_bytes=8 * GB, num_moe_layers=94,
        expert_bytes=11_830_000, moe_top_k=8, moe_num_experts=128)
    budget_raised, _, mpl_raised = compute_expert_budget(
        available_bytes=24 * GB, working_set_bytes=8 * GB, num_moe_layers=94,
        expert_bytes=11_830_000, moe_top_k=8, moe_num_experts=128,
        ceiling_bytes=16 * GB)
    assert budget_raised > budget_default
    # A bigger budget must widen the per-layer cap, which is the lever that
    # actually moved the measured hit rate (8 -> 17 slots/layer, 44.9 -> 61.3 %).
    assert mpl_raised > 8


def test_omitting_the_ceiling_keeps_the_historical_behaviour():
    from vaayu.vaayu_base import _EXPERT_CACHE_OOM_CEILING, compute_expert_budget

    budget, _, _ = compute_expert_budget(
        available_bytes=64 * GB, working_set_bytes=8 * GB, num_moe_layers=94,
        expert_bytes=11_830_000, moe_top_k=8, moe_num_experts=128)
    assert budget == _EXPERT_CACHE_OOM_CEILING
