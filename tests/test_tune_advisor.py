"""Regression guard for `tune_advisor.py`'s recommendation arithmetic.

The advisor had no tests, and a units bug lived in it undetected: it instructs
you to paste the pool budget off the engine's startup line, which prints
`_pc_budget / 1e9` (DECIMAL GB), then multiplied by `1024**3`. That inflated the
pasted figure by 7.4% — enough to cross a capacity boundary. At MiniMax's
measured 11.49 GB pool with 6 concurrent 7 751-token sessions it reported room
for 6 and recommended `prefix_compress=None`, where only 5 fit. Overshooting pool
capacity costs 100% of the reuse rather than a share of it (todo.md "Settled"),
so that recommendation would have cost the full 9.8s -> 352s cliff.

The bug was found by checking the advisor's output against Plan J Task 4's
on-device measurements — something that had never been done. These tests make
that check permanent: the four measured (K, pool, outcome) points are asserted
against what the advisor recommends for them.

Importing tune_advisor is cheap and side-effect-free: module scope only reads
env vars and extends sys.path. It loads no model and no weights.
"""
import pytest

import tune_advisor as ta


# ── Geometry fixtures: real models, derived per-session sizes (CLAUDE.md §5) ──
#
# TEST FIXTURES describing on-device models, not production constants. Per-session
# bytes are DERIVED from geometry here rather than pasted, so the numbers stay
# auditable: layers x 2 (K and V) x kv_heads x head_dim x bytes/elem x tokens.

def _per_session_bf16(layers, kv_heads, head_dim, tokens):
    return layers * 2 * kv_heads * head_dim * 2.0 * tokens


def _per_session_comp(layers, kv_heads, head_dim, tokens, bits):
    # TurboQuant pooled layout, mirroring the advisor's comp_rate: (bits-1) index
    # bits + 1 QJL sign bit per channel, plus two fp16 norms per (token, head).
    rate = bits / 8 + 4 / head_dim
    return layers * 2 * kv_heads * head_dim * rate * tokens


# MiniMax-M2.7: 62 layers, 8 KV heads, head_dim 128, at Plan J's 7 751-token shape.
_TOKENS = 7751
_MINIMAX = dict(layers=62, kv_heads=8, head_dim=128, tokens=_TOKENS)
_SESS_BF16 = _per_session_bf16(**_MINIMAX)              # ~1.968e9 bytes
_SESS_COMP9 = _per_session_comp(**_MINIMAX, bits=9)     # ~0.578x of the above


def test_gb_constant_matches_what_the_engine_prints():
    # vaayu_base.py prints the pool budget as `_pc_budget / 1e9`, and the advisor
    # tells the user to paste that number in. If this constant ever goes back to
    # 1024**3, every pasted budget is silently inflated 7.4%.
    assert ta.GB == 1e9


def test_derived_session_size_matches_the_measured_one():
    # Guards the fixtures themselves: Plan J measured 1.968 GB/session for this
    # geometry (and 2.03 GB of rss growth per session, which includes overhead).
    assert _SESS_BF16 == pytest.approx(1.968e9, rel=0.01)
    # 9-bit compression measured 0.55x from rss growth; the analytic rate is
    # 0.578x. Both are well inside the "roughly half" the advisor claims.
    assert _SESS_COMP9 / _SESS_BF16 == pytest.approx(0.578, rel=0.01)


# ── The four on-device points from Plan J Task 4 ──────────────────────────────
#
# (K, pool budget in DECIMAL GB as the engine printed it, what actually happened)
# Uncompressed runs, MiniMax, 7 751 tokens/session, R_ROUNDS=2, gate = final-round
# HIT count. Evidence: ~/vaayu-evidence/2026-08-03-pinned-roundup/multiturn-K*.log

@pytest.mark.parametrize("k, pool_gb", [
    (3, 13.76),   # 3/3 HIT, exit 0
    (4, 13.70),   # 4/4 HIT, exit 0
    (6, 12.16),   # 6/6 HIT, exit 0
])
def test_advisor_says_uncompressed_where_the_run_passed_uncompressed(k, pool_gb):
    rec = ta.recommend_compress_bits(pool_gb * ta.GB, _SESS_BF16, _SESS_COMP9, k, 9)
    assert rec is None


def test_advisor_recommends_compression_where_the_run_failed():
    # K=7 at an 11.49 GB pool: 0/7 HIT uncompressed, exit 1, round-2 TTFT stayed
    # at the cold 352 s. The same shape at prefix_compress=9 scored 7/7 in 10.4 s.
    rec = ta.recommend_compress_bits(11.49 * ta.GB, _SESS_BF16, _SESS_COMP9, 7, 9)
    assert rec == 9


def test_capacity_at_the_failing_pool_is_five_not_six():
    # The exact figure the units bug got wrong. 11.49 GB / 1.968 GB = 5.84 -> 5.
    # Read as GiB the same number is 12.34e9 bytes -> 6.27 -> 6, which is what the
    # advisor used to report, and 6 sessions at that pool thrash.
    assert ta.pool_capacity(11.49 * ta.GB, _SESS_BF16) == 5
    assert ta.pool_capacity(11.49 * 1024 ** 3, _SESS_BF16) == 6   # the old, wrong answer


@pytest.mark.parametrize("pool_gb, expected", [
    (13.76, 6),   # the largest pool measured
    (12.16, 6),   # K=6 passed here, with 0.35 GB to spare
    (11.49, 5),   # K=7 failed here
    (4.96, 2),    # Qwen's pool, unchanged by Plan J
])
def test_capacity_tracks_the_pool_across_every_measured_budget(pool_gb, expected):
    # The pool budget is not a constant of the box -- it is `free RAM at load -
    # reserve`, and it moved 2.3 GB across five loads of the SAME model on the
    # SAME day. Capacity must follow it rather than being a remembered integer.
    assert ta.pool_capacity(pool_gb * ta.GB, _SESS_BF16) == expected


def test_compression_roughly_doubles_capacity():
    # The advisor's headline claim for prefix_compress=9. Measured: K=7 fitted
    # compressed at a pool that held 5 uncompressed.
    pool = 11.49 * ta.GB
    assert ta.pool_capacity(pool, _SESS_COMP9) >= 2 * ta.pool_capacity(pool, _SESS_BF16)
    assert ta.pool_capacity(pool, _SESS_COMP9) >= 7    # the K that failed uncompressed


# ── Branch behaviour ─────────────────────────────────────────────────────────

def test_single_session_never_compresses_however_small_the_pool():
    # The eviction loop keeps at least one entry, so a lone session is cached
    # regardless of budget and compression would be pure cost.
    assert ta.recommend_compress_bits(0.0, _SESS_BF16, _SESS_COMP9, 1, 9) is None
    assert ta.recommend_compress_bits(1.0, _SESS_BF16, _SESS_COMP9, 0, 9) is None


def test_still_recommends_compression_when_even_compressed_will_not_fit():
    # Compression widens the range even when it cannot close the gap; the caller
    # reports the shortfall separately. Recommending None here would be worse.
    assert ta.recommend_compress_bits(_SESS_BF16, _SESS_BF16, _SESS_COMP9, 50, 9) == 9


def test_recommendation_honours_the_configured_width():
    # COMPRESS_BITS is user-settable; the recommendation must echo it, not 9.
    assert ta.recommend_compress_bits(11.49 * ta.GB, _SESS_BF16, _SESS_COMP9, 7, 8) == 8


def test_capacity_of_a_zero_or_negative_pool_is_zero():
    assert ta.pool_capacity(0, _SESS_BF16) == 0
    assert ta.pool_capacity(-1, _SESS_BF16) == 0
    assert ta.pool_capacity(_SESS_BF16, 0) == 0
