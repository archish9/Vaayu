"""Pure VRAM-planning tests. No CUDA, no model load -- that is the point
(METHOD NOTES: two Plan L defects lived in code 'verified' by a harness that
copied the logic instead of executing it)."""
import pytest

from vaayu.vram_plan import (
    VramPlanError,
    kv_bytes_per_token,
    kv_capacity_tokens,
    kv_elem_bytes,
)

MB = 1024 ** 2
GB = 1024 ** 3
# Measured per-token KV figures (e.g. Qwen's 0.193 MB/tok) were recorded in
# decimal units (1e6 / 1e9), while the hardware constants used elsewhere in
# this file (usable VRAM, reserve floor, expert size) are binary (1024-based).
# Use *_DEC only when comparing against a measured MB/GB-per-token figure --
# do not "fix" one family into the other.
MB_DEC = 10 ** 6
GB_DEC = 10 ** 9

# Four real on-device geometries, read off their config.json. CLAUDE.md §5:
# a fix that only proves out on Qwen is not done.
#
# `kv_capacity_buckets` records each checkpoint's REAL cache format (Task 5,
# task-5-report.md): kv_cache_mode="auto" resolves to FP8DynamicCache (a
# geometric-doubling ring buffer) on every EXL3 checkpoint, and to a plain
# transformers.DynamicCache (no over-allocation) otherwise. Three of these
# four on-device geometries are EXL3; llama-1b is the non-EXL3 format
# control and must NOT be bucketed, or the cliff-vs-no-cliff contrast this
# module exists to model collapses.
GEOMETRIES = {
    "qwen3-235b":   dict(num_hidden_layers=94, num_key_value_heads=4,
                         head_dim=128, hidden=4096, n_heads=64, vocab=151936,
                         kv_capacity_buckets=True),
    "minimax-m2.7": dict(num_hidden_layers=62, num_key_value_heads=8,
                         head_dim=128, hidden=3072, n_heads=48, vocab=200064,
                         kv_capacity_buckets=True),
    "llama-70b":    dict(num_hidden_layers=80, num_key_value_heads=8,
                         head_dim=128, hidden=8192, n_heads=64, vocab=128256,
                         kv_capacity_buckets=True),
    "llama-1b":     dict(num_hidden_layers=16, num_key_value_heads=8,
                         head_dim=64, hidden=2048, n_heads=32, vocab=128256,
                         kv_capacity_buckets=False),
}


def _kv_kwargs(geo):
    return dict(num_hidden_layers=geo["num_hidden_layers"],
                num_key_value_heads=geo["num_key_value_heads"],
                head_dim=geo["head_dim"])


def test_qwen_kv_per_token_matches_the_measured_figure():
    """todo.md records 0.193 MB/token for Qwen at bf16. If this drifts, the
    whole plan is calibrated against nothing."""
    b = kv_bytes_per_token(**_kv_kwargs(GEOMETRIES["qwen3-235b"]), kv_cache_mode="bf16")
    assert b / MB_DEC == pytest.approx(0.193, abs=0.001)


def test_auto_and_bf16_and_fp16_are_all_two_bytes_per_element():
    kw = _kv_kwargs(GEOMETRIES["minimax-m2.7"])
    assert (kv_bytes_per_token(**kw, kv_cache_mode="auto")
            == kv_bytes_per_token(**kw, kv_cache_mode="bf16")
            == kv_bytes_per_token(**kw, kv_cache_mode="fp16"))


def test_fp8_charges_the_measured_rate_not_one_byte():
    """Plan E measured fp8 mode at 3.016 B/elem (fp8 store PLUS the dequantized
    mirror attention reads), against bf16's 2.000. Charging 1 byte under-reserves
    by 3x."""
    kw = _kv_kwargs(GEOMETRIES["qwen3-235b"])
    assert (kv_bytes_per_token(**kw, kv_cache_mode="fp8")
            / kv_bytes_per_token(**kw, kv_cache_mode="bf16")) == pytest.approx(1.508, abs=0.01)


def test_turboquant_rate_is_derived_from_bits_and_head_dim():
    # bits/8 bytes of indices per element, plus 4 bytes per (token, head) of
    # norms, i.e. 4/head_dim per element.
    assert kv_elem_bytes("turboquant:9", head_dim=128) == pytest.approx(9 / 8 + 4 / 128)
    assert kv_elem_bytes("turboquant", head_dim=128) == pytest.approx(4 / 8 + 4 / 128)


def test_unknown_mode_raises_instead_of_falling_through_to_fp8():
    """CLAUDE.md §5 forbids silent degradation. A typo in kv_cache_mode must not
    quietly resolve to some other cache."""
    with pytest.raises(VramPlanError) as e:
        kv_elem_bytes("bf16_typo", head_dim=128)
    assert "bf16_typo" in str(e.value)


from vaayu.vram_plan import (
    DEFAULT_ALLOC_FACTOR,
    SCORE_ELEM_BYTES,
    VramShape,
    _kv_term_bytes,
    _peak_intercept_bytes,
    _score_term_bytes,
    predict_peak_vram,
    score_elem_bytes,
)


def shape_for(name, *, N=1, chunk=512, kv_cache_mode="bf16",
              score_elem_bytes=0.0, alloc_factor=DEFAULT_ALLOC_FACTOR,
              kv_growth_factor=2, kv_capacity_buckets=None, head_bytes=0):
    geo = GEOMETRIES[name]
    if kv_capacity_buckets is None:
        kv_capacity_buckets = geo["kv_capacity_buckets"]
    return VramShape(
        kv_per_token=kv_bytes_per_token(**_kv_kwargs(geo), kv_cache_mode=kv_cache_mode),
        N=N, chunk=chunk, hidden=geo["hidden"], n_heads=geo["n_heads"],
        vocab=geo["vocab"], elem_bytes=2,
        score_elem_bytes=score_elem_bytes, alloc_factor=alloc_factor,
        kv_growth_factor=kv_growth_factor, kv_capacity_buckets=kv_capacity_buckets,
        head_bytes=head_bytes,
    )


# --- kv_capacity_tokens -----------------------------------------------
#
# Task 5 (task-5-report.md) measured that kv_cache_mode="auto" resolves to
# FP8DynamicCache(quantize=False) on every EXL3 checkpoint, and its
# _ensure_capacity (vaayu_base.py) grows the KV ring buffer by GEOMETRIC
# DOUBLING (`new_cap = max(needed, max(1, cap) * growth)`), not per token.
# The naive S-linear KV model failed the ±10% accuracy gate by up to +41.2%
# because of this missing step term (13 of 17 measured points off by more
# than 10%). kv_capacity_tokens reproduces that growth in closed form so
# predict_peak_vram can charge the KV term against what is actually
# ALLOCATED, not the raw context length.

def test_kv_capacity_tokens_no_bucketing_returns_S_unchanged():
    """buckets=False models a plain transformers.DynamicCache (every
    non-EXL3 checkpoint) which holds exactly S tokens, no over-allocation."""
    for s in (1, 2, 511, 512, 513, 5000, 8192, 8193, 100_000):
        assert kv_capacity_tokens(s, chunk=512, growth=2, buckets=False) == s


def test_kv_capacity_tokens_sub_chunk_boundary():
    """S <= chunk means the WHOLE context is the first (and only)
    _ensure_capacity call, whose `needed` is the context's own length, not a
    full chunk: new_cap = max(needed, max(1, 0) * growth) = max(S, growth).
    So capacity for a sub-chunk context tracks S once S >= growth, and only
    floors at `growth` for very small S -- it is never a full `chunk`."""
    assert kv_capacity_tokens(1, chunk=512, growth=2, buckets=True) == 2
    assert kv_capacity_tokens(2, chunk=512, growth=2, buckets=True) == 2
    assert kv_capacity_tokens(3, chunk=512, growth=2, buckets=True) == 3
    assert kv_capacity_tokens(100, chunk=512, growth=2, buckets=True) == 100
    assert kv_capacity_tokens(512, chunk=512, growth=2, buckets=True) == 512


def test_kv_capacity_tokens_matches_the_measured_doubling_table():
    """task-5-report.md's traced simulation, chunk=512 growth=2: capacity
    locks at 8192 for any S in [4609, 8192] and jumps to 16384 the instant S
    exceeds 8192 -- exactly the S=7700->8200 boundary the GPU probe hit."""
    for s in (4609, 5000, 6000, 7000, 7700, 8192):
        assert kv_capacity_tokens(s, chunk=512, growth=2, buckets=True) == 8192
    assert kv_capacity_tokens(8193, chunk=512, growth=2, buckets=True) == 16384
    assert kv_capacity_tokens(8704, chunk=512, growth=2, buckets=True) == 16384
    assert kv_capacity_tokens(16384, chunk=512, growth=2, buckets=True) == 16384
    assert kv_capacity_tokens(16385, chunk=512, growth=2, buckets=True) == 32768


def test_kv_capacity_tokens_is_not_hardcoded_to_growth_2_or_chunk_512():
    """CLAUDE.md §5: chunk and growth must be derived/passed, never literals
    baked into the function. Hand-simulated for chunk=100, growth=3:
    i=1 needed=100 cap=0->100; i=2 needed=200 cap=100->300;
    i=3 needed=300 cap=300 (no growth); i=4(partial) needed=350 cap=300->900."""
    assert kv_capacity_tokens(100, chunk=100, growth=3, buckets=True) == 100
    assert kv_capacity_tokens(200, chunk=100, growth=3, buckets=True) == 300
    assert kv_capacity_tokens(300, chunk=100, growth=3, buckets=True) == 300
    assert kv_capacity_tokens(350, chunk=100, growth=3, buckets=True) == 900


def test_kv_capacity_tokens_is_monotone_non_decreasing():
    """A step function, never a decrease -- the ring buffer never shrinks."""
    prev = 0
    for s in range(1, 9000, 137):
        cap = kv_capacity_tokens(s, chunk=512, growth=2, buckets=True)
        assert cap >= prev
        assert cap >= s  # capacity can never hold fewer tokens than S itself
        prev = cap


def test_kv_capacity_tokens_growth_below_2_raises():
    """growth=1 would mean max(needed, cap*1) == needed exactly whenever
    growth triggers -- an always-exact-fit cache this closed form does not
    model, and not a value FP8DynamicCache._GROWTH (a doubling factor) can
    take. Raise rather than loop or silently misreport."""
    with pytest.raises(VramPlanError):
        kv_capacity_tokens(1000, chunk=512, growth=1, buckets=True)


# --- predict_peak_vram: piecewise-linear, not linear ---------------------

@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_peak_is_linear_within_a_kv_capacity_bucket_and_steps_at_the_boundary(name):
    """This test used to assert predict_peak_vram was linear in S EVERYWHERE
    ("Every S-dependent term is first-order"). Task 5's GPU probe measured
    that this is false for the KV term on every EXL3 checkpoint: capacity is
    a step function (kv_capacity_tokens), so peak is linear only WITHIN a
    capacity bucket and jumps by a whole bucket's worth of KV at each
    boundary -- confirmed within 7% of the doubling prediction on three
    independent geometries (task-5-report.md). Do not "restore" the old
    everywhere-linear assertion; it asserts something the measurement
    refuted.
    """
    sh = shape_for(name)
    if sh.kv_capacity_buckets:
        # 5000, 6000, 7000 share one capacity bucket at chunk=512 -- that is
        # WHY equal steps in S must produce equal steps in peak here.
        cap5000 = kv_capacity_tokens(5000, chunk=sh.chunk, growth=sh.kv_growth_factor,
                                      buckets=True)
        cap6000 = kv_capacity_tokens(6000, chunk=sh.chunk, growth=sh.kv_growth_factor,
                                      buckets=True)
        cap7000 = kv_capacity_tokens(7000, chunk=sh.chunk, growth=sh.kv_growth_factor,
                                      buckets=True)
        assert cap5000 == cap6000 == cap7000
    # llama-1b (buckets=False) has no bucket to share -- its KV term is
    # kv_per_token * S * N directly, which is linear in S on its own, so the
    # equal-steps assertion below holds for it too, for a different reason.
    d1 = predict_peak_vram(sh, 6000) - predict_peak_vram(sh, 5000)
    d2 = predict_peak_vram(sh, 7000) - predict_peak_vram(sh, 6000)
    assert d1 == pytest.approx(d2, rel=1e-6)

    if sh.kv_capacity_buckets:
        # Crossing 8192->8193 (bucketed models only) adds a WHOLE bucket's
        # worth of KV capacity (kv_per_token * 8192 * N) in one token -- far
        # larger than the per-1000-token delta measured above.
        jump = predict_peak_vram(sh, 8193) - predict_peak_vram(sh, 8192)
        assert jump > d1


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_scores_term_is_zero_under_flash_and_charged_under_sdpa(name):
    """The scores term is a KERNEL property, and the two kernels this engine
    actually runs disagree about it: flash_attention_2 materializes no S-sized
    buffer, sdpa materializes `N * n_heads * chunk * S` elements (todo row 10).
    Assert the exact difference, not just `on > off` -- an inequality would
    still pass if the term lost its `chunk` or `n_heads` factor."""
    flash = shape_for(name, score_elem_bytes=score_elem_bytes("flash_attention_2"))
    sdpa = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    off = predict_peak_vram(flash, 8192)
    on = predict_peak_vram(sdpa, 8192)
    expected = (sdpa.N * sdpa.n_heads * sdpa.chunk
                * score_elem_bytes("sdpa") * 8192)
    assert on - off == pytest.approx(expected, rel=1e-9)
    assert on > off


def test_score_elem_bytes_is_kernel_keyed():
    """flash_attention_2 is genuinely 0.0 (not "disabled"); sdpa is non-zero."""
    assert score_elem_bytes("flash_attention_2") == 0.0
    assert score_elem_bytes("sdpa") > 0.0


@pytest.mark.parametrize("impl", ["eager", "BetterTransformer", None, "unknown"])
def test_unmeasured_kernels_take_the_largest_measured_cost(impl):
    """An unmeasured kernel must never be charged 0 -- `eager` materializes
    scores by construction, so a 0 there would silently under-reserve. The
    conservative default is the largest MEASURED coefficient."""
    assert score_elem_bytes(impl) == max(SCORE_ELEM_BYTES.values())
    assert score_elem_bytes(impl) > 0.0


def test_alloc_factor_scales_the_whole_prediction():
    a = predict_peak_vram(shape_for("qwen3-235b", alloc_factor=1.0), 7700)
    b = predict_peak_vram(shape_for("qwen3-235b", alloc_factor=1.2), 7700)
    assert b == pytest.approx(1.2 * a, rel=1e-6)


def test_peak_grows_with_batch_width():
    """predict_peak_vram is exactly homogeneous of degree 1 in N: every term --
    KV, activations, scores, and the last-chunk logits intercept -- carries an
    explicit N factor. An inequality (e.g. `> 2.5 * one`) only catches a term
    losing its N factor if that term is large enough to move the ratio past the
    threshold; equality catches ANY term losing it, large or small. If a future
    term is legitimately not proportional to N (a fixed CUDA-context overhead,
    say), this test is *supposed* to fail -- and whoever adds that term must
    also revisit the batch-width re-check a later task builds on top of this
    N-scaling."""
    one = predict_peak_vram(shape_for("minimax-m2.7", N=1), 7700)
    three = predict_peak_vram(shape_for("minimax-m2.7", N=3), 7700)
    assert three == pytest.approx(3 * one, rel=1e-6)


def test_kv_dominates_at_long_context_on_a_flash_backend():
    """Sanity anchor against a measured number: Qwen KV at 7.7k is ~1.48 GB
    (todo.md). With scores off, KV must be the largest single term."""
    sh = shape_for("qwen3-235b", score_elem_bytes=0.0, alloc_factor=1.0)
    kv = sh.kv_per_token * 7700
    assert kv / GB_DEC == pytest.approx(1.48, abs=0.02)
    assert kv > 0.5 * predict_peak_vram(sh, 7700)


# --- _peak_intercept_bytes: lm_head weight + a FULL chunk of logits
#
# This term used to follow the TRAILING chunk (`S mod chunk`), measured against
# a table whose dense rows were last-chunk-only peaks. Against the honest
# whole-prefill data (todo rows 1 and 6, 2026-08-15) that sawtooth does not
# exist: Llama-3.2-1B's measured transient at S=4096 and S=4097 is
# byte-identical, and its `transient - KV(capacity)` is flat at 0.7917 GB from
# S=4096 to S=106987 -- which is its tied fp16 head (0.5253 GB) plus a FULL
# chunk of fp32 logits (0.2627 GB), 0.5% apart.

def test_peak_intercept_charges_a_full_chunk_not_the_trailing_one():
    """A whole-prefill peak is the max over chunks, and a full chunk always
    precedes a short trailing one, so crossing a chunk boundary must NOT drop
    the charge."""
    sh = shape_for("qwen3-235b")
    full = _peak_intercept_bytes(sh, sh.chunk)
    past = _peak_intercept_bytes(sh, sh.chunk + 1)
    assert full == sh.N * sh.chunk * sh.vocab * 4
    assert past == full


def test_peak_intercept_charges_only_S_below_one_chunk():
    """Under one chunk the whole context IS the only chunk, so there is no
    full chunk to charge for."""
    sh = shape_for("qwen3-235b")
    assert _peak_intercept_bytes(sh, 0) == 0
    assert _peak_intercept_bytes(sh, 5) == sh.N * 5 * sh.vocab * 4


def test_peak_intercept_adds_the_head_weight():
    """head_bytes is the lm_head weight that goes resident when the logits are
    computed -- 0.5253 GB of tied fp16 head on Llama-3.2-1B, a file stat at the
    call site, additive on top of the logits."""
    without = shape_for("qwen3-235b", head_bytes=0)
    with_head = shape_for("qwen3-235b", head_bytes=525_336_576)
    assert (_peak_intercept_bytes(with_head, 8192)
            - _peak_intercept_bytes(without, 8192)) == 525_336_576


def test_predict_peak_vram_is_monotone_across_a_chunk_boundary():
    """The intercept no longer resets at a chunk boundary (see the section
    comment above), so peak VRAM cannot go DOWN as S goes up. This replaces
    test_predict_peak_vram_is_not_globally_monotone_across_a_chunk_boundary,
    which asserted the opposite from the last-chunk-only data. solve_max_seq_len
    still verifies every S' <= S_max rather than binary-searching: the check is
    now redundant, not wrong, and it is what makes the sawtooth's return
    (a future term keyed on the trailing chunk) safe rather than silent."""
    sh = shape_for("qwen3-235b")
    boundary = 5 * sh.chunk  # not a KV capacity-doubling point
    cap_before = kv_capacity_tokens(boundary, chunk=sh.chunk,
                                    growth=sh.kv_growth_factor, buckets=True)
    cap_after = kv_capacity_tokens(boundary + 1, chunk=sh.chunk,
                                   growth=sh.kv_growth_factor, buckets=True)
    assert cap_before == cap_after, "premise: no KV jump at this boundary"
    assert predict_peak_vram(sh, boundary + 1) >= predict_peak_vram(sh, boundary)


from vaayu.vaayu_base import _VRAM_RESERVE_FLOOR_BYTES
from vaayu.vram_plan import (
    VramPlan,
    _largest_s_within_budget,
    _too_big_message,
    admission_headroom,
    expert_vram_budget_for,
    reserve_floor_bytes,
    plan_vram,
    solve_max_seq_len,
)

# Hardware constants for this box, not model numbers (§5).
USABLE_VRAM = int(15.47 * GB)
EXPERT_BYTES = int(11.28 * MB)   # a real Qwen expert, used only as a floor input
LAYER_SLACK = int(0.6 * GB)
POSITION_LIMIT = 131072


def _plan(name, **kw):
    kw.setdefault("free_vram", USABLE_VRAM)
    kw.setdefault("shape", shape_for(name))
    kw.setdefault("min_resident", EXPERT_BYTES)
    kw.setdefault("requested_max_seq_len", None)
    kw.setdefault("position_limit", POSITION_LIMIT)
    kw.setdefault("layer_slack", LAYER_SLACK)
    return plan_vram(**kw)


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_coordination_invariant_holds_at_the_plans_own_operating_point(name):
    """THE defect this module exists to fix: pinning and the expert VRAM cache
    were two budgets that each looked fine alone and jointly overcommitted.
    Both now subtract the same reserve, so the arithmetic must close."""
    p = _plan(name)
    # Pinning consumes its whole budget in the worst case; the expert cache then
    # sizes itself from what is left.
    free_after_pinning = USABLE_VRAM - p.pinning_budget
    expert = expert_vram_budget_for(p, free_after_pinning)
    total = p.pinning_budget + expert + p.reserve_bytes
    assert total <= USABLE_VRAM


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_reserve_covers_actual_kv_at_the_derived_context(name):
    """Bug F4, as an executable test. Measured gaps were -0.28 GB at 2k and
    -1.30 GB at 7.7k because max_seq_len was 512."""
    p = _plan(name)
    sh = shape_for(name)
    assert p.reserve_bytes >= sh.kv_per_token * p.max_seq_len


def test_the_four_geometries_produce_at_least_three_distinct_s_max():
    """Assert the fixture premise. Plan L's slab_index_of mutant passed 84/84
    because every fixture collapsed to a single chunk -- a test whose premise is
    not asserted can silently stop testing anything.

    This used to require all FOUR s_max values distinct, then (Task 5b) was
    weakened to merely "not all collapsed" because llama-70b and
    minimax-m2.7 both saturate at the same S=32768 bucket ceiling. Task 5c's
    trailing-chunk intercept correction changes the picture: verified by
    hand at USABLE_VRAM, s_max is now {llama-1b: 131072 (position-limit
    clamped), llama-70b: 32768, minimax-m2.7: 32768, qwen3-235b: 33166}. The
    llama-70b/minimax-m2.7 tie is still real (same bucket-saturation
    mechanism, demonstrated below) but qwen3-235b is no longer part of it --
    the correction broke that specific collision. So this asserts the
    three-way distinctness that IS real (one demonstrated pairwise tie, not
    the weakest possible "not all four collapsed" claim), per this task's
    instruction to show the actual colliding values rather than assume."""
    s_maxes = {name: _plan(name).s_max for name in GEOMETRIES}
    assert s_maxes["llama-70b"] == s_maxes["minimax-m2.7"], s_maxes
    distinct = {s_maxes["llama-1b"], s_maxes["llama-70b"], s_maxes["qwen3-235b"]}
    assert len(distinct) == 3, s_maxes

    unclamped = {name: solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES,
                                         shape=shape_for(name))
                for name in GEOMETRIES}
    assert unclamped["llama-70b"] == unclamped["minimax-m2.7"], unclamped
    distinct_unclamped = {unclamped["llama-1b"], unclamped["llama-70b"], unclamped["qwen3-235b"]}
    assert len(distinct_unclamped) == 3, unclamped


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_solve_max_seq_len_is_exact_at_its_own_bucket_boundary(name):
    """The whole point of an honest refusal is being exactly right at the
    edge. predict_peak_vram is now a step function of S (kv_capacity_tokens),
    so solve_max_seq_len can no longer just divide a slope -- it must return
    the largest S that TRULY fits, verified against predict_peak_vram itself:
    the chosen S_max fits, and S_max + 1 does not. An off-by-one here is the
    difference between a clean refusal and an OOM."""
    shape = shape_for(name)
    s_max = solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES,
                              shape=shape)
    assert s_max > 0
    budget = USABLE_VRAM - EXPERT_BYTES
    assert predict_peak_vram(shape, s_max) <= budget
    assert predict_peak_vram(shape, s_max + 1) > budget


def test_solve_max_seq_len_stays_safe_for_every_smaller_context():
    """The property that actually matters: max_seq_len is a CAP, and the one
    reserve plan_vram computes from it must cover any real request up to
    that length -- not just a request of exactly max_seq_len tokens. Now
    that the intercept sawtooths (predict_peak_vram is not globally
    monotone -- see test_predict_peak_vram_is_not_globally_monotone_...),
    a single-level binary search could return an S_max that fits by itself
    while some SMALLER S' < S_max does not (a local sawtooth peak). Exhaust
    every S' up to S_max on a small synthetic shape (cheap: chunk=16 keeps
    the whole range tiny) to prove that cannot happen."""
    tiny = VramShape(kv_per_token=37, N=1, chunk=16, hidden=64, n_heads=4,
                     vocab=1000, elem_bytes=2,
                     score_elem_bytes=0.0, alloc_factor=1.1,
                     kv_growth_factor=2, kv_capacity_buckets=True)
    free_vram = 400_000
    s_max = solve_max_seq_len(free_vram=free_vram, min_resident=0, shape=tiny)
    assert s_max > 0
    for s in range(1, s_max + 1):
        assert predict_peak_vram(tiny, s) <= free_vram, s
    assert predict_peak_vram(tiny, s_max + 1) > free_vram


def test_solve_max_seq_len_can_land_exactly_on_a_chunk_boundary():
    """Pin the property the task calls out explicitly: at a KV
    capacity-doubling boundary (S=8192, chunk=512, growth=2), the huge KV
    jump to the next bucket cannot be offset by the tiny intercept drop, so
    nothing past the boundary survives and solve_max_seq_len must return the
    boundary itself -- predict_peak_vram(shape, S_max) <= budget <
    predict_peak_vram(shape, S_max + 1), exactly at S_max = a chunk
    boundary, not somewhere inside the next period."""
    shape = shape_for("qwen3-235b")
    boundary = 8192
    assert kv_capacity_tokens(boundary, chunk=shape.chunk,
                              growth=shape.kv_growth_factor, buckets=True) == 8192
    assert kv_capacity_tokens(boundary + 1, chunk=shape.chunk,
                              growth=shape.kv_growth_factor, buckets=True) == 16384
    budget = predict_peak_vram(shape, boundary)
    s_max = solve_max_seq_len(free_vram=budget, min_resident=0, shape=shape)
    assert s_max == boundary
    assert predict_peak_vram(shape, s_max) <= budget
    assert predict_peak_vram(shape, s_max + 1) > budget


def test_s_max_decreases_as_kv_per_token_grows():
    """Task 5b weakened this to '<=' because llama-70b and qwen3-235b tied
    at unclamped S=32768 under the (still-uncorrected) intercept. Task 5c's
    trailing-chunk correction breaks that specific tie: verified by hand at
    USABLE_VRAM + DEFAULT_ALLOC_FACTOR, llama-70b now solves to
    32768 and qwen3-235b to 33166 (strictly larger) -- fatter KV/token (327680
    vs 192512 B/token) strictly buys less context again for this pair, so
    '<' is restored. (llama-70b still ties with minimax-m2.7 at 32768 --
    see test_the_four_geometries_produce_at_least_three_distinct_s_max --
    but that is a different pair from the one this test compares.)"""
    fat = shape_for("llama-70b")            # 0.328 MB/token
    thin = shape_for("qwen3-235b")          # 0.193 MB/token
    assert fat.kv_per_token > thin.kv_per_token
    assert (solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES, shape=fat)
            < solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES, shape=thin))


def test_s_max_is_capped_by_position_limit():
    p = _plan("llama-1b", position_limit=4096)
    assert p.max_seq_len == 4096


def test_requested_context_above_s_max_raises_and_names_the_levers():
    p = _plan("qwen3-235b")
    with pytest.raises(VramPlanError) as e:
        _plan("qwen3-235b", requested_max_seq_len=p.s_max + 1)
    msg = str(e.value)
    for lever in ("prefill_chunk_size", "max_seq_len", "expert VRAM"):
        assert lever in msg, f"missing lever {lever!r} in: {msg}"


def test_requested_context_below_s_max_is_accepted_unchanged():
    p = _plan("minimax-m2.7", requested_max_seq_len=2048)
    assert p.max_seq_len == 2048
    assert p.reserve_bytes < _plan("minimax-m2.7").reserve_bytes


def test_free_vram_too_small_for_any_context_raises():
    """Task 5c made _peak_intercept_bytes follow the real trailing chunk
    (_trailing_chunk_tokens), not a constant `chunk` -- so at S=1 the
    intercept is now a SINGLE token's logits, not a whole chunk's worth. The
    old ~0.44 GiB collapse threshold was itself an artifact of that bug (a
    phantom chunk-sized floor that never existed at S=1); the real floor at
    S=1 is now on the order of a couple MB for every geometry (it is
    N*1*vocab*(elem_bytes+4) plus a couple-token KV/activation sliver, all
    tiny by construction since they are multiplied by S=1), so the dominant
    term is simply `min_resident` (EXPERT_BYTES). A free_vram well under
    EXPERT_BYTES alone is unreconcilable regardless of act_coeff/alloc_factor
    -- solve_max_seq_len's own budget<=0 guard fires before predict_peak_vram
    is even evaluated, so this is robust to any later re-measurement of
    those constants."""
    assert EXPERT_BYTES > int(5 * MB)  # premise: min_resident alone exceeds this free_vram
    with pytest.raises(VramPlanError) as e:
        _plan("llama-70b", free_vram=int(5 * MB))
    assert "does not fit" in str(e.value)


def test_the_diagnostic_cannot_itself_raise():
    """D3: the first draft of that error interpolated a tensor shape and would
    have re-created D3 inside D3's own fix. Build the message from scalars only."""
    from vaayu.vram_plan import _too_big_message
    for free in (0, 1, USABLE_VRAM):
        for s_max in (0, 1, 10 ** 9):
            msg = _too_big_message(requested=10 ** 9, s_max=s_max,
                                   free_vram=free, shape=shape_for("qwen3-235b"),
                                   min_resident=0)
            assert isinstance(msg, str) and msg


def test_reason_string_states_where_max_seq_len_came_from():
    derived = _plan("qwen3-235b")
    given = _plan("qwen3-235b", requested_max_seq_len=2048)
    assert "derived" in derived.reason
    assert "caller-supplied" in given.reason


def test_expert_budget_is_zero_rather_than_negative_when_nothing_is_left():
    p = _plan("qwen3-235b")
    assert expert_vram_budget_for(p, free_vram_now=0) == 0


def test_sub_one_expert_budget_is_reported_not_silently_zeroed():
    """CLAUDE.md §5: when a model genuinely does not fit the hardware, WARN --
    do not silently degrade. The caller (vaayu_base.py's expert-VRAM-cache
    setup, right after its `expert_vram_budget_for` call) prints the warning;
    this pins the helper it decides on, which must return a real byte count
    (possibly below one expert) rather than clamping to 0 and hiding the case."""
    p = _plan("qwen3-235b")
    tiny = expert_vram_budget_for(p, free_vram_now=p.reserve_bytes + 1 * MB)
    assert 0 < tiny < EXPERT_BYTES, (
        "caller must be able to see 'nonzero but under one expert' and warn")


def test_reserve_never_falls_below_the_measured_decode_boundary_floor():
    """2026-07-24 RESULTS: 2 GB covered prefill but not the prefill->decode
    boundary; 3.5 GB fixed it 4/4. A short context must not undercut that.

    Must import the REAL constant, not a local literal copy: a local
    `floor = int(3.5 * GB)` would keep passing even if someone edited
    vaayu_base.py's `_VRAM_RESERVE_FLOOR_BYTES` down to the 2 GB figure that
    2026-07-24 measured OOMing at the prefill->decode boundary."""
    from vaayu.vaayu_base import _VRAM_RESERVE_FLOOR_BYTES
    assert _VRAM_RESERVE_FLOOR_BYTES >= int(3.5 * GB)

    p = _plan("llama-1b", requested_max_seq_len=512,
             reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert p.reserve_bytes >= _VRAM_RESERVE_FLOOR_BYTES


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_derived_context_sizes_for_the_reserve_floor_not_the_whole_free_vram(name):
    """Amendment 4 (task-6-brief.md): with reserve_floor=0 (Task 4's own
    default, the only value its existing tests reach), requested_max_seq_len
    =None sizes the reserve for s_max -- essentially all free VRAM -- which
    measured out to pinning_budget=0 and a sub-one-expert cache on every
    on-device geometry (Qwen, MiniMax, Llama-70B). With a real reserve_floor
    supplied, the derived max_seq_len must land well below s_max, and pinning
    must actually get real room, on every geometry (CLAUDE.md §5)."""
    floor = int(3.5 * GB)
    without_floor = _plan(name, reserve_floor=0)
    with_floor = _plan(name, reserve_floor=floor)
    assert with_floor.max_seq_len < without_floor.max_seq_len, name
    assert with_floor.reserve_bytes >= floor, name
    assert with_floor.pinning_budget > int(1 * GB), (name, with_floor)
    # The reserve must still actually cover the KV at the length it chose (F4).
    sh = shape_for(name)
    assert with_floor.reserve_bytes >= sh.kv_per_token * with_floor.max_seq_len, name


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_batch_kv_helper_agrees_with_the_planner(name):
    """batch.py:kv_bytes_for's docstring already admitted 'Same formula as
    _get_vram_pinning_budget'. Two copies of one formula is how admission ends
    up quoting a different number than the reserve it was admitted against."""
    from vaayu.batch import kv_bytes_for
    geo = GEOMETRIES[name]
    planner = kv_bytes_per_token(**_kv_kwargs(geo), kv_cache_mode="bf16") * 4096
    legacy = kv_bytes_for(4096, geo["num_hidden_layers"], geo["num_key_value_heads"],
                          geo["head_dim"], 2)
    assert planner == legacy


def test_batch_width_is_rechecked_against_the_plan():
    """The plan is sized for N=1. A fan-out of N streams multiplies KV,
    activations and logits, so admission must re-evaluate rather than assume the
    load-time answer still holds. This holds at the reserve_floor=0 operating
    point exercised here (`_plan(name)`'s default -- chosen=s_max, essentially
    the whole card), where s_max is generous enough that N=3 obviously
    overflows. It is NOT the same claim as "N=3 is rejected at the
    max_seq_len the running model actually uses in production" -- see
    test_batch_width_at_the_real_reserve_floor_xfails below, which checks
    that point directly and currently does NOT hold there."""
    import dataclasses
    sh = shape_for("minimax-m2.7")
    p = _plan("minimax-m2.7")
    wide = dataclasses.replace(sh, N=3)
    assert predict_peak_vram(wide, p.max_seq_len) > USABLE_VRAM


# --- admission headroom ------------------------------------------------
#
# Measured 2026-08-15 (scripts/fit_vram_calibration.py, four probe arms): the
# expert VRAM cache is entitled to `free_vram_now - reserve_bytes` by
# expert_vram_budget_for, and it fills LAZILY during prefill. So a request
# admitted against raw live free VRAM on a COLD cache is admitted against
# headroom the cache is about to take. Free VRAM right after MiniMax loads,
# before any prefill has warmed the cache.
FREE_VRAM_COLD_CACHE = int(14.21 * 1e9)


def test_headroom_is_capped_by_the_reserve_when_an_expert_cache_is_active():
    """The bug: on a cold cache free VRAM reads far larger than what a request
    may actually spend, because expert_vram_budget_for has already promised
    everything above the reserve to the expert VRAM cache."""
    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert FREE_VRAM_COLD_CACHE > p.reserve_bytes, "premise: cache is still cold"
    assert admission_headroom(free_vram=FREE_VRAM_COLD_CACHE, plan=p,
                              expert_cache_active=True) == p.reserve_bytes


def test_headroom_is_the_whole_free_vram_without_an_expert_cache():
    """Dense models have no expert VRAM cache, so nothing else is entitled to
    the headroom and clamping to the reserve would REFUSE contexts that fit."""
    p = _plan("llama-70b", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert admission_headroom(free_vram=FREE_VRAM_COLD_CACHE, plan=p,
                              expert_cache_active=False) == FREE_VRAM_COLD_CACHE


def test_headroom_never_exceeds_the_vram_that_is_actually_free():
    """Once the cache HAS filled, free VRAM is the binding limit, not the
    reserve -- and other processes can take VRAM too."""
    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    starved = p.reserve_bytes // 4
    assert admission_headroom(free_vram=starved, plan=p,
                              expert_cache_active=True) == starved


def test_the_refusal_does_not_offer_s_max_as_a_fix_for_a_wide_batch():
    """s_max is solved at N=1. Printing it unqualified while refusing an N=3
    request tells the reader that length would fit, which is false -- the batch
    width multiplies the request but not the reserve. Only the N=1 phrasing may
    present s_max as the thing to lower to."""
    kw = dict(requested=8192, s_max=32768, free_vram=FREE_VRAM_COLD_CACHE,
              min_resident=EXPERT_BYTES)
    wide = _too_big_message(shape=shape_for("minimax-m2.7", N=3), **kw)
    assert "at a batch width of 1" in wide
    single = _too_big_message(shape=shape_for("minimax-m2.7", N=1), **kw)
    assert "at a batch width of 1" not in single


def test_the_refusal_explains_why_the_budget_is_smaller_than_free_vram():
    """D3's lesson: a diagnostic must not mislead. Refusing a 7.5 GiB request
    with "does not fit in 14.21 GiB of free VRAM" reads as a contradiction and
    sends the reader hunting for a leak. When the reserve is what bound the
    request, the message has to say so AND still report the real free figure."""
    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    msg = _too_big_message(requested=p.max_seq_len, s_max=p.s_max,
                           free_vram=FREE_VRAM_COLD_CACHE,
                           shape=shape_for("minimax-m2.7", N=3),
                           min_resident=EXPERT_BYTES,
                           headroom=p.reserve_bytes)
    assert f"{p.reserve_bytes / GB:.2f} GiB" in msg, "must name the real budget"
    assert f"{FREE_VRAM_COLD_CACHE / GB:.2f} GiB" in msg, "must not hide free VRAM"
    assert "expert VRAM cache" in msg, "must name what took the difference"


def test_the_refusal_is_unchanged_when_nothing_clamps_the_budget():
    """Dense models pass no headroom, and their message must not grow a clause
    about a cache they do not have."""
    kw = dict(requested=99999, s_max=4096, free_vram=FREE_VRAM_COLD_CACHE,
              shape=shape_for("llama-70b"), min_resident=0)
    assert _too_big_message(**kw) == _too_big_message(**kw, headroom=None)


def test_the_admission_predicate_charges_a_wide_batch_against_the_reserve():
    """The pure helper above is only worth anything if _assert_context_fits
    actually calls it. Exercises the REAL production method against a stand-in
    holding just the attributes it reads, with a cold cache's free VRAM."""
    import types
    import torch
    from vaayu.vaayu_base import VaayuBaseModel
    from vaayu.vram_plan import VramPlanError

    if not torch.cuda.is_available():
        pytest.skip("_assert_context_fits returns early without CUDA")

    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    model = types.SimpleNamespace(
        _vram_plan=p, _vram_shape=shape_for("minimax-m2.7"),
        _moe_geometry={"expert_bytes": EXPERT_BYTES},
        _expert_vram_cache=object(),      # MoE: the cache owns the headroom
    )
    real_mem_get_info = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda *a, **k: (FREE_VRAM_COLD_CACHE, FREE_VRAM_COLD_CACHE)
    try:
        # N=1 at the plan's own max_seq_len is what the reserve was sized for.
        VaayuBaseModel._assert_context_fits(model, p.max_seq_len, N=1)
        with pytest.raises(VramPlanError):
            VaayuBaseModel._assert_context_fits(model, p.max_seq_len, N=3)
        # Same request, dense model (no expert VRAM cache): nothing else is
        # entitled to the headroom, so it must NOT be refused.
        model._expert_vram_cache = None
        VaayuBaseModel._assert_context_fits(model, p.max_seq_len, N=3)
    finally:
        torch.cuda.mem_get_info = real_mem_get_info


def test_batch_width_at_the_real_reserve_floor_is_refused():
    """Prior ON-DEVICE measurement (Plan K) established that N>=2 at ~7700
    tokens is infeasible on this box for both MoE models -- a real GPU OOM, not
    a model prediction. This checks that claim at the operating point
    production actually uses: max_seq_len as built by vaayu_base.py's __init__,
    which passes reserve_floor=_VRAM_RESERVE_FLOOR_BYTES.

    This used to be a non-strict xfail blamed on uncalibrated constants. That
    was a MISDIAGNOSIS, settled 2026-08-15: even an exact predict_peak_vram
    returns ~7.5 GiB for N=3 here, which genuinely does fit a bare 15.47 GiB
    card. The refusal cannot come from a bigger prediction -- it has to come
    from the headroom, because the expert VRAM cache owns everything above the
    reserve. Charging the request against the reserve is what makes the
    prediction and the hardware agree."""
    import dataclasses
    sh = shape_for("minimax-m2.7")
    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    headroom = admission_headroom(free_vram=FREE_VRAM_COLD_CACHE, plan=p,
                                  expert_cache_active=True)
    assert predict_peak_vram(dataclasses.replace(sh, N=1), p.max_seq_len) <= headroom, \
        "N=1 at the plan's own max_seq_len must still be admitted"
    assert predict_peak_vram(dataclasses.replace(sh, N=3), p.max_seq_len) > headroom


# --- NEXT-4: the form change (2026-08-15) ---------------------------------
#
# Measured over 26 points on four geometries (scripts/fit_vram_calibration.py):
# KV(at capacity) + last-chunk logits explains three of the four on-device
# forwards to within +-0.13 GB with NO fitted constants. The fourth, fused-EXL3,
# carries a flat ~1.5 GB more -- confirmed to be chunk-major GPU retention,
# measured at 1.514 GB = 128 Qwen experts = one full layer, constant across
# S=5000..12288 and reset per layer.


def _shape_with(**kw):
    base = dict(kv_per_token=1000, N=1, chunk=512, hidden=4096, n_heads=32,
                vocab=1000, elem_bytes=2, score_elem_bytes=0.0,
                alloc_factor=1.0, kv_capacity_buckets=False)
    base.update(kw)
    return VramShape(**base)


def test_hidden_size_no_longer_changes_the_prediction():
    """The activation term (act_coeff * S) is GONE -- it was the only term that
    depended on hidden, and it is not observable: llama-3.3-70b's measured peak
    is IDENTICAL at S=12288 and S=16384 (33 KB apart) where the term predicted
    +0.13 GB, and two of four geometries fitted it NEGATIVE."""
    small = predict_peak_vram(_shape_with(hidden=1024), 4096)
    large = predict_peak_vram(_shape_with(hidden=32768), 4096)
    assert small == large


def test_resident_experts_raise_the_peak_when_they_dominate_the_logits():
    """fused-EXL3 holds one layer's experts on the GPU across prefill chunks."""
    plain = predict_peak_vram(_shape_with(), 4096)
    withres = predict_peak_vram(_shape_with(resident_expert_bytes=2_000_000_000), 4096)
    assert withres > plain
    assert withres - plain == 2_000_000_000 - _peak_intercept_bytes(_shape_with(), 4096)


def test_residency_and_logits_do_not_BOTH_count():
    """They never coexist: retention is dropped on the layer loop's last call,
    and the logits materialize afterwards in the lm_head. Charging their sum
    would over-reserve by the smaller of the two on every fused-EXL3 model."""
    sh = _shape_with(resident_expert_bytes=2_000_000_000)
    kv = _kv_term_bytes(sh, 4096)
    assert predict_peak_vram(sh, 4096) == kv + 2_000_000_000


def test_a_big_last_chunk_of_logits_can_still_set_the_peak():
    """max(), not a one-way override: a small expert set loses to the logits."""
    sh = _shape_with(vocab=200_000, resident_expert_bytes=1000)
    kv = _kv_term_bytes(sh, 4096)
    assert predict_peak_vram(sh, 4096) == kv + _peak_intercept_bytes(sh, 4096)


def test_zero_residency_leaves_dense_and_module_list_untouched():
    a = predict_peak_vram(_shape_with(), 4096)
    b = predict_peak_vram(_shape_with(resident_expert_bytes=0), 4096)
    assert a == b


def test_residency_does_not_break_the_smaller_context_guarantee():
    """solve_max_seq_len's two-level search needs the peak non-decreasing at
    chunk boundaries and within a period. A constant term preserves both, but
    assert it rather than trust it -- weakening that search was a real latent
    bug once."""
    sh = _shape_with(kv_per_token=10_000, resident_expert_bytes=50_000_000,
                     kv_capacity_buckets=True)
    budget = predict_peak_vram(sh, 6000) + 1
    smax = solve_max_seq_len(free_vram=budget, min_resident=0, shape=sh)
    for s in range(1, smax + 1):
        assert predict_peak_vram(sh, s) <= budget, f"S={s} exceeds the budget"


def test_the_logits_term_charges_fp32_only_not_both_copies():
    """Only the fp32 upcast is resident at the peak, not the fp16 logits too.

    Measured 2026-08-15 on three independent geometries at a full trailing
    chunk, where the term is largest and cleanest to isolate: the residual
    (transient - KV - residency) is 0.2627 GB on llama-3.3-70b (S=12288 AND
    S=16384) and on llama-3.2-1b (S=16384, 32768, 49152). trailing * vocab * 4
    is 0.2627 GB; trailing * vocab * 6 -- charging the fp16 copy as well -- is
    0.3940 GB. Five points, four decimals, no fitting.
    """
    sh = _shape_with(vocab=128256, chunk=512, elem_bytes=2)
    assert _peak_intercept_bytes(sh, 4096) == 512 * 128256 * 4


# --- kernel-conditional reserve floor (todo row 8) ---------------------
#
# Rows 6/7 made the peak form kernel-aware, and that CUT the derived context on
# every sdpa load (measured 2026-08-15: Llama-3.3-70B 4373 -> 3821, Llama-3.2-1B
# 106987 -> 14052) while leaving flash loads untouched. Row 8's ruling: a load on
# a score-materializing kernel raises its reserve floor to the point where it
# derives the SAME context flash would have derived at the base floor -- paid for
# out of the expert VRAM cache / pinning budget, which both subtract this same
# reserve. Flash loads pay nothing.
#
# The raised value is DERIVED per geometry (CLAUDE.md §5), not a literal: the
# alternative was a flat 6.25 GiB read off a table of four checkpoints, which is
# a model number in disguise.

def _floor(name, impl, **kw):
    kw.setdefault("base_floor", int(3.5 * GB))
    kw.setdefault("free_vram", USABLE_VRAM)
    kw.setdefault("min_resident", EXPERT_BYTES)
    kw.setdefault("layer_slack", LAYER_SLACK)
    shape = kw.pop("shape", None) or shape_for(
        name, score_elem_bytes=score_elem_bytes(impl))
    return reserve_floor_bytes(shape=shape, **kw)


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_flash_loads_keep_the_base_reserve_floor(name):
    """flash_attention_2 materializes no S-sized score buffer, so it already
    derives its full context at the base floor. Raising it there would spend
    expert-cache VRAM to buy context nothing asked for."""
    assert _floor(name, "flash_attention_2") == int(3.5 * GB)


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_sdpa_floor_buys_back_the_context_flash_would_have_derived(name):
    """The parity rule, stated on the OUTCOME (max_seq_len), not on the floor's
    byte value -- a floor asserted in bytes would be the per-model literal §5
    forbids, and would not notice if the two shapes stopped agreeing about S.

    AT LEAST parity, not exactly it (relaxed 2026-09-09, todo row 6a). The floor
    IS `predict_peak_vram(sdpa, flash_parity)` and `plan_vram` then takes the
    largest context whose peak fits that floor. Those were the same S only while
    the peak form was monotone in S; the max-over-chunks form is not, so a
    context just past a chunk boundary -- whose trailing chunk has restarted at
    one token -- can fit the same floor too. It costs nothing: the reserve is
    `max(predict(chosen), floor)`, which is the floor either way, so this is
    strictly more context at the same bytes, and every S' <= chosen is still
    checked against that floor by _largest_s_within_budget."""
    base = int(3.5 * GB)
    flash = shape_for(name, score_elem_bytes=score_elem_bytes("flash_attention_2"))
    sdpa = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    parity = _plan(name, shape=flash, reserve_floor=base).max_seq_len
    floor = _floor(name, "sdpa")
    assert floor > base, name
    ceiling = USABLE_VRAM - EXPERT_BYTES - LAYER_SLACK
    if floor >= ceiling:
        # Clamped: llama-1b's flash parity is ~90k tokens, whose sdpa peak is
        # several times this box's whole VRAM. The clamp is what keeps the
        # floor from swallowing the pinning budget whole.
        assert floor == ceiling, name
        return
    plan = _plan(name, shape=sdpa, reserve_floor=floor)
    assert plan.max_seq_len >= parity, name
    assert plan.reserve_bytes == floor, name


@pytest.mark.parametrize("impl", ["eager", None, "unknown"])
def test_unmeasured_kernels_get_the_raised_floor_too(impl):
    """Same conservative rule row 7 applies to the score coefficient: an
    unmeasured kernel is charged the largest measured cost, so it must also be
    given the floor that cost needs -- charging the cost without the floor is
    the under-reserving half of the pair."""
    assert _floor("qwen3-235b", impl) == _floor("qwen3-235b", "sdpa")


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_parity_floor_never_starves_pinning_or_undercuts_the_base_floor(name):
    """On a box too small for parity the floor must clamp, not overrun: it is
    subtracted by BOTH budget consumers, so a floor above free VRAM would make
    the arithmetic that vram_plan.py exists to close stop closing."""
    base = int(3.5 * GB)
    tight = int(6.0 * GB)      # a smaller box than this one, to force the clamp
    floor = _floor(name, "sdpa", free_vram=tight)
    assert floor >= base, name
    assert floor <= max(base, tight - EXPERT_BYTES - LAYER_SLACK), name
    p = _plan(name, shape=shape_for(name, score_elem_bytes=score_elem_bytes("sdpa")),
              free_vram=tight, reserve_floor=floor)
    assert p.pinning_budget >= 0, name


# --- bug 11a: the caching allocator's reusable pool (2026-08-16) ----------
#
# `torch.cuda.mem_get_info()` is a DRIVER-level call: it reports VRAM free on
# the device, and structurally cannot see the blocks PyTorch's caching
# allocator holds reserved-but-unallocated. Those blocks are reusable by the
# very prefill being admitted, so charging driver-free alone under-counts.
#
# Measured on Qwen3-235B, sdpa, one process, 2026-08-16 -- driver free /
# reserved / allocated, in GiB:
#
#   after load      13.37 / 1.70 / 1.69   -> pool 0.01
#   after warm-up    5.59 / 9.44 / 9.36   -> pool 0.07
#   after a prefill  4.09 / 10.94 / 9.36  -> pool 1.57
#
# On turn 2 the guard saw 4.09 GiB and refused a request needing 4.88 -- while
# 1.57 GiB sat reusable in the pool, which is why the same prefill SUCCEEDS in
# 63.7 s with the guard bypassed. These are decimal-GiB probe readings, so the
# constants below are scaled from them rather than retyped as model numbers.
_MEASURED_11A_FREE = int(4.09 * GB)
_MEASURED_11A_POOL = int(1.57 * GB)


def _sdpa_plan(name):
    """A plan built the way production builds one on the kernel where 11a
    lives: sdpa shape, row 8's parity floor. That is what makes the reserve
    (5.60 GiB on Qwen) EXCEED the free VRAM a warm turn reports (4.09), which
    is the whole precondition of the bug -- on a flash plan the 3.50 GiB floor
    sits below free VRAM and the clamp binds either way."""
    shape = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    floor = reserve_floor_bytes(shape=shape, base_floor=_VRAM_RESERVE_FLOOR_BYTES,
                                free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES,
                                layer_slack=LAYER_SLACK)
    return _plan(name, shape=shape, reserve_floor=floor)


@pytest.mark.parametrize("name", ["qwen3-235b", "minimax-m2.7"])
def test_reusable_pool_counts_toward_headroom(name):
    """The fix. Without the pool the request is charged against driver-free
    alone; with it, against what this process can actually still allocate."""
    p = _sdpa_plan(name)
    assert p.reserve_bytes > _MEASURED_11A_FREE, (
        "premise: 11a needs a reserve ABOVE the warm turn's free VRAM")
    without = admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                                 expert_cache_active=True)
    with_pool = admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                                   expert_cache_active=True,
                                   reusable_pool=_MEASURED_11A_POOL)
    assert with_pool > without, name
    assert with_pool == min(_MEASURED_11A_FREE + _MEASURED_11A_POOL,
                            p.reserve_bytes), name


@pytest.mark.parametrize("name", ["qwen3-235b", "minimax-m2.7"])
def test_the_reserve_still_caps_headroom_however_large_the_pool(name):
    """The pool is added BEFORE the clamp, never instead of it: the expert VRAM
    cache is still entitled to everything above the reserve, and a huge pool
    must not admit a request the reserve was never sized for."""
    p = _sdpa_plan(name)
    assert admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                              expert_cache_active=True,
                              reusable_pool=100 * GB) == p.reserve_bytes, name


def test_reusable_pool_counts_on_the_dense_path_too():
    """Dense has no expert cache and no clamp, but the same driver-vs-allocator
    blind spot -- so the pool has to count there as well."""
    p = _plan("llama-70b", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                              expert_cache_active=False,
                              reusable_pool=_MEASURED_11A_POOL) == (
        _MEASURED_11A_FREE + _MEASURED_11A_POOL)


def test_omitting_the_pool_changes_nothing():
    """Default 0 -- every pre-existing caller keeps its exact behaviour."""
    p = _plan("minimax-m2.7", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert (admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                               expert_cache_active=True)
            == admission_headroom(free_vram=_MEASURED_11A_FREE, plan=p,
                                  expert_cache_active=True, reusable_pool=0))


def test_refusal_reports_the_ceiling_that_applies_now_not_the_load_time_one():
    """Second half of bug 11a: the message printed `plan.s_max` -- solved at
    LOAD against full free VRAM -- while refusing on the much smaller runtime
    headroom, so it read as a contradiction ("does not fit in 5.59 GiB ... the
    largest context that fits is max_seq_len=23415"). The ceiling named must be
    the one solved against the budget actually refused on."""
    sh = shape_for("qwen3-235b")
    p = _plan("qwen3-235b", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    headroom = _MEASURED_11A_FREE
    msg = _too_big_message(requested=p.s_max, s_max=p.s_max,
                           free_vram=int(14.21 * GB), shape=sh,
                           min_resident=EXPERT_BYTES, headroom=headroom)
    runtime_ceiling = _largest_s_within_budget(sh, headroom)
    assert runtime_ceiling < p.s_max, "premise: the runtime ceiling is lower"
    assert f"max_seq_len={runtime_ceiling}" in msg
    assert f"max_seq_len={p.s_max}" not in msg.split("Real levers")[0]


import dataclasses

# --- P10: the expert VRAM cache's giveback ------------------------------------
#
# Measured on device 2026-09-07
# (docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md). The clamp above
# hands a MoE request only `reserve_bytes` because the expert VRAM cache is
# entitled to everything above it -- but batched_generate lowers that
# entitlement for the batch's lifetime, so the clamp prices a configuration the
# batch never runs in. Forced past the guard at a ~4.15k-token agentic context,
# 3 of 4 N>1 arms ran; the 4th (MiniMax N=3) OOM'd, so the clamp cannot simply
# be dropped either.
#
# All six figures below came from one load of each checkpoint on this box.
_P10_MINIMAX = dict(reserve=5145919753, cache_budget=int(6.774 * GB),
                    working_set=int(4.100 * GB), free=int(11.567 * GB),
                    need_n2=int(7.403 * GB), need_n3=int(10.890 * GB))
_P10_QWEN = dict(reserve=6017279590, cache_budget=int(3.644 * GB),
                 working_set=int(8.285 * GB), free=int(9.248 * GB),
                 need_n2=int(7.462 * GB), need_n3=int(10.488 * GB))


def _p10_plan(name, m):
    """A plan carrying the measured reserve, so the arithmetic under test is the
    one that ran on device rather than a re-derived approximation."""
    return dataclasses.replace(_plan(name), reserve_bytes=m["reserve"])


@pytest.mark.parametrize("name,m", [("minimax-m2.7", _P10_MINIMAX),
                                    ("qwen3-235b", _P10_QWEN)])
def test_omitting_the_giveback_changes_nothing(name, m):
    """Default 0 -- every pre-existing caller keeps its exact behaviour, which
    is what lets this land without re-measuring the single-stream path."""
    p = _p10_plan(name, m)
    assert (admission_headroom(free_vram=m["free"], plan=p,
                               expert_cache_active=True)
            == admission_headroom(free_vram=m["free"], plan=p,
                                  expert_cache_active=True,
                                  expert_giveback_bytes=0))


@pytest.mark.parametrize("name,m", [("minimax-m2.7", _P10_MINIMAX),
                                    ("qwen3-235b", _P10_QWEN)])
def test_giveback_raises_headroom_by_exactly_what_the_cache_surrenders(name, m):
    p = _p10_plan(name, m)
    give = int(1.5 * GB)
    assert admission_headroom(free_vram=m["free"], plan=p,
                              expert_cache_active=True,
                              expert_giveback_bytes=give) == p.reserve_bytes + give


@pytest.mark.parametrize("name,m", [("minimax-m2.7", _P10_MINIMAX),
                                    ("qwen3-235b", _P10_QWEN)])
def test_giveback_never_exceeds_the_vram_that_is_actually_free(name, m):
    """The cache cannot surrender bytes the device does not have. The `usable`
    clamp is the one thing standing between an over-large giveback and an OOM,
    so it must survive the new term."""
    p = _p10_plan(name, m)
    assert admission_headroom(free_vram=m["free"], plan=p,
                              expert_cache_active=True,
                              expert_giveback_bytes=100 * GB) == m["free"]


@pytest.mark.parametrize("name,m", [("minimax-m2.7", _P10_MINIMAX),
                                    ("qwen3-235b", _P10_QWEN)])
def test_a_negative_giveback_cannot_shrink_the_headroom(name, m):
    """A cache budget below the working-set floor yields a negative difference
    at the call site. Charging it here would refuse contexts that fit today --
    a regression dressed as caution."""
    p = _p10_plan(name, m)
    assert admission_headroom(free_vram=m["free"], plan=p,
                              expert_cache_active=True,
                              expert_giveback_bytes=-int(4.6 * GB)) == p.reserve_bytes


def test_giveback_is_ignored_without_an_expert_cache():
    """Dense has no cache to surrender anything, and no clamp to lift."""
    p = _plan("llama-70b", reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    assert admission_headroom(free_vram=FREE_VRAM_COLD_CACHE, plan=p,
                              expert_cache_active=False,
                              expert_giveback_bytes=int(2 * GB)) == FREE_VRAM_COLD_CACHE


def test_the_measured_minimax_arms_are_decided_correctly():
    """The anti-regression case, and the whole reason the term exists. With the
    giveback its cache can really surrender (budget - working set = 2.674 GiB),
    MiniMax's headroom admits the N=2 arm that RAN on device and refuses the
    N=3 arm that OOM'd at model.layers.22. Today's clamp refuses both."""
    m = _P10_MINIMAX
    p = _p10_plan("minimax-m2.7", m)
    give = m["cache_budget"] - m["working_set"]
    assert give > 0, "premise: MiniMax's cache has something to give back"

    today = admission_headroom(free_vram=m["free"], plan=p,
                               expert_cache_active=True)
    assert today < m["need_n2"], "premise: today's clamp refuses the N=2 arm"

    headroom = admission_headroom(free_vram=m["free"], plan=p,
                                  expert_cache_active=True,
                                  expert_giveback_bytes=give)
    assert headroom >= m["need_n2"], "the N=2 arm ran on device -- admit it"
    assert headroom < m["need_n3"], "the N=3 arm OOM'd on device -- refuse it"


def test_qwen_gains_nothing_because_its_cache_is_already_under_its_working_set():
    """Not a bug, and not a case to 'fix' by lowering the floor: Qwen's expert
    VRAM cache budget (3.644 GiB) is SMALLER than one decode step's working set
    (8.285 GiB), so there is nothing to surrender. Its two false refusals are a
    need-side over-prediction, which this change deliberately does not touch --
    pinned here so a later reader does not mistake the asymmetry for a defect."""
    m = _P10_QWEN
    p = _p10_plan("qwen3-235b", m)
    give = max(0, m["cache_budget"] - m["working_set"])
    assert give == 0
    assert admission_headroom(free_vram=m["free"], plan=p,
                              expert_cache_active=True,
                              expert_giveback_bytes=give) == p.reserve_bytes


# --- which chunk sets the peak (todo row 6a / §0 P5) -------------------
#
# `predict_peak_vram` used to charge ONE chunk's terms against the FINAL KV
# capacity: `KV(cap(S)) + head + a FULL chunk of logits`. That is the max over
# chunks only when the logits term outgrows the KV growth between chunks, which
# is true on Llama-3.2-1B (0.2627 GB of logits against 16.8 MB of KV per chunk)
# and false on every other geometry measured. The honest whole-prefill data says
# so directly -- `transient - KV(capacity)` on llama-3.3-70b flash is 0.9278 GB
# at S=2048/4096/8192 (full trailing chunk) and 0.7140/0.7182 GB at S=2049/4097
# (1-token trailing chunk), reproducible to four decimals -- and the old form
# charged the full-chunk figure at both, over-reserving by 30-42% at every
# partial trailing chunk.
#
# Which chunk sets the peak is a property of the FORWARD, not of the model, and
# vaayu_base.py has exactly two:
#   * per-chunk loop (dense, or any MoE forward with output_attentions /
#     output_hidden_states): one `forward` per chunk, so every layer's KV sits at
#     THAT chunk's capacity and lm_head runs on every chunk.
#   * layer-major (`_chunked_prefill`'s MoE branch): ONE forward whose chunks are
#     batch elements, so layer L processes every chunk before layer L+1 starts --
#     by the time the widest chunk runs, every earlier layer's KV is already at
#     the FINAL capacity -- and lm_head is skipped for every chunk but the last
#     (vaayu_base.py:4581, "only the LAST chunk's logits are consumed").

def _sawtooth_shape(**kw):
    """A shape whose KV growth per chunk dwarfs a chunk of logits -- the
    llama-3.3-70b/minimax regime, where the peak sits on the LAST chunk."""
    return shape_for("llama-70b", **kw)


def test_layer_major_prefill_charges_logits_on_the_trailing_chunk_only():
    """The MoE layer-major forward runs lm_head once, on the last chunk."""
    sh = _sawtooth_shape()
    lm = dataclasses.replace(sh, layer_major_prefill=True)
    boundary = 5 * sh.chunk
    assert kv_capacity_tokens(boundary, chunk=sh.chunk, growth=sh.kv_growth_factor,
                              buckets=True) == kv_capacity_tokens(
        boundary + 1, chunk=sh.chunk, growth=sh.kv_growth_factor, buckets=True), \
        "premise: no KV jump across this boundary, so only the logits term moves"
    drop = predict_peak_vram(lm, boundary) - predict_peak_vram(lm, boundary + 1)
    assert drop == lm.N * (lm.chunk - 1) * lm.vocab * 4


def test_per_chunk_prefill_keeps_a_full_chunk_of_logits_past_a_boundary():
    """The dense loop runs lm_head on EVERY chunk, so an earlier full chunk is
    still a peak candidate once the trailing chunk goes short."""
    sh = _sawtooth_shape()
    boundary = 5 * sh.chunk
    assert predict_peak_vram(sh, boundary + 1) >= predict_peak_vram(sh, boundary)


def test_the_peak_sits_on_the_chunk_whose_KV_dominates_not_on_a_full_chunk():
    """When the KV capacity doubles at the last chunk, that chunk sets the peak
    -- with ITS logits (one token), not a full chunk's."""
    sh = _sawtooth_shape()
    S = 4 * sh.chunk + 1                       # 5 chunks; capacity doubles at the 5th
    cap = kv_capacity_tokens(S, chunk=sh.chunk, growth=sh.kv_growth_factor, buckets=True)
    assert cap > kv_capacity_tokens(4 * sh.chunk, chunk=sh.chunk,
                                    growth=sh.kv_growth_factor, buckets=True)
    last_chunk = sh.kv_per_token * cap * sh.N + _peak_intercept_bytes(sh, 1)
    assert predict_peak_vram(sh, S) == last_chunk


def test_no_chunk_is_ever_charged_more_than_the_old_flat_envelope():
    """Safety invariant, the reason this form can ship without a new on-device
    round: the new max-over-chunks value is never ABOVE the old
    `KV(cap(S)) + head + full chunk of logits + slope*S` envelope, for either
    forward. Over-reserving was the safe direction and this only ever reserves
    less."""
    for name in GEOMETRIES:
        for layer_major in (False, True):
            for score in (0.0, 10.9):
                sh = dataclasses.replace(shape_for(name, score_elem_bytes=score),
                                         layer_major_prefill=layer_major)
                for S in (1, 511, 512, 513, 1024, 2048, 2049, 4097, 8192, 8193):
                    envelope = (sh.N * sh.n_heads * sh.chunk * sh.score_elem_bytes * S
                                + max(_peak_intercept_bytes(sh, S),
                                      float(sh.resident_expert_bytes))
                                + _kv_term_bytes(sh, S))
                    assert predict_peak_vram(sh, S) <= int(round(envelope * sh.alloc_factor)) + 1, \
                        (name, layer_major, score, S)


def test_layer_major_prefill_defaults_to_the_per_chunk_loop():
    """The conservative forward is the default: a shape that does not say which
    forward it runs is charged the one that peaks higher."""
    sh = shape_for("qwen3-235b")
    assert sh.layer_major_prefill is False


# --- S14 3c: per-key attention term -------------------------------------
#
# Eager attention's transient is c*H*q*k + r*k (measured 2026-09-30,
# plans/probes/s14c_eager_coeff.py): r is the GQA repeat_kv copies of K and V,
# 2*H*D*elem_bytes per key, whatever q is. VramShape carries it as
# score_bytes_per_key; 0.0 keeps every older shape's prediction unchanged.
import dataclasses as _dc


def test_score_bytes_per_key_defaults_to_zero():
    assert shape_for("llama-1b").score_bytes_per_key == 0.0


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_per_key_term_is_keys_times_bytes_per_key(name):
    geo = GEOMETRIES[name]
    per_key = float(2 * geo["n_heads"] * geo["head_dim"] * 2)
    base = shape_for(name, score_elem_bytes=6.0)
    keyed = _dc.replace(base, score_bytes_per_key=per_key)
    for q, k in ((512, 8192), (1, 33000), (256, 4096)):
        assert _score_term_bytes(keyed, q, k) == _score_term_bytes(base, q, k) + k * per_key


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_flash_parity_floor_ignores_the_per_key_term(name):
    """The floor buys back the context FLASH would derive; flash has neither
    term, so flash parity must be solved with both zeroed."""
    base_floor = int(3.5 * GB)
    free = int(15.0 * GB)
    plain = shape_for(name, score_elem_bytes=6.0)
    keyed = _dc.replace(plain, score_bytes_per_key=16384.0)
    parity = _largest_s_within_budget(_dc.replace(plain, score_elem_bytes=0.0), base_floor)
    got = reserve_floor_bytes(shape=keyed, base_floor=base_floor, free_vram=free,
                              min_resident=0, layer_slack=0)
    assert got == max(base_floor, min(predict_peak_vram(keyed, parity), free))


# --- S15 L-b: the floor protects MoE streamed weights; a requested context caps parity ---
import dataclasses as _dc15

from vaayu.vram_plan import _largest_s_within_budget as _lswb15
from vaayu.vram_plan import reserve_floor_bytes as _rfb15

_BASE15 = int(3.5 * GB)
_FREE15 = int(15.0 * GB)


def _parity15(sh):
    return _lswb15(_dc15.replace(sh, score_elem_bytes=0.0, score_bytes_per_key=0.0), _BASE15)


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_floor_defaults_reproduce_todays_formula(name):
    sh = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    want = max(_BASE15, min(predict_peak_vram(sh, _parity15(sh)), _FREE15))
    assert _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15,
                  min_resident=0, layer_slack=0) == want


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_protected_bytes_cap_the_raise(name):
    sh = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    raised = min(predict_peak_vram(sh, _parity15(sh)), _FREE15)
    assert raised > _BASE15, "premise: this geometry's floor is raised above the base"
    protected = _FREE15 - (_BASE15 + raised) // 2      # leaves room for a floor half-way up
    got = _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15, min_resident=0,
                 layer_slack=0, protected_bytes=protected)
    assert got == max(_BASE15, _FREE15 - protected)
    assert _BASE15 < got < raised


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_protected_bytes_never_push_below_the_base_floor(name):
    sh = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    assert _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15, min_resident=0,
                  layer_slack=0, protected_bytes=_FREE15) == _BASE15


@pytest.mark.parametrize("name", ["minimax-m2.7", "llama-70b"])
def test_requested_context_caps_the_floors_parity(name):
    sh = shape_for(name, score_elem_bytes=score_elem_bytes("sdpa"))
    parity = _parity15(sh)
    req = max(1, parity // 4)
    want = max(_BASE15, min(predict_peak_vram(sh, req), _FREE15))
    assert _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15, min_resident=0,
                  layer_slack=0, requested_max_seq_len=req) == want
    # a request above parity changes nothing
    assert _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15, min_resident=0,
                  layer_slack=0, requested_max_seq_len=parity * 2) == \
        _rfb15(shape=sh, base_floor=_BASE15, free_vram=_FREE15, min_resident=0, layer_slack=0)


# --- S15 L-b trap guard ----------------------------------------------------
# A geometry with a tiny KV (DeepSeek-V4: 6,880 B/token) turns flash parity into an
# enormous context, and the unprotected floor then eats the pin budget (reserve
# ~11.3 GiB, pinning 0.01 GiB, every streamed layer re-read each token).
# `protected_bytes` is the guard. Fixture geometry: DeepSeek-V4-Flash on this box.

def _deepseek_like():
    shape = VramShape(kv_per_token=6880, N=1, chunk=512, hidden=4096, n_heads=64,
                      vocab=129280, elem_bytes=2, score_elem_bytes=1.0,
                      alloc_factor=1.0, kv_growth_factor=2, kv_capacity_buckets=False,
                      head_bytes=1_059_061_856)
    return dict(shape=shape, base_floor=int(3.5 * GB), free_vram=int(15.73e9),
                min_resident=13_369_344, layer_slack=int(3.32 * GB))


def test_protected_floor_defeats_the_tiny_kv_trap():
    kw = _deepseek_like()
    protected = int(8.24 * GB)
    assert _rfb15(**kw) >= 11 * GB                       # (a) the trap, unprotected
    floor = _rfb15(**kw, protected_bytes=protected)
    assert floor == kw["base_floor"]                     # (b) held at the base
    plan = plan_vram(free_vram=kw["free_vram"], shape=kw["shape"],
                     min_resident=kw["min_resident"], requested_max_seq_len=None,
                     position_limit=1_000_000, layer_slack=kw["layer_slack"],
                     reserve_floor=floor)
    assert plan.pinning_budget >= 7.5 * GB               # (c) the pin budget survives


@pytest.mark.parametrize("name", ["minimax-m2.7", "qwen3-235b"])
def test_a_refusal_with_reusable_pool_names_the_ceiling_it_was_refused_on(name):
    """S19 (vii): with memory the process already holds and can reuse, a request may
    spend MORE than the driver reports free. The message then printed the driver's
    figure as the budget and the load-time s_max as the ceiling: "does not fit in
    6.63 GiB of free VRAM. The largest context that fits ... is max_seq_len=52222",
    refusing 33156 tokens on a load that advertises 33155."""
    sh = shape_for(name)
    p = _plan(name, reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    budget = p.reserve_bytes - 64 * MB
    free = budget - 2 * GB                              # the rest is reusable pool
    ceiling = _largest_s_within_budget(sh, budget)
    assert 0 < ceiling < p.s_max, "premise: the budget binds below the load-time solve"
    msg = _too_big_message(requested=ceiling + 1, s_max=p.s_max, free_vram=free,
                           shape=sh, min_resident=EXPERT_BYTES, headroom=budget)
    assert f"max_seq_len={ceiling}" in msg
    assert str(p.s_max) not in msg, "the load-time ceiling is not what was refused on"
    assert f"{budget / GB:.2f} GiB" in msg, "must name the budget the request had"
    assert f"{free / GB:.2f} GiB" in msg, "must still report what the driver says is free"


@pytest.mark.parametrize("name", ["minimax-m2.7", "qwen3-235b"])
def test_the_lever_names_the_same_ceiling_as_the_opening(name):
    """ "Lower max_seq_len to <load-time s_max> or less" under an opening that names a
    smaller runtime ceiling is advice that does not fix the refusal."""
    sh = shape_for(name)
    p = _plan(name, reserve_floor=_VRAM_RESERVE_FLOOR_BYTES)
    for free in (int(14 * GB), p.reserve_bytes - 2 * GB):       # budget under / over free
        ceiling = _largest_s_within_budget(sh, p.reserve_bytes)
        assert 0 < ceiling < p.s_max
        msg = _too_big_message(requested=ceiling + 1, s_max=p.s_max, free_vram=free,
                               shape=sh, min_resident=EXPERT_BYTES,
                               headroom=p.reserve_bytes)
        assert f"lower max_seq_len to {ceiling} or less" in msg


def test_a_ceiling_is_never_offered_above_the_load_time_one():
    """s_max can be capped by the model's position limit, below what the bytes allow."""
    sh = shape_for("minimax-m2.7")
    budget = int(12 * GB)
    assert _largest_s_within_budget(sh, budget) > 2048, "premise"
    for free in (int(14 * GB), int(8 * GB)):
        msg = _too_big_message(requested=4096, s_max=2048, free_vram=free, shape=sh,
                               min_resident=EXPERT_BYTES, headroom=budget)
        assert "max_seq_len=2048" in msg
