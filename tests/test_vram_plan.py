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
    DEFAULT_ACT_COEFF,
    DEFAULT_ALLOC_FACTOR,
    VramShape,
    predict_peak_vram,
)


def shape_for(name, *, N=1, chunk=512, kv_cache_mode="bf16",
              materializes_scores=False, alloc_factor=DEFAULT_ALLOC_FACTOR,
              kv_growth_factor=2, kv_capacity_buckets=None):
    geo = GEOMETRIES[name]
    if kv_capacity_buckets is None:
        kv_capacity_buckets = geo["kv_capacity_buckets"]
    return VramShape(
        kv_per_token=kv_bytes_per_token(**_kv_kwargs(geo), kv_cache_mode=kv_cache_mode),
        N=N, chunk=chunk, hidden=geo["hidden"], n_heads=geo["n_heads"],
        vocab=geo["vocab"], elem_bytes=2, act_coeff=DEFAULT_ACT_COEFF,
        materializes_scores=materializes_scores, alloc_factor=alloc_factor,
        kv_growth_factor=kv_growth_factor, kv_capacity_buckets=kv_capacity_buckets,
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
def test_scores_term_is_zero_when_flash_is_selected(name):
    off = predict_peak_vram(shape_for(name, materializes_scores=False), 8192)
    on = predict_peak_vram(shape_for(name, materializes_scores=True), 8192)
    assert on > off


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
    sh = shape_for("qwen3-235b", materializes_scores=False, alloc_factor=1.0)
    kv = sh.kv_per_token * 7700
    assert kv / GB_DEC == pytest.approx(1.48, abs=0.02)
    assert kv > 0.5 * predict_peak_vram(sh, 7700)


from vaayu.vram_plan import (
    VramPlan,
    expert_vram_budget_for,
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


def test_the_four_geometries_are_not_collapsed_into_one_fixture():
    """Assert the fixture premise. Plan L's slab_index_of mutant passed 84/84
    because every fixture collapsed to a single chunk -- a test whose premise is
    not asserted can silently stop testing anything.

    This used to require all FOUR s_max values distinct. Task 5b's bucketed
    KV model (kv_capacity_tokens) makes that too strong: within a capacity
    bucket the KV term is flat, so two shapes can each have enough budget to
    fill a bucket up to its ceiling but not enough for the next doubling,
    landing both exactly on that ceiling. Verified by hand at USABLE_VRAM:
    llama-70b and minimax-m2.7 both solve to unclamped S=32768 even though
    their kv_per_token values differ (327680 vs 253952 B/token) -- the
    SATURATING bucket ceiling coincides, which is a real property of the
    corrected model, not a fixture defect. So this now asserts the weaker,
    still-meaningful claim the old test actually protected: the four
    fixtures do not all collapse to ONE value."""
    s_maxes = {name: _plan(name).s_max for name in GEOMETRIES}
    assert len(set(s_maxes.values())) > 1, s_maxes

    unclamped = {name: solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES,
                                         shape=shape_for(name))
                for name in GEOMETRIES}
    assert len(set(unclamped.values())) > 1, unclamped


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


def test_s_max_decreases_as_kv_per_token_grows():
    """Was a strict '<'. Task 5b's bucketed KV model (kv_capacity_tokens)
    makes the KV term flat within a capacity bucket, so a fatter KV/token
    shape can legitimately SATURATE at the same bucket ceiling as a thinner
    one (verified by hand at USABLE_VRAM + DEFAULT_ACT_COEFF/ALLOC_FACTOR:
    llama-70b and qwen3-235b both land on unclamped S=32768) without
    contradicting "more KV cost never buys more context" -- that claim is
    <=, not <, once the KV term can plateau."""
    fat = shape_for("llama-70b")            # 0.328 MB/token
    thin = shape_for("qwen3-235b")          # 0.193 MB/token
    assert fat.kv_per_token > thin.kv_per_token
    assert (solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES, shape=fat)
            <= solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES, shape=thin))


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
    """Measured collapse threshold for llama-70b is ~0.44 GiB, and it sits
    there because the floor is _peak_intercept_bytes -- the S-independent
    last-chunk logits buffer -- not KV. "Too small for any context" really
    means "too small for one chunk's logits." 0.25 GiB is deliberately well
    under that threshold rather than just under it: the intercept does not
    depend on act_coeff at all, and alloc_factor is always >= 1.0, so
    (0.25 GiB - min_resident) / alloc_factor - intercept < 0 holds for ANY
    value a later measurement task could give those constants. A value chosen
    just under 0.44 GiB would silently stop testing the raise path once the
    constants are re-measured."""
    with pytest.raises(VramPlanError) as e:
        _plan("llama-70b", free_vram=int(0.25 * GB))
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
    do not silently degrade. The caller at vaayu_base.py:1686 prints the warning;
    this pins the helper it decides on, which must return a real byte count
    (possibly below one expert) rather than clamping to 0 and hiding the case."""
    p = _plan("qwen3-235b")
    tiny = expert_vram_budget_for(p, free_vram_now=p.reserve_bytes + 1 * MB)
    assert 0 < tiny < EXPERT_BYTES, (
        "caller must be able to see 'nonzero but under one expert' and warn")
