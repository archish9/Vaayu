"""Pure VRAM-planning tests. No CUDA, no model load -- that is the point
(METHOD NOTES: two Plan L defects lived in code 'verified' by a harness that
copied the logic instead of executing it)."""
import pytest

from vaayu.vram_plan import (
    VramPlanError,
    kv_bytes_per_token,
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
GEOMETRIES = {
    "qwen3-235b":   dict(num_hidden_layers=94, num_key_value_heads=4,
                         head_dim=128, hidden=4096, n_heads=64, vocab=151936),
    "minimax-m2.7": dict(num_hidden_layers=62, num_key_value_heads=8,
                         head_dim=128, hidden=3072, n_heads=48, vocab=200064),
    "llama-70b":    dict(num_hidden_layers=80, num_key_value_heads=8,
                         head_dim=128, hidden=8192, n_heads=64, vocab=128256),
    "llama-1b":     dict(num_hidden_layers=16, num_key_value_heads=8,
                         head_dim=64, hidden=2048, n_heads=32, vocab=128256),
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
              materializes_scores=False, alloc_factor=DEFAULT_ALLOC_FACTOR):
    geo = GEOMETRIES[name]
    return VramShape(
        kv_per_token=kv_bytes_per_token(**_kv_kwargs(geo), kv_cache_mode=kv_cache_mode),
        N=N, chunk=chunk, hidden=geo["hidden"], n_heads=geo["n_heads"],
        vocab=geo["vocab"], elem_bytes=2, act_coeff=DEFAULT_ACT_COEFF,
        materializes_scores=materializes_scores, alloc_factor=alloc_factor,
    )


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_peak_is_linear_in_S(name):
    """Every S-dependent term is first-order, so equal steps in S must produce
    equal steps in peak. If this ever fails, a quadratic term was added and the
    closed-form S_max solve in plan_vram is no longer valid."""
    sh = shape_for(name)
    d1 = predict_peak_vram(sh, 6000) - predict_peak_vram(sh, 5000)
    d2 = predict_peak_vram(sh, 7000) - predict_peak_vram(sh, 6000)
    assert d1 == pytest.approx(d2, rel=1e-6)


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


def test_the_four_geometries_produce_four_distinct_s_max():
    """Assert the fixture premise. Plan L's slab_index_of mutant passed 84/84
    because every fixture collapsed to a single chunk -- a test whose premise is
    not asserted can silently stop testing anything.

    Plan-level s_max alone is not enough: at USABLE_VRAM, llama-1b's solve is
    clamped by position_limit (131072) down from its true unclamped value of
    278176. That clamp makes it distinct from the other three "for free," not
    because of its own KV/activation geometry. So assert distinctness on the
    UNCLAMPED solve_max_seq_len too -- that is the actual geometry claim --
    alongside the existing plan-level check."""
    s_maxes = {name: _plan(name).s_max for name in GEOMETRIES}
    assert len(set(s_maxes.values())) == len(GEOMETRIES), s_maxes

    unclamped = {name: solve_max_seq_len(free_vram=USABLE_VRAM, min_resident=EXPERT_BYTES,
                                         shape=shape_for(name))
                for name in GEOMETRIES}
    assert len(set(unclamped.values())) == len(GEOMETRIES), unclamped


def test_s_max_decreases_as_kv_per_token_grows():
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
