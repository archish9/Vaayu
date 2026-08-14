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
