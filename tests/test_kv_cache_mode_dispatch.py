"""_create_kv_cache must refuse an unrecognised kv_cache_mode, not guess.

H1: the method handled auto/fp16/bf16/turboquant explicitly and then FELL
THROUGH to the fp8 cache for everything else, so a typo silently changed the
cache format -- the silent degradation CLAUDE.md §5 forbids. The planner side
(vram_plan.kv_elem_bytes) already raised on the same typo, so the engine
contradicted itself: the planner refused a mode the cache accepted.
"""
import types

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vaayu.vaayu_base import (FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel,
                              resolve_kv_cache_mode)
from vaayu.vram_plan import VramPlanError


def _model(mode, *, exl3=False):
    """Stand-in carrying only what _create_kv_cache reads."""
    loader = object()
    if exl3:
        from vaayu.persist.loader_strategy import EXL3LoaderStrategy
        loader = EXL3LoaderStrategy.__new__(EXL3LoaderStrategy)
    return types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=loader)


def test_unknown_mode_raises_instead_of_silently_returning_fp8():
    with pytest.raises(VramPlanError) as e:
        VaayuBaseModel._create_kv_cache(_model("bf16_typo"))
    msg = str(e.value)
    assert "bf16_typo" in msg, "must name the offending value"
    assert "fp8" in msg, "must list the modes that ARE known"


@pytest.mark.parametrize("mode,expected", [
    ("fp16", DynamicCache),
    ("bf16", FP8DynamicCache),
    ("turboquant", TurboQuantKVCache),
    ("turboquant:9", TurboQuantKVCache),
])
def test_every_valid_mode_still_returns_its_own_cache(mode, expected):
    assert type(VaayuBaseModel._create_kv_cache(_model(mode))) is expected


def test_auto_builds_the_ring_buffer_on_every_loader():
    """auto is bf16 on EXL3 AND on everything else (row 3, 2026-09-11). It used
    to be fp16 off EXL3, which built a bare DynamicCache and so switched the
    prefix cache off at load for every non-EXL3 model."""
    for exl3 in (True, False):
        c = VaayuBaseModel._create_kv_cache(_model("auto", exl3=exl3))
        assert type(c) is FP8DynamicCache and c.quantize is False


@pytest.mark.parametrize("exl3", [True, False])
def test_auto_yields_a_cache_the_prefix_pool_can_store(exl3):
    """The point of row 3: with the server's defaults (KV_CACHE_MODE="auto",
    ENABLE_PREFIX_CACHE=True) the load-time probe must pass on any loader."""
    from vaayu.batch import cache_supports_prefix_store
    assert cache_supports_prefix_store(
        VaayuBaseModel._create_kv_cache(_model("auto", exl3=exl3))) is True


def test_an_explicit_fp16_request_still_reaches_the_refusal():
    """fp16 still means the stock DynamicCache, so the load-time refusal (the
    guard against lookup()-pops-then-store()-fails draining the pool) stays
    reachable for anyone who asks for it."""
    from vaayu.batch import cache_supports_prefix_store
    assert cache_supports_prefix_store(
        VaayuBaseModel._create_kv_cache(_model("fp16"))) is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bf16_mode_stores_the_source_dtype_and_round_trips(dtype):
    """"bf16" mode is the ring buffer, not a cast: K/V keep whatever dtype the
    model runs in (fp16 on a pre-Ampere GPU, fp32 on CPU). What auto now builds
    for non-EXL3 models must not change their KV dtype, and the pool's
    to_cpu/to_device round trip must hand back the same values."""
    c = VaayuBaseModel._create_kv_cache(_model("bf16"))
    k, v = torch.randn(1, 2, 5, 4, dtype=dtype), torch.randn(1, 2, 5, 4, dtype=dtype)
    c.update(k, v, 0)
    assert c.key_cache[0].dtype is dtype
    c.to_cpu()
    c.to_device("cpu")
    assert c.resident_bytes() == 2 * k.nbytes    # compacted to used length
    k2, v2 = c.update(k[..., :1, :], v[..., :1, :], 0)
    assert k2.dtype is dtype
    assert torch.equal(k2[..., :5, :], k) and torch.equal(v2[..., :5, :], v)


def test_fp8_is_reachable_explicitly():
    """fp8 stays a supported mode -- it just stops being the fall-through."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8_e4m3fn takes the documented fp16 fallback")
    assert type(VaayuBaseModel._create_kv_cache(_model("fp8"))) is FP8DynamicCache


# ── the resolved mode is a fact three readers share ─────────────────────────
# `_create_kv_cache` used to resolve "auto" (and fp8-without-float8) inline,
# with a second copy of the same branch in `_kv_cache_is_bucketed`. Nothing
# outside could say which dtype ran, so env.json reported the REQUEST verbatim
# ("auto") -- against its own premise that every field is derived. The
# resolution is now one pure function; these tests are what stops the three
# readers drifting apart again.

@pytest.mark.parametrize("mode,exl3,expected", [
    ("auto", True, "bf16"),
    ("auto", False, "bf16"),
    ("bf16", False, "bf16"),
    ("fp16", False, "fp16"),
    ("fp8", False, "fp8"),
    ("turboquant", False, "turboquant"),
    ("turboquant:9", False, "turboquant:9"),
    ("bf16_typo", False, "bf16_typo"),   # passed through; _create_kv_cache refuses it
])
def test_resolve_names_the_mode_that_will_actually_be_honoured(mode, exl3, expected):
    if mode == "fp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this build takes the documented fp16 fallback")
    m = _model(mode, exl3=exl3)
    assert resolve_kv_cache_mode(m.kv_cache_mode, m.loader_strategy) == expected


@pytest.mark.parametrize("mode,exl3", [
    ("auto", True), ("auto", False), ("bf16", False), ("fp16", False),
    ("fp8", False), ("turboquant", False), ("turboquant:9", False),
])
def test_the_resolved_mode_is_the_cache_that_gets_built(mode, exl3):
    """Resolve, then build with the RESOLVED string: same cache object as
    building with the requested one. A resolver that disagrees with the
    constructor would make env.json confidently wrong."""
    m = _model(mode, exl3=exl3)
    resolved = resolve_kv_cache_mode(m.kv_cache_mode, m.loader_strategy)
    direct = VaayuBaseModel._create_kv_cache(m)
    via_resolved = VaayuBaseModel._create_kv_cache(_model(resolved))
    assert type(direct) is type(via_resolved)
    assert getattr(direct, "quantize", None) == getattr(via_resolved, "quantize", None)
    assert getattr(direct, "bits_total", None) == getattr(via_resolved, "bits_total", None)


@pytest.mark.parametrize("mode,exl3", [
    ("auto", True), ("auto", False), ("bf16", False), ("fp16", False),
    ("fp8", False), ("turboquant", False),
])
def test_bucketed_matches_the_cache_the_resolved_mode_builds(mode, exl3):
    """`_kv_cache_is_bucketed` answers at load time, before any cache exists;
    the VRAM plan sizes on it. Bucketed == not a plain DynamicCache."""
    m = _model(mode, exl3=exl3)
    built = VaayuBaseModel._create_kv_cache(m)
    assert VaayuBaseModel._kv_cache_is_bucketed(m) is (type(built) is not DynamicCache)


def test_fp8_without_float8_support_resolves_to_fp16_and_still_says_so(monkeypatch, capsys):
    """The fallback moved from inside _create_kv_cache into the resolver, so
    env.json reports the fp16 that ran rather than the fp8 that was asked
    for. The operator-facing warning must survive that move."""
    monkeypatch.delattr(torch, "float8_e4m3fn", raising=False)
    m = _model("fp8")
    assert resolve_kv_cache_mode(m.kv_cache_mode, m.loader_strategy) == "fp16"
    assert type(VaayuBaseModel._create_kv_cache(m)) is DynamicCache
    assert "FP8 KV cache unavailable" in capsys.readouterr().out
    assert VaayuBaseModel._kv_cache_is_bucketed(m) is False


def test_a_plain_fp16_request_does_not_claim_an_fp8_fallback(capsys):
    """The fallback warning is keyed on the REQUESTED mode, so an ordinary
    fp16 request must not print it."""
    VaayuBaseModel._create_kv_cache(_model("fp16"))
    assert "FP8 KV cache unavailable" not in capsys.readouterr().out
