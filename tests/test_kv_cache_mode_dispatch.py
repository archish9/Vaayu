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

from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel
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


def test_auto_still_resolves_by_loader_not_by_string():
    """auto is bf16 on EXL3 and fp16 otherwise -- the branch the raise must not
    disturb."""
    assert type(VaayuBaseModel._create_kv_cache(_model("auto", exl3=True))) is FP8DynamicCache
    assert type(VaayuBaseModel._create_kv_cache(_model("auto"))) is DynamicCache


def test_fp8_is_reachable_explicitly():
    """fp8 stays a supported mode -- it just stops being the fall-through."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8_e4m3fn takes the documented fp16 fallback")
    assert type(VaayuBaseModel._create_kv_cache(_model("fp8"))) is FP8DynamicCache
