"""scripts/probe_vram_curve.py's default S sweep must probe the two places
predict_peak_vram is piecewise: KV capacity doublings and prefill-chunk
boundaries. Every point is DERIVED from chunk/growth/s_max (CLAUDE.md S5) --
no model numbers appear here or in the function under test.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "probe_vram_curve",
    Path(__file__).resolve().parent.parent / "scripts" / "probe_vram_curve.py",
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192),      # a small derived ceiling
    (512, 2, 131072),    # a large one
    (256, 3, 20000),     # neither chunk nor growth is the common case
])
def test_sweep_is_sorted_unique_and_inside_the_ceiling(chunk, growth, s_max):
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    assert pts == sorted(set(pts))
    assert all(0 < s <= s_max for s in pts)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192), (512, 2, 131072), (256, 3, 20000),
])
def test_sweep_straddles_a_chunk_boundary(chunk, growth, s_max):
    """At least one exact multiple of chunk (full trailing chunk, where the
    logits intercept is largest) AND one point just past a multiple (trailing
    chunk of 1 token, where it is smallest)."""
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    assert any(s % chunk == 0 for s in pts)
    assert any(s % chunk == 1 for s in pts)


@pytest.mark.parametrize("chunk,growth,s_max", [
    (512, 2, 8192), (512, 2, 131072), (256, 3, 20000),
])
def test_sweep_straddles_a_kv_capacity_doubling(chunk, growth, s_max):
    """Two points whose allocated KV capacity differs -- the step the KV term
    takes. Uses the engine's own capacity function so the test cannot drift
    from the implementation it is checking."""
    from vaayu.vram_plan import kv_capacity_tokens
    pts = probe.sweep_points(chunk=chunk, growth=growth, s_max=s_max)
    caps = {kv_capacity_tokens(s, chunk=chunk, growth=growth, buckets=True)
            for s in pts}
    assert len(caps) >= 2


def test_sweep_degenerates_gracefully_when_the_ceiling_is_tiny():
    """A ceiling below one chunk still yields a usable, in-range sweep."""
    pts = probe.sweep_points(chunk=512, growth=2, s_max=300)
    assert pts and all(0 < s <= 300 for s in pts)


@pytest.mark.parametrize("chunk", [0, -1, -512])
def test_sweep_raises_fast_on_nonpositive_chunk_instead_of_hanging(chunk):
    """chunk=0 is the documented 'no chunking' sentinel (prefill_chunk_size in
    vaayu_base.py) -- the chunk-boundary loop's `k * chunk` never clears
    `s_max` at chunk<=0, so this must raise before entering that loop rather
    than spin forever. Bounded wall-clock proves it raised, not looped."""
    import time

    from vaayu.vram_plan import VramPlanError

    t0 = time.monotonic()
    with pytest.raises(VramPlanError, match="chunk"):
        probe.sweep_points(chunk=chunk, growth=2, s_max=100)
    assert time.monotonic() - t0 < 1.0


@pytest.mark.parametrize("s_max", [0, -1, -100])
def test_sweep_raises_on_nonpositive_s_max(s_max):
    """A non-positive ceiling has no S values to sweep -- fail fast rather
    than silently returning an empty list."""
    from vaayu.vram_plan import VramPlanError

    with pytest.raises(VramPlanError, match="s_max"):
        probe.sweep_points(chunk=512, growth=2, s_max=s_max)
