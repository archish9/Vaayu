"""D1 Task 5: load-time WARNING when the prefix budget cannot hold ONE entry.

The pool keeps one entry whatever the budget (prefix_cache.store()'s eviction loop
is `while current > budget and len(pool) > 1`), so a budget below one full-length
entry means the pool lives permanently at its floor and swaps. This announces it
at load, before the first request. Entry size is DERIVED from the planner's own
figure (kv_bytes_per_token x max_seq_len), never a literal or a cgroup delta.
"""
import pytest

from vaayu.vaayu_base import (_PREFIX_CACHE_RAM_RESERVE,
                              compute_prefix_cache_budget, prefix_floor_warning)
from vaayu.vram_plan import kv_bytes_per_token

# SYNTHETIC geometries (layers that hold per-token KV, kv heads, head_dim, ctx).
# MiniMax-like: 62 layers. Qwen3-Next-like: hybrid, 12 of 48 layers hold KV.
_GEOS = {
    "minimax": (62, 8, 128, 32768),
    "qwen3next": (12, 2, 256, 32768),
}


def _entry(geo):
    layers, heads, hd, ctx = geo
    return ctx * kv_bytes_per_token(num_hidden_layers=layers, num_key_value_heads=heads,
                                    head_dim=hd, kv_cache_mode="bf16")


@pytest.mark.parametrize("name", list(_GEOS))
def test_fires_exactly_when_budget_below_one_entry(name):
    entry = _entry(_GEOS[name])
    msg = prefix_floor_warning(entry - 1, entry, compressed=False)
    assert msg and "WARNING" in msg
    assert f"{entry / 1e9:.2f} GB" in msg          # derived figure is quoted
    assert prefix_floor_warning(entry, entry, compressed=False) is None
    assert prefix_floor_warning(entry * 3, entry, compressed=False) is None


@pytest.mark.parametrize("name", list(_GEOS))
def test_budget_zero_leaves_the_existing_warning_alone(name):
    # The budget-0 warning already covers this load; the two must not both fire.
    assert prefix_floor_warning(0, _entry(_GEOS[name]), compressed=False) is None


def test_unknown_entry_size_does_not_warn():
    assert prefix_floor_warning(1, 0, compressed=False) is None


def test_compressed_figure_is_labelled_uncompressed():
    msg = prefix_floor_warning(1, 10 ** 9, compressed=True)
    assert "uncompressed" in msg


@pytest.mark.parametrize("free", [4 * 1024 ** 3, 20 * 1024 ** 3, 40 * 1024 ** 3])
def test_budget_value_unchanged_no_arena_resubtraction(free):
    assert compute_prefix_cache_budget(free) == max(0, free - _PREFIX_CACHE_RAM_RESERVE), (
        "compute_prefix_cache_budget no longer equals free RAM minus the reserve. "
        "Someone subtracted the expert arena here: it is ALREADY pinned and excluded "
        "from effective_available_bytes() when this runs, so subtracting it again "
        "double-counts ~12 GB (D1 Task 5 re-scope).")
