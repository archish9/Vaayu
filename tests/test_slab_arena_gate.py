"""The pinned slab arena must only be allocated for the code path that can
actually use it.

Zero-clone slab staging is implemented ONLY by `_make_fused_sparse_forward_exl3`
— it alone calls `acquire_slab`/`put_slab`, and it alone consumes the prefill
stager's `_prefill_staged` layers. The module-list forward (MiniMax-M2) and the
non-EXL3 fused forward populate the cache through `ExpertRAMCache.put()`, which
CLONES into pageable RAM and never touches the arena.

Allocating the arena for those layouts double-spends the expert budget: once as
pinned, unswappable slab memory nothing ever reads, and again as the cache's own
clone budget. Measured on this 32 GB box with MiniMax-M2: 9.19 GB arena +
up to 9.13 GB of clones for a cache specified as 9.13 GB.
"""
import pytest

from vaayu.vaayu_base import moe_supports_slab_pipeline


def test_exl3_fused_uses_slab_pipeline():
    """Qwen3-235B shape: EXL3 + fused experts — the actively-tested path.
    Must keep the arena."""
    assert moe_supports_slab_pipeline(["fused"] * 91, is_exl3=True) is True


def test_exl3_module_list_does_not():
    """MiniMax-M2 shape: EXL3 + module-list experts. The forward never calls
    acquire_slab/put_slab, so the arena is pure dead weight."""
    assert moe_supports_slab_pipeline(["module_list"] * 62, is_exl3=False) is False
    assert moe_supports_slab_pipeline(["module_list"] * 62, is_exl3=True) is False


def test_non_exl3_fused_does_not():
    """The non-EXL3 fused forward (_make_fused_sparse_forward) uses
    load_expert_sparse + ExpertRAMCache.put() — clone path, no slabs."""
    assert moe_supports_slab_pipeline(["fused"] * 32, is_exl3=False) is False


def test_mixed_layouts_do_not():
    """A model whose blocks disagree cannot be served entirely by the slab
    path; refuse rather than half-allocate."""
    assert moe_supports_slab_pipeline(["fused", "module_list"], is_exl3=True) is False


def test_no_blocks_does_not():
    """Detection found nothing (dense, or a layout we don't recognize) —
    never allocate."""
    assert moe_supports_slab_pipeline([], is_exl3=True) is False
    assert moe_supports_slab_pipeline([], is_exl3=False) is False


@pytest.mark.parametrize("styles,is_exl3", [
    (["fused"] * 91, True),
    (["module_list"] * 62, True),
    (["fused"] * 32, False),
    ([], True),
])
def test_returns_real_bool(styles, is_exl3):
    """Callers gate object construction on this; it must be a plain bool, not
    a truthy generator/np.bool_."""
    assert isinstance(moe_supports_slab_pipeline(styles, is_exl3), bool)
