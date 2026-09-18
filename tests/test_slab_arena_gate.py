"""The pinned slab arena must only be allocated for the code path that can
actually use it.

Zero-clone slab staging runs through `_dispatch_expert_misses` (acquire_slab /
load_expert_into_slab / put_slab). Its callers: the fused-EXL3 forward, the
module-list forward, and -- since row 30-P (2026-09-18) -- the NON-EXL3 fused
forward. Only the fused-EXL3 forward consumes the prefill stager's staged layers.

Allocating the arena for a layout whose forward never reads it double-spends
the expert budget: once as pinned, unswappable slab memory nothing ever reads,
and again as the cache's own clone budget. Measured on this 32 GB box with
MiniMax-M2: 9.19 GB arena + up to 9.13 GB of clones for a cache specified as 9.13 GB.
"""
import pytest

from vaayu.vaayu_base import (
    moe_supports_slab_arena,
    moe_supports_prefill_stager,
)


def test_exl3_fused_supports_both():
    """The fused-EXL3 forward uses the arena AND consumes staged layers."""
    assert moe_supports_slab_arena(["fused"] * 91, is_exl3=True) is True
    assert moe_supports_prefill_stager(["fused"] * 91, is_exl3=True) is True


def test_exl3_module_list_supports_arena_not_stager():
    """Task 4/5's shared loader routes module-list through
    acquire_slab/load_expert_into_slab/put_slab, so it can use the arena. It
    still passes staged_layer=None and never reads _prefill_staged, so staging
    for it would be pure wasted I/O."""
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=True) is True
    assert moe_supports_prefill_stager(["module_list"] * 62, is_exl3=True) is False


def test_non_exl3_fused_on_cuda_supports_arena_not_stager():
    """Row 30-P: the non-EXL3 fused forward reads the arena, and on CUDA the
    pinned slabs are what make its miss uploads async (+10.62 % decode tok/s on
    Qwen3-Next-80B int4, greedy ids identical). It never reads staged layers."""
    assert moe_supports_slab_arena(["fused"] * 48, is_exl3=False, cuda=True) is True
    assert moe_supports_prefill_stager(["fused"] * 48, is_exl3=False) is False


def test_non_exl3_fused_without_cuda_gets_no_arena():
    """CPU compute has no host-to-device copy to make async; pinning RAM for it
    would buy nothing."""
    assert moe_supports_slab_arena(["fused"] * 48, is_exl3=False, cuda=False) is False
    assert moe_supports_slab_arena(["fused"] * 48, is_exl3=False) is False


def test_non_exl3_module_list_supports_neither():
    """Never measured with an arena on a non-EXL3 checkpoint (none on this
    box) -- an unverified layout must not be assumed slab-capable."""
    for cuda in (True, False):
        assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=False, cuda=cuda) is False
    assert moe_supports_prefill_stager(["module_list"] * 62, is_exl3=False) is False


@pytest.mark.parametrize("cuda", [True, False])
def test_exl3_answers_do_not_depend_on_the_cuda_flag(cuda):
    assert moe_supports_slab_arena(["fused"] * 91, is_exl3=True, cuda=cuda) is True
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=True, cuda=cuda) is True
    assert moe_supports_slab_arena(["fused", "module_list"], is_exl3=True, cuda=cuda) is False
    assert moe_supports_slab_arena(["something_new"], is_exl3=True, cuda=cuda) is False


def test_the_engine_passes_the_compute_device_to_the_arena_gate():
    """Source-level wiring guard, same style as the conflated-gate test below:
    a call site that forgets cuda= would silently keep 30-P off."""
    import inspect
    from vaayu import vaayu_base
    src = inspect.getsource(vaayu_base)
    assert ("moe_supports_slab_arena(_styles_detected, _is_exl3, "
            "cuda=str(self.running_device).startswith(\"cuda\"))") in src


def test_mixed_layouts_support_neither():
    """A partially-usable arena is not worth the pinned RAM."""
    assert moe_supports_slab_arena(["fused", "module_list"], is_exl3=True) is False
    assert moe_supports_prefill_stager(["fused", "module_list"], is_exl3=True) is False


def test_unknown_layout_supports_neither():
    """An unrecognised style must not be assumed slab-capable."""
    assert moe_supports_slab_arena(["something_new"], is_exl3=True) is False
    assert moe_supports_prefill_stager(["something_new"], is_exl3=True) is False


def test_no_blocks_supports_neither():
    assert moe_supports_slab_arena([], is_exl3=True) is False
    assert moe_supports_slab_arena([], is_exl3=False) is False
    assert moe_supports_prefill_stager([], is_exl3=True) is False
    assert moe_supports_prefill_stager([], is_exl3=False) is False


@pytest.mark.parametrize("styles,is_exl3,cuda", [
    (["fused"], True, False), (["module_list"], True, False), ([], False, False),
    (["fused", "module_list"], True, False),
    (["fused"], False, True), (["module_list"], False, True),
])
def test_returns_real_bool(styles, is_exl3, cuda):
    """Callers use `is True` / `is False`; a truthy non-bool would break them."""
    assert isinstance(moe_supports_slab_arena(styles, is_exl3, cuda=cuda), bool)
    assert isinstance(moe_supports_prefill_stager(styles, is_exl3), bool)


def test_capability_attributes_are_independent():
    """The two attributes must come from different predicates. A model whose
    forward uses the arena but not the stager (module-list) must get an arena
    and no stager — the exact combination the single old gate could not express."""
    from vaayu.vaayu_base import (
        moe_supports_slab_arena, moe_supports_prefill_stager,
    )

    module_list = ["module_list"] * 62
    assert moe_supports_slab_arena(module_list, is_exl3=True) is True
    assert moe_supports_prefill_stager(module_list, is_exl3=True) is False

    fused = ["fused"] * 91
    assert moe_supports_slab_arena(fused, is_exl3=True) is True
    assert moe_supports_prefill_stager(fused, is_exl3=True) is True


def test_no_call_site_uses_a_single_conflated_gate():
    """Regression guard for the staleness this work fixes: the arena guard and
    the stager guard must read DIFFERENT attributes, and each must actually be
    wired to its own predicate.

    Asserting merely that the predicate names appear somewhere in the module
    source is inert — it's also satisfied by the functions' own `def` lines.
    Instead this pins down the literal CALL SITES: the two assignments that
    wire each attribute to its predicate (so unwiring one to a hardcoded
    `False` breaks this test), and the two guard checks that consume the
    attributes (so pointing both guards at the same attribute again — the
    exact conflation this work fixed — breaks this test)."""
    import inspect
    from vaayu import vaayu_base

    src = inspect.getsource(vaayu_base)
    # Each attribute must be assigned from its OWN predicate call.
    assert "self._moe_slab_capable = moe_supports_slab_arena(" in src, \
        "arena attribute not wired to moe_supports_slab_arena()"
    assert "self._moe_stager_capable = moe_supports_prefill_stager(" in src, \
        "stager attribute not wired to moe_supports_prefill_stager()"
    # Each guard must read its OWN attribute, not the other's.
    assert "if not self._moe_slab_capable:" in src, \
        "arena guard does not check _moe_slab_capable"
    assert 'getattr(self, "_moe_stager_capable", False)' in src, \
        "stager guard does not check _moe_stager_capable"
    assert "moe_supports_slab_pipeline" not in src, "old conflated gate still referenced"
