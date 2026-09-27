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


def test_non_exl3_module_list_on_cuda_supports_arena_not_stager():
    """S5-M (2026-09-26): without the arena this layout caches through
    ExpertRAMCache.put(), cloning every miss twice, and glibc malloc kept the
    evicted clones' pages. Measured on DeepSeek-V4-Flash: RSS grew ~0.19
    GB/token past a flat expert cache (live tensors flat, malloc free space
    2.52 -> 5.51 GB, malloc_trim(0) handed back 5.5 GB) and the server was
    OOM-killed on its 2nd request. On the arena RSS stayed flat. It never reads
    staged layers."""
    styles = ["module_list"] * 43
    assert moe_supports_slab_arena(styles, is_exl3=False, cuda=True) is True
    assert moe_supports_prefill_stager(styles, is_exl3=False) is False


def test_non_exl3_module_list_without_cuda_gets_no_arena():
    """Same reason as the non-EXL3 fused layout: CPU compute has no
    host-to-device copy to make async."""
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=False, cuda=False) is False
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=False) is False


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


# ---------------------------------------------------------------------------
# 47-m5: a whole-layer-dispatch model must not be charged for the arena.
#
# `moe_supports_slab_arena` answers from the expert STYLE, and a fused-3D
# checkpoint's style is legitimately "fused" -- so the predicate says yes while
# `_make_sparse_moe_forward` goes on to install `_make_whole_layer_sparse_forward`,
# which takes no expert_cache argument at all and reads neither cache. Measured
# on gpt-oss-20b (2026-09-21): a 18.45 GB pinned arena with
# `expert_cache=0hit/0miss/0evict ... usage=0.00GB/18.31GB` for every token of a
# 256-token generation, on a 32 GB box.
#
# The condition that decides the dispatch has TWO parts and both are load-bearing
# (moe_patch.py's own dispatch comment, measured 2026-09-21). `dispatches_whole_layer`
# is the one place it is written, so the guard and the dispatch cannot drift apart.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

from vaayu.moe_patch import MoEBlockInfo, dispatches_whole_layer


class _FakeIndex:
    """Only the attribute the dispatch reads."""

    def __init__(self, fused_layer_tensors):
        self.fused_layer_tensors = fused_layer_tensors


def _block(layer_idx, num_experts, fused_param_shape, expert_style="fused"):
    """A block whose experts module reports what the real one reports.

    `fused_param_shape` is the shape of the module's own direct parameter, which
    is what `fused_param_experts` reads -- never a name, never an architecture
    string (CLAUDE.md §5).
    """
    experts = nn.Module()
    experts.num_experts = num_experts
    if fused_param_shape is not None:
        experts.register_parameter(
            "gate_up_proj", nn.Parameter(torch.empty(fused_param_shape), requires_grad=False))
    moe = nn.Module()
    moe.experts = experts
    moe.gate = nn.Linear(8, num_experts, bias=False)
    return MoEBlockInfo(
        layer_idx=layer_idx, layer_name=f"model.layers.{layer_idx}",
        module_path=f"model.layers.{layer_idx}.mlp", moe_module=moe,
        router_attr="gate", experts_attr="experts", expert_style=expert_style,
    )


# Two contrasting geometries, per CLAUDE.md §5 -- and they are contrasting in
# exactly the axis that decides this, not merely in size.
def _gptoss_like(layer_idx=0):
    """Fused-3D: the module reports fused params AND the checkpoint stores the
    layer whole, so the index carries fused_layer_tensors for it. 32 experts."""
    return _block(layer_idx, 32, (32, 5760, 90))


def _qwen3next_like(layer_idx=0):
    """The measured FALSE POSITIVE: Qwen3NextExperts reports fused parameters
    (gate_up_proj [512, 1024, 2048]) but its checkpoint stores ONE TENSOR PER
    EXPERT, so the index holds no whole-layer metas and the tuned per-expert
    fused forward -- which DOES read the arena -- must keep it."""
    return _block(layer_idx, 512, (512, 1024, 2048))


def _minimax_like(layer_idx=0):
    """Module-list: 256 per-expert submodules, no direct fused parameter."""
    return _block(layer_idx, 256, None, expert_style="module_list")


def test_whole_layer_needs_both_the_index_metas_and_the_module_shape():
    b = _gptoss_like()
    with_metas = _FakeIndex({b.layer_name: {"gate_up_proj_blocks": object()}})
    assert dispatches_whole_layer(b, with_metas) is True
    # Same module, no whole-layer metas -> per-expert streaming, arena needed.
    assert dispatches_whole_layer(b, _FakeIndex({})) is False


def test_fused_params_alone_do_not_make_it_whole_layer():
    """Qwen3-Next reports fused params too. If this ever returns True its arena
    disappears and every miss upload goes pageable (row 30-P: -10.62 % decode)."""
    b = _qwen3next_like()
    assert dispatches_whole_layer(b, _FakeIndex({})) is False


def test_module_list_is_never_whole_layer():
    b = _minimax_like()
    idx = _FakeIndex({b.layer_name: {"anything": object()}})
    assert dispatches_whole_layer(b, idx) is False


def test_missing_index_is_never_whole_layer():
    """A model loaded without an offset index must not lose its arena."""
    assert dispatches_whole_layer(_gptoss_like(), None) is False


def test_returns_real_bool_for_whole_layer():
    b = _gptoss_like()
    idx = _FakeIndex({b.layer_name: {"gate_up_proj_blocks": object()}})
    assert isinstance(dispatches_whole_layer(b, idx), bool)
    assert isinstance(dispatches_whole_layer(b, _FakeIndex({})), bool)


def test_the_dispatch_and_the_arena_guard_share_one_condition():
    """Drift guard. moe_patch's own dispatch docstring warns that a second copy
    of this table drifts; the condition is written once and BOTH sites call it."""
    import inspect
    from vaayu import moe_patch, vaayu_base

    disp = inspect.getsource(moe_patch._make_sparse_moe_forward)
    assert "dispatches_whole_layer(block_info, offset_index)" in disp, \
        "the dispatch site no longer calls the shared predicate"
    assert "fused_param_experts(" not in disp, \
        "the whole-layer condition was re-inlined at the dispatch site"

    src = inspect.getsource(vaayu_base)
    assert "dispatches_whole_layer(" in src, \
        "the arena guard does not consult the whole-layer predicate"


# ---------------------------------------------------------------------------
# S8: routed fused-3D delivery DOES read the arena, so a model whose blocks take it
# must get one; the self-checks that decide it run before the arena decision and
# before patching (they drive each block's own, unpatched forward).
# ---------------------------------------------------------------------------
def test_routed_self_checks_run_before_the_arena_decision_and_the_patch():
    import inspect
    from vaayu import vaayu_base
    src = inspect.getsource(vaayu_base)
    i_check = src.index("self._routed_fused3d_ok = run_routed_self_checks(")
    i_decide = src.index("self._moe_whole_layer_only = ")
    i_patch = src.index("self._moe_blocks = patch_moe_blocks(")
    assert i_check < i_decide < i_patch


def test_a_routed_block_is_not_whole_layer_only():
    import inspect
    from vaayu import vaayu_base
    src = inspect.getsource(vaayu_base)
    assert "and not routed_delivery_enabled(b, self._expert_offset_index, self)" in src


def test_the_arena_slab_count_is_capped_at_the_indexed_experts():
    import inspect
    from vaayu import vaayu_base
    src = inspect.getsource(vaayu_base)
    assert "cap_slabs_at_indexed_experts(" in src
    assert "_n_slabs = _n_capped" in src
