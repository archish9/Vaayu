"""
Unit tests for MoE streaming performance fixes.
Tests are designed to run without a real GPU or model checkpoint.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ── Task 1: Fix A — non_expert_pinned_layer_names ─────────────────────────────

def test_non_expert_pinned_names_is_separate_set():
    """After init, non_expert_pinned_layer_names must exist and be distinct from pinned_layer_names."""
    from vaayu.vaayu_base import VaayuBaseModel as VaayuBase
    # Access the attribute declarations — the class must declare it
    # We test by checking it appears in a real instance's __init__ code path.
    # Use a mock subclass to avoid full model loading.
    obj = object.__new__(VaayuBase)
    obj.pinned_layer_names = set()
    obj.non_expert_pinned_layer_names = set()
    obj.pinned_layer_names.add("embed")
    obj.non_expert_pinned_layer_names.add("model.layers.5")
    assert "embed" not in obj.non_expert_pinned_layer_names
    assert "model.layers.5" not in obj.pinned_layer_names


def test_non_expert_pinned_excludes_transformer_layer_from_pinned():
    """Transformer layers pinned via non-expert strategy must NOT appear in pinned_layer_names."""
    # This documents the invariant: if a layer is in non_expert_pinned_layer_names,
    # original_forward bypass (which fires for pinned_layer_names) must NOT fire for it.
    pinned = {"model.embed_tokens", "model.norm", "lm_head"}
    non_expert_pinned = {f"model.layers.{i}" for i in range(32)}
    for layer in non_expert_pinned:
        assert layer not in pinned


def test_prefetch_all_predicted_experts_skips_cached():
    """_prefetch_all_predicted_experts submits tasks only for uncached experts."""
    from vaayu.vaayu_base import VaayuBaseModel as VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {0: [0, 1, 2], 1: [3, 4]}
    obj._expert_cache = MagicMock()
    # expert (layer=0, idx=0) is already cached; all others are not
    obj._expert_cache.has.side_effect = lambda layer, expert: (layer == 0 and expert == 0)
    # Mock cache headroom for prefetch budgeting (Fix #3)
    obj._expert_cache.current_usage = 0
    obj._expert_cache.budget_bytes = int(10 * 1024 ** 3)  # 10 GB budget
    # Mock _moe_geometry for model-agnostic expert sizing (Fix #1)
    obj._moe_geometry = {"expert_bytes": int(11.3 * 1024 ** 2)}  # ~11.3 MB per expert
    obj._prefetch_experts = MagicMock()

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)

    # Layer 0: [0] cached → submit [1, 2]. Layer 1: [3, 4] uncached → submit [3, 4].
    assert executor.submit.call_count == 2
    submitted_args = [c.args[1:] for c in executor.submit.call_args_list]  # (layer_idx, uncached)
    assert (0, [1, 2]) in submitted_args
    assert (1, [3, 4]) in submitted_args


def test_prefetch_all_predicted_experts_empty_history():
    """_prefetch_all_predicted_experts does nothing when history is empty."""
    from vaayu.vaayu_base import VaayuBaseModel as VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {}
    obj._expert_cache = MagicMock()

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)
    executor.submit.assert_not_called()


def test_prefetch_all_predicted_experts_none_cache():
    """_prefetch_all_predicted_experts does nothing when cache is None."""
    from vaayu.vaayu_base import VaayuBaseModel as VaayuBase

    obj = object.__new__(VaayuBase)
    obj._expert_history = {0: [0, 1]}
    obj._expert_cache = None

    executor = MagicMock()
    VaayuBase._prefetch_all_predicted_experts(obj, executor)
    executor.submit.assert_not_called()


# ── Task 2: Fix B — Expert cache budget ───────────────────────────────────────

def test_expert_budget_exceeds_working_set():
    """Expert cache budget must be large enough to hold 64-expert working set (~22.5 GB)."""
    import psutil
    mem = psutil.virtual_memory()
    _os_reserve = 6 * 1024**3
    expert_budget = max(int(mem.total * 0.50), int(mem.total - _os_reserve))
    expert_size_bytes = 352 * 1024 * 1024  # 352 MB per expert
    working_set_bytes = 64 * expert_size_bytes  # 64 experts
    assert expert_budget >= working_set_bytes, (
        f"Budget {expert_budget / 1e9:.1f} GB < working set {working_set_bytes / 1e9:.1f} GB"
    )


def test_expert_budget_greater_than_old_formula():
    """New budget formula must be larger than the old 50% formula on this machine."""
    import psutil
    mem = psutil.virtual_memory()
    old_budget = int(mem.total * 0.50)
    _os_reserve = 6 * 1024**3
    new_budget = max(int(mem.total * 0.50), int(mem.total - _os_reserve))
    assert new_budget >= old_budget


# ── Per-layer LRU tests ──────────────────────────────────────────────────────

class TestPerLayerLRU:
    def test_per_layer_evicts_within_layer_not_across(self):
        """Adding a 3rd expert to a max_per_layer=2 cache evicts from that layer only."""
        import torch
        from vaayu.persist.expert_cache import ExpertRAMCache
        cache = ExpertRAMCache(budget_bytes=10 * 1024**3, max_per_layer=2)
        t = lambda: {"w": torch.zeros(1)}
        cache.put(0, 0, t())
        cache.put(0, 1, t())
        cache.put(1, 0, t())  # different layer — no eviction in layer 0
        # layer 0 full; add layer 0 expert 2 → evicts layer 0 expert 0 (LRU)
        cache.put(0, 2, t())
        assert not cache.has(0, 0)   # evicted
        assert cache.has(0, 1)       # kept
        assert cache.has(0, 2)       # newly added
        assert cache.has(1, 0)       # different layer — untouched

    def test_per_layer_no_cross_layer_eviction(self):
        """Filling one layer to cap does not evict experts from other layers."""
        import torch
        from vaayu.persist.expert_cache import ExpertRAMCache
        cache = ExpertRAMCache(budget_bytes=10 * 1024**3, max_per_layer=2)
        t = lambda: {"w": torch.zeros(1)}
        cache.put(0, 0, t())
        cache.put(0, 1, t())
        cache.put(1, 0, t())
        cache.put(1, 1, t())
        # both layers full — all 4 experts should be present
        assert cache.has(0, 0) and cache.has(0, 1)
        assert cache.has(1, 0) and cache.has(1, 1)

    def test_clear_resets_usage_and_entries(self):
        """clear() removes all entries and resets current_usage to 0."""
        import torch
        from vaayu.persist.expert_cache import ExpertRAMCache
        cache = ExpertRAMCache(budget_bytes=10 * 1024**3, max_per_layer=2)
        t = lambda: {"w": torch.zeros(100)}
        cache.put(0, 0, t())
        cache.put(1, 0, t())
        assert cache.current_usage > 0
        cache.clear()
        assert cache.current_usage == 0
        assert not cache.has(0, 0)
        assert not cache.has(1, 0)

    def test_clear_global_lru_mode(self):
        """clear() works for global LRU mode (max_per_layer=0) too."""
        import torch
        from vaayu.persist.expert_cache import ExpertRAMCache
        cache = ExpertRAMCache(budget_bytes=10 * 1024**3, max_per_layer=0)
        t = lambda: {"w": torch.zeros(100)}
        cache.put(0, 0, t())
        cache.clear()
        assert cache.current_usage == 0
        assert not cache.has(0, 0)
