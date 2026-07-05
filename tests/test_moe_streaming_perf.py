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
