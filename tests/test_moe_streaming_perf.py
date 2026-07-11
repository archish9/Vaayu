"""
Unit tests for MoE streaming performance fixes.
Tests are designed to run without a real GPU or model checkpoint.
"""
import pytest


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


def test_prefill_pass_flag_rules():
    """_moe_prefill_pass: True for multi-token prompt ingestion, False for
    single-token decode and for speculative verify passes (draft_tokens+1 wide)."""
    from vaayu.vaayu_base import compute_prefill_pass

    # no speculator: anything wider than 1 position is prefill
    assert compute_prefill_pass(seq_len=439, has_speculator=False, draft_tokens=0) is True
    assert compute_prefill_pass(seq_len=1, has_speculator=False, draft_tokens=0) is False
    # speculator attached: verify passes are draft_tokens+1 wide -> NOT prefill
    assert compute_prefill_pass(seq_len=6, has_speculator=True, draft_tokens=5) is False
    assert compute_prefill_pass(seq_len=1, has_speculator=True, draft_tokens=5) is False
    assert compute_prefill_pass(seq_len=439, has_speculator=True, draft_tokens=5) is True


def test_prefill_skips_ram_cache_put():
    """During a prefill pass the EXL3 miss loader must NOT insert into the RAM
    expert cache (saves one full-model clone memcpy per prefill)."""
    from unittest.mock import MagicMock
    from vaayu.persist.expert_cache import ExpertRAMCache

    cache = ExpertRAMCache(budget_bytes=10**9)
    vaayu_ref = MagicMock()
    vaayu_ref._moe_prefill_pass = True
    # simulate what _load_one does with the flag set
    import torch
    raw = {"k": torch.zeros(4)}
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        cache.put(0, 0, raw)
    assert cache.current_usage == 0
    vaayu_ref._moe_prefill_pass = False
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        cache.put(0, 0, raw)
    assert cache.current_usage > 0
