"""Fix #7: VRAM expert-cache budget derived from free VRAM (hardware), reserves headroom."""
from vaayu.vaayu_base import compute_vram_expert_budget
GB = 1024 ** 3


def test_reserves_headroom():
    # 8 GB free after model load → budget must leave >=2 GB headroom
    budget = compute_vram_expert_budget(free_vram_bytes=8 * GB)
    assert budget <= 6 * GB
    assert budget >= 1 * GB


def test_tiny_free_vram_yields_zero_or_min():
    budget = compute_vram_expert_budget(free_vram_bytes=1 * GB)
    assert budget >= 0
    assert budget <= 1 * GB
