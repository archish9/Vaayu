"""S17 (S11-L4): a prefill pass READS the expert VRAM tier and never fills it.

Until S17 every per-expert MoE forward set its `_vram` handle to None on a prefill
pass. That gate was written (10a8629) to stop `put()` churn -- prefill routes ~every
expert once -- but one handle served both `get()` and `put()`, so the READ went dark
with it: a prefix HIT's diff re-uploaded experts that were already on the GPU
(measured on Qwen3-Next: 2,738-2,754 of ~9,340 routed pairs at HIT@8k).

Each of the three per-expert forwards gets the same four proofs, because a feature
wired into one forward silently no-ops in the others (CLAUDE.md §0):
  1. a resident expert is served from the tier -- nothing is uploaded for it;
  2. the tier is untouched: no put, no LRU promotion, no decode hit/miss count;
  3. a visit that mixes residents and misses is bit-identical to an all-miss visit;
  4. the miss is not put back.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch as mock_patch

import pytest
import torch

import vaayu.moe_patch as mp
from vaayu.moe_patch import (
    _make_fused_sparse_forward_exl3, detect_moe_blocks, patch_moe_blocks)
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_decode_top_k import _LogitsRouter
from tests.test_moe_patch import (
    MockFusedModel, MockFusedMoEBlock, MockVaayuRef, _make_fused_checkpoint,
    _make_mock_exl3_raw, _patch_modulelist_with_vram)
from tests.test_packed_expert_dequant import _write_packed_fused_checkpoint

BIG = 64 * 1024 * 1024

# (hidden, intermediate, num_experts): a Qwen3-Next-like shape (many small experts)
# and a MiniMax-like one (fewer, larger). Synthetic -- nothing here is a model number.
FUSED_GEOMETRIES = [(32, 64, 16), (64, 128, 8)]
# (num_experts, top_k) for the module-list mock, as tests/test_moe_patch.py uses.
MODULELIST_GEOMETRIES = [(8, 2), (16, 8)]


# ── the cache ────────────────────────────────────────────────────────────────

def _entry(n=4):
    return {r: {"w": torch.zeros(n)} for r in ("gate", "up", "down")}


def test_peek_returns_the_entry_without_promoting_or_counting():
    vram = ExpertVRAMCache(budget_bytes=BIG)
    a, b = _entry(), _entry()
    vram.put(0, 1, a)
    vram.put(0, 2, b)
    order = list(vram.cache)

    assert vram.peek(0, 1) is a
    assert vram.peek(0, 9) is None

    assert list(vram.cache) == order, "peek promoted an entry in the LRU"
    assert (vram.hits, vram.misses) == (0, 0), "peek moved the decode counters"
    assert vram.prefill_hits == 1
    assert "prefill_hit=1" in vram.stats()
