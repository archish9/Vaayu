"""Layer-major chunked prefill: past-length bookkeeping + staged multi-consume."""
import threading

import pytest
import torch

from vaayu.vaayu_base import compute_chunk_past_lens, compute_pos_range


def test_chunk_past_lens_basic():
    assert compute_chunk_past_lens([512, 512, 200], past0=0) == [0, 512, 1024]
    assert compute_chunk_past_lens([100], past0=7) == [7]
    assert compute_chunk_past_lens([], past0=0) == []


def test_chunk_past_lens_matches_cumulative_sum():
    lens = [3, 5, 2, 9]
    out = compute_chunk_past_lens(lens, past0=11)
    acc = 11
    for L, o in zip(lens, out):
        assert o == acc
        acc += L


def test_pos_range_covers_multi_chunk_total_past_max_seq_len():
    # Bug repro: a 1896-token prompt chunked into [512, 512, 512, 360] with
    # max_seq_len=512 (default) must size the RoPE cache to cover all 1896
    # positions, not just max_seq_len -- else later chunks slice past the end.
    assert compute_pos_range(max_seq_len=512, past_len=0, seq_total=1896) == 1896


def test_pos_range_floor_is_max_seq_len_for_short_sequences():
    # A short single-chunk prefill (< max_seq_len) must still get a cache sized
    # to at least max_seq_len (matches prior behavior for the common case).
    assert compute_pos_range(max_seq_len=512, past_len=0, seq_total=100) == 512


def test_pos_range_adds_existing_past():
    # Reused/decode case: past_len (already-cached tokens) plus this call's
    # new tokens must both be covered.
    assert compute_pos_range(max_seq_len=512, past_len=1024, seq_total=1) == 1025


def test_staged_layer_survives_all_chunks_and_retains_gpu(tmp_path):
    """Chunk-major prefill runs a layer's sparse_forward once per chunk. The
    staged dict must be PEEKED (not popped) and the stage semaphore held until
    the LAST chunk's call, and the layer's GPU expert tensors retained across
    chunks then freed at the last call (Steps 6 + 6b)."""
    import types
    from pathlib import Path
    from unittest.mock import MagicMock, patch as mock_patch

    # Reuse the real EXL3 fixtures from the moe_patch test module (pytest's
    # prepend import mode puts tests/ on sys.path).
    from tests.test_moe_patch import (
        MockVaayuRef, MockFusedModel, _make_fused_exl3_checkpoint,
        _make_mock_exl3_raw,
    )
    from vaayu.moe_patch import (
        _make_fused_sparse_forward_exl3, detect_moe_blocks,
    )
    from vaayu.persist.expert_cache import ExpertRAMCache
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex

    hidden, out_dim, num_experts = 8, 16, 8
    model = MockFusedModel(num_layers=1, hidden=hidden)
    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=hidden,
                                out_dim=out_dim, num_experts=num_experts)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    block_info = detect_moe_blocks(model)[0]

    vaayu_ref = MockVaayuRef()
    vaayu_ref._moe_prefill_pass = True
    vaayu_ref._prefill_stager_active = True
    vaayu_ref._chunk_major_n = 2                       # two chunks → two calls
    vaayu_ref._staged_consume_count = {}
    # Stage EVERY expert of layer 0 so any routed eidx is a staged hit (no disk).
    vaayu_ref._prefill_staged = {
        0: {e: _make_mock_exl3_raw("layers.0", e, in_dim=hidden, out_dim=out_dim)
            for e in range(num_experts)}
    }
    _ev = threading.Event(); _ev.set()
    vaayu_ref._prefill_stage_events = {0: _ev}
    vaayu_ref._prefill_staged_slabs = {}
    sem = threading.Semaphore(1)
    assert sem.acquire(blocking=False) is True          # hold the one slot
    vaayu_ref._prefill_stage_sem = sem
    # budget for chunk-major GPU retention (prefill gates the real VRAM cache off,
    # so this is only read for its budget_bytes bound).
    vaayu_ref._expert_vram_cache = types.SimpleNamespace(budget_bytes=10 ** 9)

    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = (
        lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16))

    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", return_value=mock_bc):
        fwd = _make_fused_sparse_forward_exl3(
            block_info=block_info,
            expert_cache=cache,
            offset_index=idx,
            checkpoint_path=Path(str(tmp_path)),
            top_k=2,
            device="cpu",
            dtype=torch.float32,
            vaayu_ref=vaayu_ref,
        )
        hidden_states = torch.randn(1, 4, hidden)

        # ── call 1 (not the last chunk) ─────────────────────────────────────
        fwd(hidden_states)
        assert 0 in vaayu_ref._prefill_staged, "staged dict popped on first call"
        assert sem.acquire(blocking=False) is False, "semaphore released too early"
        assert isinstance(vaayu_ref._chunk_major_gpu, dict) and vaayu_ref._chunk_major_gpu, \
            "GPU experts not retained after first chunk"

        # ── call 2 (the last chunk) ─────────────────────────────────────────
        fwd(hidden_states)
        assert 0 not in vaayu_ref._prefill_staged, "staged dict not popped on last call"
        assert vaayu_ref._chunk_major_gpu is None, "retained GPU experts not freed at last call"
        assert sem.acquire(blocking=False) is True, "semaphore not released on last call"
