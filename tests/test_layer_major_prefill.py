"""Layer-major chunked prefill: past-length bookkeeping + staged multi-consume."""
import threading
import types

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel, FP8DynamicCache, compute_chunk_past_lens, compute_pos_range


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
    from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
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
    # budget for chunk-major GPU retention. A real, empty cache: prefill reads
    # the VRAM tier (S17) and never fills it, so only budget_bytes matters here.
    vaayu_ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=10 ** 9)

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


class _FakeSelfForChunkedPrefill:
    """Minimal stand-in for VaayuBaseModel, providing only what
    _chunked_prefill's legacy (dense) per-chunk loop reads: prefill_chunk_size
    and a forward() method. is_moe/_active_attn_impl are intentionally absent
    -- getattr(self, 'is_moe', False) short-circuits the MoE layer-major
    branch's `and self._active_attn_impl in (...)` check before it's ever
    evaluated, so this object correctly forces the legacy loop without
    needing to fake attention-impl state."""

    def __init__(self, chunk_size):
        self.prefill_chunk_size = chunk_size
        self.forward_calls = []  # past_key_values object seen on each chunk call

    def forward(self, input_ids, attention_mask, past_key_values, use_cache,
                output_attentions, output_hidden_states, return_dict):
        self.forward_calls.append(past_key_values)
        # Simulate a real layer-0 KV append (single-layer model) so the cache's
        # own accumulation logic (FP8DynamicCache.update) runs for real, proving
        # the object threaded in is genuinely mutated -- not a stand-in.
        seq_len = input_ids.shape[1]
        key = torch.randn(1, 1, seq_len, 4)
        value = torch.randn(1, 1, seq_len, 4)
        past_key_values.update(key, value, layer_idx=0)
        return types.SimpleNamespace(
            logits=torch.zeros(1, seq_len, 10),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


def test_chunked_prefill_threads_caller_cache_through_legacy_loop():
    """Bug repro (vaayu_base.py _chunked_prefill): the legacy per-chunk loop
    used to hardcode `past_key_values = None` as its loop-local starting
    value, discarding whatever real cache object the caller (generate())
    passed in via _chunked_prefill's own (newly added) past_key_values
    parameter. HF's own generation loop still worked (it picks up whatever
    object forward() returns), but generate()'s `kv` variable -- passed to
    pc.store() and the session's enqueue_append hook -- stayed bound to the
    original, empty, never-populated cache. That produced a real crash
    (IndexError: list index out of range on cache.key_cache[0]) for any
    cold-start prompt long enough to trigger chunked prefill.

    This test proves the fix: the caller-supplied cache object is (a) the
    exact object handed to forward() on every chunk -- not None, not a
    replacement -- and (b) the exact object returned at the end, genuinely
    mutated with real per-chunk K/V data.
    """
    fake_self = _FakeSelfForChunkedPrefill(chunk_size=4)
    caller_cache = FP8DynamicCache()
    total_len = 10  # chunk_size=4 -> chunks of [4, 4, 2], 3 iterations
    input_ids = torch.arange(total_len).unsqueeze(0)

    output = VaayuBaseModel._chunked_prefill(
        fake_self,
        input_ids=input_ids,
        attention_mask=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        past_key_values=caller_cache,
    )

    assert len(fake_self.forward_calls) == 3
    # Every chunk -- including the first -- must see the caller's real object,
    # never None (the bug) and never a fresh replacement.
    for call_cache in fake_self.forward_calls:
        assert call_cache is caller_cache

    # Object IDENTITY: what generate() would see as `kv` after this call is
    # the SAME object it originally created/looked up.
    assert output.past_key_values is caller_cache

    # And it was genuinely mutated -- the crash-causing symptom was an empty
    # key_cache despite a "successful" chunked prefill.
    assert len(caller_cache.key_cache) > 0
    assert caller_cache.get_seq_length() == total_len


class _FakeSelfForAbortedLayerMajor:
    """Stand-in that takes _chunked_prefill's MoE layer-major branch and then
    aborts inside forward(), having already retained experts -- exactly the
    state moe_patch.py leaves behind when its `if _cm_last_call:` teardown is
    skipped by an exception (the teardown sits after the forward's finally, in
    all three MoE forwards)."""

    prefill_chunk_size = 2
    is_moe = True
    _active_attn_impl = "sdpa"
    _state_continuation_layers = 0

    def forward(self, **kwargs):
        # What the MoE forward has done by the time an expert injection raises:
        # a layer's retention dict is live and its byte count is charged.
        self._chunk_major_gpu = {0: {"weight": object()}}
        self._chunk_major_gpu_bytes = 12_345_678
        raise RuntimeError("expert injection aborted mid-layer")


def test_aborted_layer_major_prefill_clears_chunk_major_retention():
    """All three MoE forwards reset `_chunk_major_gpu`/`_chunk_major_gpu_bytes`
    OUTSIDE their own try/finally (moe_patch.py:1995, 2431, 3196), so an
    exception unwinding mid-layer skips it. What makes that safe is this
    finally, and nothing else: `_chunk_major_n` is set here and only here
    (vaayu_base.py:4741), so retention cannot be live outside this try.

    Both forwards' comments name this finally as the reason a missing dict
    always starts a fresh count, and both retention suites hand-simulate it
    ("what _chunked_prefill's finally does") rather than run it. This test runs
    it: if it is ever removed, a stale byte count would pre-charge every later
    prefill's budget check and silently shrink row 45's retention win."""
    fake_self = _FakeSelfForAbortedLayerMajor()
    input_ids = torch.arange(6).unsqueeze(0)  # chunk_size=2 -> 3 chunks

    with pytest.raises(RuntimeError, match="aborted mid-layer"):
        VaayuBaseModel._chunked_prefill(
            fake_self,
            input_ids=input_ids,
            attention_mask=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            past_key_values=None,
        )

    assert fake_self._chunk_major_gpu is None, "retained GPU experts survived the abort"
    assert fake_self._chunk_major_gpu_bytes == 0, "stale byte count survived the abort"
    # The call counter and chunk lengths gate retention on; a survivor would let
    # a later non-chunked forward think it is mid-prefill.
    assert fake_self._chunk_major_n is None
    assert fake_self._chunk_major_lens is None
