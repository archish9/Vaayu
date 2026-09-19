"""Row 45: the module-list forward keeps a layer's experts across its
layer-major prefill chunk calls.

`_chunked_prefill`'s layer-major path is style-agnostic — it calls every decoder
layer once per 512-token chunk for any MoE model — but only the two fused
forwards retained anything across those calls. Measured on MiniMax-M2.7
(EXL3 + module-list) on 2026-09-19: a 2048-token prefill issued 55,871 expert
loads against a per-layer union of 15,268 (**3.659x**), 495.86 GB read, and the
repeated per-load H2D + dequant + detach was 63 % of a 113.9 s wall.

The entries here are not the fused forward's packed tensor dicts: they are the
GPU tensors plus the EXL3 BC objects `_inject_expert_weights` returns, which the
VRAM expert cache already stores and `_attach_cached_expert` already re-binds
onto a module that `_detach_expert` sent to meta. So retention reuses this
forward's own hit path against a per-layer dict, bounded by the (idle-in-prefill)
VRAM cache budget and dropped at the layer's last chunk call.
"""
import torch

import vaayu.persist.safetensor_model_persister as smp
from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex

from tests.test_moe_patch import MockModel, MockVaayuRef, _make_checkpoint

N_CHUNKS = 4
HIDDEN = 8
NUM_EXPERTS = 16
TOP_K = 8


def _build(tmp_path, *, prefill=True, vram_budget=10**9):
    """A patched module-list MoE block in the state a layer-major prefill sees:
    `_moe_prefill_pass` set, so the VRAM cache itself is gated off. Seeded, so
    two builds on one tmp_path route identically and write the same checkpoint
    (MockModel draws random weights and `_make_checkpoint` writes them out)."""
    torch.manual_seed(0)
    model = MockModel(num_layers=1, hidden=HIDDEN, num_experts=NUM_EXPERTS)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=vram_budget)
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = prefill
    patch_moe_blocks(
        model=model, expert_cache=ExpertRAMCache(budget_bytes=100 * 1024 * 1024),
        offset_index=idx, checkpoint_path=str(tmp_path), top_k=TOP_K,
        device="cpu", dtype=torch.float32, vaayu_ref=ref,
    )
    return model, ref


def _chunks():
    g = torch.Generator().manual_seed(7)
    return [torch.randn(1, 4, HIDDEN, generator=g) for _ in range(N_CHUNKS)]


def _count_loads(monkeypatch):
    seen = []
    real = smp.load_expert_sparse

    def counting(shard_file, metas, *a, **k):
        seen.append(next(iter(metas)))
        return real(shard_file, metas, *a, **k)
    monkeypatch.setattr(smp, "load_expert_sparse", counting)
    return seen


def _layer_major(model, ref, chunks, n_calls):
    """What vaayu_base's layer-major prefill does to one MoE layer: set the
    chunk count, call the layer once per chunk, clear it again."""
    ref._chunk_major_n = n_calls
    outs, routed = [], []
    try:
        for x in chunks:
            outs.append(model.layers[0].block_sparse_moe.forward(x)[0])
            routed.append(set(ref._expert_history[0]))
    finally:
        ref._chunk_major_n = None
    return outs, routed


def test_each_expert_is_read_once_per_layer_not_once_per_chunk(tmp_path, monkeypatch):
    model, ref = _build(tmp_path)
    loads = _count_loads(monkeypatch)

    _outs, routed = _layer_major(model, ref, _chunks(), N_CHUNKS)

    union = set().union(*routed)
    assert sum(map(len, routed)) > len(union), "premise: chunks share experts"
    assert len(loads) == len(union)


def test_retained_chunks_are_bit_identical_to_rereading(tmp_path):
    m_keep, r_keep = _build(tmp_path)
    m_read, r_read = _build(tmp_path)

    kept, _ = _layer_major(m_keep, r_keep, _chunks(), N_CHUNKS)
    # _chunk_major_n None is a single-chunk forward: nothing is retained.
    reread, _ = _layer_major(m_read, r_read, _chunks(), None)

    for a, b in zip(kept, reread):
        assert torch.equal(a, b)


def test_retention_is_dropped_at_the_layers_last_chunk(tmp_path):
    model, ref = _build(tmp_path)
    _layer_major(model, ref, _chunks(), N_CHUNKS)
    assert getattr(ref, "_chunk_major_gpu", None) is None
    assert getattr(ref, "_chunk_major_gpu_bytes", 0) == 0


def test_a_single_chunk_prefill_retains_nothing(tmp_path, monkeypatch):
    model, ref = _build(tmp_path)
    loads = _count_loads(monkeypatch)
    _outs, routed = _layer_major(model, ref, _chunks(), None)
    assert len(loads) == sum(map(len, routed))
    assert getattr(ref, "_chunk_major_gpu", None) is None


def test_decode_retains_nothing(tmp_path):
    """At decode the VRAM expert cache itself is live; retention must not run."""
    model, ref = _build(tmp_path, prefill=False)
    _layer_major(model, ref, _chunks(), N_CHUNKS)
    assert getattr(ref, "_chunk_major_gpu", None) is None


def test_a_budget_smaller_than_one_expert_falls_back_to_rereading(tmp_path, monkeypatch):
    m_tiny, r_tiny = _build(tmp_path, vram_budget=1)
    m_ref, r_ref = _build(tmp_path)
    loads = _count_loads(monkeypatch)

    tiny, routed = _layer_major(m_tiny, r_tiny, _chunks(), N_CHUNKS)
    assert len(loads) == sum(map(len, routed))
    ref_out, _ = _layer_major(m_ref, r_ref, _chunks(), N_CHUNKS)
    for a, b in zip(tiny, ref_out):
        assert torch.equal(a, b)


def test_an_aborted_layer_does_not_leak_into_the_next_prefill(tmp_path, monkeypatch):
    """`_chunked_prefill`'s finally clears `_chunk_major_gpu` when an exception
    unwinds mid-layer. The next prefill of that same layer must start clean:
    count its own chunk calls from one and read each expert once."""
    model, ref = _build(tmp_path)
    chunks = _chunks()
    ref._chunk_major_n = N_CHUNKS
    model.layers[0].block_sparse_moe.forward(chunks[0])
    model.layers[0].block_sparse_moe.forward(chunks[1])
    ref._chunk_major_n = None
    ref._chunk_major_gpu = None           # what _chunked_prefill's finally does
    ref._chunk_major_gpu_bytes = 0

    loads = _count_loads(monkeypatch)
    _outs, routed = _layer_major(model, ref, chunks, N_CHUNKS)
    assert len(loads) == len(set().union(*routed))
    assert getattr(ref, "_chunk_major_gpu", None) is None
