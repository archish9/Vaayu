"""Row 44 F2: the non-EXL3 fused forward keeps a layer's experts across its
layer-major prefill chunk calls.

The layer-major chunked prefill calls every decoder layer once per 512-token
chunk, back to back. Until row 44 this forward re-routed and re-read its experts
on every one of those calls: a 2k Qwen3-Next prefill loaded 53,938 experts
against a per-layer union of 17,445 (3.09x), and every per-load phase -- disk,
H2D, dequant, 84 % of the prefill -- paid it. The fused-EXL3 forward already
retains (`_chunk_major_gpu`); this is the same mechanism for this forward:
misses uploaded on an earlier chunk call are kept as the same packed entry the
VRAM expert cache stores, served through the forward's hit branch, and dropped
at the layer's last chunk call. The VRAM cache itself stays gated off in
prefill; its (idle) budget bounds the retention, as in the EXL3 forward.
"""
import torch

import vaayu.persist.safetensor_model_persister as smp
from tests.test_fused_nonexl3_slab_arena import HIDDEN, _build, _checkpoint

N_CHUNKS = 4


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
            outs.append(model.layers[0].mlp.forward(x))
            routed.append(set(ref._expert_history[0]))
    finally:
        ref._chunk_major_n = None
    return outs, routed


def test_each_expert_is_read_once_per_layer_not_once_per_chunk(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    model, ref, _ram = _build(tmp_path, idx, arena=True, prefill=True)
    loads = _count_loads(monkeypatch)

    _outs, routed = _layer_major(model, ref, _chunks(), N_CHUNKS)

    union = set().union(*routed)
    assert sum(map(len, routed)) > len(union), "premise: chunks share experts"
    assert len(loads) == len(union)


def test_retained_chunks_are_bit_identical_to_rereading(tmp_path):
    idx = _checkpoint(tmp_path)
    m_keep, r_keep, _ = _build(tmp_path, idx, arena=True, prefill=True)
    m_read, r_read, _ = _build(tmp_path, idx, arena=True, prefill=True)

    kept, _ = _layer_major(m_keep, r_keep, _chunks(), N_CHUNKS)
    # _chunk_major_n None is a single-chunk forward: nothing is retained.
    reread, _ = _layer_major(m_read, r_read, _chunks(), None)

    for a, b in zip(kept, reread):
        assert torch.equal(a, b)


def test_retention_is_dropped_at_the_layers_last_chunk(tmp_path):
    idx = _checkpoint(tmp_path)
    model, ref, _ = _build(tmp_path, idx, arena=True, prefill=True)
    _layer_major(model, ref, _chunks(), N_CHUNKS)
    assert getattr(ref, "_chunk_major_gpu", None) is None
    assert getattr(ref, "_chunk_major_gpu_bytes", 0) == 0


def test_a_single_chunk_prefill_retains_nothing(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    model, ref, _ = _build(tmp_path, idx, arena=True, prefill=True)
    loads = _count_loads(monkeypatch)
    _outs, routed = _layer_major(model, ref, _chunks(), None)
    assert len(loads) == sum(map(len, routed))
    assert getattr(ref, "_chunk_major_gpu", None) is None


def test_decode_retains_nothing(tmp_path):
    idx = _checkpoint(tmp_path)
    model, ref, _ = _build(tmp_path, idx, arena=True, prefill=False)
    _layer_major(model, ref, _chunks(), N_CHUNKS)
    assert getattr(ref, "_chunk_major_gpu", None) is None


def test_a_budget_smaller_than_one_expert_falls_back_to_rereading(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    m_tiny, r_tiny, _ = _build(tmp_path, idx, arena=True, prefill=True, vram_budget=1)
    m_ref, r_ref, _ = _build(tmp_path, idx, arena=True, prefill=True)
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
    idx = _checkpoint(tmp_path)
    model, ref, _ = _build(tmp_path, idx, arena=True, prefill=True)
    chunks = _chunks()
    ref._chunk_major_n = N_CHUNKS
    model.layers[0].mlp.forward(chunks[0])
    model.layers[0].mlp.forward(chunks[1])
    ref._chunk_major_n = None
    ref._chunk_major_gpu = None           # what _chunked_prefill's finally does
    ref._chunk_major_gpu_bytes = 0

    loads = _count_loads(monkeypatch)
    _outs, routed = _layer_major(model, ref, chunks, N_CHUNKS)
    assert len(loads) == len(set().union(*routed))
    assert getattr(ref, "_chunk_major_gpu", None) is None
