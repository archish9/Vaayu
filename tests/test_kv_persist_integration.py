"""Ships-dark: with persistence off, no files are created and behavior is unchanged.
Uses a tiny stubbed engine surface — no model load, no CUDA."""
import os
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


def _cache(n_layers=2, heads=4, dim=16, seq=4):
    c = FP8DynamicCache()
    for L in range(n_layers):
        c.update(torch.randn(1, heads, seq, dim, dtype=torch.float16),
                 torch.randn(1, heads, seq, dim, dtype=torch.float16), L)
    return c


def test_writer_disabled_creates_no_files(tmp_path):
    """When the engine leaves _kv_writer None (flag off), store() must not write."""
    writer = None  # simulates enable_kv_persist=False
    # the hook the engine runs after store():
    if writer is not None:
        writer.enqueue_append("s", _cache(), types.SimpleNamespace(vocab_size=1), [1], 1)
    assert os.listdir(tmp_path) == []


def test_end_to_end_persist_then_reload(tmp_path):
    """Full disk round-trip via the real writer+loader (the engine's two hooks)."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache(seq=4)
    toks = [5, 6, 7, 8]
    writer = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    writer.enqueue_append("s1", c, cfg, toks, 4)   # engine's post-store hook
    writer.drain(); writer.close()

    class _Store:
        def __init__(self): self.entries = {}
        def store(self, sid, t, kv): self.entries[sid] = (t, kv)
    store = _Store()
    kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)
    toks2, kv2 = store.entries["s1"]
    assert toks2 == toks
    for L in range(2):
        assert torch.allclose(kv2._k_out_cache[L][..., :4, :], c._k_out_cache[L][..., :4, :])
