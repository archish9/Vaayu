"""KvDiskLoader: reconstruct sessions, skip geometry/quant mismatches, honor budget."""
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


class _Store:
    def __init__(self):
        self.entries = {}
    def store(self, session_id, token_ids, kv):
        self.entries[session_id] = (token_ids, kv)


def _cache(n_layers=2, heads=4, dim=16, seq=5):
    c = FP8DynamicCache()
    for L in range(n_layers):
        c.update(torch.randn(1, heads, seq, dim, dtype=torch.float16),
                 torch.randn(1, heads, seq, dim, dtype=torch.float16), L)
    return c


def _write_session(tmp_path, sid, cache, cfg, toks):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    w.enqueue_append(sid, cache, cfg, toks, len(toks))
    w.drain(); w.close()


def test_load_all_restores_sessions(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache()
    _write_session(tmp_path, "s1", c, cfg, [7, 8, 9, 10, 11])
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)
    assert n == 1 and "s1" in store.entries
    toks, kv = store.entries["s1"]
    assert toks == [7, 8, 9, 10, 11]
    assert torch.allclose(kv._k_out_cache[0][..., :5, :], c._k_out_cache[0][..., :5, :])


def test_load_all_skips_geometry_mismatch(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    _write_session(tmp_path, "s1", _cache(), cfg, [1, 2, 3, 4, 5])
    other = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=99,
                                  num_key_value_heads=4, head_dim=16)
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, other, ram_budget_bytes=10 ** 12)
    assert n == 0 and store.entries == {}


def test_load_all_respects_budget(tmp_path):
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    _write_session(tmp_path, "s1", _cache(), cfg, [1, 2, 3, 4, 5])
    _write_session(tmp_path, "s2", _cache(), cfg, [1, 2, 3, 4, 5])
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=1)
    assert n == 0  # budget of 1 byte admits nothing
