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


def test_load_all_skips_corrupt_file_and_continues(tmp_path, monkeypatch, capsys):
    """Any exception during read/reconstruct -- not just OSError/struct.error --
    must be caught and logged, so a corrupt file doesn't crash the whole startup
    scan. Proves the deliberate broad except Exception clause works; matches the
    precedent in test_kv_persist_writer.py::test_non_oserror_worker_exception_is_caught_and_logged."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache()

    # Write two valid sessions.
    _write_session(tmp_path, "s1", c, cfg, [1, 2, 3, 4, 5])
    _write_session(tmp_path, "s2", c, cfg, [6, 7, 8, 9, 10])

    # Monkeypatch build_cache to raise RuntimeError for "s1" only.
    real_build_cache = kp.build_cache

    def boom(h, buffers):
        if h.session_id == "s1":
            raise RuntimeError("simulated corrupt file")
        return real_build_cache(h, buffers)

    monkeypatch.setattr(kp, "build_cache", boom)

    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)

    # Must not raise; only s2 loads (s1 skipped due to RuntimeError).
    assert n == 1
    assert "s2" in store.entries
    assert "s1" not in store.entries
    toks, kv = store.entries["s2"]
    assert toks == [6, 7, 8, 9, 10]

    # Verify the skip message was printed.
    output = capsys.readouterr().out
    assert "[KV] skipping" in output
