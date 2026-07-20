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


# --- Final-review fix: load_all seeds the DiskWriter, closing the cross-process gap ---


def test_load_all_seeds_writer_state(tmp_path):
    """load_all(..., writer=...) must register each restored session with the
    writer, using the TRUE on-disk nrec (not 0) -- otherwise the first
    post-restart append/truncate for that session treats it as brand new."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache()
    _write_session(tmp_path, "s1", c, cfg, [7, 8, 9, 10, 11])

    # Fresh DiskWriter simulates a process restart: brand-new instance, no
    # in-memory _state, pointed at the same persist directory.
    fresh_writer = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(
        store, cfg, ram_budget_bytes=10 ** 12, writer=fresh_writer)
    assert n == 1
    assert "s1" in fresh_writer._state
    assert fresh_writer._state["s1"]["nrec"] == 5
    fresh_writer.close()


def test_load_all_without_writer_arg_still_works(tmp_path):
    """writer is optional -- omitting it (existing callers) must not break."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    _write_session(tmp_path, "s1", _cache(), cfg, [1, 2, 3, 4, 5])
    store = _Store()
    n = kp.KvDiskLoader(str(tmp_path)).load_all(store, cfg, ram_budget_bytes=10 ** 12)
    assert n == 1 and "s1" in store.entries


def test_restart_then_append_extends_tail_only_no_full_rewrite(tmp_path):
    """The actual restart scenario the finding describes. Write N records via one
    DiskWriter, close it (process exits). A FRESH DiskWriter over the same
    directory simulates the new process. load_all(..., writer=fresh_writer)
    seeds its state. A subsequent enqueue_append extending the restored session
    must start from the TRUE on-disk nrec (N), not 0:
      (a) the final nrec equals the new length,
      (b) the original first-N records' bytes are byte-identical afterward
          (proves this was a tail-only append, not a full rewrite from 0),
      (c) every position holds real data (no gap).
    """
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache(seq=10)
    toks = list(range(101, 111))   # 10 distinctive, non-zero token ids
    N = 6

    w1 = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    w1.enqueue_append("s1", c, cfg, toks[:N], N)
    w1.drain(); w1.close()

    path = w1.path_for("s1")
    with open(path, "rb") as f:
        h0 = kp.read_header(f)
    rstart = kp.record_start(h0)
    rbytes = kp.record_bytes(h0)
    with open(path, "rb") as f:
        original_prefix = f.read(rstart + N * rbytes)[rstart:]

    # Fresh writer = new process. Seed it via load_all.
    w2 = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    store = _Store()
    kp.KvDiskLoader(str(tmp_path)).load_all(
        store, cfg, ram_budget_bytes=10 ** 12, writer=w2)
    assert w2._state["s1"]["nrec"] == N   # seeded with true on-disk nrec

    # Extend the restored session past N (continue-the-session pattern).
    new_length = 10
    w2.enqueue_append("s1", c, cfg, toks[:new_length], new_length)
    w2.drain(); w2.close()

    with open(path, "rb") as f:
        h1 = kp.read_header(f)
        got_toks, _ = kp.read_records(f, h1)
    assert h1.nrec == new_length
    assert got_toks == toks[:new_length]   # every position holds real data, no gap

    with open(path, "rb") as f:
        new_prefix = f.read(rstart + N * rbytes)[rstart:]
    assert new_prefix == original_prefix   # tail-only append, NOT a full rewrite


def test_restart_then_truncate_actually_shrinks_disk(tmp_path):
    """The OTHER half of the finding: after a restart, enqueue_truncate for a
    restored session must not silently no-op. Before the fix, st is None in a
    fresh DiskWriter's _state (load_all never told it about restored sessions),
    so enqueue_truncate's `if st is None: return` guard fired and the disk
    truncate was dropped on the floor -- even though the RAM crop happened
    correctly. After register_existing seeds st, the truncate must actually
    submit and shrink the on-disk nrec."""
    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=16)
    c = _cache(seq=6)
    toks = [1, 2, 3, 4, 5, 6]

    w1 = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    w1.enqueue_append("s1", c, cfg, toks, 6)
    w1.drain(); w1.close()

    w2 = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    store = _Store()
    kp.KvDiskLoader(str(tmp_path)).load_all(
        store, cfg, ram_budget_bytes=10 ** 12, writer=w2)

    w2.enqueue_truncate("s1", 3)
    w2.drain(); w2.close()

    with open(w2.path_for("s1"), "rb") as f:
        h = kp.read_header(f)
    assert h.nrec == 3   # disk actually shrunk, not a silent no-op
