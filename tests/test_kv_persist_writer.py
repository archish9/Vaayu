"""DiskWriter: append grows nrec, nrec-last crash safety, truncate mirror."""
import os
import types
import torch
import pytest
from vaayu import kv_persist as kp
from vaayu.vaayu_base import FP8DynamicCache


def _cache(n_layers=2, heads=4, dim=16, seq=6):
    c = FP8DynamicCache()
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, dtype=torch.float16)
        c.update(k, v, L)
    return c


def _read_nrec(path):
    with open(path, "rb") as f:
        h = kp.read_header(f)
    return h.nrec


def test_append_writes_header_and_grows_nrec(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    toks = [1, 2, 3, 4, 5, 6]
    w.enqueue_append("s1", c, cfg, toks[:4], 4)
    w.drain()
    p = w.path_for("s1")
    assert os.path.exists(p) and _read_nrec(p) == 4
    # second turn appends only the new tail
    w.enqueue_append("s1", c, cfg, toks, 6)
    w.drain()
    assert _read_nrec(p) == 6
    w.close()


def test_nrec_last_survives_torn_append(tmp_path):
    """Corrupt the file by chopping the last record; header nrec still points at
    the committed prefix, so a reader that trusts nrec loads coherent data."""
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain(); w.close()
    p = w.path_for("s1")
    # simulate a torn write: append garbage bytes past the committed region
    with open(p, "ab") as f:
        f.write(b"\x00" * 32)
    assert _read_nrec(p) == 6  # nrec unchanged -> reader ignores the torn tail


def test_truncate_mirror(tmp_path):
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    cfg = types.SimpleNamespace(vocab_size=1000)
    c = _cache(seq=6)
    w.enqueue_append("s1", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()
    w.enqueue_truncate("s1", 3)
    w.drain()
    assert _read_nrec(w.path_for("s1")) == 3
    w.close()
