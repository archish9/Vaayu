"""requests.jsonl: one row per request, including the ones that failed.

Audit items 8 and 9. A failed request used to leave a REQ line with no DONE and
a traceback in server.log with no session attached.
"""
import json

import pytest

from vaayu.server import reqlog


class _Cache:
    def __init__(self, hits, misses):
        self.hits, self.misses = hits, misses


class _Model:
    """A MoE-shaped model. Dense models have none of these attributes."""
    def __init__(self):
        self._expert_disk_total_s = 2.0
        self._expert_cache = _Cache(100, 50)
        self._expert_vram_cache = _Cache(30, 70)


def test_snapshot_reads_the_monotonic_counters():
    snap = reqlog.snapshot(_Model())
    assert snap["expert_disk_total_s"] == 2.0
    assert snap["expert_ram_hits"] == 100
    assert snap["read_bytes"] >= 0


def test_snapshot_on_a_dense_model_is_nulls_not_errors():
    """A dense model legitimately has no expert caches. A healthy dense row
    must not look broken -- that was finding Item 7 of the L1 fix wave."""
    class _Dense:
        pass
    snap = reqlog.snapshot(_Dense())
    assert snap["expert_ram_hits"] is None
    assert snap["expert_disk_total_s"] is None
    assert snap["read_bytes"] >= 0          # the persister counter always works


def test_snapshot_never_resets_cuda_peak_stats():
    """vaayu_base.py:3628-3637 -- the peak counter belongs to whoever reset it
    last, and the engine must not touch it."""
    src = (reqlog.__file__).replace(".pyc", ".py")
    with open(src) as f:
        text = f.read()
    assert "reset_peak" not in text
    assert "reset_max_memory" not in text


def test_decode_block_diffs_two_snapshots():
    before = {"read_bytes": 1_000_000_000, "expert_disk_total_s": 1.0,
              "expert_ram_hits": 10, "expert_ram_misses": 5,
              "expert_vram_hits": 3, "expert_vram_misses": 7,
              "peak_rss_gb": 20.0, "peak_vram_gb_process": 15.0}
    after = {"read_bytes": 5_410_000_000, "expert_disk_total_s": 6.0,
             "expert_ram_hits": 110, "expert_ram_misses": 105,
             "expert_vram_hits": 43, "expert_vram_misses": 67,
             "peak_rss_gb": 20.7, "peak_vram_gb_process": 15.1}
    d = reqlog.decode_block(before, after, tokens=1000)
    assert d["bytes_read_gb"] == pytest.approx(4.41, abs=0.01)
    assert d["gb_per_token"] == pytest.approx(0.00441, abs=1e-5)
    assert d["mean_expert_stall_s"] == pytest.approx(0.005, abs=1e-6)
    assert d["expert_misses"] == 100
    assert d["expert_ram_hit_rate"] == pytest.approx(0.5, abs=1e-6)
    assert d["expert_vram_hit_rate"] == pytest.approx(0.4, abs=1e-6)


def test_decode_block_survives_null_sources():
    before = {"read_bytes": 0, "expert_disk_total_s": None,
              "expert_ram_hits": None, "expert_ram_misses": None,
              "expert_vram_hits": None, "expert_vram_misses": None,
              "peak_rss_gb": None, "peak_vram_gb_process": None}
    after = dict(before, read_bytes=1_000_000_000)
    d = reqlog.decode_block(before, after, tokens=10)
    assert d["bytes_read_gb"] == pytest.approx(1.0)
    assert d["expert_ram_hit_rate"] is None
    assert d["mean_expert_stall_s"] is None


def test_decode_block_with_zero_tokens_does_not_divide_by_zero():
    before = {"read_bytes": 0, "expert_disk_total_s": 0.0,
              "expert_ram_hits": 0, "expert_ram_misses": 0,
              "expert_vram_hits": 0, "expert_vram_misses": 0,
              "peak_rss_gb": None, "peak_vram_gb_process": None}
    d = reqlog.decode_block(before, dict(before), tokens=0)
    assert d["gb_per_token"] is None
    assert d["mean_expert_stall_s"] is None


def test_write_row_appends_one_json_line(tmp_path):
    reqlog.write_row(tmp_path, {"session": "a", "finish": "stop"})
    reqlog.write_row(tmp_path, {"session": "b", "finish": "error"})
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["session"] == "a"
    assert json.loads(lines[1])["finish"] == "error"


def test_write_row_never_raises_on_an_unwritable_dir(tmp_path):
    bad = tmp_path / "nope"
    bad.write_text("i am a file, not a directory")
    reqlog.write_row(bad, {"session": "a"})      # must not raise


def test_write_row_never_raises_on_unserialisable_content(tmp_path):
    reqlog.write_row(tmp_path, {"session": "a", "weird": object()})
    line = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[0])
    assert line["session"] == "a"


def test_write_row_is_a_noop_with_no_log_dir():
    reqlog.write_row(None, {"session": "a"})     # library callers pass None


def test_error_block_carries_type_message_and_traceback():
    try:
        raise ValueError("prompt is 9001 tokens; the window is 8192")
    except ValueError as e:
        block = reqlog.error_block(e)
    assert block["type"] == "ValueError"
    assert "9001" in block["message"]
    assert "test_reqlog.py" in block["traceback"]


def test_error_block_of_none_is_none():
    assert reqlog.error_block(None) is None
