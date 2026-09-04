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
    """The peak counter belongs to whoever reset it last, and the engine must
    not touch it. The reasoning lives on the comment above
    `torch.cuda.max_memory_allocated()` in vaayu_base's chunked-prefill loop --
    named rather than cited by line number, which is what the production
    counterpart in reqlog.snapshot already does."""
    src = (reqlog.__file__).replace(".pyc", ".py")
    with open(src) as f:
        text = f.read()
    assert "reset_peak" not in text
    assert "reset_max_memory" not in text


def test_decode_block_diffs_two_snapshots():
    """`bytes_read_gb`/`gb_per_token`/`mean_expert_stall_s` are diffs -- their
    two sources are genuinely monotonic. The expert-cache fields are NOT
    diffed (see test_cache_counters_are_reported_point_in_time_not_diffed):
    with `decode_stats_reset` True, they read `after`'s point-in-time value."""
    before = {"read_bytes": 1_000_000_000, "expert_disk_total_s": 1.0,
              "expert_ram_hits": 10, "expert_ram_misses": 5,
              "expert_vram_hits": 3, "expert_vram_misses": 7,
              "peak_rss_gb": 20.0, "peak_vram_gb_process": 15.0,
              "decode_stats_reset": True}
    after = {"read_bytes": 5_410_000_000, "expert_disk_total_s": 6.0,
             "expert_ram_hits": 110, "expert_ram_misses": 105,
             "expert_vram_hits": 43, "expert_vram_misses": 67,
             "peak_rss_gb": 20.7, "peak_vram_gb_process": 15.1,
             "decode_stats_reset": True}
    d = reqlog.decode_block(before, after, tokens=1000)
    assert d["bytes_read_gb"] == pytest.approx(4.41, abs=0.01)
    assert d["gb_per_token"] == pytest.approx(0.00441, abs=1e-5)
    assert d["mean_expert_stall_s"] == pytest.approx(0.005, abs=1e-6)
    assert d["expert_misses"] == 105
    assert d["expert_ram_hit_rate"] == pytest.approx(110 / 215, abs=1e-6)
    assert d["expert_vram_hit_rate"] == pytest.approx(43 / 110, abs=1e-6)


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


def test_cache_counters_are_reported_point_in_time_not_diffed():
    """The engine resets the expert-cache counters at the prefill->decode
    boundary INSIDE each request (vaayu_base.decode_stats_reset), so `before`
    holds the PREVIOUS request's totals. Diffing them yields a number that can
    be negative or plausibly wrong; `after` alone is this request's decode
    window."""
    before = {"read_bytes": 0, "expert_disk_total_s": 0.0,
              "expert_ram_hits": 900, "expert_ram_misses": 900,
              "expert_vram_hits": 400, "expert_vram_misses": 600,
              "peak_rss_gb": None, "peak_vram_gb_process": None,
              "decode_stats_reset": True}
    after = dict(before, expert_ram_hits=30, expert_ram_misses=70,
                 expert_vram_hits=20, expert_vram_misses=80)
    d = reqlog.decode_block(before, after, tokens=100)
    assert d["expert_misses"] == 70, "a diff would have given -830"
    assert d["expert_ram_hit_rate"] == pytest.approx(0.3)
    assert d["expert_vram_hit_rate"] == pytest.approx(0.2)


def test_cache_counters_are_null_when_no_reset_fired_this_request():
    """A request that failed during prefill never reached a decode pass, so no
    reset fired and the counters still describe the previous request. Report
    nothing rather than someone else's numbers."""
    before = {"read_bytes": 0, "expert_disk_total_s": 0.0,
              "expert_ram_hits": 900, "expert_ram_misses": 900,
              "expert_vram_hits": 400, "expert_vram_misses": 600,
              "peak_rss_gb": None, "peak_vram_gb_process": None,
              "decode_stats_reset": False}
    d = reqlog.decode_block(before, dict(before), tokens=0)
    assert d["expert_misses"] is None
    assert d["expert_ram_hit_rate"] is None
    assert d["expert_vram_hit_rate"] is None


def test_a_dense_model_snapshot_has_no_reset_latch_and_does_not_raise():
    class _Dense:
        pass
    snap = reqlog.snapshot(_Dense())
    assert snap["decode_stats_reset"] is None
    assert snap["expert_ram_hits"] is None


def test_cache_fields_are_null_when_the_request_produced_no_tokens():
    """The latch is STICKY: a request that raised before any forward pass ran
    never touched it, so it still carries the previous request's True. Without
    the `tokens > 0` requirement, decode_block published the previous
    request's hit rates as this one's -- a stale number that looks measured."""
    before = {"read_bytes": 0, "expert_disk_total_s": 0.0,
              "expert_ram_hits": 90, "expert_ram_misses": 10,
              "expert_vram_hits": 5, "expert_vram_misses": 5,
              "peak_rss_gb": 1.0, "peak_vram_gb_process": 1.0,
              "decode_stats_reset": True}
    d = reqlog.decode_block(before, dict(before), tokens=0)
    assert d["expert_misses"] is None
    assert d["expert_ram_hit_rate"] is None
    assert d["expert_vram_hit_rate"] is None


def test_diff_refuses_a_non_monotonic_key():
    """The diffed/point-in-time split is now enforced, not merely documented:
    wiring an expert-cache counter through `diff()` is the exact
    miscategorisation a mid-branch review already caught once."""
    assert reqlog.MONOTONIC_KEYS == {"read_bytes", "expert_disk_total_s"}
    snap = {"read_bytes": 10, "expert_ram_hits": 3}
    assert reqlog.diff(snap, {"read_bytes": 40}, "read_bytes") == 30
    with pytest.raises(AssertionError):
        reqlog.diff(snap, snap, "expert_ram_hits")
    with pytest.raises(AssertionError):
        reqlog.diff(snap, snap, "peak_vram_gb_process")
