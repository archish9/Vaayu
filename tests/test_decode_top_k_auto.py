"""DECODE_TOP_K="auto": the ladders and the cache (spec §5, §6).

The ladders are tested against cost tables shaped like the three MoE models
measured on this box (row 33 and the 2026-09-14 spike), so a regression in the
pick rule shows up as the wrong pick for a real model's numbers.
"""
import os
from pathlib import Path

import pytest

from vaayu import decode_top_k_auto as dtka


def _scorer(config_k, base, costs):
    """Fake perplexity: `costs` maps (k, p) -> relative cost. Any trial the
    ladder should NOT have run raises KeyError, so an overshoot fails loudly."""
    def score(k, p):
        if (k, p) == (config_k, None):
            return base
        return base * (1.0 + costs[(k, p)])
    return score


# ── ladders: fixed k first, cutoff only when no k fits ──────────────────────

def test_qwen3_next_like_costs_pick_k8_and_never_try_the_cutoff():
    costs = {(9, None): 0.0147, (8, None): 0.0343, (7, None): 0.06}
    out = dtka.run_ladders(10, 0.035, _scorer(10, 8.58, costs))
    assert out["pick"] == {"kind": "k", "value": 8, "cost": pytest.approx(0.0343)}
    assert [t["kind"] for t in out["trials"]] == ["k", "k", "k"]


def test_minimax_like_costs_fall_through_to_the_cutoff():
    costs = {(7, None): 0.0634, (8, 0.95): 0.0013, (8, 0.90): 0.0176, (8, 0.85): 0.05}
    out = dtka.run_ladders(8, 0.035, _scorer(8, 7.26, costs))
    assert (out["pick"]["kind"], out["pick"]["value"]) == ("p", 0.90)
    assert [(t["kind"], t["value"]) for t in out["trials"]] == [
        ("k", 7), ("p", 0.95), ("p", 0.90), ("p", 0.85)]


def test_qwen3_235b_like_costs_pick_k7():
    costs = {(7, None): 0.0340, (6, None): 0.0533}
    out = dtka.run_ladders(8, 0.035, _scorer(8, 7.78, costs))
    assert (out["pick"]["kind"], out["pick"]["value"]) == ("k", 7)


def test_nothing_in_budget_picks_none():
    costs = {(7, None): 0.10, (8, 0.95): 0.09}
    out = dtka.run_ladders(8, 0.035, _scorer(8, 7.0, costs))
    assert out["pick"] == {"kind": "none", "value": None, "cost": None}


def test_a_cost_equal_to_the_budget_is_in_budget():
    costs = {(3, None): 0.035, (2, None): 0.2}
    out = dtka.run_ladders(4, 0.035, _scorer(4, 1.0, costs))
    assert (out["pick"]["kind"], out["pick"]["value"]) == ("k", 3)


def test_the_fixed_k_ladder_stops_at_one():
    costs = {(2, None): 0.0, (1, None): 0.01}
    out = dtka.run_ladders(3, 0.035, _scorer(3, 5.0, costs))
    assert (out["pick"]["kind"], out["pick"]["value"]) == ("k", 1)
    assert [t["value"] for t in out["trials"]] == [2, 1]


# ── cache ───────────────────────────────────────────────────────────────────

def _record(key):
    return {**key, "config_k": 10, "baseline_ppl": 8.58, "trials": [],
            "pick": {"kind": "k", "value": 8, "cost": 0.0343},
            "measured_at": "2026-09-14T00:00:00Z"}


def test_cache_round_trip(tmp_path):
    key = dtka.cache_key(0.035, "abc")
    assert dtka.write_cache(tmp_path, _record(key)) is True
    assert dtka.read_cache(tmp_path, key) == _record(key)


@pytest.mark.parametrize("field,other", [
    ("method_version", 99), ("text_version", 99), ("budget", 0.02), ("fingerprint", "zzz")])
def test_any_key_field_change_is_a_miss(tmp_path, field, other):
    key = dtka.cache_key(0.035, "abc")
    dtka.write_cache(tmp_path, _record(key))
    assert dtka.read_cache(tmp_path, {**key, field: other}) is None


def test_a_missing_or_corrupt_cache_is_a_miss(tmp_path):
    key = dtka.cache_key(0.035, "abc")
    assert dtka.read_cache(tmp_path, key) is None
    (tmp_path / dtka.CACHE_FILENAME).write_text("{not json")
    assert dtka.read_cache(tmp_path, key) is None


def test_a_failed_write_leaves_no_cache_and_no_temp_file(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(dtka.os, "replace", refuse)
    assert dtka.write_cache(tmp_path, _record(dtka.cache_key(0.035, "abc"))) is False
    assert list(tmp_path.iterdir()) == []


def test_fingerprint_tracks_layer_files_and_ignores_the_cache_itself(tmp_path):
    layer = tmp_path / "model.layers.0.safetensors"
    layer.write_bytes(b"x" * 10)
    fp = dtka.fingerprint('{"a": 1}', tmp_path)
    (tmp_path / dtka.CACHE_FILENAME).write_text("{}")
    assert dtka.fingerprint('{"a": 1}', tmp_path) == fp
    layer.write_bytes(b"x" * 11)
    assert dtka.fingerprint('{"a": 1}', tmp_path) != fp
    assert dtka.fingerprint('{"a": 2}', tmp_path) != dtka.fingerprint('{"a": 1}', tmp_path)


# ── calibration text ────────────────────────────────────────────────────────

def test_calibration_text_is_long_enough_in_characters():
    assert len(dtka.CALIB_TEXT) >= 2000


@pytest.mark.parametrize("path", [
    os.environ.get("VAAYU_LLAMA_1B_CKPT", "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"),
    os.environ.get("VAAYU_QWEN3NEXT_CKPT",
                   "/home/a/code/ai-ml/models/cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"),
])
def test_calibration_text_tokenizes_to_the_minimum(path):
    if not Path(path).is_dir():
        pytest.skip(f"tokenizer not on this box: {path}")
    from transformers import AutoTokenizer
    n = len(AutoTokenizer.from_pretrained(path)(dtka.CALIB_TEXT).input_ids)
    assert n >= dtka.MIN_CALIB_TOKENS, n
