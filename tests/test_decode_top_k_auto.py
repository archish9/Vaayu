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


# ── non-finite perplexities (final review I2) ────────────────────────────────

def test_a_non_finite_trial_counts_as_over_budget():
    costs = {(9, None): float("nan"), (10, 0.95): 0.5}
    out = dtka.run_ladders(10, 0.035, _scorer(10, 8.0, costs))
    assert out["pick"] == {"kind": "none", "value": None, "cost": None}
    assert out["trials"][0]["cost"] == float("inf")


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
    tok = AutoTokenizer.from_pretrained(path)
    for name, text in dtka.CALIB_TEXTS:
        n = len(tok(text).input_ids)
        assert n >= dtka.MIN_CALIB_TOKENS, (name, n)


# ── decode_mode_ppl ─────────────────────────────────────────────────────────
from types import SimpleNamespace

import torch


class _UniformLM:
    """Uniform logits over `vocab`: perplexity is exactly `vocab`."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
        self.calls.append(input_ids.shape[1])
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], self.vocab),
                               past_key_values=object())


def test_decode_mode_ppl_prefills_once_then_scores_single_decode_steps():
    lm = _UniformLM(50)
    ids = torch.randint(0, 50, (1, 20))

    ppl = dtka.decode_mode_ppl(lm, ids, prefix_tokens=4, score_tokens=10)

    assert ppl == pytest.approx(50.0)
    assert lm.calls == [4] + [1] * 10


# ── resolve(): a fake MoE model on CPU ──────────────────────────────────────

class _FakeMoE:
    def __init__(self, ckpt, config_k=10, n_tokens=400, cfg='{"model": 1}'):
        self.is_moe = True
        self.moe_top_k = config_k
        self._decode_top_k = config_k
        self._decode_top_p = None
        self.checkpoint_path = ckpt
        self.running_device = "cpu"
        self.config = SimpleNamespace(to_json_string=lambda: cfg)
        self.tokenizer = lambda text, return_tensors=None: SimpleNamespace(
            input_ids=torch.zeros(1, n_tokens, dtype=torch.long))
        (ckpt / "model.layers.0.safetensors").write_bytes(b"x")


def _fake_ppl(monkeypatch, config_k, base, costs, calls):
    """Patches the one function that needs a real model. Records every (k, p)
    the scorer set on the model, so tests can see what was measured."""
    def fake(model, ids, prefix_tokens=dtka.PREFIX_TOKENS, score_tokens=dtka.SCORE_TOKENS):
        key = (model._decode_top_k, model._decode_top_p)
        calls.append(key)
        if key == (config_k, None):
            return base
        return base * (1.0 + costs[key])
    monkeypatch.setattr(dtka, "decode_mode_ppl", fake)


QWEN3_NEXT_LIKE = {(9, None): 0.0147, (8, None): 0.0343, (7, None): 0.06}
MINIMAX_LIKE = {(7, None): 0.0634, (8, 0.95): 0.0013, (8, 0.90): 0.0176, (8, 0.85): 0.05}


def test_a_cold_resolve_measures_applies_k_and_caches(tmp_path, monkeypatch):
    calls = []
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, calls)
    m = _FakeMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert (m._decode_top_k, m._decode_top_p) == (8, None)
    assert status == {"mode": "k", "value": 8, "cost": pytest.approx(0.0343), "source": "measured"}
    assert (tmp_path / dtka.CACHE_FILENAME).exists()
    assert calls[0] == (10, None)


def test_a_warm_resolve_reads_the_cache_and_measures_nothing(tmp_path, monkeypatch):
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, [])
    dtka.resolve(_FakeMoE(tmp_path), 0.035)
    calls = []
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, calls)
    m = _FakeMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert calls == []
    assert (m._decode_top_k, status["source"]) == (8, "cache")


def test_a_different_budget_re_measures(tmp_path, monkeypatch):
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, [])
    dtka.resolve(_FakeMoE(tmp_path), 0.035)
    calls = []
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, calls)
    m = _FakeMoE(tmp_path)

    dtka.resolve(m, 0.02)

    assert calls, "a cached pick for another budget must not be reused"
    assert m._decode_top_k == 9


def test_a_cutoff_pick_sets_p_and_keeps_k_at_the_checkpoints_own(tmp_path, monkeypatch):
    _fake_ppl(monkeypatch, 8, 7.26, MINIMAX_LIKE, [])
    m = _FakeMoE(tmp_path, config_k=8)

    status = dtka.resolve(m, 0.035)

    assert (m._decode_top_k, m._decode_top_p) == (8, 0.90)
    assert (status["mode"], status["value"]) == ("p", 0.90)


def test_nothing_in_budget_leaves_the_model_lossless(tmp_path, monkeypatch):
    costs = {(9, None): 0.10, (10, 0.95): 0.09}
    _fake_ppl(monkeypatch, 10, 8.0, costs, [])
    m = _FakeMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert (m._decode_top_k, m._decode_top_p) == (10, None)
    assert status["mode"] == "none"


def test_a_failing_measurement_warns_caches_nothing_and_stays_lossless(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(dtka, "decode_mode_ppl", boom)
    m = _FakeMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert (m._decode_top_k, m._decode_top_p) == (10, None)
    assert status == {"mode": "none", "value": None, "cost": None, "source": "failed"}
    assert not (tmp_path / dtka.CACHE_FILENAME).exists()
    out = capsys.readouterr().out
    assert "WARNING" in out and "CUDA out of memory" in out


def test_a_too_short_calibration_text_warns_and_skips(tmp_path, monkeypatch, capsys):
    calls = []
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, calls)
    m = _FakeMoE(tmp_path, n_tokens=dtka.MIN_CALIB_TOKENS - 1)

    status = dtka.resolve(m, 0.035)

    assert calls == []
    assert status["source"] == "failed"
    assert "WARNING" in capsys.readouterr().out


def test_a_nan_baseline_warns_caches_nothing_and_stays_lossless(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dtka, "decode_mode_ppl", lambda *a, **k: float("nan"))
    m = _FakeMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert status == {"mode": "none", "value": None, "cost": None, "source": "failed"}
    assert (m._decode_top_k, m._decode_top_p) == (10, None)
    assert not (tmp_path / dtka.CACHE_FILENAME).exists()
    assert "WARNING" in capsys.readouterr().out


def test_a_tokenizer_error_warns_and_stays_lossless(tmp_path, capsys):
    m = _FakeMoE(tmp_path)

    def broken(text, return_tensors=None):
        raise RuntimeError("tokenizer exploded")

    m.tokenizer = broken

    status = dtka.resolve(m, 0.035)

    assert status == {"mode": "none", "value": None, "cost": None, "source": "failed"}
    assert (m._decode_top_k, m._decode_top_p) == (10, None)
    out = capsys.readouterr().out
    assert "WARNING" in out and "tokenizer exploded" in out


def test_an_unwritable_cache_still_applies_the_pick_and_warns(tmp_path, monkeypatch, capsys):
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, [])
    monkeypatch.setattr(dtka, "write_cache", lambda *a, **k: False)
    m = _FakeMoE(tmp_path)

    dtka.resolve(m, 0.035)

    assert m._decode_top_k == 8
    assert "every load" in capsys.readouterr().out


@pytest.mark.parametrize("damage", [
    lambda r: r["pick"].pop("kind"),
    lambda r: r["pick"].update(kind="sideways"),
    lambda r: r["pick"].pop("value"),
    lambda r: r.pop("measured_at"),
])
def test_a_structurally_damaged_cache_is_a_miss(tmp_path, damage):
    """Key fields match but the pick is damaged: resolve() would crash the load
    reading it, so read_cache must treat it as a miss and let it re-measure."""
    key = dtka.cache_key(0.035, "abc")
    record = _record(key)
    damage(record)
    dtka.write_cache(tmp_path, record)
    assert dtka.read_cache(tmp_path, key) is None


# ── cache pick type/range validation (final review M1) ───────────────────────

@pytest.mark.parametrize("pick", [
    {"kind": "k", "value": 8, "cost": None},
    {"kind": "k", "value": "8", "cost": 0.03},
    {"kind": "k", "value": 0, "cost": 0.03},
    {"kind": "k", "value": 8, "cost": float("nan")},
    {"kind": "p", "value": 0, "cost": 0.02},
    {"kind": "p", "value": 1.5, "cost": 0.02},
    {"kind": "none", "value": 3, "cost": None},
])
def test_a_cache_with_a_badly_typed_pick_is_a_miss(tmp_path, pick):
    key = dtka.cache_key(0.035, "abc")
    dtka.write_cache(tmp_path, {**_record(key), "pick": pick})
    assert dtka.read_cache(tmp_path, key) is None


def test_a_cached_k_not_below_the_checkpoints_own_re_measures(tmp_path, monkeypatch):
    m = _FakeMoE(tmp_path)
    key = dtka.cache_key(0.035, dtka.fingerprint(m.config.to_json_string(), tmp_path))
    dtka.write_cache(tmp_path, {**_record(key), "pick": {"kind": "k", "value": 10, "cost": 0.0}})
    calls = []
    _fake_ppl(monkeypatch, 10, 8.58, QWEN3_NEXT_LIKE, calls)

    status = dtka.resolve(m, 0.035)

    assert calls, "a cached k >= config_k must be re-measured, not applied"
    assert status["source"] == "measured"


def test_a_top1_checkpoint_measures_nothing(tmp_path, monkeypatch):
    """One routed expert: the cutoff can never drop a slot, so measuring would spend
    ten runs to apply a no-op and print a false DEGRADED warning."""
    calls = []
    _fake_ppl(monkeypatch, 1, 5.0, {}, calls)
    m = _FakeMoE(tmp_path, config_k=1)

    status = dtka.resolve(m, 0.035)

    assert calls == []
    assert status == {"mode": "none", "value": None, "cost": None, "source": "config"}
    assert (m._decode_top_k, m._decode_top_p) == (1, None)


# ── two calibration texts: a trial's cost is the WORSE of code and prose ─────
#
# Task 7's first on-device arm priced truncation on the code text alone:
# Qwen3-Next picked top-6 at +2.26 % on code, which costs +8.97 % on row 33's
# prose passage -- 2.6x the budget. User decision 2026-09-14: score both texts
# and keep the safer pick. One ladder whose cost is the worse text's is that
# rule, with no k-versus-p comparison to make.

def _two_text_scorer(config_k, bases, costs):
    """`costs` maps (k, p) -> (code_cost, prose_cost); returns one ppl per text."""
    def score(k, p):
        if (k, p) == (config_k, None):
            return bases
        return tuple(b * (1.0 + c) for b, c in zip(bases, costs[(k, p)]))
    return score


# Qwen3-Next: code costs from Task 7's cold arm (2026-09-14), prose costs from
# row 33 (256 scored tokens). Prose k=7 was never measured; 0.06 stands in for
# "over budget", which row 33's prose k=6 (+8.97 %) already implies.
QWEN3_NEXT_TWO_TEXTS = {
    (9, None): (-0.0081, 0.0147),
    (8, None): (0.0055, 0.0343),
    (7, None): (0.0153, 0.06),
}
TWO_TEXT_BASES = (1.9005, 8.5824)


def test_the_worse_text_decides_the_pick():
    out = dtka.run_ladders(10, 0.035, _two_text_scorer(10, TWO_TEXT_BASES, QWEN3_NEXT_TWO_TEXTS))
    assert out["pick"] == {"kind": "k", "value": 8, "cost": pytest.approx(0.0343)}


def test_each_trial_records_every_texts_cost():
    out = dtka.run_ladders(10, 0.035, _two_text_scorer(10, TWO_TEXT_BASES, QWEN3_NEXT_TWO_TEXTS))
    first = out["trials"][0]
    assert first["costs"] == pytest.approx([-0.0081, 0.0147])
    assert first["cost"] == pytest.approx(0.0147)
    assert out["baseline_ppl"] == pytest.approx(list(TWO_TEXT_BASES))


class _TwoTextMoE(_FakeMoE):
    """Tokenizes the code and prose texts to different lengths, so a fake
    perplexity can tell which text it is scoring."""
    CODE_TOKENS = 400

    def __init__(self, ckpt, prose_tokens=450, **kwargs):
        super().__init__(ckpt, **kwargs)

        def tokenize(text, return_tensors=None):
            n = self.CODE_TOKENS if text == dtka.CALIB_TEXT else prose_tokens
            return SimpleNamespace(input_ids=torch.zeros(1, n, dtype=torch.long))

        self.tokenizer = tokenize


def _fake_two_text_ppl(monkeypatch, config_k, bases, costs, calls):
    def fake(model, ids, prefix_tokens=dtka.PREFIX_TOKENS, score_tokens=dtka.SCORE_TOKENS):
        text = 0 if ids.shape[1] == _TwoTextMoE.CODE_TOKENS else 1
        key = (model._decode_top_k, model._decode_top_p)
        calls.append((text, key))
        if key == (config_k, None):
            return bases[text]
        return bases[text] * (1.0 + costs[key][text])
    monkeypatch.setattr(dtka, "decode_mode_ppl", fake)


def test_resolve_scores_both_texts_and_applies_the_safer_pick(tmp_path, monkeypatch):
    calls = []
    _fake_two_text_ppl(monkeypatch, 10, TWO_TEXT_BASES, QWEN3_NEXT_TWO_TEXTS, calls)
    m = _TwoTextMoE(tmp_path)

    status = dtka.resolve(m, 0.035)

    assert (m._decode_top_k, status["value"], status["source"]) == (8, 8, "measured")
    assert {text for text, _ in calls} == {0, 1}
    assert len(calls) == 2 * (1 + 3)   # baseline + k=9, 8, 7 -- each on both texts


def test_a_too_short_prose_text_warns_naming_it(tmp_path, monkeypatch, capsys):
    calls = []
    _fake_two_text_ppl(monkeypatch, 10, TWO_TEXT_BASES, QWEN3_NEXT_TWO_TEXTS, calls)
    m = _TwoTextMoE(tmp_path, prose_tokens=dtka.MIN_CALIB_TOKENS - 1)

    status = dtka.resolve(m, 0.035)

    assert calls == []
    assert status["source"] == "failed"
    out = capsys.readouterr().out
    assert "WARNING" in out and "prose" in out


def test_calib_prose_is_row33s_passage():
    """Row 33's measured prose costs are only comparable if this is that text."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "cache_bias_ppl.py"
    spec = importlib.util.spec_from_file_location("_cb_ppl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert dtka.CALIB_PROSE == module.FIXED_TEXT
