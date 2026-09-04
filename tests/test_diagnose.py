"""`python -m vaayu.server.diagnose` -- the box, without loading weights.

Each tax is asserted to fire on a synthetic finding AND to stay quiet on a
clean box. A check that cannot stay quiet is not a check; it is a banner.
"""
import json

from vaayu.server import diagnose


class _Cfg:
    DEFAULT_STREAM = True
    DRAFTER_MODEL_PATH = None


def test_apst_tax_fires():
    data = {"storage": {"apst_active": True, "apst_ps_max_latency_us": 100000}}
    out = " ".join(diagnose.taxes(data, _Cfg()))
    assert "APST" in out and "vaayu_optimize_os.sh" in out


def test_apst_tax_is_quiet_when_inactive():
    data = {"storage": {"apst_active": False}}
    assert not any("APST" in t for t in diagnose.taxes(data, _Cfg()))


def test_foreign_gpu_process_tax_fires_below_the_refusal_threshold():
    """rustdesk's 0.14 GiB is under the 256 MiB refusal threshold but is
    permanent, and budgets are sized once at load."""
    data = {"gpu": {"foreign_processes": [{"pid": 1, "gib": 0.14, "cmd": "rustdesk"}]}}
    out = " ".join(diagnose.taxes(data, _Cfg()))
    assert "rustdesk" in out and "0.14" in out


def test_foreign_gpu_tax_is_quiet_on_an_empty_gpu():
    data = {"gpu": {"foreign_processes": []}}
    assert not any("GPU" in t for t in diagnose.taxes(data, _Cfg()))


def test_io_scheduler_tax_fires():
    data = {"storage": {"scheduler": "mq-deadline"}}
    out = " ".join(diagnose.taxes(data, _Cfg()))
    assert "scheduler" in out and "mq-deadline" in out


def test_io_scheduler_tax_is_quiet_on_none():
    data = {"storage": {"scheduler": "none"}}
    assert not any("scheduler" in t for t in diagnose.taxes(data, _Cfg()))


# ── drafter-on-MoE: detected from the checkpoint's config.json on disk,      ─
# ── NEVER from data["model"] -- that key never exists in a real invocation  ─
# ── (main() always calls collect(None, ...)), which is exactly the review   ─
# ── finding this replaces the old data["model"]["is_moe"]-based tests with. ─

def test_drafter_on_moe_tax_fires(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_experts": 128, "num_experts_per_tok": 8}))

    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/coder-1.5b"
        MODEL_PATH = str(tmp_path)
    out = " ".join(diagnose.taxes({}, C()))
    assert "drafter" in out.lower()


def test_drafter_tax_is_quiet_on_dense(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 80, "hidden_size": 8192}))

    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/llama-1b"
        MODEL_PATH = str(tmp_path)
    assert not any("drafter" in t.lower() for t in diagnose.taxes({}, C()))


def test_drafter_tax_is_quiet_when_config_json_is_missing(tmp_path):
    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/coder-1.5b"
        MODEL_PATH = str(tmp_path)   # empty dir -- no config.json at all
    assert not any("drafter" in t.lower() for t in diagnose.taxes({}, C()))


def test_drafter_tax_is_quiet_when_config_json_is_malformed(tmp_path):
    (tmp_path / "config.json").write_text("{not valid json")

    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/coder-1.5b"
        MODEL_PATH = str(tmp_path)
    assert not any("drafter" in t.lower() for t in diagnose.taxes({}, C()))


def test_diagnose_main_still_exits_0_with_a_malformed_config_json(
        tmp_path, monkeypatch, capsys):
    """diagnose is the tool for when the server will not start at all -- a
    checkpoint config.json that fails to parse must not be one more way for
    diagnose ITSELF to fail."""
    (tmp_path / "config.json").write_text("{not valid json")
    monkeypatch.setattr(diagnose.config, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(diagnose.config, "DRAFTER_MODEL_PATH", "/models/x")
    rc = diagnose.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "A drafter is configured" not in out


def test_default_stream_false_tax_fires():
    class C(_Cfg):
        DEFAULT_STREAM = False
    out = " ".join(diagnose.taxes({}, C()))
    assert "DEFAULT_STREAM" in out


def test_a_clean_box_has_no_taxes():
    data = {"storage": {"apst_active": False, "scheduler": "none"},
            "gpu": {"foreign_processes": []}}
    assert diagnose.taxes(data, _Cfg()) == []


def test_module_imports_without_cuda():
    import importlib
    importlib.reload(diagnose)      # must not raise on a CUDA-less box


def _write_config(tmp_path, payload):
    (tmp_path / "config.json").write_text(json.dumps(payload))
    return str(tmp_path)


def test_a_real_moe_config_is_detected(tmp_path):
    path = _write_config(tmp_path, {"num_local_experts": 256,
                                    "num_experts_per_tok": 8})
    assert diagnose._checkpoint_looks_like_moe(path) is True


def test_a_boolean_expert_flag_is_not_evidence_of_moe(tmp_path):
    """`bool` is a subclass of `int`, so a dense config carrying
    "expert_parallel": false used to read as MoE and bill the
    drafter-on-MoE tax to a model that has no experts at all."""
    path = _write_config(tmp_path, {"expert_parallel": False,
                                    "hidden_size": 8192})
    assert diagnose._checkpoint_looks_like_moe(path) is False


def test_zero_experts_is_not_evidence_of_moe(tmp_path):
    """A dense config that spells out `"num_experts": 0` is stating it has
    none. Counting it as MoE inverted the answer."""
    path = _write_config(tmp_path, {"num_experts": 0, "hidden_size": 8192})
    assert diagnose._checkpoint_looks_like_moe(path) is False
