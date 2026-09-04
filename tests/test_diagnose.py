"""`python -m vaayu.server.diagnose` -- the box, without loading weights.

Each tax is asserted to fire on a synthetic finding AND to stay quiet on a
clean box. A check that cannot stay quiet is not a check; it is a banner.
"""
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


def test_drafter_on_moe_tax_fires():
    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/coder-1.5b"
    data = {"model": {"is_moe": True}}
    out = " ".join(diagnose.taxes(data, C()))
    assert "drafter" in out.lower()


def test_drafter_tax_is_quiet_on_dense():
    class C(_Cfg):
        DRAFTER_MODEL_PATH = "/models/llama-1b"
    data = {"model": {"is_moe": False}}
    assert not any("drafter" in t.lower() for t in diagnose.taxes(data, C()))


def test_default_stream_false_tax_fires():
    class C(_Cfg):
        DEFAULT_STREAM = False
    out = " ".join(diagnose.taxes({}, C()))
    assert "DEFAULT_STREAM" in out


def test_a_clean_box_has_no_taxes():
    data = {"storage": {"apst_active": False, "scheduler": "none"},
            "gpu": {"foreign_processes": []},
            "model": {"is_moe": True}}
    assert diagnose.taxes(data, _Cfg()) == []


def test_module_imports_without_cuda():
    import importlib
    importlib.reload(diagnose)      # must not raise on a CUDA-less box
