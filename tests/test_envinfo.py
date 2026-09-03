"""env.json is the file that closes the support loop.

Every load-bearing fact in the 2026-09-03 "server is too slow" diagnosis came
from OUTSIDE logs/ -- above all the gitignored config_local.py, which is where
the user's real problem turned out to live. For a stranger's zip, none of it is
available. This module is the fix, so these tests care about two things:

  * it reports the RIGHT forward for each of the four dispatch rows, derived
    from the model rather than assumed (CLAUDE.md 5), and
  * it never, under any failure, prevents a server from booting.
"""
import json
import os
import platform
import re

import pytest

from vaayu.server import envinfo


# ── fake models, one per dispatch row (CLAUDE.md 0) ──────────────────────────

class _Plan:
    max_seq_len = 8192
    reserve_bytes = 3_758_096_384
    pinning_budget = 10_425_064_653
    s_max = 8192
    reason = "fake plan"


class _Cache:
    def __init__(self, budget, max_per_layer=0):
        self.budget_bytes = budget
        self.max_per_layer = max_per_layer


class _Block:
    def __init__(self, style, forward_name):
        self.expert_style = style
        self.forward_name = forward_name
        self.layer_idx = 0


class _Strategy:
    pass


class EXL3LoaderStrategy(_Strategy):
    pass


def _model(*, is_moe, style=None, forward=None, strategy=_Strategy,
           layers=94, experts=128, top_k=8, expert_bytes=12_373_196):
    m = type("FakeVaayu", (), {})()
    m.is_moe = is_moe
    m.max_seq_len = 8192
    m.kv_cache_mode = "bf16"
    m.loader_strategy = strategy()
    m.config = type("C", (), {"architectures": ["FakeForCausalLM"],
                              "num_hidden_layers": layers})()
    m._vram_plan = _Plan()
    m._expert_cache = _Cache(15_190_000_000, max_per_layer=13)
    m._expert_vram_cache = _Cache(5_900_000_000)
    m._prefix_cache = _Cache(5_270_000_000)
    if is_moe:
        m._moe_geometry = {"expert_bytes": expert_bytes, "num_moe_layers": layers,
                           "top_k": top_k, "num_experts": experts,
                           "working_set_bytes": top_k * layers * expert_bytes}
        m._moe_blocks = [_Block(style, forward)]
    else:
        m._moe_geometry = None
        m._moe_blocks = []
    return m


def _config(**over):
    from vaayu.server.app import ServerConfig
    base = dict(model_path="/home/someone/models/Big-Model-exl3",
                drafter_model_path=None, log_dir="logs")
    base.update(over)
    return ServerConfig(**base)


# ── the dispatch row is REPORTED, never guessed ──────────────────────────────

@pytest.mark.parametrize("style,forward", [
    ("fused", "_make_fused_sparse_forward_exl3"),
    ("fused", "_make_fused_sparse_forward"),
    ("module_list", "_make_modulelist_sparse_forward"),
])
def test_each_moe_dispatch_row_is_reported_as_itself(style, forward):
    data = envinfo.collect(_model(is_moe=True, style=style, forward=forward),
                           _config())
    assert data["model"]["expert_style"] == style
    assert data["model"]["forward"] == forward


def test_a_dense_model_is_reported_as_dense_with_no_expert_geometry():
    data = envinfo.collect(_model(is_moe=False, layers=80), _config())
    assert data["model"]["is_moe"] is False
    assert data["model"]["expert_style"] is None
    assert data["model"]["forward"] == "dense"
    assert data["model"]["num_experts"] is None
    assert data["model"]["num_layers"] == 80


def test_moe_geometry_is_read_from_the_model_not_assumed():
    # CLAUDE.md 5: no literal model numbers anywhere. Two different geometries
    # must produce two different reports from the same code.
    qwen = envinfo.collect(_model(is_moe=True, style="fused",
                                  forward="_make_fused_sparse_forward_exl3",
                                  layers=94, experts=128, top_k=8,
                                  expert_bytes=12_373_196), _config())
    mini = envinfo.collect(_model(is_moe=True, style="module_list",
                                  forward="_make_modulelist_sparse_forward",
                                  layers=62, experts=256, top_k=8,
                                  expert_bytes=8_850_000), _config())
    assert qwen["model"]["num_experts"] == 128
    assert mini["model"]["num_experts"] == 256
    assert qwen["model"]["num_moe_layers"] == 94
    assert mini["model"]["num_moe_layers"] == 62
    assert qwen["model"]["working_set_gb"] != mini["model"]["working_set_gb"]


def test_the_quant_strategy_is_named():
    data = envinfo.collect(_model(is_moe=True, style="fused",
                                  forward="_make_fused_sparse_forward_exl3",
                                  strategy=EXL3LoaderStrategy), _config())
    assert data["model"]["loader_strategy"] == "EXL3LoaderStrategy"


# ── budgets ──────────────────────────────────────────────────────────────────

def test_budgets_come_off_the_vram_plan_and_the_caches():
    data = envinfo.collect(_model(is_moe=True, style="fused",
                                  forward="_make_fused_sparse_forward_exl3"),
                           _config())
    b = data["budgets"]
    assert b["max_seq_len"] == 8192
    assert round(b["reserve_gib"], 2) == 3.50
    assert b["max_per_layer"] == 13
    assert b["kv_cache_mode"] == "bf16"
    assert b["expert_vram_cache_gb"] > 0 and b["prefix_cache_gb"] > 0


# ── redaction ────────────────────────────────────────────────────────────────

def test_absolute_paths_are_redacted_to_a_basename_and_a_hash():
    # Regression for the review finding: the fake config path was checked,
    # but real absolute paths (sys.executable, collect_errors messages,
    # warnings) leaked straight through on the SAME call. Scan the whole
    # dumped report, not one known literal.
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=[f"[Vaayu] saw {__file__}"])
    dumped = json.dumps(data)
    assert "/home/someone" not in dumped, "a username must not ship in the bundle"
    for pattern in (r"/home/[^\"\\]*", r"/Users/[^\"\\]*", r"/root/[^\"\\]*"):
        assert not re.search(pattern, dumped), (
            f"a raw absolute path matching {pattern!r} leaked into the report")
    assert data["config"]["model_path"]["name"] == "Big-Model-exl3"
    assert len(data["config"]["model_path"]["hash"]) == 10


def test_two_different_paths_redact_differently():
    a = envinfo.redact_path("/home/a/models/X")
    b = envinfo.redact_path("/home/b/models/X")
    assert a["name"] == b["name"] and a["hash"] != b["hash"]


def test_redacting_nothing_yields_nothing():
    assert envinfo.redact_path(None) is None


# ── fail-soft: a log file may never stop a boot ──────────────────────────────

def test_a_model_missing_everything_still_produces_a_report():
    broken = type("Broken", (), {})()
    data = envinfo.collect(broken, _config())     # must not raise
    assert data["collect_errors"], "a failure must be NAMED, not silently null"
    assert data["model"]["forward"] is None
    assert data["model"]["on_disk_gb"] is None, (
        "Path('') is Path('.'), the server's cwd -- a missing checkpoint_path "
        "must not fall through to walking the filesystem and reporting the "
        "cwd's size as the checkpoint's")


def test_a_malformed_device_string_does_not_crash_collect():
    # A typo in the user-editable config_local.py -- device="cuda:abc" --
    # must not take down an otherwise-bootable server.
    data = envinfo.collect(_model(is_moe=False), _config(device="cuda:abc"))
    assert data["gpu"]["device"] == "cuda:abc"
    assert data["model"]["is_moe"] is False        # the rest still collected
    assert any("gpu.device_index" in e for e in data["collect_errors"])


def test_an_unpatched_moe_block_records_why_forward_is_null():
    # forward_name is the single most load-bearing field in the file; a null
    # here must never look the same as "this model is dense".
    m = _model(is_moe=True, style="fused", forward=None)
    data = envinfo.collect(m, _config())
    assert data["model"]["forward"] is None
    assert any("forward_name" in e for e in data["collect_errors"])


def test_python_executable_is_redacted_not_raw():
    data = envinfo.collect(_model(is_moe=False), _config())
    exe = data["python"]["executable"]
    assert exe["name"] and exe["hash"]
    assert isinstance(data["python"]["is_venv"], bool)


def test_one_broken_config_field_does_not_null_the_whole_config_section():
    cfg = _config()
    del cfg.model_path      # simulate an attribute access failure
    data = envinfo.collect(_model(is_moe=False), cfg)
    assert data["config"]["model_path"] is None
    assert data["config"]["log_dir"]["name"] == "logs", (
        "a per-field wrapper, not a per-section one -- otherwise one broken "
        "config attribute costs the whole config section, which is the most "
        "load-bearing section in the file")
    assert any("model_path" in e for e in data["collect_errors"])


def test_one_broken_field_does_not_null_its_whole_section():
    m = _model(is_moe=True, style="fused", forward="_make_fused_sparse_forward_exl3")
    del m._expert_vram_cache
    data = envinfo.collect(m, _config())
    assert data["budgets"]["expert_vram_cache_gb"] is None
    assert data["budgets"]["max_seq_len"] == 8192, (
        "a per-field wrapper, not a per-section one -- otherwise one missing "
        "attribute costs the whole section")
    assert any("expert_vram_cache_gb" in e for e in data["collect_errors"])


# ── the rest of the snapshot ─────────────────────────────────────────────────

def test_the_vaayu_env_vars_are_dumped_including_the_unset_ones(monkeypatch):
    monkeypatch.setenv("VAAYU_LOOKAHEAD", "1")
    monkeypatch.delenv("VAAYU_RING_POOL_N", raising=False)
    data = envinfo.collect(_model(is_moe=False), _config())
    assert data["env"]["VAAYU_LOOKAHEAD"] == "1"
    assert "VAAYU_RING_POOL_N" in data["env"]
    assert data["env"]["VAAYU_RING_POOL_N"] is None


def test_an_unknown_vaayu_var_set_by_the_user_is_still_captured(monkeypatch):
    monkeypatch.setenv("VAAYU_SOMETHING_NEW", "7")
    data = envinfo.collect(_model(is_moe=False), _config())
    assert data["env"]["VAAYU_SOMETHING_NEW"] == "7"


def test_warnings_are_carried_into_the_report():
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=["[Vaayu-NVMe] WARNING: APST is active"])
    assert data["warnings"] == ["[Vaayu-NVMe] WARNING: APST is active"]


def test_the_report_is_json_serialisable(tmp_path):
    data = envinfo.collect(_model(is_moe=True, style="module_list",
                                  forward="_make_modulelist_sparse_forward"),
                           _config())
    path = tmp_path / "env.json"
    envinfo.write(path, data)
    assert json.loads(path.read_text())["model"]["expert_style"] == "module_list"


def test_writing_to_an_unwritable_place_does_not_raise(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    envinfo.write(blocker / "env.json", {"a": 1})     # must not raise


def test_the_host_and_storage_sections_are_present():
    data = envinfo.collect(_model(is_moe=False), _config())
    for section in ("vaayu", "python", "host", "gpu", "storage", "model",
                    "budgets", "config", "env", "warnings", "collect_errors"):
        assert section in data, section
    # These resolve from stdlib/module globals, not sysfs, so they hold on
    # any box -- unlike the fields above, this can actually fail.
    assert data["host"]["kernel"] == platform.release()
    assert data["host"]["cpu_count"] == os.cpu_count()
    assert isinstance(data["storage"]["o_direct"], bool)


# ── conformance: the fakes above must match the real classes ─────────────────

def test_envinfo_dependencies_exist_on_the_real_classes():
    """Ties the fakes above to reality. `on_disk_gb` reading a `name_or_path`
    attribute that does not exist on the real model is exactly the class of
    bug this test exists to catch before a stranger's zip does.
    """
    from vaayu.vram_plan import VramPlan
    from vaayu.moe_patch import MoEBlockInfo
    from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
    from vaayu.prefix_cache import PrefixCacheStore

    plan = VramPlan(max_seq_len=8192, reserve_bytes=1, pinning_budget=1,
                    s_max=8192, reason="x")
    for attr in ("reserve_bytes", "pinning_budget", "s_max", "reason"):
        assert hasattr(plan, attr)

    block = MoEBlockInfo(layer_idx=0, layer_name="model.layers.0",
                         module_path="model.layers.0.block_sparse_moe",
                         moe_module=object(), router_attr="gate",
                         experts_attr="experts", expert_style="fused")
    for attr in ("expert_style", "forward_name", "layer_idx"):
        assert hasattr(block, attr)

    ram = ExpertRAMCache(1000, max_per_layer=1)
    assert hasattr(ram, "budget_bytes") and hasattr(ram, "max_per_layer")

    vram = ExpertVRAMCache(1000)
    assert hasattr(vram, "budget_bytes")

    prefix = PrefixCacheStore(1000)
    assert hasattr(prefix, "budget_bytes")
