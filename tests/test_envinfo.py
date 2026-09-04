"""env.json is the file that closes the support loop.

Every load-bearing fact in the 2026-09-03 "server is too slow" diagnosis came
from OUTSIDE logs/ -- above all the gitignored config_local.py, which is where
the user's real problem turned out to live. For a stranger's zip, none of it is
available. This module is the fix, so these tests care about two things:

  * it reports the RIGHT forward for each of the four dispatch rows, derived
    from the model rather than assumed (CLAUDE.md 5), and
  * it never, under any failure, prevents a server from booting.
"""
import ast
import json
import os
import platform
import re
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

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


# ── vaayu's own commit/branch ─────────────────────────────────────────────────

def test_vaayu_commit_and_branch_are_reported_for_a_real_checkout():
    import subprocess
    data = envinfo.collect(_model(is_moe=False), _config())
    expected_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=5).stdout.strip()
    assert data["vaayu"]["branch"] == expected_branch
    assert data["vaayu"]["commit"]


def test_vaayu_section_is_null_without_a_dot_git(tmp_path, monkeypatch):
    # I5: `git -C <path> ...` does not require `path` itself to hold a repo --
    # if it does not, git walks UP the tree and finds whatever `.git` is
    # first, which for a venv nested inside a user's OWN project is the
    # user's application repo, not Vaayu's. Simulate the pip-installed case
    # (a real setup.py path with no `.git` two levels up) and require the
    # honest "installed, not a checkout" null rather than a foreign commit.
    from vaayu.server import envinfo as envinfo_mod
    fake_file = tmp_path / "installed" / "vaayu" / "server" / "envinfo.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("")
    monkeypatch.setattr(envinfo_mod, "__file__", str(fake_file))
    c = envinfo_mod._Collector()
    result = envinfo_mod._vaayu_section(c)
    assert result == {"commit": None, "dirty": None, "branch": None}
    assert c.errors == []


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


def test_a_model_with_no_prefix_cache_reports_clean_not_as_an_error():
    # Prefix reuse auto-disables on a non-EXL3 / dense checkpoint (the bare
    # DynamicCache has no to_cpu/to_device/resident_bytes), leaving
    # _prefix_cache legitimately None -- the same "unset, not broken"
    # condition Item 7 already covers for the expert caches. Before this fix,
    # prefix_cache_gb was the one budgets.* field that skipped the null-check
    # helper, so a perfectly healthy run reported an AttributeError.
    m = _model(is_moe=False, layers=80)
    m._prefix_cache = None
    # checkpoint_path: unrelated to this test's subject, but on_disk_gb()
    # reads it unconditionally -- give the fixture one so the full
    # collect_errors assertion below isn't tripped by a different, already
    # -covered field.
    m.checkpoint_path = "/nonexistent/fake-checkpoint"
    data = envinfo.collect(m, _config())
    assert data["budgets"]["prefix_cache_gb"] is None
    assert data["collect_errors"] == [], data["collect_errors"]


def test_a_dense_models_null_caches_report_clean_not_as_errors():
    # Item 7: on a dense model, vaayu_base.py sets _expert_cache and
    # _expert_vram_cache to None (the attribute EXISTS, it is legitimately
    # unset) -- not missing, and not broken. Before this fix, accessing
    # `.budget_bytes` off None raised AttributeError three times, so a
    # perfectly healthy dense report looked exactly like something was wrong.
    m = _model(is_moe=False, layers=80)
    m._expert_cache = None
    m._expert_vram_cache = None
    data = envinfo.collect(m, _config())
    b = data["budgets"]
    assert b["expert_ram_cache_gb"] is None
    assert b["max_per_layer"] is None
    assert b["expert_vram_cache_gb"] is None
    budget_errors = [e for e in data["collect_errors"] if e.startswith("budgets.")]
    assert budget_errors == [], budget_errors


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
    # Checkout-location independent (review round 2, item 6): a CI box at
    # /builds/... or /workspace/... contains none of the literals above, so
    # this is the assertion that holds regardless of where the repo sits.
    assert sys.executable not in dumped
    assert data["config"]["model_path"]["name"] == "Big-Model-exl3"
    assert len(data["config"]["model_path"]["hash"]) == 10


def test_two_different_paths_redact_differently():
    a = envinfo.redact_path("/home/a/models/X")
    b = envinfo.redact_path("/home/b/models/X")
    assert a["name"] == b["name"] and a["hash"] != b["hash"]


def test_redacting_nothing_yields_nothing():
    assert envinfo.redact_path(None) is None


def test_redact_path_is_insensitive_to_a_trailing_slash():
    # MODEL_PATH is hand-typed in config_local.py -- a trailing slash must
    # not make one checkpoint look like two (review round 2, item 7).
    a = envinfo.redact_path("/home/a/models/Big-exl3")
    b = envinfo.redact_path("/home/a/models/Big-exl3/")
    assert a["name"] == b["name"] and a["hash"] == b["hash"]


def test_a_colon_joined_path_list_is_fully_scrubbed():
    # The colon hole (review round 2, item 1): a colon-separated list like
    # PYTHONPATH puts its second path right after a ':', which import-error
    # text quotes verbatim.
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=["PYTHONPATH=/home/a/x:/home/b/y"])
    assert "/home/" not in data["warnings"][0]


def test_urls_are_not_mangled_by_the_scrubber():
    url = "https://huggingface.co/archish9/Model-exl3"
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=[f"downloading {url}"])
    assert url in data["warnings"][0], "a URL is public; scrubbing it breaks it"


def test_sysfs_paths_are_left_alone():
    p = "/sys/module/nvme_core/parameters/default_ps_max_latency_us"
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=[f"read {p}"])
    assert p in data["warnings"][0], "sysfs paths name no user and no checkpoint"


def test_trailing_punctuation_is_not_folded_into_the_redacted_path():
    data = envinfo.collect(_model(is_moe=False), _config(),
                           warnings=["missing /home/a/models/x/y.json."])
    w = data["warnings"][0]
    assert w.endswith("."), "the sentence's full stop must survive outside the match"
    assert "/home/" not in w


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


def test_on_disk_gb_reads_a_real_checkpoint_not_the_cwd(tmp_path):
    # `on_disk_gb is None` (above) is also what the ORIGINAL BUG produced --
    # re-break the read to model.ckpt_path and that assertion still passes.
    # This pins the fix with a real, non-None number (review round 2, item 5).
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "shard.safetensors").write_bytes(b"x" * 2_000_000)  # exactly 2 MB
    m = _model(is_moe=False)
    m.checkpoint_path = str(ckpt)
    data = envinfo.collect(m, _config())
    assert data["model"]["on_disk_gb"] == round(2_000_000 / 1e9, 2)


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


def test_writing_something_that_blows_the_recursion_limit_does_not_raise(tmp_path):
    # RecursionError is not a subclass of OSError/TypeError/ValueError -- the
    # old three-tuple except clause in write() let it straight through
    # (review round 2, item 5). 10,000 levels reliably exceeds the default
    # recursion limit (1000) inside json.dumps without needing to touch it.
    deep = []
    cur = deep
    for _ in range(10_000):
        nxt = []
        cur.append(nxt)
        cur = nxt
    envinfo.write(tmp_path / "env.json", {"deep": deep})     # must not raise


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
    """Ties the fakes above to reality for VramPlan, MoEBlockInfo and the
    three cache classes. It does NOT check the model class itself -- see
    test_envinfo_model_attrs_exist_on_the_real_init below for that; a
    `name_or_path` attribute that does not exist on the real model would not
    have been caught here.
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


def test_envinfo_model_attrs_exist_on_the_real_init():
    """ast-walks VaayuBaseModel.__init__ (no checkpoint, no GPU needed) --
    plus any `self.<method>()` it calls, transitively (e.g. `is_moe` is set
    inside `_detect_moe()`, which `__init__` calls at construction time, not
    inline) -- for every `self.X = ...` / `self.X: T = ...` assignment
    target, and asserts that every attribute envinfo._model_section /
    _budgets_section reads is among them. This is exactly the check that
    would have caught `name_or_path` -- which is not one of them -- before a
    stranger's zip did (review round 2, item 4).
    """
    import vaayu.vaayu_base as vaayu_base_mod

    tree = ast.parse(Path(vaayu_base_mod.__file__).read_text())
    class_def = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == "VaayuBaseModel")
    methods = {n.name: n for n in class_def.body if isinstance(n, ast.FunctionDef)}

    def self_attrs(target):
        if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            yield target.attr
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from self_attrs(elt)

    def self_calls(fn_node):
        for n in ast.walk(fn_node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "self"):
                yield n.func.attr

    names, visited, queue = set(), set(), ["__init__"]
    while queue:
        mname = queue.pop()
        if mname in visited or mname not in methods:
            continue
        visited.add(mname)
        fn = methods[mname]
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(self_attrs(t))
            elif isinstance(node, ast.AnnAssign):
                names.update(self_attrs(node.target))
        queue.extend(self_calls(fn))

    required = {"is_moe", "config", "loader_strategy", "_vram_plan",
               "_expert_cache", "_expert_vram_cache", "_prefix_cache",
               "_moe_geometry", "_moe_blocks", "max_seq_len", "kv_cache_mode",
               "checkpoint_path", "model_local_path"}
    missing = required - names
    assert not missing, (
        f"envinfo reads attribute(s) VaayuBaseModel.__init__ never sets "
        f"(directly or via a method it calls): {missing}")


def test_path_fields_allowlist_is_tied_to_the_real_config():
    from vaayu.server.app import ServerConfig
    assert set(envinfo._PATH_FIELDS) <= {f.name for f in dc_fields(ServerConfig)}


def test_collect_with_no_model_omits_the_model_sections():
    """--diagnose runs before any weights are loaded. A report full of
    AttributeErrors reads as a broken tool, which is the opposite of the
    point."""
    from vaayu.server import envinfo

    class _Cfg:
        device = "cuda:0"
        model_path = "/models/x"

    data = envinfo.collect(None, _Cfg())
    assert "model" not in data
    assert "budgets" not in data
    assert data["collect_errors"] == []
    assert "host" in data and "storage" in data


def test_gpu_section_reports_foreign_processes(monkeypatch):
    """Not in the Task 8 brief's given code, added while validating it end to
    end: `_gpu_section` had no field at all for another process's VRAM, so
    `diagnose`'s foreign-GPU tax could never fire against a real box (it did
    not, until this landed -- see task-8-report.md). A holder under
    `exclusive_gpu_min_mib` never reaches `collected_warnings()` (app.py's
    `_check_gpu_is_exclusive` only prints the literal word "WARNING" for a
    BLOCKING holder), so this has to be read as data, not scraped from a log
    line."""
    from vaayu import gpu_audit

    class _FakeUser:
        pid = 1229945
        name = "rustdesk"          # no leading "/" -- keeps this test clear
        mib = 148                  # of the separate path-scrubbing behaviour

        @property
        def gib(self):
            return self.mib / 1024

    monkeypatch.setattr(gpu_audit, "audit", lambda device=None: [_FakeUser()])
    data = envinfo.collect(_model(is_moe=False), _config())
    assert data["gpu"]["foreign_processes"] == [
        {"pid": 1229945, "cmd": "rustdesk", "gib": round(148 / 1024, 2)}]


def test_gpu_section_foreign_processes_is_empty_on_a_clean_box(monkeypatch):
    from vaayu import gpu_audit

    monkeypatch.setattr(gpu_audit, "audit", lambda device=None: [])
    data = envinfo.collect(_model(is_moe=False), _config())
    assert data["gpu"]["foreign_processes"] == []


def test_device_resolves_from_uppercase_DEVICE_on_a_module_shaped_config(monkeypatch):
    """`diagnose` passes the config MODULE, which defines `DEVICE` (uppercase),
    not `device` -- collect() must read it rather than silently falling back
    to `cuda:0` regardless of what the box is actually configured for."""
    from vaayu import gpu_audit
    monkeypatch.setattr(gpu_audit, "audit", lambda device=None: [])

    class _Cfg:
        DEVICE = "cuda:1"

    data = envinfo.collect(None, _Cfg())
    assert data["gpu"]["device"] == "cuda:1"


def test_device_still_resolves_from_lowercase_device_on_a_dataclass_config(monkeypatch):
    """Regression guard for the real server path: a `ServerConfig` (lowercase
    `device`) must resolve exactly as it did before the uppercase fallback
    was added."""
    from vaayu import gpu_audit
    monkeypatch.setattr(gpu_audit, "audit", lambda device=None: [])

    data = envinfo.collect(_model(is_moe=False), _config(device="cuda:1"))
    assert data["gpu"]["device"] == "cuda:1"
