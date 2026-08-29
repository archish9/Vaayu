"""Who else is holding the GPU when Vaayu loads.

Vaayu derives its pinning budget ONCE, from whatever VRAM is free at load, so a
forgotten process is a permanent tax on every token for the life of the process.
This is not hypothetical: a docker-compose stack with `restart=unless-stopped`
came back 13 s into every boot and quietly held 9.3 GiB.

Parsing is kept pure so it is testable without a GPU; only `audit()` shells out.
"""
import pytest

from vaayu.gpu_audit import (GpuUser, blocking_users, compose_report,
                             device_uuid, foreign_gpu_users,
                             parse_compute_apps, parse_container_id)

GPU_A = "GPU-a6e6ee6b-dd04-2c6f-77bc-528f104a40db"
GPU_B = "GPU-b1111111-2222-3333-4444-555555555555"

SAMPLE = f"""\
{GPU_A}, 2726, /usr/local/bin/python3.10, 590
{GPU_A}, 3179, /usr/local/bin/python3.10, 4064
{GPU_A}, 227646, ./venv/bin/python, 762
"""


def test_parse_reads_pid_name_and_megabytes():
    users = parse_compute_apps(SAMPLE)
    assert [u.pid for u in users] == [2726, 3179, 227646]
    assert users[0].name == "/usr/local/bin/python3.10"
    assert users[1].mib == 4064


def test_parse_tolerates_the_no_processes_case():
    assert parse_compute_apps("") == []
    assert parse_compute_apps("\n\n") == []


def test_parse_skips_lines_it_cannot_read_rather_than_raising():
    # nvidia-smi prints "[N/A]" for memory it cannot attribute; one odd row must
    # not cost us the whole report.
    assert len(parse_compute_apps(SAMPLE + f"{GPU_A}, 999, weird, [N/A]\n")) == 3


def test_our_own_process_is_not_a_foreign_holder():
    users = parse_compute_apps(SAMPLE)
    assert [u.pid for u in foreign_gpu_users(users, own_pid=227646)] == [2726, 3179]


def test_a_container_id_is_read_off_the_cgroup():
    cgroup = ("0::/system.slice/docker-4bd0ebbfc63559427c943ee6d5d8e1c8fe9917"
              "406b06f15a6bfe5b13f1e1950a.scope\n")
    assert parse_container_id(cgroup) == "4bd0ebbfc635"


def test_a_plain_process_has_no_container_id():
    assert parse_container_id("0::/user.slice/user-1000.slice/session-3.scope\n") is None


def test_the_report_names_the_holder_the_cost_and_the_exact_command():
    users = [GpuUser(pid=3179, name="python3.10", mib=4064, container="4bd0ebbfc635"),
             GpuUser(pid=5000, name="python", mib=762, container=None)]
    report = compose_report(users)
    assert "4.71 GiB" in report, "say the total held"
    assert "3.97 GiB" in report, "and say it per holder"
    assert "docker stop 4bd0ebbfc635" in report, "a container is stopped, not killed"
    assert "kill 5000" in report, "a plain process gets a plain kill"
    assert "REQUIRE_EXCLUSIVE_GPU" in report, "name the escape hatch"


def test_two_processes_in_one_container_give_one_stop_command():
    # A container with N worker processes is still stopped once. Printing the
    # same command twice reads like two separate things to fix.
    users = [GpuUser(pid=3299, name="python", mib=2132, container="4bd0ebbfc635"),
             GpuUser(pid=3300, name="python", mib=2132, container="4bd0ebbfc635")]
    report = compose_report(users)
    assert report.count("docker stop 4bd0ebbfc635") == 1
    assert "pid 3299" in report and "pid 3300" in report, "still name both holders"


def test_the_report_totals_the_memory():
    users = [GpuUser(pid=1, name="a", mib=1024, container=None),
             GpuUser(pid=2, name="b", mib=1024, container=None)]
    assert "2.00 GiB" in compose_report(users)


def test_no_report_when_the_gpu_is_ours_alone():
    assert compose_report([]) is None


# ── the gate ─────────────────────────────────────────────────────────────────

def test_the_server_refuses_to_load_when_the_gpu_is_shared(monkeypatch):
    """Refusing beats loading slowly and silently: the penalty is invisible in
    the output and permanent for the life of the process."""
    from vaayu.server.app import ServerConfig, _check_gpu_is_exclusive
    import vaayu.gpu_audit as ga

    monkeypatch.setattr(ga, "audit", lambda **kw: [
        GpuUser(pid=3179, name="python3.10", mib=4064, container="4bd0ebbfc635")])

    with pytest.raises(SystemExit) as e:
        _check_gpu_is_exclusive(ServerConfig(model_path="/ckpt"))
    assert "docker stop 4bd0ebbfc635" in str(e.value)


def test_the_refusal_is_the_default(monkeypatch):
    from vaayu.server.app import ServerConfig
    assert ServerConfig(model_path="/x").require_exclusive_gpu is True


def test_an_exclusive_gpu_loads_normally(monkeypatch, tmp_path):
    from vaayu.server.app import ServerConfig, create_app
    from fastapi.testclient import TestClient
    import vaayu.gpu_audit as ga

    monkeypatch.setattr(ga, "audit", lambda **kw: [])
    loaded = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kw):
            loaded["path"] = path
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert loaded["path"] == "/ckpt"


def test_the_flag_can_be_turned_off_for_a_deliberately_shared_box(monkeypatch, capsys):
    from vaayu.server.app import ServerConfig, _check_gpu_is_exclusive
    import vaayu.gpu_audit as ga

    monkeypatch.setattr(ga, "audit", lambda **kw: [
        GpuUser(pid=1, name="x", mib=4064, container=None)])
    _check_gpu_is_exclusive(ServerConfig(model_path="/ckpt",
                                         require_exclusive_gpu=False))
    assert "WARNING" in capsys.readouterr().out, "opting out still says what it costs"


def test_the_library_path_warns_but_never_refuses(monkeypatch, capsys):
    """A refusal in from_pretrained would break every existing script. The
    server is the place that says no; the library says loudly what it costs."""
    import vaayu.gpu_audit as ga
    from vaayu.auto_model import warn_if_gpu_is_shared

    monkeypatch.setattr(ga, "audit", lambda **kw: [
        GpuUser(pid=7, name="python", mib=2048, container=None)])
    warn_if_gpu_is_shared()
    out = capsys.readouterr().out
    assert "2.00 GiB" in out and "kill 7" in out


# ── which GPU, and how big (B11) ─────────────────────────────────────────────
#
# A bare CUDA context is 128 MiB on this box (measured 2026-08-29), so "any
# foreign process at any size on any GPU" refused the boot over a stray notebook
# on a GPU the server never touches.

def test_the_uuid_column_is_read():
    assert [u.uuid for u in parse_compute_apps(SAMPLE)] == [GPU_A, GPU_A, GPU_A]


def test_a_process_name_containing_a_comma_is_parsed_not_skipped():
    # nvidia-smi does not quote its CSV, so a name with a comma made the row
    # four fields and the old `len(parts) != 3` check dropped it silently --
    # under-reporting is the one thing this module must never do.
    text = f"{GPU_A}, 42, /opt/my, app/python, 2048\n"
    users = parse_compute_apps(text)
    assert len(users) == 1, "a comma in the name is not a reason to lose a holder"
    assert users[0].name == "/opt/my, app/python"
    assert users[0].mib == 2048


def test_a_holder_on_another_gpu_does_not_count():
    text = f"{GPU_A}, 1, a, 4096\n{GPU_B}, 2, b, 4096\n"
    users = foreign_gpu_users(parse_compute_apps(text), own_pid=999, uuid=GPU_B)
    assert [u.pid for u in users] == [2], "cuda:1's tenant cannot block a cuda:0 load"


def test_no_uuid_means_every_gpu_as_before():
    text = f"{GPU_A}, 1, a, 4096\n{GPU_B}, 2, b, 4096\n"
    users = foreign_gpu_users(parse_compute_apps(text), own_pid=999)
    assert [u.pid for u in users] == [1, 2], "the library path audits the whole box"


def test_a_bare_context_does_not_block_but_is_still_reported():
    users = [GpuUser(pid=1, name="notebook", mib=128, container=None)]
    assert blocking_users(users, min_mib=256) == [], "128 MiB is a context, not a tenant"
    report = compose_report(users, min_mib=256)
    assert report is not None, "below the threshold is not the same as invisible"
    assert "pid 1" in report


def test_a_real_tenant_still_blocks():
    users = [GpuUser(pid=1, name="notebook", mib=128, container=None),
             GpuUser(pid=2, name="trainer", mib=9532, container=None)]
    assert [u.pid for u in blocking_users(users, min_mib=256)] == [2]


def test_the_report_totals_only_what_actually_blocks():
    # The headline number is the argument for refusing, so it must be the
    # memory the refusal is about -- not that plus noise under the threshold.
    users = [GpuUser(pid=1, name="notebook", mib=128, container=None),
             GpuUser(pid=2, name="trainer", mib=1024, container=None)]
    report = compose_report(users, min_mib=256)
    assert "1.00 GiB" in report
    assert "kill 2" in report
    assert "kill 1" not in report, "do not tell anyone to kill a bare context"


# ── the gate, with a device and a threshold ──────────────────────────────────

def test_a_holder_below_the_threshold_does_not_refuse_the_boot(monkeypatch, capsys):
    from vaayu.server.app import ServerConfig, _check_gpu_is_exclusive
    import vaayu.gpu_audit as ga

    monkeypatch.setattr(ga, "audit", lambda **kw: [
        GpuUser(pid=1, name="python", mib=128, container=None)])
    _check_gpu_is_exclusive(ServerConfig(model_path="/ckpt"))     # must not raise
    assert "pid 1" in capsys.readouterr().out, "still say what is there"


def test_the_gate_audits_the_configured_device(monkeypatch):
    from vaayu.server.app import ServerConfig, _check_gpu_is_exclusive
    import vaayu.gpu_audit as ga

    seen = {}

    def fake_audit(device=None, **kw):
        seen["device"] = device
        return []

    monkeypatch.setattr(ga, "audit", fake_audit)
    _check_gpu_is_exclusive(ServerConfig(model_path="/ckpt", device="cuda:1"))
    assert seen["device"] == "cuda:1"


def test_the_threshold_default_is_above_a_bare_cuda_context():
    from vaayu.server.app import ServerConfig
    assert ServerConfig(model_path="/x").exclusive_gpu_min_mib >= 256


def test_a_cpu_device_has_no_uuid_to_filter_on():
    assert device_uuid("cpu") is None


def test_an_unreadable_device_falls_back_to_auditing_everything(monkeypatch):
    # Failing open on the REPORT: a device torch cannot describe must not make
    # the audit silently pass. Filtering nothing over-reports, which is safe.
    import vaayu.gpu_audit as ga

    def boom(_idx):
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(ga, "_torch_device_uuid", boom)
    assert device_uuid("cuda:0") is None
