"""env.json host section records the cgroup cap and the effective basis."""
import functools

from tests.test_cgroup_aware_budget import G, HOST, TOTAL, _leaf
from vaayu import cgroup_limits
from vaayu.server import envinfo


def _host(monkeypatch, root, proc):
    monkeypatch.setattr(envinfo, "cgroup_cap_bytes", functools.partial(
        cgroup_limits.cgroup_cap_bytes, cgroup_root=root, proc_cgroup=proc))
    monkeypatch.setattr(envinfo, "effective_available_bytes", functools.partial(
        cgroup_limits.effective_available_bytes, host_available=HOST,
        mem_total=TOTAL, cgroup_root=root, proc_cgroup=proc))
    return envinfo._host_section(envinfo._Collector())


def test_envinfo_reports_cap_and_basis_when_capped(tmp_path, monkeypatch):
    root, proc = _leaf(tmp_path, max_=28 * G, current=10 * G, anon=10 * G)
    host = _host(monkeypatch, root, proc)
    assert host["cgroup_memory_cap_bytes"] == 28 * G
    assert host["effective_available_bytes"] == 18 * G < HOST


def test_envinfo_reports_null_cap_when_uncapped(tmp_path, monkeypatch):
    root, proc = _leaf(tmp_path)
    host = _host(monkeypatch, root, proc)
    assert host["cgroup_memory_cap_bytes"] is None
    assert host["effective_available_bytes"] == HOST
