"""O_DIRECT-vs-buffered decision must use the cgroup-aware RAM basis."""
import types

import pytest

from vaayu.persist import safetensor_model_persister as smp

GIB = 1024 ** 3
HOST_TOTAL = 64 * GIB
HOST_AVAIL = 48 * GIB


@pytest.fixture
def model_dir(tmp_path):
    # Sparse 10 GiB file: only its size matters to the decision.
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.truncate(10 * GIB)
    return str(tmp_path)


@pytest.fixture(autouse=True)
def host(monkeypatch):
    vm = types.SimpleNamespace(total=HOST_TOTAL, available=HOST_AVAIL)
    monkeypatch.setattr(smp.psutil, "virtual_memory", lambda: vm)
    monkeypatch.setattr(smp, "expert_arena_bytes", lambda d: 0)


def _use(monkeypatch, model_dir, eff):
    monkeypatch.setattr(smp, "effective_available_bytes", lambda: eff)
    return smp._should_use_direct_io(model_dir, 0)


def test_tight_cgroup_cap_flips_to_direct_io(monkeypatch, model_dir):
    # Host basis: (48-8)/10 = 4.0 >= 0.40 -> buffered.
    assert _use(monkeypatch, model_dir, HOST_AVAIL) is False
    # Cgroup basis 10 GiB: (10-8)/10 = 0.2 < 0.40 -> O_DIRECT.
    assert _use(monkeypatch, model_dir, 10 * GIB) is True


def test_uncapped_matches_host_decision(monkeypatch, model_dir):
    assert _use(monkeypatch, model_dir, HOST_AVAIL) is False


def test_total_stays_host_for_ratio(monkeypatch, model_dir):
    seen = {}
    import vaayu.persist.ram_cache as rc

    def fake(**kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(rc, "auto_detect_cache_budget", fake)
    _use(monkeypatch, model_dir, 10 * GIB)
    assert seen["total_bytes"] == HOST_TOTAL
    assert seen["available_bytes"] == 10 * GIB
