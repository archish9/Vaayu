"""The exclusive-GPU guard must WAIT OUT a transient holder, not refuse forever.

SVC-REGRESSION Bug 1: `_check_gpu_is_exclusive` raised SystemExit the instant
another process held VRAM on our device, and `vaayu.service`'s `Restart=always`
+ `RestartSec=5` respawned straight back into the same refusal. Measured on this
box: 2547 failed starts over two days, max restart counter 1404, ~6.5 h of
endpoint-down -- every one of them a cold process paying ~3.5 s of torch import
to re-discover a condition that one process could simply have waited out.

The refusal itself is CORRECT and stays (a shared GPU is a permanent per-token
tax, up to 13x, invisible in the output). What changes is that a bounded wait
comes first, so a draining prior instance costs one pause instead of an outage.

No GPU needed: `gpu_audit.audit` and `time.sleep` are both monkeypatched.
"""
import time

import pytest

from vaayu.gpu_audit import GpuUser

# The real holder from the Nov 01 journal: a prior Vaayu that had not freed yet.
BLOCKER = GpuUser(pid=7679, name="./venv/bin/python", mib=10740)
# Under `exclusive_gpu_min_mib` (256): a bare CUDA context, reported not blocked.
TINY = GpuUser(pid=1234, name="python", mib=128)


def _config(**kw):
    from vaayu.server.app import ServerConfig
    return ServerConfig(model_path="/ckpt", **kw)


def test_a_holder_that_drains_is_waited_out_instead_of_refused(monkeypatch):
    """The whole bug: the holder goes away on its own, so the load must proceed."""
    import vaayu.gpu_audit as ga
    from vaayu.server.app import _check_gpu_is_exclusive

    audits = []

    def audit(**kw):
        audits.append(kw)
        return [BLOCKER] if len(audits) <= 3 else []

    monkeypatch.setattr(ga, "audit", audit)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    _check_gpu_is_exclusive(_config(exclusive_gpu_wait_s=120))   # must NOT raise

    assert len(audits) > 1, "it has to re-audit, not trust the first look"


def test_a_holder_that_never_drains_refuses_once_after_the_full_wait(monkeypatch):
    """Still a refusal -- but exactly one, and only after the wait expires."""
    import vaayu.gpu_audit as ga
    from vaayu.server.app import _check_gpu_is_exclusive

    slept = []
    monkeypatch.setattr(ga, "audit", lambda **kw: [BLOCKER])
    monkeypatch.setattr(time, "sleep", slept.append)

    with pytest.raises(SystemExit) as e:
        _check_gpu_is_exclusive(_config(exclusive_gpu_wait_s=120))

    assert "refusing to load" in str(e.value)
    assert sum(slept) == pytest.approx(120, abs=2.0), \
        f"waited {sum(slept)}s, not the configured 120s"


def test_opting_out_of_exclusivity_never_waits(monkeypatch, capsys):
    """`require_exclusive_gpu=False` is warn-and-load; a wait would be a regression."""
    import vaayu.gpu_audit as ga
    from vaayu.server.app import _check_gpu_is_exclusive

    slept = []
    monkeypatch.setattr(ga, "audit", lambda **kw: [BLOCKER])
    monkeypatch.setattr(time, "sleep", slept.append)

    _check_gpu_is_exclusive(_config(require_exclusive_gpu=False,
                                    exclusive_gpu_wait_s=120))

    assert slept == [], "a box deliberately shared has nothing to wait for"
    assert "WARNING" in capsys.readouterr().out


def test_a_holder_too_small_to_block_never_waits(monkeypatch, capsys):
    """Under `exclusive_gpu_min_mib` is reported, not blocked -- so not waited on."""
    import vaayu.gpu_audit as ga
    from vaayu.server.app import _check_gpu_is_exclusive

    slept = []
    monkeypatch.setattr(ga, "audit", lambda **kw: [TINY])
    monkeypatch.setattr(time, "sleep", slept.append)

    _check_gpu_is_exclusive(_config(exclusive_gpu_wait_s=120))

    assert slept == [], "nothing is blocking, so there is nothing to wait for"
    assert "1234" in capsys.readouterr().out, "small holders are still reported"


def test_zero_keeps_the_old_refuse_immediately_contract(monkeypatch):
    """The pre-fix behaviour stays reachable for anyone who wants it."""
    import vaayu.gpu_audit as ga
    from vaayu.server.app import _check_gpu_is_exclusive

    slept = []
    monkeypatch.setattr(ga, "audit", lambda **kw: [BLOCKER])
    monkeypatch.setattr(time, "sleep", slept.append)

    with pytest.raises(SystemExit):
        _check_gpu_is_exclusive(_config(exclusive_gpu_wait_s=0))

    assert slept == []


def test_the_wait_is_on_by_default_and_fits_under_timeoutstartsec():
    """A default of 0 would leave the boot loop in place for the real service."""
    from vaayu.server import config

    assert _config().exclusive_gpu_wait_s == 120
    assert config.EXCLUSIVE_GPU_WAIT_S == 120
    assert _config().exclusive_gpu_wait_s < 600, \
        "must expire before systemd's TimeoutStartSec=600 calls the start failed"


def test_the_installer_carves_exit_3_out_of_the_restart_policy():
    """Fix 1c: after the wait, exit 3 means `still occupied` -- respawning cannot
    help, and respawning every 5 s is what made Bug 1 invisible for 6.5 h.
    `Restart=always` stays, so crashes (the oomd case) are still covered."""
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "install-vaayu-service.sh"
    text = script.read_text()

    assert "RestartPreventExitStatus=3" in text
    assert "Restart=always" in text, "crash recovery must survive the carve-out"
