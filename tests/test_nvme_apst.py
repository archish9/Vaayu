"""APST is read from the controller's LIVE knob, not the boot-time module parameter.

nvme_core.default_ps_max_latency_us is read only when a controller initialises;
writing it after boot changes nothing. scripts/vaayu_optimize_os.sh used to write
it, and the boot check used to read it -- so after running the "fix" the check
said "disabled" while the drive still had APST on (todo.md row 25).
"""
from vaayu.auto_model import nvme_apst_tolerance


def _fake_sys(tmp_path, tolerance=None, module_param="0"):
    """A /sys tree shaped like this box's: block/nvme0n1/device -> the controller."""
    ctrl = tmp_path / "devices" / "pci0000:00" / "nvme" / "nvme0"
    (ctrl / "power").mkdir(parents=True)
    if tolerance is not None:
        (ctrl / "power" / "pm_qos_latency_tolerance_us").write_text(f"{tolerance}\n")
    (tmp_path / "block" / "nvme0n1").mkdir(parents=True)
    (tmp_path / "block" / "nvme0n1" / "device").symlink_to(ctrl)
    params = tmp_path / "module" / "nvme_core" / "parameters"
    params.mkdir(parents=True)
    (params / "default_ps_max_latency_us").write_text(f"{module_param}\n")
    return str(tmp_path)


def test_the_controller_wins_over_a_module_param_that_says_disabled(tmp_path):
    # The exact shape the old check got wrong: module param rewritten to 0
    # after boot, controller still at the kernel default.
    root = _fake_sys(tmp_path, tolerance=100000, module_param="0")
    assert nvme_apst_tolerance("nvme0n1", sys_root=root) == "100000"


def test_zero_on_the_controller_is_apst_off(tmp_path):
    root = _fake_sys(tmp_path, tolerance=0, module_param="100000")
    assert nvme_apst_tolerance("nvme0n1", sys_root=root) == "0"


def test_non_numeric_tolerances_come_back_raw(tmp_path):
    # pm_qos shows "auto" (no user constraint) or "any"; both leave APST allowed.
    root = _fake_sys(tmp_path, tolerance="auto")
    assert nvme_apst_tolerance("nvme0n1", sys_root=root) == "auto"


def test_a_controller_without_apst_control_is_none(tmp_path):
    # The kernel exposes the file only for controllers that support APST.
    root = _fake_sys(tmp_path, tolerance=None)
    assert nvme_apst_tolerance("nvme0n1", sys_root=root) is None
