"""Unit tests for _check_pcie_topology in vaayu/vaayu_base.py."""
import pytest
from unittest.mock import patch, MagicMock
import vaayu.vaayu_base as _vb


@pytest.fixture(autouse=True)
def reset_pcie_sentinel():
    _vb._pcie_topology_checked = set()
    yield
    _vb._pcie_topology_checked = set()


def _import_fn():
    from vaayu.vaayu_base import _check_pcie_topology
    return _check_pcie_topology


def _make_result(stdout, returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


def test_optimal_x16_gen4_prints_ok(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("16, 4\n")):
        fn()
    out = capsys.readouterr().out
    assert "✅" in out
    assert "Gen4 x16" in out
    assert "25.6 GB/s" in out


def test_suboptimal_x8_gen4_prints_warning(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("8, 4\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "Gen4 x8" in out
    assert "12.8 GB/s" in out
    assert "per token" in out


def test_suboptimal_x4_gen3_prints_warning(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("4, 3\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "Gen3 x4" in out
    assert "3.9 GB/s" in out


def test_nvidia_smi_nonzero_returncode_silent(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("", returncode=1)):
        fn()
    out = capsys.readouterr().out
    assert out == ""


def test_subprocess_exception_silent(capsys):
    fn = _import_fn()
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        fn()
    out = capsys.readouterr().out
    assert out == ""


def test_oserror_silent(capsys):
    fn = _import_fn()
    with patch("subprocess.run", side_effect=OSError("permission denied")):
        fn()
    out = capsys.readouterr().out
    assert out == ""


def test_unknown_pcie_combo_prints_warning_no_crash(capsys):
    fn = _import_fn()
    # Gen6 doesn't exist in the table yet
    with patch("subprocess.run", return_value=_make_result("8, 6\n")):
        fn()
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "unknown bandwidth" in out


def test_multi_gpu_parses_first_gpu_only(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("16, 4\n")) as mock_run:
        fn(gpu_index=1)
    out = capsys.readouterr().out
    assert "✅" in out
    assert "Gen4 x16" in out
    # Verify -i=1 was passed to nvidia-smi
    call_args = mock_run.call_args[0][0]
    assert "-i=1" in call_args


def test_sentinel_prevents_duplicate_nvidia_smi_calls(capsys):
    fn = _import_fn()
    with patch("subprocess.run", return_value=_make_result("16, 4\n")) as mock_run:
        fn(gpu_index=0)
        fn(gpu_index=0)
    assert mock_run.call_count == 1
