"""Unit tests for _check_pcie_topology in vaayu/vaayu_base.py."""
from unittest.mock import patch, MagicMock


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
    # nvidia-smi returns one line per GPU on multi-GPU machines
    with patch("subprocess.run", return_value=_make_result("16, 4\n16, 4\n")):
        fn()
    out = capsys.readouterr().out
    assert "✅" in out
    assert "Gen4 x16" in out
