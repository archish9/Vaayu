"""A bare `pytest` at the repo root must collect `tests/` only.

The repo root holds on-device scripts named `test_*.py` (`test_generate.py`,
`test_multiturn.py`) that load a real checkpoint at import. Without
`testpaths`, collection imports them: on 2026-09-15 that loaded Qwen3-235B
inside the unit suite and the kernel OOM-killed the run (todo.md).
"""
import configparser
from pathlib import Path


def test_bare_pytest_collects_only_the_tests_directory():
    ini = configparser.ConfigParser()
    assert ini.read(Path(__file__).resolve().parents[1] / "pytest.ini"), "pytest.ini missing"
    assert ini["pytest"]["testpaths"].split() == ["tests"]
