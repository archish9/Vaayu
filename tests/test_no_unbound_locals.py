"""`vaayu/` must be free of used-before-assignment names (todo.md row 2).

Two bugs of exactly this shape shipped GREEN: `detect_moe_blocks` and then
`EXL3LoaderStrategy` were referenced in `vaayu_base.__init__` ABOVE their
in-function imports, making them unbound locals. Both MoE models failed to load
with `UnboundLocalError` while the whole suite passed -- nothing unit-tests the
load path, and a real load costs minutes and a checkpoint.

This is a lint gate riding the suite, not a test of those bugs. `ruff`/`pyflakes`
do NOT catch the class (F821 is *undefined* name, not *used-before-assign* in a
branch; probed 2026-09-09). `pylint E0606` does, in ~3 s.

E0601 possibly-used-before-assignment's certain sibling (used-before-assignment)
E0606 possibly-used-before-assignment -- the shape that shipped twice
E0602 undefined-variable
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RULES = "E0601,E0606,E0602"


def _pylint_missing():
    try:
        import pylint  # noqa: F401
    except ImportError:
        return True
    return False


@pytest.mark.skipif(
    _pylint_missing(),
    reason="pylint is not installed -- THIS GATE IS NOT RUNNING. "
           "`./venv/bin/pip install pylint` (it is in requirements.txt).",
)
def test_vaayu_has_no_used_before_assignment():
    proc = subprocess.run(
        [sys.executable, "-m", "pylint", "--disable=all", f"--enable={RULES}",
         "--score=n", "vaayu"],
        cwd=ROOT, capture_output=True, text=True,
    )
    findings = [ln for ln in proc.stdout.splitlines()
                if ": E06" in ln or ": E0601" in ln]
    assert not findings, (
        "pylint found used-before-assign names in vaayu/ -- this is the class "
        "that broke both MoE loads while the suite stayed green:\n"
        + "\n".join(findings)
        + "\n\nIf a finding is a FALSE POSITIVE (a name bound under the same "
          "guard pylint cannot correlate), suppress it inline at the site with "
          "`# pylint: disable=possibly-used-before-assignment` and say why. "
          "Do NOT hoist the import to silence it."
    )
