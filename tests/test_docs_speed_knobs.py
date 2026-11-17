"""documentation/server.md §3 "Speed" lists every speed knob by constant name.

Found 2026-09-14: row 33's DECODE_TOP_K and row 27's KEEP_AWAKE /
PIN_ENGINE_THREADS shipped with rows in vaayu/server/README.md but none there,
so a user reading the product docs could not find them. This pins them.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("knob", [
    "DECODE_TOP_K", "DECODE_TOP_K_BUDGET", "KEEP_AWAKE", "PIN_ENGINE_THREADS"])
def test_speed_knob_has_a_row_in_server_md(knob):
    text = (ROOT / "documentation" / "server.md").read_text()
    assert f"\n| `{knob}` |" in text, f"no server.md §3 row for {knob}"


def test_tuning_guide_documents_decode_top_k_auto():
    text = (ROOT / "documentation" / "TUNING.md").read_text()
    assert "## 5c. `decode_top_k`" in text
    assert '"auto"' in text


def test_readme_keepawake_row_is_not_stale():
    # Row 27's server A/B closed 2026-09-12; the README said it was pending (K1).
    text = (ROOT / "vaayu" / "server" / "README.md").read_text()
    assert "server-level A/B is pending" not in text
    assert "+23.1 % decode tok/s on Qwen3-235B" in text


@pytest.mark.parametrize("knob", ["NVME_MAX_SQES_PER_ENTER", "NVME_MAX_CHUNKS_PER_WAVE"])
@pytest.mark.parametrize("doc", ["documentation/server.md", "vaayu/server/README.md"])
def test_the_nvme_submission_caps_have_a_row_in_both_docs(knob, doc):
    """SQE-PLUG. A workaround for a kernel bug that can hard-hang the box is the
    LAST knob an operator should have to find by reading config.py."""
    text = (ROOT / doc).read_text()
    assert f"\n| `{knob}` |" in text, f"no {doc} row for {knob}"


@pytest.mark.parametrize("doc", ["documentation/server.md", "vaayu/server/README.md"])
def test_the_nvme_cap_rows_name_the_real_fix(doc):
    """It is a WORKAROUND. A row that does not say the kernel upgrade is the
    actual fix invites someone to treat the hang as solved."""
    text = (ROOT / doc).read_text()
    assert "CVE-2026-64017" in text, doc
    assert "The real fix is a patched kernel" in text, doc
