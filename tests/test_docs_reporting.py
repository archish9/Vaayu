"""The reporting instructions must name the real directory and the real
command.

These are SUBSTRING checks over README.md and documentation/server.md, not
assertions against live code: they catch a section being deleted or renamed,
and nothing else. `tests/test_docs_ports.py` is the stronger kind -- it imports
the config value and compares it to the prose -- and nothing here does that.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_readme_has_a_reporting_section():
    text = (REPO / "README.md").read_text()
    assert "Reporting a problem" in text
    assert "logs/run-" in text
    assert "python -m vaayu.server.diagnose" in text
    assert "redact" in text.lower()


def test_server_doc_documents_the_new_artifacts():
    text = (REPO / "documentation" / "server.md").read_text()
    assert "requests.jsonl" in text
    assert "gb_tok" in text
    assert "python -m vaayu.server.diagnose" in text


def test_server_doc_says_gb_tok_is_nvme_bytes():
    """The whole point of the field. A reader who takes it for materialised
    bytes will re-derive disk traffic by hand, which is the habit this
    replaces."""
    text = (REPO / "documentation" / "server.md").read_text()
    i = text.index("gb_tok")
    window = text[i - 400:i + 400]
    assert "NVMe" in window or "disk" in window


def test_readme_says_requests_jsonl_is_not_filtered_either():
    """It already said this about `server.log`. `requests.jsonl` writes raw,
    and its `error` block embeds a full traceback whose frames name absolute
    paths -- the on-device gate used to assert the opposite."""
    text = (REPO / "README.md").read_text()
    window = text[text.index("What is redacted"):][:1000]
    assert "requests.jsonl" in window
    assert "traceback" in window
