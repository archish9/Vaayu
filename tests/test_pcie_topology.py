"""Unit tests for the PCIe health diagnostic (`vaayu_base`).

Replaces a root-level file removed 2026-08-17. That one asserted a contract the
implementation had deliberately abandoned — it warned on link TOPOLOGY (gen x
width) and computed a THEORETICAL bandwidth from it. The current code measures
real pinned H2D instead, because idle GPUs downtrain to Gen1 and native-x8 cards
can never show x16, so topology alone produced false warnings. It also lived
outside `tests/`, so `pytest tests/` never collected it and the rot went
unnoticed for two months.

These tests are PURE: the decision and the label are extracted helpers, so
nothing here allocates 512 MB on a GPU or shells out to nvidia-smi. The old file
patched only `subprocess.run` and did a real burst copy on every test.
"""
import pytest

from vaayu.vaayu_base import (PCIE_MIN_HEALTHY_GBPS, pcie_health_message,
                              pcie_link_label)


# ── link label parsing ───────────────────────────────────────────────────────

def test_a_successful_query_is_labelled_gen_then_width():
    # nvidia-smi is asked for pcie.link.gen.current,pcie.link.width.current in
    # that order, so the FIRST field is the generation. The removed test fed
    # "16, 4" and expected "Gen4 x16" -- it was written against the older query,
    # which asked for width first.
    assert pcie_link_label(0, "4, 16\n") == "Gen4 x16"


def test_a_native_x8_card_is_labelled_without_complaint():
    # An RTX 5060 Ti is x8 by construction. Width is not a defect signal.
    assert pcie_link_label(0, "4, 8\n") == "Gen4 x8"


def test_an_idle_downtrained_link_is_labelled_as_read():
    # Power-save drops an idle GPU to Gen1; the label reports it, the decision
    # ignores it. This box reported "1, 8" while idle on 2026-08-16.
    assert pcie_link_label(0, "1, 8\n") == "Gen1 x8"


def test_a_failed_query_is_unknown_rather_than_an_error():
    assert pcie_link_label(1, "") == "unknown"


def test_unparseable_output_is_unknown_rather_than_raising():
    # A diagnostic must never become the failure it is diagnosing.
    assert pcie_link_label(0, "garbage without a comma") == "unknown"
    assert pcie_link_label(0, "") == "unknown"


def test_only_the_first_gpus_row_is_parsed():
    assert pcie_link_label(0, "4, 16\n3, 8\n") == "Gen4 x16"


# ── the health decision ──────────────────────────────────────────────────────

def test_healthy_bandwidth_reports_ok_with_the_measured_figure():
    msg = pcie_health_message(19.8, "Gen1 x8", model_size_gb=132.0)
    assert msg.startswith("✅")
    assert "19.8 GB/s" in msg
    assert "Gen1 x8" in msg
    assert "⚠️" not in msg


def test_a_downtrained_narrow_link_at_full_speed_is_NOT_a_warning():
    # The whole reason the topology contract was abandoned: this box measures
    # ~20 GB/s while nvidia-smi reports Gen1 x8. The old rule called that broken.
    msg = pcie_health_message(20.1, "Gen1 x8", model_size_gb=132.0)
    assert msg.startswith("✅")


def test_slow_bandwidth_warns_and_names_the_remedy():
    msg = pcie_health_message(3.9, "Gen3 x4", model_size_gb=132.0)
    assert msg.startswith("⚠️")
    assert "3.9 GB/s" in msg
    assert "Gen3 x4" in msg
    assert "132.0GB" in msg
    # A warning the reader cannot act on is the diagnostic failure this repo
    # keeps paying for -- name what to check.
    assert "slot" in msg and "BIOS" in msg


def test_the_warning_carries_the_model_size_it_was_asked_about():
    assert "37.0GB" in pcie_health_message(2.0, "Gen1 x4", model_size_gb=37.0)


@pytest.mark.parametrize("gbps,warns", [
    (0.0, True),
    (7.9, True),
    (PCIE_MIN_HEALTHY_GBPS - 0.01, True),
    (PCIE_MIN_HEALTHY_GBPS, False),      # the threshold itself is acceptable
    (8.1, False),
    (25.6, False),
])
def test_the_threshold_is_measured_bandwidth_not_topology(gbps, warns):
    msg = pcie_health_message(gbps, "Gen4 x16", model_size_gb=132.0)
    assert msg.startswith("⚠️") is warns, msg


def test_an_unknown_link_still_reports_the_measured_bandwidth():
    # nvidia-smi failing does not invalidate the measurement, so this is NOT
    # silence -- the removed test asserted silence here.
    msg = pcie_health_message(20.1, "unknown", model_size_gb=132.0)
    assert msg.startswith("✅")
    assert "unknown" in msg


def test_the_threshold_is_documented_as_a_hardware_constant():
    # CLAUDE.md §5 permits hardware literals only, and requires them commented.
    import inspect

    import vaayu.vaayu_base as vb
    src = inspect.getsource(vb)
    i = src.index("PCIE_MIN_HEALTHY_GBPS")
    assert "hardware" in src[max(0, i - 500):i + 200].lower()


# ── the sentinel: the check runs once per GPU ────────────────────────────────

def test_the_check_is_skipped_after_the_first_call_for_a_gpu(monkeypatch):
    import vaayu.vaayu_base as vb

    monkeypatch.setattr(vb, "_pcie_topology_checked", set())
    calls = []
    monkeypatch.setattr(vb.torch.cuda, "is_available", lambda: (calls.append(1), False)[1])

    vb._check_pcie_topology(0)
    vb._check_pcie_topology(0)
    assert len(calls) == 1, "the sentinel must stop the second probe for one GPU"

    vb._check_pcie_topology(1)
    assert len(calls) == 2, "a different GPU index is a different probe"
