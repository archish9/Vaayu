"""The token summary's new bytes= / gb_tok= fields.

The line is parsed by eye during triage and by scripts in plans/, so it is
asserted as TEXT: one line, the existing fields untouched, two new ones.
"""
import re


def format_summary_fields(read_bytes_delta: int) -> str:
    """The exact substring vaayu_base appends. Kept here so the formatting is
    testable without a model: the production line is one 500-character f-string
    inside a forward pass."""
    from vaayu.vaayu_base import format_byte_fields
    return format_byte_fields(read_bytes_delta)


def test_byte_fields_render_both_values():
    assert format_summary_fields(0) == "  bytes=0  gb_tok=0.00"


def test_byte_fields_render_gigabytes_decimal():
    # Decimal GB, matching every other GB figure the engine prints.
    assert format_summary_fields(4_410_000_000) == "  bytes=4410000000  gb_tok=4.41"


def test_byte_fields_are_one_line():
    assert "\n" not in format_summary_fields(123456789)


def test_byte_fields_parse_out_of_a_summary_line():
    line = ("[VAAYU-IO] token summary: total=1.1s  io_wait=0.0s"
            + format_summary_fields(11_830_272))
    m = re.search(r"gb_tok=([0-9.]+)", line)
    assert m and abs(float(m.group(1)) - 0.01) < 1e-9

