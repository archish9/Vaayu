"""The engine must never reset the process-global CUDA peak counter.

`torch.cuda.max_memory_allocated()` is ONE high-water mark per process, and
`reset_peak_memory_stats()` is how a caller says "start measuring here". Vaayu
used to call it once per prefill chunk, inside `_chunked_prefill`'s legacy
per-chunk loop, so any caller doing the standard reset -> forward -> read saw
only the FINAL chunk of a chunked prefill.

That truncated every point in scripts/fit_vram_calibration.py's MEASURED table
for the two DENSE geometries (the MoE layer-major branch returns before the
loop and was never affected). Measured cost of the blind spot on Llama-3.2-1B
at S=32768: 2.409 GB recorded, 8.13 GB actual.

Asserted at SOURCE level on purpose: tests/ has no GPU, and the defect is
"the engine mutates shared global state", not any particular byte count. A
value-asserting test would need a GPU and would not have caught this.
"""
import inspect
from pathlib import Path

import vaayu.vaayu_base as vaayu_base

_FORBIDDEN = "reset_peak_memory_stats"


def test_chunked_prefill_does_not_reset_the_peak_counter():
    src = inspect.getsource(vaayu_base.VaayuBaseModel._chunked_prefill)
    assert _FORBIDDEN not in src, (
        "_chunked_prefill resets the process-global CUDA peak counter again; "
        "this truncates every external measurement of a chunked prefill to its "
        "last chunk. See plans/2026-08-15-vram-peak-measurement-design.md."
    )


def test_no_engine_module_resets_the_peak_counter():
    """Same invariant, package-wide: the reset must not reappear elsewhere."""
    root = Path(vaayu_base.__file__).parent
    offenders = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if _FORBIDDEN in p.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        f"these engine modules reset the global CUDA peak counter: {offenders}. "
        f"Only measurement CALLERS (scripts/, tests/) may reset it."
    )
