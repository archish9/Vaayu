"""Verify predict_peak_vram's FORM against measured VRAM. No constants are fitted.

HISTORY. This began as a calibration of DEFAULT_ACT_COEFF / DEFAULT_ALLOC_FACTOR
and REFUTED it: the FORM was wrong, not the constants. The form was then fixed
(2026-08-15) and this script became its verification. The evidence that drove it:

  * With ZERO fitted constants, `KV(at capacity) + last-chunk logits` already
    explains the measured transient to within +-0.13 GB on THREE of the four
    on-device geometries. The activation term (act_coeff * S) is not observable
    at all, and alloc_factor wants to be ~1.0.
  * The fourth, Qwen3-235B on the FUSED-EXL3 MoE forward, carries ~1.39 GB of
    residency that nothing in the model accounts for -- and its peak does not
    include the last-chunk logits at all (subtracting them makes the residual
    4x LESS flat: spread 0.536 GB vs 0.140 GB). That is todo.md item 6, now
    confirmed and localised to one forward.
  * Fitting the two constants jointly across all four geometries failed:
    13/26 points outside +-10%, worst 44.2%, and the per-model fits were
    mutually incompatible (alloc_factor 0.69-1.44, act_coeff -1.34 to +3.51 --
    NEGATIVE for two, the signature of a form error absorbed by a constant).

THE FORM NOW: peak = KV(at allocated capacity) + max(last-chunk logits,
resident experts). The activation term is gone, alloc_factor is 1.0, and the
residency term is non-zero only for the fused-EXL3 forward, derived as
num_experts * expert_bytes. This script re-checks all 26 points against it.

Why the earlier rounds could not see this: without a measured baseline, every
fit had a free per-model offset absorbing exactly this structure, which is what
let a wrong form still pass a shape-only check.

No torch, no CUDA, no model load. Run:
    PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python scripts/fit_vram_calibration.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from vaayu.vram_plan import (
    VramShape,
    kv_bytes_per_token,
    predict_peak_vram,
)

GB = 1e9  # decimal, matching the probe's own GB reporting

# The probe's operating point (scripts/probe_vram_curve.py runs a batch of 1,
# and vaayu_base.py's default prefill_chunk_size).
PROBE_N = 1
PROBE_CHUNK = 512
PROBE_ELEM_BYTES = 2

GATE = 0.10  # the +-10% bar every round has been held to

CHECKPOINTS = {
    # key: (directory under /home/a/code/ai-ml/models, is_exl3)
    # is_exl3 decides kv_capacity_buckets: _create_kv_cache resolves
    # kv_cache_mode="auto" to the doubling ring buffer FP8DynamicCache only on
    # an EXL3 loader, and to a plain transformers.DynamicCache otherwise.
    "qwen3-235b": ("Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw", True),
    "minimax-m2.7": ("NeuroSenko--MiniMax-M2.7-exl3", True),
    "llama-3.3-70b": ("archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw", True),
    "llama-3.2-1b": ("Llama-3.2-1B-Instruct", False),
}
MODELS_ROOT = "/home/a/code/ai-ml/models"

FORWARD = {
    "qwen3-235b": "MoE fused-EXL3",
    "minimax-m2.7": "MoE module-list",
    "llama-3.3-70b": "dense EXL3 streamed",
    "llama-3.2-1b": "dense non-EXL3",
}

# ---------------------------------------------------------------------------
# MEASURED arms: (S, alloc_before_bytes, alloc_peak_bytes).
#
# PROVENANCE: probed 2026-08-15 on this box, torch 2.11.0+cu130, by
# scripts/probe_vram_curve.py after it gained the per-point `alloc_before_bytes`
# column -- torch.cuda.memory_allocated() taken after empty_cache() +
# reset_peak_memory_stats(), immediately before the forward. Per-point rather
# than once at load because a MoE model's expert VRAM cache keeps growing across
# the S loop. One arm per subprocess, sequential, each under
# `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`.
#
# Header lines, driver-level vs allocator-level resident after load:
#   qwen3-235b     total=16.61GB free_after_load=14.22GB driver=2.39GB alloc=1.81GB
#   llama-3.3-70b  total=16.61GB free_after_load=13.17GB driver=3.45GB alloc=2.89GB
#   minimax-m2.7   total=16.61GB free_after_load=14.21GB driver=2.40GB alloc=1.79GB
#   llama-3.2-1b   total=16.61GB free_after_load=14.99GB driver=1.62GB alloc=1.05GB
# The driver-vs-allocator gap (0.51-0.61 GB: CUDA context plus
# reserved-but-unallocated blocks) is why the older `total - free_after_load`
# figure could never be compared against an allocator-level model.
#
# Every arm ran to completion with status=ok; no OOM at any S, including 16384.
# ---------------------------------------------------------------------------
# KNOWN STALE (2026-08-15): the dense rows below (llama-3.3-70b, llama-3.2-1b) are
# last-chunk peaks, not whole-prefill peaks -- vaayu_base.py's chunked-prefill loop
# used to reset the CUDA peak counter mid-forward (fixed on the vram-peak-measurement
# branch). The gate below still reads PASS because it is comparing predict_peak_vram
# against exactly this truncated table, which is self-consistent but not correct. The
# honest whole-prefill data already exists (`plans/2026-08-15-vram-peak-RESULTS.md`
# S3-4) and re-pointing this table at it is Phase 2, tracked as `todo.md` item 10 (and
# item 6, blocked on it) -- not done here; see `plans/2026-08-15-vram-peak-RESULTS.md`
# S6 for the full "why this PASS doesn't mean what it looks like" writeup.
# ---------------------------------------------------------------------------
MEASURED = {
    "qwen3-235b": [
        (5000, 1810343424, 4701109248), (6000, 1728843776, 4690192384),
        (7000, 1728851968, 4733883392), (7700, 1728857600, 4759347200),
        (8200, 1728861696, 6222054912), (12288, 1728893952, 6248692736),
        (16384, 1728926720, 6304004096),
    ],
    "llama-3.3-70b": [
        (5000, 2889698304, 5792254976), (7000, 2906810880, 5767645696),
        (8200, 2906820608, 8279633920), (12288, 2906852864, 8538230272),
        (16384, 2906885632, 8538263040),
    ],
    "minimax-m2.7": [
        (5000, 1788149760, 4286465024), (6000, 1702418944, 4263660032),
        (7000, 1702427136, 4240855552), (7700, 1702432768, 3895998464),
        (8200, 1702436864, 5959702528), (12288, 1702469120, 6556458496),
    ],
    "llama-3.2-1b": [
        (5000, 1050718720, 1437159936), (6000, 1072222208, 1457623040),
        (7000, 1072230400, 1478086656), (7700, 1072236032, 1334810112),
        (8200, 1072240128, 1345041920), (16384, 1072305152, 1871844352),
        (32768, 1072436224, 2408846336), (49152, 1072567296, 2945848320),
    ],
}

# The 2026-08-14 arm (F4 plan Task 5), alloc_peak in GB, NO baseline recorded.
# Kept only as a reproduction check -- its /tmp/vramcurve_*.log files are gone
# and docs/superpowers/plans/2026-08-09-vram-plan-RESULTS.md ("Raw data, all six
# arms"), which is gitignored, was the last copy.
LEGACY_PEAKS_GB = {
    "minimax-m2.7": {5000: 4.2854, 6000: 4.2626, 7000: 4.2398, 7700: 3.8949,
                     8200: 5.9586},
    "qwen3-235b": {5000: 4.7145, 6000: 4.6859, 7000: 4.6705, 7700: 4.7235,
                   8200: 6.1941},
    "llama-3.3-70b": {7000: 5.7634, 8200: 8.2754},
    "llama-3.2-1b": {5000: 1.4263, 6000: 1.4468, 7000: 1.4673, 7700: 1.3240,
                     8200: 1.3342},
}



# Bytes of expert weights held GPU-resident across prefill chunks. NON-ZERO ONLY
# for the fused-EXL3 forward -- chunk-major retention (_chunk_major_gpu) exists
# nowhere else. Mirrors vaayu_base's derivation: num_experts * expert_bytes,
# both read from _moe_geometry at load. Confirmed 2026-08-15 by instrumenting
# _chunk_major_gpu_bytes: a flat 1.514 GB on Qwen (128 experts x 11.83 MB),
# constant across S=5000..12288 and reset per layer.
RESIDENT_EXPERT_BYTES = {
    "qwen3-235b": int(128 * 11.83e6),   # measured expert_bytes at load
    "minimax-m2.7": 0,                  # module-list: no chunk-major retention
    "llama-3.3-70b": 0,                 # dense
    "llama-3.2-1b": 0,                  # dense
}

def shape_for(key: str) -> VramShape:
    """Build the probe's VramShape from the checkpoint's own config.json.

    Every geometry number is READ, never typed (CLAUDE.md S5).
    resident_expert_bytes mirrors what vaayu_base computes at load: non-zero
    only for the fused-EXL3 forward, as num_experts * expert_bytes.
    """
    directory, is_exl3 = CHECKPOINTS[key]
    with open(f"{MODELS_ROOT}/{directory}/config.json") as fh:
        raw = json.load(fh)
    cfg = raw.get("text_config", raw)

    def field(name):
        value = cfg.get(name, raw.get(name))
        if value is None:
            raise SystemExit(f"{key}: config.json has no {name!r}")
        return value

    return VramShape(
        kv_per_token=kv_bytes_per_token(
            num_hidden_layers=field("num_hidden_layers"),
            num_key_value_heads=field("num_key_value_heads"),
            head_dim=field("head_dim"),
            kv_cache_mode="auto",
        ),
        N=PROBE_N,
        chunk=PROBE_CHUNK,
        hidden=field("hidden_size"),
        n_heads=field("num_attention_heads"),
        vocab=field("vocab_size"),
        elem_bytes=PROBE_ELEM_BYTES,
        materializes_scores=False,   # MATERIALIZES_SCORES, measured on this box
        alloc_factor=1.0,
        kv_capacity_buckets=is_exl3,
        resident_expert_bytes=RESIDENT_EXPERT_BYTES[key],
    )


def main() -> int:
    print("=" * 78)
    print("FORM CHECK -- peak = KV(capacity) + max(last-chunk logits, resident experts)")
    print("no fitted constants; alloc_factor = 1.0")
    print("=" * 78)
    print(f"  {'model':<15}{'S':>7}{'transient':>11}{'predicted':>11}{'rel err':>9}")
    worst, outside, total = 0.0, 0, 0
    per_model = {}
    for key, points in MEASURED.items():
        shape = shape_for(key)
        errs = []
        for S, before, peak in sorted(points):
            transient = peak - before
            predicted = predict_peak_vram(shape, S)
            rel = (predicted - transient) / transient
            errs.append(rel)
            worst = max(worst, abs(rel))
            outside += abs(rel) > GATE
            total += 1
            flag = "" if abs(rel) <= GATE else "   <-- FAIL"
            print(f"  {key:<15}{S:>7}{transient/GB:>10.4f}G{predicted/GB:>10.4f}G"
                  f"{100*rel:>8.1f}%{flag}")
        per_model[key] = errs
        print(f"  {'':<15}-> {FORWARD[key]}, residency "
              f"{shape.resident_expert_bytes/GB:.3f}G, worst "
              f"{100*max(abs(e) for e in errs):.1f}%\n")

    print("=" * 78)
    print(f"GATE: {outside}/{total} points outside +-{GATE:.0%}, worst "
          f"{100*worst:.1f}%  ->  {'PASS' if not outside else 'FAIL'}")
    print("CAVEAT: MEASURED's dense rows are known last-chunk (not whole-prefill) peaks --")
    print("  see the comment above MEASURED, plans/2026-08-15-vram-peak-RESULTS.md S6, todo.md item 10.")
    print("=" * 78)
    if not outside:
        print("  Every geometry inside the bar with ZERO fitted constants. The two")
        print("  former constants are gone: act_coeff was unobservable, alloc_factor")
        print("  is 1.0. Do not reintroduce either without data that needs them.")
    return 0 if not outside else 1


if __name__ == "__main__":
    sys.exit(main())
