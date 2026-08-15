"""Calibrate -- or refute -- predict_peak_vram's two constants against measured VRAM.

VERDICT (2026-08-15): REFUTED. Calibrating DEFAULT_ACT_COEFF / DEFAULT_ALLOC_FACTOR
cannot close the +-10% gate, because the model's FUNCTIONAL FORM is wrong, not its
constants. Run this script for the evidence; the summary is:

  * With ZERO fitted constants, `KV(at capacity) + last-chunk logits` already
    explains the measured transient to within +-0.13 GB on THREE of the four
    on-device geometries. The activation term (act_coeff * S) is not observable
    at all, and alloc_factor wants to be ~1.0.
  * The fourth, Qwen3-235B on the FUSED-EXL3 MoE forward, carries ~1.39 GB of
    residency that nothing in the model accounts for -- and its peak does not
    include the last-chunk logits at all (subtracting them makes the residual
    4x LESS flat: spread 0.536 GB vs 0.140 GB). That is todo.md item 6, now
    confirmed and localised to one forward.
  * Fitting the two constants jointly across all four geometries therefore
    fails: 13/26 points outside +-10%, worst 44.2%, and the per-model fits are
    mutually incompatible (alloc_factor 0.69-1.44, act_coeff -1.34 to +3.51 --
    NEGATIVE for two of them, which is physically meaningless).

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
    _peak_intercept_bytes,
    _kv_term_bytes,
    kv_bytes_per_token,
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


def shape_for(key: str) -> VramShape:
    """Build the probe's VramShape from the checkpoint's own config.json.

    Every geometry number is READ, never typed (CLAUDE.md S5).
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
        act_coeff=1.0,      # unused: the fit works on the basis columns directly
        materializes_scores=False,   # MATERIALIZES_SCORES, measured on this box
        alloc_factor=1.0,   # unused, same reason
        kv_capacity_buckets=is_exl3,
    )


def basis(shape: VramShape, S: int) -> tuple[float, float]:
    """The two regressor columns at context S.

    col_x multiplies x = alloc_factor*act_coeff (the activation basis with
    act_coeff factored out); col_y multiplies y = alloc_factor (the terms whose
    size is already fully determined -- last-chunk logits and KV at capacity).
    """
    col_x = shape.N * shape.hidden * shape.elem_bytes * float(S)
    col_y = _peak_intercept_bytes(shape, S) + _kv_term_bytes(shape, S)
    return col_x, col_y


def fit(data) -> tuple[float, float]:
    """Least squares for x, y against measured transients (peak - alloc_before).

    predict(S) = alloc_factor * (act_coeff*A*S + logits + KV)
               = x*col_x + y*col_y,   x = alloc_factor*act_coeff, y = alloc_factor
    so the product of the two unknowns still leaves a linear 2-column problem.
    """
    rows, targets = [], []
    for key, points in data.items():
        shape = shape_for(key)
        for S, before, peak in points:
            col_x, col_y = basis(shape, S)
            rows.append([col_x, col_y])
            targets.append(float(peak - before))
    A = np.asarray(rows, float)
    b = np.asarray(targets, float)
    (x, y), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(x), float(y)


def predict(shape: VramShape, S: int, x: float, y: float) -> float:
    col_x, col_y = basis(shape, S)
    return x * col_x + y * col_y


def main() -> int:
    shapes = {key: shape_for(key) for key in CHECKPOINTS}

    # =====================================================================
    print("=" * 78)
    print("REPRODUCTION CHECK -- 2026-08-15 arm vs the 2026-08-14 arm")
    print("=" * 78)
    worst_repro = 0.0
    for key, legacy in LEGACY_PEAKS_GB.items():
        deltas = [abs(peak / GB - legacy[S])
                  for S, _b, peak in MEASURED[key] if S in legacy]
        worst_repro = max(worst_repro, max(deltas))
        print(f"  {key:<15} {len(deltas)} overlapping S, worst |delta| "
              f"{max(deltas):.4f} GB")
    print(f"\n  worst across all four: {worst_repro:.4f} GB -- the box reproduces,")
    print("  so the new baselines can be trusted against the old peaks.")

    # =====================================================================
    print("\n" + "=" * 78)
    print("DECOMPOSITION -- measured transient minus the two DERIVED terms.")
    print("No fitted constants at all. This is what the model gets right.")
    print("=" * 78)
    print(f"  {'model':<15}{'S':>7}{'transient':>11}{'KV':>10}{'logits':>10}"
          f"{'RESIDUAL':>11}")
    residuals = {}
    for key, points in MEASURED.items():
        shape = shapes[key]
        per_point = []
        for S, before, peak in points:
            kv = _kv_term_bytes(shape, S)
            logits = _peak_intercept_bytes(shape, S)
            transient = peak - before
            residual = transient - kv - logits
            per_point.append(residual)
            print(f"  {key:<15}{S:>7}{transient/GB:>10.4f}G{kv/GB:>9.4f}G"
                  f"{logits/GB:>9.4f}G{residual/GB:>10.4f}G")
        residuals[key] = per_point
        spread = max(per_point) - min(per_point)
        print(f"  {'':<15}-> mean {np.mean(per_point)/GB:+.3f}G  "
              f"spread {spread/GB:.3f}G   [{FORWARD[key]}]\n")

    # =====================================================================
    print("=" * 78)
    print("THE SPLIT -- three forwards need NO activation term; one needs a term")
    print("the model does not have")
    print("=" * 78)
    for key in MEASURED:
        mean = float(np.mean(residuals[key])) / GB
        spread = (max(residuals[key]) - min(residuals[key])) / GB
        verdict = ("EXPLAINED (KV + logits, no constants)" if abs(mean) < 0.15
                   else "UNEXPLAINED ~%.2fG of residency" % mean)
        print(f"  {key:<15}{FORWARD[key]:<22} mean {mean:+.3f}G "
              f"spread {spread:.3f}G  {verdict}")

    # Does the odd one out even include the logits term?
    key = "qwen3-235b"
    shape = shapes[key]
    with_logits = [peak - before - _kv_term_bytes(shape, S)
                   - _peak_intercept_bytes(shape, S)
                   for S, before, peak in MEASURED[key]]
    without_logits = [peak - before - _kv_term_bytes(shape, S)
                      for S, before, peak in MEASURED[key]]
    print(f"\n  Does {key}'s peak contain the last-chunk logits at all?")
    print(f"    subtracting them : mean {np.mean(with_logits)/GB:.3f}G  "
          f"spread {(max(with_logits)-min(with_logits))/GB:.3f}G")
    print(f"    leaving them in  : mean {np.mean(without_logits)/GB:.3f}G  "
          f"spread {(max(without_logits)-min(without_logits))/GB:.3f}G  <-- 4x flatter")
    print("    => on the fused-EXL3 forward the logits never SET the peak: a")
    print("       larger, roughly constant residency dominates them. That is")
    print("       todo.md item 6, confirmed and localised to one forward.")

    # =====================================================================
    print("\n" + "=" * 78)
    print(f"THE GATE -- fit both constants jointly across all four geometries")
    print("=" * 78)
    x, y = fit(MEASURED)
    print(f"  alloc_factor {y:.4f} (placeholder 1.18), "
          f"act_coeff {x/y:.4f} (placeholder 4.0)")
    outside, worst = 0, 0.0
    total = 0
    for key, points in MEASURED.items():
        for S, before, peak in points:
            transient = peak - before
            rel = (predict(shapes[key], S, x, y) - transient) / transient
            worst = max(worst, abs(rel))
            outside += abs(rel) > GATE
            total += 1
    print(f"  {outside}/{total} points outside +-{GATE:.0%}, worst {100*worst:.1f}%"
          f"   -> {'PASS' if not outside else 'FAIL'}")

    print("\n  per-model fits -- can ONE global pair serve all four?")
    print(f"  {'model':<15}{'alloc_factor':>14}{'act_coeff':>12}{'worst':>9}")
    for key, points in MEASURED.items():
        xi, yi = fit({key: points})
        errs = [abs(predict(shapes[key], S, xi, yi) - (peak - before))
                / (peak - before) for S, before, peak in points]
        print(f"  {key:<15}{yi:>14.3f}{xi/yi:>12.3f}{100*max(errs):>8.1f}%")
    print("  Two of the four want a NEGATIVE act_coeff -- physically meaningless,")
    print("  and the sign of a form error being absorbed by a free constant.")

    # =====================================================================
    print("\n" + "=" * 78)
    print("VERDICT: calibration REFUTED as the fix. Do not tune these constants.")
    print("=" * 78)
    print("  The +-10% gate cannot be closed by act_coeff/alloc_factor, because:")
    print("   1. act_coeff*S is not observable. llama-3.3-70b's peak is IDENTICAL")
    print("      at S=12288 and S=16384 (differs by 33 KB) where the term predicts")
    print("      +0.13 GB, and llama-3.2-1b's entire growth out to S=49152 is")
    print("      accounted for by KV alone.")
    print("   2. alloc_factor wants ~1.0: KV + logits alone lands within +-0.13 GB")
    print("      on three of four forwards with NO fitted constants.")
    print("   3. The real gap is one forward's unmodelled residency (~1.39 GB on")
    print("      fused-EXL3), which a global multiplier cannot express.")
    print("  Every earlier round had a free per-model offset absorbing exactly")
    print("  this, which is how a wrong FORM kept passing a shape-only check.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
