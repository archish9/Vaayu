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
resident experts) + the attention-score buffer. The activation term is gone,
alloc_factor is 1.0, and the residency term is non-zero only for the fused-EXL3
forward, derived as num_experts * expert_bytes.

THE KERNEL DIMENSION (todo row 6, 2026-08-15). The score term is zero under
flash_attention_2 and large under sdpa, and every calibration arm before this
one ran flash -- while batched_generate hard-requires sdpa. MEASURED is
therefore keyed by kernel first, and the score coefficient is read per kernel
from vram_plan.SCORE_ELEM_BYTES. One coefficient covers all four geometries;
it is the ONLY fitted number left in the form.

Why the earlier rounds could not see this: without a measured baseline, every
fit had a free per-model offset absorbing exactly this structure, which is what
let a wrong form still pass a shape-only check.

No torch, no CUDA, no model load. Run:
    PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python scripts/fit_vram_calibration.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from vaayu.vram_plan import (
    SCORE_ELEM_BYTES,
    VramShape,
    kv_bytes_per_token,
    predict_peak_vram,
    score_elem_bytes,
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
# MEASURED arms, keyed by ATTENTION KERNEL FIRST: {impl: {model: [(S,
# alloc_before_bytes, alloc_peak_bytes), ...]}}.
#
# Why kernel-first (todo row 6, 2026-08-15). This used to be one flat table,
# and every point in it was flash_attention_2 -- the kernel Vaayu's candidate
# order picks when flash_attn imports, which it does on this box. But
# `batched_generate` HARD-REQUIRES sdpa (vaayu_base.py's assert), and the two
# kernels' transients diverge by a factor that GROWS with S: 1.32x at S=4096
# rising to 4.66x at S=65536 on Llama-3.2-1B, where sdpa OOMs at an S flash
# serves in 4.01 GB (todo row 10,
# plans/2026-08-15-row10-attention-kernel-RESULTS.md). One table calibrated
# against either kernel is therefore wrong for the other by enough to flip
# OOM/no-OOM. Both are kept: sdpa is what the batched path must be budgeted
# for, flash is what ordinary single-stream generate() actually runs.
#
# PROVENANCE -- flash_attention_2: probed 2026-08-15, torch 2.11.0+cu130,
# transcribed from plans/2026-08-15-vram-peak-RESULTS.md S3 (its own raw
# `(S, alloc_before, alloc_peak, status)` tables, every point status=ok). These
# are WHOLE-PREFILL peaks: they replace the older rows of this table, whose
# dense entries were last-chunk-only because the chunked-prefill loop used to
# reset the CUDA peak counter mid-forward. That staleness caveat is now GONE --
# it was the defect todo row 1 existed to close.
#
# PROVENANCE -- sdpa: probed 2026-08-15 by scripts/probe_vram_curve.py with
# PROBE_ATTN=sdpa (asserted against model._active_attn_impl, so a silent
# fallback cannot corrupt a point), one arm per subprocess under
# `systemd-run --user --scope -p MemoryMax=<22G, 28G for the 70B>
# -p MemorySwapMax=0`, each checkpoint's first touch discarded.
#
# Both kernels' S lists were passed EXPLICITLY rather than left to
# probe_vram_curve.sweep_points: that helper derives its ceiling from
# model.max_seq_len, which predict_peak_vram itself decides, so a derived grid
# would move under the very form this table exists to check.
#
# Run-to-run reproducibility: row 10's independent flash arms re-measured
# Qwen3-235B at S=2048 and S=8192 and landed 0.9% and 0.4% from the S3 values
# below; 70B and MiniMax reproduced byte-identically. ~1% is the noise floor
# here, well inside the 10% gate.
#
# Header lines, driver-level vs allocator-level resident after load:
#   qwen3-235b     total=16.61GB free_after_load=14.22GB driver=2.39GB alloc=1.81GB
#   llama-3.3-70b  total=16.61GB free_after_load=13.17GB driver=3.45GB alloc=2.89GB
#   minimax-m2.7   total=16.61GB free_after_load=14.21GB driver=2.40GB alloc=1.79GB
#   llama-3.2-1b   total=16.61GB free_after_load=14.99GB driver=1.62GB alloc=1.05GB
# The driver-vs-allocator gap (0.51-0.61 GB: CUDA context plus
# reserved-but-unallocated blocks) is why the older `total - free_after_load`
# figure could never be compared against an allocator-level model.
# ---------------------------------------------------------------------------
MEASURED = {
    "flash_attention_2": {
        "qwen3-235b": [
            (512, 1810307072, 2453824512), (513, 1724540416, 3361085952),
            (1024, 1724544000, 3401497088), (2048, 1724552192, 3603365376),
            (2049, 1724552704, 3893114880), (4096, 1724568576, 3937222144),
            (4097, 1724569088, 4729782272), (8192, 1724601344, 4819416576),
        ],
        "llama-3.3-70b": [
            (512, 2889661952, 3638956544), (513, 2902499328, 3948863488),
            (1024, 2902502912, 4069401088), (2048, 2902511104, 4501422592),
            (2049, 2902511616, 4958654464), (4096, 2902527488, 5172527616),
            (4097, 2902528000, 6305042432), (8192, 2902560256, 6514737664),
        ],
        "minimax-m2.7": [
            (512, 1788113408, 2657070592), (513, 1701294080, 2042777088),
            (1024, 1701297664, 2585378304), (2048, 1701305856, 2851724800),
            (2049, 1701306368, 2825756160), (4096, 1701322240, 3384417792),
            (4097, 1701322752, 3870608896), (8192, 1701355008, 4449803776),
        ],
        "llama-3.2-1b": [
            (512, 1050682368, 1633755136), (513, 1087447552, 1633755648),
            (4096, 1087475712, 2013892608), (4097, 1087476224, 2013893120),
            (16384, 1087574016, 2416644096), (32768, 1050940416, 2953646080),
            (65537, 1087967744, 4027650560), (70656, 1088008192, 4195463168),
            (106987, 1088299008, 5370159104),
        ],
    },
    "sdpa": {
        "qwen3-235b": [
            (512, 1810307072, 2451590656), (513, 1722306560, 3288537088),
            (1024, 1722310144, 3673299968), (1025, 1722310656, 3779291648),
            (2048, 1722318336, 4295300096), (2049, 1722318848, 4640747008),
            (4096, 1722334720, 5483592192), (4097, 1722439168, 6149313024),
            (8192, 1722472960, 7777781248), (8193, 1724601856, 9260207616),
        ],
        "llama-3.3-70b": [
            (512, 2889661952, 3636826624), (513, 2900369408, 3946733568),
            (1024, 2900372992, 4383973376), (1025, 2900373504, 4383973888),
            (2048, 2900381184, 5195579392), (2049, 2900381696, 5195579904),
            (4096, 2900397568, 6625853440), (4097, 2900398080, 6625853952),
            (8192, 2900431872, 9488531456), (8193, 2902560768, 9488531968),
        ],
        "minimax-m2.7": [
            (512, 1788113408, 2656119296), (513, 1700342784, 2040345088),
            (1024, 1700346368, 2584427008), (1025, 1700346880, 2573659648),
            (2048, 1700354560, 2871456768), (2049, 1700355072, 3383168512),
            (4096, 1700370944, 3978769408), (4097, 1700371456, 5002186240),
            (8192, 1700403712, 6194345984), (8193, 1701355520, 8241174528),
        ],
        "llama-3.2-1b": [
            (512, 1050682368, 1609799680), (513, 1063492096, 1609800192),
            (1024, 1063495680, 1889249280), (1025, 1063496192, 1889249792),
            (2048, 1063503872, 1922811904), (2049, 1063504384, 1922812416),
            (4096, 1063520256, 2293000704), (4097, 1063520768, 2293001216),
            (8192, 1063553024, 3123505664), (8193, 1063553536, 3123506176),
            (16384, 1063618560, 4784551424), (16385, 1063654912, 4784551936),
            (32768, 1063787008, 8110896640), (32769, 1068111360, 8110897152),
            (65536, 1068374528, 14763587072), (65537, 1077024256, 14763587584),
        ],
    },
}

# Points EXCLUDED from the gate, with the reason each was excluded. Never
# silently dropped: the script prints them and their errors below the gate line,
# it just does not count them. The only entries allowed here are ones the
# measurement itself flags as unreliable -- see
# plans/2026-08-15-vram-peak-RESULTS.md S3's methodology caveat: the first one
# or two points of a fresh arm sit on an allocator/expert-cache floor that has
# not settled (Qwen3-235B reports alloc_before=1.81 GB at its first point
# against a stable 1.72 GB from the next point on, and its transient there is
# 0.64 G against 1.64 G one chunk-boundary step later).
_UNMEASURED = set(MEASURED) - set(SCORE_ELEM_BYTES)
if _UNMEASURED:
    raise SystemExit(
        f"MEASURED has arms for {sorted(_UNMEASURED)}, which vram_plan has no "
        f"measured score coefficient for -- score_elem_bytes() would hand them "
        f"the conservative unmeasured default, and gating measured data against "
        f"a deliberately-conservative fallback is not a check of anything.")


NOISY_FIRST_POINTS = {
    ("flash_attention_2", "qwen3-235b", 512): "first point of a fresh arm; alloc_before floor 1.81 GB vs 1.72 GB stable",
    ("sdpa", "qwen3-235b", 512): "first point of a fresh arm; same elevated floor, byte-identical to the flash arm's",
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

def head_bytes_for(key: str, cfg: dict, raw: dict) -> int:
    """Bytes of the checkpoint's lm_head weight, by the SAME rule vaayu_base
    uses at load: stat the head shard, or the embedding shard when the config
    says the weights are tied (a tied checkpoint's lm_head shard is an empty
    16-byte file). A file stat, not a typed number (CLAUDE.md S5)."""
    directory, _ = CHECKPOINTS[key]
    tied = cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))
    name = "model.embed_tokens" if tied else "lm_head"
    try:
        return os.path.getsize(f"{MODELS_ROOT}/{directory}/splitted_model/{name}.safetensors")
    except OSError:
        return 0


def shape_for(key: str, attn_impl: str) -> VramShape:
    """Build the probe's VramShape from the checkpoint's own config.json.

    Every geometry number is READ, never typed (CLAUDE.md S5).
    resident_expert_bytes mirrors what vaayu_base computes at load: non-zero
    only for the fused-EXL3 forward, as num_experts * expert_bytes.
    score_elem_bytes mirrors it too: read from the kernel this arm ran, exactly
    as vaayu_base reads it off model._active_attn_impl.
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
        score_elem_bytes=score_elem_bytes(attn_impl),
        alloc_factor=1.0,
        kv_capacity_buckets=is_exl3,
        resident_expert_bytes=RESIDENT_EXPERT_BYTES[key],
        head_bytes=head_bytes_for(key, cfg, raw),
    )


def main() -> int:
    print("=" * 78)
    print("FORM CHECK -- peak = KV(capacity) + max(last-chunk logits, resident")
    print("experts) + N*n_heads*chunk*S*score_elem_bytes[kernel]")
    print("alloc_factor = 1.0; the only fitted number is the per-kernel score")
    print("coefficient, one value for all four geometries")
    print("=" * 78)
    worst, outside, total, skipped = 0.0, 0, 0, []
    for impl in sorted(MEASURED):
        coeff = score_elem_bytes(impl)
        print(f"\n### kernel={impl}  score_elem_bytes={coeff}\n")
        print(f"  {'model':<15}{'S':>8}{'transient':>11}{'predicted':>11}{'rel err':>9}")
        for key, points in MEASURED[impl].items():
            shape = shape_for(key, impl)
            errs = []
            for S, before, peak in sorted(points):
                transient = peak - before
                predicted = predict_peak_vram(shape, S)
                rel = (predicted - transient) / transient
                reason = NOISY_FIRST_POINTS.get((impl, key, S))
                if reason is None:
                    errs.append(rel)
                    worst = max(worst, abs(rel))
                    outside += abs(rel) > GATE
                    total += 1
                    flag = "" if abs(rel) <= GATE else "   <-- FAIL"
                else:
                    skipped.append((impl, key, S, rel, reason))
                    flag = "   (excluded, see NOISY_FIRST_POINTS)"
                print(f"  {key:<15}{S:>8}{transient/GB:>10.4f}G{predicted/GB:>10.4f}G"
                      f"{100*rel:>8.1f}%{flag}")
            print(f"  {'':<15}-> {FORWARD[key]}, residency "
                  f"{shape.resident_expert_bytes/GB:.3f}G, worst "
                  f"{100*max(abs(e) for e in errs):.1f}%\n")

    print("=" * 78)
    print(f"GATE: {outside}/{total} points outside +-{GATE:.0%}, worst "
          f"{100*worst:.1f}%  ->  {'PASS' if not outside else 'FAIL'}")
    for impl, key, S, rel, reason in skipped:
        print(f"  excluded: {impl}/{key}/S={S} ({100*rel:+.1f}%) -- {reason}")
    print("=" * 78)
    return 0 if not outside else 1


if __name__ == "__main__":
    sys.exit(main())
