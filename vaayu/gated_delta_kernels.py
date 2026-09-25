"""Fused decode kernels for gated-delta-net mixers (S3, 2026-09-25).

transformers' gated-delta-net mixers (qwen3_next, qwen3_5, qwen3_5_moe,
olmo_hybrid) decode through three swappable seams -- `recurrent_gated_delta_rule`,
`causal_conv1d_update` and the gated `norm` -- that fall back to torch when `fla`
/ `causal_conv1d` are absent. The fallbacks cost 45 kernel launches per mixer per
decode token on a HOST-bound phase: on Qwen3-Next the 36 mixers' host wall was
15.8 ms of a 108.5 ms token, and one hand-written Triton kernel per seam cut a
mixer's decode call 400 -> 197 us (`fla` itself: 391 -- its Python wrappers give
the launch savings back). docs/2026-09-25-S3-census-rederive-RESULTS.md.

`install_gated_delta_kernels` puts a kernel on each seam it finds. Detection is by
the surface a seam needs, never by class or architecture name (CLAUDE.md §5), and
each fused seam is proven against the stock one on the target device before it
goes in; one that disagrees stays stock, loudly. CUDA only -- CPU compute keeps
the torch path.
"""
import math
import types

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only torch builds ship without triton
    triton = None

# Every fused seam carries the stock callable it replaced under this attribute:
# the installer's idempotency test, and the way a probe toggles a seam off.
_MARK = "_vaayu_fused"
# Self-check tolerance on seam OUTPUTS (relative L2 vs the stock seam). The
# 2026-09-25 bake-off measured 5-7e-3 at Qwen3-Next's real shapes; a changed
# formula lands orders of magnitude above it.
_CHECK_TOL = 5e-2
_RULE_BV = 32    # value-dim block per rule program (the bake-off's 32, 4 warps)
_CONV_BD = 256   # channels per conv-update program
_ATTR = {"rule": "recurrent_gated_delta_rule", "conv": "causal_conv1d_update"}


def gated_delta_seams(model) -> dict:
    """{"rule": [...], "conv": [...], "norm": [...]} -- every gated-delta mixer
    under `model` (itself included) that exposes each seam. A mixer is a module
    with a callable `recurrent_gated_delta_rule`; its `norm` counts only if it
    carries a 1-D `weight` and a float `variance_epsilon` (the gated RMSNorm's
    surface -- a dense model's top-level norm has both but is no mixer's)."""
    seams = {"rule": [], "conv": [], "norm": []}
    for module in model.modules():
        if not callable(getattr(module, "recurrent_gated_delta_rule", None)):
            continue
        seams["rule"].append(module)
        if callable(getattr(module, "causal_conv1d_update", None)):
            seams["conv"].append(module)
        norm = getattr(module, "norm", None)
        weight = getattr(norm, "weight", None)
        if (isinstance(weight, torch.Tensor) and weight.dim() == 1
                and isinstance(getattr(norm, "variance_epsilon", None), float)):
            seams["norm"].append(module)
    return seams
