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


if triton is not None:

    @triton.jit
    def _rule_kernel(q, k, v, g, beta, h0, o, ht,
                     sq_b, sq_t, sq_h, sk_b, sk_t, sk_h, sv_b, sv_t, sv_h,
                     sg_b, sg_t, sg_h, sb_b, sb_t, sb_h, sh_b, sh_h, sh_k, sh_v,
                     T, scale,
                     HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr,
                     HAS_H0: tl.constexpr, L2: tl.constexpr, STORE_HT: tl.constexpr):
        # One program owns one (batch, value-head) state slice [K, BV] and walks T.
        i_v = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // HV
        i_h = i_bh % HV
        ok = tl.arange(0, BK)
        ov = i_v * BV + tl.arange(0, BV)
        mk = ok < K
        mv = ov < V
        mkv = mk[:, None] & mv[None, :]
        if HAS_H0:
            h = tl.load(h0 + i_b * sh_b + i_h * sh_h + ok[:, None] * sh_k + ov[None, :] * sh_v,
                        mask=mkv, other=0.0).to(tl.float32)
        else:
            h = tl.zeros([BK, BV], dtype=tl.float32)
        for t in range(T):
            bq = tl.load(q + i_b * sq_b + t * sq_t + i_h * sq_h + ok,
                         mask=mk, other=0.0).to(tl.float32)
            bk = tl.load(k + i_b * sk_b + t * sk_t + i_h * sk_h + ok,
                         mask=mk, other=0.0).to(tl.float32)
            if L2:  # transformers' l2norm, eps 1e-6 -- here in fp32
                bq = bq * tl.rsqrt(tl.sum(bq * bq) + 1e-6)
                bk = bk * tl.rsqrt(tl.sum(bk * bk) + 1e-6)
            bq = bq * scale
            bv = tl.load(v + i_b * sv_b + t * sv_t + i_h * sv_h + ov,
                         mask=mv, other=0.0).to(tl.float32)
            bg = tl.load(g + i_b * sg_b + t * sg_t + i_h * sg_h).to(tl.float32)
            bb = tl.load(beta + i_b * sb_b + t * sb_t + i_h * sb_h).to(tl.float32)
            h = h * tl.exp(bg)
            delta = (bv - tl.sum(h * bk[:, None], axis=0)) * bb
            h = h + bk[:, None] * delta[None, :]
            bo = tl.sum(h * bq[:, None], axis=0)
            tl.store(o + ((i_b * T + t) * HV + i_h) * V + ov, bo.to(o.dtype.element_ty),
                     mask=mv)
        if STORE_HT:
            tl.store(ht + (i_bh * K + ok[:, None]) * V + ov[None, :], h, mask=mkv)


def fused_recurrent_rule(query, key, value, g, beta, initial_state, output_final_state,
                         use_qk_l2norm_in_kernel=False):
    """Drop-in for transformers' `torch_recurrent_gated_delta_rule`: same
    arguments, same `(output [B,T,HV,V] in query.dtype, fp32 state | None)`, one
    launch. l2norm runs in fp32 (the fallback runs it in bf16 -- further from
    fp64); `initial_state` may be None, bf16 (the real cache) or fp32."""
    B, T, HV, K = key.shape
    V = value.shape[-1]
    query, key, value = (x if x.stride(-1) == 1 else x.contiguous()
                         for x in (query, key, value))
    o = torch.empty(B, T, HV, V, device=value.device, dtype=query.dtype)
    ht = (torch.empty(B, HV, K, V, device=value.device, dtype=torch.float32)
          if output_final_state else o)
    h0 = initial_state if initial_state is not None else o
    sh = initial_state.stride() if initial_state is not None else (0, 0, 0, 0)
    BV = min(_RULE_BV, triton.next_power_of_2(V))
    _rule_kernel[(triton.cdiv(V, BV), B * HV)](
        query, key, value, g, beta, h0, o, ht,
        *query.stride()[:3], *key.stride()[:3], *value.stride()[:3],
        *g.stride(), *beta.stride(), *sh,
        T, 1 / (K ** 0.5),
        HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV,
        HAS_H0=initial_state is not None, L2=use_qk_l2norm_in_kernel,
        STORE_HT=output_final_state, num_warps=4)
    return o, (ht if output_final_state else None)
