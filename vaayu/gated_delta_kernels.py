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
    # pylint's possibly-used-before-assignment is a false positive on all three
    # kernel launches below (here, fused_conv_update, fused_gated_norm): each
    # kernel name is bound under the module-level `if triton is not None:`
    # guard, and every fused_* wrapper is reachable only after
    # install_gated_delta_kernels has already returned early when triton is
    # None -- so the kernel is always bound before its launch runs.
    _rule_kernel[(triton.cdiv(V, BV), B * HV)](  # pylint: disable=possibly-used-before-assignment
        query, key, value, g, beta, h0, o, ht,
        *query.stride()[:3], *key.stride()[:3], *value.stride()[:3],
        *g.stride(), *beta.stride(), *sh,
        T, 1 / (K ** 0.5),
        HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV,
        HAS_H0=initial_state is not None, L2=use_qk_l2norm_in_kernel,
        STORE_HT=output_final_state, num_warps=4)
    return o, (ht if output_final_state else None)


if triton is not None:

    @triton.jit
    def _conv_kernel(x, s, w, bias, o, D, sx_b, sx_d, ss_b, ss_d, ss_w, sw_d, sw_w,
                     W: tl.constexpr, BD: tl.constexpr, HAS_BIAS: tl.constexpr):
        # Shift the state left one slot, append x, depthwise dot, silu -- in place.
        i_b = tl.program_id(0)
        od = tl.program_id(1) * BD + tl.arange(0, BD)
        md = od < D
        bx = tl.load(x + i_b * sx_b + od * sx_d, mask=md, other=0.0)
        sp = s + i_b * ss_b + od * ss_d
        wp = w + od * sw_d
        acc = tl.zeros([BD], dtype=tl.float32)
        # the fallback builds cat([state, x]).to(weight.dtype) and copies its tail
        # back into the state: every value passes through weight.dtype
        for j in tl.static_range(W - 1):
            sj = tl.load(sp + (j + 1) * ss_w, mask=md, other=0.0).to(w.dtype.element_ty)
            tl.store(sp + j * ss_w, sj.to(s.dtype.element_ty), mask=md)
            acc += sj.to(tl.float32) * tl.load(wp + j * sw_w, mask=md, other=0.0).to(tl.float32)
        tl.store(sp + (W - 1) * ss_w, bx.to(w.dtype.element_ty).to(s.dtype.element_ty), mask=md)
        acc += (bx.to(w.dtype.element_ty).to(tl.float32)
                * tl.load(wp + (W - 1) * sw_w, mask=md, other=0.0).to(tl.float32))
        if HAS_BIAS:
            acc += tl.load(bias + od, mask=md, other=0.0).to(tl.float32)
        # the fallback convolves in weight.dtype: round there before silu
        acc = acc.to(w.dtype.element_ty).to(tl.float32)
        y = (acc * tl.sigmoid(acc)).to(w.dtype.element_ty)
        tl.store(o + i_b * D + od, y.to(o.dtype.element_ty), mask=md)


def fused_conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """One-token drop-in for transformers' `torch_causal_conv1d_update`: shifts
    `conv_state` in place (any strides -- HF's cache is channels-last after a
    >= W-token prefill) and returns silu(conv) as [B, D, 1]. Like the fallback,
    it applies silu whatever `activation` says."""
    B, D, _ = hidden_states.shape
    o = torch.empty(B, D, 1, device=hidden_states.device, dtype=hidden_states.dtype)
    _conv_kernel[(B, triton.cdiv(D, _CONV_BD))](  # pylint: disable=possibly-used-before-assignment
        hidden_states, conv_state, weight, bias if bias is not None else weight, o, D,
        hidden_states.stride(0), hidden_states.stride(1), *conv_state.stride(),
        *weight.stride(),
        W=conv_state.shape[-1], BD=_CONV_BD, HAS_BIAS=bias is not None, num_warps=4)
    return o


def _conv_seam(stock):
    def seam(hidden_states, conv_state, weight, bias=None, activation=None):
        if hidden_states.shape[-1] != 1:   # HF's decode path never sends more
            return stock(hidden_states, conv_state, weight, bias, activation)
        return fused_conv_update(hidden_states, conv_state, weight, bias, activation)

    setattr(seam, _MARK, stock)
    return seam


if triton is not None:

    @triton.jit
    def _norm_kernel(x, z, w, o, sx_r, sx_n, sz_r, sz_n, sw, N, eps,
                     BN: tl.constexpr, ROUND_MID: tl.constexpr):
        # One row: rmsnorm in fp32 -> round -> x weight -> x silu(gate) in fp32 -> round.
        r = tl.program_id(0)
        on = tl.arange(0, BN)
        m = on < N
        bx = tl.load(x + r * sx_r + on * sx_n, mask=m, other=0.0).to(tl.float32)
        y = bx * tl.rsqrt(tl.sum(bx * bx) / N + eps)
        y = y.to(o.dtype.element_ty).to(tl.float32)
        y = tl.load(w + on * sw, mask=m, other=0.0).to(tl.float32) * y
        if ROUND_MID:  # weight * x stays in the input dtype unless weight promotes it
            y = y.to(o.dtype.element_ty).to(tl.float32)
        bz = tl.load(z + r * sz_r + on * sz_n, mask=m, other=0.0).to(tl.float32)
        tl.store(o + r * N + on, (y * (bz * tl.sigmoid(bz))).to(o.dtype.element_ty), mask=m)


def fused_gated_norm(weight, eps, hidden_states, gate):
    """transformers' `*RMSNormGated.forward(hidden_states, gate)` in one launch,
    at the stock rounding points. Rows are everything but the last dim."""
    n = weight.numel()
    x = hidden_states.reshape(-1, n)
    z = gate.reshape(-1, n)
    o = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    bn = triton.next_power_of_2(n)
    _norm_kernel[(x.shape[0],)](  # pylint: disable=possibly-used-before-assignment
        x, z, weight, o, *x.stride(), *z.stride(), weight.stride(0), n, eps,
        BN=bn, ROUND_MID=torch.promote_types(weight.dtype, x.dtype) == x.dtype,
        num_warps=1 if bn <= 256 else 4)
    return o.view(hidden_states.shape)


def _patch_norm(norm):
    """Patch the norm's forward IN PLACE and read `norm.weight` at call time:
    a streamed layer's Parameters are REPLACED on every load
    (`set_module_tensor_to_device`), so a swapped-in module would go stale."""
    stock = norm.forward

    def forward(hidden_states, gate=None):
        if gate is None:
            return stock(hidden_states, gate)
        return fused_gated_norm(norm.weight, norm.variance_epsilon, hidden_states, gate)

    setattr(forward, _MARK, stock)
    norm.forward = forward


def _rule_seam(stock):
    def seam(query, key, value, g, beta, initial_state, output_final_state,
             use_qk_l2norm_in_kernel=False):
        return fused_recurrent_rule(query, key, value, g, beta, initial_state,
                                    output_final_state, use_qk_l2norm_in_kernel)

    setattr(seam, _MARK, stock)
    return seam


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _randn(gen, device, *shape):
    return torch.randn(*shape, generator=gen, device=device)


# The self-checks run each seam once, stock vs fused, on tiny random inputs in
# the model's dtype. Geometry does not matter to the MATH they compare; the
# kernels' geometry handling is the unit tests' job.

def _check_rule(stock, device, dtype) -> float:
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (_randn(gen, device, 1, 2, 2, 16).to(dtype) for _ in range(3))
    g = -torch.rand(1, 2, 2, generator=gen, device=device)
    beta = torch.rand(1, 2, 2, generator=gen, device=device).to(dtype)
    h0 = (_randn(gen, device, 1, 2, 16, 16) * 0.1).to(dtype)
    kw = dict(g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    so, sh = stock(q, k, v, initial_state=h0.clone(), **kw)
    fo, fh = fused_recurrent_rule(q, k, v, initial_state=h0.clone(), **kw)
    return max(_rel(fo, so), _rel(fh, sh))


def _check_conv(stock, device, dtype) -> float:
    gen = torch.Generator(device=device).manual_seed(0)
    x = _randn(gen, device, 1, 1, 64).to(dtype).transpose(1, 2)       # [B, D, 1], as HF passes it
    state = _randn(gen, device, 1, 4, 64).to(dtype).transpose(1, 2)   # channels-last, like HF's cache
    w = _randn(gen, device, 64, 4).to(dtype)
    s_state, f_state = state.clone(), state.clone()
    so = stock(x, s_state, w, None, "silu")
    fo = fused_conv_update(x, f_state, w, None, "silu")
    return _rel(fo, so) if torch.equal(f_state, s_state) else math.inf


def _check_norm(norm, device, dtype) -> float:
    gen = torch.Generator(device=device).manual_seed(0)
    n = norm.weight.numel()
    # A stand-in carries the weight: a streamed layer's own may still be on the
    # meta device at install time.
    stand_in = types.SimpleNamespace(weight=_randn(gen, device, n).to(norm.weight.dtype),
                                     variance_epsilon=norm.variance_epsilon)
    x, z = (_randn(gen, device, 4, n).to(dtype) for _ in range(2))
    so = type(norm).forward(stand_in, x, z)
    fo = fused_gated_norm(stand_in.weight, stand_in.variance_epsilon, x, z)
    return _rel(fo, so)


def _passes_self_check(seam, stock, mixer, device, dtype) -> bool:
    try:
        if seam == "norm":
            err = _check_norm(mixer.norm, device, dtype)
        elif seam == "rule":
            err = _check_rule(stock, device, dtype)
        else:
            err = _check_conv(stock, device, dtype)
    except Exception as exc:  # a Triton compile/launch failure, or a stock seam with another contract
        why = f"{type(exc).__name__}: {exc}"
    else:
        if err <= _CHECK_TOL:
            return True
        why = ("its conv state is not bit-identical to the stock seam's" if math.isinf(err)
               else f"relL2 {err:.2e} vs the stock seam, tolerance {_CHECK_TOL:.0e}")
    print(f"[Vaayu] WARNING: fused gated-delta '{seam}' kernel failed its self-check "
          f"({why}); that seam stays on the stock torch path")
    return False


def _install(seam, mixer, stock):
    if seam == "norm":
        _patch_norm(mixer.norm)
    elif seam == "rule":
        mixer.recurrent_gated_delta_rule = _rule_seam(stock)
    else:
        mixer.causal_conv1d_update = _conv_seam(stock)


def install_gated_delta_kernels(hf_model, device, dtype) -> dict:
    """Put a fused kernel on every gated-delta seam under `hf_model` whose
    on-device self-check passes. Returns {"rule": n, "conv": n, "norm": n},
    already-fused seams included, so a second call is a no-op. Nothing is
    installed off CUDA or without triton: CPU compute keeps the torch path."""
    counts = {"rule": 0, "conv": 0, "norm": 0}
    if triton is None or torch.device(device).type != "cuda":
        return counts
    verdicts = {}
    for seam, mixers in gated_delta_seams(hf_model).items():
        for mixer in mixers:
            current = mixer.norm.forward if seam == "norm" else getattr(mixer, _ATTR[seam])
            if not hasattr(current, _MARK):
                key = ((seam, type(mixer.norm), mixer.norm.weight.numel(),
                        mixer.norm.weight.dtype, mixer.norm.variance_epsilon)
                       if seam == "norm" else (seam, current))
                if key not in verdicts:
                    verdicts[key] = _passes_self_check(seam, current, mixer, device, dtype)
                if not verdicts[key]:
                    continue
                _install(seam, mixer, current)
            counts[seam] += 1
    return counts
