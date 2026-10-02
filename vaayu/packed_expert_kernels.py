"""Fused packed-expert decode kernels for the non-EXL3 fused MoE forward (S7, 2026-09-27).

`_make_fused_sparse_forward` serves a pack-quantized checkpoint (compressed-tensors
`pack-quantized`; Qwen3-Next-80B-A3B AWQ int4 on this box) one routed expert at a time:
dequantize the packed blob to dense weights, three M=1 `F.linear`, activation, routing weight,
accumulate -- ~14 launches and ~135 us of host per expert for ~33 us of GPU work. On Qwen3-Next
that was 62 % of a decode token (docs/2026-09-27-S7-dispatch-census-RESULTS.md).

For ONE token this module does the whole layer in two Triton launches that read the packed
blobs directly through a pointer table: `_gate_up_kernel` (gate and up GEMVs, SiLU, product)
and `_down_kernel` (down GEMV, routing weight, and the fp32 sum over the table's pairs in
order -- expert id, then slot -- the order `accumulate_expert_outputs` sums in). Every weight
and intermediate is rounded where the stock path rounds, so the output differs from stock only
by fp32 summation order inside each dot product (measured: 99.87 % of outputs bit-identical
over 48 layers x 8 draws). Nothing depends on whether an expert was a VRAM hit or a miss --
the table is the same either way -- which keeps the MoE output a deterministic function of its
input (row 44 F6).

CUDA + triton only. Each new (geometry, dtype) key is held to `stock_experts_decode` on its
first call; a disagreement or an unsupported geometry stays on the stock arithmetic, loudly.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # CPU-only torch builds ship without triton
    triton = None

# relL2 of a layer's output vs `stock_experts_decode`. The kernels measure ~1e-5 against stock
# at Qwen3-Next's shapes (single-ulp flips); a wrong layout or formula lands near 1.
_CHECK_TOL = 1e-2
# Tile shapes: the S7 step-2 sweep winners on this box's RTX 5060 Ti (sm_120) at Qwen3-Next's
# geometry -- GPU tuning, not a model number. BLOCK_K is re-derived per geometry (`_block_k`)
# and down's BLOCK_N shrinks (`_pow2_divisor`) so both always divide the derived dims.
_GU_BLOCK_N, _GU_WARPS, _GU_STAGES = 1, 1, 1
_DN_BLOCK_N, _DN_WARPS, _DN_STAGES = 4, 1, 3
_BK_MAX = 512
# Prefill (S20): rows of one expert per block. 16 is the smallest tile a tensor-core dot
# takes; a prefill chunk gives each routed expert ~10-14 rows, so a larger block would be
# mostly padding. The other shapes are the S20 sweep winners on the same GPU.
_PF_BLOCK_M = 16
_PF_GU_BLOCK_N, _PF_GU_WARPS, _PF_GU_STAGES = 64, 2, 1
_PF_DN_BLOCK_N, _PF_DN_WARPS, _PF_DN_STAGES = 64, 2, 1
_PF_RD_BLOCK_N = 256
_PF_MIN_TILE = 16           # tl.dot's smallest operand dimension
_PF_GU_BK_MAX, _PF_DN_BK_MAX = 256, 256
# The dtypes the kernels specialise on (rounding mode, scale pointer type).
_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_ROLES = ("gate", "down", "up")

if triton is not None:
    @triton.jit
    def _round(v, CODE: tl.constexpr):
        """fp32 `v` rounded (nearest-even) to the dtype CODE names, returned as fp32 --
        applied wherever the stock path stores into that dtype."""
        if CODE == 0:
            # bf16 in integer ops: add half an ulp (ties to even), drop the low 16 bits.
            # A NaN is passed through untouched: the integer carry would turn it into a
            # signed zero and mask a fault the stock path propagates.
            u = v.to(tl.uint32, bitcast=True)
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            return tl.where(v == v, u.to(tl.float32, bitcast=True), v)
        elif CODE == 1:
            return v.to(tl.float16).to(tl.float32)
        else:
            return v

    @triton.jit
    def _scale_ptr(p, CODE: tl.constexpr):
        if CODE == 0:
            return p.to(tl.pointer_type(tl.bfloat16))
        elif CODE == 1:
            return p.to(tl.pointer_type(tl.float16))
        else:
            return p.to(tl.pointer_type(tl.float32))

    @triton.jit
    def _dot_rows(pk, sc, rows, v_ptr, k0, K: tl.constexpr, BITS: tl.constexpr,
                  GS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  ACT: tl.constexpr):
        """fp32 dot of `rows` of one packed [*, K] weight with v[k0 : k0 + BLOCK_K].

        2-D on purpose: one static pass per code position inside a word, no reshape --
        the 3-D reshape formulation measured 2x slower (S7 step 2). Each weight is
        (code - 2^(BITS-1)) * scale, both in the activation dtype, rounded to it, exactly
        as `_fused_dequant_kernel` stores it."""
        PER: tl.constexpr = 32 // BITS
        BKW: tl.constexpr = BLOCK_K // PER
        wi = k0 // PER + tl.arange(0, BKW)
        words = tl.load(pk + rows[:, None] * (K // PER) + wi[None, :])
        s = tl.load(sc + rows[:, None] * (K // GS) + (wi // (GS // PER))[None, :])
        s = _round(s.to(tl.float32), ACT)
        acc = tl.zeros([BLOCK_N, BKW], tl.float32)
        for j in tl.static_range(PER):
            c = (words >> (j * BITS)) & ((1 << BITS) - 1)
            # code -> float without an int->float convert: bits of 2^23 + c, minus 2^23.
            f = (c | 0x4B000000).to(tl.float32, bitcast=True) - (8388608.0 + (1 << (BITS - 1)))
            v = tl.load(v_ptr + wi * PER + j).to(tl.float32)
            acc += _round(f * s, ACT) * v[None, :]
        return tl.sum(acc, axis=1)

    @triton.jit
    def _gate_up_kernel(x_ptr, tbl_ptr, h_ptr,
                        HID: tl.constexpr, INTER: tl.constexpr, BITS: tl.constexpr,
                        GS: tl.constexpr, GATE_W: tl.constexpr, GATE_S: tl.constexpr,
                        UP_W: tl.constexpr, UP_S: tl.constexpr, ACT: tl.constexpr,
                        SC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        p = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        pk = tl.load(tbl_ptr + p * 3).to(tl.pointer_type(tl.int32))
        sc = _scale_ptr(tl.load(tbl_ptr + p * 3 + 1), SC)
        ag = tl.zeros([BLOCK_N], tl.float32)
        au = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(0, HID, BLOCK_K):
            ag += _dot_rows(pk + GATE_W, sc + GATE_S, rows, x_ptr, k0, HID, BITS, GS,
                            BLOCK_N, BLOCK_K, ACT)
            au += _dot_rows(pk + UP_W, sc + UP_S, rows, x_ptr, k0, HID, BITS, GS,
                            BLOCK_N, BLOCK_K, ACT)
        g = _round(ag, ACT)
        u = _round(au, ACT)
        a = _round(g / (1.0 + tl.exp(-g)), ACT)
        tl.store(h_ptr + p * INTER + rows, _round(a * u, ACT).to(h_ptr.dtype.element_ty))

    @triton.jit
    def _down_kernel(h_ptr, tbl_ptr, w_ptr, out_ptr, n_pairs,
                     HID: tl.constexpr, INTER: tl.constexpr, BITS: tl.constexpr,
                     GS: tl.constexpr, DN_W: tl.constexpr, DN_S: tl.constexpr,
                     ACT: tl.constexpr, SC: tl.constexpr, W_ROUND: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        out = tl.zeros([BLOCK_N], tl.float32)
        # Table order IS the summation order: expert id, then slot (row 44 F6).
        for p in range(0, n_pairs):
            pk = tl.load(tbl_ptr + p * 3).to(tl.pointer_type(tl.int32)) + DN_W
            sc = _scale_ptr(tl.load(tbl_ptr + p * 3 + 1), SC) + DN_S
            slot = tl.load(tbl_ptr + p * 3 + 2)
            acc = tl.zeros([BLOCK_N], tl.float32)
            for k0 in range(0, INTER, BLOCK_K):
                acc += _dot_rows(pk, sc, rows, h_ptr + p * INTER, k0, INTER, BITS, GS,
                                 BLOCK_N, BLOCK_K, ACT)
            # mul_rn, not `*`: a plain product feeding `out +=` is contracted into one FMA,
            # which skips the fp32 rounding of the product that stock's `out * weight` then
            # `index_add_` performs (1-ulp differences, measured).
            c = libdevice.mul_rn(_round(acc, ACT), tl.load(w_ptr + slot).to(tl.float32))
            if W_ROUND:
                c = _round(c, ACT)
            out += c
        tl.store(out_ptr + rows, out)

    @triton.jit
    def _deq(words, s, j, BITS: tl.constexpr, ACT: tl.constexpr):
        """Codes at position `j` of each word, as dequantized weights rounded to the
        activation dtype (fp32 holding that value) -- `_dot_rows`' arithmetic."""
        c = (words >> (j * BITS)) & ((1 << BITS) - 1)
        f = (c | 0x4B000000).to(tl.float32, bitcast=True) - (8388608.0 + (1 << (BITS - 1)))
        return _round(f * s, ACT)

    @triton.jit
    def _gate_up_grouped(x_ptr, rows_ptr, tbl_ptr, h_ptr,
                         TOPK: tl.constexpr, HID: tl.constexpr, INTER: tl.constexpr,
                         BITS: tl.constexpr, GS: tl.constexpr, GATE_W: tl.constexpr,
                         GATE_S: tl.constexpr, UP_W: tl.constexpr, UP_S: tl.constexpr,
                         ACT: tl.constexpr, SC: tl.constexpr, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """One block = BLOCK_M (token, slot) rows of ONE expert (-1 pads the last block).
        h[row] = SiLU(x W_gate^T) * (x W_up^T), every stage rounded where stock rounds."""
        PER: tl.constexpr = 32 // BITS
        BKW: tl.constexpr = BLOCK_K // PER
        b = tl.program_id(0)
        wrows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.load(rows_ptr + b * BLOCK_M + tl.arange(0, BLOCK_M))
        live = rows >= 0
        tok = tl.where(live, rows // TOPK, 0)
        pk = tl.load(tbl_ptr + b * 2).to(tl.pointer_type(tl.int32))
        sc = _scale_ptr(tl.load(tbl_ptr + b * 2 + 1), SC)
        ag = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        au = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, HID, BLOCK_K):
            wi = k0 // PER + tl.arange(0, BKW)
            gi = wi // (GS // PER)
            gw = tl.load(pk + GATE_W + wrows[:, None] * (HID // PER) + wi[None, :])
            uw = tl.load(pk + UP_W + wrows[:, None] * (HID // PER) + wi[None, :])
            gs_ = _round(tl.load(sc + GATE_S + wrows[:, None] * (HID // GS)
                                 + gi[None, :]).to(tl.float32), ACT)
            us_ = _round(tl.load(sc + UP_S + wrows[:, None] * (HID // GS)
                                 + gi[None, :]).to(tl.float32), ACT)
            for j in tl.static_range(PER):
                v = tl.load(x_ptr + tok[:, None] * HID + (wi * PER + j)[None, :])
                ag = tl.dot(v, tl.trans(_deq(gw, gs_, j, BITS, ACT).to(v.dtype)), ag)
                au = tl.dot(v, tl.trans(_deq(uw, us_, j, BITS, ACT).to(v.dtype)), au)
        g = _round(ag, ACT)
        u = _round(au, ACT)
        a = _round(g / (1.0 + tl.exp(-g)), ACT)
        tl.store(h_ptr + rows[:, None] * INTER + wrows[None, :],
                 _round(a * u, ACT).to(h_ptr.dtype.element_ty), mask=live[:, None])

    @triton.jit
    def _down_grouped(h_ptr, rows_ptr, tbl_ptr, w_ptr, y_ptr,
                      HID: tl.constexpr, INTER: tl.constexpr, BITS: tl.constexpr,
                      GS: tl.constexpr, DN_W: tl.constexpr, DN_S: tl.constexpr,
                      ACT: tl.constexpr, SC: tl.constexpr, W_ROUND: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """y[row] = round(h[row] W_down^T) * routing_weight[row], fp32 -- one expert's
        contribution to its token, not yet summed."""
        PER: tl.constexpr = 32 // BITS
        BKW: tl.constexpr = BLOCK_K // PER
        b = tl.program_id(0)
        wrows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.load(rows_ptr + b * BLOCK_M + tl.arange(0, BLOCK_M))
        live = rows >= 0
        hr = tl.where(live, rows, 0)
        pk = tl.load(tbl_ptr + b * 2).to(tl.pointer_type(tl.int32)) + DN_W
        sc = _scale_ptr(tl.load(tbl_ptr + b * 2 + 1), SC) + DN_S
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, INTER, BLOCK_K):
            wi = k0 // PER + tl.arange(0, BKW)
            words = tl.load(pk + wrows[:, None] * (INTER // PER) + wi[None, :])
            s = _round(tl.load(sc + wrows[:, None] * (INTER // GS)
                               + (wi // (GS // PER))[None, :]).to(tl.float32), ACT)
            for j in tl.static_range(PER):
                v = tl.load(h_ptr + hr[:, None] * INTER + (wi * PER + j)[None, :])
                acc = tl.dot(v, tl.trans(_deq(words, s, j, BITS, ACT).to(v.dtype)), acc)
        w = tl.load(w_ptr + hr).to(tl.float32)
        c = libdevice.mul_rn(_round(acc, ACT), w[:, None])
        if W_ROUND:
            c = _round(c, ACT)
        tl.store(y_ptr + rows[:, None] * HID + wrows[None, :], c, mask=live[:, None])

    @triton.jit
    def _reduce_kernel(y_ptr, ord_ptr, out_ptr, TOPK: tl.constexpr, HID: tl.constexpr,
                       BLOCK_N: tl.constexpr):
        """out[t] = the fp32 sum of token t's contributions in `ord`'s order -- ascending
        expert id, the order accumulate_expert_outputs sums in (row 44 F6)."""
        t = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], tl.float32)
        for k in tl.static_range(TOPK):
            r = tl.load(ord_ptr + t * TOPK + k)
            acc += tl.load(y_ptr + r * HID + cols)
        tl.store(out_ptr + t * HID + cols, acc)


class _TableRing:
    """Pinned staging for the per-call pointer table, reused round-robin. A slot's host
    buffer is rewritten only after the H2D that last read it has completed (its event):
    overwriting a pinned buffer under an in-flight copy would corrupt the table silently.
    Each slot has its own device table too, so no queued kernel's table is overwritten."""

    def __init__(self, n_slots: int = 4):
        self._slots: list = [None] * n_slots
        self._next = 0

    def put(self, values: list, device) -> "torch.Tensor":
        i = self._next
        self._next = (i + 1) % len(self._slots)
        n = len(values)
        slot = self._slots[i]
        if slot is None or slot[0].numel() < n or slot[1].device != device:
            # Power-of-two capacity: prefill tables differ in length call to call, and a
            # pinned reallocation per growth step would cost more than the table.
            host = torch.empty(max(64, 1 << (n - 1).bit_length()), dtype=torch.int64,
                               pin_memory=True)
            slot = self._slots[i] = [host, torch.empty(host.shape, dtype=torch.int64,
                                                       device=device), torch.cuda.Event(), False]
        host, dev, done, used = slot
        if used:
            done.synchronize()
        host[:n] = torch.as_tensor(values, dtype=torch.int64)
        dev[:n].copy_(host[:n], non_blocking=True)
        done.record()
        slot[3] = True
        return dev


_TABLES = _TableRing()
_VERDICTS: dict = {}
_PF_TABLES = _TableRing()
_PF_VERDICTS: dict = {}


def _layout(geom) -> dict:
    """{role: (word offset, scale offset, rows, cols)} inside a blob's flat streams."""
    bits, gs, roles = geom
    out, w_at, s_at = {}, 0, 0
    for role, rows, cols in roles:
        out[role] = (w_at, s_at, rows, cols)
        w_at += rows * cols * bits // 32
        s_at += rows * cols // gs
    return out


def _block_k(k: int, gs: int, cap: int = _BK_MAX):
    """Largest power-of-two multiple of the group size, at most max(cap, gs), dividing k."""
    best, b = None, gs
    while b <= max(cap, gs):
        if k % b == 0:
            best = b
        b *= 2
    return best


def _pow2_divisor(n: int, cap: int) -> int:
    b = cap
    while n % b:
        b //= 2
    return b


def _unsupported(geom, act_dtype, scale_dtype, w_dtype):
    """Why the kernels cannot run a call with this geometry and these dtypes, or None."""
    bits, gs, roles = geom
    shapes = {r: (rows, cols) for r, rows, cols in roles}
    if set(shapes) != set(_ROLES):
        return f"expert roles {sorted(shapes)} are not gate/up/down"
    inter, hid = shapes["gate"]
    if shapes["up"] != (inter, hid) or shapes["down"] != (hid, inter):
        return f"projection shapes {shapes} are not a gated MLP"
    if gs & (gs - 1) or gs % (32 // bits):
        return f"group size {gs} is not a power of two holding whole {bits}-bit words"
    if _block_k(hid, gs) is None or _block_k(inter, gs) is None:
        return f"no tile divides hidden {hid} / intermediate {inter} at group size {gs}"
    for what, dt in (("activation", act_dtype), ("scale", scale_dtype),
                     ("routing-weight", w_dtype)):
        if dt not in _DTYPE_CODE:
            return f"{what} dtype {dt}"
    return None


def _table_values(pairs) -> list:
    """The pointer table, flattened: (packed address, scale address, slot) per pair."""
    return [v for blob, slot in pairs
            for v in (blob["flat"]["packed"].data_ptr(), blob["flat"]["scale"].data_ptr(), slot)]


def _gate_up(x, tbl, n, geom, sc):
    """h [n, intermediate] in the activation dtype: SiLU(gate) * up for each table row."""
    bits, gs, _roles = geom
    lay = _layout(geom)
    inter, hid = lay["gate"][2], lay["gate"][3]
    h = torch.empty(n, inter, dtype=x.dtype, device=x.device)
    # pylint's possibly-used-before-assignment is a false positive on both kernel launches
    # (here and in _down): the kernels are bound under the module-level `if triton is not
    # None:` guard, and nothing reaches a launch without install_packed_expert_kernels,
    # which returns early when triton is None (the gated_delta_kernels precedent).
    _gate_up_kernel[(n, inter // _GU_BLOCK_N)](  # pylint: disable=possibly-used-before-assignment
        x, tbl, h, hid, inter, bits, gs, lay["gate"][0], lay["gate"][1],
        lay["up"][0], lay["up"][1], _DTYPE_CODE[x.dtype], sc, _GU_BLOCK_N, _block_k(hid, gs),
        num_warps=_GU_WARPS, num_stages=_GU_STAGES)
    return h


def _down(h, tbl, routing_weights, out, n, geom, sc) -> None:
    """`out` = the fp32 sum, in table order, of each row's down projection times its weight."""
    bits, gs, _roles = geom
    lay = _layout(geom)
    inter, hid = lay["gate"][2], lay["gate"][3]
    rw = routing_weights.reshape(-1)
    bn = _pow2_divisor(hid, _DN_BLOCK_N)
    # torch promotes `out * routing_weights[...]` to fp32 unless the weight is already the
    # activation dtype; only then does the product round.
    w_round = rw.dtype == h.dtype and h.dtype != torch.float32
    _down_kernel[(hid // bn,)](  # pylint: disable=possibly-used-before-assignment
        h, tbl, rw, out, n, hid, inter, bits, gs, lay["down"][0], lay["down"][1],
        _DTYPE_CODE[h.dtype], sc, w_round, bn, _block_k(inter, gs),
        num_warps=_DN_WARPS, num_stages=_DN_STAGES)


def _launch(x, pairs, routing_weights, out) -> None:
    """The two kernels over `pairs`, into `out`. No checks: `_checked` gates this."""
    geom = pairs[0][0]["geom"]
    sc = _DTYPE_CODE[pairs[0][0]["flat"]["scale"].dtype]
    tbl = _TABLES.put(_table_values(pairs), x.device)
    h = _gate_up(x, tbl, len(pairs), geom, sc)
    _down(h, tbl, routing_weights, out, len(pairs), geom, sc)


def stock_experts_decode(x, pairs, routing_weights, out, act_fn) -> None:
    """The per-slot loop's arithmetic for one token, pair by pair, into `out`: the reference
    the kernels are held to and their fallback. Same ops as `_make_fused_sparse_forward`'s
    loop -- `expert_weights_from_entry`, three `F.linear`, `act_fn`, the routing-weight
    product (torch's promotion), an fp32 add in pair order."""
    from .moe_patch import expert_weights_from_entry
    rw = routing_weights.reshape(-1)
    out.zero_()
    for blob, slot in pairs:
        w_gate, w_down, w_up = expert_weights_from_entry(blob, x.dtype)
        y = F.linear(act_fn(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)
        out += (y * rw[slot:slot + 1]).to(out.dtype)


def _verdict(x, pairs, routing_weights, act_fn) -> bool:
    """Run the kernels AND the stock arithmetic on this call's data; True if they agree."""
    blob = pairs[0][0]
    why = _unsupported(blob["geom"], x.dtype, blob["flat"]["scale"].dtype,
                       routing_weights.dtype)
    if why is None:
        try:
            ref = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            got = torch.empty_like(ref)
            stock_experts_decode(x, pairs, routing_weights, ref, act_fn)
            _launch(x, pairs, routing_weights, got)
            err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            if not err <= _CHECK_TOL:
                why = (f"relL2 {err:.2e} vs the stock per-expert loop, "
                       f"tolerance {_CHECK_TOL:.0e}")
        except Exception as exc:  # a Triton compile/launch failure
            why = f"{type(exc).__name__}: {exc}"
    if why is None:
        return True
    print(f"[Vaayu] WARNING: fused packed-expert decode kernel refused ({why}); "
          f"these experts stay on the stock per-expert path")
    return False


def _checked(x, pairs, routing_weights, act_fn) -> bool:
    """The cached verdict for this call's (geometry, dtypes) key, taken on first sight."""
    blob = pairs[0][0]
    key = (blob["geom"], x.dtype, routing_weights.dtype, blob["flat"]["scale"].dtype)
    if key not in _VERDICTS:
        _VERDICTS[key] = _verdict(x, pairs, routing_weights, act_fn)
    return _VERDICTS[key]


def packed_experts_decode(x, pairs, routing_weights, out, act_fn) -> None:
    """One token's routed-expert sum for one MoE layer, written into `out`.

    x               [1, hidden], activation dtype, contiguous, on the current CUDA device
    pairs           [(blob, slot), ...] -- one per routed slot, sorted by (expert id, slot),
                    the order the fp32 sum runs in; `blob` is a `build_expert_blob` entry
    routing_weights this token's [1, top_k] (or [top_k]) weights; `slot` indexes them
    out             [1, hidden] float32, overwritten
    act_fn          the experts' activation; only the stock fallback calls it (the kernels
                    compute SiLU, which `install_packed_expert_kernels` verified it is)

    A call whose entries are not all blobs of ONE geometry -- an unpacked expert's dense
    entry, a packed expert with no fused pass, a layer mixing formats -- runs the stock
    arithmetic over the entries: no kernel can take it, and the stock loop would have.
    """
    geoms = {blob.get("geom") for blob, _slot in pairs}
    if len(geoms) != 1 or None in geoms:
        stock_experts_decode(x, pairs, routing_weights, out, act_fn)
    elif _checked(x, pairs, routing_weights, act_fn):
        _launch(x, pairs, routing_weights, out)
    else:
        stock_experts_decode(x, pairs, routing_weights, out, act_fn)


def _prefill_unsupported(geom, act_dtype, scale_dtype, w_dtype):
    """Why the grouped prefill kernels cannot run this geometry, or None. Everything the
    decode kernels need, plus tiles a tensor-core dot can take."""
    why = _unsupported(geom, act_dtype, scale_dtype, w_dtype)
    if why is not None:
        return why
    bits, gs, _roles = geom
    lay = _layout(geom)
    inter, hid = lay["gate"][2], lay["gate"][3]
    if act_dtype == torch.float32:
        return "fp32 activations (the tensor-core dot would run them at reduced precision)"
    for what, k in (("hidden", hid), ("intermediate", inter)):
        if _block_k(k, gs, min(_PF_GU_BK_MAX, _PF_DN_BK_MAX)) // (32 // bits) < _PF_MIN_TILE:
            return f"{what} {k} at group size {gs} leaves a K tile under {_PF_MIN_TILE} words"
        if _pow2_divisor(k, _PF_MIN_TILE) < _PF_MIN_TILE:
            return f"{what} {k} is not a multiple of {_PF_MIN_TILE}"
    return None


def _prefill_plan(eidx_cpu):
    """Host-side grouping of one call's routing: (order, live, counts).

    order   row ids (token * top_k + slot) sorted by expert id, stable -- each expert's
            rows are a contiguous run, in row order
    live    the routed expert ids, ascending
    counts  rows per live expert
    """
    eids = eidx_cpu.reshape(-1)
    order = torch.argsort(eids, stable=True)
    per_expert = torch.bincount(eids)
    live = per_expert.nonzero().reshape(-1)
    return order, live, per_expert[live]


def _prefill_tables(eidx_cpu, plan, blobs, device):
    """The call's three tables on the device, from ONE pinned copy: (tbl, rows, ord, n_blocks).

    tbl   [n_blocks, 2] (packed address, scale address) -- the expert each block belongs to
    rows  [n_blocks, BLOCK_M] row ids, each expert's run padded to whole blocks with -1
    ord   [tokens, top_k] row ids of each token's slots in ascending expert id: the order
          the fp32 sum runs in (row 44 F6)
    """
    order, live, counts = plan
    n, top_k = eidx_cpu.shape
    nblk = (counts + _PF_BLOCK_M - 1) // _PF_BLOCK_M
    blk_end = nblk.cumsum(0)
    n_blocks = int(blk_end[-1])
    which = torch.repeat_interleave(torch.arange(live.numel()), counts)
    src_start = counts.cumsum(0) - counts
    dest = ((blk_end - nblk) * _PF_BLOCK_M)[which] + torch.arange(order.numel()) - src_start[which]
    rows = torch.full((n_blocks * _PF_BLOCK_M,), -1, dtype=torch.int64)
    rows[dest] = order
    ptrs = torch.tensor([(blobs[e]["flat"]["packed"].data_ptr(),
                          blobs[e]["flat"]["scale"].data_ptr()) for e in live.tolist()],
                        dtype=torch.int64)
    tbl = ptrs[torch.repeat_interleave(torch.arange(live.numel()), nblk)].reshape(-1)
    slot_order = (torch.argsort(eidx_cpu, dim=1, stable=True)
                  + (torch.arange(n) * top_k)[:, None]).reshape(-1)
    dev = _PF_TABLES.put(torch.cat([tbl, rows, slot_order]), device)
    a, b = tbl.numel(), tbl.numel() + rows.numel()
    return dev[:a], dev[a:b], dev[b:b + slot_order.numel()], n_blocks


def _gate_up_g(x, tbl, rows, n_blocks, top_k, geom, sc, h=None):
    """h [tokens * top_k, intermediate], activation dtype: SiLU(gate) * up per live row."""
    bits, gs, _roles = geom
    lay = _layout(geom)
    inter, hid = lay["gate"][2], lay["gate"][3]
    if h is None:
        h = torch.empty(x.shape[0] * top_k, inter, dtype=x.dtype, device=x.device)
    bn = _pow2_divisor(inter, _PF_GU_BLOCK_N)
    _gate_up_grouped[(n_blocks, inter // bn)](  # pylint: disable=possibly-used-before-assignment
        x, rows, tbl, h, top_k, hid, inter, bits, gs, lay["gate"][0], lay["gate"][1],
        lay["up"][0], lay["up"][1], _DTYPE_CODE[x.dtype], sc, _PF_BLOCK_M, bn,
        _block_k(hid, gs, _PF_GU_BK_MAX), num_warps=_PF_GU_WARPS, num_stages=_PF_GU_STAGES)
    return h


def _down_g(h, tbl, rows, n_blocks, routing_weights, geom, sc, y=None):
    """y [tokens * top_k, hidden] fp32: each live row's down projection times its weight."""
    bits, gs, _roles = geom
    lay = _layout(geom)
    inter, hid = lay["gate"][2], lay["gate"][3]
    rw = routing_weights.reshape(-1)
    if y is None:
        y = torch.empty(h.shape[0], hid, dtype=torch.float32, device=h.device)
    bn = _pow2_divisor(hid, _PF_DN_BLOCK_N)
    w_round = rw.dtype == h.dtype and h.dtype != torch.float32
    _down_grouped[(n_blocks, hid // bn)](  # pylint: disable=possibly-used-before-assignment
        h, rows, tbl, rw, y, hid, inter, bits, gs, lay["down"][0], lay["down"][1],
        _DTYPE_CODE[h.dtype], sc, w_round, _PF_BLOCK_M, bn,
        _block_k(inter, gs, _PF_DN_BK_MAX), num_warps=_PF_DN_WARPS, num_stages=_PF_DN_STAGES)
    return y


def _reduce(y, order, out, top_k) -> None:
    """`out` [tokens, hidden] fp32 = each token's contributions summed in `order`."""
    hid = y.shape[1]
    bn = _pow2_divisor(hid, _PF_RD_BLOCK_N)
    _reduce_kernel[(out.shape[0], hid // bn)](  # pylint: disable=possibly-used-before-assignment
        y, order, out, top_k, hid, bn)


def _prefill_launch(x, eidx_cpu, routing_weights, blobs, out, plan=None) -> None:
    """The three kernels over one call. No checks: `_prefill_checked` gates this."""
    plan = plan or _prefill_plan(eidx_cpu)
    blob = blobs[int(plan[1][0])]
    geom, sc = blob["geom"], _DTYPE_CODE[blob["flat"]["scale"].dtype]
    top_k = eidx_cpu.shape[1]
    # The kernels index raw memory: a strided view (a slice of a wider batch) would be
    # read as if it were dense. A no-op on the contiguous tensors the forward passes.
    x = x.contiguous()
    tbl, rows, order, n_blocks = _prefill_tables(eidx_cpu, plan, blobs, x.device)
    h = _gate_up_g(x, tbl, rows, n_blocks, top_k, geom, sc)
    y = _down_g(h, tbl, rows, n_blocks, routing_weights, geom, sc)
    _reduce(y, order, out, top_k)


def _linear_fp32(a, w):
    """`F.linear` with the product summed in fp32 and rounded ONCE to the activation dtype."""
    return F.linear(a.float(), w.float()).to(a.dtype)


def stock_experts_prefill(x, eidx_cpu, routing_weights, blobs, out, act_fn,
                          linear=F.linear) -> None:
    """The per-slot loop's arithmetic for a multi-token call, into `out`: the grouped
    kernels' fallback. Same ops as `_make_fused_sparse_forward`'s loop -- per expert, in
    expert-id order: gather its rows, `expert_weights_from_entry`, three `F.linear`,
    `act_fn`, the routing-weight product (torch's promotion), `index_add_`. A routed expert
    with no entry is skipped, as the loop skips it.

    `linear=_linear_fp32` is the reference the kernels are CHECKED against. cuBLAS's
    half-precision GEMM is not correctly rounded at every shape: measured on this box at
    hidden 2048 -> intermediate 512, 2 to 32 rows give ~58 % correctly rounded outputs (1 row
    and 64+ rows give 100 %), so a check against it would measure cuBLAS, not the kernels."""
    from .moe_patch import expert_weights_from_entry
    order, live, counts = _prefill_plan(eidx_cpu)
    top_k = eidx_cpu.shape[1]
    rw = routing_weights.reshape(-1)
    order = order.to(x.device)
    out.zero_()
    at = 0
    for e, c in zip(live.tolist(), counts.tolist()):
        rows = order[at:at + c]
        at += c
        entry = blobs.get(e)
        if entry is None:
            continue
        w_gate, w_down, w_up = expert_weights_from_entry(entry, x.dtype)
        tok = rows // top_k
        xs = x[tok]
        y = linear(act_fn(linear(xs, w_gate)) * linear(xs, w_up), w_down)
        out.index_add_(0, tok, (y * rw[rows, None]).to(out.dtype))


def _prefill_verdict(x, eidx_cpu, routing_weights, blobs, act_fn, plan) -> bool:
    """Run the grouped kernels AND the stock arithmetic on this call's data; True if they agree."""
    blob = blobs[int(plan[1][0])]
    why = _prefill_unsupported(blob["geom"], x.dtype, blob["flat"]["scale"].dtype,
                               routing_weights.dtype)
    if why is None:
        try:
            ref = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            got = torch.empty_like(ref)
            stock_experts_prefill(x, eidx_cpu, routing_weights, blobs, ref, act_fn,
                                  linear=_linear_fp32)
            _prefill_launch(x, eidx_cpu, routing_weights, blobs, got, plan)
            err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            if not err <= _CHECK_TOL:
                why = (f"relL2 {err:.2e} vs the stock per-expert loop, "
                       f"tolerance {_CHECK_TOL:.0e}")
        except Exception as exc:  # a Triton compile/launch failure
            why = f"{type(exc).__name__}: {exc}"
    if why is None:
        return True
    print(f"[Vaayu] WARNING: fused packed-expert prefill kernel refused ({why}); "
          f"these experts stay on the stock per-expert path")
    return False


def _prefill_checked(x, eidx_cpu, routing_weights, blobs, act_fn, plan) -> bool:
    """The cached verdict for this call's (geometry, dtypes) key, taken on first sight."""
    blob = blobs[int(plan[1][0])]
    key = (blob["geom"], x.dtype, routing_weights.dtype, blob["flat"]["scale"].dtype)
    if key not in _PF_VERDICTS:
        _PF_VERDICTS[key] = _prefill_verdict(x, eidx_cpu, routing_weights, blobs, act_fn, plan)
    return _PF_VERDICTS[key]


def packed_experts_prefill(x, eidx_cpu, routing_weights, blobs, out, act_fn) -> None:
    """A multi-token call's routed-expert sum for one MoE layer, written into `out`.

    x               [tokens, hidden], activation dtype, contiguous, on the current CUDA device
    eidx_cpu        [tokens, top_k] int64 on the CPU: the routed expert ids
    routing_weights [tokens, top_k] on x's device; row t * top_k + s weighs slot s of token t
    blobs           {expert id: entry} for the routed experts -- `build_expert_blob` blobs
                    for the kernels; anything else (a dense entry, mixed geometries, a routed
                    expert with no entry) sends the whole call to the stock arithmetic
    out             [tokens, hidden] float32, overwritten
    act_fn          the experts' activation; only the stock fallback calls it

    The result is a function of (x, routing, weights) alone: the tables are built in expert-id
    order, so neither the order experts arrived in nor where their blobs live can reach it.
    """
    plan = _prefill_plan(eidx_cpu)
    geoms = {blobs[e].get("geom") if e in blobs else None for e in plan[1].tolist()}
    if len(geoms) != 1 or None in geoms:
        stock_experts_prefill(x, eidx_cpu, routing_weights, blobs, out, act_fn)
    elif _prefill_checked(x, eidx_cpu, routing_weights, blobs, act_fn, plan):
        _prefill_launch(x, eidx_cpu, routing_weights, blobs, out, plan)
    else:
        stock_experts_prefill(x, eidx_cpu, routing_weights, blobs, out, act_fn)


def _is_silu(act_fn, device, dtype) -> bool:
    """Behavioural: does the experts' activation compute SiLU? (the kernels hardcode it)"""
    if act_fn is None:
        return False
    t = torch.linspace(-8.0, 8.0, 257, device=device).to(dtype)
    try:
        return torch.equal(act_fn(t), F.silu(t))
    except Exception:
        return False


def _one_real_blob(vaayu_ref, block, device):
    """The VRAM-tier entry of `block`'s lowest-numbered expert, read off the checkpoint (one
    expert's bytes), or None when its experts are not pack-quantized. A packed expert
    `build_expert_blob` refuses is reported, not hidden."""
    from .moe_patch import (_group_packed_tensors_by_weight, _packed_roles_complete,
                            _upload_packed_expert, build_expert_blob)
    from .persist.safetensor_model_persister import load_expert_sparse
    metas = vaayu_ref._expert_offset_index.index.get(block.layer_name, {})
    if not metas:
        return None
    raw = load_expert_sparse(
        str(Path(vaayu_ref.checkpoint_path) / f"{block.layer_name}.safetensors"),
        metas[min(metas)])
    packed = _group_packed_tensors_by_weight(raw)
    if not packed["gate"] or not _packed_roles_complete(packed):
        return None
    why = "its projections are packed at different bit widths or group sizes"
    try:
        blob = build_expert_blob(_upload_packed_expert(packed, device))
    except NotImplementedError as exc:   # zero point / act-order: refused everywhere
        blob, why = None, str(exc)
    if blob is None:
        print(f"[Vaayu] WARNING: fused packed-expert decode kernel not installed: {why}")
    return blob


def install_packed_expert_kernels(vaayu_ref) -> bool:
    """Route one-token decode of every fused non-EXL3 pack-quantized MoE block through the
    kernels -- if they agree with the stock per-expert loop on one real expert of this
    model, on this device. Called once at load, after `patch_moe_blocks` (it reads each
    block's `forward_name`). Seeds the verdict cache for fp32 and activation-dtype routing
    weights, so the Triton JIT and the check are paid here, not at the first token.

    Nothing is installed off CUDA, without triton, or with no block on the non-EXL3 fused
    forward; a non-current CUDA device (Triton launches on the CURRENT device), a packed
    expert that cannot be fused, or a disagreeing kernel is refused with a WARNING. The
    stock path stays in every refused case."""
    device = str(vaayu_ref.running_device)
    if triton is None or torch.device(device).type != "cuda":
        return False
    blocks = [b for b in (getattr(vaayu_ref, "_moe_blocks", None) or [])
              if getattr(b, "forward_name", None) == "_make_fused_sparse_forward"]
    if not blocks or getattr(vaayu_ref, "_expert_offset_index", None) is None:
        return False
    dev_index = torch.device(device).index
    if dev_index is not None and dev_index != torch.cuda.current_device():
        print(f"[Vaayu] WARNING: fused packed-expert decode kernel not installed: model "
              f"device {device} is not the current CUDA device "
              f"({torch.cuda.current_device()}); decode keeps the stock per-expert path")
        return False
    block, dtype = blocks[0], vaayu_ref.running_dtype
    act_fn = getattr(getattr(block.moe_module, block.experts_attr, None), "act_fn", None)
    if not _is_silu(act_fn, device, dtype):
        print("[Vaayu] fused packed-expert decode kernel not used: the experts' activation "
              "is not SiLU")
        return False
    blob = _one_real_blob(vaayu_ref, block, device)
    if blob is None:
        return False
    k = int(getattr(vaayu_ref, "moe_top_k", 0) or 1)
    gen = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn(1, _layout(blob["geom"])["gate"][3], generator=gen, device=device)
         * 0.5).to(dtype)
    w = torch.rand(1, k, generator=gen, device=device) + 0.1
    w = w / w.sum(-1, keepdim=True)
    pairs = [(blob, s) for s in range(k)]
    if not all(_checked(x, pairs, rw, act_fn) for rw in (w, w.to(dtype))):
        return False
    vaayu_ref._packed_moe_kernel = blob["geom"]
    bits, gs, _roles = blob["geom"]
    print(f"[Vaayu] fused packed-expert decode kernel on {len(blocks)} MoE layers "
          f"({bits}-bit, group {gs})")
    return True
