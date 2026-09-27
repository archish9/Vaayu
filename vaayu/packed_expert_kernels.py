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
            u = v.to(tl.uint32, bitcast=True)
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            return u.to(tl.float32, bitcast=True)
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
            host = torch.empty(max(n, 64), dtype=torch.int64, pin_memory=True)
            slot = self._slots[i] = [host, torch.empty(host.shape, dtype=torch.int64,
                                                       device=device), torch.cuda.Event(), False]
        host, dev, done, used = slot
        if used:
            done.synchronize()
        host[:n] = torch.tensor(values, dtype=torch.int64)
        dev[:n].copy_(host[:n], non_blocking=True)
        done.record()
        slot[3] = True
        return dev


_TABLES = _TableRing()
_VERDICTS: dict = {}


def _layout(geom) -> dict:
    """{role: (word offset, scale offset, rows, cols)} inside a blob's flat streams."""
    bits, gs, roles = geom
    out, w_at, s_at = {}, 0, 0
    for role, rows, cols in roles:
        out[role] = (w_at, s_at, rows, cols)
        w_at += rows * cols * bits // 32
        s_at += rows * cols // gs
    return out


def _block_k(k: int, gs: int):
    """Largest power-of-two multiple of the group size, at most max(_BK_MAX, gs), dividing k."""
    best, b = None, gs
    while b <= max(_BK_MAX, gs):
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
    _gate_up_kernel[(n, inter // _GU_BLOCK_N)](
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
    _down_kernel[(hid // bn,)](
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
    """
    if _checked(x, pairs, routing_weights, act_fn):
        _launch(x, pairs, routing_weights, out)
    else:
        stock_experts_decode(x, pairs, routing_weights, out, act_fn)


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
