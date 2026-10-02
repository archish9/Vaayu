"""S20: the grouped packed-expert PREFILL kernels (vaayu/packed_expert_kernels.py).

For a multi-token call the non-EXL3 fused forward used to visit each routed expert alone:
dequantize its blob to dense weights, gather its rows, three cuBLAS GEMMs at a handful of
rows, scatter. The grouped kernels do the whole layer in three launches over the packed
blobs. They round every intermediate where the stock path rounds; what differs is the fp32
sum inside each GEMM -- and there the kernels are the accurate side: cuBLAS's half-precision
GEMM is not correctly rounded at every shape (hidden 2048 -> intermediate 512 at 2..32 rows:
~58 % of outputs, measured), so each stage is held to a CORRECTLY ROUNDED replay, and the
whole output to the stock loop and to an fp64 truth. Three contrasting geometries (CLAUDE.md
§5): Qwen3-Next's real one, MiniMax-like dims at int8 g128, and an int8 g64 one.
"""
import pytest
import torch
import torch.nn.functional as F

import vaayu.packed_expert_kernels as pk
from vaayu.moe_patch import (_fused_dequant_kernel, accumulate_expert_outputs,
                             expert_weights_from_entry)
from tests.test_packed_expert_kernels import _blobs, cuda

BF = torch.bfloat16
# (hidden, intermediate, bits, group)
GEOMS = [
    pytest.param(2048, 512, 4, 32, id="qwen3next-int4-g32"),
    pytest.param(3072, 1536, 8, 128, id="minimax-dims-int8-g128"),
    pytest.param(1024, 512, 8, 64, id="int8-g64"),
]
QWEN = (2048, 512, 4, 32)


def _case(hid, inter, bits, gs, *, n_exp=24, tokens=40, top_k=6, seed=0, w_dtype=torch.float32,
          eidx=None):
    """(x, eidx, routing_weights, blobs): `tokens` rows each routed to `top_k` distinct experts
    of `n_exp` (ids need not be dense: every third id is used, so ids != table positions)."""
    base = _blobs(hid, inter, bits, gs, 8)
    ids = [3 * i + 1 for i in range(n_exp)]
    blobs = {e: base[i % len(base)] for i, e in enumerate(ids)}
    g = torch.Generator().manual_seed(seed)
    if eidx is None:
        eidx = torch.stack([torch.tensor(ids)[torch.randperm(n_exp, generator=g)[:top_k]]
                            for _ in range(tokens)])
    gd = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(eidx.shape[0], hid, generator=gd, device="cuda") * 0.5).to(BF)
    w = torch.rand(eidx.shape, generator=gd, device="cuda") + 0.05
    return x, eidx, (w / w.sum(-1, keepdim=True)).to(w_dtype), blobs


def _launch(x, eidx, w, blobs):
    out = torch.full(x.shape, float("nan"), device="cuda")      # must be overwritten
    pk._prefill_launch(x, eidx, w, blobs, out)
    torch.cuda.synchronize()
    return out


def _stock(x, eidx, w, blobs, linear=F.linear):
    out = torch.full(x.shape, float("nan"), device="cuda")
    pk.stock_experts_prefill(x, eidx, w, blobs, out, F.silu, linear=linear)
    return out


def _prefill(x, eidx, w, blobs):
    out = torch.full(x.shape, float("nan"), device="cuda")
    pk.packed_experts_prefill(x, eidx, w, blobs, out, F.silu)
    torch.cuda.synchronize()
    return out


def _stages(x, eidx, w, blobs, h=None, y=None):
    """The kernels' own intermediates: (h, y, order, plan)."""
    plan = pk._prefill_plan(eidx)
    blob = blobs[int(plan[1][0])]
    geom, sc = blob["geom"], pk._DTYPE_CODE[blob["flat"]["scale"].dtype]
    top_k = eidx.shape[1]
    tbl, rows, order, n_blocks = pk._prefill_tables(eidx, plan, blobs, x.device)
    h = pk._gate_up_g(x, tbl, rows, n_blocks, top_k, geom, sc, h=h)
    y = pk._down_g(h, tbl, rows, n_blocks, w, geom, sc, y=y)
    torch.cuda.synchronize()
    return h, y, order, plan


def _per_expert(eidx, plan):
    """[(expert id, its row ids on the GPU)] in expert-id order."""
    order, live, counts = plan
    order, out, at = order.cuda(), [], 0
    for e, c in zip(live.tolist(), counts.tolist()):
        out.append((e, order[at:at + c]))
        at += c
    return out


def _fp64(x, eidx, w, blobs):
    """The truth: the checkpoint's exact weights, fp64 everywhere."""
    out = torch.zeros(x.shape, dtype=torch.float64, device="cuda")
    top_k, rw = eidx.shape[1], w.reshape(-1).double()
    for e, rows in _per_expert(eidx, pk._prefill_plan(eidx)):
        nb, gs, roles = blobs[e]["geom"]
        flat = _fused_dequant_kernel(blobs[e]["flat"]["packed"], blobs[e]["flat"]["scale"], nb,
                                     gs, torch.float64)
        mats, at = {}, 0
        for role, r, c in roles:
            mats[role] = flat[at:at + r * c].view(r, c)
            at += r * c
        xs = x[rows // top_k].double()
        g, u = xs @ mats["gate"].T, xs @ mats["up"].T
        out.index_add_(0, rows // top_k, ((g * torch.sigmoid(g) * u) @ mats["down"].T)
                       * rw[rows, None])
    return out


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs", GEOMS)
def test_each_stage_is_correctly_rounded(hid, inter, bits, gs):
    """Stage by stage against a replay whose GEMMs are summed in fp64 and rounded ONCE to
    the activation dtype -- what `F.linear` would return if cuBLAS rounded correctly. Every
    other op is stock's own. Measured: gate/up/SiLU/product stage 99.72-99.86 %, down stage
    99.89-99.97 %; with a stage's rounding removed the same test measures 0 %."""
    x, eidx, w, blobs = _case(hid, inter, bits, gs)
    h, y, _order, plan = _stages(x, eidx, torch.ones_like(w), blobs)     # weight 1: y = down
    h_ref, y_ref = torch.empty_like(h), torch.empty_like(y)
    top_k = eidx.shape[1]
    for e, rows in _per_expert(eidx, plan):
        w_gate, w_down, w_up = expert_weights_from_entry(blobs[e], BF)
        xs = x[rows // top_k].double()
        h_ref[rows] = F.silu((xs @ w_gate.double().T).to(BF)) * (xs @ w_up.double().T).to(BF)
        y_ref[rows] = (h[rows].double() @ w_down.double().T).to(BF).float()
    h_equal = (h == h_ref).float().mean().item()
    assert h_equal >= 0.99, f"gate/up/SiLU/product stage: {h_equal:.4%} correctly rounded"
    y_equal = (y == y_ref).float().mean().item()
    assert y_equal >= 0.99, f"down stage: {y_equal:.4%} correctly rounded"


@cuda
@pytest.mark.parametrize("w_dtype", [torch.float32, BF], ids=["fp32-w", "bf16-w"])
def test_the_contribution_is_the_rounded_down_output_times_its_weight(w_dtype):
    """EXACT, both promotions: an fp32 weight leaves an fp32 product, a weight already in
    the activation dtype makes torch round `out * weight` to it."""
    x, eidx, w, blobs = _case(*QWEN, w_dtype=w_dtype)
    _h, down, _o, _p = _stages(x, eidx, torch.ones_like(w), blobs)
    _h, y, _o, _p = _stages(x, eidx, w, blobs)
    assert torch.equal(y, (down.to(BF) * w.reshape(-1, 1)).float())


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs", GEOMS)
def test_the_token_sum_runs_in_expert_id_order(hid, inter, bits, gs):
    """EXACT: the layer output is the fp32 sum of each token's contributions in ascending
    expert id -- accumulate_expert_outputs' order (row 44 F6) -- replayed here with torch
    on the kernels' own contributions."""
    x, eidx, w, blobs = _case(hid, inter, bits, gs)
    _h, y, order, plan = _stages(x, eidx, w, blobs)
    out = torch.full(x.shape, float("nan"), device="cuda")
    pk._reduce(y, order, out, eidx.shape[1])
    expected = torch.zeros(x.shape, device="cuda")
    for _e, rows in _per_expert(eidx, plan):
        expected.index_add_(0, rows // eidx.shape[1], y[rows])
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs", GEOMS)
def test_kernel_output_is_as_accurate_as_the_stock_loop(hid, inter, bits, gs):
    x, eidx, w, blobs = _case(hid, inter, bits, gs)
    got, ref = _launch(x, eidx, w, blobs), _stock(x, eidx, w, blobs)
    precise = _stock(x, eidx, w, blobs, linear=pk._linear_fp32)
    truth = _fp64(x, eidx, w, blobs)
    assert _rel(got, ref) <= 1e-2, f"relL2 vs stock {_rel(got, ref):.2e}"
    assert _rel(got, precise) <= 2e-3, f"relL2 vs the fp32-GEMM replay {_rel(got, precise):.2e}"
    e_got, e_ref = _rel(got, truth), _rel(ref, truth)
    assert e_got <= 1.05 * e_ref + 1e-12, f"fp64 error {e_got:.3e} vs stock's {e_ref:.3e}"


@cuda
def test_a_token_does_not_depend_on_the_tokens_it_shares_a_call_with():
    """Bit-identical alone and in a batch: a row's arithmetic is its own, whichever block
    and block-mates it lands with. (The stock GEMM does not have this property -- cuBLAS
    picks its kernel by row count -- which is why a HIT diff and a cold prefill of the same
    token could differ in this layer.)"""
    x, eidx, w, blobs = _case(*QWEN, tokens=40)
    full = _launch(x, eidx, w, blobs)
    for t in (0, 17, 39):
        alone = _launch(x[t:t + 1], eidx[t:t + 1], w[t:t + 1], blobs)
        assert torch.equal(alone[0], full[t]), f"token {t}"


@cuda
def test_block_boundaries():
    """Experts with 1, exactly 16, 17 and 40 rows: a lone row, a full block, a block plus
    one row, and three blocks. Each token must equal itself computed alone."""
    counts = {1: 1, 4: 16, 7: 17, 10: 40}
    top_k = 2
    slots = [e for e, c in counts.items() for _ in range(c)]            # 74 slot entries
    filler = [13, 16, 19, 22]                                           # distinct second slot
    eidx = torch.tensor([[e, filler[i % 4]] for i, e in enumerate(slots)])
    x, eidx, w, blobs = _case(*QWEN, top_k=top_k, eidx=eidx)
    plan = pk._prefill_plan(eidx)
    assert [int(c) for c in plan[2][:4]] == [1, 16, 17, 40], "premise: the block shapes"
    full = _launch(x, eidx, w, blobs)
    assert not torch.isnan(full).any()
    for t in (0, 1, 16, 17, 33, 34, 73):
        alone = _launch(x[t:t + 1], eidx[t:t + 1], w[t:t + 1], blobs)
        assert torch.equal(alone[0], full[t]), f"token {t}"
    assert _rel(full, _stock(x, eidx, w, blobs, linear=pk._linear_fp32)) <= 2e-3


@cuda
def test_padding_rows_are_never_written():
    """A block's padding rows carry row id -1. The buffers here are views one row into a
    larger sentinel buffer, so an unmasked store lands on the guard row."""
    x, eidx, w, blobs = _case(*QWEN, tokens=5, top_k=3)                 # every block is padded
    n, inter, hid = eidx.numel(), QWEN[1], QWEN[0]
    h_big = torch.full((n + 2, inter), 7.0, dtype=BF, device="cuda")
    y_big = torch.full((n + 2, hid), 7.0, device="cuda")
    _stages(x, eidx, w, blobs, h=h_big[1:n + 1], y=y_big[1:n + 1])
    for big in (h_big, y_big):
        assert (big[0] == 7.0).all() and (big[-1] == 7.0).all(), "a guard row was written"
        assert not (big[1:n + 1] == 7.0).all(dim=1).any(), "a live row was not written"


@cuda
def test_the_output_does_not_depend_on_where_the_blobs_live_or_their_order():
    """A VRAM hit, a fresh upload and a chunk-major retained entry hand the kernels
    different tensors holding the same bytes, collected in arrival order: same bits."""
    x, eidx, w, blobs = _case(*QWEN)
    moved = {e: {"flat": {n: t.clone() for n, t in b["flat"].items()}, "geom": b["geom"]}
             for e, b in reversed(list(blobs.items()))}
    assert torch.equal(_launch(x, eidx, w, blobs), _launch(x, eidx, w, moved))


@cuda
def test_a_zero_weight_duplicate_slot_changes_nothing():
    """truncate_top_k re-points a dropped slot at the row's top-1 expert with weight 0: the
    same expert twice in one token."""
    x, eidx, w, blobs = _case(*QWEN)
    eidx2 = torch.cat([eidx, eidx[:, :1]], dim=1)
    w2 = torch.cat([w, torch.zeros_like(w[:, :1])], dim=1)
    assert torch.equal(_launch(x, eidx, w, blobs), _launch(x, eidx2, w2, blobs))


@cuda
def test_one_token():
    x, eidx, w, blobs = _case(*QWEN, tokens=1)
    assert _rel(_launch(x, eidx, w, blobs),
                _stock(x, eidx, w, blobs, linear=pk._linear_fp32)) <= 2e-3


@cuda
def test_a_nan_token_poisons_its_own_row_and_no_other():
    """Stock propagates a NaN; rounding must not turn it into a zero, and padding rows
    (which read token 0) must not carry it anywhere."""
    x, eidx, w, blobs = _case(*QWEN)
    clean = _launch(x, eidx, w, blobs)
    x = x.clone()
    x[0, 5] = float("nan")
    got = _launch(x, eidx, w, blobs)
    assert torch.isnan(got[0]).all()
    assert torch.equal(got[1:], clean[1:])


@cuda
def test_a_strided_view_of_the_activations_is_read_as_what_it_holds():
    """The kernels index raw memory. Every other row of a wider batch is a legal
    `[tokens, hidden]` tensor whose rows are not adjacent."""
    x, eidx, w, blobs = _case(*QWEN)
    wide = torch.zeros(2 * x.shape[0], x.shape[1], dtype=BF, device="cuda")
    wide[::2] = x
    view = wide[::2]
    assert not view.is_contiguous(), "premise"
    assert torch.equal(_launch(view, eidx, w, blobs), _launch(x, eidx, w, blobs))


@cuda
def test_the_blobs_may_be_freed_as_soon_as_the_launch_returns():
    """A prefill miss that nothing retains is dropped when the forward returns, with the
    kernels still queued. Their memory may be handed out again at once; stream order must
    keep the kernels ahead of whatever is written there next."""
    x, eidx, w, blobs = _case(*QWEN)
    expected = _launch(x, eidx, w, blobs)
    mine = {e: {"flat": {n: t.clone() for n, t in b["flat"].items()}, "geom": b["geom"]}
            for e, b in blobs.items()}
    sizes = [(t.shape, t.dtype) for b in mine.values() for t in b["flat"].values()]
    out = torch.empty(x.shape, device="cuda")
    pk._prefill_launch(x, eidx, w, mine, out)
    del mine
    junk = [torch.zeros(shape, dtype=dtype, device="cuda") for shape, dtype in sizes]
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert len(junk) == len(sizes)


@cuda
def test_two_geometries_get_two_verdicts(monkeypatch):
    """A mixed-precision model: each layer geometry is checked on first sight, and each
    is right."""
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    for geom in (QWEN, (1024, 512, 8, 64)):
        x, eidx, w, blobs = _case(*geom)
        assert _rel(_prefill(x, eidx, w, blobs),
                    _stock(x, eidx, w, blobs, linear=pk._linear_fp32)) <= 2e-3
    assert len(pk._PF_VERDICTS) == 2 and all(pk._PF_VERDICTS.values())


@cuda
def test_back_to_back_calls_do_not_share_a_table():
    """More unsynchronised calls than the ring has slots, of different table lengths."""
    cases = [_case(*QWEN, tokens=5 + 3 * s, seed=s) for s in range(3 * len(pk._PF_TABLES._slots))]
    outs = []
    for x, eidx, w, blobs in cases:
        out = torch.empty(x.shape, device="cuda")
        pk._prefill_launch(x, eidx, w, blobs, out)
        outs.append(out)
    torch.cuda.synchronize()
    for (x, eidx, w, blobs), out in zip(cases, outs):
        assert torch.equal(out, _launch(x, eidx, w, blobs))


@cuda
def test_prefill_uses_the_kernels_when_they_agree(monkeypatch):
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    x, eidx, w, blobs = _case(*QWEN)
    assert torch.equal(_prefill(x, eidx, w, blobs), _launch(x, eidx, w, blobs))
    assert list(pk._PF_VERDICTS.values()) == [True]


@cuda
def test_the_verdict_is_taken_once_per_geometry(monkeypatch):
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    calls = []
    real = pk._prefill_verdict
    monkeypatch.setattr(pk, "_prefill_verdict", lambda *a: calls.append(1) or real(*a))
    x, eidx, w, blobs = _case(*QWEN)
    for _ in range(3):
        _prefill(x, eidx, w, blobs)
    assert len(calls) == 1


@cuda
def test_a_wrong_kernel_is_refused_loudly(monkeypatch, capsys):
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    monkeypatch.setattr(pk, "_prefill_launch", lambda x, e, w, b, out, plan=None: out.zero_())
    x, eidx, w, blobs = _case(*QWEN)
    got = _prefill(x, eidx, w, blobs)
    assert "WARNING" in capsys.readouterr().out
    assert torch.equal(got, _stock(x, eidx, w, blobs))


@cuda
def test_a_geometry_with_no_tensor_core_tile_stays_on_the_stock_loop(monkeypatch, capsys):
    """Hidden 64 at int4 leaves an 8-word K tile: under the smallest dot the GPU takes."""
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    x, eidx, w, blobs = _case(64, 64, 4, 32, n_exp=8, top_k=2)
    got = _prefill(x, eidx, w, blobs)
    assert "WARNING" in capsys.readouterr().out
    assert torch.equal(got, _stock(x, eidx, w, blobs))


@cuda
def test_a_call_the_kernels_cannot_take_runs_the_stock_arithmetic(monkeypatch, capsys):
    """A dense entry, blobs of two geometries, or a routed expert with no entry at all (a
    missing role -- the loop skips it): the whole call runs the stock arithmetic, exactly,
    and without a WARNING."""
    monkeypatch.setattr(pk, "_PF_VERDICTS", {})
    x, eidx, w, blobs = _case(*QWEN)
    first = int(eidx[0, 0])
    dense = {"dense": dict(zip(("gate", "down", "up"),
                               expert_weights_from_entry(blobs[first], BF)))}
    other = _blobs(2048, 512, 4, 64, 1)[0]
    missing = {e: b for e, b in blobs.items() if e != first}
    for mixed in ({**blobs, first: dense}, {**blobs, first: other}, missing):
        assert torch.equal(_prefill(x, eidx, w, mixed), _stock(x, eidx, w, mixed))
    assert "WARNING" not in capsys.readouterr().out
    assert pk._PF_VERDICTS == {}, "no verdict is taken on a call the kernels cannot take"


@cuda
def test_the_fallback_is_the_per_slot_loop_arithmetic():
    """EXACT, at the shape where it matters: `stock_experts_prefill` must be the forward's
    own loop -- cuBLAS `F.linear` per expert, contributions summed in expert-id order by
    accumulate_expert_outputs -- replayed here independently, experts visited in REVERSE
    (arrival order must not matter). At hidden 2048 -> intermediate 512 cuBLAS is not
    correctly rounded, so a fallback that summed in fp32 would not match."""
    x, eidx, w, blobs = _case(*QWEN)
    top_k = eidx.shape[1]
    contribs = []
    for e, rows in reversed(_per_expert(eidx, pk._prefill_plan(eidx))):
        w_gate, w_down, w_up = expert_weights_from_entry(blobs[e], BF)
        tok = rows // top_k
        xs = x[tok]
        out = F.linear(F.silu(F.linear(xs, w_gate)) * F.linear(xs, w_up), w_down)
        contribs.append((e, tok, out * w[tok, rows % top_k, None]))
    expected = torch.zeros(x.shape, device="cuda")
    accumulate_expert_outputs(expected, contribs)
    if torch.equal(expected, _stock(x, eidx, w, blobs, linear=pk._linear_fp32)):
        pytest.skip("this cuBLAS build rounds correctly at this shape: nothing to tell apart")
    assert torch.equal(_stock(x, eidx, w, blobs), expected)


def test_tables_group_rows_by_expert_and_order_slots_by_expert_id():
    """Host-only: the layout the kernels read, on a routing small enough to write down."""
    eidx = torch.tensor([[5, 2], [2, 9], [9, 5]])                       # rows 0..5
    order, live, counts = pk._prefill_plan(eidx)
    assert live.tolist() == [2, 5, 9] and counts.tolist() == [2, 2, 2]
    assert order.tolist() == [1, 2, 0, 5, 3, 4]                         # expert 2, 5, 9
    slot_order = torch.argsort(eidx, dim=1, stable=True) + (torch.arange(3) * 2)[:, None]
    assert slot_order.tolist() == [[1, 0], [2, 3], [5, 4]]
