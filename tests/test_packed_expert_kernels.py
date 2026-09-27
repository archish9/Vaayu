"""S7: the fused packed-expert decode kernels (vaayu/packed_expert_kernels.py).

The kernels replace the non-EXL3 fused forward's per-slot loop for ONE token:
dequantize-on-the-fly GEMVs over the checkpoint's own packed blobs, reached
through a pointer table, with every intermediate rounded where the stock path
rounds. They are held to the stock loop's arithmetic (`stock_experts_decode`)
and to an fp64 truth built from the same packed codes, over three contrasting
geometries (CLAUDE.md §5): Qwen3-Next's real one, a wider MiniMax-like g128
one, and an int8 one.
"""
import functools

import pytest
import torch
import torch.nn.functional as F

import vaayu.packed_expert_kernels as pk
from vaayu.moe_patch import _fused_dequant_kernel, build_expert_blob, expert_weights_from_entry
from tests.test_packed_expert_dequant import _reference_pack

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and pk.triton is not None),
                          reason="the kernels are CUDA + triton only")

# (hidden, intermediate, bits, group, top_k)
GEOMS = [
    pytest.param(2048, 512, 4, 32, 10, id="qwen3next-int4-g32"),
    pytest.param(3072, 1536, 4, 128, 8, id="minimax-like-int4-g128"),
    pytest.param(1024, 512, 8, 64, 4, id="int8-g64"),
]


@functools.lru_cache(maxsize=None)
def _blobs(hid, inter, bits, gs, k):
    """k distinct experts, packed exactly as compressed-tensors packs them, as Vaayu's
    VRAM tier stores them (roles in gate, down, up order)."""
    torch.manual_seed(hid * 7 + inter + bits + gs)
    out = []
    for _ in range(k):
        by_role = {}
        for role, shape in (("gate", (inter, hid)), ("down", (hid, inter)), ("up", (inter, hid))):
            w = torch.randn(*shape, dtype=torch.bfloat16) * 0.05
            packed, scale, wshape, _q = _reference_pack(w, bits, gs)
            by_role[role] = {"weight_packed": packed.cuda(), "weight_scale": scale.cuda(),
                             "weight_shape": wshape}
        out.append(build_expert_blob(by_role))
    return out


def _case(hid, inter, bits, gs, k, seed=0, w_dtype=torch.float32):
    """(x, pairs, routing_weights): expert i (ascending id) routed in a shuffled slot."""
    blobs = _blobs(hid, inter, bits, gs, k)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(1, hid, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    w = torch.rand(1, k, generator=g, device="cuda") + 0.05
    w = (w / w.sum(-1, keepdim=True)).to(w_dtype)
    slots = torch.randperm(k, generator=torch.Generator().manual_seed(seed)).tolist()
    return x, [(blobs[i], slots[i]) for i in range(k)], w


def _run(fn, x, pairs, w):
    out = torch.full((1, x.shape[1]), float("nan"), device="cuda")   # must be overwritten
    fn(x, pairs, w, out)
    torch.cuda.synchronize()
    return out


def _stock(x, pairs, w, out):
    pk.stock_experts_decode(x, pairs, w, out, F.silu)


def _decode(x, pairs, w, out):
    pk.packed_experts_decode(x, pairs, w, out, F.silu)


def _fp64(x, pairs, w):
    """The truth: the checkpoint's exact weights, fp64 everywhere."""
    out = torch.zeros(x.shape[1], dtype=torch.float64, device="cuda")
    xf, rw = x[0].double(), w.reshape(-1).double()
    for blob, slot in pairs:
        nb, gs, roles = blob["geom"]
        flat = _fused_dequant_kernel(blob["flat"]["packed"], blob["flat"]["scale"], nb, gs,
                                     torch.float64)
        mats, at = {}, 0
        for role, rows, cols in roles:
            mats[role] = flat[at:at + rows * cols].view(rows, cols)
            at += rows * cols
        g, u = mats["gate"] @ xf, mats["up"] @ xf
        out += (mats["down"] @ (g * torch.sigmoid(g) * u)) * rw[slot]
    return out


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _stage_inputs(x, pairs):
    """The stock loop's own intermediates, expert by expert: h and the down output."""
    hs, outs = [], []
    for blob, _slot in pairs:
        w_gate, w_down, w_up = expert_weights_from_entry(blob, x.dtype)
        h = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
        hs.append(h)
        outs.append(F.linear(h, w_down).float())
    return hs, outs


def _one_pair_down(h, blob, w):
    """The down kernel over ONE pair: with weight 1 its output is the rounded down
    projection itself; with a real weight it is that pair's contribution."""
    geom, sc = blob["geom"], pk._DTYPE_CODE[blob["flat"]["scale"].dtype]
    tbl = pk._TABLES.put(pk._table_values([(blob, 0)]), h.device)
    out = torch.empty(1, pk._layout(geom)["gate"][3], device="cuda")
    pk._down(h, tbl, w, out, 1, geom, sc)
    torch.cuda.synchronize()
    return out


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs,k", GEOMS)
def test_each_rounding_point_matches_the_stock_loop(hid, inter, bits, gs, k):
    """Stage by stage, so fp32 summation-order noise is not amplified by cancellation in
    the sum over experts. Each stage rounds where stock rounds; the only difference left
    is the order inside each dot product. Measured on all three geometries: gate/up stage
    >= 99.95 %, down stage >= 99.98 %; the down stage with its rounding removed measures
    0 %, so the bar separates a missing rounding point from order noise."""
    x, pairs, w = _case(hid, inter, bits, gs, k)
    geom, sc = pairs[0][0]["geom"], pk._DTYPE_CODE[pairs[0][0]["flat"]["scale"].dtype]
    stock_h, stock_out = _stage_inputs(x, pairs)
    tbl = pk._TABLES.put(pk._table_values(pairs), x.device)
    h = pk._gate_up(x, tbl, len(pairs), geom, sc)
    torch.cuda.synchronize()
    h_equal = (h == torch.cat(stock_h)).float().mean().item()
    assert h_equal >= 0.99, f"gate/up/SiLU/product stage: {h_equal:.4%} bit-equal"
    one = torch.ones(1, device="cuda")
    down_equal = min((_one_pair_down(sh, blob, one) == so).float().mean().item()
                     for (blob, _slot), sh, so in zip(pairs, stock_h, stock_out))
    assert down_equal >= 0.99, f"down stage: {down_equal:.4%} bit-equal"


@cuda
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16], ids=["fp32-w", "bf16-w"])
def test_the_expert_sum_runs_in_table_order(w_dtype):
    """EXACT: the layer output is the fp32 sum, in table order, of each pair's own
    contribution -- the order accumulate_expert_outputs sums in (row 44 F6)."""
    x, pairs, w = _case(2048, 512, 4, 32, 10, w_dtype=w_dtype)
    geom, sc = pairs[0][0]["geom"], pk._DTYPE_CODE[pairs[0][0]["flat"]["scale"].dtype]
    tbl = pk._TABLES.put(pk._table_values(pairs), x.device)
    h = pk._gate_up(x, tbl, len(pairs), geom, sc)
    full = torch.empty(1, x.shape[1], device="cuda")
    pk._down(h, tbl, w, full, len(pairs), geom, sc)
    expected = torch.zeros(1, x.shape[1], device="cuda")
    for i, (blob, slot) in enumerate(pairs):
        weight = w.reshape(-1)[slot:slot + 1]
        o = _one_pair_down(h[i:i + 1], blob, torch.ones(1, device="cuda"))
        expected += (o.to(torch.bfloat16) * weight).float()
    torch.cuda.synchronize()
    assert torch.equal(full, expected)


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs,k", GEOMS)
def test_kernel_output_is_as_accurate_as_the_stock_loop(hid, inter, bits, gs, k):
    x, pairs, w = _case(hid, inter, bits, gs, k)
    got, ref = _run(pk._launch, x, pairs, w), _run(_stock, x, pairs, w)
    truth = _fp64(x, pairs, w)
    assert _rel(got, ref) <= 1e-3, f"relL2 vs stock {_rel(got, ref):.2e}"
    e_got, e_ref = _rel(got[0], truth), _rel(ref[0], truth)
    assert e_got <= 1.25 * e_ref + 1e-12, f"fp64 error {e_got:.3e} vs stock's {e_ref:.3e}"


@cuda
def test_bf16_routing_weights_round_like_the_stock_loop():
    """A router returning the activation dtype makes torch ROUND `out * weight`."""
    x, pairs, w = _case(2048, 512, 4, 32, 10, w_dtype=torch.bfloat16)
    got, ref = _run(pk._launch, x, pairs, w), _run(_stock, x, pairs, w)
    assert (got.to(torch.bfloat16) == ref.to(torch.bfloat16)).float().mean().item() >= 0.99
    assert _rel(got, ref) <= 1e-3


@cuda
def test_the_output_does_not_depend_on_where_the_blobs_live():
    """A VRAM hit and a fresh upload hand the kernel different tensors holding the same
    bytes: the result must be bit-identical (row 44 F6)."""
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    clones = [({"flat": {n: t.clone() for n, t in b["flat"].items()}, "geom": b["geom"]}, s)
              for b, s in pairs]
    assert torch.equal(_run(pk._launch, x, pairs, w), _run(pk._launch, x, clones, w))


@cuda
def test_a_zero_weight_duplicate_slot_changes_nothing():
    """truncate_top_k re-points a dropped slot at the row's top-1 expert with weight 0."""
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    k = w.shape[1]
    w2 = torch.cat([w, torch.zeros(1, 1, dtype=w.dtype, device=w.device)], dim=1)
    dup = pairs[:1] + [(pairs[0][0], k)] + pairs[1:]      # same expert, later slot
    assert torch.equal(_run(pk._launch, x, pairs, w), _run(pk._launch, x, dup, w2))


@cuda
def test_back_to_back_calls_do_not_share_a_table():
    """More unsynchronised calls than the ring has slots: each output must still be its
    own table's (a pinned buffer rewritten under an in-flight copy would not be)."""
    cases = [_case(2048, 512, 4, 32, 10, seed=s) for s in range(3 * len(pk._TABLES._slots))]
    outs = []
    for x, pairs, w in cases:
        out = torch.empty(1, x.shape[1], device="cuda")
        pk._launch(x, pairs, w, out)
        outs.append(out)
    torch.cuda.synchronize()
    for (x, pairs, w), out in zip(cases, outs):
        assert torch.equal(out, _run(pk._launch, x, pairs, w))


@cuda
def test_decode_uses_the_kernel_when_it_agrees(monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    assert torch.equal(_run(_decode, x, pairs, w), _run(pk._launch, x, pairs, w))


@cuda
def test_the_verdict_is_taken_once_per_geometry(monkeypatch):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    calls = []
    real = pk._verdict
    monkeypatch.setattr(pk, "_verdict", lambda *a: calls.append(1) or real(*a))
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    for _ in range(3):
        _run(_decode, x, pairs, w)
    assert len(calls) == 1


@cuda
def test_two_geometries_get_two_verdicts(monkeypatch):
    """A mixed-precision model: each layer geometry is checked, and each is right."""
    monkeypatch.setattr(pk, "_VERDICTS", {})
    for geom in ((2048, 512, 4, 32, 10), (1024, 512, 8, 64, 4)):
        x, pairs, w = _case(*geom)
        got, ref = _run(_decode, x, pairs, w), _run(_stock, x, pairs, w)
        assert _rel(got, ref) <= 1e-3
    assert len(pk._VERDICTS) == 2 and all(pk._VERDICTS.values())


@cuda
def test_an_unsupported_group_size_stays_on_the_stock_loop(monkeypatch, capsys):
    """Group 48 packs whole int4 words but is not a power of two: no tile exists."""
    monkeypatch.setattr(pk, "_VERDICTS", {})
    x, pairs, w = _case(192, 96, 4, 48, 2)
    got = _run(_decode, x, pairs, w)
    assert "WARNING" in capsys.readouterr().out
    assert torch.equal(got, _run(_stock, x, pairs, w))


@cuda
def test_a_wrong_kernel_is_refused_loudly(monkeypatch, capsys):
    monkeypatch.setattr(pk, "_VERDICTS", {})
    monkeypatch.setattr(pk, "_launch", lambda x, pairs, w, out: out.zero_())
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    got = _run(_decode, x, pairs, w)
    assert "WARNING" in capsys.readouterr().out
    assert torch.equal(got, _run(_stock, x, pairs, w))


def test_layout_walks_the_blob_streams_in_role_order():
    geom = (4, 32, (("gate", 512, 2048), ("down", 2048, 512), ("up", 512, 2048)))
    lay = pk._layout(geom)
    assert lay["gate"] == (0, 0, 512, 2048)
    assert lay["down"] == (512 * 2048 * 4 // 32, 512 * 2048 // 32, 2048, 512)
    assert lay["up"] == (2 * 512 * 2048 * 4 // 32, 2 * 512 * 2048 // 32, 512, 2048)
