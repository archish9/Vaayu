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


@cuda
def test_a_call_that_is_not_all_one_geometry_runs_the_stock_arithmetic(monkeypatch, capsys):
    """A dense entry, or blobs of two geometries, in one call: no kernel can take it, so
    the whole call runs the stock arithmetic -- exactly, and without a WARNING (final
    review I1 / minor 3)."""
    monkeypatch.setattr(pk, "_VERDICTS", {})
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    blob0 = pairs[0][0]
    dense = {"dense": dict(zip(("gate", "down", "up"),
                               expert_weights_from_entry(blob0, x.dtype)))}
    for mixed in ([(dense, pairs[0][1])] + pairs[1:],
                  pairs[:9] + [(_blobs(2048, 512, 4, 64, 1)[0], pairs[9][1])]):
        got = _run(_decode, x, mixed, w)
        assert torch.equal(got, _run(_stock, x, mixed, w))
    assert "WARNING" not in capsys.readouterr().out


@cuda
def test_a_nan_inside_the_experts_is_not_masked():
    """Stock propagates a NaN; rounding must not turn it into a zero (final review M1)."""
    x, pairs, w = _case(2048, 512, 4, 32, 10)
    x = x.clone()
    x[0, 5] = float("nan")
    got, ref = _run(pk._launch, x, pairs, w), _run(_stock, x, pairs, w)
    assert torch.isnan(ref).all(), "premise: a NaN input poisons every stock output"
    assert torch.isnan(got).all()


# ── S28 C5: the decode kernels as ONE CUDA graph per MoE layer ───────────────────────────────

def fresh_graphs(monkeypatch):
    """Each test starts with no graph, no refusal and zeroed counters."""
    import collections
    for name, fresh in (("_GRAPHS", {}), ("_GRAPH_REFUSED", {}), ("_GRAPH_WARNED", set()),
                        ("_GRAPH_CHURN", collections.Counter()),
                        ("GRAPH_STATS", collections.Counter())):
        monkeypatch.setattr(pk, name, fresh)


def _eager(x, pairs, w, shared, tail):
    """The forward's own eager arithmetic: the kernels, then (tail) its epilogue op for op."""
    out = torch.empty(x.shape, dtype=torch.float32, device="cuda")
    pk._launch(x, pairs, w, out)
    if not tail:
        return out
    result = out.to(x.dtype)
    return result + shared.to(result.dtype) if shared is not None else result


def _shared_out(hid, seed=5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(1, hid, generator=g, device="cuda") * 0.1).to(torch.bfloat16)


QWEN = (2048, 512, 4, 32, 10)


@cuda
@pytest.mark.parametrize("hid,inter,bits,gs,k", GEOMS)
@pytest.mark.parametrize("tail,shared", [(False, False), (True, False), (True, True)],
                         ids=["sum", "tail", "tail+shared"])
def test_the_graph_gives_the_eager_bits(monkeypatch, hid, inter, bits, gs, k, tail, shared):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(hid, inter, bits, gs, k)
    sh = _shared_out(hid) if shared else None
    want = _eager(x, pairs, w, sh, tail)
    first = pk.packed_experts_decode_graphed(0, x, pairs, w, sh, F.silu, tail)    # capture
    assert torch.equal(first, want)
    second = pk.packed_experts_decode_graphed(0, x, pairs, w, sh, F.silu, tail)   # replay
    torch.cuda.synchronize()
    assert torch.equal(second, want)
    assert (pk.GRAPH_STATS["captures"], pk.GRAPH_STATS["replays"]) == (1, 1)


@cuda
def test_a_rewritten_table_computes_the_new_experts(monkeypatch):
    fresh_graphs(monkeypatch)
    hid, inter, bits, gs, k = QWEN
    blobs = _blobs(hid, inter, bits, gs, 2 * k)
    x, _pairs, w = _case(*QWEN)
    a = [(blobs[i], i) for i in range(k)]
    b = [(blobs[k + i], i) for i in range(k)]
    pk.packed_experts_decode_graphed(0, x, a, w, None, F.silu, False)
    got = pk.packed_experts_decode_graphed(0, x, b, w, None, F.silu, False)
    torch.cuda.synchronize()
    assert pk.GRAPH_STATS["replays"] == 1, "premise: the second call replayed the graph"
    assert torch.equal(got, _eager(x, b, w, None, False))


@cuda
def test_a_new_input_tensor_recaptures(monkeypatch):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    pk.packed_experts_decode_graphed(0, x, pairs, w, None, F.silu, False)
    x2 = x * 2                       # another address AND other values; x stays alive
    got = pk.packed_experts_decode_graphed(0, x2, pairs, w, None, F.silu, False)
    assert pk.GRAPH_STATS["captures"] == 2
    assert torch.equal(got, _eager(x2, pairs, w, None, False))


@cuda
def test_inputs_that_move_every_call_refuse_the_layer(monkeypatch, capsys):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    xs = [x * (i + 1) for i in range(pk._GRAPH_CHURN_LIMIT + 2)]
    outs = [pk.packed_experts_decode_graphed(0, xi, pairs, w, None, F.silu, False) for xi in xs]
    assert all(o is not None for o in outs[:-1]) and outs[-1] is None
    assert 0 in pk._GRAPH_REFUSED and "WARNING" in capsys.readouterr().out


@cuda
def test_a_replay_that_disagrees_refuses_the_layer(monkeypatch, capsys):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    assert pk._checked(x, pairs, w, F.silu), "premise: the kernel verdict is taken unpatched"
    want = _eager(x, pairs, w, None, False)
    real = pk._launch
    monkeypatch.setattr(pk, "_launch", lambda x, p, w, out: (real(x, p, w, out), out.add_(1.0)))
    got = pk.packed_experts_decode_graphed(0, x, pairs, w, None, F.silu, False)
    assert torch.equal(got, want + 1.0), "the eager result is returned, never the graph's"
    assert "disagrees bitwise" in pk._GRAPH_REFUSED.get(0, "")
    assert "WARNING" in capsys.readouterr().out
    assert pk.packed_experts_decode_graphed(0, x, pairs, w, None, F.silu, False) is None


@cuda
def test_a_call_the_graph_cannot_take_is_left_to_the_caller(monkeypatch):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    mixed = [({"dense": {}}, pairs[0][1])] + pairs[1:]         # an unpacked expert's entry
    assert pk.packed_experts_decode_graphed(0, x, mixed, w, None, F.silu, False) is None
    assert pk.GRAPH_STATS["captures"] == 0


def _round512(nbytes):
    return -(-nbytes // 512) * 512


@cuda
def test_the_capture_allocates_only_its_static_buffers(monkeypatch):
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    sh = _shared_out(QWEN[0])
    for _ in range(len(pk._TABLES._slots)):    # the eager path's lazy state first: every
        _eager(x, pairs, w, sh, True)          # slot of its table ring,
    assert pk._checked(x, pairs, w, F.silu)    # and the verdict's (it holds ~8 MiB, S7's)
    # The capture stream's first use leaves 1 KiB of one-time lazy state (measured: once
    # per stream, 0 bytes over 20 later captures) -- take it on another layer first.
    pk.packed_experts_decode_graphed(99, x * 3, pairs, w, sh, F.silu, True)
    import gc
    gc.collect()                               # earlier tests' graphs, freed now, not mid-test
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    pk.packed_experts_decode_graphed(0, x, pairs, w, sh, F.silu, True)
    torch.cuda.synchronize()
    g = pk._GRAPHS[0]
    static = sum(_round512(t.numel() * t.element_size())
                 for t in (g.tbl, g.h, g.out, g.tmp, g.result))
    assert torch.cuda.memory_allocated() - m0 == static
    for _ in range(50):
        pk.packed_experts_decode_graphed(0, x, pairs, w, sh, F.silu, True)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - m0 == static
    assert pk.GRAPH_STATS["replays"] == 50
    del g
    pk._GRAPHS.pop(0)
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == m0, "a dropped graph returns every byte"


# ── S28 Review Focus: inputs the graph meets in a long agent session ─────────────────────────

@cuda
def test_per_request_recaptures_do_not_grow_reserved_memory(monkeypatch):
    """S12 recaptures every layer at every request, which hands C5 new static inputs: a
    long agent session recaptures each layer hundreds of times. Reserved VRAM must not
    creep (the first capture's one-time stream state aside)."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    sh = _shared_out(QWEN[0])
    xs = [x * (i + 2) for i in range(2)]
    for _ in range(len(pk._TABLES._slots)):        # the eager reference's table ring, every slot
        _eager(x, pairs, w, sh, True)
    for xi in xs:                                  # first captures: one-time lazy state
        pk.packed_experts_decode_graphed(0, xi, pairs, w, sh, F.silu, True)
        pk.packed_experts_decode_graphed(0, xi, pairs, w, sh, F.silu, True)
    import gc
    gc.collect()
    torch.cuda.synchronize()
    r0, a0 = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    for i in range(100):                           # a "request": recapture, then replay
        xi = xs[i % 2]
        pk.packed_experts_decode_graphed(0, xi, pairs, w, sh, F.silu, True)
        pk.packed_experts_decode_graphed(0, xi, pairs, w, sh, F.silu, True)
    gc.collect()
    torch.cuda.synchronize()
    assert pk.GRAPH_STATS["captures"] == 102 and 0 not in pk._GRAPH_REFUSED
    assert (torch.cuda.memory_reserved(), torch.cuda.memory_allocated()) == (r0, a0)


@cuda
def test_duplicate_expert_slots_through_the_graph(monkeypatch):
    """truncate_top_k(p=...) re-points dropped slots at the top-1 expert with weight 0, so
    one expert fills several table rows."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    dup = sorted([pairs[0]] * 3 + pairs[1:-2], key=lambda bs: id(bs[0]))
    dup = [(blob, slot) for slot, (blob, _s) in enumerate(dup)]
    w = w.clone()
    w[0, 1:3] = 0.0
    want = _eager(x, dup, w, None, False)
    pk.packed_experts_decode_graphed(0, x, dup, w, None, F.silu, False)
    got = pk.packed_experts_decode_graphed(0, x, dup, w, None, F.silu, False)
    torch.cuda.synchronize()
    assert pk.GRAPH_STATS["replays"] == 1 and torch.equal(got, want)


@cuda
@pytest.mark.parametrize("capture_fails", [False, True], ids=["ok", "capture-fails"])
def test_a_capture_inside_an_s12_capture_window_leaves_gc_as_found(monkeypatch, capture_fails):
    """The MoE seam runs this forward between two S12 graph segments, with S12's own
    gc.disable() in force: C5 must hand gc back exactly as it found it."""
    import gc
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    if capture_fails:                              # only the capture passes `h=`
        real = pk._gate_up

        def gate_up(*a, h=None):
            if h is not None:
                raise RuntimeError("boom")
            return real(*a)
        monkeypatch.setattr(pk, "_gate_up", gate_up)
    for enabled in (False, True):
        pk._GRAPHS.clear()
        pk._GRAPH_REFUSED.clear()
        (gc.enable if enabled else gc.disable)()
        try:
            out = pk.packed_experts_decode_graphed(0, x * (2 if enabled else 3), pairs, w, None,
                                                   F.silu, False)
            assert gc.isenabled() == enabled
        finally:
            gc.enable()
        assert out is not None, "a failed capture still returns the eager result"
        assert (0 in pk._GRAPH_REFUSED) == capture_fails


@cuda
def test_a_pair_count_that_changes_now_and_then_is_not_churn(monkeypatch):
    """One token whose routed experts lose a pair (a skipped expert) changes the key; the
    graph recaptures. Graphs that replayed in between are not churn: the layer stays."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    for i in range(6):
        p = pairs if i % 2 == 0 else pairs[:-1]
        pk.packed_experts_decode_graphed(0, x, p, w, None, F.silu, False)
        got = pk.packed_experts_decode_graphed(0, x, p, w, None, F.silu, False)
    torch.cuda.synchronize()
    assert 0 not in pk._GRAPH_REFUSED and pk.GRAPH_STATS["captures"] == 6
    assert torch.equal(got, _eager(x, pairs[:-1], w, None, False))


@cuda
def test_one_token_pair_count_dips_do_not_accumulate_into_a_refusal(monkeypatch):
    """Task 2 review: one token whose pair count dips captures a graph that never replays.
    Such tokens, spread over a long session with replays in between, must not add up to a
    refusal -- churn counts CONSECUTIVE graphs that never replayed."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    for _ in range(pk._GRAPH_CHURN_LIMIT + 3):
        for _ in range(3):
            got = pk.packed_experts_decode_graphed(0, x, pairs, w, None, F.silu, False)
        pk.packed_experts_decode_graphed(0, x, pairs[:-1], w, None, F.silu, False)   # one token
    got = pk.packed_experts_decode_graphed(0, x, pairs, w, None, F.silu, False)
    torch.cuda.synchronize()
    assert 0 not in pk._GRAPH_REFUSED and got is not None
    assert torch.equal(got, _eager(x, pairs, w, None, False))


@cuda
def test_short_requests_between_long_ones_do_not_refuse_the_layer(monkeypatch):
    """Task 2 review: S12 hands every request new static inputs, and a request with a single
    decode step captures a graph that never replays. A few of them among normal requests must
    not refuse the layer."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    xs = [x * (r + 2) for r in range(12)]          # held: every request gets its own address
    for r, xr in enumerate(xs):
        for _ in range(1 if r % 4 == 1 else 5):
            pk.packed_experts_decode_graphed(0, xr, pairs, w, None, F.silu, False)
    assert 0 not in pk._GRAPH_REFUSED
    assert pk.GRAPH_STATS["captures"] == 12, "premise: every request recaptured"


@cuda
def test_churn_counts_only_graphs_that_never_replayed(monkeypatch):
    """Exactly `_GRAPH_CHURN_LIMIT` never-replayed graphs after one that replayed stay within
    the limit: the replayed graph is not churn."""
    fresh_graphs(monkeypatch)
    x, pairs, w = _case(*QWEN)
    xs = [x * (i + 2) for i in range(pk._GRAPH_CHURN_LIMIT + 2)]
    pk.packed_experts_decode_graphed(0, xs[0], pairs, w, None, F.silu, False)
    pk.packed_experts_decode_graphed(0, xs[0], pairs, w, None, F.silu, False)       # replays
    for xi in xs[1:]:                     # LIMIT + 1 captures, each replacing the previous graph
        out = pk.packed_experts_decode_graphed(0, xi, pairs, w, None, F.silu, False)
    assert out is not None and 0 not in pk._GRAPH_REFUSED
