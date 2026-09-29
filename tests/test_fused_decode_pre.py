"""S12 T6: the fused non-EXL3 forward's `decode_pre` -- routing + the shared expert,
everything before the routing sync -- so a decode graph can capture it. It must be the
same arithmetic as the stock path and have no side effect."""
import torch
import torch.nn as nn

import vaayu.decode_graphs as dg
from tests.test_fused_packed_kernel_branch import HIDDEN, _cuda_model, _x, cuda
from tests.test_decode_graphs import ToyLayer, _cache, _decode, _model, _same_outputs


def _with_shared_expert(moe):
    moe.shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False).to("cuda", torch.bfloat16)
    moe.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False).to("cuda", torch.bfloat16)
    return moe


@cuda
def test_the_pre_path_is_bitwise_the_stock_path(tmp_path):
    model, _ref, _ = _cuda_model(tmp_path)
    moe = _with_shared_expert(model.layers[0].mlp)
    fwd, x = moe.forward, _x()
    stock = fwd(x)
    assert torch.equal(fwd(x, _pre=fwd.decode_pre(x)), stock)


@cuda
def test_decode_pre_has_no_side_effects(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    fwd, x = model.layers[0].mlp.forward, _x()
    fwd(x)                                          # warm the tiers
    history = dict(ref._expert_history)
    vram = ref._expert_vram_cache
    before = (len(vram.cache), vram.hits, vram.misses)
    fwd.decode_pre(x)
    assert ref._expert_history == history
    assert (len(vram.cache), vram.hits, vram.misses) == before


@cuda
def test_decode_pre_is_none_on_a_fully_pinned_layer(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    ref.pinned_layer_names = {ref._moe_blocks[0].layer_name}
    assert model.layers[0].mlp.forward.decode_pre(_x()) is None


@cuda
def test_routing_runs_inside_the_graph(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    moe = _with_shared_expert(model.layers[0].mlp)
    moe._vaayu_toy_moe = True                       # the toy _model() lists it as a MoE block
    torch.manual_seed(0)
    layer = ToyLayer(0, "attn", moe).to("cuda", torch.bfloat16)
    stock = _decode([layer], _cache(1))
    router = getattr(moe, ref._moe_blocks[0].router_attr)
    calls = []
    real = router.forward
    router.forward = lambda *a, **k: calls.append(1) or real(*a, **k)
    runner = dg.install_decode_graphs(_model([layer]))

    def reset(t):
        if t == 3:                  # warm (1), capture (2) and the checked replay (3) are done
            calls.clear()

    got = _decode([layer], _cache(1), runner, before_token=reset)
    assert _same_outputs(stock, got)
    assert "l0" not in runner._refused
    assert calls == [], "a replay ran the router in Python: routing is not inside the graph"
