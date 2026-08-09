"""free_from_gpu must not destroy buffers the checkpoint cannot restore.

Root cause of the MiniMax router crash. `MiniMaxM2SparseMoeBlock` registers
`e_score_correction_bias` with `register_buffer(..., torch.zeros(n))`, so under
`init_empty_weights()` it is created on META. The checkpoint contains no tensor
by that name (verified on NeuroSenko--MiniMax-M2.7-exl3: layer 3 has 3093 keys
and only `block_sparse_moe.gate.weight` for the router), so the ONLY thing that
ever materializes it is the load-time `named_buffers()` pass in
`vaayu_base.py`.

Both loader strategies end `free_from_gpu` with `layer_module.to("meta")`,
which re-metas BUFFERS as well as parameters. The per-layer reload injects only
checkpoint tensors, so the buffer never comes back: the layer's router works on
the FIRST forward pass and dies on every one after with

    RuntimeError: Tensor on device meta is not on the expected device cuda:0!

Parameters SHOULD go to meta -- that is the whole point of streaming, and they
are restored from the checkpoint on the next load. Buffers are tiny (256 floats
for MiniMax's bias) and may be unrestorable, so they must survive.
"""
import pytest
import torch
from torch import nn

from vaayu.persist.loader_strategy import LoaderStrategy


class _Block(nn.Module):
    """Stands in for a decoder layer: a checkpoint-backed weight plus a buffer
    that no checkpoint tensor can restore."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4))


def _strategies():
    """Every concrete strategy that implements free_from_gpu."""
    out = []
    for cls in LoaderStrategy.__subclasses__():
        if "free_from_gpu" in cls.__dict__:
            out.append(cls)
    assert out, "no strategy overrides free_from_gpu -- test is not testing anything"
    return out


@pytest.mark.parametrize("cls", _strategies(), ids=lambda c: c.__name__)
def test_buffers_survive_free_from_gpu(cls):
    block = _Block()
    strat = cls.__new__(cls)          # no __init__: free_from_gpu needs no state
    model = nn.Module()
    model.block = block

    strat.free_from_gpu(model, block, ["block.lin.weight"], hf_quantizer=None)

    assert block.e_score_correction_bias.device.type != "meta", (
        f"{cls.__name__}.free_from_gpu sent a buffer to meta; nothing can "
        f"restore it, so the layer's router dies on the next forward pass")


@pytest.mark.parametrize("cls", _strategies(), ids=lambda c: c.__name__)
def test_parameters_still_go_to_meta(cls):
    """The VRAM-freeing intent must survive the fix -- parameters are the big
    things and the checkpoint restores them on the next load."""
    block = _Block()
    strat = cls.__new__(cls)
    model = nn.Module()
    model.block = block

    strat.free_from_gpu(model, block, ["block.lin.weight"], hf_quantizer=None)

    assert block.lin.weight.device.type == "meta", (
        f"{cls.__name__}.free_from_gpu stopped freeing parameters")
