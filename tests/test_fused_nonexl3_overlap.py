"""Row 44 F6: the non-EXL3 fused forward computes while its reads are in flight.

It was the only one of the three MoE forwards that waited for EVERY expert read
of a layer call before computing any expert -- a deliberate choice, because it
accumulated with `index_add_` in ascending expert id and consuming reads in
completion order would have made that order timing-dependent. The cost is a
pure stall: 9.10 s of a 25 s Qwen3-Next 2k prefill sits in that wait while ~19 s
of compute waits behind it (row 44 post-F2 profile).

The other two forwards solved the ordering problem years ago with
`accumulate_expert_outputs`, which sums in expert-id order AFTER the loop, so
arrival order never reaches the arithmetic. This forward now does the same and
consumes through `_iter_expert_loads` like they do.
"""
import threading

import torch

import vaayu.moe_patch as mp
from vaayu.persist import safetensor_model_persister as smp
from tests.test_fused_nonexl3_slab_arena import HIDDEN, _build, _checkpoint


def _x():
    torch.manual_seed(1)
    return torch.randn(1, 4, HIDDEN)


def _delayed_loader(monkeypatch, gate):
    """`gate(expert_key) -> None` runs inside each expert's read."""
    real = smp.load_expert_sparse

    def loader(shard, metas, *a, **k):
        gate(next(iter(metas)))
        return real(shard, metas, *a, **k)
    monkeypatch.setattr(smp, "load_expert_sparse", loader)


def test_compute_starts_before_every_read_has_finished(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    model, ref, _ram = _build(tmp_path, idx, arena=False, prefill=True)

    first_compute = threading.Event()
    seen_keys = []

    def gate(key):
        # The first expert to be read sails through; every later one blocks
        # until the forward has computed something. If the forward waits for
        # all reads first, nothing can ever set the event and this raises.
        seen_keys.append(key)
        if len(seen_keys) > 1 and not first_compute.wait(timeout=10):
            raise RuntimeError(
                "compute did not start until every read had finished -- the "
                "forward is still serialising disk behind compute")

    _delayed_loader(monkeypatch, gate)
    real_weights = mp._expert_dense_weights
    monkeypatch.setattr(mp, "_expert_dense_weights",
                        lambda *a, **k: (first_compute.set(),
                                         real_weights(*a, **k))[1])

    out = model.layers[0].mlp.forward(_x())
    assert out.shape == (1, 4, HIDDEN)
    assert len(seen_keys) > 1, "premise: this layer call reads several experts"


def test_the_result_does_not_depend_on_completion_order(tmp_path, monkeypatch):
    """Experts arrive in disk completion order; the sum must not."""
    idx = _checkpoint(tmp_path)
    model, ref, _ram = _build(tmp_path, idx, arena=False, prefill=True)
    x = _x()

    def run(reverse):
        order = []
        lock = threading.Lock()

        def gate(key):
            with lock:
                order.append(key)
            # Serialise the reads in a chosen order: each expert waits until it
            # is that expert's turn, so one run completes them in the opposite
            # order to the other.
            while True:
                with lock:
                    mine = order.index(key)
                if (mine == 0) != reverse:
                    return
                with lock:
                    if len(order) == len(set(order)) and mine == len(order) - 1:
                        return
                return
        with monkeypatch.context() as m:
            _delayed_loader(m, gate)
            return model.layers[0].mlp.forward(x)

    assert torch.equal(run(False), run(True))
