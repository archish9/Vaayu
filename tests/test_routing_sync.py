"""S28 C11: a MoE visit's routing reaches the host through ONE device-to-host sync.

Before C11 both per-expert forwards did `expert_indices.flatten().unique().tolist()` --
a device sort (3 cub kernels) plus two syncs (the output's size, then the copy) -- and
the dispatch's own `expert_indices.cpu()` a few lines later: three syncs per visit,
13.44 ms of host per Qwen3-Next decode token (S27). `_routing_to_host` copies the ids
once and finds the routed set on the host. The list must be EXACTLY what the device
`unique` gave (ascending Python ints): the dispatch, the VRAM tier and the expert
history all read it.
"""
import warnings

import pytest
import torch

from vaayu.moe_patch import _routing_to_host, patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import MockModel, MockVaayuRef, _make_checkpoint
from tests.test_fused_packed_kernel_branch import _cuda_model, _x, cuda

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _id_cases():
    g = torch.Generator().manual_seed(0)
    return {
        # One decode token: a handful of distinct ids in routing (not id) order.
        "decode": torch.randperm(64, generator=g)[:10].view(1, 10),
        # A set's iteration order is NOT ascending here ({9, 3, 5} iterates 9, 3, 5).
        "set-order": torch.tensor([[9, 3, 5]]),
        # truncate_top_k(p=...) re-points dropped slots at the row's top-1 expert.
        "top-p duplicates": torch.tensor([[5, 3, 5, 5, 9, 5]]),
        # The module-list forward's (B, S, top_k) shape.
        "3-d": torch.tensor([[[7, 1, 4, 1]]]),
        # A prefill chunk: thousands of ids, every expert many times.
        "prefill": torch.randint(0, 512, (512, 10), generator=g),
    }


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", list(_id_cases()))
def test_the_host_set_is_the_device_unique(device, case):
    ids = _id_cases()[case].to(device)
    cpu, routed = _routing_to_host(ids)
    assert routed == ids.flatten().unique().tolist()
    assert all(type(e) is int for e in routed)
    assert cpu.device.type == "cpu" and torch.equal(cpu, ids.cpu())


def _syncs(fn) -> int:
    """Synchronizing CUDA operations `fn` performs, as torch's sync-debug mode counts them
    (a .cpu() / .tolist() / unique's size read each count 1; an event sync counts 0)."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(1)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            fn()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    return sum("synchronizing" in str(w.message) for w in seen)


def _warm_counts(forward, ref, x):
    """(OFF, ON) syncs of one fully warm visit: every routed expert a VRAM hit, so no
    upload (a pageable one would count) and no first-call verdict."""
    forward(x)
    forward(x)
    assert ref._expert_vram_cache.hits > 0, "premise: the visit is served from the VRAM tier"
    ref._s28_pieces = frozenset()
    off = _syncs(lambda: forward(x))
    ref._s28_pieces = frozenset({"c11"})
    on = _syncs(lambda: forward(x))
    return off, on


@cuda
def test_the_fused_forward_syncs_once_per_decode_visit(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    off, on = _warm_counts(model.layers[0].mlp.forward, ref, _x())
    assert (off, on) == (3, 1)


@cuda
def test_the_module_list_forward_drops_two_syncs(tmp_path):
    torch.manual_seed(0)
    model = MockModel(num_layers=1, hidden=8, num_experts=8)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    model = model.to("cuda")
    ref = MockVaayuRef()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=64 * 1024 * 1024)
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = False
    patch_moe_blocks(model=model, expert_cache=ExpertRAMCache(budget_bytes=64 * 1024 * 1024),
                     offset_index=idx, checkpoint_path=str(tmp_path), top_k=2, device="cuda",
                     dtype=torch.float32, vaayu_ref=ref)
    torch.manual_seed(1)
    x = torch.randn(1, 1, 8, device="cuda")
    off, on = _warm_counts(model.layers[0].block_sparse_moe.forward, ref, x)
    assert on == off - 2 and on >= 1


@cuda
def test_c11_changes_no_output_bit(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    x = _x()
    fwd = model.layers[0].mlp.forward
    off = fwd(x)
    ref._s28_pieces = frozenset({"c11"})
    assert torch.equal(fwd(x), off)
    ref._moe_prefill_pass = True                 # a multi-token prefill call: the CPU unique
    many = _x(tokens=7, seed=3)
    on_pf = fwd(many)
    ref._s28_pieces = frozenset()
    assert torch.equal(fwd(many), on_pf)
