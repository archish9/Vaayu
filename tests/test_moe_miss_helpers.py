"""S7 T2: the miss-path pieces the per-slot loop and the S7 kernel branch share.

`_settle_miss_upload` is the arena-read / staging-release / unpin / VRAM-put
sequence that used to live inline in the per-slot loop; both paths now call it
so the order cannot drift. `_upload_packed_expert` is the H2D half of
`_expert_dense_weights` (the kernel branch skips the dequant half).
"""
import torch

from vaayu.moe_patch import (
    _group_packed_tensors_by_weight, _packed_roles_complete, _settle_miss_upload,
    _upload_packed_expert, build_expert_blob)
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from tests.test_moe_patch import MockVaayuRef
from tests.test_packed_expert_dequant import _reference_pack


def _raw(hid=64, inter=32, bits=4, gs=32):
    raw = {}
    for seg, shape in (("gate_proj", (inter, hid)), ("up_proj", (inter, hid)),
                       ("down_proj", (hid, inter))):
        packed, scale, wshape, _q = _reference_pack(
            torch.randn(*shape, dtype=torch.bfloat16) * 0.1, bits, gs)
        base = f"layers.0.mlp.experts.3.{seg}"
        raw[f"{base}.weight_packed"] = packed
        raw[f"{base}.weight_scale"] = scale
        raw[f"{base}.weight_shape"] = wshape
    return raw


def test_packed_roles_complete():
    packed = _group_packed_tensors_by_weight(_raw())
    assert _packed_roles_complete(packed)
    del packed["up"]["weight_scale"]
    assert not _packed_roles_complete(packed)


def test_upload_packed_expert_copies_everything_but_the_shape():
    packed = _group_packed_tensors_by_weight(_raw())
    timing = {"h2d": 0.0}
    on_dev = _upload_packed_expert(packed, "cpu", timing)
    for role in ("gate", "down", "up"):
        assert torch.equal(on_dev[role]["weight_packed"], packed[role]["weight_packed"])
        assert on_dev[role]["weight_shape"] is packed[role]["weight_shape"]
    assert timing["h2d"] > 0.0
    blob = build_expert_blob(on_dev)
    assert torch.equal(blob["flat"]["packed"], build_expert_blob(packed)["flat"]["packed"])


def test_settle_miss_upload_unpins_and_puts():
    ram = ExpertRAMCache(budget_bytes=1 << 20)
    vram = ExpertVRAMCache(budget_bytes=1 << 26)
    ram.pin(0, 3)
    pinned = {(0, 3)}
    entry = build_expert_blob(_group_packed_tensors_by_weight(_raw()))
    _settle_miss_upload(3, _raw(), entry, layer_idx=0, expert_cache=ram,
                        vaayu_ref=MockVaayuRef(), pinned_keys=pinned, vram=vram, cm=None,
                        cm_last_call=False, cm_budget=0)
    assert pinned == set()
    assert (0, 3) not in ram._pinned
    assert vram.get(0, 3) is entry


def test_settle_miss_upload_with_no_entry_still_unpins():
    ram = ExpertRAMCache(budget_bytes=1 << 20)
    vram = ExpertVRAMCache(budget_bytes=1 << 26)
    ram.pin(0, 3)
    pinned = {(0, 3)}
    _settle_miss_upload(3, _raw(), None, layer_idx=0, expert_cache=ram,
                        vaayu_ref=MockVaayuRef(), pinned_keys=pinned, vram=vram, cm=None,
                        cm_last_call=False, cm_budget=0)
    assert pinned == set() and vram.get(0, 3) is None


def test_settle_miss_upload_fills_chunk_major_within_budget():
    ref = MockVaayuRef()
    ref._chunk_major_gpu_bytes = 0
    cm = {}
    entry = build_expert_blob(_group_packed_tensors_by_weight(_raw()))
    size = ExpertVRAMCache._group_bytes(entry)
    _settle_miss_upload(3, _raw(), entry, layer_idx=0, expert_cache=ExpertRAMCache(1 << 20),
                        vaayu_ref=ref, pinned_keys=set(), vram=None, cm=cm,
                        cm_last_call=False, cm_budget=size)
    assert cm == {3: entry} and ref._chunk_major_gpu_bytes == size
    _settle_miss_upload(4, _raw(), entry, layer_idx=0, expert_cache=ExpertRAMCache(1 << 20),
                        vaayu_ref=ref, pinned_keys=set(), vram=None, cm=cm,
                        cm_last_call=False, cm_budget=size)
    assert 4 not in cm, "over the budget: a later chunk re-reads instead"
