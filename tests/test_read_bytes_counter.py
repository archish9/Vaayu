"""The NVMe byte counter behind the token summary's gb_tok= field.

Monotonic by contract: vaayu_base takes a per-TOKEN window over it and the
server takes a per-REQUEST window over the same counter, so nothing may reset
it. These tests pin that contract, because a counter that quietly reset itself
would leave the server's requests.jsonl reporting a fraction of one token as a
whole request's disk traffic.
"""
import threading
from pathlib import Path

import torch
from safetensors.torch import save_file

from vaayu.persist import safetensor_model_persister as smp


def test_note_read_bytes_accumulates():
    before = smp.read_bytes()
    smp.note_read_bytes(4096)
    smp.note_read_bytes(1024)
    assert smp.read_bytes() - before == 5120


def test_counter_is_monotonic_and_has_no_reset():
    before = smp.read_bytes()
    smp.note_read_bytes(512)
    assert smp.read_bytes() >= before + 512
    # The whole design rests on there being no way to zero this. A future
    # helper named like a reset is the regression this asserts against.
    assert not any(n for n in dir(smp)
                   if "read_bytes" in n and "reset" in n), \
        "read_bytes() must stay monotonic -- both consumers window over it"


def test_note_read_bytes_is_thread_safe():
    before = smp.read_bytes()
    def worker():
        for _ in range(500):
            smp.note_read_bytes(1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert smp.read_bytes() - before == 4000


def _make_shard(tmp_path: Path, layer_name: str = "model.layers.0") -> Path:
    tensors = {
        f"{layer_name}.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{layer_name}.block_sparse_moe.gate.weight": torch.randn(4, 8),
        f"{layer_name}.block_sparse_moe.experts.0.w1.weight": torch.ones(8, 8),
        f"{layer_name}.block_sparse_moe.experts.0.w2.weight": torch.ones(8, 8) * 2,
        f"{layer_name}.block_sparse_moe.experts.1.w1.weight": torch.ones(8, 8) * 3,
        f"{layer_name}.block_sparse_moe.experts.1.w2.weight": torch.ones(8, 8) * 4,
    }
    shard = tmp_path / f"{layer_name}.safetensors"
    save_file(tensors, str(shard))
    return shard


def test_load_expert_sparse_counts_its_read(tmp_path):
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex
    layer_name = "model.layers.0"
    shard = _make_shard(tmp_path, layer_name)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    metas = idx.get_expert_offsets(layer_name, 0)

    before = smp.read_bytes()
    smp.load_expert_sparse(str(shard), metas, use_direct_io=False)
    read = smp.read_bytes() - before
    # At least the useful bytes, and no more than one 4096-byte alignment pad
    # per contiguous group (there are at most as many groups as tensors).
    useful = sum(m.byte_length for m in metas.values())
    assert useful <= read <= useful + 4096 * len(metas)


def test_the_count_is_taken_at_the_read_not_memoised(tmp_path):
    """Counting inside the persister rather than at the caller is what makes a
    cache hit contribute zero: every caller-side cache (LoaderStrategy._cache,
    _non_expert_tensor_cache, ExpertRAMCache) returns ABOVE this module, so a
    hit never reaches here at all. The corollary is that two real reads of the
    same expert must count twice -- this function is not memoised."""
    from vaayu.persist.expert_offset_index import ExpertOffsetIndex
    layer_name = "model.layers.0"
    shard = _make_shard(tmp_path, layer_name)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    metas = idx.get_expert_offsets(layer_name, 0)

    first = smp.read_bytes()
    smp.load_expert_sparse(str(shard), metas, use_direct_io=False)
    second = smp.read_bytes()
    smp.load_expert_sparse(str(shard), metas, use_direct_io=False)
    third = smp.read_bytes()
    assert second > first
    assert third - second == second - first, "the same read must count twice"


def test_load_model_counts_its_read(tmp_path):
    layer_name = "model.layers.0"
    _make_shard(tmp_path, layer_name)
    before = smp.read_bytes()
    result = smp.SafetensorModelPersister().load_model(layer_name, str(tmp_path))
    materialised = sum(v.nbytes for v in result.values())
    assert smp.read_bytes() - before >= materialised


def test_the_no_offset_index_fallback_is_counted_too(tmp_path):
    """load_non_expert_tensors returns early to a full-shard read when it has
    no offset index. That is a real read and must count -- it was missed when
    the counter went in per-group, below the early return."""
    layer_name = "model.layers.0"
    _make_shard(tmp_path, layer_name)
    shard = tmp_path / f"{layer_name}.safetensors"
    before = smp.read_bytes()
    result = smp.load_non_expert_tensors(str(shard), layer_name, None)
    assert smp.read_bytes() > before, "the offset_index=None fallback read nothing?"
    assert result, "the fallback returned no tensors -- wrong premise for this test"
