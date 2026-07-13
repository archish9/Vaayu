"""NVMe warm-up range selection — which shard byte ranges to pre-read to prime
the controller, derived from the offset index (model-agnostic)."""
from vaayu.vaayu_base import compute_warmup_ranges


class _FakeMeta:
    def __init__(self, file_offset, byte_length):
        self.file_offset = file_offset
        self.byte_length = byte_length


class _FakeIndex:
    """{layer_name: {expert_idx: {key: meta}}}, insertion-ordered."""
    def __init__(self, num_layers, num_experts, tensor_bytes):
        self.index = {
            f"model.layers.{L}": {
                e: {"trellis": _FakeMeta(1000 + e * tensor_bytes, tensor_bytes)}
                for e in range(num_experts)
            }
            for L in range(num_layers)
        }


def test_warmup_ranges_stops_at_budget():
    idx = _FakeIndex(num_layers=4, num_experts=8, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=2500)
    # accumulate 1000-byte tensors until >= 2500 -> exactly 3 ranges
    assert len(ranges) == 3
    assert sum(length for _, _, length in ranges) >= 2500
    # first range: first expert of the first layer, correct shard path
    shard, off, length = ranges[0]
    assert shard == "/ckpt/model.layers.0.safetensors"
    assert (off, length) == (1000, 1000)


def test_warmup_ranges_bounded_by_available_data():
    # budget larger than the whole (tiny) model — return everything, don't loop
    idx = _FakeIndex(num_layers=1, num_experts=2, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=10 ** 9)
    assert len(ranges) == 2


def test_warmup_ranges_minimax_like_geometry():
    idx = _FakeIndex(num_layers=62, num_experts=256, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=5000)
    assert len(ranges) == 5
    assert all(s.startswith("/ckpt/model.layers.0") for s, _, _ in ranges)
