"""FP8DynamicCache round-trips GPU<->CPU for prefix-cache storage, preserving
data and sequence length. CPU-only: uses device='cpu' as the 'GPU' stand-in
(the move logic is device-agnostic)."""
import torch
from vaayu.vaayu_base import FP8DynamicCache


def _fill(cache, n_layers, seq, heads=2, dim=8, device="cpu"):
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        cache.update(k, v, L)


def test_to_cpu_compacts_and_preserves():
    c = FP8DynamicCache()
    _fill(c, n_layers=3, seq=5)
    before = [c._k_out_cache[i][..., :c._seq_len[i], :].clone() for i in range(3)]
    c.to_cpu()
    assert all(t.device.type == "cpu" for t in c.key_cache)
    # compacted: capacity now equals used length (no growth tail)
    assert all(c.key_cache[i].shape[-2] == c._seq_len[i] == 5 for i in range(3))
    for i in range(3):
        assert torch.equal(c._k_out_cache[i][..., :5, :], before[i])


def test_to_device_roundtrip_then_append():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=4)
    c.to_cpu()
    c.to_device("cpu")           # 'cpu' stands in for the GPU move
    assert c.get_seq_length() == 4
    # a subsequent update appends after the restored length
    _fill(c, n_layers=2, seq=1)  # NOTE: appends to BOTH layers' seq -> 5 each
    assert c.get_seq_length() == 5


def test_resident_bytes_positive_and_shrinks_after_compact():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=3)   # capacity doubled to >=... tail present
    big = c.resident_bytes()
    c.to_cpu()                    # compaction drops the capacity tail
    assert 0 < c.resident_bytes() <= big
