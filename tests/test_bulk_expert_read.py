"""Fix #2: bulk sequential expert read must match per-expert reads byte-for-byte."""
import os
import struct
import tempfile
import torch
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import load_experts_bulk, load_expert_sparse


def _make_file(payload: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return path


def test_bulk_matches_per_expert():
    # 3 experts, each one contiguous 4096-byte tensor, laid out back to back
    data_start = 0
    experts_metas = {}
    payload = bytearray()
    for e in range(3):
        blob = bytes([e + 1]) * 4096
        off = len(payload)
        payload += blob
        experts_metas[e] = {
            f"expert.{e}.w": ExpertTensorMeta(
                file_offset=data_start + off, byte_length=4096,
                shape=[4096], dtype="U8",
            )
        }
    path = _make_file(bytes(payload))
    try:
        bulk = load_experts_bulk(path, experts_metas, use_direct_io=False)
        for e in range(3):
            single = load_expert_sparse(path, experts_metas[e], use_direct_io=False)
            bk = bulk[e][f"expert.{e}.w"]
            sk = single[f"expert.{e}.w"]
            assert torch.equal(bk.cpu().view(-1), sk.cpu().view(-1)), f"mismatch expert {e}"
    finally:
        os.unlink(path)
