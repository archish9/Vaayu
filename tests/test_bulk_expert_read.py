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


def test_bulk_matches_per_expert_with_gaps_multikey_scrambled_order():
    """Covers three untested axes at once:
    - gaps between experts (padding bytes the bulk span reads but no key claims)
    - multiple tensor keys per expert (mirrors EXL3 gate/up/down trellis tensors)
    - scrambled dict-iteration order, both for experts_metas (expert idx order
      2, 0, 1) and for each expert's sub-dict (keys inserted out of ascending
      file-offset order)

    Each tensor gets a distinct fill byte and a distinct length so a
    copy-paste'd offset/length or an off-by-one would read visibly wrong
    bytes rather than accidentally-correct ones.
    """
    payload = bytearray()

    def _write_blob(fill_value: int, length: int) -> int:
        off = len(payload)
        payload.extend(bytes([fill_value]) * length)
        return off

    # Disk layout (offsets deliberately NOT in expert/key insertion order):
    #   [pad 37B] expert2.gate(64B,0xAA) expert2.up(48B,0xBB)
    #   [pad 19B] expert0.gate(100B,0x11) expert0.down(33B,0x22)
    #   [pad 5B]  expert1.gate(16B,0x77)
    payload.extend(b"\x00" * 37)
    e2_gate_off = _write_blob(0xAA, 64)
    e2_up_off = _write_blob(0xBB, 48)
    payload.extend(b"\x00" * 19)
    e0_gate_off = _write_blob(0x11, 100)
    e0_down_off = _write_blob(0x22, 33)
    payload.extend(b"\x00" * 5)
    e1_gate_off = _write_blob(0x77, 16)

    # experts_metas inserted in scrambled expert-index order (2, 0, 1), and
    # each sub-dict inserted with keys out of ascending file-offset order.
    experts_metas = {
        2: {
            "up": ExpertTensorMeta(file_offset=e2_up_off, byte_length=48, shape=[48], dtype="U8"),
            "gate": ExpertTensorMeta(file_offset=e2_gate_off, byte_length=64, shape=[64], dtype="U8"),
        },
        0: {
            "down": ExpertTensorMeta(file_offset=e0_down_off, byte_length=33, shape=[33], dtype="U8"),
            "gate": ExpertTensorMeta(file_offset=e0_gate_off, byte_length=100, shape=[100], dtype="U8"),
        },
        1: {
            "gate": ExpertTensorMeta(file_offset=e1_gate_off, byte_length=16, shape=[16], dtype="U8"),
        },
    }

    path = _make_file(bytes(payload))
    try:
        bulk = load_experts_bulk(path, experts_metas, use_direct_io=False)
        for eidx, metas in experts_metas.items():
            single = load_expert_sparse(path, metas, use_direct_io=False)
            for key in metas:
                bk = bulk[eidx][key].cpu().view(-1)
                sk = single[key].cpu().view(-1)
                assert torch.equal(bk, sk), f"mismatch expert {eidx} key {key}"

        # Belt-and-suspenders: check the actual fill bytes directly, so a bug
        # that shifted both load_experts_bulk and load_expert_sparse
        # identically (unlikely, but the oracle-comparison alone wouldn't
        # catch it) is still caught.
        assert bulk[2]["gate"].cpu().view(-1).tolist() == [0xAA] * 64
        assert bulk[2]["up"].cpu().view(-1).tolist() == [0xBB] * 48
        assert bulk[0]["gate"].cpu().view(-1).tolist() == [0x11] * 100
        assert bulk[0]["down"].cpu().view(-1).tolist() == [0x22] * 33
        assert bulk[1]["gate"].cpu().view(-1).tolist() == [0x77] * 16
    finally:
        os.unlink(path)


def test_bulk_matches_per_expert_non_u8_dtype():
    """Covers multi-byte dtypes (F16, F32) via ExpertTensorMeta.dtype, with a
    gap between the two experts, confirming .view(dt) reconstructs correctly
    when byte_length is a multiple of the dtype's itemsize (2 and 4 bytes
    respectively) rather than 1.
    """
    payload = bytearray()

    def _write_floats(values, fmt) -> tuple:
        off = len(payload)
        blob = b"".join(struct.pack(fmt, v) for v in values)
        payload.extend(blob)
        return off, len(blob)

    # expert 0: F16 tensor, 4 values x 2 bytes = 8 bytes
    e0_off, e0_len = _write_floats([1.5, -2.25, 3.0, 0.0], "<e")
    payload.extend(b"\xff" * 12)  # gap: unrelated bytes between experts
    # (12, not an arbitrary size: keeps e1's file_offset a multiple of 4,
    # which torch's .view() requires when reinterpreting a byte slice as
    # F32 — real safetensors shards keep tensors packed with this same
    # per-dtype alignment, since each tensor's byte_length is itself a
    # multiple of its own itemsize)
    # expert 1: F32 tensor, 3 values x 4 bytes = 12 bytes
    e1_off, e1_len = _write_floats([100.5, -7.0, 0.125], "<f")

    experts_metas = {
        1: {"w": ExpertTensorMeta(file_offset=e1_off, byte_length=e1_len, shape=[3], dtype="F32")},
        0: {"w": ExpertTensorMeta(file_offset=e0_off, byte_length=e0_len, shape=[4], dtype="F16")},
    }

    path = _make_file(bytes(payload))
    try:
        bulk = load_experts_bulk(path, experts_metas, use_direct_io=False)
        for eidx, metas in experts_metas.items():
            single = load_expert_sparse(path, metas, use_direct_io=False)
            bk = bulk[eidx]["w"].cpu().view(-1)
            sk = single["w"].cpu().view(-1)
            assert torch.equal(bk, sk), f"mismatch expert {eidx}"

        assert bulk[0]["w"].dtype == torch.float16
        assert bulk[1]["w"].dtype == torch.float32
        assert bulk[0]["w"].cpu().tolist() == [1.5, -2.25, 3.0, 0.0]
        assert bulk[1]["w"].cpu().tolist() == [100.5, -7.0, 0.125]
    finally:
        os.unlink(path)
