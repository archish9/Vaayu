"""S22-A2: `load_expert_into_slab` issues ALL of one expert's byte groups in one
io_uring wave (`_READ_SEGMENTS`) instead of one round trip per group.

CLAUDE.md §5: parametrized over the group layouts of every per-expert MoE
checkpoint on this box (Qwen3-Next 3 groups, MiniMax 3, DeepSeek-V4 2, gpt-oss 6)
plus a single-group expert. Synthetic files, buffered reads, CPU only -- the ring
pool is never initialised here, so the Rust call takes its no-pool fallback; the
wave itself is tested in `cargo test` and `tests/test_rust_read_segments.py`.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("vaayu_rust")  # without the extension there is no wave to test

import vaayu.persist.safetensor_model_persister as smp
from vaayu.persist.expert_offset_index import ExpertTensorMeta

# name -> groups; a group is a run of file-adjacent tensors (leaf, dtype, shape).
# Byte sizes are the real per-group sizes of each checkpoint.
GEOMETRIES = {
    "qwen3next": [
        [("gate.weight_shape", "I64", [2]), ("up.weight_shape", "I64", [2]), ("down.weight_shape", "I64", [2])],
        [("gate.weight_packed", "I32", [131072]), ("up.weight_packed", "I32", [131072]),
         ("down.weight_packed", "I32", [131072])],
        [("gate.weight_scale", "BF16", [32768]), ("up.weight_scale", "BF16", [32768]),
         ("down.weight_scale", "BF16", [32768])],
    ],
    "minimax": [
        [("w1.mcg", "I32", [1]), ("w2.mcg", "I32", [1]), ("w3.mcg", "I32", [1])],
        [("w1.suh", "F16", [3072]), ("w1.svh", "F16", [1536]), ("w2.suh", "F16", [3072]),
         ("w2.svh", "F16", [1536]), ("w3.suh", "F16", [1536]), ("w3.svh", "F16", [3072])],
        [("w1.trellis", "I16", [1474560]), ("w2.trellis", "I16", [1474560]), ("w3.trellis", "I16", [1474560])],
    ],
    "deepseek_v4": [
        [("w1.scale", "U8", [262144]), ("w2.scale", "U8", [262144]), ("w3.scale", "U8", [262144])],
        [("w1.weight", "I8", [4194304]), ("w2.weight", "I8", [4194304]), ("w3.weight", "I8", [4194304])],
    ],
    "gpt_oss": [
        [("gate_up_proj_bias", "BF16", [5760])],
        [("down_proj_bias", "BF16", [2880])],
        [("down_proj_blocks", "U8", [2880, 90, 16])],
        [("down_proj_scales", "U8", [2880, 90])],
        [("gate_up_proj_blocks", "U8", [5760, 90, 16])],
        [("gate_up_proj_scales", "U8", [5760, 90])],
    ],
    # 400 one-byte tensors in one group: its packed span exceeds its raw read, so
    # slab_span != raw_aligned_length and the per-group slab advance is exercised.
    "many_tiny": [
        [(f"t{i:03d}.w", "U8", [1]) for i in range(400)],
        [("bulk.a", "I32", [4096]), ("bulk.b", "F16", [1000])],
    ],
    "single_group": [
        [("gate.weight", "F16", [4096]), ("up.weight", "F16", [1024])],
    ],
}
_ITEMSIZE = {"U8": 1, "I8": 1, "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "I64": 8}
_GAP = 1 << 20      # between groups: far more than _ALIGN, so each group is its own read


def _nbytes(dtype, shape):
    return _ITEMSIZE[dtype] * int(np.prod(shape))


@pytest.fixture(scope="module", params=sorted(GEOMETRIES))
def expert(request, tmp_path_factory):
    """(path, metas, payloads, n_groups): one expert of the named layout in a file.
    Group starts are 8-aligned but NOT 4096-aligned, as in a real shard."""
    rng = np.random.default_rng(len(request.param))
    blob = bytearray()
    metas, payloads = {}, {}
    cursor = 4096 + 24
    for group in GEOMETRIES[request.param]:
        for leaf, dtype, shape in group:
            raw = rng.integers(0, 256, _nbytes(dtype, shape), dtype=np.uint8).tobytes()
            blob.extend(b"\0" * (cursor - len(blob)))
            blob.extend(raw)
            metas[leaf] = ExpertTensorMeta(file_offset=cursor, byte_length=len(raw), shape=shape, dtype=dtype)
            payloads[leaf] = raw
            cursor += len(raw)
        cursor += _GAP + 8
    blob.extend(b"\0" * 8192)       # the last group's 4096-rounded read stays inside the file
    path = tmp_path_factory.mktemp(request.param) / "shard.safetensors"
    path.write_bytes(bytes(blob))
    return str(path), metas, payloads, len(GEOMETRIES[request.param])


def _slab(metas):
    return torch.empty(smp.expert_span_bytes(metas), dtype=torch.uint8)


def _raw(t):
    """The tensor's bytes (numpy has no bfloat16, so go through a uint8 view)."""
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


def _expected_segments(metas, slab):
    segs, off = [], 0
    for group in smp._group_contiguous_ranges(metas):
        aligned_offset, _pad, raw_len, _packed, slab_span = smp._group_layout(group)
        segs.append((aligned_offset, raw_len, slab.data_ptr() + off))
        off += slab_span
    return segs


def test_the_installed_extension_has_the_wave():
    assert smp._READ_SEGMENTS is not None, (
        "stale vaayu_rust: run `maturin develop --release` in vaayu_rust/")


def test_wave_views_equal_the_serial_path_and_the_clone_path(expert, monkeypatch):
    path, metas, payloads, _n = expert
    slab_w, slab_s = _slab(metas), _slab(metas)
    wave = smp.load_expert_into_slab(path, metas, slab_w, allow_direct=False, tensor_align=False)
    monkeypatch.setattr(smp, "_READ_SEGMENTS", None)
    serial = smp.load_expert_into_slab(path, metas, slab_s, allow_direct=False, tensor_align=False)
    sparse = smp.load_expert_sparse(path, metas, use_direct_io=False)
    assert set(wave) == set(metas)
    for key in metas:
        assert wave[key].dtype == serial[key].dtype and wave[key].shape == serial[key].shape
        assert _raw(wave[key]) == payloads[key], key
        assert _raw(serial[key]) == payloads[key], key
        assert _raw(sparse[key]) == payloads[key], key
        # same slab layout on both paths: the view sits at the same offset in its slab
        assert wave[key].data_ptr() - slab_w.data_ptr() == serial[key].data_ptr() - slab_s.data_ptr(), key


def test_one_rust_call_per_expert_with_one_segment_per_group(expert, monkeypatch):
    path, metas, _payloads, n_groups = expert
    slab = _slab(metas)
    calls = []
    real = smp._READ_SEGMENTS

    def spy(file_path, segments, queue_depth, use_direct):
        calls.append((file_path, list(segments), use_direct))
        return real(file_path, segments, queue_depth, use_direct)

    def boom(*a, **kw):
        raise AssertionError("the per-group read must not run when the wave is available")

    monkeypatch.setattr(smp, "_READ_SEGMENTS", spy)
    monkeypatch.setattr(smp.vaayu_rust, "io_uring_read_into", boom)
    smp.load_expert_into_slab(path, metas, slab, allow_direct=False, tensor_align=False)
    assert len(calls) == 1, f"{len(calls)} Rust calls for one expert"
    file_path, segments, use_direct = calls[0]
    assert file_path == path and use_direct is False
    assert len(segments) == n_groups
    assert segments == _expected_segments(metas, slab)


def test_both_paths_count_the_same_bytes_read(expert, monkeypatch):
    path, metas, _payloads, _n = expert
    expected = sum(seg[1] for seg in _expected_segments(metas, _slab(metas)))
    b0 = smp.read_bytes()
    smp.load_expert_into_slab(path, metas, _slab(metas), allow_direct=False, tensor_align=False)
    b1 = smp.read_bytes()
    monkeypatch.setattr(smp, "_READ_SEGMENTS", None)
    smp.load_expert_into_slab(path, metas, _slab(metas), allow_direct=False, tensor_align=False)
    b2 = smp.read_bytes()
    assert b1 - b0 == expected
    assert b2 - b1 == expected


def test_tensor_align_true_never_takes_the_wave(expert, monkeypatch):
    path, metas, payloads, _n = expert

    def boom(*a, **kw):
        raise AssertionError("tensor_align=True must keep the per-group repack path")

    monkeypatch.setattr(smp, "_READ_SEGMENTS", boom)
    views = smp.load_expert_into_slab(path, metas, _slab(metas), allow_direct=False, tensor_align=True)
    for key in metas:
        assert _raw(views[key]) == payloads[key], key


def test_without_the_symbol_the_segments_function_is_never_called(expert, monkeypatch):
    path, metas, payloads, _n = expert

    def boom(*a, **kw):
        raise AssertionError("_READ_SEGMENTS is None: the Rust wave must not be reached")

    monkeypatch.setattr(smp, "_READ_SEGMENTS", None)
    monkeypatch.setattr(smp.vaayu_rust, "io_uring_read_segments", boom, raising=False)
    views = smp.load_expert_into_slab(path, metas, _slab(metas), allow_direct=False, tensor_align=False)
    for key in metas:
        assert _raw(views[key]) == payloads[key], key


def test_an_undersized_slab_raises_before_anything_is_read(expert, monkeypatch):
    path, metas, _payloads, _n = expert
    calls = []
    monkeypatch.setattr(smp, "_READ_SEGMENTS", lambda *a, **kw: calls.append(a))
    b0 = smp.read_bytes()
    small = torch.empty(smp.expert_span_bytes(metas) - 4096, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="exceeds slab"):
        smp.load_expert_into_slab(path, metas, small, allow_direct=False, tensor_align=False)
    assert calls == [], "a read was issued into a slab that cannot hold the expert"
    assert smp.read_bytes() == b0, "bytes were counted for a read that never happened"


@pytest.mark.parametrize("use_iouring,symbol,warns", [
    (True, None, True),             # stale extension
    (True, object(), False),        # current extension
    (False, None, False),           # no io_uring at all: the preadv path, nothing to rebuild
])
def test_stale_extension_warning(monkeypatch, capsys, use_iouring, symbol, warns):
    monkeypatch.setattr(smp, "_USE_IOURING", use_iouring)
    monkeypatch.setattr(smp, "_READ_SEGMENTS", symbol)
    smp._warn_if_stale_extension()
    out = capsys.readouterr().out
    assert ("io_uring_read_segments" in out and "WARNING" in out and "maturin develop --release" in out) is warns
    if not warns:
        assert out == ""


@pytest.mark.parametrize("symbol,warns", [(None, True), (object(), False)])
def test_init_model_io_prints_the_stale_extension_warning(monkeypatch, capsys, tmp_path, symbol, warns):
    monkeypatch.setattr(smp, "_USE_IOURING", True)
    monkeypatch.setattr(smp, "_READ_SEGMENTS", symbol)
    monkeypatch.setattr(smp, "_USE_PINNED", False)          # no buffer-pool allocation
    monkeypatch.setattr(smp, "_RING_INITIALIZED", True)     # no ring init (process-global)
    monkeypatch.setattr(smp, "_FILES_REGISTERED", True)     # no file registration
    monkeypatch.setattr(smp, "_USE_DIRECT_IO", smp._USE_DIRECT_IO)  # restored after the call
    smp.init_model_io(str(tmp_path), use_direct_io=False)
    out = capsys.readouterr().out
    assert ("WARNING" in out and "io_uring_read_segments" in out) is warns


def test_io_uring_off_takes_the_preadv_path_even_with_the_symbol(expert, monkeypatch):
    path, metas, payloads, _n = expert

    def boom(*a, **kw):
        raise AssertionError("_USE_IOURING is False: the Rust wave must not be issued")

    monkeypatch.setattr(smp, "_USE_IOURING", False)
    monkeypatch.setattr(smp, "_READ_SEGMENTS", boom)
    views = smp.load_expert_into_slab(path, metas, _slab(metas), allow_direct=False, tensor_align=False)
    for key in metas:
        assert _raw(views[key]) == payloads[key], key
