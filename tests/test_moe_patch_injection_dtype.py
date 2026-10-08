"""A5.7c: the module-list expert injector REINTERPRETS packed dtypes.

`_inject_expert_weights` cast every streamed tensor to the running dtype
(`moe_patch.py:1208`). Two things break under that cast, and only the first one
raises:

1. a packed-fp4 expert weight (DeepSeek-V4-Flash ships `I8`, two fp4 values per
   byte; the skeleton declares `torch.float4_e2m1fn_x2`) reaches
   `DeepseekV4Linear.forward` with a dtype not in its `_QUANTIZED` tuple, takes
   the unquantized branch and dies in `F.linear` on the packed K axis;
2. that weight's companion `scale` is declared `torch.float8_e8m0fnu`, and
   `forward` reads the quantization format back off `scale.dtype`
   (`scale_fmt = "ue8m0" if scale.dtype == _SCALE_DTYPE else None`). `e8m0 -> bf16`
   SUCCEEDS SILENTLY, so fixing only the weight buys a crash-free WRONG token.

Why the non-expert path needed no e8m0 handling: it injects through
accelerate's `set_module_tensor_to_device`, which coerces the value back to the
declared parameter dtype (and bf16 round-trips e8m0 and e4m3 losslessly). This
injector uses `setattr(submod, base, nn.Parameter(t))` and gets no coercion.

Everything here runs on CPU with hand-built modules: what is under test is the
DECISION and the bytes, not a transfer. The non-allowlisted cases assert
byte-identity with today's behaviour, which is Qwen3-Next's and MiniMax-M2.7's
only protection at unit level (CLAUDE.md §0 — this is the forward MiniMax runs
in production).
"""
import pytest
import torch
import torch.nn as nn

import vaayu.moe_patch as mp

LAYER = "model.layers.0"
PREFIX = "model.layers.0.block_sparse_moe.experts.3."   # matches _EXPERT_PREFIX_RE
EIDX = 3
FP4 = torch.float4_e2m1fn_x2
E8M0 = torch.float8_e8m0fnu


def _expert(**submodules):
    """A module-list expert whose submodules declare the dtypes under test.

    `_expert(w1={"weight": torch.zeros(4, 2, dtype=FP4), "scale": None})` builds
    a root module with a `w1` child carrying a declared-fp4 `weight` Parameter
    and a `scale` registered as None (the shape `DeepseekV4Linear` gives a bf16
    weight). A base named in `raw` but absent here is the "no target attribute"
    case, which is what an EXL3-swapped `ExLlamaV3Linear` looks like to
    `getattr` (it has no `weight` of its own).
    """
    root = nn.Module()
    for name, bases in submodules.items():
        sub = nn.Module()
        for base, declared in bases.items():
            if declared is None:
                sub.register_parameter(base, None)
            else:
                sub.register_parameter(
                    base, nn.Parameter(declared, requires_grad=False))
        setattr(root, name, sub)
    return root


def _inject(mod, raw, *, pinned_source=False):
    return mp._inject_expert_weights(mod, raw, EIDX, LAYER, "cpu",
                                     torch.bfloat16, pinned_source=pinned_source)


def _bytes(t):
    """The byte pattern behind a tensor of ANY dtype, including the packed and
    opaque ones that have no comparison kernels."""
    return t.view(torch.uint8)


def test_the_allowlist_is_populated_on_this_torch():
    """The tuple is built with `hasattr` guards because `moe_patch` is shared
    infrastructure every model imports — a torch without these dtypes must
    degrade to today's behaviour, not fail at import. On THIS torch both exist,
    so a silently empty tuple (which would make every other test here vacuous)
    is itself a failure."""
    assert FP4 in mp.NO_CAST_INJECTION_DTYPES
    assert E8M0 in mp.NO_CAST_INJECTION_DTYPES
    assert torch.float8_e4m3fn not in mp.NO_CAST_INJECTION_DTYPES


@pytest.mark.parametrize("dtype,shape", [(FP4, (4, 2)), (FP4, (8, 16)),
                                         (E8M0, (4, 2)), (E8M0, (6, 3))])
def test_an_allowlisted_target_gets_the_bytes_reinterpreted(dtype, shape):
    """Two contrasting shapes per dtype (CLAUDE.md §5): a fix that worked by
    accident on one geometry fails on the other."""
    mod = _expert(w1={"weight": torch.zeros(shape, dtype=dtype)})
    src = torch.randint(1, 250, shape, dtype=torch.uint8)
    if dtype is FP4:
        src = src.view(torch.int8)

    entry = _inject(mod, {PREFIX + "w1.weight": src.clone()})

    assert entry is not None
    got = entry["w1"]["weight"]
    assert got.dtype == dtype
    assert tuple(got.shape) == shape
    assert torch.equal(_bytes(got), _bytes(src))
    assert mod.w1.weight.dtype == dtype


def test_an_e8m0_scale_is_not_flattened_to_the_running_dtype():
    """Defect 2, the one that does not raise: a bf16 scale turns `scale_fmt`
    into None inside `DeepseekV4Linear.forward` and hands TileLang's fp4_gemm a
    bf16 scale."""
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4),
                      "scale": torch.zeros(4, 1, dtype=E8M0)})
    w = torch.randint(-120, 120, (4, 2), dtype=torch.int8)
    s = torch.randint(120, 136, (4, 1), dtype=torch.uint8).view(E8M0)

    entry = _inject(mod, {PREFIX + "w1.weight": w.clone(),
                          PREFIX + "w1.scale": s.clone()})

    assert entry["w1"]["weight"].dtype == FP4
    assert entry["w1"]["scale"].dtype == E8M0
    assert torch.equal(_bytes(entry["w1"]["scale"]), _bytes(s))


def test_a_non_allowlisted_target_still_casts_to_the_running_dtype():
    """The MiniMax/Qwen non-regression: unchanged, byte-for-byte."""
    mod = _expert(w2={"weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                      "bias": torch.zeros(2, dtype=torch.bfloat16)})
    w = torch.tensor([[0.5, -1.5], [2.0, -3.0]], dtype=torch.float32)
    b = torch.tensor([0.25, -0.75], dtype=torch.float32)

    entry = _inject(mod, {PREFIX + "w2.weight": w.clone(),
                          PREFIX + "w2.bias": b.clone()})

    assert entry["w2"]["weight"].dtype == torch.bfloat16
    assert torch.equal(entry["w2"]["weight"], w.to(torch.bfloat16))
    assert entry["w2"]["bias"].dtype == torch.bfloat16
    assert torch.equal(entry["w2"]["bias"], b.to(torch.bfloat16))


def test_a_submodule_with_no_such_attribute_takes_the_cast_path():
    """What an EXL3-swapped `ExLlamaV3Linear` looks like to `getattr`: no
    `weight` of its own, so there is no declared dtype to read and the tensor
    must take exactly the path it took before this change."""
    mod = _expert(w3={})

    entry = _inject(mod, {PREFIX + "w3.weight":
                          torch.ones(2, 2, dtype=torch.float32)})

    assert entry["w3"]["weight"].dtype == torch.bfloat16


def test_a_none_declared_parameter_takes_the_cast_path():
    """`DeepseekV4Linear` registers `scale` as None for a bf16 weight; reading
    `.dtype` off None would be an AttributeError inside the hot path."""
    mod = _expert(w1={"weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                      "scale": None})

    entry = _inject(mod, {PREFIX + "w1.scale":
                          torch.ones(2, 2, dtype=torch.float32)})

    assert entry["w1"]["scale"].dtype == torch.bfloat16


def test_an_element_size_mismatch_raises_instead_of_rescaling_the_last_dim():
    """`torch.empty(4, 2, dtype=float32).view(torch.int8)` returns a (4, 8)
    tensor — `.view` does NOT refuse a differing element size, it silently
    rescales the last dimension. Without this guard a wrong source dtype would
    reach the kernel as a wrongly-shaped weight."""
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})

    with pytest.raises(mp.PackedInjectionError) as excinfo:
        _inject(mod, {PREFIX + "w1.weight": torch.zeros(4, 2,
                                                        dtype=torch.float32)})

    msg = str(excinfo.value)
    assert "float32" in msg and "float4_e2m1fn_x2" in msg
    assert "w1.weight" in msg and LAYER in msg and "expert 3" in msg


def test_a_shape_mismatch_against_an_allowlisted_target_raises():
    """The checkpoint disagreeing with the skeleton is a load-time bug; the
    alternative is a confusing GEMM shape error 43 layers later."""
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})

    with pytest.raises(mp.PackedInjectionError) as excinfo:
        _inject(mod, {PREFIX + "w1.weight": torch.zeros(4, 4,
                                                        dtype=torch.int8)})

    assert "(4, 4)" in str(excinfo.value) and "(4, 2)" in str(excinfo.value)


def test_a_structural_failure_still_warns_and_returns_none():
    """The OTHER failure mode is deliberately unchanged: only the new branch
    raises. Changing this one is a behaviour change on two production forwards
    and has its own todo.md row."""
    mod = _expert(w1={"weight": torch.zeros(2, 2, dtype=torch.bfloat16)})

    assert _inject(mod, {PREFIX + "nope.weight":
                         torch.zeros(2, 2, dtype=torch.float32)}) is None


def test_the_declared_dtype_survives_a_detach_and_reinject_cycle():
    """`_load_inject_run`'s `finally` calls `_detach_expert`, which does
    `expert_mod.to("meta")`; the VRAM cache re-binds entries through
    `_attach_cached_expert`. Reading the target dtype off the LIVE module is
    only safe because that cycle is a fixpoint — this is the test of that."""
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})
    src = torch.randint(-120, 120, (4, 2), dtype=torch.int8)

    _inject(mod, {PREFIX + "w1.weight": src.clone()})
    mod.to("meta")
    assert mod.w1.weight.dtype == FP4

    entry = _inject(mod, {PREFIX + "w1.weight": src.clone()})

    assert entry["w1"]["weight"].dtype == FP4
    assert torch.equal(_bytes(entry["w1"]["weight"]), _bytes(src))


def test_the_reinterpret_path_still_permits_an_async_copy(monkeypatch):
    """Row 49: slab-backed experts upload async. A cast forces the copy
    synchronous (`_copy_async_ok` refuses a dtype change, because the host cast
    makes the real source a pageable temp). Reinterpreting carries no cast, so
    `dtype=None` must reach `_copy_async_ok` and async must stay available —
    this branch is on DeepSeek-V4's per-token path (since S5-M gave its
    non-EXL3 module-list experts the slab arena), so losing it would be a
    silent perf regression."""
    seen = []
    real = mp._copy_async_ok

    def spy(pinned, tensor, dtype):
        seen.append(dtype)
        return real(pinned, tensor, dtype)

    monkeypatch.setattr(mp, "_copy_async_ok", spy)
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})

    _inject(mod, {PREFIX + "w1.weight": torch.zeros(4, 2, dtype=torch.int8)},
            pinned_source=True)

    assert seen == [None]


def test_a_same_width_wrong_format_source_is_refused_not_reinterpreted():
    """I2: `float8_e4m3fn` and `float8_e8m0fnu` are both 1 B/elem, so
    `.view` between them SUCCEEDS on this torch — verified directly:
    `torch.zeros(4, 2, dtype=torch.float8_e4m3fn).view(torch.float8_e8m0fnu)`
    does not raise. This checkpoint ships both formats (routed-expert scales as
    e8m0, shared-expert/attention weights as e4m3), so the element-size and
    shape guards alone would silently read a mislabelled e4m3 scale as ue8m0
    exponents. The source dtype must be refused before either of those guards
    runs."""
    mod = _expert(w1={"scale": torch.zeros(4, 1, dtype=E8M0)})

    with pytest.raises(mp.PackedInjectionError) as excinfo:
        _inject(mod, {PREFIX + "w1.scale":
                     torch.zeros(4, 1, dtype=torch.float8_e4m3fn)})

    msg = str(excinfo.value)
    assert "float8_e4m3fn" in msg and "float8_e8m0fnu" in msg


@pytest.mark.parametrize("src_dtype", [torch.uint8, torch.int8])
def test_raw_byte_container_sources_are_still_accepted_for_fp4(src_dtype):
    """I2 must not break the fp4 path: uint8 and int8 are the raw byte
    containers a packed fp4 checkpoint actually ships, and both must still
    reach the reinterpret."""
    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})
    src = torch.zeros(4, 2, dtype=src_dtype)

    entry = _inject(mod, {PREFIX + "w1.weight": src.clone()})

    assert entry is not None
    assert entry["w1"]["weight"].dtype == FP4


def test_a_source_already_in_the_target_dtype_is_still_accepted():
    """The e8m0 scale case: a tensor already declared `float8_e8m0fnu` (not a
    raw byte container) must still pass — I2's guard allows the target dtype
    itself, not only uint8/int8."""
    mod = _expert(w1={"scale": torch.zeros(4, 1, dtype=E8M0)})
    src = torch.randint(120, 136, (4, 1), dtype=torch.uint8).view(E8M0)

    entry = _inject(mod, {PREFIX + "w1.scale": src.clone()})

    assert entry is not None
    assert entry["w1"]["scale"].dtype == E8M0
    assert torch.equal(_bytes(entry["w1"]["scale"]), _bytes(src))


def test_a_cuda_oom_during_the_device_copy_is_not_relabelled(monkeypatch):
    """I1: `_inject_no_cast`'s blanket `except Exception` used to catch
    `torch.cuda.OutOfMemoryError` (a RuntimeError subclass) off the H2D copy
    and relabel it as `PackedInjectionError` — hiding it from
    `vaayu_base.py`'s isinstance-based OOM handler and stating the opposite of
    the truth (a transient VRAM shortage is not a packed-dtype defect). The OOM
    must propagate as itself out of `_inject_no_cast`, where
    `_inject_expert_weights`'s existing (unchanged, out of this fix's scope)
    per-group `except Exception` warns and returns None — the same
    warn-and-skip behaviour the pre-branch code had for a cast-side OOM."""

    class _OOMOnTo:
        """Stands in for a real tensor: `.to(device, ...)` raises OOM, exactly
        like a real H2D copy would when VRAM is exhausted."""
        dtype = FP4
        shape = (4, 2)

        def element_size(self):
            return FP4.itemsize

        def to(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("out of memory")

    mod = _expert(w1={"weight": torch.zeros(4, 2, dtype=FP4)})
    raw = {PREFIX + "w1.weight": _OOMOnTo()}

    entry = mp._inject_expert_weights(mod, raw, EIDX, LAYER, "cpu",
                                      torch.bfloat16, pinned_source=False)

    assert entry is None
