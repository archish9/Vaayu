"""Narrowing activations to fp16 for the EXL3 kernels must not overflow.

Root cause this guards: `ExLlamaV3Linear.forward` did an unguarded
`x.to(torch.float16)`. bf16 carries fp32's exponent range (~3e38); fp16 tops
out at 65504. MiniMax-M2's MoE computes `silu(w1(x)) * w3(x)` in bf16 and hands
the product to `w2` (an ExLlamaV3Linear), so a high-gain expert pushed that
product past 65504, the cast produced `inf`, and the kernel propagated it as
`NaN` — observed as `nan=1, inf=0` at the last MoE layer during prefill, with
no crash and largely coherent output, i.e. silent corruption.

Measured on-device: NaN appeared exactly when the product exceeded 65504
(expert 61 at input scale 171; expert 0, lower gain, only at 342).

A linear operator is homogeneous — f(x/s)*s == f(x) — so the fix rescales
rather than clamping (which would silently corrupt magnitudes).
"""
import pytest
import torch

from vaayu.persist.loader_strategy import fp16_safe_narrow

FP16_MAX = 65504.0


def test_bf16_value_above_fp16_max_does_not_become_inf():
    """The exact regression: a bf16 activation above fp16's ceiling."""
    x = torch.tensor([[1.2e5, 1.0, -3.0e5]], dtype=torch.bfloat16)
    assert torch.isinf(x.to(torch.float16)).any(), "precondition: naive cast overflows"

    x_narrow, scale = fp16_safe_narrow(x)

    assert x_narrow.dtype == torch.float16
    assert not torch.isinf(x_narrow).any()
    assert not torch.isnan(x_narrow).any()
    assert float(scale) > 1.0


def test_rescaling_reconstructs_original_magnitudes():
    """f(x/s)*s == f(x) for a linear f. Verify the round-trip recovers the
    input, which is what makes rescaling correct rather than lossy clamping."""
    x = torch.tensor([[1.2e5, 5.0e4, -3.0e5, 0.0]], dtype=torch.bfloat16)
    x_narrow, scale = fp16_safe_narrow(x)

    recovered = x_narrow.float() * float(scale)
    assert torch.allclose(recovered, x.float(), rtol=2e-2)


def test_normal_range_is_untouched():
    """Values comfortably inside fp16 range must pass through with scale 1.0 —
    byte-identical to the previous behaviour, so Qwen/Mixtral are unaffected."""
    x = torch.randn(8, 32, dtype=torch.bfloat16) * 10.0
    x_narrow, scale = fp16_safe_narrow(x)

    assert float(scale) == 1.0
    assert torch.equal(x_narrow, x.to(torch.float16))


def test_fp16_input_passes_through_unchanged():
    """Already-fp16 input cannot overflow on cast; no work, no scaling."""
    x = torch.randn(4, 16, dtype=torch.float16) * 100.0
    x_narrow, scale = fp16_safe_narrow(x)

    assert float(scale) == 1.0
    assert torch.equal(x_narrow, x)


@pytest.mark.parametrize("magnitude", [1e5, 6.6e4, 1e6, 3e7])
def test_no_overflow_across_magnitudes(magnitude):
    x = torch.full((4, 8), magnitude, dtype=torch.bfloat16)
    x_narrow, scale = fp16_safe_narrow(x)
    assert not torch.isinf(x_narrow).any()
    assert torch.allclose(x_narrow.float() * float(scale),
                          x.float(), rtol=2e-2)


def test_zero_and_empty_are_safe():
    """A degenerate all-zero row must not produce a 0 or NaN scale (div-by-zero)."""
    z = torch.zeros(2, 4, dtype=torch.bfloat16)
    x_narrow, scale = fp16_safe_narrow(z)
    assert float(scale) == 1.0
    assert not torch.isnan(x_narrow).any()

    e = torch.zeros(0, 4, dtype=torch.bfloat16)
    x_narrow, scale = fp16_safe_narrow(e)
    assert x_narrow.numel() == 0
    assert float(scale) == 1.0


def test_scale_is_tensor_not_python_float():
    """Must stay a device tensor: calling .item() here would force a host sync
    on a per-expert, per-token hot path (~350-400 calls/token)."""
    x = torch.full((2, 4), 1e5, dtype=torch.bfloat16)
    _, scale = fp16_safe_narrow(x)
    assert isinstance(scale, torch.Tensor)
