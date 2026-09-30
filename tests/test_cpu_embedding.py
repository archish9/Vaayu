"""S14 E: the embedding table lives on the CPU; only its rows cross PCIe.

Eligibility is a pure function of (module, checkpoint tensors), tested without a model load.
The wrapper must return EXACTLY what the module itself returns -- a scaled subclass still
scales -- because it replaces the module in the layer loop.
"""
import pytest
import torch

from vaayu.cpu_embedding import CpuResidentEmbedding, cpu_embedding_eligibility


class _ScaledEmbedding(torch.nn.Embedding):
    """Gemma-style: the scale lives INSIDE forward, so a raw F.embedding would miss it."""

    def __init__(self, n, d, scale=3.0):
        super().__init__(n, d)
        self.scale = scale

    def forward(self, ids):
        return super().forward(ids) * self.scale


def _meta(cls=torch.nn.Embedding, n=8, d=4, **kw):
    with torch.device("meta"):
        return cls(n, d, **kw)


def _cpu(cls=torch.nn.Embedding, **kw):
    m = cls(8, 4, **kw)
    with torch.no_grad():
        m.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    return m.to(torch.bfloat16)


def test_plain_bf16_embedding_is_eligible():
    ok, why = cpu_embedding_eligibility(_meta(), {"w": torch.zeros(8, 4, dtype=torch.bfloat16)})
    assert ok, why


def test_scaled_subclass_is_eligible():
    ok, why = cpu_embedding_eligibility(_meta(_ScaledEmbedding), {"w": torch.zeros(8, 4)})
    assert ok, why


def test_integer_weight_is_refused_with_its_dtype():
    ok, why = cpu_embedding_eligibility(_meta(), {"w": torch.zeros(8, 4, dtype=torch.int8)})
    assert not ok and "int8" in why


def test_max_norm_is_refused():
    ok, why = cpu_embedding_eligibility(_meta(max_norm=1.0), {"w": torch.zeros(8, 4)})
    assert not ok and "max_norm" in why


def test_a_module_that_is_not_an_embedding_is_refused():
    with torch.device("meta"):
        lin = torch.nn.Linear(4, 8)
    ok, why = cpu_embedding_eligibility(lin, {"w": torch.zeros(8, 4)})
    assert not ok and "Linear" in why


def test_a_shard_with_two_tensors_is_refused():
    ok, why = cpu_embedding_eligibility(_meta(), {"a": torch.zeros(8, 4), "b": torch.zeros(8, 4)})
    assert not ok and "2 tensors" in why


def test_a_shape_mismatch_is_refused():
    ok, why = cpu_embedding_eligibility(_meta(), {"w": torch.zeros(9, 4)})
    assert not ok and "shape" in why


@pytest.mark.parametrize("shape", [(3,), (1, 3), (2, 3)])
def test_wrapper_output_is_bit_equal_to_the_module(shape):
    inner = _cpu()
    ids = torch.randint(0, 8, shape)
    got = CpuResidentEmbedding(inner, "cpu")(ids)
    assert torch.equal(got, inner(ids))


def test_wrapper_keeps_a_scaled_subclass_scaling():
    inner = _cpu(_ScaledEmbedding)
    ids = torch.tensor([[5, 0, 7]])
    got = CpuResidentEmbedding(inner, "cpu")(ids)
    assert torch.equal(got, torch.nn.functional.embedding(ids, inner.weight) * 3.0)


def test_a_deepseek_style_repeat_works_on_the_output():
    """DeepSeek-V4's run_embed does layer(seq).unsqueeze(2).repeat(1, 1, hc_mult, 1)."""
    out = CpuResidentEmbedding(_cpu(), "cpu")(torch.tensor([[1, 2]]))
    rep = out.unsqueeze(2).repeat(1, 1, 4, 1)
    assert rep.shape == (1, 2, 4, 4) and torch.equal(rep[:, :, 3], out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrapper_returns_rows_on_the_target_device_from_device_ids():
    inner = _cpu()
    ids = torch.tensor([[5, 0, 7]], device="cuda")
    got = CpuResidentEmbedding(inner, "cuda:0")(ids)
    assert got.device.type == "cuda"
    assert torch.equal(got.cpu(), inner(ids.cpu()))
    assert inner.weight.device.type == "cpu"
