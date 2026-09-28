"""`use_torch_compile=True` is REFUSED at load -- warn and disable -- on every device.

On CUDA it crashes (todo.md row 48 / S10). The compiled layer callable is built
with `mode="reduce-overhead"`, which is torch.compile plus inductor's cudagraph
trees; those assume one model invocation per step, and the engine replays that
one callable once per LAYER with no step boundary. The next layer's replay
overwrites the previous layer's output before it is read:
`RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run`, on the first generation. Reproduced on
Qwen3-Next (2026-09-20, twice) and on the dense Llama-3.2-1B (2026-09-28).
Without cudagraphs it runs, but measured 24-35 % slower on Qwen3-Next (row 36).

The decision is a pure function so it is testable without loading a model.
"""
import inspect

import pytest

from vaayu.vaayu_base import VaayuBaseModel, resolve_use_torch_compile


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_a_cuda_request_is_refused_with_a_warning_naming_the_crash(device, capsys):
    # Fails if the CUDA branch ever re-enables compile without a step-boundary fix.
    assert resolve_use_torch_compile(True, device) is False
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "DISABLED" in out
    assert "CUDAGraphs" in out
    assert "row 48" in out


def test_a_cpu_request_stays_refused(capsys):
    assert resolve_use_torch_compile(True, "cpu") is False
    assert "Disabling" in capsys.readouterr().out


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_not_requesting_it_is_off_and_silent(device, capsys):
    assert resolve_use_torch_compile(False, device) is False
    assert capsys.readouterr().out == ""


def test_the_engine_resolves_the_flag_through_the_refusal():
    # §8 "is it plumbed?": the constructor must take its answer from the helper,
    # not keep a private copy of the old CUDA branch that printed "enabled".
    src = inspect.getsource(VaayuBaseModel.__init__)
    assert "self.use_torch_compile = resolve_use_torch_compile(use_torch_compile, device)" in src
    assert "torch.compile enabled" not in src
