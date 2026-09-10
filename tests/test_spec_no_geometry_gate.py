"""C1: speculation on a MoE model with no expert geometry must be REFUSED.

A MoE checkpoint loaded on a non-CUDA compute device gets no
`_expert_offset_index` (`vaayu_base.py`, built under
`if device.startswith("cuda") and self.is_moe:`), so `_moe_geometry` stays
None -- and every gate that exists to check MoE speculation reads it:

  * the S1a pre-load bound gates on `self._moe_geometry is not None`;
  * the Fix #9 post-load bound gates on that AND `_expert_cache is not None`,
    which is built under the same CUDA guard;
  * the S1d runtime `AdaptiveSpecGate` is never built (`_build_spec_gate()`
    returns None when either is None), and the generation loop reads a missing
    gate as "no restriction".

So before this gate, `use_prompt_lookup=True` or a `drafter_model_path` on a
CPU-compute MoE model ran a speculator that NOTHING checked, silently, for the
life of the process (`speculative_status.calibration` stays null, which is
indistinguishable from a model that never speculated). That is the
CLAUDE.md 5 violation -- warn, never silently degrade -- independent of how
the performance question resolves.

The fix refuses CONSTRUCTION rather than building a gate on CPU: the runtime
gate calibrates off `_expert_history`, which is never written there either
(`patch_moe_blocks` is inside the same CUDA block), so a gate built on that
path would latch `off_no_routing` every time -- a refusal disguised as a
measurement. Brief: docs/issue-C1-cpu-moe-ungated-speculation.md.
"""
import pytest

from vaayu.vaayu_base import spec_no_geometry_refusal

MB = 1024 ** 2

# The two on-device MoE geometries' shape (CLAUDE.md 5: identify the measured
# cases, derive nothing from them). Only presence is under test here.
QWEN = {"expert_bytes": int(11.28 * MB), "num_moe_layers": 94, "top_k": 8,
        "num_experts": 128, "working_set_bytes": 8 * 94 * int(11.28 * MB)}
MINIMAX = {"expert_bytes": int(8.46 * MB), "num_moe_layers": 62, "top_k": 8,
           "num_experts": 256, "working_set_bytes": 8 * 62 * int(8.46 * MB)}


def test_a_moe_model_without_geometry_is_refused():
    reason = spec_no_geometry_refusal(is_moe=True, geometry=None, device="cpu")
    assert reason, "the one configuration no gate can check must be refused here"


def test_the_refusal_names_the_cause_and_the_lever():
    """Unactionable if it does not say WHY and what the user can do."""
    reason = spec_no_geometry_refusal(is_moe=True, geometry=None, device="cpu")
    assert "cpu" in reason
    # The cause is the missing expert geometry, not the device by itself.
    assert "geometry" in reason
    # The lever: run the MoE model on CUDA, where the gates exist.
    assert "cuda" in reason.lower()


@pytest.mark.parametrize("geo", [QWEN, MINIMAX], ids=["qwen-like", "minimax-like"])
def test_a_cuda_moe_model_is_not_refused(geo):
    """The gates that CAN check it exist here -- this one must stand aside.

    MiniMax-M2.7 measures 1.307x with prompt lookup admitted by the runtime
    gate on its own evidence; refusing on a present geometry would delete that.
    """
    assert spec_no_geometry_refusal(is_moe=True, geometry=geo,
                                    device="cuda:0") is None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_a_dense_model_is_never_refused(device):
    """Dense never has an expert geometry and never needed one. A drafter on
    streamed dense is the largest measured win in the engine (5.57x); it must
    not acquire a MoE-shaped refusal on either device."""
    assert spec_no_geometry_refusal(is_moe=False, geometry=None,
                                    device=device) is None
