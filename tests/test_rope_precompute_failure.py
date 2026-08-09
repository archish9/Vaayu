"""A failed RoPE precompute must fail honestly, not return None.

Bug D3. Both precompute paths in `forward` can raise `CUDA out of memory`.
The handler used to print "RoPE will be computed per-layer" and set
`_full_position_embeddings = None` -- but no such per-layer path exists.
`get_pos_emb_args` then returns `{}` and the decoder layer dies with
`MiniMaxM2DecoderLayer.forward() missing 1 required positional argument:
'position_embeddings'`, ~30 frames from the real cause. Two things were wrong:
the message promised a mechanism that does not exist, and a VRAM exhaustion
was laundered into a signature error pointing at the wrong subsystem.

The sentinel was overloaded. `None` legitimately means "this architecture
exposes no top-level rotary_emb, so each decoder layer computes RoPE itself"
-- and for those models `{}` is the correct thing to pass. Failure has to be
distinguishable from that, and the only honest distinction is an exception.
"""
import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel


class _Stub:
    """Minimal stand-in exposing only what _precompute_rope touches."""

    def __init__(self, rotary=None, native=None, manual=None):
        self._rotary = rotary
        self._native = native
        self._manual = manual
        self.running_device = "cpu"
        self.running_dtype = None
        self.config = type("cfg", (), {"head_dim": 8, "hidden_size": 16,
                                       "num_attention_heads": 2})()

    def get_rotary_emb(self):
        return self._rotary

    def _compute_full_position_embeddings(self, position_ids):
        if isinstance(self._manual, Exception):
            raise self._manual
        return self._manual


def _positions(n=8):
    return torch.arange(n).unsqueeze(0)


def _call(stub, position_ids=None):
    return VaayuBaseModel._precompute_rope(stub, position_ids)


def test_no_rotary_emb_returns_none():
    """Architectures that compute RoPE inside each layer. NOT a failure --
    get_pos_emb_args' empty-dict path exists precisely for these."""
    assert _call(_Stub(rotary=None)) is None


def test_manual_fallback_used_when_native_unavailable(monkeypatch):
    """get_rotary_emb() is present but instantiating it fails; the manual
    computation carries the pass."""
    sentinel = ("cos", "sin")
    stub = _Stub(rotary=object(), manual=sentinel)
    assert _call(stub) is sentinel


def test_both_paths_failing_raises_instead_of_returning_none():
    """The D3 case. Must NOT return None -- that is the no-rotary_emb signal
    and would strand the caller with a TypeError 30 frames away."""
    oom = torch.OutOfMemoryError("CUDA out of memory.")
    stub = _Stub(rotary=object(), manual=oom)
    with pytest.raises(RuntimeError) as excinfo:
        _call(stub, _positions())
    assert "out of memory" in str(excinfo.value).lower()


def test_raised_error_chains_the_original_cause():
    """The OOM itself must survive; a message alone loses the traceback."""
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    stub = _Stub(rotary=object(), manual=oom)
    with pytest.raises(RuntimeError) as excinfo:
        _call(stub, _positions())
    assert excinfo.value.__cause__ is oom


def test_error_path_cannot_itself_raise():
    """A diagnostic that dies re-creates the bug it reports. Building the
    message must not depend on position_ids being well-formed."""
    stub = _Stub(rotary=object(), manual=RuntimeError("CUDA out of memory."))
    with pytest.raises(RuntimeError) as excinfo:
        _call(stub, None)
    assert "out of memory" in str(excinfo.value).lower()
