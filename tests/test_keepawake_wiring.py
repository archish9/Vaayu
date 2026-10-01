"""The keep-awake covers a generation, and only a generation (row 27).

Row 26 measured the effect INSIDE the forward: the win is the host side of each
disk read (1.22 -> 0.34 ms/read) and the slab-acquire wait (69 -> 20 ms/token).
So the spinners must be up for the whole of `generate` -- prefill included, it
is read-heavy too -- and down again the moment it returns, or an idle server
burns E-cores for nothing.

`generate` is the single convergence point: `stream_generate` calls it on a
worker thread, and the server's non-streaming path calls it directly.
"""
import inspect
import os

import pytest

from vaayu.keepawake import detect_topology
from vaayu.vaayu_base import VaayuBaseModel


@pytest.fixture
def hybrid_box():
    topo = detect_topology()
    if not topo.spin_cpus:
        pytest.skip(f"no spin CPU on this box: {topo.warning}")
    return topo


def _bare_model(**flags):
    obj = VaayuBaseModel.__new__(VaayuBaseModel)
    obj.keep_awake = flags.get("keep_awake", "auto")
    obj.pin_engine_threads = flags.get("pin_engine_threads", True)
    return obj


def test_the_keepawake_is_built_once_and_reused(hybrid_box):
    obj = _bare_model()
    first = obj._keepawake_ctx()
    assert obj._keepawake_ctx() is first, (
        "a fresh KeepAwake per call would re-detect the topology every "
        "generation and break the re-entrancy that nesting relies on")


def test_the_model_flags_reach_the_keepawake(hybrid_box):
    with _bare_model(keep_awake=1, pin_engine_threads=False)._keepawake_ctx() as ka:
        assert len(ka.pids) == 1
        assert os.sched_getaffinity(0) != set(hybrid_box.p_cpus) or \
            set(hybrid_box.p_cpus) == set(hybrid_box.all_cpus)


def test_generate_runs_with_the_spinners_up_and_drops_them_after(monkeypatch, hybrid_box):
    """The real proof: what the underlying generate sees while it runs."""
    from transformers.generation.utils import GenerationMixin

    seen = {}

    def _fake_generate(self, *args, **kwargs):
        ka = self._keepawake_obj
        seen["pids"] = list(ka.pids)
        seen["alive"] = [_live(p) for p in ka.pids]
        seen["affinity"] = os.sched_getaffinity(0)
        return [[0]]

    monkeypatch.setattr(GenerationMixin, "generate", _fake_generate)
    obj = _bare_model()
    obj._batch_decode_active = False
    obj._batch_n = 1
    obj._prefix_cache = None
    # S16: the prefix-off path now seeds Vaayu's own KV cache; this test is
    # about the keep-awake, so the stub's cache is a placeholder.
    obj._create_kv_cache = lambda: None
    obj.telemetry = type("T", (), {"dump_json": lambda self, p: None})()
    before = os.sched_getaffinity(0)

    VaayuBaseModel.generate(obj)

    assert seen["pids"] and all(seen["alive"])
    assert seen["affinity"] == set(hybrid_box.p_cpus)
    assert not any(_live(p) for p in seen["pids"])
    assert os.sched_getaffinity(0) == before


def test_batched_generate_holds_the_keepawake_across_its_own_failures(hybrid_box):
    # batched_generate is a generator, so the context opens on the first next()
    # and must close on exhaustion, on .close(), AND when the body raises --
    # its own admission asserts raise routinely.
    class _Recorder:
        def __init__(self):
            self.entered = self.exited = 0
        def __enter__(self):
            self.entered += 1
            return self
        def __exit__(self, *exc):
            self.exited += 1
            return False

    rec = _Recorder()
    obj = _bare_model()
    obj._keepawake_ctx = lambda: rec
    obj._active_attn_impl = "flash_attention_2"      # trips the sdpa assert
    with pytest.raises(AssertionError):
        list(obj.batched_generate([]))
    assert (rec.entered, rec.exited) == (1, 1)


def test_both_knobs_default_to_on(hybrid_box):
    # The shipped default IS the measured configuration (row 27: 21.5 % of
    # Qwen's decode). A default of off would bank the win, not ship it.
    params = inspect.signature(VaayuBaseModel.__init__).parameters
    assert params["keep_awake"].default == "auto"
    assert params["pin_engine_threads"].default is True


def _live(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_the_speculative_path_is_covered_too(hybrid_box):
    """`stream_generate` targets `speculative_generate` whenever a speculator is
    loaded (choose_speculative_target), so a keep-awake wired only into
    `generate` would leave every speculating request -- dense speculative
    decoding measured 5.57x, a shipped feature -- running on an idle package."""
    class _Recorder:
        def __init__(self):
            self.entered = self.exited = 0
        def __enter__(self):
            self.entered += 1
            return self
        def __exit__(self, *exc):
            self.exited += 1
            return False

    rec = _Recorder()
    obj = _bare_model()
    obj._keepawake_ctx = lambda: rec
    obj.speculator = None            # trips speculative_generate's own guard
    with pytest.raises(Exception):
        obj.speculative_generate(input_ids=None, max_new_tokens=1)
    assert (rec.entered, rec.exited) == (1, 1)
