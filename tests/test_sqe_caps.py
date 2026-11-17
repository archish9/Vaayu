"""SQE-PLUG: the io_uring submission caps that work around CVE-2026-64017.

On 2026-11-08 this box hard-hung mid-run. A kernel blk-mq race oopsed inside
`nvme_queue_rqs`, in one of Vaayu's io_uring reader threads, inside an RCU
read-side critical section -- so the grace period never ended, the dead task
held the io_uring context mutex forever, and the machine had to be
power-cycled. The defect is the KERNEL's and is fixed upstream
(`dc278e9bf2b9`); booting a patched kernel is the real fix, and these caps are
a WORKAROUND kept reachable until it moves.

Both of the race's preconditions are Vaayu's own. The one we can remove is the
block plug: io_uring starts one only when a single `io_uring_enter` submits MORE
than 2 SQEs (`io_submit_state_start`: `need_plug = max_ios > 2`), and the plug's
cached-request list is what the race corrupts.

The caps' kernel-side effect is proved with bpftrace, not here. What these tests
pin is everything between the operator and the ring: that the knob is read, that
env beats config, that it reaches `init_ring_pool`, and that a wheel too old to
take it says so instead of pretending.
"""
import dataclasses

import pytest

from vaayu.persist import safetensor_model_persister as P

ENV = ("VAAYU_MAX_SQES_PER_ENTER", "VAAYU_MAX_CHUNKS_PER_WAVE")


@pytest.fixture
def clean(monkeypatch):
    """No env, and the module defaults back to 'unlimited'."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(P, "_MAX_SQES_PER_ENTER", 0)
    monkeypatch.setattr(P, "_MAX_CHUNKS_PER_WAVE", 0)


def test_uncapped_is_the_default(clean):
    """0 on both must mean 'exactly what the code did before the caps existed'.
    Anything else turns a workaround into an unmeasured perf change for every
    user who never asked for it."""
    assert P._sqe_caps() == (0, 0)


def test_the_setter_is_what_the_server_uses(clean):
    P.set_sqe_caps(2, 2)
    assert P._sqe_caps() == (2, 2)
    P.set_sqe_caps(0, 0)
    assert P._sqe_caps() == (0, 0)


def test_the_setter_refuses_negative_caps(clean):
    """0 means UNLIMITED, so clamping a typo like -1 to 0 would switch the
    mitigation OFF on a box that had asked for it. Ignore, never clamp."""
    P.set_sqe_caps(2, 2)
    P.set_sqe_caps(-1, -5)
    assert P._sqe_caps() == (2, 2)


def test_env_overrides_the_server_config(clean, monkeypatch):
    """The A/B arms are run by env against an already-configured server, which
    is the same way VAAYU_RING_POOL_N was used for P8."""
    P.set_sqe_caps(0, 0)
    monkeypatch.setenv("VAAYU_MAX_SQES_PER_ENTER", "2")
    monkeypatch.setenv("VAAYU_MAX_CHUNKS_PER_WAVE", "3")
    assert P._sqe_caps() == (2, 3)


def test_env_can_turn_the_caps_back_off(clean, monkeypatch):
    """The `off` arm of the A/B has to be reachable without editing config.py,
    or the two arms are not the same binary."""
    P.set_sqe_caps(2, 2)
    monkeypatch.setenv("VAAYU_MAX_SQES_PER_ENTER", "0")
    monkeypatch.setenv("VAAYU_MAX_CHUNKS_PER_WAVE", "0")
    assert P._sqe_caps() == (0, 0)


@pytest.mark.parametrize("raw", ["", "two", "2.5", "-3"])
def test_an_unparseable_env_value_falls_back_instead_of_crashing(clean, monkeypatch, raw):
    P.set_sqe_caps(2, 2)
    monkeypatch.setenv("VAAYU_MAX_SQES_PER_ENTER", raw)
    assert P._sqe_caps()[0] == 2, f"{raw!r} must not change the configured cap"


class _FakeRust:
    """Takes the four-argument form, i.e. a current wheel."""

    def __init__(self):
        self.calls = []

    def init_ring_pool(self, *args):
        self.calls.append(args)


class _StaleRust:
    """A wheel built before the caps: two arguments only."""

    def __init__(self):
        self.calls = []

    def init_ring_pool(self, pool_size, queue_depth):
        self.calls.append((pool_size, queue_depth))


def test_the_caps_reach_init_ring_pool(clean, monkeypatch):
    """A cap the loader never passes is decoration (CLAUDE.md section 8). The
    rings are built ONCE, during the load, so this call is the only chance."""
    fake = _FakeRust()
    monkeypatch.setattr(P, "vaayu_rust", fake)
    P.set_sqe_caps(2, 3)
    assert P._init_iouring_ring(2, 32) is True
    assert fake.calls == [(2, 32, 2, 3)]


def test_uncapped_still_passes_the_explicit_zeros(clean, monkeypatch):
    fake = _FakeRust()
    monkeypatch.setattr(P, "vaayu_rust", fake)
    assert P._init_iouring_ring(2, 32) is True
    assert fake.calls == [(2, 32, 0, 0)]


def test_a_wheel_too_old_for_the_caps_says_so_and_still_loads(clean, monkeypatch, capsys):
    """Silently dropping the caps is the worst outcome: the operator believes
    the box is mitigated and it is not. Loading is still right -- the caps are a
    kernel workaround, not a correctness requirement."""
    stale = _StaleRust()
    monkeypatch.setattr(P, "vaayu_rust", stale)
    P.set_sqe_caps(2, 2)
    assert P._init_iouring_ring(2, 32) is True
    assert stale.calls == [(2, 32)], "it must fall back to the old arity, not fail"
    out = capsys.readouterr().out
    assert "INERT" in out, out
    assert "CVE-2026-64017" in out, out


def test_an_old_wheel_is_not_warned_about_when_nothing_was_asked_for(clean, monkeypatch, capsys):
    """Uncapped is the default, so this warning would fire for every user on
    every load and mean nothing."""
    monkeypatch.setattr(P, "vaayu_rust", _StaleRust())
    assert P._init_iouring_ring(2, 32) is True
    assert "INERT" not in capsys.readouterr().out


def test_the_caps_are_a_server_config_knob():
    """Section 8: the server is the only interface, so an I/O knob that exists
    only as an env var is not reachable for a user who edits config.py."""
    from vaayu.server import config
    from vaayu.server.app import ServerConfig

    fields = {f.name for f in dataclasses.fields(ServerConfig)}
    for field, const in (("nvme_max_sqes_per_enter", "NVME_MAX_SQES_PER_ENTER"),
                         ("nvme_max_chunks_per_wave", "NVME_MAX_CHUNKS_PER_WAVE")):
        assert field in fields
        assert hasattr(config, const)


def test_the_shipped_defaults_are_the_measured_ones():
    """`2` and `0` -- and the dataclass must agree with the tracked config file.

    Both halves were measured before this was turned on (2026-11-17). bpftrace
    on the real O_DIRECT read path showed `nr_ios > 1` and the batch allocation
    that fills `cached_rqs` present at to_submit 32 and 3 and absent at 2 and 1,
    tracking each other 1:1; the server A/B showed +0.14 % on MiniMax-M2.7 and
    -0.16 % on dense with bit-identical output in all 20 arms. The chunk cap
    stays 0 because `2` above removes the hazard on its own, leaving it only
    fewer syscalls, which measured as nothing.

    Read against the TRACKED file, not the imported module: a developer's
    machine-local `config_local.py` must not decide whether the suite passes.
    """
    import ast
    import inspect

    from vaayu.server import config as cfg
    from vaayu.server.app import ServerConfig

    tracked = {}
    for node in ast.parse(inspect.getsource(cfg)).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                tracked[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert tracked["NVME_MAX_SQES_PER_ENTER"] == 2, (
        "this is the measured mitigation for a kernel bug that hard-hung the box; "
        "set it back to 0 only once the box boots a patched kernel (>= 7.0.0-28)")
    assert tracked["NVME_MAX_CHUNKS_PER_WAVE"] == 0

    defaults = {f.name: f.default for f in dataclasses.fields(ServerConfig)}
    assert defaults["nvme_max_sqes_per_enter"] == tracked["NVME_MAX_SQES_PER_ENTER"], (
        "the dataclass default and config.py must agree, or a caller that builds a "
        "ServerConfig directly is silently unmitigated")
    assert defaults["nvme_max_chunks_per_wave"] == tracked["NVME_MAX_CHUNKS_PER_WAVE"]


def test_the_caps_are_applied_before_the_model_loads(monkeypatch):
    """`init_ring_pool` runs during the load and is a no-op afterwards, so a cap
    applied after `from_pretrained` would be inert -- the exact failure this
    whole file exists to prevent, one layer down."""
    from fastapi.testclient import TestClient

    from vaayu.server.app import ServerConfig, create_app

    order = []

    def fake_set_sqe_caps(per_enter, chunks):
        order.append(("set_sqe_caps", per_enter, chunks))

    monkeypatch.setattr(P, "set_sqe_caps", fake_set_sqe_caps)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            order.append(("from_pretrained",))
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(ga, "audit", lambda **kw: [])

    cfg = ServerConfig(model_path="/ckpt", capture_stdio=False,
                       nvme_max_sqes_per_enter=2, nvme_max_chunks_per_wave=2)
    with TestClient(create_app(cfg)):
        pass

    assert order[:2] == [("set_sqe_caps", 2, 2), ("from_pretrained",)], order
