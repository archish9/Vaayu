"""Keep-awake CPU topology + spinner lifecycle (row 27).

An idle CPU package costs Qwen3-235B ~21 % of decode: every one of the ~800 raw
read completions per token wakes a thread out of C2_ACPI (127 us declared exit
latency). A SCHED_IDLE spinner keeps the package shallow.

CLAUDE.md 5: the E-cores are DETECTED off cpuinfo_max_freq, never typed. These
tests run on a fake /sys tree for exactly that reason -- a test that asserted
"cpus 12-19" would pass on this box and be wrong everywhere else, including on
the same box with a different CPU.

The SMT rule is not a nicety: row 26's `Uspin` arm (spinners on all 20 CPUs)
SLOWED MiniMax 27 % because those spinners shared P-core SMT siblings with
engine threads. SCHED_IDLE does not protect against sibling contention -- the
two hyperthreads share execution ports whatever the scheduler priority is. So
a spin CPU may never be a sibling of a CPU the engine runs on.
"""
import pytest

from vaayu.keepawake import detect_topology


def _write_cpu(root, cpu, khz, siblings):
    d = root / "devices" / "system" / "cpu" / f"cpu{cpu}"
    (d / "cpufreq").mkdir(parents=True)
    (d / "cpufreq" / "cpuinfo_max_freq").write_text(f"{khz}\n")
    (d / "topology").mkdir(parents=True)
    (d / "topology" / "thread_siblings_list").write_text(siblings + "\n")


def _hybrid_box(tmp_path, p_cpus=range(0, 12), e_cpus=range(12, 20)):
    """This box's shape: 6 P-cores x 2 HT at 5.3 GHz, 8 E-cores at 4.0 GHz."""
    for c in p_cpus:
        sib = (c // 2) * 2
        _write_cpu(tmp_path, c, 5300000, f"{sib}-{sib + 1}")
    for c in e_cpus:
        _write_cpu(tmp_path, c, 4000000, str(c))
    return tmp_path


def test_the_e_cores_are_the_below_max_frequency_class(tmp_path):
    topo = detect_topology(sys_root=str(_hybrid_box(tmp_path)))
    assert topo.hybrid is True
    assert topo.spin_cpus == tuple(range(12, 20))
    assert topo.p_cpus == tuple(range(0, 12))


def test_the_class_is_read_off_frequency_not_off_the_cpu_number(tmp_path):
    # Same box, E-cores FIRST. An implementation that assumed "high-numbered
    # CPUs are the little ones" passes the test above and fails here.
    root = _hybrid_box(tmp_path, p_cpus=range(8, 20), e_cpus=range(0, 8))
    topo = detect_topology(sys_root=str(root))
    assert topo.spin_cpus == tuple(range(0, 8))
    assert topo.p_cpus == tuple(range(8, 20))


def test_a_spin_cpu_is_never_an_smt_sibling_of_an_engine_cpu(tmp_path):
    topo = detect_topology(sys_root=str(_hybrid_box(tmp_path)))
    siblings_of_engine = set()
    for c in topo.p_cpus:
        sib = (c // 2) * 2
        siblings_of_engine |= {sib, sib + 1}
    assert not (set(topo.spin_cpus) & siblings_of_engine)


def test_a_non_hybrid_box_offers_no_spin_cpu_and_says_why(tmp_path):
    # Uniform frequency: every core the engine wants is a core a spinner would
    # share. Row 27 was measured on a hybrid CPU only, so `auto` must decline
    # here and WARN rather than silently pick a core (CLAUDE.md 5).
    for c in range(8):
        sib = (c // 2) * 2
        _write_cpu(tmp_path, c, 4800000, f"{sib}-{sib + 1}")
    topo = detect_topology(sys_root=str(tmp_path))
    assert topo.hybrid is False
    assert topo.spin_cpus == ()
    assert topo.warning is not None and "hybrid" in topo.warning.lower()
    assert topo.p_cpus == tuple(range(8))


def test_a_non_hybrid_box_reserves_one_whole_physical_core_when_asked(tmp_path):
    # Explicit opt-in on unmeasured hardware: the spinner takes one thread of a
    # physical core, and BOTH its siblings leave the engine mask -- the
    # construction that makes row 26's Uspin confound impossible.
    for c in range(8):
        sib = (c // 2) * 2
        _write_cpu(tmp_path, c, 4800000, f"{sib}-{sib + 1}")
    topo = detect_topology(sys_root=str(tmp_path), allow_non_hybrid=True)
    assert len(topo.spin_cpus) == 1
    assert set(topo.spin_cpus) & {6, 7}
    assert topo.p_cpus == tuple(range(6))


def test_a_box_without_cpufreq_offers_no_spin_cpu_and_says_why(tmp_path):
    # VMs and many servers expose no cpufreq at all. Degrade with a warning.
    for c in range(4):
        d = tmp_path / "devices" / "system" / "cpu" / f"cpu{c}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(f"{c}\n")
    topo = detect_topology(sys_root=str(tmp_path))
    assert topo.spin_cpus == ()
    assert topo.warning is not None
    assert topo.p_cpus == tuple(range(4))


def test_a_box_with_no_sibling_information_still_classifies(tmp_path):
    # thread_siblings_list is absent on some kernels; each CPU is then its own
    # core and the hybrid split still works.
    for c in range(4):
        d = tmp_path / "devices" / "system" / "cpu" / f"cpu{c}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "cpuinfo_max_freq").write_text("5300000\n" if c < 2 else "4000000\n")
    topo = detect_topology(sys_root=str(tmp_path))
    assert topo.hybrid is True
    assert topo.spin_cpus == (2, 3)
    assert topo.p_cpus == (0, 1)


# ── spinner lifecycle (real processes on this box) ──────────────────────────

import os
import time

from vaayu.keepawake import KeepAwake


def _live(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def hybrid_box():
    topo = detect_topology()
    if not topo.spin_cpus:
        pytest.skip(f"no spin CPU on this box: {topo.warning}")
    return topo


def test_a_spinner_runs_at_sched_idle_on_the_cpu_it_was_given(hybrid_box):
    # SCHED_IDLE is what makes a busy loop admissible: it runs only when the
    # CPU would otherwise be idle, so no real thread loses time to it.
    with KeepAwake(spinners=1) as ka:
        assert len(ka.pids) == 1
        pid = ka.pids[0]
        assert os.sched_getscheduler(pid) == os.SCHED_IDLE
        assert os.sched_getaffinity(pid) == {ka.spin_cpus[0]}
        assert ka.spin_cpus[0] in hybrid_box.spin_cpus


def test_every_spinner_is_gone_after_the_block(hybrid_box):
    with KeepAwake(spinners=2) as ka:
        pids = list(ka.pids)
        assert len(pids) == 2
        assert len(set(ka.spin_cpus)) == 2   # one CPU each, never doubled up
    for pid in pids:
        assert not _live(pid)


def test_an_exception_inside_the_block_still_kills_the_spinners(hybrid_box):
    pids = []
    with pytest.raises(ZeroDivisionError):
        with KeepAwake(spinners=1) as ka:
            pids = list(ka.pids)
            1 / 0
    assert pids and not any(_live(p) for p in pids)


def test_nesting_spawns_one_set_and_keeps_it_until_the_outermost_exit(hybrid_box):
    # stream_generate calls model.generate in a worker thread; a nested
    # generate must not double the spinners, and the inner exit must not kill
    # the outer block's.
    ka = KeepAwake(spinners=1)
    with ka:
        pids = list(ka.pids)
        with ka:
            assert ka.pids == pids
        assert all(_live(p) for p in pids)
    assert not any(_live(p) for p in pids)


def test_asking_for_no_spinners_spawns_nothing(hybrid_box):
    with KeepAwake(spinners=0) as ka:
        assert ka.pids == []
        assert ka.spin_cpus == ()


def test_auto_uses_every_detected_spin_cpu(hybrid_box):
    with KeepAwake(spinners="auto") as ka:
        assert tuple(ka.spin_cpus) == hybrid_box.spin_cpus


def test_more_spinners_than_cpus_is_capped_not_doubled_up(hybrid_box):
    with KeepAwake(spinners=len(hybrid_box.spin_cpus) + 5) as ka:
        assert tuple(ka.spin_cpus) == hybrid_box.spin_cpus


def test_a_box_with_no_spin_cpu_is_a_no_op_that_warns_once(tmp_path, capsys):
    for c in range(4):
        d = tmp_path / "devices" / "system" / "cpu" / f"cpu{c}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "cpuinfo_max_freq").write_text("4800000\n")
    topo = detect_topology(sys_root=str(tmp_path))
    with KeepAwake(spinners="auto", topology=topo) as ka:
        assert ka.pids == []
    assert "keep-awake" in capsys.readouterr().out.lower()


# ── P-core pinning (row 26's UP arm: 7.6 % on its own, composes with the spinner) ──

import threading

from vaayu.keepawake import pin_process_threads, restore_thread_affinity


def test_pinning_moves_every_thread_of_this_process_and_restores_it(hybrid_box):
    before = {int(t): os.sched_getaffinity(int(t))
              for t in os.listdir("/proc/self/task")}
    saved, refused = pin_process_threads(hybrid_box.p_cpus)
    try:
        assert refused == []
        for tid in os.listdir("/proc/self/task"):
            if int(tid) in before:      # threads born since are not this test's business
                assert os.sched_getaffinity(int(tid)) == set(hybrid_box.p_cpus)
    finally:
        restore_thread_affinity(saved)
    for tid, mask in before.items():
        assert os.sched_getaffinity(tid) == mask


def test_a_thread_that_exits_mid_pin_is_reported_not_swallowed(hybrid_box, capsys):
    # The real refusal case: /proc/self/task is a snapshot, and a thread can be
    # gone by the time sched_setaffinity reaches it. Row 26's UP arm refused 0
    # of 66 threads on this box, so any refusal at all is news.
    dead_tid = 2 ** 22 - 1          # above pid_max: reliably ESRCH
    saved, refused = pin_process_threads(hybrid_box.p_cpus, tids=[dead_tid])
    restore_thread_affinity(saved)
    assert [t for t, _ in refused] == [dead_tid]
    assert str(dead_tid) in capsys.readouterr().out


def test_keepawake_pins_for_the_block_and_restores_after(hybrid_box):
    before = {int(t): os.sched_getaffinity(int(t))
              for t in os.listdir("/proc/self/task")}
    with KeepAwake(spinners=0, pin_threads=True):
        assert os.sched_getaffinity(0) == set(hybrid_box.p_cpus)
    for tid, mask in before.items():
        assert os.sched_getaffinity(tid) == mask


def test_a_thread_started_inside_the_block_inherits_the_p_core_mask(hybrid_box):
    # Why pinning the CALLING thread is enough for the Rust io_uring workers:
    # a thread inherits its creator's affinity. stream_generate's worker and
    # every reader thread are created after generate() has entered.
    seen = {}
    with KeepAwake(spinners=0, pin_threads=True):
        t = threading.Thread(target=lambda: seen.update(m=os.sched_getaffinity(0)))
        t.start()
        t.join()
    assert seen["m"] == set(hybrid_box.p_cpus)


def test_pinning_is_skipped_when_the_box_offers_no_p_core_split(tmp_path):
    for c in range(4):
        d = tmp_path / "devices" / "system" / "cpu" / f"cpu{c}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "cpuinfo_max_freq").write_text("4800000\n")
    topo = detect_topology(sys_root=str(tmp_path))
    before = os.sched_getaffinity(0)
    with KeepAwake(spinners=0, pin_threads=True, topology=topo):
        assert os.sched_getaffinity(0) == before   # no split -> nothing to pin to
    assert os.sched_getaffinity(0) == before


# ── orphan safety ───────────────────────────────────────────────────────────

import signal
import subprocess
import sys

_CHILD = """
import os, sys, time
sys.path.insert(0, {repo!r})
from vaayu.keepawake import KeepAwake
ka = KeepAwake(spinners=1)
with ka:
    print("SPINPIDS " + " ".join(str(p) for p in ka.pids), flush=True)
    time.sleep(60)
"""


def _spawn_holder(repo):
    proc = subprocess.Popen([sys.executable, "-c", _CHILD.format(repo=repo)],
                            stdout=subprocess.PIPE, text=True)
    return proc, _read_pids(proc)


def _read_pids(proc):
    """Importing `vaayu` prints a banner; the pid line is the marked one."""
    while True:
        line = proc.stdout.readline()
        assert line, "child exited before reporting its spinners"
        if line.startswith("SPINPIDS "):
            return [int(p) for p in line.split()[1:]]


def test_a_sigkilled_owner_does_not_orphan_a_spinner(hybrid_box):
    # This box OOM-kills loads (exit 137 is on the record for three separate
    # gates). SIGKILL runs no atexit handler, so a spinner that only trusts
    # atexit would burn an E-core until the next reboot.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc, pids = _spawn_holder(repo)
    assert pids and all(_live(p) for p in pids)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    deadline = time.time() + 10
    while time.time() < deadline and any(_live(p) for p in pids):
        time.sleep(0.1)
    assert not any(_live(p) for p in pids), f"orphaned spinners: {pids}"


def test_a_normal_interpreter_exit_reaps_spinners_without_the_block(hybrid_box):
    # atexit still covers the ordinary case: an exception that escapes the
    # process, or a caller that forgot __exit__.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = _CHILD.format(repo=repo).replace("time.sleep(60)", "os._exit(0) if False else None")
    proc = subprocess.Popen([sys.executable, "-c", src + "\nraise SystemExit(3)\n"],
                            stdout=subprocess.PIPE, text=True)
    pids = _read_pids(proc)
    proc.wait(timeout=30)
    deadline = time.time() + 5
    while time.time() < deadline and any(_live(p) for p in pids):
        time.sleep(0.1)
    assert not any(_live(p) for p in pids)


def test_a_thread_born_inside_the_block_does_not_stay_pinned(hybrid_box):
    """Restoring only the threads that were SAVED leaves the new ones pinned.

    The engine creates threads inside a generation (`stream_generate`'s worker,
    and any reader thread spawned under it), and they inherit the P-core mask.
    If the exit does not put them back, P-core placement -- half of row 27's
    effect -- silently persists for the life of the process: every later
    request runs part-pinned, and in an A/B every arm after the first `pin`
    arm is contaminated.
    """
    born = {}
    full = os.sched_getaffinity(0)
    with KeepAwake(spinners=0, pin_threads=True) as ka:
        assert os.sched_getaffinity(0) == set(hybrid_box.p_cpus)

        def _work():
            born["tid"] = threading.get_native_id()
            born["inside"] = os.sched_getaffinity(0)
            done.wait()

        done = threading.Event()
        t = threading.Thread(target=_work)
        t.start()
        while "tid" not in born:
            time.sleep(0.005)
    assert born["inside"] == set(hybrid_box.p_cpus)     # it did inherit the pin
    after = os.sched_getaffinity(born["tid"])
    done.set()
    t.join()
    assert after == full, (
        f"thread born inside the block is still pinned to {after}, not {full}")
