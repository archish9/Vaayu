"""Keep the CPU package out of deep idle while a generation runs (row 27).

The box idles in C2_ACPI (declared exit latency 127 us) with the CPU ~7.5 %
busy during decode. Qwen's fused-EXL3 forward issues async copies and then
SLEEPS on the read completions -- ~800 of them per token -- so it pays that
wake latency on nearly every one. A SCHED_IDLE spinner on an E-core keeps the
package shallow and bought 15.5 % (one spinner) to 21.5 % (all eight E-cores)
of decode wall-clock in row 26's arms; pinning the engine's own threads to the
P-cores bought a further 7.6 % and the two compose.

SCHED_IDLE is the whole reason a busy loop is admissible here: it runs only
when a CPU would otherwise go idle, so it takes no time from a real thread.
What it does NOT protect against is SMT sibling contention -- two hyperthreads
share execution ports regardless of scheduler priority. Row 26's `Uspin` arm
(spinners on all 20 CPUs) slowed MiniMax 27 % for exactly that reason, so a
spin CPU may never be an SMT sibling of a CPU the engine runs on.

Every CPU id here is DETECTED off cpuinfo_max_freq (CLAUDE.md 5), never typed.
"""
import atexit
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuTopology:
    """Which CPUs the engine should run on, and which may hold a spinner."""
    p_cpus: tuple          # engine threads belong here
    spin_cpus: tuple       # keep-awake spinners may run here, nowhere else
    hybrid: bool
    warning: str = None    # non-None when this box is not the measured shape
    all_cpus: tuple = ()   # every CPU seen; pinning is meaningful only if p_cpus < this


def _read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _siblings(cpu_dir, cpu):
    """Parse thread_siblings_list ("0-1" or "0,12"). Absent -> cpu alone."""
    try:
        with open(os.path.join(cpu_dir, "topology", "thread_siblings_list")) as fh:
            spec = fh.read().strip()
    except OSError:
        return {cpu}
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out or {cpu}


def detect_topology(sys_root="/sys", allow_non_hybrid=False):
    """Split the CPUs into an engine class and a spin class.

    `allow_non_hybrid` opts a uniform-frequency box in by reserving ONE whole
    physical core for the spinner; without it such a box gets no spin CPU and a
    warning, because row 27 was measured on a hybrid CPU only.
    """
    base = os.path.join(sys_root, "devices", "system", "cpu")
    try:
        cpus = sorted(int(n[3:]) for n in os.listdir(base)
                      if n.startswith("cpu") and n[3:].isdigit())
    except OSError:
        cpus = []
    if not cpus:
        return CpuTopology((), (), False, "no CPU topology under %s" % base, ())

    freqs = {c: _read_int(os.path.join(base, f"cpu{c}", "cpufreq", "cpuinfo_max_freq"))
             for c in cpus}
    sibs = {c: _siblings(os.path.join(base, f"cpu{c}"), c) for c in cpus}

    known = [f for f in freqs.values() if f is not None]
    if len(known) != len(cpus) or len(set(known)) < 2:
        why = ("no cpufreq information" if len(known) != len(cpus)
               else "CPU is not hybrid (one frequency class)")
        if not allow_non_hybrid:
            return CpuTopology(tuple(cpus), (), False,
                               f"keep-awake declined: {why}; row 27 was measured "
                               f"on a hybrid CPU and a spinner would share an SMT "
                               f"sibling with an engine thread", tuple(cpus))
        # Opt-in on unmeasured hardware: give up one whole physical core so the
        # spinner shares its siblings with nobody.
        reserved = sibs[cpus[-1]]
        return CpuTopology(tuple(c for c in cpus if c not in reserved),
                           (max(reserved),), False,
                           f"keep-awake on a non-hybrid CPU ({why}): reserved "
                           f"CPUs {sorted(reserved)} for one spinner; unmeasured",
                           tuple(cpus))

    fast = max(known)
    p_cpus = tuple(c for c in cpus if freqs[c] == fast)
    engine = set(p_cpus)
    # Enforce the SMT rule by construction rather than by assumption.
    spin = tuple(c for c in cpus if freqs[c] != fast and not (sibs[c] & engine))
    return CpuTopology(p_cpus, spin, True, None, tuple(cpus))


# ── thread placement ────────────────────────────────────────────────────────

def _comm(tid):
    try:
        with open(f"/proc/self/task/{tid}/comm") as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def pin_process_threads(cpus, tids=None):
    """Move every thread of THIS process onto `cpus`.

    Returns `(saved, refused)`: `saved` maps tid -> its previous mask (feed it
    to `restore_thread_affinity`), `refused` lists `(tid, OSError)`. Row 26's
    UP arm refused 0 of 66 threads on this box, so a refusal is news and is
    printed with the thread's name rather than counted silently -- a thread the
    kernel will not move is a thread that keeps waking a deep-idle core.
    """
    if tids is None:
        tids = os.listdir("/proc/self/task")
    mask = set(cpus)
    saved, refused = {}, []
    for tid in tids:
        tid = int(tid)
        try:
            previous = os.sched_getaffinity(tid)
            os.sched_setaffinity(tid, mask)
        except OSError as exc:
            refused.append((tid, exc))
            print(f"[Vaayu-KeepAwake] WARNING: thread {tid} ({_comm(tid)}) "
                  f"refused P-core affinity: {exc}", flush=True)
            continue
        saved[tid] = previous
    return saved, refused


def restore_thread_affinity(saved, default=None):
    """Put back exactly what `pin_process_threads` found.

    `default` is the mask for threads that did NOT exist when the pin was taken:
    the engine creates threads inside a generation (stream_generate's worker and
    any reader spawned under it), they inherit the P-core mask, and restoring
    only the saved tids would leave them pinned for the life of the process --
    half of row 27's effect applied silently, forever, and in an A/B every arm
    after the first pinned one contaminated. Threads that have exited meanwhile
    are not an error.
    """
    for tid, mask in saved.items():
        try:
            os.sched_setaffinity(tid, mask)
        except OSError:
            pass
    if default is None:
        return
    for tid in os.listdir("/proc/self/task"):
        tid = int(tid)
        if tid in saved:
            continue
        try:
            os.sched_setaffinity(tid, default)
        except OSError:
            pass


# ── the keep-awake itself ───────────────────────────────────────────────────

# A shell busy loop, not a Python one: a Python thread would hold the GIL and a
# Python subprocess would cost ~15 MB RSS per spinner against a 32 GB box.
# /bin/sh is ~1 MB and POSIX-mandatory, so this needs no util-linux (`chrt`,
# `taskset`) on PATH either -- the scheduler class and the affinity are set
# from the parent, on the child's pid, with stdlib calls.
#
# The outer `kill -0` is what stops an orphan: SIGKILL runs no atexit handler,
# and this box OOM-kills loads often enough (exit 137 is on the record for
# three separate gates) that a spinner trusting atexit alone would burn an
# E-core until reboot. The inner arithmetic loop is the actual busy work, sized
# so the owner check runs every few ms -- far below the wake latency this
# feature exists to hide (127 us), and cheap because `kill` is a shell builtin.
_SPIN_WATCH = ('p=$1; while kill -0 "$p" 2>/dev/null; do i=0; '
               'while [ "$i" -lt 200000 ]; do i=$((i+1)); done; done')


def _spin_cmd(owner_pid):
    return ["/bin/sh", "-c", _SPIN_WATCH, "vaayu-keepawake", str(owner_pid)]

_live_spinners = []      # every spinner this process owns, for the atexit net
_warned = False


def _reap_all():
    """A crash mid-generation must not leave spinners burning E-cores."""
    for proc in list(_live_spinners):
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
    _live_spinners.clear()


atexit.register(_reap_all)


class KeepAwake:
    """Hold the CPU package out of deep idle for the duration of a block.

    `spinners`: "auto" (every detected spin CPU), an int (capped at that), or
    0/False (off). Re-entrant: a nested `with` on the same instance keeps the
    spinners the outer block started and stops them only at the outermost exit.

    One generation at a time, which is what the server does (`engine.slot()`
    holds a lock for a whole run): the depth counter is not thread-safe, so two
    threads entering the SAME instance concurrently is not supported.
    """

    def __init__(self, spinners="auto", pin_threads=False, topology=None):
        self._topo = topology if topology is not None else detect_topology()
        self._want = spinners
        self._pin = pin_threads
        self._procs = []
        self._saved_affinity = None
        self._pre_pin_mask = None
        self._depth = 0
        self.spin_cpus = ()

    @property
    def pids(self):
        return [p.pid for p in self._procs]

    def _chosen_cpus(self):
        if not self._want or not self._topo.spin_cpus:
            return ()
        avail = self._topo.spin_cpus
        return avail if self._want == "auto" else tuple(avail[:int(self._want)])

    def __enter__(self):
        global _warned
        self._depth += 1
        if self._depth > 1:
            return self
        if self._pin and set(self._topo.p_cpus) < set(self._topo.all_cpus):
            # The mask threads born inside the block are put back to on exit.
            self._pre_pin_mask = os.sched_getaffinity(0)
            self._saved_affinity, _ = pin_process_threads(self._topo.p_cpus)
        self.spin_cpus = self._chosen_cpus()
        if not self.spin_cpus:
            if self._want and self._topo.warning and not _warned:
                _warned = True
                print(f"[Vaayu-KeepAwake] WARNING: {self._topo.warning}", flush=True)
            return self
        for cpu in self.spin_cpus:
            proc = subprocess.Popen(_spin_cmd(os.getpid()), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            try:
                os.sched_setscheduler(proc.pid, os.SCHED_IDLE, os.sched_param(0))
                os.sched_setaffinity(proc.pid, {cpu})
            except OSError as exc:
                # Never leave a spinner running at a priority that could steal
                # from the engine -- that is the one way this feature could
                # cost throughput instead of buying it.
                proc.kill()
                proc.wait()
                print(f"[Vaayu-KeepAwake] WARNING: spinner on CPU {cpu} refused "
                      f"({exc}); running without it", flush=True)
                continue
            self._procs.append(proc)
            _live_spinners.append(proc)
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth > 0:
            return False
        for proc in self._procs:
            proc.kill()
        for proc in self._procs:
            proc.wait()
            if proc in _live_spinners:
                _live_spinners.remove(proc)
        self._procs = []
        self.spin_cpus = ()
        if self._saved_affinity is not None:
            restore_thread_affinity(self._saved_affinity, self._pre_pin_mask)
            self._saved_affinity = None
            self._pre_pin_mask = None
        return False
