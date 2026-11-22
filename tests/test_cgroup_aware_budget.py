"""effective_available_bytes(): cgroup-aware memory basis (fake cgroup v2 trees)."""
import os

from vaayu.cgroup_limits import effective_available_bytes

G = 1024 ** 3
HOST = 30 * G
TOTAL = 31 * G
LEAF = "system.slice/vaayu.service"


def _w(root, rel, name, text):
    d = os.path.join(root, rel) if rel else str(root)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(text)


def _leaf(tmp_path, *, high="max", max_="max", current=0, anon=0, file=0,
          shmem=0, parent_max="max", rel=LEAF):
    root = str(tmp_path / "cg")
    _w(root, rel, "memory.high", f"{high}\n")
    _w(root, rel, "memory.max", f"{max_}\n")
    _w(root, rel, "memory.current", f"{current}\n")
    _w(root, rel, "memory.stat",
       f"anon {anon}\nfile {file}\nkernel_stack 0\nslab 0\nsock 0\nshmem {shmem}\n")
    _w(root, os.path.dirname(rel), "memory.max", f"{parent_max}\n")
    _w(root, os.path.dirname(rel), "memory.high", "max\n")
    proc = str(tmp_path / "proc_cgroup")
    with open(proc, "w") as f:
        f.write(f"0::/{rel}\n")
    return root, proc


def _call(root, proc):
    return effective_available_bytes(
        host_available=HOST, mem_total=TOTAL, cgroup_root=root, proc_cgroup=proc)


def test_no_cgroup_v2_returns_host_unchanged(tmp_path):
    root, proc = _leaf(tmp_path, max_=28 * G)
    with open(proc, "w") as f:
        f.write("12:memory:/foo\n0::/system.slice/vaayu.service\n")
    assert _call(root, proc) == HOST


def test_proc_cgroup_missing_returns_host(tmp_path):
    root, _ = _leaf(tmp_path, max_=28 * G)
    assert _call(root, str(tmp_path / "nope")) == HOST


def test_both_caps_max_returns_host(tmp_path):
    root, proc = _leaf(tmp_path, current=20 * G, file=12 * G)
    assert _call(root, proc) == HOST


def test_high_unset_max_set_uses_max(tmp_path):
    root, proc = _leaf(tmp_path, max_=28 * G, current=10 * G, anon=10 * G)
    assert _call(root, proc) == 18 * G


def test_cap_ge_memtotal_returns_host(tmp_path):
    root, proc = _leaf(tmp_path, max_=TOTAL, current=20 * G, anon=20 * G)
    assert _call(root, proc) == HOST


def test_cap_below_host_basis_is_cap_minus_unreclaimable(tmp_path):
    root, proc = _leaf(tmp_path, max_=28 * G, current=20 * G, file=12 * G,
                       shmem=9 * G, anon=8 * G)
    got = _call(root, proc)
    assert got == 11 * G
    assert got < HOST


def test_trap_pinned_shmem_not_treated_reclaimable(tmp_path):
    # pinned-arena shape: file is mostly shmem. Naive current-file would give 24G.
    root, proc = _leaf(tmp_path, max_=28 * G, current=14 * G, file=10 * G,
                       shmem=9 * G, anon=4 * G)
    assert _call(root, proc) == 15 * G


def test_min_cap_when_high_below_max(tmp_path):
    root, proc = _leaf(tmp_path, high=26 * G, max_=28 * G, current=10 * G,
                       anon=10 * G)
    assert _call(root, proc) == 16 * G


def test_cap_on_parent_leaf_unset_parent_binds(tmp_path):
    root, proc = _leaf(tmp_path, parent_max=20 * G, current=5 * G, anon=5 * G)
    assert _call(root, proc) == 15 * G


def test_leaf_tighter_than_parent_leaf_wins(tmp_path):
    root, proc = _leaf(tmp_path, max_=10 * G, parent_max=20 * G, current=2 * G,
                       anon=2 * G)
    assert _call(root, proc) == 8 * G


def test_degenerate_cap_minus_unreclaimable_clamps_zero(tmp_path):
    root, proc = _leaf(tmp_path, max_=10 * G, current=15 * G, anon=15 * G)
    assert _call(root, proc) == 0


def test_never_memoized_sees_later_pinning(tmp_path):
    root, proc = _leaf(tmp_path, max_=28 * G, current=5 * G, anon=5 * G)
    assert _call(root, proc) == 23 * G
    _leaf(tmp_path, max_=28 * G, current=14 * G, anon=14 * G)
    assert _call(root, proc) == 14 * G
