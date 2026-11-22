"""cgroup-v2-aware memory basis. Leaf module: imports nothing from vaayu."""
import os

import psutil


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _level_cap(directory: str):
    """min(memory.high, memory.max) at one level; None if neither is numeric."""
    vals = []
    for name in ("memory.high", "memory.max"):
        try:
            txt = _read(os.path.join(directory, name))
            if txt != "max":
                vals.append(int(txt))
        except (OSError, ValueError):
            pass
    return min(vals) if vals else None


def _resolve_cap(cgroup_root: str, proc_cgroup: str):
    """(leaf_dir, binding_cap) for our cgroup v2 leaf, or None when there is
    no v2 hierarchy or no level caps memory. The cap is the min over leaf..root
    of min(memory.high, memory.max). May raise on unreadable /proc/self/cgroup."""
    lines = [ln for ln in _read(proc_cgroup).splitlines() if ln.strip()]
    if len(lines) != 1 or not lines[0].startswith("0::"):
        return None
    rel = lines[0][3:].strip("/")
    parts = [p for p in rel.split("/") if p]

    cap = None
    for depth in range(len(parts), -1, -1):  # leaf .. root
        lvl = _level_cap(os.path.join(cgroup_root, *parts[:depth]))
        if lvl is not None:
            cap = lvl if cap is None else min(cap, lvl)
    if cap is None:
        return None
    return os.path.join(cgroup_root, *parts), cap


def cgroup_cap_bytes(
    *,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_cgroup: str = "/proc/self/cgroup",
) -> int | None:
    """Binding cgroup memory cap in bytes, or None when uncapped. Never raises."""
    try:
        resolved = _resolve_cap(cgroup_root, proc_cgroup)
        return None if resolved is None else resolved[1]
    except Exception:
        return None


def effective_available_bytes(
    *,
    host_available: int | None = None,
    mem_total: int | None = None,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_cgroup: str = "/proc/self/cgroup",
) -> int:
    """Available memory under our cgroup cap, else host available (no-op).

    Never raises and never memoizes: memory.current / memory.stat are read
    fresh on every call. No extra headroom is deducted (downstream helpers
    already reserve for it).
    """
    host = host_available
    try:
        if host is None or mem_total is None:
            vm = psutil.virtual_memory()
            host = vm.available if host is None else host
            mem_total = vm.total if mem_total is None else mem_total

        resolved = _resolve_cap(cgroup_root, proc_cgroup)
        if resolved is None:
            return host
        leaf, cap = resolved
        if cap >= mem_total:
            return host

        current = int(_read(os.path.join(leaf, "memory.current")))
        stat = {}
        for ln in _read(os.path.join(leaf, "memory.stat")).splitlines():
            k, _, v = ln.partition(" ")
            stat[k] = v
        file_b = int(stat["file"])
        shmem = int(stat["shmem"])

        # shmem (the pinned arena) sits inside `file` but is unreclaimable
        # without swap: reclaimable = file - shmem, NEVER current - file.
        reclaimable = file_b - shmem
        unreclaimable = current - reclaimable
        return max(0, min(host, cap - unreclaimable))
    except Exception:
        return host if host is not None else 0
