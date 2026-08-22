"""Who else is holding the GPU, and what to run to get it back.

Vaayu derives its pinning and cache budgets ONCE, from whatever VRAM is free at
load. A process that holds VRAM at that moment is not a transient cost -- it is
a permanently smaller budget, so more layers stream from NVMe on every token for
the life of the run. The measured spread between "nothing pinned" and "budget
filled" is 13.0x / 2.02x / 1.27x on the three on-device checkpoints.

Vaayu never frees another process's memory and cannot: GPU allocations belong to
the owning process, so only ending that process releases them. This module says
precisely what is held, by whom, and which command ends it.

Only COMPUTE processes are counted. The desktop (Xorg, gnome-shell, a browser)
holds graphics memory in the tens of MiB and must never be something an
inference engine tells you to kill.
"""
import os
import subprocess
from dataclasses import dataclass

_MIB = 1024 ** 2


@dataclass(frozen=True)
class GpuUser:
    pid: int
    name: str
    mib: int
    container: str | None = None

    @property
    def gib(self) -> float:
        return self.mib / 1024


def parse_compute_apps(csv_text: str) -> list[GpuUser]:
    """Rows of `nvidia-smi --query-compute-apps=pid,process_name,used_memory`.

    A row that cannot be read (nvidia-smi prints `[N/A]` for memory it cannot
    attribute) is skipped rather than raised on: a partial report is useful, a
    traceback during model load is not.
    """
    users = []
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            users.append(GpuUser(pid=int(parts[0]), name=parts[1], mib=int(parts[2])))
        except ValueError:
            continue
    return users


def parse_container_id(cgroup_text: str) -> str | None:
    """The 12-char docker id from /proc/<pid>/cgroup, or None if not a container.

    Worth the parse: a container must be stopped with `docker stop`, not killed.
    Killing the process inside one whose policy is `restart=unless-stopped` just
    respawns it -- which is exactly how 9.3 GiB survived a reboot.
    """
    for line in cgroup_text.splitlines():
        marker = "docker-"
        if marker in line:
            rest = line.split(marker, 1)[1]
            ident = rest.split(".")[0]
            if len(ident) >= 12:
                return ident[:12]
    return None


def foreign_gpu_users(users, own_pid=None) -> list[GpuUser]:
    own_pid = os.getpid() if own_pid is None else own_pid
    return [u for u in users if u.pid != own_pid]


def compose_report(users) -> str | None:
    """A message naming every holder and the command that ends it, or None."""
    if not users:
        return None
    total = sum(u.mib for u in users) / 1024
    lines = [f"{len(users)} other process(es) hold {total:.2f} GiB of GPU memory. "
             f"Vaayu sizes its pinning budget from what is free AT LOAD, so this "
             f"is a permanent tax on every token, not a transient one."]
    for u in users:
        where = f" (container {u.container})" if u.container else ""
        lines.append(f"  pid {u.pid}  {u.gib:.2f} GiB  {u.name}{where}")
    lines.append("Free it with:")
    seen = set()
    for u in users:
        # A container is STOPPED: killing the process inside one whose policy is
        # restart=unless-stopped only respawns it. One container can hold several
        # worker processes, and it is still stopped once.
        cmd = f"docker stop {u.container}" if u.container else f"kill {u.pid}"
        if cmd in seen:
            continue
        seen.add(cmd)
        lines.append(f"  {cmd}")
    lines.append("Then start Vaayu again -- budgets are derived at load and are "
                 "not revisited. To run anyway, set REQUIRE_EXCLUSIVE_GPU = False "
                 "in vaayu/server/config.py (or config_local.py).")
    return "\n".join(lines)


def audit() -> list[GpuUser]:
    """Foreign compute processes on the GPU. Empty when there is no GPU to read."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []          # no nvidia-smi, no GPU, or it hung: not our problem
    users = foreign_gpu_users(parse_compute_apps(out))
    return [_with_container(u) for u in users]


def _with_container(user: GpuUser) -> GpuUser:
    try:
        with open(f"/proc/{user.pid}/cgroup") as f:
            cid = parse_container_id(f.read())
    except OSError:
        cid = None
    return GpuUser(pid=user.pid, name=user.name, mib=user.mib, container=cid)
