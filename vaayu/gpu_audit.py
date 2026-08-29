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

Two things narrow it further, because "any foreign process, any size, any GPU"
refuses to boot over things that cost nothing:

* WHICH GPU. Rows are joined on `gpu_uuid`, not on nvidia-smi's index: that
  index is nvidia-smi's own enumeration and neither `CUDA_DEVICE_ORDER` nor
  `CUDA_VISIBLE_DEVICES` moves it in step with torch's `cuda:N`. The UUID is
  the one identifier both sides agree on.
* HOW BIG. A bare CUDA context is 128 MiB on this box (measured 2026-08-29), so
  a holder under the threshold is REPORTED but does not block. Reported, never
  dropped -- under-reporting is the one thing this module must not do.
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
    uuid: str | None = None

    @property
    def gib(self) -> float:
        return self.mib / 1024


def parse_compute_apps(csv_text: str) -> list[GpuUser]:
    """Rows of `--query-compute-apps=gpu_uuid,pid,process_name,used_memory`.

    A row that cannot be read (nvidia-smi prints `[N/A]` for memory it cannot
    attribute) is skipped rather than raised on: a partial report is useful, a
    traceback during model load is not.

    The name is taken as everything BETWEEN the pid and the memory rather than
    as one field: nvidia-smi does not quote its CSV, so a process name
    containing a comma makes the row five fields, and a fixed field count
    dropped such a holder silently.
    """
    users = []
    for line in csv_text.splitlines():
        raw = line.split(",")
        if len(raw) < 4:
            continue
        try:
            users.append(GpuUser(uuid=raw[0].strip(), pid=int(raw[1]),
                                 name=",".join(raw[2:-1]).strip(),
                                 mib=int(raw[-1])))
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


def foreign_gpu_users(users, own_pid=None, uuid=None) -> list[GpuUser]:
    """Holders that are not us. With `uuid`, only those on that one GPU."""
    own_pid = os.getpid() if own_pid is None else own_pid
    return [u for u in users
            if u.pid != own_pid and (uuid is None or u.uuid == uuid)]


def blocking_users(users, min_mib: int = 0) -> list[GpuUser]:
    """The holders big enough to be worth refusing over.

    Separate from `compose_report`, which reports the small ones too: "does this
    refuse the boot" and "is this worth saying" are different questions.
    """
    return [u for u in users if u.mib >= min_mib]


def _torch_device_uuid(index: int) -> str:
    import torch

    return f"GPU-{torch.cuda.get_device_properties(index).uuid}"


def device_uuid(device) -> str | None:
    """`"cuda:1"` -> the GPU uuid nvidia-smi reports for it, else None.

    None means "do not filter", and every failure resolves to it: a CPU device,
    a malformed string, no torch, no driver. Failing open OVER-reports, which
    costs a warning; failing closed would silently audit nothing.
    """
    text = str(device or "")
    if not text.startswith("cuda"):
        return None
    _, _, suffix = text.partition(":")
    try:
        return _torch_device_uuid(int(suffix) if suffix else 0)
    except Exception:                             # noqa: BLE001 -- see docstring
        return None


def compose_report(users, min_mib: int = 0) -> str | None:
    """A message naming every holder and the command that ends it, or None.

    Holders under `min_mib` are listed at the end but are left out of the total
    and out of the commands: the total is the argument for refusing, so it must
    be the memory the refusal is about, and nobody should be told to kill a bare
    CUDA context.
    """
    if not users:
        return None
    blocking = blocking_users(users, min_mib)
    minor = [u for u in users if u not in blocking]
    if not blocking:
        lines = [f"{len(minor)} other process(es) hold GPU memory, all below the "
                 f"{min_mib} MiB threshold, so the load goes ahead:"]
        for u in minor:
            lines.append(f"  pid {u.pid}  {u.gib:.2f} GiB  {u.name}")
        return "\n".join(lines)
    users, total = blocking, sum(u.mib for u in blocking) / 1024
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
    for u in minor:
        lines.append(f"Also on this GPU, below the {min_mib} MiB threshold and "
                     f"not blocking: pid {u.pid}  {u.gib:.2f} GiB  {u.name}")
    return "\n".join(lines)


def audit(device=None) -> list[GpuUser]:
    """Foreign compute processes. Empty when there is no GPU to read.

    `device` (`"cuda:0"`) narrows the result to that one GPU. None -- the
    library path, which audits the whole box -- keeps every GPU.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []          # no nvidia-smi, no GPU, or it hung: not our problem
    users = foreign_gpu_users(parse_compute_apps(out),
                              uuid=device_uuid(device) if device else None)
    return [_with_container(u) for u in users]


def _with_container(user: GpuUser) -> GpuUser:
    try:
        with open(f"/proc/{user.pid}/cgroup") as f:
            cid = parse_container_id(f.read())
    except OSError:
        cid = None
    return GpuUser(pid=user.pid, name=user.name, mib=user.mib, container=cid,
                   uuid=user.uuid)
