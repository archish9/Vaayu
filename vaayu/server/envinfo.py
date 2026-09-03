"""`env.json` -- the snapshot that lets a maintainer debug a stranger's zip.

On 2026-09-03 a "the server is too slow" report was diagnosed, and every
load-bearing fact came from OUTSIDE `logs/`: the gitignored `config_local.py`
(where the real problem lived), CLAUDE.md's hardware constants, CLAUDE.md's
dispatch table, and a benchmark band read out of the source. For a stranger's
zip none of that is available. This module writes all of it down, once, at
startup, after the load -- after, so the DERIVED values are real rather than
requested.

Two rules govern everything here:

  * **Nothing is assumed.** Layer counts, expert counts, expert bytes, budgets
    and the selected forward are READ off the loaded model (CLAUDE.md 5). The
    forward in particular is read from the tag `_make_sparse_moe_forward`
    leaves behind, never re-derived -- a second copy of that dispatch table
    would drift and then misreport which forward ran.
  * **Nothing here may stop a boot.** Every field is collected through
    `_safe`, which records the failure in `collect_errors` and yields None. A
    partial report that names what it could not collect beats no server.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

_GB = 1e9                       # decimal, to match the startup prints
_GIB = 1024 ** 3                # binary, to match the VRAM plan's own units

# Vaayu's own flag names -- not model numbers. Listed so an UNSET one is
# reported as null rather than being invisible: "the user did not set
# VAAYU_LOOKAHEAD" and "this report predates VAAYU_LOOKAHEAD" must not look the
# same to whoever reads the bundle.
_KNOWN_VAAYU_VARS = (
    "VAAYU_CACHE_BIAS", "VAAYU_LOOKAHEAD", "VAAYU_NO_WARMUP",
    "VAAYU_OVERLAP_TRACE", "VAAYU_RING_POOL_N", "VAAYU_TRACE_ACT",
    "VAAYU_TRACE_EXPERT", "VAAYU_VRAM_DIAG",
)

# Config fields holding absolute paths, which contain a username.
_PATH_FIELDS = ("model_path", "drafter_model_path", "log_dir")


def redact_path(p):
    """An absolute path reduced to what triage needs and nothing more.

    The basename identifies the checkpoint; the hash distinguishes two
    different paths with the same basename. Neither leaks a username.
    """
    if not p:
        return None
    p = str(p)
    return {"name": os.path.basename(p.rstrip("/")) or p,
            "hash": hashlib.sha1(p.encode("utf-8", "replace")).hexdigest()[:10]}


class _Collector:
    def __init__(self):
        self.errors = []

    def safe(self, name, fn):
        try:
            return fn()
        except Exception as e:                  # noqa: BLE001 -- recorded, not raised
            self.errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    def section(self, prefix, fields):
        """Per-FIELD wrapping, deliberately: one missing attribute must cost
        one field, not the whole section it happens to live in."""
        return {k: self.safe(f"{prefix}.{k}", fn) for k, fn in fields.items()}


def _vaayu_section(c):
    repo = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()

    return c.section("vaayu", {
        "commit": lambda: git("rev-parse", "--short", "HEAD") or None,
        "dirty": lambda: bool(git("status", "--porcelain")),
        "branch": lambda: git("rev-parse", "--abbrev-ref", "HEAD") or None,
    })


def _python_section(c):
    def version(pkg):
        from importlib.metadata import version as v
        try:
            return v(pkg)
        except Exception:                       # noqa: BLE001 -- absent is a fact
            return None

    return c.section("python", {
        "version": lambda: sys.version.split()[0],
        "executable": lambda: sys.executable,
        "torch": lambda: version("torch"),
        "transformers": lambda: version("transformers"),
        "flash_attn": lambda: version("flash-attn"),
        "vaayu_rust": lambda: version("vaayu_rust"),
        "fastapi": lambda: version("fastapi"),
    })


def _host_section(c):
    def meminfo(key):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1]) * 1024
        return None

    def cpu_model():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        return None

    return c.section("host", {
        "kernel": lambda: platform.release(),
        "platform": lambda: platform.platform(),
        "cpu": cpu_model,
        "cpu_count": lambda: os.cpu_count(),
        "ram_total_gb": lambda: round(meminfo("MemTotal:") / _GB, 2),
        "ram_avail_at_load_gb": lambda: round(meminfo("MemAvailable:") / _GB, 2),
        "swap_total_gb": lambda: round(meminfo("SwapTotal:") / _GB, 2),
    })


def _gpu_section(c, device):
    import torch

    idx = 0
    if isinstance(device, str) and ":" in device:
        idx = int(device.split(":", 1)[1])

    def props():
        return torch.cuda.get_device_properties(idx)

    return c.section("gpu", {
        "device": lambda: device,
        "available": lambda: torch.cuda.is_available(),
        "name": lambda: props().name,
        "vram_total_gb": lambda: round(props().total_memory / _GB, 2),
        "vram_free_now_gb": lambda: round(torch.cuda.mem_get_info(idx)[0] / _GB, 2),
        "compute_capability": lambda: f"{props().major}.{props().minor}",
        "driver": lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None,
        "cuda": lambda: torch.version.cuda,
    })


def _storage_section(c):
    """The NVMe facts that actually move tokens on this box.

    APST is the live hardware tax: MoE decode is 72-74% disk-bound, so a drive
    dropping into a low-power state between reads lands on the dominant cost.
    The two sysfs paths are the ones scripts/vaayu_optimize_os.sh writes; the
    first is the kernel-6.x location, which is this box.
    """
    from ..persist import safetensor_model_persister as smp

    def apst_us():
        for p in ("/sys/module/nvme_core/parameters/default_ps_max_latency_us",
                  "/sys/class/nvme/nvme0/power/ps_max_latency_us"):
            f = Path(p)
            if f.is_file():
                return int(f.read_text().strip())
        return None

    def devices():
        out = []
        for dev in sorted(Path("/sys/block").glob("nvme*")):
            model = dev / "device" / "model"
            sched = dev / "queue" / "scheduler"
            out.append({
                "device": dev.name,
                "model": model.read_text().strip() if model.is_file() else None,
                "scheduler": (re.search(r"\[(\w+)\]", sched.read_text()).group(1)
                              if sched.is_file()
                              and re.search(r"\[(\w+)\]", sched.read_text())
                              else None),
            })
        return out

    latency = c.safe("storage.apst_ps_max_latency_us", apst_us)
    return {
        "devices": c.safe("storage.devices", devices),
        "apst_ps_max_latency_us": latency,
        # APST active means the drive is allowed to sleep between reads.
        "apst_active": None if latency is None else latency != 0,
        **c.section("storage", {
            "o_direct": lambda: bool(smp._USE_DIRECT_IO),
            "io_uring": lambda: bool(smp._USE_IOURING),
            "ring_initialized": lambda: bool(smp._RING_INITIALIZED),
            # Sized to the DRIVE's concurrency knee, not to the caller pool:
            # this NVMe plateaus at ~7.1 GB/s around 2 readers and falls to
            # ~5.15 at 8 (P8, 2026-09-03).
            "ring_pool_n": lambda: smp._ring_pool_size(),
            "nvme_read_concurrency": lambda: smp._NVME_READ_CONCURRENCY,
            "files_registered": lambda: bool(smp._FILES_REGISTERED),
        }),
    }


def _model_section(c, model):
    geo = getattr(model, "_moe_geometry", None) or {}
    blocks = list(getattr(model, "_moe_blocks", None) or [])
    # Direct attribute access, deliberately -- NOT getattr(..., False). A
    # missing attribute must raise so it lands in collect_errors and is_moe
    # comes out None. A model we failed to read is UNKNOWN, not dense: a
    # confidently wrong "dense" beside a silent gap is exactly the kind of
    # fact this module exists to never report. See the Task 6 correction.
    is_moe = c.safe("model.is_moe", lambda: model.is_moe)

    def forward():
        if is_moe is None:
            return None          # we could not read the model; UNKNOWN, not dense
        if not is_moe:
            return "dense"
        # REPORTED, not re-derived (spec 9.2).
        return blocks[0].forward_name if blocks else None

    def on_disk_gb():
        path = Path(getattr(model, "name_or_path", "") or "")
        if not path.is_dir():
            return None
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / _GB, 2)

    return c.section("model", {
        "arch": lambda: (getattr(model.config, "architectures", None) or [None])[0],
        "loader_strategy": lambda: type(model.loader_strategy).__name__,
        "is_moe": lambda: is_moe,
        "expert_style": lambda: blocks[0].expert_style if blocks else None,
        "forward": forward,
        "num_layers": lambda: getattr(model.config, "num_hidden_layers", None),
        "num_moe_blocks": lambda: len(blocks) or None,
        "num_moe_layers": lambda: geo.get("num_moe_layers"),
        "num_experts": lambda: geo.get("num_experts"),
        "top_k": lambda: geo.get("top_k"),
        "expert_bytes": lambda: geo.get("expert_bytes"),
        "working_set_gb": lambda: (round(geo["working_set_bytes"] / _GB, 2)
                                   if geo.get("working_set_bytes") else None),
        "on_disk_gb": on_disk_gb,
    })


def _budgets_section(c, model):
    return c.section("budgets", {
        "max_seq_len": lambda: model.max_seq_len,
        "kv_cache_mode": lambda: model.kv_cache_mode,
        "reserve_gib": lambda: round(model._vram_plan.reserve_bytes / _GIB, 2),
        "pinning_budget_gib": lambda: round(model._vram_plan.pinning_budget / _GIB, 2),
        "s_max": lambda: model._vram_plan.s_max,
        "plan_reason": lambda: model._vram_plan.reason,
        "expert_ram_cache_gb": lambda: round(model._expert_cache.budget_bytes / _GB, 2),
        "max_per_layer": lambda: model._expert_cache.max_per_layer,
        "expert_vram_cache_gb": lambda: round(
            model._expert_vram_cache.budget_bytes / _GB, 2),
        "prefix_cache_gb": lambda: round(model._prefix_cache.budget_bytes / _GB, 2),
    })


def _config_section(c, config):
    def dump():
        raw = asdict(config) if is_dataclass(config) else dict(vars(config))
        for field in _PATH_FIELDS:
            if field in raw:
                raw[field] = redact_path(raw[field])
        return raw

    return c.safe("config", dump) or {}


def _env_section(c):
    def dump():
        found = {k: os.environ.get(k) for k in _KNOWN_VAAYU_VARS}
        # Plus anything else the user set that we do not know about -- a flag
        # added after this list was written is exactly what a bug report needs.
        found.update({k: v for k, v in os.environ.items()
                      if k.startswith("VAAYU_")})
        return dict(sorted(found.items()))

    return c.safe("env", dump) or {}


def collect(model, config, warnings=()) -> dict:
    """The whole snapshot. Never raises."""
    c = _Collector()
    device = getattr(config, "device", None) or "cuda:0"
    data = {
        "vaayu": _vaayu_section(c),
        "python": _python_section(c),
        "host": _host_section(c),
        "gpu": _gpu_section(c, device),
        "storage": _storage_section(c),
        "model": _model_section(c, model),
        "budgets": _budgets_section(c, model),
        "config": _config_section(c, config),
        "env": _env_section(c),
        "warnings": list(warnings),
        "collect_errors": c.errors,
    }
    return data


def write(path, data) -> None:
    """Write `env.json`. Never raises -- see the module docstring."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except (OSError, TypeError, ValueError) as e:
        print(f"[Vaayu-Server] WARNING: could not write {path} "
              f"({type(e).__name__}: {e}). Triage bundles from this run will "
              f"not describe the machine.")
