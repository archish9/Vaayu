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
  * **Nothing here may stop a boot.** Every value read off the model, config
    or environment goes through `_Collector.safe`, which records the failure
    in `collect_errors` and yields None -- imports (`torch`, the persister
    module) and string-to-int parsing (a device index off a typo'd
    `config_local.py`) included. The one thing that runs outside that
    per-field wrapper is `_scrub_paths` itself (below): it only rewrites
    values already collected, carries its own depth/cycle guards so it
    cannot blow the recursion limit, and its one external call in `collect()`
    is wrapped separately.

A third rule, added after review round 1 and sharpened in round 2: **this
file gets zipped and mailed to a stranger, so it must not leak the
filesystem it ran on** -- usernames and absolute paths, never the machine's
identity as a whole. Every value passes through `_scrub_paths` just before
`collect()` returns -- one regex pass over the whole report that rewrites any
absolute-path-shaped substring (whole field or embedded in a sentence, e.g.
an exception message or a warning) into `redact_path()`'s basename+hash form.
The per-field `_PATH_FIELDS` redaction below still runs first and stays: it
is defence in depth, not replaced by the scrub -- a known path field gets the
clean {"name", "hash"} shape, and the scrub is the net under it that catches
whatever the allowlist misses (a future config field, an OSError's
`strerror` naming a path, `sys.executable`). It deliberately does NOT touch
`https://`/`http://` URLs (public, and a maintainer needs the real host) or
paths under `/sys/`, `/proc/`, `/dev/` (kernel-owned, name no user). And it
deliberately KEEPS the checkpoint's basename -- on this box that is
`archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw`, the user's own model-hub
handle. That is correct: triage needs to know what checkpoint this is, and a
public hub handle is not privacy-sensitive the way a home directory is. What
never survives the scrub is the PATH to it.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import fields as dc_fields, is_dataclass
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

# An absolute unix path, standalone or embedded in a sentence: a leading "/"
# not itself preceded by a word/`.`/`-` character (so it doesn't fire inside
# a ratio like "3/10" or "hit/miss"), followed by at least one more "/" (so a
# bare "/" is never mistaken for a path). Deliberately does NOT exclude `:`
# in the lookbehind (review round 2, item 1): a colon-joined path list like
# "PYTHONPATH=/home/a/x:/home/b/y" has its second element starting right
# after a ':', and import-error text is exactly where colon-joined search
# paths show up. `https://` and `file:///` still match starting at their
# second/third slash regardless -- URLs are protected separately, below.
_PATH_RE = re.compile(r'(?<![\w.\-])/(?:[\w.\-]+/)+[\w.\-]*')

# Public information a maintainer needs verbatim -- never redacted.
_URL_RE = re.compile(r'https?://\S+')

# Kernel-owned paths. None of them names a user or a checkpoint; redacting
# them only makes a storage-section failure harder to read (two APST sysfs
# candidates that would otherwise collide on the same basename).
_UNSCRUBBED_PREFIXES = ("/sys/", "/proc/", "/dev/")

# Trailing punctuation that prose puts after a path but that isn't PART of
# the path -- swallowing it into the match would make "…/y.json" and
# "…/y.json." (end of a sentence) hash differently.
_TRAILING_PUNCT = ".,;:)]"

_MAX_SCRUB_DEPTH = 50   # generous for a report this shallow; a cycle guard,
                        # not a real limit -- see _scrub_paths.


def redact_path(p):
    """An absolute path reduced to what triage needs and nothing more.

    The basename identifies the checkpoint; the hash distinguishes two
    different paths with the same basename. Neither leaks a username.
    Trailing "/" is stripped before BOTH are computed (not just the name) --
    otherwise a hand-typed `MODEL_PATH=.../Big-exl3/` in config_local.py
    hashes differently from the same checkpoint without the trailing slash,
    and one checkpoint looks like two in a bundle.
    """
    if not p:
        return None
    p = str(p)
    stripped = p.rstrip("/") or p          # "/" itself stays "/"
    return {"name": os.path.basename(stripped) or stripped,
            "hash": hashlib.sha1(stripped.encode("utf-8", "replace")).hexdigest()[:10]}


def _redact_match(m):
    raw = m.group(0)
    trail = ""
    while raw and raw[-1] in _TRAILING_PUNCT:
        trail = raw[-1] + trail
        raw = raw[:-1]
    if not raw or raw.startswith(_UNSCRUBBED_PREFIXES):
        return m.group(0)
    r = redact_path(raw)
    return f"<{r['name']}:{r['hash']}>{trail}" if r else m.group(0)


def _scrub_string(s):
    # Mask URLs out before the path regex runs, then restore them verbatim --
    # https://host/a/b is public and reads as broken once its scheme/host is
    # torn away from its path.
    urls = []

    def stash(m):
        urls.append(m.group(0))
        return f"\0URL{len(urls) - 1}\0"

    masked = _URL_RE.sub(stash, s)
    scrubbed = _PATH_RE.sub(_redact_match, masked)
    for i, u in enumerate(urls):
        scrubbed = scrubbed.replace(f"\0URL{i}\0", u)
    return scrubbed


def _scrub_paths(value, _seen=None, _depth=0):
    """Recursively rewrite any absolute-path-shaped substring in `value` into
    `redact_path()`'s basename+hash form, serialised inline as `<name:hash>`
    so it still reads as a sentence when the match is embedded in one (a
    `collect_errors` message, a `warnings` line). See the module docstring.

    Bounded on two axes so it can never itself be the thing that makes
    `collect()` raise: `_depth` caps recursion (nothing this module builds is
    remotely this deep; a cycle or pathological input gets a placeholder
    instead of a RecursionError), and `_seen` (by object id) catches a
    self-referential container the same way.
    """
    if _seen is None:
        _seen = set()
    if _depth > _MAX_SCRUB_DEPTH:
        return "<max-depth-exceeded>"
    if isinstance(value, (dict, list, tuple, set)):
        vid = id(value)
        if vid in _seen:
            return "<cyclic-reference>"
        _seen = _seen | {vid}
        if isinstance(value, dict):
            return {k: _scrub_paths(v, _seen, _depth + 1) for k, v in value.items()}
        return [_scrub_paths(v, _seen, _depth + 1) for v in value]
    if isinstance(value, str):
        return _scrub_string(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    # An unknown leaf -- Path, bytes, or any object whose repr names a path.
    # write()'s json.dumps(default=str) would stringify this too, but only
    # AFTER this scrub runs, which is exactly how a raw path used to slip
    # through untouched (review round 2, item 2).
    return _scrub_paths(str(value), _seen, _depth + 1)


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
    repo = c.safe("vaayu._repo_path", lambda: Path(__file__).resolve().parents[2])
    # I5: `git -C <path> ...` does not require `path` itself to be a repo --
    # if it is not, git walks UP the directory tree looking for one. A pip
    # install ships setup.py, so this path is real, but for a venv nested
    # inside a user's OWN project that walk finds the user's `.git` instead:
    # env.json would then report the user's application branch and commit as
    # Vaayu's -- misleading for triage, and a privacy leak the path scrubber
    # does not touch (branch names carry ticket ids and employer names).
    # Trust the result only if `repo` is itself a checkout root; otherwise
    # "installed, not a checkout" is the honest answer.
    if repo is None or not (repo / ".git").exists():
        return {"commit": None, "dirty": None, "branch": None}

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
        # Redacted: a maintainer needs to know WHICH python and WHETHER it is
        # a venv, never WHERE it lives on this stranger's disk.
        "executable": lambda: redact_path(sys.executable),
        "is_venv": lambda: sys.prefix != sys.base_prefix,
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
    def parse_idx():
        if isinstance(device, str) and ":" in device:
            return int(device.split(":", 1)[1])
        return 0

    # A typo'd device string in the user's config_local.py (e.g. "cuda:abc")
    # must not take the whole boot down -- fall back to index 0 and let the
    # bad value surface in collect_errors instead.
    idx = c.safe("gpu.device_index", parse_idx)
    if idx is None:
        idx = 0

    gpu_fields = ("available", "name", "vram_total_gb", "vram_free_now_gb",
                  "compute_capability", "driver", "cuda")

    # Every other process holding VRAM on this device -- app.py's own
    # `_check_gpu_is_exclusive` refuses/warns off this same audit at boot, but
    # that report only reaches `collected_warnings()` (and so `env.json`) when
    # it is a BLOCKING holder printed with the literal word "WARNING". A
    # holder under `exclusive_gpu_min_mib` (a bare CUDA context is 128 MiB) is
    # still a permanent tax on the pinning budget -- gpu_audit.py's own
    # docstring -- so `diagnose`'s foreign-GPU tax needs it as DATA, not as a
    # log line it would have to re-parse. `gpu_audit.audit` never raises (it
    # already fails closed to `[]` with no nvidia-smi/GPU/driver), but it is
    # still routed through `c.safe` like every other field here.
    def foreign_processes():
        from .. import gpu_audit
        return [{"pid": u.pid, "cmd": u.name, "gib": round(u.gib, 2)}
                for u in gpu_audit.audit(device=device)]

    fp = c.safe("gpu.foreign_processes", foreign_processes)

    def import_torch():
        import torch
        return torch

    torch = c.safe("gpu._import_torch", import_torch)
    if torch is None:
        return {"device": device, "foreign_processes": fp or [],
                **{k: None for k in gpu_fields}}

    def props():
        return torch.cuda.get_device_properties(idx)

    return {"device": device, "foreign_processes": fp or [], **c.section("gpu", {
        "available": lambda: torch.cuda.is_available(),
        "name": lambda: props().name,
        "vram_total_gb": lambda: round(props().total_memory / _GB, 2),
        "vram_free_now_gb": lambda: round(torch.cuda.mem_get_info(idx)[0] / _GB, 2),
        "compute_capability": lambda: f"{props().major}.{props().minor}",
        "driver": lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None,
        "cuda": lambda: torch.version.cuda,
    })}


def _storage_section(c):
    """The NVMe facts that actually move tokens on this box.

    APST is the live hardware tax: MoE decode is 72-74% disk-bound, so a drive
    dropping into a low-power state between reads lands on the dominant cost.
    The two sysfs paths are the ones scripts/vaayu_optimize_os.sh writes; the
    first is the kernel-6.x location, which is this box.
    """
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
    result = {
        "devices": c.safe("storage.devices", devices),
        "apst_ps_max_latency_us": latency,
        # APST active means the drive is allowed to sleep between reads.
        "apst_active": None if latency is None else latency != 0,
    }

    smp_fields = ("o_direct", "io_uring", "ring_initialized", "ring_pool_n",
                  "nvme_read_concurrency", "files_registered")

    def import_smp():
        from ..persist import safetensor_model_persister as smp
        return smp

    smp = c.safe("storage._import_persister", import_smp)
    if smp is None:
        result.update({k: None for k in smp_fields})
    else:
        result.update(c.section("storage", {
            "o_direct": lambda: bool(smp._USE_DIRECT_IO),
            "io_uring": lambda: bool(smp._USE_IOURING),
            "ring_initialized": lambda: bool(smp._RING_INITIALIZED),
            # Sized to the DRIVE's concurrency knee, not to the caller pool:
            # this NVMe plateaus at ~7.1 GB/s around 2 readers and falls to
            # ~5.15 at 8 (P8, 2026-09-03).
            "ring_pool_n": lambda: smp._ring_pool_size(),
            "nvme_read_concurrency": lambda: smp._NVME_READ_CONCURRENCY,
            "files_registered": lambda: bool(smp._FILES_REGISTERED),
        }))
    return result


def _model_section(c, model):
    # Direct attribute access, deliberately -- NOT getattr(model, x, default).
    # A missing attribute must raise so it lands in collect_errors, and a
    # legitimate None (a dense model's real _moe_geometry/_moe_blocks) must
    # NOT: getattr-with-default cannot tell those two apart, and a bare
    # getattr-with-default also does not catch a property that raises
    # something other than AttributeError -- c.safe does.
    geo = c.safe("model._moe_geometry", lambda: model._moe_geometry) or {}
    blocks = list(c.safe("model._moe_blocks", lambda: model._moe_blocks) or [])
    is_moe = c.safe("model.is_moe", lambda: model.is_moe)

    def forward():
        if is_moe is None:
            return None          # we could not read the model; UNKNOWN, not dense
        if not is_moe:
            return "dense"
        # REPORTED, not re-derived (spec 9.2) -- but this is the single most
        # load-bearing field in the file, so a null here gets an explicit,
        # named reason instead of looking like every other silent gap.
        if not blocks:
            c.errors.append(
                "model.forward: is_moe is True but no MoE blocks were "
                "recorded (model not patched, or detect_moe_blocks found none)")
            return None
        name = blocks[0].forward_name
        if name is None:
            c.errors.append(
                "model.forward: MoE block present but forward_name is unset "
                "(block was not patched via _make_sparse_moe_forward)")
        return name

    def on_disk_gb():
        # checkpoint_path is where the split shards live; model_local_path is
        # the pre-split fallback (vaayu_base.py, VaayuBaseModel.__init__).
        # There is no `name_or_path` attribute on the real model -- that was
        # the brief's bug (review round 1, Critical 1): Path("") is Path("."),
        # so an empty/missing value walked the server's CWD, not the
        # checkpoint, and reported whatever the repo happens to weigh. Guard
        # the empty/missing case BEFORE constructing a Path, not after.
        p = c.safe("model.checkpoint_path", lambda: model.checkpoint_path)
        if not p:
            p = c.safe("model.model_local_path", lambda: model.model_local_path)
        if not p:
            return None
        path = Path(p)
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
    # Item 7: on a dense model, _expert_cache/_expert_vram_cache are
    # LEGITIMATELY None (vaayu_base.py sets them None and only the MoE path
    # ever assigns a real cache) -- a supported configuration, not a broken
    # one. Guard on that the same way _model_section guards _moe_geometry, so
    # a dense report gets a clean null with no collect_errors entry rather
    # than three AttributeErrors on a machine that ran fine.
    def cache_gb(cache):
        return round(cache.budget_bytes / _GB, 2) if cache is not None else None

    def _resolved_kv_mode(model):
        # Import inside the field, like every other heavy import here, so a
        # failure lands in collect_errors instead of nulling the section.
        from ..vaayu_base import resolve_kv_cache_mode
        return resolve_kv_cache_mode(model.kv_cache_mode, model.loader_strategy)

    return c.section("budgets", {
        "max_seq_len": lambda: model.max_seq_len,
        # The RESOLVED mode, not the requested one: "auto" is a request, and
        # this file's premise is that every field is derived (module
        # docstring). The requested value stays readable in the `config`
        # section. Asks vaayu_base's own resolver -- a second copy of that
        # branch would drift and misreport the dtype that ran.
        "kv_cache_mode": lambda: _resolved_kv_mode(model),
        "reserve_gib": lambda: round(model._vram_plan.reserve_bytes / _GIB, 2),
        "pinning_budget_gib": lambda: round(model._vram_plan.pinning_budget / _GIB, 2),
        "s_max": lambda: model._vram_plan.s_max,
        "plan_reason": lambda: model._vram_plan.reason,
        "expert_ram_cache_gb": lambda: cache_gb(model._expert_cache),
        "max_per_layer": lambda: (model._expert_cache.max_per_layer
                                  if model._expert_cache is not None else None),
        "expert_vram_cache_gb": lambda: cache_gb(model._expert_vram_cache),
        "prefix_cache_gb": lambda: cache_gb(model._prefix_cache),
    })


def _config_section(c, config):
    # Per-FIELD, like every other section -- NOT a single dump() wrapped in
    # one c.safe. config is the most load-bearing section in the file (this
    # module exists because the user's real settings live in a gitignored
    # config_local.py); one unrepresentable field must not null all of it.
    def field_names():
        if is_dataclass(config):
            return [f.name for f in dc_fields(config)]
        # `diagnose.main()` passes the `config` MODULE, not a ServerConfig
        # instance -- vars(module) is the whole namespace (imports, dunders,
        # helper functions), not just the settings. config.py's own
        # convention (README.md, every value in the file) is that a setting
        # is an UPPERCASE module-level name; filtering to that is what keeps
        # this section to the settings a maintainer actually set, instead of
        # dumping `__name__`/`__builtins__`/imported modules as "config".
        return [k for k in vars(config).keys() if k.isupper()]

    names = c.safe("config._field_names", field_names) or []

    def make_fn(name):
        def fn():
            val = getattr(config, name)
            return redact_path(val) if name in _PATH_FIELDS else val
        return fn

    return c.section("config", {name: make_fn(name) for name in names})


def _env_section(c):
    # Plus anything else the user set that we do not know about -- a flag
    # added after _KNOWN_VAAYU_VARS was written is exactly what a bug report
    # needs. Per-FIELD, like _config_section above.
    names = c.safe("env._names", lambda: sorted(
        set(_KNOWN_VAAYU_VARS) | {k for k in os.environ if k.startswith("VAAYU_")}
    )) or []
    return c.section("env", {name: (lambda n=name: os.environ.get(n))
                             for name in names})


def _speculative_section(c, model):
    """What was asked for, what is running, and why not (S1b).

    Direct attribute access like `_model_section`: a model that never set the
    field is a model this server did not load, and that belongs in
    `collect_errors` rather than being flattened into a plausible-looking
    "speculation is off". The refusal reason is prose written by the gate and
    can embed the drafter's path -- `_scrub_paths` runs over the whole tree
    afterwards, so it is redacted with everything else.
    """
    status = c.safe("model.speculative_status",
                    lambda: model.speculative_status) or {}
    return c.section("speculative", {
        "requested": lambda: status.get("requested"),
        "draft_tokens": lambda: status.get("draft_tokens"),
        "active": lambda: status.get("active"),
        "refused_because": lambda: status.get("refused_because"),
        # S1d: the RUNTIME decision, absent on a load that never speculated.
        # Null rather than omitted, so a script can index it on any report.
        "calibration": lambda: status.get("calibration"),
    })


def collect(model, config, warnings=()) -> dict:
    """The whole snapshot. Never raises."""
    c = _Collector()
    # `config` is a ServerConfig dataclass (lowercase `device`) on every real
    # server boot, but `diagnose` is the first caller to pass the config
    # MODULE instead (uppercase `DEVICE`) -- accept either spelling here so
    # `diagnose` audits the GPU actually configured rather than always
    # falling back to index 0 on a box set up for `cuda:1`. Narrow to this
    # one read; the module's other lowercase `getattr(config, ...)` reads are
    # a separate question this fix does not touch.
    device = c.safe("config.device",
                    lambda: (getattr(config, "device", None)
                             or getattr(config, "DEVICE", None))) or "cuda:0"
    data = {
        "vaayu": _vaayu_section(c),
        "python": _python_section(c),
        "host": _host_section(c),
        "gpu": _gpu_section(c, device),
        "storage": _storage_section(c),
        "config": _config_section(c, config),
        "env": _env_section(c),
        "warnings": list(warnings),
    }
    # `model=None` is `diagnose`'s case: no weights are loaded, so there is
    # nothing to derive. OMIT the two sections rather than filling
    # collect_errors with an AttributeError per field -- a report that reads
    # as broken is worse than a shorter one. (Same principle Item 7 of the L1
    # fix wave applied inside _budgets_section for a dense model's absent
    # expert caches.)
    if model is not None:
        data["model"] = _model_section(c, model)
        data["budgets"] = _budgets_section(c, model)
        data["speculative"] = _speculative_section(c, model)
    data["collect_errors"] = c.errors
    # Scrub LAST, over the whole tree: collect_errors and warnings are prose
    # that can embed a path (an exception's message, an engine warning
    # naming a checkpoint), so this cannot be folded into any one section.
    # _scrub_paths has its own depth/cycle guards, but it still gets its own
    # try/except here -- this module promises collect() never raises, and
    # that promise has to survive even a bug in the scrubber itself. If it
    # somehow fails, a report that says so beats one that leaked or crashed.
    try:
        return _scrub_paths(data)
    except Exception as e:                      # noqa: BLE001 -- recorded, not raised
        return {"collect_errors": c.errors + [f"_scrub_paths: {type(e).__name__}: {e}"]}


def write(path, data) -> None:
    """Write `env.json`. Never raises -- see the module docstring."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:                      # noqa: BLE001 -- warned, not raised
        print(f"[Vaayu-Server] WARNING: could not write {path} "
              f"({type(e).__name__}: {e}). Triage bundles from this run will "
              f"not describe the machine.")
