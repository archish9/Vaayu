"""`./venv/bin/python -m vaayu.server.diagnose`

The §4 environment snapshot without loading weights, plus the taxes an operator
can actually do something about.

A separate module, not a flag on `__main__.py`: that module's docstring promises
"No flags: every value comes from config.py", and one saved keystroke is not
worth amending it.

Runs in seconds. This is what a user runs when the server will not start at all,
which is exactly when `logs/run-*/env.json` does not exist.
"""
import json
import sys
from pathlib import Path

from . import config, envinfo


def _checkpoint_looks_like_moe(model_path) -> bool:
    """MoE-ness of a checkpoint, read from its `config.json`, WITHOUT loading
    weights -- `diagnose` never loads the model (that is the point of it), so
    `data["model"]["is_moe"]` (the field `envinfo._model_section` derives off
    the loaded model) never exists here. Reading it off disk is the only way
    this tax can ever fire for real.

    Detect, don't match architecture strings (CLAUDE.md 5): any top-level key
    whose name contains "expert" and whose value is an int is the honest
    test. Verified on this box's three checkpoints -- Qwen3-235B carries
    `num_experts`/`num_experts_per_tok`, MiniMax-M2.7 carries
    `num_local_experts`/`num_experts_per_tok`, and Llama-3.3-70B (dense)
    carries neither -- but an allow-list of the specific key names would
    silently miss the next MoE family's own naming.

    Fail-soft on every axis: no `model_path`, no `config.json`, an unreadable
    one, or one that isn't valid JSON all resolve to False (not MoE), never
    raise. This runs in the tool people reach for when the server will not
    start at all, so it must exit 0 and print a report regardless of what --
    if anything -- is sitting at MODEL_PATH.
    """
    if not model_path:
        return False
    try:
        data = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    # `not isinstance(v, bool)`: bool IS an int in Python, so a dense config
    # carrying "expert_parallel": false read as MoE. `v > 0`: "num_experts": 0
    # is a dense model spelling out that it has none. Either one used to bill
    # the drafter-on-MoE tax to a model that is not MoE at all.
    return any("expert" in k.lower() and isinstance(v, int)
               and not isinstance(v, bool) and v > 0
               for k, v in data.items())


def taxes(data: dict, cfg) -> list:
    """Findings an operator can act on, most actionable first.

    Every check must be able to stay QUIET. A check that always prints is a
    banner, and banners are what audit item 11 was about.
    """
    out = []
    storage = data.get("storage") or {}
    gpu = data.get("gpu") or {}

    if storage.get("apst_active"):
        us = storage.get("apst_ps_max_latency_us")
        out.append(
            f"NVMe APST is active (ps_max_latency_us={us}). Minor: measured with "
            f"no decode or TTFT cost here, because decode keeps the drive's idle "
            f"gaps shorter than APST's sleep timeout. To disable it anyway: "
            f"scripts/vaayu_optimize_os.sh (needs sudo).")

    for p in gpu.get("foreign_processes") or []:
        out.append(
            f"Another process holds GPU memory: {p.get('cmd')} (pid "
            f"{p.get('pid')}, {p.get('gib')} GiB). Budgets are sized ONCE at "
            f"load, so this is subtracted from the expert cache for the whole "
            f"run even if the process later exits.")

    sched = storage.get("scheduler")
    if sched not in (None, "none"):
        out.append(
            f"The NVMe I/O scheduler is '{sched}', not 'none'. Reads here are "
            f"already O_DIRECT + io_uring; a scheduler in front of them adds "
            f"latency without reordering anything useful.")

    # `data["model"]` never exists here (`main()` always calls `collect(None,
    # ...)`), so MoE-ness has to come from the checkpoint's own config.json,
    # not from `data`.
    if (getattr(cfg, "DRAFTER_MODEL_PATH", None)
            and _checkpoint_looks_like_moe(getattr(cfg, "MODEL_PATH", None))):
        out.append(
            "A drafter is configured and the model is MoE. Speculative decoding "
            "is refused at load on MoE targets (the Fix-#9 working-set gate), so "
            "the drafter loads to VRAM and is then discarded on every start. "
            "Unset DRAFTER_MODEL_PATH unless the target is dense.")

    if getattr(cfg, "DEFAULT_STREAM", True) is False:
        out.append(
            "DEFAULT_STREAM is False. A client that sends no max_tokens then "
            "gets NOTHING until the whole answer is generated -- measured at "
            "864 s of silence for an 809-token answer. Set DEFAULT_STREAM = True "
            "or have clients send stream=true.")

    return out


def main() -> int:
    data = envinfo.collect(None, config)
    print(json.dumps(data, indent=2, default=str))
    findings = taxes(data, config)
    print("\n=== actionable ===")
    if not findings:
        print("nothing to flag on this box.")
    for i, line in enumerate(findings, 1):
        print(f"\n{i}. {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
