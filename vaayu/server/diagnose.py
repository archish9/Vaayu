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

from . import config, envinfo


def taxes(data: dict, cfg) -> list:
    """Findings an operator can act on, most actionable first.

    Every check must be able to stay QUIET. A check that always prints is a
    banner, and banners are what audit item 11 was about.
    """
    out = []
    storage = data.get("storage") or {}
    gpu = data.get("gpu") or {}
    model = data.get("model") or {}

    if storage.get("apst_active"):
        us = storage.get("apst_ps_max_latency_us")
        out.append(
            f"NVMe APST is active (ps_max_latency_us={us}). The drive drops into "
            f"a low-power state between reads, and this engine's decode is "
            f"disk-stall dominated. Fix: run scripts/vaayu_optimize_os.sh once "
            f"(needs sudo).")

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

    if getattr(cfg, "DRAFTER_MODEL_PATH", None) and model.get("is_moe"):
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
