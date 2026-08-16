# Schedule Plan A: OS-Level NVMe Tweaks

**Priority:** 1 (Do it immediately)
**Effort:** 5 minutes
**Risk:** Zero

## Objective
Apply system-level configurations to the Linux kernel to optimize NVMe performance. Because these are OS-level tweaks, they are completely agnostic to the codebase and will survive the upcoming EXL3 architectural refactor.

## Cross-Machine Strategy
Vaayu runs on different machines with different NVMe device names (`nvme0n1`, `nvme1n1`, etc.). Implementation uses two complementary layers:

1. **`scripts/vaayu_optimize_os.sh`** — a one-time setup script that auto-detects the NVMe device from the model path and applies both optimizations permanently. Users run this once after cloning the repo (documented in README).
2. **Python startup check** — a passive, zero-root check inside `AutoModel.from_pretrained()` that warns the user if the machine is unoptimized, with the exact fix command to run.

This way, users who run the setup script get silence and full performance. Users who skip it get a clear warning on the first inference run.

---

## Implementation Steps

### 1. Create `scripts/vaayu_optimize_os.sh`

Auto-detects the NVMe device hosting the model path, then applies scheduler and APST optimizations. Idempotent — safe to re-run.

```bash
#!/usr/bin/env bash
# Usage: sudo bash scripts/vaayu_optimize_os.sh /path/to/model/directory
set -euo pipefail

MODEL_PATH="${1:?Usage: sudo $0 <model_directory>}"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo $0 $MODEL_PATH)"
    exit 1
fi

# Resolve the NVMe block device that hosts the model path
BLOCK_DEV=$(df --output=source "$MODEL_PATH" | tail -1)
NVME_DEV=$(lsblk -no pkname "$BLOCK_DEV" 2>/dev/null || echo "$BLOCK_DEV" | sed 's|/dev/||;s|p[0-9]*$||')
NVME_CHAR=$(echo "$NVME_DEV" | sed 's/n[0-9]*$//')  # e.g. nvme0n1 → nvme0

if [[ ! "$NVME_DEV" =~ ^nvme ]]; then
    echo "ERROR: $MODEL_PATH does not live on an NVMe device (detected: $NVME_DEV). Skipping." >&2
    exit 1
fi

echo "Detected NVMe block device : /dev/$NVME_DEV"
echo "Detected NVMe char device  : /dev/$NVME_CHAR"
echo ""

# ── 1. I/O Scheduler ────────────────────────────────────────────────────────
echo "Setting I/O scheduler to none..."
echo none | tee /sys/block/"$NVME_DEV"/queue/scheduler > /dev/null
echo "  Immediate: done"

UDEV_RULE='/etc/udev/rules.d/60-vaayu-nvme.rules'
if [[ ! -f "$UDEV_RULE" ]]; then
    echo 'ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"' \
        | tee "$UDEV_RULE" > /dev/null
    udevadm control --reload-rules
    udevadm trigger
    echo "  Permanent udev rule: installed"
else
    echo "  Permanent udev rule: already present, skipped"
fi

# ── 2. APST (power management) ──────────────────────────────────────────────
APST_PATH="/sys/class/nvme/$NVME_CHAR/power/ps_max_latency_us"
if [[ -f "$APST_PATH" ]]; then
    echo "Disabling APST..."
    echo 0 | tee "$APST_PATH" > /dev/null
    echo "  Immediate: done"
    echo "  Note: resets on reboot. For permanent fix, add nvme_core.default_ps_max_latency_us=0"
    echo "  to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then run: sudo update-grub"
else
    echo "  APST sysfs path not found at $APST_PATH — skipping"
fi

echo ""
echo "Vaayu OS optimization complete!"
echo ""
echo "Verify with:"
echo "  cat /sys/block/$NVME_DEV/queue/scheduler   # should output [none]"
```

### 2. Add Python Startup Check to `AutoModel.from_pretrained()`

Passive, zero-root check. Runs before model load. Silent on optimized machines, warns on unoptimized ones.

```python
import subprocess
import pathlib

def _check_nvme_scheduler(model_path: str) -> None:
    try:
        dev = subprocess.check_output(
            ["df", "--output=source", model_path], text=True
        ).strip().splitlines()[-1]
        nvme = subprocess.check_output(
            ["lsblk", "-no", "pkname", dev], text=True
        ).strip() or dev.replace("/dev/", "")
        if not nvme.startswith("nvme"):
            return
        scheduler = pathlib.Path(f"/sys/block/{nvme}/queue/scheduler").read_text()
        if "[none]" not in scheduler:
            print(f"[Vaayu Warning] {nvme} I/O scheduler is not optimized "
                  f"(current: {scheduler.strip()}).")
            print(f"  You are losing ~5% NVMe throughput.")
            print(f"  Fix: sudo bash scripts/vaayu_optimize_os.sh {model_path}")
    except Exception:
        pass  # non-NVMe or detection failed — silent
```

Call `_check_nvme_scheduler(model_path)` at the top of `AutoModel.from_pretrained()`, before any model loading begins.

---

## Verification
- Run `cat /sys/block/<detected_device>/queue/scheduler` — should output `[none]`.
- Re-run `test_quick.py` with an unoptimized machine to confirm the warning prints.
- Re-run `test_quick.py` after running the script to confirm the warning is gone and `io_wait` drops 2-5%.
