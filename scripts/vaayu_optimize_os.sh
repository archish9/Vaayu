#!/usr/bin/env bash
# Usage: sudo bash scripts/vaayu_optimize_os.sh /path/to/model/directory
set -euo pipefail

MODEL_PATH="${1:?Usage: sudo $0 <model_directory>}"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0 $MODEL_PATH"
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
APST_PATH=""
if [[ -f "/sys/module/nvme_core/parameters/default_ps_max_latency_us" ]]; then
    APST_PATH="/sys/module/nvme_core/parameters/default_ps_max_latency_us"  # kernel 6.x
elif [[ -f "/sys/class/nvme/$NVME_CHAR/power/ps_max_latency_us" ]]; then
    APST_PATH="/sys/class/nvme/$NVME_CHAR/power/ps_max_latency_us"          # kernel 4.x/5.x
fi

if [[ -n "$APST_PATH" ]]; then
    echo "Disabling APST..."
    echo 0 | tee "$APST_PATH" > /dev/null
    echo "  Immediate: done (path: $APST_PATH)"
    echo "  Note: resets on reboot. For permanent fix, add nvme_core.default_ps_max_latency_us=0"
    echo "  to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then run: sudo update-grub"
else
    echo "  APST sysfs path not found on this kernel — skipping"
fi

echo ""
echo "Vaayu OS optimization complete!"
echo ""
echo "Verify with:"
echo "  cat /sys/block/$NVME_DEV/queue/scheduler   # should output [none]"
