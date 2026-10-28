#!/usr/bin/env bash
# Install Vaayu as a systemd system service: survives logout, auto-restarts on
# death, and lives in system.slice — OUT of systemd-oomd's user.slice watch, so
# oomd can no longer kill the terminal scope out from under it.
#
# Usage:
#   sudo bash scripts/install-vaayu-service.sh [--headless | --gui]
#
#   --headless  (default) no desktop logged in; reach the box over SSH/Tailscale.
#               Vaayu gets almost all RAM (reserve only for the OS).
#   --gui       a graphical desktop (e.g. rustdesk) stays logged in alongside
#               Vaayu; a larger RAM reserve is left so the desktop is not starved.
set -euo pipefail

MODE="headless"
case "${1:-}" in
    --gui) MODE="gui" ;;
    --headless|"") MODE="headless" ;;
    *) echo "Usage: sudo $0 [--headless | --gui]" >&2; exit 1 ;;
esac

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0 ${1:-}" >&2
    exit 1
fi

# The invoking (non-root) user owns the venv, the repo and the model paths.
RUN_USER="${SUDO_USER:-$(logname 2>/dev/null || true)}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = "root" ]; then
    echo "ERROR: run with sudo as your normal user (need \$SUDO_USER), not as root directly." >&2
    exit 1
fi
RUN_GROUP="$(id -gn "$RUN_USER")"

# Repo root = parent of this script's dir. venv python must exist there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv python not found at $VENV_PY — create the venv first (see install.sh)." >&2
    exit 1
fi

# ── Memory caps DERIVED from this machine's RAM (never hardcoded) ────────────
# MemoryHigh = soft: above it the cgroup is throttled + reclaimed (page cache
#              first), not killed.  MemoryMax = hard: above it the kernel OOM-kills
#              inside THIS cgroup only (Vaayu, never the desktop).
TOTAL_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
if [ "$MODE" = "gui" ]; then
    RESERVE_GB=6        # leave room for GNOME + remote-desktop + a terminal
else
    RESERVE_GB=3        # headless: only the OS, sshd, tailscaled
fi
MAX_GB=$(( TOTAL_GB - RESERVE_GB ))
HIGH_GB=$(( MAX_GB - 2 ))
if [ "$MAX_GB" -lt 8 ]; then
    echo "ERROR: only ${TOTAL_GB}G RAM — too little to cap safely (need the model's working set + ${RESERVE_GB}G reserve)." >&2
    exit 1
fi

UNIT=/etc/systemd/system/vaayu.service
echo "[Vaayu] Installing $UNIT"
echo "        user=$RUN_USER group=$RUN_GROUP  repo=$REPO_DIR"
echo "        mode=$MODE  RAM=${TOTAL_GB}G  MemoryHigh=${HIGH_GB}G  MemoryMax=${MAX_GB}G"

cat > "$UNIT" <<EOF
[Unit]
Description=Vaayu LLM inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_PY -m vaayu.server

# recover after any death (the whole point)
Restart=always
RestartSec=5

# system.slice is not oomd-managed by default; be explicit
ManagedOOMMemoryPressure=auto
ManagedOOMSwap=auto

# bound memory to THIS cgroup (derived from ${TOTAL_GB}G total, ${MODE} reserve ${RESERVE_GB}G)
MemoryHigh=${HIGH_GB}G
MemoryMax=${MAX_GB}G

# lose to the desktop if the kernel ever has to pick (only on true global OOM)
OOMScoreAdjust=-200

# streaming a large model: allow load + warmup before systemd calls it failed
TimeoutStartSec=600
# let an in-flight generation drain on stop/restart instead of hard-killing it
TimeoutStopSec=120
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vaayu.service

echo ""
echo "[Vaayu] Service installed and started. Verify:"
echo "    systemctl status vaayu.service --no-pager"
echo "    oomctl | grep -A3 -i vaayu   # Vaayu should NOT be listed (out of oomd's reach)"
echo "    journalctl -u vaayu.service -f   # watch load + warmup until 'Uvicorn running'"
