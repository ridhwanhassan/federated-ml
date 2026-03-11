#!/bin/bash
set -euo pipefail

# --- System update ---
dnf update -y
dnf install -y python3.13 python3.13-pip git aws-cli

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-${NODE_NAME}

# --- uv (Python package manager) ---
curl -LsSf https://astral.sh/uv/install.sh | sh
cp /root/.local/bin/uv /usr/local/bin/uv
chmod 755 /usr/local/bin/uv

echo "[base] System update + Tailscale + uv complete for fedcost-${NODE_NAME}"
