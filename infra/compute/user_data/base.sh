#!/bin/bash
set -euo pipefail

# --- System update ---
dnf update -y
dnf install -y python3.13 python3.13-pip git aws-cli

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-${NODE_NAME}

# --- Python ---
python3.13 -m pip install --upgrade pip

echo "[base] System update + Tailscale complete for fedcost-${NODE_NAME}"
