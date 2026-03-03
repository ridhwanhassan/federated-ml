#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-fl-server

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn

echo "FL server bootstrap complete"
