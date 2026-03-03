#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-centralized

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git aws-cli
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn xgboost shap matplotlib seaborn

# --- Download full dataset from S3 ---
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/raw/" /opt/fedcost/data/ --recursive

echo "Centralized node bootstrap complete"
