#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-${HOSPITAL_NAME}

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git aws-cli
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn

# --- Download hospital data partition from S3 ---
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/partitions/${HOSPITAL_NAME}.csv" /opt/fedcost/data/

echo "Hospital ${HOSPITAL_NAME} bootstrap complete"
