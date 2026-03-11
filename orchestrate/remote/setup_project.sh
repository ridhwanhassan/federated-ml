#!/bin/bash
# Idempotent setup script — runs on each EC2 instance.
# Downloads code tarball from S3, extracts to /opt/fedcost, installs deps.
set -eu

BUCKET=$(aws ssm get-parameter \
    --name /fedcost/s3-data-bucket \
    --query Parameter.Value \
    --output text \
    --region ap-southeast-1)

echo "[setup] Bucket: ${BUCKET}"

# Download and extract code tarball
aws s3 cp "s3://${BUCKET}/deploy/fedcost-code.tar.gz" /tmp/fedcost-code.tar.gz \
    --region ap-southeast-1
mkdir -p /opt/fedcost
tar xzf /tmp/fedcost-code.tar.gz -C /opt/fedcost --strip-components=0
rm -f /tmp/fedcost-code.tar.gz

echo "[setup] Code extracted to /opt/fedcost"

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cp /root/.local/bin/uv /usr/local/bin/uv
    chmod 755 /usr/local/bin/uv
fi

# Install Python dependencies via uv
cd /opt/fedcost
uv sync 2>&1 | tail -3
chown -R ec2-user:ec2-user /opt/fedcost

echo "[setup] Dependencies installed via uv"

# Determine node role from Tailscale hostname
TS_HOSTNAME=$(tailscale status --json 2>/dev/null \
    | python3.13 -c "import sys,json; print(json.load(sys.stdin)['Self']['HostName'])")
NODE_NAME=$(echo "${TS_HOSTNAME}" | sed 's/fedcost-//')
DATA_DIR=/opt/fedcost/data
mkdir -p "${DATA_DIR}"

if [[ "${NODE_NAME}" == "centralized" ]]; then
    echo "[setup] Downloading full features.csv for centralized node"
    aws s3 cp "s3://${BUCKET}/raw/features.csv" "${DATA_DIR}/features.csv" \
        --region ap-southeast-1
elif [[ "${NODE_NAME}" =~ ^hospital-([0-9]+)$ ]]; then
    H_ID="${BASH_REMATCH[1]}"
    echo "[setup] Downloading partition for hospital ${H_ID}"
    aws s3 cp "s3://${BUCKET}/partitions/hospital_${H_ID}.csv" \
        "${DATA_DIR}/hospital_${H_ID}.csv" \
        --region ap-southeast-1
elif [[ "${NODE_NAME}" == "fl-server" ]]; then
    echo "[setup] FL server — no data download needed"
else
    echo "[setup] Unknown node: ${NODE_NAME} — skipping data download"
fi

echo "[setup] Complete on $(hostname)"
