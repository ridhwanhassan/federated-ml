#!/usr/bin/env bash
# Package code, upload to S3, distribute to all instances.
# Run from project root: bash orchestrate/deploy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

export AWS_PROFILE=ieee
export AWS_DEFAULT_REGION=ap-southeast-1

BUCKET=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter \
    --name /fedcost/s3-data-bucket \
    --query Parameter.Value \
    --output text)

TARBALL="${PROJECT_ROOT}/fedcost-code.tar.gz"

echo "=== FedCost Deploy ==="
echo "Bucket: ${BUCKET}"
echo "Project root: ${PROJECT_ROOT}"

# 1. Package code tarball (exclude heavy/sensitive dirs)
echo "[1/4] Packaging code tarball..."
tar czf "${TARBALL}" \
    --exclude='.git' \
    --exclude='data/raw' \
    --exclude='data/processed' \
    --exclude='results' \
    --exclude='.venv' \
    --exclude='infra' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='paper' \
    --exclude='node_modules' \
    -C "${PROJECT_ROOT}" \
    src experiments orchestrate pyproject.toml

echo "[2/4] Uploading code tarball to S3..."
aws s3 cp "${TARBALL}" "s3://${BUCKET}/deploy/fedcost-code.tar.gz"
rm -f "${TARBALL}"

# 3. Upload data partitions and full features
echo "[3/4] Uploading data files to S3..."
if [ -d "data/processed/partitions" ]; then
    aws s3 sync data/processed/partitions/ "s3://${BUCKET}/partitions/" \
        --exclude '*.json'
    echo "  Partitions uploaded"
fi

if [ -f "data/processed/features.csv" ]; then
    aws s3 cp data/processed/features.csv "s3://${BUCKET}/raw/features.csv"
    echo "  Features CSV uploaded"
fi

# 4. Run setup on each instance via SSM (code tarball is in S3, not yet on instances)
echo "[4/4] Running setup on all instances via SSM..."

INSTANCE_IDS=(
    "$(cd infra && pulumi stack output ec2_fl-server_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_hospital-1_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_hospital-2_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_hospital-3_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_hospital-4_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_hospital-5_id 2>/dev/null)"
    "$(cd infra && pulumi stack output ec2_centralized_id 2>/dev/null)"
)

INSTANCE_NAMES=(
    "fl-server"
    "hospital-1"
    "hospital-2"
    "hospital-3"
    "hospital-4"
    "hospital-5"
    "centralized"
)

# First extract the tarball, then run the setup script (which handles data download)
SETUP_CMD="set -e; \
BUCKET=\$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text --region ap-southeast-1); \
aws s3 cp s3://\${BUCKET}/deploy/fedcost-code.tar.gz /tmp/fedcost-code.tar.gz --region ap-southeast-1; \
mkdir -p /opt/fedcost; \
tar xzf /tmp/fedcost-code.tar.gz -C /opt/fedcost; \
rm -f /tmp/fedcost-code.tar.gz; \
bash /opt/fedcost/orchestrate/remote/setup_project.sh"

# Send command to all instances
IDS_STR="${INSTANCE_IDS[*]}"
echo "  Sending setup command to ${#INSTANCE_IDS[@]} instances..."
CMD_ID=$(MSYS_NO_PATHCONV=1 aws ssm send-command \
    --instance-ids ${IDS_STR} \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"${SETUP_CMD}\"]" \
    --timeout-seconds 600 \
    --query "Command.CommandId" \
    --output text)

echo "  SSM Command ID: ${CMD_ID}"
echo "  Waiting for completion (up to 10 min)..."

# Poll for completion (all instances in parallel check)
echo "  Polling for completion..."
REMAINING="${#INSTANCE_IDS[@]}"
for attempt in $(seq 1 60); do
    sleep 10
    ALL_DONE=true
    for i in "${!INSTANCE_IDS[@]}"; do
        ID="${INSTANCE_IDS[$i]}"
        NAME="${INSTANCE_NAMES[$i]}"
        STATUS=$(MSYS_NO_PATHCONV=1 aws ssm get-command-invocation \
            --command-id "${CMD_ID}" \
            --instance-id "${ID}" \
            --query "Status" \
            --output text 2>/dev/null) || STATUS="Pending"
        case "${STATUS}" in
            Success) ;;
            Failed|TimedOut|Cancelled)
                echo "  ${NAME}: FAILED (${STATUS})"
                ;;
            *)
                ALL_DONE=false
                ;;
        esac
    done
    if ${ALL_DONE}; then
        break
    fi
    echo "  ... attempt ${attempt}/60, still waiting"
done

# Final report
FAILED=0
for i in "${!INSTANCE_IDS[@]}"; do
    ID="${INSTANCE_IDS[$i]}"
    NAME="${INSTANCE_NAMES[$i]}"
    STATUS=$(MSYS_NO_PATHCONV=1 aws ssm get-command-invocation \
        --command-id "${CMD_ID}" \
        --instance-id "${ID}" \
        --query "Status" \
        --output text 2>/dev/null) || STATUS="Unknown"
    printf "  %-20s %s\n" "${NAME}:" "${STATUS}"
    if [[ "${STATUS}" != "Success" ]]; then
        FAILED=$((FAILED + 1))
    fi
done

if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} instance(s) failed setup"
fi

echo "=== Deploy complete ==="
