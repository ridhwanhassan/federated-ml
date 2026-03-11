#!/usr/bin/env bash
# Health-check all instances: uptime, running experiments, S3 model progress.
# Run from project root: bash orchestrate/status.sh
set -euo pipefail

export AWS_PROFILE=ieee
export AWS_DEFAULT_REGION=ap-southeast-1

SSH_USER="ec2-user"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"

HOSTS=(
    "fedcost-fl-server"
    "fedcost-hospital-1"
    "fedcost-hospital-2"
    "fedcost-hospital-3"
    "fedcost-hospital-4"
    "fedcost-hospital-5"
    "fedcost-centralized"
)

echo "=== FedCost Instance Status ==="
echo ""

for HOST in "${HOSTS[@]}"; do
    printf "%-25s " "${HOST}:"

    # Check connectivity and get uptime + python processes
    STATUS=$(ssh ${SSH_OPTS} "${SSH_USER}@${HOST}" \
        "echo 'UP'; uptime -p 2>/dev/null || uptime; echo '---'; pgrep -af 'python3.13' 2>/dev/null || echo 'no python'" \
        2>/dev/null) || STATUS="UNREACHABLE"

    if [[ "${STATUS}" == "UNREACHABLE" ]]; then
        echo "UNREACHABLE"
    else
        UPTIME=$(echo "${STATUS}" | head -2 | tail -1)
        PROCS=$(echo "${STATUS}" | sed '1,/---/d')
        echo "UP (${UPTIME})"
        if [[ "${PROCS}" != "no python" ]]; then
            echo "${PROCS}" | while read -r line; do
                printf "%-25s   %s\n" "" "${line}"
            done
        fi
    fi
done

echo ""
echo "=== S3 Model Progress ==="
BUCKET=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter \
    --name /fedcost/s3-data-bucket \
    --query Parameter.Value \
    --output text 2>/dev/null) || { echo "Cannot read SSM bucket param"; exit 1; }

# Count model files per experiment type
for EXP in fedavg gossip; do
    COUNT=$(aws s3 ls "s3://${BUCKET}/models/${EXP}/" --recursive 2>/dev/null | wc -l)
    echo "${EXP}: ${COUNT} model files in S3"
done

echo ""
echo "=== S3 Results ==="
aws s3 ls "s3://${BUCKET}/results/" --recursive 2>/dev/null | head -20 || echo "No results yet"

echo ""
echo "=== Done ==="
