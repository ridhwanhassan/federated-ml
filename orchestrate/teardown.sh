#!/usr/bin/env bash
# Kill all remote Python experiment processes on all instances.
# Run from project root: bash orchestrate/teardown.sh
set -euo pipefail

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

echo "=== Tearing down FedCost experiments ==="

for HOST in "${HOSTS[@]}"; do
    printf "%-25s " "${HOST}:"
    RESULT=$(ssh ${SSH_OPTS} "${SSH_USER}@${HOST}" \
        "pkill -f 'python.*orchestrate/remote' 2>/dev/null && echo 'killed' || echo 'nothing running'" \
        2>/dev/null) || RESULT="unreachable"
    echo "${RESULT}"
done

echo "=== Teardown complete ==="
