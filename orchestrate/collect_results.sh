#!/usr/bin/env bash
# Download distributed experiment results from S3 to local results/distributed/.
# Run from project root: bash orchestrate/collect_results.sh
set -euo pipefail

export AWS_PROFILE=ieee
export AWS_DEFAULT_REGION=ap-southeast-1

BUCKET=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter \
    --name /fedcost/s3-data-bucket \
    --query Parameter.Value \
    --output text)

DEST="results/distributed"
mkdir -p "${DEST}"

echo "=== Collecting results from s3://${BUCKET}/results/ ==="
aws s3 sync "s3://${BUCKET}/results/" "${DEST}/"

echo "=== Downloaded to ${DEST}/ ==="
find "${DEST}" -name "*.json" | sort
echo "=== Done ==="
