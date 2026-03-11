#!/bin/bash
# Role-specific setup for centralized baseline node
# Prepended by base.sh at deploy time

# Download full dataset from S3
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/raw/" /opt/fedcost/data/ --recursive

echo "[centralized] Bootstrap complete"
