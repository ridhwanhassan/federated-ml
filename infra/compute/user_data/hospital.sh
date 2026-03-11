#!/bin/bash
# Role-specific setup for hospital Flower client
# Prepended by base.sh at deploy time

# Download hospital data partition from S3
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/partitions/${NODE_NAME}.csv" /opt/fedcost/data/

echo "[${NODE_NAME}] Bootstrap complete"
