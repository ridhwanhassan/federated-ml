#!/bin/bash
# Role-specific setup for FL aggregation server
# Prepended by base.sh at deploy time

python3.13 -m pip install flwr torch numpy pandas scikit-learn

echo "[fl-server] Bootstrap complete"
