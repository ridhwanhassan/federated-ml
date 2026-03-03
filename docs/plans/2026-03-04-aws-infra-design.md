# FedCost AWS Infrastructure Design

**Date:** 2026-03-04
**Status:** Approved
**Approach:** Single Pulumi Stack, Multi-VPC

## Overview

Pulumi (Python) infrastructure to run FedCost federated learning experiments on AWS. Deploys 5 EC2 instances across 5 VPCs in `ap-southeast-1`, with VPC peering for Flower FL traffic and Tailscale for SSH management access. Pulumi state stored in S3 backend (`s3://ieee-pulumi`).

## Network Architecture

```
Region: ap-southeast-1

+-  VPC: fl-server (10.0.0.0/16) --------------------+
|  Subnet: 10.0.1.0/24 (public)                      |
|  EC2: t3.medium - Flower FedAvg aggregation server  |
|  SG: allow 8080 (Flower gRPC) from hospital VPCs    |
|      allow SSH from Tailscale                        |
+------+------------------+------------------+--------+
       | VPC Peering       | VPC Peering       | VPC Peering
+------+------+   +--------+-----+   +--------+-----+
| hospital-a  |   | hospital-b   |   | hospital-c   |
| 10.1.0.0/16 |   | 10.2.0.0/16  |   | 10.3.0.0/16  |
| Sub: .1.0/24|   | Sub: .1.0/24 |   | Sub: .1.0/24 |
| t3.medium   |   | t3.medium    |   | t3.medium    |
| SG: 8080->  |   | SG: 8080->   |   | SG: 8080->   |
|     srv only|   |     srv only |   |     srv only |
+-------------+   +--------------+   +--------------+

+-  VPC: centralized (10.4.0.0/16) -------------------+
|  Subnet: 10.4.1.0/24 (public)                       |
|  EC2: t3.large - Centralized baselines (MLP+XGB)    |
|  SG: allow SSH from Tailscale only                   |
|  No peering - isolated                               |
+------------------------------------------------------+
```

### Key networking decisions

- 5 VPCs with separate CIDR blocks (10.0-4.0.0/16) - realistic network isolation per hospital
- VPC peering only between fl-server <-> each hospital. Hospitals cannot communicate directly.
- Centralized VPC is fully isolated - no peering needed
- All instances in public subnets with public IPs (no NAT gateway cost)
- Tailscale installed via user-data for SSH/management access (not a research variable)

## Secrets & Configuration

### Pulumi Config/Secrets (encrypted in stack file)
- `tailscale-auth-key` - Tailscale pre-auth key
- `ssh-public-key` - SSH public key for instance access

### SSM Parameter Store (runtime config)
- `/fedcost/flower-server-ip` - Written by Pulumi after FL server creation; read by hospital instances
- `/fedcost/s3-data-bucket` - Data bucket name

### IAM Instance Profiles (no hardcoded AWS keys)
- **Hospital EC2s:** `s3:GetObject` on data bucket + `ssm:GetParameter`
- **FL server:** `ssm:PutParameter` (register its IP) + `s3:GetObject`
- **Centralized:** `s3:GetObject` on data bucket

Zero secrets baked into AMIs or committed to git.

## S3 Data Flow

### Buckets
1. `ieee-pulumi` - Pulumi state backend (pre-existing)
2. `fedcost-data-{stack}` - Created by Pulumi:

```
s3://fedcost-data-dev/
  raw/mimic_iv_fedcost.csv          # Full dataset (uploaded manually)
  partitions/hospital-a.csv          # Pre-partitioned
  partitions/hospital-b.csv
  partitions/hospital-c.csv
  results/                           # Experiment outputs uploaded back
```

Bucket is private, accessible only via IAM roles.

## Compute

| Instance | VPC | Type | Purpose |
|----------|-----|------|---------|
| fedcost-fl-server | fl-server (10.0.0.0/16) | t3.medium | Flower aggregation server |
| fedcost-hospital-a | hospital-a (10.1.0.0/16) | t3.medium | Flower client + local training |
| fedcost-hospital-b | hospital-b (10.2.0.0/16) | t3.medium | Flower client + local training |
| fedcost-hospital-c | hospital-c (10.3.0.0/16) | t3.medium | Flower client + local training |
| fedcost-centralized | centralized (10.4.0.0/16) | t3.large | Centralized MLP + XGBoost baselines |

- AMI: Amazon Linux 2023 or Ubuntu 22.04
- User-data installs: Python 3.13, project deps, Tailscale

## Experiment Flow

1. `pulumi up` provisions all infra
2. Upload data partitions to S3 (manual or script)
3. SSH via Tailscale to FL server, start Flower server
4. SSH to each hospital, start Flower clients (connect to FL server private IP via peering)
5. Results written locally, uploaded to S3 results/
6. Separately: SSH to centralized, run baseline experiments
7. `pulumi destroy` when done (cost control)

## Pulumi Code Structure

```
infra/
  __main__.py              # Entry point - composes all modules
  Pulumi.yaml              # Project name, runtime (python)
  Pulumi.dev.yaml          # Dev stack config
  config.py                # Pulumi config -> typed dataclass
  network/
    __init__.py
    vpcs.py                # 5 VPCs (CIDR, subnets, IGW, route tables)
    peering.py             # VPC peering + route entries
  security/
    __init__.py
    security_groups.py     # SGs per role
    iam.py                 # IAM roles + instance profiles
  compute/
    __init__.py
    instances.py           # 5 EC2 instance definitions
    user_data/
      fl_server.sh         # Bootstrap: Tailscale + Flower server
      hospital.sh          # Bootstrap: Tailscale + download data + Flower client
      centralized.sh       # Bootstrap: Tailscale + download full data
    key_pair.py            # SSH key pair resource
  storage/
    __init__.py
    s3.py                  # Data bucket + policy
  ssm/
    __init__.py
    parameters.py          # SSM parameters
```

### Module composition order in `__main__.py`
1. Storage (S3 bucket)
2. Network (VPCs, subnets, peering)
3. Security (SGs, IAM)
4. Compute (EC2 instances)
5. SSM (parameters using FL server private IP)

### Stack outputs
- All instance IDs and private IPs
- S3 bucket name
- Tailscale-accessible hostnames

## Setup Commands

```bash
# One-time backend setup
pulumi login 's3://ieee-pulumi?region=ap-southeast-1&awssdk=v2'
pulumi stack init dev

# Deploy
cd infra/
pulumi up

# Tear down
pulumi destroy
```
