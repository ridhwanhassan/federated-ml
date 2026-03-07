# FedCost AWS Infrastructure Design

**Date:** 2026-03-04
**Updated:** 2026-03-07
**Status:** Approved (implemented)
**Approach:** Single Pulumi Stack, Multi-VPC

## Overview

Pulumi (Python) infrastructure to run FedCost federated learning experiments on AWS. Deploys 7 EC2 instances across 6 VPCs in `ap-southeast-1`, with Tailscale for all inter-instance connectivity (Flower gRPC, D-PSGD gossip, SSH). No VPC peering — separate VPCs provide network isolation narrative for the paper; Tailscale handles all communication via encrypted WireGuard tunnels. Pulumi state stored in S3 backend (`s3://ieee-pulumi`).

## Network Architecture

```
Region: ap-southeast-1

+-  VPC: fl-server (10.0.0.0/16) --------------------------------+
|  Subnet: 10.0.1.0/24 (public — instances get public IPs)       |
|  EC2: t3.medium  — Flower FedAvg aggregation server             |
|  EC2: t3.large   — Centralized baselines (MLP + XGBoost)        |
|  SG: outbound only, Tailscale for all ingress                   |
+-----------------------------------------------------------------+

     All inter-VPC traffic via Tailscale (no VPC peering)

+-  hospital-1  -+   +-  hospital-2  -+   +-  hospital-3  -+
|  10.1.0.0/16   |   |  10.2.0.0/16   |   |  10.3.0.0/16   |
|  Private subnet|   |  Private subnet|   |  Private subnet|
|  .1.0/24 (EC2) |   |  .1.0/24 (EC2) |   |  .1.0/24 (EC2) |
|  .2.0/24 (NAT) |   |  .2.0/24 (NAT) |   |  .2.0/24 (NAT) |
|  t3.medium     |   |  t3.medium     |   |  t3.medium     |
|  SG: outbound  |   |  SG: outbound  |   |  SG: outbound  |
+----------------+   +----------------+   +----------------+

+-  hospital-4  -+   +-  hospital-5  -+
|  10.4.0.0/16   |   |  10.5.0.0/16   |
|  Private subnet|   |  Private subnet|
|  .1.0/24 (EC2) |   |  .1.0/24 (EC2) |
|  .2.0/24 (NAT) |   |  .2.0/24 (NAT) |
|  t3.medium     |   |  t3.medium     |
|  SG: outbound  |   |  SG: outbound  |
+----------------+   +----------------+
```

### Key networking decisions

- 6 VPCs with separate CIDR blocks (10.0–10.5.0.0/16) — realistic network isolation per hospital
- **No VPC peering** — Tailscale handles all inter-instance connectivity (Flower gRPC, D-PSGD gossip, SSH) via encrypted WireGuard tunnels
- Centralized baseline instance shares the fl-server VPC (no separate VPC needed — it has no cross-hospital communication)
- FL server VPC uses a **public subnet** — instances get public IPs and route directly via IGW
- Hospital VPCs use **private subnets** — EC2 instances have no public IPs, outbound traffic routes through NAT Gateways (one per hospital VPC)
- Each hospital VPC has two subnets: private (.1.0/24) for EC2 and public (.2.0/24) for the NAT Gateway
- Tailscale installed via user-data on all instances for SSH/management access and all research traffic
- Security groups are outbound-only (allow all egress, no ingress rules) — Tailscale handles ingress

## Secrets & Configuration

### Pulumi Config/Secrets (encrypted in stack file)
- `tailscale-auth-key` — Tailscale pre-auth key
- `ssh-public-key` — SSH public key for instance access

### SSM Parameter Store (runtime config)
- `/fedcost/flower-server-ip` — Written by Pulumi after FL server creation; read by hospital instances
- `/fedcost/s3-data-bucket` — Data bucket name
- `/fedcost/hospital-1-ip` through `/fedcost/hospital-5-ip` — Hospital private IPs for D-PSGD ring peer discovery

### IAM Instance Profiles (no hardcoded AWS keys)
- **Hospital EC2s:** `s3:GetObject`, `s3:ListBucket` on data bucket + `ssm:GetParameter` + SSM Session Manager
- **FL server:** `s3:GetObject`, `s3:ListBucket` on data bucket + `ssm:GetParameter`, `ssm:PutParameter` + SSM Session Manager
- **Centralized:** `s3:GetObject`, `s3:ListBucket`, `s3:PutObject` on data bucket + SSM Session Manager

Zero secrets baked into AMIs or committed to git.

## S3 Data Flow

### Buckets
1. `ieee-pulumi` — Pulumi state backend (pre-existing)
2. `fedcost-data-{stack}` — Created by Pulumi:

```
s3://fedcost-data-dev/
  raw/mimic_iv_fedcost.csv          # Full dataset (uploaded manually)
  partitions/hospital-1.csv          # Pre-partitioned
  partitions/hospital-2.csv
  partitions/hospital-3.csv
  partitions/hospital-4.csv
  partitions/hospital-5.csv
  results/                           # Experiment outputs uploaded back
```

Bucket is private (public access block enabled), accessible only via IAM roles.

## Compute

| Instance | VPC | Type | Purpose |
|----------|-----|------|---------|
| fedcost-fl-server | fl-server (10.0.0.0/16) | t3.medium | Flower aggregation server |
| fedcost-hospital-1 | hospital-1 (10.1.0.0/16) | t3.medium | H1 Medical — Flower client + local training |
| fedcost-hospital-2 | hospital-2 (10.2.0.0/16) | t3.medium | H2 Neuro — Flower client + local training |
| fedcost-hospital-3 | hospital-3 (10.3.0.0/16) | t3.medium | H3 Surgical — Flower client + local training |
| fedcost-hospital-4 | hospital-4 (10.4.0.0/16) | t3.medium | H4 Trauma — Flower client + local training |
| fedcost-hospital-5 | hospital-5 (10.5.0.0/16) | t3.medium | H5 Cardiac — Flower client + local training |
| fedcost-centralized | fl-server (10.0.0.0/16) | t3.large | Centralized MLP + XGBoost baselines |

- AMI: Amazon Linux 2023
- Root volume: 30 GB gp3
- User-data: modular base script (system update, Python 3.13, Tailscale) + role-specific script
- Variable injection: `${TAILSCALE_AUTH_KEY}` and `${NODE_NAME}` replaced at deploy time

## Experiment Flow

1. `pulumi up` provisions all infra (~70 AWS resources)
2. Upload data partitions to S3 (manual or script)
3. SSH via Tailscale to FL server, start Flower server
4. SSH to each hospital via Tailscale, start Flower clients (connect to FL server via Tailscale hostname)
5. For D-PSGD: hospitals discover peers via SSM parameters (hospital IPs) and communicate via Tailscale
6. Results written locally, uploaded to S3 results/
7. Separately: SSH to centralized, run baseline experiments
8. `pulumi destroy` when done (cost control)

## Pulumi Code Structure

```
infra/
  __main__.py              # Entry point — composes all modules
  Pulumi.yaml              # Project name, runtime (python)
  Pulumi.dev.yaml          # Dev stack config (gitignored — contains encrypted secrets)
  config.py                # Pulumi config -> typed dataclass
  network/
    __init__.py
    vpcs.py                # 6 VPCs (public + private with NAT)
  security/
    __init__.py
    security_groups.py     # 6 SGs (outbound-only, one per VPC)
    iam.py                 # 3 IAM roles + instance profiles
  compute/
    __init__.py
    instances.py           # 7 EC2 instance definitions
    key_pair.py            # SSH key pair resource
    user_data/
      base.sh              # Shared bootstrap: system update, Python, Tailscale
      fl_server.sh         # Role: pip install FL deps
      hospital.sh          # Role: pip install + download partition from S3
      centralized.sh       # Role: pip install + download full dataset from S3
  storage/
    __init__.py
    s3.py                  # Data bucket + public access block
  ssm/
    __init__.py
    parameters.py          # 7 SSM parameters (server IP, bucket, 5 hospital IPs)
```

### Module composition order in `__main__.py`
1. Storage (S3 bucket)
2. Network (VPCs, subnets — no peering)
3. Security (SGs, IAM)
4. Compute (EC2 instances)
5. SSM (parameters using FL server + hospital private IPs)

### Stack outputs
- Region
- S3 bucket name
- All VPC IDs (6)
- All instance IDs, private IPs, and public IPs (7 × 3 = 21)

## Resource Summary (~70 AWS resources)

- 6 VPCs, 11 subnets (1 public + 5×2 private/public), 6 IGWs
- 11 route tables + 11 route table associations
- 5 Elastic IPs + 5 NAT Gateways (one per hospital VPC)
- 6 security groups (outbound-only)
- 3 IAM roles + 3 IAM policies + 3 IAM managed policy attachments + 3 instance profiles
- 1 SSH key pair
- 7 EC2 instances
- 1 S3 bucket + 1 public access block
- 7 SSM parameters

## Setup Commands

```bash
# One-time backend setup
pulumi login 's3://ieee-pulumi?region=ap-southeast-1&awssdk=v2'
pulumi stack init dev

# Configure stack
pulumi config set aws:profile ieee
pulumi config set aws:region ap-southeast-1
pulumi config set fedcost:ssh-public-key "ssh-rsa AAAA..."
pulumi config set --secret fedcost:tailscale-auth-key "tskey-auth-..."

# Deploy
cd infra/
pulumi up

# Tear down
pulumi destroy
```
