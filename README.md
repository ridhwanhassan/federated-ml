# FedCost

Federated learning for DRG-based healthcare cost and length-of-stay prediction, validated on MIMIC-IV.

**Research question:** What accuracy do you sacrifice by using federated learning instead of centralized training for healthcare cost prediction?

## Architecture

```
                    FedAvg Server
              (aggregates model updates)
           ┌───────────┼───────────┐
           │           │           │
      Hospital A   Hospital B  Hospital C
      Medical ICU  Surgical    Cardiac
      ~18k stays   ~8k stays   ~5k stays
```

Three simulated hospitals train locally on care-unit-partitioned MIMIC-IV data, then aggregate via Flower's FedAvg. A centralized baseline (pooled data) provides the upper bound.

The model is a multi-task MLP with three heads:
- **LOS regression** (Huber loss)
- **Cost weight regression** (MSE)
- **LOS classification** (CrossEntropy) -- short/medium/long/extended

## Infrastructure

AWS infrastructure is managed with Pulumi (Python). Hospital nodes run in private subnets with NAT Gateways; the FL server and centralized baseline share a public VPC. Tailscale handles all inter-instance connectivity.

```
infra/
├── network/    # 4 VPCs, private/public subnets, NAT Gateways
├── security/   # Security groups, IAM roles + instance profiles
├── compute/    # 5 EC2 instances + user-data bootstrap scripts
├── storage/    # S3 data bucket
└── ssm/        # SSM parameters (server IP, bucket name)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run unit tests (no AWS credentials needed)
uv sync --extra test
pytest tests/test_infra_unit.py -v

# Deploy infrastructure (requires AWS + Pulumi setup -- see CLAUDE.md)
cd infra/
pulumi up
```

See [CLAUDE.md](CLAUDE.md) for full deployment walkthrough, stack configuration, and experiment commands.

## Paper

Targeting IEEE ICHI 2026 or IEEE BIBM 2026 (8-page double-column).

## License

MIT
