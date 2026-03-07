# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FedCost** is a topology-aware comparison of **FedAvg (star) vs D-PSGD (ring gossip)** for ICU length-of-stay prediction on MIMIC-IV. Targets an IEEE conference paper (ICHI or BIBM 2026). Infrastructure is deployed to **AWS** using **Pulumi** (Python).

**Research question:** How does the choice of communication topology — centralized FedAvg (star) vs decentralized D-PSGD (ring) — affect convergence and accuracy for ICU length-of-stay prediction across non-IID hospital data?

**Core contribution:** First comparison of FedAvg vs D-PSGD for ICU LOS prediction on MIMIC-IV, with a 5-hospital simulated federation using clinically motivated non-IID partitioning.

## Architecture

```
            FedAvg (Star)                        D-PSGD (Ring)

     ┌────────────────────┐
     │   Central Server    │
     │  (FedAvg aggregate) │
     └──┬──┬──┬──┬──┬─────┘
        │  │  │  │  │               H1 ── H2 ── H3 ── H4 ── H5 ── H1
     ┌──┘  │  │  │  └──┐            (each node exchanges with 2 neighbors)
     │     │  │  │     │
    H1    H2 H3 H4   H5

     ┌─────────────────────────────────────────────┐
     │          Single-Task MLP (LOS Regression)    │
     │  Backbone: 256 → 128 → 64 (BN + Dropout)   │
     │  Head: → 1 (LOS days)                       │
     │  Loss: Huber (δ=5.0)                        │
     └─────────────────────────────────────────────┘
```

## 5-Hospital Partition (care-unit-based, NOT random)

| Hospital | Care Units | Clinical Profile |
|----------|-----------|-----------------|
| H1 (Medical) | MICU, MICU/SICU | General medical ICU (largest) |
| H2 (Neuro) | Neuro Intermediate, Neuro Stepdown, Neuro SICU | Neurological care |
| H3 (Surgical) | SICU | General surgical |
| H4 (Trauma) | TSICU | Trauma surgical |
| H5 (Cardiac) | CCU, CVICU | Cardiac care (smallest) |

Unmatched care units default to H1. Creates realistic non-IID heterogeneity.

## Directory Structure

```
fedcost/
├── infra/                        # Pulumi IaC (AWS infrastructure)
│   ├── __main__.py               # Entry point — composes modules in order
│   ├── Pulumi.yaml               # Project name, runtime
│   ├── config.py                 # Pulumi config -> typed dataclass
│   ├── network/                  # VPCs, subnets, IGW (6 VPCs)
│   ├── security/                 # Security groups, IAM roles + instance profiles
│   ├── compute/                  # EC2 instances + user_data/ bootstrap scripts
│   ├── storage/                  # S3 data bucket + policy
│   └── ssm/                      # SSM parameters (server IP, hospital IPs, bucket)
├── data/
│   ├── raw/                      # MIMIC-IV CSV exports (git-ignored)
│   ├── processed/                # Feature-engineered splits
│   └── sql/
│       └── extract_cohort.sql    # MIMIC-IV extraction query
├── src/
│   ├── data/                     # extract, partition, features, loader
│   ├── models/                   # LOS MLP (single-task), XGBoost baseline
│   ├── federation/               # Flower client, FedAvg server, D-PSGD ring
│   ├── evaluation/               # Metrics (MAE, RMSE, R²), convergence tracking
│   └── visualization/            # Figures 1-5 (topology, convergence, per-hospital, ablation, cost)
├── experiments/
│   ├── run_all.py                # Main experiment runner (entry point)
│   ├── run_centralized.py        # Centralized MLP + XGBoost baselines
│   ├── run_federated.py          # Flower simulation (5 hospitals, FedAvg)
│   ├── run_gossip.py             # D-PSGD ring simulation (5 hospitals)
│   ├── run_local_only.py         # Per-hospital local training
│   └── configs/default.yaml      # Hyperparameters and experiment config
├── paper/                        # LaTeX manuscript, generated figures/tables
├── results/                      # Metric dumps, model checkpoints, logs
├── docs/plans/                   # Design documents
└── tests/                        # test_infra_unit, test_infra_deployed, test_model, test_partition
```

## Commands

```bash
# Dependency management (uv)
uv sync                      # Install/sync all dependencies
uv add <package>             # Add a dependency

# Pulumi — see "Deploying with Pulumi" section below for full walkthrough
cd infra/
pulumi up                    # Preview and deploy changes
pulumi preview               # Preview changes without deploying
pulumi destroy               # Tear down all resources
pulumi stack output --json   # Dump stack outputs (VPC IDs, instance IPs, etc.)

# Data extraction (requires MIMIC-IV DB access)
python src/data/extract.py --output data/raw/mimic_iv_fedcost.csv

# Run full experiment pipeline
python experiments/run_all.py --config experiments/configs/default.yaml

# Run individual experiments
python experiments/run_centralized.py
python experiments/run_federated.py --n-rounds 50 --local-epochs 3
python experiments/run_gossip.py --n-rounds 50 --local-epochs 3
python experiments/run_local_only.py

# Generate paper figures
python src/visualization/generate_all.py --results-dir results/metrics/ --output paper/figures/

# Tests
uv sync --extra test             # Install test deps (pytest, boto3)
pytest tests/ -v                 # Run all tests (unit pass, deployed skip)
pytest tests/test_infra_unit.py -v                    # Mocked unit tests only (no AWS creds needed)
pytest tests/test_infra_deployed.py -v --run-deployed  # Post-deploy smoke tests (needs AWS creds + deployed stack)
pytest tests/ -v -W ignore::DeprecationWarning         # Suppress pulumi-aws deprecation noise
```

### Test Suite

Two test files in `tests/`:

- **`test_infra_unit.py`** — 32 Pulumi mocked unit tests. Uses `pulumi.runtime.set_mocks()` (configured in `conftest.py`) to verify resource counts (6 VPCs, 7 instances, 6 SGs, 7 SSM params), CIDRs, instance types, tagging, IAM assume-role policies, SG egress rules, root volume config, and SSM parameter paths. Runs without AWS credentials.
- **`test_infra_deployed.py`** — Post-deploy boto3 smoke tests. Verifies live AWS resources after `pulumi up`: VPCs exist with correct CIDRs, EC2 instances running with correct types/profiles, S3 bucket has public access blocked, SSM parameters match stack outputs, SGs allow all outbound. Skipped by default; pass `--run-deployed` to enable.

`conftest.py` handles: Pulumi mock setup (`FedCostMocks` class), `sys.path` injection for `infra/` imports, and the `--run-deployed` / `@pytest.mark.deployed` gating.

### Deploying with Pulumi

**Prerequisites:**
- AWS CLI configured with an `ieee` profile (`aws configure --profile ieee`). This project always uses `AWS_PROFILE=ieee` — set in `.env` and in the Pulumi stack config (`aws:profile`).
- Pulumi CLI installed (`curl -fsSL https://get.pulumi.com | sh`)
- `PULUMI_CONFIG_PASSPHRASE` env var set (used to encrypt secrets in S3 backend — see `.env`)
- Tailscale pre-auth key (generate at https://login.tailscale.com/admin/settings/keys — use reusable + ephemeral)
- SSH public key for EC2 access

**1. One-time backend + stack setup:**

```bash
# Login to S3 backend (stores state in s3://ieee-pulumi)
pulumi login 's3://ieee-pulumi?region=ap-southeast-1&awssdk=v2'

# Create a stack (e.g., "dev")
cd infra/
pulumi stack init dev
```

**2. Configure the stack:**

```bash
# Required config
pulumi config set aws:profile ieee
pulumi config set aws:region ap-southeast-1
pulumi config set fedcost:ssh-public-key "ssh-rsa AAAA..."
pulumi config set --secret fedcost:tailscale-auth-key "tskey-auth-..."

# Optional overrides (defaults shown)
pulumi config set fedcost:instance-type-hospital t3.medium
pulumi config set fedcost:instance-type-fl-server t3.medium
pulumi config set fedcost:instance-type-centralized t3.large
```

This creates `infra/Pulumi.dev.yaml` (gitignored — contains encrypted secrets).

**3. Deploy:**

```bash
cd infra/
pulumi preview               # Dry run — review what will be created
pulumi up                    # Deploy (~70 AWS resources)
```

Deploys: 6 VPCs + subnets + IGWs + route tables, 6 security groups, 3 IAM roles + instance profiles, 7 EC2 instances (with Tailscale + Python bootstrap via user-data), 1 S3 bucket, 7 SSM parameters.

**4. Verify deployment:**

```bash
pulumi stack output --json                                    # All outputs
pytest tests/test_infra_deployed.py -v --run-deployed         # Smoke tests
```

**5. Tear down:**

```bash
cd infra/
pulumi destroy               # Remove all AWS resources
pulumi stack rm dev           # Remove stack from backend (optional)
```

**Stack config reference** (`infra/config.py` → `FedCostConfig`):

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `aws:profile` | yes | — | AWS CLI profile (always `ieee`) |
| `aws:region` | yes | — | AWS region |
| `fedcost:ssh-public-key` | yes | — | SSH public key for EC2 KeyPair |
| `fedcost:tailscale-auth-key` | yes (secret) | — | Tailscale pre-auth key |
| `fedcost:instance-type-hospital` | no | `t3.medium` | EC2 type for 5 hospital nodes |
| `fedcost:instance-type-fl-server` | no | `t3.medium` | EC2 type for FL aggregation server |
| `fedcost:instance-type-centralized` | no | `t3.large` | EC2 type for centralized baselines |

## Key Technical Decisions

### Data
- **Source:** MIMIC-IV v2.2+ via PhysioNet (credentialed access required)
- **Database:** PostgreSQL (`mimiciv_hosp`/`mimiciv_icu` schemas), BigQuery, or downloaded CSV flat files
- **Cohort:** ICU stays with LOS > 0 and ≤ 30 days, excluding in-hospital deaths
- **Features:** Demographics, admission info, first-24h vitals, first-24h labs, complexity proxies (n_procedures, n_diagnoses), DRG code as categorical feature

### 5-Hospital Partition (care-unit-based, NOT random)
- **H1 (Medical):** MICU, MICU/SICU — largest
- **H2 (Neuro):** Neuro Intermediate, Neuro Stepdown, Neuro SICU
- **H3 (Surgical):** SICU
- **H4 (Trauma):** TSICU
- **H5 (Cardiac):** CCU, CVICU — smallest, highest acuity
- Unmatched care units default to H1. Creates realistic non-IID heterogeneity.

### Model
- Single-task MLP: 256 → 128 → 64 (BatchNorm + Dropout) → 1 (LOS regression)
- Loss: Huber (δ=5.0)
- Optimizer: Adam, weight_decay=1e-4, ReduceLROnPlateau scheduler

### Algorithms
- **FedAvg (Star):** Central server aggregates all 5 hospital models each round. Weighted average by dataset size. 50 rounds, E local epochs, lr=1e-3.
- **D-PSGD (Ring):** No central server. Ring: H1↔H2↔H3↔H4↔H5↔H1. Each round: local SGD for E epochs → exchange models with 2 neighbors → average using Metropolis-Hastings mixing weights. 50 rounds, same lr and local epochs.

### Federation
- **Framework:** Flower (flwr) simulation mode for FedAvg. Custom implementation for D-PSGD ring.
- **Strategy:** FedAvg (all 5 clients every round) vs D-PSGD ring (2-neighbor exchange)
- **Privacy:** Privacy-by-architecture (data never leaves hospital). No DP-SGD this paper — future work.
- **Resources:** `client_resources={"num_cpus": 2, "num_gpus": 0.0}` — set `num_gpus=0.2` for GPU

### Experiments (5 total)
1. Centralized MLP — all data pooled, same architecture (upper bound)
2. XGBoost Centralized — strongest non-federated baseline
3. FedAvg (star, 5 hospitals) — central server, 50 rounds
4. D-PSGD (ring, 5 hospitals) — ring topology, 50 rounds
5. Local-only (×5) — each hospital trains independently (lower bound)

### Evaluation
- LOS regression: MAE, RMSE, R², Within-1-day accuracy (%predictions within 1 day of true LOS)
- Convergence: rounds to reach 95% of centralized MAE
- Communication cost: total model parameters exchanged (FedAvg: 2×N×params/round; D-PSGD: 2×2×params/node/round)
- Fairness: per-hospital MAE variance (std across 5 hospitals)
- Ablation: local epochs E = {1, 3, 5} for both FedAvg and D-PSGD
- All reported with 5-seed mean ± std (seeds 42–46). Train/val 80/20 per hospital.

### AWS Infrastructure (Pulumi)

**Design doc:** `docs/plans/2026-03-04-aws-infra-design.md`

- **Region:** `ap-southeast-1`
- **Pulumi backend:** `s3://ieee-pulumi` (S3 state backend)
- **Approach:** Single stack, multi-VPC — 6 VPCs with separate CIDR blocks
- **Infra code:** `infra/` directory with modules: network, security, compute, storage, ssm

**VPC layout:**

| VPC | CIDR | EC2 Type | Role |
|-----|------|----------|------|
| fl-server | 10.0.0.0/16 | t3.medium | Flower FedAvg aggregation server |
| hospital-1 | 10.1.0.0/16 | t3.medium | H1 Medical (MICU, MICU/SICU) |
| hospital-2 | 10.2.0.0/16 | t3.medium | H2 Neuro |
| hospital-3 | 10.3.0.0/16 | t3.medium | H3 Surgical (SICU) |
| hospital-4 | 10.4.0.0/16 | t3.medium | H4 Trauma (TSICU) |
| hospital-5 | 10.5.0.0/16 | t3.medium | H5 Cardiac (CCU, CVICU) |

- Centralized baseline instance shares the fl-server VPC (no separate VPC)
- No VPC peering — Tailscale handles all inter-instance connectivity (Flower gRPC, gossip, SSH)
- Separate VPCs provide network isolation narrative for the paper; Tailscale ACLs can enforce topology
- IAM instance profiles for S3/SSM access (no hardcoded AWS keys)
- Secrets via Pulumi config (Tailscale auth key, SSH key); runtime config via SSM Parameter Store
- SSM stores: FL server IP, all 5 hospital IPs (for gossip peer discovery), S3 bucket name
- Data bucket: `fedcost-data-{stack}` with raw/, partitions/, results/ prefixes

## What NOT to Do
- Do not implement DP-SGD / Opacus — scoped out for this paper
- Do not run privacy attacks (membership inference, etc.) — future work
- Do not use MIMIC-III — we use MIMIC-IV specifically for novelty
- Do not randomly partition data — always use care-unit-based split
- Do not include `hospital_expire_flag = 1` patients — exclude in-hospital deaths
- Do not implement multi-task learning — single-task LOS regression only
- Do not implement cost weight prediction — dropped for focused topology comparison

## Reproducibility Notes
- Global StandardScaler fit on ALL data (simulating pre-federation calibration — acknowledge in paper limitations)
- All paths via `pathlib.Path`, never hardcoded strings
- Metrics always returned as dicts, never positional tuples
- Logging over print in production code (print OK in experiment scripts)
- Type hints on all function signatures, NumPy-style docstrings on public functions

## Paper Target
- **Conference:** IEEE ICHI 2026 or IEEE BIBM 2026
- **Format:** 8-page IEEE double-column
- **Title (working):** "Decentralized vs. Federated Learning for ICU Length-of-Stay Prediction: A Ring-Topology Gossip Approach on MIMIC-IV"

### Key Novelty Claims
1. First comparison of FedAvg vs D-PSGD for ICU LOS prediction on MIMIC-IV
2. Clinically motivated non-IID partition (5 hospitals by care unit type, not random)
3. Quantifies the accuracy cost of full decentralization (ring gossip vs central server)
4. Fairness analysis: do smaller/specialized hospitals benefit differently from each topology?

### Tables & Figures
- **Table I:** Main comparison — MAE, RMSE, R², Within-1-day % for all 5 experiments
- **Table II:** Per-hospital breakdown — FedAvg vs D-PSGD vs local-only per hospital
- **Figure 1:** Architecture diagram (star vs ring topology)
- **Figure 2:** Convergence curves (MAE vs communication rounds)
- **Figure 3:** Per-hospital performance bar chart
- **Figure 4:** Local epochs ablation (E=1,3,5)
- **Figure 5:** Communication cost vs accuracy scatter

## Task Backlog

### Phase 1: Data Pipeline
- [ ] Verify MIMIC-IV PostgreSQL access and run extraction query
- [ ] Implement 5-hospital care-unit partition logic
- [ ] Confirm care unit names match MIMIC-IV v2.2+ schema
- [ ] Feature engineering (demographics, vitals, labs, DRG code as categorical)
- [ ] Export cohort CSV to `data/raw/`

### Phase 2: Model Development
- [ ] Implement single-task MLP (LOS regression, Huber loss)
- [ ] Implement XGBoost baseline
- [ ] Write unit tests for model forward pass and loss computation

### Phase 3: Federation & Gossip
- [ ] Set up Flower simulation with 5 clients (FedAvg)
- [ ] Implement D-PSGD ring topology (custom or lightweight library)
- [ ] Implement Metropolis-Hastings mixing matrix for ring
- [ ] Verify FedAvg and D-PSGD convergence on toy data

### Phase 4: Experiments
- [ ] Run all 5 experiments × 5 seeds × 3 local-epoch settings
- [ ] Collect metrics to `results/metrics/`
- [ ] Compute convergence rates and communication costs

### Phase 5: Analysis & Paper
- [ ] Generate Table I (main comparison) as LaTeX
- [ ] Generate Table II (per-hospital breakdown) as LaTeX
- [ ] Generate Figures 1-5 as PDFs
- [ ] Compute statistical significance (paired t-test or Wilcoxon across seeds)
- [ ] Draft paper in IEEE double-column LaTeX template
