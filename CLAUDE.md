# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FedCost** is a federated learning framework for DRG-based healthcare cost and length-of-stay (LOS) prediction, validated on MIMIC-IV. Targets an IEEE conference paper (ICHI or BIBM 2026). Infrastructure is deployed to **AWS** using **Pulumi** (Python).

**Research question:** What accuracy do you sacrifice by using federated learning instead of centralized training for healthcare cost prediction?

**Core contribution:** First FL system for DRG-based cost weight prediction on MIMIC-IV, with a 3-hospital simulated federation and multi-task learning (LOS regression + cost weight regression + LOS classification).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FedAvg Server                         │
│              (aggregates model updates)                  │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
    ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴───────┐
    │ Hospital A  │ │ Hospital B │ │ Hospital C │
    │ Medical ICU │ │ Surgical   │ │ Cardiac    │
    │ ~18k stays  │ │ ~8k stays  │ │ ~5k stays  │
    └─────────────┘ └────────────┘ └────────────┘
           │              │              │
    ┌──────┴──────────────┴──────────────┴────────┐
    │           FedCost Multi-Task MLP             │
    │  Shared Backbone (256→128→64, BN+Dropout)   │
    │  ┌──────────┬──────────────┬──────────────┐  │
    │  │ LOS Reg  │ Cost Weight  │  LOS Class   │  │
    │  │ (Huber)  │   (MSE)      │    (CE)      │  │
    │  └──────────┴──────────────┴──────────────┘  │
    └─────────────────────────────────────────────────┘
```

## Directory Structure

```
fedcost/
├── infra/                        # Pulumi IaC (AWS infrastructure)
│   ├── __main__.py               # Entry point — composes modules in order
│   ├── Pulumi.yaml               # Project name, runtime
│   ├── Pulumi.dev.yaml           # Dev stack config (instance types, secrets)
│   ├── config.py                 # Pulumi config -> typed dataclass
│   ├── network/                  # VPCs, subnets, IGW, peering
│   ├── security/                 # Security groups, IAM roles + instance profiles
│   ├── compute/                  # EC2 instances + user_data/ bootstrap scripts
│   ├── storage/                  # S3 data bucket + policy
│   └── ssm/                      # SSM parameters (Flower server IP, bucket name)
├── data/
│   ├── raw/                      # MIMIC-IV CSV exports (git-ignored)
│   ├── processed/                # Feature-engineered splits
│   └── sql/
│       └── extract_cohort.sql    # MIMIC-IV extraction query
├── src/
│   ├── data/                     # extract, partition, features, loader
│   ├── models/                   # FedCostModel MLP, multi-task loss, XGBoost baseline
│   ├── federation/               # Flower client, FedAvg server, param utils
│   ├── evaluation/               # Metrics (MAE, RMSE, R², F1, CM), SHAP analysis
│   └── visualization/            # Figures 1-6 (scatter, box, bar, convergence, confusion, SHAP)
├── experiments/
│   ├── run_all.py                # Main experiment runner (entry point)
│   ├── run_centralized.py        # Centralized MLP + XGBoost baselines
│   ├── run_federated.py          # Flower simulation (3 hospitals)
│   ├── run_local_only.py         # Per-hospital local training
│   └── configs/default.yaml      # Hyperparameters and experiment config
├── paper/                        # LaTeX manuscript, generated figures/tables
├── results/                      # Metric dumps, model checkpoints, logs
├── docs/plans/                   # Design documents
└── tests/                        # test_features, test_model, test_partition, test_federation
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

- **`test_infra_unit.py`** — 29 Pulumi mocked unit tests. Uses `pulumi.runtime.set_mocks()` (configured in `conftest.py`) to verify resource counts, CIDRs, instance types, tagging, IAM assume-role policies, SG egress rules, root volume config, and SSM parameter paths. Runs without AWS credentials.
- **`test_infra_deployed.py`** — 15 boto3 smoke tests. Verifies live AWS resources after `pulumi up`: VPCs exist with correct CIDRs, EC2 instances running with correct types/profiles, S3 bucket has public access blocked, SSM parameters match stack outputs, SGs allow all outbound. Skipped by default; pass `--run-deployed` to enable.

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
pulumi up                    # Deploy (creates ~52 AWS resources)
```

Deploys: 5 VPCs + subnets + IGWs + route tables, 5 security groups, 3 IAM roles + instance profiles, 5 EC2 instances (with Tailscale + Python bootstrap via user-data), 1 S3 bucket, 2 SSM parameters.

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
| `fedcost:instance-type-hospital` | no | `t3.medium` | EC2 type for 3 hospital nodes |
| `fedcost:instance-type-fl-server` | no | `t3.medium` | EC2 type for FL aggregation server |
| `fedcost:instance-type-centralized` | no | `t3.large` | EC2 type for centralized baselines |

## Key Technical Decisions

### Data
- **Source:** MIMIC-IV v2.2+ via PhysioNet (credentialed access required)
- **Database:** PostgreSQL (`mimiciv_hosp`/`mimiciv_icu` schemas), BigQuery, or downloaded CSV flat files
- **Cohort:** ICU stays with LOS > 0 and ≤ 30 days, excluding in-hospital deaths
- **Features:** Demographics, admission info, first-24h vitals, first-24h labs, complexity proxies (n_procedures, n_diagnoses), DRG info (drg_code, severity, mortality subclass)
- **Cost weight proxy:** `mean_LOS_per_DRG_severity / global_mean_LOS` — MIMIC-IV has no dollar costs. Clipped to [0.1, 10.0]. Uses APR-DRG groupings (`drg_type = 'APR'`). DRG-severity groups with <10 cases are noisy — consider merging or smoothing.

### 3-Hospital Partition (care-unit-based, NOT random)
- **Hospital A (General Medical):** MICU, MICU/SICU, Neuro Intermediate, Neuro Stepdown, Neuro SICU — largest
- **Hospital B (Surgical):** SICU, TSICU — distinct surgical/trauma LOS distribution
- **Hospital C (Cardiac):** CCU, CVICU — smallest, highest acuity
- Unmatched care units default to Hospital A. Creates realistic non-IID heterogeneity.

### Model
- 3-layer MLP backbone (256→128→64) with BatchNorm + Dropout, 3 task-specific heads
- Uncertainty-weighted multi-task loss (Kendall et al., 2018): Huber (LOS, δ=5.0) + MSE (cost weight) + CrossEntropy (LOS class)
- LOS classification buckets: short (0-2d), medium (2-5d), long (5-10d), extended (10-30d)
- Adam with weight_decay=1e-4, ReduceLROnPlateau scheduler

### Federation
- **Framework:** Flower (flwr) simulation mode — `flwr.simulation.start_simulation`, no actual network I/O
- **Strategy:** FedAvg, all 3 clients every round. 50 rounds, 3 local epochs, lr=1e-3
- **Privacy:** Privacy-by-architecture (data never leaves hospital). No DP-SGD this paper — future work.
- **Resources:** `client_resources={"num_cpus": 2, "num_gpus": 0.0}` — set `num_gpus=0.33` for GPU

### Baselines (4 total)
1. Centralized MLP — all data pooled, same architecture (upper bound)
2. XGBoost Centralized — strongest non-federated baseline
3. FedAvg (3 hospitals) — our method
4. Local-only (×3) — each hospital trains independently (lower bound)

### Evaluation
- LOS regression: MAE, RMSE, R²
- Cost weight regression: MAE, RMSE, R²
- LOS classification: F1 (macro), F1 (weighted), confusion matrix
- All reported with 5-seed mean ± std (seed=42 base). Train/val 80/20 per hospital, stratified by LOS class.

### AWS Infrastructure (Pulumi)

**Design doc:** `docs/plans/2026-03-04-aws-infra-design.md`

- **Region:** `ap-southeast-1`
- **Pulumi backend:** `s3://ieee-pulumi` (S3 state backend)
- **Approach:** Single stack, multi-VPC — 5 VPCs with separate CIDR blocks
- **Infra code:** `infra/` directory with modules: network, security, compute, storage, ssm

**VPC layout:**

| VPC | CIDR | EC2 Type | Role |
|-----|------|----------|------|
| fl-server | 10.0.0.0/16 | t3.medium | Flower FedAvg aggregation server |
| hospital-a | 10.1.0.0/16 | t3.medium | Flower client (Medical ICU) |
| hospital-b | 10.2.0.0/16 | t3.medium | Flower client (Surgical) |
| hospital-c | 10.3.0.0/16 | t3.medium | Flower client (Cardiac) |
| centralized | 10.4.0.0/16 | t3.large | Centralized baselines (MLP+XGB) |

- No VPC peering — Tailscale handles all inter-instance connectivity (Flower gRPC + SSH)
- Separate VPCs provide network isolation narrative for the paper; Tailscale ACLs can enforce topology
- IAM instance profiles for S3/SSM access (no hardcoded AWS keys)
- Secrets via Pulumi config (Tailscale auth key, SSH key); runtime config via SSM Parameter Store
- Data bucket: `fedcost-data-{stack}` with raw/, partitions/, results/ prefixes

## What NOT to Do
- Do not implement DP-SGD / Opacus — scoped out for this paper
- Do not run privacy attacks (membership inference, etc.) — future work
- Do not use MIMIC-III — we use MIMIC-IV specifically for novelty
- Do not randomly partition data — always use care-unit-based split
- Do not include `hospital_expire_flag = 1` patients — exclude in-hospital deaths

## Reproducibility Notes
- Global StandardScaler fit on ALL data (simulating pre-federation calibration — acknowledge in paper limitations)
- All paths via `pathlib.Path`, never hardcoded strings
- Metrics always returned as dicts, never positional tuples
- Logging over print in production code (print OK in experiment scripts)
- Type hints on all function signatures, NumPy-style docstrings on public functions

## Paper Target
- **Conference:** IEEE ICHI 2026 or IEEE BIBM 2026
- **Format:** 8-page IEEE double-column
- **Title:** "FedCost: Federated Learning for DRG-Based Healthcare Cost Prediction on MIMIC-IV"

### Key Novelty Claims
1. First FL framework for DRG-based cost weight prediction (not just raw LOS)
2. Multi-task federated learning (3 heads) on MIMIC-IV with care-unit-based non-IID partitioning
3. Empirical quantification of accuracy cost of federation for healthcare cost prediction
4. Smaller hospitals benefit disproportionately from federation

## Task Backlog

### Phase 1: Data Pipeline
- [ ] Verify MIMIC-IV PostgreSQL access and run extraction query
- [ ] Check `drgcodes` table coverage — how many stays have APR-DRG assignments?
- [ ] Validate cost weight distribution — check for extreme outliers
- [ ] Confirm care unit names match MIMIC-IV v2.2+ schema
- [ ] Export cohort CSV to `data/raw/`

### Phase 2: Model Development
- [ ] Refactor monolithic script into `src/` module structure
- [ ] Implement `FedCostModel` with configurable backbone width/depth
- [ ] Implement multi-task loss with uncertainty weighting
- [ ] Write unit tests for model forward pass and loss computation
- [ ] Verify XGBoost baseline runs on full feature set

### Phase 3: Federation
- [ ] Set up Flower simulation with 3 clients
- [ ] Verify FedAvg convergence on toy data before full run
- [ ] Run 5-seed experiment: centralized, federated, local-only ×3, XGBoost
- [ ] Collect all metrics into `results/metrics/`

### Phase 4: Analysis & Figures
- [ ] Generate Table I (main comparison) as LaTeX
- [ ] Generate Table II (per-hospital breakdown) as LaTeX
- [ ] Generate Figures 1-6 as PDFs
- [ ] Run SHAP analysis on federated vs centralized models
- [ ] Compute statistical significance (paired t-test or Wilcoxon across seeds)

### Phase 5: Paper Writing
- [ ] Draft Introduction and Related Work
- [ ] Draft FedCost Framework section with architecture diagram
- [ ] Draft Experimental Setup
- [ ] Draft Results with tables and figure references
- [ ] Draft Discussion and Conclusion
- [ ] Format in IEEE double-column LaTeX template
