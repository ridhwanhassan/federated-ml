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

# Pulumi (one-time backend setup)
pulumi login 's3://ieee-pulumi?region=ap-southeast-1&awssdk=v2'
pulumi stack init dev

# Pulumi stack operations (run from infra/)
cd infra/
pulumi up                    # Preview and deploy changes
pulumi preview               # Preview changes without deploying
pulumi destroy               # Tear down all resources
pulumi stack ls              # List stacks
pulumi stack select <name>   # Switch active stack

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
pytest tests/ -v
```

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
