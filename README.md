# FedCost

Topology-aware comparison of **FedAvg (star) vs D-PSGD (ring gossip)** for ICU length-of-stay prediction on MIMIC-IV, targeting an IEEE conference paper (ICHI or BIBM 2026).

**Research question:** How does the choice of communication topology -- centralized FedAvg (star) vs decentralized D-PSGD (ring) -- affect convergence and accuracy for ICU length-of-stay prediction across non-IID hospital data?

## Architecture

```
        FedAvg (Star)                    D-PSGD (Ring)

   ┌──────────────────┐
   │   Central Server  │
   └─┬──┬──┬──┬──┬────┘
     │  │  │  │  │           H1 ── H2 ── H3 ── H4 ── H5 ── H1
    H1 H2 H3 H4 H5          (each node exchanges with 2 neighbors)
```

Five simulated hospitals train locally on care-unit-partitioned MIMIC-IV data:

| Hospital | Care Units | Samples |
|----------|-----------|---------|
| H1 (Medical) | MICU, MICU/SICU | ~30k |
| H2 (Neuro) | Neuro Int., Neuro Step., Neuro SICU | ~8k |
| H3 (Surgical) | SICU | ~11k |
| H4 (Trauma) | TSICU | ~9k |
| H5 (Cardiac) | CCU, CVICU | ~23k |

The model is a single-task MLP (256 -> 128 -> 64, BatchNorm + Dropout) with Huber loss for LOS regression.

## Project Structure

```
fedcost/
├── infra/                  # Pulumi IaC (AWS, 6 VPCs, 7 EC2 instances)
├── src/
│   ├── data/               # Extract, partition, features, loader
│   ├── models/             # LOS MLP, XGBoost baseline
│   ├── federation/         # FedAvg (star), D-PSGD (ring gossip)
│   ├── evaluation/         # Post-experiment metrics and analysis
│   └── visualization/      # Paper figures and tables
├── experiments/            # Experiment runner scripts + config
├── paper/                  # LaTeX manuscript (IEEE double-column)
├── data/processed/         # Feature-engineered CSV + hospital partitions
├── results/metrics/        # Experiment output (JSON)
├── tests/                  # 104 unit tests
└── docs/plans/             # Design and implementation documents
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests (no AWS credentials needed)
uv sync --extra test
uv run pytest tests/ -v -W ignore::DeprecationWarning
```

## Running Experiments

### Smoke test (1 seed, fast)

```bash
# Centralized baselines (MLP + XGBoost) -- ~30s
uv run python experiments/run_centralized.py --seeds 42

# Local-only (each hospital trains independently) -- ~2min
uv run python experiments/run_local_only.py --seeds 42

# FedAvg with 3 rounds (quick test) -- ~20s
uv run python experiments/run_federated.py --seeds 42 --local-epochs 3 --n-rounds 3

# D-PSGD with 3 rounds (quick test) -- ~20s
uv run python experiments/run_gossip.py --seeds 42 --local-epochs 3 --n-rounds 3
```

### Full experiment run (5 seeds x 3 local epoch settings x 50 rounds)

```bash
# Runs all 5 experiments -- takes hours
uv run python experiments/run_all.py

# Or reduce scope
uv run python experiments/run_all.py --seeds 42 43
```

### What each script produces

| Command | Time | Output |
|---------|------|--------|
| `run_centralized.py --seeds 42` | ~30s | `results/metrics/centralized/results.json` |
| `run_local_only.py --seeds 42` | ~2min | `results/metrics/local_only/results.json` |
| `run_federated.py --seeds 42 --local-epochs 3 --n-rounds 3` | ~20s | `results/metrics/federated/results.json` |
| `run_gossip.py --seeds 42 --local-epochs 3 --n-rounds 3` | ~20s | `results/metrics/gossip/results.json` |
| `run_all.py` (full, 5 seeds) | hours | `results/metrics/summary.json` + `all_results_raw.json` |

## Generating Figures and Tables

After `run_all.py` completes (creates `results/metrics/all_results_raw.json`):

```bash
uv run python src/visualization/generate_all.py
```

Outputs:
- `paper/figures/fig1_topology.pdf` -- Architecture diagram (star vs ring)
- `paper/figures/fig2_convergence.pdf` -- MAE vs communication rounds
- `paper/figures/fig3_per_hospital.pdf` -- Per-hospital performance bars
- `paper/figures/fig4_ablation.pdf` -- Local epochs ablation (E=1,3,5)
- `paper/figures/fig5_cost_accuracy.pdf` -- Communication cost vs accuracy
- `paper/tables/table_i.tex` -- Main comparison table (LaTeX)
- `paper/tables/table_ii.tex` -- Per-hospital breakdown (LaTeX)

## Running Tests

```bash
# All tests (104 pass, 22 skipped)
uv run pytest tests/ -v -W ignore::DeprecationWarning

# Infrastructure unit tests only (mocked, no AWS needed)
uv run pytest tests/test_infra_unit.py -v

# Post-deploy smoke tests (requires AWS credentials + deployed stack)
uv run pytest tests/test_infra_deployed.py -v --run-deployed
```

## Infrastructure (AWS)

AWS infrastructure is managed with Pulumi (Python): 6 VPCs, 7 EC2 instances, S3 bucket, SSM parameters. Tailscale handles all inter-instance connectivity. See [CLAUDE.md](CLAUDE.md) for full deployment walkthrough and stack configuration.

## Paper

Targeting IEEE ICHI 2026 or IEEE BIBM 2026 (8-page double-column). Manuscript source in `paper/main.tex`.

## License

MIT
