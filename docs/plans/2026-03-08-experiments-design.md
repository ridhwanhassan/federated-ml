# Experiments Design — Phase 4

**Date:** 2026-03-08
**Phase:** 4 (Experiments)
**Scope:** Experiment runner scripts and YAML config in `experiments/`

## Decision

Standalone Python scripts (Option A — simple CLI scripts) rather than a framework-based experiment manager. Each script loads data via `src/data/loader`, trains via `src/models/` and `src/federation/`, and dumps JSON results. An orchestrator (`run_all.py`) runs everything and produces a summary.

## File Structure

```
experiments/
├── __init__.py
├── configs/
│   └── default.yaml          # All hyperparameters
├── run_all.py                 # Orchestrator — runs all 5 experiments
├── run_centralized.py         # Centralized MLP + XGBoost baselines
├── run_federated.py           # FedAvg (star) simulation
├── run_gossip.py              # D-PSGD (ring) simulation
└── run_local_only.py          # Per-hospital independent training
```

## Configuration (`default.yaml`)

Single YAML file with sections: `data`, `model`, `training`, `federation`, `xgboost`, `experiment`. All numeric values use explicit float notation (e.g. `0.001` not `1e-3`) for YAML `safe_load` compatibility.

Key settings:
- 5 seeds: 42–46
- Federation: 50 rounds, local epochs E ∈ {1, 3, 5}
- Training: lr=0.001, weight_decay=0.0001, patience=10, max 100 epochs
- Device: CPU (configurable)

## Experiment Scripts

### `run_centralized.py`
- Loads `features.csv` (all pooled data)
- Trains centralized MLP via `train_model()` with early stopping
- Trains centralized XGBoost via `train_xgboost()`
- Outputs per-seed: final metrics, convergence curve, epochs trained

### `run_federated.py`
- Loads per-hospital partition CSVs via `create_dataloaders()`
- Calls `run_fedavg()` from `src/federation/fedavg`
- Iterates over seeds × local_epochs settings
- Outputs: final metrics, convergence curve, per-hospital metrics, communication cost

### `run_gossip.py`
- Same data loading as `run_federated.py`
- Calls `run_gossip()` from `src/federation/gossip`
- Same iteration and output format

### `run_local_only.py`
- Loads each hospital's partition independently
- Trains via `train_model()` (same as centralized, per-hospital data)
- Outputs per-hospital per-seed: final metrics, epochs trained

### `run_all.py`
- Orchestrates all 4 scripts above
- Aggregates mean ± std across seeds
- Saves `all_results_raw.json` (full per-seed data) and `summary.json`
- Prints formatted summary table to stdout

## Output Format

All results saved as JSON to `results/metrics/<experiment>/results.json`.

Each result dict contains:
- `experiment`: string identifier
- `seed`: int
- `final_metrics`: `{"mae": float, "rmse": float, "r2": float, "within_1day": float}`
- `convergence_curve`: list of per-round/epoch MAE (where applicable)
- `per_hospital_final`: list of per-hospital metric dicts (federated/gossip only)
- `communication_cost`: int total parameters exchanged (federated/gossip only)

## CLI Interface

All scripts accept:
- `--config`: path to YAML config (default: `experiments/configs/default.yaml`)
- `--seeds`: override seed list

`run_federated.py` and `run_gossip.py` additionally accept:
- `--n-rounds`: override number of rounds
- `--local-epochs`: override local epoch settings
