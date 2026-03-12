# Hyperparameters by Model Type

All model types share the same `LOSModel` architecture. No hyperparameter optimization is performed — all values are fixed in `experiments/configs/default.yaml`.

## Shared MLP Architecture

Used by Centralized MLP, Local-only, FedAvg, and D-PSGD.

| Parameter | Value | Source |
|---|---|---|
| Architecture | Input → 256 → 128 → 64 → 1 | `model.hidden_dims` |
| Activation | ReLU | hardcoded in `LOSModel` |
| Normalization | BatchNorm1d (per hidden layer) | hardcoded in `LOSModel` |
| Dropout | 0.3 | `model.dropout` |
| Loss | HuberLoss (delta=5.0) | `model.huber_delta` |
| Optimizer | Adam | hardcoded |
| Learning rate | 0.001 | `training.lr` |
| Weight decay | 1e-4 | `training.weight_decay` |
| Batch size | 64 | `data.batch_size` |

## Centralized MLP & Local-only

| Parameter | Value |
|---|---|
| Max epochs | 100 |
| Early stopping patience | 10 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

## FedAvg (Star Topology)

| Parameter | Value |
|---|---|
| Rounds | 50 |
| Local epochs | 1, 3, 5 (ablation) |
| Aggregation | Weighted average by dataset size |
| Scheduler | None (fresh Adam optimizer each round) |
| Weight decay | 0 (not passed — Adam default) |

> **Note:** FedAvg does not pass `weight_decay` to its local Adam optimizers (`fedavg.py:116`), unlike the centralized/local-only MLP which uses `weight_decay=1e-4`. This is an inconsistency.

## D-PSGD (Ring Topology)

| Parameter | Value |
|---|---|
| Rounds | 50 |
| Local epochs | 1, 3, 5 (ablation) |
| Topology | Ring (H1↔H2↔H3↔H4↔H5↔H1) |
| Mixing weights | Metropolis-Hastings: 1/3 self, 1/3 left, 1/3 right |
| Scheduler | None (fresh Adam optimizer each round) |
| Weight decay | 0 (not passed — Adam default) |

> **Note:** Same `weight_decay` inconsistency as FedAvg (`gossip.py:109`).

## XGBoost Centralized

| Parameter | Value | Source |
|---|---|---|
| n_estimators | 500 | `xgboost.n_estimators` |
| max_depth | 6 | `xgboost.max_depth` |
| learning_rate | 0.05 | `xgboost.learning_rate` |
| early_stopping_rounds | 20 | `xgboost.early_stopping_rounds` |
| n_jobs | -1 | hardcoded |

## Experiment Settings

| Parameter | Value |
|---|---|
| Seeds | 42, 43, 44, 45, 46 |
| Validation ratio | 0.2 |
| Device | cpu |
| Number of hospitals | 5 |
