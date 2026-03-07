# Evaluation Metrics

This document describes the evaluation metrics used in FedCost for assessing ICU length-of-stay (LOS) prediction performance across all experiment configurations (centralized, federated, gossip, and local-only).

## Overview

All evaluation functions return a consistent metric dictionary with the following keys:

| Key | Metric | Range | Optimal |
|-----|--------|-------|---------|
| `mae` | Mean Absolute Error | [0, +inf) | Lower is better |
| `rmse` | Root Mean Squared Error | [0, +inf) | Lower is better |
| `r2` | R-squared (coefficient of determination) | (-inf, 1] | Higher is better |
| `within_1_day` | Within-1-day accuracy | [0, 1] | Higher is better |
| `within_2_day` | Within-2-day accuracy | [0, 1] | Higher is better |
| `within_3_day` | Within-3-day accuracy | [0, 1] | Higher is better |

## Metric Definitions

### Mean Absolute Error (MAE)

The average absolute difference between predicted and true LOS values.

```
MAE = (1/n) * sum(|y_pred_i - y_true_i|)
```

**Interpretation:** On average, how many days off is each prediction. A MAE of 1.5 means predictions are off by 1.5 days on average.

### Root Mean Squared Error (RMSE)

The square root of the average squared differences. Penalizes large errors more heavily than MAE.

```
RMSE = sqrt((1/n) * sum((y_pred_i - y_true_i)^2))
```

**Interpretation:** Similar to MAE but more sensitive to outlier predictions. RMSE >= MAE always holds; a large gap between RMSE and MAE indicates the presence of large individual errors.

### R-squared (R²)

The proportion of variance in the true LOS explained by the model.

```
R² = 1 - SS_res / SS_tot
```

Where `SS_res = sum((y_true - y_pred)^2)` and `SS_tot = sum((y_true - y_mean)^2)`.

**Interpretation:** R² = 1.0 means perfect prediction. R² = 0 means the model performs no better than predicting the mean LOS. Negative R² means the model is worse than predicting the mean.

### Within-k-days Accuracy

The fraction of predictions that fall within k days of the true LOS value.

```
within_k_accuracy = (1/n) * sum(|y_pred_i - y_true_i| <= k)
```

This metric is computed at three clinically relevant thresholds:

| Metric | Threshold | Clinical Relevance |
|--------|-----------|-------------------|
| `within_1_day` | k = 1 day | Tight accuracy — useful for short-stay discharge planning |
| `within_2_day` | k = 2 days | Moderate tolerance — covers most operational planning needs |
| `within_3_day` | k = 3 days | Broader tolerance — captures general predictive reliability |

**Interpretation:** A `within_1_day` accuracy of 0.45 means 45% of predictions are within 1 day of the true LOS. This is more clinically interpretable than MAE/RMSE because it directly answers: "How often is the model close enough to be useful?"

**Why this metric matters for ICU LOS prediction:**
- Hospital discharge planning requires predictions within an actionable tolerance window
- MAE alone can be misleading — a model with MAE = 2.0 could have most predictions very close but a few very far off, or all predictions consistently 2 days off
- Within-k-day accuracy reveals the distribution of errors, not just their average
- Different k thresholds let stakeholders pick the tolerance relevant to their use case

## Where Metrics Are Computed

### Model-level evaluation

- **MLP:** `src/models/mlp.py:evaluate()` — evaluates a PyTorch model on a DataLoader
- **XGBoost:** `src/models/xgboost_baseline.py:evaluate_xgboost()` — evaluates an XGBoost model on numpy arrays

Both return the same metric dictionary format.

### Federation-level evaluation

- **FedAvg (star):** `src/federation/fedavg.py:run_fedavg()` — evaluates the global model on each hospital's validation set after each round. Round-level metrics are computed as a **weighted average** by dataset size.
- **D-PSGD (ring):** `src/federation/gossip.py:run_gossip()` — evaluates each hospital's local model on its own validation set. Round-level metrics are computed as a **simple mean** across hospitals.

Both return per-round and per-hospital metric breakdowns.

### Standalone metric function

The `within_k_days_accuracy` function is also available standalone:

```python
from src.evaluation.metrics import within_k_days_accuracy

accuracy = within_k_days_accuracy(y_true, y_pred, k=1.0)
```

## Federation Result Structure

Both `run_fedavg()` and `run_gossip()` return:

```python
{
    "round_metrics": [
        # One dict per round with all 6 metrics averaged across hospitals
        {"mae": ..., "rmse": ..., "r2": ..., "within_1_day": ..., "within_2_day": ..., "within_3_day": ...},
        ...
    ],
    "per_hospital_metrics": [
        # per_hospital_metrics[round_idx][hospital_idx] -> metric dict
        [{"mae": ..., ...}, {"mae": ..., ...}, ...],  # round 0
        ...
    ],
    "final_global_metrics": {"mae": ..., "rmse": ..., "r2": ..., ...},
    "communication_cost": int,  # Total model parameters exchanged
}
```

## Additional Evaluation Dimensions

Beyond per-prediction metrics, FedCost tracks:

- **Convergence rate:** Rounds to reach 95% of centralized MAE
- **Communication cost:** Total model parameters exchanged (FedAvg: `2 * N * params/round`; D-PSGD: `2 * 2 * params/node/round`)
- **Fairness:** Per-hospital MAE variance (std across 5 hospitals)
- **Ablation:** Local epochs E = {1, 3, 5} for both topologies
- **Statistical reporting:** 5-seed mean +/- std (seeds 42-46)
