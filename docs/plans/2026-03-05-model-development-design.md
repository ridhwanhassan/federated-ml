# Model Development Design — MLP for ICU LOS Prediction

**Date:** 2026-03-05
**Phase:** 2 (Model Development)
**Scope:** Single-task MLP + training utilities in `src/models/mlp.py`, plus unit tests

## Architecture

Single file: `src/models/mlp.py`

### LOSModel (nn.Module)

```
Input(n_features)
  → Linear(n_features, 256) → BatchNorm1d(256) → ReLU → Dropout(p)
  → Linear(256, 128)        → BatchNorm1d(128) → ReLU → Dropout(p)
  → Linear(128, 64)         → BatchNorm1d(64)  → ReLU → Dropout(p)
  → Linear(64, 1)
```

- `n_features`: constructor arg (not hardcoded — depends on categorical encoding)
- `hidden_dims`: default `(256, 128, 64)`, configurable
- `dropout`: default `0.3`, configurable

### Functions

| Function | Purpose | Used by |
|----------|---------|---------|
| `train_one_epoch(model, loader, optimizer, criterion, device)` | One pass over training data, returns avg loss | Federation (FedAvg, D-PSGD), centralized, local-only |
| `evaluate(model, loader, device)` | Compute MAE, RMSE, R² on a loader | All experiments |
| `train_model(model, train_loader, val_loader, ...)` | Full training loop (Adam, scheduler, early stopping) | Centralized baseline, local-only baseline |

### train_model parameters

- `n_epochs`: default 100
- `lr`: default 1e-3
- `weight_decay`: default 1e-4
- `huber_delta`: default 5.0
- `patience`: default 10 (early stopping on val MAE)
- `device`: auto-detect CPU/CUDA
- Returns: `{"train_losses": [...], "val_metrics": [{"mae":, "rmse":, "r2":}], "best_val_mae": float}`

### evaluate return format

```python
{"mae": float, "rmse": float, "r2": float}
```

## Interface contract for federation

Federation layer (Phase 3) will:
1. Create `LOSModel(n_features=...)` per hospital
2. Call `train_one_epoch()` for E local epochs
3. Read/write `model.state_dict()` for aggregation or neighbor exchange
4. Call `evaluate()` after each round

`train_model()` is NOT used by federation — only by centralized/local-only baselines.

## Files to create

1. `src/models/__init__.py` — exports LOSModel, train_one_epoch, evaluate, train_model
2. `src/models/mlp.py` — all model code
3. `tests/test_model.py` — unit tests

## Test plan

- Model forward pass with random input produces correct output shape
- Model handles variable n_features
- train_one_epoch reduces loss over a few steps (smoke test with synthetic data)
- evaluate returns dict with correct keys and valid ranges
- train_model runs for a few epochs without error (synthetic data)
- Huber loss is used (not MSE)

## Dependencies

No new dependencies needed — `torch`, `numpy`, `scikit-learn` already in `pyproject.toml [ml]`.

## Decisions

- No XGBoost in this phase (deferred)
- No experiment configs yet (Phase 4)
- No experiment runner scripts yet (Phase 4)
- Dropout default 0.3 (standard for tabular MLP; ablatable later)
