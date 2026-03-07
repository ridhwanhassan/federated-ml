"""Centralized baselines: MLP and XGBoost trained on all pooled data.

These represent the upper bound — what's achievable with full data access.
Results are saved per-seed to results/metrics/centralized/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import create_dataloaders
from src.models.mlp import LOSModel, evaluate, train_model
from src.models.xgboost_baseline import evaluate_xgboost, train_xgboost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_centralized_mlp(cfg: dict, seed: int) -> dict:
    """Train centralized MLP on all data and return metrics."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    train_loader, val_loader, scaler = create_dataloaders(
        csv_path=PROJECT_ROOT / data_cfg["features_csv"],
        batch_size=data_cfg["batch_size"],
        val_ratio=data_cfg["val_ratio"],
        seed=seed,
    )

    n_features = train_loader.dataset.X.shape[1]
    model = LOSModel(
        n_features=n_features,
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=train_cfg["n_epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        huber_delta=model_cfg["huber_delta"],
        patience=train_cfg["patience"],
        device=cfg["experiment"]["device"],
    )

    final_metrics = evaluate(model, val_loader, device=cfg["experiment"]["device"])
    logger.info("Centralized MLP seed=%d: %s", seed, final_metrics)

    return {
        "experiment": "centralized_mlp",
        "seed": seed,
        "final_metrics": final_metrics,
        "best_val_mae": history["best_val_mae"],
        "n_epochs_trained": len(history["train_losses"]),
        "convergence_curve": [m["mae"] for m in history["val_metrics"]],
    }


def run_centralized_xgboost(cfg: dict, seed: int) -> dict:
    """Train centralized XGBoost on all data and return metrics."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from src.data.loader import NON_FEATURE_COLUMNS

    data_cfg = cfg["data"]
    xgb_cfg = cfg["xgboost"]

    df = pd.read_csv(PROJECT_ROOT / data_cfg["features_csv"])
    y = df["los"].values.astype(np.float32)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=data_cfg["val_ratio"], random_state=seed
    )

    # Impute NaNs with training medians
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    for col_idx in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, col_idx]), col_idx] = train_medians[col_idx]
        X_val[np.isnan(X_val[:, col_idx]), col_idx] = train_medians[col_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = train_xgboost(
        X_train, y_train,
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        X_val=X_val,
        y_val=y_val,
        seed=seed,
    )

    final_metrics = evaluate_xgboost(model, X_val, y_val)
    logger.info("Centralized XGBoost seed=%d: %s", seed, final_metrics)

    return {
        "experiment": "centralized_xgboost",
        "seed": seed,
        "final_metrics": final_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centralized baselines")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "default.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["experiment"]["seeds"]
    out_dir = PROJECT_ROOT / cfg["experiment"]["results_dir"] / "centralized"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        mlp_result = run_centralized_mlp(cfg, seed)
        xgb_result = run_centralized_xgboost(cfg, seed)
        all_results.extend([mlp_result, xgb_result])

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved centralized results to %s", out_path)


if __name__ == "__main__":
    main()
