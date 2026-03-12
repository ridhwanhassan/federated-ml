"""Main experiment runner — orchestrates all 5 experiment types.

Runs centralized baselines, federated (FedAvg), gossip (D-PSGD),
and local-only training, then produces a summary JSON with aggregated
statistics (mean +/- std across seeds).

Usage:
    python experiments/run_all.py --config experiments/configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_centralized import run_centralized_mlp, run_centralized_xgboost
from experiments.run_federated import load_hospital_loaders as load_fed_loaders
from experiments.run_federated import run_single as run_fedavg_single
from experiments.run_gossip import run_single as run_gossip_single
from experiments.run_local_only import run_single_hospital

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


HOSPITAL_NAMES = {
    1: "H1 (Medical)",
    2: "H2 (Neuro)",
    3: "H3 (Surgical)",
    4: "H4 (Trauma)",
    5: "H5 (Cardiac)",
}


def compute_data_statistics(cfg: dict) -> dict:
    """Compute per-hospital and overall data statistics."""
    from src.data.loader import NON_FEATURE_COLUMNS

    data_cfg = cfg["data"]
    partitions_dir = PROJECT_ROOT / data_cfg["partitions_dir"]
    features_csv = PROJECT_ROOT / data_cfg["features_csv"]

    hospital_stats = {}
    all_los = []

    for h_id in range(1, data_cfg["n_hospitals"] + 1):
        csv_path = partitions_dir / f"hospital_{h_id}.csv"
        df = pd.read_csv(csv_path)
        los = df["los"].values
        all_los.append(los)

        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        nan_counts = df[feature_cols].isna().sum()
        nan_pct = float(nan_counts.sum()) / (len(df) * len(feature_cols)) * 100

        hospital_stats[f"hospital_{h_id}"] = {
            "name": HOSPITAL_NAMES[h_id],
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "los": {
                "mean": float(np.mean(los)),
                "std": float(np.std(los)),
                "median": float(np.median(los)),
                "q25": float(np.percentile(los, 25)),
                "q75": float(np.percentile(los, 75)),
                "min": float(np.min(los)),
                "max": float(np.max(los)),
            },
            "missing_pct": round(nan_pct, 2),
        }

    # Overall pooled stats
    pooled_los = np.concatenate(all_los)
    pooled_df = pd.read_csv(features_csv)
    feature_cols = [c for c in pooled_df.columns if c not in NON_FEATURE_COLUMNS]
    pooled_nan_pct = float(pooled_df[feature_cols].isna().sum().sum()) / (
        len(pooled_df) * len(feature_cols)
    ) * 100

    hospital_stats["pooled"] = {
        "name": "All hospitals (pooled)",
        "n_samples": len(pooled_los),
        "n_features": len(feature_cols),
        "los": {
            "mean": float(np.mean(pooled_los)),
            "std": float(np.std(pooled_los)),
            "median": float(np.median(pooled_los)),
            "q25": float(np.percentile(pooled_los, 25)),
            "q75": float(np.percentile(pooled_los, 75)),
            "min": float(np.min(pooled_los)),
            "max": float(np.max(pooled_los)),
        },
        "missing_pct": round(pooled_nan_pct, 2),
    }

    return hospital_stats


def aggregate_metrics(results: list[dict], key: str = "final_metrics") -> dict:
    """Compute mean +/- std of metrics across seeds."""
    if not results:
        return {}
    metric_keys = results[0][key].keys()
    agg = {}
    for mk in metric_keys:
        values = [r[key][mk] for r in results]
        agg[f"{mk}_mean"] = float(np.mean(values))
        agg[f"{mk}_std"] = float(np.std(values))
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all FedCost experiments")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "default.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["experiment"]["seeds"]
    local_epochs_list = cfg["federation"]["local_epochs"]
    n_hospitals = cfg["data"]["n_hospitals"]

    results_dir = PROJECT_ROOT / cfg["experiment"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # --- Extract hyperparameters ---
    hyperparameters = {
        "data": {
            "n_hospitals": cfg["data"]["n_hospitals"],
            "val_ratio": cfg["data"]["val_ratio"],
            "batch_size": cfg["data"]["batch_size"],
        },
        "model": {
            "architecture": "MLP",
            "hidden_dims": cfg["model"]["hidden_dims"],
            "dropout": cfg["model"]["dropout"],
            "activation": "ReLU",
            "normalization": "BatchNorm1d",
            "loss": "HuberLoss",
            "huber_delta": cfg["model"]["huber_delta"],
        },
        "training": {
            "optimizer": "Adam",
            "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "n_epochs": cfg["training"]["n_epochs"],
            "patience": cfg["training"]["patience"],
        },
        "federation": {
            "n_rounds": cfg["federation"]["n_rounds"],
            "local_epochs_ablation": cfg["federation"]["local_epochs"],
            "default_local_epochs": cfg["federation"]["default_local_epochs"],
            "fedavg_aggregation": "weighted_average_by_dataset_size",
            "gossip_topology": "ring",
            "gossip_mixing": "metropolis_hastings",
            "gossip_mixing_weights": "1/3 self, 1/3 left, 1/3 right",
        },
        "xgboost": {
            "n_estimators": cfg["xgboost"]["n_estimators"],
            "max_depth": cfg["xgboost"]["max_depth"],
            "learning_rate": cfg["xgboost"]["learning_rate"],
            "early_stopping_rounds": cfg["xgboost"]["early_stopping_rounds"],
        },
        "experiment": {
            "seeds": seeds,
            "device": cfg["experiment"]["device"],
        },
    }

    # --- Compute data statistics ---
    logger.info("Computing data statistics per hospital...")
    data_stats = compute_data_statistics(cfg)

    summary = {"hyperparameters": hyperparameters, "data_statistics": data_stats}

    # --- 1. Centralized MLP ---
    logger.info("=" * 60)
    logger.info("Running Centralized MLP baselines")
    logger.info("=" * 60)
    cent_mlp_results = []
    for seed in seeds:
        cent_mlp_results.append(run_centralized_mlp(cfg, seed))
    summary["centralized_mlp"] = aggregate_metrics(cent_mlp_results)

    # --- 2. Centralized XGBoost ---
    logger.info("=" * 60)
    logger.info("Running Centralized XGBoost baselines")
    logger.info("=" * 60)
    cent_xgb_results = []
    for seed in seeds:
        cent_xgb_results.append(run_centralized_xgboost(cfg, seed))
    summary["centralized_xgboost"] = aggregate_metrics(cent_xgb_results)

    # --- 3. FedAvg (star) ---
    logger.info("=" * 60)
    logger.info("Running FedAvg experiments")
    logger.info("=" * 60)
    fed_results = {}
    for local_epochs in local_epochs_list:
        key = f"fedavg_E{local_epochs}"
        results_for_e = []
        for seed in seeds:
            results_for_e.append(run_fedavg_single(cfg, seed, local_epochs))
        fed_results[key] = results_for_e
        summary[key] = aggregate_metrics(results_for_e)

    # --- 4. D-PSGD (ring) ---
    logger.info("=" * 60)
    logger.info("Running D-PSGD gossip experiments")
    logger.info("=" * 60)
    gossip_results = {}
    for local_epochs in local_epochs_list:
        key = f"gossip_E{local_epochs}"
        results_for_e = []
        for seed in seeds:
            results_for_e.append(run_gossip_single(cfg, seed, local_epochs))
        gossip_results[key] = results_for_e
        summary[key] = aggregate_metrics(results_for_e)

    # --- 5. Local-only ---
    logger.info("=" * 60)
    logger.info("Running local-only training")
    logger.info("=" * 60)
    local_results = {}
    for h_id in range(1, n_hospitals + 1):
        key = f"local_H{h_id}"
        results_for_h = []
        for seed in seeds:
            results_for_h.append(run_single_hospital(cfg, h_id, seed))
        local_results[key] = results_for_h
        summary[key] = aggregate_metrics(results_for_h)

    elapsed = time.time() - start_time

    # --- Save all raw results ---
    all_raw = {
        "hyperparameters": hyperparameters,
        "data_statistics": data_stats,
        "centralized_mlp": cent_mlp_results,
        "centralized_xgboost": cent_xgb_results,
        "fedavg": fed_results,
        "gossip": gossip_results,
        "local_only": local_results,
    }

    raw_path = results_dir / "all_results_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_raw, f, indent=2)

    # --- Save summary ---
    summary["elapsed_seconds"] = elapsed
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("All experiments complete in %.1f seconds", elapsed)
    logger.info("Raw results: %s", raw_path)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)

    # Print summary table
    print("\n" + "=" * 70)
    print("HYPERPARAMETERS")
    print("=" * 70)
    print(f"  Model:        MLP {hyperparameters['model']['hidden_dims']} "
          f"(dropout={hyperparameters['model']['dropout']})")
    print(f"  Loss:         Huber (delta={hyperparameters['model']['huber_delta']})")
    print(f"  Optimizer:    Adam (lr={hyperparameters['training']['lr']}, "
          f"wd={hyperparameters['training']['weight_decay']})")
    print(f"  Scheduler:    ReduceLROnPlateau (factor=0.5)")
    print(f"  Training:     max_epochs={hyperparameters['training']['n_epochs']}, "
          f"patience={hyperparameters['training']['patience']}")
    print(f"  Federation:   {hyperparameters['federation']['n_rounds']} rounds, "
          f"E={hyperparameters['federation']['local_epochs_ablation']}")
    print(f"  XGBoost:      n_est={hyperparameters['xgboost']['n_estimators']}, "
          f"depth={hyperparameters['xgboost']['max_depth']}, "
          f"lr={hyperparameters['xgboost']['learning_rate']}")
    print(f"  Data:         {hyperparameters['data']['n_hospitals']} hospitals, "
          f"batch={hyperparameters['data']['batch_size']}, "
          f"val_ratio={hyperparameters['data']['val_ratio']}")
    print(f"  Seeds:        {hyperparameters['experiment']['seeds']}")

    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print(f"{'Hospital':<22} {'N':>7} {'LOS mean':>10} {'LOS med':>9} "
          f"{'LOS std':>9} {'Q25-Q75':>13} {'Miss%':>7}")
    print("-" * 82)
    for key in [f"hospital_{i}" for i in range(1, n_hospitals + 1)] + ["pooled"]:
        ds = data_stats[key]
        los = ds["los"]
        q_range = f"{los['q25']:.1f}-{los['q75']:.1f}"
        print(f"{ds['name']:<22} {ds['n_samples']:>7} {los['mean']:>10.2f} "
              f"{los['median']:>9.2f} {los['std']:>9.2f} {q_range:>13} "
              f"{ds['missing_pct']:>6.1f}%")
    print("=" * 82)

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std across %d seeds)" % len(seeds))
    print("=" * 70)
    print(f"{'Experiment':<25} {'MAE':>12} {'RMSE':>12} {'R2':>12} {'W-1d':>12}")
    print("-" * 82)
    for name, metrics in summary.items():
        if name in ("elapsed_seconds", "hyperparameters", "data_statistics"):
            continue
        mae_str = f"{metrics['mae_mean']:.3f}+/-{metrics['mae_std']:.3f}"
        rmse_str = f"{metrics['rmse_mean']:.3f}+/-{metrics['rmse_std']:.3f}"
        r2_str = f"{metrics['r2_mean']:.3f}+/-{metrics['r2_std']:.3f}"
        w1d_str = f"{metrics['within_1day_mean']:.3f}+/-{metrics['within_1day_std']:.3f}"
        print(f"{name:<25} {mae_str:>12} {rmse_str:>12} {r2_str:>12} {w1d_str:>12}")
    print("=" * 82)


if __name__ == "__main__":
    main()
