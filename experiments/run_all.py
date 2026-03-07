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
    summary = {}

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
    print("SUMMARY (mean +/- std across %d seeds)" % len(seeds))
    print("=" * 70)
    print(f"{'Experiment':<25} {'MAE':>12} {'RMSE':>12} {'R2':>12}")
    print("-" * 70)
    for name, metrics in summary.items():
        if name == "elapsed_seconds":
            continue
        mae_str = f"{metrics['mae_mean']:.3f}+/-{metrics['mae_std']:.3f}"
        rmse_str = f"{metrics['rmse_mean']:.3f}+/-{metrics['rmse_std']:.3f}"
        r2_str = f"{metrics['r2_mean']:.3f}+/-{metrics['r2_std']:.3f}"
        print(f"{name:<25} {mae_str:>12} {rmse_str:>12} {r2_str:>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
