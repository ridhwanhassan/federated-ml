"""Merge distributed experiment results into the same format as run_all.py.

Reads JSON files from results/distributed/ and produces:
- results/distributed/all_results_raw.json
- results/distributed/summary.json

These are compatible with src/visualization/generate_all.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_metrics(results: list[dict], key: str = "final_metrics") -> dict:
    """Compute mean +/- std of metrics across seeds (same as run_all.py)."""
    if not results:
        return {}
    metric_keys = results[0][key].keys()
    agg = {}
    for mk in metric_keys:
        values = [r[key][mk] for r in results if mk in r.get(key, {})]
        if values:
            agg[f"{mk}_mean"] = float(np.mean(values))
            agg[f"{mk}_std"] = float(np.std(values))
    return agg


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _load_all_results(directory: Path) -> list[dict]:
    """Load all JSON result files from a directory, handling both formats.

    Handles: single result dicts, lists of results, and {"results": [...]} wrappers.
    """
    all_results = []
    for path in sorted(directory.glob("*.json")):
        data = load_json(path)
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                all_results.extend(data["results"])
            elif "experiment" in data:
                all_results.append(data)
            else:
                # Try to treat as a single result
                all_results.append(data)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge distributed results")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/distributed"),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    summary = {}

    # --- Centralized ---
    cent_path = results_dir / "centralized" / "results.json"
    cent_mlp_results = []
    cent_xgb_results = []
    if cent_path.exists():
        data = load_json(cent_path)
        # Handle both formats: {"mlp": [...], "xgboost": [...]} or flat list
        if isinstance(data, dict):
            cent_mlp_results = data.get("mlp", [])
            cent_xgb_results = data.get("xgboost", [])
        elif isinstance(data, list):
            cent_mlp_results = [r for r in data if r.get("experiment") == "centralized_mlp"]
            cent_xgb_results = [r for r in data if r.get("experiment") == "centralized_xgboost"]

        if cent_mlp_results:
            summary["centralized_mlp"] = aggregate_metrics(cent_mlp_results)
        if cent_xgb_results:
            summary["centralized_xgboost"] = aggregate_metrics(cent_xgb_results)
        logger.info("Loaded centralized: %d MLP, %d XGBoost", len(cent_mlp_results), len(cent_xgb_results))

    # --- Local-only ---
    local_results = {}
    for h_id in range(1, 6):
        path = results_dir / "local_only" / f"hospital_{h_id}.json"
        if path.exists():
            data = load_json(path)
            results_list = data if isinstance(data, list) else [data]
            key = f"local_H{h_id}"
            local_results[key] = results_list
            summary[key] = aggregate_metrics(results_list)
            logger.info("Loaded local H%d: %d results", h_id, len(results_list))

    # --- FedAvg ---
    fed_results = {}
    fed_dir = results_dir / "federated"
    if fed_dir.exists():
        results_list = _load_all_results(fed_dir)
        by_epochs: dict[int, list] = {}
        for r in results_list:
            e = r.get("local_epochs", 3)
            by_epochs.setdefault(e, []).append(r)
        for e, rs in sorted(by_epochs.items()):
            key = f"fedavg_E{e}"
            fed_results[key] = rs
            summary[key] = aggregate_metrics(rs)
            logger.info("Loaded FedAvg E=%d: %d results", e, len(rs))

    # --- Gossip ---
    gossip_results = {}
    gossip_dir = results_dir / "gossip"
    if gossip_dir.exists():
        results_list = _load_all_results(gossip_dir)
        by_epochs: dict[int, list] = {}
        for r in results_list:
            e = r.get("local_epochs", 3)
            by_epochs.setdefault(e, []).append(r)
        for e, rs in sorted(by_epochs.items()):
            key = f"gossip_E{e}"
            gossip_results[key] = rs
            summary[key] = aggregate_metrics(rs)
            logger.info("Loaded gossip E=%d: %d results", e, len(rs))

    # --- Save all_results_raw.json ---
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
    logger.info("Saved raw results to %s", raw_path)

    # --- Save summary.json ---
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    # Print table
    print("\n" + "=" * 82)
    print("DISTRIBUTED RESULTS SUMMARY")
    print("=" * 82)
    print(f"{'Experiment':<25} {'MAE':>12} {'RMSE':>12} {'R2':>12} {'W-1d':>12}")
    print("-" * 82)
    for name, metrics in summary.items():
        if "mae_mean" not in metrics:
            continue
        mae_str = f"{metrics['mae_mean']:.3f}+/-{metrics['mae_std']:.3f}"
        rmse_str = f"{metrics['rmse_mean']:.3f}+/-{metrics['rmse_std']:.3f}"
        r2_str = f"{metrics['r2_mean']:.3f}+/-{metrics['r2_std']:.3f}"
        w1d_str = f"{metrics['within_1day_mean']:.3f}+/-{metrics['within_1day_std']:.3f}"
        print(f"{name:<25} {mae_str:>12} {rmse_str:>12} {r2_str:>12} {w1d_str:>12}")
    print("=" * 82)


if __name__ == "__main__":
    main()
