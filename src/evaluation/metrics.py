"""Post-experiment analysis: convergence, communication cost, fairness.

Loads JSON results from experiments and computes derived metrics
for paper tables and figures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> dict:
    """Load all_results_raw.json from a results directory."""
    raw_path = results_dir / "all_results_raw.json"
    with open(raw_path) as f:
        return json.load(f)


def convergence_round(
    convergence_curve: list[float],
    target_mae: float,
    fraction: float = 0.95,
) -> int | None:
    """Find the first round where MAE reaches within fraction of target.

    Parameters
    ----------
    convergence_curve : list[float]
        Per-round MAE values.
    target_mae : float
        Reference MAE (e.g. centralized baseline).
    fraction : float
        Fraction of target to reach (0.95 means within 95% of target MAE).

    Returns
    -------
    int or None
        1-indexed round number, or None if never reached.
    """
    threshold = target_mae / fraction  # MAE must go below this
    for i, mae in enumerate(convergence_curve):
        if mae <= threshold:
            return i + 1
    return None


def aggregate_over_seeds(
    results: list[dict],
    metric_key: str = "final_metrics",
) -> dict[str, dict[str, float]]:
    """Compute mean and std of metrics across seeds.

    Returns
    -------
    dict
        ``{"mae": {"mean": ..., "std": ...}, "rmse": {...}, "r2": {...}, "within_1day": {...}}``
    """
    if not results:
        return {}
    metric_names = list(results[0][metric_key].keys())
    agg = {}
    for name in metric_names:
        values = [r[metric_key][name] for r in results]
        agg[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return agg


def per_hospital_fairness(
    per_hospital_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Compute fairness metrics across hospitals for a single run.

    Parameters
    ----------
    per_hospital_metrics : list[dict]
        List of ``{"mae": ..., "rmse": ..., "r2": ...}`` per hospital.

    Returns
    -------
    dict
        ``{"mae_std": ..., "mae_range": ..., "mae_max": ..., "mae_min": ...}``
    """
    maes = [m["mae"] for m in per_hospital_metrics]
    return {
        "mae_std": float(np.std(maes)),
        "mae_range": float(max(maes) - min(maes)),
        "mae_max": float(max(maes)),
        "mae_min": float(min(maes)),
    }


def communication_cost_summary(results: list[dict]) -> dict[str, float]:
    """Aggregate communication costs across seeds.

    Parameters
    ----------
    results : list[dict]
        Results with ``"communication_cost"`` key.

    Returns
    -------
    dict
        ``{"mean": ..., "std": ...}`` in total parameters exchanged.
    """
    costs = [r["communication_cost"] for r in results]
    return {"mean": float(np.mean(costs)), "std": float(np.std(costs))}


def build_table_i(raw: dict) -> list[dict]:
    """Build Table I: main comparison across all experiments.

    Returns a list of rows with experiment name and mean+std metrics.
    """
    rows = []

    # Centralized MLP
    cent_mlp = raw["centralized_mlp"]
    agg = aggregate_over_seeds(cent_mlp)
    rows.append({"experiment": "Centralized MLP", **_format_row(agg)})

    # Centralized XGBoost
    cent_xgb = raw["centralized_xgboost"]
    agg = aggregate_over_seeds(cent_xgb)
    rows.append({"experiment": "Centralized XGBoost", **_format_row(agg)})

    # FedAvg (default E=3)
    for key, results in raw["fedavg"].items():
        local_e = results[0]["local_epochs"] if results else "?"
        agg = aggregate_over_seeds(results)
        comm = communication_cost_summary(results)
        rows.append({
            "experiment": f"FedAvg (E={local_e})",
            **_format_row(agg),
            "comm_cost": comm["mean"],
        })

    # D-PSGD (default E=3)
    for key, results in raw["gossip"].items():
        local_e = results[0]["local_epochs"] if results else "?"
        agg = aggregate_over_seeds(results)
        comm = communication_cost_summary(results)
        rows.append({
            "experiment": f"D-PSGD (E={local_e})",
            **_format_row(agg),
            "comm_cost": comm["mean"],
        })

    # Local-only (averaged across all hospitals)
    all_local = []
    for key, results in raw["local_only"].items():
        all_local.extend(results)
    agg = aggregate_over_seeds(all_local)
    rows.append({"experiment": "Local-only (avg)", **_format_row(agg)})

    return rows


def build_table_ii(raw: dict) -> list[dict]:
    """Build Table II: per-hospital breakdown for FedAvg vs D-PSGD vs local.

    Uses default local_epochs=3 (E=3).
    """
    hospital_names = ["H1 (Medical)", "H2 (Neuro)", "H3 (Surgical)",
                      "H4 (Trauma)", "H5 (Cardiac)"]
    rows = []

    # Get E=3 results
    fedavg_e3 = raw["fedavg"].get("fedavg_E3", [])
    gossip_e3 = raw["gossip"].get("gossip_E3", [])

    for h_idx, h_name in enumerate(hospital_names):
        row = {"hospital": h_name}

        # FedAvg per-hospital (average across seeds)
        if fedavg_e3:
            fed_maes = [r["per_hospital_final"][h_idx]["mae"] for r in fedavg_e3]
            row["fedavg_mae_mean"] = float(np.mean(fed_maes))
            row["fedavg_mae_std"] = float(np.std(fed_maes))

        # D-PSGD per-hospital
        if gossip_e3:
            gos_maes = [r["per_hospital_final"][h_idx]["mae"] for r in gossip_e3]
            row["gossip_mae_mean"] = float(np.mean(gos_maes))
            row["gossip_mae_std"] = float(np.std(gos_maes))

        # Local-only for this hospital
        local_key = f"local_H{h_idx + 1}"
        if local_key in raw["local_only"]:
            local_results = raw["local_only"][local_key]
            local_maes = [r["final_metrics"]["mae"] for r in local_results]
            row["local_mae_mean"] = float(np.mean(local_maes))
            row["local_mae_std"] = float(np.std(local_maes))

        rows.append(row)

    return rows


def _format_row(agg: dict[str, dict[str, float]]) -> dict[str, str]:
    """Format aggregated metrics into display strings."""
    row = {
        "mae": f"{agg['mae']['mean']:.3f} ± {agg['mae']['std']:.3f}",
        "rmse": f"{agg['rmse']['mean']:.3f} ± {agg['rmse']['std']:.3f}",
        "r2": f"{agg['r2']['mean']:.3f} ± {agg['r2']['std']:.3f}",
    }
    if "within_1day" in agg:
        row["within_1day"] = f"{agg['within_1day']['mean']:.3f} ± {agg['within_1day']['std']:.3f}"
    return row
