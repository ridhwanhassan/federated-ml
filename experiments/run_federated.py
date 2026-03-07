"""FedAvg (star topology) experiment with 5 hospitals.

Runs FedAvg simulation for each seed and local_epochs setting.
Results saved to results/metrics/federated/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import create_dataloaders
from src.federation.fedavg import run_fedavg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_hospital_loaders(
    cfg: dict, seed: int
) -> tuple[list[tuple], int]:
    """Load DataLoaders for all hospitals. Returns (loaders, n_features)."""
    data_cfg = cfg["data"]
    partitions_dir = PROJECT_ROOT / data_cfg["partitions_dir"]
    hospital_loaders = []
    n_features = None

    for h_idx in range(1, data_cfg["n_hospitals"] + 1):
        csv_path = partitions_dir / f"hospital_{h_idx}.csv"
        train_loader, val_loader, _ = create_dataloaders(
            csv_path=csv_path,
            batch_size=data_cfg["batch_size"],
            val_ratio=data_cfg["val_ratio"],
            seed=seed,
        )
        hospital_loaders.append((train_loader, val_loader))
        if n_features is None:
            n_features = train_loader.dataset.X.shape[1]

    return hospital_loaders, n_features


def run_single(cfg: dict, seed: int, local_epochs: int) -> dict:
    """Run a single FedAvg experiment."""
    hospital_loaders, n_features = load_hospital_loaders(cfg, seed)
    fed_cfg = cfg["federation"]
    model_cfg = cfg["model"]

    result = run_fedavg(
        hospital_loaders=hospital_loaders,
        n_features=n_features,
        n_rounds=fed_cfg["n_rounds"],
        local_epochs=local_epochs,
        lr=cfg["training"]["lr"],
        huber_delta=model_cfg["huber_delta"],
        device=cfg["experiment"]["device"],
        seed=seed,
    )

    logger.info(
        "FedAvg seed=%d E=%d: final=%s comm_cost=%d",
        seed, local_epochs, result["final_global_metrics"], result["communication_cost"],
    )

    return {
        "experiment": "fedavg",
        "seed": seed,
        "local_epochs": local_epochs,
        "n_rounds": fed_cfg["n_rounds"],
        "final_metrics": result["final_global_metrics"],
        "communication_cost": result["communication_cost"],
        "convergence_curve": [m["mae"] for m in result["round_metrics"]],
        "per_hospital_final": result["per_hospital_metrics"][-1] if result["per_hospital_metrics"] else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedAvg experiments")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "default.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["experiment"]["seeds"]
    local_epochs_list = args.local_epochs or cfg["federation"]["local_epochs"]
    if args.n_rounds is not None:
        cfg["federation"]["n_rounds"] = args.n_rounds

    out_dir = PROJECT_ROOT / cfg["experiment"]["results_dir"] / "federated"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for local_epochs in local_epochs_list:
        for seed in seeds:
            result = run_single(cfg, seed, local_epochs)
            all_results.append(result)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved FedAvg results to %s", out_path)


if __name__ == "__main__":
    main()
