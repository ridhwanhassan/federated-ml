"""Local-only training: each hospital trains independently.

This represents the lower bound — no collaboration between hospitals.
Results saved to results/metrics/local_only/.
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
from src.models.mlp import LOSModel, evaluate, train_model

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


def run_single_hospital(cfg: dict, hospital_id: int, seed: int) -> dict:
    """Train a local model for one hospital."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    csv_path = PROJECT_ROOT / data_cfg["partitions_dir"] / f"hospital_{hospital_id}.csv"
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=csv_path,
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
    logger.info(
        "Local-only %s seed=%d: %s",
        HOSPITAL_NAMES[hospital_id], seed, final_metrics,
    )

    return {
        "experiment": "local_only",
        "hospital_id": hospital_id,
        "hospital_name": HOSPITAL_NAMES[hospital_id],
        "seed": seed,
        "final_metrics": final_metrics,
        "best_val_mae": history["best_val_mae"],
        "n_epochs_trained": len(history["train_losses"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-only training")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "default.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["experiment"]["seeds"]
    n_hospitals = cfg["data"]["n_hospitals"]

    out_dir = PROJECT_ROOT / cfg["experiment"]["results_dir"] / "local_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        for h_id in range(1, n_hospitals + 1):
            result = run_single_hospital(cfg, h_id, seed)
            all_results.append(result)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved local-only results to %s", out_path)


if __name__ == "__main__":
    main()
