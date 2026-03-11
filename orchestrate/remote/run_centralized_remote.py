"""Centralized baselines — runs on fedcost-centralized instance.

Trains centralized MLP and XGBoost on full pooled data,
saves results to S3.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/opt/fedcost")
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from experiments.run_centralized import run_centralized_mlp, run_centralized_xgboost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_bucket() -> str:
    result = subprocess.run(
        [
            "aws", "ssm", "get-parameter",
            "--name", "/fedcost/s3-data-bucket",
            "--query", "Parameter.Value",
            "--output", "text",
            "--region", "ap-southeast-1",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def upload_to_s3(local_path: Path, s3_key: str, bucket: str) -> None:
    subprocess.run(
        [
            "aws", "s3", "cp", str(local_path),
            f"s3://{bucket}/{s3_key}",
            "--region", "ap-southeast-1",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centralized baselines on EC2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "experiments" / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override paths for remote environment
    cfg["data"]["features_csv"] = "data/features.csv"
    cfg["experiment"]["device"] = "cpu"

    bucket = get_bucket()
    results_dir = PROJECT_ROOT / "results" / "centralized"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"mlp": [], "xgboost": []}

    for seed in args.seeds:
        logger.info("=== Centralized MLP seed=%d ===", seed)
        mlp_result = run_centralized_mlp(cfg, seed)
        all_results["mlp"].append(mlp_result)

        logger.info("=== Centralized XGBoost seed=%d ===", seed)
        xgb_result = run_centralized_xgboost(cfg, seed)
        all_results["xgboost"].append(xgb_result)

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    upload_to_s3(out_path, "results/centralized/results.json", bucket)
    logger.info("Results uploaded to s3://%s/results/centralized/results.json", bucket)


if __name__ == "__main__":
    main()
