"""Local-only training — runs on each hospital-N instance.

Trains an independent MLP on a single hospital's data,
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

from experiments.run_local_only import HOSPITAL_NAMES, run_single_hospital

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
    parser = argparse.ArgumentParser(description="Run local-only training on EC2")
    parser.add_argument("--hospital-id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "experiments" / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override paths for remote environment
    cfg["data"]["partitions_dir"] = "data"
    cfg["experiment"]["device"] = "cpu"

    bucket = get_bucket()
    h_id = args.hospital_id

    results_dir = PROJECT_ROOT / "results" / "local_only"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        logger.info("=== Local-only %s seed=%d ===", HOSPITAL_NAMES[h_id], seed)
        result = run_single_hospital(cfg, h_id, seed)
        results.append(result)

    out_path = results_dir / f"hospital_{h_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    upload_to_s3(out_path, f"results/local_only/hospital_{h_id}.json", bucket)
    logger.info(
        "Results uploaded to s3://%s/results/local_only/hospital_%d.json",
        bucket, h_id,
    )


if __name__ == "__main__":
    main()
