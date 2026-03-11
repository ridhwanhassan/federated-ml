"""D-PSGD gossip coordinator — runs on fedcost-fl-server.

Orchestrates synchronous gossip rounds:
1. Upload shared init model
2. Each round: wait for all 5 trained models, then all 5 mixed models
3. Collect metrics, log progress, save final results to S3
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/opt/fedcost")
sys.path.insert(0, str(PROJECT_ROOT))

import boto3
import numpy as np
import torch
import yaml

from src.models.mlp import LOSModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REGION = "ap-southeast-1"
N_HOSPITALS = 5
POLL_INTERVAL = 2
POLL_TIMEOUT = 300


def get_bucket(ssm) -> str:
    resp = ssm.get_parameter(Name="/fedcost/s3-data-bucket")
    return resp["Parameter"]["Value"]


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def poll_all_keys(s3, bucket: str, keys: list[str]) -> None:
    start = time.time()
    remaining = set(keys)
    while remaining:
        if time.time() - start > POLL_TIMEOUT:
            raise TimeoutError(f"Timed out waiting for: {remaining}")
        found = set()
        for key in remaining:
            if s3_key_exists(s3, bucket, key):
                found.add(key)
        remaining -= found
        if remaining:
            time.sleep(POLL_INTERVAL)


def upload_state_dict(s3, bucket: str, key: str, state_dict: dict) -> None:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)


def download_json(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def upload_json(s3, bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body)


def run_one_config(
    s3, bucket: str,
    seed: int, local_epochs: int, n_rounds: int,
    n_features: int,
) -> dict:
    """Run gossip coordination for one seed/local_epochs combo."""
    prefix = f"models/gossip/seed_{seed}/E{local_epochs}"
    logger.info("Starting gossip: seed=%d, E=%d, rounds=%d", seed, local_epochs, n_rounds)

    # Upload shared init model (all hospitals start from same weights)
    torch.manual_seed(seed)
    init_model = LOSModel(n_features=n_features)
    n_params = sum(p.numel() for p in init_model.parameters())

    init_key = f"{prefix}/init/global_model.pt"
    upload_state_dict(s3, bucket, init_key, init_model.state_dict())
    logger.info("Init model uploaded to %s", init_key)

    round_metrics: list[dict] = []
    per_hospital_metrics: list[list[dict]] = []

    for rnd in range(n_rounds):
        # Wait for all trained models
        trained_keys = [
            f"{prefix}/round_{rnd}/hospital_{h}_trained.pt"
            for h in range(1, N_HOSPITALS + 1)
        ]
        logger.info("Round %d/%d: waiting for trained models...", rnd + 1, n_rounds)
        poll_all_keys(s3, bucket, trained_keys)

        # Wait for all mixed models
        mixed_keys = [
            f"{prefix}/round_{rnd}/hospital_{h}_mixed.pt"
            for h in range(1, N_HOSPITALS + 1)
        ]
        logger.info("Round %d/%d: waiting for mixed models...", rnd + 1, n_rounds)
        poll_all_keys(s3, bucket, mixed_keys)

        # Collect metrics
        metrics_keys = [
            f"{prefix}/round_{rnd}/hospital_{h}_metrics.json"
            for h in range(1, N_HOSPITALS + 1)
        ]
        poll_all_keys(s3, bucket, metrics_keys)

        hospital_evals = []
        for h in range(1, N_HOSPITALS + 1):
            metrics = download_json(
                s3, bucket, f"{prefix}/round_{rnd}/hospital_{h}_metrics.json"
            )
            hospital_evals.append(metrics)

        # Simple mean across hospitals (no central server in D-PSGD)
        avg_metrics = {}
        for key in ["mae", "rmse", "r2", "within_1day"]:
            avg_metrics[key] = float(np.mean([m[key] for m in hospital_evals]))

        round_metrics.append(avg_metrics)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "Round %d/%d: mae=%.4f, rmse=%.4f, r2=%.4f, within_1day=%.4f",
            rnd + 1, n_rounds,
            avg_metrics["mae"], avg_metrics["rmse"],
            avg_metrics["r2"], avg_metrics["within_1day"],
        )

    # Communication cost: each node sends to 2 neighbors per round
    comm_cost = 2 * 2 * n_params * N_HOSPITALS * n_rounds

    return {
        "experiment": "gossip",
        "seed": seed,
        "local_epochs": local_epochs,
        "n_rounds": n_rounds,
        "final_metrics": round_metrics[-1] if round_metrics else {},
        "communication_cost": comm_cost,
        "convergence_curve": [m["mae"] for m in round_metrics],
        "per_hospital_final": per_hospital_metrics[-1] if per_hospital_metrics else [],
        "round_metrics": round_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D-PSGD gossip coordinator")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--local-epochs", type=int, nargs="+", required=True)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-features", type=int, required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    bucket = get_bucket(ssm)

    results_dir = PROJECT_ROOT / "results" / "gossip"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for local_epochs in args.local_epochs:
        for seed in args.seeds:
            result = run_one_config(
                s3, bucket, seed, local_epochs, args.n_rounds, args.n_features,
            )
            all_results.append(result)

    # Save and upload each result with unique key
    for result in all_results:
        seed = result["seed"]
        le = result["local_epochs"]
        fname = f"results_seed{seed}_E{le}.json"
        out_path = results_dir / fname
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        upload_json(s3, bucket, f"results/gossip/{fname}", result)
        logger.info("Uploaded results/gossip/%s", fname)

    logger.info("Gossip coordinator complete.")


if __name__ == "__main__":
    main()
