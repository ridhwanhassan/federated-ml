"""FedAvg coordinator — runs on fedcost-fl-server.

Each round:
1. Upload global model to S3
2. Wait for all 5 hospital models
3. Download and aggregate via weighted_average_state_dicts
4. Collect per-hospital metrics
5. Log and save results
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
import torch
import yaml

from src.federation.fedavg import weighted_average_state_dicts
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
    """Block until all S3 keys exist."""
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


def download_state_dict(s3, bucket: str, key: str) -> dict:
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=True)


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
    """Run FedAvg coordination for one seed/local_epochs combo."""
    prefix = f"models/fedavg/seed_{seed}/E{local_epochs}"
    logger.info("Starting FedAvg: seed=%d, E=%d, rounds=%d", seed, local_epochs, n_rounds)

    # Initialize global model
    torch.manual_seed(seed)
    global_model = LOSModel(n_features=n_features)
    n_params = sum(p.numel() for p in global_model.parameters())

    round_metrics: list[dict] = []
    per_hospital_metrics: list[list[dict]] = []

    for rnd in range(n_rounds):
        # 1. Upload global model
        global_key = f"{prefix}/round_{rnd}/global_model.pt"
        upload_state_dict(s3, bucket, global_key, global_model.state_dict())
        logger.info("Round %d/%d: global model uploaded", rnd + 1, n_rounds)

        # 2. Wait for all hospital models
        hospital_keys = [
            f"{prefix}/round_{rnd}/hospital_{h}_model.pt"
            for h in range(1, N_HOSPITALS + 1)
        ]
        metrics_keys = [
            f"{prefix}/round_{rnd}/hospital_{h}_metrics.json"
            for h in range(1, N_HOSPITALS + 1)
        ]

        logger.info("Round %d/%d: waiting for %d workers...", rnd + 1, n_rounds, N_HOSPITALS)
        poll_all_keys(s3, bucket, hospital_keys + metrics_keys)

        # 3. Download hospital state_dicts and metrics
        state_dicts = []
        hospital_evals = []
        dataset_sizes = []

        for h in range(1, N_HOSPITALS + 1):
            sd = download_state_dict(s3, bucket, f"{prefix}/round_{rnd}/hospital_{h}_model.pt")
            state_dicts.append(sd)

            metrics = download_json(s3, bucket, f"{prefix}/round_{rnd}/hospital_{h}_metrics.json")
            hospital_evals.append(metrics)
            dataset_sizes.append(metrics.get("n_train_samples", 1))

        # 4. Aggregate with dataset-size weights
        total = sum(dataset_sizes)
        weights = [s / total for s in dataset_sizes]
        global_state = weighted_average_state_dicts(state_dicts, weights)
        global_model.load_state_dict(global_state)

        # 5. Compute weighted average metrics
        avg_metrics = {}
        for key in ["mae", "rmse", "r2", "within_1day"]:
            avg_metrics[key] = sum(
                w * m[key] for w, m in zip(weights, hospital_evals)
            )

        round_metrics.append(avg_metrics)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "Round %d/%d: mae=%.4f, rmse=%.4f, r2=%.4f, within_1day=%.4f",
            rnd + 1, n_rounds,
            avg_metrics["mae"], avg_metrics["rmse"],
            avg_metrics["r2"], avg_metrics["within_1day"],
        )

    comm_cost = 2 * N_HOSPITALS * n_params * n_rounds

    return {
        "experiment": "fedavg",
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
    parser = argparse.ArgumentParser(description="FedAvg coordinator on fl-server")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--local-epochs", type=int, nargs="+", required=True)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-features", type=int, required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    bucket = get_bucket(ssm)

    results_dir = PROJECT_ROOT / "results" / "federated"
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
        upload_json(s3, bucket, f"results/federated/{fname}", result)
        logger.info("Uploaded results/federated/%s", fname)

    logger.info("FedAvg coordinator complete.")


if __name__ == "__main__":
    main()
