"""FedAvg worker — runs on each hospital-N instance.

Each round:
1. Poll S3 for global model
2. Download and load into LOSModel
3. Train E local epochs
4. Upload local model and metrics to S3
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
import torch.nn as nn
import yaml

from src.data.loader import create_dataloaders
from src.models.mlp import LOSModel, evaluate, train_one_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REGION = "ap-southeast-1"
POLL_INTERVAL = 2  # seconds
POLL_TIMEOUT = 300  # 5 minutes


def get_bucket(ssm) -> str:
    resp = ssm.get_parameter(Name="/fedcost/s3-data-bucket")
    return resp["Parameter"]["Value"]


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def poll_s3_key(s3, bucket: str, key: str) -> None:
    """Block until the S3 key exists."""
    start = time.time()
    while not s3_key_exists(s3, bucket, key):
        if time.time() - start > POLL_TIMEOUT:
            raise TimeoutError(f"Timed out waiting for s3://{bucket}/{key}")
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


def upload_json(s3, bucket: str, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body)


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg worker for hospital-N")
    parser.add_argument("--hospital-id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, required=True)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=5.0)
    args = parser.parse_args()

    h_id = args.hospital_id
    seed = args.seed
    local_epochs = args.local_epochs
    n_rounds = args.n_rounds

    config_path = PROJECT_ROOT / "experiments" / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load hospital data
    data_path = PROJECT_ROOT / "data" / f"hospital_{h_id}.csv"
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=data_path,
        batch_size=cfg["data"]["batch_size"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
    )
    n_features = train_loader.dataset.X.shape[1]

    s3 = boto3.client("s3", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    bucket = get_bucket(ssm)

    prefix = f"models/fedavg/seed_{seed}/E{local_epochs}"
    criterion = nn.HuberLoss(delta=args.huber_delta)

    logger.info(
        "FedAvg worker H%d: seed=%d, E=%d, n_rounds=%d, n_features=%d",
        h_id, seed, local_epochs, n_rounds, n_features,
    )

    for rnd in range(n_rounds):
        global_key = f"{prefix}/round_{rnd}/global_model.pt"

        # 1. Wait for global model
        logger.info("H%d round %d: waiting for global model...", h_id, rnd)
        poll_s3_key(s3, bucket, global_key)

        # 2. Download and load
        global_state = download_state_dict(s3, bucket, global_key)
        model = LOSModel(n_features=n_features)
        model.load_state_dict(global_state)
        model.to("cpu")

        # 3. Local training
        torch.manual_seed(seed + rnd * 1000 + h_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(local_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, "cpu")

        # 4. Evaluate on local val set
        metrics = evaluate(model, val_loader, "cpu")
        metrics["n_train_samples"] = len(train_loader.dataset)

        # 5. Upload local model
        local_key = f"{prefix}/round_{rnd}/hospital_{h_id}_model.pt"
        upload_state_dict(s3, bucket, local_key, model.state_dict())

        # 6. Upload metrics
        metrics_key = f"{prefix}/round_{rnd}/hospital_{h_id}_metrics.json"
        upload_json(s3, bucket, metrics_key, metrics)

        logger.info(
            "H%d round %d: mae=%.4f, rmse=%.4f, r2=%.4f, within_1day=%.4f",
            h_id, rnd, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["within_1day"],
        )

    logger.info("FedAvg worker H%d complete", h_id)


if __name__ == "__main__":
    main()
