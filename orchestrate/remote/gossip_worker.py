"""D-PSGD gossip worker — runs on each hospital-N instance.

Each round:
1. Poll S3 for round start signal (or round 0 init model)
2. Train E local epochs
3. Upload trained model to S3
4. Wait for ring neighbors' trained models
5. Apply Metropolis-Hastings mixing (1/3 self + 1/3 left + 1/3 right)
6. Upload mixed model and metrics to S3
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
POLL_INTERVAL = 2
POLL_TIMEOUT = 300

# Ring topology: each hospital's (left_neighbor, right_neighbor)
RING_NEIGHBORS: dict[int, tuple[int, int]] = {
    1: (5, 2),
    2: (1, 3),
    3: (2, 4),
    4: (3, 5),
    5: (4, 1),
}

# MH mixing weight for ring with degree 2: 1/3 each
MH_WEIGHT = 1.0 / 3.0


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


def mix_state_dicts(
    self_sd: dict, left_sd: dict, right_sd: dict,
) -> dict:
    """Apply MH mixing: 1/3 * self + 1/3 * left + 1/3 * right."""
    mixed = {}
    for key in self_sd:
        mixed[key] = (
            MH_WEIGHT * self_sd[key].float()
            + MH_WEIGHT * left_sd[key].float()
            + MH_WEIGHT * right_sd[key].float()
        )
        mixed[key] = mixed[key].to(self_sd[key].dtype)
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="D-PSGD gossip worker for hospital-N")
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
    left_neighbor, right_neighbor = RING_NEIGHBORS[h_id]

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

    prefix = f"models/gossip/seed_{seed}/E{local_epochs}"
    criterion = nn.HuberLoss(delta=args.huber_delta)

    logger.info(
        "Gossip worker H%d: seed=%d, E=%d, rounds=%d, neighbors=(%d,%d)",
        h_id, seed, local_epochs, n_rounds, left_neighbor, right_neighbor,
    )

    # Wait for init model from coordinator
    init_key = f"{prefix}/init/global_model.pt"
    logger.info("H%d: waiting for init model...", h_id)
    poll_s3_key(s3, bucket, init_key)
    init_state = download_state_dict(s3, bucket, init_key)

    model = LOSModel(n_features=n_features)
    model.load_state_dict(init_state)
    model.to("cpu")

    for rnd in range(n_rounds):
        # 1. Local training
        torch.manual_seed(seed + rnd * 1000 + h_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(local_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, "cpu")

        # 2. Upload trained model
        trained_key = f"{prefix}/round_{rnd}/hospital_{h_id}_trained.pt"
        upload_state_dict(s3, bucket, trained_key, model.state_dict())
        logger.info("H%d round %d: trained model uploaded", h_id, rnd)

        # 3. Wait for neighbors' trained models
        left_key = f"{prefix}/round_{rnd}/hospital_{left_neighbor}_trained.pt"
        right_key = f"{prefix}/round_{rnd}/hospital_{right_neighbor}_trained.pt"
        poll_s3_key(s3, bucket, left_key)
        poll_s3_key(s3, bucket, right_key)

        # 4. MH mixing
        left_sd = download_state_dict(s3, bucket, left_key)
        right_sd = download_state_dict(s3, bucket, right_key)
        mixed_state = mix_state_dicts(model.state_dict(), left_sd, right_sd)
        model.load_state_dict(mixed_state)

        # 5. Upload mixed model
        mixed_key = f"{prefix}/round_{rnd}/hospital_{h_id}_mixed.pt"
        upload_state_dict(s3, bucket, mixed_key, model.state_dict())

        # 6. Evaluate and upload metrics
        metrics = evaluate(model, val_loader, "cpu")
        metrics["n_train_samples"] = len(train_loader.dataset)
        metrics_key = f"{prefix}/round_{rnd}/hospital_{h_id}_metrics.json"
        upload_json(s3, bucket, metrics_key, metrics)

        logger.info(
            "H%d round %d: mae=%.4f, rmse=%.4f, r2=%.4f, within_1day=%.4f",
            h_id, rnd, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["within_1day"],
        )

    logger.info("Gossip worker H%d complete", h_id)


if __name__ == "__main__":
    main()
