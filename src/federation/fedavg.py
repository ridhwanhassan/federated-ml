"""Pure-Python FedAvg (star topology) for ICU LOS prediction.

Implements Federated Averaging where a central server broadcasts a
global model, each hospital trains locally, and the server aggregates
via weighted average of state_dicts.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp import LOSModel, evaluate, train_one_epoch

logger = logging.getLogger(__name__)


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Compute weighted average of model state_dicts.

    Parameters
    ----------
    state_dicts : list[dict[str, torch.Tensor]]
        List of state_dicts from each client.
    weights : list[float]
        Weights for each client (should sum to 1.0).

    Returns
    -------
    dict[str, torch.Tensor]
        Averaged state_dict.
    """
    avg: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
        # Preserve original dtype (e.g. for BatchNorm num_batches_tracked)
        avg[key] = avg[key].to(state_dicts[0][key].dtype)
    return avg


def run_fedavg(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Run Federated Averaging (star topology) simulation.

    Parameters
    ----------
    hospital_loaders : list[tuple[DataLoader, DataLoader]]
        List of ``(train_loader, val_loader)`` per hospital.
    n_features : int
        Number of input features for LOSModel.
    n_rounds : int, optional
        Number of federation rounds, by default 50.
    local_epochs : int, optional
        Local training epochs per round, by default 3.
    lr : float, optional
        Learning rate for local Adam optimizers, by default 1e-3.
    huber_delta : float, optional
        Delta for HuberLoss, by default 5.0.
    device : torch.device or str, optional
        Device, by default ``"cpu"``.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    dict
        ``{"round_metrics": list[dict], "per_hospital_metrics": list[list[dict]],
        "final_global_metrics": dict, "communication_cost": int}``
    """
    torch.manual_seed(seed)
    n_hospitals = len(hospital_loaders)

    # Dataset-size weights for FedAvg
    dataset_sizes = [len(tl.dataset) for tl, _ in hospital_loaders]
    total_samples = sum(dataset_sizes)
    weights = [s / total_samples for s in dataset_sizes]

    # Initialize global model
    global_model = LOSModel(n_features=n_features)
    global_model.to(device)
    criterion = nn.HuberLoss(delta=huber_delta)

    # Count params for communication cost
    n_params = sum(p.numel() for p in global_model.parameters())

    round_metrics: list[dict[str, float]] = []
    per_hospital_metrics: list[list[dict[str, float]]] = []

    for rnd in range(n_rounds):
        local_state_dicts: list[dict[str, torch.Tensor]] = []

        # Local training
        for h_idx in range(n_hospitals):
            local_model = LOSModel(n_features=n_features)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model.to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            train_loader, _ = hospital_loaders[h_idx]

            for _ in range(local_epochs):
                train_one_epoch(local_model, train_loader, optimizer, criterion, device)

            local_state_dicts.append(local_model.state_dict())

        # Aggregate
        global_state = weighted_average_state_dicts(local_state_dicts, weights)
        global_model.load_state_dict(global_state)

        # Evaluate global model on each hospital's val set
        hospital_evals: list[dict[str, float]] = []
        for h_idx in range(n_hospitals):
            _, val_loader = hospital_loaders[h_idx]
            metrics = evaluate(global_model, val_loader, device)
            hospital_evals.append(metrics)

        # Average metrics across hospitals (weighted by dataset size)
        avg_mae = sum(w * m["mae"] for w, m in zip(weights, hospital_evals))
        avg_rmse = sum(w * m["rmse"] for w, m in zip(weights, hospital_evals))
        avg_r2 = sum(w * m["r2"] for w, m in zip(weights, hospital_evals))
        round_metric = {"mae": avg_mae, "rmse": avg_rmse, "r2": avg_r2}

        round_metrics.append(round_metric)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "FedAvg round %d/%d — mae=%.4f, rmse=%.4f, r2=%.4f",
            rnd + 1, n_rounds, avg_mae, avg_rmse, avg_r2,
        )

    # Communication cost: 2 * N * params per round (broadcast + upload)
    comm_cost = 2 * n_hospitals * n_params * n_rounds

    return {
        "round_metrics": round_metrics,
        "per_hospital_metrics": per_hospital_metrics,
        "final_global_metrics": round_metrics[-1] if round_metrics else {},
        "communication_cost": comm_cost,
    }
