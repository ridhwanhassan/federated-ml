"""D-PSGD ring gossip topology for ICU LOS prediction.

Implements Decentralized Parallel SGD where each node trains locally
and exchanges models with its 2 ring neighbors using
Metropolis-Hastings mixing weights.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp import LOSModel, evaluate, train_one_epoch

logger = logging.getLogger(__name__)


def build_mixing_matrix(n_nodes: int = 5) -> np.ndarray:
    """Build Metropolis-Hastings mixing matrix for a ring topology.

    For a ring where every node has degree 2, the MH weights simplify
    to 1/3 for self and each neighbor.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes in the ring, by default 5.

    Returns
    -------
    np.ndarray
        Doubly stochastic mixing matrix of shape ``(n_nodes, n_nodes)``.
    """
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        left = (i - 1) % n_nodes
        right = (i + 1) % n_nodes
        W[i, left] = 1.0 / 3.0
        W[i, right] = 1.0 / 3.0
        W[i, i] = 1.0 / 3.0
    return W


def run_gossip(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Run D-PSGD ring gossip simulation.

    Parameters
    ----------
    hospital_loaders : list[tuple[DataLoader, DataLoader]]
        List of ``(train_loader, val_loader)`` per hospital.
    n_features : int
        Number of input features for LOSModel.
    n_rounds : int, optional
        Number of gossip rounds, by default 50.
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
    W = build_mixing_matrix(n_hospitals)
    criterion = nn.HuberLoss(delta=huber_delta)

    # Initialize per-hospital models (all start from same init)
    init_model = LOSModel(n_features=n_features)
    models = []
    for _ in range(n_hospitals):
        m = LOSModel(n_features=n_features)
        m.load_state_dict(copy.deepcopy(init_model.state_dict()))
        m.to(device)
        models.append(m)

    # Count params for communication cost
    n_params = sum(p.numel() for p in init_model.parameters())

    round_metrics: list[dict[str, float]] = []
    per_hospital_metrics: list[list[dict[str, float]]] = []

    for rnd in range(n_rounds):
        # Local training
        for h_idx in range(n_hospitals):
            optimizer = torch.optim.Adam(models[h_idx].parameters(), lr=lr)
            train_loader, _ = hospital_loaders[h_idx]
            for _ in range(local_epochs):
                train_one_epoch(models[h_idx], train_loader, optimizer, criterion, device)

        # Gossip mixing: each node averages with neighbors per mixing matrix
        old_state_dicts = [copy.deepcopy(m.state_dict()) for m in models]
        for h_idx in range(n_hospitals):
            new_state: dict[str, torch.Tensor] = {}
            for key in old_state_dicts[0]:
                new_state[key] = sum(
                    W[h_idx, j] * old_state_dicts[j][key].float()
                    for j in range(n_hospitals)
                    if W[h_idx, j] > 0
                )
                new_state[key] = new_state[key].to(old_state_dicts[h_idx][key].dtype)
            models[h_idx].load_state_dict(new_state)

        # Evaluate each hospital's model on its own val set
        hospital_evals: list[dict[str, float]] = []
        for h_idx in range(n_hospitals):
            _, val_loader = hospital_loaders[h_idx]
            metrics = evaluate(models[h_idx], val_loader, device)
            hospital_evals.append(metrics)

        # Average metrics across hospitals (simple mean — no central server)
        avg_mae = float(np.mean([m["mae"] for m in hospital_evals]))
        avg_rmse = float(np.mean([m["rmse"] for m in hospital_evals]))
        avg_r2 = float(np.mean([m["r2"] for m in hospital_evals]))
        avg_w1 = float(np.mean([m["within_1_day"] for m in hospital_evals]))
        avg_w2 = float(np.mean([m["within_2_day"] for m in hospital_evals]))
        avg_w3 = float(np.mean([m["within_3_day"] for m in hospital_evals]))
        round_metric = {
            "mae": avg_mae,
            "rmse": avg_rmse,
            "r2": avg_r2,
            "within_1_day": avg_w1,
            "within_2_day": avg_w2,
            "within_3_day": avg_w3,
        }

        round_metrics.append(round_metric)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "D-PSGD round %d/%d — mae=%.4f, rmse=%.4f, r2=%.4f, within_1d=%.4f",
            rnd + 1, n_rounds, avg_mae, avg_rmse, avg_r2, avg_w1,
        )

    # Communication cost: each node sends to 2 neighbors per round
    # Total messages: 2 * n_hospitals per round, each message = n_params
    comm_cost = 2 * 2 * n_params * n_hospitals * n_rounds

    return {
        "round_metrics": round_metrics,
        "per_hospital_metrics": per_hospital_metrics,
        "final_global_metrics": round_metrics[-1] if round_metrics else {},
        "communication_cost": comm_cost,
    }
