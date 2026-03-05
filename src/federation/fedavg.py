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
