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
