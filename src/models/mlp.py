"""Single-task MLP for ICU length-of-stay regression.

Provides the LOSModel nn.Module and training/evaluation utilities.
The model architecture (configurable hidden dims with BatchNorm + Dropout)
is shared across all federation strategies. Federation code calls
train_one_epoch() and evaluate() directly.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LOSModel(nn.Module):
    """Multi-layer perceptron for ICU length-of-stay regression.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dims : Sequence[int], optional
        Sizes of hidden layers, by default ``(256, 128, 64)``.
    dropout : float, optional
        Dropout probability after each hidden layer, by default 0.3.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Predicted LOS of shape ``(batch_size, 1)``.
        """
        return self.net(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> float:
    """Train for one epoch over the given DataLoader.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        Training data loader yielding ``(X_batch, y_batch)`` tuples.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : nn.Module
        Loss function (e.g. ``nn.HuberLoss``).
    device : torch.device or str, optional
        Device to run on, by default ``"cpu"``.

    Returns
    -------
    float
        Average loss over the epoch.
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch).squeeze(-1)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)
