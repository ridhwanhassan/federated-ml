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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Evaluate the model on a DataLoader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        Validation/test data loader.
    device : torch.device or str, optional
        Device to run on, by default ``"cpu"``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``,
        ``"within_1day_acc"``.
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).squeeze(-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    within_1day = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    return {"mae": mae, "rmse": rmse, "r2": r2, "within_1day_acc": within_1day}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 5.0,
    patience: int = 10,
    device: torch.device | str = "cpu",
) -> dict:
    """Full training loop with Adam, ReduceLROnPlateau, and early stopping.

    Used by centralized and local-only baselines. Federation code uses
    train_one_epoch() and evaluate() directly.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    n_epochs : int, optional
        Maximum number of epochs, by default 100.
    lr : float, optional
        Learning rate for Adam, by default 1e-3.
    weight_decay : float, optional
        Weight decay for Adam, by default 1e-4.
    huber_delta : float, optional
        Delta parameter for HuberLoss, by default 5.0.
    patience : int, optional
        Early stopping patience (epochs without val MAE improvement),
        by default 10.
    device : torch.device or str, optional
        Device to run on, by default ``"cpu"``.

    Returns
    -------
    dict
        ``{"train_losses": list[float], "val_metrics": list[dict],
        "best_val_mae": float}``
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2,
    )
    criterion = nn.HuberLoss(delta=huber_delta)

    train_losses: list[float] = []
    val_metrics_history: list[dict[str, float]] = []
    best_val_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)

        train_losses.append(loss)
        val_metrics_history.append(metrics)
        scheduler.step(metrics["mae"])

        if metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_mae=%.4f, val_r2=%.4f",
            epoch + 1, n_epochs, loss, metrics["mae"], metrics["r2"],
        )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_metrics": val_metrics_history,
        "best_val_mae": best_val_mae,
    }
