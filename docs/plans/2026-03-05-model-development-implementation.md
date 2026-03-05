# MLP Model Development Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the single-task MLP for ICU LOS prediction with training utilities and unit tests.

**Architecture:** Single file `src/models/mlp.py` containing `LOSModel(nn.Module)`, `train_one_epoch()`, `evaluate()`, and `train_model()`. The model uses 3 hidden layers (256→128→64) with BatchNorm, ReLU, and Dropout. Federation code (Phase 3) will call `train_one_epoch()` and `evaluate()` directly.

**Tech Stack:** PyTorch, NumPy, scikit-learn (for r2_score)

**Design doc:** `docs/plans/2026-03-05-model-development-design.md`

---

### Task 1: LOSModel — test and implement

**Files:**
- Create: `tests/test_model.py`
- Create: `src/models/mlp.py`
- Create: `src/models/__init__.py`

**Step 1: Write the failing tests for LOSModel**

Create `tests/test_model.py`:

```python
"""Unit tests for LOSModel MLP."""

import pytest
import torch

from src.models.mlp import LOSModel


class TestLOSModel:
    def test_output_shape(self):
        """Forward pass produces (batch_size, 1) output."""
        model = LOSModel(n_features=20)
        x = torch.randn(32, 20)
        out = model(x)
        assert out.shape == (32, 1)

    def test_single_sample(self):
        """Works with batch_size=1."""
        model = LOSModel(n_features=10)
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 1)

    def test_variable_n_features(self):
        """Accepts different input dimensions."""
        for n in [5, 50, 200]:
            model = LOSModel(n_features=n)
            x = torch.randn(8, n)
            out = model(x)
            assert out.shape == (8, 1)

    def test_custom_hidden_dims(self):
        """Accepts custom hidden layer sizes."""
        model = LOSModel(n_features=10, hidden_dims=(64, 32))
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_dropout_rate(self):
        """Dropout layers use the specified rate."""
        model = LOSModel(n_features=10, dropout=0.5)
        dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(dropout_layers) > 0
        assert all(d.p == 0.5 for d in dropout_layers)

    def test_has_batchnorm(self):
        """Model contains BatchNorm1d layers."""
        model = LOSModel(n_features=10)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 3  # One per hidden layer

    def test_eval_mode_deterministic(self):
        """Eval mode produces deterministic output."""
        model = LOSModel(n_features=10)
        model.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

**Step 3: Implement LOSModel**

Create `src/models/__init__.py`:

```python
"""MLP model for ICU length-of-stay prediction."""
```

Create `src/models/mlp.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/models/__init__.py src/models/mlp.py tests/test_model.py
git commit -m "feat: add LOSModel MLP with unit tests"
```

---

### Task 2: train_one_epoch — test and implement

**Files:**
- Modify: `tests/test_model.py`
- Modify: `src/models/mlp.py`

**Step 1: Write the failing tests for train_one_epoch**

Append to `tests/test_model.py`:

```python
from torch.utils.data import DataLoader, TensorDataset
from src.models.mlp import train_one_epoch


@pytest.fixture
def synthetic_loader():
    """Create a small synthetic DataLoader for smoke tests."""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1).abs() * 5  # Positive LOS values
    ds = TensorDataset(X, y.squeeze())
    return DataLoader(ds, batch_size=16, shuffle=True)


class TestTrainOneEpoch:
    def test_returns_float(self, synthetic_loader):
        """Should return average loss as a float."""
        model = LOSModel(n_features=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=5.0)
        loss = train_one_epoch(model, synthetic_loader, optimizer, criterion)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, synthetic_loader):
        """Loss should decrease over multiple epochs."""
        model = LOSModel(n_features=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.HuberLoss(delta=5.0)
        loss_first = train_one_epoch(model, synthetic_loader, optimizer, criterion)
        for _ in range(19):
            train_one_epoch(model, synthetic_loader, optimizer, criterion)
        loss_last = train_one_epoch(model, synthetic_loader, optimizer, criterion)
        assert loss_last < loss_first

    def test_model_params_change(self, synthetic_loader):
        """Model parameters should be updated after training."""
        model = LOSModel(n_features=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=5.0)
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        train_one_epoch(model, synthetic_loader, optimizer, criterion)
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in model.named_parameters()
        )
        assert changed
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::TestTrainOneEpoch -v`
Expected: FAIL with `ImportError: cannot import name 'train_one_epoch'`

**Step 3: Implement train_one_epoch**

Add to `src/models/mlp.py` after the LOSModel class:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/models/mlp.py tests/test_model.py
git commit -m "feat: add train_one_epoch function"
```

---

### Task 3: evaluate — test and implement

**Files:**
- Modify: `tests/test_model.py`
- Modify: `src/models/mlp.py`

**Step 1: Write the failing tests for evaluate**

Append to `tests/test_model.py`:

```python
from src.models.mlp import evaluate


class TestEvaluate:
    def test_returns_metric_dict(self, synthetic_loader):
        """Should return dict with mae, rmse, r2 keys."""
        model = LOSModel(n_features=10)
        metrics = evaluate(model, synthetic_loader)
        assert set(metrics.keys()) == {"mae", "rmse", "r2"}

    def test_metrics_are_floats(self, synthetic_loader):
        """All metric values should be Python floats."""
        model = LOSModel(n_features=10)
        metrics = evaluate(model, synthetic_loader)
        for v in metrics.values():
            assert isinstance(v, float)

    def test_mae_nonnegative(self, synthetic_loader):
        """MAE should always be >= 0."""
        model = LOSModel(n_features=10)
        metrics = evaluate(model, synthetic_loader)
        assert metrics["mae"] >= 0

    def test_rmse_ge_mae(self, synthetic_loader):
        """RMSE >= MAE always holds."""
        model = LOSModel(n_features=10)
        metrics = evaluate(model, synthetic_loader)
        assert metrics["rmse"] >= metrics["mae"] - 1e-6

    def test_metrics_improve_after_training(self, synthetic_loader):
        """MAE should improve after training on the same data."""
        model = LOSModel(n_features=10)
        mae_before = evaluate(model, synthetic_loader)["mae"]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.HuberLoss(delta=5.0)
        for _ in range(30):
            train_one_epoch(model, synthetic_loader, optimizer, criterion)
        mae_after = evaluate(model, synthetic_loader)["mae"]
        assert mae_after < mae_before
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::TestEvaluate -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate'`

**Step 3: Implement evaluate**

Add to `src/models/mlp.py`:

```python
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
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``.
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
    return {"mae": mae, "rmse": rmse, "r2": r2}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 15 tests PASS

**Step 5: Commit**

```bash
git add src/models/mlp.py tests/test_model.py
git commit -m "feat: add evaluate function returning MAE/RMSE/R2"
```

---

### Task 4: train_model — test and implement

**Files:**
- Modify: `tests/test_model.py`
- Modify: `src/models/mlp.py`

**Step 1: Write the failing tests for train_model**

Append to `tests/test_model.py`:

```python
from src.models.mlp import train_model


@pytest.fixture
def synthetic_train_val():
    """Create train and val DataLoaders from synthetic data."""
    torch.manual_seed(42)
    X = torch.randn(200, 10)
    y = torch.randn(200, 1).abs() * 5
    ds = TensorDataset(X, y.squeeze())
    train_ds = TensorDataset(X[:160], y[:160].squeeze())
    val_ds = TensorDataset(X[160:], y[160:].squeeze())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    return train_loader, val_loader


class TestTrainModel:
    def test_returns_result_dict(self, synthetic_train_val):
        """Should return dict with expected keys."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        result = train_model(model, train_loader, val_loader, n_epochs=5)
        assert "train_losses" in result
        assert "val_metrics" in result
        assert "best_val_mae" in result

    def test_train_losses_length(self, synthetic_train_val):
        """train_losses has one entry per epoch."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        result = train_model(model, train_loader, val_loader, n_epochs=5, patience=10)
        assert len(result["train_losses"]) == 5

    def test_val_metrics_structure(self, synthetic_train_val):
        """val_metrics is list of dicts with mae/rmse/r2."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        result = train_model(model, train_loader, val_loader, n_epochs=3, patience=10)
        assert len(result["val_metrics"]) == 3
        for m in result["val_metrics"]:
            assert set(m.keys()) == {"mae", "rmse", "r2"}

    def test_uses_huber_loss(self, synthetic_train_val):
        """Default should use Huber loss (not MSE)."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        # Run with very large huber_delta (acts like MSE) vs small delta
        result_large = train_model(
            model, train_loader, val_loader, n_epochs=3, huber_delta=1000.0, patience=10
        )
        model2 = LOSModel(n_features=10)
        result_small = train_model(
            model2, train_loader, val_loader, n_epochs=3, huber_delta=1.0, patience=10
        )
        # Both should run without error; losses will differ
        assert result_large["train_losses"][-1] != result_small["train_losses"][-1]

    def test_early_stopping(self, synthetic_train_val):
        """Should stop before n_epochs if val MAE doesn't improve."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        result = train_model(
            model, train_loader, val_loader,
            n_epochs=200, patience=3, lr=1e-6,  # Tiny lr = no improvement
        )
        # Should stop well before 200 epochs
        assert len(result["train_losses"]) < 200

    def test_best_val_mae_is_minimum(self, synthetic_train_val):
        """best_val_mae should be the minimum across all epochs."""
        train_loader, val_loader = synthetic_train_val
        model = LOSModel(n_features=10)
        result = train_model(model, train_loader, val_loader, n_epochs=10, patience=15)
        all_maes = [m["mae"] for m in result["val_metrics"]]
        assert result["best_val_mae"] == pytest.approx(min(all_maes))
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::TestTrainModel -v`
Expected: FAIL with `ImportError: cannot import name 'train_model'`

**Step 3: Implement train_model**

Add to `src/models/mlp.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 21 tests PASS

**Step 5: Commit**

```bash
git add src/models/mlp.py tests/test_model.py
git commit -m "feat: add train_model with Adam, scheduler, early stopping"
```

---

### Task 5: Update __init__.py exports and run full test suite

**Files:**
- Modify: `src/models/__init__.py`

**Step 1: Update exports**

Replace `src/models/__init__.py` with:

```python
"""MLP model for ICU length-of-stay prediction."""

from src.models.mlp import LOSModel, evaluate, train_model, train_one_epoch

__all__ = ["LOSModel", "train_one_epoch", "evaluate", "train_model"]
```

**Step 2: Run full test suite**

Run: `pytest tests/test_model.py tests/test_data_loader.py -v`
Expected: All tests PASS (model tests + existing data loader tests)

**Step 3: Commit**

```bash
git add src/models/__init__.py
git commit -m "feat: export model API from src.models"
```
