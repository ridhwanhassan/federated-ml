"""Unit tests for LOSModel MLP."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import LOSModel, evaluate, train_one_epoch


class TestLOSModel:
    def test_output_shape(self):
        """Forward pass produces (batch_size, 1) output."""
        model = LOSModel(n_features=20)
        x = torch.randn(32, 20)
        out = model(x)
        assert out.shape == (32, 1)

    def test_single_sample(self):
        """Works with batch_size=1 in eval mode (BatchNorm requires >1 in train)."""
        model = LOSModel(n_features=10)
        model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
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
