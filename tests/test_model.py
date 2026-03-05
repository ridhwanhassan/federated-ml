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
