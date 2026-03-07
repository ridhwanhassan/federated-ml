"""Unit tests for federation strategies."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import LOSModel
from src.federation.fedavg import run_fedavg, weighted_average_state_dicts


class TestWeightedAverageStateDicts:
    def _fill_all(self, model, value):
        """Fill all parameters AND buffers with a constant value."""
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(value)
            for b in model.buffers():
                if b.is_floating_point():
                    b.fill_(value)

    def test_equal_weights(self):
        """Equal weights should produce arithmetic mean."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        m2 = LOSModel(n_features=5, hidden_dims=(8,))
        self._fill_all(m1, 2.0)
        self._fill_all(m2, 4.0)
        avg = weighted_average_state_dicts(
            [m1.state_dict(), m2.state_dict()],
            weights=[0.5, 0.5],
        )
        for key in avg:
            if avg[key].is_floating_point():
                assert torch.allclose(avg[key], torch.full_like(avg[key], 3.0))

    def test_unequal_weights(self):
        """Weighted average with 0.75/0.25 split."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        m2 = LOSModel(n_features=5, hidden_dims=(8,))
        self._fill_all(m1, 0.0)
        self._fill_all(m2, 4.0)
        avg = weighted_average_state_dicts(
            [m1.state_dict(), m2.state_dict()],
            weights=[0.75, 0.25],
        )
        for key in avg:
            if avg[key].is_floating_point():
                assert torch.allclose(avg[key], torch.full_like(avg[key], 1.0))

    def test_single_model(self):
        """Single model with weight 1.0 returns same state."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        orig = {k: v.clone() for k, v in m1.state_dict().items()}
        avg = weighted_average_state_dicts([m1.state_dict()], weights=[1.0])
        for key in avg:
            assert torch.equal(avg[key], orig[key])


def _make_hospital_loaders(n_hospitals=5, n_samples=80, n_features=10, seed=42):
    """Create synthetic hospital DataLoader pairs for testing."""
    torch.manual_seed(seed)
    loaders = []
    for i in range(n_hospitals):
        n = n_samples + i * 10  # Vary sizes for weighted avg
        X = torch.randn(n, n_features)
        y = torch.randn(n).abs() * 5
        n_train = int(n * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        loaders.append((train_loader, val_loader))
    return loaders


class TestRunFedAvg:
    def test_returns_expected_keys(self):
        """Result should have round_metrics, per_hospital_metrics, etc."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert "round_metrics" in result
        assert "per_hospital_metrics" in result
        assert "final_global_metrics" in result
        assert "communication_cost" in result

    def test_round_metrics_length(self):
        """Should have one entry per round."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=3, local_epochs=1)
        assert len(result["round_metrics"]) == 3

    def test_round_metrics_structure(self):
        """Each round metric should have mae, rmse, r2."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        for m in result["round_metrics"]:
            assert set(m.keys()) == {"mae", "rmse", "r2", "within_1day"}

    def test_per_hospital_metrics_shape(self):
        """per_hospital_metrics[round][hospital] should have metrics."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert len(result["per_hospital_metrics"]) == 2  # n_rounds
        assert len(result["per_hospital_metrics"][0]) == 3  # n_hospitals

    def test_mae_improves_over_rounds(self):
        """Global MAE should generally improve over rounds."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=10, local_epochs=2, lr=1e-2)
        first_mae = result["round_metrics"][0]["mae"]
        last_mae = result["round_metrics"][-1]["mae"]
        assert last_mae < first_mae

    def test_communication_cost_positive(self):
        """Communication cost should be positive."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert result["communication_cost"] > 0


from src.federation.gossip import build_mixing_matrix


class TestBuildMixingMatrix:
    def test_shape(self):
        """Matrix should be N x N."""
        W = build_mixing_matrix(5)
        assert W.shape == (5, 5)

    def test_doubly_stochastic(self):
        """Rows and columns should sum to 1."""
        W = build_mixing_matrix(5)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)

    def test_symmetric(self):
        """MH matrix for ring should be symmetric."""
        W = build_mixing_matrix(5)
        np.testing.assert_allclose(W, W.T, atol=1e-10)

    def test_ring_neighbors(self):
        """Each node should have non-zero weights only for self and 2 neighbors."""
        W = build_mixing_matrix(5)
        for i in range(5):
            left = (i - 1) % 5
            right = (i + 1) % 5
            for j in range(5):
                if j in (i, left, right):
                    assert W[i, j] > 0
                else:
                    assert W[i, j] == 0.0

    def test_equal_weights_for_ring(self):
        """For a ring (all degree 2), weights should be 1/3."""
        W = build_mixing_matrix(5)
        for i in range(5):
            left = (i - 1) % 5
            right = (i + 1) % 5
            assert W[i, i] == pytest.approx(1.0 / 3.0)
            assert W[i, left] == pytest.approx(1.0 / 3.0)
            assert W[i, right] == pytest.approx(1.0 / 3.0)

    def test_different_sizes(self):
        """Should work for different ring sizes."""
        for n in [3, 4, 7]:
            W = build_mixing_matrix(n)
            assert W.shape == (n, n)
            np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)


from src.federation.gossip import run_gossip


class TestRunGossip:
    def test_returns_expected_keys(self):
        """Result should have round_metrics, per_hospital_metrics, etc."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert "round_metrics" in result
        assert "per_hospital_metrics" in result
        assert "final_global_metrics" in result
        assert "communication_cost" in result

    def test_round_metrics_length(self):
        """Should have one entry per round."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=3, local_epochs=1)
        assert len(result["round_metrics"]) == 3

    def test_per_hospital_metrics_shape(self):
        """per_hospital_metrics[round][hospital] should have metrics."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert len(result["per_hospital_metrics"]) == 2
        assert len(result["per_hospital_metrics"][0]) == 3

    def test_mae_improves_over_rounds(self):
        """Average MAE should generally improve over rounds."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=10, local_epochs=2, lr=1e-2)
        first_mae = result["round_metrics"][0]["mae"]
        last_mae = result["round_metrics"][-1]["mae"]
        assert last_mae < first_mae

    def test_models_converge_toward_each_other(self):
        """After many rounds, per-hospital MAEs should be closer together."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=15, local_epochs=2, lr=1e-2)
        first_round_maes = [m["mae"] for m in result["per_hospital_metrics"][0]]
        last_round_maes = [m["mae"] for m in result["per_hospital_metrics"][-1]]
        first_spread = max(first_round_maes) - min(first_round_maes)
        last_spread = max(last_round_maes) - min(last_round_maes)
        assert last_spread <= first_spread + 0.5

    def test_communication_cost_positive(self):
        """Communication cost should be positive."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert result["communication_cost"] > 0

    def test_communication_cost_ratio(self):
        """D-PSGD ring exchanges 2x params vs FedAvg (2 neighbors vs 1 server)."""
        loaders = _make_hospital_loaders(n_hospitals=5, n_samples=40, n_features=5)
        fedavg_result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        gossip_result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        # D-PSGD: 4*params*N*rounds vs FedAvg: 2*N*params*rounds -> ratio = 2
        assert gossip_result["communication_cost"] == 2 * fedavg_result["communication_cost"]
