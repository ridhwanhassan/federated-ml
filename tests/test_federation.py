"""Unit tests for federation strategies."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import LOSModel
from src.federation.fedavg import weighted_average_state_dicts


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
