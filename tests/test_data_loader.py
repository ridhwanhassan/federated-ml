"""Unit tests for PyTorch data loader."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.loader import LOSDataset, create_dataloaders


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small features CSV and return its path."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "stay_id": range(n),
            "first_careunit": ["MICU"] * n,
            "los": np.random.exponential(3, n).clip(0.1, 30),
            "anchor_age": np.random.randint(18, 90, n).astype(float),
            "gender": np.random.randint(0, 2, n),
            "n_diagnoses": np.random.randint(0, 20, n),
            "hr_mean": np.random.normal(80, 15, n),
            "hr_min": np.random.normal(65, 10, n),
            "hr_max": np.random.normal(100, 15, n),
            "glucose_mean": np.random.normal(120, 30, n),
        }
    )
    # Add some NaNs
    df.loc[0, "hr_mean"] = np.nan
    df.loc[1, "glucose_mean"] = np.nan

    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


def test_create_dataloaders_returns_tuple(sample_csv):
    """Should return (train_loader, val_loader, scaler)."""
    train_dl, val_dl, scaler = create_dataloaders(sample_csv, batch_size=16)
    assert train_dl is not None
    assert val_dl is not None
    assert scaler is not None


def test_train_val_split_ratio(sample_csv):
    """80/20 split by default."""
    train_dl, val_dl, _ = create_dataloaders(
        sample_csv, batch_size=16, val_ratio=0.2
    )
    n_train = len(train_dl.dataset)
    n_val = len(val_dl.dataset)
    assert n_train + n_val == 100
    assert abs(n_val - 20) <= 2  # Allow rounding


def test_batch_shape(sample_csv):
    """Each batch should be (X, y) with correct dimensions."""
    train_dl, _, _ = create_dataloaders(sample_csv, batch_size=16)
    X, y = next(iter(train_dl))
    assert X.dim() == 2
    assert y.dim() == 1
    assert X.shape[0] <= 16
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


def test_no_nans_after_imputation(sample_csv):
    """NaN values should be imputed."""
    train_dl, _, _ = create_dataloaders(sample_csv, batch_size=100)
    X, y = next(iter(train_dl))
    assert not torch.isnan(X).any()
    assert not torch.isnan(y).any()


def test_reproducible_with_seed(sample_csv):
    """Same seed should produce same split."""
    train1, _, _ = create_dataloaders(sample_csv, batch_size=100, seed=42)
    train2, _, _ = create_dataloaders(sample_csv, batch_size=100, seed=42)
    X1, _ = next(iter(train1))
    X2, _ = next(iter(train2))
    assert torch.equal(X1, X2)
