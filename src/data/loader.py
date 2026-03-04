"""PyTorch data loading for ICU LOS prediction.

Provides a Dataset class and factory function for creating train/val
DataLoaders from a features CSV, with NaN imputation and standard scaling.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {"stay_id", "first_careunit", "last_careunit", "los", "hospital"}


class LOSDataset(Dataset):
    """PyTorch Dataset for ICU length-of-stay regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector of shape ``(n_samples,)`` containing LOS in days.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single ``(features, target)`` pair.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(X[idx], y[idx])`` with dtype ``float32``.
        """
        return self.X[idx], self.y[idx]


def create_dataloaders(
    csv_path: Path,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Create train and validation DataLoaders from a features CSV.

    Reads the CSV, drops non-feature columns, performs an 80/20 train/val
    split, imputes NaNs with training-set column medians, applies standard
    scaling, and wraps the result in PyTorch DataLoaders.

    Parameters
    ----------
    csv_path : Path
        Path to the features CSV.  Must contain a ``los`` column (target)
        and any number of numeric feature columns.
    batch_size : int, optional
        Mini-batch size for both loaders, by default 64.
    val_ratio : float, optional
        Fraction of data reserved for validation, by default 0.2.
    seed : int, optional
        Random seed for reproducible splitting and shuffling, by default 42.
    num_workers : int, optional
        Number of DataLoader worker processes, by default 0 (main process).

    Returns
    -------
    tuple[DataLoader, DataLoader, StandardScaler]
        ``(train_loader, val_loader, scaler)`` where *scaler* is the
        fitted ``StandardScaler`` (useful for inverse-transforming
        predictions at evaluation time).
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    y = df["los"].values.astype(np.float32)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].values.astype(np.float32)
    logger.info("Using %d feature columns: %s", len(feature_cols), feature_cols)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed
    )
    logger.info(
        "Split: %d train, %d val (val_ratio=%.2f)",
        len(y_train),
        len(y_val),
        val_ratio,
    )

    # Impute NaNs with training-set column medians
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    for col_idx in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, col_idx]), col_idx] = train_medians[col_idx]
        X_val[np.isnan(X_val[:, col_idx]), col_idx] = train_medians[col_idx]

    # Standard scaling (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = LOSDataset(X_train, y_train)
    val_ds = LOSDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(
        "Created DataLoaders: %d train batches, %d val batches (batch_size=%d)",
        len(train_loader),
        len(val_loader),
        batch_size,
    )

    return train_loader, val_loader, scaler
