"""Evaluation metrics for ICU length-of-stay prediction.

Provides within-k-days accuracy as a clinically interpretable complement
to MAE, RMSE, and R². This metric measures the fraction of predictions
that fall within k days of the true LOS value.
"""

from __future__ import annotations

import numpy as np


def within_k_days_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 1.0,
) -> float:
    """Compute the fraction of predictions within k days of the true value.

    Parameters
    ----------
    y_true : np.ndarray
        True LOS values (days).
    y_pred : np.ndarray
        Predicted LOS values (days).
    k : float, optional
        Tolerance in days, by default 1.0.

    Returns
    -------
    float
        Fraction of predictions where ``|y_pred - y_true| <= k``,
        in the range [0.0, 1.0].

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([3.0, 5.0, 7.0, 2.0])
    >>> y_pred = np.array([3.5, 6.5, 7.2, 4.0])
    >>> within_k_days_accuracy(y_true, y_pred, k=1.0)
    0.75
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred) <= k))
