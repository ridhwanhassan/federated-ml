"""XGBoost centralized baseline for ICU length-of-stay regression.

Provides train and evaluate functions that return the same metric
format as the MLP model for direct comparison in Table I.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 20,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    seed: int = 42,
) -> XGBRegressor:
    """Train an XGBoost regressor for ICU LOS prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector (LOS in days).
    n_estimators : int, optional
        Maximum number of boosting rounds, by default 500.
    max_depth : int, optional
        Maximum tree depth, by default 6.
    learning_rate : float, optional
        Boosting learning rate, by default 0.05.
    early_stopping_rounds : int, optional
        Stop if val loss doesn't improve for this many rounds,
        by default 20. Only used if X_val/y_val provided.
    X_val : np.ndarray or None, optional
        Validation features for early stopping.
    y_val : np.ndarray or None, optional
        Validation targets for early stopping.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    XGBRegressor
        Fitted model.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=-1,
    )

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(
        "Trained XGBoost: %d estimators, max_depth=%d",
        model.n_estimators,
        model.max_depth,
    )
    return model


def evaluate_xgboost(
    model: XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate XGBoost model, returning same metric format as MLP.

    Parameters
    ----------
    model : XGBRegressor
        Fitted XGBoost model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True target vector.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``.
    """
    y_pred = model.predict(X)
    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    r2 = float(r2_score(y, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}
