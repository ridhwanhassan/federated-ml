"""Unit tests for XGBoost centralized baseline."""

import numpy as np
import pytest

from src.models.xgboost_baseline import train_xgboost, evaluate_xgboost


@pytest.fixture
def synthetic_data():
    """Create synthetic train/val arrays."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10).astype(np.float32)
    y_train = rng.exponential(3, 200).astype(np.float32)
    X_val = rng.randn(50, 10).astype(np.float32)
    y_val = rng.exponential(3, 50).astype(np.float32)
    return X_train, y_train, X_val, y_val


class TestTrainXGBoost:
    def test_returns_model(self, synthetic_data):
        """Should return a fitted XGBRegressor."""
        X_train, y_train, _, _ = synthetic_data
        model = train_xgboost(X_train, y_train)
        assert hasattr(model, "predict")

    def test_can_predict(self, synthetic_data):
        """Fitted model should produce predictions."""
        X_train, y_train, X_val, _ = synthetic_data
        model = train_xgboost(X_train, y_train)
        preds = model.predict(X_val)
        assert preds.shape == (50,)

    def test_custom_params(self, synthetic_data):
        """Should accept custom XGBoost parameters."""
        X_train, y_train, _, _ = synthetic_data
        model = train_xgboost(X_train, y_train, n_estimators=10, max_depth=3)
        assert model.n_estimators == 10


class TestEvaluateXGBoost:
    def test_returns_metric_dict(self, synthetic_data):
        """Should return dict with mae, rmse, r2 keys."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_xgboost(model, X_val, y_val)
        assert set(metrics.keys()) == {"mae", "rmse", "r2", "within_1day_acc"}

    def test_metrics_are_floats(self, synthetic_data):
        """All metric values should be Python floats."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_xgboost(model, X_val, y_val)
        for v in metrics.values():
            assert isinstance(v, float)

    def test_mae_nonnegative(self, synthetic_data):
        """MAE should always be >= 0."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_xgboost(model, X_val, y_val)
        assert metrics["mae"] >= 0

    def test_rmse_ge_mae(self, synthetic_data):
        """RMSE >= MAE always holds."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = train_xgboost(X_train, y_train)
        metrics = evaluate_xgboost(model, X_val, y_val)
        assert metrics["rmse"] >= metrics["mae"] - 1e-6
