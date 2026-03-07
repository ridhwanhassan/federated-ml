"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import within_k_days_accuracy


class TestWithinKDaysAccuracy:
    def test_perfect_predictions(self):
        """All predictions exactly correct should give 1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert within_k_days_accuracy(y_true, y_pred, k=1.0) == 1.0

    def test_all_outside_tolerance(self):
        """All predictions far off should give 0.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert within_k_days_accuracy(y_true, y_pred, k=1.0) == 0.0

    def test_partial_accuracy(self):
        """Mixed predictions should give correct fraction."""
        y_true = np.array([3.0, 5.0, 7.0, 2.0])
        y_pred = np.array([3.5, 6.5, 7.2, 4.0])
        # |0.5| <= 1: yes, |1.5| <= 1: no, |0.2| <= 1: yes, |2.0| <= 1: no
        assert within_k_days_accuracy(y_true, y_pred, k=1.0) == pytest.approx(0.5)

    def test_boundary_exactly_k(self):
        """Prediction exactly k days away should count as within."""
        y_true = np.array([5.0])
        y_pred = np.array([6.0])
        assert within_k_days_accuracy(y_true, y_pred, k=1.0) == 1.0

    def test_different_k_values(self):
        """Larger k should give equal or higher accuracy."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 4.0, 3.1, 7.0, 5.5])
        acc_k1 = within_k_days_accuracy(y_true, y_pred, k=1.0)
        acc_k2 = within_k_days_accuracy(y_true, y_pred, k=2.0)
        acc_k3 = within_k_days_accuracy(y_true, y_pred, k=3.0)
        assert acc_k1 <= acc_k2 <= acc_k3

    def test_returns_float(self):
        """Should return a Python float."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        result = within_k_days_accuracy(y_true, y_pred, k=1.0)
        assert isinstance(result, float)

    def test_range_zero_to_one(self):
        """Result should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 30, size=100)
        y_pred = rng.uniform(0, 30, size=100)
        for k in [0.5, 1.0, 2.0, 5.0]:
            acc = within_k_days_accuracy(y_true, y_pred, k=k)
            assert 0.0 <= acc <= 1.0

    def test_default_k_is_one(self):
        """Default k parameter should be 1.0."""
        y_true = np.array([5.0])
        y_pred = np.array([5.8])
        assert within_k_days_accuracy(y_true, y_pred) == 1.0
        y_pred_far = np.array([6.5])
        assert within_k_days_accuracy(y_true, y_pred_far) == 0.0

    def test_negative_errors_handled(self):
        """Under-predictions should be handled the same as over-predictions."""
        y_true = np.array([5.0, 5.0])
        y_pred_over = np.array([5.5, 5.5])
        y_pred_under = np.array([4.5, 4.5])
        assert within_k_days_accuracy(y_true, y_pred_over, k=1.0) == \
               within_k_days_accuracy(y_true, y_pred_under, k=1.0)
