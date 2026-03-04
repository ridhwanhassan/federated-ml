"""Unit tests for feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.data.features import (
    extract_first_24h_vitals,
    extract_first_24h_labs,
    encode_categorical_features,
    build_feature_matrix,
)


@pytest.fixture
def sample_cohort():
    """Minimal cohort DataFrame."""
    return pd.DataFrame({
        "stay_id": [1000, 2000],
        "subject_id": [1, 2],
        "hadm_id": [100, 200],
        "gender": ["M", "F"],
        "anchor_age": [65, 72],
        "ethnicity": ["WHITE", "BLACK/AFRICAN AMERICAN"],
        "insurance": ["Medicare", "Medicaid"],
        "admission_type": ["EMERGENCY", "ELECTIVE"],
        "first_careunit": ["MICU", "SICU"],
        "last_careunit": ["MICU", "SICU"],
        "intime": pd.to_datetime(["2150-01-01 08:00", "2150-01-05 10:00"]),
        "outtime": pd.to_datetime(["2150-01-04 08:00", "2150-01-08 10:00"]),
        "los": [3.0, 3.0],
        "n_diagnoses": [5, 2],
        "n_procedures": [1, 3],
        "drg_code": ["189", "191"],
    })


@pytest.fixture
def sample_chartevents():
    """Chartevents with heart rate (220045) for two stays."""
    return pd.DataFrame({
        "stay_id": [1000, 1000, 1000, 2000, 2000],
        "itemid": [220045, 220045, 220045, 220045, 220045],
        "charttime": pd.to_datetime([
            "2150-01-01 10:00",  # within 24h of stay 1000
            "2150-01-01 14:00",  # within 24h
            "2150-01-03 08:00",  # OUTSIDE 24h — should be excluded
            "2150-01-05 12:00",  # within 24h of stay 2000
            "2150-01-05 18:00",  # within 24h
        ]),
        "valuenum": [80.0, 90.0, 70.0, 100.0, 110.0],
    })


@pytest.fixture
def sample_labevents():
    """Labevents with glucose (50931) for two stays."""
    return pd.DataFrame({
        "subject_id": [1, 1, 2],
        "hadm_id": [100, 100, 200],
        "itemid": [50931, 50931, 50931],
        "charttime": pd.to_datetime([
            "2150-01-01 09:00",
            "2150-01-01 15:00",
            "2150-01-05 11:00",
        ]),
        "valuenum": [120.0, 130.0, 150.0],
    })


def test_extract_vitals_filters_24h_window(sample_cohort, sample_chartevents):
    """Only events within 24h of ICU intime should be included."""
    vitals = extract_first_24h_vitals(sample_cohort, sample_chartevents)
    row = vitals[vitals["stay_id"] == 1000].iloc[0]
    assert row["hr_mean"] == pytest.approx(85.0)
    assert row["hr_min"] == pytest.approx(80.0)
    assert row["hr_max"] == pytest.approx(90.0)


def test_extract_vitals_returns_nan_for_missing_stay(sample_cohort, sample_chartevents):
    """Stays with no vitals data should have NaN values."""
    chart = sample_chartevents[sample_chartevents["stay_id"] != 2000]
    vitals = extract_first_24h_vitals(sample_cohort, chart)
    row = vitals[vitals["stay_id"] == 2000].iloc[0]
    assert pd.isna(row["hr_mean"])


def test_extract_labs_aggregates_correctly(sample_cohort, sample_labevents):
    """Lab values should be averaged within 24h window."""
    labs = extract_first_24h_labs(sample_cohort, sample_labevents)
    row = labs[labs["stay_id"] == 1000].iloc[0]
    assert row["glucose_mean"] == pytest.approx(125.0)


def test_encode_categorical_produces_numeric(sample_cohort):
    """Categorical encoding should produce all-numeric output."""
    encoded = encode_categorical_features(sample_cohort)
    assert encoded["gender"].dtype in [np.int64, np.int32, np.float64, int]


def test_build_feature_matrix_output_shape(sample_cohort, sample_chartevents, sample_labevents):
    """Feature matrix should have one row per stay."""
    features = build_feature_matrix(
        cohort=sample_cohort,
        chartevents=sample_chartevents,
        labevents=sample_labevents,
    )
    assert len(features) == len(sample_cohort)
    assert "los" in features.columns
    assert "first_careunit" in features.columns
    assert "stay_id" in features.columns
