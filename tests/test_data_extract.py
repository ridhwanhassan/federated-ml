"""Tests for src.data.extract — cohort extraction from MIMIC-IV tables."""

import numpy as np
import pandas as pd
import pytest

from src.data.extract import extract_cohort


# ---------------------------------------------------------------------------
# Fixtures — small synthetic DataFrames mimicking MIMIC-IV schema
# ---------------------------------------------------------------------------


@pytest.fixture()
def icustays() -> pd.DataFrame:
    """Minimal icustays table with 5 stays across 4 subjects."""
    return pd.DataFrame(
        {
            "stay_id": [100, 200, 300, 400, 500],
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [10, 20, 30, 40, 50],
            "first_careunit": ["MICU", "Neuro SICU", "SICU", "TSICU", "CCU"],
            "last_careunit": ["MICU", "Neuro SICU", "SICU", "TSICU", "CCU"],
            "intime": pd.to_datetime(
                [
                    "2150-01-01 08:00",
                    "2150-01-02 10:00",
                    "2150-01-03 12:00",
                    "2150-01-04 14:00",
                    "2150-01-05 16:00",
                ]
            ),
            "outtime": pd.to_datetime(
                [
                    "2150-01-04 08:00",  # 3 days
                    "2150-01-07 10:00",  # 5 days
                    "2150-01-13 12:00",  # 10 days
                    "2150-01-06 14:00",  # 2 days
                    "2150-01-12 16:00",  # 7 days
                ]
            ),
            "los": [3.0, 5.0, 10.0, 2.0, 7.0],
        }
    )


@pytest.fixture()
def admissions() -> pd.DataFrame:
    """Minimal admissions table — subject 2 dies in hospital."""
    return pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 40, 50],
            "subject_id": [1, 2, 3, 4, 5],
            "ethnicity": ["WHITE", "BLACK", "ASIAN", "HISPANIC", "WHITE"],
            "insurance": ["Medicare", "Medicaid", "Other", "Medicare", "Medicaid"],
            "admission_type": [
                "EMERGENCY",
                "URGENT",
                "ELECTIVE",
                "EMERGENCY",
                "EMERGENCY",
            ],
            "admittime": pd.to_datetime(
                [
                    "2150-01-01",
                    "2150-01-02",
                    "2150-01-03",
                    "2150-01-04",
                    "2150-01-05",
                ]
            ),
            "dischtime": pd.to_datetime(
                [
                    "2150-01-05",
                    "2150-01-08",
                    "2150-01-14",
                    "2150-01-07",
                    "2150-01-13",
                ]
            ),
            "hospital_expire_flag": [0, 1, 0, 0, 0],
        }
    )


@pytest.fixture()
def patients() -> pd.DataFrame:
    """Minimal patients table."""
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "gender": ["F", "M", "F", "M", "F"],
            "anchor_age": [65, 72, 55, 48, 80],
        }
    )


@pytest.fixture()
def diagnoses() -> pd.DataFrame:
    """Diagnoses ICD table — hadm_id 10 has 3 diagnoses."""
    return pd.DataFrame(
        {
            "hadm_id": [10, 10, 10, 20, 30, 40, 50, 50],
            "icd_code": ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08"],
            "seq_num": [1, 2, 3, 1, 1, 1, 1, 2],
        }
    )


@pytest.fixture()
def procedures() -> pd.DataFrame:
    """Procedures ICD table."""
    return pd.DataFrame(
        {
            "hadm_id": [10, 30, 30, 40],
            "icd_code": ["0001", "0002", "0003", "0004"],
            "seq_num": [1, 1, 2, 1],
        }
    )


@pytest.fixture()
def drgcodes() -> pd.DataFrame:
    """DRG codes table — hadm_id 50 only has APR type (no HCFA)."""
    return pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 40, 50],
            "drg_type": ["HCFA", "HCFA", "HCFA", "HCFA", "APR"],
            "drg_code": ["100", "200", "300", "400", "500"],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_cohort_excludes_deaths(
    icustays, admissions, patients, diagnoses, procedures, drgcodes
):
    """Subjects with hospital_expire_flag=1 must be excluded."""
    df = extract_cohort(icustays, admissions, patients, diagnoses, procedures, drgcodes)
    # subject 2 (hadm_id 20) died — should not appear
    assert 20 not in df["hadm_id"].values
    assert 200 not in df["stay_id"].values


def test_extract_cohort_filters_los_range(
    admissions, patients, diagnoses, procedures, drgcodes
):
    """LOS must be strictly > 0 and <= 30; stays outside are dropped."""
    icu = pd.DataFrame(
        {
            "stay_id": [601, 602, 603, 604],
            "subject_id": [1, 1, 3, 4],
            "hadm_id": [10, 10, 30, 40],
            "first_careunit": ["MICU", "MICU", "SICU", "TSICU"],
            "last_careunit": ["MICU", "MICU", "SICU", "TSICU"],
            "intime": pd.to_datetime(["2150-01-01"] * 4),
            "outtime": pd.to_datetime(["2150-01-02"] * 4),
            "los": [0.0, 31.0, 5.0, 15.0],
        }
    )
    df = extract_cohort(icu, admissions, patients, diagnoses, procedures, drgcodes)
    assert 601 not in df["stay_id"].values, "los=0 should be excluded"
    assert 602 not in df["stay_id"].values, "los=31 should be excluded"
    # los=5 and los=15 should be kept (subjects 3 and 4 are alive)
    assert 603 in df["stay_id"].values
    assert 604 in df["stay_id"].values


def test_extract_cohort_primary_key_is_stay_id(
    icustays, admissions, patients, diagnoses, procedures, drgcodes
):
    """stay_id must be unique in the output."""
    df = extract_cohort(icustays, admissions, patients, diagnoses, procedures, drgcodes)
    assert df["stay_id"].is_unique


def test_extract_cohort_has_expected_columns(
    icustays, admissions, patients, diagnoses, procedures, drgcodes
):
    """Output must contain all required columns."""
    df = extract_cohort(icustays, admissions, patients, diagnoses, procedures, drgcodes)
    expected = {
        "stay_id",
        "subject_id",
        "hadm_id",
        "gender",
        "anchor_age",
        "ethnicity",
        "insurance",
        "admission_type",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
        "n_diagnoses",
        "n_procedures",
        "drg_code",
    }
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )
    # hospital_expire_flag must NOT be in output
    assert "hospital_expire_flag" not in df.columns


def test_extract_cohort_n_diagnoses_count(
    icustays, admissions, patients, diagnoses, procedures, drgcodes
):
    """hadm_id 10 has 3 diagnoses rows => n_diagnoses should be 3."""
    df = extract_cohort(icustays, admissions, patients, diagnoses, procedures, drgcodes)
    row = df.loc[df["hadm_id"] == 10]
    assert len(row) == 1
    assert row.iloc[0]["n_diagnoses"] == 3


def test_extract_cohort_drg_filters_hcfa(
    icustays, admissions, patients, diagnoses, procedures, drgcodes
):
    """hadm_id 50 only has APR-type DRG — drg_code should be NaN."""
    df = extract_cohort(icustays, admissions, patients, diagnoses, procedures, drgcodes)
    row = df.loc[df["hadm_id"] == 50]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["drg_code"])
