# Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 4-step MIMIC-IV data pipeline (extract → features → partition → loader) that produces train-ready PyTorch DataLoaders for ICU LOS regression.

**Architecture:** Raw MIMIC-IV CSVs → pandas extraction with cohort filters → chunked feature engineering for large tables (chartevents, labevents) → care-unit-based 5-hospital partition → PyTorch DataLoader with scaling and imputation. Each step writes intermediate CSVs.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn, torch, pytest

**Config:** MIMIC-IV path is set via `MIMIC_RAW_DIR` env var (in `.env`: `MIMIC_RAW_DIR=F:/ValianceHealth/MIMIC-IV`). Scripts prefer `.csv` over `.csv.gz` when both exist (faster I/O).

---

## Task 0: Project Setup — Dependencies and Directory Structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`

**Step 1: Add ML dependencies to pyproject.toml**

Add a `ml` optional dependency group:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "boto3>=1.34",
]
ml = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "torch>=2.2",
]
```

Also update `test` to include ml deps for testing:

```toml
test = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
    "boto3>=1.34",
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "torch>=2.2",
]
```

**Step 2: Create directory structure**

```bash
mkdir -p src/data
touch src/__init__.py
touch src/data/__init__.py
mkdir -p data/processed/partitions
```

**Step 3: Install dependencies**

Run: `uv sync --extra ml`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/__init__.py src/data/__init__.py
git commit -m "chore: add ML dependencies and src/data package structure"
```

---

## Task 1: Cohort Extraction — Tests

**Files:**
- Create: `tests/test_data_extract.py`

**Context:** Tests use small synthetic DataFrames to verify join logic, cohort filtering, and output schema. No real MIMIC data needed.

**Step 1: Write tests for extract_cohort()**

```python
"""Unit tests for MIMIC-IV cohort extraction."""

import pandas as pd
import pytest

from src.data.extract import extract_cohort


@pytest.fixture
def sample_icustays():
    """Minimal icustays table."""
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4],
        "hadm_id": [100, 101, 200, 300, 400],
        "stay_id": [1000, 1001, 2000, 3000, 4000],
        "first_careunit": ["MICU", "MICU/SICU", "SICU", "CCU", "TSICU"],
        "last_careunit": ["MICU", "MICU/SICU", "SICU", "CCU", "TSICU"],
        "intime": pd.to_datetime([
            "2150-01-01 08:00", "2150-02-01 08:00",
            "2150-01-05 10:00", "2150-01-10 12:00", "2150-01-15 14:00",
        ]),
        "outtime": pd.to_datetime([
            "2150-01-04 08:00", "2150-02-03 08:00",
            "2150-01-08 10:00", "2150-01-12 12:00", "2150-01-16 14:00",
        ]),
        "los": [3.0, 2.0, 3.0, 2.0, 1.0],
    })


@pytest.fixture
def sample_admissions():
    """Minimal admissions table."""
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4],
        "hadm_id": [100, 101, 200, 300, 400],
        "admittime": pd.to_datetime([
            "2150-01-01", "2150-02-01", "2150-01-05", "2150-01-10", "2150-01-15",
        ]),
        "dischtime": pd.to_datetime([
            "2150-01-05", "2150-02-04", "2150-01-09", "2150-01-13", "2150-01-17",
        ]),
        "hospital_expire_flag": [0, 0, 1, 0, 0],  # subject 2 dies
        "insurance": ["Medicare", "Medicare", "Medicaid", "Other", "Medicare"],
        "race": ["WHITE", "WHITE", "BLACK", "ASIAN", "HISPANIC"],
        "admission_type": ["EMERGENCY", "URGENT", "EMERGENCY", "ELECTIVE", "EMERGENCY"],
    })


@pytest.fixture
def sample_patients():
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "anchor_age": [65, 72, 55, 80],
    })


@pytest.fixture
def sample_diagnoses():
    return pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3, 4, 4, 4, 4],
        "hadm_id": [100, 100, 100, 200, 200, 300, 400, 400, 400, 400],
        "icd_code": ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08", "I09", "J10"],
    })


@pytest.fixture
def sample_procedures():
    return pd.DataFrame({
        "subject_id": [1, 2, 4],
        "hadm_id": [100, 200, 400],
        "icd_code": ["0001", "0002", "0003"],
    })


@pytest.fixture
def sample_drgcodes():
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4],
        "hadm_id": [100, 101, 200, 300, 400],
        "drg_code": ["189", "190", "191", "192", "193"],
        "drg_type": ["HCFA", "HCFA", "HCFA", "APR", "HCFA"],
    })


def test_extract_cohort_excludes_deaths(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """ICU stays where patient died in hospital should be excluded."""
    cohort = extract_cohort(
        icustays=sample_icustays,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    # subject_id=2 died (hospital_expire_flag=1), should be excluded
    assert 2000 not in cohort["stay_id"].values


def test_extract_cohort_filters_los_range(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """Only stays with 0 < LOS <= 30 should be included."""
    # Modify to add edge cases
    icu = sample_icustays.copy()
    icu.loc[0, "los"] = 0.0   # Excluded: los == 0
    icu.loc[1, "los"] = 31.0  # Excluded: los > 30

    cohort = extract_cohort(
        icustays=icu,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    assert 1000 not in cohort["stay_id"].values  # los=0
    assert 1001 not in cohort["stay_id"].values  # los=31


def test_extract_cohort_primary_key_is_stay_id(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """Each row should have a unique stay_id."""
    cohort = extract_cohort(
        icustays=sample_icustays,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    assert cohort["stay_id"].is_unique


def test_extract_cohort_has_expected_columns(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """Output should contain all required columns."""
    cohort = extract_cohort(
        icustays=sample_icustays,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    required = {
        "stay_id", "subject_id", "hadm_id", "gender", "anchor_age",
        "race", "insurance", "admission_type",
        "first_careunit", "last_careunit", "intime", "outtime", "los",
        "n_diagnoses", "n_procedures", "drg_code",
    }
    assert required.issubset(set(cohort.columns))


def test_extract_cohort_n_diagnoses_count(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """n_diagnoses should be the count of ICD codes per hadm_id."""
    cohort = extract_cohort(
        icustays=sample_icustays,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    # hadm_id=100 has 3 diagnoses, hadm_id=300 has 1
    row_100 = cohort[cohort["hadm_id"] == 100]
    if len(row_100) > 0:
        assert row_100.iloc[0]["n_diagnoses"] == 3


def test_extract_cohort_drg_filters_hcfa(
    sample_icustays, sample_admissions, sample_patients,
    sample_diagnoses, sample_procedures, sample_drgcodes,
):
    """DRG code should only come from HCFA type, not APR."""
    cohort = extract_cohort(
        icustays=sample_icustays,
        admissions=sample_admissions,
        patients=sample_patients,
        diagnoses=sample_diagnoses,
        procedures=sample_procedures,
        drgcodes=sample_drgcodes,
    )
    # hadm_id=300 only has APR type DRG, so drg_code should be NaN
    row_300 = cohort[cohort["hadm_id"] == 300]
    if len(row_300) > 0:
        assert pd.isna(row_300.iloc[0]["drg_code"])
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.extract'`

**Step 3: Commit test file**

```bash
git add tests/test_data_extract.py
git commit -m "test: add cohort extraction unit tests"
```

---

## Task 2: Cohort Extraction — Implementation

**Files:**
- Create: `src/data/extract.py`

**Step 1: Implement extract_cohort() and CLI**

```python
"""MIMIC-IV cohort extraction for ICU LOS prediction.

Reads raw MIMIC-IV CSV files, joins tables, applies cohort filters,
and outputs a single cohort CSV with one row per ICU stay.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to keep in final output
COHORT_COLUMNS = [
    "stay_id", "subject_id", "hadm_id",
    "gender", "anchor_age", "race", "insurance", "admission_type",
    "first_careunit", "last_careunit", "intime", "outtime", "los",
    "n_diagnoses", "n_procedures", "drg_code",
]


def extract_cohort(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    drgcodes: pd.DataFrame,
    los_min: float = 0.0,
    los_max: float = 30.0,
) -> pd.DataFrame:
    """Extract ICU LOS cohort from MIMIC-IV tables.

    Parameters
    ----------
    icustays : pd.DataFrame
        ICU stays table with stay_id, subject_id, hadm_id, los, careunit.
    admissions : pd.DataFrame
        Admissions with hospital_expire_flag, insurance, race.
    patients : pd.DataFrame
        Patient demographics (gender, anchor_age).
    diagnoses : pd.DataFrame
        ICD diagnoses per hadm_id (for counting).
    procedures : pd.DataFrame
        ICD procedures per hadm_id (for counting).
    drgcodes : pd.DataFrame
        DRG codes per hadm_id (filtered to HCFA type).
    los_min : float
        Minimum LOS in days (exclusive). Default 0.
    los_max : float
        Maximum LOS in days (inclusive). Default 30.

    Returns
    -------
    pd.DataFrame
        One row per ICU stay with demographics, counts, and LOS target.
    """
    # Filter LOS range
    cohort = icustays[(icustays["los"] > los_min) & (icustays["los"] <= los_max)].copy()
    logger.info("After LOS filter (%s < los <= %s): %d stays", los_min, los_max, len(cohort))

    # Join admissions — exclude in-hospital deaths
    adm_cols = ["hadm_id", "hospital_expire_flag", "race", "insurance", "admission_type"]
    adm_subset = admissions[[c for c in adm_cols if c in admissions.columns]].copy()
    cohort = cohort.merge(adm_subset, on="hadm_id", how="left")
    cohort = cohort[cohort["hospital_expire_flag"] != 1].copy()
    logger.info("After excluding deaths: %d stays", len(cohort))

    # Join patients
    cohort = cohort.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")

    # Count diagnoses per hadm_id
    n_diag = diagnoses.groupby("hadm_id")["icd_code"].count().reset_index()
    n_diag.columns = ["hadm_id", "n_diagnoses"]
    cohort = cohort.merge(n_diag, on="hadm_id", how="left")
    cohort["n_diagnoses"] = cohort["n_diagnoses"].fillna(0).astype(int)

    # Count procedures per hadm_id
    n_proc = procedures.groupby("hadm_id")["icd_code"].count().reset_index()
    n_proc.columns = ["hadm_id", "n_procedures"]
    cohort = cohort.merge(n_proc, on="hadm_id", how="left")
    cohort["n_procedures"] = cohort["n_procedures"].fillna(0).astype(int)

    # DRG code (HCFA only, take first per hadm_id)
    hcfa = drgcodes[drgcodes["drg_type"] == "HCFA"][["hadm_id", "drg_code"]]
    hcfa = hcfa.drop_duplicates(subset="hadm_id", keep="first")
    cohort = cohort.merge(hcfa, on="hadm_id", how="left")

    # Drop hospital_expire_flag (used for filtering only)
    cohort = cohort.drop(columns=["hospital_expire_flag"], errors="ignore")

    # Select and order columns
    cohort = cohort[[c for c in COHORT_COLUMNS if c in cohort.columns]]

    logger.info("Final cohort: %d stays, %d columns", len(cohort), len(cohort.columns))
    return cohort


def _csv_path(directory: Path, name: str) -> Path:
    """Return path to CSV file, preferring uncompressed over .gz."""
    plain = directory / f"{name}.csv"
    if plain.exists():
        return plain
    return directory / f"{name}.csv.gz"


def load_mimic_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load required MIMIC-IV tables from CSV files.

    Parameters
    ----------
    raw_dir : Path
        Root directory containing hosp/ and icu/ subdirectories.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name to DataFrame.
    """
    tables = {}

    hosp = raw_dir / "hosp"
    icu = raw_dir / "icu"

    logger.info("Loading tables from %s", raw_dir)

    tables["icustays"] = pd.read_csv(_csv_path(icu, "icustays"), parse_dates=["intime", "outtime"])
    tables["admissions"] = pd.read_csv(_csv_path(hosp, "admissions"), parse_dates=["admittime", "dischtime"])
    tables["patients"] = pd.read_csv(_csv_path(hosp, "patients"))
    tables["diagnoses"] = pd.read_csv(_csv_path(hosp, "diagnoses_icd"))
    tables["procedures"] = pd.read_csv(_csv_path(hosp, "procedures_icd"))
    tables["drgcodes"] = pd.read_csv(_csv_path(hosp, "drgcodes"))

    for name, df in tables.items():
        logger.info("  %s: %d rows", name, len(df))

    return tables


def main():
    """CLI entry point: extract cohort from raw MIMIC-IV CSVs."""
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_raw = os.environ.get("MIMIC_RAW_DIR", "data/raw")

    parser = argparse.ArgumentParser(description="Extract ICU LOS cohort from MIMIC-IV")
    parser.add_argument("--raw-dir", type=Path, default=Path(default_raw), help="MIMIC-IV raw CSV directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed/cohort.csv"), help="Output CSV path")
    args = parser.parse_args()

    tables = load_mimic_tables(args.raw_dir)
    cohort = extract_cohort(**tables)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    logger.info("Saved cohort to %s (%d rows)", args.output, len(cohort))


if __name__ == "__main__":
    main()
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_extract.py -v`
Expected: All 6 tests PASS

**Step 3: Commit**

```bash
git add src/data/extract.py
git commit -m "feat: implement MIMIC-IV cohort extraction (extract.py)"
```

---

## Task 3: Feature Engineering — Tests

**Files:**
- Create: `tests/test_data_features.py`

**Step 1: Write tests for feature engineering**

```python
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
        "race": ["WHITE", "BLACK/AFRICAN AMERICAN"],
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
            "2150-01-01 09:00",  # within 24h of stay 1000
            "2150-01-01 15:00",  # within 24h
            "2150-01-05 11:00",  # within 24h of stay 2000
        ]),
        "valuenum": [120.0, 130.0, 150.0],
    })


def test_extract_vitals_filters_24h_window(sample_cohort, sample_chartevents):
    """Only events within 24h of ICU intime should be included."""
    vitals = extract_first_24h_vitals(sample_cohort, sample_chartevents)
    # stay_id=1000: two events within 24h (80, 90), one outside (70)
    row = vitals[vitals["stay_id"] == 1000].iloc[0]
    assert row["hr_mean"] == pytest.approx(85.0)  # mean(80, 90)
    assert row["hr_min"] == pytest.approx(80.0)
    assert row["hr_max"] == pytest.approx(90.0)


def test_extract_vitals_returns_nan_for_missing_stay(sample_cohort, sample_chartevents):
    """Stays with no vitals data should have NaN values."""
    # Remove all events for stay 2000
    chart = sample_chartevents[sample_chartevents["stay_id"] != 2000]
    vitals = extract_first_24h_vitals(sample_cohort, chart)
    row = vitals[vitals["stay_id"] == 2000].iloc[0]
    assert pd.isna(row["hr_mean"])


def test_extract_labs_aggregates_correctly(sample_cohort, sample_labevents):
    """Lab values should be averaged within 24h window."""
    labs = extract_first_24h_labs(sample_cohort, sample_labevents)
    row = labs[labs["stay_id"] == 1000].iloc[0]
    assert row["glucose_mean"] == pytest.approx(125.0)  # mean(120, 130)


def test_encode_categorical_produces_numeric(sample_cohort):
    """Categorical encoding should produce all-numeric output."""
    encoded = encode_categorical_features(sample_cohort)
    # gender should be binary numeric
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Commit**

```bash
git add tests/test_data_features.py
git commit -m "test: add feature engineering unit tests"
```

---

## Task 4: Feature Engineering — Implementation

**Files:**
- Create: `src/data/features.py`

**Context:** This is the most complex module. `chartevents.csv.gz` is ~30GB uncompressed. Must read in chunks and filter by itemid + time window. The function signatures accept DataFrames (for testing) but the CLI reads from disk in chunks.

**Step 1: Implement features.py**

```python
"""Feature engineering for ICU LOS prediction.

Extracts first-24h vitals and labs from MIMIC-IV chartevents and labevents,
encodes categorical features, and produces a single feature matrix CSV.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Vital sign itemids (MIMIC-IV MetaVision) ──────────────────────────

VITAL_ITEMS = {
    "hr": [220045],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "mbp": [220052, 220181],
    "rr": [220210, 224690],
    "spo2": [220277],
    "temp": [223761, 223762],
}

ALL_VITAL_ITEMIDS = {iid for ids in VITAL_ITEMS.values() for iid in ids}

# Reverse lookup: itemid -> vital name
ITEMID_TO_VITAL = {}
for name, ids in VITAL_ITEMS.items():
    for iid in ids:
        ITEMID_TO_VITAL[iid] = name

# ── Lab itemids ───────────────────────────────────────────────────────

LAB_ITEMS = {
    "glucose": [50931, 50809],
    "creatinine": [50912],
    "bun": [51006],
    "wbc": [51301],
    "hemoglobin": [51222],
    "platelet": [51265],
    "sodium": [50983],
    "potassium": [50971],
    "bicarbonate": [50882],
    "lactate": [50813],
}

ALL_LAB_ITEMIDS = {iid for ids in LAB_ITEMS.values() for iid in ids}

ITEMID_TO_LAB = {}
for name, ids in LAB_ITEMS.items():
    for iid in ids:
        ITEMID_TO_LAB[iid] = name


def extract_first_24h_vitals(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
) -> pd.DataFrame:
    """Extract first-24h vital sign aggregates per ICU stay.

    Parameters
    ----------
    cohort : pd.DataFrame
        Must have stay_id and intime columns.
    chartevents : pd.DataFrame
        Must have stay_id, itemid, charttime, valuenum columns.

    Returns
    -------
    pd.DataFrame
        One row per stay_id with columns like hr_mean, hr_min, hr_max, etc.
    """
    # Filter to target itemids
    ce = chartevents[chartevents["itemid"].isin(ALL_VITAL_ITEMIDS)].copy()

    # Map itemid to vital name
    ce["vital"] = ce["itemid"].map(ITEMID_TO_VITAL)

    # Join intime for 24h window filtering
    ce = ce.merge(cohort[["stay_id", "intime"]], on="stay_id", how="inner")

    # Ensure datetime
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["intime"] = pd.to_datetime(ce["intime"])

    # Filter to first 24h
    ce = ce[(ce["charttime"] >= ce["intime"]) & (ce["charttime"] < ce["intime"] + pd.Timedelta(hours=24))]

    # Aggregate per stay_id and vital
    agg = ce.groupby(["stay_id", "vital"])["valuenum"].agg(["mean", "min", "max"]).reset_index()

    # Pivot to wide format
    records = {}
    for _, row in agg.iterrows():
        sid = row["stay_id"]
        vital = row["vital"]
        if sid not in records:
            records[sid] = {"stay_id": sid}
        records[sid][f"{vital}_mean"] = row["mean"]
        records[sid][f"{vital}_min"] = row["min"]
        records[sid][f"{vital}_max"] = row["max"]

    # Ensure all stays are present (even those with no vitals)
    result = pd.DataFrame({"stay_id": cohort["stay_id"]})
    if records:
        vitals_df = pd.DataFrame(records.values())
        result = result.merge(vitals_df, on="stay_id", how="left")

    return result


def extract_first_24h_labs(
    cohort: pd.DataFrame,
    labevents: pd.DataFrame,
) -> pd.DataFrame:
    """Extract first-24h lab value averages per ICU stay.

    Parameters
    ----------
    cohort : pd.DataFrame
        Must have stay_id, subject_id, hadm_id, intime columns.
    labevents : pd.DataFrame
        Must have subject_id, hadm_id, itemid, charttime, valuenum columns.

    Returns
    -------
    pd.DataFrame
        One row per stay_id with columns like glucose_mean, creatinine_mean, etc.
    """
    # Filter to target itemids
    le = labevents[labevents["itemid"].isin(ALL_LAB_ITEMIDS)].copy()

    # Map itemid to lab name
    le["lab"] = le["itemid"].map(ITEMID_TO_LAB)

    # Labs don't have stay_id — join via subject_id + hadm_id to get intime
    stay_info = cohort[["stay_id", "subject_id", "hadm_id", "intime"]].copy()
    le = le.merge(stay_info, on=["subject_id", "hadm_id"], how="inner")

    # Ensure datetime
    le["charttime"] = pd.to_datetime(le["charttime"])
    le["intime"] = pd.to_datetime(le["intime"])

    # Filter to first 24h
    le = le[(le["charttime"] >= le["intime"]) & (le["charttime"] < le["intime"] + pd.Timedelta(hours=24))]

    # Aggregate per stay_id and lab (mean only)
    agg = le.groupby(["stay_id", "lab"])["valuenum"].mean().reset_index()

    # Pivot to wide format
    records = {}
    for _, row in agg.iterrows():
        sid = row["stay_id"]
        lab = row["lab"]
        if sid not in records:
            records[sid] = {"stay_id": sid}
        records[sid][f"{lab}_mean"] = row["valuenum"]

    # Ensure all stays are present
    result = pd.DataFrame({"stay_id": cohort["stay_id"]})
    if records:
        labs_df = pd.DataFrame(records.values())
        result = result.merge(labs_df, on="stay_id", how="left")

    return result


def encode_categorical_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features to numeric.

    Parameters
    ----------
    cohort : pd.DataFrame
        Must have gender, race, insurance, admission_type, drg_code.

    Returns
    -------
    pd.DataFrame
        Same rows with categoricals replaced by numeric encodings.
    """
    df = cohort.copy()

    # Gender: binary
    df["gender"] = (df["gender"] == "M").astype(int)

    # Ethnicity: keep top-5, rest as OTHER, then one-hot
    top_eth = df["race"].value_counts().nlargest(5).index.tolist()
    df["race"] = df["race"].where(df["race"].isin(top_eth), "OTHER")
    eth_dummies = pd.get_dummies(df["race"], prefix="eth", dtype=int)
    df = df.drop(columns=["race"])
    df = pd.concat([df, eth_dummies], axis=1)

    # Insurance: one-hot
    ins_dummies = pd.get_dummies(df["insurance"], prefix="ins", dtype=int)
    df = pd.concat([df.drop(columns=["insurance"]), ins_dummies], axis=1)

    # Admission type: one-hot
    adm_dummies = pd.get_dummies(df["admission_type"], prefix="adm", dtype=int)
    df = pd.concat([df.drop(columns=["admission_type"]), adm_dummies], axis=1)

    # DRG code: top-N one-hot + OTHER
    if "drg_code" in df.columns:
        top_drg = df["drg_code"].value_counts().nlargest(20).index
        df["drg_code"] = df["drg_code"].where(df["drg_code"].isin(top_drg), "OTHER")
        drg_dummies = pd.get_dummies(df["drg_code"], prefix="drg", dtype=int)
        df = pd.concat([df.drop(columns=["drg_code"]), drg_dummies], axis=1)

    return df


def build_feature_matrix(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete feature matrix from cohort and raw event tables.

    Parameters
    ----------
    cohort : pd.DataFrame
        Output of extract_cohort().
    chartevents : pd.DataFrame
        Raw chartevents (or chunk-filtered subset).
    labevents : pd.DataFrame
        Raw labevents (or chunk-filtered subset).

    Returns
    -------
    pd.DataFrame
        One row per stay_id with all features, los target, and first_careunit.
    """
    # Extract vitals and labs
    vitals = extract_first_24h_vitals(cohort, chartevents)
    labs = extract_first_24h_labs(cohort, labevents)

    # Encode categoricals
    encoded = encode_categorical_features(cohort)

    # Drop non-feature columns (keep stay_id, los, first_careunit)
    drop_cols = ["subject_id", "hadm_id", "last_careunit", "intime", "outtime"]
    encoded = encoded.drop(columns=[c for c in drop_cols if c in encoded.columns])

    # Merge vitals and labs
    features = encoded.merge(vitals, on="stay_id", how="left")
    features = features.merge(labs, on="stay_id", how="left")

    logger.info("Feature matrix: %d rows, %d columns", len(features), len(features.columns))
    return features


def load_chartevents_chunked(
    chartevents_path: Path,
    stay_ids: set[int],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load chartevents in chunks, filtering to target stays and vitals.

    Parameters
    ----------
    chartevents_path : Path
        Path to chartevents.csv.gz.
    stay_ids : set[int]
        Set of stay_ids to keep.
    chunksize : int
        Rows per chunk.

    Returns
    -------
    pd.DataFrame
        Filtered chartevents with stay_id, itemid, charttime, valuenum.
    """
    chunks = []
    for chunk in pd.read_csv(
        chartevents_path,
        chunksize=chunksize,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        dtype={"stay_id": "Int64", "itemid": int},
    ):
        filtered = chunk[
            (chunk["itemid"].isin(ALL_VITAL_ITEMIDS))
            & (chunk["stay_id"].isin(stay_ids))
            & (chunk["valuenum"].notna())
        ]
        if len(filtered) > 0:
            chunks.append(filtered)
        logger.debug("Processed chunk: %d rows kept", len(filtered))

    if not chunks:
        return pd.DataFrame(columns=["stay_id", "itemid", "charttime", "valuenum"])
    return pd.concat(chunks, ignore_index=True)


def load_labevents_chunked(
    labevents_path: Path,
    subject_hadm_pairs: set[tuple[int, int]],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load labevents in chunks, filtering to target subjects/admissions and labs.

    Parameters
    ----------
    labevents_path : Path
        Path to labevents.csv.gz.
    subject_hadm_pairs : set[tuple[int, int]]
        Set of (subject_id, hadm_id) pairs to keep.
    chunksize : int
        Rows per chunk.

    Returns
    -------
    pd.DataFrame
        Filtered labevents with subject_id, hadm_id, itemid, charttime, valuenum.
    """
    # Build sets for fast lookup
    subject_ids = {s for s, _ in subject_hadm_pairs}

    chunks = []
    for chunk in pd.read_csv(
        labevents_path,
        chunksize=chunksize,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        dtype={"subject_id": "Int64", "hadm_id": "Int64", "itemid": int},
    ):
        # Pre-filter by subject_id (fast) then itemid
        filtered = chunk[
            (chunk["subject_id"].isin(subject_ids))
            & (chunk["itemid"].isin(ALL_LAB_ITEMIDS))
            & (chunk["valuenum"].notna())
        ]
        # Fine-filter by exact (subject_id, hadm_id) pairs
        if len(filtered) > 0:
            mask = filtered.apply(
                lambda r: (r["subject_id"], r["hadm_id"]) in subject_hadm_pairs, axis=1
            )
            filtered = filtered[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    return pd.concat(chunks, ignore_index=True)


def _csv_path(directory: Path, name: str) -> Path:
    """Return path to CSV file, preferring uncompressed over .gz."""
    plain = directory / f"{name}.csv"
    if plain.exists():
        return plain
    return directory / f"{name}.csv.gz"


def main():
    """CLI entry point: build feature matrix from cohort + raw MIMIC-IV."""
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_raw = os.environ.get("MIMIC_RAW_DIR", "data/raw")

    parser = argparse.ArgumentParser(description="Build feature matrix for ICU LOS prediction")
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/cohort.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path(default_raw))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, parse_dates=["intime", "outtime"])
    logger.info("Loaded cohort: %d stays", len(cohort))

    # Load chartevents (chunked)
    chartevents_path = _csv_path(args.raw_dir / "icu", "chartevents")
    stay_ids = set(cohort["stay_id"])
    logger.info("Loading chartevents (chunked)...")
    chartevents = load_chartevents_chunked(chartevents_path, stay_ids, args.chunksize)
    logger.info("Loaded %d vital sign events", len(chartevents))

    # Load labevents (chunked)
    labevents_path = _csv_path(args.raw_dir / "hosp", "labevents")
    pairs = set(zip(cohort["subject_id"], cohort["hadm_id"]))
    logger.info("Loading labevents (chunked)...")
    labevents = load_labevents_chunked(labevents_path, pairs, args.chunksize)
    logger.info("Loaded %d lab events", len(labevents))

    # Build feature matrix
    features = build_feature_matrix(cohort, chartevents, labevents)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    logger.info("Saved features to %s (%d rows, %d cols)", args.output, *features.shape)


if __name__ == "__main__":
    main()
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_features.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add src/data/features.py
git commit -m "feat: implement feature engineering with chunked vitals/labs extraction"
```

---

## Task 5: Partition — Tests

**Files:**
- Create: `tests/test_data_partition.py`

**Step 1: Write tests**

```python
"""Unit tests for 5-hospital care-unit partition."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.partition import assign_hospital, partition_features, upload_to_s3, HOSPITAL_PARTITION


@pytest.fixture
def sample_features():
    """Feature matrix with known care units."""
    return pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5, 6, 7],
        "first_careunit": [
            "Medical Intensive Care Unit (MICU)",               # H1
            "Neuro Stepdown",                                   # H2
            "Surgical Intensive Care Unit (SICU)",              # H3
            "Trauma SICU (TSICU)",                              # H4
            "Coronary Care Unit (CCU)",                         # H5
            "Medical/Surgical Intensive Care Unit (MICU/SICU)", # H1
            "Unknown Unit",                                     # H1 (default)
        ],
        "los": [3.0, 2.0, 5.0, 1.0, 4.0, 2.0, 6.0],
        "anchor_age": [65, 72, 55, 80, 60, 70, 50],
        "gender": [1, 0, 1, 0, 1, 1, 0],
    })


def test_assign_hospital_known_units():
    assert assign_hospital("Medical Intensive Care Unit (MICU)") == 1
    assert assign_hospital("Neuro Stepdown") == 2
    assert assign_hospital("Surgical Intensive Care Unit (SICU)") == 3
    assert assign_hospital("Trauma SICU (TSICU)") == 4
    assert assign_hospital("Coronary Care Unit (CCU)") == 5


def test_assign_hospital_default_to_h1():
    """Unmatched care units should default to hospital 1."""
    assert assign_hospital("Unknown Unit") == 1
    assert assign_hospital("Some Other ICU") == 1


def test_partition_produces_5_hospitals(sample_features, tmp_path):
    """Partition should produce files for all 5 hospitals."""
    partition_features(sample_features, tmp_path)
    for i in range(1, 6):
        assert (tmp_path / f"hospital_{i}.csv").exists()


def test_partition_preserves_all_rows(sample_features, tmp_path):
    """Total rows across all partitions should equal input rows."""
    partition_features(sample_features, tmp_path)
    total = 0
    for i in range(1, 6):
        df = pd.read_csv(tmp_path / f"hospital_{i}.csv")
        total += len(df)
    assert total == len(sample_features)


def test_partition_stats_file(sample_features, tmp_path):
    """partition_stats.json should be created with correct structure."""
    partition_features(sample_features, tmp_path)
    stats_path = tmp_path / "partition_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert "hospital_1" in stats
    assert "n_stays" in stats["hospital_1"]
    assert "los_mean" in stats["hospital_1"]


def test_upload_to_s3_uses_hyphen_naming(sample_features, tmp_path, mocker):
    """S3 upload should use hospital-N.csv (hyphen) naming for EC2 bootstrap."""
    partition_features(sample_features, tmp_path)

    mock_s3 = mocker.MagicMock()
    mocker.patch("src.data.partition.boto3.client", return_value=mock_s3)

    upload_to_s3(tmp_path, "fedcost-data-dev")

    # Check that upload_file was called with hyphen naming
    upload_calls = mock_s3.upload_file.call_args_list
    s3_keys = [call.args[2] for call in upload_calls]
    for h in range(1, 6):
        assert f"partitions/hospital-{h}.csv" in s3_keys
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data_partition.py -v`
Expected: FAIL

**Step 3: Commit**

```bash
git add tests/test_data_partition.py
git commit -m "test: add partition unit tests"
```

---

## Task 6: Partition — Implementation

**Files:**
- Create: `src/data/partition.py`

**Step 1: Implement partition.py**

```python
"""5-hospital care-unit-based partition for federated learning.

Splits feature matrix by ICU care unit type into 5 simulated hospitals.
Unmatched care units default to Hospital 1 (Medical).
Optionally uploads partitions to S3 for EC2 bootstrap.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Care unit → hospital mapping (MIMIC-IV v2.2 first_careunit values)
HOSPITAL_PARTITION: dict[str, int] = {
    # H1 — Medical
    "Medical Intensive Care Unit (MICU)": 1,
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": 1,
    # H2 — Neuro
    "Neuro Intermediate": 2,
    "Neuro Stepdown": 2,
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": 2,
    # H3 — Surgical
    "Surgical Intensive Care Unit (SICU)": 3,
    # H4 — Trauma
    "Trauma SICU (TSICU)": 4,
    # H5 — Cardiac
    "Coronary Care Unit (CCU)": 5,
    "Cardiac Vascular Intensive Care Unit (CVICU)": 5,
}

HOSPITAL_NAMES = {
    1: "Medical",
    2: "Neuro",
    3: "Surgical",
    4: "Trauma",
    5: "Cardiac",
}

DEFAULT_HOSPITAL = 1


def assign_hospital(care_unit: str) -> int:
    """Map a care unit name to a hospital number.

    Parameters
    ----------
    care_unit : str
        MIMIC-IV first_careunit value.

    Returns
    -------
    int
        Hospital number (1-5). Defaults to 1 for unmatched units.
    """
    return HOSPITAL_PARTITION.get(care_unit, DEFAULT_HOSPITAL)


def partition_features(
    features: pd.DataFrame,
    output_dir: Path,
) -> dict[int, pd.DataFrame]:
    """Partition feature matrix into 5 hospital CSVs.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with first_careunit column.
    output_dir : Path
        Directory to write hospital_1.csv through hospital_5.csv.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of hospital number to its partition DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = features.copy()
    features["hospital"] = features["first_careunit"].map(assign_hospital)

    partitions: dict[int, pd.DataFrame] = {}
    stats: dict[str, dict] = {}

    for h in range(1, 6):
        part = features[features["hospital"] == h].drop(columns=["hospital"])
        partitions[h] = part

        path = output_dir / f"hospital_{h}.csv"
        part.to_csv(path, index=False)

        stats[f"hospital_{h}"] = {
            "name": HOSPITAL_NAMES[h],
            "n_stays": len(part),
            "los_mean": round(float(part["los"].mean()), 2) if len(part) > 0 else 0.0,
            "los_std": round(float(part["los"].std()), 2) if len(part) > 1 else 0.0,
            "care_units": sorted(part["first_careunit"].unique().tolist()) if len(part) > 0 else [],
        }

        logger.info("Hospital %d (%s): %d stays", h, HOSPITAL_NAMES[h], len(part))

    # Write stats
    stats_path = output_dir / "partition_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Partition stats saved to %s", stats_path)

    return partitions


def upload_to_s3(
    output_dir: Path,
    bucket: str,
    features_csv: Path | None = None,
) -> None:
    """Upload partition CSVs to S3 for EC2 bootstrap.

    Uploads hospital-{1-5}.csv to s3://{bucket}/partitions/ (hyphen naming
    to match EC2 bootstrap scripts). Optionally uploads the full features
    CSV to s3://{bucket}/raw/ for the centralized baseline instance.

    Parameters
    ----------
    output_dir : Path
        Local directory containing hospital_1.csv through hospital_5.csv.
    bucket : str
        S3 bucket name (e.g., 'fedcost-data-dev').
    features_csv : Path or None
        If provided, upload to s3://{bucket}/raw/features.csv for centralized instance.
    """
    import boto3

    s3 = boto3.client("s3")

    for h in range(1, 6):
        local_path = output_dir / f"hospital_{h}.csv"
        # EC2 bootstrap expects hyphen: hospital-1.csv, hospital-2.csv, etc.
        s3_key = f"partitions/hospital-{h}.csv"
        s3.upload_file(str(local_path), bucket, s3_key)
        logger.info("Uploaded %s → s3://%s/%s", local_path.name, bucket, s3_key)

    if features_csv is not None:
        s3_key = f"raw/{features_csv.name}"
        s3.upload_file(str(features_csv), bucket, s3_key)
        logger.info("Uploaded %s → s3://%s/%s", features_csv.name, bucket, s3_key)


def main():
    """CLI entry point: partition features.csv into 5 hospitals, optionally upload to S3."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Partition features into 5 hospitals")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/partitions"))
    parser.add_argument("--upload-s3", metavar="BUCKET", help="Upload partitions to S3 bucket (e.g., fedcost-data-dev)")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    logger.info("Loaded features: %d rows", len(features))

    partition_features(features, args.output_dir)

    if args.upload_s3:
        upload_to_s3(args.output_dir, args.upload_s3, features_csv=args.features)
        logger.info("S3 upload complete")


if __name__ == "__main__":
    main()
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_data_partition.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add src/data/partition.py
git commit -m "feat: implement 5-hospital care-unit partition"
```

---

## Task 7: Data Loader — Tests

**Files:**
- Create: `tests/test_data_loader.py`

**Step 1: Write tests**

```python
"""Unit tests for PyTorch data loader."""

import pandas as pd
import numpy as np
import pytest
import torch

from src.data.loader import create_dataloaders, LOSDataset


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small features CSV and return its path."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
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
    })
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
    train_dl, val_dl, _ = create_dataloaders(sample_csv, batch_size=16, val_ratio=0.2)
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
    assert X.shape[0] == 16 or X.shape[0] <= 16  # Last batch may be smaller
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
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_data_loader.py -v`
Expected: FAIL

**Step 3: Commit**

```bash
git add tests/test_data_loader.py
git commit -m "test: add data loader unit tests"
```

---

## Task 8: Data Loader — Implementation

**Files:**
- Create: `src/data/loader.py`

**Step 1: Implement loader.py**

```python
"""PyTorch data loading for ICU LOS prediction.

Reads feature CSVs, imputes missing values, scales features,
and returns train/val DataLoaders.
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

# Columns to exclude from features (metadata and target)
NON_FEATURE_COLUMNS = {"stay_id", "first_careunit", "last_careunit", "los", "hospital"}


class LOSDataset(Dataset):
    """Simple dataset for LOS regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target vector (n_samples,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    csv_path: Path,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Create train and validation DataLoaders from a features CSV.

    Parameters
    ----------
    csv_path : Path
        Path to features CSV (output of features.py or partition hospital CSV).
    batch_size : int
        Batch size for DataLoaders.
    val_ratio : float
        Fraction of data to use for validation.
    seed : int
        Random seed for reproducibility.
    num_workers : int
        Number of data loading workers.

    Returns
    -------
    tuple[DataLoader, DataLoader, StandardScaler]
        (train_loader, val_loader, fitted_scaler)
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Separate target
    y = df["los"].values.astype(np.float32)

    # Select feature columns (drop metadata and target)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].values.astype(np.float32)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed,
    )

    # Impute NaNs with training set median
    train_medians = np.nanmedian(X_train, axis=0)
    # Handle columns that are all NaN
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

    for col_idx in range(X_train.shape[1]):
        nan_mask_train = np.isnan(X_train[:, col_idx])
        nan_mask_val = np.isnan(X_val[:, col_idx])
        X_train[nan_mask_train, col_idx] = train_medians[col_idx]
        X_val[nan_mask_val, col_idx] = train_medians[col_idx]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create datasets and loaders
    train_ds = LOSDataset(X_train, y_train)
    val_ds = LOSDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    logger.info("Train: %d samples, Val: %d samples, Features: %d",
                len(train_ds), len(val_ds), X_train.shape[1])

    return train_loader, val_loader, scaler
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_data_loader.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add src/data/loader.py
git commit -m "feat: implement PyTorch data loader with imputation and scaling"
```

---

## Task 9: Integration — Run Full Pipeline on Real Data

**Files:**
- Modify: `src/data/__init__.py` (add convenience imports)

**Prerequisite:** MIMIC-IV download is complete in `data/raw/`.

**Step 1: Update __init__.py with convenience imports**

```python
"""MIMIC-IV data pipeline for ICU LOS prediction."""

from src.data.extract import extract_cohort, load_mimic_tables
from src.data.features import build_feature_matrix
from src.data.partition import partition_features
from src.data.loader import create_dataloaders

__all__ = [
    "extract_cohort",
    "load_mimic_tables",
    "build_feature_matrix",
    "partition_features",
    "create_dataloaders",
]
```

**Step 2: Run all unit tests**

Run: `uv run pytest tests/test_data_extract.py tests/test_data_features.py tests/test_data_partition.py tests/test_data_loader.py -v`
Expected: All tests PASS

**Step 3: Run pipeline on real MIMIC-IV data (manual verification)**

```bash
# MIMIC_RAW_DIR is set in .env (F:/ValianceHealth/MIMIC-IV)
# Load .env before running (or set manually: export MIMIC_RAW_DIR=F:/ValianceHealth/MIMIC-IV)

# Step 1: Extract cohort
uv run python -m src.data.extract --output data/processed/cohort.csv

# Step 2: Build features (this will take a while — chartevents is huge)
uv run python -m src.data.features --cohort data/processed/cohort.csv --output data/processed/features.csv

# Step 3: Partition into 5 hospitals (local only)
uv run python -m src.data.partition --features data/processed/features.csv --output-dir data/processed/partitions

# Step 3b (optional): Partition + upload to S3 for EC2 bootstrap
# uv run python -m src.data.partition --features data/processed/features.csv --output-dir data/processed/partitions --upload-s3 fedcost-data-dev
```

**Step 4: Verify outputs**

Check: `data/processed/cohort.csv` exists with ~40k-60k rows (expected ICU stays after filtering).
Check: `data/processed/features.csv` exists with same row count, ~40+ columns.
Check: `data/processed/partitions/hospital_{1-5}.csv` exist.
Check: `data/processed/partitions/partition_stats.json` shows reasonable distribution (H1 largest, H5 smallest).

**Step 5: Commit**

```bash
git add src/data/__init__.py
git commit -m "feat: complete data pipeline with convenience imports"
```

---

## Task 10: Deployed Data Verification Tests

**Files:**
- Modify: `tests/test_infra_deployed.py`

**Context:** These tests run post-deploy (`--run-deployed`) to verify that:
1. Partition CSVs exist in S3 with correct naming
2. Each hospital EC2 has downloaded its partition file to `/opt/fedcost/data/`
3. The centralized EC2 has downloaded the raw data

Uses SSM Run Command to check files on EC2 instances (avoids SSH key management).

**Step 1: Add S3 data tests and EC2 file verification to test_infra_deployed.py**

Add the following classes after the existing `TestSecurityGroupsDeployed` class:

```python
# ── S3 Data Tests ───────────────────────────────────────────────────


@pytest.mark.deployed
class TestS3DataDeployed:
    """Verify partition data was uploaded to S3 correctly."""

    def test_hospital_partitions_exist_in_s3(self, s3_client, stack_outputs):
        """Each hospital-{1-5}.csv should exist in s3://bucket/partitions/."""
        bucket = stack_outputs["data_bucket_name"]
        for h in range(1, 6):
            key = f"partitions/hospital-{h}.csv"
            resp = s3_client.head_object(Bucket=bucket, Key=key)
            assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200, (
                f"Missing s3://{bucket}/{key}"
            )

    def test_hospital_partitions_are_nonempty(self, s3_client, stack_outputs):
        """Each partition CSV should have content (> 100 bytes header + data)."""
        bucket = stack_outputs["data_bucket_name"]
        for h in range(1, 6):
            key = f"partitions/hospital-{h}.csv"
            resp = s3_client.head_object(Bucket=bucket, Key=key)
            size = resp["ContentLength"]
            assert size > 100, (
                f"s3://{bucket}/{key} is only {size} bytes — likely empty"
            )

    def test_raw_data_exists_for_centralized(self, s3_client, stack_outputs):
        """At least one file should exist under s3://bucket/raw/."""
        bucket = stack_outputs["data_bucket_name"]
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix="raw/", MaxKeys=1)
        assert resp.get("KeyCount", 0) > 0, (
            f"No files under s3://{bucket}/raw/ — centralized instance has no data"
        )


# ── EC2 Data Download Tests ─────────────────────────────────────────


@pytest.fixture(scope="module")
def ssm_run_client():
    return boto3.client("ssm", region_name=REGION)


def _run_command_on_instance(
    ssm_client, instance_id: str, command: str, timeout: int = 30
) -> str:
    """Run a shell command on an EC2 instance via SSM Run Command.

    Returns stdout. Raises AssertionError if command fails.
    """
    import time

    resp = ssm_client.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=timeout,
    )
    command_id = resp["Command"]["CommandId"]

    # Poll for completion
    for _ in range(timeout):
        time.sleep(1)
        result = ssm_client.get_command_invocation(
            CommandId=command_id, InstanceId=instance_id,
        )
        if result["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
            break

    assert result["Status"] == "Success", (
        f"SSM command failed on {instance_id}: {result.get('StandardErrorContent', '')}"
    )
    return result["StandardOutputContent"].strip()


@pytest.mark.deployed
class TestEC2DataDownloaded:
    """Verify each EC2 instance has its data file after bootstrap."""

    def test_hospital_instances_have_partition_csv(
        self, ssm_run_client, stack_outputs
    ):
        """Each hospital EC2 should have its CSV at /opt/fedcost/data/."""
        for name in HOSPITAL_NAMES:
            instance_id = stack_outputs[f"ec2_{name}_id"]
            # Check file exists and has content
            output = _run_command_on_instance(
                ssm_run_client,
                instance_id,
                f"ls -la /opt/fedcost/data/{name}.csv && wc -l /opt/fedcost/data/{name}.csv",
            )
            # wc -l output: "N /opt/fedcost/data/hospital-N.csv"
            lines = output.strip().split("\n")
            wc_line = lines[-1]
            row_count = int(wc_line.split()[0])
            assert row_count > 1, (
                f"EC2 {name}: partition CSV has only {row_count} lines (expected header + data)"
            )

    def test_centralized_instance_has_raw_data(
        self, ssm_run_client, stack_outputs
    ):
        """Centralized EC2 should have data files at /opt/fedcost/data/."""
        instance_id = stack_outputs["ec2_centralized_id"]
        output = _run_command_on_instance(
            ssm_run_client,
            instance_id,
            "ls /opt/fedcost/data/ | head -10",
        )
        assert len(output) > 0, (
            "EC2 centralized: /opt/fedcost/data/ is empty — no raw data downloaded"
        )

    def test_hospital_partition_has_expected_columns(
        self, ssm_run_client, stack_outputs
    ):
        """Spot-check that hospital-1 CSV has expected header columns."""
        instance_id = stack_outputs["ec2_hospital-1_id"]
        output = _run_command_on_instance(
            ssm_run_client,
            instance_id,
            "head -1 /opt/fedcost/data/hospital-1.csv",
        )
        # Should contain key feature columns
        assert "los" in output, (
            f"EC2 hospital-1: CSV header missing 'los' column. Header: {output}"
        )
        assert "stay_id" in output or "anchor_age" in output, (
            f"EC2 hospital-1: CSV header looks wrong. Header: {output}"
        )
```

**Step 2: Run deployed tests to verify (requires live stack + S3 data uploaded)**

Run: `uv run pytest tests/test_infra_deployed.py -v --run-deployed -k "Data"`
Expected: All S3 and EC2 data tests PASS (assuming `--upload-s3` was run and `pulumi up` completed)

**Step 3: Commit**

```bash
git add tests/test_infra_deployed.py
git commit -m "test: add deployed data verification (S3 partitions + EC2 file downloads)"
```

---

## Summary

| Task | Description | Tests | Files |
|------|------------|-------|-------|
| 0 | Dependencies + dirs | — | pyproject.toml, src/ |
| 1 | Extract tests | 6 tests | tests/test_data_extract.py |
| 2 | Extract impl | — | src/data/extract.py |
| 3 | Features tests | 5 tests | tests/test_data_features.py |
| 4 | Features impl | — | src/data/features.py |
| 5 | Partition tests | 6 tests | tests/test_data_partition.py |
| 6 | Partition impl + S3 upload | — | src/data/partition.py |
| 7 | Loader tests | 5 tests | tests/test_data_loader.py |
| 8 | Loader impl | — | src/data/loader.py |
| 9 | Integration | — | src/data/__init__.py + real data run |
| 10 | Deployed data verification | 6 tests | tests/test_infra_deployed.py |

Total: 28 unit/deployed tests, 4 modules, 10 commits.
