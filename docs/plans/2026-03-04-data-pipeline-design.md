# Data Pipeline Design: MIMIC-IV ICU LOS Prediction

**Date:** 2026-03-04
**Status:** Approved
**Scope:** Phase 1 — data extraction, feature engineering, partitioning, and data loading. Federation deferred to Phase 3.

## Decision: Raw CSV + Pandas (No PyHealth)

PyHealth's built-in LOS task uses 10-class classification with ICD/NDC code features. FedCost needs continuous LOS regression with numeric features (vitals, labs, demographics). PyHealth also lacks care-unit extraction needed for the 5-hospital partition. A lightweight pandas pipeline gives full control with no unnecessary dependency.

## Pipeline Overview

```
data/raw/hosp/*.csv.gz  ─┐
data/raw/icu/*.csv.gz   ─┤─→ extract.py ─→ data/processed/cohort.csv
                          │
                          └─→ features.py ─→ data/processed/features.csv
                                    │
                                    ├─→ partition.py ─→ data/processed/partitions/hospital_{1-5}.csv
                                    │
                                    └─→ loader.py ─→ PyTorch DataLoader
```

Each step writes intermediate CSVs for debuggability and resumability.

## Step 1: Cohort Extraction (`src/data/extract.py`)

### Input Tables

| File | Key Columns | Purpose |
|------|------------|---------|
| `hosp/patients.csv.gz` | subject_id, gender, anchor_age | Demographics |
| `hosp/admissions.csv.gz` | hadm_id, admittime, dischtime, hospital_expire_flag, insurance, race, admission_type | Admission info, death exclusion |
| `icu/icustays.csv.gz` | subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los | ICU stay info, care unit, LOS target |
| `hosp/diagnoses_icd.csv.gz` | subject_id, hadm_id, icd_code | Count for n_diagnoses |
| `hosp/procedures_icd.csv.gz` | subject_id, hadm_id, icd_code | Count for n_procedures |
| `hosp/drgcodes.csv.gz` | subject_id, hadm_id, drg_code, drg_type | DRG categorical feature |

### Cohort Filters

1. `los > 0` and `los <= 30` (from icustays)
2. `hospital_expire_flag != 1` (from admissions — exclude in-hospital deaths)
3. Primary key: `stay_id` (one row per ICU stay)

### Join Logic

```
icustays
  LEFT JOIN admissions ON (hadm_id)
  LEFT JOIN patients ON (subject_id)
  LEFT JOIN (diagnoses count per hadm_id) ON (hadm_id)
  LEFT JOIN (procedures count per hadm_id) ON (hadm_id)
  LEFT JOIN (DRG code per hadm_id, filter drg_type='HCFA') ON (hadm_id)
```

### Output

`data/processed/cohort.csv` — columns: stay_id, subject_id, hadm_id, gender, anchor_age, race, insurance, admission_type, first_careunit, last_careunit, intime, outtime, los, n_diagnoses, n_procedures, drg_code

## Step 2: Feature Engineering (`src/data/features.py`)

### Input

- `data/processed/cohort.csv`
- `icu/chartevents.csv.gz` (large — read in chunks, filter by itemid + time window)
- `hosp/labevents.csv.gz` (large — read in chunks, filter by itemid + time window)

### First-24h Vitals (from chartevents)

Filter: events within 24h of ICU `intime`, by specific `itemid`s.

| Vital | ItemIDs (MIMIC-IV) | Aggregations |
|-------|-------------------|--------------|
| Heart Rate | 220045 | mean, min, max |
| SBP | 220050, 220179 | mean, min, max |
| DBP | 220051, 220180 | mean, min, max |
| MBP | 220052, 220181 | mean, min, max |
| SpO2 | 220277 | mean, min, max |
| Temperature | 223761, 223762 | mean, min, max |
| Respiratory Rate | 220210, 224690 | mean, min, max |

→ 7 vitals × 3 aggregations = **21 features**

### First-24h Labs (from labevents)

Filter: events within 24h of ICU `intime`, by specific `itemid`s.

| Lab | ItemIDs (MIMIC-IV) | Aggregation |
|-----|-------------------|-------------|
| Glucose | 50931, 50809 | mean |
| Creatinine | 50912 | mean |
| BUN | 51006 | mean |
| WBC | 51301 | mean |
| Hemoglobin | 51222 | mean |
| Platelet | 51265 | mean |
| Sodium | 50983 | mean |
| Potassium | 50971 | mean |
| Bicarbonate | 50882 | mean |
| Lactate | 50813 | mean |

→ 10 labs × 1 aggregation = **10 features**

### Static Features

| Feature | Source | Encoding |
|---------|--------|----------|
| age | anchor_age from patients | numeric |
| gender | patients | binary (M=1, F=0) |
| race | admissions | top-5 one-hot + OTHER |
| insurance | admissions | one-hot |
| admission_type | admissions | one-hot |
| n_diagnoses | count from diagnoses_icd | numeric |
| n_procedures | count from procedures_icd | numeric |
| drg_code | drgcodes | top-N one-hot + OTHER |

### Memory Strategy for Large Files

`chartevents.csv.gz` (~30GB uncompressed) and `labevents.csv.gz` (~6GB):
- Read in chunks (`pd.read_csv(chunksize=500_000)`)
- Filter each chunk by `stay_id` (from cohort) and `itemid` (target vitals/labs)
- Filter by time window (event time within 24h of `intime`)
- Aggregate filtered results

### Output

`data/processed/features.csv` — one row per `stay_id`, all numeric features + target `los` + `first_careunit` (retained for partitioning)

## Step 3: 5-Hospital Partition (`src/data/partition.py`)

### Partition Mapping

```python
HOSPITAL_PARTITION = {
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
# Default (unmatched) → Hospital 1
```

**Note:** Care unit names match the MIMIC-IV v2.2 `first_careunit` column in `icustays`.

### Output

- `data/processed/partitions/hospital_1.csv` through `hospital_5.csv`
- `data/processed/partitions/partition_stats.json`:
  ```json
  {
    "hospital_1": {"n_stays": ..., "los_mean": ..., "los_std": ..., "care_units": [...]},
    ...
  }
  ```

## Step 4: Data Loading (`src/data/loader.py`)

### Responsibilities

1. Load any hospital CSV or the full features CSV
2. 80/20 train/val split (random, seed-controlled)
3. Fit `StandardScaler` on training features (global scaler — per CLAUDE.md limitations note)
4. Impute missing values with training set median
5. Return `torch.utils.data.DataLoader` with `(X_tensor, y_tensor)` batches

### Interface

```python
def create_dataloaders(
    csv_path: Path,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Returns (train_loader, val_loader, scaler)."""
```

## File Structure

```
src/data/
├── __init__.py
├── extract.py       # Raw CSVs → cohort.csv
├── features.py      # cohort.csv + chartevents/labevents → features.csv
├── partition.py      # features.csv → 5 hospital CSVs + stats
└── loader.py         # Any CSV → PyTorch DataLoader

data/processed/
├── cohort.csv
├── features.csv
└── partitions/
    ├── hospital_1.csv
    ├── hospital_2.csv
    ├── hospital_3.csv
    ├── hospital_4.csv
    ├── hospital_5.csv
    └── partition_stats.json
```

## Configuration

MIMIC-IV raw CSV path is configured via the `MIMIC_RAW_DIR` environment variable (set in `.env`):

```
MIMIC_RAW_DIR=F:/ValianceHealth/MIMIC-IV
```

All CLI scripts read this env var with a fallback to `data/raw`. Since decompressed `.csv` files are available alongside `.csv.gz`, scripts will prefer `.csv` files (faster, no decompression overhead).

## Dependencies

Add to `pyproject.toml`:
```
pandas, numpy, scikit-learn, torch
```

## Resolved Items

- ~~Verify exact `first_careunit` values in MIMIC-IV v2.2~~ — verified, full names used in partition mapping
- ~~Confirm DRG code coverage (HCFA vs APR-DRG)~~ — using `drg_type='HCFA'` by default
- ~~chartevents itemid verification~~ — IDs confirmed against MIMIC-IV MetaVision `d_items`
