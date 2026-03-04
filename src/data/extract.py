"""Cohort extraction from MIMIC-IV tables for FedCost.

Loads raw MIMIC-IV CSV files, applies inclusion/exclusion criteria,
joins demographic and clinical features, and outputs a single cohort
DataFrame ready for partitioning and feature engineering.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COHORT_COLUMNS: list[str] = [
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
]


def extract_cohort(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    drgcodes: pd.DataFrame,
    *,
    los_min: float = 0.0,
    los_max: float = 30.0,
) -> pd.DataFrame:
    """Build the FedCost cohort from raw MIMIC-IV tables.

    Parameters
    ----------
    icustays : pd.DataFrame
        ICU stays table (stay_id, subject_id, hadm_id, los, ...).
    admissions : pd.DataFrame
        Admissions table (hadm_id, hospital_expire_flag, ...).
    patients : pd.DataFrame
        Patients table (subject_id, gender, anchor_age).
    diagnoses : pd.DataFrame
        Diagnoses ICD table (hadm_id, icd_code, seq_num).
    procedures : pd.DataFrame
        Procedures ICD table (hadm_id, icd_code, seq_num).
    drgcodes : pd.DataFrame
        DRG codes table (hadm_id, drg_type, drg_code).
    los_min : float, optional
        Minimum LOS in days (exclusive). Default 0.0.
    los_max : float, optional
        Maximum LOS in days (inclusive). Default 30.0.

    Returns
    -------
    pd.DataFrame
        Cohort DataFrame with columns defined by ``COHORT_COLUMNS``.
    """
    # --- LOS filter: los > los_min AND los <= los_max ---
    df = icustays.loc[
        (icustays["los"] > los_min) & (icustays["los"] <= los_max)
    ].copy()
    logger.info("After LOS filter (%.1f, %.1f]: %d stays", los_min, los_max, len(df))

    # --- Join admissions, exclude in-hospital deaths ---
    adm_cols = ["hadm_id", "ethnicity", "insurance", "admission_type", "hospital_expire_flag"]
    adm_subset = admissions[
        [c for c in adm_cols if c in admissions.columns]
    ].copy()
    df = df.merge(adm_subset, on="hadm_id", how="left")
    df = df.loc[df["hospital_expire_flag"] != 1].copy()
    logger.info("After excluding deaths: %d stays", len(df))

    # --- Join patients for gender, anchor_age ---
    pat_cols = ["subject_id", "gender", "anchor_age"]
    df = df.merge(patients[pat_cols], on="subject_id", how="left")

    # --- Count diagnoses per hadm_id ---
    n_diag = (
        diagnoses.groupby("hadm_id")
        .size()
        .reset_index(name="n_diagnoses")
    )
    df = df.merge(n_diag, on="hadm_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0).astype(int)

    # --- Count procedures per hadm_id ---
    n_proc = (
        procedures.groupby("hadm_id")
        .size()
        .reset_index(name="n_procedures")
    )
    df = df.merge(n_proc, on="hadm_id", how="left")
    df["n_procedures"] = df["n_procedures"].fillna(0).astype(int)

    # --- DRG code (HCFA type only, first per hadm_id) ---
    hcfa = drgcodes.loc[drgcodes["drg_type"] == "HCFA"].copy()
    hcfa_first = hcfa.drop_duplicates(subset="hadm_id", keep="first")[
        ["hadm_id", "drg_code"]
    ]
    df = df.merge(hcfa_first, on="hadm_id", how="left")

    # --- Drop hospital_expire_flag, select final columns ---
    df = df.drop(columns=["hospital_expire_flag"], errors="ignore")
    df = df[COHORT_COLUMNS].copy()
    df = df.reset_index(drop=True)

    logger.info("Final cohort: %d stays, %d columns", len(df), len(df.columns))
    return df


def _csv_path(directory: Path, name: str) -> Path:
    """Return path to CSV file, preferring uncompressed over .gz.

    Parameters
    ----------
    directory : Path
        Directory containing CSV files.
    name : str
        Base name of the CSV (without extension).

    Returns
    -------
    Path
        Path to the CSV file (.csv if it exists, else .csv.gz).
    """
    plain = directory / f"{name}.csv"
    if plain.exists():
        return plain
    return directory / f"{name}.csv.gz"


def load_mimic_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw MIMIC-IV tables from CSV files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing MIMIC-IV CSV (or .csv.gz) files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: icustays, admissions, patients,
        diagnoses_icd, procedures_icd, drgcodes.
    """
    raw_dir = Path(raw_dir)
    hosp = raw_dir / "hosp"
    icu = raw_dir / "icu"

    logger.info("Loading MIMIC-IV tables from %s", raw_dir)

    icustays = pd.read_csv(
        _csv_path(icu, "icustays"),
        parse_dates=["intime", "outtime"],
    )
    admissions = pd.read_csv(
        _csv_path(hosp, "admissions"),
        parse_dates=["admittime", "dischtime"],
    )
    patients = pd.read_csv(_csv_path(hosp, "patients"))
    diagnoses_icd = pd.read_csv(_csv_path(hosp, "diagnoses_icd"))
    procedures_icd = pd.read_csv(_csv_path(hosp, "procedures_icd"))
    drgcodes = pd.read_csv(_csv_path(hosp, "drgcodes"))

    logger.info(
        "Loaded tables — icustays: %d, admissions: %d, patients: %d",
        len(icustays),
        len(admissions),
        len(patients),
    )

    return {
        "icustays": icustays,
        "admissions": admissions,
        "patients": patients,
        "diagnoses_icd": diagnoses_icd,
        "procedures_icd": procedures_icd,
        "drgcodes": drgcodes,
    }


def main() -> None:
    """CLI entry point for cohort extraction."""
    import os

    parser = argparse.ArgumentParser(
        description="Extract FedCost cohort from MIMIC-IV tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("MIMIC_RAW_DIR", "data/raw")),
        help="Directory containing raw MIMIC-IV CSVs (default: MIMIC_RAW_DIR env or data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cohort.csv"),
        help="Output path for cohort CSV (default: data/processed/cohort.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tables = load_mimic_tables(args.raw_dir)
    cohort = extract_cohort(
        icustays=tables["icustays"],
        admissions=tables["admissions"],
        patients=tables["patients"],
        diagnoses=tables["diagnoses_icd"],
        procedures=tables["procedures_icd"],
        drgcodes=tables["drgcodes"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    logger.info("Saved cohort (%d rows) to %s", len(cohort), args.output)


if __name__ == "__main__":
    main()
