"""Feature engineering for FedCost ICU length-of-stay prediction.

Extracts first-24-hour vitals and labs from MIMIC-IV chartevents and
labevents, encodes categorical features, and assembles a single feature
matrix ready for model training.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIMIC-IV item ID mappings
# ---------------------------------------------------------------------------

VITAL_ITEMS: dict[str, list[int]] = {
    "hr": [220045],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "mbp": [220052, 220181],
    "rr": [220210, 224690],
    "spo2": [220277],
    "temp": [223761, 223762],
}

LAB_ITEMS: dict[str, list[int]] = {
    "glucose": [50931, 50809],
    "creatinine": [50912],
    "bun": [51006],
    "hemoglobin": [51222],
    "platelet": [51265],
    "wbc": [51301],
    "sodium": [50983],
    "potassium": [50971],
    "bicarbonate": [50882],
    "lactate": [50813],
}

ALL_VITAL_ITEMIDS: Set[int] = {
    itemid for ids in VITAL_ITEMS.values() for itemid in ids
}
ALL_LAB_ITEMIDS: Set[int] = {
    itemid for ids in LAB_ITEMS.values() for itemid in ids
}

ITEMID_TO_VITAL: dict[int, str] = {
    itemid: name for name, ids in VITAL_ITEMS.items() for itemid in ids
}
ITEMID_TO_LAB: dict[int, str] = {
    itemid: name for name, ids in LAB_ITEMS.items() for itemid in ids
}


# ---------------------------------------------------------------------------
# First-24h vitals extraction
# ---------------------------------------------------------------------------


def extract_first_24h_vitals(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
) -> pd.DataFrame:
    """Extract first-24-hour vital sign statistics per ICU stay.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort table with at least ``stay_id`` and ``intime`` columns.
    chartevents : pd.DataFrame
        Chartevents table with ``stay_id``, ``itemid``, ``charttime``,
        ``valuenum`` columns.

    Returns
    -------
    pd.DataFrame
        One row per ``stay_id`` with columns ``{vital}_mean``,
        ``{vital}_min``, ``{vital}_max`` for each vital sign.
    """
    # Filter to target vital itemids
    ce = chartevents.loc[
        chartevents["itemid"].isin(ALL_VITAL_ITEMIDS)
    ].copy()
    ce["vital"] = ce["itemid"].map(ITEMID_TO_VITAL)
    ce["charttime"] = pd.to_datetime(ce["charttime"])

    # Join cohort for intime
    ce = ce.merge(
        cohort[["stay_id", "intime"]],
        on="stay_id",
        how="inner",
    )

    # Filter to first 24h window
    ce["intime"] = pd.to_datetime(ce["intime"])
    ce = ce.loc[
        (ce["charttime"] >= ce["intime"])
        & (ce["charttime"] < ce["intime"] + pd.Timedelta(hours=24))
    ]

    # Aggregate per stay + vital
    agg = (
        ce.groupby(["stay_id", "vital"])["valuenum"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    # Pivot to wide format
    rows = []
    for vital_name in VITAL_ITEMS:
        sub = agg.loc[agg["vital"] == vital_name, ["stay_id", "mean", "min", "max"]].copy()
        sub = sub.rename(columns={
            "mean": f"{vital_name}_mean",
            "min": f"{vital_name}_min",
            "max": f"{vital_name}_max",
        })
        rows.append(sub)

    # Start with all stay_ids to ensure every stay is present
    result = cohort[["stay_id"]].copy()
    for sub in rows:
        result = result.merge(sub, on="stay_id", how="left")

    logger.info("Extracted first-24h vitals for %d stays", len(result))
    return result


# ---------------------------------------------------------------------------
# First-24h labs extraction
# ---------------------------------------------------------------------------


def extract_first_24h_labs(
    cohort: pd.DataFrame,
    labevents: pd.DataFrame,
) -> pd.DataFrame:
    """Extract first-24-hour lab value means per ICU stay.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort table with ``stay_id``, ``subject_id``, ``hadm_id``,
        ``intime`` columns.
    labevents : pd.DataFrame
        Labevents table with ``subject_id``, ``hadm_id``, ``itemid``,
        ``charttime``, ``valuenum`` columns.

    Returns
    -------
    pd.DataFrame
        One row per ``stay_id`` with columns ``{lab}_mean`` for each lab.
    """
    # Filter to target lab itemids
    le = labevents.loc[
        labevents["itemid"].isin(ALL_LAB_ITEMIDS)
    ].copy()
    le["lab"] = le["itemid"].map(ITEMID_TO_LAB)
    le["charttime"] = pd.to_datetime(le["charttime"])

    # Join cohort via subject_id + hadm_id (labs don't have stay_id)
    le = le.merge(
        cohort[["stay_id", "subject_id", "hadm_id", "intime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # Filter to first 24h window
    le["intime"] = pd.to_datetime(le["intime"])
    le = le.loc[
        (le["charttime"] >= le["intime"])
        & (le["charttime"] < le["intime"] + pd.Timedelta(hours=24))
    ]

    # Aggregate mean per stay + lab
    agg = (
        le.groupby(["stay_id", "lab"])["valuenum"]
        .mean()
        .reset_index()
        .rename(columns={"valuenum": "mean"})
    )

    # Pivot to wide format
    result = cohort[["stay_id"]].copy()
    for lab_name in LAB_ITEMS:
        sub = agg.loc[agg["lab"] == lab_name, ["stay_id", "mean"]].copy()
        sub = sub.rename(columns={"mean": f"{lab_name}_mean"})
        result = result.merge(sub, on="stay_id", how="left")

    logger.info("Extracted first-24h labs for %d stays", len(result))
    return result


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_categorical_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as numeric features.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort table with categorical columns: ``gender``, ``race``,
        ``insurance``, ``admission_type``, ``drg_code``.

    Returns
    -------
    pd.DataFrame
        Copy of input with categorical columns replaced by numeric
        (binary or one-hot) encodings.
    """
    df = cohort.copy()

    # Gender: binary M=1, F=0
    df["gender"] = (df["gender"] == "M").astype(int)

    # Ethnicity: top-5 + OTHER, one-hot
    top_eth = df["race"].value_counts().nlargest(5).index.tolist()
    df["race"] = df["race"].where(df["race"].isin(top_eth), "OTHER")
    eth_dummies = pd.get_dummies(df["race"], prefix="eth", dtype=int)
    df = df.drop(columns=["race"])
    df = pd.concat([df, eth_dummies], axis=1)

    # Insurance: one-hot
    ins_dummies = pd.get_dummies(df["insurance"], prefix="ins", dtype=int)
    df = df.drop(columns=["insurance"])
    df = pd.concat([df, ins_dummies], axis=1)

    # Admission type: one-hot
    adm_dummies = pd.get_dummies(df["admission_type"], prefix="adm", dtype=int)
    df = df.drop(columns=["admission_type"])
    df = pd.concat([df, adm_dummies], axis=1)

    # DRG code: top-20 + OTHER, one-hot
    top_drg = df["drg_code"].value_counts().nlargest(20).index.tolist()
    df["drg_code"] = df["drg_code"].where(df["drg_code"].isin(top_drg), "OTHER")
    drg_dummies = pd.get_dummies(df["drg_code"], prefix="drg", dtype=int)
    df = df.drop(columns=["drg_code"])
    df = pd.concat([df, drg_dummies], axis=1)

    logger.info("Encoded categorical features: %d columns", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full feature matrix from cohort, vitals, and labs.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort table produced by ``extract_cohort``.
    chartevents : pd.DataFrame
        Raw chartevents (or pre-filtered) DataFrame.
    labevents : pd.DataFrame
        Raw labevents (or pre-filtered) DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per ICU stay. Retains ``stay_id``,
        ``los``, and ``first_careunit`` for downstream partitioning.
    """
    vitals = extract_first_24h_vitals(cohort, chartevents)
    labs = extract_first_24h_labs(cohort, labevents)
    encoded = encode_categorical_features(cohort)

    # Drop non-feature columns before merge
    drop_cols = ["subject_id", "hadm_id", "last_careunit", "intime", "outtime"]
    encoded = encoded.drop(columns=[c for c in drop_cols if c in encoded.columns])

    # Merge all on stay_id
    result = encoded.merge(vitals, on="stay_id", how="left")
    result = result.merge(labs, on="stay_id", how="left")

    logger.info(
        "Built feature matrix: %d rows, %d columns",
        len(result),
        len(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# Chunked loaders for large MIMIC-IV files
# ---------------------------------------------------------------------------


def load_chartevents_chunked(
    path: Path,
    stay_ids: set[int],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load chartevents CSV in chunks, filtering to relevant rows.

    Parameters
    ----------
    path : Path
        Path to chartevents CSV file.
    stay_ids : set[int]
        Set of stay_ids to keep.
    chunksize : int, optional
        Number of rows per chunk. Default 500,000.

    Returns
    -------
    pd.DataFrame
        Filtered chartevents with columns: stay_id, itemid, charttime,
        valuenum.
    """
    usecols = ["stay_id", "itemid", "charttime", "valuenum"]
    chunks = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        filtered = chunk.loc[
            chunk["itemid"].isin(ALL_VITAL_ITEMIDS)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["valuenum"].notna()
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    result = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d chartevents rows from %s", len(result), path)
    return result


def load_labevents_chunked(
    path: Path,
    subject_hadm_pairs: set[tuple[int, int]],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load labevents CSV in chunks, filtering to relevant rows.

    Parameters
    ----------
    path : Path
        Path to labevents CSV file.
    subject_hadm_pairs : set[tuple[int, int]]
        Set of (subject_id, hadm_id) tuples to keep.
    chunksize : int, optional
        Number of rows per chunk. Default 500,000.

    Returns
    -------
    pd.DataFrame
        Filtered labevents with columns: subject_id, hadm_id, itemid,
        charttime, valuenum.
    """
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    chunks = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        filtered = chunk.loc[
            chunk["itemid"].isin(ALL_LAB_ITEMIDS)
            & chunk["valuenum"].notna()
        ]
        # Filter by subject_id + hadm_id pairs
        if len(filtered) > 0:
            subject_ids = {p[0] for p in subject_hadm_pairs}
            filtered = filtered.loc[filtered["subject_id"].isin(subject_ids)]
            mask = filtered.apply(
                lambda r: (int(r["subject_id"]), int(r["hadm_id"])) in subject_hadm_pairs,
                axis=1,
            )
            filtered = filtered.loc[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    result = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d labevents rows from %s", len(result), path)
    return result


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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for feature engineering."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run feature engineering on FedCost cohort."
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("data/processed/cohort.csv"),
        help="Path to cohort CSV (default: data/processed/cohort.csv)",
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
        default=Path("data/processed/features.csv"),
        help="Output path for feature matrix CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size for reading large CSV files (default: 500000)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load cohort
    logger.info("Loading cohort from %s", args.cohort)
    cohort = pd.read_csv(args.cohort, parse_dates=["intime", "outtime"])

    stay_ids = set(cohort["stay_id"].tolist())
    subject_hadm_pairs = set(
        zip(cohort["subject_id"].tolist(), cohort["hadm_id"].tolist())
    )

    # Load chartevents and labevents in chunks
    chartevents_path = _csv_path(args.raw_dir, "chartevents")
    labevents_path = _csv_path(args.raw_dir, "labevents")

    logger.info("Loading chartevents from %s (chunksize=%d)", chartevents_path, args.chunksize)
    chartevents = load_chartevents_chunked(chartevents_path, stay_ids, args.chunksize)

    logger.info("Loading labevents from %s (chunksize=%d)", labevents_path, args.chunksize)
    labevents = load_labevents_chunked(labevents_path, subject_hadm_pairs, args.chunksize)

    # Build feature matrix
    features = build_feature_matrix(cohort, chartevents, labevents)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    logger.info("Saved feature matrix (%d rows, %d cols) to %s", len(features), len(features.columns), args.output)


if __name__ == "__main__":
    main()
