"""5-hospital care-unit partition for FedCost.

Assigns ICU stays to one of five simulated hospitals based on the
``first_careunit`` column, writes per-hospital CSV files, and optionally
uploads partitions to S3 for the federated training infrastructure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Care-unit -> hospital mapping
# ---------------------------------------------------------------------------

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

DEFAULT_HOSPITAL: int = 1

HOSPITAL_NAMES: dict[int, str] = {
    1: "Medical",
    2: "Neuro",
    3: "Surgical",
    4: "Trauma",
    5: "Cardiac",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_hospital(care_unit: str) -> int:
    """Map a care-unit name to a hospital number (1-5).

    Parameters
    ----------
    care_unit : str
        The ``first_careunit`` value from MIMIC-IV.

    Returns
    -------
    int
        Hospital number (1-5). Defaults to 1 for unrecognised units.
    """
    return HOSPITAL_PARTITION.get(care_unit, DEFAULT_HOSPITAL)


def partition_features(
    features: pd.DataFrame,
    output_dir: Path,
) -> dict[int, pd.DataFrame]:
    """Partition a feature DataFrame into 5 hospital CSV files.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with at least ``first_careunit`` and ``los`` columns.
    output_dir : Path
        Directory to write ``hospital_1.csv`` ... ``hospital_5.csv`` and
        ``partition_stats.json``.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from hospital number (1-5) to its subset DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = features.copy()
    features["hospital"] = features["first_careunit"].map(assign_hospital)

    partitions: dict[int, pd.DataFrame] = {}
    stats: dict[str, dict] = {}

    for h in range(1, 6):
        subset = features.loc[features["hospital"] == h].drop(columns=["hospital"])
        partitions[h] = subset

        out_path = output_dir / f"hospital_{h}.csv"
        subset.to_csv(out_path, index=False)
        logger.info(
            "Hospital %d (%s): %d stays -> %s",
            h,
            HOSPITAL_NAMES[h],
            len(subset),
            out_path,
        )

        care_units = (
            subset["first_careunit"].unique().tolist() if len(subset) > 0 else []
        )
        stats[f"hospital_{h}"] = {
            "name": HOSPITAL_NAMES[h],
            "n_stays": len(subset),
            "los_mean": float(subset["los"].mean()) if len(subset) > 0 else 0.0,
            "los_std": float(subset["los"].std()) if len(subset) > 1 else 0.0,
            "care_units": care_units,
        }

    stats_path = output_dir / "partition_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Partition stats written to %s", stats_path)

    return partitions


def upload_to_s3(
    output_dir: Path,
    bucket: str,
    features_csv: Path | None = None,
) -> None:
    """Upload partition files to S3.

    Parameters
    ----------
    output_dir : Path
        Local directory containing ``hospital_1.csv`` ... ``hospital_5.csv``
        and ``partition_stats.json``.
    bucket : str
        S3 bucket name (e.g. ``fedcost-data-dev``).
    features_csv : Path | None, optional
        If provided, also upload the full features CSV to ``raw/{filename}``.
    """
    output_dir = Path(output_dir)
    s3 = boto3.client("s3")

    # Upload per-hospital CSVs with HYPHEN naming to match EC2 bootstrap
    for h in range(1, 6):
        local_path = output_dir / f"hospital_{h}.csv"
        s3_key = f"partitions/hospital-{h}.csv"
        s3.upload_file(str(local_path), bucket, s3_key)
        logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, s3_key)

    # Upload partition stats
    stats_path = output_dir / "partition_stats.json"
    if stats_path.exists():
        s3.upload_file(str(stats_path), bucket, "partitions/partition_stats.json")
        logger.info("Uploaded partition_stats.json -> s3://%s/partitions/", bucket)

    # Optionally upload the raw features file
    if features_csv is not None:
        features_csv = Path(features_csv)
        s3_key = f"raw/{features_csv.name}"
        s3.upload_file(str(features_csv), bucket, s3_key)
        logger.info("Uploaded %s -> s3://%s/%s", features_csv, bucket, s3_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for partitioning features into 5 hospitals."""
    parser = argparse.ArgumentParser(
        description="Partition FedCost features into 5 hospital CSV files."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.csv"),
        help="Path to features CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/partitions"),
        help="Output directory for partition CSVs (default: data/processed/partitions)",
    )
    parser.add_argument(
        "--upload-s3",
        type=str,
        default=None,
        metavar="BUCKET",
        help="S3 bucket name — if set, upload partitions after local write",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading features from %s", args.features)
    features = pd.read_csv(args.features)

    partition_features(features, args.output_dir)

    if args.upload_s3:
        upload_to_s3(args.output_dir, args.upload_s3, features_csv=args.features)
        logger.info("S3 upload complete.")


if __name__ == "__main__":
    main()
