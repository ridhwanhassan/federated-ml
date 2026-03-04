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
            "Medical ICU",          # H1
            "Neuro Stepdown",       # H2
            "Surgical ICU",         # H3
            "Trauma SICU",          # H4
            "Coronary Care Unit",   # H5
            "MICU/SICU",            # H1
            "Unknown Unit",         # H1 (default)
        ],
        "los": [3.0, 2.0, 5.0, 1.0, 4.0, 2.0, 6.0],
        "anchor_age": [65, 72, 55, 80, 60, 70, 50],
        "gender": [1, 0, 1, 0, 1, 1, 0],
    })


def test_assign_hospital_known_units():
    assert assign_hospital("Medical ICU") == 1
    assert assign_hospital("Neuro Stepdown") == 2
    assert assign_hospital("Surgical ICU") == 3
    assert assign_hospital("Trauma SICU") == 4
    assert assign_hospital("Coronary Care Unit") == 5


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
