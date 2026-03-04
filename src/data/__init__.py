"""MIMIC-IV data pipeline for ICU LOS prediction."""

__all__ = [
    "extract_cohort",
    "load_mimic_tables",
    "build_feature_matrix",
    "partition_features",
    "create_dataloaders",
]


def __getattr__(name: str):
    if name in ("extract_cohort", "load_mimic_tables"):
        from src.data.extract import extract_cohort, load_mimic_tables
        return locals()[name]
    if name == "build_feature_matrix":
        from src.data.features import build_feature_matrix
        return build_feature_matrix
    if name == "partition_features":
        from src.data.partition import partition_features
        return partition_features
    if name == "create_dataloaders":
        from src.data.loader import create_dataloaders
        return create_dataloaders
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
