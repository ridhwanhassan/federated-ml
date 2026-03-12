"""Exploratory data analysis for the 5-hospital federated partition.

Prints per-hospital statistics and generates distribution plots
for the paper and development reference.

Usage:
    python experiments/eda.py
    python experiments/eda.py --config experiments/configs/default.yaml
    python experiments/eda.py --save-plots results/figures/eda/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import NON_FEATURE_COLUMNS

HOSPITAL_NAMES = {
    1: "H1 (Medical)",
    2: "H2 (Neuro)",
    3: "H3 (Surgical)",
    4: "H4 (Trauma)",
    5: "H5 (Cardiac)",
}

HOSPITAL_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
}


def load_hospital_dataframes(cfg: dict) -> dict[int, pd.DataFrame]:
    """Load all hospital partition CSVs into a dict."""
    data_cfg = cfg["data"]
    partitions_dir = PROJECT_ROOT / data_cfg["partitions_dir"]
    dfs = {}
    for h_id in range(1, data_cfg["n_hospitals"] + 1):
        dfs[h_id] = pd.read_csv(partitions_dir / f"hospital_{h_id}.csv")
    return dfs


def load_pooled_dataframe(cfg: dict) -> pd.DataFrame:
    """Load the full pooled features CSV."""
    return pd.read_csv(PROJECT_ROOT / cfg["data"]["features_csv"])


def compute_statistics(dfs: dict[int, pd.DataFrame], pooled_df: pd.DataFrame) -> dict:
    """Compute per-hospital and pooled summary statistics."""
    stats = {}

    for h_id, df in dfs.items():
        los = df["los"].values
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        nan_pct = df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols)) * 100

        stats[f"hospital_{h_id}"] = {
            "name": HOSPITAL_NAMES[h_id],
            "n_samples": len(df),
            "pct_of_total": round(len(df) / sum(len(d) for d in dfs.values()) * 100, 1),
            "n_features": len(feature_cols),
            "los": {
                "mean": round(float(np.mean(los)), 3),
                "std": round(float(np.std(los)), 3),
                "median": round(float(np.median(los)), 3),
                "q25": round(float(np.percentile(los, 25)), 3),
                "q75": round(float(np.percentile(los, 75)), 3),
                "min": round(float(np.min(los)), 3),
                "max": round(float(np.max(los)), 3),
                "skewness": round(float(pd.Series(los).skew()), 3),
            },
            "missing_pct": round(nan_pct, 2),
        }

        # Age statistics (if present)
        if "anchor_age" in df.columns:
            age = df["anchor_age"].values
            stats[f"hospital_{h_id}"]["age"] = {
                "mean": round(float(np.mean(age)), 1),
                "std": round(float(np.std(age)), 1),
                "median": round(float(np.median(age)), 1),
            }

        # Gender split (if present)
        if "gender" in df.columns:
            gender_counts = df["gender"].value_counts(normalize=True)
            stats[f"hospital_{h_id}"]["gender_pct"] = {
                str(k): round(v * 100, 1) for k, v in gender_counts.items()
            }

    # Pooled
    pooled_los = pooled_df["los"].values
    feature_cols = [c for c in pooled_df.columns if c not in NON_FEATURE_COLUMNS]
    pooled_nan_pct = pooled_df[feature_cols].isna().sum().sum() / (
        len(pooled_df) * len(feature_cols)
    ) * 100

    stats["pooled"] = {
        "name": "Pooled",
        "n_samples": len(pooled_df),
        "pct_of_total": 100.0,
        "n_features": len(feature_cols),
        "los": {
            "mean": round(float(np.mean(pooled_los)), 3),
            "std": round(float(np.std(pooled_los)), 3),
            "median": round(float(np.median(pooled_los)), 3),
            "q25": round(float(np.percentile(pooled_los, 25)), 3),
            "q75": round(float(np.percentile(pooled_los, 75)), 3),
            "min": round(float(np.min(pooled_los)), 3),
            "max": round(float(np.max(pooled_los)), 3),
            "skewness": round(float(pd.Series(pooled_los).skew()), 3),
        },
        "missing_pct": round(pooled_nan_pct, 2),
    }

    return stats


def print_summary_table(stats: dict, n_hospitals: int) -> None:
    """Print formatted summary tables to stdout."""
    keys = [f"hospital_{i}" for i in range(1, n_hospitals + 1)] + ["pooled"]

    # --- Table 1: Sample sizes and LOS ---
    print("\n" + "=" * 95)
    print("DATA DISTRIBUTION ACROSS HOSPITALS")
    print("=" * 95)
    print(f"{'Hospital':<22} {'N':>7} {'%':>6} {'LOS mean':>9} {'LOS med':>8} "
          f"{'LOS std':>8} {'Q25':>6} {'Q75':>6} {'Skew':>6} {'Miss%':>6}")
    print("-" * 95)
    for key in keys:
        s = stats[key]
        los = s["los"]
        print(f"{s['name']:<22} {s['n_samples']:>7} {s['pct_of_total']:>5.1f}% "
              f"{los['mean']:>9.2f} {los['median']:>8.2f} {los['std']:>8.2f} "
              f"{los['q25']:>6.1f} {los['q75']:>6.1f} {los['skewness']:>6.2f} "
              f"{s['missing_pct']:>5.1f}%")
    print("=" * 95)

    # --- Table 2: Demographics ---
    has_age = "age" in stats["hospital_1"]
    has_gender = "gender_pct" in stats["hospital_1"]
    if has_age or has_gender:
        print(f"\n{'Hospital':<22}", end="")
        if has_age:
            print(f" {'Age mean':>9} {'Age std':>8}", end="")
        if has_gender:
            print(f" {'Gender split':>20}", end="")
        print()
        print("-" * 70)
        for key in keys[:-1]:  # skip pooled for demographics
            s = stats[key]
            print(f"{s['name']:<22}", end="")
            if has_age:
                print(f" {s['age']['mean']:>9.1f} {s['age']['std']:>8.1f}", end="")
            if has_gender:
                gender_str = ", ".join(f"{k}={v}%" for k, v in s["gender_pct"].items())
                print(f" {gender_str:>20}", end="")
            print()
        print("=" * 70)

    # --- Feature-level missing data ---
    print(f"\nTotal features: {stats['pooled']['n_features']}")
    print(f"Total samples:  {stats['pooled']['n_samples']}")


def plot_los_distributions(
    dfs: dict[int, pd.DataFrame], save_dir: Path | None = None
) -> None:
    """Plot LOS histograms and box plots per hospital."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Overlaid histograms ---
    ax = axes[0]
    for h_id, df in dfs.items():
        ax.hist(
            df["los"].values, bins=60, range=(0, 30), alpha=0.5,
            label=f"{HOSPITAL_NAMES[h_id]} (n={len(df):,})",
            color=HOSPITAL_COLORS[h_id], density=True,
        )
    ax.set_xlabel("Length of Stay (days)")
    ax.set_ylabel("Density")
    ax.set_title("LOS Distribution by Hospital")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 30)

    # --- Box plots ---
    ax = axes[1]
    data = [df["los"].values for df in dfs.values()]
    labels = [HOSPITAL_NAMES[h_id] for h_id in dfs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, h_id in zip(bp["boxes"], dfs):
        patch.set_facecolor(HOSPITAL_COLORS[h_id])
        patch.set_alpha(0.7)
    ax.set_ylabel("Length of Stay (days)")
    ax.set_title("LOS Box Plot by Hospital (outliers hidden)")

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "los_distribution.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_dir / 'los_distribution.png'}")
    plt.show()


def plot_sample_sizes(
    dfs: dict[int, pd.DataFrame], save_dir: Path | None = None
) -> None:
    """Plot bar chart of sample sizes per hospital."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [HOSPITAL_NAMES[h_id] for h_id in dfs]
    sizes = [len(df) for df in dfs.values()]
    colors = [HOSPITAL_COLORS[h_id] for h_id in dfs]
    bars = ax.bar(names, sizes, color=colors, alpha=0.8)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{size:,}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of ICU Stays")
    ax.set_title("Sample Size per Hospital (Non-IID Partition)")
    plt.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "sample_sizes.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir / 'sample_sizes.png'}")
    plt.show()


def plot_feature_missingness(
    dfs: dict[int, pd.DataFrame], save_dir: Path | None = None
) -> None:
    """Plot heatmap of missing data percentage per feature group per hospital."""
    feature_groups = {
        "Vitals": ["hr_mean", "hr_min", "hr_max", "sbp_mean", "sbp_min", "sbp_max",
                    "dbp_mean", "dbp_min", "dbp_max", "mbp_mean", "mbp_min", "mbp_max",
                    "rr_mean", "rr_min", "rr_max", "spo2_mean", "spo2_min", "spo2_max",
                    "temp_mean", "temp_min", "temp_max"],
        "Labs": ["glucose_mean", "creatinine_mean", "bun_mean", "hemoglobin_mean",
                 "platelet_mean", "wbc_mean", "sodium_mean", "potassium_mean",
                 "bicarbonate_mean", "lactate_mean"],
        "Demographics": ["gender", "anchor_age"],
        "Complexity": ["n_diagnoses", "n_procedures"],
    }

    miss_data = []
    for group_name, cols in feature_groups.items():
        row = []
        for h_id, df in dfs.items():
            present_cols = [c for c in cols if c in df.columns]
            if present_cols:
                pct = df[present_cols].isna().sum().sum() / (len(df) * len(present_cols)) * 100
            else:
                pct = 0.0
            row.append(pct)
        miss_data.append(row)

    miss_arr = np.array(miss_data)

    if miss_arr.max() == 0:
        print("\nNo missing data across feature groups — skipping missingness heatmap.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(miss_arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(dfs)))
    ax.set_xticklabels([HOSPITAL_NAMES[h_id] for h_id in dfs])
    ax.set_yticks(range(len(feature_groups)))
    ax.set_yticklabels(list(feature_groups.keys()))

    for i in range(miss_arr.shape[0]):
        for j in range(miss_arr.shape[1]):
            ax.text(j, i, f"{miss_arr[i, j]:.1f}%", ha="center", va="center", fontsize=9)

    ax.set_title("Missing Data % by Feature Group and Hospital")
    fig.colorbar(im, ax=ax, label="Missing %")
    plt.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "missingness_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir / 'missingness_heatmap.png'}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for federated hospital partitions")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "default.yaml",
    )
    parser.add_argument("--save-plots", type=Path, default=None,
                        help="Directory to save plots (default: show only)")
    parser.add_argument("--save-stats", type=Path, default=None,
                        help="Path to save statistics JSON")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, print stats only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_hospitals = cfg["data"]["n_hospitals"]

    print("Loading hospital partition data...")
    dfs = load_hospital_dataframes(cfg)
    pooled_df = load_pooled_dataframe(cfg)

    # Compute and print statistics
    stats = compute_statistics(dfs, pooled_df)
    print_summary_table(stats, n_hospitals)

    # Save statistics JSON
    if args.save_stats:
        args.save_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics to {args.save_stats}")

    # Generate plots
    if not args.no_plots:
        save_dir = args.save_plots
        plot_los_distributions(dfs, save_dir)
        plot_sample_sizes(dfs, save_dir)
        plot_feature_missingness(dfs, save_dir)


if __name__ == "__main__":
    main()
