"""Generate all paper figures and tables from experiment results.

Figures:
    1. Architecture diagram (star vs ring) — static, no data needed
    2. Convergence curves (MAE vs rounds) — FedAvg vs D-PSGD
    3. Per-hospital performance bar chart
    4. Local epochs ablation (E=1,3,5)
    5. Communication cost vs accuracy scatter

Tables:
    I.  Main comparison (all experiments)
    II. Per-hospital breakdown

Usage:
    python src/visualization/generate_all.py --results-dir results/metrics/ --output paper/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    aggregate_over_seeds,
    build_table_i,
    build_table_ii,
    communication_cost_summary,
    per_hospital_fairness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HOSPITAL_NAMES = ["H1\nMedical", "H2\nNeuro", "H3\nSurgical", "H4\nTrauma", "H5\nCardiac"]
HOSPITAL_SHORT = ["H1", "H2", "H3", "H4", "H5"]


def load_raw(results_dir: Path) -> dict:
    """Load raw results JSON."""
    with open(results_dir / "all_results_raw.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Architecture diagram (star vs ring)
# ---------------------------------------------------------------------------
def figure_1_topology(output_dir: Path) -> None:
    """Generate topology comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Star topology (FedAvg)
    theta = np.linspace(0, 2 * np.pi, 6)[:-1]
    hx = 2.0 * np.cos(theta)
    hy = 2.0 * np.sin(theta)

    ax1.plot(0, 0, "s", color="#2196F3", markersize=20, zorder=5)
    ax1.annotate("Server", (0, 0), ha="center", va="center", fontsize=7,
                 fontweight="bold", color="white", zorder=6)
    for i in range(5):
        ax1.plot(hx[i], hy[i], "o", color="#4CAF50", markersize=16, zorder=5)
        ax1.annotate(HOSPITAL_SHORT[i], (hx[i], hy[i]), ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white", zorder=6)
        ax1.plot([0, hx[i]], [0, hy[i]], "-", color="#666", linewidth=1.5, zorder=1)

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect("equal")
    ax1.set_title("FedAvg (Star Topology)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Ring topology (D-PSGD)
    for i in range(5):
        ax2.plot(hx[i], hy[i], "o", color="#FF9800", markersize=16, zorder=5)
        ax2.annotate(HOSPITAL_SHORT[i], (hx[i], hy[i]), ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white", zorder=6)
        j = (i + 1) % 5
        ax2.annotate("", xy=(hx[j], hy[j]), xytext=(hx[i], hy[i]),
                     arrowprops=dict(arrowstyle="<->", color="#666", lw=1.5))

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect("equal")
    ax2.set_title("D-PSGD (Ring Topology)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_topology.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig1_topology.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 1: topology diagram")


# ---------------------------------------------------------------------------
# Figure 2: Convergence curves (MAE vs rounds)
# ---------------------------------------------------------------------------
def figure_2_convergence(raw: dict, output_dir: Path) -> None:
    """Plot MAE convergence curves for FedAvg vs D-PSGD (E=3)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    fedavg_e3 = raw["fedavg"].get("fedavg_E3", [])
    gossip_e3 = raw["gossip"].get("gossip_E3", [])

    if fedavg_e3:
        curves = np.array([r["convergence_curve"] for r in fedavg_e3])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        rounds = np.arange(1, len(mean_curve) + 1)
        ax.plot(rounds, mean_curve, "-o", label="FedAvg (E=3)", color="#2196F3",
                markersize=3, linewidth=1.5)
        ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color="#2196F3")

    if gossip_e3:
        curves = np.array([r["convergence_curve"] for r in gossip_e3])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        rounds = np.arange(1, len(mean_curve) + 1)
        ax.plot(rounds, mean_curve, "-s", label="D-PSGD (E=3)", color="#FF9800",
                markersize=3, linewidth=1.5)
        ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color="#FF9800")

    # Centralized baseline reference line
    cent_mlp = raw.get("centralized_mlp", [])
    if cent_mlp:
        cent_mae = np.mean([r["final_metrics"]["mae"] for r in cent_mlp])
        ax.axhline(y=cent_mae, color="#4CAF50", linestyle="--", linewidth=1,
                   label=f"Centralized MLP ({cent_mae:.3f})")

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Validation MAE (days)", fontsize=11)
    ax.set_title("Convergence: FedAvg vs D-PSGD", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_convergence.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig2_convergence.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 2: convergence curves")


# ---------------------------------------------------------------------------
# Figure 3: Per-hospital performance bar chart
# ---------------------------------------------------------------------------
def figure_3_per_hospital(raw: dict, output_dir: Path) -> None:
    """Bar chart of per-hospital MAE: FedAvg vs D-PSGD vs local-only."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    fedavg_e3 = raw["fedavg"].get("fedavg_E3", [])
    gossip_e3 = raw["gossip"].get("gossip_E3", [])

    x = np.arange(5)
    width = 0.25

    # Local-only
    local_means = []
    local_stds = []
    for h_id in range(1, 6):
        key = f"local_H{h_id}"
        results = raw["local_only"].get(key, [])
        maes = [r["final_metrics"]["mae"] for r in results]
        local_means.append(np.mean(maes) if maes else 0)
        local_stds.append(np.std(maes) if maes else 0)

    ax.bar(x - width, local_means, width, yerr=local_stds, label="Local-only",
           color="#9E9E9E", capsize=3)

    # FedAvg
    if fedavg_e3:
        fed_means = []
        fed_stds = []
        for h_idx in range(5):
            maes = [r["per_hospital_final"][h_idx]["mae"] for r in fedavg_e3]
            fed_means.append(np.mean(maes))
            fed_stds.append(np.std(maes))
        ax.bar(x, fed_means, width, yerr=fed_stds, label="FedAvg (E=3)",
               color="#2196F3", capsize=3)

    # D-PSGD
    if gossip_e3:
        gos_means = []
        gos_stds = []
        for h_idx in range(5):
            maes = [r["per_hospital_final"][h_idx]["mae"] for r in gossip_e3]
            gos_means.append(np.mean(maes))
            gos_stds.append(np.std(maes))
        ax.bar(x + width, gos_means, width, yerr=gos_stds, label="D-PSGD (E=3)",
               color="#FF9800", capsize=3)

    ax.set_xlabel("Hospital", fontsize=11)
    ax.set_ylabel("Validation MAE (days)", fontsize=11)
    ax.set_title("Per-Hospital Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(HOSPITAL_NAMES, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_per_hospital.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig3_per_hospital.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 3: per-hospital bar chart")


# ---------------------------------------------------------------------------
# Figure 4: Local epochs ablation (E=1,3,5)
# ---------------------------------------------------------------------------
def figure_4_ablation(raw: dict, output_dir: Path) -> None:
    """Plot final MAE for different local epoch settings."""
    fig, ax = plt.subplots(figsize=(6, 4))

    e_values = []
    fedavg_means = []
    fedavg_stds = []
    gossip_means = []
    gossip_stds = []

    for e in [1, 3, 5]:
        e_values.append(e)

        fed_key = f"fedavg_E{e}"
        fed_results = raw["fedavg"].get(fed_key, [])
        if fed_results:
            maes = [r["final_metrics"]["mae"] for r in fed_results]
            fedavg_means.append(np.mean(maes))
            fedavg_stds.append(np.std(maes))
        else:
            fedavg_means.append(0)
            fedavg_stds.append(0)

        gos_key = f"gossip_E{e}"
        gos_results = raw["gossip"].get(gos_key, [])
        if gos_results:
            maes = [r["final_metrics"]["mae"] for r in gos_results]
            gossip_means.append(np.mean(maes))
            gossip_stds.append(np.std(maes))
        else:
            gossip_means.append(0)
            gossip_stds.append(0)

    ax.errorbar(e_values, fedavg_means, yerr=fedavg_stds, fmt="-o",
                label="FedAvg", color="#2196F3", capsize=5, linewidth=2, markersize=8)
    ax.errorbar(e_values, gossip_means, yerr=gossip_stds, fmt="-s",
                label="D-PSGD", color="#FF9800", capsize=5, linewidth=2, markersize=8)

    ax.set_xlabel("Local Epochs (E)", fontsize=11)
    ax.set_ylabel("Final MAE (days)", fontsize=11)
    ax.set_title("Ablation: Effect of Local Epochs", fontsize=12, fontweight="bold")
    ax.set_xticks([1, 3, 5])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_ablation.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig4_ablation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 4: local epochs ablation")


# ---------------------------------------------------------------------------
# Figure 5: Communication cost vs accuracy scatter
# ---------------------------------------------------------------------------
def figure_5_cost_accuracy(raw: dict, output_dir: Path) -> None:
    """Scatter plot: communication cost (params exchanged) vs final MAE."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    markers = {"fedavg": ("o", "#2196F3", "FedAvg"), "gossip": ("s", "#FF9800", "D-PSGD")}

    for exp_type, (marker, color, label) in markers.items():
        plotted_label = False
        for key, results in raw[exp_type].items():
            if not results:
                continue
            costs = [r["communication_cost"] for r in results]
            maes = [r["final_metrics"]["mae"] for r in results]
            local_e = results[0]["local_epochs"]
            ax.scatter(
                np.mean(costs), np.mean(maes),
                marker=marker, color=color, s=100, zorder=5,
                label=label if not plotted_label else None,
            )
            ax.annotate(f"E={local_e}", (np.mean(costs), np.mean(maes)),
                        textcoords="offset points", xytext=(8, 4), fontsize=8)
            plotted_label = True

    ax.set_xlabel("Total Parameters Exchanged", fontsize=11)
    ax.set_ylabel("Final MAE (days)", fontsize=11)
    ax.set_title("Communication Cost vs Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_cost_accuracy.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig5_cost_accuracy.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 5: cost vs accuracy scatter")


# ---------------------------------------------------------------------------
# Tables as LaTeX
# ---------------------------------------------------------------------------
def generate_table_i_latex(raw: dict, output_dir: Path) -> None:
    """Generate Table I as LaTeX source."""
    rows = build_table_i(raw)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main comparison of all training strategies. "
        r"MAE and RMSE in days; R$^2$ dimensionless; "
        r"W-1d is \% of predictions within 1 day of true LOS. "
        r"Mean $\pm$ std over 5 seeds.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & R$^2$ $\uparrow$ & W-1d $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        w1d = row.get('within_1day', 'N/A')
        lines.append(
            f"{row['experiment']} & {row['mae']} & {row['rmse']} & {row['r2']} & {w1d} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = output_dir / "table_i.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved Table I: %s", out_path)


def generate_table_ii_latex(raw: dict, output_dir: Path) -> None:
    """Generate Table II as LaTeX source."""
    rows = build_table_ii(raw)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-hospital MAE (days) for FedAvg, D-PSGD, and local-only "
        r"(E=3, mean $\pm$ std over 5 seeds).}",
        r"\label{tab:per-hospital}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Hospital & FedAvg & D-PSGD & Local-only \\",
        r"\midrule",
    ]
    for row in rows:
        fed = f"{row.get('fedavg_mae_mean', 0):.3f} $\\pm$ {row.get('fedavg_mae_std', 0):.3f}"
        gos = f"{row.get('gossip_mae_mean', 0):.3f} $\\pm$ {row.get('gossip_mae_std', 0):.3f}"
        loc = f"{row.get('local_mae_mean', 0):.3f} $\\pm$ {row.get('local_mae_std', 0):.3f}"
        lines.append(f"{row['hospital']} & {fed} & {gos} & {loc} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = output_dir / "table_ii.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved Table II: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument(
        "--results-dir", type=Path,
        default=PROJECT_ROOT / "results" / "metrics",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "paper" / "figures",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1 is data-independent
    figure_1_topology(args.output)

    # Figures 2-5 and tables need experiment results
    raw_path = args.results_dir / "all_results_raw.json"
    if not raw_path.exists():
        logger.warning(
            "No all_results_raw.json found at %s. "
            "Run experiments/run_all.py first. Only generating Figure 1.",
            raw_path,
        )
        return

    raw = load_raw(args.results_dir)

    figure_2_convergence(raw, args.output)
    figure_3_per_hospital(raw, args.output)
    figure_4_ablation(raw, args.output)
    figure_5_cost_accuracy(raw, args.output)

    generate_table_i_latex(raw, tables_dir)
    generate_table_ii_latex(raw, tables_dir)

    logger.info("All figures and tables generated successfully.")


if __name__ == "__main__":
    main()
