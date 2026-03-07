# Analysis & Paper Design — Phase 5

**Date:** 2026-03-08
**Phase:** 5 (Analysis & Paper)
**Scope:** Evaluation metrics, visualization, and LaTeX manuscript in `src/evaluation/`, `src/visualization/`, `paper/`

## Decision

Matplotlib-based visualization (Option A — simple, no external dashboard). All figures generated as PDF + PNG. LaTeX tables generated programmatically from results JSON. IEEE double-column template for manuscript.

## File Structure

```
src/evaluation/
├── __init__.py
└── metrics.py              # Post-experiment analysis functions

src/visualization/
├── __init__.py
└── generate_all.py         # All 5 figures + 2 LaTeX tables

paper/
├── main.tex                # IEEE double-column manuscript
├── references.bib          # BibTeX citations
├── figures/                # Generated PDFs/PNGs
│   ├── fig1_topology.pdf
│   ├── fig2_convergence.pdf
│   ├── fig3_per_hospital.pdf
│   ├── fig4_ablation.pdf
│   └── fig5_cost_accuracy.pdf
└── tables/                 # Generated LaTeX table fragments
    ├── table_i.tex
    └── table_ii.tex
```

## Evaluation Module (`metrics.py`)

### Functions

- `convergence_round(curve, target_mae, fraction=0.95)` — first round MAE reaches within 95% of centralized baseline
- `aggregate_over_seeds(results)` — mean ± std of metrics across seeds
- `per_hospital_fairness(metrics)` — MAE std, range, max, min across hospitals
- `communication_cost_summary(results)` — mean ± std of comm costs
- `build_table_i(raw)` — constructs Table I rows from raw results
- `build_table_ii(raw)` — constructs Table II rows (per-hospital, E=3)

### Input Format

Reads `all_results_raw.json` produced by `experiments/run_all.py`.

## Visualization Module (`generate_all.py`)

### Figure 1: Topology Diagram
- Data-independent (static matplotlib drawing)
- Star topology (FedAvg) with central server + 5 hospital nodes
- Ring topology (D-PSGD) with 5 hospital nodes in pentagon

### Figure 2: Convergence Curves
- MAE vs communication round for FedAvg and D-PSGD (E=3)
- Mean line with ± 1 std shading across 5 seeds
- Centralized MLP baseline as dashed horizontal line

### Figure 3: Per-Hospital Bar Chart
- Grouped bars: local-only, FedAvg, D-PSGD per hospital
- Error bars showing ± 1 std across seeds
- 5 hospital groups on x-axis

### Figure 4: Local Epochs Ablation
- Line plot: final MAE vs E ∈ {1, 3, 5}
- Separate lines for FedAvg and D-PSGD
- Error bars showing ± 1 std

### Figure 5: Communication Cost vs Accuracy
- Scatter plot: total params exchanged vs final MAE
- Points for each (algorithm, E) combination
- Annotated with E value

### Tables (LaTeX)
- Table I: Main comparison — all experiments, MAE/RMSE/R² mean ± std
- Table II: Per-hospital MAE for FedAvg vs D-PSGD vs local-only (E=3)
- Both use `booktabs` package formatting

## Paper Structure (`main.tex`)

IEEE `IEEEtran` conference class, 8-page target:

1. **Abstract** — placeholder numbers to fill after full run
2. **Introduction** — RQ and 4 contributions
3. **Related Work** — LOS prediction, FL in healthcare, decentralized learning (TODOs)
4. **Methods** — Data, partition table, model, FedAvg, D-PSGD, baselines, evaluation
5. **Results** — References all 5 figures + 2 tables via `\input{}` and `\includegraphics`
6. **Discussion** — Trade-offs, fairness, limitations
7. **Conclusion** — Summary and future work

### References (`references.bib`)

8 key citations: McMahan (FedAvg), Lian (D-PSGD), Johnson (MIMIC-IV), Rieke (FL in healthcare), Li (FedProx/heterogeneous), Koloskova (decentralized SGD theory), Harutyunyan (clinical benchmarks), Sheikhalishahi (eICU benchmarks).

## Dependencies Added

- `matplotlib>=3.10.8` (for figure generation)
