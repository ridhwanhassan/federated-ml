# Data Distribution Across Hospitals

5-hospital non-IID partition based on ICU care unit type from MIMIC-IV. Total: 82,596 ICU stays across 75 features.

| Hospital | N | % | LOS mean | LOS median | Skew | Missing% |
|---|---|---|---|---|---|---|
| H1 (Medical) | 30,381 | 36.8% | 2.95 | 1.78 | 3.36 | 1.0% |
| H2 (Neuro) | 8,260 | 10.0% | 4.34 | 2.72 | 2.38 | 2.2% |
| H3 (Surgical) | 11,375 | 13.8% | 3.35 | 1.93 | 3.08 | 1.2% |
| H4 (Trauma) | 9,259 | 11.2% | 3.24 | 1.84 | 3.10 | 1.2% |
| H5 (Cardiac) | 23,321 | 28.2% | 2.90 | 1.97 | 3.59 | 1.0% |

## Key Observations

- **Sample imbalance:** H1 (Medical) is the largest partition (36.8%), ~3.7x larger than H2 (Neuro, 10.0%).
- **LOS heterogeneity:** H2 (Neuro) has the highest mean LOS (4.34 days) and widest spread, while H5 (Cardiac) has the lowest (2.90 days).
- **Right-skewed distributions:** All hospitals show positive skewness (2.38-3.59), with medians well below means.
- **Missing data:** Labs are the primary source of missingness, especially in H2 (Neuro) at 15.4% for lab features. Demographics and complexity features are complete.

## Care Unit Mapping

| Hospital | Care Units |
|---|---|
| H1 (Medical) | MICU, MICU/SICU + unmatched defaults |
| H2 (Neuro) | Neuro Intermediate, Neuro Stepdown, Neuro SICU |
| H3 (Surgical) | SICU |
| H4 (Trauma) | TSICU |
| H5 (Cardiac) | CCU, CVICU |

## Generated With

```bash
python experiments/eda.py --no-plots
python experiments/eda.py --save-plots results/figures/eda/ --save-stats results/metrics/data_statistics.json
```
