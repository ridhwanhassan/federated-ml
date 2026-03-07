# Analysis & Paper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement evaluation metrics, paper figures/tables, and LaTeX manuscript scaffold.

**Architecture:** `src/evaluation/metrics.py` for post-experiment analysis, `src/visualization/generate_all.py` for all figures and tables, `paper/` for IEEE manuscript.

**Tech Stack:** matplotlib, NumPy, LaTeX (IEEEtran)

**Design doc:** `docs/plans/2026-03-08-analysis-paper-design.md`

---

### Task 1: Evaluation metrics module

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`

Implement: `convergence_round()`, `aggregate_over_seeds()`, `per_hospital_fairness()`, `communication_cost_summary()`, `build_table_i()`, `build_table_ii()`.

**Verification:** `python -c "from src.evaluation.metrics import build_table_i, convergence_round; print('OK')"`

---

### Task 2: Figure 1 — topology diagram

**Files:**
- Create: `src/visualization/__init__.py`
- Create: `src/visualization/generate_all.py` (start with figure_1_topology)

Static matplotlib diagram, no data dependency.

**Verification:** Generate fig1_topology.pdf/png, visually inspect.

---

### Task 3: Figures 2–5

**Files:**
- Edit: `src/visualization/generate_all.py`

Add: `figure_2_convergence()`, `figure_3_per_hospital()`, `figure_4_ablation()`, `figure_5_cost_accuracy()`. All read from `all_results_raw.json`.

**Verification:** Run with smoke test data, check PDFs generated.

---

### Task 4: LaTeX table generation

**Files:**
- Edit: `src/visualization/generate_all.py`

Add: `generate_table_i_latex()`, `generate_table_ii_latex()`. Output to `paper/tables/`.

**Verification:** Check generated .tex files have valid LaTeX.

---

### Task 5: Paper manuscript

**Files:**
- Create: `paper/main.tex`
- Create: `paper/references.bib`

IEEE IEEEtran class, all sections, figure/table references, 8 BibTeX entries.

**Verification:** LaTeX structure review (compilation requires full TeX installation).

---

### Task 6: Add matplotlib dependency

**Files:**
- Edit: `pyproject.toml`

Run: `uv add matplotlib`

**Verification:** `python -c "import matplotlib; print(matplotlib.__version__)"`

---

### Final verification

- All 104 existing tests still pass
- Figure 1 generates without data
- Figures 2–5 + tables generate from smoke test results
- Paper references all figures/tables correctly
