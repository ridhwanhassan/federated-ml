# Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement experiment runner scripts for all 5 experiment types with YAML configuration and JSON output.

**Architecture:** Standalone CLI scripts in `experiments/` that compose `src/data/`, `src/models/`, and `src/federation/`. Results saved as JSON to `results/metrics/`.

**Tech Stack:** PyTorch, XGBoost, PyYAML, NumPy (existing deps)

**Design doc:** `docs/plans/2026-03-08-experiments-design.md`

---

### Task 1: YAML configuration

**Files:**
- Create: `experiments/configs/default.yaml`
- Create: `experiments/__init__.py`

Define all hyperparameters: data paths, model architecture, training params, federation settings, XGBoost params, experiment seeds. Use explicit float notation for YAML safe_load compatibility.

**Verification:** `python -c "import yaml; cfg = yaml.safe_load(open('experiments/configs/default.yaml')); assert isinstance(cfg['training']['lr'], float)"`

---

### Task 2: Centralized baselines (`run_centralized.py`)

**Files:**
- Create: `experiments/run_centralized.py`

Implement `run_centralized_mlp()` and `run_centralized_xgboost()`. Load pooled features.csv, train, evaluate, return result dicts. CLI with `--config` and `--seeds` args.

**Verification:** `python experiments/run_centralized.py --seeds 42`

---

### Task 3: FedAvg runner (`run_federated.py`)

**Files:**
- Create: `experiments/run_federated.py`

Implement `load_hospital_loaders()` (shared with gossip) and `run_single()`. Iterate over seeds × local_epochs. CLI with `--config`, `--seeds`, `--n-rounds`, `--local-epochs`.

**Verification:** `python experiments/run_federated.py --seeds 42 --local-epochs 3 --n-rounds 3`

---

### Task 4: D-PSGD runner (`run_gossip.py`)

**Files:**
- Create: `experiments/run_gossip.py`

Same structure as run_federated.py but calls `run_gossip()`. Separate `load_hospital_loaders()` for independence.

**Verification:** `python experiments/run_gossip.py --seeds 42 --local-epochs 3 --n-rounds 3`

---

### Task 5: Local-only runner (`run_local_only.py`)

**Files:**
- Create: `experiments/run_local_only.py`

Train each hospital independently via `train_model()`. Iterate seeds × hospitals.

**Verification:** `python experiments/run_local_only.py --seeds 42`

---

### Task 6: Orchestrator (`run_all.py`)

**Files:**
- Create: `experiments/run_all.py`

Import run functions from other scripts. Execute all experiments, aggregate mean ± std, save raw + summary JSON, print table.

**Verification:** Smoke test with `--seeds 42` and reduced rounds (modify config temporarily).

---

### Final verification

- All 4 scripts produce valid JSON in `results/metrics/`
- Existing 104 tests still pass
- Full run: `python experiments/run_all.py` (production run with all seeds)
