# Federation & Gossip Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement pure-Python FedAvg (star) and D-PSGD (ring gossip) federation strategies for 5-hospital ICU LOS prediction.

**Architecture:** Two files in `src/federation/` — `fedavg.py` and `gossip.py` — each exposing a single `run_*` function. Both call `train_one_epoch()` and `evaluate()` from `src.models.mlp`. No external federation library. Matching return formats enable direct comparison.

**Tech Stack:** PyTorch, NumPy (existing deps only)

**Design doc:** `docs/plans/2026-03-06-federation-design.md`

---

### Task 1: FedAvg helper — weighted_average_state_dicts + tests

**Files:**
- Create: `tests/test_federation.py`
- Create: `src/federation/__init__.py`
- Create: `src/federation/fedavg.py`

**Step 1: Write the failing tests**

Create `tests/test_federation.py`:

```python
"""Unit tests for federation strategies."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import LOSModel
from src.federation.fedavg import weighted_average_state_dicts


class TestWeightedAverageStateDicts:
    def test_equal_weights(self):
        """Equal weights should produce arithmetic mean."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        m2 = LOSModel(n_features=5, hidden_dims=(8,))
        # Set all params to known values
        with torch.no_grad():
            for p in m1.parameters():
                p.fill_(2.0)
            for p in m2.parameters():
                p.fill_(4.0)
        avg = weighted_average_state_dicts(
            [m1.state_dict(), m2.state_dict()],
            weights=[0.5, 0.5],
        )
        for key in avg:
            assert torch.allclose(avg[key], torch.full_like(avg[key], 3.0))

    def test_unequal_weights(self):
        """Weighted average with 0.75/0.25 split."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        m2 = LOSModel(n_features=5, hidden_dims=(8,))
        with torch.no_grad():
            for p in m1.parameters():
                p.fill_(0.0)
            for p in m2.parameters():
                p.fill_(4.0)
        avg = weighted_average_state_dicts(
            [m1.state_dict(), m2.state_dict()],
            weights=[0.75, 0.25],
        )
        for key in avg:
            if avg[key].is_floating_point():
                assert torch.allclose(avg[key], torch.full_like(avg[key], 1.0))

    def test_single_model(self):
        """Single model with weight 1.0 returns same state."""
        m1 = LOSModel(n_features=5, hidden_dims=(8,))
        orig = {k: v.clone() for k, v in m1.state_dict().items()}
        avg = weighted_average_state_dicts([m1.state_dict()], weights=[1.0])
        for key in avg:
            assert torch.equal(avg[key], orig[key])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_federation.py::TestWeightedAverageStateDicts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.federation'`

**Step 3: Implement weighted_average_state_dicts**

Create `src/federation/__init__.py`:

```python
"""Federation strategies for FedCost."""
```

Create `src/federation/fedavg.py`:

```python
"""Pure-Python FedAvg (star topology) for ICU LOS prediction.

Implements Federated Averaging where a central server broadcasts a
global model, each hospital trains locally, and the server aggregates
via weighted average of state_dicts.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp import LOSModel, evaluate, train_one_epoch

logger = logging.getLogger(__name__)


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Compute weighted average of model state_dicts.

    Parameters
    ----------
    state_dicts : list[dict[str, torch.Tensor]]
        List of state_dicts from each client.
    weights : list[float]
        Weights for each client (should sum to 1.0).

    Returns
    -------
    dict[str, torch.Tensor]
        Averaged state_dict.
    """
    avg: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
        # Preserve original dtype (e.g. for BatchNorm num_batches_tracked)
        avg[key] = avg[key].to(state_dicts[0][key].dtype)
    return avg
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_federation.py::TestWeightedAverageStateDicts -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/federation/__init__.py src/federation/fedavg.py tests/test_federation.py
git commit -m "feat: add weighted_average_state_dicts for FedAvg"
```

---

### Task 2: run_fedavg — test and implement

**Files:**
- Modify: `tests/test_federation.py`
- Modify: `src/federation/fedavg.py`

**Step 1: Write the failing tests**

Append to `tests/test_federation.py`:

```python
from src.federation.fedavg import run_fedavg


def _make_hospital_loaders(n_hospitals=5, n_samples=80, n_features=10, seed=42):
    """Create synthetic hospital DataLoader pairs for testing."""
    torch.manual_seed(seed)
    loaders = []
    for i in range(n_hospitals):
        n = n_samples + i * 10  # Vary sizes for weighted avg
        X = torch.randn(n, n_features)
        y = torch.randn(n).abs() * 5
        n_train = int(n * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        loaders.append((train_loader, val_loader))
    return loaders


class TestRunFedAvg:
    def test_returns_expected_keys(self):
        """Result should have round_metrics, per_hospital_metrics, etc."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert "round_metrics" in result
        assert "per_hospital_metrics" in result
        assert "final_global_metrics" in result
        assert "communication_cost" in result

    def test_round_metrics_length(self):
        """Should have one entry per round."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=3, local_epochs=1)
        assert len(result["round_metrics"]) == 3

    def test_round_metrics_structure(self):
        """Each round metric should have mae, rmse, r2."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        for m in result["round_metrics"]:
            assert set(m.keys()) == {"mae", "rmse", "r2"}

    def test_per_hospital_metrics_shape(self):
        """per_hospital_metrics[round][hospital] should have metrics."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert len(result["per_hospital_metrics"]) == 2  # n_rounds
        assert len(result["per_hospital_metrics"][0]) == 3  # n_hospitals

    def test_mae_improves_over_rounds(self):
        """Global MAE should generally improve over rounds."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=10, local_epochs=2, lr=1e-2)
        first_mae = result["round_metrics"][0]["mae"]
        last_mae = result["round_metrics"][-1]["mae"]
        assert last_mae < first_mae

    def test_communication_cost_positive(self):
        """Communication cost should be positive."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert result["communication_cost"] > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_federation.py::TestRunFedAvg -v`
Expected: FAIL with `ImportError: cannot import name 'run_fedavg'`

**Step 3: Implement run_fedavg**

Append to `src/federation/fedavg.py`:

```python
def run_fedavg(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Run Federated Averaging (star topology) simulation.

    Parameters
    ----------
    hospital_loaders : list[tuple[DataLoader, DataLoader]]
        List of ``(train_loader, val_loader)`` per hospital.
    n_features : int
        Number of input features for LOSModel.
    n_rounds : int, optional
        Number of federation rounds, by default 50.
    local_epochs : int, optional
        Local training epochs per round, by default 3.
    lr : float, optional
        Learning rate for local Adam optimizers, by default 1e-3.
    huber_delta : float, optional
        Delta for HuberLoss, by default 5.0.
    device : torch.device or str, optional
        Device, by default ``"cpu"``.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    dict
        ``{"round_metrics": list[dict], "per_hospital_metrics": list[list[dict]],
        "final_global_metrics": dict, "communication_cost": int}``
    """
    torch.manual_seed(seed)
    n_hospitals = len(hospital_loaders)

    # Dataset-size weights for FedAvg
    dataset_sizes = [len(tl.dataset) for tl, _ in hospital_loaders]
    total_samples = sum(dataset_sizes)
    weights = [s / total_samples for s in dataset_sizes]

    # Initialize global model
    global_model = LOSModel(n_features=n_features)
    global_model.to(device)
    criterion = nn.HuberLoss(delta=huber_delta)

    # Count params for communication cost
    n_params = sum(p.numel() for p in global_model.parameters())

    round_metrics: list[dict[str, float]] = []
    per_hospital_metrics: list[list[dict[str, float]]] = []

    for rnd in range(n_rounds):
        local_state_dicts: list[dict[str, torch.Tensor]] = []

        # Local training
        for h_idx in range(n_hospitals):
            local_model = LOSModel(n_features=n_features)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model.to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            train_loader, _ = hospital_loaders[h_idx]

            for _ in range(local_epochs):
                train_one_epoch(local_model, train_loader, optimizer, criterion, device)

            local_state_dicts.append(local_model.state_dict())

        # Aggregate
        global_state = weighted_average_state_dicts(local_state_dicts, weights)
        global_model.load_state_dict(global_state)

        # Evaluate global model on each hospital's val set
        hospital_evals: list[dict[str, float]] = []
        for h_idx in range(n_hospitals):
            _, val_loader = hospital_loaders[h_idx]
            metrics = evaluate(global_model, val_loader, device)
            hospital_evals.append(metrics)

        # Average metrics across hospitals (weighted by dataset size)
        avg_mae = sum(w * m["mae"] for w, m in zip(weights, hospital_evals))
        avg_rmse = sum(w * m["rmse"] for w, m in zip(weights, hospital_evals))
        avg_r2 = sum(w * m["r2"] for w, m in zip(weights, hospital_evals))
        round_metric = {"mae": avg_mae, "rmse": avg_rmse, "r2": avg_r2}

        round_metrics.append(round_metric)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "FedAvg round %d/%d — mae=%.4f, rmse=%.4f, r2=%.4f",
            rnd + 1, n_rounds, avg_mae, avg_rmse, avg_r2,
        )

    # Communication cost: 2 * N * params per round (broadcast + upload)
    comm_cost = 2 * n_hospitals * n_params * n_rounds

    return {
        "round_metrics": round_metrics,
        "per_hospital_metrics": per_hospital_metrics,
        "final_global_metrics": round_metrics[-1] if round_metrics else {},
        "communication_cost": comm_cost,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_federation.py -v`
Expected: All 9 tests PASS (3 weighted_avg + 6 run_fedavg)

**Step 5: Commit**

```bash
git add src/federation/fedavg.py tests/test_federation.py
git commit -m "feat: add run_fedavg pure-Python FedAvg simulation"
```

---

### Task 3: build_mixing_matrix + D-PSGD tests

**Files:**
- Modify: `tests/test_federation.py`
- Create: `src/federation/gossip.py`

**Step 1: Write the failing tests**

Append to `tests/test_federation.py`:

```python
from src.federation.gossip import build_mixing_matrix


class TestBuildMixingMatrix:
    def test_shape(self):
        """Matrix should be N x N."""
        W = build_mixing_matrix(5)
        assert W.shape == (5, 5)

    def test_doubly_stochastic(self):
        """Rows and columns should sum to 1."""
        W = build_mixing_matrix(5)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)

    def test_symmetric(self):
        """MH matrix for ring should be symmetric."""
        W = build_mixing_matrix(5)
        np.testing.assert_allclose(W, W.T, atol=1e-10)

    def test_ring_neighbors(self):
        """Each node should have non-zero weights only for self and 2 neighbors."""
        W = build_mixing_matrix(5)
        for i in range(5):
            left = (i - 1) % 5
            right = (i + 1) % 5
            for j in range(5):
                if j in (i, left, right):
                    assert W[i, j] > 0
                else:
                    assert W[i, j] == 0.0

    def test_equal_weights_for_ring(self):
        """For a ring (all degree 2), weights should be 1/3."""
        W = build_mixing_matrix(5)
        for i in range(5):
            left = (i - 1) % 5
            right = (i + 1) % 5
            assert W[i, i] == pytest.approx(1.0 / 3.0)
            assert W[i, left] == pytest.approx(1.0 / 3.0)
            assert W[i, right] == pytest.approx(1.0 / 3.0)

    def test_different_sizes(self):
        """Should work for different ring sizes."""
        for n in [3, 4, 7]:
            W = build_mixing_matrix(n)
            assert W.shape == (n, n)
            np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_federation.py::TestBuildMixingMatrix -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement build_mixing_matrix**

Create `src/federation/gossip.py`:

```python
"""D-PSGD ring gossip topology for ICU LOS prediction.

Implements Decentralized Parallel SGD where each node trains locally
and exchanges models with its 2 ring neighbors using
Metropolis-Hastings mixing weights.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp import LOSModel, evaluate, train_one_epoch

logger = logging.getLogger(__name__)


def build_mixing_matrix(n_nodes: int = 5) -> np.ndarray:
    """Build Metropolis-Hastings mixing matrix for a ring topology.

    For a ring where every node has degree 2, the MH weights simplify
    to 1/3 for self and each neighbor.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes in the ring, by default 5.

    Returns
    -------
    np.ndarray
        Doubly stochastic mixing matrix of shape ``(n_nodes, n_nodes)``.
    """
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        left = (i - 1) % n_nodes
        right = (i + 1) % n_nodes
        # MH weight for ring: 1 / (1 + max(deg_i, deg_j)) = 1/3
        W[i, left] = 1.0 / 3.0
        W[i, right] = 1.0 / 3.0
        W[i, i] = 1.0 / 3.0
    return W
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_federation.py::TestBuildMixingMatrix -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/federation/gossip.py tests/test_federation.py
git commit -m "feat: add build_mixing_matrix for D-PSGD ring"
```

---

### Task 4: run_gossip — test and implement

**Files:**
- Modify: `tests/test_federation.py`
- Modify: `src/federation/gossip.py`

**Step 1: Write the failing tests**

Append to `tests/test_federation.py`:

```python
from src.federation.gossip import run_gossip


class TestRunGossip:
    def test_returns_expected_keys(self):
        """Result should have round_metrics, per_hospital_metrics, etc."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert "round_metrics" in result
        assert "per_hospital_metrics" in result
        assert "final_global_metrics" in result
        assert "communication_cost" in result

    def test_round_metrics_length(self):
        """Should have one entry per round."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=3, local_epochs=1)
        assert len(result["round_metrics"]) == 3

    def test_per_hospital_metrics_shape(self):
        """per_hospital_metrics[round][hospital] should have metrics."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert len(result["per_hospital_metrics"]) == 2
        assert len(result["per_hospital_metrics"][0]) == 3

    def test_mae_improves_over_rounds(self):
        """Average MAE should generally improve over rounds."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=10, local_epochs=2, lr=1e-2)
        first_mae = result["round_metrics"][0]["mae"]
        last_mae = result["round_metrics"][-1]["mae"]
        assert last_mae < first_mae

    def test_models_converge_toward_each_other(self):
        """After many rounds, per-hospital MAEs should be closer together."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=100, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=15, local_epochs=2, lr=1e-2)
        first_round_maes = [m["mae"] for m in result["per_hospital_metrics"][0]]
        last_round_maes = [m["mae"] for m in result["per_hospital_metrics"][-1]]
        first_spread = max(first_round_maes) - min(first_round_maes)
        last_spread = max(last_round_maes) - min(last_round_maes)
        # Last round spread should be no worse (allow some tolerance)
        assert last_spread <= first_spread + 0.5

    def test_communication_cost_positive(self):
        """Communication cost should be positive."""
        loaders = _make_hospital_loaders(n_hospitals=3, n_samples=40, n_features=5)
        result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert result["communication_cost"] > 0

    def test_communication_cost_less_than_fedavg(self):
        """D-PSGD ring should exchange fewer params than FedAvg star."""
        loaders = _make_hospital_loaders(n_hospitals=5, n_samples=40, n_features=5)
        fedavg_result = run_fedavg(loaders, n_features=5, n_rounds=2, local_epochs=1)
        gossip_result = run_gossip(loaders, n_features=5, n_rounds=2, local_epochs=1)
        assert gossip_result["communication_cost"] < fedavg_result["communication_cost"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_federation.py::TestRunGossip -v`
Expected: FAIL with `ImportError: cannot import name 'run_gossip'`

**Step 3: Implement run_gossip**

Append to `src/federation/gossip.py`:

```python
def run_gossip(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Run D-PSGD ring gossip simulation.

    Parameters
    ----------
    hospital_loaders : list[tuple[DataLoader, DataLoader]]
        List of ``(train_loader, val_loader)`` per hospital.
    n_features : int
        Number of input features for LOSModel.
    n_rounds : int, optional
        Number of gossip rounds, by default 50.
    local_epochs : int, optional
        Local training epochs per round, by default 3.
    lr : float, optional
        Learning rate for local Adam optimizers, by default 1e-3.
    huber_delta : float, optional
        Delta for HuberLoss, by default 5.0.
    device : torch.device or str, optional
        Device, by default ``"cpu"``.
    seed : int, optional
        Random seed, by default 42.

    Returns
    -------
    dict
        ``{"round_metrics": list[dict], "per_hospital_metrics": list[list[dict]],
        "final_global_metrics": dict, "communication_cost": int}``
    """
    torch.manual_seed(seed)
    n_hospitals = len(hospital_loaders)
    W = build_mixing_matrix(n_hospitals)
    criterion = nn.HuberLoss(delta=huber_delta)

    # Initialize per-hospital models (all start from same init)
    init_model = LOSModel(n_features=n_features)
    models = []
    for _ in range(n_hospitals):
        m = LOSModel(n_features=n_features)
        m.load_state_dict(copy.deepcopy(init_model.state_dict()))
        m.to(device)
        models.append(m)

    # Count params for communication cost
    n_params = sum(p.numel() for p in init_model.parameters())

    round_metrics: list[dict[str, float]] = []
    per_hospital_metrics: list[list[dict[str, float]]] = []

    for rnd in range(n_rounds):
        # Local training
        for h_idx in range(n_hospitals):
            optimizer = torch.optim.Adam(models[h_idx].parameters(), lr=lr)
            train_loader, _ = hospital_loaders[h_idx]
            for _ in range(local_epochs):
                train_one_epoch(models[h_idx], train_loader, optimizer, criterion, device)

        # Gossip mixing: each node averages with neighbors per mixing matrix
        old_state_dicts = [copy.deepcopy(m.state_dict()) for m in models]
        for h_idx in range(n_hospitals):
            new_state: dict[str, torch.Tensor] = {}
            for key in old_state_dicts[0]:
                new_state[key] = sum(
                    W[h_idx, j] * old_state_dicts[j][key].float()
                    for j in range(n_hospitals)
                    if W[h_idx, j] > 0
                )
                new_state[key] = new_state[key].to(old_state_dicts[h_idx][key].dtype)
            models[h_idx].load_state_dict(new_state)

        # Evaluate each hospital's model on its own val set
        hospital_evals: list[dict[str, float]] = []
        for h_idx in range(n_hospitals):
            _, val_loader = hospital_loaders[h_idx]
            metrics = evaluate(models[h_idx], val_loader, device)
            hospital_evals.append(metrics)

        # Average metrics across hospitals (simple mean — no central server)
        avg_mae = float(np.mean([m["mae"] for m in hospital_evals]))
        avg_rmse = float(np.mean([m["rmse"] for m in hospital_evals]))
        avg_r2 = float(np.mean([m["r2"] for m in hospital_evals]))
        round_metric = {"mae": avg_mae, "rmse": avg_rmse, "r2": avg_r2}

        round_metrics.append(round_metric)
        per_hospital_metrics.append(hospital_evals)

        logger.info(
            "D-PSGD round %d/%d — mae=%.4f, rmse=%.4f, r2=%.4f",
            rnd + 1, n_rounds, avg_mae, avg_rmse, avg_r2,
        )

    # Communication cost: 2 neighbors * 2 (send+recv) * params per node per round
    # = 4 * params * n_hospitals * n_rounds
    # But each exchange is counted once per direction: each node sends to 2 neighbors
    # Total: 2 * n_hospitals * params * n_rounds (each node sends 2, receives 2)
    comm_cost = 2 * 2 * n_params * n_hospitals * n_rounds

    return {
        "round_metrics": round_metrics,
        "per_hospital_metrics": per_hospital_metrics,
        "final_global_metrics": round_metrics[-1] if round_metrics else {},
        "communication_cost": comm_cost,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_federation.py -v`
Expected: All 22 tests PASS

**Step 5: Commit**

```bash
git add src/federation/gossip.py tests/test_federation.py
git commit -m "feat: add run_gossip D-PSGD ring simulation"
```

---

### Task 5: Update __init__.py exports and run full test suite

**Files:**
- Modify: `src/federation/__init__.py`

**Step 1: Update exports**

Replace `src/federation/__init__.py` with:

```python
"""Federation strategies for FedCost."""

from src.federation.fedavg import run_fedavg, weighted_average_state_dicts
from src.federation.gossip import build_mixing_matrix, run_gossip

__all__ = [
    "run_fedavg",
    "weighted_average_state_dicts",
    "run_gossip",
    "build_mixing_matrix",
]
```

**Step 2: Run full test suite**

Run: `pytest tests/test_federation.py tests/test_model.py tests/test_xgboost_baseline.py tests/test_data_loader.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/federation/__init__.py
git commit -m "feat: export federation API from src.federation"
```
