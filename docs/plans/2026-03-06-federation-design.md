# Federation & Gossip Design — Phase 3

**Date:** 2026-03-06
**Phase:** 3 (Federation & Gossip)
**Scope:** Pure-Python FedAvg (star) and D-PSGD (ring) in `src/federation/`

## Decision

Pure-Python implementation for both FedAvg and D-PSGD (Option B). No Flower dependency. Both strategies use the same interface (`train_one_epoch`, `evaluate` from `src/models/mlp`), making the comparison fair and the code simple.

## File Structure

```
src/federation/
├── __init__.py        # exports run_fedavg, run_gossip
├── fedavg.py          # Pure-Python FedAvg (star topology)
└── gossip.py          # D-PSGD ring topology with MH mixing
```

## FedAvg (`fedavg.py`)

```python
def run_fedavg(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],  # [(train, val), ...] x5
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: str = "cpu",
    seed: int = 42,
) -> dict
```

Each round:
1. Broadcast global `state_dict()` to all 5 local models
2. Each hospital trains E local epochs via `train_one_epoch()`
3. Weighted average of all `state_dict()` by dataset size (FedAvg)
4. Evaluate global model on each hospital's val set
5. Log per-round metrics

Returns: `{"round_metrics": [...], "per_hospital_metrics": [...], "final_global_metrics": dict}`

### Weighted Averaging

```python
# Weight by number of training samples
weights = [len(loader.dataset) for loader, _ in hospital_loaders]
total = sum(weights)
weights = [w / total for w in weights]

# Weighted average of state_dicts
global_state = {}
for key in state_dicts[0]:
    global_state[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
```

## D-PSGD Ring (`gossip.py`)

```python
def run_gossip(
    hospital_loaders: list[tuple[DataLoader, DataLoader]],
    n_features: int,
    n_rounds: int = 50,
    local_epochs: int = 3,
    lr: float = 1e-3,
    huber_delta: float = 5.0,
    device: str = "cpu",
    seed: int = 42,
) -> dict
```

### Ring Topology

H1↔H2↔H3↔H4↔H5↔H1 (each node has exactly 2 neighbors)

### Metropolis-Hastings Mixing Weights

For a ring of N=5 nodes where every node has degree 2, MH simplifies to equal weights: each node averages itself with its 2 neighbors at weight 1/3 each.

```python
def build_mixing_matrix(n_nodes: int = 5) -> np.ndarray:
    """Build MH mixing matrix for a ring topology."""
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        left = (i - 1) % n_nodes
        right = (i + 1) % n_nodes
        # MH weight: 1 / (1 + max(degree_i, degree_j))
        # For ring: all degrees are 2, so weight = 1/3
        W[i, left] = 1.0 / 3.0
        W[i, right] = 1.0 / 3.0
        W[i, i] = 1.0 / 3.0
    return W
```

### Each Round

1. Each hospital trains E local epochs independently
2. Each hospital averages its `state_dict()` with its 2 ring neighbors using MH weights
3. Evaluate each hospital's model on its own val set
4. Log per-round, per-hospital metrics

Returns: same format as FedAvg for easy comparison.

## Interface Contract

Both `run_fedavg` and `run_gossip`:
- Accept `list[tuple[DataLoader, DataLoader]]` (5 hospital train/val pairs)
- Create fresh `LOSModel(n_features)` instances internally
- Call `train_one_epoch()` and `evaluate()` from `src.models.mlp`
- Return dict with `round_metrics`, `per_hospital_metrics`, `final_global_metrics`

## Communication Cost Tracking

Per CLAUDE.md spec:
- **FedAvg:** 2 × N × params per round (broadcast down + upload up, all 5 clients)
- **D-PSGD:** 2 × 2 × params per node per round (send + receive with 2 neighbors)

Both functions will count total parameters exchanged and include in return dict.

## Test Plan

- FedAvg weighted averaging produces correct global model
- FedAvg global model improves over rounds (synthetic data)
- D-PSGD mixing matrix is doubly stochastic
- D-PSGD ring neighbors are correct
- D-PSGD models converge toward each other over rounds
- Both return correct metric dict structure
- Both handle variable local_epochs (E=1,3,5)

## Dependencies

No new dependencies. Uses only PyTorch, NumPy, and existing `src.models.mlp`.
