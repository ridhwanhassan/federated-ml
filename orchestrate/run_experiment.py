"""Main CLI for dispatching distributed experiments via SSH.

Usage:
    python orchestrate/run_experiment.py --experiment fedavg --seeds 42 --local-epochs 3 --n-rounds 5
    python orchestrate/run_experiment.py --experiment all --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrate.config import (
    AWS_PROFILE,
    AWS_REGION,
    REMOTE_PROJECT_DIR,
    SSH_USER,
    TAILSCALE_HOSTS,
    get_s3_bucket,
    ssh_cmd,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ssh_nohup(host: str, remote_cmd: str, log_name: str) -> subprocess.Popen:
    """Start a remote command via SSH with nohup (non-blocking).

    Returns the SSH Popen object. The remote process continues even if SSH drops.
    """
    wrapped = (
        f"nohup bash -lc '"
        f"cd {REMOTE_PROJECT_DIR} && {remote_cmd}"
        f"' > /tmp/{log_name}.log 2>&1 &"
    )
    cmd = ssh_cmd(host, wrapped)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ssh_blocking(host: str, remote_cmd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a command on a remote host and wait for completion."""
    wrapped = f"cd {REMOTE_PROJECT_DIR} && {remote_cmd}"
    cmd = ssh_cmd(host, wrapped)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def resolve_n_features() -> int:
    """Determine n_features by running the data loader on a hospital instance."""
    host = TAILSCALE_HOSTS["hospital-1"]
    cmd = (
        "cd /opt/fedcost && uv run python -c "
        "'from src.data.loader import create_dataloaders; "
        "from pathlib import Path; "
        "tl,_,_ = create_dataloaders(Path(\"/opt/fedcost/data/hospital_1.csv\")); "
        "print(tl.dataset.X.shape[1])'"
    )
    try:
        result = ssh_blocking(host, cmd, timeout=60)
        n_features = int(result.stdout.strip().split("\n")[-1])
        logger.info("Detected n_features=%d from hospital-1", n_features)
        return n_features
    except Exception as e:
        logger.warning("Could not detect n_features: %s. Using default 75.", e)
        return 75


def run_centralized(seeds: list[int]) -> None:
    """Run centralized baselines on fedcost-centralized."""
    host = TAILSCALE_HOSTS["centralized"]
    seeds_str = " ".join(str(s) for s in seeds)
    cmd = f"uv run python orchestrate/remote/run_centralized_remote.py --seeds {seeds_str}"
    logger.info("Starting centralized on %s", host)
    result = ssh_blocking(host, cmd)
    if result.returncode != 0:
        logger.error("Centralized FAILED:\n%s", result.stderr)
    else:
        logger.info("Centralized complete")


def run_local_only(seeds: list[int]) -> None:
    """Run local-only training on all 5 hospitals in parallel."""
    seeds_str = " ".join(str(s) for s in seeds)

    def _run_one(h_id: int) -> tuple[int, subprocess.CompletedProcess]:
        host = TAILSCALE_HOSTS[f"hospital-{h_id}"]
        cmd = f"uv run python orchestrate/remote/run_local_remote.py --hospital-id {h_id} --seeds {seeds_str}"
        logger.info("Starting local-only H%d on %s", h_id, host)
        result = ssh_blocking(host, cmd)
        return h_id, result

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_run_one, h) for h in range(1, 6)]
        for fut in as_completed(futures):
            h_id, result = fut.result()
            if result.returncode != 0:
                logger.error("Local-only H%d FAILED:\n%s", h_id, result.stderr)
            else:
                logger.info("Local-only H%d complete", h_id)


def run_fedavg(
    seeds: list[int], local_epochs_list: list[int],
    n_rounds: int, n_features: int,
) -> None:
    """Run FedAvg: start coordinator on fl-server, workers on hospitals."""
    seeds_str = " ".join(str(s) for s in seeds)
    epochs_str = " ".join(str(e) for e in local_epochs_list)

    for local_epochs in local_epochs_list:
        for seed in seeds:
            logger.info("=== FedAvg seed=%d E=%d ===", seed, local_epochs)

            # Start workers on hospitals (nohup — they'll poll S3)
            worker_procs = []
            for h_id in range(1, 6):
                host = TAILSCALE_HOSTS[f"hospital-{h_id}"]
                cmd = (
                    f"uv run python orchestrate/remote/fedavg_worker.py"
                    f" --hospital-id {h_id} --seed {seed}"
                    f" --local-epochs {local_epochs} --n-rounds {n_rounds}"
                )
                log_name = f"fedavg_worker_H{h_id}_s{seed}_E{local_epochs}"
                proc = ssh_nohup(host, cmd, log_name)
                worker_procs.append((h_id, proc))
                logger.info("  Worker H%d launched on %s", h_id, host)

            # Give workers a moment to start
            time.sleep(2)

            # Run coordinator (blocking — it waits for all rounds)
            coord_host = TAILSCALE_HOSTS["fl-server"]
            coord_cmd = (
                f"uv run python orchestrate/remote/fedavg_coordinator.py"
                f" --seeds {seed} --local-epochs {local_epochs}"
                f" --n-rounds {n_rounds} --n-features {n_features}"
            )
            logger.info("  Coordinator started on %s", coord_host)
            result = ssh_blocking(coord_host, coord_cmd, timeout=7200)

            if result.returncode != 0:
                logger.error("FedAvg coordinator FAILED:\n%s", result.stderr)
            else:
                logger.info("FedAvg seed=%d E=%d complete", seed, local_epochs)

            # Clean up worker SSH processes
            for h_id, proc in worker_procs:
                proc.terminate()


def run_gossip(
    seeds: list[int], local_epochs_list: list[int],
    n_rounds: int, n_features: int,
) -> None:
    """Run D-PSGD gossip: start coordinator on fl-server, workers on hospitals."""
    for local_epochs in local_epochs_list:
        for seed in seeds:
            logger.info("=== Gossip seed=%d E=%d ===", seed, local_epochs)

            # Start workers (they'll poll for init model, then run rounds)
            worker_procs = []
            for h_id in range(1, 6):
                host = TAILSCALE_HOSTS[f"hospital-{h_id}"]
                cmd = (
                    f"uv run python orchestrate/remote/gossip_worker.py"
                    f" --hospital-id {h_id} --seed {seed}"
                    f" --local-epochs {local_epochs} --n-rounds {n_rounds}"
                )
                log_name = f"gossip_worker_H{h_id}_s{seed}_E{local_epochs}"
                proc = ssh_nohup(host, cmd, log_name)
                worker_procs.append((h_id, proc))
                logger.info("  Worker H%d launched on %s", h_id, host)

            time.sleep(2)

            # Coordinator uploads init model then monitors rounds
            coord_host = TAILSCALE_HOSTS["fl-server"]
            coord_cmd = (
                f"uv run python orchestrate/remote/gossip_coordinator.py"
                f" --seeds {seed} --local-epochs {local_epochs}"
                f" --n-rounds {n_rounds} --n-features {n_features}"
            )
            logger.info("  Coordinator started on %s", coord_host)
            result = ssh_blocking(coord_host, coord_cmd, timeout=7200)

            if result.returncode != 0:
                logger.error("Gossip coordinator FAILED:\n%s", result.stderr)
            else:
                logger.info("Gossip seed=%d E=%d complete", seed, local_epochs)

            for h_id, proc in worker_procs:
                proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch distributed FedCost experiments via SSH",
    )
    parser.add_argument(
        "--experiment",
        choices=["centralized", "local", "fedavg", "gossip", "all"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--local-epochs", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--n-rounds", type=int, default=50)
    args = parser.parse_args()

    start_time = time.time()
    experiment = args.experiment

    # Resolve n_features for federated experiments
    n_features = None
    if experiment in ("fedavg", "gossip", "all"):
        n_features = resolve_n_features()

    if experiment == "centralized":
        run_centralized(args.seeds)

    elif experiment == "local":
        run_local_only(args.seeds)

    elif experiment == "fedavg":
        run_fedavg(args.seeds, args.local_epochs, args.n_rounds, n_features)

    elif experiment == "gossip":
        run_gossip(args.seeds, args.local_epochs, args.n_rounds, n_features)

    elif experiment == "all":
        logger.info("Running ALL experiments")

        # Phase 1: centralized + local in parallel
        logger.info("Phase 1: Centralized + Local-only (parallel)")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_cent = pool.submit(run_centralized, args.seeds)
            fut_local = pool.submit(run_local_only, args.seeds)
            fut_cent.result()
            fut_local.result()

        # Phase 2: FedAvg
        logger.info("Phase 2: FedAvg")
        run_fedavg(args.seeds, args.local_epochs, args.n_rounds, n_features)

        # Phase 3: Gossip
        logger.info("Phase 3: D-PSGD Gossip")
        run_gossip(args.seeds, args.local_epochs, args.n_rounds, n_features)

    elapsed = time.time() - start_time
    logger.info("Experiment '%s' complete in %.1f seconds (%.1f min)", experiment, elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
