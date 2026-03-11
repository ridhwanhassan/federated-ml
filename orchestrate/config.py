"""Orchestration constants and SSH helpers for distributed FedCost experiments.

Maps Tailscale hostnames, S3 prefixes, and provides helper functions
for SSH command construction and S3 bucket discovery.
"""

from __future__ import annotations

import json
import os
import subprocess

# Tailscale hostnames (set by user_data scripts: tailscale up --hostname=fedcost-{name})
TAILSCALE_HOSTS: dict[str, str] = {
    "fl-server": "fedcost-fl-server",
    "hospital-1": "fedcost-hospital-1",
    "hospital-2": "fedcost-hospital-2",
    "hospital-3": "fedcost-hospital-3",
    "hospital-4": "fedcost-hospital-4",
    "hospital-5": "fedcost-hospital-5",
    "centralized": "fedcost-centralized",
}

SSH_USER = "ec2-user"
REMOTE_PROJECT_DIR = "/opt/fedcost"
REMOTE_DATA_DIR = "/opt/fedcost/data"

# S3 prefixes within the data bucket
S3_DEPLOY_PREFIX = "deploy/"
S3_PARTITIONS_PREFIX = "partitions/"
S3_RAW_PREFIX = "raw/"
S3_MODELS_PREFIX = "models/"
S3_RESULTS_PREFIX = "results/"

# SSM parameter names (must match infra/ssm/parameters.py)
SSM_BUCKET_PARAM = "/fedcost/s3-data-bucket"
SSM_FL_SERVER_IP = "/fedcost/flower-server-ip"

# AWS profile used for all CLI commands
AWS_PROFILE = "ieee"
AWS_REGION = "ap-southeast-1"

# Ring topology neighbors (1-indexed hospital IDs)
RING_NEIGHBORS: dict[int, tuple[int, int]] = {
    1: (5, 2),
    2: (1, 3),
    3: (2, 4),
    4: (3, 5),
    5: (4, 1),
}


def get_s3_bucket() -> str:
    """Read the S3 data bucket name from SSM Parameter Store."""
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    result = subprocess.run(
        [
            "aws", "ssm", "get-parameter",
            "--name", SSM_BUCKET_PARAM,
            "--query", "Parameter.Value",
            "--output", "text",
            "--profile", AWS_PROFILE,
            "--region", AWS_REGION,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def ssh_cmd(host: str, cmd: str) -> list[str]:
    """Build an SSH command list for the given Tailscale host.

    Parameters
    ----------
    host : str
        Tailscale hostname (e.g. ``"fedcost-hospital-1"``).
    cmd : str
        Shell command to execute on the remote host.

    Returns
    -------
    list[str]
        Command list suitable for ``subprocess.run()``.
    """
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{SSH_USER}@{host}",
        cmd,
    ]


def ssh_run(host: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a command on a remote host via SSH."""
    return subprocess.run(
        ssh_cmd(host, cmd),
        capture_output=True,
        text=True,
        check=check,
    )


def scp_to(local_path: str, host: str, remote_path: str) -> subprocess.CompletedProcess:
    """Copy a file to a remote host via SCP."""
    return subprocess.run(
        [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{SSH_USER}@{host}:{remote_path}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def get_pulumi_outputs() -> dict:
    """Read Pulumi stack outputs (requires pulumi CLI and active stack)."""
    result = subprocess.run(
        ["pulumi", "stack", "output", "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd="infra",
    )
    return json.loads(result.stdout)
