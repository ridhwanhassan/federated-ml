"""Pulumi mock setup for FedCost infrastructure unit tests.

This conftest runs before any test module loads, ensuring:
1. Pulumi mocks are active (no AWS credentials needed)
2. infra/ is on sys.path for module imports
3. Post-deploy smoke tests are skipped unless --run-deployed is passed
"""

import sys
from pathlib import Path

import pulumi
import pytest


# ── Post-deploy test gating ─────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--run-deployed",
        action="store_true",
        default=False,
        help="Run post-deploy smoke tests against live AWS resources",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "deployed: marks tests that require a live deployed stack"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-deployed"):
        return
    skip = pytest.mark.skip(reason="need --run-deployed option to run")
    for item in items:
        if "deployed" in item.keywords:
            item.add_marker(skip)


class FedCostMocks(pulumi.runtime.Mocks):
    """Mock Pulumi resource creation and provider calls."""

    def new_resource(
        self, args: pulumi.runtime.MockResourceArgs
    ) -> tuple[str, dict]:
        outputs = dict(args.inputs)

        # Synthesize computed outputs that the real provider would generate
        if args.typ == "aws:ec2/instance:Instance":
            outputs.setdefault("publicIp", "203.0.113.1")
            outputs.setdefault("privateIp", "10.0.1.100")

        if args.typ == "aws:s3/bucketV2:BucketV2":
            bucket_name = outputs.get("bucket", "test-bucket")
            outputs.setdefault("arn", f"arn:aws:s3:::{bucket_name}")

        return [args.name + "_id", outputs]

    def call(self, args: pulumi.runtime.MockCallArgs) -> dict:
        if args.token == "aws:ec2/getAmi:getAmi":
            return {
                "architecture": "x86_64",
                "id": "ami-0123456789abcdef0",
                "name": "al2023-ami-2023.6.20241212.0-kernel-6.1-x86_64",
            }
        return {}


pulumi.runtime.set_mocks(
    FedCostMocks(),
    preview=False,
    project="fedcost",
    stack="test",
)

# Add infra/ to Python path so tests can import infrastructure modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))
