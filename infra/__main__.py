"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config

config = load_config()

pulumi.export("region", config.region)
