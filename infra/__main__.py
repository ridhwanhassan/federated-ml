"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)
