"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config
from network import create_all_vpcs
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

# 2. Network
vpcs = create_all_vpcs()

pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)
for name, vpc_res in vpcs.items():
    pulumi.export(f"vpc_{name}_id", vpc_res.vpc.id)
