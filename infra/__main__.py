"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config
from network import create_all_vpcs, create_peering
from security import create_iam_resources, create_security_groups
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

# 2. Network
vpcs = create_all_vpcs()
peerings = create_peering(vpcs)

# 3. Security
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
sgs = create_security_groups(vpcs)

pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)
for name, vpc_res in vpcs.items():
    pulumi.export(f"vpc_{name}_id", vpc_res.vpc.id)
