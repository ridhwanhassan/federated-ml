"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi
import pulumi_aws as aws

from compute import create_instances, create_key_pair
from config import load_config
from network import create_all_vpcs
from security import create_iam_resources, create_security_groups
from ssm import create_ssm_parameters
from storage import create_data_bucket

# Show deployment target
caller = aws.get_caller_identity()
print(f"Deploying to AWS account {caller.account_id} (region: {aws.get_region().name})")

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

# 2. Network (Tailscale handles inter-VPC connectivity — no peering needed)
vpcs = create_all_vpcs()

# 3. Security
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
sgs = create_security_groups(vpcs)

# 4. Compute
key_pair = create_key_pair(config.ssh_public_key)
instances = create_instances(
    vpcs=vpcs,
    sgs=sgs,
    iam=iam,
    key_pair=key_pair,
    tailscale_auth_key=config.tailscale_auth_key,
    instance_types={
        "fl-server": config.instance_type_fl_server,
        "hospital-1": config.instance_type_hospital,
        "hospital-2": config.instance_type_hospital,
        "hospital-3": config.instance_type_hospital,
        "hospital-4": config.instance_type_hospital,
        "hospital-5": config.instance_type_hospital,
        "centralized": config.instance_type_centralized,
    },
)

# 5. SSM Parameters
hospital_ips = {
    name: instances[name].private_ip
    for name in [
        "hospital-1",
        "hospital-2",
        "hospital-3",
        "hospital-4",
        "hospital-5",
    ]
}

ssm_params = create_ssm_parameters(
    fl_server_private_ip=instances["fl-server"].private_ip,
    data_bucket_name=data_bucket.bucket,
    hospital_private_ips=hospital_ips,
)

# --- Exports ---
pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)

for name, vpc_res in vpcs.items():
    pulumi.export(f"vpc_{name}_id", vpc_res.vpc.id)

for name, inst in instances.items():
    pulumi.export(f"ec2_{name}_id", inst.instance.id)
    pulumi.export(f"ec2_{name}_private_ip", inst.private_ip)
    pulumi.export(f"ec2_{name}_public_ip", inst.public_ip)
