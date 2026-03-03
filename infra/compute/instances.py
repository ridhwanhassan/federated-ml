"""EC2 instance definitions for FedCost infrastructure."""

from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_aws as aws

from network.vpcs import VpcResources
from security.iam import IamResources

USER_DATA_DIR = Path(__file__).parent / "user_data"
BASE_SCRIPT = (USER_DATA_DIR / "base.sh").read_text()

INSTANCE_DEFINITIONS = [
    {
        "name": "fl-server",
        "vpc_name": "fl-server",
        "iam_key": "fl-server",
        "user_data_template": "fl_server.sh",
    },
    {
        "name": "hospital-a",
        "vpc_name": "hospital-a",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
    },
    {
        "name": "hospital-b",
        "vpc_name": "hospital-b",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
    },
    {
        "name": "hospital-c",
        "vpc_name": "hospital-c",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
    },
    {
        "name": "centralized",
        "vpc_name": "fl-server",
        "iam_key": "centralized",
        "user_data_template": "centralized.sh",
    },
]


@dataclass
class InstanceResult:
    """Result of creating an EC2 instance."""

    instance: aws.ec2.Instance
    private_ip: pulumi.Output
    public_ip: pulumi.Output


def _get_user_data(
    template_name: str,
    node_name: str,
    tailscale_auth_key: pulumi.Output | str,
) -> pulumi.Output:
    """Concatenate base.sh + role script and inject variables.

    Variables replaced: ${TAILSCALE_AUTH_KEY}, ${NODE_NAME}.
    """
    role_script = (USER_DATA_DIR / template_name).read_text()
    combined = BASE_SCRIPT + "\n" + role_script

    def _inject(ts_key: str) -> str:
        return combined.replace(
            "${TAILSCALE_AUTH_KEY}", ts_key
        ).replace(
            "${NODE_NAME}", node_name
        )

    if isinstance(tailscale_auth_key, str):
        return pulumi.Output.from_input(_inject(tailscale_auth_key))
    return tailscale_auth_key.apply(_inject)


def _get_ami() -> str:
    """Get the latest Amazon Linux 2023 AMI."""
    ami = aws.ec2.get_ami(
        most_recent=True,
        owners=["amazon"],
        filters=[
            aws.ec2.GetAmiFilterArgs(
                name="name",
                values=["al2023-ami-2023.*-x86_64"],
            ),
            aws.ec2.GetAmiFilterArgs(
                name="state",
                values=["available"],
            ),
        ],
    )
    return ami.id


def create_instances(
    vpcs: dict[str, VpcResources],
    sgs: dict[str, aws.ec2.SecurityGroup],
    iam: dict[str, IamResources],
    key_pair: aws.ec2.KeyPair,
    tailscale_auth_key: pulumi.Output | str,
    instance_types: dict[str, str],
) -> dict[str, InstanceResult]:
    """Create all 5 FedCost EC2 instances.

    Parameters
    ----------
    vpcs : dict
        VPC resources keyed by name.
    sgs : dict
        Security groups keyed by VPC name.
    iam : dict
        IAM resources keyed by role ("hospital", "fl-server", "centralized").
    key_pair : aws.ec2.KeyPair
        SSH key pair.
    tailscale_auth_key : str or Output
        Tailscale pre-auth key.
    instance_types : dict
        Instance type overrides keyed by name.
    """
    ami_id = _get_ami()
    results: dict[str, InstanceResult] = {}

    for defn in INSTANCE_DEFINITIONS:
        name = defn["name"]
        vpc_name = defn["vpc_name"]

        instance_type = instance_types.get(name, "t3.medium")

        user_data = _get_user_data(
            defn["user_data_template"],
            node_name=name,
            tailscale_auth_key=tailscale_auth_key,
        )

        instance = aws.ec2.Instance(
            f"ec2-{name}",
            ami=ami_id,
            instance_type=instance_type,
            subnet_id=vpcs[vpc_name].subnet.id,
            vpc_security_group_ids=[sgs[vpc_name].id],
            iam_instance_profile=iam[defn["iam_key"]].instance_profile.name,
            key_name=key_pair.key_name,
            user_data=user_data,
            root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
                volume_size=30,
                volume_type="gp3",
            ),
            tags={"Name": f"fedcost-{name}", "Project": "fedcost"},
        )

        results[name] = InstanceResult(
            instance=instance,
            private_ip=instance.private_ip,
            public_ip=instance.public_ip,
        )

    return results
