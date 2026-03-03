"""EC2 instance definitions for FedCost infrastructure."""

from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_aws as aws

from network.vpcs import VpcResources
from security.iam import IamResources

USER_DATA_DIR = Path(__file__).parent / "user_data"

INSTANCE_DEFINITIONS = [
    {
        "name": "fl-server",
        "vpc_name": "fl-server",
        "iam_key": "fl-server",
        "user_data_template": "fl_server.sh",
        "extra_env": {},
    },
    {
        "name": "hospital-a",
        "vpc_name": "hospital-a",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-a"},
    },
    {
        "name": "hospital-b",
        "vpc_name": "hospital-b",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-b"},
    },
    {
        "name": "hospital-c",
        "vpc_name": "hospital-c",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-c"},
    },
    {
        "name": "centralized",
        "vpc_name": "centralized",
        "iam_key": "centralized",
        "user_data_template": "centralized.sh",
        "extra_env": {},
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
    tailscale_auth_key: pulumi.Output | str,
    extra_env: dict[str, str],
) -> pulumi.Output:
    """Read a user-data script and inject environment variables."""
    script = (USER_DATA_DIR / template_name).read_text()

    def _inject(ts_key: str) -> str:
        result = script.replace("${TAILSCALE_AUTH_KEY}", ts_key)
        for k, v in extra_env.items():
            result = result.replace(f"${{{k}}}", v)
        return result

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
            tailscale_auth_key,
            defn["extra_env"],
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
