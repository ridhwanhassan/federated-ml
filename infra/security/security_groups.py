"""Security groups for FedCost EC2 instances.

All inter-instance communication (Flower gRPC, SSH) goes through Tailscale.
Security groups only need to allow outbound traffic for S3, apt, pip, and
Tailscale coordination. No VPC peering or cross-VPC ingress rules needed.
"""

import pulumi_aws as aws

from network.vpcs import VpcResources

ALL_VPC_NAMES = ["fl-server", "hospital-a", "hospital-b", "hospital-c"]


def create_security_groups(
    vpcs: dict[str, VpcResources],
) -> dict[str, aws.ec2.SecurityGroup]:
    """Create security groups for each VPC.

    All instances use outbound-only SGs. Tailscale handles connectivity
    (Flower gRPC, SSH) via encrypted WireGuard tunnels over the public
    internet, so no cross-VPC ingress rules are required.

    Parameters
    ----------
    vpcs : dict[str, VpcResources]
        All VPC resources keyed by name.

    Returns
    -------
    dict[str, aws.ec2.SecurityGroup]
        Security groups keyed by VPC name.
    """
    sgs: dict[str, aws.ec2.SecurityGroup] = {}

    for name in ALL_VPC_NAMES:
        sgs[name] = aws.ec2.SecurityGroup(
            f"secgrp-{name}",
            vpc_id=vpcs[name].vpc.id,
            description=f"fedcost-{name} - outbound only, Tailscale for ingress",
            egress=[
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["0.0.0.0/0"],
                    description="Allow all outbound",
                ),
            ],
            tags={
                "Name": f"fedcost-sg-{name}",
                "Project": "fedcost",
            },
        )

    return sgs
