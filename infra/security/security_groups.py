"""Security groups for FedCost EC2 instances."""

import pulumi_aws as aws

from network.vpcs import VpcResources

FLOWER_PORT = 8080
HOSPITAL_NAMES = ["hospital-a", "hospital-b", "hospital-c"]


def create_security_groups(
    vpcs: dict[str, VpcResources],
) -> dict[str, aws.ec2.SecurityGroup]:
    """Create security groups for each instance role.

    Rules:
    - FL server: accepts Flower gRPC (8080) from hospital VPC CIDRs only
    - Hospitals: allow outbound to FL server on 8080
    - Centralized: no special inbound (SSH via Tailscale only)
    - All: allow all outbound (for S3, apt, pip, Tailscale)

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

    # FL server SG — accepts Flower gRPC from hospitals
    hospital_ingress_rules = [
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=FLOWER_PORT,
            to_port=FLOWER_PORT,
            cidr_blocks=[vpcs[h].vpc.cidr_block],
            description=f"Flower gRPC from {h}",
        )
        for h in HOSPITAL_NAMES
    ]

    sgs["fl-server"] = aws.ec2.SecurityGroup(
        "sg-fl-server",
        vpc_id=vpcs["fl-server"].vpc.id,
        description="FL server — Flower gRPC from hospitals",
        ingress=hospital_ingress_rules,
        egress=[
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1",
                from_port=0,
                to_port=0,
                cidr_blocks=["0.0.0.0/0"],
                description="Allow all outbound",
            ),
        ],
        tags={"Name": "fedcost-sg-fl-server", "Project": "fedcost"},
    )

    # Hospital SGs — allow all outbound (Flower client initiates to server)
    for hospital_name in HOSPITAL_NAMES:
        sgs[hospital_name] = aws.ec2.SecurityGroup(
            f"sg-{hospital_name}",
            vpc_id=vpcs[hospital_name].vpc.id,
            description=f"{hospital_name} — Flower client",
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
                "Name": f"fedcost-sg-{hospital_name}",
                "Project": "fedcost",
            },
        )

    # Centralized SG — outbound only (SSH via Tailscale)
    sgs["centralized"] = aws.ec2.SecurityGroup(
        "sg-centralized",
        vpc_id=vpcs["centralized"].vpc.id,
        description="Centralized baselines — outbound only",
        egress=[
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1",
                from_port=0,
                to_port=0,
                cidr_blocks=["0.0.0.0/0"],
                description="Allow all outbound",
            ),
        ],
        tags={"Name": "fedcost-sg-centralized", "Project": "fedcost"},
    )

    return sgs
