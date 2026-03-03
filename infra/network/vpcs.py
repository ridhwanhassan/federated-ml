"""VPC definitions for FedCost infrastructure."""

from dataclasses import dataclass

import pulumi_aws as aws


@dataclass
class VpcResources:
    """All resources created for a single VPC."""

    vpc: aws.ec2.Vpc
    subnet: aws.ec2.Subnet
    igw: aws.ec2.InternetGateway
    route_table: aws.ec2.RouteTable


VPC_DEFINITIONS: list[dict] = [
    {"name": "fl-server", "cidr": "10.0.0.0/16", "subnet_cidr": "10.0.1.0/24"},
    {"name": "hospital-a", "cidr": "10.1.0.0/16", "subnet_cidr": "10.1.1.0/24"},
    {"name": "hospital-b", "cidr": "10.2.0.0/16", "subnet_cidr": "10.2.1.0/24"},
    {"name": "hospital-c", "cidr": "10.3.0.0/16", "subnet_cidr": "10.3.1.0/24"},
    {"name": "centralized", "cidr": "10.4.0.0/16", "subnet_cidr": "10.4.1.0/24"},
]


def create_vpc(name: str, cidr: str, subnet_cidr: str) -> VpcResources:
    """Create a VPC with one public subnet, IGW, and route table.

    Parameters
    ----------
    name : str
        Logical name (e.g., "hospital-a").
    cidr : str
        VPC CIDR block (e.g., "10.1.0.0/16").
    subnet_cidr : str
        Subnet CIDR within the VPC (e.g., "10.1.1.0/24").
    """
    vpc = aws.ec2.Vpc(
        f"vpc-{name}",
        cidr_block=cidr,
        enable_dns_support=True,
        enable_dns_hostnames=True,
        tags={"Name": f"fedcost-{name}", "Project": "fedcost"},
    )

    igw = aws.ec2.InternetGateway(
        f"igw-{name}",
        vpc_id=vpc.id,
        tags={"Name": f"fedcost-{name}-igw", "Project": "fedcost"},
    )

    route_table = aws.ec2.RouteTable(
        f"rt-{name}",
        vpc_id=vpc.id,
        routes=[
            aws.ec2.RouteTableRouteArgs(
                cidr_block="0.0.0.0/0",
                gateway_id=igw.id,
            ),
        ],
        tags={"Name": f"fedcost-{name}-rt", "Project": "fedcost"},
    )

    subnet = aws.ec2.Subnet(
        f"subnet-{name}",
        vpc_id=vpc.id,
        cidr_block=subnet_cidr,
        map_public_ip_on_launch=True,
        tags={"Name": f"fedcost-{name}-public", "Project": "fedcost"},
    )

    aws.ec2.RouteTableAssociation(
        f"rta-{name}",
        subnet_id=subnet.id,
        route_table_id=route_table.id,
    )

    return VpcResources(vpc=vpc, subnet=subnet, igw=igw, route_table=route_table)


def create_all_vpcs() -> dict[str, VpcResources]:
    """Create all 5 FedCost VPCs.

    Returns
    -------
    dict[str, VpcResources]
        Mapping of VPC name to its resources.
    """
    return {defn["name"]: create_vpc(**defn) for defn in VPC_DEFINITIONS}
