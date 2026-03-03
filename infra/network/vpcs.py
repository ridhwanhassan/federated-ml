"""VPC definitions for FedCost infrastructure.

Hospital VPCs use private subnets (with NAT Gateway for outbound).
FL server and centralized VPCs use public subnets.
"""

from dataclasses import dataclass, field

import pulumi_aws as aws


@dataclass
class VpcResources:
    """All resources created for a single VPC."""

    vpc: aws.ec2.Vpc
    subnet: aws.ec2.Subnet  # The subnet where EC2 instances are placed
    igw: aws.ec2.InternetGateway
    route_table: aws.ec2.RouteTable
    is_private: bool = False
    # NAT resources (only for private subnets)
    nat_eip: aws.ec2.Eip | None = None
    nat_gateway: aws.ec2.NatGateway | None = None
    public_subnet: aws.ec2.Subnet | None = None


VPC_DEFINITIONS: list[dict] = [
    # Public subnet — EC2 gets public IP, direct IGW route
    {"name": "fl-server", "cidr": "10.0.0.0/16", "subnet_cidr": "10.0.1.0/24", "private": False},
    # Private subnets — EC2 has no public IP, outbound via NAT Gateway
    {"name": "hospital-a", "cidr": "10.1.0.0/16", "subnet_cidr": "10.1.1.0/24", "private": True},
    {"name": "hospital-b", "cidr": "10.2.0.0/16", "subnet_cidr": "10.2.1.0/24", "private": True},
    {"name": "hospital-c", "cidr": "10.3.0.0/16", "subnet_cidr": "10.3.1.0/24", "private": True},
]


def _create_public_vpc(name: str, cidr: str, subnet_cidr: str) -> VpcResources:
    """Create a VPC with one public subnet (IGW route, public IP)."""
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


def _create_private_vpc(name: str, cidr: str, subnet_cidr: str) -> VpcResources:
    """Create a VPC with a private subnet for EC2 and a NAT Gateway for outbound.

    Layout:
        public subnet  (.2.0/24) — holds NAT Gateway only
        private subnet (.1.0/24) — holds EC2 instance, routes outbound via NAT
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

    # Public subnet (for NAT Gateway)
    public_cidr = subnet_cidr.replace(".1.0/24", ".2.0/24")
    public_subnet = aws.ec2.Subnet(
        f"subnet-{name}-public",
        vpc_id=vpc.id,
        cidr_block=public_cidr,
        map_public_ip_on_launch=False,
        tags={"Name": f"fedcost-{name}-public", "Project": "fedcost"},
    )

    public_rt = aws.ec2.RouteTable(
        f"rt-{name}-public",
        vpc_id=vpc.id,
        routes=[
            aws.ec2.RouteTableRouteArgs(
                cidr_block="0.0.0.0/0",
                gateway_id=igw.id,
            ),
        ],
        tags={"Name": f"fedcost-{name}-public-rt", "Project": "fedcost"},
    )

    aws.ec2.RouteTableAssociation(
        f"rta-{name}-public",
        subnet_id=public_subnet.id,
        route_table_id=public_rt.id,
    )

    # NAT Gateway (in public subnet)
    nat_eip = aws.ec2.Eip(
        f"eip-{name}-nat",
        domain="vpc",
        tags={"Name": f"fedcost-{name}-nat", "Project": "fedcost"},
    )

    nat_gw = aws.ec2.NatGateway(
        f"nat-{name}",
        subnet_id=public_subnet.id,
        allocation_id=nat_eip.id,
        tags={"Name": f"fedcost-{name}-nat", "Project": "fedcost"},
    )

    # Private subnet (for EC2)
    private_subnet = aws.ec2.Subnet(
        f"subnet-{name}-private",
        vpc_id=vpc.id,
        cidr_block=subnet_cidr,
        map_public_ip_on_launch=False,
        tags={"Name": f"fedcost-{name}-private", "Project": "fedcost"},
    )

    private_rt = aws.ec2.RouteTable(
        f"rt-{name}-private",
        vpc_id=vpc.id,
        routes=[
            aws.ec2.RouteTableRouteArgs(
                cidr_block="0.0.0.0/0",
                nat_gateway_id=nat_gw.id,
            ),
        ],
        tags={"Name": f"fedcost-{name}-private-rt", "Project": "fedcost"},
    )

    aws.ec2.RouteTableAssociation(
        f"rta-{name}-private",
        subnet_id=private_subnet.id,
        route_table_id=private_rt.id,
    )

    return VpcResources(
        vpc=vpc,
        subnet=private_subnet,  # EC2 goes here
        igw=igw,
        route_table=private_rt,
        is_private=True,
        nat_eip=nat_eip,
        nat_gateway=nat_gw,
        public_subnet=public_subnet,
    )


def create_all_vpcs() -> dict[str, VpcResources]:
    """Create all 4 FedCost VPCs.

    - fl-server: public subnet (direct internet, also hosts centralized instance)
    - hospital-a/b/c: private subnets (outbound via NAT Gateway)

    Returns
    -------
    dict[str, VpcResources]
        Mapping of VPC name to its resources.
    """
    result = {}
    for defn in VPC_DEFINITIONS:
        name = defn["name"]
        cidr = defn["cidr"]
        subnet_cidr = defn["subnet_cidr"]
        if defn["private"]:
            result[name] = _create_private_vpc(name, cidr, subnet_cidr)
        else:
            result[name] = _create_public_vpc(name, cidr, subnet_cidr)
    return result
