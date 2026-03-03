"""VPC peering between FL server and hospital VPCs."""

import pulumi_aws as aws

from network.vpcs import VpcResources

HOSPITAL_NAMES = ["hospital-a", "hospital-b", "hospital-c"]


def create_peering(
    vpcs: dict[str, VpcResources],
) -> list[aws.ec2.VpcPeeringConnection]:
    """Create VPC peering from fl-server to each hospital VPC.

    Also adds route table entries so traffic flows in both directions.
    Hospitals cannot peer with each other — only with the FL server.

    Parameters
    ----------
    vpcs : dict[str, VpcResources]
        All VPC resources keyed by name.
    """
    server_vpc = vpcs["fl-server"]
    peerings = []

    for hospital_name in HOSPITAL_NAMES:
        hospital_vpc = vpcs[hospital_name]

        peering = aws.ec2.VpcPeeringConnection(
            f"peer-server-{hospital_name}",
            vpc_id=server_vpc.vpc.id,
            peer_vpc_id=hospital_vpc.vpc.id,
            auto_accept=True,
            tags={
                "Name": f"fedcost-server-{hospital_name}",
                "Project": "fedcost",
            },
        )

        # Route from FL server -> hospital CIDR
        aws.ec2.Route(
            f"route-server-to-{hospital_name}",
            route_table_id=server_vpc.route_table.id,
            destination_cidr_block=hospital_vpc.vpc.cidr_block,
            vpc_peering_connection_id=peering.id,
        )

        # Route from hospital -> FL server CIDR
        aws.ec2.Route(
            f"route-{hospital_name}-to-server",
            route_table_id=hospital_vpc.route_table.id,
            destination_cidr_block=server_vpc.vpc.cidr_block,
            vpc_peering_connection_id=peering.id,
        )

        peerings.append(peering)

    return peerings
