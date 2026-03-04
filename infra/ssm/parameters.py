"""SSM Parameter Store entries for FedCost runtime config."""

import pulumi
import pulumi_aws as aws

HOSPITAL_NAMES = [
    "hospital-1",
    "hospital-2",
    "hospital-3",
    "hospital-4",
    "hospital-5",
]


def create_ssm_parameters(
    fl_server_private_ip: pulumi.Output,
    data_bucket_name: pulumi.Output,
    hospital_private_ips: dict[str, pulumi.Output],
) -> list[aws.ssm.Parameter]:
    """Create SSM parameters for runtime discovery.

    Parameters
    ----------
    fl_server_private_ip : Output
        Private IP of the FL server instance (informational — Flower
        clients connect via Tailscale hostname instead).
    data_bucket_name : Output
        Name of the S3 data bucket.
    hospital_private_ips : dict[str, Output]
        Private IPs of hospital instances, keyed by name (e.g. "hospital-1").
        Used by D-PSGD ring topology for peer discovery.
    """
    params = []

    params.append(
        aws.ssm.Parameter(
            "ssm-flower-server-ip",
            name="/fedcost/flower-server-ip",
            type=aws.ssm.ParameterType.STRING,
            value=fl_server_private_ip,
            description="Private IP of the Flower aggregation server",
            tags={"Project": "fedcost"},
        )
    )

    params.append(
        aws.ssm.Parameter(
            "ssm-s3-data-bucket",
            name="/fedcost/s3-data-bucket",
            type=aws.ssm.ParameterType.STRING,
            value=data_bucket_name,
            description="S3 bucket name for experiment data",
            tags={"Project": "fedcost"},
        )
    )

    for name in HOSPITAL_NAMES:
        params.append(
            aws.ssm.Parameter(
                f"ssm-{name}-ip",
                name=f"/fedcost/{name}-ip",
                type=aws.ssm.ParameterType.STRING,
                value=hospital_private_ips[name],
                description=f"Private IP of {name} (for gossip topology)",
                tags={"Project": "fedcost"},
            )
        )

    return params
