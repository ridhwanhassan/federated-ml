"""SSM Parameter Store entries for FedCost runtime config."""

import pulumi
import pulumi_aws as aws


def create_ssm_parameters(
    fl_server_private_ip: pulumi.Output,
    data_bucket_name: pulumi.Output,
) -> list[aws.ssm.Parameter]:
    """Create SSM parameters for runtime discovery.

    Parameters
    ----------
    fl_server_private_ip : Output
        Private IP of the FL server instance (informational — Flower
        clients connect via Tailscale hostname instead).
    data_bucket_name : Output
        Name of the S3 data bucket.
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

    return params
