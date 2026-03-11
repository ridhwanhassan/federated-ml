"""IAM roles and instance profiles for FedCost EC2 instances."""

import json
from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

EC2_ASSUME_ROLE_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": "sts:AssumeRole",
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
            }
        ],
    }
)


@dataclass
class IamResources:
    """IAM resources for a role."""

    role: aws.iam.Role
    instance_profile: aws.iam.InstanceProfile


def _create_role_and_profile(
    name: str,
    policy_document: pulumi.Output | str,
) -> IamResources:
    role = aws.iam.Role(
        f"role-{name}",
        assume_role_policy=EC2_ASSUME_ROLE_POLICY,
        tags={"Name": f"fedcost-{name}", "Project": "fedcost"},
    )

    aws.iam.RolePolicy(
        f"policy-{name}",
        role=role.id,
        policy=policy_document,
    )

    # SSM managed policy for Session Manager access (optional debugging)
    aws.iam.RolePolicyAttachment(
        f"ssm-managed-{name}",
        role=role.name,
        policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )

    profile = aws.iam.InstanceProfile(
        f"profile-{name}",
        role=role.name,
        tags={"Name": f"fedcost-{name}", "Project": "fedcost"},
    )

    return IamResources(role=role, instance_profile=profile)


def create_iam_resources(
    data_bucket_arn: pulumi.Output,
) -> dict[str, IamResources]:
    """Create IAM roles and instance profiles for all instance types.

    Parameters
    ----------
    data_bucket_arn : pulumi.Output
        ARN of the S3 data bucket.

    Returns
    -------
    dict[str, IamResources]
        Mapping of role name to IAM resources.
        Keys: "hospital", "fl-server", "centralized".
    """
    # Hospital: read/write S3 (write needed for model upload) + read SSM
    hospital_policy = data_bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:PutObject",
                        ],
                        "Resource": [arn, f"{arn}/*"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ssm:GetParameter",
                            "ssm:GetParameters",
                        ],
                        "Resource": "*",
                    },
                ],
            }
        )
    )

    # FL server: read/write S3 + read/write SSM
    fl_server_policy = data_bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:PutObject",
                        ],
                        "Resource": [arn, f"{arn}/*"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ssm:GetParameter",
                            "ssm:GetParameters",
                            "ssm:PutParameter",
                        ],
                        "Resource": "*",
                    },
                ],
            }
        )
    )

    # Centralized: read/write S3
    centralized_policy = data_bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:PutObject",
                        ],
                        "Resource": [arn, f"{arn}/*"],
                    },
                ],
            }
        )
    )

    return {
        "hospital": _create_role_and_profile("hospital", hospital_policy),
        "fl-server": _create_role_and_profile("fl-server", fl_server_policy),
        "centralized": _create_role_and_profile(
            "centralized", centralized_policy
        ),
    }
