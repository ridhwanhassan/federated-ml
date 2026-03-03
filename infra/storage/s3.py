"""S3 bucket for MIMIC-IV experiment data."""

import pulumi
import pulumi_aws as aws


def create_data_bucket() -> aws.s3.BucketV2:
    """Create the fedcost data bucket.

    Structure:
        raw/                  - Full MIMIC-IV CSV (uploaded manually)
        partitions/           - Per-hospital CSVs
        results/              - Experiment output uploads
    """
    stack = pulumi.get_stack()

    bucket = aws.s3.BucketV2(
        "fedcost-data",
        bucket=f"fedcost-data-{stack}",
        tags={"Project": "fedcost", "Stack": stack},
    )

    aws.s3.BucketPublicAccessBlock(
        "fedcost-data-public-access-block",
        bucket=bucket.id,
        block_public_acls=True,
        block_public_policy=True,
        ignore_public_acls=True,
        restrict_public_buckets=True,
    )

    return bucket
