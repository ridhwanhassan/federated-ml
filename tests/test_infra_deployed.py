"""Post-deploy smoke tests for FedCost AWS infrastructure.

Verifies live AWS resources after `pulumi up` completes.
Requires AWS credentials and a deployed stack.

Run: pytest tests/test_infra_deployed.py -v --run-deployed

These tests are skipped by default (no --run-deployed flag).
The skip logic lives in conftest.py via the "deployed" marker.
"""

import json
import subprocess

import boto3
import pytest

# ── Fixtures ─────────────────────────────────────────────────────────

REGION = "ap-southeast-1"


def _get_stack_outputs() -> dict:
    """Fetch Pulumi stack outputs via CLI."""
    result = subprocess.run(
        ["pulumi", "stack", "output", "--json"],
        capture_output=True,
        text=True,
        cwd="infra",
        check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def stack_outputs():
    return _get_stack_outputs()


@pytest.fixture(scope="module")
def ec2_client():
    return boto3.client("ec2", region_name=REGION)


@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3", region_name=REGION)


@pytest.fixture(scope="module")
def ssm_client():
    return boto3.client("ssm", region_name=REGION)


@pytest.fixture(scope="module")
def iam_client():
    return boto3.client("iam", region_name=REGION)


# ── VPC Tests ────────────────────────────────────────────────────────

EXPECTED_VPC_NAMES = [
    "fl-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
]

EXPECTED_INSTANCE_NAMES = [
    "fl-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
    "centralized",
]

EXPECTED_CIDRS = {
    "fl-server": "10.0.0.0/16",
    "hospital-a": "10.1.0.0/16",
    "hospital-b": "10.2.0.0/16",
    "hospital-c": "10.3.0.0/16",
}


@pytest.mark.deployed
class TestVPCsDeployed:
    def test_four_vpcs_exist(self, ec2_client, stack_outputs):
        vpc_ids = [stack_outputs[f"vpc_{n}_id"] for n in EXPECTED_VPC_NAMES]
        resp = ec2_client.describe_vpcs(VpcIds=vpc_ids)
        assert len(resp["Vpcs"]) == 4

    def test_vpc_cidr_blocks(self, ec2_client, stack_outputs):
        for name, expected_cidr in EXPECTED_CIDRS.items():
            vpc_id = stack_outputs[f"vpc_{name}_id"]
            resp = ec2_client.describe_vpcs(VpcIds=[vpc_id])
            actual_cidr = resp["Vpcs"][0]["CidrBlock"]
            assert actual_cidr == expected_cidr, (
                f"VPC {name}: expected {expected_cidr}, got {actual_cidr}"
            )

    def test_vpcs_have_dns_support(self, ec2_client, stack_outputs):
        for name in EXPECTED_VPC_NAMES:
            vpc_id = stack_outputs[f"vpc_{name}_id"]
            attr = ec2_client.describe_vpc_attribute(
                VpcId=vpc_id, Attribute="enableDnsSupport"
            )
            assert attr["EnableDnsSupport"]["Value"] is True, (
                f"VPC {name}: DNS support not enabled"
            )


# ── EC2 Tests ────────────────────────────────────────────────────────

EXPECTED_INSTANCE_TYPES = {
    "fl-server": "t3.medium",
    "hospital-a": "t3.medium",
    "hospital-b": "t3.medium",
    "hospital-c": "t3.medium",
    "centralized": "t3.large",
}


@pytest.mark.deployed
class TestEC2Deployed:
    def test_five_instances_running(self, ec2_client, stack_outputs):
        instance_ids = [
            stack_outputs[f"ec2_{n}_id"] for n in EXPECTED_INSTANCE_NAMES
        ]
        resp = ec2_client.describe_instances(InstanceIds=instance_ids)
        states = [
            inst["State"]["Name"]
            for res in resp["Reservations"]
            for inst in res["Instances"]
        ]
        assert all(s == "running" for s in states), (
            f"Not all instances running: {states}"
        )

    def test_instance_types(self, ec2_client, stack_outputs):
        for name, expected_type in EXPECTED_INSTANCE_TYPES.items():
            instance_id = stack_outputs[f"ec2_{name}_id"]
            resp = ec2_client.describe_instances(InstanceIds=[instance_id])
            actual_type = resp["Reservations"][0]["Instances"][0]["InstanceType"]
            assert actual_type == expected_type, (
                f"EC2 {name}: expected {expected_type}, got {actual_type}"
            )

    def test_instances_have_iam_profiles(self, ec2_client, stack_outputs):
        for name in EXPECTED_INSTANCE_NAMES:
            instance_id = stack_outputs[f"ec2_{name}_id"]
            resp = ec2_client.describe_instances(InstanceIds=[instance_id])
            inst = resp["Reservations"][0]["Instances"][0]
            assert "IamInstanceProfile" in inst, (
                f"EC2 {name}: no IAM instance profile attached"
            )

    def test_instances_have_public_ips(self, ec2_client, stack_outputs):
        for name in EXPECTED_INSTANCE_NAMES:
            public_ip = stack_outputs.get(f"ec2_{name}_public_ip")
            assert public_ip, f"EC2 {name}: no public IP in stack outputs"

    def test_root_volumes_are_gp3(self, ec2_client, stack_outputs):
        for name in EXPECTED_INSTANCE_NAMES:
            instance_id = stack_outputs[f"ec2_{name}_id"]
            resp = ec2_client.describe_instances(InstanceIds=[instance_id])
            bdm = resp["Reservations"][0]["Instances"][0]["BlockDeviceMappings"]
            root_vol_id = bdm[0]["Ebs"]["VolumeId"]

            vol_resp = ec2_client.describe_volumes(VolumeIds=[root_vol_id])
            vol = vol_resp["Volumes"][0]
            assert vol["VolumeType"] == "gp3", (
                f"EC2 {name}: root volume is {vol['VolumeType']}, expected gp3"
            )
            assert vol["Size"] == 30, (
                f"EC2 {name}: root volume is {vol['Size']}GB, expected 30GB"
            )


# ── S3 Tests ─────────────────────────────────────────────────────────


@pytest.mark.deployed
class TestS3Deployed:
    def test_data_bucket_exists(self, s3_client, stack_outputs):
        bucket_name = stack_outputs["data_bucket_name"]
        resp = s3_client.head_bucket(Bucket=bucket_name)
        assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_public_access_blocked(self, s3_client, stack_outputs):
        bucket_name = stack_outputs["data_bucket_name"]
        resp = s3_client.get_public_access_block(Bucket=bucket_name)
        config = resp["PublicAccessBlockConfiguration"]
        assert config["BlockPublicAcls"] is True
        assert config["BlockPublicPolicy"] is True
        assert config["IgnorePublicAcls"] is True
        assert config["RestrictPublicBuckets"] is True


# ── SSM Tests ────────────────────────────────────────────────────────


@pytest.mark.deployed
class TestSSMDeployed:
    def test_flower_server_ip_parameter(self, ssm_client, stack_outputs):
        resp = ssm_client.get_parameter(Name="/fedcost/flower-server-ip")
        value = resp["Parameter"]["Value"]
        expected = stack_outputs["ec2_fl-server_private_ip"]
        assert value == expected, (
            f"SSM flower-server-ip: expected {expected}, got {value}"
        )

    def test_s3_data_bucket_parameter(self, ssm_client, stack_outputs):
        resp = ssm_client.get_parameter(Name="/fedcost/s3-data-bucket")
        value = resp["Parameter"]["Value"]
        expected = stack_outputs["data_bucket_name"]
        assert value == expected, (
            f"SSM s3-data-bucket: expected {expected}, got {value}"
        )

    def test_parameters_are_string_type(self, ssm_client):
        for path in ["/fedcost/flower-server-ip", "/fedcost/s3-data-bucket"]:
            resp = ssm_client.get_parameter(Name=path)
            assert resp["Parameter"]["Type"] == "String", (
                f"SSM {path}: type is {resp['Parameter']['Type']}, expected String"
            )


# ── Security Group Tests ─────────────────────────────────────────────


@pytest.mark.deployed
class TestSecurityGroupsDeployed:
    def test_sgs_exist_per_vpc(self, ec2_client, stack_outputs):
        for name in EXPECTED_VPC_NAMES:
            vpc_id = stack_outputs[f"vpc_{name}_id"]
            resp = ec2_client.describe_security_groups(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "tag:Project", "Values": ["fedcost"]},
                ]
            )
            assert len(resp["SecurityGroups"]) >= 1, (
                f"VPC {name}: no fedcost security group found"
            )

    def test_sgs_allow_all_outbound(self, ec2_client, stack_outputs):
        for name in EXPECTED_VPC_NAMES:
            vpc_id = stack_outputs[f"vpc_{name}_id"]
            resp = ec2_client.describe_security_groups(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "tag:Project", "Values": ["fedcost"]},
                ]
            )
            sg = resp["SecurityGroups"][0]
            egress = sg["IpPermissionsEgress"]
            # At least one rule allowing all traffic to 0.0.0.0/0
            all_outbound = any(
                perm.get("IpProtocol") == "-1"
                and any(
                    r["CidrIp"] == "0.0.0.0/0"
                    for r in perm.get("IpRanges", [])
                )
                for perm in egress
            )
            assert all_outbound, f"SG in VPC {name}: no all-outbound rule"
