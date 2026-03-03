"""Pulumi mocked unit tests for FedCost AWS infrastructure.

Verifies resource counts, configuration, tagging, and properties
without requiring AWS credentials.

Run: pytest tests/test_infra_unit.py -v
"""

import json

import pulumi

from compute.instances import create_instances
from compute.key_pair import create_key_pair
from network.vpcs import create_all_vpcs
from security.iam import create_iam_resources
from security.security_groups import create_security_groups
from ssm.parameters import create_ssm_parameters
from storage.s3 import create_data_bucket


# ── Helpers ──────────────────────────────────────────────────────────


def assert_eq(actual, expected, label=""):
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


def _make_eq_check(expected, label):
    def check(actual):
        assert_eq(actual, expected, label)

    return check


def _make_tag_check(key, value, label):
    def check(tags):
        assert tags.get(key) == value, (
            f"{label}: tag {key}={tags.get(key)!r}, expected {value!r}"
        )

    return check


# ── Create the full resource graph once (mirrors infra/__main__.py) ──

data_bucket = create_data_bucket()
vpcs = create_all_vpcs()
sgs = create_security_groups(vpcs)
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
key_pair = create_key_pair("ssh-rsa AAAA test@test")

instances = create_instances(
    vpcs=vpcs,
    sgs=sgs,
    iam=iam,
    key_pair=key_pair,
    tailscale_auth_key="tskey-auth-test123",
    instance_types={
        "fl-server": "t3.medium",
        "hospital-a": "t3.medium",
        "hospital-b": "t3.medium",
        "hospital-c": "t3.medium",
        "centralized": "t3.large",
    },
)

ssm_params = create_ssm_parameters(
    fl_server_private_ip=instances["fl-server"].private_ip,
    data_bucket_name=data_bucket.bucket,
)


# ── Constants ────────────────────────────────────────────────────────

EXPECTED_VPCS = {
    "fl-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
}

EXPECTED_INSTANCES = {
    "fl-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
    "centralized",
}

EXPECTED_CIDRS = {
    "fl-server": "10.0.0.0/16",
    "hospital-a": "10.1.0.0/16",
    "hospital-b": "10.2.0.0/16",
    "hospital-c": "10.3.0.0/16",
}

EXPECTED_INSTANCE_TYPES = {
    "fl-server": "t3.medium",
    "hospital-a": "t3.medium",
    "hospital-b": "t3.medium",
    "hospital-c": "t3.medium",
    "centralized": "t3.large",
}


# ── Storage ──────────────────────────────────────────────────────────


class TestS3Bucket:
    @pulumi.runtime.test
    def test_bucket_name_includes_stack(self):
        return data_bucket.bucket.apply(
            _make_eq_check("fedcost-data-test", "bucket name")
        )

    @pulumi.runtime.test
    def test_bucket_tagged_with_project(self):
        return data_bucket.tags.apply(
            _make_tag_check("Project", "fedcost", "S3 bucket")
        )

    @pulumi.runtime.test
    def test_bucket_tagged_with_stack(self):
        return data_bucket.tags.apply(
            _make_tag_check("Stack", "test", "S3 bucket")
        )


# ── Network ──────────────────────────────────────────────────────────


class TestVPCs:
    def test_four_vpcs_created(self):
        assert len(vpcs) == 4

    def test_expected_vpc_names(self):
        assert set(vpcs.keys()) == EXPECTED_VPCS

    @pulumi.runtime.test
    def test_vpc_cidr_blocks(self):
        checks = []
        for name, expected in EXPECTED_CIDRS.items():
            checks.append(
                vpcs[name].vpc.cidr_block.apply(
                    _make_eq_check(expected, f"VPC {name} CIDR")
                )
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_dns_support_enabled(self):
        checks = []
        for name in EXPECTED_VPCS:
            checks.append(
                vpcs[name].vpc.enable_dns_support.apply(
                    _make_eq_check(True, f"VPC {name} DNS support")
                )
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_dns_hostnames_enabled(self):
        checks = []
        for name in EXPECTED_VPCS:
            checks.append(
                vpcs[name].vpc.enable_dns_hostnames.apply(
                    _make_eq_check(True, f"VPC {name} DNS hostnames")
                )
            )
        return pulumi.Output.all(*checks)

    def test_hospital_vpcs_are_private(self):
        for name in ("hospital-a", "hospital-b", "hospital-c"):
            assert vpcs[name].is_private is True, f"{name} should be private"
            assert vpcs[name].nat_gateway is not None, f"{name} missing NAT GW"

    def test_server_vpc_is_public(self):
        assert vpcs["fl-server"].is_private is False, "fl-server should be public"

    @pulumi.runtime.test
    def test_public_subnet_maps_public_ip(self):
        return vpcs["fl-server"].subnet.map_public_ip_on_launch.apply(
            _make_eq_check(True, "Subnet fl-server public IP mapping")
        )

    @pulumi.runtime.test
    def test_private_subnets_no_public_ip(self):
        checks = []
        for name in ("hospital-a", "hospital-b", "hospital-c"):
            checks.append(
                vpcs[name].subnet.map_public_ip_on_launch.apply(
                    _make_eq_check(False, f"Subnet {name} should not map public IP")
                )
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_vpcs_tagged_with_project(self):
        checks = []
        for name in EXPECTED_VPCS:
            checks.append(
                vpcs[name].vpc.tags.apply(
                    _make_tag_check("Project", "fedcost", f"VPC {name}")
                )
            )
        return pulumi.Output.all(*checks)


# ── Security Groups ──────────────────────────────────────────────────


class TestSecurityGroups:
    def test_four_sgs_created(self):
        assert len(sgs) == 4

    def test_expected_sg_names(self):
        assert set(sgs.keys()) == EXPECTED_VPCS

    @pulumi.runtime.test
    def test_egress_allows_all_outbound(self):
        checks = []
        for name in EXPECTED_VPCS:

            def _make_check(n):
                def check(rules):
                    assert len(rules) >= 1, f"SG {n}: no egress rules"
                    rule = rules[0]
                    assert rule.protocol == "-1", f"SG {n}: not all-protocol"
                    assert "0.0.0.0/0" in rule.cidr_blocks, (
                        f"SG {n}: egress not open to 0.0.0.0/0"
                    )

                return check

            checks.append(sgs[name].egress.apply(_make_check(name)))
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_sgs_tagged_with_project(self):
        checks = []
        for name in EXPECTED_VPCS:
            checks.append(
                sgs[name].tags.apply(
                    _make_tag_check("Project", "fedcost", f"SG {name}")
                )
            )
        return pulumi.Output.all(*checks)


# ── IAM ──────────────────────────────────────────────────────────────


class TestIAM:
    def test_three_roles_created(self):
        assert len(iam) == 3

    def test_expected_role_keys(self):
        assert set(iam.keys()) == {"hospital", "fl-server", "centralized"}

    @pulumi.runtime.test
    def test_roles_assume_ec2_service(self):
        checks = []
        for name, res in iam.items():

            def _make_check(n):
                def check(policy_json):
                    policy = json.loads(policy_json)
                    principals = [
                        s.get("Principal", {}).get("Service")
                        for s in policy.get("Statement", [])
                    ]
                    assert "ec2.amazonaws.com" in principals, (
                        f"IAM {n}: missing EC2 assume-role principal"
                    )

                return check

            checks.append(res.role.assume_role_policy.apply(_make_check(name)))
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_roles_tagged_with_project(self):
        checks = []
        for name, res in iam.items():
            checks.append(
                res.role.tags.apply(
                    _make_tag_check("Project", "fedcost", f"IAM role {name}")
                )
            )
        return pulumi.Output.all(*checks)


# ── EC2 Instances ────────────────────────────────────────────────────


class TestEC2Instances:
    def test_five_instances_created(self):
        assert len(instances) == 5

    def test_expected_instance_names(self):
        assert set(instances.keys()) == EXPECTED_INSTANCES

    @pulumi.runtime.test
    def test_instance_types(self):
        checks = []
        for name, expected in EXPECTED_INSTANCE_TYPES.items():
            checks.append(
                instances[name].instance.instance_type.apply(
                    _make_eq_check(expected, f"EC2 {name} instance type")
                )
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_root_volume_30gb_gp3(self):
        checks = []
        for name in EXPECTED_INSTANCES:

            def _make_check(n):
                def check(device):
                    assert device.volume_size == 30, f"EC2 {n}: root not 30GB"
                    assert device.volume_type == "gp3", f"EC2 {n}: root not gp3"

                return check

            checks.append(
                instances[name].instance.root_block_device.apply(_make_check(name))
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_instances_have_user_data(self):
        checks = []
        for name in EXPECTED_INSTANCES:

            def _make_check(n):
                def check(ud):
                    assert ud and len(ud) > 0, f"EC2 {n}: missing user data"

                return check

            checks.append(
                instances[name].instance.user_data.apply(_make_check(name))
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_instances_tagged_with_project(self):
        checks = []
        for name in EXPECTED_INSTANCES:
            checks.append(
                instances[name].instance.tags.apply(
                    _make_tag_check("Project", "fedcost", f"EC2 {name}")
                )
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_instances_use_correct_ami(self):
        checks = []
        for name in EXPECTED_INSTANCES:
            checks.append(
                instances[name].instance.ami.apply(
                    _make_eq_check("ami-0123456789abcdef0", f"EC2 {name} AMI")
                )
            )
        return pulumi.Output.all(*checks)


# ── SSM Parameters ───────────────────────────────────────────────────


class TestSSMParameters:
    def test_two_parameters_created(self):
        assert len(ssm_params) == 2

    @pulumi.runtime.test
    def test_parameter_paths(self):
        expected = {"/fedcost/flower-server-ip", "/fedcost/s3-data-bucket"}

        def check(names):
            assert set(names) == expected

        return pulumi.Output.all(*[p.name for p in ssm_params]).apply(check)

    @pulumi.runtime.test
    def test_parameters_are_string_type(self):
        checks = []
        for param in ssm_params:
            checks.append(
                param.type.apply(_make_eq_check("String", "SSM param type"))
            )
        return pulumi.Output.all(*checks)

    @pulumi.runtime.test
    def test_parameters_tagged_with_project(self):
        checks = []
        for param in ssm_params:
            checks.append(
                param.tags.apply(
                    _make_tag_check("Project", "fedcost", "SSM param")
                )
            )
        return pulumi.Output.all(*checks)
