# FedCost AWS Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stand up the full FedCost AWS infrastructure (5 VPCs, 5 EC2 instances, S3, VPC peering, IAM, SSM) via a single Pulumi Python stack, deployable with `pulumi up` from `infra/`.

**Architecture:** Single Pulumi stack in `ap-southeast-1` with 5 VPCs (fl-server, hospital-a/b/c, centralized), each containing one public subnet and one EC2 instance. VPC peering connects the FL server to each hospital. S3 bucket holds experiment data. SSM parameters share runtime config. Tailscale on all instances for SSH management.

**Tech Stack:** Python 3.13, Pulumi (pulumi + pulumi_aws), uv for dependency management.

**Design doc:** `docs/plans/2026-03-04-aws-infra-design.md`

---

### Task 1: Project scaffolding — Pulumi project files and dependencies

**Files:**
- Create: `infra/__main__.py`
- Create: `infra/Pulumi.yaml`
- Create: `infra/config.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Step 1: Add Pulumi dependencies to pyproject.toml**

Add `pulumi>=3.0` and `pulumi-aws>=6.0` to the project dependencies:

```toml
[project]
name = "federated-deployment"
version = "0.1.0"
description = "FedCost: Federated Learning for DRG-Based Healthcare Cost Prediction — AWS Infrastructure"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "pulumi>=3.0",
    "pulumi-aws>=6.0",
]
```

Run: `uv sync`

**Step 2: Update .gitignore for Pulumi**

Append to `.gitignore`:

```
# Pulumi
infra/Pulumi.*.yaml
!infra/Pulumi.yaml
```

Note: We git-ignore stack config files (`Pulumi.dev.yaml`) because they may contain encrypted secrets. The main `Pulumi.yaml` is committed.

**Step 3: Create `infra/Pulumi.yaml`**

```yaml
name: fedcost
runtime:
  name: python
  options:
    toolchain: pip
    virtualenv: ../.venv
description: FedCost AWS infrastructure for federated learning experiments
```

This tells Pulumi to use the project-level venv managed by uv (at `../.venv` relative to `infra/`).

**Step 4: Create `infra/config.py`**

Typed dataclass that reads all Pulumi config values:

```python
from dataclasses import dataclass

import pulumi


@dataclass(frozen=True)
class FedCostConfig:
    """Typed configuration read from Pulumi stack config."""

    region: str
    ssh_public_key: str
    tailscale_auth_key: str
    instance_type_hospital: str
    instance_type_centralized: str
    instance_type_fl_server: str

    # VPC CIDR blocks
    cidr_fl_server: str = "10.0.0.0/16"
    cidr_hospital_a: str = "10.1.0.0/16"
    cidr_hospital_b: str = "10.2.0.0/16"
    cidr_hospital_c: str = "10.3.0.0/16"
    cidr_centralized: str = "10.4.0.0/16"


def load_config() -> FedCostConfig:
    """Load configuration from Pulumi stack config."""
    cfg = pulumi.Config()
    aws_cfg = pulumi.Config("aws")

    return FedCostConfig(
        region=aws_cfg.require("region"),
        ssh_public_key=cfg.require("ssh-public-key"),
        tailscale_auth_key=cfg.require_secret("tailscale-auth-key"),
        instance_type_hospital=cfg.get("instance-type-hospital") or "t3.medium",
        instance_type_centralized=cfg.get("instance-type-centralized") or "t3.large",
        instance_type_fl_server=cfg.get("instance-type-fl-server") or "t3.medium",
    )
```

**Step 5: Create `infra/__main__.py` (skeleton)**

```python
"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config

config = load_config()

pulumi.export("region", config.region)
```

**Step 6: Run `pulumi preview` to verify scaffolding**

```bash
cd infra/
pulumi login 's3://ieee-pulumi?region=ap-southeast-1&awssdk=v2'
pulumi stack init dev
pulumi config set aws:region ap-southeast-1
pulumi config set fedcost:ssh-public-key "$(cat ~/.ssh/id_ed25519.pub)"
pulumi config set --secret fedcost:tailscale-auth-key "tskey-auth-PLACEHOLDER"
pulumi preview
```

Expected: Preview completes with 0 resources, 1 output (`region`).

**Step 7: Commit**

```bash
git add pyproject.toml .gitignore infra/Pulumi.yaml infra/__main__.py infra/config.py
git commit -m "feat(infra): scaffold Pulumi project with config dataclass"
```

---

### Task 2: Storage — S3 data bucket

**Files:**
- Create: `infra/storage/__init__.py`
- Create: `infra/storage/s3.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/storage/__init__.py`**

```python
from storage.s3 import create_data_bucket

__all__ = ["create_data_bucket"]
```

**Step 2: Create `infra/storage/s3.py`**

```python
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

    aws.s3.BucketPublicAccessBlockV2(
        "fedcost-data-public-access-block",
        bucket=bucket.id,
        block_public_acls=True,
        block_public_policy=True,
        ignore_public_acls=True,
        restrict_public_buckets=True,
    )

    return bucket
```

**Step 3: Wire into `__main__.py`**

```python
"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: 2 resources to create (BucketV2, BucketPublicAccessBlockV2), 2 outputs.

**Step 5: Commit**

```bash
git add infra/storage/ infra/__main__.py
git commit -m "feat(infra): add S3 data bucket with public access block"
```

---

### Task 3: Network — VPCs, subnets, internet gateways

**Files:**
- Create: `infra/network/__init__.py`
- Create: `infra/network/vpcs.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/network/vpcs.py`**

This creates all 5 VPCs, each with one public subnet, an internet gateway, and a route table.

```python
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
```

**Step 2: Create `infra/network/__init__.py`**

```python
from network.vpcs import VpcResources, create_all_vpcs

__all__ = ["VpcResources", "create_all_vpcs"]
```

**Step 3: Wire into `__main__.py`**

```python
"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from config import load_config
from network import create_all_vpcs
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

# 2. Network
vpcs = create_all_vpcs()

pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)
for name, vpc_res in vpcs.items():
    pulumi.export(f"vpc_{name}_id", vpc_res.vpc.id)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: ~27 resources (5 VPCs + 5 subnets + 5 IGWs + 5 route tables + 5 route table associations + 2 S3 resources).

**Step 5: Commit**

```bash
git add infra/network/ infra/__main__.py
git commit -m "feat(infra): add 5 VPCs with public subnets and IGWs"
```

---

### Task 4: Network — VPC peering (fl-server <-> hospitals)

**Files:**
- Create: `infra/network/peering.py`
- Modify: `infra/network/__init__.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/network/peering.py`**

```python
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
```

**Step 2: Update `infra/network/__init__.py`**

```python
from network.peering import create_peering
from network.vpcs import VpcResources, create_all_vpcs

__all__ = ["VpcResources", "create_all_vpcs", "create_peering"]
```

**Step 3: Wire into `__main__.py`**

Add after the `vpcs = create_all_vpcs()` line:

```python
from network import create_all_vpcs, create_peering

# ... (existing code)

# 2. Network
vpcs = create_all_vpcs()
peerings = create_peering(vpcs)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: +9 new resources (3 peering connections + 6 routes).

**Step 5: Commit**

```bash
git add infra/network/ infra/__main__.py
git commit -m "feat(infra): add VPC peering between FL server and hospitals"
```

---

### Task 5: Security — IAM roles and instance profiles

**Files:**
- Create: `infra/security/__init__.py`
- Create: `infra/security/iam.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/security/iam.py`**

Three IAM roles: hospital (S3 read + SSM read), fl-server (S3 read + SSM write), centralized (S3 read).

```python
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
    # Hospital: read S3 + read SSM
    hospital_policy = data_bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
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

    # FL server: read S3 + read/write SSM
    fl_server_policy = data_bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
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

    # Centralized: read S3 only
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
```

**Step 2: Create `infra/security/__init__.py`**

```python
from security.iam import IamResources, create_iam_resources

__all__ = ["IamResources", "create_iam_resources"]
```

**Step 3: Wire into `__main__.py`**

Add after network section:

```python
from security import create_iam_resources

# 3. Security — IAM
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: +9 new resources (3 roles + 3 policies + 3 instance profiles).

**Step 5: Commit**

```bash
git add infra/security/ infra/__main__.py
git commit -m "feat(infra): add IAM roles and instance profiles for 3 node types"
```

---

### Task 6: Security — Security groups

**Files:**
- Create: `infra/security/security_groups.py`
- Modify: `infra/security/__init__.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/security/security_groups.py`**

```python
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
```

**Step 2: Update `infra/security/__init__.py`**

```python
from security.iam import IamResources, create_iam_resources
from security.security_groups import create_security_groups

__all__ = ["IamResources", "create_iam_resources", "create_security_groups"]
```

**Step 3: Wire into `__main__.py`**

```python
from security import create_iam_resources, create_security_groups

# 3. Security
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
sgs = create_security_groups(vpcs)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: +5 new security groups.

**Step 5: Commit**

```bash
git add infra/security/ infra/__main__.py
git commit -m "feat(infra): add security groups for FL server, hospitals, and centralized"
```

---

### Task 7: Compute — SSH key pair and user-data scripts

**Files:**
- Create: `infra/compute/__init__.py`
- Create: `infra/compute/key_pair.py`
- Create: `infra/compute/user_data/fl_server.sh`
- Create: `infra/compute/user_data/hospital.sh`
- Create: `infra/compute/user_data/centralized.sh`

**Step 1: Create `infra/compute/key_pair.py`**

```python
"""SSH key pair for EC2 instances."""

import pulumi_aws as aws


def create_key_pair(public_key: str) -> aws.ec2.KeyPair:
    """Create an EC2 key pair from the provided SSH public key.

    Parameters
    ----------
    public_key : str
        SSH public key content (e.g., "ssh-ed25519 AAAA...").
    """
    return aws.ec2.KeyPair(
        "fedcost-key",
        public_key=public_key,
        tags={"Name": "fedcost-key", "Project": "fedcost"},
    )
```

**Step 2: Create `infra/compute/user_data/fl_server.sh`**

```bash
#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-fl-server

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn

echo "FL server bootstrap complete"
```

**Step 3: Create `infra/compute/user_data/hospital.sh`**

```bash
#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-${HOSPITAL_NAME}

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git aws-cli
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn

# --- Download hospital data partition from S3 ---
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/partitions/${HOSPITAL_NAME}.csv" /opt/fedcost/data/

echo "Hospital ${HOSPITAL_NAME} bootstrap complete"
```

**Step 4: Create `infra/compute/user_data/centralized.sh`**

```bash
#!/bin/bash
set -euo pipefail

# --- Tailscale ---
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname=fedcost-centralized

# --- Python 3.13 + pip ---
dnf install -y python3.13 python3.13-pip git aws-cli
python3.13 -m pip install --upgrade pip

# --- Project dependencies ---
python3.13 -m pip install flwr torch numpy pandas scikit-learn xgboost shap matplotlib seaborn

# --- Download full dataset from S3 ---
BUCKET=$(aws ssm get-parameter --name /fedcost/s3-data-bucket --query Parameter.Value --output text)
mkdir -p /opt/fedcost/data
aws s3 cp "s3://${BUCKET}/raw/" /opt/fedcost/data/ --recursive

echo "Centralized node bootstrap complete"
```

**Step 5: Create `infra/compute/__init__.py`**

```python
from compute.key_pair import create_key_pair

__all__ = ["create_key_pair"]
```

**Step 6: Commit**

```bash
git add infra/compute/
git commit -m "feat(infra): add SSH key pair and user-data bootstrap scripts"
```

---

### Task 8: Compute — EC2 instances

**Files:**
- Create: `infra/compute/instances.py`
- Modify: `infra/compute/__init__.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/compute/instances.py`**

```python
"""EC2 instance definitions for FedCost infrastructure."""

from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_aws as aws

from network.vpcs import VpcResources
from security.iam import IamResources

USER_DATA_DIR = Path(__file__).parent / "user_data"

INSTANCE_DEFINITIONS = [
    {
        "name": "fl-server",
        "vpc_name": "fl-server",
        "iam_key": "fl-server",
        "user_data_template": "fl_server.sh",
        "extra_env": {},
    },
    {
        "name": "hospital-a",
        "vpc_name": "hospital-a",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-a"},
    },
    {
        "name": "hospital-b",
        "vpc_name": "hospital-b",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-b"},
    },
    {
        "name": "hospital-c",
        "vpc_name": "hospital-c",
        "iam_key": "hospital",
        "user_data_template": "hospital.sh",
        "extra_env": {"HOSPITAL_NAME": "hospital-c"},
    },
    {
        "name": "centralized",
        "vpc_name": "centralized",
        "iam_key": "centralized",
        "user_data_template": "centralized.sh",
        "extra_env": {},
    },
]


@dataclass
class InstanceResult:
    """Result of creating an EC2 instance."""

    instance: aws.ec2.Instance
    private_ip: pulumi.Output
    public_ip: pulumi.Output


def _get_user_data(
    template_name: str,
    tailscale_auth_key: pulumi.Output | str,
    extra_env: dict[str, str],
) -> pulumi.Output:
    """Read a user-data script and inject environment variables."""
    script = (USER_DATA_DIR / template_name).read_text()

    def _inject(ts_key: str) -> str:
        result = script.replace("${TAILSCALE_AUTH_KEY}", ts_key)
        for k, v in extra_env.items():
            result = result.replace(f"${{{k}}}", v)
        return result

    if isinstance(tailscale_auth_key, str):
        return pulumi.Output.from_input(_inject(tailscale_auth_key))
    return tailscale_auth_key.apply(_inject)


def _get_ami() -> pulumi.Output:
    """Get the latest Amazon Linux 2023 AMI."""
    ami = aws.ec2.get_ami(
        most_recent=True,
        owners=["amazon"],
        filters=[
            aws.ec2.GetAmiFilterArgs(
                name="name",
                values=["al2023-ami-2023.*-x86_64"],
            ),
            aws.ec2.GetAmiFilterArgs(
                name="state",
                values=["available"],
            ),
        ],
    )
    return ami.id


def create_instances(
    vpcs: dict[str, VpcResources],
    sgs: dict[str, aws.ec2.SecurityGroup],
    iam: dict[str, IamResources],
    key_pair: aws.ec2.KeyPair,
    tailscale_auth_key: pulumi.Output | str,
    instance_types: dict[str, str],
) -> dict[str, InstanceResult]:
    """Create all 5 FedCost EC2 instances.

    Parameters
    ----------
    vpcs : dict
        VPC resources keyed by name.
    sgs : dict
        Security groups keyed by VPC name.
    iam : dict
        IAM resources keyed by role ("hospital", "fl-server", "centralized").
    key_pair : aws.ec2.KeyPair
        SSH key pair.
    tailscale_auth_key : str or Output
        Tailscale pre-auth key.
    instance_types : dict
        Instance type overrides keyed by name.
    """
    ami_id = _get_ami()
    results: dict[str, InstanceResult] = {}

    for defn in INSTANCE_DEFINITIONS:
        name = defn["name"]
        vpc_name = defn["vpc_name"]

        instance_type = instance_types.get(name, "t3.medium")

        user_data = _get_user_data(
            defn["user_data_template"],
            tailscale_auth_key,
            defn["extra_env"],
        )

        instance = aws.ec2.Instance(
            f"ec2-{name}",
            ami=ami_id,
            instance_type=instance_type,
            subnet_id=vpcs[vpc_name].subnet.id,
            vpc_security_group_ids=[sgs[vpc_name].id],
            iam_instance_profile=iam[defn["iam_key"]].instance_profile.name,
            key_name=key_pair.key_name,
            user_data=user_data,
            root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
                volume_size=30,
                volume_type="gp3",
            ),
            tags={"Name": f"fedcost-{name}", "Project": "fedcost"},
        )

        results[name] = InstanceResult(
            instance=instance,
            private_ip=instance.private_ip,
            public_ip=instance.public_ip,
        )

    return results
```

**Step 2: Update `infra/compute/__init__.py`**

```python
from compute.instances import InstanceResult, create_instances
from compute.key_pair import create_key_pair

__all__ = ["InstanceResult", "create_instances", "create_key_pair"]
```

**Step 3: Wire into `__main__.py` — full file**

```python
"""FedCost AWS Infrastructure — Pulumi entry point."""

import pulumi

from compute import create_instances, create_key_pair
from config import load_config
from network import create_all_vpcs, create_peering
from security import create_iam_resources, create_security_groups
from storage import create_data_bucket

config = load_config()

# 1. Storage
data_bucket = create_data_bucket()

# 2. Network
vpcs = create_all_vpcs()
peerings = create_peering(vpcs)

# 3. Security
iam = create_iam_resources(data_bucket_arn=data_bucket.arn)
sgs = create_security_groups(vpcs)

# 4. Compute
key_pair = create_key_pair(config.ssh_public_key)
instances = create_instances(
    vpcs=vpcs,
    sgs=sgs,
    iam=iam,
    key_pair=key_pair,
    tailscale_auth_key=config.tailscale_auth_key,
    instance_types={
        "fl-server": config.instance_type_fl_server,
        "hospital-a": config.instance_type_hospital,
        "hospital-b": config.instance_type_hospital,
        "hospital-c": config.instance_type_hospital,
        "centralized": config.instance_type_centralized,
    },
)

# --- Exports ---
pulumi.export("region", config.region)
pulumi.export("data_bucket_name", data_bucket.bucket)

for name, vpc_res in vpcs.items():
    pulumi.export(f"vpc_{name}_id", vpc_res.vpc.id)

for name, inst in instances.items():
    pulumi.export(f"ec2_{name}_id", inst.instance.id)
    pulumi.export(f"ec2_{name}_private_ip", inst.private_ip)
    pulumi.export(f"ec2_{name}_public_ip", inst.public_ip)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: +6 new resources (5 instances + 1 key pair).

**Step 5: Commit**

```bash
git add infra/compute/ infra/__main__.py
git commit -m "feat(infra): add 5 EC2 instances with user-data bootstrap"
```

---

### Task 9: SSM — Parameter Store for runtime config

**Files:**
- Create: `infra/ssm/__init__.py`
- Create: `infra/ssm/parameters.py`
- Modify: `infra/__main__.py`

**Step 1: Create `infra/ssm/parameters.py`**

```python
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
        Private IP of the FL server instance.
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
```

**Step 2: Create `infra/ssm/__init__.py`**

```python
from ssm.parameters import create_ssm_parameters

__all__ = ["create_ssm_parameters"]
```

**Step 3: Wire into `__main__.py`**

Add after the compute section:

```python
from ssm import create_ssm_parameters

# 5. SSM Parameters
ssm_params = create_ssm_parameters(
    fl_server_private_ip=instances["fl-server"].private_ip,
    data_bucket_name=data_bucket.bucket,
)
```

**Step 4: Run `pulumi preview`**

```bash
cd infra/ && pulumi preview
```

Expected: +2 SSM parameters. Total resources should be ~50.

**Step 5: Commit**

```bash
git add infra/ssm/ infra/__main__.py
git commit -m "feat(infra): add SSM parameters for Flower server IP and data bucket"
```

---

### Task 10: End-to-end validation — `pulumi preview` on full stack

**Step 1: Run full preview**

```bash
cd infra/ && pulumi preview --diff
```

Expected: ~50 resources total. Verify:
- 5 VPCs, 5 subnets, 5 IGWs, 5 route tables, 5 route table associations
- 3 VPC peering connections, 6 routes (peering)
- 1 S3 bucket, 1 public access block
- 3 IAM roles, 3 IAM policies, 3 IAM policy attachments, 3 instance profiles
- 5 security groups
- 1 key pair, 5 EC2 instances
- 2 SSM parameters

**Step 2: Verify exports**

Check that all expected outputs are listed:
- `region`
- `data_bucket_name`
- `vpc_*_id` (5 entries)
- `ec2_*_id`, `ec2_*_private_ip`, `ec2_*_public_ip` (5 × 3 = 15 entries)

**Step 3: Commit final state**

```bash
git add -A infra/
git commit -m "feat(infra): complete FedCost AWS infrastructure — ready for pulumi up"
```

---

## Summary of all files created

```
infra/
├── __main__.py
├── Pulumi.yaml
├── config.py
├── storage/
│   ├── __init__.py
│   └── s3.py
├── network/
│   ├── __init__.py
│   ├── vpcs.py
│   └── peering.py
├── security/
│   ├── __init__.py
│   ├── iam.py
│   └── security_groups.py
├── compute/
│   ├── __init__.py
│   ├── key_pair.py
│   ├── instances.py
│   └── user_data/
│       ├── fl_server.sh
│       ├── hospital.sh
│       └── centralized.sh
└── ssm/
    ├── __init__.py
    └── parameters.py
```

## First deploy checklist (manual, after implementation)

1. Ensure AWS credentials are configured (`aws sts get-caller-identity`)
2. Ensure the `ieee-pulumi` S3 bucket exists in `ap-southeast-1`
3. Set stack config values (SSH key, Tailscale auth key)
4. Run `pulumi up` and review the plan
5. After deploy: verify instances appear in EC2 console
6. Verify Tailscale shows 5 nodes
7. Upload test data to S3: `aws s3 cp test.csv s3://fedcost-data-dev/partitions/hospital-a.csv`
8. SSH via Tailscale to fl-server: `ssh ec2-user@fedcost-fl-server`
