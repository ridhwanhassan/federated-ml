"""Typed configuration read from Pulumi stack config."""

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

    # VPC CIDR blocks (fl-server + 5 hospital VPCs)
    cidr_fl_server: str = "10.0.0.0/16"
    cidr_hospital_1: str = "10.1.0.0/16"
    cidr_hospital_2: str = "10.2.0.0/16"
    cidr_hospital_3: str = "10.3.0.0/16"
    cidr_hospital_4: str = "10.4.0.0/16"
    cidr_hospital_5: str = "10.5.0.0/16"


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
