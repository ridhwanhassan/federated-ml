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
