import hashlib
import json
import os
from dataclasses import dataclass
from typing import List

from superduper.base import exceptions
from superduper.base.status import STATUS_FAILED, STATUS_PENDING, STATUS_RUNNING


@dataclass
class SecretStatus:
    """Status information for a secret."""

    Name: str
    Phase: str
    Reason: str
    Msg: str


@dataclass
class SecretStatusReport:
    """Report containing status for all secrets."""

    secrets: List[SecretStatus]


def build_secret_status_report(db) -> SecretStatusReport:
    """Check if secrets are updated in Snowflake and return structured status.

    :param db: The database connection object.
    :return: SecretStatusReport with status for each secret
    """
    result = db.databackend.execute_native("CALL v1.wrapper('SHOW SECRETS IN ADMIN')")

    lookup = {
        # Replace underscores with hyphens
        r["name"].replace("-", "_").upper(): json.loads(r["comment"])['status']['hash']
        for r in result
    }

    secrets_list = []

    for secret_name in lookup:
        if secret_name not in os.environ:
            secrets_list.append(
                SecretStatus(
                    Name=secret_name.lower(),
                    Phase=STATUS_FAILED,
                    Reason="NotFound",
                    Msg=f"Secret {secret_name} not found in environment variables",
                )
            )
            continue

        local_value = os.environ[secret_name]
        local_hash = hashlib.sha256(local_value.encode()).hexdigest()
        remote_hash = lookup[secret_name]

        if remote_hash == local_hash:
            secrets_list.append(
                SecretStatus(
                    Name=secret_name.lower(),
                    Phase=STATUS_RUNNING,
                    Reason="InSync",
                    Msg=f"hash: {local_hash[:8]}",
                )
            )
        else:
            secrets_list.append(
                SecretStatus(
                    Name=secret_name.lower(),
                    Phase=STATUS_PENDING,
                    Reason="Updating",
                    Msg=f"Expected {remote_hash[:8]}... got {local_hash[:8]}...",
                )
            )

    return SecretStatusReport(secrets=secrets_list)


def check_secret_updates(db) -> SecretStatusReport:
    """Check the status of secrets in Snowflake and return a report.

    :param db: The database connection object.
    :return: SecretStatusReport with status for each secret
    """
    report = build_secret_status_report(db)
    raise_if_secrets_pending(report)


def raise_if_secrets_pending(report: SecretStatusReport):
    """Check if any secrets are pending and raise exception if so.

    :param report: SecretStatusReport to check
    :raises UpdatingSecretException: If any secrets are still updating
    """
    pending_secrets = [
        secret for secret in report.secrets if secret.Phase == STATUS_PENDING
    ]

    if pending_secrets:
        raise exceptions.Conflict(
            "secret",
            ", ".join(secret.Name for secret in pending_secrets),
            "Some secrets are still updating",
        )


def secrets_not_ready(report: SecretStatusReport) -> bool:
    """Check if any secrets are not in 'Running' phase.

    :param report: SecretStatusReport to check
    :return: True if any secret is not in 'Running' phase, False if all are Running
    """
    return any(secret.Phase != STATUS_RUNNING for secret in report.secrets)
