import hashlib
import json
import os

from superduper.base import exceptions


class UpdatingSecretException(Exception):
    """Exception raised when secrets are still updating."""


def check_secret_updates(db):
    """Check if secrets are updated in Snowflake.

    :param db: The database connection object.
    """
    result = db.databackend.raw_sql("CALL v1.wrapper('SHOW SECRETS IN ADMIN')")

    lookup = {r[1]: json.loads(r[5])['status']['hash'] for r in result}

    updating = []
    for k in lookup:
        if k not in os.environ:
            raise exceptions.NotFound("secret", k)

        value = os.environ[k]
        target = hashlib.sha256(value.encode()).hexdigest()
        if lookup[k] != target:
            updating.append(k)

    if updating:
        raise UpdatingSecretException(f'Secrets {updating} are still updating.')
