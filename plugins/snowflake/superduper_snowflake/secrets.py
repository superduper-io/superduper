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
    result = db.databackend.execute_native("CALL v1.wrapper('SHOW SECRETS IN ADMIN')")

    lookup = {
        r["name"].replace("-", "_").upper(): json.loads(r["comment"])['status']['hash']
        for r in result
    }

    updating = {}
    for k in lookup:
        if k not in os.environ:
            raise exceptions.NotFound("secret", k)

        value = os.environ[k]
        target = hashlib.sha256(value.encode()).hexdigest()
        if lookup[k] != target:
            updating[k] = {'current': lookup[k], 'target': target}

    if updating:
        msg = ', '.join(f"{k} (Expected {v['current']} -> Got{v['target']})" for k, v in updating.items())
        raise UpdatingSecretException(f'Secrets {list(updating.keys())} are still updating. {msg}')
