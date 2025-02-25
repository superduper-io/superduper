import hashlib
import json
import os

from superduper.base.exceptions import MissingSecretsException


class UpdatingSecretException(Exception):
    ...


def check_secret_updates(db):
    result = db.databackend.raw_sql("CALL v1.wrapper('SHOW SECRETS')")

    lookup = {r[1]: json.loads(r[5])['status']['hash'] for r in result}

    updating = []
    for k in lookup:
        if k not in os.environ:
            raise MissingSecretsException(f'Secret {k} is missing')

        value = os.environ[k]
        target = hashlib.sha256(value.encode()).hexdigest()
        if lookup[k] != target:
            updating.append(k)

    if updating:
        raise UpdatingSecretException(f'Secrets {updating} are still updating.')
