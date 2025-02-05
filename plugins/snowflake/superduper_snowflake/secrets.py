import json


class UpdatingSecretException(Exception):
    ...


def check_secret_updates(db):
    result = db.databackend.conn.raw_sql("CALL v1.wrapper('SHOW SECRETS')")

    lookup = {
        r[1]: json.loads(r[5])['status']['phase']
        for r in result
    }

    updating = []
    for k in lookup:
        if lookup[k] != 'RUNNING':
            updating.append(k)
    if updating:
        raise UpdatingSecretException(f'Secrets {updating} are still updating')
