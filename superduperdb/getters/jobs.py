from superduperdb import cf
import requests


def process(database, collection, method, *args, dependencies=(), **kwargs):
    if 'ids' in kwargs:
        kwargs['ids'] = [str(id_) for id_ in kwargs['ids']]
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/process',
        json={
            'collection': collection,
            'database': database,
            'method': method,
            'args': list(args),
            'kwargs': kwargs,
            'dependencies': list(dependencies),
        }
   )
    if r.status_code == 500:
        raise Exception(r.text)
    else:
        return r.text
