import json
import os
from urllib.parse import urlencode
from superduper import CFG

HOST = CFG.cluster.rest.uri
VERBOSE = os.environ.get('SUPERDUPER_VERBOSE', '1')


def make_params(params):
    return '?' + urlencode(params)



def insert(client, data):
    query = {'query': 'coll.insert_many(documents)', 'documents': data}
    return client.post('/db/execute', json=query)


def apply(client, component):
    return client.post('/db/apply', json=component)


def delete(client):
    return client.post('/db/execute', json={'query': 'coll.delete_many({})'})


def remove(client, type_id, identifier):
    return client.post(f'/db/remove?type_id={type_id}&identifier={identifier}', json={})


def setup(client):
    from superduper.base.build import build_datalayer
    db = build_datalayer()
    client.app.state.pool = db
    data = [
        {"x": [1, 2, 3, 4, 5], "y": 'test'},
        {"x": [6, 7, 8, 9, 10], "y": 'test'},
    ]
    insert(client, data)
    return client


def teardown(client):
    delete(client)
    remove(client, 'datatype', 'image')


if __name__ == '__main__':
    import sys

    if sys.argv[1] == 'setup':
        setup()
    elif sys.argv[1] == 'teardown':
        teardown()
    else:
        raise NotImplementedError
