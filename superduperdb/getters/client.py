import json
import pickle

from bson import ObjectId, BSON
import requests

from superduperdb import cf


def find_nearest(database, collection, filter, ids=None):
    json_ = {
        'database': database,
        'collection': collection,
        'filter': bytes(BSON.encode(filter)).decode('iso-8859-1'),
    }
    if ids is not None:  # pragma: no cover
        json_['ids'] = [str(id_) for id_ in json_['ids']]

    response = requests.get(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/find_nearest',
        headers=None,
        stream=True,
        json=json_,
    )
    results = json.loads(response.text)
    for i, id_ in enumerate(results['_ids']):
        results['_ids'][i] = ObjectId(id_)
    return results


def unset_hash_set(database, collection):
    json_ = {
        'database': database,
        'collection': collection,
    }
    response = requests.put(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/unset_hash_set',
        headers=None,
        json=json_,
    )
    if response.status_code != 200:
        raise Exception('Unsetting hash set failed...')


def clear_remote_cache():
    response = requests.put(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/clear_remote_cache',
        headers=None,
    )
    if response.status_code != 200:
        raise Exception('Unsetting hash set failed...')


def apply_model(database, collection, name, input_, **kwargs):
    input_ = pickle.dumps(input_).decode('iso-8859-1')
    json_ = {
        'database': database,
        'collection': collection,
        'input_': input_,
        'name': name,
        'kwargs': kwargs,
    }
    response = requests.get(
        f'http://{cf["model_server"]["host"]}:{cf["model_server"]["port"]}/apply_model',
        json=json_
    )
    return pickle.loads(response.content)
