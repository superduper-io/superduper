import json
import pickle

from bson import ObjectId, BSON
import requests

from superduperdb import cf
from superduperdb.types.utils import convert_from_bytes_to_types
from superduperdb.utils import get_database_from_database_type

databases = {}


def find_nearest(database, collection, filter, ids=None, semantic_index=None):
    json_ = {
        'database': database,
        'collection': collection,
        'filter': bytes(BSON.encode(filter)).decode('iso-8859-1'),
        'semantic_index': semantic_index,
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


def apply_model(database_type, database, name, input_, **kwargs):
    database_name = database
    if database not in databases:
        database = get_database_from_database_type(database_type, database)
        databases[database_name] = database
    input_ = pickle.dumps(input_).decode('iso-8859-1')
    json_ = {
        'database': database_name,
        'input_': input_,
        'name': name,
        'kwargs': kwargs,
    }
    response = requests.get(
        f'http://{cf["model_server"]["host"]}:{cf["model_server"]["port"]}/apply_model',
        json=json_
    )
    out = pickle.loads(response.content)
    if isinstance(out, dict):
        out = convert_from_bytes_to_types(out, converters=database.types)
    return out

