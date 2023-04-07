import json
import pickle

from bson import ObjectId, BSON
import requests

from superduperdb import cf
from superduperdb.types.utils import convert_from_bytes_to_types
from superduperdb.utils import get_database_from_database_type

databases = {}


def find_nearest(database, collection, like, ids=None, semantic_index=None):
    """
    Send off request to remote to find-nearest item based on ``like`` item and ``semantic_index``
    Supports MongoDB.

    :param database: name of database
    :param collection: name of collection
    :param like: name of like
    :param ids: name of ids
    :param semantic_index: name of semantic-index
    """
    json_ = {
        'database': database,
        'collection': collection,
        'filter': bytes(BSON.encode(like)).decode('iso-8859-1'),
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


def clear_remote_cache():
    response = requests.put(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/clear_remote_cache',
        headers=None,
    )
    if response.status_code != 200:
        raise Exception(f'Unsetting hash set failed with status-code {response.status_code}...:'
                        + str(response.content))


def apply_model(database_type, database, name, input_, **kwargs):
    """
    Apply model which is saved in the database to the input.

    :param database_type: type of databas3
    :param database: database name
    :param name: name of model
    :param input_: input_ to model
    :param kwargs: kwargs passed to ``superduperdb.models.utils.apply_model``
    """
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

