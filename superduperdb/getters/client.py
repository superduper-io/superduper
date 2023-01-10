import json

from bson import ObjectId, BSON
import requests

from superduperdb import cf


def _find_nearest(database, collection, filter, ids=None):
    json_ = {
        'database': database,
        'collection': collection,
        'filter': bytes(BSON.encode(filter)).decode('iso-8859-1'),
    }
    if ids is not None:  # pragma: no cover
        json_['ids'] = [str(id_) for id_ in json_['ids']]

    response = requests.get(
        f'http://{cf["master"]["host"]}:{cf["master"]["port"]}/_find_nearest',
        headers=None,
        stream=True,
        json=json_,
    )
    results = json.loads(response.text)
    for i, id_ in enumerate(results['_ids']):
        results['_ids'][i] = ObjectId(id_)
    print(results)
    return results
