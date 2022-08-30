from sddb import cf
import requests
import base64
from bson import ObjectId

from sddb.models.converters import FloatTensor


def find_nearest_from_hash(
    database,
    collection,
    semantic_index,
    h,
    n
):
    r = requests.get(
        f'{cf["hash_set"]["url"]}/find_nearest_from_hash',
        params={
            'collection': collection,
            'database': database,
            'semantic_index': semantic_index,
            'n': n,
            'h': base64.b64encode(FloatTensor.encode(h)),
        }
    )
    d = r.json()
    d['ids'] = [ObjectId(id_) for id_ in d['ids']]
    return d

