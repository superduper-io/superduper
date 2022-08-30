from sddb import cf
import requests
from bson import ObjectId


from sddb.models.converters import FloatTensor


def find_nearest_from_hash(
    database,
    collection,
    semantic_index,
    h,
    n
):
    r = requests.post(
        f'http://{cf["hash_set"]["host"]}:{cf["hash_set"]["port"]}/find_nearest_from_hash',
        json={
            'collection': collection,
            'database': database,
            'semantic_index': semantic_index,
            'n': n,
            'h': FloatTensor.encode(h).decode('iso-8859-1'),
        }
    )
    print(r.content)
    d = r.json()
    d['ids'] = [ObjectId(id_) for id_ in d['ids']]
    return d

